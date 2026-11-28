"""KPP mixing provider — per-step KPP diffusivities from model state, for steppers.

Wraps the op-level KPP driver (:func:`locomgcm.dynamics.kpp.kpp_calc_diffusivities`)
with the grid-aware halo exchange and convention assembly needed by a stepper.
itself is grid-agnostic and pure-columnar+stencil; this module supplies the
MITgcm-convention haloed inputs and maps the cell-centred output to the staggered
fields the implicit-diffusion solver consumes.

Conventions (docs/NUMERICS.md):
  * Inputs to the driver are MITgcm-convention: theta/salt/u/v haloed C-grid slab
    (u west-face at index i, v south-face at index j); zgrid/hwide MIT-indexed
    (negative, z-up); surface forcing sfT=-Qnet/(Cp*rho0), sfU/sfV=tau*mass2rUnit.
  * Driver returns cell-centred KappaR(k) = diffusivity at the TOP interface of
    layer k (KPPviscAz/diffKzS/diffKzT) + KPPghat + hbl.
  * Cube valence-3 corner: dbloc's 9-point smoothing needs the cube-vertex
    diagonal; scalar edge-exchange leaves it wrong, so theta/salt/maskC get a
    ``fill_cs_corner_tr_rl(mode=2)`` pass (validated bitwise vs MIT).  The velocity
    vector exchange already handles its valence-3 corner (shsq bitwise vs MIT).

The provider assembles both initial-state and subsequent-step KPP inputs
using the same MITgcm conventions.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import torch

from locomgcm.accel.constant_cache import constant_cache
from locomgcm.dynamics.base import TendencyModule, TendencyPhase, TendencyStepDomain
from locomgcm.dynamics.kpp import (
    _canonical_kpp_eos,
    jmd95z_layer_pressure,
    kpp_calc_diffusivities,
    kpp_wscale_tables,
)
from locomgcm.dynamics.planetary_coriolis import spherical_coriolis
from locomgcm.operators.fluxes.tracer_flux import TracerFluxScheme
from locomgcm.geometry.cubed_sphere_corner import fill_cs_corner_tr_rl
from locomgcm.geometry.exchange import (
    FillSpec,
    HaloExtentV1,
    exchange,
    exchange_cubed_sphere_vector,
)
from locomgcm.geometry.vertical_coordinates import ZLevelCoordinate, reference_layer_centres
from locomgcm.state import StateTendency

# Stock KPP parameters and KPP_OPTIONS switches use MITgcm package defaults.
# Vtc/cg are derived once below.
# conc1/phepsi/heat_capacity_cp and the three CPP-equivalent switches are
# carried through to the driver.
#
# CAUTION -- minKPPhbl below is NOT a true MITgcm default.  MITgcm derives it
# from the vertical grid: kpp_init_fixed.F:162-164 sets minKPPhbl = -rC(1) when
# data.kpp leaves it UNSET, and ini_vertical_grid.F:123,144 make that half the
# top layer.  The 25.0 here corresponds to a 50 m top layer and is wrong for any
# other grid -- SCS_CLM's 5 m top layer needs 2.5, and the constant floored hbl
# ten times too deep, splitting kbl on 74% of wet columns.
# The lat-lon and
# cube steps both take their parameters from
# ``locomgcm.steppers.ifs.kpp_inputs.kpp_driver_params``, which replaces minKPPhbl by
# ``KppStatic.min_hbl_default`` (built below from the grid) unless data.kpp
# sets it; any other caller of this table must override minKPPhbl from its own
# grid.
_VONK, _EPSILON, _RICR = 0.4, 0.1, 0.3
_CONCV, _CONCS, _CSTAR = 1.8, 98.96, 10.0
KPP_STOCK_PARAMS: dict = dict(
    gravity=9.81,
    rho_const=1035.0,
    eos_type="jmd95",
    # Parameters of eos_type="linear" (kpp.KppLinearEos); no other family reads them.
    linear_eos=None,
    # KPPuseDoubleDiff and its ddmix parameters (kpp_readparms.F:84,142-143).
    use_double_diff=False,
    Rrho0=1.9,
    dsfmax=1.0e-2,
    Ricr=_RICR,
    epsilon=_EPSILON,
    Vtc=_CONCV * math.sqrt(0.2 / _CONCS / _EPSILON) / (_VONK * _VONK) / _RICR,
    cekman=0.7,
    cmonob=1.0,
    minKPPhbl=25.0,
    vonk=_VONK,
    Riinfty=0.7,
    BVSQcon=-2.0e-5,
    difm0=5.0e-3,
    difs0=5.0e-3,
    dift0=5.0e-3,
    difmcon=0.1,
    difscon=0.1,
    diftcon=0.1,
    visc_arnr=1.0e-3,
    cg=_CSTAR * _VONK * (_CONCS * _VONK * _EPSILON) ** (1.0 / 3.0),
    conc1=5.0,
    phepsi=1.0e-10,
    heat_capacity_cp=3994.0,
    penetrating_sw=True,
    # Fortran SIGN(a, -0.0) of the reference build (kpp.KPP_SIGN_ZERO_MODES).
    # Not a pkg/kpp parameter: the compiler decides it.  Runs set it through
    # ImplicitFreeSurfaceStepper.kpp_sign_zero.
    sign_zero="signbit",
    num_v_smooth_Ri=0,
    kpp_estimate_uref=False,
    kpp_smooth_shsq=True,
    kpp_smooth_dbloc=True,
)


def resolve_kpp_runtime_parameters(params: dict) -> dict:
    """Return KPP parameters with MIT's data.kpp-derived ``Vtc`` refreshed.

    Formula: ``Vtc = concv*sqrt(0.2/concs/epsilon)/vonk**2/Ricr``.
    Source: references/MITgcm/pkg/kpp/kpp_init_fixed.F:121-126,
    sha256=782a793f20d21c59346c49981e75c939ee8b073489164e78eafa352931cdaad2.
    """
    resolved = dict(params)
    ricr, epsilon, vonk = (
        value if isinstance(value, torch.Tensor) else float(value)
        for value in (resolved["Ricr"], resolved["epsilon"], resolved["vonk"])
    )
    root_argument = 0.2 / _CONCS / epsilon
    root = (torch.sqrt(root_argument) if isinstance(root_argument, torch.Tensor)
            else math.sqrt(root_argument))
    resolved["Vtc"] = _CONCV * root / (vonk * vonk) / ricr
    return resolved

_OL = 4  # KPP halo width (MITgcm OLx=4, SMOOTH_SHSQ/DBLOC)


# --- static-field halo cache -----------------------------------------------
# Only a Coriolis field the fixed grid built itself owns a reusable halo.  A
# caller-supplied tensor, and any later assignment, is an explicit control
# input: it recomputes the current halo and never enters this cache, whether or
# not it carries requires_grad.
def _kpp_static_halo_cache_enabled() -> bool:
    """Toggle: cache the haloed static KPP fields across steps (default ON).

    ``LOCOMGCM_KPP_STATIC_HALO_CACHE=0`` re-exchanges maskC/coriolis/kmtj every step
    (A/B + OFF==ON bitwise check), same discipline as ``LOCOMGCM_KPP_FACE_BATCH`` /
    ``LOCOMGCM_CUBE_LOCAL_GRID_CACHE``.  The enable decision is owned by the unified
    control plane (``locomgcm.accel``), not by this module."""
    from locomgcm.accel import gate

    return gate("kpp_static_halo_cache").effective_value


def _static_halo_admitted(static) -> bool:
    """Whether this owner's haloed static fields may be reused across steps.

    build_kpp_static_from_grid sets the flag when it derived Coriolis itself.
    Assigning ``coriolis`` afterwards clears it through ``__setattr__``, because
    a replacement is a new explicit input even under the same owner.  An owner
    carrying no Coriolis at all declares no control here, so its halo -- masks
    and the level count, both fixed by the grid -- stays reusable.
    """
    if not hasattr(static, "coriolis"):
        return True
    return static._coriolis_from_static_fields


def _static_halo(grid, static, sample: torch.Tensor, build):
    """Return the cached (maskC_h, fCori_h, kmtj_h) tuple, building once.

    ``build`` is a zero-arg closure that performs the (per-step-identical) halo
    exchange + corner fill and returns the 3-tuple.  ``sample`` carries the
    dtype/device for the key.  When the toggle is OFF, ``build`` is called every
    time (no cache, no key insertion) — the OFF==ON fallback path."""
    if not _kpp_static_halo_cache_enabled():
        return build()
    if not _static_halo_admitted(static):
        from locomgcm.accel import current_rank, record_dispatch

        record_dispatch(
            "kpp_static_halo_cache",
            outcome="inapplicable",
            variant="maskC_fCori_kmtj",
            site="kpp_mixing._static_halo",
            backend="torch",
            rank=current_rank(),
            reason="explicit_coriolis",
        )
        return build()
    ctx = getattr(grid, "_cube_step_ctx", None)
    # ``static`` belongs to one grid and context; it owns the halo it implies.
    key = (getattr(ctx, "rank", None), sample.dtype, str(sample.device))
    return constant_cache.get_or_build(
        ("kpp_mixing._static_halo", key),
        build,
        variant="maskC_fCori_kmtj",
        site="kpp_mixing._static_halo",
        owner=static,
        selector="kpp_static_halo_cache",
    )


def _background_diffusivity_fields(
    grid,
    static,
    sample: torch.Tensor,
    background_diffusivity_s: float | torch.Tensor,
    background_diffusivity_t: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the two background diffusivity fields for this step.

    Both fields are one repeated constant, so they are built as stride-0 views
    of a 0-d tensor rather than as dense volumes.  Every consumer reads them
    through explicit per-axis strides -- the eager path via ``index_select``
    and a trailing-index slice (``kpp.py:1113`` / ``kpp.py:1661``), the CUDA
    path via ``ld3(bg_s, sbgs, ...)`` with ``S3`` strides taken from the tensor
    itself (``_kpp_fused/kernel.cu:28-31,246,342``) -- so a zero stride is read
    exactly like a materialised copy and every value is bit-identical.  At the
    production shape (nz=90, 960x480, float64) this is 8 B per field instead of
    331.8 MB, i.e. 663.6 MB of DRAM traffic per KPP call that no longer moves.

    ``expand`` on a 0-d tensor cannot be spelled with ``full_like``: that
    allocates the dense volume this is avoiding.

    ``LOCOMGCM_KPP_BG_DENSE=1`` restores the dense spelling.  It exists so the two
    layouts can be timed against each other in one job; the two arms are
    bitwise identical, so it is a
    measurement control, not a numerical option, and nothing should read it to
    decide physics.
    """
    dense = os.environ.get("LOCOMGCM_KPP_BG_DENSE", "") == "1"

    def _build() -> tuple[torch.Tensor, torch.Tensor]:
        def _constant(value: float | torch.Tensor) -> torch.Tensor:
            if isinstance(value, torch.Tensor):
                # An explicit control: keep the caller's tensor on the graph
                # rather than reading a number off it.
                return value.to(dtype=sample.dtype, device=sample.device).expand_as(sample)
            if dense:
                return torch.full_like(sample, float(value))
            scalar = torch.full(
                (), float(value), dtype=sample.dtype, device=sample.device
            )
            return scalar.expand(sample.shape)

        return (
            _constant(background_diffusivity_s),
            _constant(background_diffusivity_t),
        )

    # A tensor here is an explicit Ks/Kt control, not a fixed coefficient.  Its
    # type says so on its own -- requires_grad would be read under tracing, and a
    # cache keyed on its value would hand back a field built from an earlier one.
    if isinstance(background_diffusivity_s, torch.Tensor) or isinstance(
        background_diffusivity_t, torch.Tensor
    ):
        return _build()

    key = (
        sample.dtype, str(sample.device), tuple(sample.shape),
        float(background_diffusivity_s), float(background_diffusivity_t), dense,
    )
    return constant_cache.get_or_build(
        ("kpp_mixing._background_diffusivity_fields", key),
        _build,
        variant="kpp_background_diffusivity_s_t",
        site="kpp_mixing._background_diffusivity_fields",
        owner=grid,
        tensor_only=True,
    )


@dataclass
class KppStatic:
    """Time-invariant KPP grid tensors (build once, reuse every step)."""

    zgrid: torch.Tensor        # (nz+2,) MIT-indexed, z-up negative
    hwide: torch.Tensor        # (nz+2,)
    layer_thickness: torch.Tensor  # (nz,) positive
    maskC: torch.Tensor        # (..., nz, ny, nx) wet mask, native face layout
    kmtj: torch.Tensor         # (..., ny, nx) deepest-wet count
    coriolis: torch.Tensor     # (..., ny, nx) f at cell centre
    wmt: torch.Tensor
    wst: torch.Tensor
    # MITgcm kpp_init_fixed.F:162-163 uses -rC(1) when minKPPhbl is
    # unset. With rF(1)=0 and rkSign=-1, this is 0.5*drF[0]. Keep the
    # preparation-time host float here so the per-step consumer never syncs.
    min_hbl_default: float
    # drF(1) as a preparation-time host float for ``kpp_surface_friction_velocity``
    # (``float(layer_thickness[0])`` in the per-step body is a D2H sync
    # and a graph break; ``None`` keeps the historical spelling for callers that
    # build ``KppStatic`` by hand).
    layer1_thickness: float | None = None
    # Set by build_kpp_static_from_grid when it derived Coriolis itself.
    _coriolis_from_static_fields: bool = field(default=False, init=False, repr=False)

    def __setattr__(self, name, value) -> None:
        object.__setattr__(self, name, value)
        if name == "coriolis":
            object.__setattr__(self, "_coriolis_from_static_fields", False)


def build_kpp_static_from_grid(
    grid,
    layer_thickness: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
    coriolis: torch.Tensor | None = None,
    omega: float,
) -> KppStatic:
    """Prepare the static KPP tensors from a LOCOMgcm grid.

    ``layer_thickness`` is the positive drF (nz,).  zgrid/hwide are built MIT-style
    (rC = -cumulative centre depth).  Coriolis is taken from the grid centre
    latitude (``metrics.face_geometry['yC']``) when present, else from
    ``masks``/metrics; kmtj is the wet-layer count from hFacC.

    Call this builder eagerly during preparation. Tracing the CPU tensor
    constructors allows Inductor to move only some operands to CUDA before
    ``aten.cat``, producing inconsistent operand devices.
    """
    drF_host = layer_thickness.detach().to(device="cpu", dtype=torch.float64)
    nz = int(drF_host.shape[0])
    vert = ZLevelCoordinate.from_delR(drF_host)
    rF = -vert.interface_depth
    _, centers = reference_layer_centres(drF_host)
    rC = -centers
    phepsi = KPP_STOCK_PARAMS["phepsi"]
    zg = torch.empty(nz + 2, dtype=torch.float64, device="cpu")
    zg[0] = phepsi; zg[1:nz + 1] = rC; zg[nz + 1] = rC[-1] * 100.0
    hw = torch.empty(nz + 2, dtype=torch.float64, device="cpu")
    hw[0] = phepsi; hw[1:nz + 1] = drF_host; hw[nz + 1] = phepsi

    hFacC = grid.masks.hFacC.to(dtype=dtype, device=device)
    maskC = (hFacC > 0).to(dtype)
    kmtj = (hFacC > 0).sum(dim=-3).to(dtype)  # count over k -> (..., ny, nx)

    coriolis_from_static_fields = coriolis is None
    if coriolis is not None:
        # caller-supplied f at cell centre (regional spherical_polar does not
        # expose yC, so the harness/stepper recomputes f from latitude).
        coriolis = coriolis.to(dtype=dtype, device=device)
    else:
        yC = grid.metrics.face_geometry["yC"] if hasattr(grid.metrics, "face_geometry") else None
        if yC is not None:
            coriolis = spherical_coriolis(yC.to(dtype=dtype, device=device), omega=omega)
        else:
            latC = grid.metrics.yC_for(dtype, device) if hasattr(grid.metrics, "yC_for") else None
            if latC is None:
                raise ValueError("cannot derive KPP coriolis: pass coriolis= or expose grid yC")
            coriolis = spherical_coriolis(latC, omega=omega)

    wmt, wst = kpp_wscale_tables()
    wmt, wst = wmt.to(dtype=dtype, device=device), wst.to(dtype=dtype, device=device)
    static = KppStatic(
        zgrid=zg.to(dtype=dtype, device=device),
        hwide=hw.to(dtype=dtype, device=device),
        layer_thickness=drF_host.to(dtype=dtype, device=device),
        maskC=maskC, kmtj=kmtj, coriolis=coriolis, wmt=wmt, wst=wst,
        min_hbl_default=0.5 * float(drF_host[0]),
        layer1_thickness=float(drF_host[0]),
    )
    # This builder runs during eager step preparation, so it can read gradient
    # metadata; the traced consumer reads only the flag it writes here.
    static._coriolis_from_static_fields = (
        coriolis_from_static_fields and not coriolis.requires_grad
    )
    return static


@dataclass
class KppFields:
    """KPP output mapped to the staggered fields the stepper applies."""

    kappa_ru: torch.Tensor   # (..., nz, ny, nx+1)  u-face implicit viscosity KappaRU
    kappa_rv: torch.Tensor   # (..., nz, ny+1, nx)  v-face implicit viscosity KappaRV
    diff_kzt: torch.Tensor   # (..., nz, ny, nx)    temp implicit vertical diffusivity
    diff_kzs: torch.Tensor   # (..., nz, ny, nx)    salt implicit vertical diffusivity
    ghat: torch.Tensor       # (..., nz, ny, nx)    KPPghat nonlocal transport
    hbl: torch.Tensor        # (..., ny, nx)        boundary-layer depth
    visc_cell: torch.Tensor  # (..., nz, ny, nx)    cell-centre KPPviscAz (source of KappaRU/RV)
    surf_forc_t: torch.Tensor  # (..., ny, nx)      sfT (non-shortwave), for the surface heat apply
    surf_forc_s: torch.Tensor  # (..., ny, nx)      sfS, surface salt apply + ghat nonlocal flux
    surf_forc_t_nl: torch.Tensor  # (..., ny, nx)   COMPLETE sfT for the T ghat nonlocal flux
    #                                                 = sfT - Qsw*recip_Cp*recip_rhoConst*KPPfrac


@dataclass(frozen=True)
class RegionalKppInputs:
    """Scientific inputs fixed during one regional KPP evaluation interval.

    Surface fields retain the existing MIT forcing conventions. Pressure is in
    Pa at the EOS sampling time chosen by the caller; it is not sampled again
    from a model during a tracer substep. T/S and outer U/V remain explicit
    arguments because their time levels differ in the tracer loop.
    """

    static: KppStatic
    surf_forc_t: torch.Tensor
    surf_forc_s: torch.Tensor
    surf_forc_u: torch.Tensor
    surf_forc_v: torch.Tensor
    surf_qsw: torch.Tensor | None
    eos_pressure_pa: torch.Tensor | None
    background_diffusivity_s: float | torch.Tensor
    background_diffusivity_t: float | torch.Tensor
    params: dict

    def diagnose(self, temp, salt, u, v, *, grid) -> KppFields:
        return regional_kpp_fields(
            temp, salt, u, v,
            self.surf_forc_t, self.surf_forc_s,
            self.surf_forc_u, self.surf_forc_v, self.surf_qsw,
            grid=grid, static=self.static,
            background_diffusivity_s=self.background_diffusivity_s,
            background_diffusivity_t=self.background_diffusivity_t,
            eos_pressure_pa=self.eos_pressure_pa, params=self.params,
        )


@dataclass
class KppNonlocalTransport(TendencyModule):
    """GAD nonlocal source from this step's diagnosed KPP fields.

    Uses MIT surfaceForcingT/S conventions and positive layer thickness.
    The shared tracer policy places this explicit source inside AB; the
    diagnosed diffusivities remain inputs to the later column solve.
    ``temp_diffusivity``/``salt_diffusivity`` multiply the nonlocal flux:
    KPPdiffKzT/S, plus GM/Redi Kwz under KPP_ghatUseTotalDiffus
    (``kpp_transport_t.F:69-82``, ``kpp_transport_s.F:86-95``), as the caller chose.
    """

    kpp: KppFields
    layer_thickness: torch.Tensor
    tracer_names: tuple[str, ...]
    temp_diffusivity: torch.Tensor
    salt_diffusivity: torch.Tensor
    name = "kpp_nonlocal_transport"
    stepper_domain = TendencyStepDomain.SLOW
    tendency_phase = TendencyPhase.TRACER
    ab2_eligible = True

    def evaluate(self, state, diagnostics, grid) -> StateTendency:
        from .implicit_vertical_advection import tracer_hfac_factors

        temp = state.temp.data
        _, recip_hfac = tracer_hfac_factors(
            field=temp, hfac_c=grid.masks.hFacC.to(dtype=temp.dtype, device=temp.device),
        )
        tracers = {}
        for name, diff, forcing in (
            ("temp", self.temp_diffusivity, self.kpp.surf_forc_t_nl),
            ("salt", self.salt_diffusivity, self.kpp.surf_forc_s),
        ):
            if name in self.tracer_names:
                tracers[name] = kpp_ghat_tendency(
                    self.kpp.ghat, diff, forcing, self.layer_thickness, recip_hfac,
                )
            else:
                tracers[name] = torch.zeros_like(getattr(state, name).data)
        return StateTendency(
            eta=torch.zeros_like(state.eta.data), u=torch.zeros_like(state.u.data),
            v=torch.zeros_like(state.v.data), temp=tracers["temp"], salt=tracers["salt"],
        )


def mix_kpp_tracers(
    temp_rhs: torch.Tensor,
    salt_rhs: torch.Tensor,
    *,
    temp_reference: torch.Tensor,
    salt_reference: torch.Tensor,
    temp_limiter_reference: torch.Tensor,
    salt_limiter_reference: torch.Tensor,
    kpp: KppFields,
    temp_diffusivity: torch.Tensor,
    salt_diffusivity: torch.Tensor,
    nonlocal_temp_diffusivity: torch.Tensor,
    nonlocal_salt_diffusivity: torch.Tensor,
    layer_thickness: torch.Tensor,
    grid,
    dt,
    tracer_w,
    temperature_implicit_advection: bool,
    salinity_implicit_advection: bool,
    temperature_advection_scheme: TracerFluxScheme | None,
    salinity_advection_scheme: TracerFluxScheme | None,
    multi_dim_advection: bool,
    nonlocal_tracer_names: tuple[str, ...],
    continuity_tracer_names: tuple[str, ...],
    recip_hfac: torch.Tensor | None,
    mask_c_current: torch.Tensor | None,
    recip_hfac_current_c: torch.Tensor | None,
    recip_hfac_continuity: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply nonlocal sources and the configured T/S column solves.

    RHS fields already include the outer tendency. The reference tracers enter
    the raw Euler continuity correction; the limiter references select the
    current substep tracer when global multiDimAdvection is false. Otherwise
    the limiter takes the full RHS after the nonlocal and continuity sources.
    w uses MIT upward-positive units.
    KPP fields and layer thickness belong to this evaluation, not a step cache.
    ``temp_diffusivity``/``salt_diffusivity`` are the column solves' KappaRT/RS
    (``calc_3d_diffusivity.F``); the nonlocal source multiplies
    ``nonlocal_temp_diffusivity``/``nonlocal_salt_diffusivity``, KPPdiffKzT/S
    plus GM/Redi Kwz under KPP_ghatUseTotalDiffus (``kpp_transport_t.F``,
    ``kpp_transport_s.F``).
    Only named tracers still need their nonlocal source here. Sources already
    included in the outer AB bucket must not be added a second time.
    continuity_tracer_names explicitly selects the Euler fields still needing C.
    ``recip_hfac`` explicitly selects the thickness used by the nonlocal
    source and column matrix (``thermodynamics.F:236-244``).
    ``recip_hfac_continuity`` selects the post-AB Euler continuity volume;
    ``recip_hfac_current_c`` retains the current column transport metric.
    """
    from locomgcm.dynamics.implicit_vertical_advection import (
        add_tracer_vertical_continuity,
        solve_tracer_implicit_vertical_advection,
        solve_tracer_vertical_diffusion,
    )

    if "temp" in nonlocal_tracer_names:
        temp_rhs = temp_rhs + dt * kpp_ghat_tendency(
            kpp.ghat, nonlocal_temp_diffusivity, kpp.surf_forc_t_nl, layer_thickness, recip_hfac,
        )
    if "salt" in nonlocal_tracer_names:
        salt_rhs = salt_rhs + dt * kpp_ghat_tendency(
            kpp.ghat, nonlocal_salt_diffusivity, kpp.surf_forc_s, layer_thickness, recip_hfac,
        )
    if temperature_implicit_advection:
        if "temp" in continuity_tracer_names:
            temp_rhs = add_tracer_vertical_continuity(
                tracer_n=temp_reference, rhs=temp_rhs,
                w=tracer_w, dt_lev=dt, grid=grid,
                mask_c_current=mask_c_current,
                recip_hfac_matrix_c=recip_hfac_continuity,
            )
        temp = solve_tracer_implicit_vertical_advection(
            rhs=temp_rhs,
            advection_scheme=temperature_advection_scheme,
            limiter_tracer=temp_rhs if multi_dim_advection else temp_limiter_reference,
            w=tracer_w, dt_lev=dt, diffusivity=temp_diffusivity, grid=grid,
            mask_c_current=mask_c_current,
            recip_hfac_matrix_c=recip_hfac,
            recip_hfac_current_c=recip_hfac_current_c,
        )
    else:
        temp = solve_tracer_vertical_diffusion(
            temp_rhs, grid, dt, temp_diffusivity, recip_hFac=recip_hfac,
        )
    if salinity_implicit_advection:
        if "salt" in continuity_tracer_names:
            salt_rhs = add_tracer_vertical_continuity(
                tracer_n=salt_reference, rhs=salt_rhs,
                w=tracer_w, dt_lev=dt, grid=grid,
                mask_c_current=mask_c_current,
                recip_hfac_matrix_c=recip_hfac_continuity,
            )
        salt = solve_tracer_implicit_vertical_advection(
            rhs=salt_rhs,
            advection_scheme=salinity_advection_scheme,
            limiter_tracer=salt_rhs if multi_dim_advection else salt_limiter_reference,
            w=tracer_w, dt_lev=dt, diffusivity=salt_diffusivity, grid=grid,
            mask_c_current=mask_c_current,
            recip_hfac_matrix_c=recip_hfac,
            recip_hfac_current_c=recip_hfac_current_c,
        )
    else:
        salt = solve_tracer_vertical_diffusion(
            salt_rhs, grid, dt, salt_diffusivity, recip_hFac=recip_hfac,
        )
    return temp, salt


def subcycle_kpp_tracers(
    temp_initial: torch.Tensor,
    salt_initial: torch.Tensor,
    *,
    outer_u: torch.Tensor,
    outer_v: torch.Tensor,
    temp_tendency: torch.Tensor,
    salt_tendency: torch.Tensor,
    tracer_w,
    kpp_inputs: RegionalKppInputs,
    grid,
    dt,
    substeps: int,
    temperature_implicit_advection: bool,
    salinity_implicit_advection: bool,
    temperature_advection_scheme: TracerFluxScheme | None,
    salinity_advection_scheme: TracerFluxScheme | None,
    multi_dim_advection: bool,
    outer_nonlocal: StateTendency | None,
    nonlocal_ab_names: tuple[str, ...],
    continuity_tracer_names: tuple[str, ...],
    mask_c_current: torch.Tensor,
    recip_hfac_current_c: torch.Tensor,
    recip_hfac: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance T/S with refreshed KPP and fixed outer velocity and tendencies.

    Continuity correction uses the initial tracers, matching the fixed outer
    horizontal tendencies. KPP and the evolving RHS use the current substep
    tracers. Momentum consumes its original KPP evaluation and is not updated.
    AB(C) stays in the fixed outer tendency; Euler C is added after each new N.
    The caller selects the nonlocal and column volume. Masks and transport
    metrics use the caller's current geometry and remain fixed during substeps.
    """
    # The outer AB bucket already contains N^n plus its history correction.
    # Replace only raw N^n by the refreshed substep source below, retaining
    # the outer AB correction. Fields without AB keep the original arithmetic.
    if "temp" in nonlocal_ab_names:
        temp_tendency = temp_tendency - outer_nonlocal.temp
    if "salt" in nonlocal_ab_names:
        salt_tendency = salt_tendency - outer_nonlocal.salt
    h = dt / substeps
    temp, salt = temp_initial, salt_initial
    for _ in range(substeps):
        kpp = kpp_inputs.diagnose(temp, salt, outer_u, outer_v, grid=grid)
        temp_rhs = temp + h * temp_tendency
        salt_rhs = salt + h * salt_tendency
        temp, salt = mix_kpp_tracers(
            temp_rhs, salt_rhs,
            temp_reference=temp_initial, salt_reference=salt_initial,
            temp_limiter_reference=temp, salt_limiter_reference=salt,
            kpp=kpp, temp_diffusivity=kpp.diff_kzt, salt_diffusivity=kpp.diff_kzs,
            nonlocal_temp_diffusivity=kpp.diff_kzt, nonlocal_salt_diffusivity=kpp.diff_kzs,
            layer_thickness=kpp_inputs.static.layer_thickness,
            grid=grid, dt=h, tracer_w=tracer_w,
            temperature_implicit_advection=temperature_implicit_advection,
            salinity_implicit_advection=salinity_implicit_advection,
            temperature_advection_scheme=temperature_advection_scheme,
            salinity_advection_scheme=salinity_advection_scheme,
            multi_dim_advection=multi_dim_advection,
            nonlocal_tracer_names=("temp", "salt"),
            continuity_tracer_names=continuity_tracer_names,
            recip_hfac=recip_hfac,
            mask_c_current=mask_c_current,
            recip_hfac_current_c=recip_hfac_current_c,
            recip_hfac_continuity=recip_hfac,
        )
    return temp, salt


def _nonlocal_temp_forcing(
    surf_forc_t: torch.Tensor,
    surf_qsw: torch.Tensor | None,
    hbl: torch.Tensor,
    params: dict,
) -> torch.Tensor:
    """Complete temperature surface forcing for the KPP nonlocal ghat flux.

    The KPP counter-gradient flux carries the COMPLETE surface heat forcing, not
    just the non-shortwave part: the shortwave absorbed *within* the mixing layer
    also drives nonlocal transport.  MITgcm ``kpp_transport_t.F`` uses

        sf_eff = surfaceForcingT - Qsw*recip_Cp*recip_rhoConst*KPPfrac

    with ``KPPfrac = 1 - swfrac(hbl)`` the fraction of shortwave heating the mixing
    layer (``KPP.h:21``; ``kpp_calc.F:736-746``, the ``KPPuseSWfrac3D=.FALSE.``
    branch = ``1 - swfrac(KPPhbl)``).  ``surf_qsw`` is MITgcm upward-positive net
    shortwave. Disabled ``params['penetrating_sw']`` returns ``surf_forc_t``
    unchanged; enabled mode retains the radiative expression even at zero flux.
    Dropping this term is op-level invisible (KPP fields/penetration
    all still match) but biases the top-of-column tracer by O(1e-3) per step,
    largest where the boundary layer reaches the bottom (shallow columns).
    """
    if not params["penetrating_sw"]:
        return surf_forc_t
    if surf_qsw is None:
        surf_qsw = torch.zeros_like(surf_forc_t)
    from locomgcm.dynamics.surface_flux import swfrac

    # jwtype=2 (MITgcm hardcodes the Jerlov type; matches the penetration term and
    # kpp_bldepth).  hbl is positive-down depth, the swfrac() convention.
    kpp_frac = 1.0 - swfrac(hbl, 2)
    recip_cp = 1.0 / params["heat_capacity_cp"]
    recip_rho = 1.0 / params["rho_const"]
    return surf_forc_t - surf_qsw * recip_cp * recip_rho * kpp_frac


def kpp_ghat_tendency(
    ghat: torch.Tensor,
    diff_kz: torch.Tensor,
    sf_x: torch.Tensor,
    layer_thickness: torch.Tensor,
    recip_hFacC: torch.Tensor,
) -> torch.Tensor:
    """Explicit nonlocal KPP transport tendency (MIT kpp_transport_t + gad_calc_rhs).

    Nonlocal vertical flux at interface k (top of layer k):
      ``G[k] = -diff_kz[k] * ghat[k-1] * sf_x``   (k=1..nz-1; G[0]=G[nz]=0)
    tendency: ``tend[k] = -recip_hFacC[k] * recip_drF[k] * (G[k] - G[k+1])``
    (MITgcm gad_calc_rhs.F:769-781 ``gTracer -= recip_VV*(fVerT(kDown)-fVerT(kUp))*rkSign``,
    rkSign=-1; kpp_transport_t.F:91-100 ``df=-rA*KPPdiffKzT(k)*KPPghat(k-1)*sfT``; rA cancels).

    Shapes: ``ghat``/``diff_kz``/``recip_hFacC`` (..., nz, ny, nx); ``sf_x`` (..., ny, nx);
    ``layer_thickness`` (nz,).  Returns (..., nz, ny, nx).  Apply BEFORE the implicit
    vertical-diffusion solve (explicit), matching MIT's explicit-then-implicit order.
    """
    nz = ghat.shape[-3]
    lead = ghat.shape[:-3]
    G = ghat.new_zeros(lead + (nz + 1,) + ghat.shape[-2:])
    G[..., 1:nz, :, :] = -diff_kz[..., 1:nz, :, :] * ghat[..., 0:nz - 1, :, :] * sf_x.unsqueeze(-3)
    recip_drF = (1.0 / layer_thickness.reshape(-1)).reshape((1,) * len(lead) + (nz, 1, 1))
    return -recip_hFacC * recip_drF * (G[..., 0:nz, :, :] - G[..., 1:nz + 1, :, :])


def _kpp_face_batch_enabled() -> bool:
    """Face-batch toggle: collapse the per-face KPP driver loop into ONE batched
    call on the (n_faces, nz, ny, nx) slab.  Default ON; ``LOCOMGCM_KPP_FACE_BATCH=0``
    forces the per-face loop (A/B + OFF==HEAD verification).  The driver is
    vertical-axis-aware so the batched call is BITWISE vs the loop (op-level gate
    ).  The enable decision is owned by
    the unified control plane (``locomgcm.accel``), not by this module."""
    from locomgcm.accel import gate

    return gate("kpp_face_batch").effective_value


def _record_face_batch(outcome: str, *, reason: str | None = None) -> None:
    """Dispatch record for LOCOMGCM_KPP_FACE_BATCH, taken at the driver call."""
    from locomgcm.accel import current_rank, record_dispatch

    record_dispatch(
        "kpp_face_batch",
        outcome=outcome,
        variant="cube_six_face_slab",
        site="kpp_mixing.cube_kpp_fields",
        backend="torch",
        rank=current_rank(),
        reason=reason,
    )


def cube_kpp_fields(
    temp: torch.Tensor,
    salt: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    surf_forc_t: torch.Tensor,
    surf_forc_s: torch.Tensor,
    surf_forc_u: torch.Tensor,
    surf_forc_v: torch.Tensor,
    surf_qsw: torch.Tensor | None = None,
    *,
    grid,
    static: KppStatic,
    params: dict = KPP_STOCK_PARAMS,
    background_diffusivity_s: float,
    background_diffusivity_t: float,
) -> KppFields:
    """KPP diffusivities for a cubed-sphere step (per-face, validated vs MITgcm).

    The background diffusivities (diffKrS/T) enter KPP as diffusKzS/T, as on the
    regional path: CALC_3D_DIFFUSIVITY writes them into KPPdiffKzS/T before KPPMIX
    (kpp_calc.F), which uses them for the interior and boundary-layer matching
    (kpp_routines.F:295,1195-1208), and KPPdiffKzT is then the whole tracer
    diffusivity (kpp_calc_diff_t.F:50).

    ``temp/salt`` (6,nz,ny,nx), ``u`` (6,nz,ny,nx+1), ``v`` (6,nz,ny+1,nx).  The KPP
    surface buoyancy forcing is the COMPLETE forcing, not the post-split fields:
    ``surf_forc_t/s`` (6,ny,nx, sfT/sfS centre) → ``bo``; ``surf_forc_u/v`` (u/v face
    slot) → ``ustar``; ``surf_qsw`` (6,ny,nx, net shortwave, MITgcm upward-positive)
    → ``bosol`` (the penetrating-shortwave buoyancy, MITgcm kpp_forcing_surf.F).  When
    ``surf_qsw`` is None / absent it supplies zero flux; ``params['penetrating_sw']``
    independently selects the physical branch.  Returns :class:`KppFields` with KappaRU/RV
    averaged from cell-centre KPPviscAz (haloed slab so cross-face faces are correct).
    """
    params = resolve_kpp_runtime_parameters(params)
    n = grid.nx
    # Face count is the LEADING dim of the inputs: 6 single-rank, n_local multi-rank
    # (face-local KPP).  The cube exchanges auto-route to the distributed
    # P2P path when ``grid._cube_step_ctx`` is set (localized grid); the per-face
    # KPP kernel is unchanged -- it just loops the owned faces.
    n_faces = temp.shape[0]
    nz = temp.shape[-3]
    dtype, device = temp.dtype, temp.device
    ext = HaloExtentV1.uniform(_OL)

    def sc(fld):
        return exchange(fld, grid=grid, location="T", extent=ext, fill=FillSpec.ZERO)

    def vec(uu, vv):
        # Route to the distributed C-grid vector exchange when the grid is rank-
        # local; single-rank otherwise.  ``exchange_cubed_sphere_vector`` is
        # a direct primitive (no auto-route wrapper like ``exchange``).
        kctx = getattr(grid, "_cube_step_ctx", None)
        if kctx is not None:
            from locomgcm.geometry.exchange import exchange_cubed_sphere_vector_distributed
            return exchange_cubed_sphere_vector_distributed(
                uu, vv, grid=grid, extent=ext, ctx=kctx
            )
        return exchange_cubed_sphere_vector(uu, vv, grid=grid, extent=ext)

    temp_h = fill_cs_corner_tr_rl(sc(temp), 2, grid)
    salt_h = fill_cs_corner_tr_rl(sc(salt), 2, grid)
    # maskC/coriolis/kmtj are time-invariant -> halo once, reuse every step
    # (the exact same fields the per-step exchange would produce; bitwise).
    def _build_static_h():
        return (
            fill_cs_corner_tr_rl(sc(static.maskC), 2, grid),
            sc(static.coriolis),
            sc(static.kmtj),
        )
    maskC_h, fCori_h, kmtj_h = _static_halo(grid, static, temp, _build_static_h)
    sfT_h, sfS_h = sc(surf_forc_t), sc(surf_forc_s)
    qsw_h = sc(surf_qsw) if surf_qsw is not None else None   # net shortwave -> bosol
    u_h, v_h = vec(u, v)
    sfU_stag = u.new_zeros(n_faces, n, n + 1); sfU_stag[..., :, 0:n] = surf_forc_u
    sfV_stag = v.new_zeros(n_faces, n + 1, n); sfV_stag[..., 0:n, :] = surf_forc_v
    sfU_h, sfV_h = vec(sfU_stag, sfV_stag)
    bg_kzs, bg_kzt = _background_diffusivity_fields(
        grid, static, temp_h, background_diffusivity_s, background_diffusivity_t,
    )

    # The KPP column driver broadcasts over a leading face-batch dim (the
    # exchanges above already produced the full (n_faces, ...) haloed slab in ONE
    # pass — NOT per face).  Default: ONE batched call over all owned faces, cutting
    # the per-step driver launch count ~n_faces×.  The driver is vertical-axis-aware
    # (kpp.py) so this is BITWISE vs the per-face loop; ``LOCOMGCM_KPP_FACE_BATCH=0`` and
    # the non-broadcastable guard fall back to the loop.  Outputs slabbed (face,...).
    cut = (slice(None), slice(_OL, _OL + n), slice(_OL, _OL + n))
    cut2 = (slice(_OL, _OL + n), slice(_OL, _OL + n))
    # Guard: all per-face haloed slabs must carry the same leading face-batch dim
    # (they do — built by the single-pass exchanges above).  If something upstream
    # ever breaks that invariant, fall back to the per-face loop rather than feed a
    # non-broadcastable slab to the driver.
    _batch_ok = (
        temp_h.shape[0] == n_faces and salt_h.shape[0] == n_faces
        and u_h.shape[0] == n_faces and v_h.shape[0] == n_faces
        and maskC_h.shape[0] == n_faces and kmtj_h.shape[0] == n_faces
        and (qsw_h is None or qsw_h.shape[0] == n_faces)
    )
    _face_batch_requested = _kpp_face_batch_enabled()
    if _face_batch_requested and _batch_ok:
        viscAz, diffKzS, diffKzT, gh, hb = kpp_calc_diffusivities(
            temp_h, salt_h, u_h, v_h, sfT_h, sfS_h, sfU_h, sfV_h,
            fCori_h, maskC_h, kmtj_h, static.layer_thickness, static.zgrid,
            static.hwide, static.wmt, static.wst, qsw=qsw_h, bg_KzS=bg_kzs, bg_KzT=bg_kzt,
            layer1_thickness=getattr(static, "layer1_thickness", None), **params,
        )
        # The single batched driver call above IS the accelerated path;
        # record the hit after it ran, not at the predicate.
        _record_face_batch("hit")
        visc_h = viscAz                                 # keep haloed for face averaging
        kzt, kzs = diffKzT[(slice(None),) + cut], diffKzS[(slice(None),) + cut]
        ghat = gh[(slice(None),) + cut]
        hbl = hb[(slice(None),) + cut2]
    else:
        if _face_batch_requested:
            # Requested but the per-face slabs are not batch-shaped: declare the
            # decline with a reason instead of falling back silently.
            _record_face_batch(
                "inapplicable", reason="face_slab_leading_dim_mismatch"
            )
        visc_h = temp.new_zeros(n_faces, nz, n + 2 * _OL, n + 2 * _OL)
        kzt = temp.new_zeros(n_faces, nz, n, n)
        kzs = temp.new_zeros(n_faces, nz, n, n)
        ghat = temp.new_zeros(n_faces, nz, n, n)
        hbl = temp.new_zeros(n_faces, n, n)
        for f in range(n_faces):
            viscAz, diffKzS, diffKzT, gh, hb = kpp_calc_diffusivities(
                temp_h[f], salt_h[f], u_h[f], v_h[f], sfT_h[f], sfS_h[f], sfU_h[f], sfV_h[f],
                fCori_h[f], maskC_h[f], kmtj_h[f], static.layer_thickness, static.zgrid,
                static.hwide, static.wmt, static.wst,
                qsw=(qsw_h[f] if qsw_h is not None else None),
                bg_KzS=bg_kzs[f], bg_KzT=bg_kzt[f], **params,
            )
            visc_h[f] = viscAz                          # keep haloed for face averaging
            kzt[f] = diffKzT[cut]; kzs[f] = diffKzS[cut]
            ghat[f] = gh[cut]; hbl[f] = hb[cut2]

    # cell-centre viscAz -> u/v faces (MIT KappaRU/RV): 2-pt horizontal average,
    # using the haloed viscAz so the west/south face at a face edge picks up the
    # cross-face neighbour cell.
    rows = visc_h[..., _OL:_OL + n, _OL - 1:_OL + n + 1]   # (6,nz,ny,nx+2)
    avg_ru = 0.5 * (rows[..., :-1] + rows[..., 1:])        # (6,nz,ny,nx+1)
    cols = visc_h[..., _OL - 1:_OL + n + 1, _OL:_OL + n]   # (6,nz,ny+2,nx)
    avg_rv = 0.5 * (cols[..., :-1, :] + cols[..., 1:, :])  # (6,nz,ny+1,nx)
    visc_arnr = float(params["visc_arnr"])
    bg_ru = torch.full_like(avg_ru, visc_arnr)
    bg_rv = torch.full_like(avg_rv, visc_arnr)
    mask_w = grid.masks.u_float(dtype=dtype, device=device)
    mask_s = grid.masks.v_float(dtype=dtype, device=device)
    kappa_ru = torch.maximum(bg_ru, bg_ru - visc_arnr + mask_w * avg_ru)
    kappa_rv = torch.maximum(bg_rv, bg_rv - visc_arnr + mask_s * avg_rv)
    surf_forc_t_nl = _nonlocal_temp_forcing(surf_forc_t, surf_qsw, hbl, params)
    return KppFields(kappa_ru=kappa_ru, kappa_rv=kappa_rv, diff_kzt=kzt,
                     diff_kzs=kzs, ghat=ghat, hbl=hbl,
                     visc_cell=visc_h[..., _OL:_OL + n, _OL:_OL + n],
                     surf_forc_t=surf_forc_t, surf_forc_s=surf_forc_s,
                     surf_forc_t_nl=surf_forc_t_nl)


def regional_kpp_fields(
    temp: torch.Tensor,
    salt: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    surf_forc_t: torch.Tensor,
    surf_forc_s: torch.Tensor,
    surf_forc_u: torch.Tensor,
    surf_forc_v: torch.Tensor,
    surf_qsw: torch.Tensor | None = None,
    *,
    grid,
    static: KppStatic,
    background_diffusivity_s: float,
    background_diffusivity_t: float,
    eos_pressure_pa: torch.Tensor | None,
    params: dict = KPP_STOCK_PARAMS,
) -> KppFields:
    """KPP diffusivities for a regional structured-grid step (single patch, no seam).

    ``temp/salt`` (B,nz,ny,nx), ``u`` (B,nz,ny,nx+1), ``v`` (B,nz,ny+1,nx),
    ``surf_forc_t/s`` (B,ny,nx), ``surf_forc_u`` (B,ny,nx+1), and
    ``surf_forc_v`` (B,ny+1,nx).
    Standard :func:`exchange` fills the OL=4 halo; regional grids have no cube
    valence-3 corner so no ``fill_cs_corner`` pass.  Full staggered u/v arrays enter
    the MIT-convention slab, including the real east/north outer faces.  The
    background tracer diffusivities are complete case values, not additions
    performed by the downstream solver.  Returns batched :class:`KppFields`.
    """
    params = resolve_kpp_runtime_parameters(params)
    ext = HaloExtentV1.uniform(_OL)
    u_ext = HaloExtentV1(_OL, _OL - 1, _OL, _OL)
    v_ext = HaloExtentV1(_OL, _OL, _OL, _OL - 1)
    B = temp.shape[0]
    ny, nx = temp.shape[-2], temp.shape[-1]

    def sc(fld, *, constant=False):  # scalar/compact halo, REPLICATE (regional convention, tracers.py:511)
        return exchange(fld, grid=grid, location="T", extent=ext, fill=FillSpec.REPLICATE,
                        constant=constant)

    def sc_mask(fld):  # the static land mask: a constant on every rank
        return exchange(fld, grid=grid, location="T", extent=ext, fill=FillSpec.ZERO, constant=True)

    # static (grid) halos: maskC/coriolis/kmtj are batch-free (nz/2-D) and
    # time-invariant -> the cache halos them ONCE and reuses every step (bitwise; same
    # fields the per-step exchange would produce).  maskC wraps on periodic axes
    # through exchange() and stays dry at OPEN/WALL physical ghost cells.
    # Coriolis is constant unless the caller supplied it as a control.
    def _build_static_h():
        return (sc_mask(static.maskC), sc(static.coriolis, constant=not static.coriolis.requires_grad),
                sc(static.kmtj, constant=True))
    mk_h, fc_h, km_h = _static_halo(grid, static, temp, _build_static_h)
    cut3 = (slice(None), slice(_OL, _OL + ny), slice(_OL, _OL + nx))
    cut2 = (slice(_OL, _OL + ny), slice(_OL, _OL + nx))
    acc = {k: [] for k in ("ru", "rv", "kzt", "kzs", "ghat", "hbl", "vc")}
    visc_arnr = float(params["visc_arnr"])
    mask_w = grid.masks.u_float(dtype=temp.dtype, device=temp.device)
    mask_s = grid.masks.v_float(dtype=temp.dtype, device=temp.device)
    for b in range(B):
        th, sa = sc(temp[b]), sc(salt[b])
        uu = exchange(
            u[b], grid=grid, location="U", extent=u_ext, fill=FillSpec.REPLICATE,
        )
        vv = exchange(
            v[b], grid=grid, location="V", extent=v_ext, fill=FillSpec.REPLICATE,
        )
        # Sampled surface forcing is a constant unless it is a control; the
        # flag is rank-uniform because the forcing exists on every tile.
        sft = sc(surf_forc_t[b], constant=not surf_forc_t.requires_grad)
        sfs = sc(surf_forc_s[b], constant=not surf_forc_s.requires_grad)
        # sfU/sfV are full face-located arrays (nx+1 / ny+1).  The asymmetric
        # extents retain their real east/north outer faces while producing the
        # same-size MIT slab compute_ustar consumes.
        #
        # Face-located stress must vanish at a WALL, so its halo is filled with
        # zero rather than by replicating the outermost wet face -- the same
        # convention exchange_velocity_pair() applies to u/v on a regional grid
        # (geometry/exchange.py:243-248).  Replicating instead doubled the
        # neighbour sum sfU(i)+sfU(i+1) on every water column touching land,
        # inflating ustar by up to sqrt(2); that showed up as 87 coastal columns
        # whose hbl disagreed with MITgcm by metres, all of them land-adjacent
        # Limit each column using its diagnosed boundary-layer depth.
        sfu = exchange(
            surf_forc_u[b], grid=grid, location="U",
            extent=u_ext, fill=FillSpec.ZERO, constant=not surf_forc_u.requires_grad,
        )
        sfv = exchange(
            surf_forc_v[b], grid=grid, location="V",
            extent=v_ext, fill=FillSpec.ZERO, constant=not surf_forc_v.requires_grad,
        )
        qsw_b = sc(surf_qsw[b], constant=not surf_qsw.requires_grad) if surf_qsw is not None else None   # net shortwave -> bosol
        pressure_b = sc(eos_pressure_pa[b]) if eos_pressure_pa is not None else None
        bg_kzs, bg_kzt = _background_diffusivity_fields(
            grid, static, th, background_diffusivity_s, background_diffusivity_t,
        )
        fused = None
        from locomgcm.accel import (
            consumer_snapshot,
            in_accelerated_dispatch,
            requested,
        )

        if not in_accelerated_dispatch():
            with consumer_snapshot():
                if requested(("kpp_fused_cuda",)):
                    if params.get("surface_buoyancy", "statekpp") != "statekpp":
                        raise ValueError(
                            "kpp_fused_cuda does not implement "
                            "kpp_surface_buoyancy='divided_difference'; set LOCOMGCM_KPP_FUSED_CUDA=0")
                    from locomgcm.dynamics._kpp_fused import dispatcher

                    eos = _canonical_kpp_eos(str(params["eos_type"]))
                    fused_qsw = qsw_b if qsw_b is not None else torch.zeros_like(sft)
                    fused_pressure = pressure_b
                    if fused_pressure is None:
                        fused_pressure = jmd95z_layer_pressure(
                            static.layer_thickness,
                            gravity=float(params["gravity"]),
                            rho_const=float(params["rho_const"]),
                        )
                    fused = dispatcher.regional_kpp_fused_dispatch(
                        (
                            th, sa, uu, vv, sft, sfs, sfu, sfv, fc_h, mk_h,
                            km_h, static.layer_thickness, static.zgrid,
                            static.hwide, static.wmt, static.wst, fused_qsw,
                            fused_pressure, bg_kzs, bg_kzt,
                        ),
                        eos=eos,
                        params=params,
                    )
        if fused is None:
            viscAz, diffKzS, diffKzT, gh, hb = kpp_calc_diffusivities(
                th, sa, uu, vv, sft, sfs, sfu, sfv, fc_h, mk_h, km_h,
                static.layer_thickness, static.zgrid, static.hwide, static.wmt, static.wst,
                qsw=qsw_b, eos_pressure_pa=pressure_b, bg_KzS=bg_kzs, bg_KzT=bg_kzt,
                layer1_thickness=getattr(static, "layer1_thickness", None), **params,
            )
        else:
            viscAz, diffKzS, diffKzT, gh, hb = fused
        acc["kzt"].append(diffKzT[cut3]); acc["kzs"].append(diffKzS[cut3])
        acc["ghat"].append(gh[cut3]); acc["hbl"].append(hb[cut2]); acc["vc"].append(viscAz[cut3])
        rows = viscAz[..., _OL:_OL + ny, _OL - 1:_OL + nx + 1]
        avg_ru = 0.5 * (rows[..., :-1] + rows[..., 1:])
        bg_ru = torch.full_like(avg_ru, visc_arnr)
        acc["ru"].append(torch.maximum(
            bg_ru, bg_ru - visc_arnr + mask_w * avg_ru,
        ))
        cols = viscAz[..., _OL - 1:_OL + ny + 1, _OL:_OL + nx]
        avg_rv = 0.5 * (cols[..., :-1, :] + cols[..., 1:, :])
        bg_rv = torch.full_like(avg_rv, visc_arnr)
        acc["rv"].append(torch.maximum(
            bg_rv, bg_rv - visc_arnr + mask_s * avg_rv,
        ))
    st = lambda key: torch.stack(acc[key], 0)
    hbl_b = st("hbl")
    surf_forc_t_nl = _nonlocal_temp_forcing(surf_forc_t, surf_qsw, hbl_b, params)
    return KppFields(kappa_ru=st("ru"), kappa_rv=st("rv"), diff_kzt=st("kzt"),
                     diff_kzs=st("kzs"), ghat=st("ghat"), hbl=hbl_b, visc_cell=st("vc"),
                     surf_forc_t=surf_forc_t, surf_forc_s=surf_forc_s,
                     surf_forc_t_nl=surf_forc_t_nl)
