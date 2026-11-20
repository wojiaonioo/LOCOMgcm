"""GM/Redi as a tracer tendency module (MITgcm ``DO_OCEANIC_PHYS`` GM part plus ``GAD_CALC_RHS`` GM fluxes).

Per evaluation, from the step-start ``T, S`` the module sees (the same time
level MITgcm's ``DO_OCEANIC_PHYS`` reads, ``forward_step.F:657``, in both
synchronous and staggered stepping):

1. densities as ``do_oceanic_phys.F:759-823``: ``rho_in_situ`` row k at level
   k's reference, ``rho_upper`` row k is level k-1 water at level k's
   reference, with the model's equation-of-state module (bound by the
   builder, ``io/runner/builders/from_manifest.py``);
2. halos of width ``_OL`` by ``exchange`` (topology from the grid,
   ``docs/NUMERICS.md``); the pointwise density commutes with the
   exchange, so densities are computed on the interior and exchanged;
3. ``grad_sigma`` and ``gmredi_calc_tensor`` on the haloed arrays;
4. for the stepped tracers (``tracer_names``) the x, y and r GM fluxes (``gad_calc_rhs.F:343-351,472-480,
   624-632``) and their divergence (``gad_calc_rhs.F:772-781``) as the
   explicit tendency.  The module keeps the default ``ab2_eligible``: it is
   in the AB bucket with advection, and which tracer is extrapolated is
   decided by the stepper's per-tracer AB flags (``gad_init_fixed.F:143-163``).
5. ``Kwz`` (W points, ``[Nr, ny, nx]``, row k the top of layer k, m^2/s) is
   published as diagnostics tensor ``gm_kwz`` for the vertical diffusivity
   assembly, which adds ``Kwz*maskInC`` to T's and S's ``kappaRk`` after KPP
   (``calc_3d_diffusivity.F:203-209``, ``gmredi_calc_diff.F:49-66``).

ldd97 reads ``fCori`` at cell centres (``gmredi_calc_tensor.F:163-211``), a
grid constant of ``ini_cori.F``: the module takes it as ``coriolis_profile_c``,
one value per row (latitude-longitude, beta- or f-plane maps, constant in x),
and gives it the tracer halo.

6. under GM_AdvForm, the bolus velocity of ``gmredi_residual_flow.F:58-98``
   from the stream functions, published as ``gm_bolus_u`` ``[1, Nr, ny, nx+1]``,
   ``gm_bolus_v`` ``[1, Nr, ny+1, nx]`` (both outer faces) and ``gm_bolus_w``
   ``[1, Nr+1, ny, nx]`` (MIT upward-positive, row k the top of layer k, last row
   zero), m/s.  The tracer advection adds them to its Eulerian velocity
   (``thermodynamics.F:258-276``); the increments are formed alone and added
   by the consumer, the same rounding as MITgcm's ``uFld = uFld + ...``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from defgcm.dynamics._host_constants import host_profile_to_tensor
from defgcm.dynamics._registry import register_dynamics_module
from defgcm.dynamics.base import TendencyModule, TendencyPhase, TendencyStepDomain
from defgcm.dynamics.equation_of_state import (
    JMD95EquationOfState,
    LinearEquationOfState,
    linear_density_anomaly,
)
from defgcm.geometry import StructuredCGrid
from defgcm.geometry.grid import _compute_interior_mask
from defgcm.geometry.exchange import FillSpec, HaloExtentV1, exchange
from defgcm.state import DiagnosticState, PrognosticState, StateTendency

from .calc_tensor import GMRediParameters, gmredi_calc_tensor
from .grad_sigma import grad_sigma
from .bolus import residual_flow
from .levels import previous_level
from .transport import (
    add_r_offdiagonal_flux,
    add_x_diagonal_flux,
    add_x_offdiagonal_flux,
    add_y_diagonal_flux,
    add_y_offdiagonal_flux,
)

# Halo width: the tensor's inner range [2-OL : s+OL-1] then reaches the east and
# north faces of the interior, the widest point the fluxes read.
_OL = 2
_RK_SIGN = -1.0


def _recip(x: torch.Tensor) -> torch.Tensor:
    nonzero = x != 0
    return torch.where(nonzero, 1.0 / torch.where(nonzero, x, torch.ones_like(x)), torch.zeros_like(x))


@dataclass
@register_dynamics_module("gmredi")
class GMRedi(TendencyModule):
    """``data.gmredi`` (after MITgcm's defaults, ``gmredi_readparms.F:110-125,223``) and CPP switches."""

    taper_scheme: str
    adv_form: bool
    isopyc_k: float
    background_k: float
    max_slope: float
    kmin_horiz: float
    small_number: float
    slope_sq_cutoff: float
    s_crit: float
    s_d: float
    non_unity_diagonal: bool
    extra_diagonal: bool
    exclude_tapering: bool
    # GM_Visbeck_* after MITgcm's defaults and GM_isoFac_calcK (gmredi_readparms.F:125,150-157,225-227).
    visbeck_alpha: float
    visbeck_length: float
    visbeck_depth: float
    visbeck_min_depth: float
    visbeck_max_slope: float
    visbeck_min_val_k: float
    visbeck_max_val_k: float
    iso_fac_calc_k: float
    # gravity and rhoConst of data PARM01, read by the Visbeck N^2 (gmredi_calc_tensor.F:488).
    gravity: float
    rho_const: float
    # The stepped tracers (tempStepping, saltStepping): MITgcm adds the GM fluxes
    # inside TEMP_INTEGRATE / SALT_INTEGRATE, each called only for its stepped
    # tracer (thermodynamics.F:317-337).
    tracer_names: tuple[str, ...]
    # fCori at cell centres, one value per row (ny), MITgcm ini_cori.F; read only
    # by ldd97 (gmredi_calc_tensor.F:163-211), None declaring it is not read.
    coriolis_profile_c: tuple[float, ...] | None
    name: str = "gmredi"
    stepper_domain: TendencyStepDomain = TendencyStepDomain.SLOW
    tendency_phase: TendencyPhase = TendencyPhase.TRACER
    # The model's equation-of-state module, bound by the builder.
    equation_of_state: TendencyModule | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.parameters = GMRediParameters(
            taper_scheme=self.taper_scheme,
            adv_form=self.adv_form,
            isopyc_k=self.isopyc_k,
            background_k=self.background_k,
            max_slope=self.max_slope,
            kmin_horiz=self.kmin_horiz,
            small_number=self.small_number,
            slope_sq_cutoff=self.slope_sq_cutoff,
            s_crit=self.s_crit,
            s_d=self.s_d,
            non_unity_diagonal=self.non_unity_diagonal,
            extra_diagonal=self.extra_diagonal,
            exclude_tapering=self.exclude_tapering,
            visbeck_alpha=self.visbeck_alpha,
            visbeck_length=self.visbeck_length,
            visbeck_depth=self.visbeck_depth,
            visbeck_min_depth=self.visbeck_min_depth,
            visbeck_max_slope=self.visbeck_max_slope,
            visbeck_min_val_k=self.visbeck_min_val_k,
            visbeck_max_val_k=self.visbeck_max_val_k,
            iso_fac_calc_k=self.iso_fac_calc_k,
        )
        unknown = set(self.tracer_names) - {"temp", "salt"}
        if unknown:
            raise ValueError(
                f"gmredi: tracer_names {sorted(unknown)} are not ported; GM/Redi acts on temp and salt only"
            )
        if (self.taper_scheme == "ldd97") != (self.coriolis_profile_c is not None):
            raise ValueError(
                f"gmredi: coriolis_profile_c is read by ldd97 only (gmredi_calc_tensor.F:163-211); "
                f"GM_taper_scheme={self.taper_scheme!r} with "
                f"coriolis_profile_c {'unset' if self.coriolis_profile_c is None else 'set'}"
            )

    def _densities(
        self, temp: torch.Tensor, salt: torch.Tensor, state: PrognosticState,
        diagnostics: DiagnosticState, grid: StructuredCGrid,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(rho_in_situ, rho_upper)`` on the interior, ``do_oceanic_phys.F:759-823``."""
        eos = self.equation_of_state
        temp_up, salt_up = previous_level(temp), previous_level(salt)
        if isinstance(eos, LinearEquationOfState):
            # find_rho.F LINEAR branch: tRef(kRef), sRef(kRef) of the row's level.
            def rho_of(t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
                # The anomaly w.r.t. rhoNil: the module holds no rhoConst, and only
                # vertical and horizontal differences of it are used.
                return linear_density_anomaly(
                    t, s,
                    reference_density=eos.reference_density,
                    rho_const=eos.reference_density,
                    thermal_expansion=eos.thermal_expansion,
                    haline_contraction=eos.haline_contraction,
                    reference_temperature=eos.reference_temperature,
                    reference_salinity=eos.reference_salinity,
                )
            return rho_of(temp, salt), rho_of(temp_up, salt_up)
        elif isinstance(eos, JMD95EquationOfState):
            # PRESSURE_FOR_EOS at kRef: the model EOS pressure of the row's level.
            inputs = diagnostics.inputs
            (pressure,) = eos._pressure_field(
                grid, dtype=temp.dtype, device=temp.device, target=state.temp.data,
                prev_total_phi_hyd=None if inputs is None else inputs.total_phi_hyd,
            )
            return (
                eos.density_anomaly_at_pressure(temp, salt, pressure),
                eos.density_anomaly_at_pressure(temp_up, salt_up, pressure),
            )

    def populate_diagnostic_state(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
    ) -> None:
        """Tensor, ``gm_kwz`` and the stepped tracers' GM tendencies (``gm_tendency_temp``,
        ``gm_tendency_salt``), before any tracer module evaluates: the advective form's
        residual velocity comes from the same stream functions and the tracer advection
        reads it (``thermodynamics.F:269-276``)."""
        p = self.parameters
        (temp,) = state.temp.data
        (salt,) = state.salt.data
        dtype, device = temp.dtype, temp.device
        nx, ny = grid.nx, grid.ny
        extent = HaloExtentV1.uniform(_OL)

        def halo(f: torch.Tensor, location: str, *, constant: bool = False) -> torch.Tensor:
            return exchange(f, grid=grid, location=location, extent=extent,
                            fill=FillSpec.REPLICATE, constant=constant)

        def metric(f: torch.Tensor, location: str) -> torch.Tensor:
            return halo(f, location, constant=not f.requires_grad)

        rho, rho_up = self._densities(temp, salt, state, diagnostics, grid)
        rho, rho_up = halo(rho, "T"), halo(rho_up, "T")

        hfac_c = grid.masks.hFacC.to(dtype=dtype, device=device)
        hfac_w = metric(grid.hfac_w_for(dtype, device)[..., :nx], "U")
        hfac_s = metric(grid.hfac_s_for(dtype, device)[..., :ny, :], "V")
        mask_c = metric((hfac_c > 0).to(dtype), "T")
        mask_w = (hfac_w > 0).to(dtype)
        mask_s = (hfac_s > 0).to(dtype)
        m = grid.metrics
        recip_dxc = _recip(metric(m.dxC_for(dtype, device)[..., :nx], "U"))
        recip_dyc = _recip(metric(m.dyC_for(dtype, device)[..., :ny, :], "V"))
        dyg = metric(m.dyG_for(dtype, device)[..., :nx], "U")
        dxg = metric(m.dxG_for(dtype, device)[..., :ny, :], "V")
        r_a = metric(m.rA_for(dtype, device), "T")
        mask_in_c = metric(_compute_interior_mask(grid, dtype, device)[0], "T")
        drf = grid.layer_thickness_for(dtype, device).view(-1, 1, 1)
        drc, zc = grid.vertical_coordinate.reference_centres_for(dtype, device)
        drc = drc.view(-1, 1, 1)
        recip_drc = _recip(drc)

        sigma_x, sigma_y, sigma_r = grad_sigma(
            rho, rho_up, maskC=mask_c, maskW=mask_w, maskS=mask_s,
            recip_dxC=recip_dxc, recip_dyC=recip_dyc, recip_drC=recip_drc,
        )
        if p.taper_scheme == "ldd97":
            # ini_cori.F fills the halo from the halo yC; the exchange gives the neighbour's
            # rows, and beyond a wall the replicated edge row, read only by masked faces;
            # a periodic-y wrap that a wet seam face would read is refused by the builder.
            f_cori = metric(
                host_profile_to_tensor(self.coriolis_profile_c, dtype=dtype, device=device)
                .view(ny, 1).expand(ny, nx),
                "T",
            )
        else:
            # gmredi_calc_tensor.F:212-220: the Rossby radii are not read.
            f_cori = torch.zeros_like(r_a)
        tensor = gmredi_calc_tensor(
            p, sigma_x=sigma_x, sigma_y=sigma_y, sigma_r=sigma_r,
            maskC=mask_c, maskW=mask_w, maskS=mask_s,
            f_cori=f_cori,
            r_depth_f=grid.interface_depth.to(dtype=dtype, device=device)[:-1].view(-1, 1, 1),
            r_depth_c=zc.view(-1, 1, 1),
            drc=drc,
            gravity=self.gravity,
            rho_const=self.rho_const,
        )

        # gad_advection.F:300-303: xA = dyG*drF*hFacW, yA = dxG*drF*hFacS.
        x_area = dyg * drf * hfac_w
        y_area = dxg * drf * hfac_s
        inner = (slice(None), slice(_OL, -_OL), slice(_OL, -_OL))
        recip_hfac_c = _recip(hfac_c)
        recip_ra = _recip(r_a[inner[1:]])
        mask_in_inner = mask_in_c[inner[1:]]

        def divergence(tracer: torch.Tensor) -> torch.Tensor:
            zero = torch.zeros_like(tracer)
            dfx = add_x_diagonal_flux(zero, tracer, kux=tensor.kux, xA=x_area, recip_dxC=recip_dxc)
            dfy = add_y_diagonal_flux(zero, tracer, kvy=tensor.kvy, yA=y_area, recip_dyC=recip_dyc)
            if p.extra_diagonal and p.extra_diag:
                dfx = add_x_offdiagonal_flux(dfx, tracer, kuz=tensor.kuz, xA=x_area,
                                             maskC=mask_c, recip_drC=recip_drc)
                dfy = add_y_offdiagonal_flux(dfy, tracer, kvz=tensor.kvz, yA=y_area,
                                             maskC=mask_c, recip_drC=recip_drc)
            dfr = add_r_offdiagonal_flux(
                zero, tracer, kwx=tensor.kwx, kwy=tensor.kwy, maskC=mask_c, maskW=mask_w, maskS=mask_s,
                recip_dxC=recip_dxc, recip_dyC=recip_dyc, rA=r_a, maskInC=mask_in_c,
            )
            # gad_calc_rhs.F:772-781 for the GM fluxes: fVerT(kUp) is the row's own
            # interface, fVerT(kDown) the next one, zero below the bottom cell.
            dfr = dfr[inner]
            dfr_down = torch.cat([dfr[1:], torch.zeros_like(dfr[:1])])
            flux_div = (
                (dfx[:, _OL:-_OL, _OL + 1:-_OL + 1] - dfx[inner]) * mask_in_inner
                + (dfy[:, _OL + 1:-_OL + 1, _OL:-_OL] - dfy[inner]) * mask_in_inner
                + (dfr_down - dfr) * _RK_SIGN
            )
            return -(recip_hfac_c * drf.reciprocal() * recip_ra * flux_div)

        diagnostics.set_tensor("gm_kwz", tensor.kwz[inner])
        if p.adv_form:
            # gmredi_residual_flow.F:58-98 on the haloed arrays, added to zero velocities so
            # the outputs are the increments; the east (north) face sits at halo column
            # (row) _OL + nx (ny), inside the stream functions' inner range.
            zero = torch.zeros_like(tensor.psi_x)
            bolus_u, bolus_v, bolus_w = residual_flow(
                zero, zero, zero, tensor.psi_x, tensor.psi_y,
                recip_drF=drf.reciprocal(), recip_hFacW=_recip(hfac_w), recip_hFacS=_recip(hfac_s),
                dxG=dxg, dyG=dyg, recip_rA=_recip(r_a),
            )
            bolus_w = bolus_w[inner]
            diagnostics.set_tensor(
                "gm_bolus_u", bolus_u[:, _OL:_OL + ny, _OL:_OL + nx + 1].unsqueeze(0))
            diagnostics.set_tensor(
                "gm_bolus_v", bolus_v[:, _OL:_OL + ny + 1, _OL:_OL + nx].unsqueeze(0))
            diagnostics.set_tensor(
                "gm_bolus_w", torch.cat([bolus_w, torch.zeros_like(bolus_w[:1])]).unsqueeze(0))
        if "temp" in self.tracer_names:
            diagnostics.set_tensor("gm_tendency_temp", divergence(halo(temp, "T")).unsqueeze(0))
        if "salt" in self.tracer_names:
            diagnostics.set_tensor("gm_tendency_salt", divergence(halo(salt, "T")).unsqueeze(0))

    def evaluate(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
    ) -> StateTendency:
        """The GM tendencies ``populate_diagnostic_state`` formed for this evaluation."""
        del grid
        temp_tend = torch.zeros_like(state.temp.data)
        salt_tend = torch.zeros_like(state.salt.data)
        if "temp" in self.tracer_names:
            temp_tend = diagnostics.tensors["gm_tendency_temp"]
        if "salt" in self.tracer_names:
            salt_tend = diagnostics.tensors["gm_tendency_salt"]
        return StateTendency(
            eta=torch.zeros_like(state.eta.data),
            u=torch.zeros_like(state.u.data),
            v=torch.zeros_like(state.v.data),
            temp=temp_tend,
            salt=salt_tend,
        )
