"""KPP vertical mixing (Large, McWilliams & Doney 1994) — MITgcm ``pkg/kpp`` port.

Operator-by-operator translation of MITgcm's K-profile parameterization, written
**grid-independent**: every routine operates along the vertical axis (axis 0 = k)
and broadcasts over an arbitrary horizontal shape, so one implementation runs
unchanged across regional/global, lat-lon/cubed-sphere, single/multi-GPU, and any
resolution (cs32 -> higher-res cubed-sphere).  Horizontal-stencil pieces (shear,
121 smoothing) use an injected exchange callback (the "narrow-waist" operator
contract); the columnar pieces here need no horizontal information at all.

Contents: STATEKPP (density / buoyancy-gradient inputs), forcing, shear,
bldepth, wscale, ri_iwmix, blmix/enhance and the KPPMIX assembly.

All sign / index / pressure conventions follow MITgcm (docs/NUMERICS.md); convention
conversions happen at the producer-consumer boundary, not inside these kernels.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from locomgcm.deferred_checks import strict_checks
from locomgcm.dynamics._rounded_product import rounded_multiply
from locomgcm.dynamics.equation_of_state import (
    _MDJWF_DEN,
    _MDJWF_NUM,
    _SI_TO_DBAR,
    _mdjwf_density_and_reciprocal,
    _mdjwf_multiply,
    jmd95_alpha_beta,
    jmd95_density_anomaly,
    jmd95_density_difference,
    linear_density_anomaly,
    mdjwf_density_anomaly,
    reference_pressure_from_layers,
)


def _kpp_wscale_batch_enabled() -> bool:
    """Toggle: hoist the per-level ``kpp_wscale`` table lookups OUT of the
    bldepth ``for kl`` Rib scan and the blmix ``for ki`` loop into ONE batched
    lookup over the level dim (~79 of the 81 wscale calls/face collapse to 2).
    Default ON; ``LOCOMGCM_KPP_WSCALE_BATCH=0`` forces the per-level loop calls
    (A/B + OFF==HEAD verification).  wscale is element-wise in
    (sigma, hbl, ustar, bfsfc) so the batched call (extra leading LEVEL axis) is
    BITWISE vs the per-level loop — the serial kbl threshold scan / Rib
    accumulation / hbl interpolation order is UNCHANGED (only the level-
    independent table lookup is hoisted).  The enable decision is owned by the
    unified control plane (``locomgcm.accel``), not by this module."""
    from locomgcm.accel import gate

    return gate("kpp_wscale_batch").effective_value


def _record_wscale_batch(site: str, variant: str) -> None:
    """Dispatch record for LOCOMGCM_KPP_WSCALE_BATCH.

    Taken immediately after the single batched ``kpp_wscale`` call that
    replaces the per-level loop, i.e. at the accelerated work itself.
    """
    from locomgcm.accel import current_rank, record_dispatch

    record_dispatch(
        "kpp_wscale_batch",
        outcome="hit",
        variant=variant,
        site=site,
        backend="torch",
        rank=current_rank(),
    )


#: Arithmetic of STATEKPP's adjacent-layer density difference behind dbloc.
#: ``"statekpp"``: two in-situ densities at the deeper level's pressure subtracted,
#: as MITgcm STATEKPP subtracts two FIND_RHO_2D results (pkg/kpp/kpp_routines.F).
#: ``"divided_difference"``: the JMD95 polynomial divided difference
#: (:func:`locomgcm.dynamics.equation_of_state.jmd95_density_difference`, JMD95 only),
#: algebraically equal and free of the cancellation, which keeps the near-neutral
#: derivatives accurate.  Where |dbloc| is a few tens of ULP of g*rho/rho the two
#: forms differ by 1e-3 relative.
KPP_DBLOC_FORMS = ("statekpp", "divided_difference")
#: Arithmetic of KPP's surface buoyancy difference dbsfc, the same two forms as
#: ``KPP_DBLOC_FORMS`` applied to the surface reference (dispatch in kpp_calc_diffusivities).
KPP_SURFACE_BUOYANCY_FORMS = ("statekpp", "divided_difference")


def check_kpp_dbloc_form(dbloc_form: str, eos: str | None = None) -> str:
    """Validate an explicit dbloc form; ``eos`` (canonical) also checks the EOS family."""
    if dbloc_form not in KPP_DBLOC_FORMS:
        raise ValueError(f"KPP dbloc_form must be one of {KPP_DBLOC_FORMS}, got {dbloc_form!r}")
    if dbloc_form == "divided_difference" and eos is not None and eos != "jmd95":
        raise ValueError("KPP dbloc_form='divided_difference' supports JMD95 only")
    return dbloc_form


def _canonical_kpp_eos(eos_type: str) -> str:
    eos = str(eos_type).strip().lower()
    if eos in {"jmd95", "jmd95z", "jmd95p"}:
        return "jmd95"
    if eos == "mdjwf":
        return "mdjwf"
    if eos == "linear":
        return "linear"
    raise ValueError(f"KPP does not support equation of state {eos_type!r}")


@dataclass(frozen=True)
class KppLinearEos:
    """Linear equation of state as STATEKPP evaluates it (``eos_type='linear'``).

    ``reference_density`` is rhoNil [kg/m^3]; ``thermal_expansion`` tAlpha
    [1/degC] and ``haline_contraction`` sBeta [1/PSU]; ``reference_temperature``
    and ``reference_salinity`` are tRef/sRef, a scalar or one value per level
    (surface first), indexed by the pressure-reference level kRef
    (``find_rho.F:98-99``).
    """

    reference_density: float | torch.Tensor
    thermal_expansion: float
    haline_contraction: float
    reference_temperature: float | torch.Tensor | tuple
    reference_salinity: float | torch.Tensor | tuple


def mdjwf_alpha_beta(
    temp: torch.Tensor,
    salt: torch.Tensor,
    pressure_pa: torch.Tensor | float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MDJWF ``d(rho)/d(theta)`` and ``d(rho)/d(salt)`` without ``1/rho``.

    This is the analytic MDJWF branch of MITgcm ``FIND_ALPHA`` and
    ``FIND_BETA`` (``model/src/find_alpha.F:222-276,539-586``). Pressure is
    supplied in Pa and converted to dbar, matching ``SItodBar``. As in the
    density kernel, non-float64 inputs are evaluated in float64 and cast back.
    """
    out_dtype = temp.dtype
    if out_dtype is not torch.float64:
        temp = temp.to(torch.float64)
        salt = salt.to(torch.float64)
        if isinstance(pressure_pa, torch.Tensor):
            pressure_pa = pressure_pa.to(torch.float64)

    t1 = temp
    s1 = torch.clamp(salt, min=0.0)
    s_pos = s1 > 0
    s_safe = torch.where(s_pos, s1, torch.ones_like(s1))
    sp5 = torch.where(s_pos, torch.sqrt(s_safe), torch.zeros_like(s1))
    t2 = t1 * t1
    p1 = pressure_pa * _SI_TO_DBAR
    p1t1 = p1 * t1

    n, d = _MDJWF_NUM, _MDJWF_DEN
    mul = _mdjwf_multiply
    rho, rho_den = _mdjwf_density_and_reciprocal(t1, s1, sp5, p1, t2, p1t1)
    dnum_dtheta = (
        n[1]
        + mul(t1, 2.0 * n[2] + mul(3.0 * n[3], t1))
        + mul(n[5], s1)
        + mul(p1t1, 2.0 * n[8] + mul(2.0 * n[11], p1))
    )
    dden_dtheta = (
        d[1]
        + mul(t1, 2.0 * d[2] + mul(t1, 3.0 * d[3] + mul(4.0 * d[4], t1)))
        + mul(s1, d[6] + mul(t1, mul(3.0 * d[7], t1) + mul(2.0 * d[9], sp5)))
        + mul(p1 * p1, mul(3.0 * d[11], t2) + mul(d[12], p1))
    )
    alpha = mul(rho_den, dnum_dtheta - mul(rho, dden_dtheta))

    dnum_dsalt = n[4] + mul(n[5], t1) + mul(2.0 * n[6], s1) + mul(n[9], p1)
    dden_dsalt = (
        d[5]
        + mul(t1, d[6] + mul(d[7], t2))
        + mul(1.5 * sp5, d[8] + mul(d[9], t2))
    )
    beta = mul(rho_den, dnum_dsalt - mul(rho, dden_dsalt))

    if out_dtype is not torch.float64:
        alpha = alpha.to(out_dtype)
        beta = beta.to(out_dtype)
    return alpha, beta


def kpp_statekpp(
    theta: torch.Tensor,
    salt: torch.Tensor,
    layer_pressure_pa: torch.Tensor,
    *,
    gravity: float,
    rho_const: float,
    eos_type: str,
    dbloc_form: str,
    linear_eos: KppLinearEos | None = None,
):
    """STATEKPP: buoyancy gradients + expansion coefficients.

    Ports MITgcm ``STATEKPP`` (``pkg/kpp/kpp_routines.F`` L1766-1958).  Density is
    the in-situ density evaluated **at each layer's own pressure** (potential
    density referenced to layer k), exactly as MITgcm calls ``FIND_RHO_2D`` with
    ``kRef = k``. ``eos_type`` explicitly selects the JMD95 or MDJWF density and
    derivative branch. Unsupported EOS families fail rather than falling back.

    Parameters
    ----------
    theta, salt : Tensor, shape ``(nz, *spatial)``
        Potential temperature [degC] and salinity [PSU] at the current time level.
    layer_pressure_pa : Tensor, shape ``(nz,)`` or broadcastable to ``theta``
        Hydrostatic pressure [Pa] at each layer centre.  For JMD95Z:
        ``rho_const * gravity * |z_center(k)|``.
    gravity : float
        MITgcm ``gravity`` (e.g. 9.81).
    rho_const : float
        MITgcm ``rhoConst`` (e.g. 1035); also the EOS anomaly reference so that
        ``anomaly + rho_const`` is the true in-situ density in the denominators.
    dbloc_form : str
        Required, one of :data:`KPP_DBLOC_FORMS`: ``"statekpp"`` subtracts the two
        densities as MITgcm does; ``"divided_difference"`` (JMD95 only) evaluates the
        same difference as a polynomial divided difference.  dbsfc, rho1, alpha and
        beta do not depend on it.
    linear_eos : KppLinearEos
        rhoNil, tAlpha, sBeta, tRef and sRef of ``eos_type='linear'``, the only
        family that reads it.  Its density is MITgcm's linear FIND_RHO_2D
        anomaly w.r.t. ``rho_const`` (``find_rho.F:92-110``: tRef/sRef of the
        pressure-reference level plus ``rhoNil - rhoConst``); alpha and beta are
        ``-rhoNil*tAlpha`` and ``rhoNil*sBeta`` (``find_alpha.F:75-81,408-414``).

    Returns
    -------
    dbloc : Tensor ``(nz, *spatial)``
        Local buoyancy gradient at the interface below layer k:
        ``dbloc[k] = g*(rho_{k+1} - rho_{k @ p_{k+1}}) / rho_{k+1}`` for
        ``k = 0..nz-2``; ``dbloc[nz-1] = 0``.  (MITgcm stores this at ``DBLOC(k)``;
        masking/121-smoothing are applied by the caller, not here.)
    dbsfc : Tensor ``(nz, *spatial)``
        Buoyancy difference w.r.t. the surface layer:
        ``dbsfc[k] = g*(rho_k - rho_{0 @ p_k}) / rho_k``; ``dbsfc[0] = 0``.
        (MITgcm stores this in ``Ritop`` before the ``(zgrid(1)-zgrid(k))`` scale.)
    rho1 : Tensor ``(*spatial)``
        Surface-layer in-situ density [kg/m^3].
    alpha, beta : Tensor ``(nz+1, *spatial)``
        ``d(rho)/d(theta)`` and ``d(rho)/d(salt)`` (no 1/rho factor) at each layer's
        own pressure; the trailing virtual layer (Nrp1) duplicates layer Nr, as in
        MITgcm.  alpha is usually negative but can be positive in cold,
        low-salinity water; beta is positive in the supported ocean regime.
    """
    # Reduced-precision guardrail (the buoyancy half of the EOS float64 island).
    # ``jmd95_density_anomaly`` already evaluates its polynomials in float64 for
    # a non-float64 input, but it casts the ANOMALY back before returning, and
    # dbloc/dbsfc are differences of two such anomalies.  In the weakly
    # stratified water where KPP decides the boundary-layer depth those
    # anomalies are of order 10 kg/m^3 and their difference five decades
    # smaller, so narrowing before the subtraction costs 1.6e-02 relative on
    # dbloc (a strongly stratified column hides it at 1.3e-06).  Evaluating
    # STATEKPP in float64 and casting the five outputs back at the end keeps
    # the difference in float64 and costs one widening of theta/salt/p.
    # dbloc_form="divided_difference" additionally evaluates the JMD95
    # adjacent-layer difference as a divided difference, which keeps
    # near-neutral contrasts and their derivatives within that island.
    _out_dtype = theta.dtype
    if _out_dtype is not torch.float64:
        theta = theta.to(torch.float64)
        salt = salt.to(torch.float64)
        layer_pressure_pa = layer_pressure_pa.to(torch.float64)
    nz = theta.shape[-3]
    p = layer_pressure_pa
    if p.ndim == 1:  # (nz,) -> (nz, 1, 1): aligns with the vertical axis (-3) of theta,
        p = p.reshape(nz, 1, 1)  # broadcasting over any optional leading face-batch dim

    eos = _canonical_kpp_eos(eos_type)
    check_kpp_dbloc_form(dbloc_form, eos)
    # ``levels`` selects the pressure-reference levels kRef of the evaluated
    # layers: all of them, or 1.. for the layer above evaluated at the level below.
    if eos == "linear":
        rho_nil = torch.as_tensor(linear_eos.reference_density, dtype=theta.dtype, device=theta.device)
        t_ref = torch.as_tensor(linear_eos.reference_temperature, dtype=theta.dtype, device=theta.device)
        s_ref = torch.as_tensor(linear_eos.reference_salinity, dtype=theta.dtype, device=theta.device)

        def rho(t, s, levels):
            # FIND_RHO_2D LINEAR: anomaly w.r.t. rhoConst at reference level kRef.
            return linear_density_anomaly(
                t, s, reference_density=rho_nil, rho_const=rho_const,
                thermal_expansion=linear_eos.thermal_expansion,
                haline_contraction=linear_eos.haline_contraction,
                reference_temperature=t_ref if t_ref.ndim == 0 else t_ref[levels],
                reference_salinity=s_ref if s_ref.ndim == 0 else s_ref[levels],
            )
    elif eos == "jmd95" or eos == "mdjwf":
        density_kernel = jmd95_density_anomaly if eos == "jmd95" else mdjwf_density_anomaly

        def rho(t, s, levels):
            # in-situ density anomaly w.r.t. rho_const; the reference cancels in every
            # numerator (density difference) and (anomaly + rho_const) = true density
            # in every denominator -> matches MITgcm STATEKPP exactly.
            return density_kernel(t, s, p[..., levels, :, :], reference_density=rho_const)

    # vertical axis is -3 (trailing spatial = (NY, NX)); ``...`` slicing passes an
    # optional leading face-batch dim through every step bit-for-bit identically.
    rho_diag = rho(theta, salt, slice(None))       # rho_k (layer k T/S at p_k)
    rho_surf = rho(theta[..., :1, :, :].expand_as(theta),
                   salt[..., :1, :, :].expand_as(salt), slice(None))

    dbloc = torch.zeros_like(theta)
    if dbloc_form == "divided_difference":
        adjacent_difference = jmd95_density_difference(
            theta[..., 1:, :, :], salt[..., 1:, :, :],
            theta[..., :-1, :, :], salt[..., :-1, :, :], p[..., 1:, :, :],
        )
    else:
        rho_upper = rho(theta[..., :-1, :, :], salt[..., :-1, :, :], slice(1, None))
        adjacent_difference = rho_diag[..., 1:, :, :] - rho_upper
    dbloc[..., :-1, :, :] = gravity * adjacent_difference / (rho_diag[..., 1:, :, :] + rho_const)
    # dbloc[-1] stays 0  (MITgcm DBLOC(Nr) = 0)

    dbsfc = gravity * (rho_diag - rho_surf) / (rho_diag + rho_const)
    dbsfc[..., 0, :, :] = 0.0                       # MITgcm DBSFC(1) = 0

    rho1 = rho_diag[..., 0, :, :] + rho_const

    if eos == "linear":
        alpha_nr = (-(rho_nil * linear_eos.thermal_expansion)).expand_as(theta)
        beta_nr = (rho_nil * linear_eos.haline_contraction).expand_as(theta)
    elif eos == "jmd95":
        alpha_nr, beta_nr = jmd95_alpha_beta(theta, salt, p)
    elif eos == "mdjwf":
        alpha_nr, beta_nr = mdjwf_alpha_beta(theta, salt, p)
    alpha = torch.cat([alpha_nr, alpha_nr[..., -1:, :, :]], dim=-3)   # Nrp1 virtual layer = Nr
    beta = torch.cat([beta_nr, beta_nr[..., -1:, :, :]], dim=-3)

    if _out_dtype is not torch.float64:
        return (
            dbloc.to(_out_dtype),
            dbsfc.to(_out_dtype),
            rho1.to(_out_dtype),
            alpha.to(_out_dtype),
            beta.to(_out_dtype),
        )
    return dbloc, dbsfc, rho1, alpha, beta


def kpp_double_diffusion(
    theta: torch.Tensor,
    salt: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    kappa_t: torch.Tensor,
    kappa_s: torch.Tensor,
    *,
    Rrho0: float,
    dsfmax: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """KPP_DOUBLEDIFF: double-diffusive diffusivities added to the background.

    Ports MITgcm ``KPP_DOUBLEDIFF`` (``pkg/kpp/kpp_routines.F`` L1962-2088,
    Large et al. 1994 eqs. 31-34), which ``kpp_calc.F`` L542-553 applies to
    ``KPPdiffKzT``/``KPPdiffKzS`` before KPPMIX reads them.

    ``theta``/``salt`` ``(*lead, nz, NY, NX)`` are the state STATEKPP read;
    ``alpha``/``beta`` ``(*lead, nz+1, NY, NX)`` its TTALPHA/SSBETA
    (``d(rho)/d(theta)``, ``d(rho)/d(salt)``, no 1/rho factor);
    ``kappa_t``/``kappa_s`` ``(*lead, nz, NY, NX)`` the background
    diffusivities [m^2/s], row k at the top interface of layer k.  Row k >= 1
    gains the salt-fingering or diffusive-convection term of the interface
    between layers k-1 and k; the surface row is unchanged.  No mask: dry rows
    are zeroed later by KPPMIX and the KappaR cell mask, as in MITgcm.
    """
    numol = 1.5e-06
    rfac = 1.0 / (Rrho0 - 1.0)
    nz = theta.shape[-3]
    up = (..., slice(0, nz - 1), slice(None), slice(None))
    dn = (..., slice(1, nz), slice(None), slice(None))
    alpha_dt = (theta[up] - theta[dn]) * 0.5 * torch.abs(alpha[up] + alpha[dn])
    beta_ds = (salt[up] - salt[dn]) * 0.5 * (beta[up] + beta[dn])
    finger = (alpha_dt > beta_ds) & (beta_ds > 0.0)
    diffusive = (~finger) & (alpha_dt < 0.0) & (beta_ds < 0.0) & (alpha_dt > beta_ds)
    one = torch.ones_like(beta_ds)
    zero = torch.zeros_like(beta_ds)
    # salt fingering (salinity destabilizes), eqs. 31a and 31c
    rrho = torch.clamp(alpha_dt / torch.where(finger, beta_ds, one), max=Rrho0)
    nutmp = 1.0 - (rrho - 1.0) * rfac
    nudds_finger = dsfmax * nutmp * nutmp * nutmp
    nuddt_finger = 0.7 * nudds_finger
    # diffusive convection (temperature destabilizes), eqs. 32 and 34
    rrho = torch.where(diffusive, alpha_dt, one) / torch.where(diffusive, beta_ds, one)
    nuddt_diffusive = numol * 0.909 * torch.exp(4.6 * torch.exp(-5.4 * (1.0 / rrho - 1.0)))
    nudds_diffusive = nuddt_diffusive * torch.maximum(0.15 * rrho, 1.85 * rrho - 0.85)
    nuddt = torch.where(finger, nuddt_finger, torch.where(diffusive, nuddt_diffusive, zero))
    nudds = torch.where(finger, nudds_finger, torch.where(diffusive, nudds_diffusive, zero))
    kappa_t = torch.cat((kappa_t[..., :1, :, :], kappa_t[..., 1:, :, :] + nuddt), dim=-3)
    kappa_s = torch.cat((kappa_s[..., :1, :, :], kappa_s[..., 1:, :, :] + nudds), dim=-3)
    return kappa_t, kappa_s


def jmd95z_layer_pressure(
    layer_thickness: torch.Tensor, *, gravity: float, rho_const: float
) -> torch.Tensor:
    """Return the static pressure profile at MITgcm's stored cell centers in Pa."""
    return reference_pressure_from_layers(layer_thickness, rho_const=rho_const, gravity=gravity)


# ──────────────────────────────────────────────────────────────────────
# surface forcing + velocity shear + horizontal 121 smoothing.
# All operate on a haloed C-grid slab (..., NY, NX) and read horizontal
# neighbours by array shift; the halo is supplied by the caller (the dumped
# tile slab for op-level, or the exchange callback for multi-tile/cube). The
# interior result region is slab [1:NY-1, 1:NX-1] (reads ±1); outer ring is
# left zero/unfilled (unused). uVel[..,i+1]=east u-face, vVel[..,j+1]=north
# v-face of t-cell (i,j) (MITgcm C-grid). No sign flips on u/v/surfForc.
# ──────────────────────────────────────────────────────────────────────


def kpp_surface_buoyancy_divided_difference(theta, salt, pressure, *, gravity, rho_const):
    """Surface-relative buoyancy g*(rho(T1,S1,p_k)-rho(T_k,S_k,p_k))/rho(T_k,S_k,p_k).

    The same quantity as STATEKPP's dbsfc, evaluated as the JMD95 density
    difference at the common pressure of level k (one rounding of the
    difference instead of two densities). Level 0 is exactly zero.
    ``theta/salt`` ``(..., nz, NY, NX)``; ``pressure`` the same or ``(nz,)``.
    """
    from locomgcm.dynamics import equation_of_state as eos

    dtype = theta.dtype
    t, s, p = theta.double(), salt.double(), pressure.double()
    if p.ndim == 1:
        p = p.reshape(-1, 1, 1)
    denominator = eos.jmd95_density_anomaly(t, s, p, reference_density=rho_const) + rho_const
    difference = eos.jmd95_density_difference(
        t, s, t[..., 0:1, :, :].expand_as(t), s[..., 0:1, :, :].expand_as(s), p)
    result = gravity * difference / denominator
    result = torch.cat((torch.zeros_like(result[..., 0:1, :, :]), result[..., 1:, :, :]), dim=-3)
    return result.to(dtype)


def kpp_surface_friction_velocity(
    surf_forc_u: torch.Tensor,
    surf_forc_v: torch.Tensor,
    *,
    layer1_thickness: float,
    phepsi: float = 1.0e-10,
) -> torch.Tensor:
    """Friction velocity ustar [m/s] (MITgcm kpp_forcing_surf.F:187-211).

    ``work3 = (sfU(i,j)+sfU(i+1,j))^2 + (sfV(i,j)+sfV(i,j+1))^2``;
    ``ustar = sqrt(0.5*sqrt(work3))`` with a small-shear floor ``epsLoc``.
    ``surf_forc_{u,v}`` = surface wind stress / rho on the C-grid (..., NY, NX).
    ``layer1_thickness`` = drF(1) [m].
    """
    NY, NX = surf_forc_u.shape[-2:]
    sj, si = slice(1, NY - 1), slice(1, NX - 1)
    u_c = surf_forc_u[..., sj, si]
    u_ip1 = surf_forc_u[..., sj, 2:NX]
    v_c = surf_forc_v[..., sj, si]
    v_jp1 = surf_forc_v[..., 2:NY, si]
    work3 = (u_c + u_ip1) ** 2 + (v_c + v_jp1) ** 2
    eps_sq = phepsi * phepsi * layer1_thickness * layer1_thickness
    eps_loc = (0.5 * phepsi * layer1_thickness) ** 0.5
    below_floor = work3 < eps_sq
    # Protect the unselected sqrt branch: zero stress otherwise yields 0/0
    # in its pullback even though the outer where selects the finite floor.
    work3_safe = torch.where(below_floor, torch.ones_like(work3), work3)
    out = torch.zeros_like(surf_forc_u)
    out[..., sj, si] = torch.where(
        below_floor,
        torch.full_like(work3, eps_loc),
        torch.sqrt(torch.sqrt(work3_safe) * 0.5),
    )
    return out


def kpp_relative_velocity_shear(u_vel: torch.Tensor, v_vel: torch.Tensor) -> torch.Tensor:
    """dVsq [m^2/s^2]: velocity shear of each level vs the surface, on t-grid.

    MITgcm kpp_forcing_surf.F:472-480 (default branch; KPP_ESTIMATE_UREF /
    KPP_SMOOTH_DVSQ off). ``dVsq[k] = 0.5*[ (u0-uk)^2|_(i) + (u0-uk)^2|_(i+1)
    + (v0-vk)^2|_(j) + (v0-vk)^2|_(j+1) ]``; ``dVsq[0]=0``. u/v haloed
    ``(*lead, nz, NY, NX)`` (optional leading face-batch dim).
    """
    NY, NX = u_vel.shape[-2:]
    sj, si = slice(1, NY - 1), slice(1, NX - 1)
    du = u_vel[..., 0:1, :, :] - u_vel  # u_surf - u_k  (..., nz, NY, NX)
    dv = v_vel[..., 0:1, :, :] - v_vel
    out = torch.zeros_like(u_vel)
    out[..., sj, si] = 0.5 * (
        du[..., sj, si] ** 2
        + du[..., sj, 2:NX] ** 2      # i+1 (east u-face)
        + dv[..., sj, si] ** 2
        + dv[..., 2:NY, si] ** 2      # j+1 (north v-face)
    )
    return out


def kpp_relative_velocity_shear_estimate_uref(
    u_vel: torch.Tensor,
    v_vel: torch.Tensor,
    dbloc: torch.Tensor,
    ustar: torch.Tensor,
    surf_forc_u: torch.Tensor,
    surf_forc_v: torch.Tensor,
    layer_thickness: torch.Tensor,
    maskC: torch.Tensor,
    nzmax: torch.Tensor,
    *,
    vonk: float,
    epsilon: float,
    phepsi: float = 1.0e-10,
    epsln: float = 1.0e-20,
    dB_dz: float = 5.2e-5,
) -> torch.Tensor:
    """dVsq with the resolution-independent KPP reference velocity estimate.

    Ports MITgcm ``KPP_ESTIMATE_UREF``
    (``pkg/kpp/kpp_forcing_surf.F:309-461``). Inputs use the haloed C-grid
    layout ``(*lead, nz, NY, NX)``: U is the west face at ``i`` and V the
    south face at ``j``. ``dbloc`` is the buoyancy difference at the bottom of
    each cell, ``maskC`` is the wet-cell mask, and ``nzmax`` is the one-based
    deepest-wet level. ``layer_thickness`` is positive-down ``drF``. MITgcm's
    negative-down interfaces ``rF`` and the positive reciprocals
    ``recip_drF``/``recip_drC`` are derived from it in the source operation
    order; the resulting reference depths remain positive-down.

    MITgcm's depth-average ``DO WHILE`` has no ``k <= Nr`` guard. Valid inputs
    exit above the bottom interface; if they do not, this port raises instead
    of indexing beyond the allocated vertical arrays.
    """
    nz = u_vel.shape[-3]
    if nz < 2:
        raise ValueError("KPP_ESTIMATE_UREF requires at least two vertical levels")

    NY, NX = u_vel.shape[-2:]
    sj, si = slice(1, NY - 1), slice(1, NX - 1)
    drF = layer_thickness.to(dtype=u_vel.dtype, device=u_vel.device)
    p0 = u_vel.new_tensor(0.0)
    p1 = u_vel.new_tensor(1.0)
    p5 = u_vel.new_tensor(0.5)
    vonk_t = u_vel.new_tensor(vonk)
    epsilon_t = u_vel.new_tensor(epsilon)
    phepsi_t = u_vel.new_tensor(phepsi)
    epsln_t = u_vel.new_tensor(epsln)
    dB_dz_t = u_vel.new_tensor(dB_dz)

    rF_values = [p0]
    for k in range(nz):
        rF_values.append(rF_values[-1] + (-p1) * drF[k])
    rF = torch.stack(rF_values)
    recip_drF = p1 / drF
    drC_values = [p5 * drF[0]]
    for k in range(1, nz):
        drC_values.append(p5 * (drF[k - 1] + drF[k]))
    drC_values.append(p5 * drF[nz - 1])
    recip_drC = p1 / torch.stack(drC_values)

    zFac = torch.abs(rF[2]) * torch.log(rF[2] / rF[1]) * recip_drF[1]
    dbloc_c = dbloc[..., :, sj, si]
    mask_c = maskC[..., :, sj, si]
    nzmax_c = nzmax[..., sj, si].long()

    kTmp = nzmax_c.clone()
    for k_one_based in range(nz, 0, -1):
        gradient = dbloc_c[..., k_one_based - 1, :, :] * recip_drC[k_one_based]
        select = (
            (k_one_based < nzmax_c)
            & (mask_c[..., k_one_based - 1, :, :] > p0)
            & (gradient > dB_dz_t)
        )
        kTmp = torch.where(select, k_one_based, kTmp)

    zRef = torch.zeros_like(ustar[..., sj, si])
    zero_case = (kTmp == 0) | (nzmax_c == 1)
    one_case = (~zero_case) & (kTmp == 1)
    interior_case = (~zero_case) & (~one_case) & (kTmp < nzmax_c)
    bottom_case = ~(zero_case | one_case | interior_case)

    dBdz2 = dbloc_c[..., 0, :, :] * recip_drC[1]
    dBdz2_safe = torch.where(one_case, dBdz2, torch.ones_like(dBdz2))
    zRef_one = drF[0] * dB_dz_t / dBdz2_safe
    zRef = torch.where(one_case, zRef_one, zRef)

    km2 = torch.clamp(kTmp - 2, 0, nz - 1)
    km1 = torch.clamp(kTmp - 1, 0, nz - 1)
    dBdz1 = _kpp_gather_k(dbloc_c, km2) * recip_drC[
        torch.clamp(kTmp - 1, 0, nz)
    ]
    dBdz2 = _kpp_gather_k(dbloc_c, km1) * recip_drC[
        torch.clamp(kTmp, 0, nz)
    ]
    denominator = torch.maximum(phepsi_t, dBdz2 - dBdz1)
    zRef_interior = torch.abs(rF[torch.clamp(kTmp - 1, 0, nz)])
    zRef_interior = zRef_interior + drF[km1] * (dB_dz_t - dBdz1) / denominator
    zRef = torch.where(interior_case, zRef_interior, zRef)

    bottom_depth = torch.abs(rF[torch.clamp(kTmp, 0, nz)])
    zRef = torch.where(bottom_case, bottom_depth, zRef)

    u_w = u_vel[..., :, sj, si]
    u_e = u_vel[..., :, sj, 2:NX]
    v_s = v_vel[..., :, sj, si]
    v_n = v_vel[..., :, 2:NY, si]
    du_w = u_w[..., 0, :, :] - u_w[..., 1, :, :]
    du_e = u_e[..., 0, :, :] - u_e[..., 1, :, :]
    dv_s = v_s[..., 0, :, :] - v_s[..., 1, :, :]
    dv_n = v_n[..., 0, :, :] - v_n[..., 1, :, :]
    tempVar1 = du_w * du_w
    tempVar1 = tempVar1 + du_e * du_e
    tempVar1 = tempVar1 + dv_s * dv_s
    tempVar1 = tempVar1 + dv_n * dv_n
    tempVar1 = p5 * tempVar1
    # MITgcm's IF/ELSE evaluates SQRT only on the lanes it takes; a where
    # evaluates both.  On a dry column tempVar1 is exactly 0, where the floor
    # is taken and d sqrt/dx is infinite, so the zero cotangent of the
    # unselected branch would become 0*inf = NaN.  The floor lanes take the
    # square root of 1 instead; the selected lanes see the same argument.
    below = tempVar1 < epsln_t * epsln_t
    tempVar2 = torch.where(
        below,
        torch.full_like(tempVar1, epsln),
        torch.sqrt(torch.where(below, torch.ones_like(tempVar1), tempVar1)),
    )

    ustar_c = ustar[..., sj, si]
    z0 = drF[0] * (zFac - tempVar2 * vonk_t / ustar_c)
    z0 = torch.maximum(z0, phepsi_t)
    zRef = torch.maximum(epsilon_t * zRef, z0)

    uRef = p5 * (u_w[..., 0, :, :] + u_e[..., 0, :, :])
    vRef = p5 * (v_s[..., 0, :, :] + v_n[..., 0, :, :])
    shallow = zRef < drF[0]
    deep = zRef >= drF[0]

    ustarX = (surf_forc_u[..., sj, si] + surf_forc_u[..., sj, 2:NX]) * p5
    ustarX = ustarX * recip_drF[0]
    ustarY = (surf_forc_v[..., sj, si] + surf_forc_v[..., 2:NY, si]) * p5
    ustarY = ustarY * recip_drF[0]
    tempVar1 = ustarX * ustarX + ustarY * ustarY
    # The same floor for a zero surface stress (dry columns, calm cells): it
    # matters once the stress is a differentiated input.
    below = tempVar1 < epsln_t * epsln_t
    tempVar2 = torch.where(
        below,
        torch.full_like(tempVar1, epsln),
        torch.sqrt(torch.where(below, torch.ones_like(tempVar1), tempVar1)),
    )
    profile = torch.log(zRef * recip_drF[0])
    profile = profile + z0 / zRef
    profile = profile - z0 * recip_drF[0]
    correction = ustar_c * profile
    correction = correction / vonk_t
    correction = correction / tempVar2
    uRef = torch.where(shallow, uRef + ustarX * correction, uRef)
    vRef = torch.where(shallow, vRef + ustarY * correction, vRef)

    weighted_u = uRef * drF[0]
    weighted_v = vRef * drF[0]
    exit_layer = torch.ones_like(kTmp)
    still_accumulating = deep.clone()
    for layer in range(1, nz):
        full_layer = still_accumulating & (torch.abs(rF[layer + 1]) <= zRef)
        u_term = drF[layer] * p5
        u_term = u_term * (u_w[..., layer, :, :] + u_e[..., layer, :, :])
        v_term = drF[layer] * p5
        v_term = v_term * (v_s[..., layer, :, :] + v_n[..., layer, :, :])
        weighted_u = weighted_u + torch.where(
            full_layer, u_term, torch.zeros_like(u_term)
        )
        weighted_v = weighted_v + torch.where(
            full_layer, v_term, torch.zeros_like(v_term)
        )
        exit_layer = torch.where(full_layer, layer + 1, exit_layer)
        still_accumulating = full_layer
    # **Strict level only**: a whole-field reduction and a host branch inside
    # the traced step. ``still_accumulating``
    # is consumed by the loop above, so only the verdict is dropped here.
    if strict_checks() and torch.any(still_accumulating):
        raise ValueError("zRef reaches the bottom interface; MITgcm loop would overrun")

    u_last = _kpp_gather_k(u_w, exit_layer)
    u_last_e = _kpp_gather_k(u_e, exit_layer)
    v_last = _kpp_gather_k(v_s, exit_layer)
    v_last_n = _kpp_gather_k(v_n, exit_layer)
    partial = torch.maximum(p0, zRef - torch.abs(rF[exit_layer]))
    u_term = partial * p5
    u_term = u_term * (u_last + u_last_e)
    v_term = partial * p5
    v_term = v_term * (v_last + v_last_n)
    weighted_u = weighted_u + torch.where(deep, u_term, torch.zeros_like(u_term))
    weighted_v = weighted_v + torch.where(deep, v_term, torch.zeros_like(v_term))
    uRef = torch.where(deep, weighted_u / zRef, uRef)
    vRef = torch.where(deep, weighted_v / zRef, vRef)

    du_w = uRef.unsqueeze(-3) - u_w
    du_e = uRef.unsqueeze(-3) - u_e
    dv_s = vRef.unsqueeze(-3) - v_s
    dv_n = vRef.unsqueeze(-3) - v_n
    term = du_w * du_w
    term = term + du_e * du_e
    term = term + dv_s * dv_s
    term = term + dv_n * dv_n
    out = torch.zeros_like(u_vel)
    out[..., sj, si] = p5 * term
    return out


def kpp_local_velocity_shear(
    u_vel: torch.Tensor, v_vel: torch.Tensor, *, smooth: bool = True
) -> torch.Tensor:
    """shsq [m^2/s^2]: local interface velocity shear squared, on t-grid.

    MITgcm kpp_calc.F:454-492. Base (C-grid 4-term avg of (u_k-u_{k+1})^2 etc.);
    with ``smooth`` (KPP_SMOOTH_SHSQ, default ON) adds the 8-neighbour 121 term:
    ``shsq = 0.25*base + 0.125*(8 neighbour shears)``. ``shsq[nz-1]=0``.
    u/v haloed ``(*lead, nz, NY, NX)`` (optional leading face-batch dim).
    """
    NY, NX = u_vel.shape[-2:]
    sj, si = slice(1, NY - 1), slice(1, NX - 1)
    du = u_vel[..., :-1, :, :] - u_vel[..., 1:, :, :]  # u_k - u_{k+1}  (..., nz-1, NY, NX)
    dv = v_vel[..., :-1, :, :] - v_vel[..., 1:, :, :]
    base = (
        du[..., sj, si] ** 2
        + du[..., sj, 2:NX] ** 2      # (i+1, j)
        + dv[..., sj, si] ** 2
        + dv[..., 2:NY, si] ** 2      # (i, j+1)
    )
    out = torch.zeros_like(u_vel)
    if not smooth:
        out[..., :-1, sj, si] = 0.5 * base
        return out
    extra = (
        du[..., 0:NY - 2, si] ** 2    # (i,   j-1)
        + du[..., 0:NY - 2, 2:NX] ** 2  # (i+1, j-1)
        + du[..., 2:NY, si] ** 2      # (i,   j+1)
        + du[..., 2:NY, 2:NX] ** 2    # (i+1, j+1)
        + dv[..., sj, 0:NX - 2] ** 2  # (i-1, j)
        + dv[..., 2:NY, 0:NX - 2] ** 2  # (i-1, j+1)
        + dv[..., sj, 2:NX] ** 2      # (i+1, j)
        + dv[..., 2:NY, 2:NX] ** 2    # (i+1, j+1)
    )
    out[..., :-1, sj, si] = 0.5 * (0.5 * base) + 0.125 * extra
    return out


def kpp_smooth_horiz(fld: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mask-weighted horizontal 121 (3x3) smoothing (MITgcm kpp_routines.F:1311-1391).

    ``fld``, ``mask`` haloed (..., NY, NX) (mask = maskC at the relevant level).
    Center weight 0.25, edge 0.125, corner 0.0625, renormalised by the masked
    weight sum W; where W < 0.25 the value is left unchanged. Outer ring keeps fld.
    """
    NY, NX = fld.shape[-2:]
    sj, si = slice(1, NY - 1), slice(1, NX - 1)
    m_c = mask[..., sj, si]
    m_im, m_ip = mask[..., sj, 0:NX - 2], mask[..., sj, 2:NX]
    m_jm, m_jp = mask[..., 0:NY - 2, si], mask[..., 2:NY, si]
    m_mm, m_mp = mask[..., 0:NY - 2, 0:NX - 2], mask[..., 2:NY, 0:NX - 2]
    m_pm, m_pp = mask[..., 0:NY - 2, 2:NX], mask[..., 2:NY, 2:NX]
    W = (0.25 * m_c + 0.125 * (m_im + m_ip + m_jm + m_jp)
         + 0.0625 * (m_mm + m_mp + m_pm + m_pp))
    f_c = fld[..., sj, si]
    num = (
        0.25 * f_c * m_c
        + 0.125 * (fld[..., sj, 0:NX - 2] * m_im + fld[..., sj, 2:NX] * m_ip
                   + fld[..., 0:NY - 2, si] * m_jm + fld[..., 2:NY, si] * m_jp)
        + 0.0625 * (fld[..., 0:NY - 2, 0:NX - 2] * m_mm + fld[..., 2:NY, 0:NX - 2] * m_mp
                    + fld[..., 0:NY - 2, 2:NX] * m_pm + fld[..., 2:NY, 2:NX] * m_pp)
    )
    w_ok = W >= 0.25
    w_safe = torch.where(w_ok, W, torch.ones_like(W))  # avoid 0-div in unused branch
    out = fld.clone()
    out[..., sj, si] = torch.where(w_ok, num / w_safe, f_c)
    return out


def kpp_smooth_dbloc(
    dbloc_raw: torch.Tensor, maskC: torch.Tensor, *, smooth: bool = True
) -> torch.Tensor:
    """SMOOTH_DBLOC + masking (MITgcm kpp_calc.F:270-346): produce ``dblocSm``.

    With ``smooth`` (KPP_SMOOTH_DBLOC, default ON), smooth dbloc level k with
    maskC level k+1 (kpp_calc.F:277 uses mask k+1, since dbloc(k) lives at the
    cell bottom), for k=0..nz-2; then mask by
    maskC[k]*maskC[k+1] (the below-bottom maskC=0 also enforces DBLOC(nzmax)=0).
    ``dbloc_raw``, ``maskC`` haloed ``(*lead, nz, NY, NX)`` (optional leading
    face-batch dim). Returns the same shape.
    """
    nz = dbloc_raw.shape[-3]
    g = dbloc_raw.clone()
    if smooth:
        # vertical-axis (axis -3) slices; ``...`` passes through an optional face-batch.
        g[..., :-1, :, :] = kpp_smooth_horiz(
            dbloc_raw[..., :-1, :, :], maskC[..., 1:nz, :, :]
        )  # levels 0..nz-2, mask 1..nz-1
    kp1 = torch.clamp(torch.arange(nz, device=dbloc_raw.device) + 1, max=nz - 1)  # min(k+1, nz-1)
    return g * maskC * maskC.index_select(-3, kp1)


_apply_kpp_smooth_dbloc = kpp_smooth_dbloc


def kpp_surface_buoyancy_forcing(
    alpha_surf: torch.Tensor,
    beta_surf: torch.Tensor,
    surf_forc_t: torch.Tensor,
    surf_forc_s: torch.Tensor,
    rho_surf: torch.Tensor,
    *,
    gravity: float,
    qsw: torch.Tensor | None = None,
    heat_capacity_cp: float = 3994.0,
    rho_const: float = 1035.0,
):
    """Surface turbulent (bo) and radiative (bosol) buoyancy forcing [m^2/s^3].

    MITgcm kpp_forcing_surf.F:225-242. Pointwise (no horizontal stencil):
      ``bo    = -g * (alpha*surfForcT + beta*surfForcS) / rhoSurf``
      ``bosol = +g *  alpha*Qsw*recip_Cp*recip_rhoConst  / rhoSurf``   (if Qsw given)
    ``alpha_surf``/``beta_surf`` = ttalpha/ssbeta at the surface level (k=1); ``rho_surf``
    = surface in-situ density (sdens). All MIT-convention (consumed from the dump);
    producer-side ingest does heat_flux=-Qnet etc. at the boundary. The
    ``+gravity`` in bosol follows the MITgcm code (Qsw upward-positive), not the stale
    MITgcm comment. With no shortwave file Qsw=0 => bosol=0.
    """
    bo = -gravity * (alpha_surf * surf_forc_t + beta_surf * surf_forc_s) / rho_surf
    if qsw is None:
        bosol = torch.zeros_like(bo)
    else:
        recip_cp = 1.0 / heat_capacity_cp
        recip_rho = 1.0 / rho_const
        bosol = gravity * alpha_surf * qsw * recip_cp * recip_rho / rho_surf
    return bo, bosol


# ──────────────────────────────────────────────────────────────────────
# turbulent velocity scales (wscale) + boundary-layer depth (bldepth).
# Fully columnar (axis 0 = k); broadcasts over arbitrary horizontal shape, so
# one implementation runs unchanged across cs32 -> higher-res cube, regional/
# global, single/multi-GPU.  No horizontal stencil (◆ purely columnar in MITgcm).
#
# Index convention: LOCOMgcm array index k = 0..nz-1 maps to MITgcm 1-based layer
# kl = k+1; ``zgrid``/``hwide`` are passed MIT-indexed (length nz+2, index
# 0..Nr+1) so the bldepth arithmetic mirrors the Fortran 1:1.  All values are in
# MITgcm convention (fed from the op-level dumps): zgrid = rC (negative, z-up),
# hwide = drF (positive).
#
# FP-bitwise fidelity (the whole point — FP-floor vs MITgcm dumps): integer
# powers use explicit multiplication (Fortran ``x**3`` -> x*x*x); Fortran
# ``SIGN(a,b)`` is :func:`_fortran_sign` (NOT torch.sign, which returns 0 at
# b==0).  SIGN(a, -0.0) is decided by the compiler, not the Fortran source, and
# MITgcm's bo is -0.0 whenever the surface fluxes are zero, so the KPP option
# ``sign_zero`` names the reference build's semantics (see
# :data:`KPP_SIGN_ZERO_MODES`); the three
# bldepth bfsfc evaluations are kept in order. Disabled penetrating shortwave
# uses bo directly; enabled shortwave keeps the addition even for zero bosol,
# since adding +0 to -0 changes the subsequent SIGN branch. SWFRAC is applied
# via ``swfrac(depth, jwtype)`` (MITgcm bldepth.F hbf=1).
# ──────────────────────────────────────────────────────────────────────

# stock pkg/kpp constants: kpp_readparms.F:80-155, KPP_PARAMS.h:152,
# kpp_init_fixed.F:125.  Grid- and resolution-independent.
_KPP_WSCALE_DEFAULTS = dict(
    nni=890, nnj=480, zmin=-4.0e-7, zmax=0.0, umin=0.0, umax=4.0e-2,
    vonk=0.4, conc1=5.0, conc2=16.0, conc3=16.0,
    conam=1.257, concm=8.38, conas=-28.86, concs=98.96,
    zetam=-0.2, zetas=-1.0, phepsi=1.0e-10,
)

# Fortran SIGN(a, b) for b = -0.0, as compiled into the reference MITgcm:
#   "signbit"   the IEEE sign bit decides, SIGN(a, -0.0) = -|a|: gfortran's
#               default (-fsign-zero) and ifx/ifort with -assume minus0.
#   "positive"  -0.0 counts as non-negative, SIGN(a, -0.0) = +|a|: the
#               ifx/ifort default and gfortran -fno-sign-zero.
# With zero surface fluxes bo is -0.0 (kpp_forcing_surf.F:225), so the choice
# decides bldepth's stable branch and whether the Ekman/Monin-Obukhov limit
# applies (kpp_routines.F:765-788).  The option names the semantics rather than
# a compiler because the same compiler switches semantics with one flag.
KPP_SIGN_ZERO_MODES = ("signbit", "positive")


def check_kpp_sign_zero(sign_zero: str) -> str:
    """Return ``sign_zero`` if it names a SIGN semantics, else raise."""
    if sign_zero not in KPP_SIGN_ZERO_MODES:
        raise ValueError(
            f"KPP sign_zero must be one of {KPP_SIGN_ZERO_MODES}, got {sign_zero!r}"
        )
    return sign_zero


def _fortran_sign(a: float, b: torch.Tensor, sign_zero: str) -> torch.Tensor:
    """Fortran ``SIGN(a, b)``: ``|a|`` with the sign of ``b`` (``a`` a scalar).

    ``sign_zero`` only matters for ``b = -0.0`` (and NaN); see
    :data:`KPP_SIGN_ZERO_MODES`.  "signbit" is spelled copysign rather than
    where(signbit(b)): torch 2.9 inductor lowers a float64 signbit to b < 0 on
    CUDA (_inductor/codegen/triton.py:1317), which reads -0.0 as positive.
    """
    aa = abs(float(a))
    if check_kpp_sign_zero(sign_zero) == "positive":
        return torch.where(b >= 0, torch.full_like(b, aa), torch.full_like(b, -aa))
    return torch.copysign(torch.full_like(b, aa), b)


def _kpp_gather_k(arr: torch.Tensor, kidx: torch.Tensor, n_spatial: int = 2) -> torch.Tensor:
    """Gather ``arr[..., kidx, ...]`` along the vertical axis.

    ``arr`` shape ``(*lead, nz, *spatial)``; ``kidx`` shape ``(*lead, *spatial)``
    long in ``[0, nz-1]``; ``n_spatial`` = number of trailing spatial dims (2 for the
    C-grid (NY, NX)).  Returns ``(*lead, *spatial)``.  The vertical axis is taken from
    the BACK (``-(n_spatial+1)``) so an optional leading batch dim (the face batch)
    passes through bit-for-bit identically to the no-batch ``axis 0`` case.
    """
    vax = -(n_spatial + 1)
    return torch.gather(arr, vax, kidx.long().unsqueeze(vax)).squeeze(vax)


def kpp_wscale_tables(*, dtype: torch.dtype = torch.float64, device=None, **overrides):
    """Build the wm/ws turbulent-velocity-scale lookup tables.

    Ports MITgcm ``kpp_init_fixed.F`` L132-156.  Grid- and time-independent:
    build once.  Returns ``(wmt, wst)`` each shape ``(nni+2, nnj+2)`` indexed
    ``[i_zehat, j_ustar]`` (= Fortran ``wmt(0:nni+1, 0:nnj+1)``), consumed by
    :func:`kpp_wscale`.  Stable (``zehat>=0``) and the two unstable regimes
    (momentum/scalar, weak/strong) follow the Fortran branch-for-branch.
    Integer powers (``usta**3``) use explicit multiplication to match Fortran.
    """
    p = {**_KPP_WSCALE_DEFAULTS, **overrides}
    nni, nnj = p["nni"], p["nnj"]
    zmin, zmax, umin, umax = p["zmin"], p["zmax"], p["umin"], p["umax"]
    vonk, conc1, conc2, conc3 = p["vonk"], p["conc1"], p["conc2"], p["conc3"]
    conam, concm, conas, concs = p["conam"], p["concm"], p["conas"], p["concs"]
    zetam, zetas, phepsi = p["zetam"], p["zetas"], p["phepsi"]
    p25, p33 = 0.25, 1.0 / 3.0

    deltaz = (zmax - zmin) / (nni + 1)
    deltau = (umax - umin) / (nnj + 1)
    ii = torch.arange(nni + 2, dtype=dtype, device=device).reshape(-1, 1)
    jj = torch.arange(nnj + 2, dtype=dtype, device=device).reshape(1, -1)
    zehat = deltaz * ii + zmin                     # (nni+2, 1)
    usta = deltau * jj + umin                      # (1, nnj+2)
    usta3 = usta * usta * usta                     # Fortran usta**3 (integer power)
    zeta = zehat / torch.clamp(usta3, min=phepsi)  # zehat / max(phepsi, usta**3)
    zehat_b, usta_b = torch.broadcast_tensors(zehat, usta)
    usta3_b = usta_b * usta_b * usta_b

    # momentum scale wmt
    wm_stable = vonk * usta_b / (1.0 + conc1 * zeta)
    wm_weak = vonk * usta_b * (1.0 - conc2 * zeta) ** p25
    wm_strong = vonk * (conam * usta3_b - concm * zehat_b) ** p33
    wmt = torch.where(
        zehat_b >= 0.0, wm_stable, torch.where(zeta > zetam, wm_weak, wm_strong)
    )
    # scalar scale wst (stable branch identical to wmt)
    ws_weak = vonk * usta_b * torch.sqrt(1.0 - conc3 * zeta)
    ws_strong = vonk * (conas * usta3_b - concs * zehat_b) ** p33
    wst = torch.where(
        zehat_b >= 0.0, wm_stable, torch.where(zeta > zetas, ws_weak, ws_strong)
    )
    return wmt, wst


def kpp_wscale(sigma, hbl, ustar, bfsfc, wmt, wst, **overrides):
    """Turbulent velocity scales ``wm``, ``ws`` (MITgcm wscale, kpp_routines.F
    L923-1028).

    ``sigma``/``ustar``/``bfsfc`` are column-shaped (``*spatial``); ``hbl`` is the
    boundary-layer depth (a scalar ``-zgrid(kl)`` inside the bldepth Rib loop, or
    a column tensor elsewhere).  ``zehat = vonk*sigma*hbl*bfsfc``: unstable/neutral
    (``zehat<=zmax``) bilinearly interpolates the (zehat,ustar) lookup tables with
    **nearest-neighbour extrapolation** (the clamp at L988-993, NOT linear
    extrapolation); stable (``zehat>zmax``) uses the analytic form.
    """
    p = {**_KPP_WSCALE_DEFAULTS, **overrides}
    nni, nnj = p["nni"], p["nnj"]
    zmin, zmax, umin, umax = p["zmin"], p["zmax"], p["umin"], p["umax"]
    vonk, conc1 = p["vonk"], p["conc1"]
    deltaz = (zmax - zmin) / (nni + 1)
    deltau = (umax - umin) / (nnj + 1)

    zehat = vonk * sigma * hbl * bfsfc

    # ---- table branch (zehat <= zmax): bilinear lookup, nearest-extrap clamp ----
    zdiff = zehat - zmin
    iz = torch.clamp((zdiff / deltaz).long(), 0, nni)   # INT(.) toward 0, then clamp
    izp1 = iz + 1
    udiff = ustar - umin
    ju = torch.clamp((udiff / deltau).long(), 0, nnj)
    jup1 = ju + 1
    zfrac = zdiff / deltaz - iz.to(zdiff.dtype)
    ufrac = udiff / deltau - ju.to(udiff.dtype)
    fzfrac = 1.0 - zfrac

    def _bilin(tab):
        wa = fzfrac * tab[iz, jup1] + zfrac * tab[izp1, jup1]
        wb = fzfrac * tab[iz, ju] + zfrac * tab[izp1, ju]
        return (1.0 - ufrac) * wb + ufrac * wa

    wm_tab = _bilin(wmt)
    ws_tab = _bilin(wst)

    # ---- stable branch (zehat > zmax): analytic ----
    u3 = ustar * ustar * ustar
    use_tab = zehat <= zmax
    # On table lanes the analytic denominator can be exactly 0 (dry column:
    # ustar = 0 and zehat = 0), which is harmless in the forward pass (the
    # ``where`` below discards it) but puts 0*inf = NaN into the backward pass
    # through DivBackward.  Divide by 1 on those lanes instead; the selected
    # lanes see the same divisor as before, so forward values are unchanged.
    denom = u3 + conc1 * zehat
    denom_safe = torch.where(use_tab, torch.ones_like(denom), denom)
    wm_st = vonk * ustar * u3 / denom_safe

    wm = torch.where(use_tab, wm_tab, wm_st)
    ws = torch.where(use_tab, ws_tab, wm_st)   # stable ws == wm
    return wm, ws


# Diagnostic tap for the Ekman/Monin-Obukhov limiter inside kpp_bldepth.  None
# on the production path; a probe assigns a dict to collect the limiter's inputs
# and the hbl on either side of it.  Set and cleared by the caller, never here.
_BLDEPTH_PROBE: dict | None = None


def kpp_bldepth(
    kmtj: torch.Tensor,
    dvsq: torch.Tensor,
    dbloc: torch.Tensor,
    Ritop: torch.Tensor,
    ustar: torch.Tensor,
    bo: torch.Tensor,
    bosol: torch.Tensor,
    coriol: torch.Tensor,
    zgrid: torch.Tensor,
    hwide: torch.Tensor,
    wmt: torch.Tensor,
    wst: torch.Tensor,
    *,
    Ricr: float,
    epsilon: float,
    Vtc: float,
    cekman: float,
    cmonob: float,
    minKPPhbl: float,
    vonk: float,
    phepsi: float = 1.0e-10,
    limit_hbl_stable: bool = True,
    jwtype: int = 2,
    penetrating_sw: bool = True,
    sign_zero: str = "signbit",
    **wscale_overrides,
):
    """Boundary-layer depth ``hbl`` and ``kbl``/``bfsfc``/``stable``/``casea``.

    Ports MITgcm ``bldepth`` (kpp_routines.F L309-919).  Columnar; broadcasts over
    arbitrary horizontal shape.  All inputs are MITgcm-convention (fed from the
    op-level dumps).

    Parameters
    ----------
    kmtj : Tensor ``(*spatial)`` int
        Number of wet layers (= MITgcm ``nzmax``/``kLowC``); per-column (cs32 is
        NOT flat-bottom, so ``kmtj`` varies and the search/interp are bounded by it).
    dvsq, dbloc, Ritop : Tensor ``(nz, *spatial)``
        bldepth inputs at LOCOMgcm index ``k`` = MIT layer ``kl=k+1``: velocity shear re
        surface squared, masked local buoyancy gradient, scaled bulk-Ri numerator.
    ustar, bo, bosol, coriol : Tensor ``(*spatial)``
    zgrid, hwide : Tensor ``(nz+2,)``  MIT-indexed (0..Nr+1)
        ``zgrid[0]=phepsi, zgrid[1..Nr]=rC (negative), zgrid[Nr+1]=rC[Nr]*100``;
        ``hwide[0]=phepsi, hwide[1..Nr]=drF, hwide[Nr+1]=phepsi``.
    wmt, wst : Tensor   lookup tables from :func:`kpp_wscale_tables`.
    Ricr, epsilon, Vtc, cekman, cmonob, minKPPhbl, vonk, phepsi : float
        Stock pkg/kpp params (``Vtc = concv*sqrt(0.2/concs/epsilon)/vonk**2/Ricr``).
    jwtype : int   Jerlov water type for the penetrating-shortwave fraction in
        ``bfsfc = bo + bosol*(1 - SWFRAC(hbf*depth))`` (MITgcm bldepth.F:505-519,
        hbf=1).
    penetrating_sw : bool
        Effective SHORTWAVE_HEATING / selectPenetratingSW >= 1 mode. When
        disabled, use bo directly at every stage, preserving signed zero.
        Enabled but zero solar flux still executes the radiative addition.
    sign_zero : str
        Fortran SIGN semantics for a -0.0 second argument, one of
        :data:`KPP_SIGN_ZERO_MODES`; zero surface forcing is stable under
        "positive" and unstable under "signbit".

    Returns
    -------
    hbl : Tensor ``(*spatial)``         boundary-layer depth [m] (>0, downward)
    kbl : Tensor ``(*spatial)`` long    first grid level below hbl (MIT 1-based)
    bfsfc, stable, casea : Tensor ``(*spatial)``   final (clipped) forcing diagnostics
    Rib : Tensor ``(nz, *spatial)``     bulk Richardson number (Rib[k]=MIT Rib(k+1))
    """
    from locomgcm.accel import (
        consumer_snapshot,
        in_accelerated_dispatch,
        requested,
    )

    if not in_accelerated_dispatch():
        with consumer_snapshot():
            if requested(("kpp_bldepth_cuda",)):
                from locomgcm.dynamics._kpp_bldepth_cuda import dispatcher

                return dispatcher.kpp_bldepth_cuda(
                    kmtj,
                    dvsq,
                    dbloc,
                    Ritop,
                    ustar,
                    bo,
                    bosol,
                    coriol,
                    zgrid,
                    hwide,
                    wmt,
                    wst,
                    Ricr=Ricr,
                    epsilon=epsilon,
                    Vtc=Vtc,
                    cekman=cekman,
                    cmonob=cmonob,
                    minKPPhbl=minKPPhbl,
                    vonk=vonk,
                    phepsi=phepsi,
                    limit_hbl_stable=limit_hbl_stable,
                    jwtype=jwtype,
                    penetrating_sw=penetrating_sw,
                    sign_zero=sign_zero,
                    **wscale_overrides,
                )
    nz = dvsq.shape[-3]
    Nr = nz
    lead = dvsq.shape[:-3]            # optional leading face-batch dims
    hsp = dvsq.shape[-2:]            # (NY, NX); vertical axis is -3
    dtype, device = dvsq.dtype, dvsq.device
    kmtj = kmtj.long()

    def fsign(a, b):
        """Fortran SIGN(a,b) under this call's ``sign_zero``.

        With zero surface fluxes MITgcm's bo is -0.0 (kpp_forcing_surf.F:225);
        kpp_routines.F:765 then takes the unstable branch and skips the
        Ekman/Monin-Obukhov limit under "signbit" and
        the stable branch with the limit under "positive".
        """
        return _fortran_sign(a, b, sign_zero)

    def kk(arr, k):                  # vertical-axis (-3) scalar index -> (*lead, NY, NX)
        return arr[..., k, :, :]

    # ---- Rib over kl=2..Nr (1st bfsfc = raw bo, not clipped) ----
    Rib = torch.zeros(lead + (nz,) + hsp, dtype=dtype, device=device)  # Rib[k]=MIT Rib(k+1); Rib[0]=0
    from locomgcm.dynamics.surface_flux import swfrac

    # Hoist the per-level wscale lookup out of the Rib scan.  bfsfc/stable/sigma
    # and the hbl arg (-zgrid[kl]) at level kl are functions of (static zgrid[kl] +
    # forcing bo/bosol/ustar) ONLY — independent of the running Rib accumulation —
    # so ws for ALL kl=2..Nr is computable in ONE batched wscale call (extra leading
    # LEVEL axis), bitwise vs the per-level loop.  The serial Rib write order below
    # is unchanged; it just READS the precomputed ws_all.
    ws_all = None
    if _kpp_wscale_batch_enabled():
        kls = torch.arange(2, Nr + 1, device=device)              # (Nlev,)
        nzkl = (-zgrid[kls]).reshape((-1,) + (1,) * (len(lead) + 2))  # -zgrid(kl), (Nlev,1,...,1)
        bfsfc_b = bo + bosol * (1.0 - swfrac(nzkl, jwtype)) if penetrating_sw else bo
        stable_b = 0.5 + fsign(0.5, bfsfc_b)
        sigma_b = stable_b + (1.0 - stable_b) * epsilon
        _, ws_all = kpp_wscale(sigma_b, nzkl, ustar, bfsfc_b, wmt, wst, **wscale_overrides)
        _record_wscale_batch("kpp.kpp_bldepth", "bldepth_level_batch")

    for kl in range(2, Nr + 1):                       # MIT kl = 2..Nr
        k = kl - 1                                    # LOCOMgcm index
        # Per-level surface buoyancy forcing: solar absorbed down to hbf*zgrid(kl)
        # (MITgcm bldepth.F:505-519, hbf=1; SWFRAC = fraction penetrating BELOW the
        # depth, so 1-SWFRAC is the part absorbed above).
        if ws_all is not None:
            ws = ws_all[kl - 2]                       # batch idx kl-2 == level kl
        else:
            bfsfc = bo + bosol * (1.0 - swfrac(-zgrid[kl], jwtype)) if penetrating_sw else bo
            stable = 0.5 + fsign(0.5, bfsfc)
            sigma = stable + (1.0 - stable) * epsilon
            _, ws = kpp_wscale(sigma, -zgrid[kl], ustar, bfsfc, wmt, wst, **wscale_overrides)
        dz_km1 = zgrid[kl - 1] - zgrid[kl]
        dz_k = zgrid[kl] - zgrid[kl + 1]
        bvsq = 0.5 * (kk(dbloc, k - 1) / dz_km1 + kk(dbloc, k) / dz_k)  # dbloc(kl-1), dbloc(kl)
        # sqrt only sees a non-zero argument: the ``bvsq == 0`` lanes take the
        # zero branch below anyway, but ``d sqrt/dx`` at 0 is infinite and the
        # unselected branch would put 0*inf = NaN into the backward pass (same
        # guard as ``s_safe`` in kpp_wscale_tables).  Forward values unchanged.
        bvsq_safe = torch.where(bvsq == 0.0, torch.ones_like(bvsq), torch.abs(bvsq))
        vtsq = torch.where(
            bvsq == 0.0,
            torch.zeros_like(bvsq),
            (-zgrid[kl]) * ws * torch.sqrt(bvsq_safe) * Vtc,
        )
        tempVar2 = torch.clamp(kk(dvsq, k) + vtsq, min=phepsi)  # MAX(dvsq+vtsq, phepsi)
        Rib[..., k, :, :] = kk(Ritop, k) / tempVar2

    # ---- 1st kbl: first kl in 2..Nr with Rib(kl) > Ricr ----
    kbl = kmtj.clone()
    for kl in range(2, Nr + 1):
        cond = (kbl == kmtj) & (kk(Rib, kl - 1) > Ricr)
        kbl = torch.where(cond, torch.full_like(kbl, kl), kbl)

    # ---- initial hbl = -zgrid(kbl_init); kbl_init = kmtj (or 1 if kmtj<1) ----
    kbl_init = torch.where(kmtj < 1, torch.ones_like(kmtj), kmtj)
    hbl = -zgrid[torch.clamp(kbl_init, 0, nz + 1)]

    # ---- linear interp where 1 < kbl < kmtj ----
    cond = (kbl > 1) & (kbl < kmtj)
    zk = zgrid[torch.clamp(kbl, 0, nz + 1)]
    zkm1 = zgrid[torch.clamp(kbl - 1, 0, nz + 1)]
    Rk = _kpp_gather_k(Rib, torch.clamp(kbl - 1, 0, nz - 1))    # Rib(kl)   = Rib[kl-1]
    Rkm1 = _kpp_gather_k(Rib, torch.clamp(kbl - 2, 0, nz - 1))  # Rib(kl-1) = Rib[kl-2]
    denom = Rk - Rkm1
    denom = torch.where(denom == 0.0, torch.ones_like(denom), denom)  # safe; unused where cond=False
    hbl_interp = -zkm1 + (zkm1 - zk) * (Ricr - Rkm1) / denom
    hbl = torch.where(cond, hbl_interp, hbl)

    # ---- 2nd bfsfc + clip (never 0); Ekman/Monin-Obukhov limit if stable ----
    # MITgcm bldepth.F (post-loop): solar absorbed down to hbf*hbl at the final BL depth.
    bfsfc = bo + bosol * (1.0 - swfrac(hbl, jwtype)) if penetrating_sw else bo
    stable = 0.5 + fsign(0.5, bfsfc)
    bfsfc = fsign(1.0, bfsfc) * torch.clamp(torch.abs(bfsfc), min=phepsi)
    # Cross-validation checkpoint, mirroring the MIT-side hook this limiter is
    # bracketed by (:_patch_bldepth_hlimit).  The
    # dict is populated only when a caller supplies one, so the production path
    # allocates nothing and the arithmetic below is untouched either way.
    if _BLDEPTH_PROBE is not None:
        _BLDEPTH_PROBE["hblPre"] = hbl
    if limit_hbl_stable:
        hekman = cekman * ustar / torch.clamp(torch.abs(coriol), min=phepsi)
        hmonob = cmonob * ustar * ustar * ustar / vonk / bfsfc
        hlimit = stable * torch.minimum(hekman, hmonob) + (stable - 1.0) * zgrid[nz]
        hbl = torch.where(bfsfc > 0.0, torch.minimum(hbl, hlimit), hbl)
        if _BLDEPTH_PROBE is not None:
            _BLDEPTH_PROBE.update(
                hekman=hekman, hmonob=hmonob, hlimit=hlimit,
                ustar=ustar, hblPost=hbl,
            )

    # ---- floor; reset & re-find kbl (first kl with -zgrid(kl) > hbl) ----
    hbl = torch.clamp(hbl, min=minKPPhbl)
    kbl = kmtj.clone()
    for kl in range(2, Nr + 1):
        cond = (kbl == kmtj) & ((-zgrid[kl]) > hbl)
        kbl = torch.where(cond, torch.full_like(kbl, kl), kbl)

    # ---- 3rd bfsfc (solar absorbed to the final hbl) + clip; casea ----
    bfsfc = bo + bosol * (1.0 - swfrac(hbl, jwtype)) if penetrating_sw else bo
    stable = 0.5 + fsign(0.5, bfsfc)
    bfsfc = fsign(1.0, bfsfc) * torch.clamp(torch.abs(bfsfc), min=phepsi)
    kbl_c = torch.clamp(kbl, 0, nz + 1)
    casea = 0.5 + fsign(0.5, -zgrid[kbl_c] - 0.5 * hwide[kbl_c] - hbl)

    return hbl, kbl, bfsfc, stable, casea, Rib


# ──────────────────────────────────────────────────────────────────────
# interior mixing (Ri_iwmix) + optional vertical Ri smoothing (z121).
# Columnar; output is the interior viscosity/diffusivity at interfaces ki=1..Nr
# (MITgcm diffus(:,ki,1:3)), captured BEFORE blmix/enhance overwrite the boundary
# layer.  Index: LOCOMgcm k = 0..nz-1 == MIT interface kl = k+1 (bottom of layer k+1).
# ──────────────────────────────────────────────────────────────────────


def kpp_z121_smooth_ri(
    v: torch.Tensor, *, Riinfty: float, sign_zero: str = "signbit"
) -> torch.Tensor:
    """Vertical 1-2-1 smoothing of the gradient Richardson number (MITgcm z121,
    kpp_routines.F L1250-1327).

    ``v`` shape ``(nz, *spatial)`` is Ri at interfaces ki=1..Nr (LOCOMgcm index k=ki-1).
    Only points with ``0 <= Ri <= Riinfty`` participate (the ``KRi_range`` gate,
    two Fortran SIGNs, so ``sign_zero`` decides whether Ri = -0.0 does);
    the bottom uses ``v(Nrp1)=v(Nr)``.  MITgcm's ``zwork`` always references the
    ORIGINAL neighbour values (single pass), so this is a pure
    original-value stencil — vectorised here, bit-equivalent to the serial loop.

    DORMANT in the stock cs32 config (``num_v_smooth_Ri=0`` => never called by
    :func:`kpp_ri_iwmix`); code-verified only (no MIT dump to validate against).

    ``v`` may carry a leading face-batch dim ``(*lead, nz, NY, NX)``; the
    vertical axis is ``-3`` (trailing spatial = (NY, NX)).  The serial loop indexes
    only that axis, so an optional leading batch dim passes through identically.
    """
    nz = v.shape[-3]
    sl = lambda k: (..., slice(k, k + 1), slice(None), slice(None))  # noqa: E731

    def s05(x):  # 0.5 + SIGN(0.5, x): 1.0 for x > 0, 0.0 for x < 0, x = -0.0 per sign_zero
        return 0.5 + _fortran_sign(0.5, x, sign_zero)

    kri = s05(v) * s05(Riinfty - v)               # KRi_range(ki=1..Nr) on ORIGINAL v
    out = v.clone()
    z = torch.zeros_like(v[sl(0)])
    for ki in range(1, nz + 1):                    # MIT ki = 1..Nr
        k = ki - 1
        kc = kri[sl(k)]
        kp = kri[sl(k + 1)] if ki < nz else z      # KRi(ki+1); KRi(Nrp1)=0
        vp = v[sl(k + 1)] if ki < nz else v[sl(nz - 1)]  # v(ki+1); v(Nrp1)=v(Nr)
        if ki == 1:                                # no upper neighbour (v(0) dummy)
            out[sl(k)] = (2.0 * v[sl(k)] + kc * kp * vp) / (2.0 + kc * kp)
        else:
            km = kri[sl(k - 1)]                    # KRi(ki-1)
            vm = v[sl(k - 1)]                      # v(ki-1)  (ORIGINAL, via zwork)
            out[sl(k)] = (2.0 * v[sl(k)] + kc * kp * vp + kc * km * vm) / (2.0 + kc * (kp + km))
    return out


def kpp_ri_iwmix(
    kmtj: torch.Tensor,
    shsq: torch.Tensor,
    dbloc: torch.Tensor,
    dblocSm: torch.Tensor,
    diffusKzS: torch.Tensor,
    diffusKzT: torch.Tensor,
    zgrid: torch.Tensor,
    *,
    Riinfty: float,
    BVSQcon: float,
    difm0: float,
    difs0: float,
    dift0: float,
    difmcon: float,
    difscon: float,
    diftcon: float,
    visc_arnr: float,
    phepsi: float = 1.0e-10,
    num_v_smooth_Ri: int = 0,
    sign_zero: str = "signbit",
):
    """Interior viscosity/diffusivity from shear, convective and internal-wave
    mixing (MITgcm ``Ri_iwmix``, kpp_routines.F L1052-1246).

    Columnar; broadcasts over arbitrary horizontal shape.  All inputs are
    MITgcm-convention (fed from the op-level dumps); the ``dblocSm`` argument is
    the horizontally-smoothed dbloc (KPPMIX passes ``ghat`` = dump ``dblocSm``),
    ``dbloc`` is the masked dbloc (dump ``dbloc_masked``), and
    ``diffusKzS``/``diffusKzT`` are the background diffusivities (dumps
    ``bgKzS``/``bgKzT``).

    Parameters
    ----------
    kmtj : Tensor ``(*spatial)`` int        number of wet layers.
    shsq, dbloc, dblocSm : Tensor ``(nz, *spatial)``   at interfaces ki=1..Nr.
    diffusKzS, diffusKzT : Tensor ``(nz, *spatial)``   background diffusivities.
    zgrid : Tensor ``(nz+2,)``  MIT-indexed (see :func:`kpp_bldepth`).
    Riinfty, BVSQcon, difm0, difs0, dift0, difmcon, difscon, diftcon, visc_arnr :
        stock pkg/kpp params (``visc_arnr`` = ``viscArNr(1)``).
    num_v_smooth_Ri : z121 vertical-Ri smoothing passes (stock 0 => z121 dormant).
    sign_zero : Fortran SIGN semantics inside z121 (:data:`KPP_SIGN_ZERO_MODES`).

    Returns
    -------
    visc, KzS, KzT : Tensor ``(nz, *spatial)``  interior mixing coeffs at
        interfaces ki=1..Nr (= MITgcm ``diffus(:,1:Nr,1:3)``; surface ki=0 is 0).
    """
    nz = shsq.shape[-3]
    dtype, device = shsq.dtype, shsq.device
    kmtj = kmtj.long()
    # vertical axis is -3 (trailing spatial = (NY, NX)); a leading face-batch dim,
    # when present, passes through every op below identically.

    # ---- gradient Ri (Rig) and Brunt-Vaisala^2 (Nsq) at interfaces ki=1..Nr ----
    # dz(ki) = zgrid(ki) - zgrid(ki+1); LOCOMgcm: zgrid[k+1]-zgrid[k+2] for k=0..nz-1.
    dz = (zgrid[1 : nz + 1] - zgrid[2 : nz + 2]).reshape(nz, 1, 1)
    Rig_raw = dblocSm * dz / torch.clamp(shsq, min=phepsi)   # MAX(shsq, phepsi)
    Nsq_raw = dbloc / dz
    # "set values at bottom and below to nearest value above bottom": for MIT
    # ki>=kmtj (LOCOMgcm k>=kmtj-1), copy the ki=kmtj-1 (LOCOMgcm k=kmtj-2) value -> forward-fill.
    karange = torch.arange(nz, device=device).reshape(nz, 1, 1)
    src = torch.clamp(torch.minimum(karange, (kmtj - 2).unsqueeze(-3)), 0, nz - 1)
    src = src.expand_as(Rig_raw)
    Rig = torch.gather(Rig_raw, -3, src)
    Nsq = torch.gather(Nsq_raw, -3, src)
    shallow = (kmtj <= 1).unsqueeze(-3)             # kmtj<=1 -> Rig=Nsq=0
    Rig = torch.where(shallow, torch.zeros_like(Rig), Rig)
    Nsq = torch.where(shallow, torch.zeros_like(Nsq), Nsq)

    # ---- optional vertical smoothing of Ri (dormant: num_v_smooth_Ri=0) ----
    for _ in range(num_v_smooth_Ri):
        Rig = kpp_z121_smooth_ri(Rig, Riinfty=Riinfty, sign_zero=sign_zero)

    # ---- convective switch fcon (note: MAX(Nsq, BVSQcon), NOT min) ----
    Rig_c = torch.clamp(Nsq, min=BVSQcon)
    ratio = torch.clamp((BVSQcon - Rig_c) / BVSQcon, max=1.0)
    fcon = 1.0 - ratio * ratio
    fcon = fcon * fcon * fcon
    # ---- shear switch fRi ----
    Rig_s = torch.clamp(Rig, min=0.0)
    ratio = torch.clamp(Rig_s / Riinfty, max=1.0)
    fRi = 1.0 - ratio * ratio
    fRi = fRi * fRi * fRi

    # ---- compose; background uses diffusKzS(kp1), kp1=MIN(ki+1,Nr) ----
    kp1 = torch.clamp(torch.arange(nz, device=device) + 1, max=nz - 1)  # LOCOMgcm idx of kp1
    bgS = diffusKzS.index_select(-3, kp1)
    bgT = diffusKzT.index_select(-3, kp1)
    visc = visc_arnr + fcon * difmcon + fRi * difm0
    KzS = bgS + fcon * difscon + fRi * difs0
    KzT = bgT + fcon * diftcon + fRi * dift0
    return visc, KzS, KzT


# ──────────────────────────────────────────────────────────────────────
# boundary-layer K-profile + nonlocal transport (blmix) + kbl-1
# interface matching (enhance).  Columnar; reference layer kn and the kbl-1
# interface are per-column -> gather/scatter along the vertical axis.
# diffus is the interior mixing from Ri_iwmix AS blmix sees it (post seafloor-
# zero, MIT-indexed 0:Nrp1 on axis 1), stacked (3, nz+2, *spatial) = [visc,KzS,KzT].
# ──────────────────────────────────────────────────────────────────────


def _gather_mit(arr: torch.Tensor, idx: torch.Tensor, n_spatial: int = 2) -> torch.Tensor:
    """Index a MIT-indexed (0..Nrp1) vertical array at per-column index ``idx``
    (``(*lead, *spatial)``), clamped -> ``(*lead, *spatial)``.  ``arr`` may be 1-D
    ``(M,)`` (a shared z-profile: zgrid/hwide; plain fancy-index, broadcasts over any
    leading dims of ``idx``) or ``(*lead, M, *spatial)`` (a field; gather on the
    vertical axis taken from the BACK so an optional leading face-batch dim passes
    through bit-for-bit identically to the no-batch case)."""
    if arr.ndim == 1:
        idx = torch.clamp(idx.long(), 0, arr.shape[0] - 1)
        return arr[idx]
    vax = -(n_spatial + 1)
    idx = torch.clamp(idx.long(), 0, arr.shape[vax] - 1)
    return torch.gather(arr, vax, idx.unsqueeze(vax)).squeeze(vax)


def kpp_blmix(
    ustar: torch.Tensor,
    bfsfc: torch.Tensor,
    hbl: torch.Tensor,
    stable: torch.Tensor,
    casea: torch.Tensor,
    kbl: torch.Tensor,
    diffus: torch.Tensor,
    zgrid: torch.Tensor,
    hwide: torch.Tensor,
    wmt: torch.Tensor,
    wst: torch.Tensor,
    *,
    epsilon: float,
    conc1: float,
    cg: float,
    phepsi: float = 1.0e-10,
    sign_zero: str = "signbit",
    **wscale_overrides,
):
    """Boundary-layer mixing coefficients ``blmc`` + nonlocal transport ``ghat`` +
    ``dkm1`` (MITgcm ``blmix``, kpp_routines.F L1429-1726).

    ``sign_zero`` is the Fortran SIGN semantics of the velocity-scale clip
    (:data:`KPP_SIGN_ZERO_MODES`).

    Columnar; broadcasts over arbitrary horizontal shape.  All inputs MITgcm-
    convention (fed from dumps): ``ustar/bfsfc/hbl/stable/casea/kbl`` are bldepth
    outputs; ``diffus`` is the interior mixing (Ri_iwmix) as blmix sees it —
    stacked ``(3, nz+2, *spatial)`` = [visc, KzS, KzT], MIT-indexed ki=0..Nrp1
    on axis 1 (i.e. the post-seafloor-zero diffus, dump ``blVisc``/``blKzS``/``blKzT``).

    With an optional leading face-batch dim the layouts are
    ``diffus (3, *lead, nz+2, NY, NX)``, ``blmc (3, *lead, nz, NY, NX)``,
    ``ghat (*lead, nz, NY, NX)``, ``dkm1 (3, *lead, NY, NX)``; the vertical axis is
    ``-3`` throughout, so the ``_gather_mit`` lookups and the per-level writes pass an
    optional batch dim through identically.

    Returns
    -------
    blmc : Tensor ``(3, *lead, nz, NY, NX)``   boundary-layer mixing coeffs at ki=1..Nr.
    ghat : Tensor ``(*lead, nz, NY, NX)``       nonlocal transport coefficient.
    dkm1 : Tensor ``(3, *lead, NY, NX)``        BL diffusivity at the kbl-1 level.
    """
    from locomgcm.accel import (
        consumer_snapshot,
        in_accelerated_dispatch,
        requested,
    )

    if not in_accelerated_dispatch():
        with consumer_snapshot():
            if requested(("kpp_blmix_cuda",)):
                from locomgcm.dynamics._kpp_blmix_cuda import dispatcher

                return dispatcher.kpp_blmix(
                    ustar,
                    bfsfc,
                    hbl,
                    stable,
                    casea,
                    kbl,
                    diffus,
                    zgrid,
                    hwide,
                    wmt,
                    wst,
                    epsilon=epsilon,
                    conc1=conc1,
                    cg=cg,
                    phepsi=phepsi,
                    sign_zero=sign_zero,
                    **wscale_overrides,
                )
    nz = diffus.shape[-3] - 2         # diffus vertical axis (-3) = 0..Nrp1 = nz+2
    Nr = nz
    lead = ustar.shape[:-2]           # optional leading face-batch dims (() if none)
    hsp = ustar.shape[-2:]            # (NY, NX)
    dtype, device = ustar.dtype, ustar.device
    kbl = kbl.long()

    def fsign(a, b):                  # Fortran SIGN(a,b) under sign_zero, as in kpp_bldepth
        return _fortran_sign(a, b, sign_zero)

    # ---- velocity scales at hbl (clipped) ----
    sigma = stable * 1.0 + (1.0 - stable) * epsilon
    wm, ws = kpp_wscale(sigma, hbl, ustar, bfsfc, wmt, wst, **wscale_overrides)
    wm = fsign(1.0, wm) * torch.clamp(torch.abs(wm), min=phepsi)
    ws = fsign(1.0, ws) * torch.clamp(torch.abs(ws), min=phepsi)
    w_of = [wm, ws, ws]               # m=0 viscosity uses wm; m=1,2 use ws

    # ---- reference layer kn = caseA ? kbl-1 : kbl  (INT(caseA+phepsi)) ----
    caseA_int = (casea + phepsi).long()
    kn = caseA_int * (kbl - 1) + (1 - caseA_int) * kbl    # MIT index
    hw_kn = _gather_mit(hwide, kn)
    hw_knp1 = _gather_mit(hwide, kn + 1)
    delhat = 0.5 * hw_kn - _gather_mit(zgrid, kn) - hbl
    R = 1.0 - delhat / hw_kn
    f1 = stable * conc1 * bfsfc / torch.clamp(ustar * ustar * ustar * ustar, min=phepsi)

    gat1, dat1 = [], []
    for m in range(3):
        dm = diffus[m]                                # (nz+2, *spatial)
        d_knm1 = _gather_mit(dm, kn - 1)
        d_kn = _gather_mit(dm, kn)
        d_knp1 = _gather_mit(dm, kn + 1)
        dvdzup = (d_knm1 - d_kn) / hw_kn
        dvdzdn = (d_kn - d_knp1) / hw_knp1
        # 0.5*((1-R)*(x+|x|) + R*(y+|y|)) = (1-R)*max(x,0) + R*max(y,0)
        p_m = 0.5 * ((1.0 - R) * (dvdzup + torch.abs(dvdzup))
                     + R * (dvdzdn + torch.abs(dvdzdn)))
        h_m = d_kn + p_m * delhat
        wm_m = w_of[m]
        gat1.append(h_m / hbl / wm_m)
        dat1.append(torch.clamp(-p_m / wm_m + f1 * h_m, max=0.0))   # MIN(dat1, 0)

    # face-batch layout: vertical axis is -3 (faces sit in *lead, before nz).
    blmc = torch.zeros((3,) + lead + (nz,) + hsp, dtype=dtype, device=device)
    ghat = torch.zeros(lead + (nz,) + hsp, dtype=dtype, device=device)

    # Hoist the per-level wscale lookup out of the ki loop.  sigma_i at level ki
    # is a function of (static zgrid[ki]/hwide[ki] + the loop-INVARIANT bldepth
    # outputs hbl/stable) ONLY — it never reads the running blmc/ghat writes — so
    # wm_i/ws_i for ALL ki=1..Nr is computable in ONE batched wscale call (leading
    # LEVEL axis), bitwise vs the loop.  sig/a1/a2/a3 stay per-level (HEAD order).
    wm_i_all = ws_i_all = None
    if _kpp_wscale_batch_enabled():
        kis = torch.arange(1, Nr + 1, device=device)
        sigb = ((-zgrid[kis] + 0.5 * hwide[kis])
                .reshape((-1,) + (1,) * (len(lead) + 2)) / hbl)   # (Nlev, *lead, NY, NX)
        sigma_ib = stable * sigb + (1.0 - stable) * torch.clamp(sigb, max=epsilon)
        wm_i_all, ws_i_all = kpp_wscale(
            sigma_ib, hbl, ustar, bfsfc, wmt, wst, **wscale_overrides)
        _record_wscale_batch("kpp.kpp_blmix", "blmix_level_batch")

    if wm_i_all is not None:
        # The ki loop carries no recurrence: level ki reads only sig(ki) and the
        # loop-invariant gat1/dat1/hbl/stable/w_i, never the running blmc/ghat
        # writes.  With every level's wscale already batched above, the whole
        # loop is one broadcast expression with the level axis in front.  Each
        # scalar operation keeps its operand order and association (the products
        # below associate left to right exactly as the per-level spelling did)
        # and no reduction is introduced, so the result is bitwise the loop's --
        # the level index only moves which lane a value lands in.
        # 4*Nr slice assignments become a handful of whole-array writes.
        lev = (slice(1, Nr + 1),) + (None,) * (len(lead) + 2)
        sig_all = (-zgrid[lev] + 0.5 * hwide[lev]) / hbl   # (Nlev, *lead, NY, NX)
        w_i_all = [wm_i_all, ws_i_all, ws_i_all]
        a1, a2, a3 = sig_all - 2.0, 3.0 - 2.0 * sig_all, sig_all - 1.0
        for m in range(3):
            G = a1 + a2 * gat1[m] + a3 * dat1[m]
            # blmc[m] is (*lead, nz, NY, NX) with the vertical at -3; the batched
            # value has the level axis in front, so it moves back into place.
            blmc[m][..., 0:Nr, :, :] = torch.movedim(
                hbl * w_i_all[m] * sig_all * (1.0 + sig_all * G), 0, -3
            )
        ghat[..., 0:Nr, :, :] = torch.movedim(
            (1.0 - stable) * cg / torch.clamp(ws_i_all * hbl, min=phepsi), 0, -3
        )
    else:
        for ki in range(1, Nr + 1):                   # MIT interface ki=1..Nr
            sig = (-zgrid[ki] + 0.5 * hwide[ki]) / hbl
            sigma_i = stable * sig + (1.0 - stable) * torch.clamp(sig, max=epsilon)
            wm_i, ws_i = kpp_wscale(sigma_i, hbl, ustar, bfsfc, wmt, wst, **wscale_overrides)
            w_i = [wm_i, ws_i, ws_i]
            a1, a2, a3 = sig - 2.0, 3.0 - 2.0 * sig, sig - 1.0
            for m in range(3):
                G = a1 + a2 * gat1[m] + a3 * dat1[m]
                blmc[m][..., ki - 1, :, :] = hbl * w_i[m] * sig * (1.0 + sig * G)
            ghat[..., ki - 1, :, :] = (1.0 - stable) * cg / torch.clamp(ws_i * hbl, min=phepsi)

    # ---- dkm1 at the kbl-1 grid level ----
    zg_klm1 = _gather_mit(zgrid, kbl - 1)
    sig = -zg_klm1 / hbl
    sigma_d = stable * sig + (1.0 - stable) * torch.clamp(sig, max=epsilon)
    wm_d, ws_d = kpp_wscale(sigma_d, hbl, ustar, bfsfc, wmt, wst, **wscale_overrides)
    w_d = [wm_d, ws_d, ws_d]
    a1, a2, a3 = sig - 2.0, 3.0 - 2.0 * sig, sig - 1.0
    dkm1 = torch.zeros((3,) + lead + hsp, dtype=dtype, device=device)
    for m in range(3):
        G = a1 + a2 * gat1[m] + a3 * dat1[m]
        dkm1[m] = hbl * w_d[m] * sig * (1.0 + sig * G)
    return blmc, ghat, dkm1


def _enhance_diffusivity(delta, casea, diffus_ki, blmc_ki, dkm1):
    """The three two-product sums in MITgcm ENHANCE, also used by its tangent."""
    outer = 1.0 - delta
    dkmp5 = rounded_multiply(casea, diffus_ki) + rounded_multiply(1.0 - casea, blmc_ki)
    dstar = rounded_multiply(outer.square(), dkm1) + rounded_multiply(delta.square(), dkmp5)
    return rounded_multiply(outer, diffus_ki) + rounded_multiply(delta, dstar)


def kpp_enhance(
    dkm1: torch.Tensor,
    hbl: torch.Tensor,
    kbl: torch.Tensor,
    casea: torch.Tensor,
    diffus: torch.Tensor,
    blmc: torch.Tensor,
    ghat: torch.Tensor,
    zgrid: torch.Tensor,
):
    """Enhance the diffusivity at the ``kbl-1`` interface (MITgcm ``enhance``,
    kpp_routines.F L1730-1796).  Modifies ``blmc`` and ``ghat`` at the single
    per-column interface ``ki=kbl-1`` (only where ``1 <= ki < Nr``); a smooth
    blend to the interior diffusivity, and zeroes ``ghat`` there in case B.

    Layouts carry an optional leading face-batch dim: ``diffus
    (3, *lead, nz+2, NY, NX)``, ``blmc (3, *lead, nz, NY, NX)``, ``ghat
    (*lead, nz, NY, NX)``, ``dkm1 (3, *lead, NY, NX)``.  The single per-column
    interface write uses the vertical axis ``-3`` (scatter), so a batch dim passes
    through identically.  Returns updated (blmc, ghat).
    """
    nz = diffus.shape[-3] - 2
    Nr = nz
    kbl = kbl.long()
    ki = kbl - 1                                      # MIT interface index
    valid = (ki >= 1) & (ki < Nr)
    to = torch.clamp(ki - 1, 0, nz - 1)               # LOCOMgcm index of MIT ki (blmc/ghat)
    delta = (hbl + _gather_mit(zgrid, ki)) / (_gather_mit(zgrid, ki) - _gather_mit(zgrid, ki + 1))

    # Each of the three coefficient slots is rebuilt out of place and the three
    # are stacked: assigning ``blmc[m] = ...`` back into the shared (3, ...)
    # buffer is what autograd rejects once ``blmc`` carries a gradient (the
    # gather above saved a view of that same buffer).  Values are unchanged.
    blmc_out = []
    for m in range(3):
        diffus_ki = _gather_mit(diffus[m], ki)
        blmc_ki = _kpp_gather_k(blmc[m], to)
        new = _enhance_diffusivity(delta, casea, diffus_ki, blmc_ki, dkm1[m])
        new = torch.where(valid, new, blmc_ki)        # no-op where invalid
        blmc_out.append(blmc[m].scatter(-3, to.unsqueeze(-3), new.unsqueeze(-3)))
    blmc = torch.stack(blmc_out, dim=0)

    ghat_ki = _kpp_gather_k(ghat, to)
    new_g = torch.where(valid, (1.0 - casea) * ghat_ki, ghat_ki)
    ghat = ghat.scatter(-3, to.unsqueeze(-3), new_g.unsqueeze(-3))
    return blmc, ghat


# ──────────────────────────────────────────────────────────────────────
# Combine interior (Ri_iwmix) + boundary-layer (blmix/enhance) mixing
# coefficients into the final KPP diffusivities + nonlocal term (the tail of
# MITgcm KPPMIX, kpp_routines.F L352-362).  Still columnar / op-level; the
# cell-centre mapping (KPPviscAz(k)=vddiff(k-1)*mask) is the kpp_calc producer-
# consumer boundary applied by the model, not here.
# ──────────────────────────────────────────────────────────────────────


def kpp_combine(
    blmc: torch.Tensor,
    diffus: torch.Tensor,
    ghat: torch.Tensor,
    kbl: torch.Tensor,
    *,
    visc_arnr: float,
    bg_KzS_bottom: torch.Tensor,
    bg_KzT_bottom: torch.Tensor,
):
    """Combine interior + boundary-layer mixing (MITgcm KPPMIX L352-362).

    Within the boundary layer (MIT interface ``ki < kbl``) the final diffusivity is
    ``MAX(blmc, background)`` (background = ``viscArNr(1)`` for viscosity, the
    bottom-layer ``diffusKzS(Nr)``/``diffusKzT(Nr)`` for scalars/heat); at and below
    the boundary-layer base (``ki >= kbl``) the interior Ri_iwmix value is kept and
    the nonlocal term ``ghat`` is zeroed.

    Parameters
    ----------
    blmc : Tensor ``(3, nz, *spatial)``        post-enhance BL mixing coeffs (ki=1..Nr).
    diffus : Tensor ``(3, nz+2, *spatial)``     interior diffus as blmix saw it
        (post seafloor-zero, MIT-indexed ki=0..Nrp1) — kept for ki>=kbl.
    ghat : Tensor ``(nz, *spatial)``            post-enhance nonlocal coeff (ki=1..Nr).
    kbl : Tensor ``(*spatial)`` int.
    bg_KzS_bottom, bg_KzT_bottom : Tensor ``(*spatial)``  diffusKzS/T at ki=Nr.

    Returns
    -------
    diffus_out : Tensor ``(3, nz+2, *spatial)``  combined diffusivity (MIT-indexed;
        surface ki=0 and ki>=kbl carry the interior value, ki<kbl the MAX).
    ghat_out : Tensor ``(nz, *spatial)``          ghat with ki>=kbl zeroed.
    """
    # vertical axis -3 throughout (optional leading face-batch dim).
    nz = blmc.shape[-3]
    device = blmc.device
    kbl = kbl.long()
    ki = (torch.arange(nz, device=device) + 1).reshape(nz, 1, 1)
    within = ki < kbl.unsqueeze(-3)                      # MIT ki < kbl (in boundary layer)

    bg = torch.stack([torch.full_like(bg_KzS_bottom, visc_arnr), bg_KzS_bottom, bg_KzT_bottom])
    blmc_max = torch.maximum(blmc, bg.unsqueeze(-3))     # (3, *lead, nz, NY, NX)

    diffus_out = diffus.clone()
    diffus_out[..., 1 : nz + 1, :, :] = torch.where(
        within.unsqueeze(0), blmc_max, diffus[..., 1 : nz + 1, :, :])
    ghat_out = torch.where(within, ghat, torch.zeros_like(ghat))
    return diffus_out, ghat_out


# ──────────────────────────────────────────────────────────────────────
# KPP driver — assemble the full kpp_calc preprocessing + KPPMIX from
# raw model state + surface forcing into the final cell-centred KPP diffusivities
# + nonlocal coefficient (MITgcm KPP_CALC + KPPMIX end-to-end).  This is the
# model-facing entry; it orchestrates every operator above and the input
# producers (buoyancy gradients, smoothing, scaling, friction velocity, surface
# buoyancy forcing, shear) that the op-level gates fed from dumps.  Columnar +
# per-tile horizontal stencils (the haloed slab supplies neighbours; the cross-
# face cube exchange supplies the cube halos).
# ──────────────────────────────────────────────────────────────────────


def kpp_calc_diffusivities(
    theta: torch.Tensor,
    salt: torch.Tensor,
    u_vel: torch.Tensor,
    v_vel: torch.Tensor,
    surf_forc_t: torch.Tensor,
    surf_forc_s: torch.Tensor,
    surf_forc_u: torch.Tensor,
    surf_forc_v: torch.Tensor,
    coriol: torch.Tensor,
    maskC: torch.Tensor,
    kmtj: torch.Tensor,
    layer_thickness: torch.Tensor,
    zgrid: torch.Tensor,
    hwide: torch.Tensor,
    wmt: torch.Tensor,
    wst: torch.Tensor,
    *,
    gravity: float,
    rho_const: float,
    eos_type: str,
    linear_eos: KppLinearEos | None,
    dbloc_form: str,
    use_double_diff: bool,
    Rrho0: float,
    dsfmax: float,
    layer1_thickness: float | None = None,
    Ricr: float,
    epsilon: float,
    Vtc: float,
    cekman: float,
    cmonob: float,
    minKPPhbl: float,
    vonk: float,
    Riinfty: float,
    BVSQcon: float,
    difm0: float,
    difs0: float,
    dift0: float,
    difmcon: float,
    difscon: float,
    diftcon: float,
    visc_arnr: float,
    cg: float,
    conc1: float = 5.0,
    phepsi: float = 1.0e-10,
    qsw: torch.Tensor | None = None,
    penetrating_sw: bool = True,
    sign_zero: str = "signbit",
    eos_pressure_pa: torch.Tensor | None = None,
    bg_KzS: torch.Tensor | None = None,
    bg_KzT: torch.Tensor | None = None,
    num_v_smooth_Ri: int = 0,
    heat_capacity_cp: float = 3994.0,
    kpp_estimate_uref: bool = False,
    kpp_smooth_shsq: bool = True,
    kpp_smooth_dbloc: bool = True,
    epsln: float = 1.0e-20,
    dB_dz: float = 5.2e-5,
    surface_buoyancy: str = "statekpp",
):
    """Full KPP driver: state + surface forcing -> KPP diffusivities + ghat.

    Ports MITgcm ``KPP_CALC`` (preprocessing) + ``KPPMIX`` (kpp_calc.F + kpp_routines.F)
    end-to-end.  All inputs MITgcm-convention; ``theta/salt/u_vel/v_vel/maskC`` are
    the haloed C-grid slab (``(nz, NY, NX)``; u east-face at ``i+1``, v north-face at
    ``j+1``); ``surf_forc_*`` are surfaceForcingT/S/U/V (``(NY, NX)``); ``zgrid``/``hwide``
    MIT-indexed ``(nz+2,)``; ``wmt/wst`` from :func:`kpp_wscale_tables`.

    Returns ``(KPPviscAz, KPPdiffKzS, KPPdiffKzT, KPPghat, hbl)`` — the cell-centred
    fields (``(nz, NY, NX)``; ``KappaR(k)`` = interface ``k-1`` value × maskC(k)·maskC(k-1),
    matching kpp_calc.F:639-650) and the boundary-layer depth.

    ``kpp_estimate_uref``, ``kpp_smooth_shsq``, and ``kpp_smooth_dbloc``
    mirror the corresponding ``KPP_OPTIONS.h`` switches. Their defaults are
    the pkg/kpp defaults: OFF, ON, and ON respectively.  ``sign_zero`` is the
    reference build's Fortran SIGN semantics for -0.0, applied at every SIGN
    in bldepth, z121 and blmix (:data:`KPP_SIGN_ZERO_MODES`).
    ``linear_eos`` carries the parameters of ``eos_type='linear'``
    (:class:`KppLinearEos`, read by :func:`kpp_statekpp` only).
    ``use_double_diff`` is MITgcm ``KPPuseDoubleDiff``: the background
    diffusivities gain :func:`kpp_double_diffusion` (``Rrho0``, ``dsfmax``)
    before Ri_iwmix and the boundary-layer combination read them.
    ``dbloc_form`` (required, :data:`KPP_DBLOC_FORMS`) selects the arithmetic of
    STATEKPP's adjacent-layer density difference (:func:`kpp_statekpp`); the
    CUDA STATEKPP kernel serves ``"divided_difference"`` only and declines the
    MITgcm form (``_kpp_statekpp_cuda/dispatcher.py``).
    ``surface_buoyancy`` selects the surface-relative buoyancy behind Ritop:
    ``"statekpp"`` (STATEKPP's dbsfc) or ``"divided_difference"``
    (:func:`kpp_surface_buoyancy_divided_difference`, JMD95 only).
    """
    # Vertical axis is -3 throughout; an OPTIONAL leading face-batch dim (the
    # cube face-batch) sits in ``*lead`` (() when called per-face) and passes through
    # every op below bit-for-bit identically to the per-face call.
    nz = theta.shape[-3]
    lead = theta.shape[:-3]
    hsp = theta.shape[-2:]
    spatial = lead + hsp            # 2D forcing shape (*lead, NY, NX)
    dtype, device = theta.dtype, theta.device
    kmtj = kmtj.long()
    if qsw is None:
        qsw = torch.zeros(spatial, dtype=dtype, device=device)
    if bg_KzS is None:
        bg_KzS = torch.zeros(lead + (nz,) + hsp, dtype=dtype, device=device)
    if bg_KzT is None:
        bg_KzT = torch.zeros(lead + (nz,) + hsp, dtype=dtype, device=device)

    # --- buoyancy quantities (STATEKPP) ---
    eos = _canonical_kpp_eos(eos_type)
    p = (
        eos_pressure_pa
        if eos_pressure_pa is not None
        else jmd95z_layer_pressure(layer_thickness, gravity=gravity, rho_const=rho_const)
    )
    # STATEKPP is the largest single piece of the cube KPP per-step cost and is
    # HBM-bandwidth bound (hundreds of fp64 element-wise ops, each round-tripping
    # the whole (nz, NY, NX) volume).  ``kpp_statekpp_cuda`` fuses the whole
    # per-cell map into one kernel and is BITWISE against the eager reference
    # (the kernel is compiled ``-fmad=false``; see the package header).  It is
    # jmd95-only: every other EOS family stays on the eager path below, exactly
    # as before.
    from locomgcm.accel import (
        consumer_snapshot,
        in_accelerated_dispatch,
        requested,
    )

    statekpp = None
    if eos == "jmd95" and not in_accelerated_dispatch():
        with consumer_snapshot():
            if requested(("kpp_statekpp_cuda",)):
                from locomgcm.dynamics._kpp_statekpp_cuda import dispatcher

                statekpp = dispatcher.kpp_statekpp_cuda(
                    theta, salt, p, gravity=gravity, rho_const=rho_const,
                    dbloc_form=dbloc_form,
                )
    if statekpp is None:
        statekpp = kpp_statekpp(
            theta, salt, p, gravity=gravity, rho_const=rho_const, eos_type=eos,
            dbloc_form=dbloc_form, linear_eos=linear_eos,
        )
    dbloc_raw, dbsfc_raw, rho1, alpha, beta = statekpp
    if surface_buoyancy == "divided_difference":
        if eos != "jmd95":
            raise ValueError("surface_buoyancy='divided_difference' supports JMD95 only")
        dbsfc_raw = kpp_surface_buoyancy_divided_difference(
            theta, salt, p, gravity=gravity, rho_const=rho_const)
    elif surface_buoyancy != "statekpp":
        raise ValueError(f"unknown KPP surface_buoyancy {surface_buoyancy!r}")

    # --- kpp_calc preprocessing: smoothed/masked dbloc + scaled Ritop ---
    dblocSm = _apply_kpp_smooth_dbloc(
        dbloc_raw, maskC, smooth=kpp_smooth_dbloc
    )
    kp1 = torch.clamp(torch.arange(nz, device=device) + 1, max=nz - 1)
    dbloc_masked = dbloc_raw * maskC * maskC.index_select(-3, kp1)  # nzmax auto-zeroed (maskC[kp1]=0 / DBLOC(Nr)=0)
    kc = torch.arange(nz, device=device).reshape(nz, 1, 1)
    scale = (zgrid[1] - zgrid[1 : nz + 1]).reshape(nz, 1, 1)        # zgrid(1)-zgrid(k)
    Ritop = scale * dbsfc_raw * maskC * maskC[..., 0:1, :, :]
    Ritop = torch.where(kc + 1 >= kmtj.unsqueeze(-3), torch.zeros_like(Ritop), Ritop)   # nzmax zero

    # --- surface forcing (ustar, buoyancy forcing) ---
    ustar = kpp_surface_friction_velocity(
        surf_forc_u,
        surf_forc_v,
        # Host float from ``KppStatic`` when the caller has it (no D2H
        # sync in the step body); ``float(tensor[0])`` is the historical spelling.
        layer1_thickness=(
            layer1_thickness
            if layer1_thickness is not None
            else float(layer_thickness[0])
        ),
        phepsi=phepsi
    )
    bo, bosol = kpp_surface_buoyancy_forcing(
        alpha[..., 0, :, :], beta[..., 0, :, :], surf_forc_t, surf_forc_s, rho1,
        gravity=gravity, qsw=qsw, heat_capacity_cp=heat_capacity_cp, rho_const=rho_const,
    )
    # ustar/bo/bosol live on the t-cell interior (slab ring left unfilled); the
    # surface friction velocity 2D field is computed on the same haloed slab.

    # --- shear ---
    if kpp_estimate_uref:
        dVsq = kpp_relative_velocity_shear_estimate_uref(
            u_vel, v_vel, dbloc_raw, ustar, surf_forc_u, surf_forc_v,
            layer_thickness, maskC, kmtj,
            vonk=vonk, epsilon=epsilon, phepsi=phepsi, epsln=epsln,
            dB_dz=dB_dz,
        )
    else:
        dVsq = kpp_relative_velocity_shear(u_vel, v_vel)
    shsq = kpp_local_velocity_shear(u_vel, v_vel, smooth=kpp_smooth_shsq)

    # --- double diffusion joins the background before KPPMIX (kpp_calc.F:532-554) ---
    if use_double_diff:
        bg_KzT, bg_KzS = kpp_double_diffusion(
            theta, salt, alpha, beta, bg_KzT, bg_KzS, Rrho0=Rrho0, dsfmax=dsfmax,
        )

    # --- bldepth ---
    hbl, kbl, bfsfc, stable, casea, _Rib = kpp_bldepth(
        kmtj, dVsq, dbloc_masked, Ritop, ustar, bo, bosol, coriol, zgrid, hwide, wmt, wst,
        Ricr=Ricr, epsilon=epsilon, Vtc=Vtc, cekman=cekman, cmonob=cmonob,
        minKPPhbl=minKPPhbl, vonk=vonk, phepsi=phepsi,
        penetrating_sw=penetrating_sw, sign_zero=sign_zero,
    )

    # --- interior mixing (Ri_iwmix) -> diffus(3, nz+2) MIT-indexed, post seafloor-zero ---
    ri_args = (kmtj, shsq, dbloc_masked, dblocSm, bg_KzS, bg_KzT, zgrid)
    ri_kwargs = dict(
        Riinfty=Riinfty, BVSQcon=BVSQcon, difm0=difm0, difs0=difs0, dift0=dift0,
        difmcon=difmcon, difscon=difscon, diftcon=diftcon, visc_arnr=visc_arnr,
        phepsi=phepsi, num_v_smooth_Ri=num_v_smooth_Ri, sign_zero=sign_zero,
    )
    visc, KzS, KzT = kpp_ri_iwmix(*ri_args, **ri_kwargs)
    diffus = torch.zeros((3,) + lead + (nz + 2,) + hsp, dtype=dtype, device=device)
    diffus[0][..., 1 : nz + 1, :, :] = visc
    diffus[1][..., 1 : nz + 1, :, :] = KzS
    diffus[2][..., 1 : nz + 1, :, :] = KzT
    below = (kc + 1 >= kmtj.unsqueeze(-3))                  # seafloor-zero: ki>=kmtj -> 0
    diffus[..., 1 : nz + 1, :, :] = torch.where(
        below.unsqueeze(0), torch.zeros_like(diffus[..., 1 : nz + 1, :, :]),
        diffus[..., 1 : nz + 1, :, :])

    # --- boundary-layer mixing (blmix) + enhance + combine ---
    bl_args = (ustar, bfsfc, hbl, stable, casea, kbl, diffus, zgrid, hwide, wmt, wst)
    bl_kwargs = dict(epsilon=epsilon, conc1=conc1, cg=cg, phepsi=phepsi, sign_zero=sign_zero)
    blmc, ghat, dkm1 = kpp_blmix(*bl_args, **bl_kwargs)
    blmc, ghat = kpp_enhance(dkm1, hbl, kbl, casea, diffus, blmc, ghat, zgrid)
    diffus_out, ghat_out = kpp_combine(
        blmc, diffus, ghat, kbl, visc_arnr=visc_arnr,
        bg_KzS_bottom=bg_KzS[..., nz - 1, :, :], bg_KzT_bottom=bg_KzT[..., nz - 1, :, :],
    )

    # --- cell-centre map: KappaR(k) = vddiff(k-1) * maskC(k)*maskC(km1) ---
    km1 = torch.clamp(torch.arange(nz, device=device) - 1, min=0)
    cellmask = maskC * maskC.index_select(-3, km1)          # (*lead, nz, NY, NX)
    KPPviscAz = diffus_out[0][..., 0:nz, :, :] * cellmask
    KPPdiffKzS = diffus_out[1][..., 0:nz, :, :] * cellmask
    KPPdiffKzT = diffus_out[2][..., 0:nz, :, :] * cellmask
    KPPghat = ghat_out * cellmask
    return KPPviscAz, KPPdiffKzS, KPPdiffKzT, KPPghat, hbl
