"""Surface heat and salt flux forcing for tracers.

Implements MITgcm's ``external_forcing.F`` L229-250 algorithm for
temperature and salinity surface forcing, applied only to the
topmost wet layer with proper ``hFacC`` normalization.

Two modes are supported:
1. **Net flux mode**: directly apply prescribed Q_net [W/m²] and
   salt_flux [PSU·m/s].
2. **SST restoring mode**: ``Q_net = λ · (T_obs - T_model)`` with
   a relaxation rate λ [W/m²/K].

Reference
---------
MITgcm ``external_forcing.F`` L229-250::

    gT(i,j,1) = gT(i,j,1) + surfaceForcingT(i,j)
                * recip_drF(1) * _recip_hFacC(i,j,1)

where ``surfaceForcingT`` has units [K·m/s] = Q_net / (ρ₀ · Cp).
"""
from __future__ import annotations

from locomgcm.sampling import SampleClock, SamplingRule

from dataclasses import dataclass, field

import torch

from locomgcm.geometry import StructuredCGrid
from locomgcm.geometry.grid import _compute_interior_mask, runtime_memo_constant, top_layer_thickness_value
from locomgcm.state import DiagnosticState, PrognosticState, StateTendency
from locomgcm.constants import RHO_CONST_DEFAULT

from .base import TendencyModule, TendencyPhase, TendencyStepDomain
from locomgcm.dynamics._registry import register_dynamics_module


# ---------------------------------------------------------------------------
# Penetrating shortwave radiation (MITgcm ``swfrac.F`` + ``external_forcing.F``
# L527-551, ``SHORTWAVE_HEATING``). The Jerlov profile is shared by this
# temperature tendency and the KPP ``bosol``/``swfrac`` calculations.
# ---------------------------------------------------------------------------
# Jerlov water-type table (Paulson & Simpson 1977), ``swfrac.F`` L74-76:
#   jwtype:     1(I)   2(IA)  3(IB)  4(II)  5(III)
_JERLOV_RFAC = (0.58, 0.62, 0.67, 0.77, 0.78)
_JERLOV_A1 = (0.35, 0.6, 1.0, 1.5, 1.4)
_JERLOV_A2 = (23.0, 20.0, 17.0, 14.0, 7.9)


def swfrac(depth_m: torch.Tensor, jwtype: int = 2) -> torch.Tensor:
    """Fraction of surface shortwave flux that penetrates to ``depth_m``.

    Ports MITgcm ``swfrac.F`` L97-105 (two-band exponential decay)::

        facz = fact * swdk = (-1) * abs(rF) = -depth        # <= 0
        swdk = rfac*exp(facz/a1) + (1-rfac)*exp(facz/a2)    # 0 below 200 m

    Parameters
    ----------
    depth_m : Tensor
        Depth below the surface in metres, **positive downward** (LOCOMgcm
        ``interface_depth`` convention, ``= abs(rF)``).  Any shape.
    jwtype : int
        Jerlov water type 1..5.  Default 2 (type IA), matching MITgcm's
        hardcoded ``jwtype=2`` (``swfrac.F`` L92-94); exposed for a future
        spatial map.

    Returns
    -------
    Tensor
        Penetrating fraction, same shape as ``depth_m``: 1.0 at the surface
        (depth 0), 0.0 below 200 m.
    """
    rfac = _JERLOV_RFAC[jwtype - 1]
    a1 = _JERLOV_A1[jwtype - 1]
    a2 = _JERLOV_A2[jwtype - 1]
    facz = -depth_m
    frac = rfac * torch.exp(facz / a1) + (1.0 - rfac) * torch.exp(facz / a2)
    # swfrac.F L99-100: below 200 m the penetration is forced to zero.
    return torch.where(facz < -200.0, torch.zeros_like(frac), frac)


def shortwave_penetration_tendency(
    qsw: torch.Tensor,
    interface_depth: torch.Tensor,
    layer_thickness: torch.Tensor,
    hFacC: torch.Tensor,
    *,
    heat_capacity: float = 3994.0,
    rho_const: float = RHO_CONST_DEFAULT,
    jwtype: int = 2,
) -> torch.Tensor:
    """Per-layer temperature tendency from penetrating shortwave radiation.

    Ports the MITgcm **active** forcing path (``USE_OLD_EXTERNAL_FORCING`` undef):
    ``apply_forcing.F`` L683-695 (the ``APPLY_FORCING_T`` ``SHORTWAVE_HEATING``
    block) over the precomputed ``SWFrac3D`` from ``ini_forcing.F`` L149-187,
    with the Jerlov profile from ``swfrac.F``::

        SWFrac3D(k) = swfrac(rF(k)) * maskC(MIN(k,Nr)) ; SWFrac3D(Nr+1)=0
        gT(k) += Qsw*gravitySign*(SWFrac3D(k) - SWFrac3D(k+1))
                 *recip_Cp*mass2rUnit*recip_drF(k)*recip_hFacC(k)

    ``gravitySign=-1`` for z-coords (ini_vertical_grid.F L54) => the ``-Qsw``;
    ``recip_Cp*mass2rUnit = 1/(Cp*rhoConst)`` (set_defaults.F L174
    ``HeatCapacity_Cp=3994``; ini_parms.F L1573 ``mass2rUnit=recip_rhoConst``).
    The two-mask form computed below, ``swfracb1*maskC(k) - swfracb2*maskC(k+1)``,
    is algebraically (and bitwise at op level) identical to the baked
    ``SWFrac3D(k)-SWFrac3D(k+1)`` difference.  Shortwave heats *every* layer it
    reaches; partial/interior seafloors zero the down-flux via ``maskC(k+1)``,
    the absolute bottom (k=Nr) via ``SWFrac3D(Nr+1)=0`` (ini_forcing.F L174).

    Sign convention (docs/NUMERICS.md): ``qsw`` is **MITgcm
    upward-positive** (``Qsw>0`` cools), matching ``kpp.py`` ``bosol``.  The
    producer negates the LOCOMgcm downward-positive ``shortwave_down`` once at the
    data boundary; a wrong sign here is op-level invisible but blows up
    a multi-step run.

    Parameters
    ----------
    qsw : Tensor ``[..., ny, nx]``
        Net surface shortwave flux [W/m^2], MITgcm upward-positive.
    interface_depth : Tensor ``[nz+1]``
        Layer-interface depths [m], positive downward (``= abs(rF)``).
    layer_thickness : Tensor ``[nz]``
        Layer thickness ``drF`` [m], positive.
    hFacC : Tensor ``[..., nz, ny, nx]``
        Centre-cell open fraction (``maskC = hFacC>0``).  Grid-agnostic: ``[nz,
        ny, nx]`` (regional), ``[6, nz, ny, nx]`` (cube, single/multi-card), or
        ``[B, nz, ny, nx]``; ``qsw`` must carry the matching leading dims.
    heat_capacity, rho_const : float
        ``Cp`` and ``rhoConst``.
    jwtype : int
        Jerlov water type (default 2).

    Returns
    -------
    Tensor ``[..., nz, ny, nx]``
        Shortwave contribution to the temperature tendency [K/s].
    """
    dtype = hFacC.dtype
    device = hFacC.device
    nz = int(layer_thickness.shape[0])
    iface = interface_depth.to(dtype=dtype, device=device)

    swfracb1 = swfrac(iface[:nz], jwtype)            # [nz] top interface of layer k
    swfracb2 = swfrac(iface[1:nz + 1], jwtype).clone()  # [nz] bottom interface
    swfracb2[nz - 1] = 0.0                           # external_forcing.F L539: k=Nr

    # Layout-agnostic over the vertical axis dim=-3: hFacC may be [nz, ny, nx]
    # (regional/lat-lon), [6, nz, ny, nx] (cube, single- or multi-card face-local),
    # or [B, nz, ny, nx].  The k+1 shift and the swfracb broadcast both act on the
    # nz axis only, so this term is per-column and needs no halo / cross-rank comm.
    maskC = (hFacC > 0.0).to(dtype=dtype)
    maskC_kp1 = torch.cat(
        [maskC[..., 1:, :, :], torch.zeros_like(maskC[..., :1, :, :])], dim=-3
    )

    absorbed = swfracb1.view(nz, 1, 1) * maskC - swfracb2.view(nz, 1, 1) * maskC_kp1

    recip_drF = (1.0 / layer_thickness.to(dtype=dtype, device=device)).view(nz, 1, 1)
    recip_hFacC = torch.where(
        hFacC > 0.0, 1.0 / hFacC.clamp(min=1e-14), torch.zeros_like(hFacC)
    )
    recip_cp = 1.0 / heat_capacity                   # MITgcm recip_Cp
    mass2runit = 1.0 / rho_const                      # MITgcm mass2rUnit (z: recip_rhoConst)

    qsw_nz = qsw.to(dtype=dtype, device=device).unsqueeze(-3)  # [..., 1, ny, nx]
    # Factor order mirrors apply_forcing.F:686-689 left-to-right (=> bitwise).
    return -qsw_nz * absorbed * recip_cp * mass2runit * recip_drF * recip_hFacC


def surface_heat_forcing(
    qnet: torch.Tensor,
    qsw: torch.Tensor,
    *,
    heat_capacity: float = 3994.0,
    rho_const: float = RHO_CONST_DEFAULT,
    penetrating_sw: bool = True,
) -> torch.Tensor:
    r"""Surface (non-shortwave) heat forcing field ``surfaceForcingT`` [K·m/s].

    Ports MITgcm ``external_forcing_surf.F`` L226-231 (no surface relaxation)::

        surfaceForcingT = -( Qnet - Qsw*tmpFac ) * recip_Cp * mass2rUnit

    with ``tmpFac=1`` when ``selectPenetratingSW>0`` (``SHORTWAVE_HEATING``): the
    shortwave ``Qsw`` is removed from the surface here and redistributed over depth
    by :func:`shortwave_penetration_tendency`.  ``recip_Cp=1/HeatCapacity_Cp``
    (``external_forcing_surf.F`` L79), ``mass2rUnit=1/rhoConst`` (z-coords).

    This **field** [K·m/s] is the single quantity consumed by *both*
    (a) the dynamics surface apply ``gT(1) += surfaceForcingT·recip_drF·recip_hFacC``
    (``apply_forcing.F``) and (b) KPP ``bo`` (``sfT``, ``kpp_mixing.py`` L13); the
    shortwave part is handled separately (penetration term + KPP ``bosol``).

    Sign (docs/NUMERICS.md): ``qnet``/``qsw`` are **MITgcm upward-positive**
    (``Qnet>0`` cools, ``Qsw>0`` cools); the producer converts the LOCOMgcm
    downward-positive fluxes once at the data boundary.  A wrong sign is op-level
    invisible but blows up a multi-step run.

    Parameters
    ----------
    qnet, qsw : Tensor ``[..., ny, nx]``
        Net surface heat flux and net surface shortwave [W/m^2], MIT upward-positive.
    heat_capacity, rho_const : float
        ``Cp`` and ``rhoConst``.
    penetrating_sw : bool
        ``selectPenetratingSW>0`` ⟹ remove ``Qsw`` from the surface term (default
        True).  False folds all shortwave into the surface layer (no penetration).

    Returns
    -------
    Tensor
        ``surfaceForcingT`` [K·m/s], same shape as ``qnet``.
    """
    tmp_fac = 1.0 if penetrating_sw else 0.0
    recip_cp = 1.0 / heat_capacity
    mass2runit = 1.0 / rho_const
    # Factor order mirrors external_forcing_surf.F:226-231 (=> bitwise).
    return -((qnet - qsw * tmp_fac) * recip_cp * mass2runit)


def surface_salt_forcing(
    empmr: torch.Tensor,
    *,
    salt_flux: torch.Tensor | float = 0.0,
    convert_fw2salt: float = 35.0,
    salt_evprn: float = 0.0,
    rho_const: float = RHO_CONST_DEFAULT,
    surface_salinity: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Surface salt forcing field ``surfaceForcingS`` [psu·m/s] (virtual salt flux).

    Ports MITgcm ``external_forcing_surf.F`` L233-234 + L296-319 + L336-343.
   The uniform virtual-salt-flux branch is::

        surfaceForcingS = -saltFlux*mass2rUnit
                          + EmPmR*(convertFW2Salt - salt_EvPrRn)*mass2rUnit

    When ``convertFW2Salt=-1``, MITgcm instead uses the current local surface
    salinity::

        surfaceForcingS = -saltFlux*mass2rUnit
                          + EmPmR*(salt(surface) - salt_EvPrRn)*mass2rUnit

    where ``EmPmR`` [kg/m²/s] is the net freshwater flux (evaporation−precip,
    ``>0`` = net evaporation), ``convertFW2Salt`` [g/kg] the reference salinity of
    the virtual flux (default 35), ``mass2rUnit=1/rhoConst``. The freshwater→
    salinity effect is this virtual salt flux; the linFS volume-conservation
    ``SsurfCor`` global surface correction in :class:`SurfaceHeatSaltFlux`
    is a separate term.

    Sign (docs/NUMERICS.md): ``EmPmR>0`` increases salinity in BOTH LOCOMgcm and MITgcm
    (direct, no flip — net evaporation concentrates salt).  ``saltFlux>0``
    DECREASES salinity in MITgcm (opposite to LOCOMgcm), so a prescribed ``salt_flux``
    must be passed in MITgcm convention (it enters as ``-saltFlux`` here); a
    wrong sign is op-level invisible but blows up a multi-step run.
    ``surfaceForcingS`` feeds the dynamics salt apply (``gS(1)+=···``) and KPP
    ``bo`` (``sfS``).

    Parameters
    ----------
    empmr : Tensor ``[..., ny, nx]``
        Net freshwater flux EmPmR [kg/m²/s] (MITgcm global; = exf flux × rhoConstFresh).
    salt_flux : Tensor or float
        Prescribed salt flux [psu·kg/m²/s] in MITgcm convention (``saltFlux>0``
        cools salinity).  Default 0 (no prescribed salt flux).
    convert_fw2salt, salt_evprn : float
        ``convertFW2Salt`` and ``salt_EvPrRn`` [g/kg].
    rho_const : float
        ``rhoConst``.
    surface_salinity : Tensor or None
        Current local surface salinity [g/kg], required when
        ``convert_fw2salt == -1``. Must have the same shape as ``empmr``.

    Returns
    -------
    Tensor
        ``surfaceForcingS`` [psu·m/s], same shape as ``empmr``.
    """
    mass2runit = 1.0 / rho_const
    if convert_fw2salt == -1.0:
        if surface_salinity is None:
            raise ValueError(
                "surface_salinity is required when convert_fw2salt == -1"
            )
        local_salinity = surface_salinity.to(
            dtype=empmr.dtype, device=empmr.device
        )
        salt_evprn_local = torch.as_tensor(
            salt_evprn, dtype=empmr.dtype, device=empmr.device
        )
        mass2runit_local = torch.as_tensor(
            mass2runit, dtype=empmr.dtype, device=empmr.device
        )
        # Exact factor order from external_forcing_surf.F:313-316.
        local_term = (
            empmr * (local_salinity - salt_evprn_local) * mass2runit_local
        )
        if isinstance(salt_flux, torch.Tensor) or salt_flux != 0.0:
            # external_forcing_surf.F:233-234 precedes the local addition.
            s = -salt_flux * mass2runit
            s = s + local_term
        else:
            s = local_term
    else:
        # Factor order mirrors external_forcing_surf.F:340-343 (=> bitwise).
        s = empmr * (convert_fw2salt - salt_evprn) * mass2runit
        if isinstance(salt_flux, torch.Tensor) or salt_flux != 0.0:
            s = s - salt_flux * mass2runit      # Pre-uniform path: bitwise.
    return s


def surface_salt_forcing_real_freshwater(
    pmep_r: torch.Tensor,
    *,
    salt_flux: torch.Tensor | float = 0.0,
    salt_evprn: float = 0.0,
    rho_const: float = RHO_CONST_DEFAULT,
    surface_salinity: torch.Tensor,
) -> torch.Tensor:
    r"""``surfaceForcingS`` [psu·m/s] of a nonlinear free surface with real freshwater flux.

    MITgcm ``external_forcing_surf.F:233-234`` and ``:261-288`` (the
    ``nonlinFreeSurf>0 .AND. useRealFreshWaterFlux`` branch)::

        surfaceForcingS = -saltFlux*mass2rUnit
                          + PmEpR*(salt_EvPrRn - salt(surface))*mass2rUnit

    The freshwater changes the column height, so EmPmR enters no virtual salt
    flux; the term is the salt the water brings in at ``salt_EvPrRn`` less the
    dilution of the thicker column.  ``PmEpR`` [kg/m²/s] is ``-EmPmR*maskInC``:
    the previous continuity's under synchronous stepping
    (``integr_continuity.F:173-178``; zero on the first forward step of a cold
    start), the current step's under staggerTimeStep
    (``external_forcing_surf.F:158-166``).  The stepper chooses it
    (``ImplicitFreeSurfaceStepper._pmep_r_for_step``).  The temperature
    counterpart needs ``temp_EvPrRn``, which MITgcm leaves unset by default and
    LOCOMgcm does not carry.
    """
    dtype, device = pmep_r.dtype, pmep_r.device
    mass2runit = torch.as_tensor(1.0 / rho_const, dtype=dtype, device=device)
    salt_evprn_local = torch.as_tensor(salt_evprn, dtype=dtype, device=device)
    local_salinity = surface_salinity.to(dtype=dtype, device=device)
    # Factor order of external_forcing_surf.F:283-285.
    volume_term = pmep_r * (salt_evprn_local - local_salinity) * mass2runit
    if isinstance(salt_flux, torch.Tensor) or salt_flux != 0.0:
        return -salt_flux * mass2runit + volume_term
    return volume_term


@dataclass(frozen=True)
class SurfaceForcing:
    """Bundle of MITgcm surfaceForcing fields from one producer call (single source).

    All MITgcm conventions: ``sfT`` [K·m/s], ``sfS`` [psu·m/s], ``sfU``/``sfV``
    [m²/s²-equiv], ``qsw`` [W/m²] upward-positive.  Both the dynamics surface apply
    (:func:`apply_wpflux_surface_tendency`) and the KPP forcing harvest read THIS
    bundle, so the two consumers never re-derive the fields independently (no
    dual-entry drift; :func:`surface_forcing_from_fluxes` is the ONLY sfT/sfS
    computation point).

    ``sfU``/``sfV`` are ``None`` for tracer-only producers (era5_bulk: momentum
    stays with the wind path).  The raw ``qnet``/``empmr`` fields and conversion
    constants ride along for the legacy cube hand-off; its KPP consumer still
    reads the already-produced ``sfT``/``sfS`` rather than re-deriving them.
    """

    sfT: torch.Tensor                 # surfaceForcingT
    sfS: torch.Tensor                 # surfaceForcingS
    qsw: torch.Tensor                 # net shortwave (pass-through: penetration + KPP bosol)
    sfU: torch.Tensor | None = None   # surfaceForcingU = cgrid_taux * mass2rUnit
    sfV: torch.Tensor | None = None   # surfaceForcingV
    rho_const: float = RHO_CONST_DEFAULT
    heat_capacity: float = 3994.0
    qnet: torch.Tensor | None = None  # raw MIT Qnet, for the cube cache hand-off
    empmr: torch.Tensor | None = None  # raw bulk freshwater volume flux [m/s]
    rho_const_fresh: float = RHO_CONST_DEFAULT
    convert_fw2salt: float = 35.0
    salt_evprn: float = 0.0
    penetrating_sw: bool = True  # physical mode, independent of instantaneous Qsw


@dataclass(frozen=True)
class PrescribedSurfaceFluxSource:
    """Time-interpolated LOCOMgcm-convention heat/freshwater fields at tracer centres.

    ``heat_flux`` is positive into the ocean.  ``freshwater_flux`` retains the
    core MITgcm ``EmPmRFile`` volume-flux bytes [m/s], positive for evaporation;
    :func:`surface_forcing_from_bulk` performs the one ``rhoConstFresh`` mass
    conversion when the module consumes a sample.  ``sample`` also returns the
    net shortwave ``Qsw`` (MITgcm upward-positive), zero for this source.
    """

    sampling_rule = SamplingRule(SampleClock.START)

    times: torch.Tensor
    time_values: tuple[str, ...]
    heat_flux: torch.Tensor
    freshwater_flux: torch.Tensor
    period_seconds: float | None = None

    def sample(
        self, model_time_seconds: float | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = torch.as_tensor(
            model_time_seconds, dtype=self.times.dtype, device=self.times.device
        )
        if self.period_seconds is not None and self.period_seconds > 0.0:
            t = t % self.period_seconds
        t = torch.clamp(t, self.times[0], self.times[-1])
        upper = torch.clamp(
            torch.searchsorted(self.times, t.unsqueeze(0), right=False).squeeze(),
            0,
            self.times.shape[0] - 1,
        )
        lower = torch.clamp(upper - 1, 0, self.times.shape[0] - 1)
        t_low, t_high = self.times[lower], self.times[upper]
        diff = torch.where(t_high == t_low, torch.ones_like(t_high), t_high - t_low)
        weight = torch.where(
            t_high == t_low, torch.zeros_like(diff), (t - t_low) / diff
        )
        heat_flux = (
            (1.0 - weight) * self.heat_flux[lower]
            + weight * self.heat_flux[upper]
        )
        freshwater_flux = (
            (1.0 - weight) * self.freshwater_flux[lower]
            + weight * self.freshwater_flux[upper]
        )
        return heat_flux, freshwater_flux, torch.zeros_like(heat_flux)

    def to_runtime(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> "PrescribedSurfaceFluxSource":
        tensor_dtype = self.heat_flux.dtype if dtype is None else dtype
        return PrescribedSurfaceFluxSource(
            times=self.times.to(device=device, dtype=torch.float64),
            time_values=self.time_values,
            heat_flux=self.heat_flux.to(device=device, dtype=tensor_dtype),
            freshwater_flux=self.freshwater_flux.to(
                device=device, dtype=tensor_dtype
            ),
            period_seconds=self.period_seconds,
        )


def surface_forcing_from_fluxes(
    qnet: torch.Tensor,
    qsw: torch.Tensor,
    empmr: torch.Tensor,
    *,
    fu: torch.Tensor | None,
    fv: torch.Tensor | None,
    salt_flux: torch.Tensor | float,
    heat_capacity: float,
    rho_const: float,
    convert_fw2salt: float,
    salt_evprn: float,
    penetrating_sw: bool,
    surface_salinity: torch.Tensor | None,
    pmep_r: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """MITgcm surface fluxes -> ``(surfaceForcingT, surfaceForcingS, surfaceForcingU, surfaceForcingV)``.

    The one place the surfaceForcing fields are formed from the FFIELDS.h fluxes
    (``external_forcing_surf.F:222-234,256-345``); the dynamics apply and KPP
    both read its output.  All inputs are MITgcm conventions: ``qnet``/``qsw``
    [W/m2] upward-positive (positive cools), ``empmr`` the MITgcm ``EmPmR`` mass
    flux [kg/m2/s] (positive = net evaporation, increases salinity),
    ``salt_flux`` the MITgcm ``saltFlux`` [g/m2/s] (positive upward, decreases
    salinity), ``fu``/``fv`` the stress [N/m2] on the C-grid U/V faces.
    ``fu``/``fv`` are ``None`` for a producer that leaves momentum to another
    module; the returned ``surfaceForcingU/V`` are then ``None``.  ``pmep_r``
    selects the salt branch as in :func:`surface_salt_forcing_real_freshwater`.
    """
    sfT = surface_heat_forcing(qnet, qsw, heat_capacity=heat_capacity,
                               rho_const=rho_const, penetrating_sw=penetrating_sw)
    if pmep_r is not None:
        if surface_salinity is None:
            raise ValueError("the real-freshwater salt forcing needs the surface salinity")
        sfS = surface_salt_forcing_real_freshwater(
            pmep_r, salt_flux=salt_flux, salt_evprn=salt_evprn, rho_const=rho_const,
            surface_salinity=surface_salinity,
        )
    else:
        sfS = surface_salt_forcing(empmr, salt_flux=salt_flux,
                                   convert_fw2salt=convert_fw2salt, salt_evprn=salt_evprn,
                                   rho_const=rho_const,
                                   surface_salinity=surface_salinity)
    mass2runit = 1.0 / rho_const
    sfU = fu * mass2runit if fu is not None else None
    sfV = fv * mass2runit if fv is not None else None
    return sfT, sfS, sfU, sfV


def surface_forcing_from_bulk(
    qnet: torch.Tensor,
    qsw: torch.Tensor,
    empmr: torch.Tensor,
    cgrid_taux: torch.Tensor | None = None,
    cgrid_tauy: torch.Tensor | None = None,
    *,
    salt_flux: torch.Tensor | float = 0.0,
    heat_capacity: float = 3994.0,
    rho_const: float = RHO_CONST_DEFAULT,
    rho_const_fresh: float = RHO_CONST_DEFAULT,
    convert_fw2salt: float = 35.0,
    salt_evprn: float = 0.0,
    penetrating_sw: bool = True,
    surface_salinity: torch.Tensor | None = None,
    pmep_r: torch.Tensor | None = None,
) -> SurfaceForcing:
    """Bulk fluxes (+ C-grid wind stress) -> the :class:`SurfaceForcing` bundle.

   The convention boundary (docs/NUMERICS.md): the bulk outputs are ALREADY MITgcm
    upward-positive (``Qnet``/``Qsw``>0 cool, ``empmr``>0 = net evaporation) because
    the real-data ingest negates the LOCOMgcm downward-positive data once, upstream.  The
    fields themselves come from :func:`surface_forcing_from_fluxes`, so the dynamics
    apply and KPP read identical fields.

    EmPmR unit: the bulk ``empmr`` is the EXF freshwater VOLUME flux [m/s]; the MITgcm
    ``EmPmR`` global is a MASS flux [kg/m²/s] = ``empmr × rhoConstFresh`` (EXF_MAPFIELDS;
    verified bitwise on mit_cs32_exf) — the ONLY place that conversion happens.
    Momentum: ``cgrid_tau{x,y}`` is the wind stress already staggered to the C-grid
    u/v faces (the wind ingest owns the staggering — MIT ``fu``/``fv``, which differ
    from the raw exf ustress/vstress); omit for tracer-only producers.

    Parameters
    ----------
    qnet, qsw : Tensor
        Net heat / net shortwave [W/m²], MITgcm upward-positive.
    empmr : Tensor
        Bulk freshwater flux [m/s] (= exf flux; ``BulkFluxes.empmr``).
    cgrid_taux, cgrid_tauy : Tensor or None
        Wind stress on the C-grid u/v faces [N/m²] (already staggered); ``None``
        for tracer-only producers (bundle carries ``sfU=sfV=None``).
    salt_flux : Tensor or float
        Prescribed salt flux [psu·kg/m²/s] in MITgcm convention (default 0).
    heat_capacity, rho_const, rho_const_fresh, convert_fw2salt, salt_evprn : float
        MITgcm constants (see :func:`surface_heat_forcing` / :func:`surface_salt_forcing`).
    surface_salinity : Tensor or None
        Current local surface salinity, required for ``convert_fw2salt=-1``.
    penetrating_sw : bool
        ``selectPenetratingSW>0`` (default True).
    pmep_r : Tensor or None
        ``PmEpR`` [kg/m²/s] of a nonlinear free surface with real freshwater
        flux; the salt forcing is then
        :func:`surface_salt_forcing_real_freshwater` and ``empmr`` adds no
        virtual salt flux.  ``None`` is the virtual-flux branch
        (``external_forcing_surf.F:290-345``); the caller decides from the
        resolved model configuration.

    Returns
    -------
    SurfaceForcing
        sfT, sfS, qsw (+ sfU/sfV, rho_const, heat_capacity) — the single-source bundle.
    """
    sfT, sfS, sfU, sfV = surface_forcing_from_fluxes(
        qnet, qsw, empmr * rho_const_fresh, fu=cgrid_taux, fv=cgrid_tauy,
        salt_flux=salt_flux, heat_capacity=heat_capacity, rho_const=rho_const,
        convert_fw2salt=convert_fw2salt, salt_evprn=salt_evprn,
        penetrating_sw=penetrating_sw, surface_salinity=surface_salinity, pmep_r=pmep_r,
    )
    return SurfaceForcing(
        sfT=sfT,
        sfS=sfS,
        qsw=qsw,
        sfU=sfU,
        sfV=sfV,
        rho_const=rho_const,
        heat_capacity=heat_capacity,
        qnet=qnet,
        empmr=empmr,
        rho_const_fresh=rho_const_fresh,
        convert_fw2salt=convert_fw2salt,
        salt_evprn=salt_evprn,
        penetrating_sw=penetrating_sw,
    )


def apply_wpflux_surface_tendency(
    forcing: SurfaceForcing,
    state: PrognosticState,
    grid: StructuredCGrid,
    *,
    jwtype: int = 2,
) -> StateTendency:
    """Apply a :class:`SurfaceForcing` bundle as MIT surface tracer forcing.

    Single source of truth for the surface apply math — non-shortwave
    surface heat (k=0) + shortwave penetration over depth + virtual salt flux::

        gT(1) += sfT · recip_drF · recip_hFacC
        gT(k) += shortwave_penetration_tendency(qsw, …)
        gS(1) += sfS · recip_drF · recip_hFacC

    This function only APPLIES: ``sfT``/``sfS`` come pre-computed from the one
    producer :func:`surface_forcing_from_bulk` (consumers never re-derive
    them), and the penetration term reads the bundle's
    ``rho_const``/``heat_capacity`` so apply and produce cannot disagree on
    constants.  Shared by :class:`WPFluxSurfaceForcing` (static fields) and
    :class:`~locomgcm.dynamics.era5_bulk_forcing.ERA5BulkForcing` (per-step
    fields from bulk formulae + model SST).  All bundle fields are MITgcm
    upward-positive (docs/NUMERICS.md).  Per-cell / per-column → grid- and
    parallelism-agnostic (regional ``[B,nz,ny,nx]`` and cube ``[6,nz,ny,nx]``).
    """
    dtype = state.temp.data.dtype
    device = state.temp.data.device

    drF = grid.layer_thickness.to(dtype=dtype, device=device)            # [nz]
    hFacC = grid.masks.hFacC.to(dtype=dtype, device=device)
    if hFacC.ndim == state.temp.data.ndim - 1:                            # [nz,ny,nx]
        hFacC = hFacC.unsqueeze(0)                                        # -> [1,nz,ny,nx]
    recip_hFacC = torch.where(hFacC > 0, 1.0 / hFacC.clamp(min=1e-14), torch.zeros_like(hFacC))
    recip_drF0 = 1.0 / drF[0]

    sfT = forcing.sfT.to(dtype=dtype, device=device)
    sfS = forcing.sfS.to(dtype=dtype, device=device)
    qsw = forcing.qsw.to(dtype=dtype, device=device)

    # Heat: non-shortwave surface (k=0) + shortwave penetration over depth.
    temp_tendency = torch.zeros_like(state.temp.data)
    temp_tendency[:, 0] = sfT * recip_drF0 * recip_hFacC[:, 0]
    if forcing.penetrating_sw:
        iface = torch.cat([drF.new_zeros(1), torch.cumsum(drF, 0)])
        temp_tendency = temp_tendency + shortwave_penetration_tendency(
            qsw, iface, drF, hFacC, heat_capacity=forcing.heat_capacity,
            rho_const=forcing.rho_const, jwtype=jwtype,
        )

    # Salt: surface (k=0) virtual salt flux (sfS pre-computed by the producer).
    salt_tendency = torch.zeros_like(state.salt.data)
    salt_tendency[:, 0] = sfS * recip_drF0 * recip_hFacC[:, 0]

    return StateTendency(
        eta=torch.zeros_like(state.eta.data),
        u=torch.zeros_like(state.u.data),
        v=torch.zeros_like(state.v.data),
        temp=temp_tendency,
        salt=salt_tendency,
    )


@dataclass(frozen=True)
@register_dynamics_module("surface_heat_salt_flux")
class SurfaceHeatSaltFlux(TendencyModule):
    """Surface heat and salt flux applied to the top model layer.

    Parameters
    ----------
    heat_flux : float
        Net surface heat flux Q_net [W/m²].  Positive into ocean.
        Converted to temperature tendency: dT/dt = Q/(ρ₀·Cp·Δz·hFac).
    salt_flux : float
        Surface salt flux [PSU·m/s].  Positive increases salinity.
    reference_density : float
        Reference density ρ₀ [kg/m³].
    heat_capacity : float
        Specific heat capacity Cp [J/(kg·K)].  Default: 3994.0.
    sst_restoring_rate : float
        SST restoring rate λ [W/m²/K] at the resting surface thickness.  If > 0,
        adds restoring term; a moving (nonlinear) surface scales the flux by the
        current over the resting surface hFacC, as MITgcm's FORCING_SURF_RELAX.
    sst_target : float or None
        Target SST for restoring [°C].
    """

    sampling_rule = SamplingRule(SampleClock.INDEPENDENT)

    heat_flux: float = 0.0
    salt_flux: float = 0.0
    reference_density: float = RHO_CONST_DEFAULT
    heat_capacity: float = 3994.0
    sst_restoring_rate: float = 0.0
    sst_target: float | None = None
    lin_fs_conserve_tr: bool = False
    name: str = "surface_heat_salt_flux"
    stepper_domain: TendencyStepDomain = TendencyStepDomain.SLOW
    tendency_phase: TendencyPhase = TendencyPhase.TRACER
    # MIT tracForcingOutAB=1 puts surface tracer forcing after AB2
    # (apply_forcing.F:144-146 / temp_integrate.F:404-410).
    # AssemblyPolicy uses this tracer declaration in both synchronous and
    # staggered stepping, independently of momentum forcing placement.
    ab2_eligible: bool = False

    def surface_heat_forcing(
        self, state: PrognosticState, grid: StructuredCGrid, *,
        surface_hfac_c: torch.Tensor | None,
    ) -> torch.Tensor | float:
        """Heat flux including SST restoring, positive into the ocean [K m/s].

        Both the tracer tendency and KPP consume this field before dividing
        by wet surface thickness (MITgcm external_forcing_surf.F).

        ``surface_hfac_c`` [ny, nx] is the surface-level hFacC the restoring
        flux carries: that of the tracer stage which applies it.  MITgcm
        generates the flux with the step-start ``_hFacC`` and, under
        staggerTimeStep with a nonlinear free surface, multiplies it by
        ``hFac_surfC/_hFacC`` (``forcing_surf_relax.F:95-104,113-151``).
        The tracer evaluation passes the grid's own hFacC, which already holds
        that layer when it runs; KPP, which reads the flux before the geometry
        update, gets it from the stepper.  ``None`` only for a grid without
        hFacC.
        """
        rho_cp = self.reference_density * self.heat_capacity
        q_net = self.heat_flux
        if self.sst_restoring_rate > 0.0 and self.sst_target is not None:
            t_surface = state.temp.data[:, 0:1, :, :]
            q_restoring = self.sst_restoring_rate * (self.sst_target - t_surface)
            # FORCING_SURF_RELAX multiplies by wet thickness; the generated
            # lambda cancels that thickness only on wet cells. Prescribed
            # base heat flux is added separately in EXTERNAL_FORCING_SURF.
            if grid.masks.hFacC is not None:
                surface_wet = (grid.masks.hFacC > 0).select(dim=-3, index=0)
                q_restoring = torch.where(
                    surface_wet.to(device=q_restoring.device).unsqueeze(-3),
                    q_restoring, torch.zeros_like(q_restoring),
                )
                # The rate is the coefficient at the resting thickness (the
                # generated lambda divides by the resting drF hFacC).  MITgcm
                # multiplies by drF times the surface hFacC of the applying
                # tracer stage (forcing_surf_relax.F:95-104,113-151), so once a
                # nonlinear free surface moves the surface level the flux
                # scales by surface_hfac_c/h0FacC and the relaxation rate of
                # the cell stays lambda.
                resting = grid.masks.resting_hfac
                if resting is not None:
                    h_now = surface_hfac_c.to(
                        dtype=q_restoring.dtype, device=q_restoring.device)
                    h_rest = resting[0].select(dim=-3, index=0).to(
                        dtype=q_restoring.dtype, device=q_restoring.device)
                    safe = torch.where(h_rest > 0, h_rest, torch.ones_like(h_rest))
                    ratio = torch.where(h_rest > 0, h_now / safe, torch.zeros_like(h_rest))
                    q_restoring = q_restoring * ratio.unsqueeze(-3)
            return (q_net / rho_cp) + q_restoring / rho_cp
        return q_net / rho_cp

    def evaluate(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
    ) -> StateTendency:
        dtype = state.temp.data.dtype
        device = state.temp.data.device

        temp_tendency = torch.zeros_like(state.temp.data)
        salt_tendency = torch.zeros_like(state.salt.data)

        top_dz = top_layer_thickness_value(grid)

        # hFacC normalization at surface layer
        if grid.masks.hFacC is not None:
            hfc = grid.masks.hFacC.to(dtype=dtype, device=device)
            if hfc.ndim == 3:
                hfc = hfc.unsqueeze(0)
            hfc_top = hfc[:, 0:1, :, :]  # [1, 1, ny, nx]
            recip_hfc_top = torch.where(
                hfc_top > 0,
                1.0 / hfc_top.clamp(min=1e-14),
                torch.zeros_like(hfc_top),
            )
        else:
            recip_hfc_top = torch.ones(1, 1, grid.ny, grid.nx, dtype=dtype, device=device)

        # ── Temperature tendency ──
        # surfaceForcingT = Q_net / (rho0 * Cp)   [K·m/s]
        # dT/dt = surfaceForcingT * recip_drF * recip_hFacC
        surface_forcing_t = self.surface_heat_forcing(
            state, grid,
            surface_hfac_c=(None if grid.masks.hFacC is None
                            else grid.masks.hFacC.select(dim=-3, index=0)),
        )

        # Apply: only to surface layer k=0
        temp_tendency[:, 0:1, :, :] = surface_forcing_t / top_dz * recip_hfc_top

        # ── Salinity tendency ──
        if self.salt_flux != 0.0:
            salt_tendency[:, 0:1, :, :] = self.salt_flux / top_dz * recip_hfc_top

        # CALC_WSURF_TR averages rA*w(surface)*tracer(surface) over globalArea.
        # The evaluated state's U/V select the THERMODYNAMICS time layer:
        # old velocity for synchronous stepping, corrected velocity for staggered
        # stepping. etaN-etaH also contains implicDiv2DFlow and is not dt*w.
        if self.lin_fs_conserve_tr:
            from locomgcm.dynamics.implicit_vertical_advection import diagnose_mit_vertical_velocity
            from locomgcm.distributed.comm_adjoint import constant_transport, session_for_grid
            from locomgcm.distributed.reductions import global_sum

            w_surface = diagnose_mit_vertical_velocity(state, grid)[:, 0:1]
            mask_in_c, _, _ = _compute_interior_mask(grid, dtype, device)
            area = grid.metrics.cell_area_for(dtype, device)
            wet_area = torch.where(mask_in_c > 0, area, torch.zeros_like(area))
            wet = mask_in_c.unsqueeze(0).unsqueeze(0) > 0
            t_sfc = torch.where(wet, state.temp.data[:, 0:1], 0.0)
            s_sfc = torch.where(wet, state.salt.data[:, 0:1], 0.0)
            w_surface = torch.where(wet, w_surface, 0.0)
            t_integral = (wet_area * w_surface * t_sfc).sum(dim=(-2, -1), keepdim=True)
            s_integral = (wet_area * w_surface * s_sfc).sum(dim=(-2, -1), keepdim=True)
            layout = getattr(grid, "parallel_layout", None)
            distributed = layout is not None and layout.world_size > 1
            def surface_area():
                total = wet_area.sum()
                if distributed:
                    total = global_sum(total, session=session_for_grid(grid))
                return total

            if wet_area.requires_grad or getattr(grid.masks, "hfac_time_varying", False):
                total_area = surface_area()
            else:
                def constant_area():
                    # Static globalArea is built once, outside adjoint windows.
                    with constant_transport():
                        return surface_area()
                total_area = runtime_memo_constant(
                    grid, ("surface_tracer_global_area", dtype, str(device)), constant_area,
                )
            if distributed:
                # Centre arrays contain owned cells only, without halos.
                session = session_for_grid(grid)
                t_integral = global_sum(t_integral, session=session)
                s_integral = global_sum(s_integral, session=session)
            safe_area = torch.where(total_area > 0, total_area, torch.ones_like(total_area))
            TsurfCor = torch.where(total_area > 0, t_integral / safe_area, 0.0)
            SsurfCor = torch.where(total_area > 0, s_integral / safe_area, 0.0)
            temp_tendency[:, 0:1] = temp_tendency[:, 0:1] + TsurfCor / top_dz * recip_hfc_top
            salt_tendency[:, 0:1] = salt_tendency[:, 0:1] + SsurfCor / top_dz * recip_hfc_top

        return StateTendency(
            eta=torch.zeros_like(state.eta.data),
            u=torch.zeros_like(state.u.data),
            v=torch.zeros_like(state.v.data),
            temp=temp_tendency,
            salt=salt_tendency,
        )


@dataclass
@register_dynamics_module("wpflux_surface_forcing")
class WPFluxSurfaceForcing(TendencyModule):
    """Grid-agnostic surface forcing (heat split + salt + shortwave penetration).

   The tendency-framework counterpart of the cube stepper's in-step forcing
    (``implicit_free_surface_cube.py``); applies the MITgcm surface tracer forcing
    from spatially-varying ``Qnet``/``Qsw``/``EmPmR`` fields::

        gT(1) += surface_heat_forcing(Qnet, Qsw) · recip_drF · recip_hFacC   (non-shortwave)
        gT(k) += shortwave_penetration_tendency(Qsw, …)                      (Qsw over depth)
        gS(1) += surface_salt_forcing(EmPmR·rhoConstFresh) · recip_drF · recip_hFacC

    All three operators are per-cell / per-column, so this module is grid-agnostic
    (regional lat-lon ``[B,nz,ny,nx]`` and cube ``[6,nz,ny,nx]``) and parallelism-
    agnostic (no halo / gather).  ``qnet``/``qsw`` are MITgcm upward-positive,
    ``empmr`` is the bulk volume flux [m/s] (the ingest converts at the boundary,
    docs/NUMERICS.md); set them per step (e.g. from an ERA5 bulk source + model SST).
    """

    @property
    def sampling_rule(self) -> SamplingRule:
        # Static qnet/qsw/empmr are read directly, without a sampled bundle.
        inputs = () if self.source is None else ("_current_fluxes",)
        return SamplingRule(SampleClock.START, inputs, providers=("source",))

    source: PrescribedSurfaceFluxSource | None = None
    qnet: torch.Tensor | None = None
    qsw: torch.Tensor | None = None
    empmr: torch.Tensor | None = None
    heat_capacity: float = 3994.0
    rho_const: float = RHO_CONST_DEFAULT
    rho_const_fresh: float = RHO_CONST_DEFAULT
    convert_fw2salt: float = 35.0
    salt_evprn: float = 0.0
    jwtype: int = 2
    penetrating_sw: bool = True
    name: str = "wpflux_surface_forcing"
    stepper_domain: TendencyStepDomain = TendencyStepDomain.SLOW
    tendency_phase: TendencyPhase = TendencyPhase.TRACER
    ab2_eligible: bool = False
    current_time_seconds: float = field(default=0.0, init=False, repr=False)
    _current_fluxes: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = field(
        default=None, init=False, repr=False
    )
    _current_forcing: SurfaceForcing | None = field(
        default=None, init=False, repr=False
    )
    # MITgcm's (nonlinFreeSurf>0 .AND. useRealFreshWaterFlux) branch of the
    # salt forcing, resolved from the stepper at model assembly
    # (resolve_volume_freshwater); None when no model resolved it, which keeps
    # the virtual flux and is refused by a stepper whose run takes the branch.
    _volume_freshwater: bool | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        static = (self.qnet, self.qsw, self.empmr)
        if self.source is None and any(value is None for value in static):
            raise ValueError(
                "wpflux_surface_forcing requires source or all of qnet/qsw/empmr"
            )
        if self.source is not None and any(value is not None for value in static):
            raise ValueError(
                "wpflux_surface_forcing source and static qnet/qsw/empmr are mutually exclusive"
            )

    def update_model_time(self, time_seconds: float) -> None:
        self.current_time_seconds = float(time_seconds)
        if self.source is not None:
            self._current_fluxes = self.source.sample(time_seconds)

    def to_runtime(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> "WPFluxSurfaceForcing":
        if self.source is not None:
            self.source = self.source.to_runtime(device, dtype)
            self.update_model_time(self.current_time_seconds)
        else:
            tensor_dtype = self.qnet.dtype if dtype is None else dtype
            self.qnet = self.qnet.to(device=device, dtype=tensor_dtype)
            self.qsw = self.qsw.to(device=device, dtype=tensor_dtype)
            self.empmr = self.empmr.to(device=device, dtype=tensor_dtype)
        return self

    def _forcing_fields(
        self, *, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.source is not None:
            sampled = self._current_fluxes
            if sampled is None:
                sampled = self.source.sample(self.current_time_seconds)
            heat_flux, freshwater_flux, qsw = sampled
            heat_flux = heat_flux.to(dtype=dtype, device=device)
            freshwater_flux = freshwater_flux.to(dtype=dtype, device=device)
            # The source heat flux is a LOCOMgcm convention boundary (positive warms);
            # the existing producer consumes MIT Qnet (positive cools).  Qsw
            # arrives in the MIT convention.
            return -heat_flux, qsw.to(dtype=dtype, device=device), freshwater_flux
        assert self.qnet is not None and self.qsw is not None and self.empmr is not None
        return (
            self.qnet.to(dtype=dtype, device=device),
            self.qsw.to(dtype=dtype, device=device),
            self.empmr.to(dtype=dtype, device=device),
        )

    def surface_forcing(
        self,
        state: PrognosticState,
        *,
        pmep_r: torch.Tensor | None,
    ) -> SurfaceForcing:
        """The surfaceForcing bundle of one evaluation; the tendency apply and
        the stepper's KPP harvest both read it.

        ``pmep_r`` is required exactly when the module is resolved for a
        nonlinear free surface with real freshwater flux.
        """
        if self._volume_freshwater and pmep_r is None:
            raise ValueError(
                "wpflux_surface_forcing under a nonlinear free surface with real freshwater "
                "flux reads PmEpR (EvaluationInputs.pmep_r), which this evaluation did not pass")
        if not self._volume_freshwater and pmep_r is not None:
            raise ValueError(
                "PmEpR passed to a wpflux_surface_forcing not resolved for a nonlinear free "
                "surface with real freshwater flux (resolve_volume_freshwater)")
        # Cast the static fields to the state dtype/device BEFORE producing, so
        # sfT/sfS are computed at the run precision (same float-op order as the
        # separate apply, which cast first — bitwise-preserving).
        dtype = state.temp.data.dtype
        device = state.temp.data.device
        qnet, qsw, empmr = self._forcing_fields(dtype=dtype, device=device)
        return surface_forcing_from_bulk(
            qnet,
            qsw,
            empmr,
            heat_capacity=self.heat_capacity, rho_const=self.rho_const,
            rho_const_fresh=self.rho_const_fresh,
            convert_fw2salt=self.convert_fw2salt, salt_evprn=self.salt_evprn,
            penetrating_sw=self.penetrating_sw,
            surface_salinity=state.salt.data.select(dim=-3, index=0),
            pmep_r=pmep_r,
        )

    def evaluate(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
    ) -> StateTendency:
        inputs = diagnostics.inputs
        forcing = self.surface_forcing(
            state, pmep_r=None if inputs is None else inputs.pmep_r,
        )
        self._current_forcing = forcing
        return apply_wpflux_surface_tendency(forcing, state, grid, jwtype=self.jwtype)
