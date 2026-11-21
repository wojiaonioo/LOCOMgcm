"""Atmospheric surface pressure loading (MITgcm ``pLoadFile``, ATMOSPHERIC_LOADING).

``pload`` is MITgcm ``pLoad``: the atmospheric pressure anomaly relative to
``surf_pRef`` at tracer centres, [ny, nx] in Pa, positive for higher pressure,
unmasked (MITgcm reads the file verbatim, ``ini_forcing.F:191-195``).  The same
sign holds in both conventions.  ``phi0surf`` is the surface geopotential it
adds to the hydrostatic potential, [ny, nx] in m2 s-2, in the dtype of ``pload``.
"""
from __future__ import annotations

import torch


def surface_pressure_load_potential(pload: torch.Tensor, *, recip_rho_const: float) -> torch.Tensor:
    """``phi0surf = pLoad*recip_rhoConst`` (external_forcing_surf.F:376-381, z-coordinate ocean
    without useRealFreshWaterFlux)."""
    return pload * recip_rho_const


def validate_surface_pressure_load_composition(
    *,
    modules,
    face_count: int,
    use_real_freshwater_flux: bool,
) -> None:
    """Refuse the compositions the pLoad port does not cover (SPEC_pload section 1).

    useRealFreshWaterFlux adds the sea-ice load to phi0surf (external_forcing_surf.F:367-373);
    a dynamic-pressure EOS reads totPhiHyd, which holds phi0surf from the second step on but not
    in the INI_PRESSURE seed (ini_fields.F:33 runs before ini_forcing.F reads pLoad); the cube
    takes its pressure gradient outside HydrostaticPressureGradient.evaluate.
    """
    if not any(getattr(m, "surface_pressure_load", False) for m in modules):
        return
    if face_count != 1:
        raise ValueError("surface_pressure_load (MITgcm pLoadFile) is not ported to the cubed sphere")
    if use_real_freshwater_flux:
        raise ValueError(
            "surface_pressure_load (MITgcm pLoadFile) with use_real_freshwater_flux: the sea-ice "
            "load term of phi0surf (external_forcing_surf.F:367-373) is not ported")
    dynamic = [m.name for m in modules if getattr(m, "dynamic_pressure", False)]
    if dynamic:
        raise ValueError(
            f"surface_pressure_load (MITgcm pLoadFile) with the dynamic-pressure EOS {dynamic}: "
            "totPhiHyd with phi0surf (diags_phi_hyd.F:62-64) and its phi0surf-free INI_PRESSURE "
            "seed are not ported")
