"""SEAICE inventory transport and regularization, for the T and L branches."""

import torch
from torch import Tensor

from ._records import (
    RegularizationParameters, RegularizationResult, Scalar,
    SeaiceInventory, TransportGeometry,
)
from ._stencils import check_field, extensive_sweep, flux41, flux77, interior_write
from ..thermo_pw79.records import Salinity

#: SEAICEadvSnow true: HSNOW is transported with HEFF and AREA
#: (seaice_advdiff.F:262,351,617).
ADVECT_SNOW = (True,)


def seaice_advdiff(inventory: SeaiceInventory, UICE: Tensor, VICE: Tensor,
                   geometry: TransportGeometry, *, deltaT: Scalar,
                   scheme: int) -> SeaiceInventory:
    """Return T3/L9: extensive HEFF/AREA/HSNOW, interior writes, old halo.

    The selected first-batch schemes are 41 (monotone PPM) and 77
    (Superbee). The configured path has multidimensional advection,
    all three tracers enabled, no diffusion, no ITD and no OBCS.
    Velocities remain live inputs even when identically zero.
    """
    if scheme not in (41, 77):
        raise ValueError("SEAICE transport supports schemes 41 and 77")
    for field in (inventory.HEFF, inventory.AREA, inventory.HSNOW, UICE, VICE):
        check_field(field)
    g = geometry
    # seaice_advdiff.F:100-106,172-176: masked face areas and transports.
    utrans = UICE * (g.dyG * g.SIMaskU)
    vtrans = VICE * (g.dxG * g.SIMaskV)
    recip_area = torch.div(g.rA.new_ones(()), g.rA)

    def advect(q: Tensor) -> Tensor:
        local = q
        # seaice_advection.F:340-341: X then Y with no intervening exchange.
        for axis, velocity, transport, face_mask, dx_c, dx_f in (
            (-1, UICE, utrans, g.SIMaskU, g.dxC, g.dxF),
            (-2, VICE, vtrans, g.SIMaskV, g.dyC, g.dyF),
        ):
            if scheme == 77:
                af = flux77(local, transport, (velocity * deltaT * torch.div(dx_c.new_ones(()), dx_c)).abs(), face_mask, axis=axis)
            else:
                af = flux41(local, transport, velocity, g.maskC, torch.div(dx_f.new_ones(()), dx_f), deltaT, axis=axis)
            local = extensive_sweep(local, af, g.maskInC, recip_area, deltaT, axis=axis)
        # seaice_advection.F:759-764; seaice_advdiff.F:314-319,341-346,368-373.
        tendency = (local - q) / deltaT
        return interior_write(q, g.HEFFM * (q + deltaT * tendency))

    return SeaiceInventory(advect(inventory.HEFF), advect(inventory.AREA), advect(inventory.HSNOW))


def seaice_reg_ridge(inventory: SeaiceInventory, TICES: Tensor,
                     parameters: RegularizationParameters) -> RegularizationResult:
    """Return T4/L10.reg including the signed effective-thickness corrections.

    The caller invokes this only in SEAICE's inventory branch, never H/B
    (seaice_model.F:209-214,249). Corrections have units m, not m/s.
    """
    for field in (inventory.HEFF, inventory.AREA, inventory.HSNOW):
        check_field(field)
    check_field(TICES, categories=True)
    p = parameters
    H, A, S = inventory.HEFF, inventory.AREA, inventory.HSNOW
    # seaice_reg_ridge.F:99-100,181-185: reset corrections, repair negatives.
    dH, dS = (-H).clamp(min=0.0), (-S).clamp(min=0.0)
    H, S, A = H + dH, S + dS, A.clamp(min=0.0)
    # seaice_reg_ridge.F:220-232: thin effective ice removes snow as well.
    thin = H <= p.siEps
    removeH = torch.where(thin, -H, torch.zeros_like(H))
    removeS = torch.where(thin, -S, torch.zeros_like(S))
    temperature = torch.where(thin.unsqueeze(-3), TICES.new_ones(()) * p.celsius2K, TICES)
    H, S, dH, dS = H + removeH, S + removeS, dH + removeH, dS + removeS
    # seaice_reg_ridge.F:255-256,286-287,374-381: empty, floor, then cap.
    A = torch.where((H == 0.0) & (S == 0.0), torch.zeros_like(A), A)
    floor = A.new_ones(()) * p.area_floor
    A = torch.where((H > 0.0) | (S > 0.0), torch.maximum(A, floor), A)
    preridge = interior_write(torch.zeros_like(A), A)
    A = torch.minimum(A, A.new_ones(()) * p.area_max)
    return RegularizationResult(
        SeaiceInventory(interior_write(inventory.HEFF, H), interior_write(inventory.AREA, A), interior_write(inventory.HSNOW, S)),
        interior_write(TICES, temperature),
        interior_write(torch.zeros_like(dH), dH), interior_write(torch.zeros_like(dS), dS), preridge,
    )


def regularize_salinity(HSALT: Tensor, HEFF: Tensor, HEFFM: Tensor, *, deltaT: Scalar) -> Salinity:
    """SEAICE_VARIABLE_SALINITY part of REG_RIDGE (seaice_reg_ridge.F:106,302-307).

    ``HEFF`` is the regularized inventory. Negative HSALT, or any HSALT left
    without ice, returns to the ocean as saltFluxAdjust (g/m2/s, upward
    positive) and HSALT is set to zero; only the interior is written.
    """
    for field in (HSALT, HEFF, HEFFM):
        check_field(field)
    clip = (HSALT < 0.0) | (HEFF == 0.0)
    adjust = torch.where(clip, -HEFFM * HSALT * (1.0 / deltaT), torch.zeros_like(HSALT))
    return Salinity(interior_write(HSALT, torch.where(clip, torch.zeros_like(HSALT), HSALT)),
                    interior_write(torch.zeros_like(adjust), adjust))
