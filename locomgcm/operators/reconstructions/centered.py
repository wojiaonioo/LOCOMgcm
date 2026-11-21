from __future__ import annotations

import torch

from locomgcm.geometry import StructuredCGrid
from locomgcm.geometry.exchange import FillSpec, HaloExtentV1, exchange

from ._boundary import prepare_obc_tracer


_X_FACE_EXTENT = HaloExtentV1(
    west_left_extent=1,
    east_right_extent=1,
    south_bottom_extent=0,
    north_top_extent=0,
)
_Y_FACE_EXTENT = HaloExtentV1(
    west_left_extent=0,
    east_right_extent=0,
    south_bottom_extent=1,
    north_top_extent=1,
)


def center_to_u_centered(
    tracer: torch.Tensor,
    grid: StructuredCGrid,
    *,
    face_mask: torch.Tensor | None,
    obc_west_tracer: torch.Tensor | None = None,
    obc_east_tracer: torch.Tensor | None = None,
    constant: bool = False,
) -> torch.Tensor:
    """Reconstruct center values on U faces with an explicit mask policy.

    ``face_mask=None`` explicitly requests the unmasked arithmetic average.
    Otherwise the mask is applied to the completed average, matching MITgcm
    ``exf_mapfields.F:246``; it must not be folded into the two weights.
    """
    obc_west = prepare_obc_tracer(
        obc_west_tracer,
        tracer=tracer,
        side_len=grid.ny,
        side_name="west",
    )
    obc_east = prepare_obc_tracer(
        obc_east_tracer,
        tracer=tracer,
        side_len=grid.ny,
        side_name="east",
    )
    obc = {}
    if obc_west is not None:
        obc["west"] = obc_west
    if obc_east is not None:
        obc["east"] = obc_east
    tracer_halo = exchange(
        tracer,
        grid=grid,
        location="T",
        extent=_X_FACE_EXTENT,
        fill=FillSpec.OBC_VALUE,
        obc=obc,
        constant=constant,
    )
    out = 0.5 * (tracer_halo[..., :, :-1] + tracer_halo[..., :, 1:])
    return out if face_mask is None else out * face_mask


def center_to_v_centered(
    tracer: torch.Tensor,
    grid: StructuredCGrid,
    *,
    face_mask: torch.Tensor | None,
    obc_south_tracer: torch.Tensor | None = None,
    obc_north_tracer: torch.Tensor | None = None,
    constant: bool = False,
) -> torch.Tensor:
    """Reconstruct center values on V faces with an explicit mask policy.

    ``face_mask=None`` explicitly requests the unmasked arithmetic average.
    Otherwise the mask is applied to the completed average, matching MITgcm
    ``exf_mapfields.F:287``; it must not be folded into the two weights.
    """
    obc_south = prepare_obc_tracer(
        obc_south_tracer,
        tracer=tracer,
        side_len=grid.nx,
        side_name="south",
    )
    obc_north = prepare_obc_tracer(
        obc_north_tracer,
        tracer=tracer,
        side_len=grid.nx,
        side_name="north",
    )
    obc = {}
    if obc_south is not None:
        obc["south"] = obc_south
    if obc_north is not None:
        obc["north"] = obc_north
    tracer_halo = exchange(
        tracer,
        grid=grid,
        location="T",
        extent=_Y_FACE_EXTENT,
        fill=FillSpec.OBC_VALUE,
        obc=obc,
        constant=constant,
    )
    out = 0.5 * (tracer_halo[..., :-1, :] + tracer_halo[..., 1:, :])
    return out if face_mask is None else out * face_mask
