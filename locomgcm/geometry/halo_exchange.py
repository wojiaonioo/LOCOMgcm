"""Narrow-waist halo-exchange injection for geometry-neutral operators.

An operator depends only on ``{tensor, 2D metrics, HaloExchange}``.  Geometry
lives in the metric construction + the exchange implementation, NOT in the operator.

``HaloExchange`` is a thin grid-bound facade over the existing halo primitives
(``geometry.exchange.exchange`` / ``exchange_velocity_pair`` and
``geometry.cubed_sphere_corner.fill_cs_corner_tr_rl``).  Those primitives ALREADY
self-dispatch region (per-axis fill) vs cube single-rank vs cube multi-rank
distributed (keyed by ``grid.boundary_topology`` SEAM + ``grid._cube_step_ctx``),
so a single grid-bound facade serves both geometries — the operator calls
``exchange.center / .vector / .corner`` and never branches on geometry.
"""

from __future__ import annotations

from typing import Any

import torch

from .cubed_sphere_corner import fill_cs_corner_tr_rl
from .exchange import (
    CGridLocation,
    FillSpec,
    HaloExtentV1,
    exchange,
    exchange_velocity_pair,
)


class HaloExchange:
    """Grid-bound halo-fill facade for geometry-neutral operators.

    Methods forward to the self-dispatching primitives bound to ``grid``:

    - :meth:`center` — scalar/centred field halo (region per-axis fill OR cube
      cross-face scalar exchange, single- or multi-rank).
    - :meth:`vector` — coupled (u, v) C-grid face halo (region independent fill
      OR cube stagger-swap + sign vector exchange).
    - :meth:`corner` — cube valence-3 corner / cross-seam scalar fill
      (``FILL_CS_CORNER_TR_RL``).  **No-op on region** (the underlying primitive
      returns ``field`` unchanged when ``grid.face_count != 6``).

    The facade adds no numerics; it is bitwise-identical to calling the
    primitives directly (verified).
    """

    __slots__ = ("grid",)

    def __init__(self, grid: Any) -> None:
        self.grid = grid

    @property
    def is_cube(self) -> bool:
        return getattr(self.grid, "face_count", 1) == 6

    def center(
        self,
        field: torch.Tensor,
        *,
        location: CGridLocation = "T",
        extent: HaloExtentV1,
        fill: FillSpec = FillSpec.ZERO,
        obc: Any = None,
        constant: bool = False,
    ) -> torch.Tensor:
        """Fill a scalar/centred field's halo; returns a new haloed tensor."""
        return exchange(
            field,
            grid=self.grid,
            location=location,
            extent=extent,
            fill=fill,
            obc=obc,
            constant=constant,
        )

    def vector(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        *,
        extent: HaloExtentV1,
        with_signs: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fill a coupled (u, v) C-grid pair's halo (cube swap+sign safe)."""
        return exchange_velocity_pair(
            u,
            v,
            grid=self.grid,
            extent=extent,
            with_signs=with_signs,
        )

    def corner(self, field: torch.Tensor, *, mode: int) -> torch.Tensor:
        """Cube scalar corner / cross-seam fill (``FILL_CS_CORNER_TR_RL``).

        ``mode=1`` prepares the X-gradient stencil, ``mode=2`` the Y-gradient
        stencil (MIT ``gad_advection.F``).  Region: returns ``field`` unchanged.
        """
        return fill_cs_corner_tr_rl(field, mode, self.grid)
