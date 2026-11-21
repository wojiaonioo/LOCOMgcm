from __future__ import annotations

from dataclasses import dataclass

import torch

from locomgcm.geometry import StructuredCGrid
from locomgcm.operators import average_u_to_center, average_v_to_center, center_to_u, center_to_v
from locomgcm.state import DiagnosticState, PrognosticState, StateTendency

from .base import TendencyModule, TendencyStepDomain
from locomgcm.dynamics._registry import register_dynamics_module


@dataclass(frozen=True)
@register_dynamics_module("beta_plane_coriolis")
class BetaPlaneCoriolis(TendencyModule):
    """Beta-plane tendency with exchanged f*u at periodic and rank seams.

    Center f uses global coordinates; interpolation exchanges the product
    with velocity, matching MIT selectCoriMap=3, selectCoriScheme=2 semantics.
    """

    beta: float
    y_equator: float | None = None
    f0: float = 0.0
    name: str = "beta_plane_coriolis"
    stepper_domain: TendencyStepDomain = TendencyStepDomain.SLOW

    def _f_center(
        self,
        *,
        grid: StructuredCGrid,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if grid.metrics.del_y is None or grid.metrics.is_uniform_y:
            layout = grid.parallel_layout
            y_start = 0 if layout is None else layout.tile_offset_j
            global_ny = grid.ny if layout is None else layout.global_ny
            y_center = (
                torch.arange(y_start, y_start + grid.ny, dtype=dtype, device=device)
                + 0.5
            ) * float(grid.dy)
            domain_length = float(global_ny) * float(grid.dy)
        else:
            del_y = grid.metrics.del_y.to(dtype=dtype, device=device)
            y_center = torch.cumsum(del_y, dim=0) - 0.5 * del_y
            domain_length = grid.metrics.meridional_length_value(dtype=del_y.dtype, device=del_y.device)
        y_eq = 0.5 * domain_length if self.y_equator is None else float(self.y_equator)
        return self.f0 + self.beta * (y_center - y_eq)

    def evaluate(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
    ) -> StateTendency:
        del diagnostics
        if self.beta == 0.0 and self.f0 == 0.0:
            return StateTendency.zeros_like(state)

        u_center = average_u_to_center(state.u.data)
        v_center = average_v_to_center(state.v.data)
        f_center = self._f_center(
            grid=grid,
            dtype=state.u.data.dtype,
            device=state.u.data.device,
        ).view(1, 1, grid.ny, 1)
        f_v_on_u = center_to_u(f_center * v_center, grid)
        f_u_on_v = center_to_v(f_center * u_center, grid)

        return StateTendency(
            eta=torch.zeros_like(state.eta.data),
            u=f_v_on_u,
            v=-f_u_on_v,
            temp=torch.zeros_like(state.temp.data),
            salt=torch.zeros_like(state.salt.data),
        )
