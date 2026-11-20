from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from defgcm.distributed.comm_adjoint import session_for_grid
from defgcm.distributed.reductions import global_sum
from defgcm.geometry import StructuredCGrid
from defgcm.state import DiagnosticState, PrognosticState, StateTendency

from .base import TendencyModule, TendencyStepDomain
from defgcm.dynamics._registry import register_dynamics_module


@dataclass(frozen=True)
@register_dynamics_module("gaussian_eta_mass_source")
class GaussianEtaMassSource(TendencyModule):
    """Idealized free-surface Gaussian mass source.

    This helper is deliberately scoped to single-face, uniform Cartesian,
    all-wet grids.  It is not a metric-aware spherical/cubed-sphere source and
    it does not compute a wet-mask-weighted ocean-only mean on domains with dry
    columns.  Coordinates, default centres and the area-weighted mean refer
    to the global domain when the grid is partitioned.
    """

    amplitude: float
    x_center: float | None = None
    y_center: float | None = None
    x_width: float = 50_000.0
    y_width: float = 50_000.0
    remove_area_weighted_mean: bool = True
    name: str = "gaussian_eta_mass_source"
    stepper_domain: TendencyStepDomain = TendencyStepDomain.FAST

    def __post_init__(self) -> None:
        for name in ("amplitude", "x_width", "y_width"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.x_width <= 0.0 or self.y_width <= 0.0:
            raise ValueError("x_width and y_width must be positive")
        for name in ("x_center", "y_center"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite when provided")

    def _validate_supported_grid(
        self,
        *,
        grid: StructuredCGrid,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        message = (
            "GaussianEtaMassSource supports only single-face uniform Cartesian "
            "all-wet grids; use a metric-aware/wet-mask-aware source for "
            "spherical, cubed-sphere, nonuniform, or dry-column domains"
        )

        if int(getattr(grid, "face_count", 1)) != 1:
            raise ValueError(message)
        if not grid.metrics.is_uniform_x or not grid.metrics.is_uniform_y:
            raise ValueError(message)

        dx = float(grid.dx)
        dy = float(grid.dy)
        if not (
            math.isfinite(dx)
            and math.isfinite(dy)
            and dx > 0.0
            and dy > 0.0
        ):
            raise ValueError(message)

        dx_center = grid.metrics.dx_center.to(dtype=dtype, device=device)
        dy_center = grid.metrics.dy_center.to(dtype=dtype, device=device)
        cell_area = grid.metrics.cell_area.to(dtype=dtype, device=device)
        expected_shape = (grid.ny, grid.nx)
        if (
            tuple(dx_center.shape) != expected_shape
            or tuple(dy_center.shape) != expected_shape
        ):
            raise ValueError(message)
        if tuple(cell_area.shape) != expected_shape:
            raise ValueError(message)

        is_low_precision = dtype in (torch.float16, torch.bfloat16, torch.float32)
        rtol = 1.0e-6 if is_low_precision else 1.0e-12
        atol = 1.0e-8 if is_low_precision else 1.0e-12
        if not torch.allclose(
            dx_center, torch.full_like(dx_center, dx), rtol=rtol, atol=atol
        ):
            raise ValueError(message)
        if not torch.allclose(
            dy_center, torch.full_like(dy_center, dy), rtol=rtol, atol=atol
        ):
            raise ValueError(message)
        if not torch.allclose(
            cell_area,
            torch.full_like(cell_area, dx * dy),
            rtol=rtol,
            atol=atol,
        ):
            raise ValueError(message)

        center_mask = grid.masks.center.to(device=device)
        if center_mask.ndim == 3:
            surface_wet = center_mask.any(dim=0)
        elif center_mask.ndim == 2:
            surface_wet = center_mask
        else:
            raise ValueError(message)
        if (
            tuple(surface_wet.shape) != expected_shape
            or not bool(surface_wet.bool().all().item())
        ):
            raise ValueError(message)

    def _center_coordinates(
        self,
        *,
        grid: StructuredCGrid,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layout = grid.parallel_layout
        x_start = 0 if layout is None else layout.tile_offset_i
        y_start = 0 if layout is None else layout.tile_offset_j
        x = (
            torch.arange(x_start, x_start + grid.nx, dtype=dtype, device=device) + 0.5
        ) * float(grid.dx)
        y = (
            torch.arange(y_start, y_start + grid.ny, dtype=dtype, device=device) + 0.5
        ) * float(grid.dy)
        return x, y

    def _source_pattern(
        self,
        *,
        grid: StructuredCGrid,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        self._validate_supported_grid(grid=grid, dtype=dtype, device=device)

        layout = grid.parallel_layout
        global_nx = grid.nx if layout is None else layout.global_nx
        global_ny = grid.ny if layout is None else layout.global_ny
        x_center = (
            0.5 * float(global_nx) * float(grid.dx)
            if self.x_center is None
            else float(self.x_center)
        )
        y_center = (
            0.5 * float(global_ny) * float(grid.dy)
            if self.y_center is None
            else float(self.y_center)
        )
        x, y = self._center_coordinates(grid=grid, dtype=dtype, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        x_arg = (xx - x_center) / float(self.x_width)
        y_arg = (yy - y_center) / float(self.y_width)
        source = float(self.amplitude) * torch.exp(
            -0.5 * (x_arg.square() + y_arg.square())
        )
        if self.remove_area_weighted_mean:
            area = grid.metrics.cell_area.to(dtype=dtype, device=device)
            source_area = (source * area).sum()
            total_area = area.sum()
            if layout is not None and layout.world_size > 1:
                # Cell-centre arrays contain owned cells only, with no halos.
                session = session_for_grid(grid)
                source_area = global_sum(source_area, session=session)
                total_area = global_sum(total_area, session=session)
            source = source - source_area / total_area
        return source

    def evaluate(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
    ) -> StateTendency:
        del diagnostics
        pattern = self._source_pattern(
            grid=grid,
            dtype=state.eta.data.dtype,
            device=state.eta.data.device,
        )
        eta_tendency = pattern.view(1, grid.ny, grid.nx).expand(
            state.batch_size, grid.ny, grid.nx
        ).clone()
        return StateTendency(
            eta=eta_tendency,
            u=torch.zeros_like(state.u.data),
            v=torch.zeros_like(state.v.data),
            temp=torch.zeros_like(state.temp.data),
            salt=torch.zeros_like(state.salt.data),
        )
