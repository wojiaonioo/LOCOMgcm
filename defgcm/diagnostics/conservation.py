from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import sqrt

import torch

from defgcm.geometry import StructuredCGrid
from defgcm.operators import (
    average_u_to_center,
    average_v_to_center,
    divergence_transport_to_center,
    vertical_velocity_from_divergence,
)
from defgcm.state import DiagnosticState, PrognosticState

from .edge_metrics import boundary_budget_metrics


class DiagnosticModule(ABC):
    name: str

    @abstractmethod
    def compute(
        self,
        state: PrognosticState,
        grid: StructuredCGrid,
        *,
        gravity: float,
        dt: float | None = None,
        context: DiagnosticState | None = None,
    ) -> dict[str, float]:
        raise NotImplementedError


@dataclass(frozen=True)
class SurfaceMassDiagnostic(DiagnosticModule):
    name: str = "surface_mass"

    def compute(
        self,
        state: PrognosticState,
        grid: StructuredCGrid,
        *,
        gravity: float,
        dt: float | None = None,
        context: DiagnosticState | None = None,
    ) -> dict[str, float]:
        del context
        area = grid.metrics.cell_area.to(state.eta.data)
        mass = (state.eta.data * area).sum().item()
        return {self.name: float(mass)}


@dataclass(frozen=True)
class TracerMassDiagnostic(DiagnosticModule):
    field_name: str
    name: str

    def compute(
        self,
        state: PrognosticState,
        grid: StructuredCGrid,
        *,
        gravity: float,
        dt: float | None = None,
        context: DiagnosticState | None = None,
    ) -> dict[str, float]:
        del context
        tracer = getattr(state, self.field_name).data
        thickness = grid.layer_thickness.to(tracer).view(1, grid.nz, 1, 1)
        area = grid.metrics.cell_area.to(tracer).view(1, 1, grid.ny, grid.nx)
        mass = (tracer * thickness * area).sum().item()
        return {self.name: float(mass)}


@dataclass(frozen=True)
class KineticEnergyDiagnostic(DiagnosticModule):
    name: str = "kinetic_energy"

    def compute(
        self,
        state: PrognosticState,
        grid: StructuredCGrid,
        *,
        gravity: float,
        dt: float | None = None,
        context: DiagnosticState | None = None,
    ) -> dict[str, float]:
        del context
        thickness = grid.layer_thickness.to(state.u.data).view(1, grid.nz, 1, 1)
        u_center = average_u_to_center(state.u.data)
        v_center = average_v_to_center(state.v.data)
        area = grid.metrics.cell_area.to(u_center).view(1, 1, grid.ny, grid.nx)
        kinetic_energy = 0.5 * ((u_center.square() + v_center.square()) * thickness * area).sum().item()
        return {self.name: float(kinetic_energy)}


@dataclass(frozen=True)
class FreeSurfaceEnergyDiagnostic(DiagnosticModule):
    name: str = "free_surface_energy"

    def compute(
        self,
        state: PrognosticState,
        grid: StructuredCGrid,
        *,
        gravity: float,
        dt: float | None = None,
        context: DiagnosticState | None = None,
    ) -> dict[str, float]:
        del context
        area = grid.metrics.cell_area.to(state.eta.data)
        energy = 0.5 * gravity * (state.eta.data.square() * area).sum().item()
        return {self.name: float(energy)}


@dataclass(frozen=True)
class CFLEstimateDiagnostic(DiagnosticModule):
    """Coarse, report-only CFL estimate — NOT a time-step guard.

    Reports a single scalar barotropic-wave CFL number for logging/monitoring
    only: it never gates, adapts, or validates ``dt``. The estimate uses the
    smallest centre spacing of the grid (``dxC`` / ``dyC``) and the full-column
    depth (sum of all layer thicknesses), so it makes no per-cell or per-layer
    distinction.
    """

    name: str = "cfl"

    def compute(
        self,
        state: PrognosticState,
        grid: StructuredCGrid,
        *,
        gravity: float,
        dt: float | None = None,
        context: DiagnosticState | None = None,
    ) -> dict[str, float]:
        del context
        if dt is None:
            return {self.name: 0.0}
        max_u = float(state.u.data.abs().max().item())
        max_v = float(state.v.data.abs().max().item())
        depth = float(grid.layer_thickness.sum().item())
        wave_speed = sqrt(gravity * depth)
        dx = float(grid.metrics.dxC_for(torch.float64, "cpu").min().item())
        dy = float(grid.metrics.dyC_for(torch.float64, "cpu").min().item())
        cfl = dt * ((max_u + wave_speed) / dx + (max_v + wave_speed) / dy)
        return {self.name: float(cfl)}


@dataclass(frozen=True)
class VelocityDivergenceDiagnostic(DiagnosticModule):
    name: str = "velocity_divergence"

    def compute(
        self,
        state: PrognosticState,
        grid: StructuredCGrid,
        *,
        gravity: float,
        dt: float | None = None,
        context: DiagnosticState | None = None,
    ) -> dict[str, float]:
        if context is not None and "transport_velocity_u" in context.tensors and "transport_velocity_v" in context.tensors:
            u = context.tensors["transport_velocity_u"]
            v = context.tensors["transport_velocity_v"]
        else:
            u = state.u.data
            v = state.v.data
        div = divergence_transport_to_center(u, v, grid)
        rms = float(div.square().mean().sqrt().item())
        amax = float(div.abs().max().item())
        mean = float(div.mean().item())
        return {
            f"{self.name}_rms": rms,
            f"{self.name}_max": amax,
            f"{self.name}_mean": mean,
        }


@dataclass(frozen=True)
class ThreeDContinuityDiagnostic(DiagnosticModule):
    name: str = "three_d_continuity"

    def compute(
        self,
        state: PrognosticState,
        grid: StructuredCGrid,
        *,
        gravity: float,
        dt: float | None = None,
        context: DiagnosticState | None = None,
    ) -> dict[str, float]:
        del gravity, dt
        if context is not None and "horizontal_divergence" in context.tensors:
            divergence = context.tensors["horizontal_divergence"]
        else:
            divergence = divergence_transport_to_center(state.u.data, state.v.data, grid)
        if context is not None and "vertical_velocity_interface" in context.tensors:
            vertical_velocity_interface = context.tensors["vertical_velocity_interface"]
        else:
            vertical_velocity_interface = vertical_velocity_from_divergence(divergence, grid.layer_thickness)
        thickness = grid.layer_thickness.to(divergence).view(1, grid.nz, 1, 1)
        residual = divergence + (vertical_velocity_interface[:, 1:, :, :] - vertical_velocity_interface[:, :-1, :, :]) / thickness
        bottom_velocity = vertical_velocity_interface[:, -1, :, :]
        top_velocity = vertical_velocity_interface[:, 0, :, :]
        return {
            f"{self.name}_rms": float(residual.square().mean().sqrt().item()),
            f"{self.name}_max": float(residual.abs().max().item()),
            "vertical_velocity_top_max_abs": float(top_velocity.abs().max().item()),
            "vertical_velocity_bottom_rms": float(bottom_velocity.square().mean().sqrt().item()),
            "vertical_velocity_bottom_max_abs": float(bottom_velocity.abs().max().item()),
        }


@dataclass
class BoundaryFluxBudgetDiagnostic(DiagnosticModule):
    name: str = "boundary_budget"

    def compute(
        self,
        state: PrognosticState,
        grid: StructuredCGrid,
        *,
        gravity: float,
        dt: float | None = None,
        context: DiagnosticState | None = None,
    ) -> dict[str, float]:
        del context
        metrics = boundary_budget_metrics(state, grid)
        flattened: dict[str, float] = {
            "boundary_q_net": float(metrics["q_net"]),
            "boundary_q_in": float(metrics["q_in"]),
            "boundary_q_out": float(metrics["q_out"]),
            "boundary_temp_flux_net": float(metrics["temp_flux_net"]),
            "boundary_salt_flux_net": float(metrics["salt_flux_net"]),
        }
        for side, value in metrics["transport_net_by_side"].items():
            flattened[f"boundary_q_net_{side}"] = float(value)
        for side, value in metrics["transport_in_by_side"].items():
            flattened[f"boundary_q_in_{side}"] = float(value)
        for side, value in metrics["transport_out_by_side"].items():
            flattened[f"boundary_q_out_{side}"] = float(value)
        for side, value in metrics["inflow_fraction_by_side"].items():
            flattened[f"boundary_inflow_fraction_{side}"] = float(value)
        for side, value in metrics["outflow_fraction_by_side"].items():
            flattened[f"boundary_outflow_fraction_{side}"] = float(value)
        for side, value in metrics["temp_flux_by_side"].items():
            flattened[f"boundary_temp_flux_{side}"] = float(value)
        for side, value in metrics["salt_flux_by_side"].items():
            flattened[f"boundary_salt_flux_{side}"] = float(value)
        return flattened


@dataclass
class DiagnosticSuite:
    modules: list[DiagnosticModule] = field(default_factory=list)

    def compute_all(
        self,
        state: PrognosticState,
        grid: StructuredCGrid,
        *,
        gravity: float,
        dt: float | None = None,
        context: DiagnosticState | None = None,
    ) -> dict[str, float]:
        diagnostics: dict[str, float] = {}
        for module in self.modules:
            diagnostics.update(module.compute(state, grid, gravity=gravity, dt=dt, context=context))
        return diagnostics
