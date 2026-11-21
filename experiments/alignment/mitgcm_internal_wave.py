#!/usr/bin/env python3
"""
MITgcm internal_wave Verification for LOCOMgcm
==================================================

Reconstructs the MITgcm ``verification/internal_wave`` test case:
- 2D x-z section (ny=1): internal wave forced by open boundary over sloping topography
- West boundary injects a mode-1 internal wave signal
- East boundary uses Orlanski radiation
- Sloping bathymetry from 200 m to 40 m with easternmost cell = land

Uses split-explicit time stepping: slow (baroclinic) dynamics at dt, fast
(barotropic free-surface) dynamics at dt/fast_substeps.

Usage:
    python experiments/alignment/mitgcm_internal_wave.py --device cpu --hours 14 --dt 50
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from locomgcm.boundaries.open import SideBoundaryTensorSet
from locomgcm.boundaries.internal_wave_boundary import (
    build_vertical_structure,
    west_temp_profile,
    west_u_profile,
)

# ---------------------------------------------------------------------------
# MITgcm internal_wave setup parameters
# ---------------------------------------------------------------------------
NX = 60
NY = 1
NZ = 20
H = 200.0           # total depth (m)
LX = 13.3e3         # domain length (m)
DY = 5.0e3           # y cell size (m)
DZ = H / NZ          # 10 m
GRAVITY = 9.81
F0 = 0.0
ALPHA = 2.0e-4
N2 = 1.0e-6
OB_TIMESCALE = 44567.0
U_INFLOW = 0.024


@dataclass(frozen=True)
class AnalyticalIWBoundarySample:
    temp: SideBoundaryTensorSet
    salt: SideBoundaryTensorSet
    u: SideBoundaryTensorSet
    v: SideBoundaryTensorSet | None = None


@dataclass
class AnalyticalIWBoundarySource:
    """Analytical OBC source for MITgcm ``internal_wave`` experiment."""

    nz: int
    ny: int
    Tref: torch.Tensor
    u_inflow: float = 0.024
    ob_timescale: float = 44567.0
    f0: float = 0.0
    N2: float = 1.0e-6
    gravity: float = 9.81
    alpha: float = 2.0e-4
    total_depth: float = 200.0
    salt_value: float = 35.0

    def __post_init__(self) -> None:
        if self.Tref.ndim != 1 or self.Tref.shape[0] != self.nz:
            raise ValueError(f"Tref must have shape [{self.nz}], got {tuple(self.Tref.shape)}")

    def _mode1_vertical_structure(self, *, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        """``vertStructWst`` on this source's uniform layers (obcs_calc.F:59-67), the
        boundary condition's own function; rC, rF(Nr+1) negative, as MITgcm."""
        dz = self.total_depth / float(self.nz)
        rC = -((torch.arange(self.nz, dtype=dtype, device=device) + 0.5) * dz)
        drF = torch.full((self.nz,), dz, dtype=dtype, device=device)
        rF_bottom = torch.tensor(-self.total_depth, dtype=dtype, device=device)
        return build_vertical_structure(rC, rF_bottom, drF)

    def _west_u(self, model_time_seconds: float, *, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        vert = self._mode1_vertical_structure(dtype=dtype, device=device)
        u_boundary = west_u_profile(
            vert, u_inflow=self.u_inflow, ob_timescale=self.ob_timescale, f0=self.f0,
            N2=self.N2, west_dx=LX / (NX + 1), time_seconds=float(model_time_seconds),
        )
        return u_boundary.unsqueeze(-1).expand(self.nz, self.ny)

    def _west_temp(self, model_time_seconds: float, *, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        t_boundary = west_temp_profile(
            self.Tref.to(dtype=dtype, device=device), u_inflow=self.u_inflow,
            ob_timescale=self.ob_timescale, f0=self.f0, N2=self.N2, gravity=self.gravity,
            alpha=self.alpha, time_seconds=float(model_time_seconds),
        )
        return t_boundary.unsqueeze(-1).expand(self.nz, self.ny)

    def _east_temp(self, *, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        tref = self.Tref.to(dtype=dtype, device=device)
        return tref.unsqueeze(-1).expand(self.nz, self.ny)

    def _salt(self, *, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        return torch.full((self.nz, self.ny), self.salt_value, dtype=dtype, device=device)

    def sample(
        self,
        model_time_seconds: float,
        *,
        mode: str = "linear",
        update_interval_seconds: float | None = None,
    ) -> AnalyticalIWBoundarySample:
        del mode, update_interval_seconds
        dtype = self.Tref.dtype
        device = self.Tref.device

        west_u = self._west_u(model_time_seconds, dtype=dtype, device=device)
        east_u = torch.zeros((self.nz, self.ny), dtype=dtype, device=device)
        west_temp = self._west_temp(model_time_seconds, dtype=dtype, device=device)
        east_temp = self._east_temp(dtype=dtype, device=device)
        west_salt = self._salt(dtype=dtype, device=device)
        east_salt = west_salt.clone()

        return AnalyticalIWBoundarySample(
            temp=SideBoundaryTensorSet(west=west_temp, east=east_temp),
            salt=SideBoundaryTensorSet(west=west_salt, east=east_salt),
            u=SideBoundaryTensorSet(west=west_u, east=east_u),
            v=SideBoundaryTensorSet(),
        )

    def to_runtime(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> "AnalyticalIWBoundarySource":
        move_kwargs: dict[str, object] = {"device": device}
        if dtype is not None:
            move_kwargs["dtype"] = dtype
        return AnalyticalIWBoundarySource(
            nz=self.nz,
            ny=self.ny,
            Tref=self.Tref.to(**move_kwargs),
            u_inflow=self.u_inflow,
            ob_timescale=self.ob_timescale,
            f0=self.f0,
            N2=self.N2,
            gravity=self.gravity,
            alpha=self.alpha,
            total_depth=self.total_depth,
            salt_value=self.salt_value,
        )

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> "AnalyticalIWBoundarySource":
        return self.to_runtime(device, dtype)


def build_bathymetry(nx: int, dx: float) -> np.ndarray:
    """Reconstruct sloping bathymetry from gendata.m."""
    slope = 0.03
    offset = 2.5e3
    dmax = 40.0  # minimum depth (positive)

    # Cell center x-coordinates (MITgcm convention)
    x = np.zeros(nx)
    x[0] = dx
    for i in range(1, nx):
        x[i] = x[i - 1] + dx

    depth = np.zeros(nx)
    for i in range(nx):
        d = H - slope * (x[i] - offset)  # positive depth
        d = min(d, H)
        d = max(d, dmax)
        depth[i] = d
    depth[-1] = 0.0  # land at east end
    return depth


_MITGCM_TREF_20 = np.array([
    0.04842, 0.0433231, 0.0382263, 0.0331295, 0.0280326,
    0.0229358, 0.0178389, 0.0127421, 0.00764526, 0.00254842,
    -0.00254842, -0.00764526, -0.0127421, -0.0178389, -0.0229358,
    -0.0280326, -0.0331295, -0.0382263, -0.0433231, -0.04842,
])


def build_tref(nz: int) -> np.ndarray:
    """Return tRef matching MITgcm's ``data`` file (truncated precision).

    MITgcm reads tRef from its namelist at ~7 significant digits. Using a
    formula-derived value introduces ~4e-8 drift that propagates through
    EOS → HPG → velocity over time.  Return the exact namelist values when
    nz matches the standard 20-layer setup; fall back to formula otherwise.
    """
    if nz == 20:
        return _MITGCM_TREF_20.copy()
    Tz = N2 / (GRAVITY * ALPHA)
    z = np.array([-(k + 0.5) * DZ for k in range(nz)])
    Tz_z = Tz * z
    return Tz_z - Tz_z.mean()


def build_initial_temperature(nz: int, nx: int, Tref: np.ndarray) -> np.ndarray:
    """T(x,z) = Tref(z) uniform in x."""
    T = np.zeros((1, nz, 1, nx))  # [batch, nz, ny=1, nx]
    for k in range(nz):
        T[0, k, 0, :] = Tref[k]
    return T


# ---------------------------------------------------------------------------
# Build LOCOMgcm model
# ---------------------------------------------------------------------------
def build_model(*, device: str = "cpu", dt: float = 50.0, fast_substeps: int = 13):
    from locomgcm import RegionalOceanConfig, TorchRegionalOceanModel
    from locomgcm.boundaries.open import Prescribed3DBoundaryCondition
    from locomgcm.steppers import SplitExplicitStepper
    from locomgcm.geometry import BoundaryTopology, BoundaryType, StructuredCGrid

    dx = LX / (NX + 1)  # MITgcm convention: dx = Lx / (nx+1)
    lx = dx * NX         # actual domain length for NX cells
    ly = DY * NY

    # Build bathymetry
    bathy_1d = build_bathymetry(NX, dx)
    bathy_2d = torch.tensor(bathy_1d.reshape(NY, NX), dtype=torch.float64)

    # Build grid with bathymetry
    grid = StructuredCGrid.with_bathymetry(
        nx=NX,
        ny=NY,
        nz=NZ,
        lx=lx,
        ly=ly,
        depth=H,
        bathymetry=bathy_2d,
        boundary_topology=BoundaryTopology(
            west_east=BoundaryType.OPEN,
            south_north=BoundaryType.WALL,
        ),
        device=device,
        dtype=torch.float64,
    )

    # Build reference temperature
    Tref = build_tref(NZ)
    Tref_tensor = torch.tensor(Tref, dtype=torch.float64, device=device)
    Sref_tensor = torch.full((NZ,), 35.0, dtype=torch.float64, device=device)

    # Build boundary condition source (prescribed 3D OBCS)
    boundary_source = AnalyticalIWBoundarySource(
        nz=NZ,
        ny=NY,
        Tref=Tref_tensor,
        ob_timescale=OB_TIMESCALE,
        u_inflow=U_INFLOW,
        f0=F0,
        N2=N2,
        gravity=GRAVITY,
        alpha=ALPHA,
        total_depth=H,
    )
    bc = Prescribed3DBoundaryCondition(tracer_source=boundary_source)

    # Build model (bypass RegionalOceanConfig.rectangular by building directly)
    from locomgcm.dynamics import (
        DiagnosticVerticalVelocity,
        FCoriolis,
        HorizontalTracerAdvection,
        HorizontalTracerDiffusion,
        HorizontalViscosity,
        HydrostaticPressureGradient,
        LinearEquationOfState,
        LinearFreeSurfaceContinuity,
        LinearFreeSurfacePressureGradient,
        VerticalMixing,
        VerticalTracerTransport,
    )
    from locomgcm.operators import TracerFluxScheme

    config = RegionalOceanConfig(
        nx=NX,
        ny=NY,
        nz=NZ,
        lx=lx,
        ly=ly,
        depth=H,
        gravity=GRAVITY,
        reference_density=1025.0,
        thermal_expansion=ALPHA,
        haline_contraction=0.0,
        reference_temperature=0.0,
        reference_salinity=35.0,
        vertical_diffusivity=1.0e-3,  # diffKzT
        coriolis_parameter=F0,
        default_dt=dt,
        dtype=torch.float64,
        device=device,
        west_east_boundary=BoundaryType.OPEN,
        south_north_boundary=BoundaryType.WALL,
        tracer_advection_scheme=TracerFluxScheme.UPWIND,
    )

    dynamics = [
        LinearFreeSurfaceContinuity(),
        LinearFreeSurfacePressureGradient(gravity=GRAVITY),
        LinearEquationOfState(
            reference_density=1025.0,
            thermal_expansion=ALPHA,
            haline_contraction=0.0,
            reference_temperature=Tref_tensor,
            reference_salinity=Sref_tensor,
        ),
        HydrostaticPressureGradient(
            gravity=GRAVITY,
            reference_density=1025.0,
            thermal_expansion=ALPHA,
            haline_contraction=0.0,
            reference_temperature=Tref_tensor,
            reference_salinity=Sref_tensor,
        ),
        DiagnosticVerticalVelocity(),
        HorizontalTracerAdvection(
            scheme=TracerFluxScheme.UPWIND,
            tracer_names=("temp",),  # salt stepping off
        ),
        VerticalTracerTransport(),
        VerticalMixing(diffusivity=1.0e-3),  # viscAz = diffKzT
        HorizontalViscosity(viscosity=1.0e-2),  # viscAh
        HorizontalTracerDiffusion(diffusivity=1.0e-2, tracer_names=("temp",)),  # diffKhT
        FCoriolis(coriolis_parameter=F0),
    ]

    stepper = SplitExplicitStepper(fast_substeps=fast_substeps)

    model = TorchRegionalOceanModel(
        config,
        dynamics_modules=dynamics,
        stepper=stepper,
        boundary_condition=bc,
    )
    # Override grid with bathymetry grid
    model.grid = grid

    # Build initial state
    T_init = build_initial_temperature(NZ, NX, Tref)
    temp = torch.tensor(T_init, dtype=torch.float64, device=device)

    # Apply mask to initial temperature
    if grid.masks.has_3d_masks:
        cmask = grid.masks.center_float(dtype=torch.float64, device=device)
        temp = temp * cmask

    # Salt = 35 everywhere (MITgcm: sRef=20*35)
    salt_init = torch.full_like(temp, 35.0)
    if grid.masks.has_3d_masks:
        salt_init = salt_init * cmask

    initial_state = model.create_state(temp=temp, salt=salt_init)

    return model, initial_state, config, {
        "Tref": Tref,
        "bathy": bathy_1d,
        "dx": dx,
        "x": np.array([dx * (i + 1) for i in range(NX)]),
        "zc": np.array([-(k + 0.5) * DZ for k in range(NZ)]),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_results(result, model, grid_info: dict, output_dir: Path):
    os.makedirs(output_dir, exist_ok=True)
    x = grid_info["x"] / 1e3  # km
    zc = grid_info["zc"]       # m (negative)

    n_snaps = result.eta_snapshots.shape[0]
    snap_times = result.snapshot_times

    # --- Figure 1: Temperature cross-sections ---
    n_panels = min(n_snaps, 6)
    snap_indices = np.linspace(0, n_snaps - 1, n_panels, dtype=int)

    fig, axes = plt.subplots(2, n_panels, figsize=(4 * n_panels, 8),
                             squeeze=False, constrained_layout=True)
    for col, si in enumerate(snap_indices):
        T = result.temp_snapshots[si, 0, :, 0, :].numpy()  # [nz, nx]

        # Mask land cells with NaN for clean plotting
        if model.grid.masks.has_3d_masks:
            mask = model.grid.masks.center[:, 0, :].numpy()
            T = np.where(mask, T, np.nan)

        ax = axes[0, col]
        im = ax.pcolormesh(x, zc, T, cmap="RdYlBu_r", shading="nearest")
        ax.set_title(f"T @ t={snap_times[si]/3600:.1f}h", fontsize=10)
        ax.set_ylabel("depth (m)")
        ax.set_xlabel("x (km)")
        plt.colorbar(im, ax=ax, label="°C")

        # Plot bathymetry line
        bathy = grid_info["bathy"]
        ax.fill_between(x, -bathy, zc[-1] - 5, color="0.4", alpha=0.5)

        # u velocity cross-section
        u = result.u_snapshots[si, 0, :, 0, :].numpy()  # [nz, nx+1]
        u_c = 0.5 * (u[:, :-1] + u[:, 1:])  # center average -> [nz, nx]
        if model.grid.masks.has_3d_masks:
            u_c = np.where(mask, u_c, np.nan)

        ax = axes[1, col]
        vmax = max(abs(np.nanmin(u_c)), abs(np.nanmax(u_c)), 1e-6)
        im = ax.pcolormesh(x, zc, u_c, cmap="RdBu_r", shading="nearest",
                           vmin=-vmax, vmax=vmax)
        ax.set_title(f"u @ t={snap_times[si]/3600:.1f}h", fontsize=10)
        ax.set_ylabel("depth (m)")
        ax.set_xlabel("x (km)")
        plt.colorbar(im, ax=ax, label="m/s")
        ax.fill_between(x, -bathy, zc[-1] - 5, color="0.4", alpha=0.5)

    fig.suptitle("MITgcm internal_wave → LOCOMgcm: T & u Cross-Sections (x-z)",
                 fontsize=13, fontweight="bold")
    fig.savefig(output_dir / "internal_wave_T_u_sections.png", dpi=150)
    plt.close(fig)

    # --- Figure 2: Conservation diagnostics ---
    history = result.history
    times_h = [h["time"] / 3600.0 for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    # KE
    ke = [h.get("ke", 0.0) for h in history]
    axes[0, 0].plot(times_h, ke, "r-", lw=1.5)
    axes[0, 0].set_ylabel("KE (J/m²)")
    axes[0, 0].set_xlabel("time (h)")
    axes[0, 0].set_title("Kinetic Energy")
    axes[0, 0].grid(True, alpha=0.3)

    # Free surface
    fse = [h.get("free_surface_energy", 0.0) for h in history]
    axes[0, 1].plot(times_h, fse, "b-", lw=1.5)
    axes[0, 1].set_ylabel("FSE (J/m²)")
    axes[0, 1].set_xlabel("time (h)")
    axes[0, 1].set_title("Free Surface Energy")
    axes[0, 1].grid(True, alpha=0.3)

    # Temp mass
    tm = [h.get("temp_mass", 0.0) for h in history]
    if tm[0] != 0:
        tm_rel = [(t - tm[0]) / abs(tm[0]) for t in tm]
        axes[1, 0].plot(times_h, tm_rel, "g-", lw=1.5)
        axes[1, 0].set_ylabel("relative drift")
    else:
        axes[1, 0].plot(times_h, tm, "g-", lw=1.5)
        axes[1, 0].set_ylabel("temp mass")
    axes[1, 0].set_xlabel("time (h)")
    axes[1, 0].set_title("Temperature Mass Conservation")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].ticklabel_format(style="sci", axis="y", scilimits=(-3, 3))

    # η range
    eta_max = [h.get("eta_max", 0.0) if "eta_max" in h
               else float(result.eta_snapshots[min(i, n_snaps - 1)].abs().max())
               for i, h in enumerate(history)]
    axes[1, 1].plot(times_h, eta_max, "m-", lw=1.5)
    axes[1, 1].set_ylabel("|η|_max (m)")
    axes[1, 1].set_xlabel("time (h)")
    axes[1, 1].set_title("Free Surface Amplitude")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].ticklabel_format(style="sci", axis="y", scilimits=(-3, 3))

    fig.suptitle("MITgcm internal_wave → LOCOMgcm: Diagnostics",
                 fontsize=13, fontweight="bold")
    fig.savefig(output_dir / "internal_wave_conservation.png", dpi=150)
    plt.close(fig)

    # --- Figure 3: Bathymetry profile ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 4), constrained_layout=True)
    bathy = grid_info["bathy"]
    ax.fill_between(x, -bathy, 0, color="0.6", alpha=0.7, label="land")
    ax.plot(x, -bathy, "k-", lw=2)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("depth (m)")
    ax.set_title("Bathymetry Profile")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / "internal_wave_bathymetry.png", dpi=150)
    plt.close(fig)

    print(f"\n✓ Plots saved to {output_dir}/")


def print_summary(result, model) -> dict:
    """Print quantitative validation summary and return metrics dict."""
    history = result.history
    n_steps = int(history[-1]["step"])
    total_time_h = history[-1]["time"] / 3600.0

    tm = [h.get("temp_mass", 0.0) for h in history]
    ke = [h.get("ke", 0.0) for h in history]

    # For open-boundary case, temp mass will change due to boundary fluxes
    # So we track the magnitude of change, not strict conservation
    tm_drift_abs = abs(tm[-1] - tm[0])
    tm_drift_rel = tm_drift_abs / max(abs(tm[0]), 1e-30) if tm[0] != 0 else tm_drift_abs

    final_temp = result.temp_snapshots[-1]
    is_finite = bool(torch.isfinite(final_temp).all())
    final_temp_range = float(final_temp.max() - final_temp.min())

    initial_temp = result.temp_snapshots[0]
    temp_changed = float((final_temp - initial_temp).abs().max())
    ke_final = ke[-1] if ke else 0.0
    ke_max = max(ke) if ke else 0.0
    u_final_max = float(result.u_snapshots[-1].abs().max())

    # Check land mask
    if model.grid.masks.has_3d_masks:
        land_mask = ~model.grid.masks.center  # True where land
        land_temp = final_temp[0][land_mask]  # batch 0
        land_all_zero = bool((land_temp.abs() < 1e-30).all()) if land_temp.numel() > 0 else True
    else:
        land_all_zero = True

    metrics = {
        "total_steps": n_steps,
        "total_time_hours": total_time_h,
        "is_finite": is_finite,
        "temp_drift_relative": tm_drift_rel,
        "temp_range_final": final_temp_range,
        "max_temp_change": temp_changed,
        "ke_final": ke_final,
        "ke_max": ke_max,
        "u_max_final": u_final_max,
        "land_mask_correct": land_all_zero,
        "mitgcm_case": "internal_wave",
        "mitgcm_params": {
            "gravity": GRAVITY,
            "f0": F0,
            "alpha": ALPHA,
            "N2": N2,
            "ob_timescale": OB_TIMESCALE,
            "u_inflow": U_INFLOW,
            "H": H,
            "nx": NX, "ny": NY, "nz": NZ,
        },
    }

    verdicts = {}
    verdicts["stability"] = "PASS" if is_finite else "FAIL"
    # Open boundary means temp mass will change (wave inflow carries heat)
    verdicts["temp_boundary_flux"] = "PASS" if tm_drift_rel < 1.0 else "WARN"
    verdicts["physical_evolution"] = "PASS" if temp_changed > 1e-4 else "FAIL"
    verdicts["wave_signal"] = "PASS" if u_final_max > 1e-4 else "FAIL"
    verdicts["land_mask"] = "PASS" if land_all_zero else "FAIL"

    print("\n" + "=" * 70)
    print(" MITgcm internal_wave → LOCOMgcm Verification Report")
    print("=" * 70)
    print(f"\n  Case:           MITgcm verification/internal_wave")
    print(f"  Physics:        2D x-z, linear EOS, open boundary, sloping topography")
    print(f"  Grid:           nx={NX}, ny={NY}, nz={NZ}")
    print(f"  Domain:         Lx={model.config.lx/1e3:.1f} km × H={H:.0f} m")
    print(f"  Duration:       {total_time_h:.1f} hours ({n_steps} steps)")
    print(f"  dt:             {model.config.default_dt:.1f} s")

    print(f"\n{'  Metric':<35} {'Value':>15} {'Verdict':>10}")
    print(f"  {'─' * 60}")
    print(f"  {'Numerical stability':<35} {'finite' if is_finite else 'NaN/Inf':>15} {verdicts['stability']:>10}")
    print(f"  {'Temp mass drift (relative)':<35} {tm_drift_rel:>15.2e} {verdicts['temp_boundary_flux']:>10}")
    print(f"  {'Max ΔT evolution':<35} {temp_changed:>15.6f} {verdicts['physical_evolution']:>10}")
    print(f"  {'KE peak (wave signal)':<35} {ke_max:>15.4e} {verdicts['wave_signal']:>10}")
    print(f"  {'Land mask correct':<35} {str(land_all_zero):>15} {verdicts['land_mask']:>10}")
    print(f"  {'Max |u|':<35} {u_final_max:>15.6f}")
    print(f"  {'Temp range final':<35} {final_temp_range:>15.6f}")

    overall = "PASS" if all(v == "PASS" for v in verdicts.values()) else (
        "WARN" if all(v != "FAIL" for v in verdicts.values()) else "FAIL"
    )
    print(f"\n  {'Overall verdict':<35} {'':>15} {overall:>10}")
    print("=" * 70)

    metrics["verdicts"] = verdicts
    metrics["overall"] = overall
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="MITgcm internal_wave verification for LOCOMgcm"
    )
    parser.add_argument("--hours", type=float, default=14.0,
                        help="Simulation duration in hours")
    parser.add_argument("--dt", type=float, default=50.0,
                        help="Baroclinic time step in seconds (split-explicit handles barotropic CFL)")
    parser.add_argument("--fast-substeps", type=int, default=13,
                        help="Number of barotropic fast substeps per baroclinic step")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu or cuda)")
    parser.add_argument("--output-dir", type=str,
                        default="outputs/mitgcm_internal_wave",
                        help="Output directory")
    parser.add_argument("--snapshot-interval-hours", type=float, default=2.0,
                        help="Snapshot interval in hours")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("━" * 70)
    print("  MITgcm internal_wave ──▶ LOCOMgcm Verification")
    print("━" * 70)

    # Step 1: Build model
    print("\n[1/3] Building LOCOMgcm model with bathymetry...")
    model, initial_state, config, grid_info = build_model(
        device=args.device, dt=args.dt, fast_substeps=args.fast_substeps,
    )
    print(f"  Grid: nx={NX}, ny={NY}, nz={NZ}")
    print(f"  Domain: Lx={config.lx/1e3:.1f} km × H={H:.0f} m")
    print(f"  Bathymetry: {grid_info['bathy'][0]:.0f}m (west) → "
          f"{grid_info['bathy'][-2]:.0f}m → land (east)")
    print(f"  3D mask: {model.grid.masks.has_3d_masks}")
    if model.grid.masks.has_3d_masks:
        n_wet = int(model.grid.masks.center.sum().item())
        n_total = NZ * NY * NX
        print(f"  Wet cells: {n_wet}/{n_total} ({100*n_wet/n_total:.1f}%)")
    print(f"  dt={args.dt:.0f} s, duration={args.hours:.1f} h")
    print(f"  Device: {args.device}")

    # Step 2: Run
    print("\n[2/3] Running simulation...")
    from locomgcm import ModelRunner

    steps = int(args.hours * 3600 / args.dt)
    snap_interval = max(1, int(args.snapshot_interval_hours * 3600 / args.dt))

    t0 = time.time()
    runner = ModelRunner()
    result = runner.run(
        model, initial_state,
        steps=steps,
        dt=args.dt,
        snapshot_interval=snap_interval,
    )
    elapsed = time.time() - t0
    print(f"\n  Simulation completed in {elapsed:.1f} s ({steps} steps)")

    # Step 3: Plot and report
    print("\n[3/3] Generating plots and report...")
    plot_results(result, model, grid_info, output_dir)
    metrics = print_summary(result, model)

    metrics_path = output_dir / "verification_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\n  Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
