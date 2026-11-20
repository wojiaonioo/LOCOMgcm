"""Cube-global conservation diagnostics.

The regional ``DiagnosticSuite`` defaults (conservation.py) assume a
single-patch ``[batch, nz, ny, nx]`` state with ``[ny, nx]`` metrics and
would mis-broadcast on the FACE-FIRST cube layout (``[6, nz, ny, nx]``
state, ``[6, ny, nx]`` cell_area).  These modules are their cube-aware
counterparts, written in negative-index/broadcast form so the leading
face dim is just a batch axis — they work unchanged on the global
``[6, ...]`` state and on a rank-local ``[d, ...]`` face subset
(``SUPPORTED_RANKS`` {2,3,6} -> d in {3,2,1}).

Two deliberate metric-convention differences vs the regional modules
(documented, not drift):

* Tracer mass and kinetic energy are **hFacC-weighted** (partial-cell
  volumes). The regional ``TracerMassDiagnostic`` uses unweighted
  ``drF * rA``; on the cube's realistic bathymetry an unweighted sum is
  not a conserved quantity, so the hFac weighting is the meaningful one.
* Kinetic energy uses a plain two-point face->center average at
  monitor-level fidelity.  It is NOT a MITgcm-aligned KE and must not be
  used as an alignment metric (see docs/VALIDATION.md for that protocol).

``CubeConservationMonitor`` is the runtime front-end: per-interval
compute, a <=2-collective all_reduce when distributed, one rank-0 log
line and a ``cube_conservation.jsonl`` record.  It is an observer — it
must never mutate the state or the stepper cache.  Failure policy is
mode-dependent: single-rank failures degrade the monitor to OFF (an
observer must not kill the run); distributed failures propagate, because
a rank-local degrade would desync the collective schedule and hang the
other ranks — a crash is diagnosable, a deadlock is not.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist

from defgcm.state import PrognosticState


def _hfac_cell_volume(grid: object, like: torch.Tensor) -> torch.Tensor:
    """Partial-cell volume ``hFacC * drF * rA`` shaped ``[d, nz, ny, nx]``.

    ``hFacC`` carries the leading face dim (``[d/6, nz, ny, nx]``);
    ``cell_area`` is ``[d/6, ny, nx]`` -> ``unsqueeze(-3)`` = ``[.., 1, ny, nx]``;
    ``layer_thickness`` is ``[nz]`` -> ``view(nz, 1, 1)``.
    """
    hfac = grid.masks.hFacC.to(dtype=like.dtype, device=like.device)
    drf = grid.layer_thickness.to(dtype=like.dtype, device=like.device).view(-1, 1, 1)
    area = grid.metrics.cell_area.to(dtype=like.dtype, device=like.device).unsqueeze(-3)
    return hfac * drf * area


@dataclass(frozen=True)
class CubeSurfaceMassDiagnostic:
    """Free-surface volume ``sum(eta * rA)`` (m^3)."""

    name: str = "cube_eta_volume"
    reductions = {"cube_eta_volume": "sum"}

    def compute(
        self,
        state: PrognosticState,
        grid: object,
        *,
        gravity: float,
        dt: float | None = None,
        context: object | None = None,
    ) -> dict[str, float]:
        del gravity, dt, context
        eta = state.eta.data
        area = grid.metrics.cell_area.to(dtype=eta.dtype, device=eta.device)
        return {self.name: float((eta * area).sum().item())}


@dataclass(frozen=True)
class CubeTracerMassDiagnostic:
    """hFacC-weighted tracer content ``sum(tr * hFacC * drF * rA)``."""

    field_name: str
    name: str
    # populated per-instance below (dataclass frozen -> property style)

    @property
    def reductions(self) -> dict[str, str]:
        return {self.name: "sum"}

    def compute(
        self,
        state: PrognosticState,
        grid: object,
        *,
        gravity: float,
        dt: float | None = None,
        context: object | None = None,
    ) -> dict[str, float]:
        del gravity, dt, context
        tracer = getattr(state, self.field_name).data
        volume = _hfac_cell_volume(grid, tracer)
        return {self.name: float((tracer * volume).sum().item())}


@dataclass(frozen=True)
class CubeKineticEnergyDiagnostic:
    """hFacC-weighted KE ``0.5 * sum((u_c^2 + v_c^2) * hFacC * drF * rA)``.

    ``u_c`` / ``v_c`` are plain two-point face->center averages
    (monitor-level; NOT a MITgcm-aligned KE — do not use as an alignment
    metric).
    """

    name: str = "cube_kinetic_energy"
    reductions = {"cube_kinetic_energy": "sum"}

    def compute(
        self,
        state: PrognosticState,
        grid: object,
        *,
        gravity: float,
        dt: float | None = None,
        context: object | None = None,
    ) -> dict[str, float]:
        del gravity, dt, context
        u = state.u.data
        v = state.v.data
        u_c = 0.5 * (u[..., :-1] + u[..., 1:])
        v_c = 0.5 * (v[..., :-1, :] + v[..., 1:, :])
        volume = _hfac_cell_volume(grid, u_c)
        ke = 0.5 * ((u_c.square() + v_c.square()) * volume).sum()
        return {self.name: float(ke.item())}


@dataclass(frozen=True)
class CubeStateExtremaDiagnostic:
    """Global field extrema (blow-up watch + T/S range drift)."""

    name: str = "cube_state_extrema"
    # min entries are reduced as max(-x) then negated back at report time.
    reductions = {
        "cube_eta_max_abs": "max",
        "cube_u_max_abs": "max",
        "cube_v_max_abs": "max",
        "cube_temp_min": "min",
        "cube_temp_max": "max",
        "cube_salt_min": "min",
        "cube_salt_max": "max",
    }

    def compute(
        self,
        state: PrognosticState,
        grid: object,
        *,
        gravity: float,
        dt: float | None = None,
        context: object | None = None,
    ) -> dict[str, float]:
        del grid, gravity, dt, context
        return {
            "cube_eta_max_abs": float(state.eta.data.abs().max().item()),
            "cube_u_max_abs": float(state.u.data.abs().max().item()),
            "cube_v_max_abs": float(state.v.data.abs().max().item()),
            "cube_temp_min": float(state.temp.data.min().item()),
            "cube_temp_max": float(state.temp.data.max().item()),
            "cube_salt_min": float(state.salt.data.min().item()),
            "cube_salt_max": float(state.salt.data.max().item()),
        }


_DEFAULT_MODULES = (
    CubeSurfaceMassDiagnostic(),
    CubeTracerMassDiagnostic(field_name="temp", name="cube_temp_mass"),
    CubeTracerMassDiagnostic(field_name="salt", name="cube_salt_mass"),
    CubeKineticEnergyDiagnostic(),
    CubeStateExtremaDiagnostic(),
)


class CubeConservationMonitor:
    """Per-interval cube conservation watch for the cube run loop.

    All ranks must call :meth:`maybe_record` at the same steps (it may
    issue collectives); only rank 0 logs/writes.  Single-rank: a failure
    inside the monitor degrades it to OFF with a one-shot warning
    (observer, not part of the stepping).  Distributed: failures
    propagate — see the module docstring's failure policy.
    """

    def __init__(
        self,
        grid: object,
        *,
        gravity: float,
        out_path: Path | None,
        is_rank0: bool,
        interval: int,
    ) -> None:
        self.grid = grid
        self.gravity = float(gravity)
        self.out_path = Path(out_path) if out_path is not None else None
        self.is_rank0 = bool(is_rank0)
        self.interval = max(1, int(interval))
        self.modules = _DEFAULT_MODULES
        self._disabled = False
        # Stable key order for the packed all_reduce tensors.
        self._sum_keys: list[str] = []
        self._max_keys: list[str] = []  # min keys ride along as max(-x)
        self._min_keys: set[str] = set()
        for module in self.modules:
            for key, red in module.reductions.items():
                if red == "sum":
                    self._sum_keys.append(key)
                else:
                    self._max_keys.append(key)
                    if red == "min":
                        self._min_keys.add(key)

    def maybe_record(self, step: int, state: PrognosticState) -> dict[str, float] | None:
        if self._disabled or step % self.interval != 0:
            return None
        if dist.is_available() and dist.is_initialized():
            # Distributed: do NOT catch.  A rank-local degrade would desync the
            # packed all_reduce schedule and hang every other rank at the next
            # interval — a deterministic crash on all ranks is strictly better
            # than an undiagnosable deadlock.
            return self._record(step, state)
        try:
            return self._record(step, state)
        except Exception as exc:  # noqa: BLE001 — single-rank observer must not kill the run
            self._disabled = True
            if self.is_rank0:
                print(
                    f"  [conserv] WARN: monitor failed ({exc!r}); disabling for the "
                    "rest of the run",
                    flush=True,
                )
            return None

    def _record(self, step: int, state: PrognosticState) -> dict[str, float]:
        local: dict[str, float] = {}
        for module in self.modules:
            local.update(
                module.compute(state, self.grid, gravity=self.gravity, dt=None)
            )

        scalars = dict(local)
        if dist.is_available() and dist.is_initialized():
            # <=2 collectives: one packed SUM, one packed MAX (min as max(-x)).
            if self._sum_keys:
                buf = torch.tensor(
                    [local[k] for k in self._sum_keys], dtype=torch.float64
                )
                from defgcm.distributed.transport_failure import data_all_reduce
                data_all_reduce(buf, op=dist.ReduceOp.SUM)
                scalars.update(
                    {k: float(v) for k, v in zip(self._sum_keys, buf.tolist())}
                )
            if self._max_keys:
                vals = [
                    -local[k] if k in self._min_keys else local[k]
                    for k in self._max_keys
                ]
                buf = torch.tensor(vals, dtype=torch.float64)
                from defgcm.distributed.transport_failure import data_all_reduce
                data_all_reduce(buf, op=dist.ReduceOp.MAX)
                for k, v in zip(self._max_keys, buf.tolist()):
                    scalars[k] = float(-v) if k in self._min_keys else float(v)

        if self.is_rank0:
            print(
                f"  [conserv] step {step:6d}: "
                f"eta_vol={scalars['cube_eta_volume']:.9e} "
                f"T_mass={scalars['cube_temp_mass']:.9e} "
                f"S_mass={scalars['cube_salt_mass']:.9e} "
                f"KE={scalars['cube_kinetic_energy']:.9e} "
                f"|u|max={scalars['cube_u_max_abs']:.4e}",
                flush=True,
            )
            if self.out_path is not None:
                record = {"step": int(step), **scalars}
                with self.out_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, sort_keys=True) + "\n")
        return scalars
