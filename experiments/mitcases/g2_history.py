"""G2 Tier-1 diagnostics and ``to-g2-history/v1`` artifact production."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from experiments.mitcases.g2_compare import (
    ARTIFACT_SCHEMA,
    EXECUTION_IDENTITY_REQUIRED_FIELDS,
    TIER0_KEYS,
    TIER1_KEYS,
)
from locomgcm.dynamics.vertical_velocity import diagnose_vertical_velocity
from locomgcm.geometry.metric_bundle import build_tracer_metric_bundle
from locomgcm.steppers.base import TimeStepBundle


def _require_float64_finite(value: torch.Tensor, label: str) -> None:
    if value.dtype != torch.float64:
        raise ValueError(f"{label} must be torch.float64, got {value.dtype}")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{label} contains non-finite values")


def advcfl_diagnostics(
    state: Any,
    grid: Any,
    *,
    time_step: TimeStepBundle,
) -> dict[str, float]:
    """Compute MIT ``MON_ADVCFL`` maxima from the current step-start state.

    The vertical velocity returned by LOCOMgcm is downward-positive while
    MITgcm's ``wVel`` is upward-positive.  The monitor takes an absolute value,
    so no convention conversion is needed here.
    """
    u = state.u.data
    v = state.v.data
    _require_float64_finite(u, "state.u")
    _require_float64_finite(v, "state.v")
    metrics = build_tracer_metric_bundle(grid, u.dtype, u.device)
    d_t_adv = max(float(time_step.dt_tracer), float(time_step.dt_mom))

    # MIT loops I=1..sNx/J=1..sNy.  LOCOMgcm carries one extra east/north
    # face, which is outside that physical monitor domain.
    u_physical = u[..., : grid.ny, : grid.nx]
    recip_dx_c = metrics.recip_dxC[..., : grid.ny, : grid.nx]
    u_cfl = u_physical.abs() * recip_dx_c
    u_cfl = u_cfl * d_t_adv

    v_physical = v[..., : grid.ny, : grid.nx]
    recip_dy_c = metrics.recip_dyC[..., : grid.ny, : grid.nx]
    v_cfl = v_physical.abs() * recip_dy_c
    v_cfl = v_cfl * d_t_adv

    _, w_interface, _ = diagnose_vertical_velocity(state, grid)
    _require_float64_finite(w_interface, "continuity vertical velocity")
    layer_thickness = grid.layer_thickness_for(u.dtype, u.device)
    center_depth = grid.center_depth.to(dtype=u.dtype, device=u.device)
    dr_c = torch.empty_like(layer_thickness)
    dr_c[0] = 0.5 * layer_thickness[0]
    if grid.nz > 1:
        dr_c[1:] = center_depth[1:] - center_depth[:-1]
    if not bool((dr_c > 0.0).all().item()):
        raise ValueError("vertical center spacing must be positive")
    recip_dr_c = 1.0 / dr_c
    recip_shape = (1,) * (w_interface.ndim - 3) + (grid.nz, 1, 1)
    w_physical = w_interface[..., : grid.nz, : grid.ny, : grid.nx]
    w_cfl = w_physical.abs() * recip_dr_c.view(recip_shape)
    w_cfl = w_cfl * d_t_adv

    maxima = torch.stack((u_cfl.max(), v_cfl.max(), w_cfl.max()))
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
    result = {
        key: float(value.item()) for key, value in zip(TIER1_KEYS, maxima, strict=True)
    }
    if any(not math.isfinite(value) for value in result.values()):
        raise ValueError("advection CFL diagnostic is non-finite")
    return result


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _validated_tier0(tier0: Mapping[str, object]) -> dict[str, object]:
    values_raw = tier0.get("values")
    unavailable_raw = tier0.get("unavailable")
    if not isinstance(values_raw, Mapping) or not isinstance(unavailable_raw, Mapping):
        raise ValueError("tier0 must contain values and unavailable mappings")
    value_keys = set(values_raw)
    unavailable_keys = set(unavailable_raw)
    expected = set(TIER0_KEYS)
    if value_keys & unavailable_keys:
        raise ValueError("tier0 values and unavailable keys must not overlap")
    if value_keys | unavailable_keys != expected:
        missing = sorted(expected - value_keys - unavailable_keys)
        extra = sorted((value_keys | unavailable_keys) - expected)
        raise ValueError(f"tier0 key mismatch: missing={missing}, extra={extra}")
    values = {
        key: _finite_number(values_raw[key], f"tier0.values.{key}")
        for key in TIER0_KEYS
        if key in values_raw
    }
    unavailable: dict[str, str] = {}
    for key in TIER0_KEYS:
        if key not in unavailable_raw:
            continue
        reason = unavailable_raw[key]
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"tier0.unavailable.{key} must be a non-empty string")
        unavailable[key] = reason
    return {"values": values, "unavailable": unavailable}


def g2_history_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return the explicit G2 side-channel switch, rejecting ambiguous values."""
    env = os.environ if environ is None else environ
    raw = env.get("LOCOMGCM_G2_HISTORY", "0")
    if raw not in {"0", "1"}:
        raise ValueError("LOCOMGCM_G2_HISTORY must be 0 or 1")
    return raw == "1"


def runtime_producer(
    provenance: Mapping[str, object],
    *,
    state: Any,
    distributed_world_size: int,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Build the artifact producer block from stage provenance and runtime facts."""
    git = provenance.get("git")
    environment = provenance.get("environment")
    if not isinstance(git, Mapping) or not isinstance(environment, Mapping):
        raise ValueError("provenance must contain git and environment mappings")
    if (
        isinstance(distributed_world_size, bool)
        or not isinstance(distributed_world_size, int)
        or distributed_world_size < 1
    ):
        raise ValueError("distributed_world_size must be a positive integer")
    env = os.environ if environ is None else environ
    ivadv_backend = env.get("LOCOMGCM_IVADV_FUSED_BACKEND", "")
    identity: dict[str, object] = {
        "device": state.u.data.device.type,
        "dtype": str(state.u.data.dtype).removeprefix("torch."),
        "distributed_world_size": distributed_world_size,
        "IVADV": int(ivadv_backend not in {"", "0", "off", "disabled"}),
        # legacy identity key name kept to preserve frozen G2 fingerprints;
        # reads the successor switch (the Triton STATEKPP implementation is retired)
        "TRITON_KPP": int(env.get("LOCOMGCM_KPP_STATEKPP_CUDA") == "1"),
        "FUSED": int(env.get("LOCOMGCM_CG3D_FUSED_PCG") == "1"),
    }
    for key in ("torch_version", "cuda_available", "cuda_version", "gpu_name"):
        if key in environment:
            identity[key] = environment[key]
    return {
        "commit": git.get("commit"),
        "dirty": git.get("dirty"),
        "execution_identity": identity,
    }


class G2HistoryCollector:
    """Collect step-start Tier-1 records on an explicit absolute clock."""

    def __init__(
        self,
        *,
        base_time_seconds: float,
        start_step: int,
        time_step: TimeStepBundle,
    ) -> None:
        self.base_time_seconds = _finite_number(base_time_seconds, "base_time_seconds")
        if (
            isinstance(start_step, bool)
            or not isinstance(start_step, int)
            or start_step < 0
        ):
            raise ValueError("start_step must be a non-negative integer")
        self.start_step = start_step
        self.time_step = time_step
        self.records: list[dict[str, object]] = []

    def record(self, local_step: int, values: Mapping[str, object]) -> None:
        if (
            isinstance(local_step, bool)
            or not isinstance(local_step, int)
            or local_step < 0
        ):
            raise ValueError("local_step must be a non-negative integer")
        if self.records and local_step <= int(self.records[-1]["local_step"]):
            raise ValueError("local_step must be strictly increasing")
        if set(values) != set(TIER1_KEYS):
            missing = sorted(set(TIER1_KEYS) - set(values))
            extra = sorted(set(values) - set(TIER1_KEYS))
            raise ValueError(f"Tier-1 key mismatch: missing={missing}, extra={extra}")
        normalized = {
            key: _finite_number(values[key], f"values.{key}") for key in TIER1_KEYS
        }
        global_step = self.start_step + local_step
        dt_clock = float(self.time_step.dt_clock)
        self.records.append(
            {
                "local_step": local_step,
                "global_step": global_step,
                "elapsed_seconds": local_step * dt_clock,
                "model_time_seconds": self.base_time_seconds + global_step * dt_clock,
                "values": normalized,
            }
        )

    def sample(self, local_step: int, state: Any, grid: Any) -> None:
        self.record(
            local_step,
            advcfl_diagnostics(state, grid, time_step=self.time_step),
        )

    def build_artifact(
        self,
        *,
        case: str,
        variant: str | None,
        producer: Mapping[str, object],
        tier0: Mapping[str, object],
    ) -> dict[str, object]:
        if not isinstance(case, str) or not case.strip() or case.strip() != case:
            raise ValueError("case must be a non-empty trimmed string")
        if variant is not None and (
            not isinstance(variant, str)
            or not variant.strip()
            or variant.strip() != variant
        ):
            raise ValueError("variant must be None or a non-empty trimmed string")
        if not self.records:
            raise ValueError("history collector has no records")
        commit = producer.get("commit")
        if (
            not isinstance(commit, str)
            or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)
        ):
            raise ValueError("producer.commit must be a lowercase 40-hex commit")
        dirty = producer.get("dirty")
        if not isinstance(dirty, bool):
            raise ValueError("producer.dirty must be boolean")
        identity = producer.get("execution_identity")
        if not isinstance(identity, Mapping):
            raise ValueError("producer.execution_identity must be a mapping")
        missing_identity = sorted(EXECUTION_IDENTITY_REQUIRED_FIELDS - identity.keys())
        if missing_identity:
            raise ValueError(
                f"producer.execution_identity missing fields: {missing_identity}"
            )
        dt_clock = float(self.time_step.dt_clock)
        return {
            "schema": ARTIFACT_SCHEMA,
            "case": case,
            "variant": variant,
            "producer": {
                "commit": commit,
                "dirty": dirty,
                "execution_identity": dict(identity),
            },
            "clock": {
                "base_time_seconds": self.base_time_seconds,
                "start_time_seconds": (
                    self.base_time_seconds + self.start_step * dt_clock
                ),
                "start_iteration": self.start_step,
                "start_step": self.start_step,
                "dt_clock": dt_clock,
            },
            "records": list(self.records),
            "tier0": _validated_tier0(tier0),
        }


def write_history_artifact(
    path: Path | str,
    artifact: Mapping[str, object],
) -> Path:
    """Atomically serialize one deterministic JSON history artifact.

    Returns the written path; the path and the producer commit identify the
    artifact.
    """
    target = Path(path)
    payload = (
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(target)
    return target


__all__ = [
    "G2HistoryCollector",
    "advcfl_diagnostics",
    "g2_history_enabled",
    "runtime_producer",
    "write_history_artifact",
]
