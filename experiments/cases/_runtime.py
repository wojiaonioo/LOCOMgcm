"""Case-agnostic yaml framework runtime for pipeline stage2."""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import re
import socket
import statistics
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import numpy as np
import torch
import torch.distributed as dist
import yaml

from defgcm.distributed import (
    DistributedContext,
    ParallelLayout,
    gather_prognostic_state_to_rank0,
    init_distributed,
    layout_from_rank,
    localize_boundary_bundle_source,
    localize_bulk_source,
    localize_ic_tensors,
    localize_surface_flux_source,
    require_minimum_tile_extent,
    localize_structured_grid,
    localize_tidal_config,
    localize_wind_source,
)
from defgcm.distributed.bootstrap import shutdown_distributed
from defgcm.distributed.localize import localize_center_field
from defgcm.operators.fluxes.tracer_flux import TRACER_SCHEME_BY_NAME
from defgcm.io.runner.builders import (
    BoundaryBundleSource,
    NCVarSpec,
    build_grid_from_run_cfg,
    build_model_from_manifest,
    current_frame_time_seconds,
    default_state_variables,
    make_nc_writer,
    replace_variable,
)
from defgcm.io.runner.builders.from_manifest import _tracer_flux_scheme
from defgcm.io.runner.builders.nc_snapshot import latlon_centre_axes
from defgcm.io.runner.config._schema import DynamicsEntry, ModelManifest
from defgcm.io.runner.layer_a import (
    DataBundle,
    LocationKind,
    ProvenanceStep,
    VariableSpec,
)
from defgcm.io.runner.layer_b import DATA_SOURCES, DataQuery, Grid, SourceContext
from defgcm.io.era5 import ERA5BulkSource, ERA5WindSource, load_era5_wind_source_by_time
from defgcm.io.runner._canonical_helpers import (
    BarotropicBoundaryBundleSource as _BarotropicBoundaryBundleSource,
    boundary_source_from_bundle as _boundary_source_from_bundle,
    build_bulk_source as _build_bulk_source,
    enrich_manifest as _enrich_manifest_core,
    reference_profile_from_tensors as _reference_profile_from_tensors,
    resolve_reference_profiles as _resolve_reference_profiles,
    runtime_kwargs as _enrich_runtime_kwargs,
    seconds_from_time_coord as _seconds_from_time_coord,
    surface_flux_source_from_bundle as _surface_flux_source_from_bundle,
    surface_pressure_load_from_bundle as _surface_pressure_load_from_bundle,
    uses_era5_bulk_forcing as _uses_era5_bulk_forcing,
    uses_surface_pressure_load as _uses_surface_pressure_load,
    wind_source_from_bundle as _wind_source_from_bundle,
)
from defgcm.io.runner.stages._common import _steps_from_config, build_run_metadata
from defgcm.runner import BenchmarkStepTimer, ModelRunner, RunnerCompileConfig
from defgcm.steppers.base import TimeStepBundle


_DUMP_GLOB = "Eta.*.001.001.data"

_BENCHMARK_ADVANCES = 100
_BENCHMARK_WARMUP = 20
_BENCHMARK_ACTIVITY_COVERAGE = (
    "frame_output",
    "snapshot_output",
    "checkpoint",
    "diagnostic_reduction",
    "station_output",
    "progress_output",
    "debug_output",
    "other_io",
)
_BENCHMARK_ACTIVITY_COUNTERS = (
    "frame_writes",
    "snapshot_writes",
    "checkpoint_operations",
    "diagnostic_reductions",
    "station_output_operations",
    "progress_output_operations",
    "debug_output_operations",
    "other_io_operations",
)
_BENCHMARK_ACCELERATOR_GATES = (
    "DEFGCM_IVADV_FUSED_BACKEND",
    "DEFGCM_KPP_STATEKPP_CUDA",
    # Tombstone: the Triton STATEKPP switch is retired (CUDA successor above).
    # A frozen benchmark check pins this key in its OFF inventory; keep emitting
    # it (always "unset") until that check is re-frozen, then delete this entry.
    "DEFGCM_TRITON_KPP",
    "DEFGCM_CG3D_FUSED_PCG",
    "DEFGCM_CG3D_USE_FAST_THOMAS",
    "DEFGCM_KPP_WSCALE_BATCH",
    "DEFGCM_KPP_STATIC_HALO_CACHE",
    "DEFGCM_KPP_FACE_BATCH",
    "DEFGCM_VECTORIZE_VERTSHEAR",
    "DEFGCM_IMPLDIFF_BUILD_BATCH",
    "DEFGCM_CG2D_CUDA_GRAPH",
    "DEFGCM_TRITON_VISC",
    "DEFGCM_DEVICE_RESIDENT_MASKS",
    "DEFGCM_CACHE_CONSTANTS",
    "DEFGCM_CUBE_LOCAL_GRID_CACHE",
)


def _benchmark_run_uuid(distributed_context: DistributedContext | None) -> str:
    value = str(uuid4()) if _rank0(distributed_context) else None
    if distributed_context is not None and distributed_context.is_distributed:
        values = [value]
        dist.broadcast_object_list(values, src=0)
        value = values[0]
    if not isinstance(value, str):
        raise RuntimeError("benchmark run UUID broadcast failed")
    return value


def _configure_benchmark_backend(device: str) -> None:
    resolved = torch.device(device)
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("benchmark requested CUDA but CUDA is unavailable")
        torch.cuda.reset_peak_memory_stats(resolved)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def _host_memory_peak() -> tuple[int, str]:
    """Resolve this task's host peak RSS, trying each cgroup generation in turn.

    Try cgroup v2 ``memory.peak``, then cgroup v1
    ``memory.max_usage_in_bytes``, then ``VmHWM``.

    ``VmHWM`` is a last resort and is labelled as such in the returned source
    string: it is the high-water mark of *this process only*, so under a
    multi-rank launch it understates the task total.  A caller that needs the
    whole-task figure must reject a ``proc:`` source rather than treat it as
    equivalent.
    """
    relative_v2 = "/"
    relative_v1 = None
    for line in Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        if fields[0] == "0" and fields[1] == "":
            relative_v2 = fields[2]
        elif "memory" in fields[1].split(","):
            relative_v1 = fields[2]

    root = Path("/sys/fs/cgroup")
    candidates: list[Path] = [
        root / relative_v2.lstrip("/") / "memory.peak",
        root / "memory.peak",
    ]
    if relative_v1 is not None:
        candidates.append(
            root / "memory" / relative_v1.lstrip("/") / "memory.max_usage_in_bytes"
        )
    candidates.append(root / "memory" / "memory.max_usage_in_bytes")

    seen: set[Path] = set()
    errors: list[str] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            raw = candidate.read_text(encoding="utf-8").strip()
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
            continue
        if raw.isdigit() and int(raw) > 0:
            return int(raw), str(candidate)
        errors.append(f"{candidate}: expected a positive byte count, got {raw!r}")

    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmHWM:"):
                kib = int(line.split()[1])
                if kib > 0:
                    return kib * 1024, "proc:/proc/self/status:VmHWM"
    except OSError as exc:  # pragma: no cover - /proc always present on Linux
        errors.append(f"/proc/self/status: {exc}")

    raise RuntimeError("cannot resolve host memory peak: " + "; ".join(errors))


def _classify_host_peak_source(node_peaks) -> str:
    """Name the file the host peak actually came from.

    Derive the label from the resolved paths so a fallback to cgroup v1
    or process RSS is reported accurately.
    """
    kinds = set()
    for _hostname, path in node_peaks:
        text = str(path)
        if text.startswith("proc:"):
            kinds.add("proc_vmhwm")
        elif text.endswith("memory.peak"):
            kinds.add("cgroup_v2_memory.peak")
        elif text.endswith("memory.max_usage_in_bytes"):
            kinds.add("cgroup_v1_max_usage_in_bytes")
        else:
            kinds.add("unknown")
    if not kinds:
        return "unknown"
    if len(kinds) == 1:
        return kinds.pop()
    return "mixed:" + ",".join(sorted(kinds))


def _benchmark_layout(
    distributed_context: DistributedContext | None,
    layout: ParallelLayout | None,
) -> tuple[int, int, int]:
    world_size = (
        distributed_context.world_size
        if distributed_context is not None and distributed_context.is_distributed
        else 1
    )
    if layout is None:
        return world_size, 1, 1
    return world_size, int(layout.nPx), int(layout.nPy)


def _posture_field(value: object) -> bool | None:
    return None if value is None else bool(value)


def _counter_field(value: object) -> int | None:
    """A counter this run recorded, or ``None`` when it recorded none.

    Never a default zero: an unrecorded counter and a measured zero read the
    same downstream, which is the mistake ``_BenchmarkActivityLedger`` below
    exists to prevent.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _benchmark_compile_record(metadata: Mapping[str, object]) -> dict[str, object]:
    return {
        "target": str(metadata.get("target", "disabled")),
        "enabled": bool(metadata.get("enabled", False)),
        "requested_target": metadata.get("requested_target", metadata.get("target", "disabled")),
        "requested_fullgraph": bool(metadata.get("requested_fullgraph", metadata.get("fullgraph", False))),
        "reason": metadata.get("reason"),
        # Shape posture of the traced step: ``False`` = static shapes (the
        # contract posture, the static-shape compilation contract),
        # ``True`` = the diagnostic dynamic-shape trace, ``None`` = the
        # automatic-dynamic diagnostic (``DEFGCM_COMPILE_DYNAMIC_AUTO=1``).  An
        # absent field is the static posture.  Additive, schema_version stays
        # 1.0.0.
        "dynamic": _posture_field(metadata.get("dynamic", False)),
        # Record whether the Inductor reordering passes were off
        # for the compiled step (multi-rank launch-order pin).  An absent field
        # is ``False`` (torch's default scheduling).  Additive, schema_version
        # stays 1.0.0.
        "pin_schedule": bool(metadata.get("pin_schedule", False)),
        # Record trace-time graph breaks and compiled graphs separately from
        # recompiles so changes in graph structure remain visible.
        "graph_break": _counter_field(metadata.get("graph_break")),
        "unique_graphs": _counter_field(metadata.get("unique_graphs")),
        "eager_warmup_steps": _counter_field(metadata.get("eager_warmup_steps")),
    }


class _BenchmarkActivityLedger:
    """Provenance for the benchmark's per-advance activity counters.

    The artifact used to build ``per_advance``
    out of ``dict.fromkeys(_BENCHMARK_ACTIVITY_COUNTERS, 0)``, i.e. the zeroes
    were *written*, never *read*.  A frozen benchmark check then reported "eight
    forbidden activity classes were observed ... with zero operations", a claim
    the document could not support: a fabricated zero and a measured zero are
    the same bytes, so no consumer could tell "no operation" from "no
    observation".  The frozen tree happens to have no producer for any of the
    eight classes, but the
    artifact would keep printing zeroes if one appeared.

    Every covered class must therefore arrive here one of two ways before the
    document can be emitted:

    * ``watch()`` -- its production site calls ``observe()``, so a non-zero
      count reaches the artifact and fails C8 loudly instead of vanishing.
    * ``declare_suppressed()`` -- the producer is provably absent in this run,
      and the *evidence* (the object that would have produced it) is checked
      here rather than asserted in prose.

    Anything else raises: the writer refuses to publish a zero it cannot back.
    """

    __slots__ = ("_advances", "_observed", "_suppressed", "_post_loop")

    def __init__(self, advances: int) -> None:
        self._advances = int(advances)
        self._observed: dict[str, dict[int, int]] = {}
        self._suppressed: dict[str, str] = {}
        self._post_loop: dict[str, int] = {}

    @staticmethod
    def _check_name(counter: str) -> None:
        if counter not in _BENCHMARK_ACTIVITY_COUNTERS:
            raise ValueError(
                f"{counter!r} is not one of the covered activity counters "
                f"{_BENCHMARK_ACTIVITY_COUNTERS}"
            )

    def watch(self, counter: str) -> None:
        """Register ``counter`` as instrumented; its site must call ``observe``."""
        self._check_name(counter)
        self._observed.setdefault(counter, {})

    def observe(self, counter: str, advance_index: int) -> None:
        """Record one operation.

        An index outside 0..advances-1 is the post-loop tail (the benchmark's
        one final frame write lands there); it is kept separately so it neither
        inflates an advance nor disappears.
        """
        self._check_name(counter)
        bucket = self._observed.setdefault(counter, {})
        if 0 <= advance_index < self._advances:
            bucket[advance_index] = bucket.get(advance_index, 0) + 1
        else:
            self._post_loop[counter] = self._post_loop.get(counter, 0) + 1

    def declare_suppressed(self, counter: str, *, producer_absent: bool, reason: str) -> None:
        self._check_name(counter)
        if not producer_absent:
            raise RuntimeError(
                f"benchmark activity class {counter!r} was declared suppressed "
                f"({reason}) but its producer is live; the artifact must not "
                "report zero operations for it"
            )
        self._suppressed[counter] = reason

    def provenance(self) -> dict[str, str]:
        unbacked = [
            name
            for name in _BENCHMARK_ACTIVITY_COUNTERS
            if name not in self._observed and name not in self._suppressed
        ]
        if unbacked:
            raise RuntimeError(
                "benchmark activity classes have neither instrumentation nor a "
                f"suppression declaration: {unbacked}; a zero for them would be "
                "unobserved, not measured"
            )
        record: dict[str, str] = {}
        for name in _BENCHMARK_ACTIVITY_COUNTERS:
            parts: list[str] = []
            if name in self._observed:
                tail = self._post_loop.get(name, 0)
                parts.append(
                    "observed at its call site"
                    + (
                        f"; {tail} post-loop operation(s) outside advances "
                        f"0..{self._advances - 1}"
                        if tail
                        else ""
                    )
                )
            if name in self._suppressed:
                parts.append(f"suppressed: {self._suppressed[name]}")
            record[name] = "; ".join(parts)
        return record

    def per_advance(self) -> list[dict[str, object]]:
        self.provenance()
        return [
            {
                "advance_index": index,
                **{
                    name: self._observed.get(name, {}).get(index, 0)
                    for name in _BENCHMARK_ACTIVITY_COUNTERS
                },
            }
            for index in range(self._advances)
        ]


def _write_benchmark_artifact(
    *,
    path: Path,
    run_uuid: str,
    timer: BenchmarkStepTimer,
    model: Any,
    global_grid: Any,
    compile_metadata: Mapping[str, object],
    distributed_context: DistributedContext | None,
    layout: ParallelLayout | None,
    activity: "_BenchmarkActivityLedger",
) -> None:
    timer.finalize()
    host_peak, cgroup_path = _host_memory_peak()
    device = torch.device(model.config.device)
    if device.type == "cuda":
        device_allocated = int(torch.cuda.max_memory_allocated(device))
        device_reserved = int(torch.cuda.max_memory_reserved(device))
    else:
        device_allocated = 0
        device_reserved = 0
    local = {
        "hostname": socket.gethostname(),
        "cgroup_path": cgroup_path,
        "host_peak": host_peak,
        "device_allocated": device_allocated,
        "device_reserved": device_reserved,
        "host_wall": timer.synchronized_host_wall_seconds,
        "samples": list(timer.per_step),
        "uses_cuda_events": timer.uses_cuda_events,
    }
    rank_records = [local]
    if distributed_context is not None and distributed_context.is_distributed:
        gathered: list[object] = [None] * distributed_context.world_size
        dist.all_gather_object(gathered, local)
        rank_records = [record for record in gathered if isinstance(record, dict)]
        if len(rank_records) != distributed_context.world_size:
            raise RuntimeError("benchmark rank telemetry gather was incomplete")
    if not _rank0(distributed_context):
        return

    sample_maps = [dict(record["samples"]) for record in rank_records]
    expected_indices = set(range(_BENCHMARK_ADVANCES))
    if any(set(values) != expected_indices for values in sample_maps):
        raise RuntimeError(
            "benchmark samples do not cover advance indices 0..99 on every rank"
        )
    samples = [
        (index, max(float(values[index]) for values in sample_maps))
        for index in range(_BENCHMARK_ADVANCES)
    ]
    host_wall = max(float(record["host_wall"]) for record in rank_records)
    node_peaks: dict[tuple[str, str], int] = {}
    for record in rank_records:
        key = (str(record["hostname"]), str(record["cgroup_path"]))
        node_peaks[key] = max(node_peaks.get(key, 0), int(record["host_peak"]))
    host_peak_total = sum(node_peaks.values())
    device_allocated_total = sum(
        int(record["device_allocated"]) for record in rank_records
    )
    device_reserved_total = sum(
        int(record["device_reserved"]) for record in rank_records
    )

    nx = int(global_grid.nx)
    ny = int(global_grid.ny)
    nz = int(global_grid.nz)
    world_size, npx, npy = _benchmark_layout(distributed_context, layout)
    device_kind = "G" + str(world_size) if device.type == "cuda" else "C"
    retained_indices = list(range(_BENCHMARK_WARMUP, _BENCHMARK_ADVANCES))
    retained_times = [
        seconds for index, seconds in samples if index >= _BENCHMARK_WARMUP
    ]
    accelerator_gates = {
        key: os.environ.get(key, "unset") for key in _BENCHMARK_ACCELERATOR_GATES
    }
    # The frozen tuple above is a legacy surface kept for
    # a pinned benchmark check; it neither resolves effective values
    # nor closes over registry growth, so an "all OFF" claim built on it cannot
    # exclude a key it does not name, and library-default-ON switches hide
    # behind "unset".  The map below enumerates the whole accelerator registry
    # through the runtime's own gate() resolver (single source with the
    # dispatch predicate) and records raw spelling, effective value, default
    # source, and observed ignition state per env key.  Additive field: the
    # frozen gate ignores it, so schema_version stays 1.0.0.
    from defgcm.accel import REGISTRY as _accel_registry, gate as _accel_gate

    accelerator_effective = {}
    for switch_key in sorted(_accel_registry):
        view = _accel_gate(switch_key)
        accelerator_effective[_accel_registry[switch_key].env_var] = {
            "switch": switch_key,
            "raw": view.raw_spelling,
            "effective": bool(view.effective_value),
            "source": view.default_source,
            "ignition": view.ignition_state,
        }
    # A key the frozen check pins but the registry no
    # longer serves keeps being echoed with whatever the environment spells,
    # so `DEFGCM_TRITON_VISC=1` reads as "enabled" in the artifact while switching
    # nothing.  The key cannot be dropped (the pinned check would report it
    # missing), so name the retired set instead of leaving the reader to
    # difference two maps.  Derived from the registry, never a hand list:
    # retiring a third kernel shows up here without an edit.  Additive, so
    # schema_version stays 1.0.0.
    live_env_vars = {spec.env_var for spec in _accel_registry.values()}
    accelerator_retired = sorted(
        set(_BENCHMARK_ACCELERATOR_GATES) - live_env_vars
    )
    compile_record = _benchmark_compile_record(compile_metadata)
    document = {
        "schema_version": "1.0.0",
        "benchmark_point": f"LL{nx}-{device_kind}",
        "protocol": {
            "total_advances": _BENCHMARK_ADVANCES,
            "warmup_boundary": _BENCHMARK_WARMUP,
            "retained_count": len(retained_indices),
            "median_source_indices": retained_indices,
        },
        "samples": [
            {
                "advance_index": index,
                "retained": index >= _BENCHMARK_WARMUP,
                "cuda_event_seconds": seconds,
            }
            for index, seconds in samples
        ],
        "timing": {
            "cuda_event_total_seconds": sum(seconds for _, seconds in samples),
            "synchronized_host_wall_seconds": host_wall,
            "median_cuda_event_seconds": statistics.median(retained_times),
            "primary_clock": (
                "cuda_event"
                if all(bool(record["uses_cuda_events"]) for record in rank_records)
                else "perf_counter_fallback"
            ),
        },
        "speed_record": {"run_uuid": run_uuid},
        "memory_record": {
            "run_uuid": run_uuid,
            "host_peak_bytes": host_peak_total,
            "host_peak_source": _classify_host_peak_source(node_peaks),
            "host_cgroup_path": ";".join(
                f"{hostname}:{cgroup}" for hostname, cgroup in sorted(node_peaks)
            ),
            "device_peak_allocated_bytes": device_allocated_total,
            "device_peak_reserved_bytes": device_reserved_total,
            "gross_peak_bytes": host_peak_total + device_reserved_total,
        },
        "grid": {"nx": nx, "ny": ny, "nz": nz, "cell_count": nx * ny * nz},
        "loop_activity": {
            "coverage": list(_BENCHMARK_ACTIVITY_COVERAGE),
            # Counts come from the ledger, never from dict.fromkeys(..., 0):
            # a zero must be one this run established, not one this writer
            # invented.  ``provenance`` names how each class got its
            # number so a reader can tell measurement from suppression.
            "per_advance": activity.per_advance(),
            "provenance": activity.provenance(),
        },
        "switch_inventory": {
            "precision": str(model.config.dtype).removeprefix("torch."),
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "compile": compile_record,
            "rank_layout": {"world_size": world_size, "npx": npx, "npy": npy},
            "accelerator_gates": accelerator_gates,
            "accelerator_effective": accelerator_effective,
            "accelerator_retired": accelerator_retired,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    print(f"[benchmark] timing artifact -> {path}", flush=True)


def _production_time_step_bundle(model: Any, dt: float) -> TimeStepBundle:
    """Assemble the 4-deltaT bundle the stepper must actually receive.

    Value priority: an explicit bundle (caller-supplied) > the per-equation
    values already resolved onto ``model.config`` > uniform expansion of the
    scalar clock ``dt``.

    Why this lives here (production assembly) and not in
    ``model.step`` / ``ModelRunner.run``: both coerce a **scalar** ``dt`` via
    ``coerce_time_step_bundle`` -> ``TimeStepBundle.all_equal(dt)``, which
    overwrites every per-equation value.  Handing them a scalar therefore
    silently discarded ``model.config.dt_mom`` — fatal for MITgcm
    "accelerated spin-up" configurations where ``deltaTtracer != deltaTmom``
    (``tutorial_global_oce_latlon``: deltaTmom=1800 vs deltaTtracer=86400,
    a factor 48 on the momentum update; step-1 |u|max was 48.6x MIT and the
    run hit NaN by step 7).  The coerce semantics in
    ``defgcm/steppers/base.py`` are deliberately left untouched.

    Synchronous cases (``dt_mom == dt_tracer == dt_freesurf == dt_clock ==
    dt``) reproduce ``TimeStepBundle.all_equal(dt)`` value-for-value, so this
    is a bitwise no-op for every case that was already running.
    """
    config = getattr(model, "config", None)

    def _resolved(name: str) -> float:
        value = getattr(config, name, None)
        return dt if value is None else float(value)

    return TimeStepBundle(
        dt=dt,
        dt_mom=_resolved("dt_mom"),
        dt_tracer=_resolved("dt_tracer"),
        dt_freesurf=_resolved("dt_freesurf"),
        dt_clock=_resolved("dt_clock"),
    )


def build_model(
    run_cfg: Any,
    run_dir: Path | str,
    *,
    steps: int,
    device: str | None = None,
    logger: logging.Logger | None = None,
) -> tuple[Any, Any, TimeStepBundle]:
    """Build the model, its initial state and the time-step bundle of a prepared run.

    ``run_dir`` holds the outputs of pipeline stages 1a and 1b for ``run_cfg``.
    This is the assembly stage 2 performs before stepping, for scripts that
    drive the model directly: ``ModelRunner().run(model, state, steps=steps, dt=bundle)``.
    ``device`` defaults to ``run_cfg.run.device``.
    """
    dt = float(run_cfg.time.dt)
    model, state, _ctx = _load_canonical_and_build_model(
        run_cfg,
        output_dir=Path(run_dir),
        mitgcm_run_dir=None,
        n_steps=int(steps),
        dt=dt,
        device=str(device or run_cfg.run.device),
        logger=logger or logging.getLogger("defgcm.build_model"),
    )
    return model, state, _production_time_step_bundle(model, dt)


def _case_kind(run_cfg: Any) -> str:
    return str(getattr(run_cfg.case, "kind", ""))


def _extras(run_cfg: Any) -> dict[str, Any]:
    return dict(getattr(run_cfg.case, "extras", {}) or {})


def _extra(run_cfg: Any, key: str, default: Any = None) -> Any:
    extras = _extras(run_cfg)
    for candidate in (key, key.replace("_", "-")):
        if candidate in extras and extras[candidate] is not None:
            return extras[candidate]
    return default


def _required_extra(run_cfg: Any, key: str) -> Any:
    value = _extra(run_cfg, key, None)
    if value is None:
        raise ValueError(f"case.extras.{key} is required for framework runtime")
    return value


def _extra_bool(run_cfg: Any, key: str, default: bool) -> bool:
    value = _extra(run_cfg, key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return bool(value)


def _resolve_device(run_cfg: Any) -> str:
    if run_cfg.run.device != "auto":
        return str(run_cfg.run.device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _rank0(distributed_context: DistributedContext | None) -> bool:
    return distributed_context is None or distributed_context.rank == 0


def _barrier(distributed_context: DistributedContext | None) -> None:
    if distributed_context is not None and distributed_context.is_distributed:
        dist.barrier()


def _run_cfg_with_device(run_cfg: Any, device: str) -> Any:
    return replace(run_cfg, run=replace(run_cfg.run, device=device))


#: Floor every distributed run carries whatever the tracer scheme is.  Not a
#: halo width: it is the widest narrow-tile early return left in the horizontal
#: core, which is the same exchange-count-splitting shape the stencil floor exists
#: for.  ``_compute_hFacZ_4neighbor`` returns before its two halo exchanges at
#: ``ny < 2`` (``dynamics/horizontal_diffusion.py:240``), and the u/v no-slip
#: side drag returns zeros -- skipping that same hFacZ call -- at ``ny < 2`` /
#: ``nx < 2`` (``:3113``, ``:3335``).
_BASE_MIN_TILE_EXTENT = 2


def _minimum_tile_extent(run_cfg: Any) -> tuple[int, str]:
    """Narrowest tile the operators THIS run selects can survive, and why.

    The builder feeds ``advection.tracer`` to ``HorizontalTracerAdvection``
    (``io/runner/builders/from_manifest.py:_tracer_flux_scheme``), whose face
    stencil reads ``halo`` cells on each side of a face and asks every tile for
    ``2*halo`` cells (``tracer_flux._require_stencil_width``); the floor reads
    that ``halo`` from the scheme's declaration.  A two-point scheme (``halo``
    1) posts no wide-halo exchange, so its floor is ``_BASE_MIN_TILE_EXTENT``.
    The returned minimum is a global-number verdict every rank computes
    identically, and it never drops below ``_BASE_MIN_TILE_EXTENT``.
    """
    scheme = TRACER_SCHEME_BY_NAME[_tracer_flux_scheme(run_cfg)]
    stencil = 2 * scheme.halo
    if stencil <= _BASE_MIN_TILE_EXTENT:
        return (
            _BASE_MIN_TILE_EXTENT,
            f"the {scheme.name} tracer scheme's horizontal core "
            f"(side-drag / hFacZ stencils)",
        )
    return stencil, f"the {stencil}-cell {scheme.name} tracer-advection stencil"


def _distributed_layout(
    run_cfg: Any,
    distributed_context: DistributedContext | None,
) -> ParallelLayout | None:
    if distributed_context is None or not distributed_context.is_distributed:
        return None
    nPx = int(os.environ.get("DEFGCM_NPX", run_cfg.mit_parallel.nPx))
    nPy = int(os.environ.get("DEFGCM_NPY", run_cfg.mit_parallel.nPy))
    # Refuse a decomposition finer than the horizontal stencils can survive.
    # Checked here, at the one place a real run commits to nPx/nPy, because the
    # failure it prevents is a silent deadlock rather than a wrong answer, and
    # because every rank derives the same verdict from the same global numbers.
    # The floor follows the scheme this run selected (G3_distributed-M-1); the
    # refusal itself stays unconditional and global-number based.
    minimum, reason = _minimum_tile_extent(run_cfg)
    require_minimum_tile_extent(
        global_nx=int(run_cfg.grid.nx),
        global_ny=int(run_cfg.grid.ny),
        nPx=nPx,
        nPy=nPy,
        minimum=minimum,
        reason=reason,
    )
    return layout_from_rank(
        global_nx=int(run_cfg.grid.nx),
        global_ny=int(run_cfg.grid.ny),
        nPx=nPx,
        nPy=nPy,
        rank=distributed_context.rank,
        local_rank=distributed_context.local_rank,
        world_size=distributed_context.world_size,
    )


def _target_grid(run_cfg: Any) -> Grid:
    return Grid(
        nx=int(run_cfg.grid.nx),
        ny=int(run_cfg.grid.ny),
        nz=int(run_cfg.grid.nz),
        center_lat=float(run_cfg.grid.center_lat),
        center_lon=float(run_cfg.grid.center_lon),
        resolution_deg=float(run_cfg.grid.resolution_deg),
        depth_m=float(run_cfg.grid.depth_m),
        vertical_layer_thicknesses=tuple(
            float(v) for v in run_cfg.grid.vertical_layer_thicknesses
        ),
        grid_id=(
            f"{run_cfg.run.experiment_name}_"
            f"{run_cfg.grid.nx}x{run_cfg.grid.ny}x{run_cfg.grid.nz}"
        ),
    )


def _model_input_dir(manifest_run_dir: Path) -> Path:
    return manifest_run_dir / "_pipeline_meta" / "model_input"


def _model_input_initial_state_path(manifest_run_dir: Path) -> Path:
    return _model_input_dir(manifest_run_dir) / "initial_state.zarr"


def _model_input_grid_path(manifest_run_dir: Path) -> Path:
    return _model_input_dir(manifest_run_dir) / "grid"


def _source_context(
    run_cfg: Any, target_grid: Grid, logger: logging.Logger
) -> SourceContext:
    def grid_resolver(grid_id: str) -> Grid:
        if grid_id in {target_grid.grid_id, "model_grid"}:
            return target_grid
        raise ValueError(f"unknown grid_id {grid_id!r}")

    return SourceContext(
        cache_root=Path(run_cfg.cache.root),
        target_grid_resolver=grid_resolver,
        logger=logger,
    )


def _stage1b_consumer_path(manifest_run_dir: Path, consumer_name: str) -> Path:
    output_names = {
        "torchocean_ic_zarr": "torchocean_ic.zarr",
        "torchocean_obc_zarr": "torchocean_obc.zarr",
        "torchocean_forcing_zarr": "torchocean_forcing.zarr",
    }
    filename = output_names.get(consumer_name, f"{consumer_name}.zarr")
    return manifest_run_dir / "_pipeline_meta" / "torch_inputs" / filename


def _resolve_source_extras(
    run_cfg: Any,
    manifest_run_dir: Path,
    prefix: str,
) -> dict[str, Any]:
    raw = _extra(run_cfg, f"{prefix}_source_extras", {}) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"case.extras.{prefix}_source_extras must be a mapping")
    resolved = dict(raw)
    consumer = resolved.pop("stage1b_consumer", None)
    if consumer is not None:
        resolved.setdefault(
            "path",
            _stage1b_consumer_path(manifest_run_dir, str(consumer)).resolve(),
        )
    for key in ("path", "bundle_path"):
        if key in resolved:
            value = Path(str(resolved[key]))
            resolved[key] = value if value.is_absolute() else manifest_run_dir / value
    if "bundle_path" in resolved and "path" not in resolved:
        resolved["path"] = resolved.pop("bundle_path")
    return resolved


def _tensor_from_bundle(
    bundle: Any,
    name: str,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    values = np.asarray(bundle.data[name].values)
    if name != "bathymetry" and values.ndim > 0 and values.shape[0] == 1:
        values = values[0]
    return torch.as_tensor(values, dtype=dtype, device=device)


def _initial_frame(
    values: Any,
    *,
    expected_ndim: int,
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == expected_ndim + 1:
        arr = arr[0]
    if arr.ndim != expected_ndim:
        raise ValueError(
            f"expected {expected_ndim}D initial field, got shape {arr.shape}"
        )
    return arr


def _canonical_ic_tensors(
    manifest_run_dir: Path,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    path = _model_input_initial_state_path(manifest_run_dir)
    if not path.exists():
        return {}
    from defgcm.io.runner.model_input import read_initial_state_dataset

    ds = read_initial_state_dataset(path)
    try:
        aliases = {
            "temperature": "temp",
            "salinity": "salt",
            "eta": "eta",
            "u": "u",
            "v": "v",
        }
        tensors: dict[str, torch.Tensor] = {}
        for target_name, source_name in aliases.items():
            if source_name in ds:
                tensors[target_name] = torch.as_tensor(
                    np.asarray(ds[source_name].values),
                    dtype=dtype,
                    device=device,
                )
        grid_tensors_path = _model_input_grid_path(manifest_run_dir) / "tensors.npz"
        if grid_tensors_path.exists() and "bathymetry" not in tensors:
            with np.load(grid_tensors_path) as grid_tensors:
                if "depth_center" in grid_tensors.files:
                    # float64 on purpose: the bathymetry is a geometry input,
                    # not a prognostic field.  The grid constructor derives the
                    # land mask from it by an untoleranced comparison, so
                    # narrowing it to the run dtype here moves the mask itself
                    # (grid.py ``_GEOMETRY_DTYPE_NOTE``).
                    tensors["bathymetry"] = torch.as_tensor(
                        np.asarray(grid_tensors["depth_center"]),
                        dtype=torch.float64,
                        device=device,
                    )
        return tensors
    finally:
        ds.close()


def _bundle_with_bathymetry(bundle: Any, bathymetry: torch.Tensor) -> Any:
    if "bathymetry" in bundle.data:
        return bundle
    bathy_np = bathymetry.detach().cpu().numpy().astype(np.float64)
    data = bundle.data.assign(
        bathymetry=(("latitude", "longitude"), bathy_np),
    )
    variables = tuple(bundle.meta.variables) + (
        VariableSpec(
            name="bathymetry",
            cf_standard_name="sea_floor_depth_below_geoid",
            units="m",
            dtype="float64",
            dims=("latitude", "longitude"),
        ),
    )
    meta_no_prov = dataclasses.replace(
        bundle.meta,
        variables=variables,
        provenance=(),
    )
    prov = ProvenanceStep(
        kind="operator",
        name="RuntimeBathymetryFromDepth",
        params={"depth_m": float(np.nanmax(bathy_np)) if bathy_np.size else 0.0},
        timestamp=datetime.utcnow(),
        bundle_hash="00000000",
    )
    meta = dataclasses.replace(
        meta_no_prov,
        provenance=tuple(bundle.meta.provenance) + (prov,),
    )
    return DataBundle(data=data, meta=meta)


def _fetch_ic_bundle(
    run_cfg: Any,
    manifest_run_dir: Path,
    target_grid: Grid,
    *,
    logger: logging.Logger,
) -> Any:
    # There is no stage1b to_ic pipeline, so the stage1b_consumer:
    # torchocean_ic_zarr product is not generated. The canonical IC zarr is the single
    # source; every real downstream consumer (_ic_tensor_map fast-path,
    # reference_profile_from_tensors, _coord_from_bundle gated by mitgcm_run_dir) reads it
    # before the ic_bundle fallback. Return None when the canonical file exists, so the
    # fallback path fails loudly instead of hanging on a silent fetch.
    canonical_path = _model_input_initial_state_path(manifest_run_dir)
    if canonical_path.exists():
        logger.info(
            "canonical IC present at %s — skipping legacy zarr_bundle fetch",
            canonical_path,
        )
        return None
    source_name = str(_required_extra(run_cfg, "ic_source"))
    source = DATA_SOURCES.instantiate(
        source_name,
        _source_context(run_cfg, target_grid, logger),
        **_source_kwargs(run_cfg, manifest_run_dir, "ic"),
    )
    bundle = source.fetch(
        DataQuery(
            location_kind=LocationKind.GRIDDED_3D_OCEAN,
            variables=("temperature", "salinity", "eta", "u", "v", "bathymetry"),
            time_range=None,
            target_grid=target_grid,
        )
    )
    if "bathymetry" not in bundle.data:
        bathy = torch.full(
            (int(run_cfg.grid.ny), int(run_cfg.grid.nx)),
            float(run_cfg.grid.depth_m),
            dtype=torch.float64,
        )
        bundle = _bundle_with_bathymetry(bundle, bathy)
    return bundle


def _source_kwargs(
    run_cfg: Any,
    manifest_run_dir: Path,
    prefix: str,
) -> dict[str, Any]:
    source_name = _extra(run_cfg, f"{prefix}_source", None)
    specific = _resolve_source_extras(run_cfg, manifest_run_dir, prefix)
    if source_name is not None:
        merged = _source_config_map(run_cfg, str(source_name))
        merged.update(specific)
        if merged:
            return merged
    if specific:
        return specific
    extras = _extras(run_cfg)
    excluded = {
        "framework",
        "ic_source",
        "ic_source_extras",
        "obcs_source",
        "obcs_source_extras",
        "forcing_source",
        "forcing_source_extras",
        "west_east_boundary",
        "south_north_boundary",
        "lx",
        "ly",
        "case_type",
        "case_name",
        "description",
        "reference_path",
        "hfac_min",
        "hfac_min_dz",
        "obcs_sides",
        "obcs_along",
        "external_pnh_dir",
        "external_pnh_mode",
    }
    return {k: v for k, v in extras.items() if k not in excluded}


def _resolve_external_pnh_dir(run_cfg: Any, mitgcm_run_dir: Path | None) -> Path | None:
    mode = str(_extra(run_cfg, "external_pnh_mode", "off"))
    if mode == "off":
        return None
    if mode not in {"auto", "on"}:
        raise ValueError(
            "case.extras.external_pnh_mode must be one of 'auto', 'on', or 'off'"
        )

    candidates: list[Path] = []
    configured = _extra(run_cfg, "external_pnh_dir", None)
    if configured is not None and str(configured) != "":
        candidates.append(Path(str(configured)))
    if mitgcm_run_dir is not None:
        candidates.append(mitgcm_run_dir)

    for candidate in candidates:
        if candidate.exists() and sorted(candidate.glob("PNH.*.001.001.meta")):
            return candidate

    if mode == "on":
        raise FileNotFoundError(
            "--external-pnh-mode=on but no PNH.*.001.001.meta dumps found in: "
            + ", ".join(str(path) for path in candidates)
        )
    return None


def _external_pnh_source(
    run_cfg: Any,
    mitgcm_run_dir: Path | None,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
) -> Any | None:
    pnh_dir = _resolve_external_pnh_dir(run_cfg, mitgcm_run_dir)
    if pnh_dir is None:
        return None

    from defgcm.io.mitgcm_pnh import MITgcmPNHSource

    return MITgcmPNHSource.from_run_dir(
        pnh_dir,
        dt=float(run_cfg.time.dt),
        base_time=float(_extra(run_cfg, "base_time", 0.0)),
        extrapolation="clamp",
        device=device,
        dtype=dtype,
    )


def _fetch_obcs_source(
    run_cfg: Any,
    manifest_run_dir: Path,
    target_grid: Grid,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
    n_steps: int,
    dt: float,
    logger: logging.Logger,
) -> BoundaryBundleSource:
    source_name = str(_required_extra(run_cfg, "obcs_source"))
    source = DATA_SOURCES.instantiate(
        source_name,
        _source_context(run_cfg, target_grid, logger),
        **_source_kwargs(run_cfg, manifest_run_dir, "obcs"),
    )
    side_names = tuple(str(v) for v in _extra(run_cfg, "obcs_sides", ("W", "E")))
    start_date = datetime.fromisoformat(str(run_cfg.run.start_date))
    end_date = start_date + timedelta(seconds=float(n_steps) * float(dt))
    bundle = source.fetch(
        DataQuery(
            location_kind=LocationKind.BOUNDARY_TIMESERIES,
            variables=("temperature", "salinity", "eta", "u", "v", "w"),
            time_range=(start_date, end_date),
            target_grid=target_grid,
            extras={
                "dt_seconds": float(dt),
                "sides": side_names,
                "along": int(_extra(run_cfg, "obcs_along", run_cfg.grid.nx)),
            },
        )
    )
    if "side" in bundle.data.coords:
        available = {str(v) for v in bundle.data.coords["side"].values}
        side_names = tuple(side for side in side_names if side in available)
    obcs_source = _boundary_source_from_bundle(
        bundle,
        side_names=side_names,
        device=device,
        dtype=dtype,
        n_steps=n_steps,
        dt=dt,
    )
    if "eta" not in obcs_source.fields and "eta" in bundle.data:
        fields = dict(obcs_source.fields)
        fields["eta"] = torch.as_tensor(
            np.asarray(bundle.data["eta"].values),
            dtype=dtype,
            device=device,
        )
        obcs_source = BoundaryBundleSource(
            times_seconds=obcs_source.times_seconds,
            side_names=obcs_source.side_names,
            fields=fields,
            valid_along=obcs_source.valid_along,
            forcing_period_seconds=obcs_source.forcing_period_seconds,
            forcing_cycle_seconds=obcs_source.forcing_cycle_seconds,
        )
    return obcs_source


def _source_config(
    run_cfg: Any, source_name: str, field: str, default: Any = None
) -> Any:
    for source in getattr(run_cfg, "sources", ()):
        if source.name == source_name or source.type == source_name:
            return dict(source.config).get(field, default)
    return default


def _source_config_map(run_cfg: Any, source_name: str) -> dict[str, Any]:
    for source in getattr(run_cfg, "sources", ()):
        if source.name == source_name or source.type == source_name:
            return dict(source.config)
    return {}


def _era5_wind_request(
    run_cfg: Any,
    manifest_run_dir: Path,
    *,
    n_steps: int,
    dt: float,
) -> tuple[Any, datetime, int] | None:
    """``(era5_dir, start_date, window_seconds)`` when ``_fetch_wind_source``
    reads the yearly ERA5 wind archive, otherwise ``None``."""
    source_name = _extra(run_cfg, "forcing_source", None)
    if source_name is None:
        return None
    source_kwargs = _source_kwargs(run_cfg, manifest_run_dir, "forcing")
    time_mode = str(source_kwargs.pop("time_mode", "linear"))
    source_type = _source_config(run_cfg, str(source_name), "type", str(source_name))
    if source_type == "era5_netcdf" and time_mode == "linear":
        start_date = datetime.fromisoformat(str(run_cfg.run.start_date))
        # Wind-stress time-interpolation alignment.  MIT EXF
        # re-interpolates the (here 6-hourly) stress records at EVERY model
        # step, so LOCOMgcm must load a window holding >=2 records that bracket the
        # whole run.  Sizing the window to only the active run length starves
        # the source to a single record whenever the run is shorter than the
        # forcing cadence (e.g. short alignment/comparison runs).  Then
        # ``ERA5WindSource.sample`` clamps every step to that one record -> TO
        # holds wind stress CONSTANT while MIT drifts linearly, injecting a
        # spurious surface-trapped ``du ~ -d(tau)/dt * t^2`` (corr -0.98 with
        # the stress time-derivative; see 
        # wind_stress_interp_rootcause.md).  Load at least the configured case
        # horizon so TO interpolates the same records MIT EXF does.
        #
        # Work in INTEGER SECONDS, never float days: ``timedelta(days=float)``
        # is lossy and would drop/add an ERA5 record exactly at a cadence
        # boundary (same class as the ``time.days`` step-count trap that
        # motivated the integer ``time.seconds`` schema field).
        active_seconds = int(round(float(n_steps) * float(dt)))
        cfg_seconds = getattr(run_cfg.time, "seconds", None)
        configured_seconds = (
            int(round(float(cfg_seconds)))
            if cfg_seconds
            else int(round(float(run_cfg.time.days) * 86400.0))
        )
        window_seconds = max(active_seconds, configured_seconds)
        raw_path = source_kwargs.get(
            "path",
            _source_config(run_cfg, str(source_name), "path"),
        )
        if raw_path is not None:
            return raw_path, start_date, window_seconds
    return None


def _fetch_wind_source(
    run_cfg: Any,
    manifest_run_dir: Path,
    target_grid: Grid,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
    n_steps: int,
    dt: float,
    logger: logging.Logger,
) -> ERA5WindSource | ERA5BulkSource | None:
    source_name = _extra(run_cfg, "forcing_source", None)
    if source_name is None:
        return None
    request = _era5_wind_request(run_cfg, manifest_run_dir, n_steps=n_steps, dt=dt)
    if request is not None:
        raw_path, start_date, window_seconds = request
        bulk_entries = [entry for entry in run_cfg.model.dynamics if entry.name == "era5_bulk_forcing"]
        if bulk_entries and dict(bulk_entries[0].kwargs).get("use_atm_wind", True):
            # Preserve the velocity source for coupled EXF. No legacy
            # wind-to-stress preprocessing participates in this path.
            return _build_bulk_source(run_cfg, device=device, dtype=dtype,
                                      window_seconds=int(n_steps * dt))
        return load_era5_wind_source_by_time(
            raw_path,
            start_date=start_date,
            window_seconds=window_seconds,
            center_lat=float(target_grid.center_lat),
            center_lon=float(target_grid.center_lon),
            nx=int(target_grid.nx),
            ny=int(target_grid.ny),
            resolution_deg=float(target_grid.resolution_deg),
            cyclic=False,
        ).to_runtime(device, dtype)
    source_kwargs = _source_kwargs(run_cfg, manifest_run_dir, "forcing")
    time_mode = str(source_kwargs.pop("time_mode", "linear"))
    source = DATA_SOURCES.instantiate(
        str(source_name),
        _source_context(run_cfg, target_grid, logger),
        **source_kwargs,
    )
    bundle = source.fetch(
        DataQuery(
            location_kind=LocationKind.GRIDDED_2D_SURFACE,
            variables=("tau_x", "tau_y", "wind_u", "wind_v"),
            time_range=None,
            target_grid=target_grid,
        )
    )
    return _wind_source_from_bundle(
        bundle,
        device=device,
        dtype=dtype,
        time_mode=time_mode,
        end_time_seconds=float(n_steps) * float(dt),
    )


def _uses_surface_flux_source(run_cfg: Any) -> bool:
    """Whether ``_fetch_surface_flux_source`` reads the forcing source."""
    return any(
        entry.name == "wpflux_surface_forcing" for entry in run_cfg.model.dynamics
    ) and _extra(run_cfg, "forcing_source", None) is not None


def _files_under(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(Path(root, name) for root, _, names in os.walk(path, followlinks=True)
                  for name in names)


def external_input_files(
    run_cfg: Any,
    manifest_run_dir: Path,
    *,
    mitgcm_run_dir: Path | None,
    n_steps: int,
    dt: float,
) -> list[Path]:
    """Files that ``_load_canonical_and_build_model`` reads besides ``manifest_run_dir``.

    The yearly ERA5 wind files of ``_fetch_wind_source``, every file under the
    OBCS source path and, without a canonical initial state, under the IC
    source path (these paths usually lie in the prepared run; a caller may
    drop duplicates), and the OBzonalW overlay.  A case that activates a
    reader whose files are not listed here (another forcing reader, surface
    fluxes, ERA5 bulk forcing, external PNH dumps, a data source without a
    path) raises NotImplementedError, so a caller binding inputs cannot omit
    them silently.
    """
    from defgcm.io.era5 import era5_wind_paths

    files: list[Path] = []
    unlisted: list[str] = []
    request = _era5_wind_request(run_cfg, manifest_run_dir, n_steps=n_steps, dt=dt)
    if request is not None:
        raw_path, start_date, window_seconds = request
        for pair in zip(*era5_wind_paths(raw_path, start_date=start_date,
                                         window_seconds=window_seconds)):
            files.extend(pair)
    elif _extra(run_cfg, "forcing_source", None) is not None:
        unlisted.append("forcing source other than the linear ERA5 wind archive")
    prefixes = ["obcs"]
    if not _model_input_initial_state_path(manifest_run_dir).exists():
        prefixes.append("ic")
    for prefix in prefixes:
        path = _source_kwargs(run_cfg, manifest_run_dir, prefix).get("path")
        if path is None:
            unlisted.append(f"{prefix} source without a path")
        else:
            files.extend(_files_under(Path(str(path))))
    if _uses_surface_flux_source(run_cfg):
        unlisted.append("surface flux source")
    if _uses_era5_bulk_forcing(run_cfg):
        unlisted.append("ERA5 bulk forcing")
    if _resolve_external_pnh_dir(run_cfg, mitgcm_run_dir) is not None:
        unlisted.append("external PNH dumps")
    overlay = _west_w_overlay_path(run_cfg)
    if overlay is not None:
        files.append(overlay)
    if unlisted:
        raise NotImplementedError("files read by " + ", ".join(unlisted) + " are not listed")
    return files


def _fetch_surface_flux_source(
    run_cfg: Any,
    manifest_run_dir: Path,
    target_grid: Grid,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
    n_steps: int,
    dt: float,
    logger: logging.Logger,
) -> Any | None:
    if not _uses_surface_flux_source(run_cfg):
        return None
    source_name = _extra(run_cfg, "forcing_source", None)
    source_kwargs = _source_kwargs(run_cfg, manifest_run_dir, "forcing")
    time_mode = str(source_kwargs.pop("time_mode", "linear"))
    source = DATA_SOURCES.instantiate(
        str(source_name),
        _source_context(run_cfg, target_grid, logger),
        **source_kwargs,
    )
    bundle = source.fetch(
        DataQuery(
            location_kind=LocationKind.GRIDDED_2D_SURFACE,
            variables=("heat_flux", "freshwater_flux"),
            time_range=None,
            target_grid=target_grid,
        )
    )
    return _surface_flux_source_from_bundle(
        bundle,
        device=device,
        dtype=dtype,
        time_mode=time_mode,
        end_time_seconds=float(n_steps) * float(dt),
    )


def _fetch_surface_pressure_load(
    run_cfg: Any,
    manifest_run_dir: Path,
    target_grid: Grid,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
    logger: logging.Logger,
) -> torch.Tensor | None:
    """MITgcm pLoad from ``case.extras.forcing_source`` for a declaring pressure gradient."""
    if not _uses_surface_pressure_load(run_cfg):
        return None
    source_kwargs = _source_kwargs(run_cfg, manifest_run_dir, "forcing")
    source_kwargs.pop("time_mode", None)
    source = DATA_SOURCES.instantiate(
        str(_extra(run_cfg, "forcing_source", None)),
        _source_context(run_cfg, target_grid, logger),
        **source_kwargs,
    )
    bundle = source.fetch(
        DataQuery(
            location_kind=LocationKind.GRIDDED_2D_SURFACE,
            variables=("pload",),
            time_range=None,
            target_grid=target_grid,
        )
    )
    return _surface_pressure_load_from_bundle(bundle, device=device, dtype=dtype)


def _spherical_yg_origin(run_cfg: Any) -> float:
    """South-west corner latitude of a spherical-polar case [deg]."""
    return float(
        _extra(
            run_cfg,
            "yg_origin",
            float(run_cfg.grid.center_lat)
            - (int(run_cfg.grid.ny) / 2.0) * float(run_cfg.grid.resolution_deg),
        )
    )


def _legacy_cartesian_spacing(run_cfg: Any, manifest_run_dir: Path | None) -> dict:
    """Use the prepared mesh axes, or reconstruct them through their declared pipeline."""
    if getattr(run_cfg.grid, "kind", "latlon") == "cubed_sphere":
        return {}  # Cube face geometry comes from its own grid files.
    axes = ("del_x", "del_y")
    if manifest_run_dir is not None:
        from defgcm.io.runner.stages._common import stage_output_dir_for
        import xarray as xr

        stage = stage_output_dir_for("stage1a_prepare_common", manifest_run_dir, run_cfg)
        artifact = stage / "prepared_mesh.zarr"
        if artifact.is_dir():
            with xr.open_zarr(artifact, consolidated=False) as dataset:
                spacing = {axis: dataset[axis].values for axis in axes if axis in dataset}
            if spacing:
                return spacing

    pipelines = getattr(run_cfg, "pipelines", {}).get("stage1a_prepare_common", ())
    mesh = next((pipeline for pipeline in pipelines if pipeline.name == "mesh"), None)
    if mesh is None:
        return {}
    spec = next((source for source in run_cfg.sources if source.name == mesh.primary_source), None)
    if spec is None or spec.type != "mit_input_binary_mesh" or not any(
        key in spec.config for key in ("del_x", "del_y", "del_x_file", "del_y_file")
    ):
        return {}
    from defgcm.io.runner.model_input import _run_stage1a_pipeline_bundle

    bundle = _run_stage1a_pipeline_bundle(run_cfg, "mesh", target_grid=_target_grid(run_cfg),
                                          logger=logging.getLogger(__name__))
    return {axis: bundle.data[axis].values for axis in axes if axis in bundle.data}


def _build_grid(
    run_cfg: Any,
    *,
    manifest_run_dir: Path | None = None,
    bathymetry: torch.Tensor,
    device: str | torch.device,
    dtype: torch.dtype,
) -> Any:
    grid_mode = str(_extra(run_cfg, "grid_mode", "cartesian"))
    if manifest_run_dir is not None:
        canonical = _build_grid_from_model_input(
            manifest_run_dir,
            run_cfg=run_cfg,
            device=device,
            dtype=dtype,
        )
        if canonical is not None:
            return canonical

    hfac_min = float(_extra(run_cfg, "hfac_min", 0.2))
    hfac_min_dz = float(_extra(run_cfg, "hfac_min_dz", 0.0))
    if grid_mode != "spherical_polar":
        return build_grid_from_run_cfg(
            run_cfg,
            bathymetry=bathymetry,
            **_legacy_cartesian_spacing(run_cfg, manifest_run_dir),
            device=device,
            dtype=dtype,
            hfac_min=hfac_min,
            hfac_min_dz=hfac_min_dz,
        )

    from defgcm.geometry import BoundaryTopology, BoundaryType, StructuredCGrid

    boundary_cfg = getattr(run_cfg, "boundary", None)
    fallback = str(getattr(boundary_cfg, "type", "periodic"))
    west_east = str(_extra(run_cfg, "west_east_boundary", fallback))
    south_north = str(_extra(run_cfg, "south_north_boundary", fallback))
    return StructuredCGrid.spherical_polar(
        nx=int(run_cfg.grid.nx),
        ny=int(run_cfg.grid.ny),
        nz=int(run_cfg.grid.nz),
        ygOrigin_deg=_spherical_yg_origin(run_cfg),
        delY_deg=float(run_cfg.grid.resolution_deg),
        delX_deg=float(run_cfg.grid.resolution_deg),
        depth=float(run_cfg.grid.depth_m),
        # float64 on purpose: this fallback (grid artifact missing) feeds the
        # same untoleranced wet/dry closure test the canonical builder does, so
        # narrowing here would make the topology depend on the run precision
        # (grid.py ``_GEOMETRY_DTYPE_NOTE``).  ``spherical_polar`` narrows the
        # derived metrics to ``dtype`` once the masks are built.
        bathymetry=torch.as_tensor(bathymetry, dtype=torch.float64, device=device),
        vertical_layer_thicknesses=tuple(
            float(v) for v in run_cfg.grid.vertical_layer_thicknesses
        ),
        vertical_center_spacing=tuple(run_cfg.grid.vertical_center_spacing),
        r_sphere=float(_extra(run_cfg, "r_sphere", 6_370_000.0)),
        boundary_topology=BoundaryTopology(
            west_east=BoundaryType(west_east),
            south_north=BoundaryType(south_north),
        ),
        obcs_fix_topo=bool(getattr(boundary_cfg, "obcs_fix_topo", False)),
        hfac_min=hfac_min,
        hfac_min_dz=hfac_min_dz,
        device=str(device),
        dtype=dtype,
    )


def _build_grid_from_model_input(
    manifest_run_dir: Path,
    *,
    run_cfg: Any,
    device: str | torch.device,
    dtype: torch.dtype,
) -> Any | None:
    """The grid stage 1a built, from its artifact (model_input.write_grid_artifacts).

    A run takes its grid from the prepared model input, not from the yaml, so the
    preparation's bathymetry, partial cells and masks are the ones it steps on.  The
    artifact stores tensors and a few scalars; the grid fields it does not store come
    from the run configuration that prepared it: the explicit cell-centre spacing
    (grid.vertical_center_spacing, MITgcm drC = delRc, ini_vertical_grid.F:103-107)
    and, on a spherical grid, the latitude rows StructuredCGrid.spherical_polar sets.
    The stored centre and interface depths are then required to equal the rebuilt
    grid's, so a run configuration that does not match the artifact fails here.
    """
    grid_dir = _model_input_grid_path(manifest_run_dir)
    scalars_path = grid_dir / "scalars.yaml"
    tensors_path = grid_dir / "tensors.npz"
    if not (scalars_path.exists() and tensors_path.exists()):
        return None

    from defgcm.geometry import (
        BoundaryTopology,
        BoundaryType,
        MaskSet,
        MetricSet,
        StructuredCGrid,
    )
    from defgcm.geometry.vertical_coordinates import ZLevelCoordinate

    scalars = yaml.safe_load(scalars_path.read_text()) or {}
    arrays = np.load(tensors_path)

    def tensor(name: str, *, bool_dtype: bool = False) -> torch.Tensor:
        arr = arrays[name]
        target_dtype = torch.bool if bool_dtype else dtype
        # Rebuild products from the original spacing, before run-dtype rounding.
        if name in ("del_x", "del_y"):
            target_dtype = torch.float64
        return torch.as_tensor(arr, dtype=target_dtype, device=device)

    topology_raw = dict(scalars.get("boundary_topology", {}) or {})
    boundary_topology = BoundaryTopology(
        west_east=BoundaryType(str(topology_raw.get("west_east", "periodic"))),
        south_north=BoundaryType(str(topology_raw.get("south_north", "periodic"))),
    )
    vector_names = (
        "del_x",
        "del_y",
        "dxF",
        "dyF",
        "dxG",
        "dyG",
        "dxC",
        "dyC",
        "dxV",
        "dyU",
        "rA",
        "rAw",
        "rAs",
        "rAz",
    )
    spacing_mode = scalars.get("spacing_mode")
    present_vector_names = [name for name in vector_names if name in arrays.files]
    if spacing_mode is None and present_vector_names:
        raise ValueError(
            "grid artifact contains vector tensors without spacing_mode=vector"
        )
    if spacing_mode not in (None, "vector"):
        raise ValueError(f"unsupported grid artifact spacing_mode={spacing_mode!r}")

    if spacing_mode == "vector":
        missing = [name for name in vector_names if name not in arrays.files]
        if missing:
            raise ValueError(
                "vector grid artifact missing required tensors: " + ", ".join(missing)
            )
        # The spacing products are rebuilt from float64 spacing and narrowed
        # once.  An artifact written before spacing was kept in float64 has
        # lost that input; its products cannot be rebuilt.
        narrowed = [name for name in ("del_x", "del_y") if arrays[name].dtype != np.float64]
        if narrowed:
            raise ValueError(
                "vector grid artifact stores " + ", ".join(narrowed)
                + " below float64; prepare the model input again"
            )
        metrics = MetricSet.with_bathymetry(
            nx=int(scalars["nx"]),
            ny=int(scalars["ny"]),
            del_x=tensor("del_x"),
            del_y=tensor("del_y"),
            bathymetry=tensor("depth_center"),
            device=str(device),
            dtype=dtype,
        )
        inconsistent = [
            name
            for name in vector_names
            if not torch.equal(getattr(metrics, name), tensor(name))
        ]
        if inconsistent:
            raise ValueError(
                "vector grid artifact tensors disagree with del_x/del_y: "
                + ", ".join(inconsistent)
            )
    else:
        metrics = MetricSet(
            dx_scalar=float(scalars["dx"]),
            dy_scalar=float(scalars["dy"]),
            dx_center=tensor("dx_center"),
            dy_center=tensor("dy_center"),
            cell_area=tensor("cell_area"),
            cell_volume=tensor("cell_volume"),
            depth_center=tensor("depth_center"),
            face_area_u=tensor("face_area_u"),
            face_area_v=tensor("face_area_v"),
            dxG_row=tensor("dxG_row"),
            dxC_row=tensor("dxC_row"),
            dyG_scalar_1d=float(scalars["dy"]),
            dxG_row_ext=tensor("dxG_row_ext"),
            rAz_row=tensor("rAz_row"),
        )
    masks = MaskSet(
        center=tensor("mask_center", bool_dtype=True),
        u=tensor("mask_u", bool_dtype=True),
        v=tensor("mask_v", bool_dtype=True),
        hFacC=tensor("hFacC") if "hFacC" in arrays.files else None,
        hFacW=tensor("hFacW") if "hFacW" in arrays.files else None,
        hFacS=tensor("hFacS") if "hFacS" in arrays.files else None,
        maskInC=tensor("maskInC") if "maskInC" in arrays.files else None,
        maskInW=tensor("maskInW") if "maskInW" in arrays.files else None,
        maskInS=tensor("maskInS") if "maskInS" in arrays.files else None,
    )
    if str(_extra(run_cfg, "grid_mode", "cartesian")) == "spherical_polar":
        # KPP's Ekman depth reads the latitudes (prepare_kpp_static).
        from defgcm.geometry.metrics import (
            spherical_centre_latitudes,
            spherical_corner_latitudes,
        )

        latitude_rows = (
            _spherical_yg_origin(run_cfg),
            float(run_cfg.grid.resolution_deg),
            int(run_cfg.grid.ny),
        )
        metrics = replace(
            metrics,
            yC_row=spherical_centre_latitudes(*latitude_rows),
            yG_row=spherical_corner_latitudes(*latitude_rows),
        )
    vertical_coordinate = ZLevelCoordinate(
        layer_thickness=tensor("layer_thickness"),
        center_spacing_given=tuple(float(v) for v in run_cfg.grid.vertical_center_spacing),
    )
    grid = StructuredCGrid(
        nx=int(scalars["nx"]),
        ny=int(scalars["ny"]),
        metrics=metrics,
        vertical_coordinate=vertical_coordinate,
        masks=masks,
        boundary_topology=boundary_topology,
    )
    # In the artifact's own dtype: the depths stage 1a wrote come from these layers.
    written = ZLevelCoordinate(
        layer_thickness=torch.as_tensor(arrays["layer_thickness"]),
        center_spacing_given=vertical_coordinate.center_spacing_given,
    )
    stale = [name for name in ("center_depth", "interface_depth")
             if name in arrays.files
             and not torch.equal(getattr(written, name), torch.as_tensor(arrays[name]))]
    if stale:
        raise ValueError(
            "grid artifact " + ", ".join(stale) + " disagree with the run configuration's "
            "vertical grid (grid.vertical_center_spacing); prepare the model input again"
        )
    return grid


def _ic_tensor_map(
    ic_bundle: Any,
    *,
    manifest_run_dir: Path | None = None,
    device: str | torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    if manifest_run_dir is not None:
        canonical = _canonical_ic_tensors(
            manifest_run_dir,
            device=device,
            dtype=dtype,
        )
        if canonical:
            return canonical
    if ic_bundle is None:
        raise RuntimeError(
            "canonical IC missing and ic_bundle is None — "
            "stage1a must produce _pipeline_meta/model_input/initial_state.zarr "
            "for production paths, or the case must provide a non-canonical ic_source."
        )
    # ``bathymetry`` is the odd one out: it is a geometry input consumed only by
    # the grid constructor, which derives the land mask from it by an
    # untoleranced comparison.  Narrowing it to the run dtype here moves the
    # mask itself (grid.py ``_GEOMETRY_DTYPE_NOTE``), so it stays float64 while
    # the prognostic fields carry the run dtype.
    tensors = {
        name: torch.as_tensor(
            _initial_frame(
                ic_bundle.data[name].values,
                expected_ndim=2 if name in {"eta", "bathymetry"} else 3,
            ),
            dtype=torch.float64 if name == "bathymetry" else dtype,
            device=device,
        )
        for name in ("temperature", "salinity", "eta", "u", "v", "bathymetry")
        if name in ic_bundle.data
    }
    # No zero-pad fallback. This fallback is dead in production
    # (canonical fast-path at line 809-816 always returns first), but keep the
    # safety: if caller ever lands here, they must do proper face reconstruction
    # via `center_to_u_centered` / `center_to_v_centered` after grid is built.
    return tensors


def _reference_profile(run_cfg: Any, ic_bundle: Any, name: str) -> torch.Tensor:
    extras = _extras(run_cfg)
    aliases = [name, name.replace("_", "-")]
    if name == "reference_temperature":
        aliases.extend(("t_ref", "t-ref"))
    elif name == "reference_salinity":
        aliases.extend(("s_ref", "s-ref"))
    for key in aliases:
        if key in extras and extras[key] is not None:
            raw = extras[key]
            if np.isscalar(raw):
                return torch.full(
                    (int(run_cfg.grid.nz),),
                    float(raw),
                    dtype=torch.float64,
                )
            values = np.asarray(raw, dtype=np.float64)
            if values.shape == (int(run_cfg.grid.nz),):
                return torch.as_tensor(values, dtype=torch.float64)
    profile = _compute_reference_profile_from_ic(ic_bundle, name)
    if profile is not None:
        return profile
    value = 35.0 if name == "reference_salinity" else 0.0
    return torch.full((int(run_cfg.grid.nz),), value, dtype=torch.float64)


def _compute_reference_profile_from_ic(
    ic_bundle: Any,
    name: str,
) -> torch.Tensor | None:
    # Canonical IC fast-path (via reference_profile_from_tensors on
    # ic_tensors) always returns valid profiles in production, so this
    # ic_bundle-based fallback is only used by non-canonical test/exp cases.
    if ic_bundle is None:
        return None
    data_name = "temperature" if name == "reference_temperature" else "salinity"
    if data_name not in ic_bundle.data:
        return None
    values = torch.as_tensor(
        np.asarray(ic_bundle.data[data_name].values), dtype=torch.float64
    )
    return _reference_profile_from_tensors({data_name: values}, name)


def _enrich_manifest(
    run_cfg: Any,
    ic_bundle: Any,
    ic_tensors: dict[str, torch.Tensor],
    *,
    obcs_source: BoundaryBundleSource,
    wind_source: ERA5WindSource | None,
    bulk_source: Any | None = None,
    surface_flux_source: Any | None = None,
    surface_pressure_load: torch.Tensor | None = None,
    device: str,
    tidal_data: Any | None = None,
    tidal_config_transform: Callable[[Any], Any] | None = None,
    layout: ParallelLayout | None = None,
    reference_profiles: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> Any:
    """L5 wrapper around _canonical_helpers.enrich_manifest.

    Adds the ``t_ref`` / ``s_ref`` ``case.extras`` fallback (via
    ``_reference_profile``) used by experiment configs that pre-date the
    canonical-IC-T/S guarantee. Modern canonical-pipeline cases never
    trigger the fallback because ic_tensors always carries T/S.

    ``tidal_data`` is the canonical tidal lane output (prepared_tidal zarr
    Dataset); when present the OBCS tidal config is built from it instead of
    the legacy per-constituent bins under ``boundary.tidal_data_dir``.
    """
    return _enrich_manifest_core(
        run_cfg,
        ic_bundle,
        ic_tensors,
        obcs_source=obcs_source,
        wind_source=wind_source,
        surface_flux_source=surface_flux_source,
        bulk_source=bulk_source,
        surface_pressure_load=surface_pressure_load,
        device=device,
        reference_profile_fallback=_reference_profile,
        reference_profiles=reference_profiles,
        tidal_data=tidal_data,
        tidal_config_transform=tidal_config_transform,
        layout=layout,
    )


def _fetch_tidal_dataset(run_cfg: Any, output_dir: Path) -> Any | None:
    """Open the canonical prepared_tidal.zarr from the stage1a lane output.

    Returns None when the case has no tidal constituents or the lane did not
    run (pre-lane cases that still rely on boundary.tidal_data_dir bins).
    """
    if not getattr(run_cfg.boundary, "tidal_constituents", ()):
        return None
    from defgcm.io.runner.stages._common import stage_output_dir_for

    zarr_path = (
        stage_output_dir_for("stage1a_prepare_common", Path(output_dir), run_cfg)
        / "prepared_tidal.zarr"
    )
    if not zarr_path.exists():
        return None
    import xarray as xr

    return xr.open_zarr(zarr_path, consolidated=False)


def _discover_dump_steps(mitgcm_run_dir: Path) -> list[int]:
    steps: set[int] = set()
    pattern = re.compile(r"Eta\.(\d{10})\.001\.001\.data")
    for path in mitgcm_run_dir.glob(_DUMP_GLOB):
        match = pattern.match(path.name)
        if match:
            steps.add(int(match.group(1)))
    if not steps:
        for path in mitgcm_run_dir.glob("Eta.*.data"):
            match = re.match(r"Eta\.(\d{10})\.data", path.name)
            if match:
                steps.add(int(match.group(1)))
    return sorted(steps)


def _snapshot_interval_seconds(run_cfg: Any) -> float:
    # Prefer integer-seconds cadence (OutputConfig.resolved_*), fall back
    # to the legacy float-hours fields. resolved_snapshot_seconds already
    # applies the seconds>hours precedence.
    out = run_cfg.output
    if hasattr(out, "resolved_snapshot_seconds"):
        interval = out.resolved_snapshot_seconds()
        if interval > 0.0:
            return interval
        interval = out.resolved_nc_output_seconds()
        if interval > 0.0:
            return interval
        raise ValueError("snapshot/nc_output interval must be positive")
    interval = float(getattr(out, "snapshot_interval_hours", 0.0))
    if interval <= 0.0:
        interval = float(getattr(out, "nc_output_interval_hours", 0.0))
    if interval <= 0.0:
        raise ValueError(
            "snapshot_interval_hours or nc_output_interval_hours must be positive"
        )
    return interval * 3600.0


def _dump_freq_seconds(run_cfg: Any) -> float:
    out = run_cfg.output
    if hasattr(out, "resolved_nc_output_seconds"):
        interval = out.resolved_nc_output_seconds()
        if interval > 0.0:
            return interval
        interval = out.resolved_snapshot_seconds()
        if interval > 0.0:
            return interval
        raise ValueError("nc_output/snapshot interval must be positive")
    interval = float(getattr(out, "nc_output_interval_hours", 0.0))
    if interval <= 0.0:
        interval = float(getattr(out, "snapshot_interval_hours", 0.0))
    if interval <= 0.0:
        raise ValueError(
            "nc_output_interval_hours or snapshot_interval_hours must be positive"
        )
    return interval * 3600.0


def _records_per_nc_file(
    run_cfg: Any,
    snapshot_steps: list[int],
    dt: float,
) -> int | None:
    if _case_kind(run_cfg) != "mit_verification":
        return None
    positive_steps = [int(step) for step in snapshot_steps if int(step) > 0]
    if not positive_steps:
        return 1
    dump_seconds = positive_steps[0] * float(dt)
    return max(1, int((24.0 * 3600.0) / max(dump_seconds, 1.0)))


def _snapshot_steps_from_interval(run_cfg: Any, n_steps: int, dt: float) -> list[int]:
    interval_steps = max(1, int(round(_snapshot_interval_seconds(run_cfg) / dt)))
    steps = set(range(0, n_steps + 1, interval_steps))
    steps.add(n_steps)
    return sorted(steps)


def _monitor_interval(run_cfg: Any) -> int:
    return max(1, int(run_cfg.output.monitor_interval_steps))


def _write_metadata(
    run_cfg: Any, output_dir: Path, mitgcm_run_dir: Path | None
) -> Path:
    metadata = build_run_metadata(
        run_cfg,
        description=str(_extra(run_cfg, "description", run_cfg.run.experiment_name)),
        mitgcm_run_dir=mitgcm_run_dir,
    )
    meta_path = output_dir / "metadata.yaml"
    meta_path.write_text(yaml.safe_dump(metadata, sort_keys=False))
    print(f"[metadata] Written to {meta_path}", flush=True)
    return meta_path


def _run_preprocess_pipeline(output_dir: Path) -> None:
    print("[preprocess] Running schema pipeline ...", flush=True)
    from helper.preprocess.pipeline import run_pipeline

    output_zarr = output_dir / "preprocessed.zarr"
    run_pipeline(experiment_dir=output_dir, output_zarr=output_zarr)
    print(f"[preprocess] Done -> {output_zarr}", flush=True)


def _reference_run_available(output_dir: Path) -> bool:
    run_dir = output_dir / "mitgcm" / "run"
    return run_dir.exists() and any(run_dir.glob("*.data"))


def _overlay_obcs_w(
    w_arr: np.ndarray,
    obcs_source: BoundaryBundleSource,
    sample_time_seconds: float,
) -> np.ndarray:
    if "w" not in obcs_source.fields:
        return w_arr

    try:
        sample = obcs_source.sample_field("w", sample_time_seconds)
    except (KeyError, ValueError):
        return w_arr

    result = w_arr.copy()
    for attr, axis_index in (("west", 0), ("east", -1)):
        values = getattr(sample, attr, None)
        if values is None:
            continue
        arr = values.detach().cpu().numpy()
        if np.isnan(arr).all():
            continue
        result[:, : arr.shape[-1], axis_index] = arr
    return result


def _west_w_overlay_path(run_cfg: Any) -> Path | None:
    """The OBzonalW.bin ``_external_west_w_overlay`` reads, or ``None`` when it is off or absent."""
    overlay_flag = _extra(run_cfg, "apply_obwfile_overlay", None)
    if overlay_flag is None:
        overlay_enabled = str(_extra(run_cfg, "external_pnh_mode", "off")) != "off"
    else:
        overlay_enabled = bool(overlay_flag)
    if not overlay_enabled:
        return None
    input_dir = _extra(run_cfg, "mitgcm_input_dir", None)
    if input_dir is None:
        verification_dir = getattr(run_cfg.case, "mit_verification_dir", None)
        if verification_dir is None:
            return None
        input_dir = Path(str(verification_dir)) / "input"
    path = Path(str(input_dir)) / "OBzonalW.bin"
    return path if path.exists() else None


def _external_west_w_overlay(
    run_cfg: Any,
    sample_time_seconds: float,
) -> np.ndarray | None:
    path = _west_w_overlay_path(run_cfg)
    if path is None:
        return None
    nz = int(run_cfg.grid.nz)
    ny = int(run_cfg.grid.ny)
    series = np.fromfile(path, dtype=">f8").reshape(-1, nz, ny)
    if series.shape[0] == 1:
        return series[0].astype(np.float64, copy=False)
    base_time = float(_extra(run_cfg, "base_time", 0.0))
    forcing_period = float(_extra(run_cfg, "forcing_period", 1.0))
    forcing_cycle = float(
        _extra(run_cfg, "forcing_cycle", forcing_period * series.shape[0])
    )
    phase = (float(sample_time_seconds) - base_time) % forcing_cycle
    slab = int(math.floor(phase / forcing_period)) % series.shape[0]
    next_slab = (slab + 1) % series.shape[0]
    frac = (phase - slab * forcing_period) / forcing_period
    return ((1.0 - frac) * series[slab] + frac * series[next_slab]).astype(
        np.float64,
        copy=False,
    )


def _coord_depth(grid: Any) -> np.ndarray:
    return grid.center_depth.detach().cpu().numpy().astype(np.float64)


def _coord_from_bundle(ic_bundle: Any, name: str) -> np.ndarray | None:
    if ic_bundle is None:
        return None  # Caller must derive coords from grid when canonical IC supplants ic_bundle
    if name not in ic_bundle.data.coords:
        return None
    return np.asarray(ic_bundle.data.coords[name].values, dtype=np.float64)


def _nc_horizontal_coords(
    run_cfg: Any, ic_bundle: Any, *, ny: int, nx: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """``lat`` and ``lon`` for the nc snapshots.

    The IC bundle's ``latitude`` / ``longitude`` when it carries them. A
    staged run has none (stage 1a's canonical IC makes ``_fetch_ic_bundle``
    return None), so a spherical-polar lat-lon grid gets the cell-centre
    degrees of ``run_cfg.grid``. A Cartesian grid gets None, which leaves the
    writer's row-index and metre defaults (``nc_snapshot._default_coords``).
    """
    lat = _coord_from_bundle(ic_bundle, "latitude")
    lon = _coord_from_bundle(ic_bundle, "longitude")
    spherical_latlon = (
        str(getattr(run_cfg.grid, "kind", "latlon")) == "latlon"
        # The grid builders' test: anything but spherical_polar is Cartesian.
        and str(_extra(run_cfg, "grid_mode", "cartesian")) == "spherical_polar"
    )
    if (lat is None or lon is None) and spherical_latlon:
        grid = run_cfg.grid
        centre_lat, centre_lon = latlon_centre_axes(
            grid.center_lat, grid.center_lon, grid.resolution_deg, ny=ny, nx=nx
        )
        lat = centre_lat if lat is None else lat
        lon = centre_lon if lon is None else lon
    return lat, lon


def _obcs_w_extract_factory(
    obcs_source: BoundaryBundleSource,
    base_time_seconds: float,
    *,
    base_extract: Callable[[Any], np.ndarray],
    run_cfg: Any | None,
) -> Callable[[Any], np.ndarray]:
    """Return a W extractor that overlays OBCS / external-west onto diagnostic w.

    ``base_extract`` supplies the diagnostic vertical velocity (typically
    the default ``default_state_variables`` w spec). We then sample
    OBCS w for the current frame time, and — when the run enables the
    external obwfile path — splice the west-boundary slab into column 0.
    """

    def extract(state_obj: Any) -> np.ndarray:
        sample_time = base_time_seconds + current_frame_time_seconds()
        w_arr = _overlay_obcs_w(base_extract(state_obj), obcs_source, sample_time)
        if run_cfg is not None:
            west_w = _external_west_w_overlay(run_cfg, sample_time)
            if west_w is not None:
                # copy() guards against in-place writing the diagnostic
                # array when _overlay_obcs_w returned it unmodified.
                w_arr = w_arr.copy()
                w_arr[:, : west_w.shape[-1], 0] = west_w
        return w_arr

    return extract


def _require_g2_history_authorized(
    *requested_values: object | None,
    internal_override: bool,
) -> None:
    """Require the production env switch for G2; override is test-only."""

    if internal_override or not any(value is not None for value in requested_values):
        return
    from experiments.mitcases.g2_history import g2_history_enabled

    if not g2_history_enabled():
        raise ValueError("direct G2 history injection requires DEFGCM_G2_HISTORY=1")


def run_simulation(
    run_cfg: Any,
    manifest_run_dir: Path,
    logger: logging.Logger,
    *,
    g2_history_path: Path | None = None,
    timing_artifact_path: Path | None = None,
    _g2_internal_override: bool = False,
) -> int:
    """Run a yaml-driven framework simulation. Returns 0 on success.

    Five phases:

    1. ``_resolve_snapshot_steps`` — derive ``n_steps``, ``dt``, and
       ``snapshot_steps`` (MIT-dump-discovered or interval-driven).
    2. ``_load_canonical_and_build_model`` — fetch IC bundle, build grid
       and runtime sources, enrich manifest, instantiate model + state.
    3. ``_build_nc_writer`` — choose the OBCS-W overlay variant when the
       run owns its own diagnostic ``w`` field; otherwise the plain
       writer.
    4. ``_run_integration_loop`` — case-kind-aware step loop
       (mit_verification step-by-step vs ModelRunner full loop).
    5. ``_run_postproc`` — write metadata, defer preprocess pipeline to
       stage4_postproc.

    A cubed-sphere run (grid.kind='cubed_sphere') is dispatched to the dedicated
    cube runtime before any of the above — the five regional phases (OBCS / wind /
    nc_writer / MIT) do not apply to the closed global cube.

    ``_g2_internal_override`` is reserved for tests that inject the private G2
    side channel without the production environment switch.
    """

    _require_g2_history_authorized(
        g2_history_path,
        internal_override=_g2_internal_override,
    )

    benchmark_enabled = timing_artifact_path is not None
    if benchmark_enabled and g2_history_path is not None:
        raise ValueError("benchmark timing and G2 per-step history cannot run together")
    if str(getattr(run_cfg.grid, "kind", "latlon")) == "cubed_sphere":
        if benchmark_enabled:
            raise ValueError(
                "benchmark timing is not implemented for the cubed-sphere loop"
            )
        from experiments.cases._cube_runtime import run_cube_simulation

        return run_cube_simulation(run_cfg, manifest_run_dir, logger)

    distributed_context = init_distributed()
    if distributed_context is not None:
        run_cfg = _run_cfg_with_device(run_cfg, str(distributed_context.device))
    benchmark_uuid = (
        _benchmark_run_uuid(distributed_context) if benchmark_enabled else None
    )

    output_dir = Path(manifest_run_dir)
    mit_dir = getattr(run_cfg.case, "mit_verification_dir", None)
    mitgcm_run_dir = None if mit_dir is None else Path(str(mit_dir)) / "run"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        # config_snapshot/ (source_config.yaml + resolved_args.yaml + real
        # git_state.txt) is written once by stage1a's write_legacy_metadata;
        # stage2 must NOT re-write it (the old _save_config_snapshot emitted a
        # 4-field resolved_args.json that nothing read and clobbered git_state
        # with "git info unavailable").

        print("=" * 98)
        print(f"LOCOMgcm yaml framework run: {run_cfg.run.experiment_name}")
        if distributed_context is not None:
            print(
                "Distributed runtime: "
                f"rank={distributed_context.rank}/{distributed_context.world_size} "
                f"local_rank={distributed_context.local_rank} "
                f"backend={distributed_context.backend} "
                f"device={distributed_context.device}",
                flush=True,
            )
        print("=" * 98)

        n_steps, dt, snapshot_steps = _resolve_snapshot_steps(run_cfg, mitgcm_run_dir)
        if benchmark_enabled:
            if n_steps != _BENCHMARK_ADVANCES:
                raise ValueError(
                    f"benchmark timing requires exactly {_BENCHMARK_ADVANCES} advances; "
                    f"resolved {n_steps}"
                )
            # Timing runs deliver the timing artifact only: no NetCDF frames
            # (a forced final frame cost 1.5 GB per run at N=960 and was never read).
            snapshot_steps = []

        device = _resolve_device(run_cfg)
        if benchmark_enabled:
            _configure_benchmark_backend(device)
        print(f"\n[2/4] Building LOCOMgcm model (device={device}) ...")
        model, state, ctx = _load_canonical_and_build_model(
            run_cfg,
            output_dir=output_dir,
            mitgcm_run_dir=mitgcm_run_dir,
            n_steps=n_steps,
            dt=dt,
            device=device,
            logger=logger,
            distributed_context=distributed_context,
        )

        g2_artifact_metadata: Mapping[str, object] | None = None
        if g2_history_path is not None:
            from experiments.mitcases.g2_history import runtime_producer
            from experiments.mitcases.g2_tier0 import produce_regional_runtime_tier0

            provenance_path = output_dir / "provenance.yaml"
            if not provenance_path.is_file():
                raise ValueError(
                    f"G2 history requires stage provenance at {provenance_path}"
                )
            provenance = yaml.safe_load(provenance_path.read_text())
            if not isinstance(provenance, Mapping):
                raise ValueError("G2 stage provenance must be a mapping")
            extras = _extras(run_cfg)
            case_name = extras.get("case_name")
            if not isinstance(case_name, str) or not case_name.strip():
                raise ValueError(
                    "G2 history requires a non-empty case.extras.case_name"
                )
            variant = extras.get("variant")
            world_size = (
                distributed_context.world_size if distributed_context is not None else 1
            )
            tier0 = produce_regional_runtime_tier0(
                model.grid,
                run_cfg,
                model.dynamics_modules,
            )
            g2_artifact_metadata = {
                "case": case_name,
                "variant": variant,
                "producer": runtime_producer(
                    provenance,
                    state=state,
                    distributed_world_size=world_size,
                ),
                "tier0": tier0,
            }

        nc_writer, flush_nc = _build_nc_writer(
            run_cfg,
            output_dir=output_dir,
            mitgcm_run_dir=mitgcm_run_dir,
            grid=ctx["output_grid"],
            obcs_source=ctx["obcs_source"],
            ic_bundle=ctx["ic_bundle"],
            base_time_seconds=ctx["base_time_seconds"],
            has_external_w_overlay=ctx["has_external_w_overlay"],
            snapshot_steps=snapshot_steps,
            dt=dt,
            distributed_context=distributed_context,
            layout=ctx["layout"],
        )

        integration_kwargs: dict[str, Any] = {
            "model": model,
            "state": state,
            "n_steps": n_steps,
            "dt": dt,
            "snapshot_steps": snapshot_steps,
            "base_time_seconds": ctx["base_time_seconds"],
            "pickup_step_state": ctx["pickup_step_state"],
            "nc_writer": nc_writer,
            "nc_flush": flush_nc,
            "output_dir": output_dir,
            "distributed_context": distributed_context,
            "g2_history_path": g2_history_path,
            "g2_artifact_metadata": g2_artifact_metadata,
            "_g2_internal_override": _g2_internal_override,
        }
        if benchmark_enabled:
            assert timing_artifact_path is not None
            assert benchmark_uuid is not None
            integration_kwargs.update(
                {
                    "timing_artifact_path": timing_artifact_path,
                    "benchmark_run_uuid": benchmark_uuid,
                    # Timing artifacts consume dimensions, not grid arrays.
                    "global_grid": SimpleNamespace(
                        nx=ctx["output_grid"].nx,
                        ny=ctx["output_grid"].ny,
                        nz=ctx["output_grid"].nz,
                    ),
                    "layout": ctx["layout"],
                }
            )
        # Writers now own only their actual inputs; do not keep the loader's
        # global OBCS, IC package or unused output grid throughout integration.
        del ctx
        elapsed, captured_snapshots = _run_integration_loop(
            run_cfg, **integration_kwargs
        )
        flush_nc()
        _barrier(distributed_context)
        if _rank0(distributed_context):
            print(f"  LOCOMgcm runtime: {elapsed:.1f}s")
            print(f"  Captured {captured_snapshots} LOCOMgcm snapshots")

            _run_postproc(run_cfg, output_dir=output_dir, mitgcm_run_dir=mitgcm_run_dir)

            print("\nDone.")
        return 0
    except Exception as exc:
        logger.exception("universal framework run failed")
        print(f"[framework] failed: {exc}", flush=True)
        return 5
    finally:
        shutdown_distributed(distributed_context)


def _resolve_snapshot_steps(
    run_cfg: Any,
    mitgcm_run_dir: Path | None,
) -> tuple[int, float, list[int]]:
    """derive (n_steps, dt, snapshot_steps).

    When ``mitgcm_run_dir`` is set we mirror the MIT dump steps so TO/MIT
    snapshots align cell-by-cell. Otherwise the requested snapshot interval
    in yaml drives the sample schedule.
    """
    n_steps = _steps_from_config(run_cfg)
    dt = float(run_cfg.time.dt)
    if mitgcm_run_dir is not None:
        print("\n[1/4] Discovering MITgcm dump steps ...")
        dump_steps = _discover_dump_steps(mitgcm_run_dir)
        if not dump_steps:
            raise ValueError(f"no MITgcm Eta dumps found in {mitgcm_run_dir}")
        print(f"  Dump steps: {dump_steps}")
        requested = sorted({step for step in dump_steps if step <= n_steps} | {0})
    else:
        print("\n[1/4] Deriving snapshot steps from output interval ...")
        requested = _snapshot_steps_from_interval(run_cfg, n_steps, dt)
        print(f"  Snapshot steps: {requested}")
    snapshot_steps = sorted(set(requested) | {0})
    return n_steps, dt, snapshot_steps


def _start_from_mit_pickup(pickup: Any, run_cfg: Any, model: Any, state: Any,
                           layout: ParallelLayout | None) -> tuple[Any, dict[str, Any]]:
    """The state and step state READ_PICKUP starts the run from (ini_fields.F:27-38).

    The five prognostic records replace the IC fields.  GuNm1, GvNm1 and, for each tracer whose
    tendency the stepper extrapolates, GtNm1 or GsNm1 become the first step's AB2 history, each
    with MITgcm's start flag (``pickup_ab_start``); a missing record leaves a zero history that
    its false flag never reads.  A synchronous step keeps the tracer histories with the momentum
    one; a staggered step's tracer stage keeps its own (``_tracer_ab``), holding a tracer's
    history only when its flag is set.  With the CD scheme, pickup_cd's five fields are its
    state (CD_CODE_READ_PICKUP, cd_code_ini_vars.F:52-54), so the first step takes the AB
    weights as MITgcm's does at myIter = nIter0 != 0 (cd_code_scheme.F:69-75).  u, v and the
    D-grid velocities take LOCOMgcm's redundant outer face as the IC path does, and every field is
    sliced to the rank's tile as the IC tensors are.  The step state is returned as the restart
    codec's mapping, which ``initialize_fresh_step_state`` installs over the fresh head.
    """
    from defgcm.dynamics.cd_scheme.module import cd_scheme_of
    from defgcm.dynamics.cd_scheme.state import CdState, encode_cd_state
    from defgcm.io.mit_pickup import (
        AB_HISTORY_RECORDS, CD_PICKUP_RECORDS, PROGNOSTIC_RECORDS, pickup_ab_start,
        read_mit_pickup, read_mit_pickup_cd,
    )
    from defgcm.io.runner.model_input import append_redundant_face
    from defgcm.steppers.ifs.tendency import MomentumABHistory, encode_momentum_ab_history
    from defgcm.steppers.ifs.tracer_stage import TracerABHistory, encode_tracer_ab_history

    stepper = model.stepper
    cd = cd_scheme_of(model.dynamics_modules) is not None
    # read_pickup.F:251-463 with SPEC_pickup section 1: the records of these runs are not ported.
    refused = [reason for reason, present in (
        ("the cubed sphere", model.grid.face_count != 1),
        (f"the {getattr(stepper, 'name', type(stepper).__name__)} stepper",
         getattr(stepper, "name", None) != "implicit_free_surface"),
        ("nonHydrostatic (Phi_NHyd, GwNm1, dPhiNH)", stepper.nonhydrostatic.nonhydrostatic),
        ("nonlinFreeSurf > 0 (EtaH)", stepper.nonlin_free_surf > 0),
        ("useRealFreshWaterFlux (dEtaHdt, integr_continuity.F:141-162)", stepper.use_real_freshwater_flux),
        ("a dynamic-pressure equation of state (PhiHyd)",
         any(getattr(m, "dynamic_pressure", False) for m in model.dynamics_modules)),
        # cd_code_scheme.F:69-71 keys its first weights on myIter = 0, not on the pickup.
        ("the CD scheme at nIter0 = 0", cd and pickup.n_iter0 == 0),
        ("the CD scheme without pickup_cd (cd_data)", cd and pickup.cd_data is None),
        ("cd_data (pickup_cd) without the CD scheme", not cd and pickup.cd_data is not None),
    ) if present]
    if refused:
        raise ValueError(f"a start from a MITgcm pickup is not ported with {', '.join(refused)}")
    nx, ny, nz = int(run_cfg.grid.nx), int(run_cfg.grid.ny), int(run_cfg.grid.nz)
    fields = read_mit_pickup(pickup.meta, pickup.data, nx=nx, ny=ny, nz=nz)

    def local(values, location: str, like: torch.Tensor) -> torch.Tensor:
        # location: the IC tensor-map name whose grid point the values sit on.
        if location in ("u", "v"):
            values = append_redundant_face(location, values, nx if location == "u" else ny)
        tensor = torch.as_tensor(values)
        if layout is not None:
            tensor = localize_ic_tensors({location: tensor}, layout)[location]
        return tensor.to(dtype=like.dtype, device=like.device).reshape(like.shape)

    state = state.with_tensors(**{name: local(fields[record], name, getattr(state, name).data)
                                  for name, record in PROGNOSTIC_RECORDS})
    start = pickup_ab_start(fields, n_iter0=pickup.n_iter0)
    extrapolated = {"u": True, "v": True, "temp": stepper.ab2_on_temperature_tendency,
                    "salt": stepper.ab2_on_salinity_tendency}
    history = {name: (local(fields[record], name, getattr(state, name).data) if record in fields
                      else torch.zeros_like(getattr(state, name).data))
               for name, record in AB_HISTORY_RECORDS if extrapolated[name]}
    step_state: dict[str, Any] = {}
    if stepper.stagger_tracer_stepping:
        tracers = {name: history.pop(name) for name in ("temp", "salt") if name in history}
        valid = {name: tensor for name, tensor in tracers.items() if start[name]}
        if valid:
            step_state.update(encode_tracer_ab_history(TracerABHistory(
                valid.get("temp"), valid.get("salt"), False)))
    device = state.u.data.device
    flags = {f"{kind}_valid": torch.tensor(start[kind], device=device)
             for kind in ("mom", "temp", "salt") if kind == "mom" or kind in history}
    step_state.update(encode_momentum_ab_history(MomentumABHistory(**history, **flags)))
    if cd:
        cd_fields = read_mit_pickup_cd(pickup.cd_data, nx=nx, ny=ny, nz=nz)
        step_state.update(encode_cd_state(CdState(**{
            name: local(cd_fields[record], location, getattr(state, location).data)
            for name, record, location in CD_PICKUP_RECORDS})))
    return state, step_state


def _load_canonical_and_build_model(
    run_cfg: Any,
    *,
    output_dir: Path,
    mitgcm_run_dir: Path | None,
    n_steps: int,
    dt: float,
    device: str,
    logger: logging.Logger,
    distributed_context: DistributedContext | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """load canonical IC, build grid and runtime sources, instantiate model.

    Returns ``(model, state, ctx)`` where ``ctx`` carries the values the
    later phases need (grid, obcs_source, ic_bundle, base_time_seconds,
    has_external_w_overlay, and pickup_step_state, the step state a start from
    a MITgcm pickup installs, else None).
    """
    # Single dtype source for the regional line: resolve the configured
    # precision (run.precision, default "float64") to a torch.dtype.  Default
    # keeps every downstream builder (IC / grid / obcs / wind / bulk source) on
    # float64 exactly as before; "float32" threads reduced precision through the
    # same already-parameterised dtype argument.  Unknown values are rejected at
    # config load (RunIdentity.__post_init__); guard here too for defence.
    _precision = getattr(run_cfg.run, "precision", "float64")
    if _precision == "float64":
        dtype = torch.float64
    elif _precision == "float32":
        dtype = torch.float32
    else:
        raise ValueError(
            f"unsupported run.precision {_precision!r}; expected 'float64' or 'float32'"
        )
    target_grid = _target_grid(run_cfg)
    layout = _distributed_layout(run_cfg, distributed_context)
    staging_device = "cpu" if layout is not None else device

    ic_bundle = _fetch_ic_bundle(run_cfg, output_dir, target_grid, logger=logger)
    ic_tensors = _ic_tensor_map(
        ic_bundle, manifest_run_dir=output_dir, device=staging_device, dtype=dtype
    )
    global_grid = _build_grid(
        run_cfg,
        manifest_run_dir=output_dir,
        bathymetry=ic_tensors["bathymetry"],
        device=staging_device,
        dtype=dtype,
    )
    # NB: cmask is applied in stage1a `build_model_state`; stage2
    # consumes the already-masked canonical IC, so no second-pass clobber.
    obcs_source = _fetch_obcs_source(
        run_cfg,
        output_dir,
        target_grid,
        device=staging_device,
        dtype=dtype,
        n_steps=n_steps,
        dt=dt,
        logger=logger,
    )
    output_obcs_source = obcs_source
    wind_source = _fetch_wind_source(
        run_cfg,
        output_dir,
        target_grid,
        device=staging_device,
        dtype=dtype,
        n_steps=n_steps,
        dt=dt,
        logger=logger,
    )
    surface_flux_source = _fetch_surface_flux_source(
        run_cfg,
        output_dir,
        target_grid,
        device=staging_device,
        dtype=dtype,
        n_steps=n_steps,
        dt=dt,
        logger=logger,
    )
    surface_pressure_load = _fetch_surface_pressure_load(
        run_cfg,
        output_dir,
        target_grid,
        device=staging_device,
        dtype=dtype,
        logger=logger,
    )
    grid = global_grid
    reference_profiles = None
    if layout is not None:
        # Density references belong to the global initial condition. Resolve
        # them once before slicing; only the two nz profiles move to the GPU.
        reference_profiles = _resolve_reference_profiles(
            run_cfg, ic_bundle, ic_tensors, device=device,
            reference_profile_fallback=_reference_profile,
        )
        grid = localize_structured_grid(global_grid, layout)
        if not _rank0(distributed_context):
            # Only rank 0 writes gathered frames. Other ranks need the global
            # dimensions for metadata, not the global metrics and masks.
            global_grid = SimpleNamespace(
                nx=global_grid.nx, ny=global_grid.ny, nz=global_grid.nz,
            )
        ic_tensors = localize_ic_tensors(ic_tensors, layout)
        obcs_source = localize_boundary_bundle_source(obcs_source, layout)
        wind_source = (localize_bulk_source(wind_source, layout)
                       if isinstance(wind_source, ERA5BulkSource)
                       else localize_wind_source(wind_source, layout))
        surface_flux_source = localize_surface_flux_source(surface_flux_source, layout)
        if surface_pressure_load is not None:
            surface_pressure_load = localize_center_field(surface_pressure_load, layout)
        print(
            "  [distributed] "
            f"rank {layout.rank}/{layout.world_size} tile=({layout.tile_i},{layout.tile_j}) "
            f"x={layout.x_start}:{layout.x_stop} y={layout.y_start}:{layout.y_stop} "
            f"local={layout.nx}x{layout.ny}",
            flush=True,
        )
    # OPEN-B: time-varying ERA5 bulk heat/salt/shortwave source (model-grid),
    # injected into era5_bulk_forcing by enrich_manifest. None unless the case lists it.
    bulk_source = wind_source if isinstance(wind_source, ERA5BulkSource) else _build_bulk_source(
        run_cfg,
        device=staging_device,
        dtype=dtype,
        window_seconds=int(n_steps * dt) if n_steps else None,
    )
    if bulk_source is not None and layout is not None and bulk_source is not wind_source:
        bulk_source = localize_bulk_source(bulk_source, layout)
    tidal_data = _fetch_tidal_dataset(run_cfg, output_dir)
    tidal_config_transform = None
    if layout is not None:
        tidal_config_transform = lambda config: localize_tidal_config(config, layout)
    enriched_cfg = _enrich_manifest(
        run_cfg,
        ic_bundle,
        ic_tensors,
        obcs_source=obcs_source,
        wind_source=wind_source,
        surface_flux_source=surface_flux_source,
        bulk_source=bulk_source,
        surface_pressure_load=surface_pressure_load,
        device=device,
        tidal_data=tidal_data,
        tidal_config_transform=tidal_config_transform,
        layout=layout,
        reference_profiles=reference_profiles,
    )
    if ic_bundle is not None:
        # Reference-profile fallbacks have finished. The only remaining IC
        # consumer reads snapshot coordinates, never the initial fields.
        ic_bundle = replace(
            ic_bundle,
            data=ic_bundle.data.drop_vars(list(ic_bundle.data.data_vars)),
            meta=replace(ic_bundle.meta, variables=()),
        )
    external_pnh_source = _external_pnh_source(
        run_cfg,
        mitgcm_run_dir,
        device=device,
        dtype=dtype,
    )
    model, state = build_model_from_manifest(
        enriched_cfg,
        grid=grid,
        ic_bundle=ic_tensors,
        obcs_source=obcs_source,
        forcing_source=external_pnh_source,
    )

    pickup_step_state = None
    if run_cfg.initial_condition.mit_pickup is not None:
        state, pickup_step_state = _start_from_mit_pickup(
            run_cfg.initial_condition.mit_pickup, run_cfg, model, state, layout)
    base_time_seconds = float(_extra(run_cfg, "base_time", 0.0))
    if _case_kind(run_cfg) == "mit_verification":
        # The fresh-start initializer of the step loop (initialize_fresh_step_state,
        # below) applies the boundaries of this time after the INI_PRESSURE seed.
        model.prepare_step_time(base_time_seconds)
    has_external_w_overlay = (
        _external_west_w_overlay(run_cfg, base_time_seconds) is not None
    )

    ctx = {
        "grid": grid,
        "output_grid": global_grid,
        "obcs_source": output_obcs_source,
        "ic_bundle": ic_bundle,
        "base_time_seconds": base_time_seconds,
        "has_external_w_overlay": has_external_w_overlay,
        "layout": layout,
        "pickup_step_state": pickup_step_state,
    }
    return model, state, ctx


def _build_nc_writer(
    run_cfg: Any,
    *,
    output_dir: Path,
    mitgcm_run_dir: Path | None,
    grid: Any,
    obcs_source: Any,
    ic_bundle: Any,
    base_time_seconds: float,
    has_external_w_overlay: bool,
    snapshot_steps: list[int],
    dt: float,
    distributed_context: DistributedContext | None = None,
    layout: ParallelLayout | None = None,
) -> tuple[Any, Any]:
    """assemble variables and return the unified NC writer."""

    if not snapshot_steps:
        return lambda step, time_s, state_obj: None, lambda: None
    if (
        distributed_context is not None
        and layout is not None
        and distributed_context.rank != 0
    ):

        def distributed_write_nonroot(step: int, time_s: float, state_obj: Any) -> None:
            del step, time_s
            gather_prognostic_state_to_rank0(state_obj, layout)

        return distributed_write_nonroot, lambda: None

    records_per_file = _records_per_nc_file(run_cfg, snapshot_steps, dt)
    diagnose_w = (mitgcm_run_dir is None) or has_external_w_overlay
    variables = default_state_variables(grid, diagnose_w=diagnose_w)
    dataset_attrs: dict[str, Any] = {}
    if diagnose_w:
        dataset_attrs["w_convention"] = "MITgcm upward-positive"

    coords: dict[str, np.ndarray | None] | None = None
    time_units = "seconds"
    if mitgcm_run_dir is None:
        lat, lon = _nc_horizontal_coords(
            run_cfg, ic_bundle, ny=int(grid.ny), nx=int(grid.nx)
        )
        coords = {"depth": _coord_depth(grid), "lat": lat, "lon": lon}
    if diagnose_w:
        base_w_extract = next(spec.extract for spec in variables if spec.name == "w")
        w_extract = _obcs_w_extract_factory(
            replace(
                obcs_source,
                fields={
                    name: tensor
                    for name, tensor in obcs_source.fields.items()
                    if name == "w"
                },
            ),
            base_time_seconds,
            base_extract=base_w_extract,
            run_cfg=run_cfg,
        )
        variables = replace_variable(
            variables,
            "w",
            NCVarSpec(
                name="w",
                dims=("time", "depth", "lat", "lon"),
                extract=w_extract,
                attrs=(
                    {"convention": "MITgcm upward-positive (OBCS overlay)"}
                    if has_external_w_overlay
                    else {}
                ),
            ),
        )
    if mitgcm_run_dir is None and not has_external_w_overlay:
        time_units = "hours"

    write, flush = make_nc_writer(
        output_dir,
        grid,
        _dump_freq_seconds(run_cfg),
        variables=variables,
        coords=coords,
        dataset_attrs=dataset_attrs,
        time_units=time_units,
        records_per_file=records_per_file,
    )
    if distributed_context is None or layout is None:
        return write, flush

    def distributed_write(step: int, time_s: float, state_obj: Any) -> None:
        global_state = gather_prognostic_state_to_rank0(state_obj, layout)
        if distributed_context.rank == 0:
            if global_state is None:
                raise RuntimeError("rank 0 did not receive gathered state")
            write(step, time_s, global_state)

    def distributed_flush() -> None:
        if distributed_context.rank == 0:
            flush()

    return distributed_write, distributed_flush


def _resolve_compile_dynamic(
    *, compile_enabled: bool, rank: int, world_size: int = 1
) -> bool | None:
    """Resolve the shape posture handed to ``torch.compile``.

    Static shapes are the posture (the accelerator reachability contract section 7.1
    rule 3): an unset or ``0`` ``DEFGCM_COMPILE_DYNAMIC`` traces at
    ``dynamic=False``.  ``DEFGCM_COMPILE_DYNAMIC=1`` asks for an explicitly dynamic
    trace and is a *diagnostic* lever: it exists to reproduce symbolic-shape
    defects.  It is not part of the production posture -- the library default
    is static and the driver injects nothing for this key -- and the two
    postures measured speed-neutral ( (3)). The one thing dynamic shapes
    used to buy, headroom against the recompile limit, now comes from the
    posture-independent ``RECOMPILE_LIMIT`` in ``defgcm/runner.py``,
    without the Inductor unbound-symbol defect of.

    **On more than one rank the dynamic posture is refused, at the start of the
    run.**  Under a fully symbolic trace the halo exchange's peer and length
    arguments are derived from tensor shapes and reach ``defgcm::halo_p2p``
    as symbolic integers, which its ``List[int]`` schema does not accept: rank 0
    fails to compile while the others wait in a collective, so the run ends in a
    watchdog timeout minutes later with the compile error buried on one rank.
    Refusing here turns that into one clear
    message before any step runs.  A single rank keeps the lever, which is where
    a symbolic-shape defect is reproduced anyway; widening the operator's schema
    is a different piece of work and is not what this diagnostic switch is for.

    **A differentiated run refuses the dynamic posture too**
    (``defgcm.runner.refuse_dynamic_derivative``): the lever is a forward
    diagnostic.  This function does not know the caller's derivative intent
    (the case generator resolves one posture for a row's forward and gradient
    phases), so the refusal sits where the intent is stated: ModelRunner.run's
    entry (``track_gradients``), the checkpoint window's entry, and the
    compile context, which takes it from its caller.

    ``DEFGCM_COMPILE_DYNAMIC_AUTO=1`` is a second diagnostic lever, outside the
    switch registry: it returns ``None``, torch's automatic-dynamic posture
    (static on the first trace, symbolic sizes only after a recompile).  That
    is the posture the  reading was taken in; the first static-shape
    reading on the same script came out 20% slower () and the two can
    only be told apart on one tree, so the old posture stays reachable as a
    named diagnostic rather than by rolling the mapping back.  Automatic-
    dynamic traces statically first, so it does not hand ``halo_p2p`` symbolic
    lists and is allowed on any rank count; it still masks contract rule 1
    the way the dynamic posture does, so the gate never runs in it.

   The returned value is exactly what reaches ``RunnerCompileConfig.dynamic``.
    The ``compile_dynamic`` dispatch is recorded by the runner: a hit where the
    dynamic posture is handed to torch.compile, a refusal where a
    differentiated run is turned away.  Resolving the switch records nothing.
    """
    from defgcm.accel import gate as _accel_gate
    from defgcm.accel.runtime import parse_bool as _parse_bool

    dynamic = bool(_accel_gate("compile_dynamic").effective_value)
    automatic = _parse_bool(
        os.environ.get("DEFGCM_COMPILE_DYNAMIC_AUTO"),
        default=False,
        name="DEFGCM_COMPILE_DYNAMIC_AUTO",
    )
    if automatic and dynamic:
        raise ValueError(
            "DEFGCM_COMPILE_DYNAMIC=1 and DEFGCM_COMPILE_DYNAMIC_AUTO=1 name two "
            "different postures; set at most one of them."
        )
    if dynamic and compile_enabled and world_size > 1:
        raise ValueError(
            "DEFGCM_COMPILE_DYNAMIC=1 is not available on more than one rank "
            f"(world_size={world_size}): a fully symbolic trace hands "
            "defgcm::halo_p2p symbolic peer and length arguments, which its "
            "List[int] schema rejects, and the ranks that did compile then wait "
            "in a collective until the watchdog fires. "
            "Static shapes are the posture -- unset DEFGCM_COMPILE_DYNAMIC or set it "
            "to 0; the dynamic lever is a single-rank diagnostic."
        )
    if automatic and compile_enabled:
        return None
    return dynamic


def _resolve_compile_pin_schedule(
    *, compile_enabled: bool, rank: int, world_size: int = 1
) -> bool:
    """Resolve whether the compiled step pins its launch order, and record it.

    ``DEFGCM_COMPILE_PIN_SCHEDULE`` (library default ON) switches off Inductor's
    peak-memory scheduler reordering pass for the compiled step **on more than
    one rank**.  The ranks compile different graphs -- the open boundaries live
    on one tile -- and that pass moved the heavy external kernels (KPP+EOS,
    the fused vertical implicit momentum solve, horizontal viscosity, the VIM
    kernel) to different sides of the halo p2p rounds on each rank; each such
    kernel was then paid serially by both ranks inside the NCCL wait: 277.8 vs
    169.0 ms/step at N=960 on two cards. The locality pass stays on because
    it reduced the measured runtime.
    It is not a numerics switch: the launch order changes, no operator does.

    On a single rank there is no peer to wait for, so the switch is inert
    there -- torch's defaults stay and nothing is booked -- which is what the
    registry's ``distributed=("multi",)`` says.  The returned value is exactly
    what reaches ``RunnerCompileConfig.pin_schedule``; the dispatch record
    carries the same value, so the ignition ledger cannot disagree with what
    the runner applied.
    """
    from defgcm.accel import gate as _accel_gate
    from defgcm.accel import record_dispatch as _record_dispatch

    requested = bool(_accel_gate("compile_pin_schedule").effective_value)
    pinned = requested and compile_enabled and world_size > 1
    if pinned:
        _record_dispatch(
            "compile_pin_schedule",
            outcome="hit",
            variant="multi",
            site="experiments.cases._runtime.compile_config",
            backend="inductor",
            rank=rank,
        )
    return pinned


def _run_integration_loop(
    run_cfg: Any,
    *,
    model: Any,
    state: Any,
    n_steps: int,
    dt: float,
    snapshot_steps: list[int],
    base_time_seconds: float,
    pickup_step_state: dict[str, Any] | None,
    nc_writer: Any,
    output_dir: Path,
    nc_flush: Callable[[], None] | None = None,
    distributed_context: DistributedContext | None = None,
    g2_history_path: Path | None = None,
    g2_artifact_metadata: Mapping[str, object] | None = None,
    timing_artifact_path: Path | None = None,
    benchmark_run_uuid: str | None = None,
    global_grid: Any = None,
    layout: ParallelLayout | None = None,
    _g2_internal_override: bool = False,
) -> tuple[float, int]:
    """run the integration loop, returning (elapsed_s, captured_count).

    ``mit_verification`` case kind drives the step-by-step model.step path so
    each TO step lines up with a known MIT dump for cross-validation;
    other case kinds delegate to ``ModelRunner`` which can compile the
    tendency graph for speed.

    ``nc_flush`` is the flush of the writer behind ``nc_writer``.  When station
    output is on, it is called before the station file is written, so no
    snapshot frame is still being appended at that moment.

    Checkpointing (``run_cfg.checkpoint``) is woven in here without touching
    the physics: a ``CheckpointManager`` (when enabled) optionally resumes
    from a prior checkpoint (overriding the seed state, restoring the
    stepper's cross-step caches, and advancing the time origin), then
    captures full-state checkpoints at its scheduled steps via the same
    per-step callback that already streams NetCDF frames.  All step
    bookkeeping below is in *global* step units; ModelRunner runs the
    remaining ``n_steps - start_step`` local steps with a matching
    ``time_offset`` so absolute times line up bitwise with a continuous run.

    ``_g2_internal_override`` is reserved for tests that inject the private G2
    side channel without the production environment switch.
    """
    _require_g2_history_authorized(
        g2_history_path,
        g2_artifact_metadata,
        internal_override=_g2_internal_override,
    )
    from defgcm.checkpoint import CheckpointManager

    benchmark_values = (timing_artifact_path, benchmark_run_uuid, global_grid)
    benchmark_enabled = timing_artifact_path is not None
    if any(value is not None for value in benchmark_values) != all(
        value is not None for value in benchmark_values
    ):
        raise ValueError(
            "timing_artifact_path, benchmark_run_uuid, and global_grid must be supplied together"
        )
    if benchmark_enabled and n_steps != _BENCHMARK_ADVANCES:
        raise ValueError(
            f"benchmark timing requires exactly {_BENCHMARK_ADVANCES} advances; "
            f"resolved {n_steps}"
        )
    section_config = getattr(
        getattr(run_cfg, "observations", None), "section_transport_u", None
    )
    chunked_layout = layout is not None and (layout.nPx > 1 or layout.nPy > 1)
    if section_config is not None and (
        chunked_layout
        or (
            distributed_context is not None
            and distributed_context.world_size > 1
        )
    ):
        raise ValueError("section observation supports single-process only")
    snapshot_step_set = set(snapshot_steps)

    # The four activity classes produced inside this function are
    # instrumented at their call sites below, so a zero in the artifact is a
    # reading.  The classes produced inside ModelRunner are declared just
    # before the document is written.
    activity_ledger = (
        _BenchmarkActivityLedger(_BENCHMARK_ADVANCES) if benchmark_enabled else None
    )
    if activity_ledger is not None:
        activity_ledger.watch("frame_writes")
        activity_ledger.watch("checkpoint_operations")
        activity_ledger.watch("station_output_operations")
        activity_ledger.watch("progress_output_operations")

    ckpt_rank = (
        distributed_context.rank
        if distributed_context is not None and distributed_context.is_distributed
        else None
    )
    ckpt_mgr = (
        None
        if benchmark_enabled
        else CheckpointManager.from_run_config(run_cfg, output_dir, rank=ckpt_rank)
    )
    start_step = 0
    if ckpt_mgr is not None:
        plan = ckpt_mgr.resolve_resume(model, n_steps)
        if plan is not None:
            state = ckpt_mgr.apply_resume(model, plan)
            start_step = plan.start_step
            print(
                f"  [checkpoint] resume from step {start_step} "
                f"(t={plan.model_time_seconds:.1f}s) <- {plan.source_path}",
                flush=True,
            )

    from defgcm.io.station_output import build_station_recorder

    station_rec = None
    if not benchmark_enabled:
        station_rec = build_station_recorder(
            run_cfg,
            model.grid,
            dt,
            distributed_context=distributed_context,
        )
    if station_rec is not None:
        print(
            f"  [station] high-freq output: {len(station_rec.names)} station(s) "
            f"every {station_rec.interval_steps} step(s) -> station_output.nc",
            flush=True,
        )

    g2_collector = None
    section_accumulator = None
    section_geometry = None
    if section_config is not None:
        from defgcm.diagnostics.section_transport import (
            SectionTransportWindowAccumulator,
            section_transport_window_steps,
        )

        u = state.u.data
        section_accumulator = SectionTransportWindowAccumulator(
            section_config.i_face,
            section_config.j0,
            section_config.j1,
            window_steps=section_transport_window_steps(dt),
        )
        section_geometry = (
            model.grid.hfac_w_for(u.dtype, u.device),
            model.grid.metrics.dyG_for(u.dtype, u.device),
            model.grid.layer_thickness_for(u.dtype, u.device),
        )

    def step_callback(local_step: int, _local_time: float, state_obj: Any) -> None:
        if g2_collector is not None:
            g2_collector.sample(local_step, state_obj, model.grid)
        g = start_step + local_step
        abs_time = base_time_seconds + g * dt
        if g in snapshot_step_set:
            if activity_ledger is not None:
                activity_ledger.observe("frame_writes", local_step)
            nc_writer(g, abs_time, state_obj)
        if ckpt_mgr is not None:
            if activity_ledger is not None:
                activity_ledger.observe("checkpoint_operations", local_step)
            ckpt_mgr.maybe_capture(
                model,
                state_obj,
                global_step=g,
                model_time=abs_time,
                dt=dt,
                n_steps=n_steps,
            )
        if station_rec is not None and station_rec.should_record(g):
            if activity_ledger is not None:
                activity_ledger.observe("station_output_operations", local_step)
            station_rec.record(abs_time, state_obj)
        if (
            not benchmark_enabled
            and g > 0
            and g % 100 == 0
            and _rank0(distributed_context)
        ):
            if activity_ledger is not None:
                activity_ledger.observe("progress_output_operations", local_step)
            eta = state_obj.eta.data
            temp = state_obj.temp.data
            pct = 100.0 * g / max(n_steps, 1)
            print(
                f"    step {g}/{n_steps} ({pct:.0f}%) "
                f"eta=[{float(eta.min()):.3e},{float(eta.max()):.3e}] "
                f"T=[{float(temp.min()):.6f},{float(temp.max()):.6f}]",
                flush=True,
            )
        if section_accumulator is not None:
            assert section_geometry is not None
            section_accumulator.observe(
                g,
                state_obj.u.data,
                section_geometry[0],
                section_geometry[1],
                section_geometry[2],
            )

    local_n_steps = n_steps - start_step
    time_offset = base_time_seconds + start_step * dt

    # The stepper must receive the resolved per-equation deltaTs, not the bare
    # scalar clock (which model.step/ModelRunner.run would expand uniformly and
    # thereby overwrite config.dt_mom).  See _production_time_step_bundle.
    time_step = _production_time_step_bundle(model, dt)
    if (g2_history_path is None) != (g2_artifact_metadata is None):
        raise ValueError(
            "g2_history_path and g2_artifact_metadata must be supplied together"
        )
    if g2_history_path is not None:
        from experiments.mitcases.g2_history import G2HistoryCollector

        g2_collector = G2HistoryCollector(
            base_time_seconds=base_time_seconds,
            start_step=start_step,
            time_step=time_step,
        )

    benchmark_timer = (
        BenchmarkStepTimer(device=model.config.device, expected_steps=local_n_steps)
        if benchmark_enabled
        else None
    )
    benchmark_compile_metadata: Mapping[str, object] = {
        "enabled": False,
        "target": "disabled",
    }

    print(
        f"\n[3/4] Running LOCOMgcm ({local_n_steps} steps from global "
        f"step {start_step}, dt={dt}s) ...",
        flush=True,
    )
    if (
        time_step.dt_mom != dt
        or time_step.dt_tracer != dt
        or time_step.dt_freesurf != dt
        or time_step.dt_clock != dt
    ):
        print(
            "  asynchronous time steps: "
            f"dt_mom={time_step.dt_mom}s dt_tracer={time_step.dt_tracer}s "
            f"dt_freesurf={time_step.dt_freesurf}s dt_clock={time_step.dt_clock}s",
            flush=True,
        )
    start = time.perf_counter()
    try:
        if _case_kind(run_cfg) == "mit_verification":
            # This loop bypasses ModelRunner, so it must perform the same fresh
            # boundary/shared-face pass and D6.1 initializer before the step-0
            # callback (which may capture a checkpoint).  A resumed run keeps
            # the restart state apply_resume installed.
            if start_step == 0:
                from defgcm.runner import initialize_fresh_step_state

                state = initialize_fresh_step_state(model, state, pickup_step_state=pickup_step_state)
            captured = 0
            for local_step in range(0, local_n_steps + 1):
                g = start_step + local_step
                if g in snapshot_step_set:
                    captured += 1
                if not benchmark_enabled:
                    step_callback(local_step, local_step * dt, state)
                if local_step == local_n_steps:
                    if benchmark_enabled:
                        step_callback(local_step, local_step * dt, state)
                    break
                if benchmark_timer is not None:
                    benchmark_timer.start_step(local_step)
                state = model.step(
                    state, dt=time_step, time_seconds=time_offset + local_step * dt
                )
                if benchmark_timer is not None:
                    benchmark_timer.end_step(local_step)
                    benchmark_timer.poll()
            if benchmark_timer is not None:
                benchmark_timer.finalize(simulated_dt=time_step.dt_clock)
        else:
            # Local-index snapshot schedule for ModelRunner bookkeeping (the
            # callback itself gates NetCDF/checkpoint on the *global* sets).
            local_snaps = sorted(
                {s - start_step for s in snapshot_steps if start_step <= s <= n_steps}
            )
            if not local_snaps:
                local_snaps = sorted({0, local_n_steps})
            # DEFGCM_MEM_DEBUG=1: record the CUDA allocation history so an OOM dumps a
            # snapshot (load at pytorch.org/memory_viz, or tmp parse) pinpointing
            # which code holds the memory. Also logs post-build baseline + peak.
            import os as _os

            _mem_dbg = None if benchmark_enabled else _os.environ.get("DEFGCM_MEM_DEBUG")
            if _mem_dbg:
                import torch as _t

                if _t.cuda.is_available():
                    _t.cuda.empty_cache()
                    _t.cuda.reset_peak_memory_stats()
                    try:
                        _t.cuda.memory._record_memory_history(max_entries=300000)
                    except Exception as _e:
                        print(
                            f"  [mem-debug] record_memory_history failed: {_e}",
                            flush=True,
                        )
                    print(
                        f"  [mem-debug] post-build baseline: allocated="
                        f"{_t.cuda.memory_allocated() / 1024**3:.2f} GiB reserved="
                        f"{_t.cuda.memory_reserved() / 1024**3:.2f} GiB",
                        flush=True,
                    )
            try:
                # The posture below has never been A/B'd in-model: single-card runs
                # have compiled the tendency since it was written, multi-card runs
                # never have, and no reading exists for what either choice costs.
                # DEFGCM_COMPILE_TARGET makes the posture measurable without editing
                # this line per experiment.  An unknown value raises rather than
                # falling back, because a silently-ignored override would make the
                # arm it names unmeasurable while still producing a number.
                _compile_default = (
                    "disabled"
                    if benchmark_enabled
                    or (
                        distributed_context is not None
                        and distributed_context.is_distributed
                    )
                    else "tendency"
                )
                _compile_override = _os.environ.get("DEFGCM_COMPILE_TARGET")
                if _compile_override is not None:
                    if _compile_override not in ("disabled", "tendency", "model_step"):
                        raise ValueError(
                            f"DEFGCM_COMPILE_TARGET={_compile_override!r} is not one of "
                            "'disabled', 'tendency', 'model_step'"
                        )
                    _compile_target = _compile_override
                else:
                    _compile_target = _compile_default
                from defgcm.accel import gate as _accel_gate
                from defgcm.accel import record_dispatch as _record_dispatch

                # Shape posture: static by default, dynamic only when the
                # diagnostic switch asks for it (see ``_resolve_compile_dynamic``
                # and the static-shape compilation contract).
                _compile_dynamic = _resolve_compile_dynamic(
                    compile_enabled=_compile_target != "disabled",
                    rank=(
                        getattr(distributed_context, "rank", 0)
                        if distributed_context is not None
                        else 0
                    ),
                    world_size=(
                        getattr(distributed_context, "world_size", 1)
                        if distributed_context is not None
                        else 1
                    ),
                )
                # DEFGCM_COMPILE_FULLGRAPH=1 asks for
                # one graph per step (hard error on any break) with TWO eager
                # warm-up steps so first-step host work never meets the tracer.
                # Two, not one: the AB2 second-step path (and any other
                # branch a first step does not take) can still perform a cold
                # constant build, and a cold ``_build_eager`` call inside the
                # traced step is a hard error under ``error_on_graph_break``.
                # The runner records the value in ``compile_metadata``
                # (``defgcm/runner.py`` "eager_warmup_steps").
                _fg_view = _accel_gate("compile_fullgraph")
                _compile_fullgraph = bool(_fg_view.effective_value)
                # Pin the launch order of the compiled
                # step on more than one rank (see ``_resolve_compile_pin_schedule``).
                _compile_pin_schedule = _resolve_compile_pin_schedule(
                    compile_enabled=_compile_target != "disabled",
                    rank=(
                        getattr(distributed_context, "rank", 0)
                        if distributed_context is not None
                        else 0
                    ),
                    world_size=(
                        getattr(distributed_context, "world_size", 1)
                        if distributed_context is not None
                        else 1
                    ),
                )
                compile_config = RunnerCompileConfig(
                    target=_compile_target,
                    dynamic=_compile_dynamic,
                    fullgraph=_compile_fullgraph,
                    pin_schedule=_compile_pin_schedule,
                    eager_warmup_steps=2 if _compile_fullgraph else 0,
                )
                if _compile_fullgraph and _compile_target != "disabled":
                    _record_dispatch(
                        "compile_fullgraph",
                        outcome="hit",
                        variant="model_step" if _compile_target == "model_step" else _compile_target,
                        site="experiments.cases._runtime.compile_config",
                        backend="dynamo",
                        rank=(
                            getattr(distributed_context, "rank", 0)
                            if distributed_context is not None
                            else 0
                        ),
                    )
                result = ModelRunner().run(
                    model,
                    state,
                    steps=local_n_steps,
                    dt=time_step,
                    time_offset=time_offset,
                    snapshot_steps=local_snaps,
                    monitor_interval=_monitor_interval(run_cfg),
                    step_callback=step_callback,
                    step_callback_interval=1,
                    # The step_callback streams each frame to nc_snapshots/ already,
                    # and only the snapshot COUNT is read below — so skip
                    # ModelRunner's O(N_frames) in-memory snapshot accumulation
                    # (stage2 OOM driver for long/large runs; ~16 GB at 720²×50×31,
                    # scales with frame count).
                    collect_snapshots=False,
                    step_timer=benchmark_timer,
                    compile_config=compile_config,
                    benchmark_mode=benchmark_enabled,
                    resume=start_step > 0,
                    pickup_step_state=pickup_step_state,
                )
                benchmark_compile_metadata = result.metadata["compile"]
                if _mem_dbg:
                    import torch as _t

                    if _t.cuda.is_available():
                        print(
                            "  [mem-debug] peak: "
                            f"allocated={_t.cuda.max_memory_allocated() / 1024**3:.2f} GiB "
                            f"reserved={_t.cuda.max_memory_reserved() / 1024**3:.2f} GiB",
                            flush=True,
                        )
            except Exception as _err:
                if _mem_dbg and "out of memory" in str(_err).lower():
                    import torch as _t

                    _snap = output_dir / "to_mem_snapshot.pickle"
                    try:
                        _t.cuda.memory._dump_snapshot(str(_snap))
                        print(
                            f"  [mem-debug] OOM: peak_allocated="
                            f"{_t.cuda.max_memory_allocated() / 1024**3:.2f} GiB "
                            f"peak_reserved={_t.cuda.max_memory_reserved() / 1024**3:.2f} GiB; "
                            f"snapshot -> {_snap}",
                            flush=True,
                        )
                        print(_t.cuda.memory_summary(abbreviated=True), flush=True)
                    except Exception as _e2:
                        print(f"  [mem-debug] snapshot dump failed: {_e2}", flush=True)
                raise
            finally:
                if _mem_dbg:
                    import torch as _t

                    try:
                        _t.cuda.memory._record_memory_history(enabled=None)
                    except Exception:
                        pass
            captured = len(result.snapshot_steps)
    finally:
        if ckpt_mgr is not None:
            ckpt_mgr.close()
        if station_rec is not None:
            # The nc writer's worker appends frames through netCDF4 without
            # xarray's HDF5 lock, and HDF5 here is not thread-safe: drain the
            # writer before xarray writes the station file

            # if the drain raises, so every rank reaches the station gather.
            # The drain is final; run_simulation's later flush returns at once.
            try:
                if nc_flush is not None:
                    nc_flush()
            finally:
                sp = station_rec.flush(output_dir / "station_output.nc")
                if sp is not None:
                    print(
                        f"  [station] wrote {station_rec.n_records} records -> {sp}",
                        flush=True,
                    )
    elapsed = time.perf_counter() - start
    if benchmark_enabled:
        assert timing_artifact_path is not None
        assert benchmark_run_uuid is not None
        assert global_grid is not None
        assert benchmark_timer is not None
        assert activity_ledger is not None
        # The remaining four classes are produced inside ModelRunner, which
        # gates each of them on ``benchmark_mode`` (defgcm/runner.py: the
        # diagnostics block, the log-interval progress line, and the
        # DEFGCM_DIV_DIAG / DEFGCM_WIN_DUMP dumps), and it is entered with
        # ``collect_snapshots=False``.  This loop cannot observe them, so they
        # are declared -- and the declaration is what lands in the artifact's
        # provenance map, rather than being passed off as a measurement.
        activity_ledger.declare_suppressed(
            "snapshot_writes",
            producer_absent=True,
            reason="ModelRunner runs with collect_snapshots=False; the "
            "mit_verification loop keeps no snapshot buffer either",
        )
        activity_ledger.declare_suppressed(
            "diagnostic_reductions",
            producer_absent=True,
            reason="ModelRunner benchmark_mode empties the diagnostic step set",
        )
        activity_ledger.declare_suppressed(
            "debug_output_operations",
            producer_absent=True,
            reason="ModelRunner benchmark_mode skips the DEFGCM_DIV_DIAG and "
            "DEFGCM_WIN_DUMP dumps regardless of the environment",
        )
        activity_ledger.declare_suppressed(
            "progress_output_operations",
            producer_absent=True,
            reason="this loop's own progress line is instrumented above; "
            "ModelRunner's log-interval line is gated on benchmark_mode",
        )
        activity_ledger.declare_suppressed(
            "other_io_operations",
            producer_absent=g2_collector is None,
            reason="the only other in-loop writer is the G2 history collector, "
            "which benchmark mode refuses to construct",
        )
        _write_benchmark_artifact(
            path=timing_artifact_path,
            run_uuid=benchmark_run_uuid,
            timer=benchmark_timer,
            model=model,
            global_grid=global_grid,
            compile_metadata=benchmark_compile_metadata,
            distributed_context=distributed_context,
            layout=layout,
            activity=activity_ledger,
        )
    if g2_collector is not None and _rank0(distributed_context):
        from experiments.mitcases.g2_history import write_history_artifact

        assert g2_history_path is not None
        assert g2_artifact_metadata is not None
        artifact = g2_collector.build_artifact(**g2_artifact_metadata)
        write_history_artifact(g2_history_path, artifact)
    if section_accumulator is not None and _rank0(distributed_context):
        from defgcm.diagnostics.section_transport import write_section_transport_json

        section_path = (
            output_dir
            / "_pipeline_meta"
            / "stage2_torch_run"
            / "section_transport.json"
        )
        write_section_transport_json(
            section_path,
            section_accumulator.records(base_time_s=base_time_seconds, dt_s=dt),
        )
    return elapsed, captured


def _run_postproc(
    run_cfg: Any,
    *,
    output_dir: Path,
    mitgcm_run_dir: Path | None,
) -> None:
    """write metadata, defer preprocess to stage4_postproc."""
    print("\n[4/4] Writing metadata and running preprocess pipeline ...")
    _write_metadata(run_cfg, output_dir, mitgcm_run_dir)
    print(
        "[preprocess] Skipped in stage2; stage4_postproc will run it "
        "after the reference stage is available.",
        flush=True,
    )
