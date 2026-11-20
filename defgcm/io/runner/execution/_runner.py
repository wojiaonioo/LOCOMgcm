"""Pipeline execution framework — runs a PipelineSpec through Layer B → C → D.

Used by all stage CLIs.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Any, Mapping

from defgcm.io.runner.config import (
    OperatorSpec,
    PipelineSpec,
    RunConfig,
)
from defgcm.io.runner.layer_a import (
    BundleMeta,
    DataBundle,
    ProvenanceStep,
)
from defgcm.io.runner.layer_a import LocationKind
from defgcm.io.runner.layer_b import (
    DATA_SOURCES,
    DataQuery,
    DataSource,
    Grid,
    SourceContext,
    validate_query_against_capabilities,
)
from defgcm.io.runner.layer_c import MERGERS, OPERATORS
from defgcm.io.runner.layer_d import CONSUMERS, ArtifactRef, ConsumerContext


@dataclass(frozen=True)
class ExecutionContext:
    """Per-stage execution context shared by framework + plugins."""

    run_id: str
    run_cfg: RunConfig
    stage_name: str
    output_dir: Path
    source_context: SourceContext
    consumer_context: ConsumerContext
    logger: Logger
    persist_intermediate: bool = False
    dry_run: bool = False


def _append_provenance(
    bundle: DataBundle, step_kind: str, step_name: str, params: Mapping[str, Any]
) -> DataBundle:
    """Append a ProvenanceStep to bundle (framework-side; ops/consumers don't do this)."""
    new_prov = bundle.meta.provenance + (
        ProvenanceStep(
            kind=step_kind,
            name=step_name,
            params={k: str(v) for k, v in params.items()},  # stringify for hashability
            timestamp=datetime.utcnow(),
            bundle_hash="00000000",
        ),
    )
    new_meta = dataclasses.replace(bundle.meta, provenance=new_prov)
    return DataBundle(data=bundle.data, meta=new_meta)


def framework_call_operator(
    op_spec: OperatorSpec,
    bundle: DataBundle,
    ctx: ExecutionContext,
) -> DataBundle:
    """Pre-validate capabilities + call op + append ProvenanceStep."""
    op_cls = OPERATORS.get(op_spec.op)
    caps = op_cls.capabilities
    # Capability pre-validation
    if bundle.meta.location_kind not in caps.accepts:
        raise ValueError(
            f"operator {op_spec.op!r}: bundle location_kind {bundle.meta.location_kind} "
            f"not in accepts {caps.accepts}"
        )
    if caps.convention_in is not None and bundle.meta.convention != caps.convention_in:
        raise ValueError(
            f"operator {op_spec.op!r}: bundle convention {bundle.meta.convention} "
            f"!= required {caps.convention_in}"
        )

    # Resolve string grid_id refs (e.g. yaml `target_grid: model_grid`) to
    # actual Grid object via ctx.source_context.target_grid_resolver.
    resolved_config = dict(op_spec.config)
    if "target_grid" in resolved_config and isinstance(resolved_config["target_grid"], str):
        resolved_config["target_grid"] = ctx.source_context.target_grid_resolver(
            resolved_config["target_grid"]
        )

    # Instantiate + call
    op = OPERATORS.instantiate(op_spec.op, None, **resolved_config)
    new_bundle = op(bundle)
    # Auto-append provenance (framework's job, not op's)
    new_bundle = _append_provenance(new_bundle, "operator", op_spec.op, op_spec.config)
    return new_bundle


def framework_call_merger(
    op_spec: OperatorSpec,
    primary: DataBundle,
    secondary: DataBundle,
    ctx: ExecutionContext,
) -> DataBundle:
    """Same as framework_call_operator but for N→1 mergers."""
    merger_cls = MERGERS.get(op_spec.op)
    merger = MERGERS.instantiate(op_spec.op, None, **op_spec.config)
    new_bundle = merger(primary, secondary)
    new_bundle = _append_provenance(new_bundle, "operator", op_spec.op + ":merger", op_spec.config)
    return new_bundle


def framework_call_consumer(
    consumer_name: str,
    consumer_config: Mapping[str, Any],
    bundle: DataBundle,
    ctx: ExecutionContext,
) -> ArtifactRef:
    """Pre-validate consumer caps + call consume."""
    consumer_cls = CONSUMERS.get(consumer_name)
    caps = consumer_cls.capabilities
    if bundle.meta.location_kind not in caps.accepts_location_kinds:
        raise ValueError(
            f"consumer {consumer_name!r}: bundle location_kind "
            f"{bundle.meta.location_kind} not in {caps.accepts_location_kinds}"
        )
    if caps.required_convention != bundle.meta.convention:
        # OPAQUE consumers are permissive
        from defgcm.io.runner.layer_a import ConventionTag

        if caps.required_convention != ConventionTag.OPAQUE:
            raise ValueError(
                f"consumer {consumer_name!r}: bundle convention "
                f"{bundle.meta.convention} != required {caps.required_convention}"
            )
    bundle_vars = {v.name for v in bundle.meta.variables}
    missing = set(caps.required_variables) - bundle_vars
    if missing:
        raise ValueError(
            f"consumer {consumer_name!r}: bundle missing required variables {missing}"
        )

    consumer = CONSUMERS.instantiate(consumer_name, ctx.consumer_context, **consumer_config)
    return consumer.consume(bundle)


def _warn_fetch_window_vs_effective_duration(
    run_cfg: RunConfig,
    pipeline: PipelineSpec,
    fetch_days: float,
    logger: Any,
) -> None:
    """Warn when the resolved step count implies a longer run than the fetch window.

    When STEPS= or
    time.steps overrides the step count to a run longer than time.days, the
    source fetch window (derived from time.days) may not cover the full run.
    For time_mode='constant' / IC-only pipelines this is benign; for
    time-varying sources it silently truncates the available forcing/boundary
    data.

    Conditions for warning:
      1. Resolved step count * dt > fetch_days * 86400 (window is short).
      2. The pipeline's primary source has time_mode != 'cyclic' AND
         the pipeline is not the IC-only 'state' fetch (which uses
         first_time_only=True and ignores the window length).

    No behaviour change: the fetch window is not modified.
    """
    from defgcm.io.runner._time_steps import compute_n_steps, exact_total_seconds

    time_cfg = run_cfg.time
    try:
        n_steps = compute_n_steps(
            days=time_cfg.days,
            dt_seconds=time_cfg.dt,
            steps_override=time_cfg.steps,
            seconds_override=time_cfg.seconds if isinstance(time_cfg.seconds, (int, float)) else None,
        )
    except Exception:
        return  # trap errors are raised elsewhere; don't double-emit here

    effective_seconds = exact_total_seconds(n_steps, time_cfg.dt)
    window_seconds = fetch_days * 86400.0

    if effective_seconds <= window_seconds:
        return  # window is big enough

    # Check whether this pipeline/source is time-varying.
    # State pipeline always uses first_time_only=True — window length is irrelevant.
    if pipeline.name == "state":
        return

    # Look up the source spec to check time_mode.
    source_spec = next(
        (s for s in run_cfg.sources if s.name == pipeline.primary_source),
        None,
    )
    if source_spec is not None and source_spec.time_mode == "cyclic":
        # Cyclic sources repeat their data — window length doesn't cap coverage.
        return

    logger.warning(
        "run config: source fetch window (%.1f days = %.0f s, from time.days) "
        "is shorter than the effective run duration (%d steps × %.1f s/step = %.0f s "
        "= %.2f days). "
        "Pipeline %r (source %r, time_mode=%r) may receive insufficient time frames. "
        "Set time.days (or time.seconds) to at least %.1f days to silence this warning.",
        fetch_days,
        window_seconds,
        n_steps,
        time_cfg.dt,
        effective_seconds,
        effective_seconds / 86400.0,
        pipeline.name,
        pipeline.primary_source,
        source_spec.time_mode if source_spec is not None else "unknown",
        effective_seconds / 86400.0,
    )


def run_pipeline_spec(
    pipeline: PipelineSpec,
    run_cfg: RunConfig,
    sources: Mapping[str, DataSource],
    ctx: ExecutionContext,
) -> ArtifactRef:
    """Execute one pipeline (source → ops → consumer) and return artifact ref.

    Algorithm:
      1. primary_bundle = sources[pipeline.primary_source].fetch(query)
      2. for op_spec in pipeline.operators:
          - if op_spec.op == 'merge': call merger with secondary
          - else: call operator
      3. consumer = CONSUMERS.instantiate(...); consume bundle
      4. Return ArtifactRef
    """
    if ctx.dry_run:
        ctx.logger.info(f"[dry-run] would execute pipeline {pipeline.name}")
        return ArtifactRef(
            name=pipeline.consumer + "_dry_run",
            path=None,
            artifact_kind=CONSUMERS.get(pipeline.consumer).capabilities.artifact_kind,
            bundle_hash="dry_run",
            schema_version="0.1.0",
        )

    src = sources.get(pipeline.primary_source)
    if src is None:
        raise ValueError(f"pipeline {pipeline.name}: source {pipeline.primary_source!r} not in sources map")

    # Build the source query: pick location_kind such that:
    #   (1) source produces it, AND
    #   (2) the FIRST non-pass-through operator (if any) accepts it, ELSE
    #   (3) consumer accepts it.
    # (1)+(2) protects against pipelines like `boundary` whose source can produce
    # both GRIDDED_3D_OCEAN and BOUNDARY_TIMESERIES, but the first operator
    # `subset_boundary` only accepts GRIDDED_3D_OCEAN (it's the transformer).
    # ConventionConverter accepts ALL kinds → still falls through to (3).
    consumer_caps = CONSUMERS.get(pipeline.consumer).capabilities
    src_caps = src.capabilities

    initial_kind = None
    # Find first operator that NARROWS location kinds (not pass-through)
    for op_spec in pipeline.operators:
        if op_spec.op == "merge":
            continue
        op_cls = OPERATORS.get(op_spec.op)
        op_accepts = op_cls.capabilities.accepts
        # Skip pass-through operators (accept everything)
        if set(op_accepts) == set(LocationKind):
            continue
        # Found a narrowing operator — pick kind in src.produces ∩ op.accepts
        for kind in op_accepts:
            if kind in src_caps.produces:
                initial_kind = kind
                break
        break  # only consult the first narrowing operator
    # Fallback: try consumer.accepts ∩ src.produces
    if initial_kind is None:
        for kind in consumer_caps.accepts_location_kinds:
            if kind in src_caps.produces:
                initial_kind = kind
                break
    if initial_kind is None:
        if src_caps.produces:
            initial_kind = src_caps.produces[0]
        else:
            raise ValueError(
                f"pipeline {pipeline.name}: source produces nothing"
            )

    # Build time_range from run_cfg (start_date + days window).
    # Without this, sources scanning multi-year archives (HYCOM 2019-2023)
    # try to open every file → OOM on 1781 .nc handles.
    from datetime import timedelta
    start_d = ctx.run_cfg.run.start_date
    fetch_days = max(1.0, ctx.run_cfg.time.days)
    end_d = start_d + timedelta(days=fetch_days)
    query_time_range = (
        datetime.combine(start_d, datetime.min.time()),
        datetime.combine(end_d, datetime.min.time()),
    )

    # Guard: warn when the effective run duration (steps * dt) exceeds the
    # fetch window and a time-varying source is configured for this pipeline.
    # The fetch window is derived from time.days verbatim; when STEPS= overrides
    # the step count, time.steps wins over time.days but the window is not
    # updated — the source may silently return fewer frames than the run needs.
    _warn_fetch_window_vs_effective_duration(
        ctx.run_cfg, pipeline, fetch_days, ctx.logger
    )

    # Resolve target_grid from ctx so sources can pre-slice domain before
    # materialize (HycomNetcdfSource avoids 22GB allocation if it knows the bbox).
    try:
        target_grid_obj = ctx.source_context.target_grid_resolver("model_grid")
    except Exception:
        target_grid_obj = None

    # Streaming hint: the "state" pipeline only consumes t=0 (IC) via
    # `_initial_frame(arr)[0]` in model_input.py, so let the source skip
    # materializing the full time-series (24× memory saving for HYCOM).
    # Boundary pipelines fetch as GRIDDED_3D_OCEAN too (they slice 4 sides
    # via the layer_c `subset_boundary` op AFTER the fetch), so they MUST
    # get the full time-series — do not pass first_time_only there.
    fetch_extras: dict[str, Any] = {}
    if pipeline.name == "state":
        fetch_extras["first_time_only"] = True

    query = DataQuery(
        location_kind=initial_kind,
        variables=src_caps.variables,
        time_range=query_time_range,
        target_grid=target_grid_obj,
        extras=fetch_extras,
    )
    validate_query_against_capabilities(
        query, src_caps, source_name=pipeline.primary_source
    )

    # Content-keyed reuse (opt-in cache.reuse): only external-data regrid
    # pipelines whose output is a pure function of {source files, grid, time,
    # ops, consumer}. On a hit (and not verify mode) we skip fetch+ops+consume
    # entirely and symlink the cached artifact. See execution/_pipeline_cache.py.
    from ._pipeline_cache import PipelineCache

    cache = PipelineCache(run_cfg, ctx.logger)
    cache_payload: dict | None = None
    cache_entry = None
    if cache.active and cache.is_cacheable(pipeline):
        cache_payload = cache.compute_key(pipeline, src, query)
        if not cache.rebuild:
            cache_entry = cache.lookup(pipeline.consumer, cache_payload)
        if cache_entry is not None and not cache.verify:
            ctx.logger.info(
                f"[cache HIT] {pipeline.name} consumer={pipeline.consumer} entry={cache_entry.name}"
            )
            return cache.materialize(cache_entry, ctx)

    bundle = src.fetch(query)

    # Run operator chain
    for op_spec in pipeline.operators:
        if op_spec.op == "merge":
            # Find secondary source
            secondary_name = op_spec.config.get("with")
            if secondary_name is None:
                raise ValueError(f"merge step missing 'with' for secondary source")
            secondary_src = sources.get(secondary_name)
            if secondary_src is None:
                raise ValueError(f"merge step: unknown secondary source {secondary_name!r}")
            sec_query = DataQuery(
                location_kind=bundle.meta.location_kind,
                variables=tuple(v.name for v in bundle.meta.variables),
                time_range=query_time_range, target_grid=None, extras={},
            )
            secondary_bundle = secondary_src.fetch(sec_query)
            bundle = framework_call_merger(op_spec, bundle, secondary_bundle, ctx)
        else:
            bundle = framework_call_operator(op_spec, bundle, ctx)

    # Call consumer
    artifact = framework_call_consumer(
        pipeline.consumer, pipeline.consumer_config, bundle, ctx
    )

    # Cache publish / verify (only when keyed above).
    if cache_payload is not None:
        if cache_entry is not None and cache.verify:
            # Recomputed despite a hit — prove the key is complete.
            cache.assert_bitwise_equal(cache_entry, artifact, pipeline)
        elif cache_entry is None or cache.rebuild:
            entry = cache.publish(
                pipeline.consumer, artifact, cache_payload, overwrite=cache.rebuild,
            )
            ctx.logger.info(
                f"[cache {'REBUILD' if cache.rebuild else 'MISS→publish'}] "
                f"{pipeline.name} consumer={pipeline.consumer} entry={entry.name}"
            )
    return artifact
