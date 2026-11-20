"""Accelerator gates of the sea-ice LSR kernels, with the eager reference fallback.

``seaice_lsr_cuda`` serves both ``defgcm_seaice::lsr_factor`` and
``defgcm_seaice::lsr_sweep`` (variants ``factor`` and ``sweep``);
``seaice_ordered_sum_cuda`` serves ``defgcm_seaice::ordered_sum`` and
``::ordered_sum_pair`` (both residual means in one launch);
``seaice_lsr_graph`` serves replayed linear iterations (variant ``block``) and
replayed whole passes (variant ``pass``, see :class:`.graph.PassGraphs`). A disabled
gate runs the reference silently; an enabled gate records exactly one hit or
one decline per call, and a selected kernel's failure is recorded and raised.
Program conditions are admitted before the segment runs
(``experiments/seaice/run/kernels.py``). A call that still declines falls back
to the reference only for the driver's auto request; an explicit request
raises, as admission would have.
"""

from __future__ import annotations

import torch

from defgcm.accel import REGISTRY, current_rank, dispatch_scope, gate, record_dispatch
from defgcm.accel.tensor_context import grad_mode_requires_reference

LSR_KEY = "seaice_lsr_cuda"
SUM_KEY = "seaice_ordered_sum_cuda"
GRAPH_KEY = "seaice_lsr_graph"
_SITE = {LSR_KEY: "defgcm.seaice.dynamics.sweep", SUM_KEY: "defgcm.seaice.space.ordered_sum",
         GRAPH_KEY: "defgcm.seaice.dynamics.lsr.solve_lsr"}


def _record(key: str, variant: str, outcome: str, reason: str | None = None) -> None:
    record_dispatch(key, outcome=outcome, variant=variant, site=_SITE[key], backend="cuda",
                    rank=current_rank(), reason=reason)


def _decline_reason(values: tuple[torch.Tensor, ...], *, contiguous: bool = True) -> str | None:
    reference = values[0]
    if any(value.device.type != "cuda" for value in values):
        return "non_cuda"
    if any(value.device != reference.device for value in values):
        return "device_mismatch"
    if any(value.dtype != torch.float64 for value in values):
        return "dtype_not_float64"
    if contiguous and any(not value.is_contiguous() for value in values if value.ndim):
        return "non_contiguous"
    if torch.is_grad_enabled() and any(value.requires_grad for value in values):
        return "grad_enabled"
    return None


def _decline(key: str, variant: str, reason: str, view) -> None:
    """Record an inapplicable call; an explicit request raises instead of using the reference."""
    _record(key, variant, "inapplicable", reason)
    if view.default_source == "user":
        raise RuntimeError(
            f"accelerator {key!r} was requested explicitly "
            f"({REGISTRY[key].env_var}={view.raw_spelling!r}), but this {variant} call is "
            f"inapplicable: {reason}; unset the variable to use the reference")


def _run(key: str, variant: str, values: tuple[torch.Tensor, ...], native, reference, *,
         contiguous: bool = True):
    view = gate(key)
    if not view.effective_value:
        return reference()
    with dispatch_scope(key):
        reason = _decline_reason(values, contiguous=contiguous)
        if reason is not None:
            _decline(key, variant, reason, view)
            return reference()
        try:
            result = native()
        except Exception as error:
            _record(key, variant, "error", type(error).__name__)
            raise
        _record(key, variant, "hit")
        return result


def decline_for_forward_mode(key: str, variant: str, values: tuple[torch.Tensor, ...], *,
                             contiguous: bool = True) -> None:
    """A caller takes the eager reference before reaching the operator: forward mode is live.

    The operators register no autograd, so a dual input would come back without its tangent.
    The decline is recorded as :func:`_run` records one, with the same device and dtype
    precedence and ``grad_enabled`` for the derivative; an explicit request raises.
    """
    view = gate(key)
    if not view.effective_value:
        return
    with dispatch_scope(key):
        reason = _decline_reason(values, contiguous=contiguous) or "grad_enabled"
        _decline(key, variant, reason, view)


def factor(a, b, c, transpose: bool, reference):
    from ._fast import factor_cuda

    return _run(LSR_KEY, "factor", (a, b, c), lambda: factor_cuda(a, b, c, transpose), reference)


def sweep_both(coef_u, factors_u, old_u, relax_u, coef_v, factors_v, old_v, relax_v,
               swept_u: bool, swept_v: bool, reference):
    from ._fast import sweep_cuda

    values = (coef_u, factors_u, old_u, relax_u, coef_v, factors_v, old_v, relax_v)
    return _run(LSR_KEY, "sweep", values,
                lambda: sweep_cuda(*values, swept_u, swept_v), reference)


def ordered_sum(block, reference):
    from ._fast import ordered_sum_cuda

    return _run(SUM_KEY, "ordered_sum", (block,), lambda: ordered_sum_cuda(block), reference,
                contiguous=False)


def ordered_sum_pair(first, second, reference):
    from ._fast import ordered_sum_pair_cuda

    return _run(SUM_KEY, "ordered_sum_pair", (first, second),
                lambda: ordered_sum_pair_cuda(first, second), reference, contiguous=False)


@torch.compiler.disable(reason="host gate decision and graph cache")
def linear_blocks(space, reference: torch.Tensor, check_interval: int, flex: bool,
                  print_residual: bool, residual_means):
    """The replayed block context of this program, or None for the per-call loop.

    The context is cached on ``space``, which owns the native topology for the
    whole run; the key holds no tensor identity. A declined request records
    its reason; as in :func:`_run`, an explicit request raises instead.
    """
    from .graph import BLOCK, LinearBlocks

    view = gate(GRAPH_KEY)
    if not view.effective_value:
        return None
    with dispatch_scope(GRAPH_KEY):
        # A replay has no tape and no tangent: an enabled gradient mode or a live
        # forward mode (the shared policy) keeps the per-call loop.
        reason = ("grad_enabled" if grad_mode_requires_reference() else _decline_reason((reference,))
                  or (None if BLOCK % check_interval == 0 else "check_interval"))
        if reason is not None:
            _record(GRAPH_KEY, "block", "inapplicable", reason)
            if view.default_source == "user":
                raise RuntimeError(
                    f"accelerator {GRAPH_KEY!r} was requested explicitly "
                    f"({REGISTRY[GRAPH_KEY].env_var}={view.raw_spelling!r}), but this solve is "
                    f"inapplicable: {reason}; unset the variable to use the per-call loop")
            return None
    cache = space.__dict__.setdefault("_lsr_blocks", {})
    key = (str(reference.device), flex, check_interval, bool(print_residual))
    blocks = cache.get(key)
    if blocks is None:
        try:
            blocks = LinearBlocks(space, residual_means, flex=flex, check_interval=check_interval,
                                  print_residual=print_residual, reference=reference)
        except Exception as error:
            _record(GRAPH_KEY, "block", "error", type(error).__name__)
            raise
        cache[key] = blocks
    return blocks


@torch.compiler.disable(recursive=False, reason="host replay; a capture compiles its flex residual")
def replay_block(blocks, **kwargs):
    """Replayed blocks of one pass, one hit per replay; a failure is recorded and raised."""
    try:
        result = blocks.run(**kwargs)
    except Exception as error:
        _record(GRAPH_KEY, "block", "error", type(error).__name__)
        raise
    if result["replays"]:
        record_dispatch(GRAPH_KEY, outcome="hit", variant="block", site=_SITE[GRAPH_KEY], backend="cuda",
                        rank=current_rank(), reason=None, count=result["replays"])
    return result


@torch.compiler.disable(reason="host pass graph cache")
def pass_graphs(blocks, parameters, stages, space, live):
    """The pass graphs of ``blocks`` for these parameters, built once per program.

    They bake the parameter values and hold the parameter object; another
    parameter object rebuilds them.
    """
    from .graph import PassGraphs

    graphs = blocks.passes
    if graphs is None or graphs.parameters is not parameters:
        try:
            graphs = PassGraphs(blocks, parameters, stages, space, live)
        except Exception as error:
            _record(GRAPH_KEY, "pass", "error", type(error).__name__)
            raise
        blocks.passes = graphs
    return graphs


# A replayed pass runs as: capture_pass once per program (a skipped frame, so
# the stages compile), launch_pass for a solve's first replayed pass or one
# after a per-call pass, then replay_pass, whose end (PassGraphs.finish)
# launches the epilogue and the next pass. The per-pass calls are fully
# disabled frames: a recursive=False frame repeats Dynamo's frame checks on
# every call.

@torch.compiler.disable(recursive=False, reason="CUDA graph capture of the pass graphs")
def capture_pass(graphs, variants, live) -> None:
    try:
        graphs.capture(variants, live)
    except Exception as error:
        _record(GRAPH_KEY, "pass", "error", type(error).__name__)
        raise


@torch.compiler.disable(reason="host launch of a captured pass prologue and its first block")
def launch_pass(graphs, variant, live, current: bool) -> None:
    """Launch the prologue and first block of one prepared pass; a failure is recorded and raised."""
    try:
        graphs.prologue(variant, live, current)
        graphs.blocks.launch_first(graphs.read_tests())
    except Exception as error:
        _record(GRAPH_KEY, "pass", "error", type(error).__name__)
        raise


@torch.compiler.disable(reason="host replay of a prepared pass")
def replay_pass(blocks, **kwargs):
    """The blocks of a prepared pass; :func:`count_pass` records its hits."""
    try:
        return blocks.run_prepared(**kwargs)
    except Exception as error:
        _record(GRAPH_KEY, "pass", "error", type(error).__name__)
        raise


@torch.compiler.disable(reason="host dispatch record of a replayed pass")
def count_pass(result) -> None:
    """One hit per replayed graph of the pass: its prologue and blocks.

    Recorded after the next pass is on the stream.
    """
    record_dispatch(GRAPH_KEY, outcome="hit", variant="pass", site=_SITE[GRAPH_KEY], backend="cuda",
                    rank=current_rank(), reason=None, count=result["replays"] + 1)
