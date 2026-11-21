"""Replayed blocks of LSR linear iterations with device-resident stop state.

The CG2D and CG3D replays (``locomgcm/solvers/cg2d.py``,
``locomgcm/solvers/_cg3d_fused/graph.py``) are the model. One CUDA graph holds a
block of ``BLOCK`` iterations of seaice_lsr.F:875-987: the ordered sweep, the
check every SOLV_NCHECK iterations, the record row and X6. The executed
iteration, the swept directions, the stop tests, the counts and the
relaxation safeguard live in device state, so the block needs no host read;
the host reads the block's record rows once after each replay. An iteration
after both directions stopped, or past SEAICElinearIterMax, changes nothing.

Every value keeps the operands and order of the per-call path: the sweep and
X6 are the same arithmetic and copies, a flex check calls the same
``residual`` (in a compiled step its compiled kernels), and the two residual
means share one ordered reduction. The graph reads only buffers it owns and
long-lived module constants; each pass refreshes coefficients, factors,
geometry and state, as CG2D refreshes its static operands.

A pass without a requested scan snapshot also replays its preparation and
end (``PassGraphs``): seaice_lsr.F:308-779 writes the block operands, the flex
tests and the block counters on the device, and the pass outputs land in one
record buffer that the host copies once. The tests reach the host with the
first block's read, and the next pass's prologue is launched before the host
assembles the finished pass's records, so a pass costs no read of its own.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
import math

import numpy as np
import torch

from locomgcm.accel import cuda_graphs
from locomgcm.accel.tensor_context import ordinary_tensor_mode

from ..types import DynamicsCarry, LinearSystem, Record, scalar

#: Iterations per replay, a multiple of SOLV_NCHECK in the supported programs.
BLOCK = 16

# Device state layout, mirrored from ``lsr_state`` in kernel.cu.
ACTIVE_U, ACTIVE_V, COUNT_U, COUNT_V, M, M_MAX, RAN, SWEPT_U, SWEPT_V, INTS = range(10)
S1, S2, S1A, S2A, WU, WV, ERROR_U, ERROR_V, REALS = range(9)
(LOG_RAN, LOG_SWEPT_U, LOG_SWEPT_V, LOG_CHECKED_U, LOG_CHECKED_V, LOG_ACTIVE_U, LOG_ACTIVE_V,
 LOG_COUNT_U, LOG_COUNT_V, LOG_M, LOG_INTS) = range(11)
_RECORD_COLUMNS = [LOG_M, LOG_SWEPT_U, LOG_SWEPT_V, LOG_CHECKED_U, LOG_CHECKED_V,
                   LOG_ACTIVE_U, LOG_ACTIVE_V, LOG_COUNT_U, LOG_COUNT_V]


@torch.compiler.disable(recursive=False, reason="CUDA graph capture")
def _captured(body, *args) -> torch.cuda.CUDAGraph:
    """Warm ``body`` up on a side stream, then capture it; the CG2D sequence.

    ``body`` must be a skipped host frame (``recursive=False``) so that the
    stages it calls compile as they do on the per-call path. The tensor mode
    is entered here rather than as a decorator, whose wrapper frame would be
    traced.
    """
    with ordinary_tensor_mode():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            # Warm-up compiles and allocates; callers reload every replay input.
            body(*args)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode=cuda_graphs.CAPTURE_ERROR_MODE):
            body(*args)
    cuda_graphs.register(graph)
    return graph


class LinearBlocks:
    """Graph-owned operands, device state and the captured block of one program."""

    @torch.compiler.disable(reason="eager construction of graph-owned buffers")
    @ordinary_tensor_mode()
    def __init__(self, space, residual_means, *, flex: bool, check_interval: int,
                 print_residual: bool, reference: torch.Tensor) -> None:
        from ._fast import load_extension

        if BLOCK % check_interval:
            raise ValueError("the replayed block must hold whole check intervals")
        self.ext = load_extension()
        self.flex, self.check_interval, self.print_residual = flex, check_interval, print_residual
        field = torch.zeros_like(reference)
        rows, cols = reference.shape[-2:]
        self.coef_u = field.new_zeros((7, *field.shape))
        self.coef_v = field.new_zeros((7, *field.shape))
        self.factors_u = field.new_zeros((2, *field.shape[:-2], rows - 4, cols - 4))
        self.factors_v = field.new_zeros((2, *field.shape[:-2], cols - 4, rows - 4))
        self.u, self.v, self.pre_u, self.pre_v = (field.clone() for _ in range(4))
        self.rAw, self.rAs, self.maskInW, self.maskInS = (field.clone() for _ in range(4))
        self.area = field.new_ones(())
        self.zero = field.new_zeros(())
        self.state = torch.zeros(INTS, dtype=torch.int32, device=field.device)
        self.reals = field.new_zeros((REALS,))
        self.log_ints = torch.zeros((BLOCK, LOG_INTS), dtype=torch.int32, device=field.device)
        self.log_reals = field.new_zeros((BLOCK, 4))
        self.host_state = torch.zeros(INTS, dtype=torch.int32, pin_memory=True)
        self.host_log = torch.zeros((BLOCK, LOG_INTS), dtype=torch.int32, pin_memory=True)
        self.host_reals = torch.zeros((BLOCK, 4), dtype=torch.float64, pin_memory=True)
        self.event = torch.cuda.Event()
        # X6 is a pure copy of interior cells into the three halo layers.
        index = torch.arange(field.numel(), dtype=torch.float64, device=field.device).reshape(field.shape)
        source, _ = space.exchange_vector(index, index.clone(), event="X6")
        self.exchange = source.to(torch.int32).reshape(-1)
        self.space, self.residual_means = space, residual_means
        self.system = LinearSystem(rhsU=self.coef_u[5], rhsV=self.coef_v[5], AU=self.coef_u[0],
                                   BU=self.coef_u[1], CU=self.coef_u[2], AV=self.coef_v[0],
                                   BV=self.coef_v[1], CV=self.coef_v[2], uRt1=self.coef_u[3],
                                   uRt2=self.coef_u[4], vRt1=self.coef_v[3], vRt2=self.coef_v[4])
        self.graph = None
        self.passes = None  # PassGraphs of the program's parameters, see dispatcher.pass_graphs

    # Host frames below keep ``recursive=False``: a flex check inside a compiled
    # step captures the same compiled residual kernels as the per-call loop.
    @torch.compiler.disable(recursive=False, reason="host body of the captured block")
    def _iterations(self, geometry) -> None:
        for k in range(BLOCK):
            self.ext.sweep_state(self.coef_u, self.factors_u, self.u, self.coef_v, self.factors_v,
                                 self.v, self.reals, self.state, self.pre_u, self.pre_v)
            check = (k + 1) % self.check_interval == 0
            su = sv = None
            if check and self.flex:
                # flex_check with calculate_mean=printResidual; false means 0.
                su, sv = (self.residual_means(self.system, self.pre_u, self.pre_v, geometry, self.space)
                          if self.print_residual else (self.zero, self.zero))
            self.ext.advance(self.state, self.reals, self.log_ints, self.log_reals, self.pre_u,
                             self.pre_v, self.u, self.v, self.exchange, self.coef_u[6], self.coef_v[6],
                             su, sv, k, check, self.flex)

    @torch.compiler.disable(recursive=False, reason="CUDA graph capture")
    def _capture(self, geometry) -> None:
        self.graph = _captured(self._iterations, geometry)

    @torch.compiler.disable(reason="eager refresh of graph-owned operands")
    def _refresh(self, lines, g, u, v, reals, ints) -> None:
        torch._foreach_copy_(
            [self.coef_u, self.coef_v, self.factors_u, self.factors_v, self.u, self.v,
             self.rAw, self.rAs, self.maskInW, self.maskInS, self.area],
            [lines.coef_u, lines.coef_v, lines.factors_u, lines.factors_v, u, v,
             g.rAw, g.rAs, g.maskInW, g.maskInS, g.globalArea])
        self.reals.copy_(torch.stack(reals))
        self._load_state(ints)

    @torch.compiler.disable(reason="eager load of the device counters")
    def _load_state(self, ints) -> None:
        # The previous load has completed: every replay ends with a host wait.
        self.host_state.copy_(torch.tensor(ints, dtype=torch.int32))
        self.state.copy_(self.host_state, non_blocking=True)

    @torch.compiler.disable(recursive=False, reason="host replay; only a capture compiles its flex residual")
    def run(self, lines, g, u, v, *, relax_u, relax_v, s1, s2, s1a, s2a, error_u, error_v,
            active_u: bool, active_v: bool, count_u: int, count_v: int, m: int, m_max: int,
            my_iter: int, ipass: int, sequence: int, stop_before: int | None):
        """Replay blocks from iteration ``m`` until the loop stops or would reach
        ``stop_before``; return the published state and the record rows.

        Operands and state are refreshed once; later blocks of the pass start
        from the graph's own state. Each block costs one host read.
        """
        reals = (s1, s2, s1a, s2a, relax_u, relax_v, error_u, error_v)
        ints = (int(active_u), int(active_v), count_u, count_v, m, m_max, 0, 0, 0)
        self._refresh(lines, g, u, v, reals, ints)
        if self.graph is None:
            self._capture(self._geometry(g))
            self._refresh(lines, g, u, v, reals, ints)
        result = self._replay(m, my_iter, ipass, sequence, stop_before,
                              active_u, active_v, count_u, count_v)
        result.update(u=self.u.clone(), v=self.v.clone(), records=self.records(result))
        return result

    @torch.compiler.disable(reason="host replay of a prepared pass")
    def run_prepared(self, *, my_iter: int, ipass: int, sequence: int, tests, on_end):
        """Replay a whole pass from m=1 on the operands and counters its prologue wrote.

        The first block is already launched (:meth:`launch_first`); ``on_end``
        runs right after the last read (``PassGraphs.finish``) and its result
        is returned as ``ended``. The returned directions and counts are those
        of the last executed iteration; with none executed the caller keeps
        its own.
        """
        return self._replay(1, my_iter, ipass, sequence, None, False, False, 0, 0, tests=tests,
                            launched=True, on_end=on_end)

    @torch.compiler.disable(reason="eager view of the pass geometry")
    def _geometry(self, g):
        return replace(g, rAw=self.rAw, rAs=self.rAs, maskInW=self.maskInW,
                       maskInS=self.maskInS, globalArea=self.area)

    def _launch(self, tests=None) -> None:
        """One block replay and its read, up to the event (called from disabled frames)."""
        self.graph.replay()
        # The record rows arrive with the block's single read, so their
        # reals are host values; a device copy would cost a read per record.
        self.host_log.copy_(self.log_ints, non_blocking=True)
        self.host_reals.copy_(self.log_reals, non_blocking=True)
        if tests is not None:
            # A replayed pass's flex tests share the first block's read.
            tests[1].copy_(tests[0], non_blocking=True)
        self.event.record()

    @torch.compiler.disable(reason="host launch of a pass's first block")
    def launch_first(self, tests) -> None:
        self._launch(tests)

    @torch.compiler.disable(reason="host replay and record rows")
    def _replay(self, m, my_iter, ipass, sequence, stop_before, active_u, active_v, count_u, count_v,
                tests=None, launched=False, on_end=None):
        logs, reals, replays = [], [], 0
        log, values = self.host_log.numpy(), self.host_reals.numpy()
        while True:
            if not launched:
                self._launch(tests if replays == 0 else None)
            launched = False
            replays += 1
            self.event.synchronize()
            ran = int(log[:, LOG_RAN].sum())
            logs.append(log[:ran].copy())
            reals.append(values[:ran].copy())
            m += ran
            if ran < BLOCK or (stop_before is not None and m + BLOCK > stop_before):
                break
        # A replayed pass ends here, before this host work (see PassGraphs.finish).
        ended = None if on_end is None else on_end(sum(len(rows) for rows in logs))
        log, values = np.concatenate(logs), np.concatenate(reals)
        executed = len(log)
        if executed:
            active_u, active_v = bool(log[-1, LOG_ACTIVE_U]), bool(log[-1, LOG_ACTIVE_V])
            count_u, count_v = int(log[-1, LOG_COUNT_U]), int(log[-1, LOG_COUNT_V])
        published = self.reals.clone()
        # ``records`` builds the Record objects; a replayed pass builds them
        # after the next pass is on the stream.
        return dict(tests=None if tests is None else [bool(value) for value in tests[1].tolist()],
                    s1=published[S1], s2=published[S2],
                    s1a=published[S1A], s2a=published[S2A], relax_u=published[WU],
                    relax_v=published[WV], active_u=active_u, active_v=active_v,
                    count_u=count_u, count_v=count_v, executed=executed, replays=replays,
                    rows=(log, values, sequence, my_iter, ipass), ended=ended)

    @staticmethod
    @torch.compiler.disable(reason="host record rows of replayed iterations")
    def records(result) -> list[Record]:
        log, values, sequence, my_iter, ipass = result["rows"]
        executed = len(log)
        rows = np.empty((executed, 12), dtype=np.int32)
        rows[:, 0] = np.arange(sequence + 1, sequence + executed + 1)
        rows[:, 1], rows[:, 2] = my_iter, ipass
        rows[:, 3:] = log[:, _RECORD_COLUMNS]
        return [Record(i, r) for i, r in zip(torch.from_numpy(rows).unbind(0),
                                             torch.from_numpy(values).unbind(0))]


# Record buffer of a replayed pass: the fields first, then the control reals
# (7), the flex control (5) and the final residuals (2), each scalar group on
# a 16-byte boundary.
_SLOTS = ("uIceC", "vIceC", *(f"carry.{f.name}" for f in fields(DynamicsCarry)),
          *(f"system.{f.name}" for f in fields(LinearSystem)), "UICE", "VICE")
_CONTROL, _FLEX, _FINAL, _SCALARS = 0, 8, 14, 16
# Pass inputs held in graph-owned buffers: tensors, then dataclasses of tensors.
_LIVE = ("u", "v", "uc", "vc", "old_u", "old_v", "initial_nonlinear")
_LIVE_RECORDS = ("carry", "ice", "prepared", "forcing", "g")


@dataclass(frozen=True)
class PassOutputs:
    """One replayed pass's published values, views of one copy of the record buffer."""

    uIceC: torch.Tensor
    vIceC: torch.Tensor
    carry: DynamicsCarry
    system: LinearSystem
    UICE: torch.Tensor
    VICE: torch.Tensor
    control: torch.Tensor
    flex: torch.Tensor
    final: torch.Tensor


def _layout(buffer: torch.Tensor, shape: tuple[int, ...]) -> PassOutputs:
    n = math.prod(shape)
    view = {name: buffer[k * n:(k + 1) * n].view(shape) for k, name in enumerate(_SLOTS)}
    base = len(_SLOTS) * n
    return PassOutputs(
        view["uIceC"], view["vIceC"],
        DynamicsCarry(*(view[f"carry.{f.name}"] for f in fields(DynamicsCarry))),
        LinearSystem(*(view[f"system.{f.name}"] for f in fields(LinearSystem))),
        view["UICE"], view["VICE"], buffer[base + _CONTROL:base + _CONTROL + 7],
        buffer[base + _FLEX:base + _FLEX + 5], buffer[base + _FINAL:base + _FINAL + 2])


def _tensors(value):
    """The tensors of a dataclass of tensors, nested dataclasses included, in field order."""
    for item in (getattr(value, f.name) for f in fields(value)):
        if is_dataclass(item):
            yield from _tensors(item)
        else:
            yield item


def _mirror(value):
    """Graph-owned buffers with the strides of a dataclass of tensors."""
    return type(value)(*(_mirror(item) if is_dataclass(item) else torch.empty_like(item)
                         for item in (getattr(value, f.name) for f in fields(value))))


class PassGraphs:
    """The captured preparation and end of one program's prepared passes.

    The prologue is seaice_lsr.F:308-351 and :382-779 of a pass whose
    preparation runs: predictor, linearization, matrix, line factors, initial
    residual and the flex control with its three tests. It writes the block
    operands and counters, the carry the next pass linearizes from, and the
    record buffer; each predictor branch, with or without the :745 test, has
    its own graph. The epilogue is the final residual and the L7 masking. The
    stages are the per-call functions, so a compiled step captures their
    compiled kernels, and the scalar operations are the per-call ones on the
    same constants.
    """

    @torch.compiler.disable(reason="eager construction of graph-owned buffers")
    @ordinary_tensor_mode()
    def __init__(self, blocks, parameters, stages, space, live) -> None:
        self.blocks, self.parameters, self.stages, self.space = blocks, parameters, stages, space
        self.flex, self.print_residual = blocks.flex, blocks.print_residual
        for name in _LIVE:
            setattr(self, name, torch.empty_like(live[name]))
        for name in _LIVE_RECORDS:
            setattr(self, name, _mirror(live[name]))
        reference, p = live["u"], parameters
        # The per-call solve's constants.
        self.one, self.zero, self.tenths8 = reference.new_tensor(1.0), reference.new_zeros(()), reference.new_tensor(0.8)
        self.tiny, self.cap = reference.new_tensor(1.0e-20), reference.new_tensor(0.99)
        self.wfau, self.wfav = scalar(p.relax_u, reference), scalar(p.relax_v, reference)
        self.lsr_error = scalar(p.lsr_error, reference)
        self.shape = tuple(reference.shape)
        self.buffer = reference.new_zeros(len(_SLOTS) * reference.numel() + _SCALARS)
        # The epilogue's copy of the record buffer, cloned once the next pass is launched.
        self.ended = torch.zeros_like(self.buffer)
        self.outputs = _layout(self.buffer, self.shape)
        self.tests = torch.zeros(3, dtype=torch.bool, device=reference.device)
        self.host_tests = torch.zeros(3, dtype=torch.bool, pin_memory=True)
        n = p.linear_iterations
        # Block counters of a pass without flex tests: both directions from m=1.
        self.counters = torch.tensor((1, 1, n, n, 1, n, 0, 0, 0), dtype=torch.int32, device=reference.device)
        self.prologues, self.end = {}, None

    @property
    def ready(self) -> bool:
        return self.end is not None

    @torch.compiler.disable(reason="eager load of the pass inputs")
    def load(self, live) -> None:
        statics = [getattr(self, name) for name in _LIVE]
        values = [live[name] for name in _LIVE]
        for name in _LIVE_RECORDS:
            statics.extend(_tensors(getattr(self, name)))
            values.extend(_tensors(live[name]))
        torch._foreach_copy_(statics, values)

    @torch.compiler.disable(recursive=False, reason="host body of the captured pass prologue")
    def _prologue_body(self, variant: tuple[str, bool]) -> None:
        variant, small_gate = variant
        linearize, assemble_system, prepare_lines, residual = self.stages
        p, space, b, out = self.parameters, self.space, self.blocks, self.outputs
        u, v, uc, vc, g = self.u, self.v, self.uc, self.vc, self.g
        if variant == "first":
            uc.copy_(u)
            vc.copy_(v)
        elif variant == "euler":
            u.copy_(0.5 * (u + self.old_u))
            v.copy_(0.5 * (v + self.old_v))
            uc.copy_(u)
            vc.copy_(v)
        else:
            uc.copy_(0.5 * (u + uc))
            vc.copy_(0.5 * (v + vc))
        lin = linearize(uc, vc, self.carry, self.prepared, self.forcing, g, p, space)
        system = assemble_system(self.ice, self.old_u, self.old_v, lin, self.prepared, self.forcing, g, p, space)
        lines = prepare_lines(system, g)
        ru, rv, _, _ = residual(system, u, v, g, space, calculate_mean=self.print_residual)
        s1 = s2 = self.zero
        error_u = error_v = self.lsr_error
        if self.flex:
            initial = torch.sqrt(ru * ru + rv * rv)
            if variant == "first":
                self.initial_nonlinear.copy_(initial)
            self.tests.copy_(torch.stack((ru == 0.0, rv == 0.0,
                                          initial < p.nonlinear_tolerance * self.initial_nonlinear)))
            # A zero initial residual stops its direction with S1/S2 = that residual.
            s1, s2 = torch.where(self.tests[0], ru, s1), torch.where(self.tests[1], rv, s2)
            initial = torch.where(initial == 0.0, self.tiny, initial)
            factor = torch.minimum(self.one / (1.0 + torch.abs(torch.log10(initial))), self.cap)
            error_u, error_v = ru * factor, rv * factor
            out.flex.copy_(torch.stack((initial, self.initial_nonlinear, factor, error_u, error_v)))
            # seaice_lsr.F:725-750 on the device: a zero residual stops its
            # direction, and no direction or (ipass > 2) a small residual
            # leaves the linear loop closed (SEAICElinearIterMax = 0).
            zu, zv, small = (test.to(torch.int32) for test in self.tests.unbind())
            go = torch.clamp((1 - zu) + (1 - zv), max=1) * ((1 - small) if small_gate else 1)
            n = p.linear_iterations
            b.state.copy_(torch.stack((1 - zu, 1 - zv, (1 - zu) * n, (1 - zv) * n, torch.ones_like(zu),
                                       go * n, zu * 0, zu * 0, zu * 0)))
        else:
            b.state.copy_(self.counters)
        out.control.copy_(torch.stack((ru, rv, self.lsr_error, self.wfau, self.wfav, self.tenths8, self.tenths8)))
        b.reals.copy_(torch.stack((s1, s2, self.tenths8, self.tenths8, self.wfau, self.wfav, error_u, error_v)))
        torch._foreach_copy_(
            [b.coef_u, b.coef_v, b.factors_u, b.factors_v, b.u, b.v, b.rAw, b.rAs, b.maskInW,
             b.maskInS, b.area, out.uIceC, out.vIceC, *_tensors(out.carry), *_tensors(out.system),
             out.UICE, out.VICE],
            [lines.coef_u, lines.coef_v, lines.factors_u, lines.factors_v, u, v, g.rAw, g.rAs,
             g.maskInW, g.maskInS, g.globalArea, uc, vc, *_tensors(lin.carry), *_tensors(system), u, v])
        # The next pass linearizes from this pass's carry; every read of it is above.
        torch._foreach_copy_(list(_tensors(self.carry)), list(_tensors(lin.carry)))

    @torch.compiler.disable(recursive=False, reason="host body of the captured pass epilogue")
    def _epilogue_body(self) -> None:
        b, g, out = self.blocks, self.g, self.outputs
        if self.print_residual:
            final_u, final_v, _, _ = self.stages[3](b.system, b.u, b.v, g, self.space, calculate_mean=True)
            out.final.copy_(torch.stack((final_u, final_v)))
        # L7, seaice_lsr.F:1070: mask the native storage, no exchange.
        u, v = b.u * g.seaiceMaskU, b.v * g.seaiceMaskV
        torch._foreach_copy_([self.u, self.v, out.UICE, out.VICE], [u, v, u, v])
        self.ended.copy_(self.buffer)

    @torch.compiler.disable(recursive=False, reason="CUDA graph capture of the pass stages")
    def capture(self, variants, live) -> None:
        """Capture the block, the epilogue and every prologue variant, then load ``live``.

        A skipped frame (``recursive=False``) so that the stages compile. Each
        warm-up advances the buffers, so ``live`` loads last. The replays are
        fully disabled frames, since every call of a ``recursive=False`` frame
        repeats Dynamo's frame checks, and a pass's end launches the next pass
        from inside the block loop.
        """
        blocks = self.blocks
        if blocks.graph is None:
            # Before any prologue replay: the block warm-up writes the block buffers.
            blocks._capture(blocks._geometry(self.g))
        self.end = _captured(self._epilogue_body)
        for variant in variants:
            self.load(live)
            self.prologues[variant] = _captured(self._prologue_body, variant)
        self.load(live)

    @torch.compiler.disable(reason="host launch of a captured pass prologue")
    def prologue(self, variant: tuple[str, bool], live, current: bool) -> None:
        """Launch the prologue of ``variant`` (predictor branch, small-test gate).

        Nothing is read back: the flex tests travel with the first block's
        read. ``current`` means the buffers hold this solve's state from the
        previous replayed pass; otherwise ``live`` loads first.
        """
        if not current:
            self.load(live)
        self.prologues[variant].replay()

    def read_tests(self):
        """The (device, pinned host) pair of the flex tests, or None without flex."""
        return (self.tests, self.host_tests) if self.flex else None

    def finish(self, executed: int, following) -> torch.Tensor:
        """End a pass right after its last block read; called from the block loop.

        An executed loop gets the epilogue, which copies the record buffer;
        with a ``following`` prologue variant the next pass's prologue and
        first block go on the stream next, and the copy is cloned after them,
        before the host assembles this pass's records. A pass whose loop did
        not execute has no epilogue and no following pass.
        """
        if not executed:
            return self.buffer.clone()
        self.end.replay()
        if following is not None:
            self.prologues[following].replay()
            self.blocks._launch(self.read_tests())
        return self.ended.clone()

    @torch.compiler.disable(reason="views of a copied record buffer")
    def views(self, buffer: torch.Tensor) -> PassOutputs:
        return _layout(buffer, self.shape)
