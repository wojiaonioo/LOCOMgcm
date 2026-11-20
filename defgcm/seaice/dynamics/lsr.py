"""Finite Picard/LSR program and its public per-pass/per-scan solve state."""

from dataclasses import fields

import torch
from torch import Tensor

from .matrix import assemble_system
from .physics import linearize
from .sweep import prepare_lines, sweep
from .types import (
    INNER, DynamicsCarry, DynamicsForcing, DynamicsGeometry, DynamicsParameters,
    DynamicsSpace, IceState, Linearization, LinearSystem, LsrResult, PassState,
    PreparedDynamics, Record, ScanState, record, scalar,
)


def scan_tiles(system: LinearSystem, u: Tensor, v: Tensor,
               swept_u: bool, swept_v: bool, relax_u: Tensor, relax_v: Tensor,
               g: DynamicsGeometry, space: DynamicsSpace) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """L6 pre-exchange boundary, seaice_lsr.F:875-910, for a single scan.

    Prepares the pass operands itself; ``solve_lsr`` prepares them once per
    pass. The entry velocities are the uTmp/vTmp snapshots and are returned.
    """
    del space
    new_u, new_v = sweep(prepare_lines(system, g), u, v, swept_u, swept_v, relax_u, relax_v)
    return new_u, new_v, u, v


def residual(system: LinearSystem, u: Tensor, v: Tensor, g: DynamicsGeometry,
              space: DynamicsSpace, *, calculate_mean: bool) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """seaice_lsr.F:1237-1284; one continuous tile,j,i sum per direction."""
    r, w, q = space.read, space.write, INNER
    ru = (r(system.rhsU, q) + r(system.uRt1, q) * r(u, q, dj=-1)
          + r(system.uRt2, q) * r(u, q, dj=1)
          - (r(system.AU, q) * r(u, q, di=-1) + r(system.BU, q) * r(u, q)
             + r(system.CU, q) * r(u, q, di=1)))
    rv = (r(system.rhsV, q) + r(system.vRt1, q) * r(v, q, di=-1)
          + r(system.vRt2, q) * r(v, q, di=1)
          - (r(system.AV, q) * r(v, q, dj=-1) + r(system.BV, q) * r(v, q)
             + r(system.CV, q) * r(v, q, dj=1)))
    zero = torch.zeros_like(u)
    su, sv = u.new_zeros(()), v.new_zeros(())
    if calculate_mean:
        # One reduction folds both directions; each keeps its own accumulator.
        su, sv = space.ordered_sum_pair(w(zero, ru * ru * r(g.rAw, q) * r(g.maskInW, q), q),
                                        w(zero, rv * rv * r(g.rAs, q) * r(g.maskInS, q), q), q)
        # The source only takes sqrt for a strictly positive accumulation.
        su = torch.where(su > 0.0, torch.sqrt(torch.where(su > 0.0, su / g.globalArea, torch.ones_like(su))), su)
        sv = torch.where(sv > 0.0, torch.sqrt(torch.where(sv > 0.0, sv / g.globalArea, torch.ones_like(sv))), sv)
    return su, sv, w(zero, ru, q), w(zero, rv, q)


def residual_means(system: LinearSystem, u: Tensor, v: Tensor, g: DynamicsGeometry,
                   space: DynamicsSpace) -> tuple[Tensor, Tensor]:
    """The two means of :func:`residual` with calculate_mean, for a replayed check."""
    s1, s2, _, _ = residual(system, u, v, g, space, calculate_mean=True)
    return s1, s2


def flex_check(system: LinearSystem, u: Tensor, v: Tensor, g: DynamicsGeometry,
               space: DynamicsSpace, error_u: Tensor, error_v: Tensor,
               calculate_mean: bool) -> tuple[Tensor, Tensor, Tensor]:
    """seaice_lsr.F:917-933 tensor part: residuals and both stop tests."""
    s1, s2, _, _ = residual(system, u, v, g, space, calculate_mean=calculate_mean)
    return s1, s2, torch.stack((s1 < error_u, s2 < error_v))


def change_check(u: Tensor, v: Tensor, old_u: Tensor, old_v: Tensor, g: DynamicsGeometry,
                 space: DynamicsSpace, s1: Tensor, s2: Tensor, s1a: Tensor, s2a: Tensor,
                 relax_u: Tensor, relax_v: Tensor, error: Tensor, *, check_u: bool,
                 check_v: bool, safeguard: bool) -> tuple[Tensor, ...]:
    """seaice_lsr.F:936-983 tensor part; the WFAU2=0 safeguard stays on the device."""
    if check_u:
        s1 = space.max_abs((u - old_u) * g.seaiceMaskU, INNER)
        if safeguard:
            relax_u = torch.where(s1 > s1a, torch.zeros_like(relax_u), relax_u)
        s1a = s1
    if check_v:
        s2 = space.max_abs((v - old_v) * g.seaiceMaskV, INNER)
        if safeguard:
            relax_v = torch.where(s2 > s2a, torch.zeros_like(relax_v), relax_v)
        s2a = s2
    return s1, s2, s1a, s2a, relax_u, relax_v, torch.stack((s1 < error, s2 < error))


@torch.compiler.disable(reason="host read of packed solver tests")
def host_flags(flags: Tensor) -> list[bool]:
    """The single device-to-host read of an executed convergence check."""
    return [bool(value) for value in flags.tolist()]


@torch.compiler.disable(recursive=False, reason="host LSR convergence and relaxation branches")
def check_scan(system: LinearSystem, u: Tensor, v: Tensor, old_u: Tensor, old_v: Tensor,
               g: DynamicsGeometry, p: DynamicsParameters, space: DynamicsSpace,
               *, m: int, active_u: bool, active_v: bool, count_u: int, count_v: int,
               s1: Tensor, s2: Tensor, s1a: Tensor, s2a: Tensor, relax_u: Tensor,
               relax_v: Tensor, error_u: Tensor, error_v: Tensor, print_residual: bool
               ) -> tuple[bool, bool, bool, bool, int, int, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """L6(a) stopping and relaxation changes, seaice_lsr.F:917-985.

    ``error_u``/``error_v`` are LSR_ERRORfu/fv for flex and LSR_ERROR otherwise.
    """
    checked_u = checked_v = False
    if m % p.check_interval == 0:
        if p.use_lsr_flex:
            checked_u = checked_v = active_u or active_v
            if checked_u:
                s1, s2, flags = flex_check(system, u, v, g, space, error_u, error_v, print_residual)
                stop_u, stop_v = host_flags(flags)
                # These assignments intentionally also apply to a stopped direction.
                if stop_u:
                    count_u, active_u = m, False
                if stop_v:
                    count_v, active_v = m, False
        else:
            checked_u, checked_v = active_u, active_v
            if checked_u or checked_v:
                s1, s2, s1a, s2a, relax_u, relax_v, flags = change_check(
                    u, v, old_u, old_v, g, space, s1, s2, s1a, s2a, relax_u, relax_v, error_u,
                    check_u=checked_u, check_v=checked_v, safeguard=m > 1)
                stop_u, stop_v = host_flags(flags)
                if checked_u and stop_u:
                    count_u, active_u = m, False
                if checked_v and stop_v:
                    count_v, active_v = m, False
    return (checked_u, checked_v, active_u, active_v, count_u, count_v,
            s1, s2, s1a, s2a, relax_u, relax_v)


@torch.compiler.disable(recursive=False, reason="host LSR control record")
def lsr_control(integers: tuple[int, ...], common_reals: tuple,
                flex_reals: Tensor | None, reference: Tensor) -> tuple[Record, Tensor | None]:
    """L5 boundary after seaice_lsr.F:776; separate common/flex payloads."""
    return record(integers, common_reals, reference), flex_reals


@torch.compiler.disable(recursive=False, reason="host LSR scan index; exchange remains compilable")
def lsr_scan(old_u: Tensor, old_v: Tensor, pre_u: Tensor, pre_v: Tensor,
              space: DynamicsSpace, *, my_iter: int, ipass: int, m: int) -> ScanState:
    """L6(b) boundary: execute X6 after the completed L6(a) check at :985."""
    u, v = space.exchange_vector(pre_u, pre_v, event="X6")
    return ScanState(my_iter, ipass, m, old_u, old_v, pre_u, pre_v, u, v)


@torch.compiler.disable(recursive=False, reason="host LSR pass count record")
def lsr_pass_end(u: Tensor, v: Tensor, g: DynamicsGeometry, *, sequence: int,
                  my_iter: int, ipass: int, count_u: int, count_v: int,
                  active_u: bool, active_v: bool) -> tuple[Tensor, Tensor, Record]:
    """L7 boundary at seaice_lsr.F:1070: mask the native storage, no exchange."""
    u, v = u * g.seaiceMaskU, v * g.seaiceMaskV
    end = record((sequence, my_iter, ipass, count_u, count_v, int(active_u), int(active_v)), (), u)
    return u, v, end


@torch.compiler.disable(recursive=False, reason="finite LSR host loop; compile called tensor kernels")
def solve_lsr(ice: IceState, carry: DynamicsCarry, prepared: PreparedDynamics,
              forcing: DynamicsForcing, g: DynamicsGeometry, p: DynamicsParameters,
              space: DynamicsSpace, *, my_iter: int, print_residual: bool,
              capture_scans: frozenset[tuple[int, int, int]] = frozenset({(0, 1, 1), (0, 1, 2)})) -> LsrResult:
    """L3-L7/B5 finite iteration with complete common and flex records.

    capture_scans selects L6(b) snapshots by (entry_iter, ipass, m). All L5,
    L6(a), and L7 records are always returned. Initial/final residuals obey
    printResidual as an algorithm input, including its false/zero behavior.
    """
    my_iter = int(my_iter)  # Compiled entry clocks are scalar tensors.
    u, v = ice.UICE.clone(), ice.VICE.clone()
    old_u, old_v = u.clone(), v.clone()
    uc, vc = u, v
    do_nonlinear = True
    # Host constants are copied to the device once per solve, not once per use.
    one, zero, tenths8 = u.new_tensor(1.0), u.new_zeros(()), u.new_tensor(0.8)
    tiny, cap = u.new_tensor(1.0e-20), u.new_tensor(0.99)
    wfau, wfav, lsr_error = scalar(p.relax_u, u), scalar(p.relax_v, v), scalar(p.lsr_error, u)
    initial_nonlinear = one
    passes = []
    check_sequence = end_sequence = 0
    # LSR_mixIniGuess=1 is required; hence residGate at :606 is always true.
    resid_gate = print_residual or p.mix_initial_guess >= 1 or p.use_lsr_flex
    from ._lsr_cuda.dispatcher import (capture_pass, count_pass, launch_pass, linear_blocks,
                                       pass_graphs, replay_block, replay_pass)
    from ._lsr_cuda.graph import BLOCK

    blocks = None
    # current: the pass graph buffers hold u, v, uc, vc and the carry of the
    # previous pass, which ran as replayed graphs; launched: the pass whose
    # prologue is already on the stream.
    current = replayed = False
    launched = None

    def variant(k: int) -> tuple[str, bool]:
        # seaice_lsr.F:308-351 predictor branch, and whether :745 applies.
        return ("first" if k == 1 else "euler" if k == 2 and p.nonlinear_iterations <= 2
                else "relaxed", k > 2)

    def replayable(k: int) -> bool:
        return not any((i, j) == (my_iter, k) for i, j, _ in capture_scans)

    for ipass in range(1, p.nonlinear_iterations + 1):
        prep_ran = do_nonlinear
        snapshots = sorted(k for i, j, k in capture_scans if (i, j) == (my_iter, ipass))
        if prep_ran and launched != ipass:
            blocks = linear_blocks(space, u, p.check_interval, p.use_lsr_flex, print_residual, residual_means)
        # A replayed pass's end keys on an executed block, which needs at least one linear iteration.
        if prep_ran and blocks is not None and not snapshots and p.linear_iterations > 0:
            # The whole pass replays: prologue graph, blocks, epilogue graph.
            if launched != ipass:
                live = dict(u=u, v=v, uc=uc, vc=vc, old_u=old_u, old_v=old_v,
                            initial_nonlinear=initial_nonlinear, carry=carry, ice=ice,
                            prepared=prepared, forcing=forcing, g=g)
                graphs = pass_graphs(blocks, p, (linearize, assemble_system, prepare_lines, residual), space, live)
                if not graphs.ready:
                    capture_pass(graphs, sorted({variant(k) for k in range(1, p.nonlinear_iterations + 1)}), live)
                    current = False
                launch_pass(graphs, variant(ipass), live, current)
            current = replayed = True
            following = (variant(ipass + 1) if ipass < p.nonlinear_iterations and replayable(ipass + 1)
                         else None)
            out = replay_pass(blocks, my_iter=my_iter, ipass=ipass, sequence=check_sequence,
                              tests=graphs.read_tests(), on_end=lambda executed: graphs.finish(executed, following))
            wu, wv, s1, s2, s1a, s2a = wfau, wfav, zero, zero, tenths8, tenths8
            active_u = active_v = True
            count_u = count_v = p.linear_iterations
            if p.use_lsr_flex:
                zero_u, zero_v, small = out["tests"]
                if zero_u:
                    active_u, count_u = False, 0
                if zero_v:
                    active_v, count_v = False, 0
                do_nonlinear = do_nonlinear and (active_u or active_v)
                do_nonlinear = do_nonlinear and not (ipass > 2 and small)
            integers = (ipass, my_iter, ipass, int(prep_ran), int(do_nonlinear), int(active_u), int(active_v),
                        count_u, count_v, p.linear_iterations, int(print_residual), int(resid_gate))
            checks, end = (), None
            if out["executed"]:
                s1, s2, s1a, s2a, wu, wv = out["s1"], out["s2"], out["s1a"], out["s2a"], out["relax_u"], out["relax_v"]
                active_u, active_v, count_u, count_v = out["active_u"], out["active_v"], out["count_u"], out["count_v"]
                check_sequence += out["executed"]
            # finish() ran the epilogue of an executed loop and launched the
            # next pass; an executed loop is exactly do_nonlinear.
            buffer = out["ended"]
            launched = ipass + 1 if out["executed"] and following is not None else None
            count_pass(out)
            if out["executed"]:
                checks = tuple(blocks.records(out))
            got = graphs.views(buffer)
            u, v, uc, vc, carry, system = got.UICE, got.VICE, got.uIceC, got.vIceC, got.carry, got.system
            ru, rv = got.control[0], got.control[1]
            flex = None
            if p.use_lsr_flex:
                flex = got.flex
                initial_nonlinear, error_u, error_v = flex[1], flex[3], flex[4]
                if not do_nonlinear:
                    s1, s2 = (ru if zero_u else s1), (rv if zero_v else s2)
            if do_nonlinear:
                end_sequence += 1
                end = record((end_sequence, my_iter, ipass, count_u, count_v, int(active_u), int(active_v)), (), u)
            final_u, final_v = (got.final[0], got.final[1]) if end is not None and print_residual else (None, None)
            control = Record(torch.tensor(integers, dtype=torch.int32), got.control)
            passes.append(PassState(my_iter, ipass, Linearization(uc, vc, carry), system, control, flex,
                                    checks, (), end, u, v, final_u, final_v))
            continue
        current = False
        if prep_ran:
            # seaice_lsr.F:308-351: predictor, modified Euler, relaxed Picard.
            wu, wv = wfau, wfav
            s1, s2 = zero, zero
            s1a, s2a = tenths8, tenths8
            if ipass == 1:
                uc, vc = u.clone(), v.clone()
            elif ipass == 2 and p.nonlinear_iterations <= 2:
                u, v = 0.5 * (u + old_u), 0.5 * (v + old_v)
                uc, vc = u.clone(), v.clone()
            else:
                uc, vc = 0.5 * (u + uc), 0.5 * (v + vc)
            lin = linearize(uc, vc, carry, prepared, forcing, g, p, space)
            carry = lin.carry
            system = assemble_system(ice, old_u, old_v, lin, prepared, forcing, g, p, space)
            lines = prepare_lines(system, g)
            ru, rv, _, _ = residual(system, u, v, g, space, calculate_mean=print_residual)
            active_u = active_v = True
            count_u = count_v = p.linear_iterations
        # Both the outer control point and its cached inputs survive a closed
        # preparation gate: seaice_lsr.F:719-779.
        flex = None
        error_u = error_v = lsr_error
        if p.use_lsr_flex:
            initial = torch.sqrt(ru * ru + rv * rv)
            if ipass == 1:
                initial_nonlinear = initial
            # One packed read for the three tests at :725, :730 and :745.
            zero_u, zero_v, small = host_flags(torch.stack(
                (ru == 0.0, rv == 0.0, initial < p.nonlinear_tolerance * initial_nonlinear)))
            if zero_u:
                active_u, count_u, s1 = False, 0, ru
            if zero_v:
                active_v, count_v, s2 = False, 0, rv
            do_nonlinear = do_nonlinear and (active_u or active_v)
            do_nonlinear = do_nonlinear and not (ipass > 2 and small)
            initial = torch.where(initial == 0.0, tiny, initial)
            factor = torch.minimum(one / (1.0 + torch.abs(torch.log10(initial))), cap)
            error_u, error_v = ru * factor, rv * factor
            flex = torch.stack((initial, initial_nonlinear, factor, error_u, error_v))
        control, flex = lsr_control(
            (ipass, my_iter, ipass, int(prep_ran), int(do_nonlinear), int(active_u), int(active_v),
             count_u, count_v, p.linear_iterations, int(print_residual), int(resid_gate)),
            (ru, rv, lsr_error, wu, wv, s1a, s2a), flex, u)
        checks: list[Record] = []
        scans: list[ScanState] = []
        end = None
        final_u = final_v = None
        if do_nonlinear:
            m = 1
            while m <= p.linear_iterations and (active_u or active_v):
                # Replays start on a check-interval boundary and hold no
                # requested L6(b) snapshot; other iterations run one at a time.
                stop_before = next((k for k in snapshots if k >= m), None)
                if (blocks is not None and (m - 1) % p.check_interval == 0
                        and (stop_before is None or m + BLOCK <= stop_before)):
                    out = replay_block(
                        blocks, lines=lines, g=g, u=u, v=v, relax_u=wu, relax_v=wv, s1=s1, s2=s2,
                        s1a=s1a, s2a=s2a, error_u=error_u, error_v=error_v, active_u=active_u,
                        active_v=active_v, count_u=count_u, count_v=count_v, m=m,
                        m_max=p.linear_iterations, my_iter=my_iter, ipass=ipass,
                        sequence=check_sequence, stop_before=stop_before)
                    u, v, s1, s2, s1a, s2a = out["u"], out["v"], out["s1"], out["s2"], out["s1a"], out["s2a"]
                    wu, wv = out["relax_u"], out["relax_v"]
                    active_u, active_v, count_u, count_v = out["active_u"], out["active_v"], out["count_u"], out["count_v"]
                    checks.extend(out["records"])
                    check_sequence += out["executed"]
                    m += out["executed"]
                    continue
                swept_u, swept_v = active_u, active_v
                before_u, before_v = u, v
                u, v = sweep(lines, u, v, swept_u, swept_v, wu, wv)
                (checked_u, checked_v, active_u, active_v, count_u, count_v,
                 s1, s2, s1a, s2a, wu, wv) = check_scan(
                    system, u, v, before_u, before_v, g, p, space, m=m,
                    active_u=active_u, active_v=active_v, count_u=count_u, count_v=count_v,
                    s1=s1, s2=s2, s1a=s1a, s2a=s2a, relax_u=wu, relax_v=wv,
                    error_u=error_u, error_v=error_v, print_residual=print_residual)
                check_sequence += 1
                checks.append(record(
                    (check_sequence, my_iter, ipass, m, int(swept_u), int(swept_v),
                     int(checked_u), int(checked_v), int(active_u), int(active_v), count_u, count_v),
                    (s1, s2, wu, wv), u))
                # X6 follows the check even when that check stopped both directions.
                scan = lsr_scan(before_u, before_v, u, v, space, my_iter=my_iter, ipass=ipass, m=m)
                u, v = scan.after_exchange_u, scan.after_exchange_v
                if (my_iter, ipass, m) in capture_scans:
                    scans.append(scan)
                m += 1
            if print_residual:
                final_u, final_v, _, _ = residual(system, u, v, g, space, calculate_mean=True)
            end_sequence += 1
            u, v, end = lsr_pass_end(u, v, g, sequence=end_sequence, my_iter=my_iter,
                                     ipass=ipass, count_u=count_u, count_v=count_v,
                                     active_u=active_u, active_v=active_v)
        passes.append(PassState(my_iter, ipass, lin, system, control, flex, tuple(checks), tuple(scans), end, u, v, final_u, final_v))
    if replayed:
        # Replayed passes publish views of their record buffers; the step's
        # outputs are standalone tensors, as on the per-call path.
        u, v = u.clone(), v.clone()
        carry = DynamicsCarry(*(getattr(carry, f.name).clone() for f in fields(carry)))
    return LsrResult(u, v, carry, tuple(passes))
