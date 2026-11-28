"""Run 4D-Var, online training, network analyses and forecast evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from . import network as n2, scores, training
from .control import FIELDS
from .features import Grid, Normalisation, background_channels, frame_channels
from .model import Window, native_fields
from .optimizer import TrialFailure, lbfgs_backtracking


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=training.jdefault, allow_nan=False) + '\n')


def specs(path):
    path = Path(path).resolve()
    rows = json.loads(path.read_text())['windows']
    for row in rows:
        for key in ('case', 'run_dir', 'grid', 'background', 'truth', 'mit_truth', 'obs'):
            p = Path(row[key])
            row[key] = str(p if p.is_absolute() else path.parent / p)
    if len({r['name'] for r in rows}) != len(rows):
        raise ValueError('Window names must be unique')
    return rows


def selected(args):
    rows = specs(args.windows)
    if args.select:
        missing = set(args.select) - {r['name'] for r in rows}
        if missing:
            raise ValueError(f'Unknown windows: {sorted(missing)}')
        rows = [r for r in rows if r['name'] in args.select]
    if args.split:
        rows = [r for r in rows if r['split'] in args.split]
    if args.smoke:
        rows = rows[:1]
    if not rows:
        raise ValueError('No windows selected')
    return rows


def open_window(row, args, momentum=None):
    return Window(row, args.bstd, segment=args.segment,
                  device_bytes=int(args.device_gib * 1024**3), compiled=not args.eager,
                  momentum=momentum)


def minimize(w, args):
    out = args.out / w.name
    out.mkdir(parents=True, exist_ok=True)
    evaluations, accepted, failures = [], [], []
    last = dict(x=np.zeros(w.C.size))
    def fun(x):
        try:
            value, gradient, row = w.prob.value_grad(x, 432)
        except FloatingPointError as e:
            if not evaluations:
                raise
            raise TrialFailure(dict(error=type(e).__name__, message=str(e),
                                    evaluation=len(evaluations))) from e
        evaluations.append(row)
        last['trial_cost'] = value
        if 'cost' not in last:
            last['cost'] = value
        print(w.name, 'evaluation', len(evaluations), 'J', value, flush=True)
        return value, gradient
    def callback(x):
        value = last['trial_cost']
        decrease = (last['cost'] - value) / abs(last['cost'])
        accepted.append(dict(iteration=len(accepted) + 1, cost=value, relative_decrease=decrease))
        last.update(x=x.copy(), cost=value)
        if len(accepted) >= 2 and all(r['relative_decrease'] < .01 for r in accepted[-2:]):
            raise StopIteration
    try:
        result = lbfgs_backtracking(fun, last['x'], 2 if args.smoke else 30, callback, failures.append)
        v, message = result.x, result.message
    except TrialFailure as e:
        v, message = last['x'], str(e)
    np.savez(out / 'analysis_v.npz', v=v)
    fields, _ = w.prob.fields(v)
    np.savez(out / 'analysis_fields.npz', **{n: f.numpy() for n, f in fields.items()})
    write_json(out / 'result.json', dict(termination=message, evaluations=evaluations,
        accepted=accepted, failures=failures, smoke=args.smoke))


def static_grid(w):
    grid = Grid(w.grid_npz)
    static = n2.Static(grid, float(w.prob.cfg.grid.center_lat), float(w.prob.cfg.grid.resolution_deg))
    static.level, static.column, static.surface_wet = (
        static.level.cuda(), static.column.cuda(), static.surface_wet.cuda())
    return grid, static


def inputs(w, grid, norm, gradient, scale):
    frames = frame_channels(grid, w.rows, w.innovation, w.steps, w.frame_step)
    frames = torch.cat((frames, n2.frame_extras(frames, w.frame_step, w.dt, w.steps)), 1)
    return n2.Inputs2(background_channels(grid, norm, {n: w.prob.xb[n].numpy() for n in FIELDS}),
        n2.field_channels(grid, w.C.unpack(torch.as_tensor(gradient)), scale), frames)


def train(args):
    if args.phase == 2 and not args.smoke and not args.init_net and not args.resume:
        raise ValueError('Phase two requires --init-net or --resume')
    rows = selected(args)
    if not args.smoke:
        rows = [r for r in rows if r['split'] != 'test']
        if args.phase == 1:
            rows = [r for r in rows if r['t0'].startswith('2019')]
    wins = {r['name']: open_window(r, args) for r in rows}
    try:
        first = next(iter(wins.values()))
        grid, static = static_grid(first)
        train_names = [k for k, w in wins.items() if w.split == 'train']
        val_names = [k for k, w in wins.items() if w.split == ('report' if args.phase == 1 else 'val')]
        if args.smoke:
            train_names, val_names = list(wins), []
        elif not train_names or not val_names:
            raise ValueError('Training requires training and validation windows')
        args.stages = args.stages or ('36:8,72:6,144:4,432:15' if args.phase == 1 else '432:16')
        args.no_uv, args.output_space = True, 'v'
        args.until_steps, args.switch_by = 0, 0
        args.max_updates = 1 if args.smoke else 0
        if args.smoke:
            args.stages, args.batch, args.min_success = '432:1', 1, 1
        args.trial_errors = (FloatingPointError,)
        eng = training.Engine(args, wins, train_names, val_names, [], grid, static,
            training.Tree({'windows': rows, 'case_text': Path(rows[0]['case']).read_text()}),
            training.Dist(), 'cuda', lambda tag, rec: print(tag, json.dumps(rec, default=str), flush=True))
        eng.backgrounds()
        net = n2.AnalysisNet2().cuda()
        status, cur = training.train(eng, net, args, eng.log)
        if args.smoke:
            training.atomic_save(dict(net=net.state_dict(), args={'no_uv': True, 'output_space': 'v'},
                steps=432, g0_scale=eng.stage_info['g0_scale']), args.out / 'smoke.pt')
        write_json(args.out / 'training.json', dict(status=status, curriculum=cur,
            parameters=n2.parameter_count(net), phase=args.phase))
    finally:
        for w in wins.values():
            w.prob.close()


def analyze(w, args, saved, norm, net):
    grid, static = static_grid(w)
    _, g0, _ = w.background_and_g0()
    inp = inputs(w, grid, norm, g0, saved['g0_scale'])
    with torch.no_grad():
        v, _ = n2.VMap(w.C, 'cuda', uv=False).control(net(inp.to('cuda'), static))
    out = args.out / w.name
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / 'analysis_v.npz', v=v)
    fields, _ = w.prob.fields(v)
    np.savez(out / 'analysis_fields.npz', **{n: f.numpy() for n, f in fields.items()})
    value, _ = w.prob.value(v, 432)
    write_json(out / 'result.json', dict(J=value, J_per_obs=value / w.m, J0=w.j0))


def forecast_scores(w, fields, truth, weights, evalobs):
    by_step = {}
    def score(step, tensors):
        fs = {scores.TO_NAME_INV[n]: t.detach().cpu().reshape(w.C.shapes[n]).numpy() for n, t in tensors.items()}
        by_step[step] = dict(mse=weights.mse(fs, truth.at(step)), observations=evalobs.score(step, fs))
    score(0, fields)
    def observe(model, state):
        step = round((model.current_time_seconds - w.prob.origin) / w.dt)
        if step % 36 == 0:
            score(step, {n: getattr(state, n).data for n in FIELDS})
        return state.eta.data.sum() * 0
    w.prob.forward(fields, 1152, observe)
    return by_step


def evaluate(args):
    rows = selected(args)
    scale_rows = [r for r in specs(args.scale_windows) if r['split'] == 'train' and r['t0'].startswith('2019')]
    if len(scale_rows) != 40:
        raise ValueError(f'Q requires the 40 training windows of 2019; found {len(scale_rows)}')
    weights = scores.Weights(rows[0]['grid'])
    scale = scores.background_scale(weights, [(r['background'], r['truth']) for r in scale_rows])
    results = []
    for row in rows:
        w = open_window(row, args)
        try:
            truth = scores.Truth(row['mit_truth'], w.dt)
            evalobs = scores.EvalObs(row['obs'])
            methods = dict(background=w.prob.xb, truth_ic=native_fields(row['truth']))
            for method, directory in (('fourdvar', args.var_dir), ('network', args.network_dir)):
                with np.load(directory / w.name / 'analysis_v.npz') as z:
                    methods[method] = w.prob.fields(z['v'])[0]
            rec = {name: forecast_scores(w, fields, truth, weights, evalobs) for name, fields in methods.items()}
            tend = truth.at(432)
            rec['persistence'] = {k: dict(mse=weights.mse(tend, truth.at(k))) for k in range(432, 1153, 36)}
            write_json(args.out / (w.name + '.json'), rec)
            results.append(rec)
        finally:
            w.prob.close()
    pooled, gap = {}, {}
    for method in ('background', 'truth_ic', 'fourdvar', 'network', 'persistence'):
        pooled[method] = {}
        for step in range(432, 1153, 36):
            ms = [r[method][step]['mse'] for r in results]
            mse = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
            pooled[method][step] = scores.summary(mse, scale)
    for step in pooled['background']:
        b, f, n = (pooled[m][step]['Q'] for m in ('background', 'fourdvar', 'network'))
        gap[step] = (b - n) / (b - f) if b != f else None
    write_json(args.out / 'summary.json', dict(scale=scale, pooled=pooled, gap_closed=gap,
        window_count=len(rows), forecast_lead_days={s: (s - 432) / 144 for s in gap}))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('var', 'train', 'analyze', 'evaluate'):
        q = sub.add_parser(name)
        q.add_argument('--windows', type=Path, required=True)
        q.add_argument('--bstd', type=Path, required=True)
        q.add_argument('--out', type=Path, required=True)
        q.add_argument('--select', nargs='+')
        q.add_argument('--split', nargs='+', choices=['train', 'val', 'report', 'test'])
        q.add_argument('--segment', type=int, default=1)
        q.add_argument('--device-gib', type=float, default=32)
        q.add_argument('--eager', action='store_true')
        q.add_argument('--smoke', action='store_true')
        if name == 'train':
            q.add_argument('--phase', type=int, choices=[1, 2], default=1)
            q.add_argument('--stages')
            q.add_argument('--seed', type=int, default=20261018)
            q.add_argument('--batch', type=int, default=4)
            q.add_argument('--lr', type=float, default=1e-3)
            q.add_argument('--warmup', type=int, default=20)
            q.add_argument('--rewarm', type=int, default=10)
            q.add_argument('--ema', type=float, default=.9)
            q.add_argument('--min-success', type=int, default=2)
            q.add_argument('--plateau-rel', type=float, default=.01)
            q.add_argument('--plateau-n', type=int, default=2)
            q.add_argument('--min-epochs', type=int, default=3)
            q.add_argument('--halvings', type=int, default=2)
            q.add_argument('--max-failures-per-epoch', type=int, default=5)
            q.add_argument('--max-failure-fraction', type=float, default=.05)
            q.add_argument('--time-limit-s', type=float, default=float('inf'))
            q.add_argument('--init-net', type=Path)
            q.add_argument('--resume', action='store_true')
            q.add_argument('--cache-dir', type=Path)
        if name == 'analyze':
            q.add_argument('--checkpoint', type=Path, required=True)
            q.add_argument('--norm', type=Path, required=True)
        if name == 'evaluate':
            q.add_argument('--scale-windows', type=Path, required=True)
            q.add_argument('--var-dir', type=Path, required=True)
            q.add_argument('--network-dir', type=Path, required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == 'train':
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        train(args)
    elif args.command == 'evaluate':
        evaluate(args)
    else:
        if args.command == 'analyze':
            saved = torch.load(args.checkpoint, map_location='cuda', weights_only=False)
            net = n2.AnalysisNet2().cuda()
            net.load_state_dict(saved['net'])
            net.eval()
            norm = Normalisation.from_json(args.norm)
        for row in selected(args):
            w = open_window(row, args)
            try:
                if args.command == 'var':
                    minimize(w, args)
                else:
                    analyze(w, args, saved, norm, net)
            finally:
                w.prob.close()


if __name__ == '__main__':
    main()
