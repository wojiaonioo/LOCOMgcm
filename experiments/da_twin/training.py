"""Online normalized-cost training, curriculum and EMA model selection."""
from __future__ import annotations
import datetime
import json
import math
import os
from pathlib import Path
import random
import time
import traceback
import numpy as np
import torch
import torch.distributed as dist
from . import network as n2
from .features import FIELDS, Grid, Normalisation, background_channels, frame_channels

class Dist:

    def __init__(self):
        if int(os.environ.get('WORLD_SIZE', '1')) != 1:
            raise ValueError('Run training in one process on one GPU')
        self.world, self.rank, self.on = 1, 0, False

    def sum_(self, t):
        if self.on:
            dist.all_reduce(t)
        return t

    def gather(self, obj):
        if not self.on:
            return [obj]
        out = [None] * self.world
        dist.all_gather_object(out, obj)
        return out

    def barrier(self):
        if self.on:
            dist.barrier()

    def close(self):
        if self.on:
            dist.destroy_process_group()

def parse_stages(text):
    out = []
    for item in text.split(','):
        (s, c) = item.split(':')
        out.append((int(s), int(c)))
    if [s for (s, _) in out] != sorted({s for (s, _) in out}):
        raise ValueError(f'stages must grow: {text}')
    return out

def atomic_save(obj, path):
    tmp = Path(str(path) + '.partial')
    torch.save(obj, tmp)
    os.replace(tmp, path)

def jdefault(o):
    if hasattr(o, 'tolist'):
        return o.tolist()
    return str(o)

def epoch_order(seed, stage, epoch, names):
    rng = np.random.default_rng([seed, stage, epoch])
    return [names[i] for i in rng.permutation(len(names))]

def stage_rows(w, steps):
    sel = w.rows['step'] <= steps
    return ({k: v[sel] for (k, v) in w.rows.items()}, w.innovation[sel], sel)

def j0_at(w, steps):
    (_, inn, sel) = stage_rows(w, steps)
    return (float(0.5 * np.square(inn / w.rows['sigma'][sel]).sum()), int(sel.sum()))

def segment_rms(C, g):
    return {n: float(np.sqrt(np.mean(np.square(g[C.offsets[i]:C.offsets[i + 1]])))) for (i, n) in enumerate(FIELDS)}

class Tree:

    def __init__(self, record):
        self.text = json.dumps(record, sort_keys=True)

def g0_path(out, name, steps):
    return Path(out) / 'g0' / f'{name}_s{steps}.npz'

def compute_g0(w, steps, path, tree):
    t0 = time.perf_counter()
    (J, g, row) = w.prob.value_grad(np.zeros(w.C.size), steps)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + '.partial.npz')
    np.savez(tmp, g=g.astype(np.float32), J=J, steps=steps, tree=tree.text)
    os.replace(tmp, path)
    return dict(window=w.name, steps=steps, J=J, wall_s=time.perf_counter() - t0, kpp_solver=row.get('kpp_solver'))

def load_g0(path, tree):
    with np.load(path, allow_pickle=False) as z:
        if str(z['tree']) != tree.text:
            return None
        return (np.asarray(z['g'], np.float64), float(z['J']))

def flat_grads(params):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in params])

def set_grads(params, flat):
    i = 0
    for p in params:
        n = p.numel()
        p.grad = flat[i:i + n].view_as(p).clone()
        i += n

class Engine:

    def __init__(self, args, wins, train_names, val_names, report_names, grid, static, tree, dd, device, log):
        (self.args, self.wins, self.grid, self.static, self.tree, self.dd, self.device) = (args, wins, grid, static, tree, dd, device)
        (self.train_names, self.val_names, self.report_names) = (train_names, val_names, report_names)
        self.log = log
        self.out = Path(args.out)
        self.cache = Path(getattr(args, 'cache_dir', None) or args.out)
        self.names = list(wins)
        self.C = wins[self.names[0]].C
        (self.inputs, self.stage_info) = ({}, {})

    def owner(self, name):
        return self.names.index(name) % self.dd.world

    def backgrounds(self):
        cache = self.cache / 'background'
        recs = {}
        for name in self.names:
            if self.owner(name) == self.dd.rank:
                recs[name] = self.wins[name].background(cache)
        self.dd.barrier()
        for name in self.names:
            if self.owner(name) != self.dd.rank:
                recs[name] = self.wins[name].background(cache)
                if 'cached' not in recs[name]:
                    raise RuntimeError(f"{name}: background not taken from the owner's cache ({recs[name]})")
        norm_path = self.out / 'norm.json'
        if norm_path.exists():
            self.norm = Normalisation.from_json(norm_path)
        else:
            self.norm = Normalisation.from_backgrounds(self.grid, [{n: self.wins[k].prob.xb[n].numpy() for n in FIELDS} for k in self.train_names])
            self.dd.barrier()
            if self.dd.rank == 0:
                self.norm.to_json(norm_path, dict(windows=self.train_names))
        self.background_in = {k: background_channels(self.grid, self.norm, {n: self.wins[k].prob.xb[n].numpy() for n in FIELDS}) for k in self.names}
        return recs

    def prepare_stage(self, steps):
        recs = []
        for name in self.names:
            p = g0_path(self.cache, name, steps)
            if self.owner(name) == self.dd.rank and (not p.exists() or load_g0(p, self.tree) is None):
                recs.append(compute_g0(self.wins[name], steps, p, self.tree))
                self.log('G0', recs[-1])
        self.dd.barrier()
        (g0, info) = ({}, {})
        for name in self.names:
            got = load_g0(g0_path(self.cache, name, steps), self.tree)
            if got is None:
                raise RuntimeError(f'{name}: g0 at {steps} steps missing or from another model tree')
            g0[name] = got[0]
            (j0, m) = j0_at(self.wins[name], steps)
            info[name] = dict(J0=j0, m=m, J0_value_grad=got[1], J0_relative_difference=abs(got[1] - j0) / j0)
        scale = {n: float(np.median([segment_rms(self.C, g0[k])[n] for k in self.train_names])) for n in FIELDS}
        scale = {n: s if s > 0 else 1.0 for (n, s) in scale.items()}
        self.inputs = {}
        for name in self.names:
            w = self.wins[name]
            (rows, inn, _) = stage_rows(w, steps)
            frames = frame_channels(self.grid, rows, inn, steps, w.frame_step)
            frames = torch.cat([frames, n2.frame_extras(frames, w.frame_step, w.dt, steps)], 1)
            g = self.C.unpack(torch.as_tensor(g0[name]))
            self.inputs[name] = n2.Inputs2(self.background_in[name], n2.field_channels(self.grid, g, scale), frames)
        self.stage_info = dict(steps=steps, g0_scale=scale, windows=info, computed=recs)
        return self.stage_info

    def window_update(self, net, vmap, name, steps):
        (w, info) = (self.wins[name], self.stage_info['windows'][name])
        t0 = time.perf_counter()
        raw = net(self.inputs[name].to(self.device), self.static)
        (v, handle) = vmap.control(raw)
        t1 = time.perf_counter()
        (J, g, row) = w.prob.value_grad(v, steps)
        t2 = time.perf_counter()
        vmap.backward(handle, g, scale=1.0 / info['J0'])
        kpp = row.get('kpp_solver') or {}
        return dict(window=name, ok=True, J=J, J_per_obs=J / info['m'], loss=J / info['J0'], v_l2=float(np.linalg.norm(v)), v_share=n2.v_share(self.C, v), network_s=t1 - t0, value_grad_s=t2 - t1, backward_s=time.perf_counter() - t2, zero_v=bool(not np.any(v)), J_equals_g0_J=J == info['J0_value_grad'] if not np.any(v) else None, recovered_columns=kpp.get('recovered_columns'), dense_fallback_columns=kpp.get('dense_fallback_columns'), dense_max_relative_residual=kpp.get('dense_max_relative_residual'))

    def window_value(self, net, vmap, name, steps):
        (w, info) = (self.wins[name], self.stage_info['windows'][name])
        t0 = time.perf_counter()
        with torch.no_grad():
            (v, _) = vmap.control(net(self.inputs[name].to(self.device), self.static))
        (J, rec) = w.prob.value(v, steps)
        return dict(window=name, ok=True, J=J, J_per_obs=J / info['m'], loss=J / info['J0'], v_l2=float(np.linalg.norm(v)), v_share=n2.v_share(self.C, v), wall_s=time.perf_counter() - t0)

def learning_rate(cur, args):
    peak = args.lr * 0.5 ** cur['halvings'] * 0.5 ** cur['clip_halved']
    warm = min(1.0, (cur['updates'] + 1) / args.warmup) if args.warmup else 1.0
    k = cur['updates'] - cur['stage_start_update']
    rewarm = 1.0 if cur['stage'] == 0 else min(1.0, 0.3 + 0.7 * k / args.rewarm)
    return peak * warm * rewarm

def clip_bound(norms):
    if not norms:
        return math.inf
    if len(norms) < 20:
        return 10.0 * norms[0]
    return 3.0 * float(np.median(norms[-20:]))

def new_curriculum():
    return dict(stage=0, epoch=0, update_in_epoch=0, updates=0, attempted=0, skipped=0, stage_start_update=0, halvings=0, clip_halved=0, clipped_in_stage=0, updates_in_stage=0, norms=[], best={}, stall=0, val_history={}, stage_entries=[], failures=dict(total=0, by_epoch={}, by_window={}), done=False, status=None)

def train(eng, net, args, log):
    (dd, out) = (eng.dd, Path(args.out))
    stages = parse_stages(args.stages)
    params = list(net.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), eps=1e-06, weight_decay=0.0001)
    ema = [p.detach().clone() for p in params]
    vmap = n2.VMap(eng.C, eng.device, space=args.output_space, uv=not args.no_uv)
    state_path = out / 'state.pt'
    cur = new_curriculum()
    t_start = time.perf_counter()
    if args.resume and state_path.exists():
        st = torch.load(state_path, map_location=eng.device, weights_only=False)
        net.load_state_dict(st['net'])
        opt.load_state_dict(st['opt'])
        for (e, s) in zip(ema, st['ema']):
            e.copy_(s)
        cur = st['curriculum']
        cur['done'] = False
        rngs = st.get('rng') or []
        if len(rngs) == dd.world:
            r = rngs[dd.rank]
            torch.set_rng_state(r['torch'].cpu())
            np.random.set_state(r['numpy'])
            random.setstate(r['python'])
        log('RESUME', dict(stage=cur['stage'], epoch=cur['epoch'], update_in_epoch=cur['update_in_epoch'], updates=cur['updates'], world=dd.world, saved_world=len(rngs)))
    elif getattr(args, 'init_net', None):
        st = torch.load(args.init_net, map_location=eng.device, weights_only=False)
        net.load_state_dict(st['net'])
        for (e, p) in zip(ema, params):
            e.copy_(p.detach())
        log('INIT_NET', dict(path=str(args.init_net), stage=st.get('stage'), epoch=st.get('epoch'), val_J_per_obs=st.get('val_J_per_obs')))
    planned = sum((c for (_, c) in stages)) * (len(eng.train_names) // args.batch) * args.batch

    def rng_record():
        return dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate())

    def save_state():
        rngs = dd.gather(rng_record())
        if dd.rank == 0:
            atomic_save(dict(net=net.state_dict(), opt=opt.state_dict(), ema=[e.cpu() for e in ema], curriculum=cur, rng=rngs, args=vars(args), stage_info=eng.stage_info, world=dd.world), state_path)

    def ema_net():
        saved = [p.detach().clone() for p in params]
        with torch.no_grad():
            for (p, e) in zip(params, ema):
                p.copy_(e)
        return saved

    def restore(saved):
        with torch.no_grad():
            for (p, s) in zip(params, saved):
                p.copy_(s)

    def progress(rec):
        if dd.rank == 0:
            with open(out / 'progress.jsonl', 'a') as f:
                f.write(json.dumps(rec, default=jdefault) + '\n')

    def time_up():
        flag = torch.tensor([1.0 if time.perf_counter() - t_start > args.time_limit_s else 0.0], device=eng.device)
        return bool(dd.sum_(flag).item() > 0)
    status = None
    durations = cur.setdefault('epoch_seconds', {})

    def epoch_estimate(steps):
        if durations.get(str(steps)):
            return durations[str(steps)][-1]
        known = [(int(k), v[-1]) for (k, v) in durations.items() if v]
        if not known:
            return None
        (k, d) = max(known)
        return d * steps / k

    def past_switch(steps, entering):
        if not args.switch_by:
            return False
        est = epoch_estimate(steps)
        return est is not None and time.time() + est * (2 if entering else 1) > args.switch_by

    def advance_stage(why):
        log('STAGE_ADVANCE', dict(stage=cur['stage'], steps=stages[cur['stage']][0], epoch=cur['epoch'], why=why))
        cur['stage'] += 1
        cur['epoch'] = cur['update_in_epoch'] = 0
        cur['clip_halved'] = 0
        save_state()
    while cur['stage'] < len(stages):
        (steps, cap) = stages[cur['stage']]
        final = cur['stage'] == len(stages) - 1
        if args.until_steps and steps > args.until_steps:
            status = 'paused_before_stage'
            break
        if not final and cur['update_in_epoch'] == 0 and (cur['epoch'] >= cap or past_switch(steps, entering=cur['epoch'] == 0)):
            advance_stage('cap' if cur['epoch'] >= cap else 'switch_by')
            continue
        if final and cur['epoch'] >= cap:
            status = 'cap'
            break
        if cur['epoch'] == 0 and cur['update_in_epoch'] == 0 and (not any((e['stage'] == cur['stage'] for e in cur['stage_entries']))):
            cur['stage_start_update'] = cur['updates']
            cur['clipped_in_stage'] = cur['updates_in_stage'] = cur['stall'] = 0
            cur['stage_entries'].append(dict(stage=cur['stage'], steps=steps, update=cur['updates'], time=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
        t0 = time.perf_counter()
        info = eng.prepare_stage(steps)
        log('STAGE', dict(stage=cur['stage'], steps=steps, cap=cap, final=final, prepare_s=time.perf_counter() - t0, g0_scale=info['g0_scale'], computed=len(info['computed']), J0_relative_difference_max=max((v['J0_relative_difference'] for v in info['windows'].values()))))
        progress(dict(kind='stage', stage=cur['stage'], steps=steps, cap=cap, info=info))
        n_updates = len(eng.train_names) // args.batch
        while cur['epoch'] < cap:
            t_mark = time.time()
            order = epoch_order(args.seed, cur['stage'], cur['epoch'], eng.train_names)
            ep_key = f"{cur['stage']}:{cur['epoch']}"
            while cur['update_in_epoch'] < n_updates:
                u = cur['update_in_epoch']
                batch = order[u * args.batch:(u + 1) * args.batch]
                mine = [name for (j, name) in enumerate(batch) if j % dd.world == dd.rank]
                opt.zero_grad(set_to_none=False)
                for p in params:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                recs = []
                for name in mine:
                    try:
                        recs.append(eng.window_update(net, vmap, name, steps))
                    except args.trial_errors as e:
                        recs.append(dict(window=name, ok=False, error=f'{type(e).__name__}: {e}'[:500]))
                ok = sum((1 for r in recs if r['ok']))
                buf = torch.cat([flat_grads(params), torch.tensor([float(ok)], device=eng.device)])
                dd.sum_(buf)
                n_ok = int(round(float(buf[-1])))
                allrecs = [r for rs in dd.gather(recs) for r in rs]
                lr = learning_rate(cur, args)
                upd = dict(kind='update', stage=cur['stage'], steps=steps, epoch=cur['epoch'], update=u, global_update=cur['updates'], batch=batch, successes=n_ok, lr=lr, windows=allrecs)
                cur['attempted'] += 1
                for r in allrecs:
                    if not r['ok']:
                        f = cur['failures']
                        f['total'] += 1
                        f['by_epoch'][ep_key] = f['by_epoch'].get(ep_key, 0) + 1
                        f['by_window'][r['window']] = f['by_window'].get(r['window'], 0) + 1
                if n_ok >= min(args.min_success, args.batch):
                    flat = buf[:-1] / n_ok
                    norm = float(flat.double().norm())
                    bound = clip_bound(cur['norms'])
                    clipped = norm > bound
                    cur['norms'].append(norm)
                    if clipped:
                        flat = flat * (bound / norm)
                        cur['clipped_in_stage'] += 1
                    set_grads(params, flat)
                    for g in opt.param_groups:
                        g['lr'] = lr
                    opt.step()
                    with torch.no_grad():
                        for (e, p) in zip(ema, params):
                            e.mul_(args.ema).add_(p.detach(), alpha=1.0 - args.ema)
                    cur['updates'] += 1
                    cur['updates_in_stage'] += 1
                    upd.update(grad_norm=norm, clip_bound=bound, clipped=clipped, loss=float(np.mean([r['loss'] for r in allrecs if r['ok']])), J_per_obs=float(np.mean([r['J_per_obs'] for r in allrecs if r['ok']])))
                else:
                    cur['skipped'] += 1
                    upd.update(skipped=True)
                cur['update_in_epoch'] += 1
                progress(upd)
                log('UPDATE', {k: upd.get(k) for k in ('stage', 'steps', 'epoch', 'update', 'global_update', 'successes', 'lr', 'loss', 'J_per_obs', 'grad_norm', 'clipped', 'skipped')})
                fails_epoch = cur['failures']['by_epoch'].get(ep_key, 0)
                if fails_epoch > args.max_failures_per_epoch or cur['failures']['total'] > args.max_failure_fraction * planned:
                    status = 'failure_limit'
                cur['epoch_elapsed'] = cur.get('epoch_elapsed', 0.0) + time.time() - t_mark
                t_mark = time.time()
                save_state()
                if args.max_updates and cur['updates'] >= args.max_updates:
                    status = 'smoke_complete'
                if status or time_up():
                    status = status or 'time_limit'
                    break
            if status:
                break
            saved = ema_net()
            mine = [k for k in eng.val_names + eng.report_names if eng.owner(k) == dd.rank]
            recs = []
            for name in mine:
                try:
                    recs.append(eng.window_value(net, vmap, name, steps))
                except args.trial_errors as e:
                    recs.append(dict(window=name, ok=False, error=f'{type(e).__name__}: {e}'[:500]))
            allrecs = {r['window']: r for rs in dd.gather(recs) for r in rs}
            sel = [allrecs[k] for k in eng.val_names]
            failed = [r['window'] for r in sel if not r['ok']]
            val = None if failed else float(np.mean([r['J_per_obs'] for r in sel]))
            hist = cur['val_history'].setdefault(str(cur['stage']), [])
            hist.append(val)
            best = cur['best'].get(str(cur['stage']))
            improved = False
            if val is not None:
                if best is None or val < best['J_per_obs'] * (1.0 - args.plateau_rel):
                    cur['stall'] = 0
                else:
                    cur['stall'] += 1
                if best is None or val < best['J_per_obs']:
                    improved = True
                    cur['best'][str(cur['stage'])] = dict(J_per_obs=val, epoch=cur['epoch'], update=cur['updates'])
                    if dd.rank == 0:
                        atomic_save(dict(net=net.state_dict(), stage=cur['stage'], steps=steps, epoch=cur['epoch'], val_J_per_obs=val, args=vars(args), g0_scale=eng.stage_info['g0_scale']), out / f'best_s{steps}.pt')
            if val is None:
                f = cur['failures']
                f['total'] += len(failed)
                f['by_epoch'][ep_key] = f['by_epoch'].get(ep_key, 0) + len(failed)
            restore(saved)
            train_recs = []
            ep_updates = []
            if dd.rank == 0 and (out / 'progress.jsonl').exists():
                for line in (out / 'progress.jsonl').read_text().splitlines():
                    r = json.loads(line)
                    if r.get('kind') == 'update' and r['stage'] == cur['stage'] and (r['epoch'] == cur['epoch']) and (not r.get('skipped')):
                        ep_updates.append(r)
                ep_updates = list({r['update']: r for r in ep_updates}.values())
                train_recs = [w for r in ep_updates for w in r['windows'] if w['ok']]
            share = cur['clipped_in_stage'] / max(cur['updates_in_stage'], 1)
            vrec = dict(kind='validation', stage=cur['stage'], steps=steps, epoch=cur['epoch'], updates=cur['updates'], val_J_per_obs=val, failed=failed, improved=improved, stall=cur['stall'], windows=allrecs, clipped_share=share, val_v_share_uv={k: r['v_share']['u'] + r['v_share']['v'] for (k, r) in allrecs.items() if r['ok']}, train_loss=float(np.mean([r['loss'] for r in train_recs])) if train_recs else None, train_J_per_obs=float(np.mean([r['J_per_obs'] for r in train_recs])) if train_recs else None)
            progress(vrec)
            log('VALIDATION', {k: vrec[k] for k in ('stage', 'steps', 'epoch', 'updates', 'val_J_per_obs', 'failed', 'improved', 'stall', 'clipped_share', 'train_loss', 'train_J_per_obs', 'val_v_share_uv')})
            if share > 0.3 and cur['clip_halved'] == 0 and (cur['updates_in_stage'] >= 5):
                cur['clip_halved'] = 1
                log('CLIP_HALVING', dict(stage=cur['stage'], share=share))
            cur['epoch'] += 1
            cur['update_in_epoch'] = 0
            durations.setdefault(str(steps), []).append(cur.get('epoch_elapsed', 0.0) + time.time() - t_mark)
            cur['epoch_elapsed'] = 0.0
            plateau = cur['stall'] >= args.plateau_n and cur['epoch'] >= args.min_epochs
            advance = False
            if final and plateau:
                if cur['halvings'] < args.halvings:
                    cur['halvings'] += 1
                    cur['stall'] = 0
                    log('LR_HALVING', dict(stage=cur['stage'], halvings=cur['halvings']))
                else:
                    status = 'converged'
            elif plateau:
                advance = True
            if cur['epoch'] >= cap and (not status):
                if final:
                    status = 'cap'
                else:
                    advance = True
            if not final and (not advance) and past_switch(steps, entering=False):
                log('STAGE_ADVANCE', dict(stage=cur['stage'], steps=steps, epoch=cur['epoch'], why='switch_by'))
                advance = True
            if advance:
                cur['stage'] += 1
                cur['epoch'] = cur['update_in_epoch'] = 0
                cur['clip_halved'] = 0
            save_state()
            if status:
                break
            if time_up():
                status = 'time_limit'
                break
            if advance:
                break
        if status:
            break
    if status is None:
        status = 'completed'
    if status in ('converged', 'cap', 'completed'):
        cur['done'] = True
        if dd.rank == 0:
            last = stages[min(cur['stage'], len(stages) - 1)][0]
            src = out / f'best_s{last}.pt'
            if src.exists():
                atomic_save(torch.load(src, map_location='cpu', weights_only=False), out / 'best.pt')
    cur['status'] = status
    save_state()
    return (status, cur)
