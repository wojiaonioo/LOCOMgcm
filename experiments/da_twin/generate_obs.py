"""Synthetic SSH, SST and profile observations at grid points."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
(SEED_POS, SEED_NOISE) = (2026101801, 2026101802)
(DT, SIX) = (600.0, 36)
ASSIM_STEPS = list(range(0, 5 * 144 + 1, SIX))
EVAL_STEPS = list(range(0, 8 * 144 + 1, SIX))
EDGE = 7
SIGMA = dict(ssh=0.03, sst=0.4, tprof=0.1, sprof=0.02)
KIND_CODE = dict(ssh=0, sst=1, tprof=2, sprof=3)
FIELD = dict(ssh='eta', sst='temp', tprof='temp', sprof='salt')
VAR = dict(eta=0, temp=1, salt=2)
ROLE_CODE = dict(assim=0, eval=1)
TRACKS = (2, 4, 8)
SST_FRAC = (0.1, 0.25, 0.5)
(EVAL_TRACKS, EVAL_SST_FRAC) = (2, 0.1)
BLOCK = 8
(TRACK_SLOPE, TRACK_JC) = (0.5, 76)
PROF_MAX_DEPTH = 1000.0
(SHELF, SLOPE) = (200.0, 2000.0)
STATIONS = [('assim', 0, (3, 4, 5)), ('assim', 1, (3, 4, 5)), ('assim', 2, (6, 8, 10)), ('eval', -1, (3, 4, 5))]
MIN_SEP = 8
CORE_RULE = f'wet cells with {EDGE} <= i <= nx-{EDGE + 1} and {EDGE} <= j <= ny-{EDGE + 1}: outside the open-boundary cell (index 0), the LOCOMgcm mitgcm-mode sponge (1..3, tracer_sponge.py _band_slice) and 3 more cells'

def noise_seed(role, kind, n, obj):
    return ((ROLE_CODE[role] * 8 + KIND_CODE[kind]) * 1000 + n) * 1000 + obj

def noise(seed, count, sigma):
    return np.random.default_rng([SEED_NOISE, seed]).standard_normal(count) * sigma

class Grid:

    def __init__(self, path):
        with np.load(path) as g:
            self.hfac = np.asarray(g['hFacC'], dtype=np.float64)
            self.dz = np.asarray(g['layer_thickness'], dtype=np.float64)
            self.zc = np.asarray(g['center_depth'], dtype=np.float64)
        (self.nz, self.ny, self.nx) = self.hfac.shape
        (jj, ii) = np.meshgrid(np.arange(self.ny), np.arange(self.nx), indexing='ij')
        self.core = (ii >= EDGE) & (ii <= self.nx - EDGE - 1) & (jj >= EDGE) & (jj <= self.ny - EDGE - 1)
        self.wet0 = self.hfac[0] > 0
        self.core0 = self.core & self.wet0
        self.depth = (self.hfac * self.dz[:, None, None]).sum(0)
        self.lon = 99.0 + (np.arange(self.nx) + 0.5) * 0.1875
        self.lat = 1.5 + (np.arange(self.ny) + 0.5) * 0.1875

class Rows:
    COLS = ('kind', 'step', 'i', 'j', 'k', 'role', 'track', 'station', 'block', 'noise_seed', 'level', 'obj', 'n')

    def __init__(self):
        self.parts = {c: [] for c in self.COLS}

    def add(self, count, **kw):
        for c in self.COLS:
            v = kw[c]
            self.parts[c].append(np.asarray(v) if np.ndim(v) else np.full(count, v))

    def table(self):
        return {c: np.concatenate(v) for (c, v) in self.parts.items()}

def ssh_tracks(g, rows, role, steps, n_tracks, level_of):
    for (n, step) in enumerate(steps):
        for q in range(n_tracks):
            rng = np.random.default_rng([SEED_POS, ROLE_CODE[role], KIND_CODE['ssh'], n, q])
            ic = rng.uniform(EDGE, g.nx - EDGE - 1)
            sign = 1.0 if q % 2 == 0 else -1.0
            j = np.arange(EDGE, g.ny - EDGE)
            i = np.rint(ic + sign * TRACK_SLOPE * (j - TRACK_JC)).astype(int)
            ok = (i >= EDGE) & (i <= g.nx - EDGE - 1)
            (j, i) = (j[ok], i[ok])
            keep = g.core0[j, i]
            (j, i) = (j[keep], i[keep])
            if len(j) == 0:
                continue
            rows.add(len(j), kind='ssh', step=step, i=i, j=j, k=0, role=role, track=q, station=-1, block=-1, noise_seed=noise_seed(role, 'ssh', n, q), level=level_of(q), obj=q, n=n)

def sst_blocks(g, rows):
    (nbx, nby) = (-(-g.nx // BLOCK), -(-g.ny // BLOCK))
    (jj, ii) = np.nonzero(g.core0)
    bid = jj // BLOCK * nbx + ii // BLOCK
    counts = np.bincount(bid, minlength=nbx * nby)
    total = int(g.core0.sum())
    cover = {}
    for (n, step) in enumerate(EVAL_STEPS):
        order = np.random.default_rng([SEED_POS, 9, KIND_CODE['sst'], n]).permutation(nbx * nby)
        order = order[counts[order] > 0]
        cum = np.cumsum(counts[order])
        cuts = [int(np.searchsorted(cum, f * total) + 1) for f in SST_FRAC]
        cut_e = int(np.searchsorted(cum, (SST_FRAC[-1] + EVAL_SST_FRAC) * total) + 1)
        sets = []
        if step <= ASSIM_STEPS[-1]:
            lo = 0
            for (lev, hi) in enumerate(cuts):
                sets += [('assim', b, lev) for b in order[lo:hi]]
                lo = hi
        sets += [('eval', b, -1) for b in order[cuts[-1]:cut_e]]
        assim = step <= ASSIM_STEPS[-1]
        cover[step] = dict(D0=int(cum[cuts[0] - 1]) if assim else 0, D1=int(cum[cuts[1] - 1]) if assim else 0, D2=int(cum[cuts[2] - 1]) if assim else 0, eval=int(cum[cut_e - 1] - cum[cuts[-1] - 1]), total=total)
        for (role, b, lev) in sets:
            m = bid == b
            (j, i) = (jj[m], ii[m])
            o = np.lexsort((i, j))
            (j, i) = (j[o], i[o])
            rows.add(len(j), kind='sst', step=step, i=i, j=j, k=0, role=role, track=-1, station=-1, block=int(b), noise_seed=noise_seed(role, 'sst', n, int(b)), level=lev, obj=int(b), n=n)
    return cover

def choose_stations(g):
    rng = np.random.default_rng([SEED_POS, 7])
    (jj, ii) = np.nonzero(g.core0)
    h = g.depth[jj, ii]
    classes = [h < SHELF, (h >= SHELF) & (h < SLOPE), h >= SLOPE]
    pools = []
    for c in classes:
        idx = np.nonzero(c)[0]
        pools.append(list(idx[rng.permutation(len(idx))]))
    (taken, out) = ([], [])
    for (role, lev, per_class) in STATIONS:
        for (cls, want) in enumerate(per_class):
            got = 0
            while got < want:
                if not pools[cls]:
                    raise SystemExit(f'class {cls}: not enough separated columns')
                p = pools[cls].pop(0)
                (j, i) = (int(jj[p]), int(ii[p]))
                if all((max(abs(j - a), abs(i - b)) >= MIN_SEP for (a, b) in taken)):
                    taken.append((j, i))
                    out.append(dict(role=role, level=lev, cls=['shelf', 'slope', 'deep'][cls], j=j, i=i, depth=float(g.depth[j, i])))
                    got += 1
    sid = {'assim': 0, 'eval': 0}
    for s in out:
        s['station'] = sid[s['role']]
        sid[s['role']] += 1
    return out

def profiles(g, rows, stations):
    for s in stations:
        k = np.nonzero((g.hfac[:, s['j'], s['i']] > 0) & (g.zc <= PROF_MAX_DEPTH))[0]
        s['levels'] = int(len(k))
        steps = ASSIM_STEPS if s['role'] == 'assim' else EVAL_STEPS
        for (n, step) in enumerate(steps):
            for kind in ('tprof', 'sprof'):
                rows.add(len(k), kind=kind, step=step, i=s['i'], j=s['j'], k=k, role=s['role'], track=-1, station=s['station'], block=-1, noise_seed=noise_seed(s['role'], kind, n, s['station']), level=s['level'], obj=s['station'], n=n)

def generate(grid_path, truth_dir, out_dir, t0):
    grid_path, truth_dir, out_dir = map(Path, (grid_path, truth_dir, out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    g = Grid(grid_path)
    rows = Rows()
    ssh_tracks(g, rows, 'assim', ASSIM_STEPS, TRACKS[-1], lambda q: int(np.searchsorted(TRACKS, q + 1)))
    ssh_tracks(g, rows, 'eval', EVAL_STEPS, EVAL_TRACKS, lambda q: -1)
    cover = sst_blocks(g, rows)
    stations = choose_stations(g)
    profiles(g, rows, stations)
    t = rows.table()
    order = np.lexsort((t['k'], t['obj'], np.array([KIND_CODE[x] for x in t['kind']]), t['step'], np.array([ROLE_CODE[x] for x in t['role']])))
    t = {c: v[order] for (c, v) in t.items()}
    nrow = len(order)
    t['obs_id'] = np.arange(nrow, dtype=np.int64)
    t['field'] = np.array([FIELD[x] for x in t['kind']])
    t['var'] = np.array([VAR[x] for x in t['field']], dtype=np.int8)
    t['sigma'] = np.array([SIGMA[x] for x in t['kind']])
    t['time_s'] = t['step'].astype(np.float64) * DT
    (t['lon'], t['lat']) = (g.lon[t['i']], g.lat[t['j']])
    t['depth'] = np.where(t['kind'] == 'ssh', 0.0, g.zc[t['k']])
    tsteps = list(np.load(truth_dir / 'steps.npy'))
    arr = {'temp': np.load(truth_dir / 'T.npy', mmap_mode='r'), 'salt': np.load(truth_dir / 'S.npy', mmap_mode='r'), 'eta': np.load(truth_dir / 'Eta.npy', mmap_mode='r')}
    truth = np.empty(nrow)
    for step in np.unique(t['step']):
        n = tsteps.index(int(step))
        for (f, a) in arr.items():
            m = (t['step'] == step) & (t['field'] == f)
            if not m.any():
                continue
            snap = np.asarray(a[n], dtype=np.float64)
            truth[m] = snap[t['j'][m], t['i'][m]] if f == 'eta' else snap[t['k'][m], t['j'][m], t['i'][m]]
    if not np.isfinite(truth).all():
        raise SystemExit('non-finite truth')
    wet = np.where(t['kind'] == 'ssh', g.wet0[t['j'], t['i']], g.hfac[t['k'], t['j'], t['i']] > 0)
    if not wet.all() or not g.core[t['j'], t['i']].all():
        raise SystemExit('an observation outside the wet core')
    nz_ = np.empty(nrow)
    by_seed = np.argsort(t['noise_seed'], kind='stable')
    cuts = np.nonzero(np.diff(t['noise_seed'][by_seed]))[0] + 1
    for m in np.split(by_seed, cuts):
        nz_[m] = noise(int(t['noise_seed'][m[0]]), len(m), float(t['sigma'][m[0]]))
    (t['truth'], t['noise'], t['value']) = (truth, nz_, truth + nz_)
    cols = dict(obs_id=np.int64, kind='U5', field='U4', var=np.int8, step=np.int32, time_s=np.float64, i=np.int16, j=np.int16, k=np.int16, lon=np.float64, lat=np.float64, depth=np.float64, value=np.float64, sigma=np.float64, truth=np.float64, noise=np.float64, noise_seed=np.int64, role='U5', track=np.int32, station=np.int32, block=np.int32)
    scal = dict(t0_utc=t0, dt_s=DT, nx=g.nx, ny=g.ny, nz=g.nz, core_rule=CORE_RULE, truth_dir=str(truth_dir), format_version=1, seed_base=SEED_NOISE)
    summary = dict(rows_total=nrow, stations=stations, sst_cover_cells=cover, tiers={})
    for tier in range(3):
        m = (t['role'] == 'assim') & (t['level'] <= tier) | (t['role'] == 'eval')
        data = {c: t[c][m].astype(dt) for (c, dt) in cols.items()}
        np.savez_compressed(out_dir / f'D{tier}.npz', tier=f'D{tier}', **scal, **data)
        rec = {}
        for role in ('assim', 'eval'):
            for kind in KIND_CODE:
                mm = (data['role'] == role) & (data['kind'] == kind)
                if mm.any():
                    z = data['noise'][mm] / data['sigma'][mm]
                    rec[f'{role}_{kind}'] = dict(rows=int(mm.sum()), times=int(len(np.unique(data['step'][mm]))), noise_sd_over_sigma=float(z.std()), noise_mean_over_sigma=float(z.mean()))
        summary['tiers'][f'D{tier}'] = rec
    d = [np.load(out_dir / f'D{n}.npz') for n in range(3)]
    for (lo, hi) in ((0, 1), (1, 2)):
        (a, b) = (d[lo], d[hi])
        ida = a['obs_id'][a['role'] == 'assim']
        pos = np.searchsorted(b['obs_id'], ida)
        same = all((np.array_equal(a[c][a['role'] == 'assim'], b[c][pos]) for c in ('value', 'noise', 'i', 'j', 'k', 'step')))
        summary[f'nested_D{lo}_in_D{hi}'] = bool(same and np.array_equal(b['obs_id'][pos], ida))
    ev = [x['obs_id'][x['role'] == 'eval'] for x in d]
    summary['eval_identical_across_tiers'] = bool(all((np.array_equal(ev[0], e) for e in ev[1:])))
    (out_dir / 'SUMMARY.json').write_text(json.dumps(summary, indent=1, default=int) + '\n')
    print('OBSGEN', json.dumps({k: v for (k, v) in summary.items() if k != 'sst_cover_cells' and k != 'stations'}, default=int), flush=True)
