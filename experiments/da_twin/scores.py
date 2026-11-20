"""Volume and area weighted forecast scores against MITgcm truth."""
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
EDGE = 7
VARS = ('T', 'S', 'u', 'v', 'eta')
TO_NAME = dict(T='temp', S='salt', u='u', v='v', eta='eta')
TO_NAME_INV = {value: key for key, value in TO_NAME.items()}
MIT_FILE = dict(T='T', S='S', u='U', v='V', eta='Eta')
LAYERS = (('upper1000', 0.0, 1000.0), ('0-200', 0.0, 200.0), ('200-1000', 200.0, 1000.0), ('below1000', 1000.0, 1000000000.0))

class Weights:

    def __init__(self, grid_npz):
        with np.load(grid_npz) as g:
            (hc, hw, hs) = (np.asarray(g[k], np.float64) for k in ('hFacC', 'hFacW', 'hFacS'))
            drf = np.asarray(g['layer_thickness'], np.float64)[:, None, None]
            zc = np.asarray(g['center_depth'], np.float64)
            (ac, au, av) = (np.asarray(g[k], np.float64) for k in ('cell_area', 'face_area_u', 'face_area_v'))
        (nz, ny, nx) = hc.shape
        (self.nz, self.ny, self.nx) = (nz, ny, nx)
        (jj, ii) = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
        core = (ii >= EDGE) & (ii <= nx - EDGE - 1) & (jj >= EDGE) & (jj <= ny - EDGE - 1)
        vol = dict(T=hc * drf * ac, S=hc * drf * ac, u=hw[..., :nx] * drf * au[:, :nx], v=hs[:, :ny] * drf * av[:ny])
        self.w = {}
        for n in ('T', 'S', 'u', 'v'):
            for (lab, lo, hi) in LAYERS:
                band = ((zc >= lo) & (zc < hi))[:, None, None]
                self.w[n, lab] = np.where(core[None] & band & (vol[n] > 0), vol[n], 0.0)
        self.w['eta', 'surface'] = np.where(core & (hc[0] > 0), ac, 0.0)

    def mse(self, to_fields, mit_fields):
        out = {}
        for ((n, lab), w) in self.w.items():
            a = np.asarray(to_fields[n], np.float64)
            if n == 'u':
                a = a[..., :self.nx]
            elif n == 'v':
                a = a[:, :self.ny, :]
            d = a - np.asarray(mit_fields[n], np.float64)
            out[f'{n}|{lab}'] = float((w * d * d).sum() / w.sum())
        return out

def summary(mse, scale):
    e = dict(T=mse['T|upper1000'], S=mse['S|upper1000'], U=mse['u|upper1000'] + mse['v|upper1000'], eta=mse['eta|surface'])
    q = 0.25 * sum((e[k] / scale[k] for k in e)) if scale else None
    return dict(E=e, rmse={k: float(np.sqrt(v)) for (k, v) in e.items()}, Q=q)

def mit_npz_fields(path):
    with np.load(path) as z:
        return dict(T=np.asarray(z['temp'], np.float64), S=np.asarray(z['salt'], np.float64), u=np.asarray(z['u'], np.float64), v=np.asarray(z['v'], np.float64), eta=np.asarray(z['eta'], np.float64))

def background_scale(weights, pairs):
    rows = []
    for (xb, xt) in pairs:
        (b, t) = (mit_npz_fields(xb), mit_npz_fields(xt))
        b['u'] = np.concatenate([b['u'], b['u'][..., -1:]], axis=-1)
        b['v'] = np.concatenate([b['v'], b['v'][..., -1:, :]], axis=-2)
        rows.append(summary(weights.mse(b, t), None)['E'])
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}

class EvalObs:
    FIELD = dict(eta='eta', temp='T', salt='S')

    def __init__(self, path):
        with np.load(path) as z:
            keep = np.asarray(z['role']).astype(str) == 'eval'
            self.kind = np.asarray(z['kind']).astype(str)[keep]
            self.field = np.asarray(z['field']).astype(str)[keep]
            self.step = np.asarray(z['step'])[keep].astype(np.int64)
            (self.i, self.j, self.k) = (np.asarray(z[c])[keep].astype(np.int64) for c in ('i', 'j', 'k'))
            (self.value, self.sigma, self.truth) = (np.asarray(z[c], np.float64)[keep] for c in ('value', 'sigma', 'truth'))

    def score(self, step, fields):
        out = {}
        sel = self.step == step
        for kind in sorted((str(x) for x in set(self.kind[sel]))):
            m = sel & (self.kind == kind)
            name = self.FIELD[self.field[m][0]]
            a = np.asarray(fields[name], np.float64)
            hx = a[self.j[m], self.i[m]] if name == 'eta' else a[self.k[m], self.j[m], self.i[m]]
            out[kind] = dict(n=int(m.sum()), rms_normalized=float(np.sqrt(np.mean(((hx - self.value[m]) / self.sigma[m]) ** 2))), rmse_truth=float(np.sqrt(np.mean((hx - self.truth[m]) ** 2))))
        return out

class Truth:

    def __init__(self, directory, dt=None):
        d = Path(directory)
        if dt is not None:
            readme = (d / 'README.txt').read_text() if (d / 'README.txt').exists() else ''
            found = re.search('dt (\\d+(?:\\.\\d+)?) s', readme)
            if not found or float(found.group(1)) != float(dt):
                raise ValueError(f"{d}: trajectory dt {(found.group(1) if found else 'unknown')} s, the case has {dt} s")
        self.steps = [int(s) for s in np.load(d / 'steps.npy')]
        self.arr = {n: np.load(d / f'{MIT_FILE[n]}.npy', mmap_mode='r') for n in VARS}

    def at(self, step):
        k = self.steps.index(step)
        return {n: self.arr[n][k] for n in VARS}
