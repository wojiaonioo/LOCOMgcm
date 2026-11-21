"""Background normalization and six-hour observation frames."""
from __future__ import annotations
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .control import FIELDS, ControlTransform, load_b_std
KINDS = ('ssh', 'sst', 'tprof', 'sprof')
PROFILE_LEVELS = 30
FRAME_SECONDS = 21600.0
FRAME_CHANNELS = 2 + 2 + 2 * PROFILE_LEVELS + 2 * PROFILE_LEVELS + 1
GEOMETRY_CHANNELS = 4

class Grid:

    def __init__(self, npz):
        with np.load(npz) as z:
            self.mask = dict(temp=z['mask_center'], salt=z['mask_center'], u=z['mask_u'], v=z['mask_v'], eta=z['mask_center'][0])
            self.depth = np.asarray(z['depth_center'], np.float64)
            self.hfac = dict(c=np.asarray(z['hFacC']), w=np.asarray(z['hFacW']), s=np.asarray(z['hFacS']))
            self.drf = np.asarray(z['layer_thickness'], np.float64)
            self.zc = np.asarray(z['center_depth'], np.float64)
            self.area = dict(c=np.asarray(z['cell_area']), w=np.asarray(z['face_area_u']), s=np.asarray(z['face_area_v']))
        (self.nz, self.ny, self.nx) = self.mask['temp'].shape
        self.shapes = {n: tuple(m.shape) for (n, m) in self.mask.items()}
        (self.H, self.W) = (self.ny + 1, self.nx + 1)
        self.levels = {n: 1 if len(s) == 2 else s[0] for (n, s) in self.shapes.items()}
        self.channels = sum(self.levels.values())

def frame_step(dt):
    steps = FRAME_SECONDS / float(dt)
    if abs(steps - round(steps)) > 1e-09:
        raise ValueError(f'dt {dt} s does not divide the {FRAME_SECONDS:g} s observation cadence')
    return round(steps)

def load_rows(path, max_step, dt, role='assim'):
    with np.load(path, allow_pickle=False) as z:
        if 'dt_s' in z.files and float(z['dt_s']) != float(dt):
            raise ValueError(f"{path}: observations are on dt {float(z['dt_s'])} s, the case has {dt} s")
        keep = (z['role'] == role) & (z['step'] <= max_step)
        rows = {n: np.asarray(z[n][keep]) for n in ('kind', 'step', 'i', 'j', 'k', 'value', 'sigma', 'obs_id', 'time_s')}
    rows['index'] = np.nonzero(keep)[0]
    if not np.allclose(rows['time_s'], rows['step'] * float(dt), rtol=0, atol=1e-06):
        raise ValueError(f'{path}: time_s differs from step * {dt}')
    fs = frame_step(dt)
    if (rows['step'] % fs).any():
        raise ValueError(f'{path}: an observation step is not a multiple of {fs}')
    prof = np.isin(rows['kind'], ('tprof', 'sprof'))
    if (rows['k'][prof] >= PROFILE_LEVELS).any():
        raise ValueError(f'{path}: profile level beyond {PROFILE_LEVELS - 1}')
    return rows

class Normalisation:

    def __init__(self, mean, std):
        (self.mean, self.std) = (mean, std)

    @classmethod
    def from_backgrounds(cls, grid, backgrounds, floor=1e-06):
        (mean, std) = ({}, {})
        for n in FIELDS:
            m = grid.mask[n].reshape(grid.levels[n], -1)
            x = np.stack([np.asarray(b[n], np.float64).reshape(grid.levels[n], -1) for b in backgrounds])
            (mu, sd) = ([], [])
            for lev in range(grid.levels[n]):
                vals = x[:, lev, m[lev]]
                mu.append(float(vals.mean()) if vals.size else 0.0)
                sd.append(max(float(vals.std()) if vals.size else 1.0, floor))
            (mean[n], std[n]) = (mu, sd)
        return cls(mean, std)

    def to_json(self, path, extra=None):
        Path(path).write_text(json.dumps(dict(mean=self.mean, std=self.std, **extra or {}), indent=1) + '\n')

    @classmethod
    def from_json(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(d['mean'], d['std'])

def pad(x, H, W):
    return F.pad(x, (0, W - x.shape[-1], 0, H - x.shape[-2]))

def background_channels(grid, norm, xb):
    out = []
    for n in FIELDS:
        x = torch.as_tensor(np.asarray(xb[n], np.float64)).reshape(grid.levels[n], *grid.shapes[n][-2:])
        mu = torch.as_tensor(norm.mean[n], dtype=torch.float64)[:, None, None]
        sd = torch.as_tensor(norm.std[n], dtype=torch.float64)[:, None, None]
        m = torch.as_tensor(grid.mask[n]).reshape(x.shape)
        out.append(pad(torch.where(m, (x - mu) / sd, 0.0), grid.H, grid.W))
    return torch.cat(out).float()

def frame_channels(grid, rows, innovation, max_step, fstep):
    frames = max_step // fstep + 1
    val = np.zeros((frames, FRAME_CHANNELS - 1, grid.H, grid.W))
    cnt = np.zeros_like(val)
    base = dict(ssh=0, sst=2, tprof=4, sprof=4 + 2 * PROFILE_LEVELS)
    f = rows['step'] // fstep
    w = innovation / rows['sigma']
    for kd in KINDS:
        sel = rows['kind'] == kd
        if not sel.any():
            continue
        lev = rows['k'][sel] if kd in ('tprof', 'sprof') else np.zeros(int(sel.sum()), np.int64)
        span = PROFILE_LEVELS if kd in ('tprof', 'sprof') else 1
        ch_val = base[kd] + lev
        ch_cnt = base[kd] + span + lev
        idx = (f[sel], ch_val, rows['j'][sel], rows['i'][sel])
        np.add.at(val, idx, w[sel])
        np.add.at(cnt, (f[sel], ch_cnt, rows['j'][sel], rows['i'][sel]), 1.0)
        np.add.at(cnt, idx, 1.0)
    for kd in KINDS:
        span = PROFILE_LEVELS if kd in ('tprof', 'sprof') else 1
        vs = slice(base[kd], base[kd] + span)
        val[:, vs] = val[:, vs] / np.maximum(cnt[:, vs], 1.0)
        cs = slice(base[kd] + span, base[kd] + 2 * span)
        val[:, cs] = cnt[:, cs]
    t = np.broadcast_to((np.arange(frames) * fstep / max_step)[:, None, None, None], (frames, 1, grid.H, grid.W))
    return torch.as_tensor(np.concatenate([val, t], axis=1)).float()
