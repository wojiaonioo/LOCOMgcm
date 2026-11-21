"""Encode-process-decode analysis network; output is a whitened control."""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .features import FIELDS, FRAME_CHANNELS, Grid, pad
S_OUT = 0.03
LATENT = 192
OMEGA = 7.2921e-05
LEVEL_IN = 11
COLUMN_IN = 2 + 4
FRAME_IN = FRAME_CHANNELS + 2
FRAME_FEATURES = 64

class Static:

    def __init__(self, grid, center_lat, resolution_deg):
        (H, W) = (grid.H, grid.W)
        wet3 = torch.as_tensor(grid.mask['temp'], dtype=torch.float64)
        zc = torch.as_tensor(grid.zc / 5500.0)[:, None, None].expand_as(wet3)
        ldr = torch.as_tensor(np.log(grid.drf) / 5.0)[:, None, None].expand_as(wet3)
        self.level = pad(torch.stack([wet3, zc * wet3, ldr * wet3]), H, W).float()
        wet = torch.as_tensor(grid.mask['eta'], dtype=torch.float64)
        lat = center_lat + (np.arange(grid.ny) - (grid.ny - 1) / 2.0) * resolution_deg
        f = 2 * OMEGA * np.sin(np.radians(lat)) / (2 * OMEGA * math.sin(math.radians(15.0)))
        (jj, ii) = np.meshgrid(np.arange(grid.ny), np.arange(grid.nx), indexing='ij')
        dist = np.minimum.reduce([jj, ii, grid.ny - 1 - jj, grid.nx - 1 - ii]) / 20.0
        col = torch.stack([wet, torch.as_tensor(grid.depth / 5500.0) * wet, torch.as_tensor(np.broadcast_to(f[:, None], (grid.ny, grid.nx)).copy()) * wet, torch.as_tensor(dist, dtype=torch.float64) * wet])
        self.column = pad(col, H, W).float()
        self.surface_wet = pad(wet[None], H, W)[0].float()

def field_channels(grid, fields, scale=None):
    out = []
    for n in FIELDS:
        x = torch.as_tensor(np.asarray(fields[n], np.float64)).reshape(grid.levels[n], *grid.shapes[n][-2:])
        m = torch.as_tensor(grid.mask[n]).reshape(x.shape)
        s = 1.0 if scale is None else float(scale[n])
        out.append(pad(torch.where(m, x / s, 0.0), grid.H, grid.W))
    return torch.cat(out).float()

def frame_extras(frames, fstep, dt, max_step):
    (n, _, H, W) = frames.shape
    tau = torch.arange(n, dtype=torch.float32) * fstep * dt / 86400.0
    length = torch.full((n,), max_step * dt / 86400.0)
    return torch.stack([tau, length], 1)[:, :, None, None].expand(n, 2, H, W)

class Inputs2:

    def __init__(self, background, g0, frames):
        (self.background, self.g0, self.frames) = (background, g0, frames)

    def to(self, device):
        return Inputs2(self.background.to(device), self.g0.to(device), self.frames.to(device))

    def double(self):
        return Inputs2(self.background.double(), self.g0.double(), self.frames.double())

def mlp(a, b, c):
    return nn.Sequential(nn.Linear(a, b), nn.SiLU(), nn.Linear(b, c))

class MixBlock(nn.Module):

    def __init__(self, ch=LATENT):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch)
        self.mlp = mlp(ch, 2 * ch, ch)
        self.norm = nn.LayerNorm(ch)

    def forward(self, x, mask):
        y = self.dw(x * mask) * mask
        y = self.mlp(y.permute(0, 2, 3, 1))
        return x + self.norm(y).permute(0, 3, 1, 2) * mask

class AnalysisNet2(nn.Module):

    def __init__(self, levels=41, out_channels=165, latent=LATENT, use_g0=True):
        super().__init__()
        (self.levels, self.use_g0) = (levels, use_g0)
        self.frame = mlp(FRAME_IN, FRAME_FEATURES, FRAME_FEATURES)
        self.score = nn.Linear(FRAME_FEATURES, 1)
        (convs, c) = ([], LEVEL_IN)
        for (k, o) in enumerate((64, 64, 64, 64, 32)):
            convs.append(nn.Conv1d(c, o, 3, padding=1))
            if k < 4:
                convs.append(nn.SiLU())
            c = o
        self.column = nn.Sequential(*convs)
        self.latent = nn.Linear(levels * 32 + FRAME_FEATURES + 1 + COLUMN_IN, latent)
        self.fine = nn.ModuleList((MixBlock(latent) for _ in range(4)))
        self.coarse = nn.ModuleList((MixBlock(latent) for _ in range(4)))
        self.dec1 = nn.Linear(latent, 2 * latent)
        self.dec2 = nn.Linear(2 * latent, out_channels, bias=False)
        nn.init.zeros_(self.dec2.weight)

    def forward(self, inp, static):
        L = self.levels
        fr = inp.frames
        (Fn, _, H, W) = fr.shape
        x = fr.permute(0, 2, 3, 1).reshape(Fn, H * W, -1)
        e = self.frame(x)
        cnt = fr[:, 1] + fr[:, 3] + fr[:, 4 + 30:4 + 60].sum(1) + fr[:, 64 + 30:64 + 60].sum(1)
        has = cnt.reshape(Fn, H * W) > 0
        s = self.score(e)[..., 0].masked_fill(~has, float('-inf'))
        w = torch.softmax(s, dim=0)
        w = torch.where(has.any(0, keepdim=True), w, torch.zeros_like(w))
        pooled = (w[..., None] * e).sum(0)
        logc = torch.log1p(cnt.reshape(Fn, H * W).sum(0))[:, None]
        g0 = inp.g0 if self.use_g0 else torch.zeros_like(inp.g0)
        lev = torch.cat([inp.background[:4 * L].reshape(4, L, H, W), g0[:4 * L].reshape(4, L, H, W), static.level], 0)
        col = self.column(lev.permute(2, 3, 0, 1).reshape(H * W, LEVEL_IN, L))
        surf = torch.cat([inp.background[4 * L:4 * L + 1], g0[4 * L:4 * L + 1], static.column], 0)
        z = torch.cat([col.reshape(H * W, -1), pooled, logc, surf.reshape(COLUMN_IN, H * W).T], 1)
        z = self.latent(z).T.reshape(1, -1, H, W)
        mask = static.surface_wet[None, None]
        z = z * mask
        for blk in self.fine:
            z = blk(z, mask)
        (hc, wc) = (math.ceil(H / 4), math.ceil(W / 4))
        zp = F.pad(z, (0, wc * 4 - W, 0, hc * 4 - H))
        mp = F.pad(mask, (0, wc * 4 - W, 0, hc * 4 - H))
        c = F.avg_pool2d(zp, 4) / F.avg_pool2d(mp, 4).clamp_min(1e-06)
        cm = (F.avg_pool2d(mp, 4) > 0).to(z.dtype)
        c = c * cm
        for blk in self.coarse:
            c = blk(c, cm)
        up = c.repeat_interleave(4, -2).repeat_interleave(4, -1)[..., :H, :W]
        z = (z + up) * mask
        y = self.dec2(F.silu(self.dec1(z[0].permute(1, 2, 0))))
        return y.permute(2, 0, 1)

def parameter_count(net):
    return sum((p.numel() for p in net.parameters()))

class VMap:

    def __init__(self, C, device, space='v', s_out=S_OUT, uv=True):
        (self.C, self.space, self.s_out, self.device, self.uv) = (C, space, s_out, torch.device(device), uv)
        self.support = {n: C.support[n].to(self.device) for n in FIELDS}
        self.std = {n: C.std[n].to(self.device) for n in FIELDS}
        self.levels = {n: 1 if len(C.shapes[n]) == 2 else C.shapes[n][0] for n in FIELDS}

    def fields(self, raw):
        (out, c) = ({}, 0)
        for n in FIELDS:
            shape = self.C.shapes[n]
            g = raw[c:c + self.levels[n], :shape[-2], :shape[-1]].reshape(shape).to(torch.float64) * self.s_out
            out[n] = torch.where(self.support[n], g if self.space == 'v' else g / self.std[n], 0.0)
            if not self.uv and n in ('u', 'v'):
                out[n] = out[n] * 0.0
            c += self.levels[n]
        return out

    def control(self, raw):
        f = self.fields(raw)
        if self.space == 'v':
            v = torch.cat([f[n][self.support[n]] for n in FIELDS])
            return (v.detach().cpu().numpy(), v)
        v = self.C.adjoint({n: f[n].detach().to(self.C.device) for n in FIELDS})
        return (v.numpy(), f)

    def backward(self, handle, grad_v, scale=1.0):
        g = torch.as_tensor(np.asarray(grad_v), dtype=torch.float64)
        if self.space == 'v':
            handle.backward((scale * g).to(handle.device))
            return
        ga = self.C.forward(g.to(self.C.device))
        names = list(FIELDS)
        torch.autograd.backward([handle[n] for n in names], [(scale * ga[n]).to(handle[n].device) for n in names])

def v_share(C, v):
    sq = np.square(np.asarray(v, np.float64))
    total = float(sq.sum())
    return {n: float(sq[C.offsets[i]:C.offsets[i + 1]].sum()) / total if total > 0 else None for (i, n) in enumerate(FIELDS)}
