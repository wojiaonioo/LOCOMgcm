"""Masked Gaussian control transform and its explicit transpose."""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import torch
FIELDS = ('temp', 'salt', 'u', 'v', 'eta')
LENGTH_KM = dict(temp=100.0, salt=100.0, eta=100.0, u=70.0, v=70.0)
VERT_MIN_M = 30.0
EDGE = 4
CUT_SIGMA = 4.0

def support_masks(mask_c, mask_u, mask_v, edge=EDGE):
    (nz, ny, nx) = mask_c.shape
    ci = np.zeros(nx, bool)
    ci[edge:nx - edge] = True
    cj = np.zeros(ny, bool)
    cj[edge:ny - edge] = True
    fu = np.zeros(nx + 1, bool)
    fu[1:nx] = ci[:-1] & ci[1:]
    fv = np.zeros(ny + 1, bool)
    fv[1:ny] = cj[:-1] & cj[1:]
    cell = cj[:, None] & ci[None, :]
    return dict(temp=mask_c & cell, salt=mask_c & cell, u=mask_u & (cj[:, None] & fu[None, :]), v=mask_v & (fv[:, None] & ci[None, :]), eta=mask_c[0] & cell)

def run_labels(mask, axis):
    m = mask.to(torch.int64)
    first = torch.zeros_like(m.narrow(axis, 0, 1))
    change = torch.diff(m, dim=axis, prepend=first).abs()
    return torch.where(mask, torch.cumsum(change, dim=axis), -1)

def shifted(x, axis, d, fill):
    size = x.shape[axis]
    if abs(d) >= size:
        return torch.full_like(x, fill)
    pad_shape = list(x.shape)
    pad_shape[axis] = abs(d)
    pad = torch.full(pad_shape, fill, dtype=x.dtype, device=x.device)
    if d >= 0:
        return torch.cat([x.narrow(axis, d, size - d), pad], dim=axis)
    return torch.cat([pad, x.narrow(axis, 0, size + d)], dim=axis)

class ControlTransform:

    def __init__(self, grid_npz, b_std, device='cpu', dtype=torch.float64):
        z = np.load(grid_npz)
        (self.device, self.dtype) = (torch.device(device), dtype)
        sup = support_masks(z['mask_center'], z['mask_u'], z['mask_v'])
        dz = np.asarray(z['layer_thickness'], np.float64)
        zc = np.asarray(z['center_depth'], np.float64)
        dy = float(np.asarray(z['dy_center']).mean())
        if np.ptp(z['dx_center'], axis=1).max() > 1e-06 * dy or np.ptp(z['dy_center']) > 1e-06 * dy:
            raise ValueError('the passes assume one dx per row and one dy')
        dx_rows = dict(temp=z['dx_center'][:, 0], salt=z['dx_center'][:, 0], eta=z['dx_center'][:, 0], u=z['dx_center'][:, 0], v=z['dxG_row_ext'])
        self.dx_rows = {n: np.asarray(r, np.float64) for (n, r) in dx_rows.items()}
        self.nz = len(dz)
        s = np.maximum(VERT_MIN_M, dz) / math.sqrt(2.0)
        sig2 = 0.5 * (s[:, None] ** 2 + s[None, :] ** 2)
        self.gz = torch.as_tensor(np.exp(-(zc[:, None] - zc[None, :]) ** 2 / (2.0 * sig2)), dtype=dtype, device=self.device)
        (self.support, self.labels, self.kx, self.ky, self.norm, self.std) = ({}, {}, {}, {}, {}, {})
        (self.shapes, self.counts) = ({}, {})
        for n in FIELDS:
            m = torch.as_tensor(sup[n], device=self.device)
            if m.dim() == 3 and (not bool((m[1:] <= m[:-1]).all())):
                raise ValueError(f'{n}: wet levels are not contiguous from the surface')
            self.support[n] = m
            self.shapes[n] = tuple(m.shape)
            self.counts[n] = int(m.sum())
            self.labels[n] = {a: run_labels(m, a) for a in (-1, -2)}
            sigma = LENGTH_KM[n] * 1000.0 / math.sqrt(2.0)
            self.kx[n] = self._kernel(sigma, np.asarray(dx_rows[n], np.float64))
            self.ky[n] = self._kernel(sigma, np.array([dy]))[:, 0]
            sd = np.asarray(b_std[n], np.float64).reshape(-1)
            if sd.shape != ((self.nz,) if m.dim() == 3 else (1,)):
                raise ValueError(f'B standard deviation of {n} has shape {sd.shape}')
            if not (np.isfinite(sd).all() and (sd > 0).all()):
                raise ValueError(f'invalid B standard deviation of {n}')
            self.std[n] = torch.as_tensor(sd.reshape((-1, 1, 1) if m.dim() == 3 else ()), dtype=dtype, device=self.device)
            diag = self._s(m.to(dtype), n, power=2)
            if (diag[m] <= 0).any():
                raise ValueError(f'zero normalisation on the support of {n}')
            self.norm[n] = torch.where(m, diag.clamp_min(1e-300).rsqrt(), 0.0)
        self.offsets = np.cumsum([0] + [self.counts[n] for n in FIELDS])
        self.size = int(self.offsets[-1])

    def _kernel(self, sigma, spacing):
        radius = int(math.ceil(CUT_SIGMA * sigma / spacing.min()))
        d = np.arange(-radius, radius + 1, dtype=np.float64)[:, None] * spacing[None, :]
        w = np.where(np.abs(d) <= CUT_SIGMA * sigma, np.exp(-d ** 2 / (2.0 * sigma ** 2)), 0.0)
        return torch.as_tensor(w, dtype=self.dtype, device=self.device)

    def _pass_h(self, x, n, axis, power):
        lab = self.labels[n][axis]
        w = (self.kx[n] if axis == -1 else self.ky[n]) ** power
        radius = (w.shape[0] - 1) // 2
        out = torch.zeros_like(x)
        for d in range(-radius, radius + 1):
            if abs(d) >= x.shape[axis]:
                continue
            same = (shifted(lab, axis, d, -2) == lab) & (lab >= 0)
            wd = w[d + radius].reshape(-1, 1) if axis == -1 else w[d + radius]
            out = out + torch.where(same, shifted(x, axis, d, 0.0) * wd, 0.0)
        return out

    def _pass_z(self, x, power):
        return torch.einsum('pq,qji->pji', self.gz ** power, x)

    def _s(self, x, n, power=1):
        m = self.support[n]
        y = torch.where(m, x, 0.0)
        y = self._pass_h(y, n, -1, power)
        y = self._pass_h(y, n, -2, power)
        if y.dim() == 3:
            y = self._pass_z(y, power)
        return torch.where(m, y, 0.0)

    def _st(self, x, n):
        m = self.support[n]
        y = torch.where(m, x, 0.0)
        if y.dim() == 3:
            y = torch.where(m, self._pass_z(y, 1), 0.0)
        y = self._pass_h(y, n, -2, 1)
        y = self._pass_h(y, n, -1, 1)
        return torch.where(m, y, 0.0)

    def unpack(self, v):
        out = {}
        for (i, n) in enumerate(FIELDS):
            full = torch.zeros(self.shapes[n], dtype=v.dtype, device=v.device)
            out[n] = full.masked_scatter(self.support[n], v[self.offsets[i]:self.offsets[i + 1]])
        return out

    def pack(self, fields):
        return torch.cat([fields[n].reshape(self.shapes[n])[self.support[n]] for n in FIELDS])

    def correlation_sqrt(self, v):
        u = self.unpack(v)
        return {n: self.norm[n] * self._s(u[n], n) for n in FIELDS}

    def forward(self, v):
        s = self.correlation_sqrt(v)
        return {n: self.std[n] * s[n] for n in FIELDS}
    __call__ = forward

    def adjoint(self, fields):
        return self.pack({n: self._st(self.norm[n] * self.std[n] * fields[n].reshape(self.shapes[n]), n) for n in FIELDS})

    def check(self, seed=20261018, points=64, probes=4):
        gen = torch.Generator(device='cpu').manual_seed(seed)
        rec = dict(size=self.size, counts=self.counts, length_km=LENGTH_KM, vertical_min_m=VERT_MIN_M, cut_sigma=CUT_SIGMA, edge=EDGE, kernel_radius_cells={n: dict(x=(self.kx[n].shape[0] - 1) // 2, y=(self.ky[n].shape[0] - 1) // 2) for n in FIELDS})
        v = torch.randn(self.size, generator=gen, dtype=self.dtype).to(self.device)
        w = {n: torch.randn(self.shapes[n], generator=gen, dtype=self.dtype).to(self.device) for n in FIELDS}
        cv = self.forward(v)
        ctw = self.adjoint(w)
        lhs = float(sum(((cv[n] * w[n]).sum() for n in FIELDS)))
        rhs = float(torch.dot(v, ctw))
        vv = v.clone().requires_grad_(True)
        out = self.forward(vv)
        auto = torch.autograd.grad(sum(((out[n] * w[n]).sum() for n in FIELDS)), vv)[0]
        rec['dot_test'] = dict(Cv_dot_w=lhs, v_dot_CTw=rhs, relative=abs(lhs - rhs) / abs(lhs), autograd_vs_explicit_CT_relative=float((auto - ctw).norm() / ctw.norm()))
        diag = {}
        for n in FIELDS:
            idx = torch.nonzero(self.support[n].flatten()).flatten()
            pick = idx[torch.randperm(len(idx), generator=gen)[:points].to(idx.device)]
            vals = []
            for p in pick.tolist():
                e = torch.zeros(math.prod(self.shapes[n]), dtype=self.dtype, device=self.device)
                e[p] = 1.0
                vals.append(float(self._st(self.norm[n] * e.reshape(self.shapes[n]), n).square().sum()))
            vals = np.array(vals)
            diag[n] = dict(points=len(vals), min=float(vals.min()), max=float(vals.max()), max_abs_error=float(np.abs(vals - 1.0).max()))
        rec['diag_L'] = diag
        rec['white_v_rms_over_D'] = {n: float((cv[n] / self.std[n].expand(self.shapes[n]))[self.support[n]].square().mean().sqrt()) for n in FIELDS}
        rec['correlation_along_x'] = self._probe(gen, probes)
        return rec

    def _probe(self, gen, count, reach=12):
        out = {}
        for n in ('temp', 'u', 'eta'):
            m2 = self.support[n] if self.support[n].dim() == 2 else self.support[n][0]
            (ny, nx) = m2.shape
            cand = [(j, i) for j in range(reach, ny - reach) for i in range(reach, nx - reach) if bool(m2[j, i - reach:i + reach + 1].all()) and bool(m2[j - reach:j + reach + 1, i].all())]
            rows = []
            for k in torch.randperm(len(cand), generator=gen)[:count].tolist():
                (j, i) = cand[k]
                e = torch.zeros(self.shapes[n], dtype=self.dtype, device=self.device)
                e[(0, j, i) if e.dim() == 3 else (j, i)] = 1.0
                col = self.norm[n] * self._s(self._st(self.norm[n] * e, n), n)
                line = (col[0, j] if col.dim() == 3 else col[j]).cpu().numpy()
                dx = float(self.dx_rows[n][j])
                rows.append(dict(j=j, i=i, dx_km=dx / 1000.0, corr=[round(float(line[i + d]), 4) for d in range(reach + 1)], target=[round(math.exp(-(d * dx) ** 2 / (2.0 * (LENGTH_KM[n] * 1000.0) ** 2)), 4) for d in range(reach + 1)]))
            out[n] = dict(length_km=LENGTH_KM[n], points=rows)
        return out

def load_b_std(path):
    d = json.loads(Path(path).read_text())
    return {n: d[n] for n in FIELDS}
