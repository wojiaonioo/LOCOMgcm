"""Grid-point observation cost and initial-state observation gradient."""
import numpy as np
import torch
from .control import FIELDS
KIND_FIELD = dict(ssh="eta", sst="temp", tprof="temp", sprof="salt")
OBS_DEVICE = "cuda"
class Observations:

    def __init__(self, path, shapes, wet, max_step, role='assim', device='cuda'):
        z = np.load(path, allow_pickle=False)
        keep = (z['role'] == role) & (z['step'] <= max_step)
        (self.path, self.max_step) = (str(path), max_step)
        self.dropped_after_window = int(((z['role'] == role) & (z['step'] > max_step)).sum())
        kind = z['kind'][keep]
        unknown = sorted(set(kind.tolist()) - set(KIND_FIELD))
        if unknown:
            raise ValueError(f'unknown observation kinds {unknown}')
        if 'field' in z.files and any((f != KIND_FIELD[kd] for (kd, f) in zip(kind.tolist(), z['field'][keep].tolist()))):
            raise ValueError("the file's field column disagrees with the kind -> field map")
        (step, i, j, k) = (z[n][keep].astype(np.int64) for n in ('step', 'i', 'j', 'k'))
        (y, sigma) = (z['value'][keep].astype(np.float64), z['sigma'][keep].astype(np.float64))
        if (step < 0).any() or not (sigma > 0).all() or (not np.isfinite(y).all()):
            raise ValueError('negative step, non-positive sigma or non-finite value')
        self.count = int(keep.sum())
        self.counts = {kd: int((kind == kd).sum()) for kd in KIND_FIELD}
        self.steps = sorted(set(step.tolist()))
        self.groups = {}
        for (kd, field) in KIND_FIELD.items():
            sel = kind == kd
            if not sel.any():
                continue
            nzyx = shapes[field]
            if kd in ('ssh', 'sst') and (k[sel] != 0).any():
                raise ValueError(f'{kd}: surface observation with level != 0')
            if len(nzyx) == 2:
                flat = j[sel] * nzyx[1] + i[sel]
            else:
                flat = (k[sel] * nzyx[1] + j[sel]) * nzyx[2] + i[sel]
            if not wet[field].reshape(-1)[torch.as_tensor(flat)].all():
                raise ValueError(f'{kd}: observation on a dry point')
            for s in sorted(set(step[sel].tolist())):
                r = sel & (step == s)
                fl = flat[step[sel] == s]
                dev = 'cpu' if s == 0 else device
                self.groups.setdefault(s, []).append((kd, field, torch.as_tensor(fl, device=dev), torch.as_tensor(y[r], device=dev), torch.as_tensor(sigma[r], device=dev)))

    def record(self):
        return dict(path=self.path, count=self.count, counts=self.counts, steps=self.steps, t0_count=sum((len(g[2]) for g in self.groups.get(0, []))), dropped_after_window=self.dropped_after_window)

    def t0(self, fields):
        grads = {n: torch.zeros_like(fields[n]) for n in FIELDS}
        (parts, total) = ({}, 0.0)
        for (kd, field, idx, y, s) in self.groups.get(0, []):
            r = (fields[field].reshape(-1)[idx] - y) / s
            term = float(0.5 * r.square().sum())
            parts[kd] = parts.get(kd, 0.0) + term
            total += term
            grads[field].view(-1).index_add_(0, idx, r / s)
        return (total, parts, grads)

    def objective(self, session, steps, parts, hx=None):
        (origin, dtc) = (session.origin, session.bundle.dt_clock)

        def observe(model, st):
            zero = st.eta.data.sum() * 0.0
            k = round((model.current_time_seconds - origin) / dtc)
            groups = self.groups.get(k) if 1 <= k <= steps else None
            if not groups:
                return zero
            out = zero
            for (kd, field, idx, y, s) in groups:
                val = getattr(st, field).data.reshape(-1)[idx]
                if hx is not None:
                    hx[k, kd] = val.detach().clone()
                term = 0.5 * ((val - y) / s).square().sum()
                parts[k, kd] = float(term.detach())
                out = out + term
            return out
        return observe
