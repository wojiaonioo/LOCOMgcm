"""Pointwise momentum forcing inside each model time step."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from locomgcm.dynamics.base import MomentumTermClass, TendencyModule, TendencyPhase, TendencyStepDomain
from locomgcm.state import StateTendency
HIDDEN = 32
AMPLITUDE = 1e-6
U0, T0, DT, S0, DS = 0.5, 15.0, 10.0, 34.5, 1.0
class MomentumMLP(nn.Module):
    shapes = ((HIDDEN, 4), (HIDDEN,), (HIDDEN, HIDDEN), (HIDDEN,), (2, HIDDEN), (2,))

    def __init__(self, seed):
        super().__init__()
        gen = torch.Generator().manual_seed(int(seed))
        values = []
        for shape in self.shapes:
            x = torch.randn(shape, generator=gen, dtype=torch.float64)
            x = x / math.sqrt(shape[1]) if len(shape) == 2 else x * 0.1
            values.append(x.reshape(-1))
        self.theta = nn.Parameter(torch.cat(values))

    def layers(self, theta):
        (out, start) = ([], 0)
        for shape in self.shapes:
            n = math.prod(shape)
            out.append(theta[start:start + n].view(shape))
            start += n
        return out

    def forward(self, x):
        (w1, b1, w2, b2, w3, b3) = self.layers(self.theta)
        h = F.silu(F.linear(x, w1, b1))
        h = F.silu(F.linear(h, w2, b2))
        return torch.tanh(F.linear(h, w3, b3))

class NetworkMomentumTendency(TendencyModule):
    name = 'instep_network_momentum'
    stepper_domain = TendencyStepDomain.SLOW
    tendency_phase = TendencyPhase.MOMENTUM
    momentum_term_class = MomentumTermClass.FORCING

    def __init__(self, law, model):
        g = model.grid
        (nz, ny, nx) = (int(g.nz), int(g.ny), int(g.nx))
        dev = torch.device(model.config.device)
        c = g.hfac_c_for(torch.float64, dev).detach().reshape(nz, ny, nx) > 0
        w = (g.hfac_w_for(torch.float64, dev).detach().reshape(nz, ny, nx + 1) > 0).clone()
        s = (g.hfac_s_for(torch.float64, dev).detach().reshape(nz, ny + 1, nx) > 0).clone()
        w[..., 0] = False
        w[..., -1] = False
        s[..., 0, :] = False
        s[..., -1, :] = False
        (self.law, self.wet_c, self.wet_w, self.wet_s, self.shape) = (law, c, w, s, (nz, ny, nx))

    def evaluate(self, state, diagnostics, grid):
        (nz, ny, nx) = self.shape
        u = state.u.data.reshape(nz, ny, nx + 1)
        v = state.v.data.reshape(nz, ny + 1, nx)
        t = state.temp.data.reshape(nz, ny, nx)
        s = state.salt.data.reshape(nz, ny, nx)
        uc = 0.5 * (u[..., :-1] + u[..., 1:])
        vc = 0.5 * (v[..., :-1, :] + v[..., 1:, :])
        x = torch.stack((uc / U0, vc / U0, (t - T0) / DT, (s - S0) / DS), dim=-1)
        x = torch.where(self.wet_c[..., None], x, 0.0)
        out = AMPLITUDE * self.law(x)
        fx = torch.where(self.wet_c, out[..., 0], 0.0)
        fy = torch.where(self.wet_c, out[..., 1], 0.0)
        du = torch.where(self.wet_w, F.pad(0.5 * (fx[..., :-1] + fx[..., 1:]), (1, 1)), 0.0)
        dv = torch.where(self.wet_s, F.pad(0.5 * (fy[..., :-1, :] + fy[..., 1:, :]), (0, 0, 1, 1)), 0.0)
        return StateTendency(eta=torch.zeros_like(state.eta.data), u=du.reshape(state.u.data.shape), v=dv.reshape(state.v.data.shape), temp=torch.zeros_like(state.temp.data), salt=torch.zeros_like(state.salt.data))
