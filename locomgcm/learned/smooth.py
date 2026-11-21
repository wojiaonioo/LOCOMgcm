"""Continuous closure on regularised Richardson numbers, with slope-limited Ri tables.

Long cooling windows are sensitive to the Richardson-number input: where N^2 and S^2 both approach 0, Ri = N^2 / S^2 is
the ratio of two near-zero numbers. A steep MLP map can amplify its state
derivative, while hard bins in ``RegimeEmbeddingClosure`` make the loss
jump when a small parameter change moves an interface across a bin edge.
This closure keeps the dependence on Ri continuous and bounds it:

* regularised input: at every interior interface the gradient Richardson
  number becomes N^2 / (S^2 + s0) and the bulk one N_b^2 / (S_b^2 + s0), both
  through atan, with s0 = ``RI_S0`` = 1e-7 s^-2 fixed in advance (the weak
  stratification threshold of the regime closure).  Its partial derivatives with respect
  to N^2 and S^2 are bounded by 1 / s0 everywhere, and at N^2 = S^2 = 0 it is
  0 instead of the ratio of two roundings;
* slope-limited tables (``ri_nodes`` > 0, default ``RI_NODES`` = 24): each of
  the two atan values goes through its own piecewise-linear table
  (``SlopeLimitedTable``) with ``ri_nodes`` equally spaced nodes on
  [-pi/2, pi/2] and learned node values of dimension ``EMBED_DIM``; in the
  forward pass the difference of adjacent node values is clipped to
  [-``MAX_NODE_STEP``, ``MAX_NODE_STEP``] per component, so the 8-vector
  changes by at most 1 per component between two nodes;
* control (``ri_nodes`` = 0): the two atan values enter the MLP head directly
  as continuous channels, regularised but without a slope limit.

The bulk Richardson number of ``features.column_features`` is
(z_c[k] - z_c[0]) dbsfc[k] / |V[0] - V[k]|^2 = N_b^2 / S_b^2 with
N_b^2 = dbsfc[k] / (z_c[k] - z_c[0]) and S_b^2 = |V[0] - V[k]|^2 /
(z_c[k] - z_c[0])^2, so "s0 added to its denominator" is S_b^2 + s0 in s^-2.
``ColumnFeatures`` does not carry dbsfc and |V[0] - V[k]|^2, and features.py
stays unchanged, so ``features_from_inputs`` (the hook ``hybrid.learned_diagnose``
calls) appends the two channels ``BULK_CHANNELS`` of every interior interface,
asinh(N_b^2 / 1e-5 s^-2) and asinh(S_b^2 / 1e-5 s^-2) (``bulk_channels``, the
definitions of column_features), after the column scalars of the standard
feature vector.  N^2 and S^2 come from the standard ``n2`` and ``s2``
channels by inverting their asinh scaling.

No input is a hard token: the N^2-sign and forcing-sign tokens of the regime
closure are left out, since a sign token of N^2 jumps at N^2 = 0, while
the regularised atan(N^2 / (S^2 + s0)) carries the sign of N^2
continuously.  The head input of an interface is the continuous channels
``KEPT_CHANNELS`` (the interface channels without ri_g and ri_b)
followed by the two table outputs (2 x 8, zero on dry interfaces) or the two
atan values; the column scalars follow the interface block once.  That vector
feeds a ``LearnedKppClosure`` head (``ColumnMLP``), so outputs, caps,
backgrounds and index conventions match that head, and the closure
takes the keyword arguments of ``LearnedKppClosure.forward``.  The head's own
gain is not limited: the tables bound the slope of the embedding, not of
log10 K.
"""
from __future__ import annotations

import dataclasses
import math

import torch
from torch import nn

from locomgcm.dynamics.kpp import jmd95z_layer_pressure, kpp_statekpp
from locomgcm.learned.closure_kpp import LearnedKppClosure
from locomgcm.learned.features import (
    _N2_SCALE,
    _S2_SCALE,
    COLUMN_CHANNELS,
    INTERFACE_CHANNELS,
    ColumnFeatures,
)
from locomgcm.learned.hybrid import _crop_center
from locomgcm.learned.hybrid import features_from_inputs as standard_features_from_inputs
from locomgcm.learned.regime import KEPT_CHANNELS
from locomgcm.operators import average_u_to_center, average_v_to_center

RI_S0 = 1.0e-7            # s^-2, added to S^2 and S_b^2
RI_NODES = 24             # nodes of each Ri table on [-pi/2, pi/2]
EMBED_DIM = 8             # output dimension of each Ri table
MAX_NODE_STEP = 1.0       # bound of |adjacent node difference| per component
#: Channels appended per interior interface after the column scalars.
BULK_CHANNELS = ("nb2", "sb2")   # asinh(N_b^2 / 1e-5 s^-2), asinh(S_b^2 / 1e-5 s^-2)


def regularized_atan_ri(n2: torch.Tensor, s2: torch.Tensor, s0: float = RI_S0) -> torch.Tensor:
    """atan(N^2 / (S^2 + s0)) for N^2, S^2 >= 0 in s^-2; |d/dN^2| <= 1/s0, |d/dS^2| <= 1/(2 s0)."""
    return torch.atan(n2 / (s2 + s0))


def bulk_channels(
    temp: torch.Tensor,
    salt: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    *,
    eos: str,
    eos_pressure_pa: torch.Tensor | None,
    drf: torch.Tensor,
    mask_c: torch.Tensor,
    gravity: float,
    rho0: float,
) -> torch.Tensor:
    """(B, ny, nx, 2 (nz - 1)): asinh(N_b^2 / 1e-5), asinh(S_b^2 / 1e-5) per interior interface.

    Arguments as in ``features.column_features``.  Interface k = 1..nz-1 with
    zr = z_c[k] - z_c[0] (the Ritop scale of column_features): N_b^2 =
    dbsfc[k] / zr (dbsfc of ``kpp_statekpp`` with the pressure choice of
    column_features) and S_b^2 = |V[0] - V[k]|^2 / zr^2 (V the cell-centred
    velocity), so N_b^2 / S_b^2 is the Ri_b of column_features without its
    1e-10 guard.  Interface-major, (nb2, sb2) per interface, both 0 on dry
    interfaces by selection.
    """
    batch, nz, ny, nx = temp.shape
    dtype, device = temp.dtype, temp.device
    drf = drf.to(dtype=dtype, device=device)
    mask = mask_c.to(dtype=dtype, device=device).expand(batch, nz, ny, nx)
    pressure = (eos_pressure_pa if eos_pressure_pa is not None
                else jmd95z_layer_pressure(drf, gravity=gravity, rho_const=rho0))
    # only dbsfc is read here, and it does not depend on the dbloc form
    _, dbsfc, _, _, _ = kpp_statekpp(temp, salt, pressure, gravity=gravity, rho_const=rho0, eos_type=eos,
                                     dbloc_form="statekpp")
    centre = torch.cumsum(drf, 0) - 0.5 * drf
    zr = (centre[1:] - centre[0]).reshape(1, nz - 1, 1, 1)
    u_c = average_u_to_center(u)
    v_c = average_v_to_center(v)
    du0 = u_c[:, :1] - u_c[:, 1:]
    dv0 = v_c[:, :1] - v_c[:, 1:]
    nb2 = dbsfc[:, 1:] / zr
    sb2 = (du0 * du0 + dv0 * dv0) / (zr * zr)
    channels = torch.stack((torch.asinh(nb2 / _N2_SCALE), torch.asinh(sb2 / _S2_SCALE)), dim=-1)
    wet = (mask[:, 1:] * mask[:, :-1] > 0).unsqueeze(-1)
    channels = torch.where(wet, channels, torch.zeros_like(channels))
    return channels.permute(0, 2, 3, 1, 4).reshape(batch, ny, nx, (nz - 1) * len(BULK_CHANNELS))


class SlopeLimitedTable(nn.Module):
    """Piecewise-linear map of [-pi/2, pi/2] to R^dim with a bounded step between nodes.

    ``nodes`` equally spaced nodes (``positions``, fixed); the parameter
    ``values`` (nodes, dim) holds the learned node values.  ``node_values()``
    keeps the first row and rebuilds the others from the adjacent differences
    clipped to [-max_step, max_step] per component, so the interpolant is
    continuous and its slope is at most max_step / spacing per component.
    Initial values are the linear ramps positions * a with a ~ N(0, 1) per
    component, a step of |a| * spacing (inside the clip unless |a| > 7.3 at 24
    nodes).
    """

    def __init__(self, nodes: int = RI_NODES, dim: int = EMBED_DIM, max_step: float = MAX_NODE_STEP):
        super().__init__()
        if nodes < 2:
            raise ValueError(f"a table needs at least 2 nodes, got {nodes}")
        if not max_step > 0.0:
            raise ValueError(f"max_step must be > 0, got {max_step}")
        self.nodes, self.dim, self.max_step = int(nodes), int(dim), float(max_step)
        self.spacing = math.pi / (self.nodes - 1)
        self.register_buffer("positions", torch.linspace(-0.5 * math.pi, 0.5 * math.pi, self.nodes,
                                                         dtype=torch.float64), persistent=False)
        self.values = nn.Parameter(self.positions.unsqueeze(-1)
                                   * torch.randn(self.dim, dtype=torch.float64))

    def node_values(self) -> torch.Tensor:
        step = torch.diff(self.values, dim=0).clamp(-self.max_step, self.max_step)
        return torch.cat((self.values[:1], self.values[:1] + torch.cumsum(step, dim=0)), dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (...) in [-pi/2, pi/2] -> (..., dim)."""
        v = self.node_values()
        t = (x + 0.5 * math.pi) / self.spacing
        i = t.detach().floor().clamp(0, self.nodes - 2).long()
        w = (t - i.to(t.dtype)).unsqueeze(-1)
        return v[i] * (1.0 - w) + v[i + 1] * w


class SmoothRiClosure(nn.Module):
    """Regularised atan(Ri) inputs, slope-limited tables (or none), then a ``ColumnMLP`` head.

    Built for ``n_iface`` interior interfaces (one vertical grid); ``hidden``,
    ``depth``, ``k_cap`` and ``tie_salt_to_temp`` are those of the head.
    ``ri_nodes`` > 1 gives the two tables with that many nodes, 0 the control
    without tables; ``s0`` (s^-2) regularises both Richardson numbers (module
    docstring).  The input is the feature vector of ``features_from_inputs``:
    the ``ColumnFeatures`` of hybrid.features_from_inputs with the
    ``BULK_CHANNELS`` block appended; a standard feature vector is refused.
    """

    architecture = "smooth"

    def __init__(self, n_iface: int, hidden: int = 256, depth: int = 3, k_cap: float = 5.0,
                 tie_salt_to_temp: bool = True, *, ri_nodes: int = RI_NODES, s0: float = RI_S0):
        super().__init__()
        if n_iface < 1:
            raise ValueError(f"n_iface must be >= 1, got {n_iface}")
        if ri_nodes != 0 and ri_nodes < 2:
            raise ValueError(f"ri_nodes must be 0 (no tables) or >= 2, got {ri_nodes}")
        if not s0 > 0.0:
            raise ValueError(f"s0 must be > 0 s^-2, got {s0}")
        self.n_iface = int(n_iface)
        self.ri_nodes = int(ri_nodes)
        self.s0 = float(s0)
        c, c_col = len(INTERFACE_CHANNELS), len(COLUMN_CHANNELS)
        self.feature_dim = self.n_iface * (c + len(BULK_CHANNELS)) + c_col
        ri_width = 2 * EMBED_DIM if self.ri_nodes else 2
        self.head_dim = self.n_iface * (len(KEPT_CHANNELS) + ri_width) + c_col
        self.register_buffer("kept_index", torch.tensor(
            [INTERFACE_CHANNELS.index(name) for name in KEPT_CHANNELS]), persistent=False)
        self.tables = (nn.ModuleDict({name: SlopeLimitedTable(self.ri_nodes) for name in ("ri_g", "ri_b")})
                       if self.ri_nodes else None)
        self.head = LearnedKppClosure(self.n_iface, self.head_dim, hidden=hidden, depth=depth,
                                      k_cap=k_cap, tie_salt_to_temp=tie_salt_to_temp)

    @property
    def k_cap(self) -> float:
        return self.head.k_cap

    def features_from_inputs(self, inputs, temp, salt, u, v, *, grid) -> ColumnFeatures:
        """``hybrid.features_from_inputs`` with ``bulk_channels`` of the same inputs appended."""
        feats = standard_features_from_inputs(inputs, temp, salt, u, v, grid=grid)
        params = inputs.params
        ny, nx = temp.shape[-2], temp.shape[-1]
        bulk = bulk_channels(
            temp, salt, u, v, eos=str(params["eos_type"]), eos_pressure_pa=inputs.eos_pressure_pa,
            drf=inputs.static.layer_thickness,
            mask_c=_crop_center((grid.masks.hFacC > 0).to(temp.dtype), ny, nx),
            gravity=float(params["gravity"]), rho0=float(params["rho_const"]),
        )
        x = torch.cat((feats.x, bulk), dim=-1)
        return dataclasses.replace(feats, x=x, dim=x.shape[-1])

    def _check(self, feats: ColumnFeatures) -> None:
        if feats.n_iface != self.n_iface or feats.dim != self.feature_dim:
            raise ValueError(f"features have n_iface={feats.n_iface}, dim={feats.dim}; the closure was "
                             f"built for n_iface={self.n_iface}, feature_dim={self.feature_dim} (the "
                             "standard features plus the bulk channels of features_from_inputs)")

    def ri_inputs(self, feats: ColumnFeatures) -> torch.Tensor:
        """(B, ny, nx, n_iface, 2): atan of the regularised gradient and bulk Ri, 0 on dry interfaces."""
        self._check(feats)
        n, c, c_col = self.n_iface, len(INTERFACE_CHANNELS), len(COLUMN_CHANNELS)
        iface = feats.x[..., : n * c].unflatten(-1, (n, c))
        bulk = feats.x[..., n * c + c_col:].unflatten(-1, (n, len(BULK_CHANNELS)))
        n2 = _N2_SCALE * torch.sinh(iface[..., INTERFACE_CHANNELS.index("n2")])
        s2 = _S2_SCALE * torch.sinh(iface[..., INTERFACE_CHANNELS.index("s2")])
        nb2 = _N2_SCALE * torch.sinh(bulk[..., 0])
        sb2 = _S2_SCALE * torch.sinh(bulk[..., 1])
        r = torch.stack((regularized_atan_ri(n2, s2, self.s0), regularized_atan_ri(nb2, sb2, self.s0)), dim=-1)
        wet = (feats.wet_iface[:, 1:-1] > 0).permute(0, 2, 3, 1).unsqueeze(-1)
        return torch.where(wet, r, torch.zeros_like(r))

    def head_input(self, feats: ColumnFeatures) -> torch.Tensor:
        """(B, ny, nx, head_dim): per interface the kept channels and the Ri part, then the column scalars."""
        r = self.ri_inputs(feats)
        n, c, c_col = self.n_iface, len(INTERFACE_CHANNELS), len(COLUMN_CHANNELS)
        kept = feats.x[..., : n * c].unflatten(-1, (n, c)).index_select(-1, self.kept_index)
        if self.tables is None:
            ri = r
        else:
            ri = torch.cat((self.tables["ri_g"](r[..., 0]), self.tables["ri_b"](r[..., 1])), dim=-1)
            wet = (feats.wet_iface[:, 1:-1] > 0).permute(0, 2, 3, 1).unsqueeze(-1)
            ri = torch.where(wet, ri, torch.zeros_like(ri))
        column = feats.x[..., n * c: n * c + c_col]
        return torch.cat((torch.cat((kept, ri), dim=-1).flatten(-2), column), dim=-1)

    def forward(self, feats: ColumnFeatures, **kwargs) -> dict[str, torch.Tensor]:
        x = self.head_input(feats)
        return self.head(dataclasses.replace(feats, x=x, dim=x.shape[-1]), **kwargs)
