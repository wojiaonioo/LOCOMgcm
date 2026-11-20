"""Regime-embedding closure: the Richardson-number dependence as piecewise constants.

In hidden-label learning, the tangent model of cooling windows diverged within
18 to 36 steps from the PP81 fit.  The hypothesis behind this closure is that
the divergence runs through the closure's derivative with respect to the state
along the steep part of PP81 near its Richardson-number threshold.  This
closure takes the gradient and bulk Richardson numbers, the sign of N^2 and
the sign of the surface buoyancy flux as fixed-bin tokens and looks each up in
a learned embedding table, so the derivative of the closure output with
respect to the state is exactly 0 along those inputs.

Tokens, from the detached features under ``torch.no_grad`` (``torch.bucketize``
with ``right=False``: bin i holds edges[i-1] < x <= edges[i]):

* ``ri_g``, ``ri_b``: the ``ri_g``/``ri_b`` channels of ``column_features``
  are atan(Ri), in (-pi/2, pi/2); ``ri_bins`` (default ``RI_BINS``) equal bins
  of that interval, the edges fixed here (``ri_edges``) and never fitted.
  With ``weak_threshold`` set (s^-2), an
  interface with |N^2| < weak_threshold or S^2 < weak_threshold takes the
  extra token ``ri_bins`` in both ri_g and ri_b instead of its Ri bin: in the
  cooling windows studied, 81% to 94% of the token changes of a cooling window at |N^2| < 1e-7,
  where N^2 and S^2 are both near 0 and Ri is the ratio of two near-zero
  numbers.  The test is on the ``n2`` and ``s2`` channels, |asinh(N^2 /
  1e-5)| < asinh(weak_threshold / 1e-5) and likewise for S^2, which is the
  test on N^2 and S^2 because asinh is increasing and odd;
* ``n2_sign``: the ``n2`` channel (asinh of N^2 / 1e-5 s^-2), N^2 <= 0 -> 0,
  N^2 > 0 -> 1;
* ``forcing``: one token per column from the ``b_f`` column scalar (asinh of
  B_f / 1e-7 m^2/s^3, B_f < 0 destabilising): cooling (B_f <= -B0) -> 0, zero
  flux (-B0 < B_f <= B0) -> 1, heating (B_f > B0) -> 2, with
  B0 = ``ZERO_FLUX_BUOYANCY`` = 1e-12 m^2/s^3, a Qnet of about 0.002 W/m^2 for
  a thermal expansion coefficient of 2e-4 1/K.  The zero-flux forcing records
  of the truth cases are exactly 0, so any small B0 gives the same
  tokens away from the few hours where a record ramps to or from 0.

Every token type has its own embedding table of dimension ``EMBED_DIM``.  At
each interior interface the four embedding vectors (the column's forcing
vector repeated on every interface; all four set to 0 on dry interfaces by
selection) follow the kept continuous channels ``KEPT_CHANNELS``
(``INTERFACE_CHANNELS`` without ``ri_g`` and ``ri_b``); the column scalars
``COLUMN_CHANNELS`` follow the interface block once.  That vector is the input
of a ``LearnedKppClosure`` head (``ColumnMLP``), so the outputs (diff_kzt,
diff_kzs, visc_cell, ghat, hbl, gamma_iface), their caps, backgrounds and
index conventions match that head. The closure accepts the same
``ColumnFeatures`` and keyword arguments as ``LearnedKppClosure``, so it
works with ``LearnedKppInjection``, ``FrozenInputClosure`` and ``ClosureTap``.
"""
from __future__ import annotations

import dataclasses
import math

import torch
from torch import nn

from defgcm.learned.closure_kpp import LearnedKppClosure
from defgcm.learned.features import (
    _N2_SCALE,
    _S2_SCALE,
    COLUMN_CHANNELS,
    INTERFACE_CHANNELS,
    ColumnFeatures,
)

RI_BINS = 24
EMBED_DIM = 8
ZERO_FLUX_BUOYANCY = 1.0e-12       # m^2/s^3
_B_F_SCALE = 1.0e-7                # m^2/s^3, the scale of the b_f feature (features.py)
#: Interface channels that stay continuous; ri_g and ri_b enter as tokens only.
KEPT_CHANNELS = tuple(name for name in INTERFACE_CHANNELS if name not in ("ri_g", "ri_b"))
#: Token types and their number of bins with the default options.
TOKEN_BINS = {"ri_g": RI_BINS, "ri_b": RI_BINS, "n2_sign": 2, "forcing": 3}


def ri_edges(dtype: torch.dtype = torch.float64, bins: int = RI_BINS) -> torch.Tensor:
    """The ``bins - 1`` inner edges of ``bins`` equal bins of (-pi/2, pi/2)."""
    return torch.linspace(-0.5 * math.pi, 0.5 * math.pi, bins + 1, dtype=dtype)[1:-1]


class RegimeEmbeddingClosure(nn.Module):
    """Tokens and embeddings for Ri, N^2 sign and forcing sign, then a ``ColumnMLP`` head.

    Built for ``n_iface`` interior interfaces (one vertical grid), like the
    ``"mlp"`` ``LearnedKppClosure``; ``hidden``, ``depth``, ``k_cap`` and
    ``tie_salt_to_temp`` are those of the head.  ``ri_bins`` and
    ``weak_threshold`` (s^-2, ``None``: no weak-stratification token) define
    the ri_g and ri_b tokens (module docstring). ``token_bins`` holds the
    number of tokens of every type, the
    rows of its embedding table, so a checkpoint loads only into a closure
    with the same ``ri_bins`` and with or without the weak token alike.
    ``tokens(feats)`` returns the integer tokens (no gradient); ``forward``
    has the signature and output of ``LearnedKppClosure.forward``.
    """

    architecture = "regime"

    def __init__(self, n_iface: int, hidden: int = 256, depth: int = 3, k_cap: float = 5.0,
                 tie_salt_to_temp: bool = True, *, ri_bins: int = RI_BINS,
                 weak_threshold: float | None = None):
        super().__init__()
        if n_iface < 1:
            raise ValueError(f"n_iface must be >= 1, got {n_iface}")
        if ri_bins < 1:
            raise ValueError(f"ri_bins must be >= 1, got {ri_bins}")
        if weak_threshold is not None and not weak_threshold > 0.0:
            raise ValueError(f"weak_threshold must be > 0 s^-2 or None, got {weak_threshold}")
        self.n_iface = int(n_iface)
        self.ri_bins = int(ri_bins)
        self.weak_threshold = None if weak_threshold is None else float(weak_threshold)
        ri_tokens = self.ri_bins + (0 if self.weak_threshold is None else 1)
        self.token_bins = {"ri_g": ri_tokens, "ri_b": ri_tokens, "n2_sign": 2, "forcing": 3}
        self.feature_dim = self.n_iface * len(INTERFACE_CHANNELS) + len(COLUMN_CHANNELS)
        self.head_dim = (self.n_iface * (len(KEPT_CHANNELS) + len(self.token_bins) * EMBED_DIM)
                         + len(COLUMN_CHANNELS))
        # Constants, not parameters: moved by .to(), not saved in the state_dict.
        self.register_buffer("ri_edges", ri_edges(bins=self.ri_bins), persistent=False)
        self.register_buffer("n2_edges", torch.zeros(1, dtype=torch.float64), persistent=False)
        b0 = math.asinh(ZERO_FLUX_BUOYANCY / _B_F_SCALE)
        self.register_buffer("forcing_edges", torch.tensor([-b0, b0], dtype=torch.float64),
                             persistent=False)
        if self.weak_threshold is not None:
            # |n2| and s2 below these are |N^2| and S^2 below weak_threshold.
            self.register_buffer("weak_edges", torch.tensor(
                [math.asinh(self.weak_threshold / _N2_SCALE), math.asinh(self.weak_threshold / _S2_SCALE)],
                dtype=torch.float64), persistent=False)
        self.register_buffer("kept_index", torch.tensor(
            [INTERFACE_CHANNELS.index(name) for name in KEPT_CHANNELS]), persistent=False)
        self.embed = nn.ModuleDict({name: nn.Embedding(bins, EMBED_DIM, dtype=torch.float64)
                                    for name, bins in self.token_bins.items()})
        self.head = LearnedKppClosure(self.n_iface, self.head_dim, hidden=hidden, depth=depth,
                                      k_cap=k_cap, tie_salt_to_temp=tie_salt_to_temp)

    @property
    def k_cap(self) -> float:
        return self.head.k_cap

    def _check(self, feats: ColumnFeatures) -> None:
        if feats.n_iface != self.n_iface or feats.dim != self.feature_dim:
            raise ValueError(f"features have n_iface={feats.n_iface}, dim={feats.dim}; the closure "
                             f"was built for n_iface={self.n_iface}, feature_dim={self.feature_dim}")

    def tokens(self, feats: ColumnFeatures) -> dict[str, torch.Tensor]:
        """Integer tokens: ri_g, ri_b, n2_sign (B, ny, nx, n_iface) and forcing (B, ny, nx)."""
        self._check(feats)
        n, c = self.n_iface, len(INTERFACE_CHANNELS)
        with torch.no_grad():
            x = feats.x.detach()
            iface = x[..., : n * c].unflatten(-1, (n, c))
            b_f = x[..., n * c + COLUMN_CHANNELS.index("b_f")]

            def channel(name: str) -> torch.Tensor:
                return iface[..., INTERFACE_CHANNELS.index(name)].contiguous()

            ri_g = torch.bucketize(channel("ri_g"), self.ri_edges)
            ri_b = torch.bucketize(channel("ri_b"), self.ri_edges)
            if self.weak_threshold is not None:
                weak = (channel("n2").abs() < self.weak_edges[0]) | (channel("s2") < self.weak_edges[1])
                ri_g = ri_g.masked_fill(weak, self.ri_bins)
                ri_b = ri_b.masked_fill(weak, self.ri_bins)
            return {
                "ri_g": ri_g,
                "ri_b": ri_b,
                "n2_sign": torch.bucketize(channel("n2"), self.n2_edges),
                "forcing": torch.bucketize(b_f.contiguous(), self.forcing_edges),
            }

    def head_input(self, feats: ColumnFeatures) -> torch.Tensor:
        """(B, ny, nx, head_dim): per interface the kept channels and four embeddings, then the column scalars."""
        tok = self.tokens(feats)
        n, c = self.n_iface, len(INTERFACE_CHANNELS)
        iface = feats.x[..., : n * c].unflatten(-1, (n, c))
        kept = iface.index_select(-1, self.kept_index)                       # (B, ny, nx, n, 8)
        forcing = self.embed["forcing"](tok["forcing"]).unsqueeze(-2)        # (B, ny, nx, 1, E)
        emb = torch.cat((self.embed["ri_g"](tok["ri_g"]), self.embed["ri_b"](tok["ri_b"]),
                         self.embed["n2_sign"](tok["n2_sign"]),
                         forcing.expand(*kept.shape[:-1], EMBED_DIM)), dim=-1)
        wet = (feats.wet_iface[:, 1:-1] > 0).permute(0, 2, 3, 1).unsqueeze(-1)
        emb = torch.where(wet, emb, torch.zeros_like(emb))
        return torch.cat((torch.cat((kept, emb), dim=-1).flatten(-2), feats.x[..., n * c:]), dim=-1)

    def forward(self, feats: ColumnFeatures, **kwargs) -> dict[str, torch.Tensor]:
        x = self.head_input(feats)
        return self.head(dataclasses.replace(feats, x=x, dim=x.shape[-1]), **kwargs)
