"""Learned replacement of the KPP diagnosis.

``LearnedKppClosure`` takes the place of ``kpp_calc_diffusivities`` (kpp.py:1510):
from the column features it returns the fields that driver returns, in the same
index conventions, so every consumer (implicit vertical viscosity and diffusion,
the explicit non-local tendency) is unchanged.

Index conventions (LOCOMgcm, k = 0 at the surface):

* ``diff_kzt``, ``diff_kzs``, ``visc_cell`` (B, nz, ny, nx): index k is the top
  interface of layer k; index 0 is the sea surface and is 0; the bottom
  interface of the last layer is not stored (kpp_calc.F:578-583).
* ``ghat`` (B, nz, ny, nx): index k is the bottom interface of layer k, in s/m^2
  (kpp_calc.F:584-585).  ``kpp_ghat_tendency`` (kpp_mixing.py:525-549) forms the
  non-local flux on interface k as ``diff_kz[k] * ghat[k-1] * sf``.
* ``gamma_iface`` (B, nz+1, ny, nx): dimensionless non-local fraction Gamma on
  interfaces 0..nz; for k < nz it is the quantity MITgcm writes as ``KPPghatK``
  at level k, ``KPPghat(k-1) * KPPdiffKzS(k)`` (kpp_output.F:204-216).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from defgcm.dynamics.kpp_mixing import KppFields
from defgcm.learned.features import COLUMN_CHANNELS, INTERFACE_CHANNELS, ColumnFeatures
from defgcm.learned.networks import ColumnConv, ColumnMLP
from defgcm.operators.reconstructions import center_to_u_centered, center_to_v_centered

ARCHITECTURES = ("mlp", "conv")

# Value of every sigmoid of the untrained closure.  The output bias of the
# coefficient and Gamma slots starts at its logit, so the untrained closure is
# close to background mixing: K - K_bg = 1e-5 * k_cap (5e-5 m^2/s at the default
# cap) and Gamma = 1e-5; the hbl slot starts at softplus(0) = ln 2 above min_hbl.
_INIT_FRACTION = 1.0e-5
_INIT_LOGIT = math.log(_INIT_FRACTION / (1.0 - _INIT_FRACTION))


def _full_interfaces(inner: torch.Tensor) -> torch.Tensor:
    """(B, nz-1, ny, nx) on interfaces 1..nz-1 -> (B, nz+1, ny, nx), 0 at 0 and nz."""
    return F.pad(inner, (0, 0, 0, 0, 1, 1))


class LearnedKppClosure(nn.Module):
    """Column network mapped to KPP mixing coefficients and non-local fraction.

    The network output of one column is, in order, ``a_T``, ``a_visc``, ``c``
    (and ``a_S`` when ``tie_salt_to_temp`` is false), each on the ``n_iface``
    interior interfaces, then one scalar ``e``.  With ``w`` the interface wet
    mask ``maskC(k) * maskC(k-1)``:

    * K_T = (K_bg,T + k_cap * sigmoid(a_T)) * w, visc = (visc_arnr +
      k_cap * sigmoid(a_visc)) * w, and K_S likewise from ``a_S``, or K_S = K_T
      when ``tie_salt_to_temp``;
    * Gamma = [B_f < 0] * sigmoid(c) * w;
    * all of them are 0 on interfaces 0 and nz;
    * ghat[k] = Gamma[k+1] / K_S[k+1] on wet interfaces, 0 elsewhere, so that
      the consumer's non-local flux ``K_S * ghat * sf`` equals ``Gamma * sf``;
    * hbl = min_hbl + softplus(e).

    ``background_diff_t``, ``background_diff_s`` and ``visc_arnr`` are the
    complete background values the regional stepper passes to the reference
    KPP (``tracer_vertical_diffusivity_kr`` and ``implicit_viscosity``), as
    floats or 0-d tensors.  They must be positive, or k_cap * sigmoid must stay
    above zero, for ghat to be finite on wet interfaces.

    ``architecture`` selects the column network:

    * ``"mlp"``: ``ColumnMLP`` on the flat feature vector ``feats.x``; the
      closure is built for one vertical grid (``n_iface``, ``feature_dim``).
    * ``"conv"``: ``ColumnConv`` with ``hidden`` channels, ``depth`` layers,
      ``kernel_size`` and ``dilation_base``; ``n_iface`` and ``feature_dim``
      must be None, since no weight depends on the number of interfaces.  The
      interface block of ``feats.x`` becomes ``len(INTERFACE_CHANNELS)``
      channels along the interior interfaces, the ``len(COLUMN_CHANNELS)``
      column scalars are repeated on every interface as further channels, dry
      interfaces are masked inside the network, and the hbl scalar is pooled
      over the wet interfaces weighted by the centre spacing ``dz_c`` (a depth
      average, so grids of different resolution weigh the column alike).  The
      per-interface output is laid out as the ``"mlp"`` output (slot-major,
      then the hbl scalar), so both share the mapping below.
    """

    def __init__(
        self,
        n_iface: int | None = None,
        feature_dim: int | None = None,
        hidden: int = 256,
        depth: int = 3,
        k_cap: float = 5.0,
        tie_salt_to_temp: bool = True,
        architecture: str = "mlp",
        kernel_size: int = 5,
        dilation_base: int = 1,
    ):
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {ARCHITECTURES}, got {architecture!r}")
        self.architecture = architecture
        self.k_cap = float(k_cap)
        self.tie_salt_to_temp = bool(tie_salt_to_temp)
        self.n_slots = 3 if self.tie_salt_to_temp else 4
        if architecture == "conv":
            if n_iface is not None or feature_dim is not None:
                raise ValueError("the conv closure does not depend on the number of "
                                 "interfaces; pass n_iface=None and feature_dim=None")
            self.n_iface = None
            self.feature_dim = None
            self.net = ColumnConv(
                len(INTERFACE_CHANNELS) + len(COLUMN_CHANNELS), self.n_slots, 1,
                hidden=hidden, depth=depth, kernel_size=kernel_size,
                dilation_base=dilation_base,
            )
            with torch.no_grad():
                self.net.out.bias[:] = _INIT_LOGIT
            return
        if n_iface is None or feature_dim is None:
            raise ValueError("the mlp closure needs n_iface and feature_dim")
        if n_iface < 1:
            raise ValueError(f"n_iface must be >= 1, got {n_iface}")
        self.n_iface = int(n_iface)
        self.feature_dim = int(feature_dim)
        self.net = ColumnMLP(
            self.feature_dim, self.n_slots * self.n_iface + 1, hidden=hidden, depth=depth,
        )
        with torch.no_grad():
            self.net.out.bias[: self.n_slots * self.n_iface] = _INIT_LOGIT

    def _conv_output(self, feats: ColumnFeatures) -> torch.Tensor:
        """``ColumnConv`` output in the ``ColumnMLP`` layout, (B, ny, nx, n_slots*n + 1)."""
        n = feats.n_iface
        c_iface, c_col = len(INTERFACE_CHANNELS), len(COLUMN_CHANNELS)
        if feats.dim != n * c_iface + c_col:
            raise ValueError(f"features have dim={feats.dim}, expected {n} interfaces x "
                             f"{c_iface} channels + {c_col} column scalars")
        batch, ny, nx, _ = feats.x.shape
        columns = batch * ny * nx
        # ColumnFeatures.x: interface k, channel c at k * c_iface + c (interface-major).
        iface = feats.x[..., : n * c_iface].reshape(columns, n, c_iface).transpose(1, 2)
        scalars = feats.x[..., n * c_iface:].reshape(columns, c_col, 1).expand(columns, c_col, n)
        wet = feats.wet_iface[:, 1:-1].permute(0, 2, 3, 1).reshape(columns, n) > 0
        dz_c = iface[:, INTERFACE_CHANNELS.index("dz_c")]
        weight = torch.where(wet, dz_c, torch.zeros_like(dz_c))
        per_iface, scalar = self.net(torch.cat((iface, scalars), dim=1), mask=wet,
                                     pool_weight=weight)
        # (N, n_slots, n) flattens slot-major, index s * n + k, as the MLP output.
        return torch.cat((per_iface.reshape(batch, ny, nx, self.n_slots * n),
                          scalar.reshape(batch, ny, nx, 1)), dim=-1)

    def forward(
        self,
        feats: ColumnFeatures,
        *,
        background_diff_t,
        background_diff_s,
        visc_arnr,
        min_hbl,
    ) -> dict[str, torch.Tensor]:
        if self.architecture == "mlp" and (
            feats.n_iface != self.n_iface or feats.dim != self.feature_dim
        ):
            raise ValueError(
                f"features have n_iface={feats.n_iface}, dim={feats.dim}; the closure "
                f"was built for n_iface={self.n_iface}, feature_dim={self.feature_dim}"
            )
        if (
            self.tie_salt_to_temp
            and not isinstance(background_diff_t, torch.Tensor)
            and not isinstance(background_diff_s, torch.Tensor)
            and float(background_diff_s) != float(background_diff_t)
        ):
            raise ValueError(
                "tie_salt_to_temp=True sets diff_kzs = diff_kzt, so background_diff_s "
                f"({background_diff_s}) must equal background_diff_t ({background_diff_t})"
            )
        n = feats.n_iface
        if self.architecture == "mlp":
            y = self.net(feats.x)                               # (B, ny, nx, n_slots*n + 1)
        else:
            y = self._conv_output(feats)
        slots = y[..., :-1].unflatten(-1, (self.n_slots, n))
        slots = torch.sigmoid(slots.permute(3, 0, 4, 1, 2))    # (n_slots, B, n, ny, nx)
        wet = feats.wet_iface[:, 1:-1]

        k_t = _full_interfaces((background_diff_t + self.k_cap * slots[0]) * wet)
        visc = _full_interfaces((visc_arnr + self.k_cap * slots[1]) * wet)
        gamma = _full_interfaces(feats.unstable.unsqueeze(1) * slots[2] * wet)
        if self.tie_salt_to_temp:
            k_s = k_t
        else:
            k_s = _full_interfaces((background_diff_s + self.k_cap * slots[3]) * wet)

        # ghat[k] sits on interface k+1; divide only on wet interfaces
        # (Gamma and K_S are both 0 on the others).
        wet_below = feats.wet_iface[:, 1:] > 0
        k_s_below = k_s[:, 1:]
        k_s_safe = torch.where(wet_below, k_s_below, torch.ones_like(k_s_below))
        ghat = torch.where(wet_below, gamma[:, 1:] / k_s_safe, torch.zeros_like(k_s_below))

        return {
            "diff_kzt": k_t[:, :-1],
            "diff_kzs": k_s[:, :-1],
            "visc_cell": visc[:, :-1],
            "ghat": ghat,
            "hbl": min_hbl + F.softplus(y[..., -1]),
            "gamma_iface": gamma,
        }


def to_kpp_fields(
    out: dict[str, torch.Tensor],
    sf_t: torch.Tensor,
    sf_s: torch.Tensor,
    sf_t_nl: torch.Tensor,
    *,
    grid,
    visc_arnr: float,
) -> KppFields:
    """Map the closure output to the ``KppFields`` the regional stepper consumes.

    ``kappa_ru``/``kappa_rv`` follow kpp_mixing.py:858-869: the two-point
    average of ``visc_cell`` onto the u/v faces, times the face wet mask
    (``grid.masks.u_float``/``v_float``), floored at ``visc_arnr`` as
    ``maximum(bg, bg - visc_arnr + mask * avg)``.  kpp_mixing.py has no
    standalone function for this, so the average comes from
    ``center_to_u_centered``/``center_to_v_centered``, whose halo is
    ``exchange()`` of the grid topology: the wrap neighbour on a periodic axis,
    the edge value otherwise.  On a periodic axis this is the reference halo.
    At an open edge the reference averages with the KPP output of its
    replicated halo column instead; at a wall the face mask is 0 either way.

    ``sf_t``, ``sf_s`` (B, ny, nx) are the surface forcing passed to the
    features; ``sf_t_nl`` is the complete temperature forcing of the non-local
    flux, ``kpp_mixing._nonlocal_temp_forcing(sf_t, qsw, out["hbl"], params)``
    in the reference (equal to ``sf_t`` without shortwave).  Regional
    (single-patch) grids only.
    """
    visc = out["visc_cell"]
    mask_w = grid.masks.u_float(dtype=visc.dtype, device=visc.device)
    mask_s = grid.masks.v_float(dtype=visc.dtype, device=visc.device)
    masked_ru = center_to_u_centered(visc, grid, face_mask=mask_w)
    bg_ru = torch.full_like(masked_ru, visc_arnr)
    kappa_ru = torch.maximum(bg_ru, bg_ru - visc_arnr + masked_ru)
    masked_rv = center_to_v_centered(visc, grid, face_mask=mask_s)
    bg_rv = torch.full_like(masked_rv, visc_arnr)
    kappa_rv = torch.maximum(bg_rv, bg_rv - visc_arnr + masked_rv)
    return KppFields(
        kappa_ru=kappa_ru,
        kappa_rv=kappa_rv,
        diff_kzt=out["diff_kzt"],
        diff_kzs=out["diff_kzs"],
        ghat=out["ghat"],
        hbl=out["hbl"],
        visc_cell=visc,
        surf_forc_t=sf_t,
        surf_forc_s=sf_s,
        surf_forc_t_nl=sf_t_nl,
    )
