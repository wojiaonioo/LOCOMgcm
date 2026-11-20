"""Hidden-label learning of the KPP closure on the h160 regional case.

The closure is trained only from temperature and
salinity observations through the LOCOMgcm model; the KPP diagnostics of the
truth run stay hidden until the end-of-stage evaluation.  This module holds the
pieces the h160 training and evaluation scripts share:

* ``H160ObservationOperator``: every ``stride``-th wet column in x and in y
  (full-depth T and S) plus the SST of every wet column, and the masks the
  statistics use (wet cells, the outer sponge ring);
* ``H160Truth``: loader of the recorded truth that opens only ``states/`` and
  ``obs/`` and refuses any hidden label by path or by key;
* ``H160Split``: training, validation and held-out time segments;
* ``observation_weights`` and ``observation_loss``;
* ``pp81_coefficients_h160``: PP81 viscosity and diffusivity as MITgcm
  ``pkg/pp81`` computes them, with the EOS pressure the model hands to its
  vertical mixing (JMD95P uses the dynamic pressure);
* ``RiAsinhClosure``: input transform of the ``ri_g`` and ``ri_b`` channels
  around a ``LearnedKppClosure`` (the atan scaling saturates on h160), with an
  optional zero probe on the diffusivity output for gradient-signal
  measurements;
* ``BackgroundClosure``: KPP switched off, background mixing only.

The single-column stage has its own module (``hidden.py``); nothing is
shared with it so that the two stages can change independently.  Truth layout
under one run directory: ``manifest.json``, ``static.npz``, ``states/step_XXXXXX.pt``,
``obs/step_XXXXXX.npz`` and ``hidden/step_XXXXXX.npz``.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from defgcm.dynamics.equation_of_state import reference_pressure_from_depth
from defgcm.dynamics.vertical_mixing_schemes import pp81 as pp81_scheme, richardson
from defgcm.geometry.vertical_coordinates import reference_layer_centres
from defgcm.learned.features import INTERFACE_CHANNELS, ColumnFeatures

#: Observation operator: every 4th wet column in x and in y.
OBS_STRIDE = 4
#: Outer rows and columns of the open-boundary sponge, excluded from statistics.
SPONGE_RING = 3
#: Arrays an observation frame may hold; anything else is refused on load.
OBS_KEYS = frozenset({"obs_j", "obs_i", "T", "S", "sst", "wet_c", "ring", "step",
                      "time_seconds"})
#: Names of the hidden KPP diagnostics (lower case); a key containing "kpp" is
#: refused as well.
HIDDEN_KEYS = frozenset({"diff_kzt", "diff_kzs", "visc_cell", "gamma_iface", "ghat",
                         "hbl", "kappa_ru", "kappa_rv"})
TRUTH_SUBDIRS = ("states", "obs")
HIDDEN_SUBDIR = "hidden"


class HiddenLabelError(RuntimeError):
    """A training-side code path asked for a hidden KPP label."""


def _is_hidden_key(name: str) -> bool:
    low = str(name).lower()
    return low in HIDDEN_KEYS or "kpp" in low


# ----------------------------------------------------------------------------- observations
@dataclass(frozen=True)
class H160ObservationOperator:
    """Which truth values are observed, and the masks of the statistics.

    ``obs_j``/``obs_i`` (n_obs,) are the wet columns (surface cell wet) with
    ``j % stride == offset[0]`` and ``i % stride == offset[1]``, ring columns
    included (the operator observes them; the loss weights drop them).
    ``wet_c`` (nz, ny, nx) is hFacC > 0; ``ring`` (ny, nx) is True on the
    outer ``ring_width`` rows and columns.
    """

    obs_j: torch.Tensor
    obs_i: torch.Tensor
    wet_c: torch.Tensor
    ring: torch.Tensor
    stride: int = OBS_STRIDE
    offset: tuple[int, int] = (0, 0)

    @classmethod
    def from_wet_mask(cls, wet_c: torch.Tensor, *, stride: int = OBS_STRIDE,
                      offset: tuple[int, int] = (0, 0),
                      ring_width: int = SPONGE_RING) -> "H160ObservationOperator":
        wet_c = wet_c.to(torch.bool).cpu()
        _, ny, nx = wet_c.shape
        pick = torch.zeros((ny, nx), dtype=torch.bool)
        pick[offset[0]::stride, offset[1]::stride] = True
        pick &= wet_c[0]
        obs_j, obs_i = torch.nonzero(pick, as_tuple=True)
        ring = torch.ones((ny, nx), dtype=torch.bool)
        ring[ring_width:ny - ring_width, ring_width:nx - ring_width] = False
        return cls(obs_j=obs_j, obs_i=obs_i, wet_c=wet_c, ring=ring,
                   stride=int(stride), offset=(int(offset[0]), int(offset[1])))

    def observe(self, temp: torch.Tensor, salt: torch.Tensor) -> dict[str, torch.Tensor]:
        """T and S (nz, n_obs) of the observed columns and SST (ny, nx) of one state.

        ``temp``/``salt`` are (nz, ny, nx) or (1, nz, ny, nx).  Dry cells keep
        whatever value the state holds; the weights of ``observation_weights``
        are 0 there.
        """
        if temp.ndim == 4:
            temp, salt = temp[0], salt[0]
        j = self.obs_j.to(temp.device)
        i = self.obs_i.to(temp.device)
        return {"T": temp[:, j, i], "S": salt[:, j, i], "sst": temp[0]}


def observation_weights(op: H160ObservationOperator, *, r_a: torch.Tensor,
                        hfac_c: torch.Tensor, drf: torch.Tensor) -> dict[str, torch.Tensor]:
    """Volume weights of the observed profiles and area weights of the SST, ring excluded.

    ``profile`` (nz, n_obs) = rA * hFacC * drF at the observed columns (0 on
    dry cells and ring columns); ``sst`` (ny, nx) = rA on wet surface cells
    outside the ring.  float64, on the device of ``r_a``.
    """
    r_a = r_a.to(torch.float64)
    hfac_c = hfac_c.to(dtype=torch.float64, device=r_a.device)
    drf = drf.to(dtype=torch.float64, device=r_a.device)
    keep = (~op.ring).to(r_a.device)
    j, i = op.obs_j.to(r_a.device), op.obs_i.to(r_a.device)
    profile = r_a[j, i] * hfac_c[:, j, i] * drf[:, None] * keep[j, i]
    sst = r_a * (hfac_c[0] > 0) * keep
    return {"profile": profile, "sst": sst}


def observation_loss(pred: Mapping[str, torch.Tensor], obs: Mapping[str, torch.Tensor], *,
                     weights: Mapping[str, torch.Tensor], sigma: Mapping[str, float],
                     var_weight: Mapping[str, float]) -> torch.Tensor:
    """Weighted mean squared error of one observation time, each term divided by sigma^2.

    ``L = sum_v var_weight[v] * sum(w_v (pred_v - obs_v)^2) / (sigma_v^2 sum(w_v))``
    for v in ``var_weight`` (``T`` and ``S`` with the profile weights, ``sst``
    with the SST weights).  A variable with ``sigma`` below 1e-12 is skipped.
    """
    total = None
    for name, factor in var_weight.items():
        if float(sigma[name]) < 1e-12 or float(factor) == 0.0:
            continue
        w = weights["sst" if name == "sst" else "profile"]
        err = pred[name] - obs[name]
        term = float(factor) * (w * err * err).sum() / (float(sigma[name]) ** 2 * w.sum())
        total = term if total is None else total + term
    if total is None:
        raise ValueError("observation_loss: every variable is skipped")
    return total


# ----------------------------------------------------------------------------- truth loader
def _refuse_hidden_path(root: Path, path: Path) -> None:
    """Raise unless ``path`` lies in root/states or root/obs (symlinks resolved)."""
    resolved = Path(path).resolve()
    allowed = [(Path(root) / sub).resolve() for sub in TRUTH_SUBDIRS]
    hidden = (Path(root) / HIDDEN_SUBDIR).resolve()
    if resolved == hidden or hidden in resolved.parents:
        raise HiddenLabelError(f"{path} is a hidden KPP label; training code may not open it")
    if not any(base in resolved.parents for base in allowed):
        raise HiddenLabelError(f"{path} is outside {TRUTH_SUBDIRS} of {root}")
    if HIDDEN_SUBDIR in Path(path).parts:
        raise HiddenLabelError(f"{path} names the hidden directory")


class H160Truth:
    """Recorded h160 truth, restricted to full states and observations.

    ``state(step)`` returns the ``ModelCheckpoint`` of the state after
    ``step`` steps (step_state limited to the checkpoint restart allowlist, so
    it holds no KPP output); ``obs(step)`` the observation frame.  Both refuse
    any path under ``hidden/`` and any array name of a KPP diagnostic
    (``HiddenLabelError``).
    """

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        for key in self.manifest:
            if _is_hidden_key(key):
                raise HiddenLabelError(f"manifest key {key!r} names a KPP diagnostic")
        self.every = int(self.manifest["every"])
        self.steps = int(self.manifest["steps"])
        self.dt = float(self.manifest["dt_seconds"])

    def path(self, sub: str, step: int) -> Path:
        if sub not in TRUTH_SUBDIRS:
            raise HiddenLabelError(f"subdirectory {sub!r} is not one of {TRUTH_SUBDIRS}")
        suffix = ".pt" if sub == "states" else ".npz"
        path = self.root / sub / f"step_{int(step):06d}{suffix}"
        _refuse_hidden_path(self.root, path)
        return path

    def state(self, step: int):
        from defgcm.checkpoint import RESTART_STATE_KEYS_V5
        from defgcm.checkpoint.types import PROGNOSTIC_FIELDS

        ckpt = torch.load(self.path("states", step), map_location="cpu", weights_only=False)
        extra = set(ckpt.step_state) - set(RESTART_STATE_KEYS_V5)
        bad = [k for k in (*ckpt.prognostic, *ckpt.step_state) if _is_hidden_key(k)]
        if extra or bad or set(ckpt.prognostic) != set(PROGNOSTIC_FIELDS):
            raise HiddenLabelError(f"state {step}: unexpected keys {sorted(extra | set(bad))}")
        if ckpt.meta.global_step != int(step):
            raise ValueError(f"states/step_{step:06d}.pt holds step {ckpt.meta.global_step}")
        return ckpt

    def obs(self, step: int) -> dict[str, np.ndarray]:
        with np.load(self.path("obs", step)) as f:
            names = list(f.files)
            bad = [k for k in names if _is_hidden_key(k) or k not in OBS_KEYS]
            if bad:
                raise HiddenLabelError(f"obs {step}: refused keys {bad}")
            out = {k: f[k] for k in names}
        if int(out["step"]) != int(step):
            raise ValueError(f"obs/step_{step:06d}.npz holds step {int(out['step'])}")
        return out

    def static(self) -> dict[str, np.ndarray]:
        with np.load(self.root / "static.npz") as f:
            bad = [k for k in f.files if _is_hidden_key(k)]
            if bad:
                raise HiddenLabelError(f"static.npz: refused keys {bad}")
            return {k: f[k] for k in f.files}


# ----------------------------------------------------------------------------- split
@dataclass(frozen=True)
class H160Split:
    """Three consecutive time segments in steps: (0, t1], (t1, t2], (t2, total].

    A window of n steps from ``start`` belongs to a segment when start and
    start + n both lie in its closed interval; its observation times are the
    multiples of ``every`` in (start, start + n], so no observation of one
    segment enters the loss of another.  The state at a boundary is the last
    observation of one segment and the initial state of the next.
    """

    total_steps: int
    t1: int
    t2: int
    every: int = 60

    @classmethod
    def for_days(cls, days: int, *, dt: float = 120.0, every: int = 60) -> "H160Split":
        per_day = int(round(86400.0 / dt))
        table = {10: (6, 8), 5: (3, 4)}      # days of train / validation ends
        if days not in table:
            raise ValueError(f"no split for {days} days")
        a, b = table[days]
        return cls(total_steps=days * per_day, t1=a * per_day, t2=b * per_day, every=every)

    def __post_init__(self):
        if not 0 < self.t1 < self.t2 < self.total_steps:
            raise ValueError(f"segments must satisfy 0 < {self.t1} < {self.t2} < {self.total_steps}")
        if any(t % self.every for t in (self.t1, self.t2, self.total_steps)):
            raise ValueError("segment boundaries must be observation times")

    def segments(self) -> dict[str, tuple[int, int]]:
        return {"train": (0, self.t1), "validation": (self.t1, self.t2),
                "holdout": (self.t2, self.total_steps)}

    def fraction(self, name: str) -> float:
        a, b = self.segments()[name]
        return (b - a) / self.total_steps

    def contains(self, name: str, start: int, n_steps: int) -> bool:
        a, b = self.segments()[name]
        return a <= start and start + n_steps <= b and n_steps > 0

    def observation_steps(self, start: int, n_steps: int) -> list[int]:
        first = (start // self.every + 1) * self.every
        return list(range(first, start + n_steps + 1, self.every))

    def window_starts(self, name: str, n_steps: int, *, stride: int | None = None) -> list[int]:
        """Recorded-state starts (multiples of ``every``) of whole windows in segment ``name``."""
        a, b = self.segments()[name]
        stride = self.every if stride is None else int(stride)
        return [s for s in range(a, b - n_steps + 1, stride)
                if s % self.every == 0 and self.contains(name, s, n_steps)
                and self.observation_steps(s, n_steps)]


# ----------------------------------------------------------------------------- PP81
@dataclass(frozen=True)
class Pp81H160Params:
    """PP81 parameters: ``pp81_readparms.F`` defaults plus the case background.

    ``visc_background`` is viscArNr, ``diff_background_t``/``_s`` diffKrNrT and
    diffKrNrS of the case; ``eos`` is ``"jmd95"`` (JMD95Z/JMD95P) or
    ``"mdjwf"``.  RiLimit is set as ``pp81_readparms.F:118-122`` does when
    it is not given.
    """

    visc_background: float
    diff_background_t: float
    diff_background_s: float
    rho_const: float
    gravity: float
    eos: str = "jmd95"
    nu0: float = 1.0e-2
    alpha: float = 5.0
    n_ri: int = 2
    visc_max: float = 1.0


def _density_kernel(eos: str):
    from defgcm.dynamics.equation_of_state import jmd95_density_anomaly, mdjwf_density_anomaly

    name = str(eos).strip().lower()
    if name in ("jmd95", "jmd95z", "jmd95p"):
        return jmd95_density_anomaly
    if name == "mdjwf":
        return mdjwf_density_anomaly
    raise ValueError(f"unsupported EOS {eos!r}")


def pp81_coefficients_h160(
    temp: torch.Tensor,
    salt: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    pressure_pa: torch.Tensor | None,
    *,
    drf: torch.Tensor,
    mask_c: torch.Tensor,
    params: Pp81H160Params,
) -> dict[str, torch.Tensor]:
    """PP81 viscosity and diffusivities (``pp81_ri_number.F``, ``pp81_calc.F``).

    Shapes: ``temp``, ``salt`` (B, nz, ny, nx); ``u`` (B, nz, ny, nx+1) and
    ``v`` (B, nz, ny+1, nx) with both outer faces; ``pressure_pa`` the EOS
    pressure of every layer in Pa, (B, nz, ny, nx), (nz, ny, nx) or (nz,), or
    None for the static rhoConst * g * z_c; ``drf`` (nz,); ``mask_c``
    (nz, ny, nx).  MITgcm level K = 2..Nr is interface k = K - 1 (top of
    layer k).

    Ri: densities of layers k-1 and k both at the pressure of layer k
    (``FIND_RHO_2D`` with kRef = K), N^2 = -g / rhoConst (rho_{k-1} - rho_k) /
    drC(K); shear of the cell-centred velocity differenced over drC(K); Ri =
    N^2 / max(shear^2, 1e-10).  Below RiLimit the viscosity is PPviscMax and
    the denominator uses RiLimit, otherwise PPnu0 / (1 + PPalpha Ri)^PPnRi;
    visc = max(., viscArNr); diff = max(visc / denominator, diffKrNrS); both
    times maskC(K) (``ALLOW_PP81_LOWERBOUND`` and ``PP81_SMOOTH_RI`` are off
    in ``PP81_OPTIONS.h``).

    Returned in the closure convention (index k = top interface of layer k, 0
    at k = 0): ``visc_cell`` = PPviscAr, ``diff_kzt`` = diffKrNrT + (PPdiffKr
    - diffKrNrS), ``diff_kzs`` = PPdiffKr (the totals ``pp81_calc_diff.F``
    forms on top of the background), and ``ri``.
    """
    if params.n_ri < 1 or int(params.n_ri) != params.n_ri:
        raise ValueError(f"PPnRi must be a positive integer, got {params.n_ri}")
    batch, nz, ny, nx = temp.shape
    dtype, device = temp.dtype, temp.device
    drf = drf.to(dtype=dtype, device=device)
    drc, zc = reference_layer_centres(drf)
    recip_drc = torch.cat((drc.new_ones(1), drc[1:].reciprocal()))
    if pressure_pa is None:
        p = reference_pressure_from_depth(
            zc, rho_const=params.rho_const, gravity=params.gravity,
        ).reshape(1, nz, 1, 1)
    else:
        p = pressure_pa.to(dtype=dtype, device=device)
        if p.ndim == 1:
            p = p.reshape(1, nz, 1, 1)
        elif p.ndim == 3:
            p = p.unsqueeze(0)
    density = _density_kernel(params.eos)
    mask_c = mask_c.to(dtype=dtype, device=device)
    minus_n2 = richardson.negative_buoyancy_frequency(
        density(temp, salt, p, reference_density=params.rho_const),
        density(temp[:, :-1], salt[:, :-1], p[:, 1:], reference_density=params.rho_const),
        recip_drc=recip_drc, gravity=params.gravity, mass2r_unit=1.0 / params.rho_const)
    ri = richardson.richardson_number(
        minus_n2, richardson.vertical_shear_squared(u, v, recip_drc=recip_drc))
    n_ri = int(params.n_ri)
    visc, diff = pp81_scheme.pp81_coefficients(
        ri, mask_c=mask_c, alpha=params.alpha, nu0=params.nu0, n_ri=n_ri, visc_max=params.visc_max,
        ri_limit=pp81_scheme.pp81_ri_limit(nu0=params.nu0, visc_arnr_surface=params.visc_background,
                                           visc_max=params.visc_max, n_ri=n_ri, alpha=params.alpha),
        visc_arnr=drf.new_full((nz,), params.visc_background),
        diffkr_nr_s=drf.new_full((nz,), params.diff_background_s))
    visc, diff = visc[:, 1:], diff[:, 1:]
    mask = mask_c[1:].expand(batch, nz - 1, ny, nx)

    def full(inner: torch.Tensor) -> torch.Tensor:
        return torch.cat((inner.new_zeros(batch, 1, ny, nx), inner), dim=1)

    kzt = (params.diff_background_t + (diff - params.diff_background_s)) * mask
    return {"visc_cell": full(visc), "diff_kzs": full(diff), "diff_kzt": full(kzt),
            "ri": full(ri * mask)}


# ----------------------------------------------------------------------------- closures
RI_CHANNELS = ("ri_g", "ri_b")


def ri_asinh_features(feats: ColumnFeatures, scale: float) -> ColumnFeatures:
    """Replace the ``ri_g`` and ``ri_b`` channels x = atan(Ri) by asinh(tan(x) / scale).

    Dry interfaces hold 0, which maps to 0.  float64 recovers Ri from its
    atan to about 1e-8 relative up to Ri ~ 1e8 (the largest N^2 / 1e-10 on
    h160), so no information is lost by inverting the saturated channel.
    """
    n, c = feats.n_iface, len(INTERFACE_CHANNELS)
    x = feats.x
    iface = x[..., : n * c].unflatten(-1, (n, c))
    idx = [INTERFACE_CHANNELS.index(name) for name in RI_CHANNELS]
    cols = []
    for ch in range(c):
        col = iface[..., ch]
        if ch in idx:
            col = torch.asinh(torch.tan(col) / scale)
        cols.append(col)
    new_iface = torch.stack(cols, dim=-1).flatten(-2)
    return replace(feats, x=torch.cat((new_iface, x[..., n * c:]), dim=-1))


class RiAsinhClosure(nn.Module):
    """A ``LearnedKppClosure`` whose ri channels are re-scaled first (``ri_asinh_features``).

    ``probe`` (None or a (1, nz, ny, nx) tensor, normally zeros with
    requires_grad) is added to ``diff_kzt`` and ``diff_kzs`` of every call, so
    its gradient is the sum over steps of the loss gradient with respect to
    the closure's diffusivity output (tied salt: both consumers).

    ``detach_inputs = True`` detaches the features before the network: the
    forward values are unchanged, and in the backward pass the gradient
    reaches the closure parameters through the dynamical core only, not
    through the dependence of the closure output on the state.
    """

    def __init__(self, inner: nn.Module, scale: float = 1.0):
        super().__init__()
        self.inner = inner
        self.scale = float(scale)
        self.probe: torch.Tensor | None = None
        self.detach_inputs = False

    @property
    def n_iface(self):
        return self.inner.n_iface

    @property
    def feature_dim(self):
        return self.inner.feature_dim

    def forward(self, feats: ColumnFeatures, **kwargs) -> dict[str, torch.Tensor]:
        if self.detach_inputs:
            feats = replace(feats, x=feats.x.detach())
        out = self.inner(ri_asinh_features(feats, self.scale), **kwargs)
        if self.probe is not None:
            out = dict(out)
            out["diff_kzt"] = out["diff_kzt"] + self.probe
            out["diff_kzs"] = out["diff_kzs"] + self.probe
        return out

    def raw_slots(self, feats: ColumnFeatures) -> torch.Tensor:
        """sigmoid of the network's coefficient slots, (n_slots, B, n_iface, ny, nx) (MLP only)."""
        y = self.inner.net(ri_asinh_features(feats, self.scale).x)
        slots = y[..., :-1].unflatten(-1, (self.inner.n_slots, feats.n_iface))
        return torch.sigmoid(slots.permute(3, 0, 4, 1, 2))


class BackgroundClosure(nn.Module):
    """KPP switched off: background diffusivity and viscosity, Gamma = 0, hbl = min_hbl.

    Same output layout as ``LearnedKppClosure``; wet interfaces only.
    """

    def forward(self, feats: ColumnFeatures, *, background_diff_t, background_diff_s,
                visc_arnr, min_hbl) -> dict[str, torch.Tensor]:
        wet = feats.wet_iface[:, :-1]                       # (B, nz, ny, nx), 0 at k = 0
        batch, nz, ny, nx = wet.shape
        return {
            "diff_kzt": background_diff_t * wet,
            "diff_kzs": background_diff_s * wet,
            "visc_cell": visc_arnr * wet,
            "ghat": torch.zeros_like(wet),
            "hbl": torch.full((batch, ny, nx), float(min_hbl), dtype=wet.dtype, device=wet.device),
            "gamma_iface": torch.zeros_like(feats.wet_iface),
        }


# ----------------------------------------------------------------------------- misc
def weighted_quantile(values: torch.Tensor, weights: torch.Tensor, q: float) -> float:
    """Quantile ``q`` of ``values`` under non-negative ``weights`` (lower interpolation)."""
    v = values.reshape(-1).double()
    w = weights.reshape(-1).double()
    keep = w > 0
    v, w = v[keep], w[keep]
    if v.numel() == 0:
        return math.nan
    order = torch.argsort(v)
    v, w = v[order], w[order]
    cum = torch.cumsum(w, 0)
    idx = int(torch.searchsorted(cum, q * cum[-1]).clamp(max=v.numel() - 1))
    return float(v[idx])


def interior_wet_interfaces(wet_c: torch.Tensor, ring: torch.Tensor) -> torch.Tensor:
    """maskC(k) * maskC(k-1) on the top interface of layer k (0 at k = 0), ring excluded; (nz, ny, nx)."""
    wet = wet_c.to(torch.bool)
    out = torch.zeros_like(wet)
    out[1:] = wet[1:] & wet[:-1]
    return out & ~ring.to(wet.device)


__all__ = [
    "BackgroundClosure", "H160ObservationOperator", "H160Split", "H160Truth",
    "HiddenLabelError", "OBS_KEYS", "OBS_STRIDE", "Pp81H160Params", "RiAsinhClosure",
    "SPONGE_RING", "interior_wet_interfaces", "observation_loss", "observation_weights",
    "pp81_coefficients_h160", "ri_asinh_features", "weighted_quantile",
]
