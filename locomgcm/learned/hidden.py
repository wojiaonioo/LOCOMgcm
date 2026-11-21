"""Hidden-label learning of the KPP closure: data layer, observation operator, PP81.

The closure is learned from temperature and salinity
observations only.  Every KPP diagnostic of the truth run (KPPdiffT,
KPPdiffS, KPPviscA, KPPghat, KPPghatK, KPPhbl and whatever derives from them)
is a hidden label: the training, fitting and window-measurement entry points
never read it, and the evaluation script reads it once, at the end of a stage,
for a frozen checkpoint.  The reference KPP of the twin is hidden in the same
way, since its output is the same kind of label.

Pieces shared by those entry points:

* ``HiddenLabelError`` and ``install_hidden_label_guard``: process-wide
  refusal of ``MitTruth.kpp``, of the ``kpp3d``/``kpp2d`` files and of the
  reference ``RegionalKppInputs.diagnose`` (a ``LearnedKppInjection`` still
  replaces the latter inside its ``with`` block);
* ``HiddenTruth``: the truth as the training side sees it: full states at
  window starts (pickups), T and S observations every ``obs_every`` steps,
  the MITgcm surface heat flux Qnet for grouping windows;
* ``TimeSplit``: three disjoint segments of consecutive iterations;
  ``regime_window_starts``: windows whose length follows the forcing regime;
* ``observation_loss``, ``observation_sigma``, ``observation_rollout`` (with
  optional gradient truncation every ``grad_segment`` steps);
* ``ClosureTap``: keeps the features and output of every closure call;
  ``FrozenInputClosure``: the closure seen by autograd as a function of its
  parameters only (features detached);
* ``pp81_coefficients``: Pacanowski and Philander (1981) as MITgcm
  ``pkg/pp81`` computes it, the starting closure of the online training.
"""
from __future__ import annotations

import contextlib
import dataclasses
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from locomgcm.geometry.vertical_coordinates import mit_layer_centres, reference_layer_centres
from locomgcm.dynamics.equation_of_state import reference_pressure_from_depth
from locomgcm.dynamics.vertical_mixing_schemes import pp81 as pp81_scheme, richardson
from locomgcm.learned.data import MitTruth
from locomgcm.learned.hybrid import features_from_inputs as standard_features_from_inputs
from locomgcm.learned.losses import trajectory_loss

#: Variables of the observation operator: key -> MITgcm pickup field.
OBSERVED_FIELDS = {"temp": "Theta", "salt": "Salt"}
#: A variable whose truth standard deviation is below this is constant and
#: carries no observation (case_3 salinity).
SIGMA_FLOOR = 1.0e-12
#: Truth files that hold only KPP diagnostics.
KPP_PREFIXES = ("kpp3d", "kpp2d")


class HiddenLabelError(RuntimeError):
    """A hidden KPP label was requested on the training side."""


def _is_kpp_key(name: str) -> bool:
    return str(name).upper().startswith("KPP")


# ----------------------------------------------------------------------------- guard
def install_hidden_label_guard() -> Callable[[], None]:
    """Refuse every KPP label in this process; returns the undo function.

    Replaces ``MitTruth.kpp`` by a function that raises ``HiddenLabelError``,
    wraps ``MitTruth.fields`` so that the ``kpp3d``/``kpp2d`` prefixes and
    any ``KPP*`` field raise, and replaces the reference
    ``RegionalKppInputs.diagnose`` by a function that raises.  A
    ``LearnedKppInjection`` saves and restores whatever ``diagnose`` it finds,
    so inside its block the learned closure runs and outside it the reference
    KPP stays refused.  Entry points call this once at start and never undo;
    the returned function exists for tests.
    """
    from locomgcm.dynamics.kpp_mixing import RegionalKppInputs

    original_kpp = MitTruth.kpp
    original_fields = MitTruth.fields
    original_diagnose = RegionalKppInputs.diagnose

    def kpp(self, it):
        raise HiddenLabelError(f"MitTruth.kpp({it}) reads hidden KPP labels")

    def fields(self, prefix, it):
        if prefix in KPP_PREFIXES:
            raise HiddenLabelError(f"MitTruth.fields({prefix!r}, {it}) reads hidden KPP labels")
        out = original_fields(self, prefix, it)
        leaked = [name for name in out if _is_kpp_key(name)]
        if leaked:
            raise HiddenLabelError(f"{prefix} file at {it} carries KPP fields {leaked}")
        return out

    def diagnose(self, *args, **kwargs):
        raise HiddenLabelError("the reference KPP diagnosis is a hidden label; run the "
                               "model inside a LearnedKppInjection")

    MitTruth.kpp = kpp  # type: ignore[method-assign]
    MitTruth.fields = fields  # type: ignore[method-assign]
    RegionalKppInputs.diagnose = diagnose  # type: ignore[method-assign]

    def undo() -> None:
        MitTruth.kpp = original_kpp  # type: ignore[method-assign]
        MitTruth.fields = original_fields  # type: ignore[method-assign]
        RegionalKppInputs.diagnose = original_diagnose  # type: ignore[method-assign]

    return undo


# ----------------------------------------------------------------------------- truth
class _StateFields(dict):
    """Fields of one pickup; asking for a KPP key raises ``HiddenLabelError``."""

    def __getitem__(self, name):
        if _is_kpp_key(name):
            raise HiddenLabelError(f"field {name!r} is a hidden KPP label")
        return super().__getitem__(name)

    def get(self, name, default=None):
        if _is_kpp_key(name):
            raise HiddenLabelError(f"field {name!r} is a hidden KPP label")
        return super().get(name, default)


_NAMELIST_VALUE = r"^\s*{}\s*=\s*'?([^,'\n]+)'?\s*,?\s*$"


def _namelist_value(text: str, name: str) -> str:
    match = re.search(_NAMELIST_VALUE.format(name), text, flags=re.MULTILINE | re.IGNORECASE)
    if match is None:
        raise KeyError(f"{name} not set in the MITgcm data namelist")
    return match.group(1).strip()


def _fortran_float(token: str) -> float:
    return float(token.lower().replace("d", "e").rstrip("."))


class HiddenTruth:
    """The truth run as the training side may see it.

    ``pickup(it)`` (and ``fields("pickup", it)``, which
    ``training.state_from_pickup`` calls) returns the full state of pickup
    ``it``: prognostic fields and Adams-Bashforth history, the window start.
    ``observations(it)`` returns T and S at an observation instant, a
    multiple of ``obs_every``.  ``qnet(t)`` is the MITgcm surface heat flux
    (W/m^2, positive = the ocean loses heat) at model time ``t`` seconds,
    read from ``surfQnetFile`` of the run's ``data`` namelist and
    interpolated as MITgcm ``GET_PERIODIC_INTERVAL`` does (record n centred
    at (n - 1/2) * externForcingPeriod, linear, cyclic over
    externForcingCycle).  ``kpp`` and the ``kpp3d``/``kpp2d`` files raise
    ``HiddenLabelError``; a pickup that carried a ``KPP*`` field would raise
    as well.
    """

    def __init__(self, truth: MitTruth, *, obs_every: int):
        if int(obs_every) < 1:
            raise ValueError(f"obs_every must be >= 1, got {obs_every}")
        self._truth = truth
        self.run_dir = truth.run_dir
        self.nx, self.ny, self.nz = truth.nx, truth.ny, truth.nz
        self.obs_every = int(obs_every)
        spacing = {b - a for a, b in zip(truth.iters("pickup"), truth.iters("pickup")[1:])}
        if len(spacing) != 1 or self.obs_every % spacing.pop():
            raise ValueError(f"obs_every={obs_every} is not a multiple of the pickup spacing")
        text = (self.run_dir / "data").read_text()
        self.dt = _fortran_float(_namelist_value(text, "deltaT"))
        self.forcing_period = _fortran_float(_namelist_value(text, "externForcingPeriod"))
        self.forcing_cycle = _fortran_float(_namelist_value(text, "externForcingCycle"))
        qnet_file = _namelist_value(text, "surfQnetFile")
        n_rec = int(round(self.forcing_cycle / self.forcing_period))
        records = np.fromfile(self.run_dir / qnet_file, dtype=">f8").astype(np.float64)
        records = records.reshape(-1, self.ny, self.nx)[:n_rec]
        # GET_PERIODIC_INTERVAL as two extra records around the cycle, so one
        # linear interpolation over t mod cycle covers the seam
        # (locomgcm/io/runner/layer_b/sources/mit_input_binary.py, _forcing_record_times).
        self._qnet_times = np.concatenate([[-0.5 * self.forcing_period],
                                           (np.arange(n_rec) + 0.5) * self.forcing_period,
                                           [(n_rec + 0.5) * self.forcing_period]])
        self._qnet_records = np.concatenate([records[-1:], records, records[:1]])

    # --- states
    def iters(self, prefix: str = "pickup") -> list[int]:
        if prefix != "pickup":
            raise HiddenLabelError(f"iterations of {prefix!r} files are hidden-label bookkeeping")
        return self._truth.iters("pickup")

    def window(self, it0: int, n_out: int) -> list[int]:
        return self._truth.window(it0, n_out)

    def fields(self, prefix: str, it: int) -> _StateFields:
        if prefix != "pickup":
            raise HiddenLabelError(f"{prefix} files hold hidden KPP labels")
        out = self._truth.fields("pickup", it)
        leaked = [name for name in out if _is_kpp_key(name)]
        if leaked:
            raise HiddenLabelError(f"pickup {it} carries KPP fields {leaked}")
        return _StateFields(out)

    def pickup(self, it: int) -> _StateFields:
        return self.fields("pickup", it)

    def observations(self, it: int) -> dict[str, torch.Tensor]:
        """T and S (nz, ny, nx) at observation instant ``it``."""
        if int(it) % self.obs_every:
            raise ValueError(f"iteration {it} is not an observation instant "
                             f"(every {self.obs_every} steps)")
        fields = self.fields("pickup", it)
        return {key: fields[name] for key, name in OBSERVED_FIELDS.items()}

    def observation_times(self, it0: int, n_steps: int) -> list[int]:
        """Observation instants after ``it0`` up to ``it0 + n_steps`` (both multiples)."""
        if int(it0) % self.obs_every or int(n_steps) % self.obs_every or n_steps < 1:
            raise ValueError(f"window ({it0}, {n_steps}) does not start and end on "
                             f"observation instants (every {self.obs_every} steps)")
        return list(range(int(it0) + self.obs_every, int(it0) + int(n_steps) + 1, self.obs_every))

    def kpp(self, it: int):
        raise HiddenLabelError(f"KPP diagnostics of iteration {it} are hidden labels")

    # --- forcing
    def qnet(self, time_seconds: float) -> np.ndarray:
        """MITgcm Qnet (ny, nx) in W/m^2 at model time ``time_seconds`` (> 0 cools)."""
        phase = float(time_seconds) % self.forcing_cycle
        k = int(np.searchsorted(self._qnet_times, phase, side="right")) - 1
        k = min(max(k, 0), len(self._qnet_times) - 2)
        t0, t1 = self._qnet_times[k], self._qnet_times[k + 1]
        w = (phase - t0) / (t1 - t0)
        return (1.0 - w) * self._qnet_records[k] + w * self._qnet_records[k + 1]

    def mean_qnet(self, it0: int, n_steps: int) -> float:
        """Domain and window mean of Qnet over the step start times it0 .. it0+n_steps-1."""
        return float(np.mean([self.qnet((int(it0) + n) * self.dt).mean() for n in range(int(n_steps))]))

    def domain_qnet(self, it_first: int, it_last: int) -> np.ndarray:
        """Domain-mean Qnet (W/m^2, > 0 cools) at the step start times it_first .. it_last-1."""
        return np.array([self.qnet(it * self.dt).mean() for it in range(int(it_first), int(it_last))])


# ----------------------------------------------------------------------------- split
@dataclass(frozen=True)
class TimeSplit:
    """Three disjoint segments of iterations, each ``(first, last)`` inclusive.

    A window of ``n_steps`` from ``it0`` belongs to a segment when its start
    and every observation instant lie in it: ``first <= it0`` and
    ``it0 + n_steps <= last``.  Disjoint means no instant is in two
    segments (``last`` of one < ``first`` of the next), so a window of one
    segment never shares its start or an observation with a window of another.
    """

    train: tuple[int, int]
    validation: tuple[int, int]
    holdout: tuple[int, int]

    def __post_init__(self):
        segments = sorted(self.segments().items(), key=lambda kv: kv[1][0])
        for name, (first, last) in segments:
            if not (0 <= first < last):
                raise ValueError(f"segment {name} = ({first}, {last}) is empty or negative")
        for (name_a, (_, last_a)), (name_b, (first_b, _)) in zip(segments, segments[1:]):
            if last_a >= first_b:
                raise ValueError(f"segments {name_a} and {name_b} overlap: "
                                 f"{name_a} ends at {last_a}, {name_b} starts at {first_b}")

    @classmethod
    def from_config(cls, cfg: Mapping) -> "TimeSplit":
        return cls(**{name: (int(cfg[name][0]), int(cfg[name][1]))
                      for name in ("train", "validation", "holdout")})

    def segments(self) -> dict[str, tuple[int, int]]:
        return {"train": self.train, "validation": self.validation, "holdout": self.holdout}

    def segment_of(self, it: int) -> str | None:
        for name, (first, last) in self.segments().items():
            if first <= int(it) <= last:
                return name
        return None

    def fraction(self, name: str, total_steps: int) -> float:
        first, last = self.segments()[name]
        return (last - first) / float(total_steps)

    def window_starts(self, name: str, n_steps: int, *, every: int, stride: int | None = None,
                      pickups: Sequence[int] | None = None) -> list[int]:
        """Starts of ``n_steps`` windows inside segment ``name``, ascending.

        Starts are multiples of ``every`` (the observation interval) with a
        pickup when ``pickups`` is given.  ``stride`` keeps a start only when
        it is at least ``stride`` steps after the previous kept one (no
        overlap for ``stride = n_steps``).
        """
        first, last = self.segments()[name]
        have = None if pickups is None else set(int(p) for p in pickups)
        out: list[int] = []
        it0 = -(-first // every) * every
        while it0 + n_steps <= last:
            if (have is None or it0 in have) and (stride is None or not out or it0 >= out[-1] + stride):
                out.append(it0)
            it0 += every
        return out


CONVECTIVE = "convective"
NON_CONVECTIVE = "non_convective"


def regime_window_starts(truth: HiddenTruth, split: TimeSplit, name: str, *, steps: int,
                         convective_steps: int,
                         pickups: Sequence[int] | None = None) -> list[tuple[int, int, str]]:
    """``(it0, n_steps, regime)`` of the windows of segment ``name``, the length set by forcing.

    Every observation instant of the segment (with a pickup when ``pickups``
    is given) may start one window.  The start is ``non_convective`` when the
    domain-mean MITgcm Qnet is <= 0 (no cooling) at every step start time of
    its ``steps``-step window, and the window then has ``steps`` steps.  A
    start whose ``steps``-step window meets a cooling step (Qnet > 0) is
    ``convective`` and gets ``convective_steps``: the gradient of a long
    window is valid in the non-convective regime only, so a non-convective start just before the cooling
    onset takes the convective length too.  A window that does not fit
    before the segment end is dropped.  Every kept window lies in the
    segment (``it0 + n_steps <= last``).
    """
    first, last = split.segments()[name]
    every = truth.obs_every
    steps, convective_steps = int(steps), int(convective_steps)
    # cooling[i] = number of cooling step start times in first .. first+i-1
    cooling = np.concatenate([[0], np.cumsum(truth.domain_qnet(first, last) > 0.0)])
    have = None if pickups is None else set(int(p) for p in pickups)
    out: list[tuple[int, int, str]] = []
    it0 = -(-first // every) * every
    while it0 < last:
        if have is None or it0 in have:
            i = it0 - first
            reach = min(it0 + steps, last) - first
            if cooling[reach] == cooling[i]:
                if it0 + steps <= last:
                    out.append((it0, steps, NON_CONVECTIVE))
            elif it0 + convective_steps <= last:
                out.append((it0, convective_steps, CONVECTIVE))
        it0 += every
    return out


# ----------------------------------------------------------------------------- loss
def observation_sigma(truth: HiddenTruth, iters: Sequence[int],
                      volume: torch.Tensor) -> dict[str, float]:
    """Volume-weighted standard deviation of T and S over the observation instants ``iters``."""
    w = volume.numpy()
    sums = {name: np.zeros(3) for name in OBSERVED_FIELDS}   # sum w, sum w x, sum w x^2
    for it in iters:
        obs = truth.observations(it)
        for name in OBSERVED_FIELDS:
            x = obs[name].numpy()
            sums[name] += (w.sum(), (w * x).sum(), (w * x * x).sum())
    out = {}
    for name, (s0, s1, s2) in sums.items():
        mean = s1 / s0
        out[name] = math.sqrt(max(s2 / s0 - mean * mean, 0.0))
    return out


def observed_variables(sigma: Mapping[str, float]) -> list[str]:
    """Observed variables that vary in the truth (sigma >= ``SIGMA_FLOOR``)."""
    names = [name for name in OBSERVED_FIELDS if float(sigma[name]) >= SIGMA_FLOOR]
    if not names:
        raise ValueError(f"every observed variable is constant: sigma={dict(sigma)}")
    return names


def observation_loss(pred: Mapping[str, torch.Tensor], obs: Mapping[str, torch.Tensor], *,
                     volume: torch.Tensor, sigma: Mapping[str, float]) -> torch.Tensor:
    """Sum over the varying observed variables of the volume-weighted MSE / sigma^2.

    ``pred`` and ``obs`` map ``temp``/``salt`` to (nz, ny, nx) fields at one
    observation instant; ``volume`` is the wet-cell volume rA * hFacC * drF.
    Variables with sigma below ``SIGMA_FLOOR`` are skipped.  No u, v.
    """
    names = observed_variables(sigma)
    return trajectory_loss(pred, obs, volume_weight=volume, sigma=sigma,
                           var_weight={name: 1.0 for name in names})


def detach_rollout_state(model, state):
    """Cut the autograd graph between two steps: the prognostic state and the
    stepper's restart-critical cross-step state (AB history, etaH, lagged w,
    ...), both detached, the latter through the checkpoint contract
    (``export_step_state(detach=True)`` then ``import_step_state``; the
    grid-attached mirrors are re-derived from it at the next step's start).
    """
    stepper = model.stepper
    carried = stepper.export_step_state(detach=True)
    stepper.import_step_state(carried, device=model.config.device, dtype=model.config.dtype)
    return state.with_tensors(eta=state.eta.data.detach(), u=state.u.data.detach(),
                              v=state.v.data.detach(), temp=state.temp.data.detach(),
                              salt=state.salt.data.detach())


def observation_rollout(twin, injection, truth: HiddenTruth, it0: int, n_steps: int, *,
                        volume: torch.Tensor, sigma: Mapping[str, float],
                        keep_final: bool = False,
                        grad_segment: int | None = None) -> tuple[torch.Tensor, dict]:
    """Observation loss of one window of ``n_steps`` from the truth state at ``it0``.

    The window starts from the full truth state of pickup ``it0``
    (``training.state_from_pickup``, AB history included) and runs the twin
    with ``injection`` (``None`` runs the twin's reference KPP, which the
    guard refuses in training processes).  The loss is the mean over the
    observation instants ``truth.observation_times(it0, n_steps)`` of
    ``observation_loss``.  With ``grad_segment = N`` the graph is cut every
    N steps (``detach_rollout_state`` before steps N, 2N, ...): the forward
    is the whole window, but the gradient of each observation reaches only
    the closure calls of its own segment.  Metrics (no gradient):
    ``temp_end``/``salt_end`` RMSE at the last instant, ``temp``/``salt``
    RMSE over all instants, in physical units, volume weighted; ``n_obs``;
    with ``keep_final`` also ``final`` = {temp, salt} numpy fields at the
    window end.
    """
    from locomgcm.learned.training import state_from_pickup

    if grad_segment is not None and int(grad_segment) < 1:
        raise ValueError(f"grad_segment must be >= 1, got {grad_segment}")
    model = twin.model
    targets = truth.observation_times(it0, n_steps)
    target_set = set(targets)
    state = state_from_pickup(twin, truth, it0)
    loss = state.temp.data.new_zeros(())
    device = state.temp.data.device      # observations are read on the CPU
    w = volume.to(device)
    w_sum = float(w.sum())
    sq = {name: 0.0 for name in OBSERVED_FIELDS}
    sq_end: dict[str, float] = {}
    final = None
    calls0 = injection.calls if injection is not None else 0
    with injection if injection is not None else contextlib.nullcontext():
        for n in range(n_steps):
            if grad_segment is not None and n and n % int(grad_segment) == 0:
                state = detach_rollout_state(model, state)
            state = model.step(state, dt=twin.time_step, time_seconds=(it0 + n) * twin.dt)
            it = it0 + n + 1
            if it not in target_set:
                continue
            obs = {name: value.to(device) for name, value in truth.observations(it).items()}
            pred = {"temp": state.temp.data[0], "salt": state.salt.data[0]}
            loss = loss + observation_loss(pred, obs, volume=w, sigma=sigma)
            with torch.no_grad():
                for name in OBSERVED_FIELDS:
                    e2 = float(((pred[name] - obs[name]) ** 2 * w).sum()) / w_sum
                    sq[name] += e2
                    if it == targets[-1]:
                        sq_end[name] = e2
                if it == targets[-1] and keep_final:
                    final = {name: pred[name].detach().cpu().numpy().copy() for name in OBSERVED_FIELDS}
    if injection is not None and injection.calls - calls0 != n_steps:
        raise RuntimeError(f"closure called {injection.calls - calls0} times in {n_steps} steps; "
                           "the injection did not replace the KPP diagnosis")
    loss = loss / len(targets)
    metrics: dict = {name: math.sqrt(sq[name] / len(targets)) for name in OBSERVED_FIELDS}
    metrics.update({f"{name}_end": math.sqrt(value) for name, value in sq_end.items()})
    metrics["n_obs"] = len(targets)
    if keep_final:
        metrics["final"] = final
    return loss, metrics


def _inner_features(inner: nn.Module, inputs, temp, salt, u, v, *, grid):
    """The input ``hybrid.learned_diagnose`` builds for ``inner``: its own
    ``features_from_inputs`` if it has one, else ``hybrid.features_from_inputs``."""
    build = getattr(inner, "features_from_inputs", None)
    if build is None:
        build = standard_features_from_inputs
    return build(inputs, temp, salt, u, v, grid=grid)


class ClosureTap(nn.Module):
    """Wrap a closure and keep ``(features, keyword arguments, output)`` of every call.

    ``retain`` names output tensors whose gradient ``.grad`` is kept after
    backward (``Tensor.retain_grad``), for example ``("diff_kzt",)``.  The
    wrapped closure's parameters are this module's parameters, so a
    ``LearnedKppInjection(ClosureTap(closure))`` trains ``closure``.
    ``features_from_inputs`` passes on the wrapped closure's input builder
    (``hybrid.learned_diagnose``), so a closure with its own input
    (``smooth.SmoothRiClosure``, ``pp81_closure.Pp81Closure``) gets it through
    the wrapper.
    """

    def __init__(self, inner: nn.Module, retain: Sequence[str] = ()):
        super().__init__()
        self.inner = inner
        self.retain = tuple(retain)
        self.calls: list[tuple] = []

    def features_from_inputs(self, inputs, temp, salt, u, v, *, grid):
        return _inner_features(self.inner, inputs, temp, salt, u, v, grid=grid)

    def forward(self, feats, **kwargs):
        out = self.inner(feats, **kwargs)
        for key in self.retain:
            if out[key].requires_grad:
                out[key].retain_grad()
        self.calls.append((feats, kwargs, out))
        return out


class FrozenInputClosure(nn.Module):
    """Wrap a closure and detach its input features (``ColumnFeatures.x``).

    The forward output is the wrapped closure's.  In the backward pass the
    gradient still crosses every step of the dynamical core (the implicit
    mixing solve included) and reaches the closure parameters, but not the
    feedback "the state changed, so the mixing coefficients change": d K /
    d state is set to zero.  The other feature fields are masks.  The
    wrapped closure's parameters are this module's parameters, and
    ``features_from_inputs`` is the wrapped closure's input builder, as in
    ``ClosureTap``.
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def features_from_inputs(self, inputs, temp, salt, u, v, *, grid):
        return _inner_features(self.inner, inputs, temp, salt, u, v, grid=grid)

    def forward(self, feats, **kwargs):
        return self.inner(dataclasses.replace(feats, x=feats.x.detach()), **kwargs)


# ----------------------------------------------------------------------------- PP81
@dataclass(frozen=True)
class Pp81Params:
    """PP81 parameters; defaults are ``pp81_readparms.F:64-70`` (no data.pp81).

    ``visc_background`` is viscArNr, ``diff_background_t``/``_s`` are
    diffKrNrT/diffKrNrS of the case; ``gravity`` and ``rho_const`` the
    MITgcm values; RiLimit (``pp81_readparms.F:117-123``) is ``pp81.pp81_ri_limit``.
    ALLOW_PP81_LOWERBOUND is undefined in ``PP81_OPTIONS.h``, so PPviscMin
    and PPdiffMin never enter.
    """

    visc_background: float
    diff_background_t: float
    diff_background_s: float
    gravity: float
    rho_const: float
    nu0: float = 1.0e-2
    alpha: float = 5.0
    n_ri: int = 2
    visc_max: float = 1.0
    eos: str = "mdjwf"


def pp81_coefficients(
    temp: torch.Tensor,
    salt: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    *,
    drf: torch.Tensor,
    mask_c: torch.Tensor,
    params: Pp81Params,
) -> dict[str, torch.Tensor]:
    """PP81 viscosity and diffusivities as MITgcm ``pkg/pp81`` computes them.

    Shapes: ``temp``, ``salt`` (B, nz, ny, nx); ``u`` (B, nz, ny, nx+1) and
    ``v`` (B, nz, ny+1, nx) with both outer faces, as in the LOCOMgcm state;
    ``drf`` (nz,); ``mask_c`` (nz, ny, nx).  Levels K = 2..Nr of MITgcm are
    LOCOMgcm interfaces k = K-1 = 1..nz-1 (top of layer k).

    ``pp81_ri_number.F``: the densities of layers k-1 and k at the pressure
    of layer k (``FIND_RHO_2D`` with kRef = K; pressure pRef4EOS(K) =
    rhoConst |rC(K)| g, ``set_ref_state.F:96``, selectP_inEOS_Zc <= 1 and
    surf_pRef = eosRefP0); N^2 = -g/rhoConst (rho_{k-1} - rho_k) / drC; the
    shear of the cell-centred velocity differenced over drC; Ri = N^2 /
    max(shear^2, 1e-10).  ``pp81_calc.F:73-86``: below RiLimit the viscosity
    is PPviscMax and the denominator uses RiLimit, otherwise PPnu0 /
    (1 + PPalpha Ri)^PPnRi; viscosity >= viscArNr, diffusivity =
    max(visc / denominator, diffKrNrS); both times maskC(K).

    Returned in the closure's index convention (index k = top interface of
    layer k, 0 at k = 0): ``visc_cell`` = PPviscAr; ``diff_kzt`` =
    diffKrNrT + (PPdiffKr - diffKrNrS) and ``diff_kzs`` = diffKrNrS +
    (PPdiffKr - diffKrNrS), the totals ``calc_3d_diffusivity.F`` and
    ``pp81_calc_diff.F`` form; ``ri`` the Richardson number.  No gradient is
    needed; the operations follow the Fortran order so a line-by-line
    transcription agrees bit for bit.
    """
    from locomgcm.dynamics.equation_of_state import mdjwf_density_anomaly

    if params.eos != "mdjwf":
        # ponytail: MDJWF only (every truth case uses it); JMD95 needs
        # the jmd95 kernel and its bar pressure unit.
        raise NotImplementedError(f"pp81_coefficients supports eos='mdjwf', got {params.eos!r}")
    if params.n_ri < 1 or int(params.n_ri) != params.n_ri:
        raise ValueError(f"PPnRi must be a positive integer, got {params.n_ri}")
    batch, nz, ny, nx = temp.shape
    dtype, device = temp.dtype, temp.device
    drf = drf.to(dtype=dtype, device=device)
    drc, zc = reference_layer_centres(drf)
    recip_drc = torch.cat((drc.new_ones(1), 1.0 / drc[1:]))
    pressure = reference_pressure_from_depth(
        zc, rho_const=params.rho_const, gravity=params.gravity,
    ).reshape(nz, 1, 1)

    def rho(t, s, p):
        return mdjwf_density_anomaly(t, s, p, reference_density=params.rho_const)

    minus_n2 = richardson.negative_buoyancy_frequency(
        rho(temp, salt, pressure), rho(temp[:, :-1], salt[:, :-1], pressure[1:]),
        recip_drc=recip_drc, gravity=params.gravity, mass2r_unit=1.0 / params.rho_const)
    ri = richardson.richardson_number(
        minus_n2, richardson.vertical_shear_squared(u, v, recip_drc=recip_drc))
    n_ri = int(params.n_ri)
    visc, diff = pp81_scheme.pp81_coefficients(
        ri, mask_c=mask_c.to(dtype=dtype, device=device), alpha=params.alpha, nu0=params.nu0,
        n_ri=n_ri, visc_max=params.visc_max,
        ri_limit=pp81_scheme.pp81_ri_limit(nu0=params.nu0, visc_arnr_surface=params.visc_background,
                                           visc_max=params.visc_max, n_ri=n_ri, alpha=params.alpha),
        visc_arnr=drf.new_full((nz,), params.visc_background),
        diffkr_nr_s=drf.new_full((nz,), params.diff_background_s))
    visc, diff = visc[:, 1:], diff[:, 1:]

    def full(inner: torch.Tensor) -> torch.Tensor:
        return torch.cat((inner.new_zeros(batch, 1, ny, nx), inner), dim=1)

    return {
        "visc_cell": full(visc),
        "diff_kzt": full(params.diff_background_t + (diff - params.diff_background_s)),
        "diff_kzs": full(params.diff_background_s + (diff - params.diff_background_s)),
        "ri": full(ri),
    }


def pickup_state_tensors(truth: HiddenTruth, it: int) -> dict[str, torch.Tensor]:
    """T, S (1, nz, ny, nx) and u (1, nz, ny, nx+1), v (1, nz, ny+1, nx) of pickup ``it``.

    The far face of each periodic axis is the image of face 0, as in
    ``training.state_from_pickup``.  # ponytail: periodic axes only.
    """
    fields = truth.pickup(it)
    uvel = fields["Uvel"].unsqueeze(0)
    vvel = fields["Vvel"].unsqueeze(0)
    return {
        "temp": fields["Theta"].unsqueeze(0),
        "salt": fields["Salt"].unsqueeze(0),
        "u": torch.cat((uvel, uvel[..., :1]), dim=-1),
        "v": torch.cat((vvel, vvel[..., :1, :]), dim=-2),
    }
