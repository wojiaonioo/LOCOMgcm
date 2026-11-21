"""Online multi-step training of the learned KPP closure.

The hybrid model is the LOCOMgcm twin of the MITgcm truth case with
``RegionalKppInputs.diagnose`` routed to the closure (``hybrid.py``).  Every
training window starts from an MITgcm pickup, runs the hybrid model for a
fixed number of steps under the same forcing, and compares the state with the
truth pickups written every ``output_every_steps`` steps; the gradient reaches
the network through the whole rollout (implicit vertical solves included).

Pieces:

* ``state_from_pickup``: the LOCOMgcm state and cross-step restart state of
  MITgcm iteration ``it``;
* ``WindowSampler``: forcing-block split of the year into training and held-out
  window starts;
* ``rollout_loss``: one window, trajectory loss plus the optional log K term;
* ``train``: curriculum, Adam, gradient clipping, cosine learning rate,
  validation-driven stage advance, ``log.jsonl`` and ``ckpt.pt``;
* ``taylor_test``: second-order Taylor remainder of a loss along a random
  parameter direction.

The twin model is built by ``experiments/learned_kpp/twin/build_to.py``;
``open_case`` imports it lazily, the only reference from this module to the
experiment tree.
"""
from __future__ import annotations

import contextlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from locomgcm.learned.closure_kpp import LearnedKppClosure
from locomgcm.learned.data import MitTruth
from locomgcm.learned.features import COLUMN_CHANNELS, INTERFACE_CHANNELS
from locomgcm.learned.hybrid import LearnedKppInjection
from locomgcm.learned.losses import log_k_loss, trajectory_loss
from locomgcm.learned.regime import RI_BINS, RegimeEmbeddingClosure
from locomgcm.learned.smooth import RI_NODES, SmoothRiClosure

#: Prognostic variables compared with the truth: key -> MITgcm pickup field.
TRUTH_FIELDS = {"temp": "Theta", "salt": "Salt", "u": "Uvel", "v": "Vvel"}

# Stepper switches whose restart needs cross-step state that an MITgcm pickup
# of the first truth case does not carry (tracer AB history GtNm1/GsNm1,
# non-hydrostatic pressure, implicit vertical advection wrappers, r* geometry).
_UNSUPPORTED_RESTART_FLAGS = (
    "ab2_on_temperature_tendency",
    "ab2_on_salinity_tendency",
    "momentum_implicit_vertical_advection",
    "temperature_implicit_vertical_advection",
    "salinity_implicit_vertical_advection",
)


# ----------------------------------------------------------------------------- set-up
def open_case(cfg: Mapping[str, Any]):
    """Build the twin model (without fresh-run preparation) and open the truth.

    Checks that the yaml, the truth run and the twin agree on grid size, time
    step and pickup spacing.  ``optim.device`` (default ``cpu``) places the
    twin on that device (``run.device`` of the case yaml overridden).
    Returns ``(twin, truth)``.
    """
    from experiments.learned_kpp.twin.build_to import build_model

    tc = cfg["truth"]
    device = str(cfg.get("optim", {}).get("device", "cpu"))
    twin = build_model(tc["case_yaml"], initialize=False,
                       sets=() if device == "cpu" else (f"run.device={device}",))
    grid = twin.model.grid
    sizes = (int(grid.nx), int(grid.ny), int(grid.nz))
    if sizes != (int(tc["nx"]), int(tc["ny"]), int(tc["nz"])):
        raise ValueError(f"twin grid (nx, ny, nz) = {sizes}, config says "
                         f"{(tc['nx'], tc['ny'], tc['nz'])}")
    if twin.dt != float(tc["dt_seconds"]):
        raise ValueError(f"twin dt {twin.dt} s, config dt_seconds {tc['dt_seconds']}")
    truth = MitTruth(tc["run_dir"], *sizes)
    every = pickup_spacing(truth)
    if every != int(tc["output_every_steps"]):
        raise ValueError(f"truth pickups every {every} steps, config says "
                         f"output_every_steps={tc['output_every_steps']}")
    return twin, truth


def build_closure(cfg: Mapping[str, Any]) -> LearnedKppClosure:
    """The closure of ``cfg["network"]``, initialised from ``cfg["optim"]["seed"]``.

    Seeding immediately before construction gives training and evaluation
    the same untrained closure. ``network.architecture`` selects ``mlp``
    (default), built for ``truth.nz`` levels, ``conv``, which takes
    ``network.kernel_size`` (default 5) and ``network.dilation_base``
    (default 1) and whose weights fit any number of levels, or ``regime``
    (``RegimeEmbeddingClosure``: Ri and forcing-sign tokens with embeddings
    in front of the ``mlp`` head), which
    takes ``network.ri_bins`` (default 24) and ``network.weak_threshold``
    (s^-2, default none: no weak-stratification token), or
    ``smooth`` (``SmoothRiClosure``: regularised atan(Ri) inputs through
    slope-limited tables), which takes ``network.ri_nodes``
    (default 24; 0: no tables, the regularised inputs enter the MLP directly).
    """
    net = cfg["network"]
    architecture = str(net.get("architecture", "mlp"))
    common = {
        "hidden": int(net["hidden"]),
        "depth": int(net["depth"]),
        "k_cap": float(net["k_cap"]),
        "tie_salt_to_temp": bool(net["tie_salt_to_temp"]),
    }
    torch.manual_seed(int(cfg["optim"]["seed"]))
    if architecture == "conv":
        return LearnedKppClosure(
            architecture="conv",
            kernel_size=int(net.get("kernel_size", 5)),
            dilation_base=int(net.get("dilation_base", 1)),
            **common,
        )
    nz = int(cfg["truth"]["nz"])
    if architecture == "regime":
        weak = net.get("weak_threshold")
        return RegimeEmbeddingClosure(nz - 1, ri_bins=int(net.get("ri_bins", RI_BINS)),
                                      weak_threshold=None if weak is None else float(weak), **common)
    if architecture == "smooth":
        return SmoothRiClosure(nz - 1, ri_nodes=int(net.get("ri_nodes", RI_NODES)), **common)
    return LearnedKppClosure(
        n_iface=nz - 1,
        feature_dim=(nz - 1) * len(INTERFACE_CHANNELS) + len(COLUMN_CHANNELS),
        architecture=architecture,
        **common,
    )


def pickup_spacing(truth: MitTruth) -> int:
    """Iterations between consecutive pickups; raises if the spacing varies."""
    its = truth.iters("pickup")
    if len(its) < 2:
        raise ValueError(f"{truth.run_dir} has fewer than two pickups")
    steps = {b - a for a, b in zip(its, its[1:])}
    if len(steps) != 1:
        raise ValueError(f"pickups in {truth.run_dir} are not evenly spaced: {sorted(steps)}")
    return steps.pop()


def volume_weights(grid) -> dict[str, torch.Tensor]:
    """Wet volume of every compared point, (nz, ny, nx) for each variable.

    temp/salt: rA * hFacC * drF; u: rAw * hFacW * drF on the nx west faces; v:
    rAs * hFacS * drF on the ny south faces.  On a periodic axis the dropped
    far face is the image of face 0, so each face is counted once.
    """
    dtype, device = torch.float64, torch.device("cpu")
    nx, ny = int(grid.nx), int(grid.ny)
    drf = grid.layer_thickness.to(dtype=dtype, device=device).view(-1, 1, 1)
    masks, metrics = grid.masks, grid.metrics

    def cpu(t: torch.Tensor) -> torch.Tensor:   # the grid may live on cuda
        return t.to(dtype=dtype, device=device)

    return {
        "temp": metrics.rA_for(dtype, device) * cpu(masks.hFacC) * drf,
        "salt": metrics.rA_for(dtype, device) * cpu(masks.hFacC) * drf,
        "u": metrics.rAw_for(dtype, device)[:, :nx] * cpu(masks.hFacW)[..., :nx] * drf,
        "v": metrics.rAs_for(dtype, device)[:ny, :] * cpu(masks.hFacS)[..., :ny, :] * drf,
    }


def interface_wet_mask(grid) -> torch.Tensor:
    """maskC(k) * maskC(k-1) on the top interface of layer k, 0 at k = 0; (nz, ny, nx)."""
    wet_c = (grid.masks.hFacC > 0).to(torch.float64)
    out = torch.zeros_like(wet_c)
    out[1:] = wet_c[1:] * wet_c[:-1]
    return out


def truth_sigma(
    truth: MitTruth,
    volume_weight: Mapping[str, torch.Tensor],
    *,
    names: Sequence[str] = ("temp", "u", "v"),
    cache: Path | None = None,
) -> dict[str, float]:
    """Volume-weighted standard deviation of each variable over every truth pickup.

    ``cache`` is a json file: read when it was written for the same run
    directory and pickup range, otherwise computed (one pass over all pickups)
    and written.
    """
    its = truth.iters("pickup")
    source = {"run_dir": str(truth.run_dir), "first": its[0], "last": its[-1], "count": len(its)}
    if cache is not None and Path(cache).exists():
        stored = json.loads(Path(cache).read_text())
        if stored.get("source") == source and set(names) <= set(stored.get("sigma", {})):
            return {name: float(stored["sigma"][name]) for name in names}
    weights = {name: volume_weight[name].numpy() for name in names}
    sums = {name: np.zeros(3) for name in names}   # sum w, sum w x, sum w x^2
    for it in its:
        fields = truth.fields("pickup", it)
        for name in names:
            x = fields[TRUTH_FIELDS[name]].numpy()
            w = weights[name]
            sums[name] += (w.sum(), (w * x).sum(), (w * x * x).sum())
    sigma = {}
    for name, (s0, s1, s2) in sums.items():
        mean = s1 / s0
        sigma[name] = math.sqrt(max(s2 / s0 - mean * mean, 0.0))
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        Path(cache).write_text(json.dumps({"source": source, "sigma": sigma}, indent=1) + "\n")
    return sigma


# ----------------------------------------------------------------------------- restart
def state_from_pickup(twin, truth: MitTruth, it: int):
    """LOCOMgcm state at MITgcm iteration ``it``, with the stepper's restart state.

    Prognostic fields: Uvel, Vvel, Theta, Salt, EtaN of pickup ``it``.  MITgcm
    stores nx west faces of u (ny south faces of v); the LOCOMgcm state has
    nx+1 (ny+1), and on a periodic axis the far face is the image of face 0.
    Restart keys (``locomgcm/checkpoint/state.py:12-25``), written through
    ``stepper.import_step_state``, the decode/apply path of a checkpoint
    restart:

    * ``_ab2_prev_tendency`` {u, v} from GuNm1, GvNm1 (same face mapping);
    * ``_etaH``, the eta of the pressure solve's mass term: EtaH when the
      stepper runs ``exact_conserv``, EtaN otherwise.  Without exactConserv
      MITgcm never updates etaH (integr_continuity.F:325-332, the pickup holds
      its initial 0) and its solve reads etaN (solve_for_pressure.F:216-235);
      LOCOMgcm then caches eta itself as ``_etaH``
      (implicit_free_surface.py, ``eta_H_new = eta_new``);
    * ``__cached_total_phi_hyd`` from PhiHyd;
    * ``_step_index`` = ``it``;
    * ``_w_interface_prev``: minus the LOCOMgcm continuity w of the pickup
      velocities, which is what the stepper caches at the end of a step
      (implicit_free_surface.py, final vertical-velocity cache; MITgcm sign).

    The pairing of the three MITgcm fields with the LOCOMgcm caches is the
    one the twin's MITgcm comparison uses (cache after step
    n <-> pickup n, first nx faces).  The model clock is set to ``it * dt``;
    the returned state passes through ``resume_step_state`` for ordinary tensor
    storage and shared-face synchronization, without reapplying physical boundaries.
    Imported step state is retained; the next step applies its sampled boundaries.

    Supported: both horizontal axes periodic, linear free surface, hydrostatic,
    no tracer AB, no implicit vertical advection; anything else raises.
    """
    from locomgcm.dynamics.vertical_velocity import diagnose_vertical_velocity
    from locomgcm.geometry import BoundaryType
    from locomgcm.runner import resume_step_state

    model = twin.model
    grid = model.grid
    stepper = model.stepper
    topo = grid.boundary_topology
    if topo.west_east != BoundaryType.PERIODIC or topo.south_north != BoundaryType.PERIODIC:
        # ponytail: periodic axes only; a walled axis needs its far face set
        # from the face mask instead of the periodic image.
        raise NotImplementedError(f"state_from_pickup needs periodic axes, got {topo}")
    on = [flag for flag in _UNSUPPORTED_RESTART_FLAGS if getattr(stepper, flag, False)]
    if getattr(getattr(stepper, "nonhydrostatic", None), "nonhydrostatic", False):
        on.append("nonhydrostatic")
    if getattr(stepper, "nonlin_free_surf", 0):
        on.append("nonlin_free_surf")
    if on:
        raise NotImplementedError(f"pickup restart not implemented for stepper switches {on}")

    fields = truth.fields("pickup", it)
    eta_h_name = "EtaH" if stepper.exact_conserv else "EtaN"
    missing = [name for name in ("Uvel", "Vvel", "Theta", "Salt", "EtaN", eta_h_name,
                                 "GuNm1", "GvNm1") if name not in fields]
    if missing:
        raise KeyError(f"pickup {it} lacks {missing}")
    # PhiHyd is in the pickup only when MITgcm applies the pressure gradient
    # (momPressureForcing); without it the case runs the static EOS pressure
    # (selectP_inEOS_Zc=0, LOCOMgcm ``dynamic_pressure: false``) and the lagged
    # total_phi_hyd cache is never read.
    has_phi_hyd = "PhiHyd" in fields
    dtype, device = model.config.dtype, model.config.device

    def cell(name: str) -> torch.Tensor:
        return fields[name].to(dtype=dtype, device=device).unsqueeze(0)

    def u_faces(name: str) -> torch.Tensor:
        a = cell(name)
        return torch.cat((a, a[..., :1]), dim=-1)

    def v_faces(name: str) -> torch.Tensor:
        a = cell(name)
        return torch.cat((a, a[..., :1, :]), dim=-2)

    seed = twin.state.with_tensors(
        eta=cell("EtaN"), u=u_faces("Uvel"), v=v_faces("Vvel"),
        temp=cell("Theta"), salt=cell("Salt"),
    )
    seed.validate_against(grid)
    _, w_iface, _ = diagnose_vertical_velocity(seed, grid)
    restart = {
        "_step_index": int(it),
        "_ab2_prev_tendency": {"u": u_faces("GuNm1"), "v": v_faces("GvNm1")},
        "_etaH": cell(eta_h_name),
        "_w_interface_prev": -w_iface,
    }
    if has_phi_hyd:
        restart["__cached_total_phi_hyd"] = cell("PhiHyd")
    stepper.import_step_state(restart, device=device, dtype=dtype)
    time_seconds = int(it) * twin.dt
    model.reset_runtime_budget()
    model.current_time_seconds = time_seconds
    model.prepare_step_time(time_seconds)
    return resume_step_state(model, seed)


# ----------------------------------------------------------------------------- windows
class WindowSampler:
    """Training and held-out window starts of one truth run.

    The year is cut into blocks of ``block_seconds`` (one forcing record);
    block b is held out when ``b % holdout_every_nth_block == 0``, which for
    the 72 five-day blocks of case_1 holds out 15 blocks, 20.8 % of the year.
    A window of n steps from pickup ``it0`` covers the pickups ``it0``,
    ``it0 + every``, ..., ``it0 + n``; it is a training (held-out) window when
    every block containing one of these instants is a training (held-out)
    block, so no instant of a held-out block starts a training window or
    enters its loss.  ``seed`` fixes the order of ``train_order``.
    """

    def __init__(self, truth: MitTruth, dt: float, block_seconds: float,
                 holdout_every_nth_block: int, seed: int):
        steps_per_block = float(block_seconds) / float(dt)
        if steps_per_block != int(steps_per_block) or steps_per_block < 1:
            raise ValueError(f"block_seconds {block_seconds} is not a whole number of "
                             f"{dt} s steps")
        if int(holdout_every_nth_block) < 2:
            raise ValueError("holdout_every_nth_block must be >= 2")
        self.steps_per_block = int(steps_per_block)
        self.nth = int(holdout_every_nth_block)
        self.seed = int(seed)
        self.every = pickup_spacing(truth)
        self.pickups = truth.iters("pickup")
        self._pickup_set = frozenset(self.pickups)

    def is_holdout_block(self, block: int) -> bool:
        return block % self.nth == 0

    def block_of(self, it: int) -> int:
        return int(it) // self.steps_per_block

    def _starts(self, n_steps: int, holdout: bool) -> list[int]:
        if n_steps < self.every or n_steps % self.every:
            raise ValueError(f"window of {n_steps} steps is not a multiple of the "
                             f"pickup spacing {self.every}")
        out = []
        for it0 in self.pickups:
            it1 = it0 + n_steps
            if it1 not in self._pickup_set:
                continue
            blocks = range(self.block_of(it0), self.block_of(it1) + 1)
            if all(self.is_holdout_block(b) == holdout for b in blocks):
                out.append(it0)
        return out

    def train_starts(self, n_steps: int) -> list[int]:
        """Window starts whose whole window lies in training blocks, ascending."""
        return self._starts(n_steps, holdout=False)

    def holdout_starts(self, n_steps: int) -> list[int]:
        """Window starts whose whole window lies in held-out blocks, ascending."""
        return self._starts(n_steps, holdout=True)

    def validation_starts(self, n_steps: int, count: int) -> list[int]:
        """``count`` training-window starts evenly spread over the year (fixed per length)."""
        starts = self.train_starts(n_steps)
        if count < 1 or count > len(starts):
            raise ValueError(f"cannot pick {count} validation windows from {len(starts)}")
        if count == 1:
            return [starts[len(starts) // 2]]
        return [starts[round(i * (len(starts) - 1) / (count - 1))] for i in range(count)]

    def train_order(self, n_steps: int, exclude: Sequence[int] = ()) -> list[int]:
        """Training-window starts minus ``exclude``, shuffled by (seed, n_steps)."""
        skip = set(exclude)
        starts = [it0 for it0 in self.train_starts(n_steps) if it0 not in skip]
        rng = np.random.default_rng([self.seed, int(n_steps)])
        return [starts[i] for i in rng.permutation(len(starts))]


# ----------------------------------------------------------------------------- one window
def rollout_loss(
    twin,
    injection: LearnedKppInjection | None,
    truth: MitTruth,
    it0: int,
    n_steps: int,
    *,
    sigma: Mapping[str, float],
    var_weight: Mapping[str, float],
    volume_weight: Mapping[str, torch.Tensor],
    log_k_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Loss of one window of ``n_steps`` steps started from pickup ``it0``.

    Step n of the window runs at model time ``(it0 + n) * dt``.  At every truth
    instant (each pickup after ``it0``) ``trajectory_loss`` of the variables in
    ``var_weight`` is added: temp on cells, u and v on the nx (ny) unique faces.
    ``log_k_weight > 0`` adds ``log_k_weight * log_k_loss`` between the closure's
    ``diff_kzt`` of the first step (diagnosed from the state at ``it0``) and
    MITgcm ``KPPdiffT`` of kpp snapshot ``it0`` (same state, time pairing in
    ``locomgcm/learned/data.py``).

    ``injection=None`` runs the model's own (reference) KPP.  With an
    injection, the closure must be called once per step; a different count
    raises, since it means the stepper did not go through
    ``RegionalKppInputs.diagnose``.

    Returns ``(loss, metrics)``: ``metrics[name]`` is the volume-weighted RMSE
    over all truth instants of the window and ``metrics[name + "_end"]`` the
    RMSE at the last instant, for temp, salt, u, v in physical units;
    ``metrics["log_k"]`` is the unweighted log K term when it was added.
    """
    model = twin.model
    grid = model.grid
    nx, ny = int(grid.nx), int(grid.ny)
    every = truth.window(it0, 2)[1] - it0
    if n_steps < every or n_steps % every:
        raise ValueError(f"n_steps={n_steps} is not a multiple of the pickup spacing {every}")
    targets = set(truth.window(it0, n_steps // every + 1)[1:])
    if log_k_weight > 0 and injection is None:
        raise ValueError("the log K term needs a learned-closure injection")

    state = state_from_pickup(twin, truth, it0)
    loss = state.temp.data.new_zeros(())
    sq = {name: 0.0 for name in TRUTH_FIELDS}
    vol = {name: 0.0 for name in TRUTH_FIELDS}
    sq_end: dict[str, float] = {}
    first_out = None
    calls0 = injection.calls if injection is not None else 0
    with injection if injection is not None else contextlib.nullcontext():
        for n in range(n_steps):
            state = model.step(state, dt=twin.time_step, time_seconds=(it0 + n) * twin.dt)
            if n == 0 and injection is not None:
                first_out = injection.last_out
            it = it0 + n + 1
            if it not in targets:
                continue
            fields = truth.fields("pickup", it)
            pred = {
                "temp": state.temp.data[0],
                "salt": state.salt.data[0],
                "u": state.u.data[0, ..., :nx],
                "v": state.v.data[0, ..., :ny, :],
            }
            tgt = {name: fields[mit] for name, mit in TRUTH_FIELDS.items()}
            loss = loss + trajectory_loss(pred, tgt, volume_weight=volume_weight,
                                          sigma=sigma, var_weight=var_weight)
            with torch.no_grad():
                for name in TRUTH_FIELDS:
                    w = volume_weight[name]
                    e2 = float(((pred[name] - tgt[name]) ** 2 * w).sum())
                    sq[name] += e2
                    vol[name] += float(w.sum())
                    if it == it0 + n_steps:
                        sq_end[name] = e2 / float(w.sum())
    if injection is not None and injection.calls - calls0 != n_steps:
        raise RuntimeError(f"closure called {injection.calls - calls0} times in "
                           f"{n_steps} steps; the injection did not replace the KPP diagnosis")
    metrics = {name: math.sqrt(sq[name] / vol[name]) for name in TRUTH_FIELDS}
    metrics.update({f"{name}_end": math.sqrt(value) for name, value in sq_end.items()})
    if log_k_weight > 0:
        truth_k = truth.kpp(it0)["KPPdiffT"].unsqueeze(0)
        term = log_k_loss(first_out["diff_kzt"], truth_k, interface_wet_mask(grid))
        loss = loss + log_k_weight * term
        metrics["log_k"] = term.item()
    return loss, metrics


# ----------------------------------------------------------------------------- training
class NonFiniteUpdate(RuntimeError):
    """A window's loss or gradient norm is not finite; no parameter was updated."""

    def __init__(self, loss_value: float, grad_norm: float):
        super().__init__(f"non-finite loss {loss_value} or gradient norm {grad_norm}")
        self.loss_value, self.grad_norm = loss_value, grad_norm


def descend(loss: torch.Tensor, params: Sequence[torch.Tensor],
            optimizer: torch.optim.Optimizer, grad_clip_norm: float) -> tuple[float, float]:
    """One window's update: backward, clip, step; returns ``(loss, gradient norm)``.

    The owned reverse entry publishes gradients only after its numerical
    verdict, including when an earlier node interrupts the engine before its
    final callbacks. A non-finite loss or norm raises
    :class:`NonFiniteUpdate`.  Either way ``optimizer.step`` is not reached.
    """
    from locomgcm.checkpoint.window import backward
    backward(loss)
    loss_value = loss.item()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(params, grad_clip_norm))
    if not (math.isfinite(loss_value) and math.isfinite(grad_norm)):
        raise NonFiniteUpdate(loss_value, grad_norm)
    optimizer.step()
    return loss_value, grad_norm


def _cosine(start: float, end: float, index: int, total: int) -> float:
    if total <= 1:
        return start
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * index / (total - 1)))


def _mean_metrics(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Mean of each key over the rows that have it (``log_k`` stops at the end of warm-up)."""
    keys = dict.fromkeys(key for row in rows for key in row)
    return {key: float(np.mean([row[key] for row in rows if key in row])) for key in keys}


def train(cfg: Mapping[str, Any], out_dir: Path, *, max_windows: int | None = None,
          init_ckpt: Path | None = None) -> dict:
    """Curriculum training of the closure of ``cfg`` (``configs/train_v1.yaml`` layout).

    Per curriculum stage {steps, windows}: a fixed set of
    ``validation.windows`` training-block windows is evaluated at the start of
    the stage and after every ``validation.every_windows`` training windows;
    the stage ends when its window budget is spent or the validation loss has
    not reached a new minimum ``validation.patience`` times in a row.  Each
    training window is one Adam step on ``rollout_loss`` (``eps`` =
    ``optim.adam_eps``: the trajectory loss of a 3-step window is about 5e-6
    and every gradient entry of the untrained closure is below 1e-8, so
    Adam's default 1e-8 would scale the steps down by orders of magnitude),
    gradient norm clipped to ``optim.grad_clip_norm``, learning rate on a cosine from
    ``optim.lr`` to ``optim.lr_min`` over the stage budget (restarting each
    stage).  The log K term has weight ``loss.log_k_warmup_weight`` at the
    first window, decreasing linearly to 0 at ``loss.log_k_warmup_windows``.
    Training windows come from ``WindowSampler.train_order`` with the
    validation starts excluded.

    Every ``log.every_windows`` windows one line with the mean over those
    windows goes to ``out_dir/log.jsonl`` and ``out_dir/ckpt.pt`` is rewritten
    with the closure ``state_dict`` and ``cfg``.  ``max_windows`` stops after
    that many training windows in total (debugging).  A non-finite loss or
    gradient raises.  Returns a summary dict.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    optim_cfg, loss_cfg, val_cfg = cfg["optim"], cfg["loss"], cfg["validation"]
    if optim_cfg["dtype"] != "float64" or optim_cfg["device"] != "cpu":
        raise ValueError("training runs on cpu in float64 (optim.dtype, optim.device)")
    log_every = int(cfg["log"]["every_windows"])

    twin, truth = open_case(cfg)
    closure = build_closure(cfg)
    if init_ckpt is not None:
        # Continue from an earlier run's weights; the curriculum, optimizer
        # state and log start afresh.
        closure.load_state_dict(torch.load(Path(init_ckpt), weights_only=False)["closure"])
    injection = LearnedKppInjection(closure)
    sampler = WindowSampler(truth, twin.dt, float(cfg["split"]["block_seconds"]),
                            int(cfg["split"]["holdout_every_nth_block"]), int(optim_cfg["seed"]))
    volume = volume_weights(twin.model.grid)
    var_weight = {name: float(w) for name, w in loss_cfg["var_weight"].items()}
    sigma = truth_sigma(truth, volume, names=tuple(var_weight), cache=out_dir / "sigma.json")
    loss_kw = {"sigma": sigma, "var_weight": var_weight, "volume_weight": volume}
    optimizer = torch.optim.Adam(closure.parameters(), lr=float(optim_cfg["lr"]),
                                 eps=float(optim_cfg["adam_eps"]))
    params = list(closure.parameters())

    log_path = out_dir / "log.jsonl"
    log_path.write_text("")   # one run per log; train() does not resume

    def write(record: dict) -> None:
        with log_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

    def save(stage: int, windows: int) -> None:
        torch.save({"closure": closure.state_dict(), "cfg": dict(cfg), "stage": stage,
                    "windows": windows}, out_dir / "ckpt.pt")

    def validate(starts: Sequence[int], n_steps: int) -> tuple[float, dict[str, float]]:
        losses, rows = [], []
        with torch.no_grad():
            for it0 in starts:
                loss, metrics = rollout_loss(twin, injection, truth, it0, n_steps, **loss_kw)
                losses.append(float(loss))
                rows.append(metrics)
        return float(np.mean(losses)), _mean_metrics(rows)

    write({"kind": "start", "sigma": sigma, "params": sum(p.numel() for p in params),
           "curriculum": list(cfg["curriculum"]), "max_windows": max_windows})
    done = 0
    stages = []
    t_train = time.perf_counter()
    for stage_index, stage in enumerate(cfg["curriculum"]):
        if max_windows is not None and done >= max_windows:
            break
        n_steps, budget = int(stage["steps"]), int(stage["windows"])
        val_starts = sampler.validation_starts(n_steps, int(val_cfg["windows"]))
        order = sampler.train_order(n_steps, exclude=val_starts)
        val_loss, val_metrics = validate(val_starts, n_steps)
        best, stale = val_loss, 0
        write({"kind": "validation", "stage": stage_index, "steps": n_steps, "windows": done,
               "loss": val_loss, "rmse": val_metrics, "starts": val_starts})
        reason = "budget"
        pending: list[dict] = []
        for w in range(budget):
            if max_windows is not None and done >= max_windows:
                reason = "max_windows"
                break
            lr = _cosine(float(optim_cfg["lr"]), float(optim_cfg["lr_min"]), w, budget)
            for group in optimizer.param_groups:
                group["lr"] = lr
            warm = max(0.0, 1.0 - done / float(loss_cfg["log_k_warmup_windows"]))
            log_k_weight = float(loss_cfg["log_k_warmup_weight"]) * warm
            it0 = order[w % len(order)]
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = rollout_loss(twin, injection, truth, it0, n_steps,
                                         log_k_weight=log_k_weight, **loss_kw)
            try:
                loss_value, grad_norm = descend(loss, params, optimizer,
                                                float(optim_cfg["grad_clip_norm"]))
            except NonFiniteUpdate as failure:
                save(stage_index, done)
                raise RuntimeError(f"{failure} at stage {stage_index}, window start {it0}") from None
            seconds = time.perf_counter() - t0
            done += 1
            pending.append({"loss": loss_value, "grad_norm": grad_norm, "seconds": seconds,
                            "lr": lr, "log_k_weight": log_k_weight, **metrics})
            if len(pending) == log_every:
                write({"kind": "train", "stage": stage_index, "steps": n_steps, "windows": done,
                       "last_start": it0, **_mean_metrics(pending),
                       "seconds_per_step": float(np.mean([p["seconds"] for p in pending])) / n_steps})
                pending = []
                save(stage_index, done)
            if (w + 1) % int(val_cfg["every_windows"]) == 0:
                val_loss, val_metrics = validate(val_starts, n_steps)
                if val_loss < best:
                    best, stale = val_loss, 0
                else:
                    stale += 1
                write({"kind": "validation", "stage": stage_index, "steps": n_steps,
                       "windows": done, "loss": val_loss, "rmse": val_metrics, "stale": stale})
                if stale >= int(val_cfg["patience"]):
                    reason = "stalled"
                    break
        if pending:
            write({"kind": "train", "stage": stage_index, "steps": n_steps, "windows": done,
                   **_mean_metrics(pending),
                   "seconds_per_step": float(np.mean([p["seconds"] for p in pending])) / n_steps})
        save(stage_index, done)
        stages.append({"stage": stage_index, "steps": n_steps, "end": reason,
                       "windows_total": done, "best_validation": best})
        write({"kind": "stage_end", **stages[-1]})
        if reason == "max_windows":
            break
    summary = {"windows": done, "stages": stages, "wall_seconds": time.perf_counter() - t_train,
               "out_dir": str(out_dir)}
    write({"kind": "end", **summary})
    return summary


# ----------------------------------------------------------------------------- Taylor test
def taylor_test(
    loss_fn: Callable[[], torch.Tensor],
    closure: torch.nn.Module,
    eps_list: Sequence[float],
    seed: int,
) -> list[dict[str, float]]:
    """Taylor remainders of ``loss_fn`` along a random direction of the closure parameters.

    The direction v has, per parameter tensor p, independent N(0, 1) entries
    times the root mean square of p (1 where that is 0), so ``eps`` is a
    relative perturbation size.  With g the autograd gradient at theta and L0
    the loss at theta, every row reports, for theta + eps v,
    ``r1 = |L - L0|`` and ``r2 = |L - L0 - eps <g, v>|``; for a smooth loss r1
    falls by about 2 and r2 by about 4 when eps halves (``ratio_r1``,
    ``ratio_r2`` against the previous row).  L0 and all perturbed losses are
    evaluated under ``torch.no_grad``; the first row also reports
    ``l0_grad_mode_diff``, the difference between L0 with and without
    gradient recording.  The parameters are restored afterwards.
    """
    if not eps_list:
        raise ValueError("eps_list is empty")
    params = [p for p in closure.parameters() if p.requires_grad]
    base = [p.detach().clone() for p in params]
    gen = torch.Generator().manual_seed(int(seed))
    direction = []
    for p in base:
        rms = float(p.pow(2).mean().sqrt()) if p.numel() else 0.0
        # Drawn on the CPU generator for every device, so a seed gives one direction.
        direction.append((torch.randn(p.shape, generator=gen, dtype=p.dtype) * (rms or 1.0)).to(p.device))
    for p in params:
        p.grad = None
    loss_grad = loss_fn()
    from locomgcm.checkpoint.window import backward
    backward(loss_grad)
    slope = float(sum((p.grad * d).sum() for p, d in zip(params, direction)
                      if p.grad is not None))
    for p in params:
        p.grad = None
    with torch.no_grad():
        l0 = float(loss_fn())
    rows: list[dict[str, float]] = []
    try:
        for eps in eps_list:
            with torch.no_grad():
                for p, b, d in zip(params, base, direction):
                    p.copy_(b + eps * d)
                value = float(loss_fn())
            row = {"eps": float(eps), "loss": value, "r1": abs(value - l0),
                   "r2": abs(value - l0 - eps * slope)}
            if rows:
                prev = rows[-1]
                row["ratio_r1"] = prev["r1"] / row["r1"] if row["r1"] else math.inf
                row["ratio_r2"] = prev["r2"] / row["r2"] if row["r2"] else math.inf
            rows.append(row)
    finally:
        with torch.no_grad():
            for p, b in zip(params, base):
                p.copy_(b)
    rows[0].update({"l0": l0, "slope": slope, "l0_grad_mode_diff": loss_grad.item() - l0})
    return rows
