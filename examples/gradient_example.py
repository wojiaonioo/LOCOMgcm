"""Gradient of a pureocean objective with respect to the initial temperature.

The objective is the mean top-layer temperature in a box around the domain
centre after ``--steps`` steps.  The script differentiates it with
``torch.autograd.grad`` and checks one directional derivative against a
central finite difference, all in float64 on the CPU.

Usage (from the repository root)::

    python examples/gradient_example.py [--steps 10] [--h 1e-6] [--save DIR]

``--save DIR`` also writes the gradient as ``DIR/gradient.npy`` and the scalar
results as ``DIR/metrics.json``; ``bin/verify`` uses them.

Upwind advection makes the objective piecewise smooth, so a finite-difference
step can cross a branch change and the check then degrades to about 1e-5
relative difference.  Observed relative AD-FD differences:

    steps  h=1e-3   h=1e-4   h=1e-5   h=1e-6
    10     4.5e-5   9.5e-10  2.0e-9   2.9e-9
    40       -      3.6e-5   6.0e-5   1.3e-8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _pureocean  # noqa: E402  (also puts the repository root on sys.path)
import torch  # noqa: E402

from defgcm import ModelRunner  # noqa: E402

BOX = (slice(14, 22), slice(14, 22))  # 8 x 8 cells around the centre of 36 x 36


def objective(run_cfg, run_dir, steps, temp0, *, grad):
    """Build a fresh model, replace its initial temperature, run, return J.

    A model instance keeps its time-stepping history after a run, so every
    evaluation builds a new one.
    """
    model, state, dt = _pureocean.build(run_cfg, run_dir, steps)
    state = state.with_tensors(temp=temp0)
    result = ModelRunner().run(
        model, state, steps=steps, dt=dt, track_gradients=grad,
        snapshot_steps=[steps], collect_snapshots=False, monitor_interval=steps,
    )
    top = result.final_state.temp.data[0, 0]
    return top[BOX].mean(), model, state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--h", type=float, default=1.0e-6,
                        help="finite-difference step in degC (direction max-norm is 1)")
    parser.add_argument("--work", type=Path, default=_pureocean.DEFAULT_WORK)
    parser.add_argument("--save", type=Path, default=None,
                        help="write gradient.npy and metrics.json to this directory")
    args = parser.parse_args(argv)

    run_cfg, run_dir = _pureocean.prepare(args.work, args.steps)
    _, state0, _ = _pureocean.build(run_cfg, run_dir, args.steps)
    temp_ref = state0.temp.data.detach().clone()

    started = time.perf_counter()
    temp0 = temp_ref.clone().requires_grad_(True)
    J, model, _ = objective(run_cfg, run_dir, args.steps, temp0, grad=True)
    (g,) = torch.autograd.grad(J, temp0)
    t_grad = time.perf_counter() - started

    wet = model.grid.hfac_c_for(g.dtype, g.device) > 0
    generator = torch.Generator().manual_seed(0)
    direction = torch.rand(temp_ref.shape, generator=generator, dtype=temp_ref.dtype)
    direction = direction * wet

    started = time.perf_counter()
    with torch.no_grad():
        j_plus, _, _ = objective(run_cfg, run_dir, args.steps,
                                 temp_ref + args.h * direction, grad=False)
        j_minus, _, _ = objective(run_cfg, run_dir, args.steps,
                                  temp_ref - args.h * direction, grad=False)
    t_fd = time.perf_counter() - started

    ad = float((g * direction).sum())
    fd = float((j_plus - j_minus) / (2.0 * args.h))
    top_box = g[0, 0][BOX]
    print(f"J = {float(J.detach()):.12f} degC (top-layer box mean after {args.steps} steps)")
    print(f"|dJ/dT0|_2 = {float(g.norm()):.6e}, sum = {float(g.sum()):.6e}, "
          f"top-layer box sum = {float(top_box.sum()):.6e}")
    print(f"nonzero gradient entries: {int((g != 0).sum())} of {g.numel()}")
    print(f"directional derivative: AD = {ad:.12e}, FD = {fd:.12e}, "
          f"relative difference = {abs(ad - fd) / abs(fd):.3e} (h = {args.h:g})")
    print(f"wall time: forward+backward {t_grad:.2f} s, two FD forwards {t_fd:.2f} s")
    if args.save is not None:
        import numpy as np

        args.save.mkdir(parents=True, exist_ok=True)
        np.save(args.save / "gradient.npy", g.detach().cpu().numpy())
        metrics = {
            "J": float(J.detach()), "ad": ad, "fd": fd,
            "ad_fd_rel": abs(ad - fd) / abs(fd), "h": args.h, "steps": args.steps,
        }
        (args.save / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
