"""Run the built-in pureocean case on the CPU for a few steps.

Usage (from the repository root)::

    python examples/minimal_run.py [--steps 20] [--work outputs/examples]

The first call writes synthetic inputs and runs pipeline stages 1a and 1b under
``--work``; later calls reuse them.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make the sibling helper importable even under ``python -P`` / PYTHONSAFEPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _pureocean  # noqa: E402  (also puts the repository root on sys.path)
import torch  # noqa: E402

from locomgcm import ModelRunner  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--work", type=Path, default=_pureocean.DEFAULT_WORK)
    args = parser.parse_args(argv)

    run_cfg, run_dir = _pureocean.prepare(args.work, args.steps)

    started = time.perf_counter()
    model, state, dt = _pureocean.build(run_cfg, run_dir, args.steps)
    built = time.perf_counter()
    result = ModelRunner().run(
        model, state, steps=args.steps, dt=dt,
        snapshot_steps=[args.steps], collect_snapshots=False,
        monitor_interval=max(1, args.steps // 4),
    )
    finished = time.perf_counter()

    final = result.final_state
    grid = model.grid
    # State tensors carry a leading batch axis of length 1:
    # eta [1, ny, nx], temp/salt [1, nz, ny, nx],
    # u [1, nz, ny, nx + 1] on west faces, v [1, nz, ny + 1, nx] on south faces.
    eta = final.eta.data[0]
    temp = final.temp.data[0]
    u, v = final.u.data[0], final.v.data[0]
    wet = grid.hfac_c_for(temp.dtype, temp.device)[0]
    area = grid.metrics.cell_area.to(temp) * wet
    sst = float((temp[0] * area).sum() / area.sum())

    print(f"grid {grid.nx} x {grid.ny} x {grid.nz}, dtype {temp.dtype}, "
          f"device {temp.device}, dt {dt.dt_clock:g} s")
    print(f"after {args.steps} steps ({args.steps * dt.dt_clock / 3600:.2f} h):")
    print(f"  max |u| = {float(u.abs().max()):.4e} m/s, "
          f"max |v| = {float(v.abs().max()):.4e} m/s")
    print(f"  area-mean SST = {sst:.6f} degC")
    print(f"  max |eta| = {float(eta.abs().max()):.4e} m")
    print(f"wall time: build {built - started:.2f} s, "
          f"{args.steps} steps {finished - built:.2f} s")
    if not all(torch.isfinite(t).all() for t in (eta, temp, u, v)):
        raise SystemExit("non-finite values in the final state")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
