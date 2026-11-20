"""Memory-budget-bounded dask scheduler config for the stage1a/1b prep pipelines.

The prep operator chain is now dask-backed (sources chunk along time; the whole
mask/fill/interp/subset chain runs lazily and the zarr consumer streams it). That
removes the old OOM (the entire HYCOM time series materialised at once).
But dask's DEFAULT threaded scheduler uses ~n_cores workers, so it would compute
~n_cores time-blocks concurrently — re-introducing the blow-up by concurrency
(one 1696x960x90 block ≈ several GB × dozens of workers ≫ the SLURM --mem).

So we cap dask's worker count from a MEMORY BUDGET, mirroring the proven pattern in
helper/preprocess/metrics/_parallel.py (we re-derive it here rather than import it,
to keep defgcm from depending on helper/):

    workers = clamp(budget*SAFETY // per_block_footprint, 1, usable_cores)

Budget source (first hit wins): DEFGCM_PREP_MEM_BUDGET_GB env → cgroup memory limit
(the real SLURM --mem, robust because SLURM_MEM_PER_NODE is not exported on every
cluster) → psutil available → conservative 8 GB. Cores = sched_getaffinity (respects
--cpus-per-task cpuset). Env overrides: DEFGCM_PREP_DASK_WORKERS (force worker count),
DEFGCM_PREP_DASK_SCHED ('threads'|'synchronous'|'processes').
"""
from __future__ import annotations

import os
from pathlib import Path

_SAFETY = 0.5            # at most half the budget held in concurrent block transients
_COPY_FACTOR = 4         # interp in/out + fill copy transients per variable per block
_N_VARS = 5              # T, S, u, v, eta — conservative upper bound


def _cgroup_mem_limit_bytes() -> int | None:
    """The process's cgroup memory limit (bytes), or None if unlimited/unreadable.

    cgroup v2 (memory.max) then v1 (walk up to the job-level memory.limit_in_bytes).
    'max' / the v1 unlimited sentinel → None (caller falls back to node RAM).
    """
    try:  # cgroup v2
        v = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if v.isdigit():
            return int(v)
    except Exception:
        pass
    try:  # cgroup v1
        rel = None
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            f = line.split(":")
            if len(f) == 3 and "memory" in f[1].split(","):
                rel = f[2]
                break
        if rel:
            base = Path("/sys/fs/cgroup/memory") / rel.lstrip("/")
            root = Path("/sys/fs/cgroup/memory")
            for d in [base, *base.parents]:
                try:
                    v = int((d / "memory.limit_in_bytes").read_text().strip())
                    if 0 < v < (1 << 62):
                        return v
                except Exception:
                    pass
                if d == root:
                    break
    except Exception:
        pass
    return None


def mem_budget_bytes() -> int:
    env = os.environ.get("DEFGCM_PREP_MEM_BUDGET_GB")
    if env:
        try:
            return int(float(env) * 1e9)
        except ValueError:
            pass
    cg = _cgroup_mem_limit_bytes()
    if cg and cg > 0:
        return cg
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        return 8 * 10**9


def usable_cores() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def per_block_footprint_bytes(nz: int, ny: int, nx: int) -> int:
    """Conservative peak RAM for one time-block flowing through the gridded chain.

    One target-grid copy = nz*ny*nx*8; the chain holds ~_COPY_FACTOR copies
    (interp input+output, post-interp fill copy) across ~_N_VARS variables.
    Over-estimating only spawns FEWER workers (safe).
    """
    return _COPY_FACTOR * _N_VARS * int(nz) * int(ny) * int(nx) * 8


def plan_workers(nz: int, ny: int, nx: int) -> int:
    """Adaptive dask worker count = clamp(budget*SAFETY // footprint, 1, cores)."""
    forced = os.environ.get("DEFGCM_PREP_DASK_WORKERS")
    if forced:
        try:
            return max(1, int(forced))
        except ValueError:
            pass
    cores = usable_cores()
    fp = per_block_footprint_bytes(nz, ny, nx)
    if fp <= 0:
        return cores
    budget = int(mem_budget_bytes() * _SAFETY)
    return max(1, min(cores, budget // fp))


def configure_dask_for_prep(run_cfg, logger) -> None:
    """Set a memory-bounded global dask scheduler for the prep stage.

    Default scheduler = 'threads' (zero IPC, shared memory; the prep hot path
    releases the GIL on the numpy/scipy interp + bottleneck fill). Override with
    DEFGCM_PREP_DASK_SCHED. Worker count is the memory knob (see plan_workers).
    """
    import dask

    g = run_cfg.grid
    workers = plan_workers(g.nz, g.ny, g.nx)
    sched = os.environ.get("DEFGCM_PREP_DASK_SCHED", "threads")
    if sched == "synchronous":
        dask.config.set(scheduler="synchronous")
    else:
        dask.config.set(scheduler=sched, num_workers=workers)
    fp_gb = per_block_footprint_bytes(g.nz, g.ny, g.nx) / 1e9
    budget_gb = mem_budget_bytes() / 1e9
    logger.info(
        f"[dask] scheduler={sched} num_workers={workers} "
        f"(budget={budget_gb:.0f}G safety={_SAFETY} per-block≈{fp_gb:.1f}G "
        f"cores={usable_cores()} grid={g.nz}x{g.ny}x{g.nx})"
    )
