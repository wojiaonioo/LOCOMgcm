"""Memory-budget-bounded parallelism for the full-mode per-variable metric loops.

The per-variable work in verification / categorical / neighborhood is independent,
so it parallelizes across the ~6 variables (benchmark on 448²×90×49:
verification 406→110 s, categorical 201→66 s with 6 threads — ~3.0–3.7×). But
parallelism MUST NOT blow memory: each concurrent worker holds its variable's
footprint (the full-mode verification per-cell accumulator is O(grid) — single-digit
GB), so N workers = N × footprint. We therefore derive the worker count FROM a
memory budget rather than assuming the node has spare RAM:

    workers = clamp(budget // per_var_footprint, 1, min(n_vars, cores))

- Small-memory node / huge grid → 1 worker (= the safe sequential footprint).
- Memory-rich node → up to n_vars workers.
- A single variable whose full footprint exceeds the budget → the caller degrades
  it to global-only (no per-cell accumulator), so it never OOMs.

Threads, not processes: numpy releases the GIL during the C-level reductions and
np.percentile, so threads gave ~3× with ZERO pickling/IPC (ctx, hFac and the mmap'd
fields are shared natively); processes were only ~15% faster but need fork+globals
or pickling the grid. Memory accounting is identical either way.

Budget source (first hit wins): ``METRICS_MEM_BUDGET_GB`` env → ``SLURM_MEM_PER_NODE``
(respects the cgroup --mem, so we never exceed the SLURM allocation) → psutil
available RAM → a conservative 8 GB default. A safety factor leaves headroom for
the transient frame copies and the rest of the process.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Sequence

_SAFETY = 0.5  # at most half the budget for accumulators; rest = transients + base


def mem_budget_bytes() -> int:
    """Best estimate of RAM this process may safely use, in bytes.

    Source order (first hit wins):
      1. METRICS_MEM_BUDGET_GB — explicit override.
      2. cgroup memory limit — the TRUE --mem allocation enforced by SLURM, read
         straight from the cgroup file. Robust even when SLURM_MEM_PER_NODE is not
         exported. This limit is enforced by the OOM killer.
      3. SLURM_MEM_PER_NODE (MB) — env fallback; "0" (= --mem=0 whole node) is
         non-positive → skipped → falls through to the real node RAM below.
      4. psutil available RAM — for --mem=0 / login-node / no-cgroup cases, where
         the whole node really IS the limit.
      5. conservative 8 GB default.
    """
    env = os.environ.get("METRICS_MEM_BUDGET_GB")
    if env:
        try:
            return int(float(env) * 1e9)
        except ValueError:
            pass
    cg = _cgroup_mem_limit_bytes()
    if cg and cg > 0:
        return cg
    slurm_mb = _parse_mem_mb(os.environ.get("SLURM_MEM_PER_NODE"))
    if slurm_mb is not None and slurm_mb > 0:
        return int(slurm_mb * 1e6)
    try:
        import psutil  # optional
        return int(psutil.virtual_memory().available)
    except Exception:
        return 8 * 10**9


def _cgroup_mem_limit_bytes() -> int | None:
    """The process's cgroup memory limit (bytes), or None if unlimited/unreadable.

    Handles cgroup v2 (``/sys/fs/cgroup/memory.max``) and v1 (resolve the memory
    controller path from /proc/self/cgroup, then read ``memory.limit_in_bytes``,
    walking up to the job level since the limit is set there). The v1 'unlimited'
    sentinel (~2^63) and v2 'max' are treated as None → caller uses node RAM.
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


def _parse_mem_mb(value: str | None) -> float | None:
    """Parse a SLURM memory string (MB) → megabytes; tolerates K/M/G/T suffixes."""
    if not value:
        return None
    s = value.strip().upper()
    mult = {"K": 1.0 / 1024, "M": 1.0, "G": 1024.0, "T": 1024.0 * 1024.0}
    try:
        if s and s[-1] in mult:
            return float(s[:-1]) * mult[s[-1]]
        return float(s)  # plain MB
    except ValueError:
        return None


def affordable(footprint_bytes: int) -> bool:
    """Can a single variable of this footprint fit the safety-scaled budget?"""
    return footprint_bytes <= int(mem_budget_bytes() * _SAFETY)


def usable_cores() -> int:
    """Cores this process may actually run on.

    Under SLURM ``--cpus-per-task=N`` the process is pinned to an N-CPU cpuset,
    so ``os.sched_getaffinity`` reports N — whereas ``os.cpu_count()`` reports
    the WHOLE node (e.g. 160 on the CPU node). Using cpu_count would let a loop over
    >N items (e.g. w_diagnostics / per-timestep-sampled over the sampled nodes)
    spawn more worker threads than the allocation has cores, oversubscribing
    the N-core cpuset. Fall back to cpu_count() where affinity is unavailable
    (non-Linux). Verified on the CPU node: cpu_count()=160, sched_getaffinity=16 under
    --cpus-per-task=16.
    """
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def plan_workers(footprint_bytes: int, n_items: int) -> int:
    """Worker count = clamp(budget // footprint, 1, min(n_items, cores))."""
    if n_items <= 1:
        return 1
    cap = min(n_items, usable_cores())
    if footprint_bytes <= 0:
        return cap
    budget = int(mem_budget_bytes() * _SAFETY)
    return max(1, min(cap, budget // footprint_bytes))


def map_bounded(worker: Callable[[Any], Any], items: Sequence[Any], footprint_bytes: int) -> list:
    """Map ``worker`` over ``items`` with a memory-budget-bounded thread pool.

    Falls back to a plain sequential comprehension when the budget allows only one
    worker (small node / huge grid) — same footprint as the old sequential loop.
    Result order matches ``items`` (ThreadPoolExecutor.map preserves order), so the
    parallel output is identical to sequential.
    """
    items = list(items)
    w = plan_workers(footprint_bytes, len(items))
    if w <= 1:
        return [worker(x) for x in items]
    # Pin numpy/BLAS to 1 thread per worker for the duration of the pool: N worker
    # threads EACH spawning BLAS/OpenMP threads oversubscribes the cores and NEGATES
    # the speedup (measured: an unpinned full regen showed ~0 gain vs ~4-9× pinned).
    # threadpool_limits caps the loaded pools at runtime, so it works regardless of
    # how python was launched (no reliance on OMP_NUM_THREADS being set upstream).
    with _pin_threads(1), ThreadPoolExecutor(max_workers=w) as ex:
        return list(ex.map(worker, items))


def _pin_threads(n: int):
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=n)
    except Exception:
        import contextlib
        return contextlib.nullcontext()


def var_footprint_bytes(grid, *, full: bool, n_accum: int = 8, transient: int = 8,
                        surface: bool = False) -> int:
    """Conservative per-variable peak memory (bytes) for one variable's reduction.

    * ``full``    → ~(n_accum per-cell accumulators + ~transient frame copies) × cell × 8 B
    * not ``full``→ only ~transient frame copies × 8 B (global-only / categorical)
    * ``surface`` → 2-D field (neighborhood / FSS): drop the depth factor.
    """
    ny, nx = int(grid.ny), int(grid.nx)
    cells = (ny * nx) if surface else int(grid.nz) * ny * nx
    factor = (n_accum + transient) if full else transient
    return factor * cells * 8
