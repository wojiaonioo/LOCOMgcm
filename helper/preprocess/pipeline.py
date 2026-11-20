from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from .adapters import detect_adapter
from .diag_conservation import compute_conservation
from .diag_energy       import compute_energy
from .diag_spectra      import compute_spectra
from .diag_transport    import compute_transport
from .metrics import write_comparison_metrics
from .stats import compute_stats
from .zarr_writer import write_zarr_store


def run_pipeline(
    experiment_dir: Path | str,
    output_zarr: Path | str,
    compute_diag: bool = True,
) -> None:
    """Steps 1-9: detect adapter → scan → align → write Zarr → stats → diagnostics.

    Args:
        compute_diag: If True (default), run advanced diagnostics after stats.
                      Set False to skip for speed during testing.

    Never loads more than one time step of reference data into RAM.
    """
    experiment_dir = Path(experiment_dir)
    output_zarr = Path(output_zarr)

    print(f"[pipeline] Detecting adapter for {experiment_dir.name} ...", flush=True)
    adapter = detect_adapter(experiment_dir)
    _patch_legacy_adapter_paths(adapter)
    print(f"[pipeline] Using {type(adapter).__name__}", flush=True)

    grid = adapter.load_grid()
    print(f"[pipeline] Grid: {grid.nx}×{grid.ny}×{grid.nz}", flush=True)

    # Open model data lazily (no RAM spike)
    print("[pipeline] Opening model NC (lazy) ...", flush=True)
    model_ds = adapter.load_model_data()
    model_time = model_ds.time.values  # float64 seconds, cheap

    print("[pipeline] Scanning reference time steps ...", flush=True)
    # Duck-typed adapter contract: any adapter that exposes
    # `_scan_mitgcm_steps() -> (steps, time_vals)` AND `iter_reference_steps()`
    # uses the streaming-zarr path (memory-efficient, one ref step at a time).
    # Adapters without the iterator must implement load_reference_data() and
    # we'll do an in-memory align (small cases only).
    if hasattr(adapter, "_scan_mitgcm_steps") and hasattr(adapter, "iter_reference_steps"):
        ref_steps, ref_time = adapter._scan_mitgcm_steps()
        n_ref = len(ref_time)
        use_iter = True
    else:
        ref_ds = adapter.load_reference_data()
        ref_time = ref_ds.time.values
        n_ref = len(ref_time)
        ref_steps = None
        use_iter = False

    print(f"[pipeline] Model steps: {len(model_time)}, Ref steps: {n_ref}", flush=True)

    # Time alignment (index-only, no data loaded)
    aligned_model_idx, aligned_ref_idx = _align_indices(model_time, ref_time)
    n_aligned = len(aligned_model_idx)
    print(f"[pipeline] Aligned time steps: {n_aligned}", flush=True)
    if n_aligned == 0:
        raise ValueError(
            f"No aligned time steps. Model has {len(model_time)}, "
            f"Ref has {n_ref} steps. Check metadata.yaml time configuration."
        )
    # Last-line defense: refuse to compute metrics when LOCOMgcm and
    # reference snapshot times do not match to FP64 noise. See the
    # docstring of _assert_no_step_mismatch.
    _assert_no_step_mismatch(
        model_time=model_time,
        ref_time=ref_time,
        aligned_model_idx=aligned_model_idx,
        aligned_ref_idx=aligned_ref_idx,
    )

    availability_hints: dict = {
        "aligned_steps": n_aligned,
        "model_variables": sorted({_standard_var_name(v) for v in model_ds.data_vars}),
        "ref_observed_steps": {},
        "ref_valid_counts": {},
    }

    print(f"[pipeline] Writing Zarr → {output_zarr} ...", flush=True)
    if use_iter:
        ref_iter = _track_ref_iter(
            adapter.iter_reference_steps(),
            set(aligned_ref_idx),
            availability_hints,
        )
        write_zarr_store(
            store_path=output_zarr,
            model_ds=model_ds,
            ref_iter=ref_iter,
            ref_time_vals=ref_time,
            model_time_vals=model_time,
            aligned_model_idx=aligned_model_idx,
            aligned_ref_idx=aligned_ref_idx,
            grid=grid,
        )
    else:
        # In-memory fallback: convert Dataset to per-step iterator on the fly.
        ref_ds_aligned = ref_ds.isel(time=aligned_ref_idx)  # type: ignore[union-attr]

        def _ds_iter():
            for ti in range(ref_ds_aligned.sizes["time"]):
                t = float(ref_ds_aligned.time.values[ti])
                arrays = {
                    name: ref_ds_aligned[name].isel(time=ti).values
                    for name in ref_ds_aligned.data_vars
                }
                yield t, arrays

        # Pretend each step is at consecutive index for ref_step_to_zarr lookup.
        synth_aligned_ref_idx = list(range(n_aligned))
        tracked_ref_iter = _track_ref_iter(
            _ds_iter(),
            set(synth_aligned_ref_idx),
            availability_hints,
        )
        write_zarr_store(
            store_path=output_zarr,
            model_ds=model_ds,
            ref_iter=tracked_ref_iter,
            ref_time_vals=ref_ds_aligned.time.values,
            model_time_vals=model_time,
            aligned_model_idx=aligned_model_idx,
            aligned_ref_idx=synth_aligned_ref_idx,
            grid=grid,
        )

    # Steps 4-5: spatial statistics + Taylor data
    print("[pipeline] Computing spatial statistics ...", flush=True)
    compute_stats(output_zarr, grid.lat, getattr(grid, "hfac", None))

    # Steps 6-9: advanced diagnostics (optional)
    if compute_diag:
        print("[pipeline] Computing advanced diagnostics ...", flush=True)
        compute_conservation(output_zarr, grid, adapter.metadata)
        compute_energy(output_zarr, grid, adapter.metadata)
        try:
            compute_spectra(output_zarr, grid)
        except Exception as e:
            print(f"  [diag] spectra skipped (scipy not available?): {e}", flush=True)
        compute_transport(output_zarr, grid, adapter.metadata)

    print("[pipeline] Writing schema comparison metrics ...", flush=True)
    summary_path, details_path = write_comparison_metrics(
        experiment_dir,
        output_zarr,
        grid,
        adapter.metadata,
        availability_hints=availability_hints,
    )
    print(
        f"[pipeline] Metrics written: {summary_path.name} + {details_path.name}",
        flush=True,
    )

    print("[pipeline] Done.", flush=True)


def _align_indices(
    model_time: np.ndarray,
    ref_time: np.ndarray,
    tol: float = 1.0e-3,
) -> tuple[list[int], list[int]]:
    """Return parallel index lists: ``model_idx[i]`` aligns with ``ref_idx[i]``.

    The default ``tol`` is 1 ms — far below any physically meaningful step
    size. This is intentional and is the last-line defense against a
    step mismatch: LOCOMgcm and MIT both stamp their snapshot
    times as exact integer multiples of ``dt`` seconds, so legitimate runs
    should align to FP64 noise. A larger tolerance silently turns an
    off-by-one-step run into a "PASS" with arbitrarily large physical
    differences (e.g. LOCOMgcm step 5 at t=75 s paired with MIT step 6 at
    t=90 s, a 3.21 m SSH "error").
    Callers that genuinely need a coarser tolerance must pass it explicitly
    AND document why.
    """
    midx, ridx = [], []
    rp = 0
    for i, t in enumerate(model_time):
        while rp < len(ref_time) and ref_time[rp] < t - tol:
            rp += 1
        if rp < len(ref_time) and abs(float(ref_time[rp]) - float(t)) <= tol:
            midx.append(i)
            ridx.append(rp)
            rp += 1
    return midx, ridx


def _assert_no_step_mismatch(
    model_time: np.ndarray,
    ref_time: np.ndarray,
    aligned_model_idx: list[int],
    aligned_ref_idx: list[int],
) -> None:
    """Hard-fail when LOCOMgcm and reference snapshot times do not line up.

    A silent step shift masquerades as a "small numerical error", so stage4
    refuses to write metrics when the LOCOMgcm and reference time grids disagree
    by more than FP64 noise. The error message includes the side-by-side
    table so the user can immediately spot which side ran the wrong
    number of steps.

    The check has two prongs:

    1. Every LOCOMgcm snapshot must have a reference partner AND vice versa.
       If ``len(aligned) < max(len(model_time), len(ref_time))`` then either
       a frame is dropped (step-shift) or the two sides dumped a different
       number of frames (e.g. mismatched dump cadence) — both are refused.
       Using ``max`` (not ``min``) is deliberate: with min() a 31-vs-2 frame
       asymmetry passes silently as a degraded 2-frame comparison.
    2. Among the aligned pairs, each pairwise time gap must be < 1 ms
       (the helper above already enforces this when matching, but we
       re-check defensively in case a caller passed a coarser tolerance).
    """
    n_model = len(model_time)
    n_ref = len(ref_time)
    n_aligned = len(aligned_model_idx)
    # EVERY frame on BOTH sides must pair, not just the smaller set — use
    # max(), not min(). A coarser dump cadence on one side (e.g. MIT dumpFreq
    # left stale in run_params.yaml while LOCOMgcm honored the live config) leaves
    # LOCOMgcm with 31 frames and MIT with 2; min() would see aligned==2==min and
    # silently PASS over a degraded 2-frame comparison. max() refuses it.
    expected = max(n_model, n_ref)

    if n_aligned < expected:
        lines = ["snapshot times do not line up between LOCOMgcm and reference run."]
        lines.append(
            f"  model snapshots: {n_model}   ref snapshots: {n_ref}   "
            f"aligned: {n_aligned}   expected: {expected}"
        )
        lines.append("  side-by-side (first 20):")
        lines.append("    idx    model_time          ref_time            "
                     "Δ (s)")
        aligned_map_model = dict(zip(aligned_model_idx, aligned_ref_idx))
        for i in range(min(20, max(n_model, n_ref))):
            mt = f"{float(model_time[i]):>18.6f}" if i < n_model else " " * 18
            rt = f"{float(ref_time[i]):>18.6f}" if i < n_ref else " " * 18
            delta = ""
            if i < n_model and i in aligned_map_model:
                ri = aligned_map_model[i]
                delta = f"{float(ref_time[ri]) - float(model_time[i]):+.6e}"
            elif i < n_model and i < n_ref:
                delta = f"{float(ref_time[i]) - float(model_time[i]):+.6e}  (unaligned)"
            lines.append(f"    {i:>4}  {mt}  {rt}  {delta}")
        lines.append(
            "  This is the step-mismatch fingerprint: LOCOMgcm and MIT"
            " produced a different number of comparable frames. Two common"
            " causes: (a) different STEP count — route both ends through"
            " defgcm.io.runner._time_steps.compute_n_steps and use"
            " `time.steps: N` rather than a `time.days` ULP-trap float; or"
            " (b) different DUMP cadence — the MIT export reads dumpFreq from"
            " model_input/run_params.yaml (frozen at stage1b), so re-running"
            " stage1c alone with a changed output.*_interval keeps the stale"
            " cadence while LOCOMgcm honors the live config. Re-run stage1b (or the"
            " full prep) so run_params.yaml is regenerated."
        )
        raise ValueError("\n".join(lines))

    # Pairwise tolerance re-check.
    worst = 0.0
    worst_i = -1
    for mi, ri in zip(aligned_model_idx, aligned_ref_idx):
        d = abs(float(ref_time[ri]) - float(model_time[mi]))
        if d > worst:
            worst = d
            worst_i = mi
    if worst > 1.0e-3:
        raise ValueError(
            "aligned snapshot pair has time mismatch above 1 ms — "
            f"max delta = {worst!r} s at model_idx={worst_i}, "
            f"model_time={float(model_time[worst_i])!r}. "
            "LOCOMgcm and MIT clocks must agree to FP64 noise."
        )


def _track_ref_iter(ref_iter, aligned_ref_idx: set[int], hints: dict):
    """Yield reference steps unchanged while recording aligned var availability."""
    for ref_seq_pos, (t_ref, ref_arrays) in enumerate(ref_iter):
        if ref_seq_pos in aligned_ref_idx:
            for name, arr in ref_arrays.items():
                var = _standard_var_name(name)
                hints["ref_observed_steps"][var] = hints["ref_observed_steps"].get(var, 0) + 1
                hints["ref_valid_counts"][var] = hints["ref_valid_counts"].get(var, 0) + int(
                    np.isfinite(np.asarray(arr)).sum()
                )
        yield t_ref, ref_arrays


def _standard_var_name(name: str) -> str:
    return "ssh" if name == "eta" else name


def _patch_legacy_adapter_paths(adapter) -> None:
    """Compatibility hook retained for older callers."""
    del adapter
