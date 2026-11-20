"""Systematic LOCOMgcm-vs-MIT verification metrics (CNT line type).

Computes **continuous deterministic
skill** (METplus CNT line type) of a model field vs a reference, with one
finite-volume weighting convention everywhere, instead of per-module
hand-rolled rmse/corr (each with its own inconsistent weighting + masking).

Two layers:

* **Single-pass weighted-moment core** (``cnt_accumulate`` -> ``aggregate_sums``
  -> ``cnt_from_sums``) is the production engine. It reads each variable's
  (time[, depth], y, x) array ONCE, accumulating per-cell time-sums + per-frame
  weighted sums; every CNT metric (bias/rmse/mae/corr/r2/slope/std-ratio) is a
  closed form of those sums. This is what makes 720x720x50x31 grids tractable:
  a benchmark showed dense per-metric ``xskillscore`` at 40-280 s
  *per metric per variable* (pearson_r alone 249 s), vs ~40 s for ALL metrics
  here — and the per-cell sums make the stratified breakdown almost free
  (re-weight the same sums by each bin mask, no extra field pass).
* **xskillscore adapters** (``cnt_metrics``/``cnt_series``/``cnt_lean``) remain
  as the validation oracle: the moment-core formulas are unit-tested to match
  them bitwise on small arrays. They are also the entry point for the
  categorical (CTS) / neighborhood (FSS via ``scores``) work.

Weighting convention: ``hFac`` (partial-cell, land-sea) x ``cos(lat)`` (spherical
area) x ``drF`` (layer thickness, 3D). Land/dry cells get zero weight.

Scope: CNT only. CTS, FSS, and spectral-skill live in sibling modules.
Probabilistic metrics are omitted — the comparison is deterministic (no ensemble).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import xarray as xr
import xskillscore as xs

__all__ = [
    "CNT_METRICS",
    "cnt_accumulate",
    "aggregate_sums",
    "global_from_frames",
    "cnt_from_sums",
    "series_from_frames",
    "cnt_metrics",
    "cnt_series",
    "cnt_lean",
    "area_hfac_weights",
    "to_dataarray",
    "bootstrap_cnt",
]

# Per-cell time-sum accumulators (one entry per quantity reduced over time).
_CELL_KEYS = ("cn", "sm", "sr", "smm", "srr", "smr", "sabsd", "sd2")

# CNT metric -> xskillscore callable. Kept explicit so the schema/docs and the
# dashboard legend stay in lock-step with what is actually computed.
CNT_METRICS: tuple[str, ...] = (
    "me",          # mean error = bias (model - ref)
    "mae",         # mean absolute error
    "rmse",        # root mean squared error
    "mse",         # mean squared error
    "pearson_r",   # linear correlation
    "r2",          # coefficient of determination
    "linslope",    # least-squares slope (model on ref)
)

# Small floor below which correlation/slope denominators are degenerate.
_STD_FLOOR = 1.0e-14


# ---------------------------------------------------------------------------
# Single-pass weighted-moment core (production CNT engine)
# ---------------------------------------------------------------------------

def cnt_accumulate(
    model: Any,
    ref: Any,
    w3: np.ndarray | None,
    wet: np.ndarray | None,
    *,
    want_cell: bool = True,
    want_series: bool = True,
) -> tuple[dict[str, Any] | None, list[dict[str, float]] | None]:
    """One pass over the leading (time) axis of ``model``/``ref``.

    ``model``/``ref`` are ``(nt[, nz], ny, nx)`` array-likes (mmap fine — read
    frame by frame, so peak memory is one slice, not the whole field). ``w3`` is
    the time-constant per-cell weight (spatial shape) or ``None``; ``wet`` is a
    spatial bool mask (or ``None``). NaNs in either field are dropped pairwise.

    Returns ``(cell, frames)``:

    * ``cell`` — per-cell time-sums (spatial-shaped arrays under :data:`_CELL_KEYS`
      plus a scalar ``maxabs``), or ``None`` when ``want_cell`` is False. These
      are needed ONLY for the stratified breakdown (every stratum re-weights
      them); the global metric also derives from ``frames``, so the cheap
      fast-path skips the per-cell accumulation entirely (``want_cell=False``).
    * ``frames`` — per-frame weighted sums (global = their sum; also the
      per-snapshot skill curve). ``None`` when ``want_series`` is False.
    """
    nt = int(model.shape[0])
    spatial = tuple(model.shape[1:])
    cell = {k: np.zeros(spatial, dtype=np.float64) for k in _CELL_KEYS} if want_cell else None
    maxabs = 0.0
    frames: list[dict[str, float]] | None = [] if want_series else None
    w3a = None if w3 is None else np.asarray(w3, dtype=np.float64)
    weta = None if wet is None else np.asarray(wet, dtype=bool)

    for t in range(nt):
        mt = np.asarray(model[t], dtype=np.float64)
        rt = np.asarray(ref[t], dtype=np.float64)
        valid = np.isfinite(mt) & np.isfinite(rt)
        if weta is not None:
            valid &= weta
        m = np.where(valid, mt, 0.0)
        r = np.where(valid, rt, 0.0)
        d = m - r
        ad = np.abs(d)
        frame_max = float(ad[valid].max()) if valid.any() else 0.0
        maxabs = max(maxabs, frame_max)
        if cell is not None:
            cell["cn"] += valid
            cell["sm"] += m
            cell["sr"] += r
            cell["smm"] += m * m
            cell["srr"] += r * r
            cell["smr"] += m * r
            cell["sabsd"] += ad
            cell["sd2"] += d * d
        if frames is not None:
            wt = (np.where(valid, w3a, 0.0) if w3a is not None else valid.astype(np.float64))
            frames.append({
                "W": float(wt.sum()),
                "n": int(valid.sum()),
                "swm": float((wt * m).sum()),
                "swr": float((wt * r).sum()),
                "swmm": float((wt * m * m).sum()),
                "srr": float((wt * r * r).sum()),
                "smr": float((wt * m * r).sum()),
                "sabsd": float((wt * ad).sum()),
                "sd2": float((wt * d * d).sum()),
                "maxabs": frame_max,
            })
    if cell is not None:
        cell["maxabs"] = maxabs
    return cell, frames


def global_from_frames(frames: list[dict[str, float]]) -> dict[str, float]:
    """Global weighted sums = sum of the per-frame sums (additive); maxabs = max.

    Lets the fast path compute the all-time global CNT without the per-cell
    accumulation (which is only needed to stratify)."""
    keys = ("n", "W", "swm", "swr", "swmm", "srr", "smr", "sabsd", "sd2")
    out: dict[str, float] = {k: 0.0 for k in keys}
    maxabs = 0.0
    for f in frames:
        for k in keys:
            out[k] += f[k]
        maxabs = max(maxabs, f.get("maxabs", 0.0))
    out["n"] = int(out["n"])
    out["maxabs"] = maxabs
    return out


def aggregate_sums(cell: dict[str, Any], weight: np.ndarray | None) -> dict[str, float]:
    """Collapse per-cell time-sums to global weighted sums under ``weight``.

    ``weight`` is the per-cell weight field (``w3`` for the global metric, or
    ``w3 * bin_mask`` for a stratum); ``None`` means uniform unit weight.
    """
    if weight is None:
        weight = np.ones_like(cell["cn"])
    w = np.asarray(weight, dtype=np.float64)
    # n counts valid (t,cell) samples in the *selected* region — i.e. cells the
    # weight actually includes (w>0), so a stratum weight (w3*bin_mask) yields
    # that bin's own count, not the global one.
    return {
        "n": int(((w > 0.0) * cell["cn"]).sum()),
        "W": float((w * cell["cn"]).sum()),
        "swm": float((w * cell["sm"]).sum()),
        "swr": float((w * cell["sr"]).sum()),
        "swmm": float((w * cell["smm"]).sum()),
        "srr": float((w * cell["srr"]).sum()),
        "smr": float((w * cell["smr"]).sum()),
        "sabsd": float((w * cell["sabsd"]).sum()),
        "sd2": float((w * cell["sd2"]).sum()),
        "maxabs": cell.get("maxabs"),
    }


def cnt_from_sums(
    *,
    n: int,
    W: float,
    swm: float,
    swr: float,
    swmm: float,
    srr: float,
    smr: float,
    sabsd: float,
    sd2: float,
    maxabs: float | None = None,
    lean: bool = False,
) -> dict[str, Any]:
    """Closed-form CNT metrics from weighted sums (see :func:`aggregate_sums`).

    Matches the corresponding ``xskillscore`` weighted metric bitwise (pinned in
    tests). ``lean`` returns only me/mae/rmse/pearson_r/n_valid (the per-stratum
    path); the full payload adds mse/r2/linslope/ref_rms/std_ratio/rel_rmse.
    """
    if n == 0 or W <= 0.0:
        out = {"me": None, "mae": None, "rmse": None, "pearson_r": None, "n_valid": int(n)}
        if lean:
            return out
        out.update({"mse": None, "r2": None, "linslope": None, "ref_rms": None,
                    "model_rms": None, "rel_rmse": None, "std_ratio": None, "max_abs": maxabs})
        return out

    mbar = swm / W
    rbar = swr / W
    me = (swm - swr) / W
    mae = sabsd / W
    mse = sd2 / W
    rmse = float(np.sqrt(mse))
    var_m = max(swmm / W - mbar * mbar, 0.0)
    var_r = max(srr / W - rbar * rbar, 0.0)
    cov = smr / W - mbar * rbar
    std_m = float(np.sqrt(var_m))
    std_r = float(np.sqrt(var_r))
    if std_m > _STD_FLOOR and std_r > _STD_FLOOR:
        pearson = float(np.clip(cov / (std_m * std_r), -1.0, 1.0))
    else:
        pearson = None
    out: dict[str, Any] = {"me": me, "mae": mae, "rmse": rmse, "pearson_r": pearson, "n_valid": int(n)}
    if lean:
        return out
    ref_rms = float(np.sqrt(srr / W))
    model_rms = float(np.sqrt(swmm / W))
    out.update({
        "mse": mse,
        # r2 and linslope follow xskillscore's convention, which uses the FIRST
        # argument (the model) as the regression basis: SS_tot = W * var(model),
        # slope = cov / var(model). Pinned bitwise to xs.r2 / xs.linslope.
        "r2": (1.0 - sd2 / (W * var_m)) if var_m > _STD_FLOOR else None,
        "linslope": (cov / var_m) if var_m > _STD_FLOOR else None,
        "ref_rms": ref_rms,
        "model_rms": model_rms,
        "rel_rmse": (rmse / ref_rms) if ref_rms > 1.0e-30 else None,
        "std_ratio": (std_m / std_r) if std_r > _STD_FLOOR else None,
        "max_abs": maxabs,
    })
    return out


def series_from_frames(
    frames: list[dict[str, float]],
    metrics: Sequence[str] = ("rmse", "me", "pearson_r"),
) -> dict[str, list[float | None]]:
    """Per-snapshot CNT curve from the per-frame weighted sums."""
    out: dict[str, list[float | None]] = {m: [] for m in metrics}
    for f in frames:
        blk = cnt_from_sums(
            n=f["n"], W=f["W"], swm=f["swm"], swr=f["swr"], swmm=f["swmm"],
            srr=f["srr"], smr=f["smr"], sabsd=f["sabsd"], sd2=f["sd2"], lean=True,
        )
        for m in metrics:
            v = blk.get(m)
            out[m].append(None if v is None or not np.isfinite(v) else float(v))
    return out


def to_dataarray(
    arr: np.ndarray,
    *,
    is3d: bool,
    name: str = "field",
) -> xr.DataArray:
    """Wrap a raw (nt[, nz], ny, nx) numpy/mmap array as a labelled DataArray.

    Land/missing cells must already be NaN (xskillscore skips them pairwise).
    """
    a = np.asarray(arr, dtype=np.float64)
    if is3d:
        if a.ndim != 4:
            raise ValueError(f"{name}: expected (nt,nz,ny,nx), got {a.shape}")
        dims = ("time", "depth", "y", "x")
    else:
        if a.ndim != 3:
            raise ValueError(f"{name}: expected (nt,ny,nx), got {a.shape}")
        dims = ("time", "y", "x")
    return xr.DataArray(a, dims=dims, name=name)


def area_hfac_weights(
    hfac: np.ndarray | None,
    lat: np.ndarray | None,
    *,
    is3d: bool,
    shape: tuple[int, ...],
    drf: np.ndarray | None = None,
) -> xr.DataArray | None:
    """Build the canonical finite-volume weight field.

    The weight of each cell is its true wet volume fraction:

        w = hFac (partial-cell fraction, land-sea + bathymetry)
            x cos(lat)   (spherical horizontal cell area; dx = R cos(lat) dlon)
            x drF[k]     (layer thickness — 3D only, makes the depth reduction a
                          genuine volume integral rather than a per-level count)

    * ``hfac`` (nz,ny,nx) in [0,1]; absent -> uniform wet weight 1. For a 2D
      surface field the any-wet column fraction (max over depth) is used.
    * ``lat`` (deg, len ny) -> cos-lat area weighting; skipped (uniform) when
      non-geographic (|lat|>90, e.g. the index/metre axis bug) or missing.
    * ``drf`` (nz,) layer thicknesses (m) -> volume weighting for 3D fields.

    Returns a DataArray (dims (depth,y,x) or (y,x)) broadcastable against
    :func:`to_dataarray` output, or ``None`` when no weight source is available
    (xskillscore then reduces unweighted over the NaN-masked field).
    """
    if is3d:
        nz, ny, nx = shape[1], shape[2], shape[3]
        dims = ("depth", "y", "x")
        base = (
            np.asarray(hfac, dtype=np.float64)
            if hfac is not None and np.asarray(hfac).shape == (nz, ny, nx)
            else np.ones((nz, ny, nx), dtype=np.float64)
        )
        if drf is not None:
            drf_arr = np.asarray(drf, dtype=np.float64)
            if drf_arr.size == nz:
                base = base * drf_arr[:, None, None]  # -> wet volume per cell
    else:
        ny, nx = shape[1], shape[2]
        dims = ("y", "x")
        if hfac is not None and np.asarray(hfac).ndim == 3:
            base = np.asarray(hfac, dtype=np.float64).max(axis=0)  # any-wet column
        else:
            base = np.ones((ny, nx), dtype=np.float64)

    have_hfac = hfac is not None
    cosw = None
    if lat is not None:
        lat_arr = np.asarray(lat, dtype=np.float64)
        if lat_arr.size == ny and float(np.nanmax(np.abs(lat_arr))) <= 90.0:
            cosw = np.clip(np.cos(np.deg2rad(lat_arr)), 0.0, None)

    if not have_hfac and cosw is None and drf is None:
        return None

    w = base
    if cosw is not None:
        w = w * (cosw[None, :, None] if is3d else cosw[:, None])
    return xr.DataArray(w, dims=dims, name="weights")


def cnt_metrics(
    model: xr.DataArray,
    ref: xr.DataArray,
    *,
    dim: Sequence[str],
    weights: xr.DataArray | None = None,
    skipna: bool = True,
) -> dict[str, Any]:
    """Continuous deterministic skill of ``model`` vs ``ref`` over ``dim``.

    Returns a flat dict of CNT metrics (see :data:`CNT_METRICS`) plus context:
    n_eff (effective sample size accounting for autocorrelation), pearson_r
    p-value, ref/model rms, rel_rmse, std_ratio, and the valid-pair count. All
    weighted + masked identically via ``weights`` + NaN-skipping.
    """
    dim = list(dim)
    kw = dict(dim=dim, skipna=skipna)
    wkw = _wkw(kw, weights, model)

    def f(name, **extra):
        try:
            return float(getattr(xs, name)(model, ref, **{**wkw, **extra}).values)
        except Exception:
            return None

    out: dict[str, Any] = {
        "me": f("me"),
        "mae": f("mae"),
        "rmse": f("rmse"),
        "mse": f("mse"),
        "pearson_r": f("pearson_r"),
        "r2": f("r2"),
        "linslope": f("linslope"),
    }
    # significance / sample size (autocorrelation-aware) — no weights kwarg
    try:
        out["pearson_r_p_value"] = float(xs.pearson_r_p_value(model, ref, **kw).values)
    except Exception:
        out["pearson_r_p_value"] = None
    try:
        out["n_eff"] = float(xs.effective_sample_size(model, ref, **kw).values)
    except Exception:
        out["n_eff"] = None

    # derived context (ref_rms / rel_rmse / std_ratio) computed consistently
    ref_rms = _weighted_rms(ref, dim, weights, skipna)
    model_rms = _weighted_rms(model, dim, weights, skipna)
    out["ref_rms"] = ref_rms
    out["model_rms"] = model_rms
    out["rel_rmse"] = (
        out["rmse"] / ref_rms if (out["rmse"] is not None and ref_rms and ref_rms > 1e-30) else None
    )
    rstd = _weighted_std(ref, dim, weights, skipna)
    mstd = _weighted_std(model, dim, weights, skipna)
    out["std_ratio"] = (mstd / rstd if (rstd and rstd > _STD_FLOOR) else None)
    out["n_valid"] = int(np.isfinite(model.values + ref.values).sum())
    return out


def cnt_series(
    model: xr.DataArray,
    ref: xr.DataArray,
    *,
    dim: Sequence[str],
    weights: xr.DataArray | None = None,
    skipna: bool = True,
    metrics: Sequence[str] = ("rmse", "me", "pearson_r"),
) -> dict[str, list[float | None]]:
    """CNT metrics reduced over ``dim`` while *keeping* the remaining axis.

    With ``dim=["depth","y","x"]`` on a ``(time, depth, y, x)`` field this is the
    per-snapshot global skill curve (one value per ``time``). Returns
    ``{metric: [v0, v1, ...]}`` aligned to the surviving axis; a metric that
    raises maps to ``None`` (and its per-step list is ``None``).
    """
    kw = dict(dim=list(dim), skipna=skipna)
    wkw = _wkw(kw, weights, model)
    out: dict[str, list[float | None]] = {}
    for name in metrics:
        try:
            da = getattr(xs, name)(model, ref, **wkw)
            out[name] = [
                None if not np.isfinite(v) else float(v)
                for v in np.atleast_1d(np.asarray(da.values, dtype=np.float64))
            ]
        except Exception:
            out[name] = None
    return out


def cnt_lean(
    model: xr.DataArray,
    ref: xr.DataArray,
    *,
    dim: Sequence[str],
    weights: xr.DataArray | None = None,
    skipna: bool = True,
    names: Sequence[str] = ("me", "mae", "rmse", "pearson_r"),
) -> dict[str, float | None]:
    """Reduced CNT (no derived ref_rms/std/p-value context) for the per-bin
    stratified loop, where computing the full :func:`cnt_metrics` payload x ~33
    bins x N variables would be wasteful. One scalar per requested metric.
    """
    kw = dict(dim=list(dim), skipna=skipna)
    wkw = _wkw(kw, weights, model)
    out: dict[str, float | None] = {}
    for name in names:
        try:
            out[name] = float(getattr(xs, name)(model, ref, **wkw).values)
        except Exception:
            out[name] = None
    return out


def _wkw(kw: dict[str, Any], weights: xr.DataArray | None, like: xr.DataArray) -> dict[str, Any]:
    """Return ``kw`` augmented with a ``weights`` entry spanning exactly ``dim``.

    xskillscore requires ``weights`` to match the *reduced* dims — no fewer (it
    does not auto-broadcast a missing ``time``) and no more (a kept axis such as
    ``time`` in a per-snapshot reduction raises). So a (depth,y,x) volume-weight
    field is broadcast against ``like`` restricted to the reduce dims: it gains
    ``time`` for a full reduction, stays (depth,y,x) for a per-snapshot one.
    """
    if weights is None:
        return kw
    dims = list(kw["dim"])
    template = like.isel({d: 0 for d in like.dims if d not in dims}, drop=True)
    return dict(kw, weights=weights.broadcast_like(template))


def _weighted_rms(a: xr.DataArray, dim, weights, skipna) -> float | None:
    try:
        sq = a * a
        if weights is not None:
            m = sq.weighted(weights.fillna(0.0)).mean(dim=dim, skipna=skipna)
        else:
            m = sq.mean(dim=dim, skipna=skipna)
        return float(np.sqrt(float(m.values)))
    except Exception:
        return None


def _weighted_std(a: xr.DataArray, dim, weights, skipna) -> float | None:
    try:
        if weights is not None:
            return float(a.weighted(weights.fillna(0.0)).std(dim=dim, skipna=skipna).values)
        return float(a.std(dim=dim, skipna=skipna).values)
    except Exception:
        return None


def bootstrap_cnt(
    model: xr.DataArray,
    ref: xr.DataArray,
    *,
    dim: Sequence[str],
    weights: xr.DataArray | None = None,
    metrics: Sequence[str] = ("rmse", "me", "pearson_r"),
    n_boot: int = 200,
    ci: float = 0.95,
) -> dict[str, dict[str, float]]:
    """Block-bootstrap CI for selected CNT metrics by resampling whole ``time``
    slices (preserves spatial structure + respects temporal autocorrelation).

    Returns ``{metric: {"lo": .., "hi": .., "mean": ..}}``. Falls back to a
    point estimate (lo==hi) when there is no time dim or only one slice.
    """
    dim = list(dim)
    if "time" not in model.dims or model.sizes.get("time", 1) < 3:
        pt = cnt_metrics(model, ref, dim=dim, weights=weights)
        return {m: {"lo": pt.get(m), "hi": pt.get(m), "mean": pt.get(m)} for m in metrics}

    # Library-native block bootstrap: resample whole `time` slices (preserves
    # spatial structure + temporal autocorrelation) with xskillscore, keeping
    # the metric reduction vectorized over the new `iteration` dim.
    ds = xr.Dataset({"m": model, "r": ref})
    ds_b = xs.resample_iterations_idx(ds, int(n_boot), dim="time")  # adds 'iteration'
    lo_q, hi_q = (1.0 - ci) / 2.0, 1.0 - (1.0 - ci) / 2.0
    wkw = _wkw(dict(dim=dim, skipna=True), weights, ds_b["m"])  # weights span `dim`, not 'iteration'
    out: dict[str, dict[str, float]] = {}
    for m in metrics:
        try:
            dist = getattr(xs, m)(ds_b["m"], ds_b["r"], **wkw)
            q = dist.quantile([lo_q, hi_q], dim="iteration")
            out[m] = {
                "lo": float(q.isel(quantile=0).values),
                "hi": float(q.isel(quantile=1).values),
                "mean": float(dist.mean(dim="iteration").values),
            }
        except Exception:
            pt = cnt_metrics(model, ref, dim=dim, weights=weights).get(m)
            out[m] = {"lo": pt, "hi": pt, "mean": pt}
    return out
