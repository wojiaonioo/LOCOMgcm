"""Declarative physical-plausibility data contract for prepared ocean fields.

The preparation boundary validates initial and boundary data before writing
``prepared_*.zarr``. Checks report physically implausible values before the
model consumes them.

The contract uses declarative checks and a shared validation engine:

  * The CONTRACT is declarative DATA (a per-variable list of named checks +
    params), kept separate from the generic ENGINE that interprets it.  Adding
    a check means adding a spec row, never new control flow.
  * LAZY / collect-all: every check on every variable is evaluated and ALL
    violations are aggregated into one report, then raised together — mirrors
    pandera's ``lazy=True`` -> ``SchemaErrors`` (failing on the first bad cell
    would hide the rest).
  * Tiered severity, QARTOD-style: ``fail`` aborts the run (loud); ``warn``
    logs and continues (the "suspect" tier) — so genuine borderline physics
    does not block, but gross implausibilities do.
  * Rich reporting: each violation records variable, check, severity,
    n_violating_cells, fraction, min/max offending value and a few sample
    ``(value, depth, indices)`` — turning a step-N bisection into a one-line
    diagnosis (``temperature depth_band_range FAIL: value 29.0 @ z=1250m``).

The check taxonomy follows the Argo Real-Time QC manual (global range, gross
range, vertical gradient) and IOOS QARTOD (gross/climatology range, flag
tiers); the engine is native numpy/xarray (zero new deps) so it is *adopting*
the established pattern, not reinventing it.

Thresholds are configurable envelopes, not verbatim literature limits.
Per-depth temperature bands allow warm deep water, and salinity limits allow
hypersaline water such as the Red Sea. Check these envelopes against the
physical domain when configuring a dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import xarray as xr

CONTRACT_VERSION = "1.0"

# Severity tiers (QARTOD-style).  Higher value = more severe.
FAIL = "fail"
WARN = "warn"
_SEVERITY_RANK = {WARN: 1, FAIL: 2}


@dataclass(frozen=True)
class Violation:
    """One failed check, with enough detail to diagnose without re-running."""

    variable: str
    check: str
    severity: str
    n_cells: int
    fraction: float
    vmin: float
    vmax: float
    samples: tuple[str, ...]
    message: str


class ContractViolationError(ValueError):
    """Raised when prepared data fails the physical-plausibility contract.

    Carries the full structured violation list (``.violations``) plus any
    non-blocking warnings (``.warnings``) so callers can inspect or re-render.
    """

    def __init__(
        self,
        violations: list[Violation],
        *,
        warnings: list[Violation] | None = None,
        version: str = CONTRACT_VERSION,
    ) -> None:
        self.violations = violations
        self.warnings = warnings or []
        self.version = version
        super().__init__(self._format())

    def _format(self) -> str:
        lines = [
            f"prepared data failed physical-plausibility contract v{self.version} "
            f"— {len(self.violations)} hard violation(s):"
        ]
        lines += [f"  - {v.message}" for v in self.violations]
        if self.warnings:
            lines.append(f"  ({len(self.warnings)} warning(s), not blocking):")
            lines += [f"    · {v.message}" for v in self.warnings]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _depth_axis(da: xr.DataArray) -> tuple[int | None, np.ndarray | None]:
    """Return (z-axis index, |depth| values) or (None, None) if no z dim."""
    if "z" not in da.dims:
        return None, None
    zaxis = da.dims.index("z")
    z = np.abs(np.asarray(da["z"].values, dtype="float64")) if "z" in da.coords else None
    if z is None:
        # No coordinate values: fabricate a level index so band checks degrade
        # gracefully (treat as 0 m — they simply will not band-discriminate).
        z = np.zeros(da.sizes["z"], dtype="float64")
    return zaxis, z


def _summarize(
    vals: np.ndarray,
    badmask: np.ndarray,
    dims: tuple[str, ...],
    zvals: np.ndarray | None,
    n_samples: int = 3,
) -> tuple[int, float, float, float, tuple[str, ...]]:
    """Count + locate the worst offenders for a boolean ``badmask``.

    Returns (n_cells, fraction, vmin_offending, vmax_offending, sample_strings).
    Samples are the cells with the largest |value|, formatted with depth +
    per-dim indices, e.g. ``29.000 @ k=33(z=1250m), latitude=300, longitude=700``.
    """
    n = int(badmask.sum())
    frac = n / badmask.size if badmask.size else 0.0
    off = vals[badmask]
    vmin = float(np.min(off)) if n else float("nan")
    vmax = float(np.max(off)) if n else float("nan")

    flat = np.flatnonzero(badmask.ravel())
    order = flat[np.argsort(-np.abs(vals.ravel()[flat]))][:n_samples]
    zaxis = dims.index("z") if "z" in dims else None
    samples: list[str] = []
    for fi in order:
        idx = np.unravel_index(int(fi), badmask.shape)
        parts: list[str] = []
        for ax, (d, ix) in enumerate(zip(dims, idx)):
            if ax == zaxis and zvals is not None:
                parts.append(f"k={ix}(z={float(zvals[ix]):.0f}m)")
            else:
                parts.append(f"{d}={ix}")
        samples.append(f"{float(vals.ravel()[fi]):.3f} @ " + ", ".join(parts))
    return n, frac, vmin, vmax, tuple(samples)


# ---------------------------------------------------------------------------
# Check primitives.  Each: (var, da, severity, **params) -> list[Violation].
# A check that passes returns [].
# ---------------------------------------------------------------------------


def _chk_not_all_nan(var: str, da: xr.DataArray, severity: str) -> list[Violation]:
    vals = np.asarray(da.values, dtype="float64")
    if np.isnan(vals).all():
        msg = f"[{var}] not_all_nan {severity.upper()}: every cell is NaN (empty/failed regrid)"
        return [Violation(var, "not_all_nan", severity, vals.size, 1.0,
                          float("nan"), float("nan"), (), msg)]
    return []


def _chk_max_nan_fraction(var: str, da: xr.DataArray, severity: str, *, frac: float = 0.0) -> list[Violation]:
    vals = np.asarray(da.values, dtype="float64")
    nan_frac = float(np.isnan(vals).mean())
    if nan_frac > frac:
        msg = (f"[{var}] max_nan_fraction {severity.upper()}: NaN fraction "
               f"{nan_frac:.4%} exceeds allowed {frac:.4%}")
        return [Violation(var, "max_nan_fraction", severity, int(np.isnan(vals).sum()),
                          nan_frac, float("nan"), float("nan"), (), msg)]
    return []


def _chk_global_range(var: str, da: xr.DataArray, severity: str, *, lo: float, hi: float) -> list[Violation]:
    vals = np.asarray(da.values, dtype="float64")
    finite = np.isfinite(vals)
    bad = finite & ((vals < lo) | (vals > hi))
    if not bad.any():
        return []
    n, fr, vmin, vmax, samples = _summarize(vals, bad, da.dims, _depth_axis(da)[1])
    msg = (f"[{var}] global_range {severity.upper()}: {n} cell(s) ({fr:.2%}) "
           f"outside [{lo}, {hi}]; offending [{vmin:.3f}, {vmax:.3f}]; e.g. " + "; ".join(samples))
    return [Violation(var, "global_range", severity, n, fr, vmin, vmax, samples, msg)]


def _chk_depth_band_range(
    var: str, da: xr.DataArray, severity: str, *, bands: list[tuple[float, float, float, float]],
) -> list[Violation]:
    """Per-depth-band [lo, hi] range check (Argo/QARTOD climatology-range style).

    ``bands`` = list of ``(z_lo, z_hi, v_lo, v_hi)`` with z in metres
    (positive-down).  Cells whose depth falls in [z_lo, z_hi) must lie in
    [v_lo, v_hi].  This is the decisive catch for deep-warm "poison water".
    """
    zaxis, z = _depth_axis(da)
    if zaxis is None or z is None:
        return []
    vals = np.asarray(da.values, dtype="float64")
    lo_arr = np.full(z.shape, -np.inf, dtype="float64")
    hi_arr = np.full(z.shape, np.inf, dtype="float64")
    for z_lo, z_hi, v_lo, v_hi in bands:
        sel = (z >= z_lo) & (z < z_hi)
        lo_arr[sel] = v_lo
        hi_arr[sel] = v_hi
    shape = [1] * vals.ndim
    shape[zaxis] = z.size
    lo_b = lo_arr.reshape(shape)
    hi_b = hi_arr.reshape(shape)
    finite = np.isfinite(vals)
    bad = finite & ((vals < lo_b) | (vals > hi_b))
    if not bad.any():
        return []
    n, fr, vmin, vmax, samples = _summarize(vals, bad, da.dims, z)
    msg = (f"[{var}] depth_band_range {severity.upper()}: {n} cell(s) ({fr:.2%}) "
           f"outside per-depth bounds; offending [{vmin:.3f}, {vmax:.3f}]; e.g. " + "; ".join(samples))
    return [Violation(var, "depth_band_range", severity, n, fr, vmin, vmax, samples, msg)]


def _chk_vertical_gradient(
    var: str, da: xr.DataArray, severity: str, *,
    deep_m: float = 500.0, d_shallow: float = 35.0, d_deep: float = 22.0,
) -> list[Violation]:
    """Adjacent-level |Δ| check (Argo Test 11 family) — catches local spikes.

    Thresholds are deliberately loose (above the known-good field's land-fill
    discontinuities) so this is a soft signal; the hard catch is
    ``depth_band_range``.  Pairs shallower than ``deep_m`` use ``d_shallow``.
    """
    zaxis, z = _depth_axis(da)
    if zaxis is None or z is None or da.sizes["z"] < 2:
        return []
    vals = np.asarray(da.values, dtype="float64")
    dvals = np.abs(np.diff(vals, axis=zaxis))
    zmid = z[:-1]
    thr = np.where(zmid < deep_m, d_shallow, d_deep)
    shape = [1] * dvals.ndim
    shape[zaxis] = zmid.size
    thr_b = thr.reshape(shape)
    finite = np.isfinite(dvals)
    bad = finite & (dvals > thr_b)
    if not bad.any():
        return []
    n, fr, vmin, vmax, samples = _summarize(dvals, bad, da.dims, zmid)
    msg = (f"[{var}] vertical_gradient {severity.upper()}: {n} level-pair(s) "
           f"with |Δ| > threshold (shallow {d_shallow}/deep {d_deep}); "
           f"max |Δ|={vmax:.3f}; e.g. " + "; ".join(samples))
    return [Violation(var, "vertical_gradient", severity, n, fr, vmin, vmax, samples, msg)]


_CHECKS: dict[str, Callable[..., list[Violation]]] = {
    "not_all_nan": _chk_not_all_nan,
    "max_nan_fraction": _chk_max_nan_fraction,
    "global_range": _chk_global_range,
    "depth_band_range": _chk_depth_band_range,
    "vertical_gradient": _chk_vertical_gradient,
}


# ---------------------------------------------------------------------------
# The default declarative contract for prepared ocean state / boundary fields.
# Thresholds anchored to the clean global-p25 envelope + margin (see module
# docstring).  Variables absent from a given bundle are simply skipped.
# ---------------------------------------------------------------------------

OCEAN_IC_CONTRACT: dict[str, list[dict[str, Any]]] = {
    "temperature": [
        {"check": "not_all_nan", "severity": FAIL},
        {"check": "max_nan_fraction", "severity": FAIL, "frac": 0.0},
        # Argo Test 6 gross range (degC).
        {"check": "global_range", "severity": FAIL, "lo": -3.0, "hi": 40.0},
        # Depth-banded climatology range — the decisive poison catch.
        # Deep-water cap allows warm Red Sea water while rejecting hotter outliers.
        {"check": "depth_band_range", "severity": FAIL, "bands": [
            (0.0, 300.0, -3.0, 40.0),
            (300.0, 1000.0, -3.0, 33.0),
            (1000.0, 2000.0, -3.0, 27.0),
            (2000.0, 1.0e9, -3.0, 26.0),
        ]},
        # Soft local-spike signal (loose; clean field has ~19 degC land-fill jumps).
        {"check": "vertical_gradient", "severity": WARN,
         "deep_m": 500.0, "d_shallow": 35.0, "d_deep": 22.0},
    ],
    "salinity": [
        {"check": "not_all_nan", "severity": FAIL},
        {"check": "max_nan_fraction", "severity": FAIL, "frac": 0.0},
        # lo permissive (coastal/fill ~0 PSU); hi 45 passes Red-Sea/Gulf ~42.3.
        {"check": "global_range", "severity": FAIL, "lo": -0.1, "hi": 45.0},
    ],
    "u": [{"check": "global_range", "severity": FAIL, "lo": -6.0, "hi": 6.0}],
    "v": [{"check": "global_range", "severity": FAIL, "lo": -6.0, "hi": 6.0}],
    "eta": [{"check": "global_range", "severity": FAIL, "lo": -6.0, "hi": 6.0}],
}


def validate_contract(
    ds: xr.Dataset,
    contract: dict[str, list[dict[str, Any]]] | None = None,
    *,
    raise_on: str = FAIL,
    logger: Any | None = None,
) -> list[Violation]:
    """Run every check on every contracted variable; collect ALL violations.

    Variables in ``contract`` but absent from ``ds`` are skipped (so the same
    contract serves state, boundary and partial bundles).  Warnings are logged
    via ``logger`` (if given); any violation with severity >= ``raise_on`` makes
    this raise :class:`ContractViolationError`.  Returns the full violation list
    (including warnings) when nothing reaches the raise threshold.
    """
    contract = contract if contract is not None else OCEAN_IC_CONTRACT
    threshold = _SEVERITY_RANK.get(raise_on, _SEVERITY_RANK[FAIL])

    all_violations: list[Violation] = []
    for var, checks in contract.items():
        if var not in ds.data_vars:
            continue
        da = ds[var]
        for spec in checks:
            name = spec["check"]
            severity = spec.get("severity", FAIL)
            params = {k: v for k, v in spec.items() if k not in ("check", "severity")}
            fn = _CHECKS.get(name)
            if fn is None:
                raise KeyError(f"unknown data-contract check '{name}' for variable '{var}'")
            all_violations.extend(fn(var, da, severity, **params))

    blocking = [v for v in all_violations if _SEVERITY_RANK.get(v.severity, 0) >= threshold]
    warnings = [v for v in all_violations if _SEVERITY_RANK.get(v.severity, 0) < threshold]

    if logger is not None and warnings:
        for v in warnings:
            logger.warning("data-contract: %s", v.message)

    if blocking:
        raise ContractViolationError(blocking, warnings=warnings, version=CONTRACT_VERSION)
    return all_violations
