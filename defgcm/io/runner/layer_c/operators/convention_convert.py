"""ConventionConverter — LOCOMgcm ↔ MIT ↔ CF sign / layout / dtype dispatch.

This is the highest-silent-failure-risk operator in the IO pipeline.
Each sign-convention row in ``docs/NUMERICS.md``
table has a dispatch entry below. ConventionEvent is emitted to provenance
so manifest can audit every conversion.

Implemented conventions:

| Field | LOCOMgcm | MIT | CF |
|---|---|---|---|
| Vertical velocity `w` | downward-positive | upward-positive | CF standard = upward-positive (matches MIT) |
| Bathymetry | positive ocean depth | runtime positive depth (bathyFile.bin negative) | GEBCO 'elevation' = positive on land, negative below sea level |
| Heat flux (`heat_flux` / Qnet) | LOCOMgcm `heat_flux > 0` warms ocean | MIT Qnet > 0 cools (upward flux) | CF ambiguous; we assume input bundle declares its own |
| Salt flux | LOCOMgcm `salt_flux > 0` increases salinity | MIT EmPmR > 0 increases salinity (same); saltFlux > 0 decreases | n/a here |

Heat flux handling assumes the input bundle's `heat_flux` variable carries
the source's own sign semantics. ConventionConverter flips sign when crossing
LOCOMgcm ↔ MIT boundary.

Sign conventions per the MDS reader semantics: MDS raw output already in MIT
convention; this operator converts MIT → LOCOMgcm for stage4 zarr.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from defgcm.io.runner.layer_a import (
    BundleMeta,
    ConventionTag,
    DataBundle,
    LocationKind,
    VariableSpec,
)

from .._registry import register_operator
from .._types import OperatorCapabilities


# ============================================================
# Dispatch tables
# ============================================================
# Each entry: variable name → action when converting (src_conv → dst_conv)
#
# Actions:
#   "negate"          — multiply array by -1
#   "passthrough"     — copy as-is
#   "negate_ocean"    — set negative values to positive (or vice versa); used for
#                       bathy_elevation → bathy_positive_depth (LOCOMgcm/MIT runtime)
#                       OR bathy_positive_depth → bathy_negative_ocean (MIT input file)
#
# The conversion graph (only direct conversions supported; 2-step via CF):
#   CF_STANDARD → TORCH_OCEAN  (uses _CF_TO_TO_TABLE)
#   CF_STANDARD → MITGCM       (uses _CF_TO_MIT_TABLE)
#   MITGCM → TORCH_OCEAN       (uses _MIT_TO_TO_TABLE)
#   TORCH_OCEAN → MITGCM       (uses _TO_TO_MIT_TABLE)
#   TORCH_OCEAN → CF_STANDARD  (uses _TO_TO_CF_TABLE)
#   OPAQUE → any               (assume identity; warn via ConventionEvent)


def _negate(arr: np.ndarray) -> np.ndarray:
    return -arr


def _passthrough(arr: np.ndarray) -> np.ndarray:
    return arr


def _gebco_to_positive_depth(arr: np.ndarray) -> np.ndarray:
    """GEBCO 'elevation' (positive land, negative sea level) → positive depth (LOCOMgcm/MIT runtime).

    Negate the negative ocean values; clamp positive (land) to 0.
    """
    out = np.where(arr < 0, -arr, 0.0).astype(arr.dtype)
    return out


def _gebco_to_mit_input_file(arr: np.ndarray) -> np.ndarray:
    """GEBCO 'elevation' → MIT bathyFile.bin (negative ocean, land 0).

    Goes through LOCOMgcm positive depth, so the values written to
    bathyFile.bin come from the one LOCOMgcm → bathyFile conversion
    (``_to_positive_depth_to_mit_input_file``).  The float64 view keeps both
    negations exact for integer elevation, whose most negative value has no
    positive counterpart in its own dtype.
    """
    depth = _gebco_to_positive_depth(np.asarray(arr, dtype=np.float64))
    return _to_positive_depth_to_mit_input_file(depth).astype(arr.dtype)


def _to_positive_depth_to_mit_input_file(arr: np.ndarray) -> np.ndarray:
    """LOCOMgcm positive ocean depth (e.g. 1000.0) → MIT bathyFile.bin negative (-1000.0).

    Delegates to ``experiments/alignment/mitgcm_export.py``
    ``project_bathymetry_to_mitgcm``, the only bathyFile.bin sign conversion
    numerics rule 9 allows; ``layer_d/consumers/mit_binary.py`` imports the same
    module for its other MITgcm writers.  That function squeezes size-1 axes
    and returns float64, so the input shape and dtype are restored here.
    """
    from experiments.alignment.mitgcm_export import project_bathymetry_to_mitgcm

    return project_bathymetry_to_mitgcm(arr).reshape(arr.shape).astype(arr.dtype)


def _to_positive_depth_to_cf_elevation(arr: np.ndarray) -> np.ndarray:
    """LOCOMgcm positive ocean depth → CF elevation (negative below sea level); land stays 0."""
    out = np.where(arr > 0, -arr, 0.0).astype(arr.dtype)
    return out


# CF_STANDARD → TORCH_OCEAN
_CF_TO_TO_TABLE: dict[str, Any] = {
    "bathymetry": _gebco_to_positive_depth,   # elevation → positive depth
    "w": _negate,                              # CF upward → LOCOMgcm downward
    # heat_flux: depends on source; CF standard varies. We treat CF as
    # "downward into ocean = positive" (CF surface_downward_heat_flux),
    # matching LOCOMgcm semantics → passthrough.
    "heat_flux": _passthrough,
    "salt_flux": _passthrough,
    "freshwater_flux": _passthrough,
}

# CF_STANDARD → MITGCM
_CF_TO_MIT_TABLE: dict[str, Any] = {
    "bathymetry": _gebco_to_mit_input_file,   # elevation → MIT bathyFile.bin negative
    "w": _passthrough,                          # CF upward = MIT wVel upward
    "heat_flux": _negate,                       # CF downward = LOCOMgcm heat_flux > 0 warms;
                                                # MIT Qnet > 0 cools → negate
    "salt_flux": _passthrough,                  # CF → MIT EmPmR semantics
    "freshwater_flux": _passthrough,
}

# MITGCM → TORCH_OCEAN
_MIT_TO_TO_TABLE: dict[str, Any] = {
    "w": _negate,                               # MIT upward → LOCOMgcm downward
    # bathymetry: at MIT runtime, Depth.data is positive (same as LOCOMgcm); no flip
    "bathymetry": _passthrough,
    "heat_flux": _negate,                       # MIT Qnet (cool-positive) → LOCOMgcm heat_flux (warm-positive)
    "salt_flux": _passthrough,
}

# TORCH_OCEAN → MITGCM
_TO_TO_MIT_TABLE: dict[str, Any] = {
    "w": _negate,                               # LOCOMgcm downward → MIT upward
    "bathymetry": _to_positive_depth_to_mit_input_file,
    "heat_flux": _negate,                       # LOCOMgcm heat_flux (warm-pos) → MIT Qnet (cool-pos)
    "salt_flux": _passthrough,
}


# TORCH_OCEAN → CF_STANDARD  (exact inverse of _CF_TO_TO_TABLE)
#
# Needed by any source that reads native model-grid fields (e.g. the MITgcm
# verification binaries) and has to hand them to the CF_STANDARD stage1a
# consumers.  Every entry is the inverse of the CF → LOCOMgcm row above, so a
# CF → LOCOMgcm → CF round trip is the identity for ocean points; land is 0 on both
# sides of the bathymetry rule (CF elevation would be positive on land, which
# neither MITgcm's bathyFile nor LOCOMgcm's depth carries in the first place).
_TO_TO_CF_TABLE: dict[str, Any] = {
    "bathymetry": _to_positive_depth_to_cf_elevation,    # positive depth → negative elevation
    "w": _negate,                                        # LOCOMgcm downward → CF upward
    "heat_flux": _passthrough,
    "salt_flux": _passthrough,
    "freshwater_flux": _passthrough,
}


_DISPATCH: dict[tuple[ConventionTag, ConventionTag], dict[str, Any]] = {
    (ConventionTag.CF_STANDARD, ConventionTag.TORCH_OCEAN): _CF_TO_TO_TABLE,
    (ConventionTag.CF_STANDARD, ConventionTag.MITGCM): _CF_TO_MIT_TABLE,
    (ConventionTag.MITGCM, ConventionTag.TORCH_OCEAN): _MIT_TO_TO_TABLE,
    (ConventionTag.TORCH_OCEAN, ConventionTag.MITGCM): _TO_TO_MIT_TABLE,
    (ConventionTag.TORCH_OCEAN, ConventionTag.CF_STANDARD): _TO_TO_CF_TABLE,
}


@register_operator("convention_convert")
class ConventionConverter:
    """Convert a DataBundle between conventions per per-variable dispatch table.

    Variables not in the dispatch table pass through unchanged.

    Emits `convention_event` in bundle.meta.extras recording: (from, to, fields).
    """

    capabilities = OperatorCapabilities(
        accepts=tuple(LocationKind),
        produces=tuple(LocationKind),
        convention_in=None,   # any input convention
        convention_out=None,  # output convention is configured per instance
        preserves_variables=True,
    )

    def __init__(self, *, to: str, **_: Any) -> None:
        self._target = ConventionTag(to)

    def __call__(self, bundle: DataBundle) -> DataBundle:
        src = bundle.meta.convention
        if src == self._target:
            # No-op; still emit event for audit trail
            new_meta = dataclasses.replace(
                bundle.meta,
                extras={
                    **dict(bundle.meta.extras),
                    "convention_event": {
                        "from": src.value,
                        "to": self._target.value,
                        "fields": [],
                        "noop": True,
                    },
                },
            )
            return DataBundle(data=bundle.data, meta=new_meta)

        # OPAQUE → any: pass-through with audit warning
        if src == ConventionTag.OPAQUE:
            new_meta = dataclasses.replace(
                bundle.meta,
                convention=self._target,
                extras={
                    **dict(bundle.meta.extras),
                    "convention_event": {
                        "from": src.value,
                        "to": self._target.value,
                        "fields": [],
                        "opaque_assumed_identity": True,
                    },
                },
            )
            return DataBundle(data=bundle.data, meta=new_meta)

        key = (src, self._target)
        if key not in _DISPATCH:
            raise ValueError(
                f"ConventionConverter: no dispatch table for {src.value} → {self._target.value}"
            )
        table = _DISPATCH[key]

        new_ds = bundle.data.copy()
        converted_fields: list[str] = []
        for var in bundle.meta.variables:
            if var.name in table:
                action = table[var.name]
                old_arr = np.asarray(bundle.data[var.name].values)
                new_arr = action(old_arr).astype(old_arr.dtype)
                new_ds[var.name] = (bundle.data[var.name].dims, new_arr)
                converted_fields.append(var.name)

        new_meta = dataclasses.replace(
            bundle.meta,
            convention=self._target,
            extras={
                **dict(bundle.meta.extras),
                "convention_event": {
                    "from": src.value,
                    "to": self._target.value,
                    "fields": converted_fields,
                },
            },
        )
        return DataBundle(data=new_ds, meta=new_meta)
