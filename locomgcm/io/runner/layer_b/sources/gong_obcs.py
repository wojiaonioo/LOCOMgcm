"""Gong et al. 2023 (ISWFM-NSCS v2.0) native tidal-OBCS source.

Reads Gong's per-side barotropic tidal harmonics directly from the published
MITgcm input binaries::

    OB_{East,West,North,South}_{u,v}_{Amp,Gph}.bin

and emits the SAME canonical TIDAL_HARMONICS boundary bundle that the
``extract_tidal_obcs`` operator produces — dims ``(constituent, side, along)``,
amplitude in m/s, phase in SECONDS, ``side`` order N, S, E, W, with an
``along_valid`` coord per side. Because Gong's files are already per-boundary
amplitude/phase (MITgcm's native OBCS form), this source emits the
POST-operator schema: the tidal lane wires it source → consumer with NO
operator (``operators: []``), exactly like the d4 ``mit_ic`` lane.

The downstream consumers then give BOTH models Gong's identical tides:
  - ``prepared_tidal_zarr``  → LOCOMgcm runtime tidal injection
  - ``mitgcm_tidal_binary``  → MIT per-constituent ``tides_raw`` bins, later
                               stacked + V0-corrected by ``mitgcm_obcs_namelist``

UNITS / CONVENTIONS (verified against the input package):
  * Endianness: Gong compiled MITgcm with ``-convert big_endian`` → all OB bins
    are BIG-endian float64. Read with ``>f8``.
  * Layout: each bin is ``(nConstituents=8, along_len)`` C-order — constituent
    is the OUTER axis (verified: reshape(8, len) recovers the physical
    hierarchy M2 > K1 > O1 > S2 > P1 > N2 > K2 > Q1; reshape(len, 8) does not).
  * Side lengths: East/West run along y (== ny = 960), North/South along x
    (== nx = 1696). Matches the canonical ``along_valid = [nx, nx, ny, ny]``.
  * along ordering: OB_North/South are indexed by MITgcm i (west→east) and
    OB_East/West by j (south→north) — identical to extract_tidal_obcs's
    ``np.linspace`` boundary lines, so a direct copy lands in the right slot.
  * ``_Amp`` is the tidal VELOCITY amplitude (m/s) — Gong's OBCS reads these
    directly as velocities — so it maps to ``u/v_tide_amplitude`` unchanged.
  * ``_Gph`` is the Greenwich phase expressed as a TIME LAG in SECONDS (per
    constituent the values track that constituent's own period, e.g. M2 ≈
    44.7k s ≈ its 44714 s period; degrees/radians are ruled out by magnitude).
    This is exactly the ``phase_seconds`` the prepared_tidal schema and LOCOMgcm's
    ``boundaries/tidal.py`` (``A·cos(ω·(t−phase_s))``) expect, so ``_Gph`` is
    copied verbatim into ``u/v_tide_phase``.

V0 NOTE: Gong's ``_Gph`` ALREADY includes the V0 astronomical argument — Gong's
``obcs_add_tides.F`` uses it as-is in ``cos(w*(t-phase))`` with NO V0 correction
(``code/obcs_add_tides.F:76``).  This source emits ``_Gph`` verbatim, and the
case sets ``boundary.tidal_phase_includes_v0: true`` so that BOTH the LOCOMgcm runtime
injection and the MIT export SKIP their ``V0(start_date)`` correction.  Adding it
would double-count (+14564 s on M2 ≈ 33% of the period vs Gong).  Skipping on
both sides stays alignment-neutral (LOCOMgcm==MIT) AND now matches Gong's forcing.
(The earlier assumption that ``_Gph`` was raw Greenwich with no V0 was wrong.)

CONSTITUENT ORDER is intrinsic to the binary layout — Gong's data.obcs
``tidalPeriod`` line fixes it as M2, S2, N2, K2, K1, O1, P1, Q1. Periods are
the authoritative Gong values; they must match the yaml ``boundary.
tidal_constituents`` so LOCOMgcm and MIT share one ω per constituent.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from locomgcm.external_paths import resolve_input_path
from typing import Any

import numpy as np
import xarray as xr

from locomgcm.io.runner.layer_a import (
    BundleMeta,
    ConventionTag,
    DataBundle,
    LocationKind,
    ProvenanceStep,
    VariableSpec,
)

from .._registry import register_source
from .._types import (
    DataQuery,
    SourceCapabilities,
    SourceContext,
    TimeCoverage,
)

# Gong data.obcs `tidalPeriod` (seconds), in the fixed OB-bin constituent order.
# name → period; insertion order == the bins' outer-axis order.
GONG_CONSTITUENTS: dict[str, float] = {
    "m2": 44714.16,
    "s2": 43200.0,
    "n2": 45570.05,
    "k2": 43082.05,
    "k1": 86164.2,
    "o1": 92949.48,
    "p1": 86637.20,
    "q1": 96726.08,
}
_N_CONST = len(GONG_CONSTITUENTS)

# Canonical schema side order (matches extract_tidal_obcs._SIDES) and the Gong
# file-name token for each side. N/S run along x (nx), E/W along y (ny).
_SIDES = ("N", "S", "E", "W")
_SIDE_FILE = {"N": "North", "S": "South", "E": "East", "W": "West"}

# bundle var name → Gong (component, kind) file tokens.
_VAR_TO_GONG = {
    "u_tide_amplitude": ("u", "Amp"),
    "u_tide_phase": ("u", "Gph"),
    "v_tide_amplitude": ("v", "Amp"),
    "v_tide_phase": ("v", "Gph"),
}


def _read_ob_bin(path: Path, along_len: int) -> np.ndarray:
    """Read one Gong OB bin → (nConstituents, along_len) native float64.

    Big-endian f8, constituent-outer layout. Validates the element count so a
    grid/file mismatch fails loudly instead of silently mis-reshaping.
    """
    flat = np.fromfile(str(path), dtype=">f8")
    expected = _N_CONST * along_len
    if flat.size != expected:
        raise ValueError(
            f"gong_tidal_obcs_native: {path.name} has {flat.size} f8 values, "
            f"expected {_N_CONST}*{along_len}={expected} "
            f"(constituents × side length)"
        )
    return np.ascontiguousarray(flat.reshape(_N_CONST, along_len).astype("float64"))


@register_source("gong_tidal_obcs_native")
class GongTidalObcsNativeSource:
    """Gong native per-side OB tidal bins → canonical TIDAL_HARMONICS bundle."""

    capabilities = SourceCapabilities(
        produces=(LocationKind.TIDAL_HARMONICS,),
        variables=(
            "u_tide_amplitude", "u_tide_phase",
            "v_tide_amplitude", "v_tide_phase",
        ),
        # Per-boundary harmonics are a static field — no time axis. INSTANT
        # keeps the pipeline query time_range out of the way (reusable across
        # simulation start dates; V0 is applied downstream).
        time_coverage_strategy=TimeCoverage.INSTANT,
        # Tidal amplitude is a magnitude and phase a time offset — neither
        # carries the LOCOMgcm↔MIT sign flip, so the lane needs no convention_convert
        # (mirrors the tpxo9 lane, whose consumers require CF_STANDARD).
        convention=ConventionTag.CF_STANDARD,
        requires_target_grid=True,
    )

    def __init__(
        self,
        ctx: SourceContext,
        *,
        path: Path | str,
        **_unused: Any,
    ) -> None:
        self._ctx = ctx
        self._path = resolve_input_path(path)
        if not self._path.is_dir():
            raise ValueError(
                f"gong_tidal_obcs_native: path must be Gong's input directory "
                f"(containing OB_*_{{Amp,Gph}}.bin), got {path}"
            )
        # Resolve + existence-check all 16 bins up front; expose as `_files` so
        # the pipeline CAS fingerprint keys on each file's (path, size, mtime).
        self._bin_paths: dict[tuple[str, str, str], Path] = {}
        missing: list[str] = []
        for side in _SIDES:
            stoken = _SIDE_FILE[side]
            for comp in ("u", "v"):
                for kind in ("Amp", "Gph"):
                    fp = self._path / f"OB_{stoken}_{comp}_{kind}.bin"
                    self._bin_paths[(side, comp, kind)] = fp
                    if not fp.exists():
                        missing.append(fp.name)
        if missing:
            raise FileNotFoundError(
                f"gong_tidal_obcs_native: missing OB bins under {self._path}: "
                f"{', '.join(missing)}"
            )
        self._files = list(self._bin_paths.values())

    def fetch(self, query: DataQuery) -> DataBundle:
        g = query.target_grid
        if g is None or not hasattr(g, "center_lat"):
            raise ValueError(
                "gong_tidal_obcs_native: requires query.target_grid with grid params"
            )
        nx, ny = int(g.nx), int(g.ny)
        # Side true lengths — N/S along x (nx), E/W along y (ny). Identical to
        # extract_tidal_obcs.side_valid; `along` is zero-padded to max(nx, ny).
        side_valid = {"N": nx, "S": nx, "E": ny, "W": ny}
        n_along = max(nx, ny)

        names = tuple(GONG_CONSTITUENTS.keys())
        periods = np.asarray(list(GONG_CONSTITUENTS.values()), dtype="float64")

        out = {
            v: np.zeros((_N_CONST, len(_SIDES), n_along), dtype="float64")
            for v in _VAR_TO_GONG
        }
        for s_idx, side in enumerate(_SIDES):
            vlen = side_valid[side]
            for var, (comp, kind) in _VAR_TO_GONG.items():
                arr = _read_ob_bin(self._bin_paths[(side, comp, kind)], vlen)
                # arr is (const, vlen); land cells in dry phase points carry
                # fill but their amplitude is 0, so they inject nothing.
                out[var][:, s_idx, :vlen] = arr

        dims = ("constituent", "side", "along")
        ds = xr.Dataset(
            data_vars={v: (dims, out[v]) for v in _VAR_TO_GONG},
            coords={
                "constituent": np.array(names),
                "period": ("constituent", periods),
                "side": np.array(list(_SIDES)),
                "along": np.arange(n_along),
                "along_valid": ("side", np.array([side_valid[s] for s in _SIDES])),
            },
        )
        var_specs = [
            VariableSpec("u_tide_amplitude", "sea_water_x_velocity_amplitude_due_to_tides",
                         "m s-1", "float64", dims),
            VariableSpec("u_tide_phase", "sea_water_x_velocity_phase_due_to_tides",
                         "s", "float64", dims),
            VariableSpec("v_tide_amplitude", "sea_water_y_velocity_amplitude_due_to_tides",
                         "m s-1", "float64", dims),
            VariableSpec("v_tide_phase", "sea_water_y_velocity_phase_due_to_tides",
                         "s", "float64", dims),
        ]
        return self._wrap(ds, var_specs, getattr(g, "grid_id", None))

    def _wrap(
        self, ds: xr.Dataset, var_specs: list[VariableSpec], grid_id: Any
    ) -> DataBundle:
        meta_no_prov = BundleMeta(
            schema_version="0.1.0",
            location_kind=LocationKind.TIDAL_HARMONICS,
            convention=ConventionTag.CF_STANDARD,
            variables=tuple(var_specs),
            time_range=None,
            target_grid_id=grid_id,
            provenance=(),
        )
        prov = ProvenanceStep(
            kind="source",
            name="GongTidalObcsNativeSource",
            params={
                "path": str(self._path),
                "constituents": tuple(GONG_CONSTITUENTS.keys()),
                "n_files": len(self._files),
            },
            timestamp=datetime.utcnow(),
            bundle_hash="00000000",
        )
        meta = BundleMeta(
            schema_version=meta_no_prov.schema_version,
            location_kind=meta_no_prov.location_kind,
            convention=meta_no_prov.convention,
            variables=meta_no_prov.variables,
            time_range=meta_no_prov.time_range,
            target_grid_id=meta_no_prov.target_grid_id,
            provenance=(prov,),
            extras=meta_no_prov.extras,
        )
        return DataBundle(data=ds, meta=meta)
