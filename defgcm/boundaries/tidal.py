"""OBCS Tidal Forcing — barotropic tidal velocity overlay for open boundaries.

Implements MITgcm's OBCS_ADD_TIDES mechanism: harmonic reconstruction of
barotropic tidal velocities from precomputed amplitude/phase coefficients,
added to open-boundary velocity arrays AFTER OBCS_BALANCE_FLOW.

MITgcm reference: pkg/obcs/obcs_add_tides.F:53-149
                   pkg/obcs/obcs_init_fixed.F:536-582
                   pkg/obcs/obcs_adjust.F:42-51 (ordering)
"""
from __future__ import annotations

from defgcm.sampling import SampleClock, SamplingRule

import math
from dataclasses import dataclass, field

import torch

from defgcm.geometry import StructuredCGrid

from .open import SideBoundaryTensorSet


@dataclass(frozen=True)
class TidalComponent:
    """Precomputed Cs/Sn coefficients for one tidal constituent.

    After initialization from amplitude (m/s) and phase (seconds):
      Cs = A * cos(omega * phase_seconds)
      Sn = A * sin(omega * phase_seconds)
    where omega = 2*pi / period_seconds.

    At runtime: tide(t) = Cs*cos(omega*t) + Sn*sin(omega*t) = A*cos(omega*(t - phase))
    """
    period_seconds: float
    u_tide_cs: SideBoundaryTensorSet
    u_tide_sn: SideBoundaryTensorSet
    v_tide_cs: SideBoundaryTensorSet
    v_tide_sn: SideBoundaryTensorSet


@dataclass
class OBCSTidalConfig:
    """Configuration for OBCS barotropic tidal velocity injection.

    Equivalent to MITgcm: useOBCStides + OBCS_tidalPeriod + 16 amplitude/phase files.
    Each TidalComponent holds precomputed Cs/Sn tensors for all four boundaries.
    """

    sampling_rule = SamplingRule(SampleClock.END)
    components: list[TidalComponent] = field(default_factory=list)

    @staticmethod
    def from_amplitude_phase(
        periods: list[float],
        u_amplitude: dict[str, torch.Tensor | None] | None = None,
        u_phase: dict[str, torch.Tensor | None] | None = None,
        v_amplitude: dict[str, torch.Tensor | None] | None = None,
        v_phase: dict[str, torch.Tensor | None] | None = None,
    ) -> "OBCSTidalConfig":
        """Create tidal config from amplitude (m/s) and phase (seconds) arrays.

        Args:
            periods: List of constituent periods in seconds (e.g. M2=44712.0).
                     Zero periods are skipped.
            u_amplitude: Per-side u-velocity amplitude tensors.
                         Keys: 'north','south','east','west'. Shape: [n_comp, n_boundary_pts]
                         or [n_boundary_pts] if single component.
            u_phase: Per-side u-velocity phase tensors (units: seconds).
            v_amplitude: Per-side v-velocity amplitude tensors.
            v_phase: Per-side v-velocity phase tensors (units: seconds).

        The conversion formula (MITgcm obcs_init_fixed.F:540-546):
            omega = 2*pi / T
            Cs = A * cos(omega * phase_seconds)
            Sn = A * sin(omega * phase_seconds)

        Note: Python computes Cs and Sn from IMMUTABLE amplitude/phase,
        avoiding the Fortran in-place overwrite order dependency.
        """
        u_amplitude = u_amplitude or {}
        u_phase = u_phase or {}
        v_amplitude = v_amplitude or {}
        v_phase = v_phase or {}

        components: list[TidalComponent] = []
        for c, period in enumerate(periods):
            if period == 0.0:
                continue
            omega = 2.0 * math.pi / period

            u_cs_dict: dict[str, torch.Tensor | None] = {}
            u_sn_dict: dict[str, torch.Tensor | None] = {}
            v_cs_dict: dict[str, torch.Tensor | None] = {}
            v_sn_dict: dict[str, torch.Tensor | None] = {}

            for side in ('north', 'south', 'east', 'west'):
                u_cs_dict[side], u_sn_dict[side] = _convert_one(
                    u_amplitude.get(side), u_phase.get(side), c, omega, len(periods)
                )
                v_cs_dict[side], v_sn_dict[side] = _convert_one(
                    v_amplitude.get(side), v_phase.get(side), c, omega, len(periods)
                )

            components.append(TidalComponent(
                period_seconds=period,
                u_tide_cs=SideBoundaryTensorSet(**u_cs_dict),
                u_tide_sn=SideBoundaryTensorSet(**u_sn_dict),
                v_tide_cs=SideBoundaryTensorSet(**v_cs_dict),
                v_tide_sn=SideBoundaryTensorSet(**v_sn_dict),
            ))
        return OBCSTidalConfig(components=components)

    def to(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> "OBCSTidalConfig":
        """Move all coefficient tensors to the specified device/dtype."""
        moved = []
        for comp in self.components:
            moved.append(TidalComponent(
                period_seconds=comp.period_seconds,
                u_tide_cs=comp.u_tide_cs.to(device, dtype),
                u_tide_sn=comp.u_tide_sn.to(device, dtype),
                v_tide_cs=comp.v_tide_cs.to(device, dtype),
                v_tide_sn=comp.v_tide_sn.to(device, dtype),
            ))
        return OBCSTidalConfig(components=moved)


def add_tidal_velocities(
    balanced_u: SideBoundaryTensorSet,
    balanced_v: SideBoundaryTensorSet,
    grid: StructuredCGrid,
    time_seconds: float,
    tidal_config: OBCSTidalConfig,
) -> tuple[SideBoundaryTensorSet, SideBoundaryTensorSet]:
    """Add barotropic tidal velocity to already-balanced OB arrays.

    MITgcm equivalent: OBCS_ADD_TIDES (obcs_add_tides.F:53-149)
    Called AFTER OBCS_BALANCE_FLOW so tidal transport is intentionally unbalanced.

    The tidal velocity is barotropic (depth-independent). When OB arrays are
    3D [nz, n_pts], the 1D [n_pts] tidal velocity broadcasts across all layers.
    The tidal increment is masked at the same OB-array face used by MITgcm
    before either face-clamp or sponge consumers read the corrected targets.

    Args:
        balanced_u: Already-balanced u boundary values (SideBoundaryTensorSet).
        balanced_v: Already-balanced v boundary values (SideBoundaryTensorSet).
        grid: Structured grid providing MITgcm maskW/maskS-equivalent hFac masks.
        time_seconds: Model time at which to evaluate tides (usually futureTime).
        tidal_config: Precomputed tidal coefficients.

    Returns:
        (new_u, new_v) with tidal increments added. Does not mutate inputs.
    """
    if not tidal_config.components:
        return balanced_u, balanced_v

    u_deltas: dict[str, torch.Tensor | None] = {
        'west': None, 'east': None, 'south': None, 'north': None
    }
    v_deltas: dict[str, torch.Tensor | None] = {
        'west': None, 'east': None, 'south': None, 'north': None
    }

    for comp in tidal_config.components:
        if comp.period_seconds == 0.0:
            continue
        omega_t = time_seconds * 2.0 * math.pi / comp.period_seconds
        time_cos = math.cos(omega_t)
        time_sin = math.sin(omega_t)

        for side in ('west', 'east', 'south', 'north'):
            u_cs = getattr(comp.u_tide_cs, side)
            u_sn = getattr(comp.u_tide_sn, side)
            if u_cs is not None and u_sn is not None:
                u_tide = u_cs * time_cos + u_sn * time_sin
                u_deltas[side] = u_tide if u_deltas[side] is None else u_deltas[side] + u_tide

            v_cs = getattr(comp.v_tide_cs, side)
            v_sn = getattr(comp.v_tide_sn, side)
            if v_cs is not None and v_sn is not None:
                v_tide = v_cs * time_cos + v_sn * time_sin
                v_deltas[side] = v_tide if v_deltas[side] is None else v_deltas[side] + v_tide

    def _face_mask(side: str, component: str, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if component == 'u':
            mask_w = (grid.hfac_w_for(dtype, device) != 0.0).to(dtype=dtype)
            if side == 'west':
                # obcs_add_tides.F:88-96, OBWu uses maskW(1+iB).
                return mask_w[:, :, 1]
            if side == 'east':
                # obcs_add_tides.F:68-76, OBEu uses maskW(iB); -2 = east OB
                # normal face (nx-1), NOT outermost -1 (=nx, dry).
                return mask_w[:, :, -2]
            if side == 'south':
                # obcs_add_tides.F:128-136, OBSu uses maskW(i,jB).
                return mask_w[:, 0, :grid.nx]
            # obcs_add_tides.F:108-116, OBNu uses maskW(i,jB).
            return mask_w[:, -1, :grid.nx]

        mask_s = (grid.hfac_s_for(dtype, device) != 0.0).to(dtype=dtype)
        if side == 'west':
            # obcs_add_tides.F:88-96, OBWv uses maskS(iB).
            return mask_s[:, :grid.ny, 0]
        if side == 'east':
            # obcs_add_tides.F:68-76, OBEv uses maskS(iB).
            return mask_s[:, :grid.ny, -1]
        if side == 'south':
            # obcs_add_tides.F:128-136, OBSv uses maskS(1+jB).
            return mask_s[:, 1, :]
        # obcs_add_tides.F:108-116, OBNv uses maskS(jB); -2 = north OB
        # normal face (ny-1), NOT outermost -1 (=ny, dry).
        return mask_s[:, -2, :]

    def _masked_delta(delta: torch.Tensor | None, side: str, component: str) -> torch.Tensor | None:
        if delta is None:
            return None
        mask = _face_mask(side, component, delta.dtype, delta.device)
        return delta * mask

    def _add(
        base: torch.Tensor | None,
        delta: torch.Tensor | None,
        side: str,
        component: str,
    ) -> torch.Tensor | None:
        if base is None and delta is None:
            return None
        delta = _masked_delta(delta, side, component)
        if base is None:
            return delta
        if delta is None:
            return base
        # Barotropic broadcast: delta is [n_pts], base may be [nz, n_pts]
        return base + delta

    new_u = SideBoundaryTensorSet(
        west=_add(balanced_u.west, u_deltas['west'], 'west', 'u'),
        east=_add(balanced_u.east, u_deltas['east'], 'east', 'u'),
        south=_add(balanced_u.south, u_deltas['south'], 'south', 'u'),
        north=_add(balanced_u.north, u_deltas['north'], 'north', 'u'),
    )
    new_v = SideBoundaryTensorSet(
        west=_add(balanced_v.west, v_deltas['west'], 'west', 'v'),
        east=_add(balanced_v.east, v_deltas['east'], 'east', 'v'),
        south=_add(balanced_v.south, v_deltas['south'], 'south', 'v'),
        north=_add(balanced_v.north, v_deltas['north'], 'north', 'v'),
    )
    return new_u, new_v


def _compute_v0_correction(name: str, period: float, start_date: str) -> float:
    """Astronomical-argument (V0) phase correction in seconds for a constituent.

    TPXO9 phase files are referenced to a fixed Greenwich epoch: the runtime
    reconstruction is ``A·cos(ω·(t − phase_seconds))`` with ``phase_seconds`` the
    Greenwich phase LAG. The physically-correct equilibrium tide at model time
    ``t`` is ``A·cos(ω·t + V0 − G)``, where ``V0`` is the constituent's
    equilibrium (astronomical) argument at the model start. That equals
    ``A·cos(ω·(t − (phase_seconds − V0/ω)))``, so the correction ADDED to
    ``phase_seconds`` is ``−V0/ω`` — this function returns exactly that (the
    equilibrium argument as a time, negated), so callers keep doing
    ``phase_corrected = phase_seconds + v0_correction``.

    MITgcm has no built-in V0 correction; the MIT export bakes in the SAME value
    (layer_d/consumers/mit_binary.py), so LOCOMgcm↔MIT stay bit-aligned for any start
    date. The Gong path already includes V0 and skips this (tidal_phase_includes_v0).

    Conventions — verified against pyTMD's equilibrium argument (nodal off, two
    independent reductions; ):
      • T is mean SOLAR time τ+s−h (NOT GMST sidereal time).
      • JD is the start-date 00:00 = JDN−0.5 (the integer formula yields the
        Julian Day Number, i.e. JD at 12:00 UT noon).
      • V0 is SUBTRACTED from the phase lag (the return is negated).
    """
    from datetime import datetime

    import numpy as _np
    import pyTMD.astro as _astro

    dt = datetime.strptime(start_date[:10], "%Y-%m-%d")
    # Fliegel–Van Flandern yields the Julian Day NUMBER (JD at 12:00 UT noon);
    # −0.5 shifts to the start-date 00:00, the instant the model calls t=0.
    a = (14 - dt.month) // 12
    y = dt.year + 4800 - a
    m = dt.month + 12 * a - 3
    JD = dt.day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045 - 0.5
    MJD = JD - 2400000.5

    s, h, p, n, pp = _astro.mean_longitudes(MJD)
    s, h, p = float(s), float(h), float(p)
    # Doodson's T is mean SOLAR time = τ + s − h (τ = mean lunar time, radians
    # from doodson_arguments). The old code used GMST (sidereal) here, shifting
    # the whole tide by ~−4.5 h — a near-constant offset invisible to op-level
    # and LOCOMgcm↔MIT parity tests because both models shared the identical bug.
    tau = math.degrees(float(_np.atleast_1d(_astro.doodson_arguments(_np.atleast_1d(MJD))[0])[0]))
    T = (tau + s - h) % 360

    # Equilibrium (Doodson) arguments in degrees.
    _V0 = {
        "m2": (2 * T - 2 * s + 2 * h) % 360,
        "s2": (2 * T) % 360,
        "n2": (2 * T - 3 * s + 2 * h + p) % 360,
        "k2": (2 * T + 2 * h) % 360,
        "k1": (T + h + 90) % 360,
        "o1": (T - 2 * s + h - 90) % 360,
        "p1": (T - h - 90) % 360,  # solar diurnal: −90 (old +90 was 180° wrong)
        "q1": (T - 3 * s + h + p - 90) % 360,
    }
    v0_deg = _V0.get(name.lower(), 0.0)
    # The equilibrium argument advances the tide → subtract it from the phase lag.
    return -(v0_deg / 360.0 * period)


def tidal_phase_correction_seconds(
    constituent: dict,
    start_date: str | None,
) -> float:
    """Resolve a constituent's phase origin for runtime and MIT export.

    Undeclared phases retain the historical Greenwich convention, regardless
    of whether values come from arrays or files. Model-time phase lags must
    declare ``phase_reference="model"``; canonical datasets use the same
    Greenwich default when converted to arrays at ingestion.
    """
    reference = constituent.get("phase_reference", "greenwich")
    if reference not in ("model", "greenwich"):
        raise ValueError(f"unsupported tidal phase_reference: {reference!r}")
    name = constituent.get("name", "")
    if reference == "model" or not start_date or not name:
        return 0.0
    try:
        return _compute_v0_correction(name, float(constituent["period"]), start_date)
    except Exception:
        # Retain the legacy Greenwich-file fallback when astronomy is unavailable.
        return 0.0


def build_tidal_config_from_constituents(
    constituents: list[dict],
    data_dir: str | None = None,
    start_date: str | None = None,
) -> OBCSTidalConfig | None:
    """Build OBCSTidalConfig from a list of constituent dicts (YAML-friendly).

    Each constituent dict:
        name: str (e.g. "M2"), used for Greenwich astronomical arguments
        period: float (seconds)
        files: dict with keys like "north_u_amplitude", "north_u_phase", etc.
               Values are file paths (relative to data_dir) to binary float64 arrays.
        arrays: dict — alternative to files, direct tensor values
        phase_reference: "model" or "greenwich"; defaults to "greenwich"
                         for both arrays and files

    Args:
        start_date: Model start date (e.g. "2019-01-01"). Applies the V0
            astronomical correction only to Greenwich-referenced phases.
            Model-time phase lags are used directly in A*cos(omega*(t-phase)).

    Returns None if constituents list is empty.
    """
    if not constituents:
        return None

    import numpy as np
    from pathlib import Path

    periods: list[float] = []
    phase_corrections: list[float] = []
    u_amp: dict[str, list[torch.Tensor | None]] = {s: [] for s in ('north', 'south', 'east', 'west')}
    u_ph: dict[str, list[torch.Tensor | None]] = {s: [] for s in ('north', 'south', 'east', 'west')}
    v_amp: dict[str, list[torch.Tensor | None]] = {s: [] for s in ('north', 'south', 'east', 'west')}
    v_ph: dict[str, list[torch.Tensor | None]] = {s: [] for s in ('north', 'south', 'east', 'west')}

    for comp in constituents:
        period = float(comp['period'])
        periods.append(period)

        v0_sec = tidal_phase_correction_seconds(comp, start_date)
        phase_corrections.append(v0_sec)

        files = comp.get('files', {})
        arrays = comp.get('arrays', {})
        base = Path(data_dir) if data_dir else Path('.')

        for side in ('north', 'south', 'east', 'west'):
            for var, store in [('u_amplitude', u_amp), ('u_phase', u_ph),
                               ('v_amplitude', v_amp), ('v_phase', v_ph)]:
                key = f"{side}_{var}"
                if key in arrays:
                    val = arrays[key]
                    t = val if isinstance(val, torch.Tensor) else torch.tensor(val, dtype=torch.float64)
                    if 'phase' in var and v0_sec != 0.0:
                        t = t + v0_sec
                    store[side].append(t)
                elif key in files and files[key]:
                    fpath = base / files[key]
                    arr = np.fromfile(str(fpath), dtype=np.float64)
                    t = torch.from_numpy(arr)
                    if 'phase' in var and v0_sec != 0.0:
                        t = t + v0_sec
                    store[side].append(t)
                else:
                    store[side].append(None)

    if any(phase_corrections):
        print(f"[tidal] V0 correction applied for {start_date}:")
        for comp, v0 in zip(constituents, phase_corrections):
            name = comp.get('name', '?')
            period = float(comp['period'])
            if v0 != 0.0:
                print(f"  {name}: V0={v0/period*360:.1f}°, correction={v0:.0f}s")

    def _stack_or_none(tensors: list[torch.Tensor | None]) -> torch.Tensor | None:
        if all(t is None for t in tensors):
            return None
        filled = [t if t is not None else torch.zeros_like(next(x for x in tensors if x is not None)) for t in tensors]
        return torch.stack(filled, dim=0)

    return OBCSTidalConfig.from_amplitude_phase(
        periods=periods,
        u_amplitude={s: _stack_or_none(u_amp[s]) for s in ('north', 'south', 'east', 'west')},
        u_phase={s: _stack_or_none(u_ph[s]) for s in ('north', 'south', 'east', 'west')},
        v_amplitude={s: _stack_or_none(v_amp[s]) for s in ('north', 'south', 'east', 'west')},
        v_phase={s: _stack_or_none(v_ph[s]) for s in ('north', 'south', 'east', 'west')},
    )


def _convert_one(
    amplitude: torch.Tensor | None,
    phase: torch.Tensor | None,
    comp_index: int,
    omega: float,
    n_total: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Convert amplitude+phase for one component/side to Cs/Sn coefficients.

    Handles multi-component stacked arrays [n_comp, n_pts] and single [n_pts].
    """
    if amplitude is None:
        return None, None

    if amplitude.dim() >= 2 and amplitude.shape[0] == n_total:
        amp = amplitude[comp_index]
    elif amplitude.dim() == 1:
        amp = amplitude
    else:
        raise ValueError(
            f"amplitude shape {tuple(amplitude.shape)} incompatible with "
            f"n_total={n_total}, comp_index={comp_index}"
        )

    if phase is not None:
        if phase.dim() >= 2 and phase.shape[0] == n_total:
            ph = phase[comp_index]
        elif phase.dim() == 1:
            ph = phase
        else:
            raise ValueError(
                f"phase shape {tuple(phase.shape)} incompatible with "
                f"n_total={n_total}, comp_index={comp_index}"
            )
    else:
        ph = torch.zeros_like(amp)

    loc_ph = ph * omega
    cs = amp * torch.cos(loc_ph)
    sn = amp * torch.sin(loc_ph)
    return cs, sn
