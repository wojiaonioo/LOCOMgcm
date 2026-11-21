"""MITgcm forcing-record clock and the core surface-climatology source.

``periodic_interval`` is ``GET_PERIODIC_INTERVAL``
(``eesupp/src/get_periodic_interval.F:65-134``): the two records around a time
in a regularly spaced sequence and their linear weights.  The core periodic
forcing (``model/src/external_fields_load.F:69-72``) and the non-calendar EXF
fields (``pkg/exf/exf_getffieldrec.F:219-231``) both call it.  Record numbers
are one-based as in MITgcm; times and weights are float64 host scalars.

``SurfaceClimatologySource`` holds the ``thetaClimFile`` / ``saltClimFile``
records on the tracer grid, ``[record, ny, nx]`` in degC and psu, and samples
them at the step-start model time ``myTime = start_time + t``
(``forward_step.F:430``, ``load_fields_driver.F:234``).  A tracer without a
file has an empty record axis and samples MITgcm's initial zero
(``ini_ffields.F``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from locomgcm.sampling import SampleClock, SamplingRule


def _nint(value: float) -> int:
    """Fortran ``NINT``: round half away from zero."""
    return int(math.copysign(math.floor(abs(value) + 0.5), value))


def _f90modulo(value: float, period: float) -> float:
    """``F90MODULO(a,p) = MOD(MOD(a,p)+p,p)`` (get_periodic_interval.F:61)."""
    return math.fmod(math.fmod(value, period) + period, period)


def periodic_interval(current_time: float, *, cycle: float,
                      spacing: float) -> tuple[int, int, float, float]:
    """``(tRec1, tRec2, wght1, wght2)`` of GET_PERIODIC_INTERVAL, one-based.

    ``cycle=0`` is the non-periodic sequence (``:96-108``), otherwise the
    periodic one (``:115-128``).  Record ``n`` is centred at
    ``(n - 1/2)*spacing``.  The previous-step record ``tRec0`` only drives
    MITgcm's reload bookkeeping and is not returned.
    """
    if cycle < 0.0 or spacing <= 0.0:
        raise ValueError(
            f"GET_PERIODIC_INTERVAL requires cycle >= 0 and spacing > 0, got "
            f"cycle={cycle!r}, spacing={spacing!r} (get_periodic_interval.F:71-81)")
    nb_rec = _nint(cycle / spacing)
    if cycle != nb_rec * spacing:
        raise ValueError(
            f"cycle {cycle!r} is not a multiple of spacing {spacing!r} "
            "(get_periodic_interval.F:85-94)")
    if cycle == 0.0:
        loc_time = current_time - spacing * 0.5
        mod_time = _f90modulo(loc_time, spacing)
        rec1 = 1 + _nint((loc_time - mod_time) / spacing)
        rec2 = 1 + rec1
        wght2 = mod_time / spacing
    else:
        loc_time = current_time - spacing * 0.5 + cycle * (2 - _nint(current_time / cycle))
        tmp_time = math.fmod(loc_time, cycle)
        rec1 = 1 + int(tmp_time / spacing)
        rec2 = 1 + int(math.fmod(rec1, nb_rec))
        wght2 = (tmp_time - spacing * (rec1 - 1)) / spacing
    wght1 = 1.0 - wght2
    return rec1, rec2, wght1, wght2


@dataclass(frozen=True)
class SurfaceClimatologySource:
    """SST/SSS climatology of the core forcing, sampled at step start.

    ``periodic`` is ``periodicExternalForcing``: when false the first record
    holds for the whole run (``ini_forcing.F:100-105``); when true the records
    cycle with ``period``/``cycle`` (``externForcingPeriod``/
    ``externForcingCycle``) and ``SST = bWght*SST0 + aWght*SST1``
    (``external_fields_load.F:249-264``).  ``start_time`` is MITgcm's
    ``startTime``: the model time at LOCOMgcm time zero.
    """

    sampling_rule = SamplingRule(SampleClock.START)

    sst: torch.Tensor
    sss: torch.Tensor
    periodic: bool
    period: float
    cycle: float
    start_time: float

    def __post_init__(self) -> None:
        # ini_parms.F:1069-1085.
        if self.periodic and (
            self.cycle * self.period == 0.0
            or int(self.cycle / self.period) != self.cycle / self.period
            or self.cycle < self.period
        ):
            raise ValueError(
                "periodicExternalForcing needs externForcingCycle a whole multiple of "
                f"externForcingPeriod > 0, got period={self.period!r}, cycle={self.cycle!r}")

    @property
    def has_sst(self) -> bool:
        return self.sst.shape[0] > 0

    @property
    def has_sss(self) -> bool:
        return self.sss.shape[0] > 0

    def _sample(self, records: torch.Tensor, time_seconds: float) -> torch.Tensor:
        if records.shape[0] == 0:
            return records.new_zeros(records.shape[1:])
        if not self.periodic:
            return records[0]
        rec1, rec2, wght1, wght2 = periodic_interval(
            self.start_time + time_seconds, cycle=self.cycle, spacing=self.period)
        return wght1 * records[rec1 - 1] + wght2 * records[rec2 - 1]

    def sample(self, time_seconds: float) -> tuple[torch.Tensor, torch.Tensor]:
        """(SST, SSS) at LOCOMgcm elapsed time ``time_seconds``."""
        time_seconds = float(time_seconds)
        return self._sample(self.sst, time_seconds), self._sample(self.sss, time_seconds)

    def to_runtime(self, device: torch.device | str,
                   dtype: torch.dtype | None = None) -> "SurfaceClimatologySource":
        return SurfaceClimatologySource(
            sst=self.sst.to(device=device, dtype=dtype or self.sst.dtype),
            sss=self.sss.to(device=device, dtype=dtype or self.sss.dtype),
            periodic=self.periodic, period=self.period, cycle=self.cycle,
            start_time=self.start_time,
        )
