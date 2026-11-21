"""EXF forcing fields: record clock, field start time and the bulk-input source.

``exf_record_interval`` is ``EXF_GetFFieldRec`` (``pkg/exf/exf_getffieldrec.F``):
record ``n`` of a field lies at ``fldStartTime + (n-1)*fldPeriod``; ``fac`` is
the weight of ``count0``.  Without the calendar it calls the shared
``GET_PERIODIC_INTERVAL`` clock (:func:`locomgcm.io.forcing_records.periodic_interval`)
at ``myTime - fldStartTime + fldPeriod/2`` (``:219-231``); with it, it counts
seconds from ``fldStartTime`` (``:112-149``).  ``fldPeriod=0`` reads record 1
once (``exf_init_fld.F:98-145``).  ``exf_interpolate`` is ``exf_set_fld.F:305-310``.

``FieldSeries`` holds one field's records ``[record, ny, nx]`` (float64, EXF
units and signs: K, kg/kg, m/s, W/m2 downward, N/m2) with its namelist clock.
``ExfBulkSource`` samples the bulk-formula inputs, ``ExfFluxSource`` and
``ExfStressSource`` the heat/freshwater fluxes and cell-centre stress read
from files without ALLOW_ATM_TEMP, ``ExfClimatologySource`` the climsst/climsss
restoring targets of ALLOW_CLIMSST/SSS_RELAXATION, all at the step-start model time
``myTime = start_time + t`` (``load_fields_driver.F:214``).  Times are float64
host scalars; record numbers are one-based as in MITgcm.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from datetime import datetime, timedelta

import torch

from locomgcm.io.era5 import ERA5BulkInputs
from locomgcm.io.forcing_records import periodic_interval
from locomgcm.sampling import SampleClock, SamplingRule

# pkg/cal calendars this port converts start dates with.
EXF_CALENDARS = ("gregorian",)
# EXF_OPTIONS.h macros whose branches are not ported; a build must leave them undefined.
EXF_OPTIONS_UNDEFINED = ("ALLOW_BULK_LARGEYEAGER04", "ALLOW_ZENITHANGLE", "ALLOW_RUNOFTEMP",
                         "USE_EXF_INTERPOLATION")


def _cal_date(date1: int, date2: int) -> datetime:
    """``YYYYMMDD``, ``HHMMSS`` integers as a Gregorian datetime (cal_FullDate)."""
    return datetime(date1 // 10000, date1 // 100 % 100, date1 % 100,
                    date2 // 10000, date2 // 100 % 100, date2 % 100)


def exf_field_start_time(startdate1: int, startdate2: int, *, calendar: str,
                         base_date1: int, base_date2: int, start_time: float) -> float:
    """``fld_start_time`` from a field's start date (exf_getffield_start.F:99-103).

    ``startTime + (field date - model start date)`` in seconds, where the model
    start date is ``startDate_1/2`` plus ``startTime`` (cal_set.F:205-210).
    """
    if calendar not in EXF_CALENDARS:
        raise ValueError(f"calendar {calendar!r} is not one of {EXF_CALENDARS}")
    model_start = _cal_date(base_date1, base_date2) + timedelta(seconds=start_time)
    return start_time + (_cal_date(startdate1, startdate2) - model_start).total_seconds()


def exf_record_interval(time: float, *, start_time: float, period: float,
                        repeat_cycle: float, use_cal: bool) -> tuple[int, int, float]:
    """``(count0, count1, fac)`` of EXF_GetFFieldRec for one field at ``myTime``."""
    if period == 0.0:
        return 1, 1, 1.0
    if period < 0.0:
        raise ValueError(f"fldPeriod={period!r}: monthly calendar records are not ported")
    if use_cal:
        fldsectot = time - start_time
        if repeat_cycle == 0.0:
            if fldsectot < 0.0:
                raise ValueError(f"myTime={time!r} precedes the first record at {start_time!r}")
            count0 = int((fldsectot + 0.5) / period) + 1
            count1 = count0 + 1
            fldsecs = math.fmod(fldsectot, period)
        else:
            if fldsectot < 0.0:
                fldsectot = fldsectot + repeat_cycle
            fldsecs0 = math.fmod(fldsectot, repeat_cycle)
            count0 = int((fldsecs0 + 0.5) / period) + 1
            fldsecs1 = math.fmod(fldsectot + period, repeat_cycle)
            count1 = int((fldsecs1 + 0.5) / period) + 1
            fldsecs = math.fmod(fldsecs0, period)
        return count0, count1, 1.0 - fldsecs / period
    count0, count1, fac, _ = periodic_interval(
        time - start_time + period * 0.5, cycle=repeat_cycle, spacing=period)
    if count0 <= 0:
        raise ValueError(f"myTime={time!r} precedes the first record at {start_time!r} "
                         "(exf_getffieldrec.F:233-260)")
    return count0, count1, fac


def exf_interpolate(fld0: torch.Tensor, fld1: torch.Tensor, fac: float, *,
                    in_scale: float, remove_intercept: float, remove_slope: float,
                    elapsed: float) -> torch.Tensor:
    """``fldArr`` of exf_set_fld.F:305-310; ``elapsed`` is ``myTime - startTime``."""
    value = in_scale * (fac * fld0 + (1.0 - fac) * fld1)
    return value - in_scale * (remove_intercept + remove_slope * elapsed)


@dataclass(frozen=True)
class FieldSeries:
    """One EXF field: records ``[record, ny, nx]``, or an empty record axis for a constant field.

    ``start_time`` is the resolved ``fldStartTime``; ``center_mask`` is a
    ``'c'`` ``fldMask`` (exf_init_fixed.F:52-115), applied by the consumer
    that owns the mask (exf_filter_rl.F:57-65).
    """
    records: torch.Tensor
    has_file: bool
    constant: float | torch.Tensor
    period: float
    start_time: float
    repeat_cycle: float
    input_scale: float | torch.Tensor
    remove_intercept: float | torch.Tensor
    remove_slope: float | torch.Tensor
    center_mask: bool


def exf_field_value(series: FieldSeries, my_time: float, *, use_cal: bool,
                    start_time: float) -> torch.Tensor:
    """One field at ``myTime``: ``fldConst`` without a file (exf_init_fld.F:86-96),
    ``inScale*record 1`` for period 0, else the interpolated records.  Dry
    cells keep their file values; see the consumers' masks.
    """
    if not series.has_file:
        return torch.full(series.records.shape[1:], float(series.constant),
                          dtype=series.records.dtype, device=series.records.device)
    if series.period == 0.0:
        return series.input_scale * series.records[0]
    count0, count1, fac = exf_record_interval(
        my_time, start_time=series.start_time, period=series.period,
        repeat_cycle=series.repeat_cycle, use_cal=use_cal)
    return exf_interpolate(
        series.records[count0 - 1], series.records[count1 - 1], fac,
        in_scale=series.input_scale, remove_intercept=series.remove_intercept,
        remove_slope=series.remove_slope, elapsed=my_time - start_time)


def _series_to(series: FieldSeries, device: torch.device | str,
               dtype: torch.dtype | None) -> FieldSeries:
    return replace(series, records=series.records.to(device=device, dtype=dtype or series.records.dtype))


@dataclass(frozen=True)
class ExfBulkSource:
    """EXF bulk-formula inputs (ALLOW_ATM_TEMP, ALLOW_ATM_WIND, ALLOW_DOWNWARD_RADIATION).

    Dry cells keep their file values: the bulk formulae and the tracer
    forcing are per cell, and the wind consumer zeroes dry centre stress by
    ``maskC`` selection before its face average (``era5_wind_forcing``), so
    the ``'c'`` dry-cell filter changes no wet-cell result on this path.
    ``atemp_offset`` is ``exf_offset_atemp`` (exf_getffields.F:199-206); with
    a snowprecip file ``precip = MAX(precip, snowPrecip)`` (:314-324).
    ``runoff`` [m/s, into the ocean] leaves the bulk freshwater flux
    (ALLOW_RUNOFF, exf_getforcing.F:313), from its file or ``runoffconst``.
    ``lwflux``/``swflux`` [W/m2, upward] are the net radiation fields (file or
    ``lwfluxconst``/``swfluxconst``, exf_init_varia.F:229-290,
    exf_getffields.F:252-343); EXF replaces each by the one formed from
    lwdown/swdown only where the down file is given and the flux file is not
    (exf_radiation.F:60, :107), as ``longwave_from_down``/``shortwave_from_down``.
    """

    sampling_rule = SamplingRule(SampleClock.START)
    # Records are chosen with host arithmetic (exf_record_interval); the
    # consumer samples eagerly when it updates its model time.
    samples_on_host = True

    atemp: FieldSeries
    aqh: FieldSeries
    uwind: FieldSeries
    vwind: FieldSeries
    swdown: FieldSeries
    lwdown: FieldSeries
    precip: FieldSeries
    snowprecip: FieldSeries
    runoff: FieldSeries
    lwflux: FieldSeries
    swflux: FieldSeries
    atemp_offset: float
    use_cal: bool
    start_time: float

    @property
    def longwave_from_down(self) -> bool:
        return self.lwdown.has_file and not self.lwflux.has_file

    @property
    def shortwave_from_down(self) -> bool:
        return self.swdown.has_file and not self.swflux.has_file

    def sample(self, model_time_seconds: float | torch.Tensor) -> ERA5BulkInputs:
        my_time = self.start_time + float(model_time_seconds)
        value = {name: exf_field_value(getattr(self, name), my_time, use_cal=self.use_cal,
                                       start_time=self.start_time)
                 for name in ("atemp", "aqh", "uwind", "vwind", "swdown", "lwdown", "precip",
                              "runoff", "lwflux", "swflux")}
        value["atemp"] = value["atemp"] + self.atemp_offset
        if self.snowprecip.has_file:
            value["precip"] = torch.maximum(value["precip"], exf_field_value(
                self.snowprecip, my_time, use_cal=self.use_cal, start_time=self.start_time))
        return ERA5BulkInputs(**value)

    def to_runtime(self, device: torch.device | str,
                   dtype: torch.dtype | None = None) -> "ExfBulkSource":
        return ExfBulkSource(**{
            item.name: (_series_to(getattr(self, item.name), device, dtype)
                        if isinstance(getattr(self, item.name), FieldSeries)
                        else getattr(self, item.name))
            for item in fields(self)})


@dataclass(frozen=True)
class ExfFluxSource:
    """Heat and freshwater fluxes read from files, without ALLOW_ATM_TEMP.

    ``Qnet = exf_outscal_hflux*hflux`` and ``EmPmR = exf_outscal_sflux*sflux
    *rhoConstFresh`` (exf_mapfields.F:99-129).  A field without a file then
    loses ``exf_outscal*(intercept + slope*(myTime-startTime))`` of its
    ``_exfremo`` trend (:105-111, :122-129); a file field had it removed when
    read (exf_set_fld.F:308-310).  ``sample`` returns the
    ``wpflux_surface_forcing`` pair: the heat flux in LOCOMgcm convention,
    positive into the ocean (``heat_flux = -Qnet``, docs/NUMERICS.md),
    and the freshwater volume flux [m/s] whose ``rhoConstFresh`` product the
    consumer forms, so the freshwater trend is removed before that product
    (MIT multiplies each term by rhoConstFresh first; the orders differ by
    rounding only).  The third value is ``Qsw = exf_outscal_swflux*swflux``
    [W/m2, MIT upward-positive] (exf_mapfields.F:289-296, with ALLOW_ATM_TEMP
    or SHORTWAVE_HEATING; ``exf_init_varia.F:278`` fills ``swfluxconst``
    without a file).
    """

    sampling_rule = SamplingRule(SampleClock.START)

    hflux: FieldSeries
    sflux: FieldSeries
    swflux: FieldSeries
    hflux_scale: float
    sflux_scale: float
    swflux_scale: float
    use_cal: bool
    start_time: float

    def sample(self, model_time_seconds: float | torch.Tensor
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        my_time = self.start_time + float(model_time_seconds)
        elapsed = my_time - self.start_time
        hflux = exf_field_value(self.hflux, my_time, use_cal=self.use_cal, start_time=self.start_time)
        sflux = exf_field_value(self.sflux, my_time, use_cal=self.use_cal, start_time=self.start_time)
        qnet = self.hflux_scale * hflux
        if not self.hflux.has_file:
            qnet = qnet - self.hflux_scale * (
                self.hflux.remove_intercept + self.hflux.remove_slope * elapsed)
        freshwater = self.sflux_scale * sflux
        if not self.sflux.has_file:
            freshwater = freshwater - self.sflux_scale * (
                self.sflux.remove_intercept + self.sflux.remove_slope * elapsed)
        swflux = exf_field_value(self.swflux, my_time, use_cal=self.use_cal, start_time=self.start_time)
        return -qnet, freshwater, self.swflux_scale * swflux

    def to_runtime(self, device: torch.device | str,
                   dtype: torch.dtype | None = None) -> "ExfFluxSource":
        return replace(self, hflux=_series_to(self.hflux, device, dtype),
                       sflux=_series_to(self.sflux, device, dtype),
                       swflux=_series_to(self.swflux, device, dtype))


@dataclass(frozen=True)
class ExfStressSource:
    """Cell-centre ``ustress``/``vstress`` files [N/m2], east and north positive.

    For ``era5_wind_forcing`` with ``stress_location: center``, which zeroes dry
    centres by ``maskC``, clips to ``windstressmax`` and averages onto the
    masked faces as exf_mapfields.F:216-284 (``exf_outscal_ustress`` and
    ``exf_outscal_vstress`` are 1 on this path).
    """

    sampling_rule = SamplingRule(SampleClock.START)

    ustress: FieldSeries
    vstress: FieldSeries
    use_cal: bool
    start_time: float

    def sample(self, model_time_seconds: float | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        my_time = self.start_time + float(model_time_seconds)
        return (exf_field_value(self.ustress, my_time, use_cal=self.use_cal, start_time=self.start_time),
                exf_field_value(self.vstress, my_time, use_cal=self.use_cal, start_time=self.start_time))

    def to_runtime(self, device: torch.device | str,
                   dtype: torch.dtype | None = None) -> "ExfStressSource":
        return replace(self, ustress=_series_to(self.ustress, device, dtype),
                       vstress=_series_to(self.vstress, device, dtype))


@dataclass(frozen=True)
class ExfClimatologySource:
    """SST/SSS targets of EXF's surface restoring (ALLOW_CLIMSST/SSS_RELAXATION).

    ``climsst``/``climsss`` [degC, psu] follow the EXF record clock of every
    EXF field (exf_getclim.F:72-82, :101-111), ``climsstconst``/``climsssconst``
    without a file.  climsst below ``climtempfreeze`` is raised to it
    (:84-94), then ``SST = exf_outscal_sst*climsst`` and ``SSS =
    exf_outscal_sss*climsss`` (exf_mapfields.F:298-310), sampled at step start
    (exf_getforcing.F:196).  The core FORCING_SURF_RELAX forms the restoring
    flux from them (``surface_climate_relaxation``).  Dry cells keep their file
    values instead of the ``'c'`` filter's zero; the flux multiplies by hFacC.
    """

    sampling_rule = SamplingRule(SampleClock.START)

    climsst: FieldSeries
    climsss: FieldSeries
    climtempfreeze: float
    sst_scale: float
    sss_scale: float
    use_cal: bool
    start_time: float

    @property
    def has_sst(self) -> bool:
        return self.climsst.has_file

    @property
    def has_sss(self) -> bool:
        return self.climsss.has_file

    def sample(self, time_seconds: float | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(SST, SSS) at LOCOMgcm elapsed time ``time_seconds``."""
        my_time = self.start_time + float(time_seconds)
        climsst = exf_field_value(self.climsst, my_time, use_cal=self.use_cal,
                                  start_time=self.start_time)
        climsst = torch.where(climsst < self.climtempfreeze,
                              torch.full_like(climsst, self.climtempfreeze), climsst)
        climsss = exf_field_value(self.climsss, my_time, use_cal=self.use_cal,
                                  start_time=self.start_time)
        return self.sst_scale * climsst, self.sss_scale * climsss

    def to_runtime(self, device: torch.device | str,
                   dtype: torch.dtype | None = None) -> "ExfClimatologySource":
        return replace(self, climsst=_series_to(self.climsst, device, dtype),
                       climsss=_series_to(self.climsss, device, dtype))
