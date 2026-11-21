"""Bridge Layer B boundary bundles into sample-capable runtime sources."""

from __future__ import annotations

from locomgcm.sampling import SampleClock, SamplingRule

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from locomgcm.io.hycom_open_boundary import BoundaryInterpolationInfo, BoundaryTimeMode


_SIDE_ATTRS = {"W": "west", "E": "east", "S": "south", "N": "north"}


@dataclass(frozen=True)
class BoundaryBundleSample:
    """One clock of a bundle, with the same fields as ``HYCOMOpenBoundarySample``.

    Boundary readers (``TracerOpenBoundaryCondition``, the sponges) take
    ``temp``/``salt``/``u``/``v`` and the ``interpolation`` record from any
    source's sample; ``interpolation`` reports the records and weights this
    bundle actually used.
    """

    temp: Any
    salt: Any
    u: Any
    v: Any
    interpolation: BoundaryInterpolationInfo


@dataclass(frozen=True)
class BoundaryFieldSample:
    """Requested canonical field names and their common interpolation record."""

    fields: Mapping[str, Any]
    interpolation: BoundaryInterpolationInfo


@dataclass(frozen=True)
class BoundaryBundleSource:
    """Generic sample-capable bridge for Layer B OBC bundles.

    The source stores pre-fetched time series. Linear mode uses exact-time
    lookup or interpolation; constant mode holds the first record, matching
    EXF's zero-period OBC input. Fields have leading
    dimensions ``(time, side, ...)``. A sample never aliases the stored series:
    every clock returns tensors made in the caller's autograd mode.
    """

    sampling_rule = SamplingRule(SampleClock.END)

    times_seconds: tuple[float, ...]
    side_names: tuple[str, ...]
    fields: Mapping[str, torch.Tensor]
    valid_along: Mapping[str, int] | None = None
    forcing_period_seconds: float | None = None
    forcing_cycle_seconds: float | None = None

    #: Memoised ``torch.equal`` verdicts, keyed by (field, side, lo, hi) and
    #: stamped with the field tensor's content stamp.  See ``_records_equal``.
    #: Excluded from init/repr/compare so the dataclass contract is unchanged.
    _equal_cache: dict[
        tuple[str, str, int, int], tuple[tuple[int, int], bool]
    ] = field(default_factory=dict, init=False, repr=False, compare=False)

    def _time_bracket(self, model_time_seconds: float) -> tuple[int, int, float]:
        time_value = float(model_time_seconds)
        times = np.asarray(self.times_seconds, dtype=np.float64)
        if times.ndim != 1 or times.size == 0:
            raise ValueError("boundary bundle must contain at least one time value")

        def bracket(value: float) -> tuple[int, int, float] | None:
            for index, known in enumerate(times):
                if abs(float(known) - value) <= 1.0e-9:
                    return index, index, 0.0
            if value < float(times[0]) or value > float(times[-1]):
                return None
            hi = int(np.searchsorted(times, value, side="right"))
            lo = hi - 1
            if lo < 0 or hi >= times.size:
                return None
            span = float(times[hi] - times[lo])
            if abs(span) <= 1.0e-30:
                return lo, hi, 0.0
            return lo, hi, float((value - float(times[lo])) / span)

        raw = bracket(time_value)
        if raw is not None:
            return raw

        def cyclic_bracket(value: float) -> tuple[int, int, float] | None:
            if times.size < 2:
                return None
            if (
                self.forcing_period_seconds is None
                and self.forcing_cycle_seconds is None
            ):
                return None
            origin = float(times[0])
            frame_period = self.forcing_period_seconds
            if frame_period is None:
                frame_period = float(times[-1] - times[0]) / float(times.size - 1)
            frame_period = float(frame_period)
            cycle_period = self.forcing_cycle_seconds
            if cycle_period is None:
                cycle_period = frame_period * float(times.size)
            cycle_period = float(cycle_period)
            if frame_period <= 0.0 or cycle_period <= 0.0:
                return None
            phase = (float(value) - origin) % cycle_period
            slab = int(np.floor(phase / frame_period)) % int(times.size)
            next_slab = (slab + 1) % int(times.size)
            alpha = (phase - float(slab) * frame_period) / frame_period
            return slab, next_slab, float(alpha)

        cyclic = cyclic_bracket(time_value)
        if cyclic is not None:
            return cyclic

        # Some Layer B sources publish native absolute forcing times while the
        # runtime calls sources with relative model seconds during boundary
        # object construction. Accept that relative clock by shifting to the
        # first cached frame; exact absolute requests still took the raw branch.
        if abs(float(times[0])) > 1.0e-9:
            shifted = bracket(time_value + float(times[0]))
            if shifted is not None:
                return shifted
            shifted_cyclic = cyclic_bracket(time_value + float(times[0]))
            if shifted_cyclic is not None:
                return shifted_cyclic

        raise ValueError(
            f"OBC sample time {time_value} is outside the prefetched Layer B "
            "boundary bundle"
        )

    def _time_index(self, model_time_seconds: float) -> int:
        lo, hi, alpha = self._time_bracket(model_time_seconds)
        if lo == hi or alpha == 0.0:
            return lo
        if alpha == 1.0:
            return hi
        time_value = float(model_time_seconds)
        for index, known in enumerate(self.times_seconds):
            if abs(known - time_value) <= 1.0e-9:
                return index
        raise ValueError(
            f"OBC sample time {time_value} requires interpolation, not a single index"
        )

    def _slice_along(self, values: torch.Tensor, side_name: str) -> torch.Tensor:
        if self.valid_along is None:
            return values
        side_key = _SIDE_ATTRS.get(side_name, side_name)
        n_valid = self.valid_along.get(side_name, self.valid_along.get(side_key))
        if n_valid is None:
            return values
        return values[..., : int(n_valid)].contiguous()

    def _side_set(self, values: torch.Tensor, index: int) -> Any:
        from locomgcm.boundaries.open import SideBoundaryTensorSet

        kwargs: dict[str, torch.Tensor] = {}
        for side_index, side_name in enumerate(self.side_names):
            attr = _SIDE_ATTRS.get(side_name)
            if attr is not None:
                kwargs[attr] = self._slice_along(values[index, side_index], side_name)
        return SideBoundaryTensorSet(**kwargs)

    @staticmethod
    def _content_stamp(values: torch.Tensor) -> tuple[int, int]:
        """Identity-and-mutation stamp for one field's whole time series.

        ``frozen=True`` freezes attribute rebinding, not content: ``fields`` is a
        Mapping whose entries can be replaced, and the tensors it holds can be
        written in place.  Either makes a cached verdict stale, and a stale
        "equal" returns the lower record verbatim where the caller asked for an
        interpolation.

        ``_version`` is torch's version counter -- shared between a tensor and
        every view of it, and bumped by every in-place write -- and ``data_ptr``
        separates a replaced entry from the one that was measured.  Both are
        O(1) host reads with no device synchronisation, which is the property
        the memo exists to buy; a per-sample content comparison would reinstate
        the 240 syncs/step it removes.

        Not covered: a write through a tensor that aliases the same storage but
        was built outside this one's version counter (``as_strided`` on the
        storage, ``from_blob``).  Catching that needs the content compare this
        stamp replaces.
        """
        return (values.data_ptr(), values._version)

    def _records_equal(
        self,
        cache_key: str,
        side_name: str,
        lo: int,
        hi: int,
        stamp: tuple[int, int],
        lower: torch.Tensor,
        upper: torch.Tensor,
    ) -> bool:
        """``torch.equal(lower, upper)``, memoised per record pair.

        ``torch.equal`` on CUDA tensors is a blocking device->host
        synchronisation.  This test runs once per (field, side) on every sample,
        which measured 240 syncs/step -- 65% of the whole step's total, and the
        same 240 at N=160 as at N=960, so it is structural rather than a
        function of problem size.

        The verdict depends on the bundle contents and the two integer record
        indices ``_time_bracket`` picked on the host, never on ``alpha`` or on
        model state, so it is computed once per (pair, content) and reused.  The
        branch selected is identical either way, hence so is every bit
        downstream.
        """
        key = (cache_key, side_name, lo, hi)
        cached = self._equal_cache.get(key)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        verdict = bool(torch.equal(lower, upper))
        self._equal_cache[key] = (stamp, verdict)
        return verdict

    def _interpolated_side_set(
        self,
        values: torch.Tensor,
        lo: int,
        hi: int,
        alpha: float,
        cache_key: str,
    ) -> Any:
        from locomgcm.boundaries.open import SideBoundaryTensorSet

        # One stamp per call: every side reads the same field tensor, so an
        # in-place write to any record invalidates all of that field's verdicts.
        stamp = self._content_stamp(values)
        kwargs: dict[str, torch.Tensor] = {}
        for side_index, side_name in enumerate(self.side_names):
            attr = _SIDE_ATTRS.get(side_name)
            if attr is None:
                continue
            lower = self._slice_along(values[lo, side_index], side_name)
            upper = self._slice_along(values[hi, side_index], side_name)
            if lo == hi or alpha == 0.0:
                record = lower
            elif alpha == 1.0:
                record = upper
            elif self._records_equal(
                cache_key, side_name, lo, hi, stamp, lower, upper
            ):
                record = lower
            else:
                record = None
            # Every sample is a new tensor made in the caller's autograd mode.
            # This keeps the dispatch-key set consistent for record-aligned
            # and interpolated samples under inference mode. Copying preserves
            # the record bits, including signed zeros.
            kwargs[attr] = (
                (1.0 - alpha) * lower + alpha * upper
                if record is None
                else record.clone()
            )
        return SideBoundaryTensorSet(**kwargs)

    def sample_field(self, name: str, model_time_seconds: float) -> Any:
        """Return one side tensor field at ``model_time_seconds``."""

        if name not in self.fields:
            raise KeyError(f"boundary bundle does not contain field {name!r}")
        lo, hi, alpha = self._time_bracket(model_time_seconds)
        return self._interpolated_side_set(self.fields[name], lo, hi, alpha, name)

    @property
    def supports_field_projection(self) -> bool:
        """Whether consumers may replace full sampling with this projection.

        An existing subclass can override only ``sample`` to apply controls or
        other transformations. Its inherited projection must not bypass that
        behavior. Overriding ``sample_fields`` explicitly declares a matching
        projection protocol; otherwise consumers keep the full-sample path.
        """

        full_impl = getattr(self.sample, "__func__", None)
        projection_impl = getattr(self.sample_fields, "__func__", None)
        return (
            full_impl is BoundaryBundleSource.sample
            or projection_impl is not BoundaryBundleSource.sample_fields
        )

    def sample_fields(
        self,
        model_time_seconds: float,
        *,
        fields: tuple[str, ...],
        mode: str | BoundaryTimeMode = BoundaryTimeMode.LINEAR,
        **kwargs: Any,
    ) -> BoundaryFieldSample:
        """Sample only the named fields at one common boundary clock.

        Names are the canonical keys in ``self.fields`` (for example,
        ``temperature`` and ``salinity``). Each requested field keeps the full
        sample's interpolation and fresh-tensor contract. No sampled tensors
        are shared with another consumer or retained for a later clock.
        Consumers must also check ``supports_field_projection`` before using
        this method in place of a potentially overridden full ``sample``.
        """

        # The prefetched bundle's clock is already resolved by its producer;
        # update_interval_seconds does not resample that time axis.
        del kwargs
        actual_mode = BoundaryTimeMode(mode)
        if actual_mode == BoundaryTimeMode.CONSTANT:
            if not self.times_seconds:
                raise ValueError("boundary bundle must contain at least one time value")
            lo, hi, alpha = 0, 0, 0.0
        else:
            lo, hi, alpha = self._time_bracket(model_time_seconds)
        lower_time = float(self.times_seconds[lo])
        upper_time = float(self.times_seconds[hi])
        interpolation = BoundaryInterpolationInfo(
            mode=actual_mode.value,
            model_time_seconds=float(model_time_seconds),
            sample_time_seconds=float(model_time_seconds),
            update_interval_seconds=None,
            lower_index=lo,
            upper_index=hi,
            lower_time_seconds=lower_time,
            upper_time_seconds=upper_time,
            lower_time_value=f"{lower_time}s",
            upper_time_value=f"{upper_time}s",
            lower_weight=1.0 - alpha,
            upper_weight=alpha,
        )
        return BoundaryFieldSample(
            fields={
                name: self._interpolated_side_set(
                    self.fields[name], lo, hi, alpha, name
                )
                for name in fields
            },
            interpolation=interpolation,
        )

    def sample(
        self,
        model_time_seconds: float,
        *,
        mode: str | BoundaryTimeMode = BoundaryTimeMode.LINEAR,
        **kwargs: Any,
    ) -> BoundaryBundleSample:
        """Sample the declared mode at the requested boundary clock."""

        # Preserve this API's ignored-keyword contract, including names used
        # by the projection API. Full samples always request all four fields.
        del kwargs
        sample = BoundaryBundleSource.sample_fields(
            self,
            model_time_seconds,
            fields=("temperature", "salinity", "u", "v"),
            mode=mode,
        )
        return BoundaryBundleSample(
            temp=sample.fields["temperature"],
            salt=sample.fields["salinity"],
            u=sample.fields["u"],
            v=sample.fields["v"],
            interpolation=sample.interpolation,
        )

    def to_runtime(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> "BoundaryBundleSource":
        kwargs: dict[str, object] = {"device": device}
        if dtype is not None:
            kwargs["dtype"] = dtype
        return BoundaryBundleSource(
            times_seconds=self.times_seconds,
            side_names=self.side_names,
            fields={name: tensor.to(**kwargs) for name, tensor in self.fields.items()},
            valid_along=self.valid_along,
            forcing_period_seconds=self.forcing_period_seconds,
            forcing_cycle_seconds=self.forcing_cycle_seconds,
        )

    def to(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> "BoundaryBundleSource":
        return self.to_runtime(device, dtype)
