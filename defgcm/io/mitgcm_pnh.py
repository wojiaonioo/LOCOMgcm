from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import warnings

import numpy as np
import torch


def parse_meta(meta_path: Path) -> dict:
    """Parse a MITgcm .meta companion file."""
    text = meta_path.read_text()

    def _field(name: str) -> str:
        match = re.search(rf"{name}\s*=\s*\[(.*?)\]\s*;", text, flags=re.S)
        if match is None:
            raise ValueError(f"{meta_path} is missing {name}")
        return match.group(1)

    dim_values = [int(value) for value in re.findall(r"[-+]?\d+", _field("dimList"))]
    if len(dim_values) != 9:
        raise ValueError(f"{meta_path} dimList expected 9 integers, found {len(dim_values)}")
    nx, ny, nz = dim_values[0], dim_values[3], dim_values[6]

    dataprec_match = re.search(r"'([^']+)'", _field("dataprec"))
    if dataprec_match is None:
        raise ValueError(f"{meta_path} dataprec is missing quoted precision")
    dataprec = dataprec_match.group(1)
    if dataprec not in {"float32", "float64"}:
        raise ValueError(f"{meta_path} unsupported dataprec {dataprec!r}")

    def _int_field(name: str) -> int:
        values = re.findall(r"[-+]?\d+", _field(name))
        if not values:
            raise ValueError(f"{meta_path} {name} has no integer value")
        return int(values[0])

    return {
        "dimList": (nx, ny, nz),
        "dataprec": dataprec,
        "nrecords": _int_field("nrecords"),
        "timeStepNumber": _int_field("timeStepNumber"),
    }


def load_mitgcm_pnh_dump(meta_path: Path) -> tuple[np.ndarray, dict]:
    """Load PNH .data + .meta. Returns (array [nz, ny, nx] float64, meta)."""
    meta = parse_meta(meta_path)
    data_path = meta_path.with_suffix(".data")
    nx, ny, nz = meta["dimList"]
    nrecords = meta["nrecords"]
    if nrecords != 1:
        raise ValueError(f"{meta_path} expected nrecords=1 for PNH, found {nrecords}")
    dtype = ">f4" if meta["dataprec"] == "float32" else ">f8"
    raw = np.fromfile(data_path, dtype=dtype)
    expected = nx * ny * nz * nrecords
    if raw.size != expected:
        raise ValueError(f"{data_path} expected {expected} values, found {raw.size}")
    arr = raw.reshape(nz, ny, nx).astype(np.float64)
    return arr, meta


@dataclass(frozen=True)
class MITgcmPNHSource:
    """Time-interpolated source for MITgcm phi_nh dumps.

    Note: although declared ``frozen=True``, the ``_index`` / ``_cache`` /
    ``_warned`` fields are mutable and populated lazily on first use. This is
    therefore not a true immutable value object — it is not thread-safe and must
    not be shared as if it were an immutable constant.
    """

    run_dir: Path
    dt: float
    base_time: float = 0.0
    extrapolation: str = "clamp"
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    dtype: torch.dtype = torch.float64
    _index: list = field(default_factory=list)
    _cache: dict = field(default_factory=dict)
    _warned: set = field(default_factory=set)

    @classmethod
    def from_run_dir(
        cls,
        run_dir: Path,
        dt: float,
        base_time: float = 0.0,
        extrapolation: str = "clamp",
        device=None,
        dtype=torch.float64,
    ) -> "MITgcmPNHSource":
        """Scan run_dir for PNH.<step>.001.001.meta and build a source."""
        if extrapolation not in {"clamp", "raise"}:
            raise ValueError("extrapolation must be 'clamp' or 'raise'")
        pattern = re.compile(r"PNH\.(\d{10})\.001\.001\.meta")
        index = []
        for meta_path in sorted(run_dir.glob("PNH.*.001.001.meta")):
            match = pattern.fullmatch(meta_path.name)
            if match is not None:
                index.append((int(match.group(1)), meta_path))
        if not index:
            raise FileNotFoundError(f"no PNH.*.001.001.meta dumps found in {run_dir}")
        index.sort()
        src = cls(
            run_dir=run_dir,
            dt=dt,
            base_time=base_time,
            extrapolation=extrapolation,
            device=torch.device("cpu") if device is None else torch.device(device),
            dtype=dtype,
        )
        object.__setattr__(src, "_index", index)
        return src

    @property
    def step_indices(self) -> np.ndarray:
        return np.array([step for step, _ in self._index], dtype=np.int64)

    def _load_step(self, step: int) -> np.ndarray:
        """Load or fetch a step from a small LRU cache."""
        if step in self._cache:
            arr = self._cache.pop(step)
            self._cache[step] = arr
            return arr
        meta_by_step = {indexed_step: path for indexed_step, path in self._index}
        try:
            meta_path = meta_by_step[step]
        except KeyError as exc:
            raise KeyError(f"PNH step {step} not indexed in {self.run_dir}") from exc
        arr, _ = load_mitgcm_pnh_dump(meta_path)
        if len(self._cache) >= 2:
            self._cache.pop(next(iter(self._cache)))
        self._cache[step] = arr
        return arr

    def _as_tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(arr)).to(dtype=self.dtype, device=self.device)

    def sample(self, time_seconds: float) -> torch.Tensor:
        """Linear-interp phi_nh at absolute time. Returns [nz, ny, nx] tensor."""
        step_real = (float(time_seconds) - self.base_time) / self.dt
        steps = self.step_indices

        if step_real < float(steps[0]):
            if self.extrapolation == "raise":
                raise ValueError(
                    f"requested PNH time {time_seconds} before first dump step {steps[0]}"
                )
            if "low" not in self._warned:
                warnings.warn(
                    f"PNH request at step {step_real:.6g} before first dump; clamping to {steps[0]}",
                    UserWarning,
                    stacklevel=2,
                )
                self._warned.add("low")
            return self._as_tensor(self._load_step(int(steps[0])))

        if step_real > float(steps[-1]):
            if self.extrapolation == "raise":
                raise ValueError(
                    f"requested PNH time {time_seconds} after last dump step {steps[-1]}"
                )
            if "high" not in self._warned:
                warnings.warn(
                    f"PNH request at step {step_real:.6g} after last dump; clamping to {steps[-1]}",
                    UserWarning,
                    stacklevel=2,
                )
                self._warned.add("high")
            return self._as_tensor(self._load_step(int(steps[-1])))

        pos = int(np.searchsorted(steps, step_real, side="left"))
        if pos < len(steps) and np.isclose(step_real, steps[pos], rtol=0.0, atol=1.0e-12):
            return self._as_tensor(self._load_step(int(steps[pos])))

        low_step = int(steps[pos - 1])
        high_step = int(steps[pos])
        arr_lo = self._load_step(low_step)
        arr_hi = self._load_step(high_step)
        frac = (step_real - low_step) / (high_step - low_step)
        arr = (1.0 - frac) * arr_lo + frac * arr_hi
        return self._as_tensor(arr)
