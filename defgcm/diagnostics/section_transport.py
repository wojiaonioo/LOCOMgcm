"""Zonal transport through a meridional U-face section.

The observation is ``1e-6 * sum(u * hfac_w * dyG * drF)`` in Sv.  It
corresponds to MITgcm ``UVELMASS*dyG*drF``, where ``UVELMASS`` is defined as
``uVel*hFacW`` in ``diagnostics_fill_state.F:550-551``.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import torch

__all__ = [
    "SECTION_TRANSPORT_SCHEMA_VERSION",
    "SECTION_TRANSPORT_WINDOW_SECONDS",
    "SectionTransportWindowAccumulator",
    "section_transport_window_steps",
    "section_transport_u",
    "write_section_transport_json",
]

SECTION_TRANSPORT_SCHEMA_VERSION = "1.0.0"
SECTION_TRANSPORT_WINDOW_SECONDS = 21600.0


def section_transport_window_steps(dt_s: float) -> int:
    """Resolve the fixed six-hour observation window for a model time step."""
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("section transport requires a finite positive dt")
    steps = round(SECTION_TRANSPORT_WINDOW_SECONDS / dt_s)
    if steps <= 0 or not math.isclose(
        steps * dt_s,
        SECTION_TRANSPORT_WINDOW_SECONDS,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError("six-hour section window must contain a whole number of steps")
    return steps


def section_transport_u(
    u: torch.Tensor,
    hfac_w: torch.Tensor,
    dyG: torch.Tensor,
    drF: torch.Tensor,
    i_face: int,
    j0: int,
    j1: int,
) -> torch.Tensor:
    """Return eastward-positive section transport per batch member in Sv."""
    if u.ndim != 4:
        raise ValueError("u must have shape [batch, nz, ny, nx+1]")
    _, nz, ny, nx_faces = u.shape
    expected_shapes = (
        (hfac_w, (nz, ny, nx_faces), "hfac_w"),
        (dyG, (ny, nx_faces), "dyG"),
        (drF, (nz,), "drF"),
    )
    for tensor, shape, name in expected_shapes:
        if tensor.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if tensor.dtype != u.dtype or tensor.device != u.device:
            raise ValueError(f"{name} must share u dtype and device")
    if not all(type(index) is int for index in (i_face, j0, j1)):
        raise ValueError("section indices must be integers")
    if not 0 <= i_face < nx_faces or not 0 <= j0 <= j1 < ny:
        raise ValueError("section indices are out of bounds")

    transport = u[:, :, j0 : j1 + 1, i_face] * hfac_w[:, j0 : j1 + 1, i_face]
    transport = transport * dyG[j0 : j1 + 1, i_face]
    transport = transport * drF[:, None]
    return transport.sum(dim=(1, 2)) * 1e-6


class SectionTransportWindowAccumulator:
    """Accumulate completed-state section transports into fixed windows.

    Resume does not reconstruct a partial window; windows with missing samples
    emit no record.
    """

    def __init__(
        self,
        i_face: int,
        j0: int,
        j1: int,
        *,
        window_steps: int = 180,
    ) -> None:
        if window_steps <= 0:
            raise ValueError("window_steps must be positive")
        self.i_face = i_face
        self.j0 = j0
        self.j1 = j1
        self.window_steps = window_steps
        self.window_sum_Sv: torch.Tensor | None = None
        self.window_count = 0
        self.window_means_Sv: list[torch.Tensor] = []
        self._window_end_steps: list[int] = []

    def observe(
        self,
        global_step: int,
        u: torch.Tensor,
        hfac_w: torch.Tensor,
        dyG: torch.Tensor,
        drF: torch.Tensor,
    ) -> None:
        if global_step == 0:
            return
        value = section_transport_u(
            u, hfac_w, dyG, drF, self.i_face, self.j0, self.j1
        )
        if self.window_sum_Sv is None:
            self.window_sum_Sv = torch.zeros_like(value)
        self.window_sum_Sv.add_(value)
        self.window_count += 1
        if global_step % self.window_steps == 0:
            if self.window_count == self.window_steps:
                self.window_means_Sv.append(
                    (self.window_sum_Sv / self.window_steps).detach()
                )
                self._window_end_steps.append(global_step)
            self.window_sum_Sv.zero_()
            self.window_count = 0

    def records(self, *, base_time_s: float, dt_s: float) -> list[dict[str, float]]:
        if not self.window_means_Sv:
            return []
        values = torch.stack(self.window_means_Sv).detach().cpu()
        if values.shape[1] != 1:
            raise ValueError("section transport JSON supports batch size one")
        host_values = values[:, 0].tolist()
        return [
            {
                "t_end_s": float(base_time_s + step * dt_s),
                "Q_Sv": float(value),
            }
            for step, value in zip(self._window_end_steps, host_values, strict=True)
        ]


def write_section_transport_json(
    path: Path,
    records: Sequence[Mapping[str, float]],
) -> None:
    """Atomically write section-transport records as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(records, stream, allow_nan=False, separators=(",", ":"))
            stream.write("\n")
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
