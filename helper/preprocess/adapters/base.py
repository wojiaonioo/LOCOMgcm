from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class StandardGrid:
    """Unified grid descriptor produced by adapters."""

    nx: int
    ny: int
    nz: int
    lat: np.ndarray
    lon: np.ndarray
    depth: np.ndarray
    cell_area: np.ndarray | None
    hfac: np.ndarray | None

    def __eq__(self, other):
        if not isinstance(other, StandardGrid):
            return NotImplemented
        return self.nx == other.nx and self.ny == other.ny and self.nz == other.nz

    def __hash__(self):
        return hash((self.nx, self.ny, self.nz))


class ExperimentAdapter(ABC):
    """Converts one experiment type's raw output to standard xarray Datasets."""

    def __init__(self, experiment_dir: Path, metadata: dict):
        self.experiment_dir = Path(experiment_dir)
        self.metadata = metadata

    @classmethod
    @abstractmethod
    def can_handle(cls, metadata: dict) -> bool:
        """Return True if this adapter handles the given experiment type."""

    @abstractmethod
    def load_model_data(self) -> xr.Dataset:
        """Load model (LOCOMgcm NC) output as a standardized Dataset.

        Required coords: time (float64, seconds), depth (float64, m), lat, lon.
        Required vars: subset of {temp, salt, u, v, eta}. All float64.
        eta is 3D (time, lat, lon). Others are 4D (time, depth, lat, lon).
        """

    @abstractmethod
    def load_reference_data(self) -> xr.Dataset:
        """Load reference (MITgcm binary) output in the same standard format."""

    @abstractmethod
    def load_grid(self) -> StandardGrid:
        """Return the unified grid descriptor."""

    def grid_needs_interpolation(self) -> bool:
        return False

