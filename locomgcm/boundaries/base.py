from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import torch

from locomgcm.geometry import StructuredCGrid
from locomgcm.state import PrognosticState, StateTendency


class BoundaryCondition(ABC):
    name: str
    accepts_tidal_config: ClassVar[bool] = False

    @abstractmethod
    def enforce(self, state: PrognosticState, grid: StructuredCGrid) -> PrognosticState:
        raise NotImplementedError

    def enforce_tendency(self, tendency: StateTendency, grid: StructuredCGrid) -> StateTendency:
        """Base default: no-op pass-through (returns ``tendency`` unchanged).

        Subclasses that need to zero boundary tendencies must override this.
        """
        del grid
        return tendency

    def to_runtime(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> "BoundaryCondition":
        del device, dtype
        return self
