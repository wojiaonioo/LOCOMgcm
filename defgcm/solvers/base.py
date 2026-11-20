from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import torch


class LinearOperator(Protocol):
    def __call__(self, vector: torch.Tensor) -> torch.Tensor:
        ...


class LinearSolver(ABC):
    name: str

    @abstractmethod
    def solve(self, operator: LinearOperator, rhs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
