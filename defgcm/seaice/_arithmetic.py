"""Division for coefficients that may be Python scalars or tensors."""
import torch
from torch import Tensor


def divide(numerator: float | Tensor, denominator: float | Tensor):
    # Keep scalar coefficients as Python floats, and preserve tensor dtype,
    # device and derivatives without Tensor.__rtruediv__'s reciprocal multiply.
    if isinstance(denominator, Tensor):
        return torch.div(numerator, denominator)
    return numerator/denominator
