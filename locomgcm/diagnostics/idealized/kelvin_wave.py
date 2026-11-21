"""Kelvin-wave diagnostics (idealized wave-validation only).

Analytic-solution / fitting tools for the coastal/equatorial Kelvin wave
benchmark. NOT metric-aware; assumes a uniform grid only; NOT a member of the
production ``DiagnosticSuite``.
"""

from __future__ import annotations

import math

import torch

from ._fit import _loglinear_lstsq


def barotropic_phase_speed(*, gravity: float, depth: float) -> float:
    return float(math.sqrt(gravity * depth))


def barotropic_rossby_radius(
    *,
    gravity: float,
    depth: float,
    coriolis_parameter: float,
) -> float:
    if coriolis_parameter == 0.0:
        raise ValueError("coriolis_parameter must be nonzero for Rossby radius")
    return barotropic_phase_speed(gravity=gravity, depth=depth) / abs(coriolis_parameter)


def kelvin_wave_frequency(
    *,
    gravity: float,
    depth: float,
    wavelength: float,
) -> float:
    c = barotropic_phase_speed(gravity=gravity, depth=depth)
    k = 2.0 * math.pi / wavelength
    return float(c * k)


def fit_exponential_decay_scale(
    y_values: torch.Tensor,
    amplitude_values: torch.Tensor,
) -> dict[str, float]:
    """Fit A * exp(-y / Rd) to amplitude_values(y) via log-linear regression."""
    y = torch.as_tensor(y_values, dtype=torch.float64)
    amp = torch.as_tensor(amplitude_values, dtype=torch.float64)
    positive = amp > 0.0
    if positive.sum() < 2:
        raise ValueError("at least two positive amplitude values are required")
    y_pos = y[positive]
    log_amp = torch.log(amp[positive])
    a, b = _loglinear_lstsq(y_pos, log_amp)
    if b >= 0.0:
        return {"decay_scale": float("inf"), "amplitude": float(math.exp(a)), "slope": b}
    return {
        "decay_scale": -1.0 / b,
        "amplitude": float(math.exp(a)),
        "slope": b,
    }
