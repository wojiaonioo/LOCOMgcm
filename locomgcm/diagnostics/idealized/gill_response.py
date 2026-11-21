"""Gill equatorial response diagnostics (idealized wave-validation only).

Analytic-solution / fitting tools for the Gill equatorial response benchmark.
NOT metric-aware; assumes a uniform grid only; NOT a member of the production
``DiagnosticSuite``.
"""

from __future__ import annotations

import math

import torch

from ._fit import _loglinear_lstsq


def equatorial_deformation_radius(
    *,
    phase_speed: float,
    beta: float,
) -> float:
    if beta <= 0.0:
        raise ValueError("beta must be positive for equatorial deformation radius")
    if phase_speed <= 0.0:
        raise ValueError("phase_speed must be positive for equatorial deformation radius")
    return float(math.sqrt(phase_speed / beta))


def wrapped_longitudinal_offset(
    x_values: torch.Tensor,
    *,
    x_center: float,
    period: float,
) -> torch.Tensor:
    if period <= 0.0:
        raise ValueError("period must be positive")
    x = torch.as_tensor(x_values, dtype=torch.float64)
    half_period = 0.5 * float(period)
    return torch.remainder(x - float(x_center) + half_period, float(period)) - half_period


def fit_gaussian_waveguide_scale(
    y_values: torch.Tensor,
    amplitude_values: torch.Tensor,
    *,
    y_equator: float | None = None,
    min_relative_amplitude: float = 0.05,
    max_abs_offset: float | None = None,
) -> dict[str, float]:
    if not (0.0 < min_relative_amplitude < 1.0):
        raise ValueError("min_relative_amplitude must be in (0, 1)")
    if max_abs_offset is not None and max_abs_offset <= 0.0:
        raise ValueError("max_abs_offset must be positive when provided")

    y = torch.as_tensor(y_values, dtype=torch.float64)
    amplitude = torch.as_tensor(amplitude_values, dtype=torch.float64).abs()
    if y.numel() != amplitude.numel():
        raise ValueError("y_values and amplitude_values must have the same length")
    if y.numel() < 3:
        raise ValueError("at least three points are required")

    peak = float(amplitude.max().item())
    if peak <= 0.0:
        raise ValueError("amplitude_values must contain at least one nonzero value")

    y_eq = float(0.5 * (y.min().item() + y.max().item())) if y_equator is None else float(y_equator)
    fit_mask = (
        torch.ones_like(amplitude, dtype=torch.bool)
        if max_abs_offset is None
        else (y - y_eq).abs() <= float(max_abs_offset)
    )
    if int(fit_mask.sum().item()) < 3:
        raise ValueError("max_abs_offset leaves fewer than three samples for fitting")

    mask = fit_mask & (amplitude >= min_relative_amplitude * peak)
    if int(mask.sum().item()) < 3:
        mask = fit_mask & (amplitude > 0.0)
    if int(mask.sum().item()) < 3:
        raise ValueError("not enough positive points for Gaussian scale fit")

    y_offset_squared = (y[mask] - y_eq).square()
    log_amp = torch.log(torch.clamp(amplitude[mask], min=peak * 1.0e-12))
    intercept, slope = _loglinear_lstsq(y_offset_squared, log_amp)
    if slope >= 0.0:
        scale = float("inf")
    else:
        scale = float(math.sqrt(-0.5 / slope))
    return {
        "scale": scale,
        "slope": slope,
        "amplitude": float(math.exp(intercept)),
        "y_equator": y_eq,
    }


def estimate_e_folding_scale(
    y_values: torch.Tensor,
    amplitude_values: torch.Tensor,
    *,
    y_equator: float,
) -> float:
    y = torch.as_tensor(y_values, dtype=torch.float64)
    amplitude = torch.as_tensor(amplitude_values, dtype=torch.float64).abs()
    if y.numel() != amplitude.numel():
        raise ValueError("y_values and amplitude_values must have the same length")
    if y.numel() < 2:
        raise ValueError("at least two points are required")

    distance = (y - float(y_equator)).abs()
    order = torch.argsort(distance)
    distance_sorted = distance[order]
    amplitude_sorted = amplitude[order]
    peak = float(amplitude_sorted[0].item())
    if peak <= 0.0:
        raise ValueError("amplitude_values must contain at least one nonzero value")
    target = peak / math.e

    for idx in range(1, distance_sorted.numel()):
        left_amp = float(amplitude_sorted[idx - 1].item())
        right_amp = float(amplitude_sorted[idx].item())
        if right_amp > target:
            continue
        left_dist = float(distance_sorted[idx - 1].item())
        right_dist = float(distance_sorted[idx].item())
        if abs(right_amp - left_amp) < 1.0e-12:
            return right_dist
        weight = (target - left_amp) / (right_amp - left_amp)
        return left_dist + weight * (right_dist - left_dist)

    return float(distance_sorted[-1].item())
