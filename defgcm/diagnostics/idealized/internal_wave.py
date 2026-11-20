"""Internal-wave diagnostics (idealized wave-validation only).

Analytic-solution / modal-projection / fitting tools for the two-layer and
continuously-stratified internal wave benchmarks. NOT metric-aware; assumes a
uniform grid only; NOT a member of the production ``DiagnosticSuite``.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from defgcm.dynamics import linear_density, linear_density_anomaly
from defgcm.geometry.vertical_coordinates import ZLevelCoordinate
from defgcm.operators import hydrostatic_pressure_potential


def reduced_gravity_from_density_jump(
    *,
    gravity: float,
    reference_density: float,
    upper_density: float,
    lower_density: float,
) -> float:
    return float(gravity * (lower_density - upper_density) / reference_density)


def reduced_gravity_from_linear_eos(
    *,
    gravity: float,
    reference_density: float,
    thermal_expansion: float,
    haline_contraction: float,
    reference_temperature: float,
    reference_salinity: float,
    upper_temperature: float,
    lower_temperature: float,
    upper_salinity: float,
    lower_salinity: float,
) -> float:
    upper_density = linear_density(
        torch.tensor(upper_temperature, dtype=torch.float64),
        torch.tensor(upper_salinity, dtype=torch.float64),
        reference_density=reference_density,
        thermal_expansion=thermal_expansion,
        haline_contraction=haline_contraction,
        reference_temperature=reference_temperature,
        reference_salinity=reference_salinity,
    )
    lower_density = linear_density(
        torch.tensor(lower_temperature, dtype=torch.float64),
        torch.tensor(lower_salinity, dtype=torch.float64),
        reference_density=reference_density,
        thermal_expansion=thermal_expansion,
        haline_contraction=haline_contraction,
        reference_temperature=reference_temperature,
        reference_salinity=reference_salinity,
    )
    return reduced_gravity_from_density_jump(
        gravity=gravity,
        reference_density=reference_density,
        upper_density=float(upper_density.item()),
        lower_density=float(lower_density.item()),
    )


def two_layer_long_wave_phase_speed(
    *,
    reduced_gravity: float,
    upper_layer_thickness: float,
    lower_layer_thickness: float,
) -> float:
    total_depth = upper_layer_thickness + lower_layer_thickness
    return float(math.sqrt(reduced_gravity * upper_layer_thickness * lower_layer_thickness / total_depth))


def two_layer_long_wave_frequency(
    *,
    reduced_gravity: float,
    upper_layer_thickness: float,
    lower_layer_thickness: float,
    wavelength: float,
) -> float:
    phase_speed = two_layer_long_wave_phase_speed(
        reduced_gravity=reduced_gravity,
        upper_layer_thickness=upper_layer_thickness,
        lower_layer_thickness=lower_layer_thickness,
    )
    return float(phase_speed * (2.0 * math.pi / wavelength))


def continuous_stratification_mode_frequency(
    *,
    brunt_vaisala_frequency: float,
    wavelength: float,
    depth: float,
    mode_number: int,
) -> float:
    if mode_number < 1:
        raise ValueError("mode_number must be positive")
    wavenumber = 2.0 * math.pi / wavelength
    vertical_wavenumber = mode_number * math.pi / depth
    omega_squared = (brunt_vaisala_frequency**2) * (wavenumber**2) / (wavenumber**2 + vertical_wavenumber**2)
    return float(math.sqrt(omega_squared))


def _collapse_to_profile(temp: torch.Tensor) -> torch.Tensor:
    if temp.ndim == 2:
        return temp
    if temp.ndim == 3:
        return temp.mean(dim=1)
    if temp.ndim == 4:
        return temp.mean(dim=(0, 2))
    raise ValueError(f"expected temp to have 2, 3, or 4 dimensions, got {temp.ndim}")


def interface_depth_from_isotherm(
    temp: torch.Tensor,
    center_depth: torch.Tensor,
    *,
    threshold: float,
) -> torch.Tensor:
    profile = _collapse_to_profile(torch.as_tensor(temp, dtype=torch.float64))
    if profile.ndim != 2:
        raise ValueError("collapsed profile must be 2D [z, x]")
    if profile.shape[0] != center_depth.numel():
        raise ValueError("center_depth length must match the vertical dimension of temp")
    if profile.shape[0] < 2:
        raise ValueError("at least two vertical levels are required to diagnose an interface depth")

    diff = profile - float(threshold)
    crossings = diff[:-1] * diff[1:] <= 0.0
    has_crossing = crossings.any(dim=0)
    first_crossing = torch.argmax(crossings.to(torch.int64), dim=0)

    t0 = profile[:-1].gather(0, first_crossing.unsqueeze(0)).squeeze(0)
    t1 = profile[1:].gather(0, first_crossing.unsqueeze(0)).squeeze(0)
    z0 = center_depth[first_crossing]
    z1 = center_depth[first_crossing + 1]
    denominator = t1 - t0
    fraction = torch.where(
        denominator.abs() > 1.0e-14,
        (float(threshold) - t0) / denominator,
        torch.full_like(denominator, 0.5),
    )
    interpolated_depth = z0 + fraction * (z1 - z0)

    nearest_depth = center_depth[torch.argmin(diff.abs(), dim=0)]
    return torch.where(has_crossing, interpolated_depth, nearest_depth)


def modal_projection(
    values: torch.Tensor | Sequence[float],
    mode: torch.Tensor | Sequence[float],
    *,
    remove_mean: bool = False,
) -> float:
    values_tensor = torch.as_tensor(values, dtype=torch.float64)
    mode_tensor = torch.as_tensor(mode, dtype=torch.float64)
    if values_tensor.shape != mode_tensor.shape:
        raise ValueError("values and mode must have the same shape")
    projected = values_tensor - values_tensor.mean() if remove_mean else values_tensor
    normalization = torch.mean(mode_tensor.square())
    if float(normalization.item()) <= 0.0:
        raise ValueError("mode must not be identically zero")
    return float((torch.mean(projected * mode_tensor) / normalization).item())


def separable_modal_projection(
    values: torch.Tensor,
    vertical_mode: torch.Tensor | Sequence[float],
    horizontal_mode: torch.Tensor | Sequence[float],
    *,
    remove_mean: bool = False,
) -> float:
    values_tensor = _collapse_to_profile(torch.as_tensor(values, dtype=torch.float64))
    if values_tensor.ndim != 2:
        raise ValueError("values must collapse to a 2D [z, x] field")
    vertical_mode_tensor = torch.as_tensor(vertical_mode, dtype=torch.float64)
    horizontal_mode_tensor = torch.as_tensor(horizontal_mode, dtype=torch.float64)
    if values_tensor.shape != (vertical_mode_tensor.numel(), horizontal_mode_tensor.numel()):
        raise ValueError("values shape must match the supplied vertical and horizontal modes")
    basis = vertical_mode_tensor.view(-1, 1) * horizontal_mode_tensor.view(1, -1)
    return modal_projection(values_tensor, basis, remove_mean=remove_mean)


def interface_mode_amplitude(
    temp: torch.Tensor,
    center_depth: torch.Tensor,
    *,
    threshold: float,
    mode: torch.Tensor | Sequence[float],
    reference_depth: float,
) -> float:
    interface_depth = interface_depth_from_isotherm(temp, center_depth, threshold=threshold)
    return modal_projection(interface_depth - float(reference_depth), mode)


def zero_crossing_times(
    time_seconds: Sequence[float] | torch.Tensor,
    signal: Sequence[float] | torch.Tensor,
) -> list[float]:
    time_tensor = torch.as_tensor(time_seconds, dtype=torch.float64)
    signal_tensor = torch.as_tensor(signal, dtype=torch.float64)
    if time_tensor.ndim != 1 or signal_tensor.ndim != 1:
        raise ValueError("time_seconds and signal must be 1D")
    if time_tensor.numel() != signal_tensor.numel():
        raise ValueError("time_seconds and signal must have the same length")
    if time_tensor.numel() < 2:
        raise ValueError("at least two samples are required")

    crossings: list[float] = []
    for index in range(1, int(signal_tensor.numel())):
        previous = float(signal_tensor[index - 1].item())
        current = float(signal_tensor[index].item())
        if previous == 0.0:
            crossings.append(float(time_tensor[index - 1].item()))
            continue
        if previous * current < 0.0:
            t0 = float(time_tensor[index - 1].item())
            t1 = float(time_tensor[index].item())
            fraction = -previous / (current - previous)
            crossings.append(t0 + fraction * (t1 - t0))
    return crossings


def estimate_angular_frequency_from_zero_crossings(
    time_seconds: Sequence[float] | torch.Tensor,
    signal: Sequence[float] | torch.Tensor,
) -> dict[str, float | list[float]]:
    crossings = zero_crossing_times(time_seconds, signal)
    if len(crossings) < 2:
        raise ValueError("at least two zero crossings are required to estimate the period")
    half_period = crossings[1] - crossings[0]
    period_seconds = 2.0 * half_period
    omega = 2.0 * math.pi / period_seconds
    return {
        "omega": float(omega),
        "period_seconds": float(period_seconds),
        "zero_crossing_times_seconds": crossings,
    }


def estimate_angular_frequency_from_all_zero_crossings(
    time_seconds: Sequence[float] | torch.Tensor,
    signal: Sequence[float] | torch.Tensor,
) -> dict[str, float | list[float]]:
    """Estimate angular frequency using the full span of zero crossings.

    Uses (last_crossing - first_crossing) / (n_crossings - 1) as the mean
    half-period, which is more robust than the two-crossing estimate when the
    signal has slight even-harmonic asymmetry.
    """
    crossings = zero_crossing_times(time_seconds, signal)
    if len(crossings) < 2:
        raise ValueError("at least two zero crossings are required to estimate the period")
    total_span = crossings[-1] - crossings[0]
    num_half_periods = len(crossings) - 1
    mean_half_period = total_span / num_half_periods
    period_seconds = 2.0 * mean_half_period
    omega = 2.0 * math.pi / period_seconds
    return {
        "omega": float(omega),
        "period_seconds": float(period_seconds),
        "zero_crossing_times_seconds": crossings,
    }


def estimate_angular_frequency_from_first_zero_crossing(
    time_seconds: Sequence[float] | torch.Tensor,
    signal: Sequence[float] | torch.Tensor,
) -> dict[str, float]:
    crossings = zero_crossing_times(time_seconds, signal)
    if not crossings:
        raise ValueError("at least one zero crossing is required to estimate the quarter period")
    quarter_period_seconds = float(crossings[0])
    if quarter_period_seconds <= 0.0:
        raise ValueError("the first zero crossing must occur after the initial sample")
    period_seconds = 4.0 * quarter_period_seconds
    omega = 2.0 * math.pi / period_seconds
    return {
        "omega": float(omega),
        "period_seconds": float(period_seconds),
        "quarter_period_seconds": float(quarter_period_seconds),
        "first_zero_crossing_time_seconds": float(crossings[0]),
    }


def baroclinic_free_surface_compensation(
    temp: torch.Tensor,
    salt: torch.Tensor,
    layer_thickness: torch.Tensor,
    *,
    gravity: float,
    reference_density: float,
    thermal_expansion: float,
    haline_contraction: float,
    reference_temperature: float,
    reference_salinity: float,
) -> torch.Tensor:
    density_anomaly = linear_density_anomaly(
        torch.as_tensor(temp, dtype=torch.float64),
        torch.as_tensor(salt, dtype=torch.float64),
        reference_density=reference_density,
        rho_const=reference_density,
        thermal_expansion=thermal_expansion,
        haline_contraction=haline_contraction,
        reference_temperature=reference_temperature,
        reference_salinity=reference_salinity,
    )
    vertical = ZLevelCoordinate(layer_thickness=torch.as_tensor(layer_thickness, dtype=torch.float64))
    pressure_potential = hydrostatic_pressure_potential(
        density_anomaly,
        vertical.center_spacing,
        vertical.bottom_half_thickness,
        gravity=gravity,
        reference_density=reference_density,
    )
    return -pressure_potential.mean(dim=1) / float(gravity)


def one_sided_power_spectrum(
    time_seconds: Sequence[float] | torch.Tensor,
    signal: Sequence[float] | torch.Tensor,
) -> dict[str, list[float]]:
    time_tensor = torch.as_tensor(time_seconds, dtype=torch.float64)
    signal_tensor = torch.as_tensor(signal, dtype=torch.float64)
    if time_tensor.ndim != 1 or signal_tensor.ndim != 1:
        raise ValueError("time_seconds and signal must be 1D")
    if time_tensor.numel() != signal_tensor.numel():
        raise ValueError("time_seconds and signal must have the same length")
    if time_tensor.numel() < 2:
        raise ValueError("at least two samples are required")

    sample_spacing = torch.diff(time_tensor)
    dt = float(sample_spacing.mean().item())
    if not torch.allclose(sample_spacing, torch.full_like(sample_spacing, dt), atol=1.0e-9, rtol=0.0):
        raise ValueError("time_seconds must have uniform spacing")

    detrended = signal_tensor - signal_tensor.mean()
    frequencies = torch.fft.rfftfreq(detrended.numel(), d=dt)
    power = torch.fft.rfft(detrended).abs().pow(2)
    angular_frequency = 2.0 * math.pi * frequencies
    return {
        "frequency_hz": frequencies.tolist(),
        "angular_frequency": angular_frequency.tolist(),
        "power": power.tolist(),
    }


def fit_sinusoid_at_angular_frequency(
    time_seconds: Sequence[float] | torch.Tensor,
    signal: Sequence[float] | torch.Tensor,
    *,
    omega: float,
) -> dict[str, float | list[float]]:
    time_tensor = torch.as_tensor(time_seconds, dtype=torch.float64)
    signal_tensor = torch.as_tensor(signal, dtype=torch.float64)
    design_matrix = torch.stack(
        (
            torch.cos(float(omega) * time_tensor),
            torch.sin(float(omega) * time_tensor),
            torch.ones_like(time_tensor),
        ),
        dim=1,
    )
    coefficients = torch.linalg.lstsq(design_matrix, signal_tensor).solution
    fitted = design_matrix @ coefficients
    residual = signal_tensor - fitted
    return {
        "fitted_signal": fitted.tolist(),
        "cos_coefficient": float(coefficients[0].item()),
        "sin_coefficient": float(coefficients[1].item()),
        "offset": float(coefficients[2].item()),
        "rmse": float(torch.sqrt(torch.mean(residual.square())).item()),
    }
