"""Shared log-linear least-squares helper for idealized wave-fit diagnostics.

Pure structural extraction of the common "design matrix + lstsq + extract
intercept/slope" segment shared by
``gill_response.fit_gaussian_waveguide_scale`` and
``kelvin_wave.fit_exponential_decay_scale``. Mask / clamp / threshold logic
stays in each caller; this helper performs only the regression step.
"""

from __future__ import annotations

import torch


def _loglinear_lstsq(
    feature: torch.Tensor,
    log_amp: torch.Tensor,
) -> tuple[float, float]:
    """Fit ``log_amp ~= intercept + slope * feature`` by ordinary least squares.

    Builds the ``[ones, feature]`` design matrix, solves with
    ``torch.linalg.lstsq``, and returns ``(intercept, slope)`` as Python floats.
    Callers own any masking / clamping of ``feature`` and ``log_amp`` upstream.
    """
    design = torch.stack([torch.ones_like(feature), feature], dim=1)
    coeffs = torch.linalg.lstsq(design, log_amp).solution
    intercept = float(coeffs[0].item())
    slope = float(coeffs[1].item())
    return intercept, slope
