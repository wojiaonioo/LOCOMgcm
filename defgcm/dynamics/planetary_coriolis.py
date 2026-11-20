"""Planetary vorticity from the grid's geography.

Kernels that turn latitudes or metric rows into Coriolis parameters.
``spherical_coriolis`` is ``f = 2 * omega * sin(lat)`` in s^-1 for a
latitude field in degrees, at whatever points the latitudes are given.
``f_cori_cos_rows`` is the horizontal component ``fCoriCos = 2 * omega *
cos(lat)`` broadcast as ``[1, ny, 1]`` cell-centre rows, from a centre-latitude
cosine that ``latitude_center_cos_from_metrics`` infers from spherical-polar
``dxC`` rows and ``dyC`` (``dxC = rSphere*cos(yC)*dlon``; uniform ``dxC``
rows give a cosine of one).  ``omega`` is the planetary rotation rate in
rad/s, which the caller derives from its configured rotation period
(``defgcm.constants.omega_from_rotation_period``).  Everything stays in the
inputs' dtype.
"""

from __future__ import annotations

import torch


def spherical_coriolis(lat_deg: torch.Tensor, *, omega: float) -> torch.Tensor:
    """``2 omega sin(lat)`` for a latitude field in degrees."""
    return 2.0 * omega * torch.sin(torch.deg2rad(lat_deg))


def latitude_center_cos_from_metrics(
    dx_c: torch.Tensor,
    dy: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Infer center-latitude cosine from caller-provided dxC and dy metrics.

    Uniformity is a property of the metrics, decided on the device: a select,
    not a host branch.  The branch form read the predicate through ``bool()``
    inside the traced step (``Data-dependent branching`` at this line, four
    times a step in the non-hydrostatic case; torch 2.9's strict tracing makes
    it an error).  ``where`` returns the same bytes as the branch it selects.
    """
    if dx_c.numel() <= 1:
        return torch.ones_like(dx_c)
    is_uniform = torch.max(torch.abs(dx_c - dx_c.mean())) <= torch.finfo(dtype).eps
    return torch.where(
        is_uniform, torch.ones_like(dx_c), torch.clamp(dx_c / dy, min=-1.0, max=1.0)
    )


def f_cori_cos_rows(
    latitude_cos: torch.Tensor,
    *,
    ny: int,
    omega: float,
) -> torch.Tensor:
    """Broadcast nonhydrostatic U-face Coriolis cosine factor ``2 omega cos(lat)``."""
    return (2.0 * omega * latitude_cos).view(1, ny, 1)
