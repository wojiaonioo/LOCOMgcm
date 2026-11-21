"""The two records crossing between the sea-ice component and an ocean.

``OceanSurface`` is what the ice reads from the ocean: the step-entry surface
layer (MIT kSurface) after the ocean's end-of-step halo exchange, in native
tile storage. ``PublishedFluxes`` is what the ocean reads back: the final
versions of the surface fluxes, in MIT units and signs, consumed once per
outer interval. Both the single-layer host and a three-dimensional ocean
produce and consume these records only.
"""

from dataclasses import dataclass, fields

import torch

from .records import OceanFluxes, OpenWaterFluxes
from .space import SeaiceSpace


@dataclass(frozen=True)
class OceanSurface:
    """Step-entry ocean surface read by EXF and both ice packages, MIT signs.

    theta (degC) and salt (g/kg) at C points, uVel/vVel (m/s, positive
    east/north) at the west/south faces, etaN (m, positive up) at C points;
    float64 native G fields. Dry cells hold what the ocean holds (+0.0 for
    tracers, face-masked velocities).
    """

    theta: torch.Tensor
    salt: torch.Tensor
    uVel: torch.Tensor
    vVel: torch.Tensor
    etaN: torch.Tensor


@dataclass(frozen=True)
class PublishedFluxes:
    values: OceanFluxes
    entry_iter: int
    start: float
    end: float
    chain: str


@dataclass(frozen=True)
class OceanConventionFluxes:
    heat_flux: torch.Tensor
    qsw: torch.Tensor
    empmr: torch.Tensor
    salt_flux: torch.Tensor
    fu: torch.Tensor
    fv: torch.Tensor
    ice_load: torch.Tensor


def open_water_version(open_water: OpenWaterFluxes, sIceLoad: torch.Tensor, AREA: torch.Tensor) -> OceanFluxes:
    """EXF's first version; later stages replace values, never add a second copy."""
    return OceanFluxes(open_water.Qnet_ow, open_water.Qsw_ow, open_water.EmPmR_ow,
                       open_water.saltFlux_ow, open_water.fu_ow, open_water.fv_ow, sIceLoad, AREA)


def publish_fluxes(values: OceanFluxes, *, entry_iter: int, start: float, end: float, chain: str,
                   space: SeaiceSpace) -> PublishedFluxes:
    if chain not in ("T", "L", "H", "B") or end <= start:
        raise ValueError("invalid flux interval or chain")
    for field in fields(values):
        value = getattr(values, field.name)
        space.validate(value)
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"nonfinite final flux {field.name}")
    return PublishedFluxes(values, entry_iter, start, end, chain)


def to_ocean_convention(values: OceanFluxes) -> OceanConventionFluxes:
    """Adapt only consumers requiring heat_flux/salt_flux conventions.

    FFIELDS.h:29-53: EmPmR stays positive upward; Qsw retains MIT sign.
    The single-layer host and surface_flux.py consume MIT values directly.
    """
    return OceanConventionFluxes(-values.Qnet, values.Qsw, values.EmPmR,
                                 -values.saltFlux, values.fu, values.fv, values.sIceLoad)
