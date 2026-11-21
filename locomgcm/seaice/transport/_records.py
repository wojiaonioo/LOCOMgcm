"""Explicit inputs and stage results for the native sea-ice transport tiles.

Horizontal tensors are float64 [bj, bi, j, i] native tiles (SeaiceSpace.shape; offline [2, 2, 27, 46]).
Temperatures add the singleton category axis before j. All signs are MITgcm
signs: Qnet, EmPmR and saltFlux are positive out of the ocean. The three
oce* melt corrections are positive into the ocean (heat may be negative).
"""

from dataclasses import dataclass

from torch import Tensor

from ..records import Geometry as TransportGeometry

Scalar = float | Tensor


@dataclass(frozen=True)
class SeaiceInventory:
    HEFF: Tensor
    AREA: Tensor
    HSNOW: Tensor


@dataclass(frozen=True)
class RegularizationParameters:
    siEps: Scalar
    area_floor: Scalar
    area_max: Scalar
    celsius2K: Scalar


@dataclass(frozen=True)
class RegularizationResult:
    inventory: SeaiceInventory
    TICES: Tensor
    d_HEFFbyNEG: Tensor
    d_HSNWbyNEG: Tensor
    SIareaPR: Tensor


@dataclass(frozen=True)
class ThsiceFields:
    """Five advected fields plus snow age; thicknesses are actual, not effective."""

    iceMask: Tensor
    iceHeight: Tensor
    snowHeight: Tensor
    Qice1: Tensor
    Qice2: Tensor
    snowAge: Tensor


@dataclass(frozen=True)
class ThsiceParameters:
    deltaT: Scalar
    iceMaskMin: Scalar
    hIceMin: Scalar
    rhoi: Scalar
    rhos: Scalar
    saltIce: Scalar
    qsnow: Scalar


@dataclass(frozen=True)
class CapacityAdvectionResult:
    field: Tensor
    iceVol: Tensor
    afx: Tensor
    afy: Tensor


@dataclass(frozen=True)
class CapacityStage:
    """A B7 event; oldVol is undefined before the ice event in MITgcm."""

    name: str
    field: Tensor
    iceVol_in: Tensor
    iceVol_out: Tensor
    oldVol_in: Tensor | None
    oldVol_out: Tensor | None
    afx: Tensor
    afy: Tensor


@dataclass(frozen=True)
class FiveFieldTransport:
    """B7 outputs before interior fraction/thickness adjustment."""

    fields: ThsiceFields
    iceFrc: Tensor
    oldVol: Tensor
    iceVol: Tensor
    stages: tuple[CapacityStage, ...]


@dataclass(frozen=True)
class ThsiceAdjustment:
    fields: ThsiceFields
    iceVol: Tensor
    oceQnet: Tensor
    oceFWfx: Tensor
    oceSflx: Tensor


@dataclass(frozen=True)
class OceanTransportFields:
    Qnet: Tensor
    EmPmR: Tensor
    saltFlux: Tensor
    sIceLoad: Tensor


@dataclass(frozen=True)
class ThsiceTransportResult:
    five_fields: FiveFieldTransport
    adjusted: ThsiceAdjustment
    ocean: OceanTransportFields
