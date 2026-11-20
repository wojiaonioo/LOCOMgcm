from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import torch


class FieldLocation(StrEnum):
    CENTER = "center"
    U_FACE = "u_face"
    V_FACE = "v_face"
    W_FACE = "w_face"


@dataclass(frozen=True)
class Field:
    name: str
    data: torch.Tensor
    location: FieldLocation
    units: str
    halo_shape: tuple[int, int] = (0, 0)
    mask_ref: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def with_data(self, data: torch.Tensor) -> "Field":
        # Direct construction, not ``dataclasses.replace`` (a Dynamo "skip
        # function" inside the compiled step); same fields, same sharing of
        # ``metadata`` as ``replace`` would give.
        return Field(
            self.name,
            data,
            self.location,
            self.units,
            self.halo_shape,
            self.mask_ref,
            self.metadata,
        )
