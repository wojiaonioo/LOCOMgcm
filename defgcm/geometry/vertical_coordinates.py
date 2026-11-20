from __future__ import annotations

from dataclasses import dataclass, field

import torch


def mit_layer_centres(drf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """MITgcm drC and |rC| of a z-coordinate grid (ini_vertical_grid.F:123-146).

    drC(1) = drF(1)/2, drC(k) = (drF(k-1) + drF(k))/2; rC accumulates
    rkSign * drC.  Returned as positive (nz,) tensors ``drc`` and ``zc``.
    """
    drc = torch.empty_like(drf)
    drc[0] = 0.5 * drf[0]
    drc[1:] = 0.5 * (drf[:-1] + drf[1:])
    zc = torch.empty_like(drf)
    acc = drc[0].clone()
    zc[0] = acc
    for k in range(1, drf.shape[0]):
        acc = acc + drc[k]
        zc[k] = acc
    return drc, zc


def reference_layer_centres(drf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate reference geometry in binary64 and return the model's coordinate dtype."""
    spacing, centers = mit_layer_centres(drf.to(torch.float64))
    return spacing.to(drf.dtype), centers.to(drf.dtype)


def layer_thickness_from_center_spacing(delrc) -> tuple[float, ...]:
    """MITgcm drF from delRc alone, interfaces midway between centres (ini_vertical_grid.F:91-100).

    ``delrc`` holds the ``Nr+1`` centre spacings drC(1..Nr+1) in metres.
    """
    delrc = [float(v) for v in delrc]
    nr = len(delrc) - 1
    drf = [0.0] * nr
    drf[0] = delrc[0]
    for k in range(1, nr):
        drf[k] = 0.5 * delrc[k]
        drf[k - 1] = 0.5 * delrc[k] + drf[k - 1]
    drf[nr - 1] = delrc[nr] + drf[nr - 1]
    return tuple(drf)


@dataclass(frozen=True)
class ZLevelCoordinate:
    """Positive-down z-level vertical coordinates.

    ``layer_thickness`` stores one thickness per model layer in meters.
    Uniform grids use :meth:`uniform`; MITgcm-style non-uniform ``delR``
    vectors use :meth:`from_delR`.  Interface depths are derived from the
    layer thicknesses.  The centre spacing drC(1..Nr+1) is MITgcm's delRc when
    ``center_spacing_given`` holds it (ini_vertical_grid.F:102-106), otherwise
    the centres sit midway between interfaces (:120-127); centre depths
    accumulate it in the given case (:139-142).
    """

    layer_thickness: torch.Tensor
    # MITgcm delRc, Nr+1 values in metres; empty when delRc is not set.
    center_spacing_given: tuple[float, ...] = ()
    top_layer_thickness_value: float = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.center_spacing_given:
            # ini_vertical_grid.F:104-117: Nr+1 values, each > 0.
            if len(self.center_spacing_given) != self.nz + 1:
                raise ValueError(
                    f"center_spacing_given has {len(self.center_spacing_given)} values, "
                    f"needs nz+1 = {self.nz + 1}"
                )
            if not all(v > 0.0 for v in self.center_spacing_given):
                raise ValueError("center_spacing_given values must be strictly positive")
        # Host metadata must exist before the first compiled consumer, including after .to().
        object.__setattr__(self, "top_layer_thickness_value", float(self.layer_thickness[0].item()))

    @classmethod
    def uniform(
        cls,
        nz: int,
        depth: float,
        *,
        center_spacing_given: tuple[float, ...] = (),
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> "ZLevelCoordinate":
        if nz < 1:
            raise ValueError("nz must be positive")
        layer_thickness = torch.full(
            (nz,),
            depth / nz,
            device=device,
            dtype=dtype,
        )
        return cls(
            layer_thickness=layer_thickness,
            center_spacing_given=tuple(float(v) for v in center_spacing_given),
        )

    @classmethod
    def from_delR(
        cls,
        delR_array,
        *,
        center_spacing_given: tuple[float, ...] = (),
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> "ZLevelCoordinate":
        layer_thickness = torch.as_tensor(delR_array, dtype=dtype, device=device)
        if layer_thickness.ndim != 1:
            raise ValueError("delR_array must be 1-D")
        if layer_thickness.numel() == 0:
            raise ValueError("delR_array must be non-empty")
        if not torch.isfinite(layer_thickness).all():
            raise ValueError("delR_array values must be finite")
        if not (layer_thickness > 0).all():
            raise ValueError("delR_array values must be strictly positive")
        return cls(
            layer_thickness=layer_thickness.clone(),
            center_spacing_given=tuple(float(v) for v in center_spacing_given),
        )

    @property
    def nz(self) -> int:
        return int(self.layer_thickness.shape[0])

    @property
    def total_depth(self) -> torch.Tensor:
        return self.layer_thickness.sum()

    @property
    def interface_depth(self) -> torch.Tensor:
        zeros = torch.zeros(
            (1,),
            dtype=self.layer_thickness.dtype,
            device=self.layer_thickness.device,
        )
        return torch.cat((zeros, torch.cumsum(self.layer_thickness, dim=0)), dim=0)

    @property
    def center_spacing(self) -> torch.Tensor:
        """MITgcm drC(1..Nr+1), ``[nz+1]`` in the coordinate's dtype."""
        dr = self.layer_thickness
        if self.center_spacing_given:
            return torch.tensor(self.center_spacing_given, dtype=dr.dtype, device=dr.device)
        return torch.cat((0.5 * dr[:1], 0.5 * (dr[:-1] + dr[1:]), 0.5 * dr[-1:]))

    def reference_centres_for(
        self, dtype: torch.dtype, device: torch.device | str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MITgcm drC(1..Nr) and |rC| accumulated in binary64, returned in ``dtype``
        (``reference_layer_centres`` of the layers, or of the given delRc)."""
        if not self.center_spacing_given:
            return reference_layer_centres(self.layer_thickness.to(dtype=dtype, device=device))
        drc = torch.tensor(self.center_spacing_given[: self.nz], dtype=torch.float64, device=device)
        centres = [drc[0]]
        for k in range(1, self.nz):
            centres.append(centres[-1] + drc[k])
        return drc.to(dtype), torch.stack(centres).to(dtype)

    @property
    def bottom_half_thickness(self) -> torch.Tensor:
        """``[1]``: the distance from the last centre to the floor, rC(Nr) - rF(Nr+1)."""
        if self.center_spacing_given:
            return self.interface_depth[-1:] - self.center_depth[-1:]
        return 0.5 * self.layer_thickness[-1:]

    @property
    def center_depth(self) -> torch.Tensor:
        if self.center_spacing_given:
            # rC(1) = rF(1) + rkSign*drC(1), rC(k) = rC(k-1) + rkSign*drC(k), positive down.
            drc = self.center_spacing
            centres = [drc[0]]
            for k in range(1, self.nz):
                centres.append(centres[-1] + drc[k])
            return torch.stack(centres)
        interface_depth = self.interface_depth
        return 0.5 * (interface_depth[1:] + interface_depth[:-1])

    @property
    def interface_spacing(self) -> torch.Tensor:
        if self.nz < 2:
            return self.layer_thickness.new_empty((0,))
        if self.center_spacing_given:
            return self.center_spacing[1:-1]
        return 0.5 * (self.layer_thickness[:-1] + self.layer_thickness[1:])
