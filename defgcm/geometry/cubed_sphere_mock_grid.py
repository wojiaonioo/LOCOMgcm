"""Cubed-sphere mock `Grid` placeholder.

This module ships a placeholder class that is
``isinstance(grid, Grid)`` ``True`` and ``isinstance(grid, StructuredCGrid)``
``False``, demonstrating that the topology-agnostic `Grid` Protocol
(`defgcm.geometry.grid.Grid`) is sufficient to
drive the operator stack without leaking the structured lat-lon backend
type. Test fixture; production code should use CubedSphereGrid via from_cs32_grid.
Real cubed-sphere work uses `CubedSphereGrid` (per-face metrics + true
`tile_connectivity` descriptor + seam exchange).

Design choices
--------------
* **Not a subclass of `StructuredCGrid`**.  The whole point of the mock is
  to prove the abstract `Grid` Protocol is exercisable on its own.
  Subclassing or composition with `StructuredCGrid` would defeat that.
  Every Protocol member at ``geometry/grid.py:46-113`` is therefore
  implemented directly here.
* **Frozen dataclass** mirroring `StructuredCGrid`'s style — the operator
  stack only ever reads from a `Grid`, so immutability is the natural
  default. Mutable runtime caches live in a per-instance ``_runtime_cache``
  attribute, outside the frozen geometry fields.
* **Internal data** is a plain uniform lat-lon-ish layout built at
  ``__init__`` time from numpy / torch arrays via ``MetricSet.rectangular``
  + a hand-built ``MaskSet`` (a small interior-land patch keeps
  ``maskInC`` non-trivial so any operator that consults it exercises a
  realistic path).  Float64 throughout.
* **Topology choice** is configurable via the ``boundary_topology``
  constructor argument so the same mock can drive a periodic, wall, or
  open scenario.  The class deliberately does *not* use
  `BoundaryType.SEAM`; the cubed-sphere seam case belongs to `CubedSphereGrid`.
* **Accessor semantics** for the four behavioural Protocol predicates
  (`edge_face_couples`, `physical_wall`, `open_boundary`,
  `w_face_thickness_edge_couples`) are the **lat-lon analogue** of the
  eventual cubed-sphere intent.  Each accessor's docstring in `grid.py`
  records the cubed-sphere-seam contract; this mock returns the same
  per-axis answer `StructuredCGrid` would for the structured topology it
  picks (so the test suite can verify *behavioural* equivalence on the
  picked topology without coupling to the lat-lon implementation type).

This mock is intentionally **not** registered as a default factory; the
runnability test instantiates it directly.  Tests that need a real
`Grid` must keep using `StructuredCGrid` factory helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Mapping

import torch

from .grid import BoundaryTopology, BoundaryType
from .masks import MaskSet
from .metrics import MetricSet, _cache_device_key
from .vertical_coordinates import ZLevelCoordinate


@dataclass(frozen=True)
class CubedSphereMockGrid:
    """A `Grid`-Protocol-conformant placeholder that is NOT a `StructuredCGrid`.

    Construct via :meth:`from_uniform_lat_lon` for the common case.  The
    dataclass fields mirror the Protocol surface so consumers can spot the
    1:1 mapping without indirection.
    """

    nx: int
    ny: int
    metrics: MetricSet
    masks: MaskSet
    boundary_topology: BoundaryTopology
    vertical_coordinate: ZLevelCoordinate
    face_count: int = 1
    tile_connectivity: Mapping[tuple[int, str], tuple[int, str, int]] = field(default_factory=dict)
    cube_corner_indices: tuple[tuple[int, int, int], ...] = ()
    distributed_neighbors: Mapping[str, object] = field(default_factory=dict)
    parallel_layout: object | None = None

    def __post_init__(self) -> None:
        # Match production grids without exposing unrelated memo keys to Dynamo.
        object.__setattr__(self, "_runtime_cache", {})

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_uniform_lat_lon(
        cls,
        *,
        nx: int = 6,
        ny: int = 5,
        nz: int = 3,
        lx: float = 6.0,
        ly: float = 5.0,
        depth: float = 30.0,
        boundary_topology: BoundaryTopology | None = None,
        interior_land_patch: bool = True,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> "CubedSphereMockGrid":
        """Build a minimal uniform lat-lon-ish mock grid.

        Parameters
        ----------
        nx, ny, nz :
            Horizontal and vertical extents.  Defaults pick a small but
            non-trivial size (``ny >= 2`` and ``nz >= 2`` are required by
            several operator branches; defaults satisfy both).
        lx, ly, depth :
            Domain extents in metres.
        boundary_topology :
            Topology to advertise to the operator stack.  Defaults to
            ``BoundaryTopology(PERIODIC, PERIODIC)``.
        interior_land_patch :
            When ``True`` (default) a single cell at the geometric centre
            is forced dry, so 3-D ``maskInC`` carries a non-trivial wet /
            land contrast.  Disable to fall back to an all-wet column.
        device, dtype :
            Tensor placement.
        """
        if nx < 2 or ny < 2:
            raise ValueError("CubedSphereMockGrid needs nx>=2 and ny>=2")
        if nz < 1:
            raise ValueError("CubedSphereMockGrid needs nz>=1")
        topo = boundary_topology or BoundaryTopology()

        # Bathymetry: full-depth ocean with one interior land patch.
        bathy = torch.full((ny, nx), float(depth), dtype=dtype, device=device)
        if interior_land_patch:
            bathy[ny // 2, nx // 2] = 0.0

        vert = ZLevelCoordinate.uniform(nz, depth, device=device, dtype=dtype)

        # Build a `MetricSet.rectangular`-style metric block.  We use the
        # bathymetry to populate `cell_volume` and `depth_center` directly
        # so the masks and metrics agree on the same bathy view.
        metrics = MetricSet.rectangular(
            nx=nx,
            ny=ny,
            lx=lx,
            ly=ly,
            depth=depth,
            device=device,
            dtype=dtype,
        )
        # Replace depth_center / cell_volume so the interior land patch
        # shows up in metric-driven operators (no public mutator on
        # MetricSet — we substitute by hand since this is mock data).
        depth_center = bathy.clone()
        cell_volume = depth_center * (lx / nx) * (ly / ny)
        metrics_obj = MetricSet(
            dx_scalar=metrics.dx_scalar,
            dy_scalar=metrics.dy_scalar,
            dx_center=metrics.dx_center,
            dy_center=metrics.dy_center,
            cell_area=metrics.cell_area,
            cell_volume=cell_volume,
            depth_center=depth_center,
            face_area_u=metrics.face_area_u,
            face_area_v=metrics.face_area_v,
            dxG_row=metrics.dxG_row,
            dxC_row=metrics.dxC_row,
            dyG_scalar_1d=metrics.dyG_scalar_1d,
            dxG_row_ext=metrics.dxG_row_ext,
            rAz_row=metrics.rAz_row,
        )

        # Masks: use MaskSet.from_bathymetry_3d so hFacC/hFacW/hFacS and
        # maskInC/W/S are all populated consistently with the chosen
        # topology, matching what StructuredCGrid.with_bathymetry would
        # build.
        masks = MaskSet.from_bathymetry_3d(
            bathy,
            vert.interface_depth,
            device=device,
            boundary_topology=topo,
            hfac_min=0.3,
            hfac_min_dz=5.0,
        )

        return cls(
            nx=nx,
            ny=ny,
            metrics=metrics_obj,
            masks=masks,
            boundary_topology=topo,
            vertical_coordinate=vert,
        )

    # ------------------------------------------------------------------
    # Protocol properties — shapes
    # ------------------------------------------------------------------
    @property
    def nz(self) -> int:
        return self.vertical_coordinate.nz

    @property
    def dx(self) -> float:
        return self.metrics.dx

    @property
    def dy(self) -> float:
        return self.metrics.dy

    @property
    def center_shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    @property
    def u_shape(self) -> tuple[int, int]:
        return (self.ny, self.nx + 1)

    @property
    def v_shape(self) -> tuple[int, int]:
        return (self.ny + 1, self.nx)

    @property
    def tracer_shape(self) -> tuple[int, int, int]:
        return (self.nz, self.ny, self.nx)

    # ------------------------------------------------------------------
    # Protocol properties — vertical structure
    # ------------------------------------------------------------------
    @property
    def layer_thickness(self) -> torch.Tensor:
        return self.vertical_coordinate.layer_thickness

    @property
    def center_depth(self) -> torch.Tensor:
        return self.vertical_coordinate.center_depth

    @property
    def interface_depth(self) -> torch.Tensor:
        return self.vertical_coordinate.interface_depth

    # ------------------------------------------------------------------
    # Protocol runtime accessors (cached)
    # ------------------------------------------------------------------
    def _cached_runtime_tensor(
        self,
        name: str,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
        factory: Callable[[], torch.Tensor],
    ) -> torch.Tensor:
        key = (name, dtype, _cache_device_key(device))
        cached = self._runtime_cache.get(key)
        if cached is None:
            cached = factory()
            self._runtime_cache[key] = cached
        return cached

    def layer_thickness_for(self, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        return self._cached_runtime_tensor(
            "layer_thickness",
            dtype=dtype,
            device=device,
            factory=lambda: self.layer_thickness.to(dtype=dtype, device=device),
        )

    def hfac_w_for(self, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        return self._cached_runtime_tensor(
            "hfac_w",
            dtype=dtype,
            device=device,
            factory=lambda: self._build_hfac_w(dtype=dtype, device=device),
        )

    def hfac_s_for(self, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        return self._cached_runtime_tensor(
            "hfac_s",
            dtype=dtype,
            device=device,
            factory=lambda: self._build_hfac_s(dtype=dtype, device=device),
        )

    def hfac_c_for(self, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        return self._cached_runtime_tensor(
            "hfac_c",
            dtype=dtype,
            device=device,
            factory=lambda: self._build_hfac_c(dtype=dtype, device=device),
        )

    def _build_hfac_w(self, *, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        if self.masks.has_3d_masks and self.masks.hFacW is not None:
            return self.masks.hFacW.to(dtype=dtype, device=device)
        return torch.ones((self.nz, self.ny, self.nx + 1), dtype=dtype, device=device)

    def _build_hfac_s(self, *, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        if self.masks.has_3d_masks and self.masks.hFacS is not None:
            return self.masks.hFacS.to(dtype=dtype, device=device)
        return torch.ones((self.nz, self.ny + 1, self.nx), dtype=dtype, device=device)

    def _build_hfac_c(self, *, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        if self.masks.has_3d_masks and self.masks.hFacC is not None:
            return self.masks.hFacC.to(dtype=dtype, device=device)
        return torch.ones((self.nz, self.ny, self.nx), dtype=dtype, device=device)

    # ------------------------------------------------------------------
    # Protocol behavioural accessors
    # ------------------------------------------------------------------
    # Each method's contract is documented on the Protocol in `grid.py`.
    # For the lat-lon-style structured topology this mock advertises, the
    # answers reduce to the same boolean reductions `StructuredCGrid`
    # exposes — `physical_wall` is True only for a WALL axis,
    # `open_boundary` is True only for an OPEN axis, and the two
    # edge-couple predicates collapse to `periodic_x/y` or `open_x/y` per
    # the per-method docstrings.  A future cubed-sphere `Grid` will
    # override these with seam-aware truth values; the mock keeps the
    # structured semantics so the operator stack runs without changes.

    def face_seam_rotation(self, face: int, side: Literal["west", "east", "south", "north"]) -> int:
        raise NotImplementedError(
            f"CubedSphereMockGrid is single-patch; no face seam rotation (face={face}, side={side})"
        )

    def edge_face_couples(self, axis: Literal["x", "y"]) -> bool:
        if axis == "x":
            return self.boundary_topology.periodic_x
        if axis == "y":
            return self.boundary_topology.periodic_y
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")

    def physical_wall(self, axis: Literal["x", "y"]) -> bool:
        if axis == "x":
            return self.boundary_topology.west_east == BoundaryType.WALL
        if axis == "y":
            return self.boundary_topology.south_north == BoundaryType.WALL
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")

    def open_boundary(self, axis: Literal["x", "y"]) -> bool:
        if axis == "x":
            return self.boundary_topology.open_x
        if axis == "y":
            return self.boundary_topology.open_y
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")

    def w_face_thickness_edge_couples(self, axis: Literal["x", "y"]) -> bool:
        if axis == "x":
            return self.boundary_topology.open_x or self.boundary_topology.periodic_x
        if axis == "y":
            return self.boundary_topology.open_y or self.boundary_topology.periodic_y
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")


__all__ = ["CubedSphereMockGrid"]
