from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable, Literal, Mapping, Protocol, runtime_checkable

import torch

from defgcm.accel.constant_cache import _build_eager
from defgcm.accel.tensor_context import ordinary_tensor_mode

from .masks import MaskSet
from .metrics import MetricSet, _cache_device_key
from .vertical_coordinates import ZLevelCoordinate

#: Runtime-cache key of the memoised top-layer thickness (a Python float).
_TOP_LAYER_THICKNESS_KEY = ("layer_thickness_top", torch.float64, "scalar")
# Key names of the hfac_*_for copies (StructuredCGrid._hfac_runtime) and the mask attribute each copies.
_HFAC_RUNTIME_ATTRS = {"hfac_c": "hFacC", "hfac_w": "hFacW", "hfac_s": "hFacS"}


def top_layer_thickness_value(grid: object) -> float:
    """Python float of ``grid.layer_thickness[0]``, read from the tensor once.

    Production grids own this host constant in their vertical coordinate,
    initialized at construction rather than by a first eager model step.
    Duck-typed grids retain the legacy inline/memoised fallback.
    """
    vertical = getattr(grid, "vertical_coordinate", None)
    if isinstance(vertical, ZLevelCoordinate):
        return vertical.top_layer_thickness_value
    cache = getattr(grid, "_runtime_cache", None)
    if cache is None:  # grids without a runtime cache keep the inline read
        return float(grid.layer_thickness[0].item())  # type: ignore[attr-defined]
    cached = cache.get(_TOP_LAYER_THICKNESS_KEY)
    if cached is None:
        cached = float(grid.layer_thickness[0].item())  # type: ignore[attr-defined]
        cache[_TOP_LAYER_THICKNESS_KEY] = cached  # type: ignore[assignment]
    return float(cached)


def runtime_memo_constant(
    grid: object,
    key: tuple[object, ...],
    build: Callable[[], object],
) -> object:
    """Memoise one grid-constant product in the grid runtime cache.

    Same contract as :meth:`StructuredCGrid._runtime_memo` -- eager cold build,
    warm hit is a plain dictionary lookup and therefore traceable -- but it
    takes the grid by argument and accepts any product (a tuple of tensors, a
    scalar), so a caller outside ``geometry`` can memoise without reaching into
    the cache dictionary itself.  A grid with no runtime cache keeps the inline
    recompute, which is bit-for-bit the pre-memo path.
    """
    cache = getattr(grid, "_runtime_cache", None)
    if cache is None:
        return build()
    cached = cache.get(key)  # type: ignore[arg-type]
    if cached is None:
        cached = _build_eager(build)
        cache[key] = cached  # type: ignore[index,assignment]
    return cached


def device_resident_masks_enabled() -> bool:
    """Toggle for holding per-grid masks/metrics on the compute device (default ON).

    The cube operators read the large constant tensors ``masks.hFacC/W/S``, the
    boolean ``masks.{center,u,v}`` / ``maskIn{C,W,S}``, and the per-face
    ``metrics.face_geometry`` arrays — all CPU-resident — and ``.to(device=...)``
    them on **every** operator call (~841 ms/step of GPU memcpy at cs384, where
    each hFac is ~285 MB and gets re-copied by momentum, viscosity, vertical
    shear AND the per-tracer GAD pass).  When ON, :func:`ensure_grid_device_resident`
    migrates those raw tensors onto the device **once** (at the first step), so
    every downstream ``.to(device=...)`` becomes a same-device no-op.

    Set ``DEFGCM_DEVICE_RESIDENT_MASKS=0`` to fall back to the per-step
    ``masks.hFac*.to(device=...)`` path (bitwise-identical; for ablation).

    Note: ``vertical_coordinate.layer_thickness`` is deliberately **left on the
    host** (it is a tiny ``[nz]`` vector, so its per-step transfer is negligible)
    so that cheap host reads like ``float(grid.layer_thickness[0])`` in the
    steppers stay host-side and do not turn into a ``.item()`` device→host sync.

    The enable decision is owned by the unified control plane
    (``defgcm.accel``), not by this module.
    """
    from defgcm.accel import gate

    return gate("device_resident_masks").effective_value


def _record_device_resident(event: str, *, variant: str, site: str) -> None:
    """Cache record for DEFGCM_DEVICE_RESIDENT_MASKS (the cache-observation contract).

    Only ``warm-reuse`` is ignition: the cold migration is the host->device
    cost, the reuse on every later step is the win.  ``use-of-cached`` marks an
    operator actually reading a migrated tensor -- evidence that the migration
    is consumed, but explicitly NOT ignition.
    """
    from defgcm.accel import current_rank, record_cache_event

    record_cache_event(
        "device_resident_masks",
        event=event,
        variant=variant,
        site=site,
        backend="torch",
        rank=current_rank(),
    )


def record_device_resident_use(site: str, variant: str) -> None:
    """Record one operator read of a device-resident constant (use-of-cached).

    Called from the branch that actually reads the memoized device tensor, so
    the ledger separates "the switch was requested" from "an operator really
    consumed the migrated constant".  Never call it from the gate predicate.
    """
    _record_device_resident("use-of-cached", variant=variant, site=site)


# Tensors migrated onto the compute device by ``ensure_grid_device_resident``.
# Deliberately EXCLUDES ``vertical_coordinate.layer_thickness`` (tiny [nz] vector;
# kept host-side so ``float(layer_thickness[i])`` stays a host read — see the note
# in ``device_resident_masks_enabled``).  Each entry is a large constant that the
# cube operators otherwise re-transfer host→device every call.
_DEVICE_RESIDENT_MASK_ATTRS = (
    "center", "u", "v",
    "hFacC", "hFacW", "hFacS",
    "maskInC", "maskInW", "maskInS",
)


@ordinary_tensor_mode()
def ensure_grid_device_resident(grid: object, device: torch.device | str) -> None:
    """Move the grid's large constant mask/metric tensors onto ``device`` once.

    Idempotent and gated by :func:`device_resident_masks_enabled`.  Called at the
    top of the stepper ``step()`` so the first step on a given device pays the
    host→device copy of the (constant) masks / ``face_geometry`` exactly once;
    every subsequent ``masks.hFac*.to(device=...)`` / ``face_geometry[k].to(...)``
    in the operators is then a same-device no-op.

    Bit-exactness: replacing a CPU constant with its ``.to(device)`` copy changes
    nothing numerically (float64 / bool host→device is exact); the operators would
    have produced that exact device tensor on their own ``.to()``.
    ``DEFGCM_DEVICE_RESIDENT_MASKS=0`` skips it entirely (every operator then
    re-transfers per step, the original behaviour).

    Differentiability: the migrated tensors are constants **for an initial-state
    or parameter control**, but NOT for a geometry control -- under
    ``ALLOW_DEPTH_CONTROL`` ``hFacC/hFacW/hFacS`` are built from the bathymetry
    leaf and carry its graph.  The migration therefore runs in the caller's grad
    mode instead of under ``no_grad``: a ``requires_grad=False`` tensor's
    ``.to()`` records nothing whatever the mode, so the constant case pays
    exactly the copy it always paid, while a tensor that does carry a graph
    keeps it through the copy.  Migrating under ``no_grad`` replaced the
    differentiable ``hFacC`` with a detached leaf IN PLACE on the shared
    ``MaskSet``, so every consumer after the first step read a constant and the
    geometry gradient came back zero with no error anywhere.

    One consequence of keeping the graph: a migrated tensor is memoised on the
    grid for the life of that grid's runtime caches (``_runtime_cache``, and
    ``_hfac_runtime`` for the hFac entries), so it pins the graph that produced
    it.  A second ``backward()`` over a grid whose bathymetry graph was already
    freed raises the usual "backward through the graph a second time" --
    loudly, which is the point; rebuild the grid (or both runtime caches) per
    optimisation iteration, as the control drivers do.
    """
    if not device_resident_masks_enabled():
        return
    dev = torch.device(device)
    cache = getattr(grid, "_runtime_cache", None)
    if cache is None:  # grids without a runtime cache opt out (per-step .to path)
        # Requested but structurally unreachable on this grid -- declared with a
        # reason rather than passing silently (capability + fallback).
        from defgcm.accel import current_rank, record_dispatch

        record_dispatch(
            "device_resident_masks",
            outcome="inapplicable",
            variant="grid_migration",
            site="grid.ensure_grid_device_resident",
            backend="torch",
            rank=current_rank(),
            reason="grid_without_runtime_cache",
        )
        return
    sentinel = ("__device_resident__", _cache_device_key(dev))
    if cache.get(sentinel) is True:
        # The masks are already resident: this step reuses them instead of
        # re-transferring.  Warm reuse is the ignition event.
        _record_device_resident(
            "warm-reuse",
            variant="grid_migration",
            site="grid.ensure_grid_device_resident",
        )
        return

    # No ``no_grad`` here, and no ``requires_grad`` branch either: the fast path
    # IS the plain ``.to()``.  Autograd records nothing for a tensor that does
    # not require grad, so a constant costs the same copy it always cost and the
    # loop has one less context manager than before; a tensor that does require
    # grad keeps its ``ToCopyBackward`` edge.  A branch here would only add a
    # Dynamo guard on a per-tensor flag.
    masks = getattr(grid, "masks", None)
    if masks is not None:
        for attr in _DEVICE_RESIDENT_MASK_ATTRS:
            t = getattr(masks, attr, None)
            if torch.is_tensor(t) and t.device != dev:
                object.__setattr__(masks, attr, t.to(device=dev))

    metrics = getattr(grid, "metrics", None)
    fg = getattr(metrics, "face_geometry", None) if metrics is not None else None
    if isinstance(fg, dict):
        for key, t in list(fg.items()):
            if torch.is_tensor(t) and t.device != dev:
                fg[key] = t.to(device=dev)

    cache[sentinel] = True
    _record_device_resident(
        "cold-construction",
        variant="grid_migration",
        site="grid.ensure_grid_device_resident",
    )


class BoundaryType(StrEnum):
    PERIODIC = "periodic"
    WALL = "wall"
    OPEN = "open"
    SEAM = "seam"
    DISTRIBUTED = "distributed"


@dataclass(frozen=True)
class BoundaryTopology:
    west_east: BoundaryType = BoundaryType.PERIODIC
    south_north: BoundaryType = BoundaryType.PERIODIC

    @property
    def periodic_x(self) -> bool:
        return self.west_east == BoundaryType.PERIODIC

    @property
    def periodic_y(self) -> bool:
        return self.south_north == BoundaryType.PERIODIC

    @property
    def open_x(self) -> bool:
        return self.west_east == BoundaryType.OPEN

    @property
    def open_y(self) -> bool:
        return self.south_north == BoundaryType.OPEN

    @property
    def wall_x(self) -> bool:
        return self.west_east == BoundaryType.WALL

    @property
    def wall_y(self) -> bool:
        return self.south_north == BoundaryType.WALL

    @property
    def has_open_boundaries(self) -> bool:
        return self.open_x or self.open_y


def _side_is_open(side: str, grid: StructuredCGrid) -> bool:
    """Whether the ``side`` edge ("west", "east", "south", "north") of this tile is a physical open boundary.

    A side shared with a distributed neighbour rank is never open; otherwise the
    axis topology decides.
    """
    neighbours = getattr(grid, "distributed_neighbors", {}) or {}
    if neighbours.get(side) is not None:
        return False
    if side in ("west", "east"):
        return grid.boundary_topology.open_x
    return grid.boundary_topology.open_y


@runtime_checkable
class Grid(Protocol):
    nx: int
    ny: int
    metrics: MetricSet
    masks: MaskSet
    boundary_topology: BoundaryTopology
    face_count: int
    """Number of horizontal faces: 1 for single-patch grids, 6 for cubed-sphere grids."""
    tile_connectivity: Mapping[tuple[int, str], tuple[int, str, int]]
    """Mapping ``(face, side) -> (neighbour_face, neighbour_side, rotation)``.

    ``side`` is one of west/east/south/north and ``rotation`` is 0, 90,
    180, or 270 degrees. Single-patch grids return an empty mapping.
    """
    cube_corner_indices: tuple[tuple[int, int, int], ...]
    """Tuple of ``(face, i, j)`` entries for 3-valent cube vertices.

    Single-patch grids return ``()``; cubed-sphere grids list 8 vertices.
    """
    distributed_neighbors: Mapping[str, object]
    """Optional distributed transport neighbours keyed by side name.

    Single-process grids return an empty mapping.  Distributed regional tiles
    use this as the transport table; physical global-edge semantics still come
    from ``boundary_topology``.
    """
    parallel_layout: object | None
    """Optional rank-local decomposition metadata."""

    @property
    def nz(self) -> int:
        ...

    @property
    def dx(self) -> float:
        ...

    @property
    def dy(self) -> float:
        ...

    @property
    def center_shape(self) -> tuple[int, ...]:
        ...

    @property
    def u_shape(self) -> tuple[int, ...]:
        ...

    @property
    def v_shape(self) -> tuple[int, ...]:
        ...

    @property
    def tracer_shape(self) -> tuple[int, ...]:
        ...

    @property
    def layer_thickness(self) -> torch.Tensor:
        ...

    @property
    def center_depth(self) -> torch.Tensor:
        ...

    @property
    def interface_depth(self) -> torch.Tensor:
        ...

    def layer_thickness_for(self, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        ...

    def hfac_w_for(self, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        ...

    def hfac_s_for(self, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        ...

    def edge_face_couples(self, axis: Literal["x", "y"]) -> bool:
        ...

    def physical_wall(self, axis: Literal["x", "y"]) -> bool:
        ...

    def open_boundary(self, axis: Literal["x", "y"]) -> bool:
        ...

    def w_face_thickness_edge_couples(self, axis: Literal["x", "y"]) -> bool:
        ...

    def face_seam_rotation(self, face: int, side: Literal["west", "east", "south", "north"]) -> int:
        """Return seam rotation angle 0, 90, 180, or 270 for ``(face, side)``.

        This is the ergonomic shortcut for ``tile_connectivity[(face, side)][2]``.
        """
        ...


@dataclass(frozen=True)
class StructuredCGrid:
    nx: int
    ny: int
    metrics: MetricSet
    vertical_coordinate: ZLevelCoordinate
    masks: MaskSet
    boundary_topology: BoundaryTopology
    face_count: int = 1
    tile_connectivity: Mapping[tuple[int, str], tuple[int, str, int]] = field(default_factory=dict)
    cube_corner_indices: tuple[tuple[int, int, int], ...] = ()
    distributed_neighbors: Mapping[str, object] = field(default_factory=dict)
    parallel_layout: object | None = None
    comm_session: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Runtime memo contents are not geometry fields. Dynamo expands every
        # frozen dataclass field, even keys used only by an eager observer.
        # Keep the same per-instance memo; actual lookups still get guards.
        object.__setattr__(self, "_runtime_cache", {})
        # The hfac_*_for copies, apart: refresh_hfac_runtime rewrites them
        # inside a traced step, and a traced write to a dict guards on all of
        # its keys.
        object.__setattr__(self, "_hfac_runtime", {})

    @classmethod
    def rectangular(
        cls,
        *,
        nx: int,
        ny: int,
        nz: int,
        lx: float | None = None,
        ly: float | None = None,
        depth: float,
        del_x: torch.Tensor | Sequence[float] | None = None,
        del_y: torch.Tensor | Sequence[float] | None = None,
        vertical_layer_thicknesses=None,
        boundary_topology: BoundaryTopology | None = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> "StructuredCGrid":
        vert = _vertical_coordinate(
            nz,
            depth,
            vertical_layer_thicknesses=vertical_layer_thicknesses,
            device=device,
            dtype=dtype,
        )
        return cls(
            nx=nx,
            ny=ny,
            metrics=MetricSet.rectangular(
                nx=nx,
                ny=ny,
                lx=lx,
                ly=ly,
                depth=depth,
                del_x=del_x,
                del_y=del_y,
                device=device,
                dtype=dtype,
            ),
            vertical_coordinate=vert,
            masks=MaskSet.all_wet(ny, nx, device=device),
            boundary_topology=boundary_topology or BoundaryTopology(),
        )

    @classmethod
    def with_bathymetry(
        cls,
        *,
        nx: int,
        ny: int,
        nz: int,
        lx: float | None = None,
        ly: float | None = None,
        depth: float,
        bathymetry: torch.Tensor,
        del_x: torch.Tensor | Sequence[float] | None = None,
        del_y: torch.Tensor | Sequence[float] | None = None,
        vertical_layer_thicknesses=None,
        vertical_center_spacing: tuple[float, ...] = (),
        boundary_topology: BoundaryTopology | None = None,
        hfac_min: float = 0.3,
        hfac_min_dz: float = 5.0,
        obcs_fix_topo: bool = False,
        smooth_abs_func_range: float | None = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> "StructuredCGrid":
        """Create a grid with variable bathymetry and 3D masks.

        Parameters
        ----------
        bathymetry : Tensor [ny, nx]
            Ocean depth at each cell (positive = ocean, zero/negative = land).
        depth : float
            Maximum / reference depth for vertical coordinate.
        vertical_layer_thicknesses : sequence or Tensor, optional
            Per-layer z-level thicknesses in meters. Defaults to uniform
            ``depth / nz`` spacing when omitted.
        vertical_center_spacing : tuple of float
            MITgcm ``delRc`` (``nz+1`` centre spacings in metres) when given;
            empty puts the centres midway between interfaces.
        hfac_min : float
            Minimum fraction of a cell that must be wet (MITgcm hFacMin).
        hfac_min_dz : float
            Minimum dimensional thickness of a partial cell in metres
            (MITgcm hFacMinDr).
        obcs_fix_topo : bool
            MITgcm ``OBCSfixTopo``.  Default **False** = MITgcm's own default
            (``pkg/obcs/obcs_readparms.F:291``).  Applied identically to the
            MetricSet and MaskSet views below.
        smooth_abs_func_range : float or None
            MITgcm ``smoothAbsFuncRange``; forwarded verbatim to
            ``MaskSet.from_bathymetry_3d``, which documents the geometry it
            selects.  Default ``None`` = the existing (non-depth-control)
            hFac construction.
        """
        vert = _vertical_coordinate(
            nz,
            depth,
            vertical_layer_thicknesses=vertical_layer_thicknesses,
            vertical_center_spacing=vertical_center_spacing,
            device=device,
            dtype=dtype,
        )
        from defgcm.geometry.masks import obcs_fix_topo_and_clamp
        # Convention regression guard (a stage1a mesh pipeline that emitted
        # CF_STANDARD elevation
        # which feeds clamp(min=0) inside obcs_fix_topo_and_clamp →
        # land 200m elevation became "200m-deep ocean" and HYCOM IC filled
        # 28°C T onto Philippine islands).
        #
        # Three valid conventions for the bathy tensor:
        #   LOCOMgcm     positive=ocean depth, 0=land       → max>0, min=0
        #   unclamped -elevation → max>0, min<0,
        #                                              and max ≈ deep-ocean
        #                                              dominates |min|
        #   CF_STANDARD elevation (raw GEBCO)        → max>0, min<0, with
        #                                              |min| (=deep ocean
        #                                              depth) ≫ max
        #                                              (=continental height)
        #
        # Reject only the CF_STANDARD signature.
        #
        # Geometry generation runs in float64 whatever ``dtype`` the run uses:
        # see ``_GEOMETRY_DTYPE_NOTE``.
        b_in = bathymetry.to(dtype=_GEOMETRY_DTYPE, device=device)
        b_max = float(b_in.max().item())
        b_min = float(b_in.min().item())
        if b_max > 50.0 and b_min < -50.0 and -b_min > b_max * 1.5:
            raise ValueError(
                f"StructuredCGrid.with_bathymetry: bathymetry looks like CF_STANDARD "
                f"elevation (max={b_max:.1f} m, min={b_min:.1f} m; "
                f"negative-ocean dominates positive-land).  Convert to torch_ocean "
                f"(positive=ocean depth, 0=land) before calling — otherwise "
                f"clamp(min=0) will flip land into deep ocean.  Use "
                f"convention_convert(to='torch_ocean') in stage1a mesh pipeline."
            )
        vert_geom = _geometry_vertical_coordinate(
            vert,
            nz,
            depth,
            vertical_layer_thicknesses=vertical_layer_thicknesses,
            device=device,
            dtype=dtype,
        )
        column_floor = float(vert_geom.interface_depth[-1].item())
        bathy_geom = obcs_fix_topo_and_clamp(
            b_in,
            column_floor=column_floor,
            boundary_topology=boundary_topology,
            obcs_fix_topo=obcs_fix_topo,
        )
        # The metrics are storage, so they carry the run dtype; the topology
        # above is not, so it stays float64.  NOT bit-identical to narrowing
        # first on a reduced-precision run -- see ``_GEOMETRY_DTYPE_NOTE``.
        bathy_eff = bathy_geom.to(dtype=dtype)
        return cls(
            nx=nx,
            ny=ny,
            metrics=MetricSet.with_bathymetry(
                nx=nx,
                ny=ny,
                lx=lx,
                ly=ly,
                bathymetry=bathy_eff,
                del_x=del_x,
                del_y=del_y,
                device=device,
                dtype=dtype,
            ),
            vertical_coordinate=vert,
            masks=MaskSet.from_bathymetry_3d(
                bathy_geom,
                vert_geom.interface_depth,
                device=device,
                boundary_topology=boundary_topology,
                hfac_min=hfac_min,
                hfac_min_dz=hfac_min_dz,
                # Keep this value equal to the MetricSet setting above.
                obcs_fix_topo=obcs_fix_topo,
                smooth_abs_func_range=smooth_abs_func_range,
            ),
            boundary_topology=boundary_topology or BoundaryTopology(),
        )

    @classmethod
    def spherical_polar(
        cls,
        *,
        nx: int,
        ny: int,
        nz: int,
        ygOrigin_deg: float,
        delY_deg: float,
        delX_deg: float,
        depth: float,
        bathymetry: torch.Tensor | None = None,
        vertical_layer_thicknesses=None,
        vertical_center_spacing: tuple[float, ...] = (),
        r_sphere: float = 6_370_000.0,
        boundary_topology: BoundaryTopology | None = None,
        hfac_min: float = 0.3,
        hfac_min_dz: float = 5.0,
        obcs_fix_topo: bool = False,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> "StructuredCGrid":
        """MITgcm-compatible spherical polar grid.

        Builds a grid whose horizontal metrics match MITgcm's
        ``usingSphericalPolarGrid=.TRUE.`` path bit-for-bit (default
        ``rSphere = 6_370_000.0 m``).  See :meth:`MetricSet.spherical_polar`
        for the exact formulas.

        Parameters
        ----------
        nx, ny, nz : int
            Grid dimensions.
        ygOrigin_deg : float
            South-west corner latitude (Fortran ``ygOrigin``).
        delY_deg : float
            Latitude cell size (uniform).
        delX_deg : float
            Longitude cell size (uniform).
        depth : float
            Reference total depth for the vertical coordinate.
        bathymetry : Tensor [ny, nx] or None
            Per-column ocean depth in metres, LOCOMgcm sign: positive =
            ocean, zero or negative = land (the sign of MITgcm's runtime
            ``Depth``; a MITgcm ``bathyFile`` is negative in the ocean and must
            be negated first).  ``obcs_fix_topo_and_clamp`` clamps land to 0
            and depths below the column floor to the floor.  If ``None``,
            defaults to a flat ``depth``-deep ocean (wall-boundary test
            configuration).
        vertical_layer_thicknesses : sequence or Tensor, optional
            Per-layer z-level thicknesses in meters. Defaults to uniform
            ``depth / nz`` spacing when omitted.
        r_sphere : float
            Earth radius in metres.  Default ``6_370_000`` m, matching MITgcm's
            ``rSphere`` default.
        hfac_min : float
            Minimum fraction of a cell that must be wet (MITgcm hFacMin).
        hfac_min_dz : float
            Minimum dimensional thickness of a partial cell in metres
            (MITgcm hFacMinDr).
        obcs_fix_topo : bool
            MITgcm ``OBCSfixTopo``.  Default **False** = MITgcm's own default
            (``pkg/obcs/obcs_readparms.F:291``).  Applied identically to the
            MetricSet and MaskSet views below.
        """
        vert = _vertical_coordinate(
            nz,
            depth,
            vertical_layer_thicknesses=vertical_layer_thicknesses,
            vertical_center_spacing=vertical_center_spacing,
            device=device,
            dtype=dtype,
        )
        if bathymetry is None:
            bathy_eff = torch.full((ny, nx), float(depth), dtype=dtype, device=device)
            masks = MaskSet.all_wet(ny, nx, device=device)
        else:
            # Apply the same OBCSfixTopo and column-floor clamp as
            # MaskSet.from_bathymetry_3d so metrics and masks agree at
            # open-boundary strips whose interior neighbour is land.
            from defgcm.geometry.masks import obcs_fix_topo_and_clamp
            # Geometry generation runs in float64 whatever ``dtype`` the run
            # uses; see ``_GEOMETRY_DTYPE_NOTE``.
            vert_geom = _geometry_vertical_coordinate(
                vert,
                nz,
                depth,
                vertical_layer_thicknesses=vertical_layer_thicknesses,
                device=device,
                dtype=dtype,
            )
            column_floor = float(vert_geom.interface_depth[-1].item())
            bathy_geom = obcs_fix_topo_and_clamp(
                bathymetry.to(dtype=_GEOMETRY_DTYPE, device=device),
                column_floor=column_floor,
                boundary_topology=boundary_topology,
                obcs_fix_topo=obcs_fix_topo,
            )
            # Metrics are storage and carry the run dtype.  NOT bit-identical
            # to narrowing first on a reduced-precision run -- see
            # ``_GEOMETRY_DTYPE_NOTE``.
            bathy_eff = bathy_geom.to(dtype=dtype)
            masks = MaskSet.from_bathymetry_3d(
                bathy_geom,
                vert_geom.interface_depth,
                device=device,
                boundary_topology=boundary_topology,
                hfac_min=hfac_min,
                hfac_min_dz=hfac_min_dz,
                # Keep this value equal to the MetricSet setting below.
                obcs_fix_topo=obcs_fix_topo,
            )
        return cls(
            nx=nx,
            ny=ny,
            metrics=MetricSet.spherical_polar(
                nx=nx,
                ny=ny,
                ygOrigin_deg=ygOrigin_deg,
                delY_deg=delY_deg,
                delX_deg=delX_deg,
                bathymetry=bathy_eff,
                depth=depth,
                r_sphere=r_sphere,
                device=device,
                dtype=dtype,
            ),
            vertical_coordinate=vert,
            masks=masks,
            boundary_topology=boundary_topology or BoundaryTopology(),
        )

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

    def to_runtime(
        self,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> "StructuredCGrid":
        """Move a regional grid without dropping Cartesian vector metrics."""
        def metric_tensor(name: str) -> torch.Tensor:
            # Spacing is a geometry input; only the derived metrics narrow.
            target_dtype = torch.float64 if name in ("del_x", "del_y") else dtype
            return getattr(self.metrics, name).to(device=device, dtype=target_dtype)

        optional_metric_names = (
            "del_x",
            "del_y",
            "dxF",
            "dyF",
            "dxG",
            "dyG",
            "dxC",
            "dyC",
            "dxV",
            "dyU",
            "rA",
            "rAw",
            "rAs",
            "rAz",
        )
        optional_metrics = {
            name: (
                None
                if getattr(self.metrics, name) is None
                else metric_tensor(name)
            )
            for name in optional_metric_names
        }
        metrics = MetricSet(
            dx_scalar=self.metrics.dx_scalar,
            dy_scalar=self.metrics.dy_scalar,
            dx_center=metric_tensor("dx_center"),
            dy_center=metric_tensor("dy_center"),
            cell_area=metric_tensor("cell_area"),
            cell_volume=metric_tensor("cell_volume"),
            depth_center=metric_tensor("depth_center"),
            face_area_u=metric_tensor("face_area_u"),
            face_area_v=metric_tensor("face_area_v"),
            dxG_row=metric_tensor("dxG_row"),
            dxC_row=metric_tensor("dxC_row"),
            dyG_scalar_1d=self.metrics.dyG_scalar_1d,
            dxG_row_ext=metric_tensor("dxG_row_ext"),
            rAz_row=metric_tensor("rAz_row"),
            # Host float64 latitudes: neither moved nor narrowed.
            yC_row=self.metrics.yC_row,
            yG_row=self.metrics.yG_row,
            **optional_metrics,
        )
        masks = MaskSet(
            center=self.masks.center.to(device=device),
            u=self.masks.u.to(device=device),
            v=self.masks.v.to(device=device),
            hFacC=(
                None
                if self.masks.hFacC is None
                else self.masks.hFacC.to(device=device, dtype=dtype)
            ),
            hFacW=(
                None
                if self.masks.hFacW is None
                else self.masks.hFacW.to(device=device, dtype=dtype)
            ),
            hFacS=(
                None
                if self.masks.hFacS is None
                else self.masks.hFacS.to(device=device, dtype=dtype)
            ),
            maskInC=(
                None
                if self.masks.maskInC is None
                else self.masks.maskInC.to(device=device, dtype=dtype)
            ),
            maskInW=(
                None
                if self.masks.maskInW is None
                else self.masks.maskInW.to(device=device, dtype=dtype)
            ),
            maskInS=(
                None
                if self.masks.maskInS is None
                else self.masks.maskInS.to(device=device, dtype=dtype)
            ),
            # Geometry MODE travels with the tensors it shaped.  hFacC/W/S are
            # copied with the depth-control formulas already applied, but the
            # corner hFacZ is computed by its consumer from this flag
            # (horizontal_diffusion._hfacz_smooth_geometry). Preserve the flag
            # so the corner uses the same geometry rule as the face fractions.
            smooth_abs_func_range=self.masks.smooth_abs_func_range,
        )
        return StructuredCGrid(
            nx=self.nx,
            ny=self.ny,
            metrics=metrics,
            vertical_coordinate=ZLevelCoordinate(
                layer_thickness=self.vertical_coordinate.layer_thickness.to(
                    device=device, dtype=dtype
                ),
                center_spacing_given=self.vertical_coordinate.center_spacing_given,
            ),
            masks=masks,
            boundary_topology=self.boundary_topology,
            face_count=self.face_count,
            tile_connectivity=self.tile_connectivity,
            cube_corner_indices=self.cube_corner_indices,
            distributed_neighbors=self.distributed_neighbors,
            parallel_layout=self.parallel_layout,
        )

    @property
    def tracer_shape(self) -> tuple[int, int, int]:
        return (self.nz, self.ny, self.nx)

    @property
    def layer_thickness(self) -> torch.Tensor:
        return self.vertical_coordinate.layer_thickness

    @property
    def center_depth(self) -> torch.Tensor:
        return self.vertical_coordinate.center_depth

    @property
    def interface_depth(self) -> torch.Tensor:
        return self.vertical_coordinate.interface_depth

    def _runtime_memo(
        self,
        key: tuple[str, torch.dtype, str],
        build: Callable[[], torch.Tensor],
    ) -> torch.Tensor:
        """Route cold builds outside traced CUDAGraph ownership.
        The eager builder owns the tensor before it enters the runtime memo.
        Warm hits remain a plain dictionary lookup and therefore stay traceable.
        The unchanged key and build expression keep the result bitwise neutral.
        """
        cached = self._runtime_cache.get(key)
        if cached is None:
            cached = _build_eager(build)
            self._runtime_cache[key] = cached
        return cached

    def _cached_runtime_tensor(
        self,
        name: str,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
        factory: Callable[[], torch.Tensor],
    ) -> torch.Tensor:
        key = (name, dtype, _cache_device_key(device))
        return self._runtime_memo(key, factory)

    def layer_thickness_for(self, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        return self._cached_runtime_tensor(
            "layer_thickness",
            dtype=dtype,
            device=device,
            factory=lambda: self.layer_thickness.to(dtype=dtype, device=device),
        )

    def _cached_hfac(self, name: str, *, dtype: torch.dtype, device: torch.device | str,
                     build: Callable[[], torch.Tensor]) -> torch.Tensor:
        """:meth:`_runtime_memo` for an hFac copy, held in ``_hfac_runtime``.

        :meth:`refresh_hfac_runtime` rewrites these entries inside the traced
        step; kept out of ``_runtime_cache``, they leave that memo read-only
        there, so adding an unrelated key does not invalidate the step.
        """
        key = (name, dtype, _cache_device_key(device))
        cached = self._hfac_runtime.get(key)
        if cached is None:
            cached = _build_eager(build)
            self._hfac_runtime[key] = cached
        return cached

    def hfac_w_for(self, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        return self._cached_hfac(
            "hfac_w",
            dtype=dtype,
            device=device,
            build=lambda: self._build_hfac_w(dtype=dtype, device=device),
        )

    def hfac_s_for(self, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        return self._cached_hfac(
            "hfac_s",
            dtype=dtype,
            device=device,
            build=lambda: self._build_hfac_s(dtype=dtype, device=device),
        )

    def hfac_c_for(self, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        """Device-resident, memoized ``hFacC`` (mirrors :meth:`hfac_w_for`).

        Removes the per-step ``masks.hFacC.to(device=...)`` host→device transfer
        in the momentum / viscosity operators.  Bit-identical to the per-step
        ``.to()`` (float64 H2D is exact).
        """
        return self._cached_hfac(
            "hfac_c",
            dtype=dtype,
            device=device,
            build=lambda: self._build_hfac_c(dtype=dtype, device=device),
        )

    def refresh_hfac_runtime(self) -> None:
        """Rebuild every ``hfac_*_for`` entry from the replaced ``masks.hFac*``.

        Called with autograd enabled, right after
        ``MaskSet.replace_surface_hfac`` (``UPDATE_SURF_DR`` and the
        checkpoint codec).  An entry of the masks' dtype and device is the
        replaced tensor itself and is rebound to the new one; an entry of
        another dtype or device is a converted copy and is converted again
        from the new tensor.  Every read on the gradient path then carries the
        current surface level and its graph.

        Only ``_hfac_runtime`` is visited and written (:meth:`_cached_hfac`), so
        a traced step guards on those few entries and not on ``_runtime_cache``.

        The no-grad path writes the surface level in place and does not call
        this: an entry that is the mask tensor sees the write, a converted copy
        keeps the old surface level.  A consumer reads the mask tensor itself
        only when it requests the masks' dtype and device; grids built by the
        runtime hold the masks at the run's dtype and device
        (``experiments/cases/_runtime.py``).
        """
        for key, value in self._hfac_runtime.items():
            current = getattr(self.masks, _HFAC_RUNTIME_ATTRS[key[0]])
            if value is not current:
                self._hfac_runtime[key] = current.to(dtype=key[1], device=key[2])

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

    def face_seam_rotation(self, face: int, side: Literal["west", "east", "south", "north"]) -> int:
        raise NotImplementedError(
            f"StructuredCGrid is single-patch; no face seam rotation (face={face}, side={side})"
        )

    def edge_face_couples(self, axis: Literal["x", "y"]) -> bool:
        """Whether the two domain-edge u/v faces on ``axis`` couple two cells.

        A topology-agnostic query used by face-interpolation operators
        (``center_to_u``/``center_to_v``) to decide how to treat the first
        and last face on an axis. After a 1-cell ghost ``exchange()``, the
        interpolated edge face value depends on the ghost-fill policy:

        * On a **periodic** axis the ghost holds the wrap neighbour, so the
          edge face genuinely couples two interior cells and its
          interpolated (wrap-average) value must be kept.
        * On a **wall** or **open** axis the ghost is zero, so the edge face
          does not couple two cells and its interpolated value (half the
          interior value) is meaningless and must be zeroed.
        * On a future cubed-sphere **seam** the ghost holds the neighbouring
          face's value, so the edge face again couples two cells and its
          value must be kept; such a ``Grid`` returns ``True`` for the
          seam-bearing axis.

        Returning a per-axis ``bool`` keeps the contract purely behavioural:
        operators consult it instead of reading ``boundary_topology``, so a
        non-lat-lon ``Grid`` plugs in without touching the operator.

        Parameters
        ----------
        axis : {"x", "y"}
            ``"x"`` queries the west/east u-face edges; ``"y"`` queries the
            south/north v-face edges.
        """
        if axis == "x":
            return self.boundary_topology.periodic_x
        if axis == "y":
            return self.boundary_topology.periodic_y
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")

    def physical_wall(self, axis: Literal["x", "y"]) -> bool:
        """Whether ``axis`` is bounded by a physical solid wall.

        A topology-agnostic predicate for operators that apply a no-slip
        lateral term only at a *solid* boundary (e.g. the W side-drag in
        ``vertical_momentum.py``). It is ``True`` only for a ``WALL`` axis:

        * **wall** — a solid boundary; the term fires.
        * **open** — an open boundary; the term must *not* fire (the open
          strip is owned by OBCS / removed from the solved interior).
        * **periodic** — no boundary; the term must not fire.
        * a future cubed-sphere **seam** is not a wall either, so such a
          ``Grid`` returns ``False`` and the term correctly does not fire.

        Operators consult this instead of reading ``boundary_topology``,
        so a non-lat-lon ``Grid`` plugs in without touching the operator.

        Parameters
        ----------
        axis : {"x", "y"}
            ``"x"`` queries the west/east boundary; ``"y"`` the south/north.
        """
        if axis == "x":
            return self.boundary_topology.west_east == BoundaryType.WALL
        if axis == "y":
            return self.boundary_topology.south_north == BoundaryType.WALL
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")

    def open_boundary(self, axis: Literal["x", "y"]) -> bool:
        """Whether ``axis`` carries an open boundary owned by OBCS.

        A topology-agnostic predicate for operators that apply an
        open-boundary-only treatment (e.g. the OBC first-order upwind tracer
        flux override in ``fluxes/tracer_flux.py``). It is ``True`` only for an
        ``OPEN`` axis:

        * **open** — an open boundary; OBCS owns the boundary strip and the
          open-boundary treatment fires on this axis.
        * **wall** — a solid boundary; OBCS does not own it, the treatment
          must not fire (the boundary is closed, no flux crosses it).
        * **periodic** — no boundary; the treatment must not fire.
        * a future cubed-sphere **seam** is not an open boundary either, so
          such a ``Grid`` returns ``False`` and the treatment does not fire.

        This is deliberately distinct from ``w_face_thickness_edge_couples``,
        which answers a different question ("do the domain-end W-cell faces
        couple two cells") and is ``True`` on every axis except a wall —
        a periodic axis and a cubed-sphere seam return ``True`` there but
        ``False`` here. Operators consult this instead of reading
        ``boundary_topology``, so a non-lat-lon ``Grid`` plugs in without
        touching the operator.

        Parameters
        ----------
        axis : {"x", "y"}
            ``"x"`` queries the west/east boundary; ``"y"`` the south/north.
        """
        neighbours = self.distributed_neighbors or {}
        if axis == "x":
            return self.boundary_topology.open_x and (
                neighbours.get("west") is None or neighbours.get("east") is None
            )
        if axis == "y":
            return self.boundary_topology.open_y and (
                neighbours.get("south") is None or neighbours.get("north") is None
            )
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")

    def w_face_thickness_edge_couples(self, axis: Literal["x", "y"]) -> bool:
        """Whether the domain-edge W-cell face on ``axis`` pairs two cells.

        Used by ``vertical_momentum._w_cell_thickness_{w,s}_face`` to build
        the partial-cell ``r_low`` at the first/last W-cell face on an axis.
        An interior W-cell face combines its two adjacent cell columns; the
        domain-edge face has only one interior neighbour, so its second
        operand is topology-dependent:

        * On an axis whose edge faces *couple* the two domain-end cells the
          edge face combines cell ``0`` with cell ``N-1`` (a wrap pairing).
        * Otherwise the edge face replicates its single interior cell.

        Only a ``wall`` axis replicates. A wall has no neighbour column and
        ``MaskSet.from_bathymetry_3d`` zeroes ``hFacW``/``hFacS`` there, so
        its edge face carries no geometry at all. Every other axis type puts
        a genuine neighbour column in the ghost slot, and MITgcm builds the
        face geometry from it identically in each case:
        ``rLowW = MAX(R_low(i-1), R_low(i))`` over the halo-extended index
        range (``ini_masks_etc.F:332,342``), the ``rSurfW/S`` clamp against
        that paired low value (``:334,344``), then the exchange on MITgcm's
        doubly periodic process topology (``:402-404``).

        * **periodic** — the ghost holds the wrap neighbour, so the edge face
          is an ordinary two-column face and its seam slot carries the paired
          minimum-depth geometry rather than the local column's.
        * **open** — the verbatim baseline arm; the OBCS-owned column plays
          the neighbour's role.
        * a cubed-sphere **seam** couples the two end cells through the
          adjacent face, so a seam-bearing axis pairs as well.

        This predicate is the sole authority on that decision. An operator
        that had to combine it with ``edge_face_couples`` to recover the
        periodic case would hold half the topology contract outside the
        ``Grid``, where a new ``Grid`` implementation cannot reach it; the
        distinct question ``edge_face_couples`` answers (about the u/v faces,
        where an ``open`` axis does *not* couple) is documented there.
        ``vertical_momentum`` deliberately does not fold
        ``distributed_neighbors`` in here: a distributed tile edge is
        resolved per side from the exchanged halo further downstream, which
        knows the real neighbour rank's column, whereas an all-or-nothing
        per-axis answer would wrap-pair the other, physical edge of a
        one-sided tile.

        Returning a per-axis ``bool`` keeps the contract behavioural:
        operators consult it instead of reading ``boundary_topology``.

        Parameters
        ----------
        axis : {"x", "y"}
            ``"x"`` queries the west/east W-cell face edges; ``"y"`` the
            south/north W-cell face edges.
        """
        if axis == "x":
            return self.boundary_topology.open_x or self.boundary_topology.periodic_x
        if axis == "y":
            return self.boundary_topology.open_y or self.boundary_topology.periodic_y
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")


#: Dtype every grid constructor generates wet/dry geometry in, whatever the run
#: precision is.  See ``_GEOMETRY_DTYPE_NOTE``.
_GEOMETRY_DTYPE = torch.float64

_GEOMETRY_DTYPE_NOTE = """Generate geometry in float64 even for a float32 run.

The hFacMin closure comparison has no tolerance. Narrowing bathymetry before
computing hFac can move a column across that threshold and change its wet mask.
The clamp, column floor, hFac and masks therefore use float64 inputs; stored
metrics and hFac products use the run dtype. A float64 run needs no casts.

A float32 cumulative sum of layer thicknesses can differ from the float64
column floor and thus change cell volumes and reciprocal column depths.
Grid artifacts store interface_depth in the run dtype, while masks derive
from the float64 geometry. Consumers must read the stored masks rather than
reconstruct them from narrowed interface depths.
"""


def _geometry_vertical_coordinate(
    vert: ZLevelCoordinate,
    nz: int,
    depth: float,
    *,
    vertical_layer_thicknesses,
    device: str,
    dtype: torch.dtype,
) -> ZLevelCoordinate:
    """Return the float64 twin of ``vert`` used for mask/hFac construction.

    Widening ``vert.interface_depth`` after the fact would not do: the interface
    depths are a cumulative sum, so a float32 coordinate has already lost the
    low bits that decide which side of a closure threshold a column falls on.

    Every keyword is required on purpose.  A ``dtype`` default would make an
    omitted argument return ``vert`` unchanged and turn the whole guard into a
    silent no-op; a ``vertical_layer_thicknesses`` default of ``None`` would
    silently rebuild the twin as a UNIFORM column and hand the mask a different
    set of layer interfaces.  Both are the shape the explicit-input requirement exists to
    stop: an omitted argument translated into "the thing does not exist".
    """
    if dtype is _GEOMETRY_DTYPE:
        return vert
    return _vertical_coordinate(
        nz,
        depth,
        vertical_layer_thicknesses=vertical_layer_thicknesses,
        device=device,
        dtype=_GEOMETRY_DTYPE,
    )


def _vertical_coordinate(
    nz: int,
    depth: float,
    *,
    vertical_layer_thicknesses=None,
    vertical_center_spacing: tuple[float, ...] = (),
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> ZLevelCoordinate:
    if vertical_layer_thicknesses is None:
        return ZLevelCoordinate.uniform(
            nz, depth, center_spacing_given=tuple(vertical_center_spacing), device=device, dtype=dtype
        )
    vert = ZLevelCoordinate.from_delR(
        vertical_layer_thicknesses,
        center_spacing_given=tuple(vertical_center_spacing),
        device=device,
        dtype=dtype,
    )
    if vert.nz != nz:
        raise ValueError("vertical_layer_thicknesses length must equal nz")
    return vert


def _compute_interior_mask(
    grid: StructuredCGrid,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute OBCS interior masks (MITgcm maskInC/maskInW/maskInS).

    For open boundaries, the boundary-strip center cells are excluded from the
    elliptic solve and later re-imposed by the boundary-condition layer. The
    excluded rows/columns become trivial outside-solve rows in the Helmholtz
    operator (`aC=1`, masked RHS / initial guess).

    The masks are grid constants, as in MITgcm obcs_init_fixed.F:154-156.
    Memoize them per (grid, dtype, device); cold construction runs eagerly.

    The memo is bypassed when the grid declares its ``hFac`` fields time-varying
    (``nonlinFreeSurf>=1`` rewrites the surface level in place every step,
    ``MaskSet.mark_hfac_time_varying``), because ``maskInW``/``maskInS`` carry an
    ``hFacW/hFacS > 0`` factor.  That predicate is a one-way per-run latch, so it
    is constant across steps and cannot guard-fail under ``torch.compile``.

    Returns
    -------
    maskInC : [ny, nx]    — 1 for interior wet solve cells, 0 for OBC strip/land
    maskInW : [ny, nx+1]  — 1 for active u-faces, 0 for inactive faces
    maskInS : [ny+1, nx]  — 1 for active v-faces, 0 for inactive faces
    """
    if not getattr(grid.masks, "hfac_time_varying", False):
        return runtime_memo_constant(  # type: ignore[return-value]
            grid,
            ("cg2d_interior_masks", dtype, _cache_device_key(device)),
            lambda: _build_interior_mask(grid, dtype, device),
        )
    return _build_interior_mask(grid, dtype, device)


def _build_interior_mask(
    grid: StructuredCGrid,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The unchanged body of :func:`_compute_interior_mask` (the cold build)."""
    ny, nx = grid.ny, grid.nx

    # Start from surface wet mask
    if grid.masks.has_3d_masks:
        surface_mask = grid.masks.center_float(dtype=dtype, device=device)[0]  # [ny, nx]
    else:
        surface_mask = torch.ones(ny, nx, dtype=dtype, device=device)

    maskInC = surface_mask.clone()

    # MITgcm OBCS excludes the open-boundary cell strip from the elliptic
    # solve domain via maskInC, while retaining the first interior normal face
    # through the MAX rule used below for maskInW/maskInS.  Distributed tiles
    # keep internal rank seams in the solve domain; only physical global OBC
    # sides are removed.
    west_open = _side_is_open("west", grid)
    east_open = _side_is_open("east", grid)
    south_open = _side_is_open("south", grid)
    north_open = _side_is_open("north", grid)
    # ``.zero_()`` rather than ``= 0.0``: the assignment form goes through
    # ``Tensor.fill_(cpu 0-d tensor)``, which reads the scalar back with
    # ``value.item()`` when the devices differ.  Same
    # ``aten::fill_`` kernel, same +0.0.  It matters on the ``hfac_time_varying``
    # bypass, where this body still runs on every call.
    if west_open:
        maskInC[:, 0].zero_()
    if east_open:
        maskInC[:, -1].zero_()
    if south_open:
        maskInC[0, :].zero_()
    if north_open:
        maskInC[-1, :].zero_()

    # maskInW: interior u-face mask
    # MITgcm obcs_init_fixed.F:154-156:
    #   maskInW(i,j) = maskW(i,j,1) * MAX(insideMask(i-1), insideMask(i))
    # The maskW factor zeros faces where hFacW=0 (e.g. OB faces where
    # MITgcm sets hFacW=0 in the halo/OBC strip). LOCOMgcm has no halo,
    # but hFacW at OBC boundary faces can be 0 depending on the setup.
    maskInW = torch.zeros(ny, nx + 1, dtype=dtype, device=device)
    maskInW[:, 1:-1] = torch.max(maskInC[:, :-1], maskInC[:, 1:])
    neighbours = getattr(grid, "distributed_neighbors", {}) or {}
    if grid.boundary_topology.periodic_x or neighbours.get("west") is not None:
        maskInW[:, 0] = maskInC[:, 0]
    if grid.boundary_topology.periodic_x or neighbours.get("east") is not None:
        maskInW[:, -1] = maskInC[:, -1]

    # Multiply by surface-layer face wet mask (maskW at k=0)
    if grid.masks.hFacW is not None:
        hFacW_surf = grid.masks.hFacW[0].to(dtype=dtype, device=device)
        face_wet_W = (hFacW_surf > 0).to(dtype=dtype)
        maskInW = maskInW * face_wet_W

    # maskInS: interior v-face mask (same convention)
    maskInS = torch.zeros(ny + 1, nx, dtype=dtype, device=device)
    maskInS[1:-1, :] = torch.max(maskInC[:-1, :], maskInC[1:, :])
    if grid.boundary_topology.periodic_y or neighbours.get("south") is not None:
        maskInS[0, :] = maskInC[0, :]
    if grid.boundary_topology.periodic_y or neighbours.get("north") is not None:
        maskInS[-1, :] = maskInC[-1, :]

    if grid.masks.hFacS is not None:
        hFacS_surf = grid.masks.hFacS[0].to(dtype=dtype, device=device)
        face_wet_S = (hFacS_surf > 0).to(dtype=dtype)
        maskInS = maskInS * face_wet_S

    return maskInC, maskInW, maskInS
