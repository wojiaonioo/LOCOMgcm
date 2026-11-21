"""Cubed-sphere `Grid` peer of `StructuredCGrid`.

`CubedSphereGrid` is a frozen dataclass that satisfies the locked
`Grid` Protocol without inheriting from `StructuredCGrid`.  It carries the
per-face metrics, tile connectivity, cube-corner indices and bathymetry-aware
masks; vector rotation and cross-face exchange live in
:mod:`locomgcm.geometry.exchange`.
"""

from __future__ import annotations

from locomgcm.external_paths import resolve

from dataclasses import dataclass, field
from typing import Callable, Literal, Mapping

import torch

from locomgcm.accel.constant_cache import _build_eager

from .grid import _GEOMETRY_DTYPE, BoundaryTopology, BoundaryType
from .masks import MaskSet
from .metrics import MetricSet, _cache_device_key
from .tiling import CubeTileLayout, MitTiling
from .vertical_coordinates import ZLevelCoordinate


_CUBED_SPHERE_FACE_COUNT = 6

#: The two horizontal-advection topologies a cube grid may declare.
#: ``mit_12tile`` = MIT's SIZE.h tile GAD pass schedule on the tiling the grid
#: declares (``mit_tiling``; the name records the audited cs32x15 instance,
#: 12 tiles of 32x16 with OL=4, the only layout checked bitwise against MIT);
#: ``six_face_approx`` = the six-face OL=1 approximation every cube size may run.
ADVECTION_TOPOLOGIES = ("mit_12tile", "six_face_approx")

_CUBE_SIDES = ("west", "east", "south", "north")


def _default_boundary_topology() -> BoundaryTopology:
    return BoundaryTopology(
        west_east=BoundaryType.SEAM,
        south_north=BoundaryType.SEAM,
    )


def _run_dtype_vertical_coordinate(
    vertical: ZLevelCoordinate,
    dtype: torch.dtype,
) -> ZLevelCoordinate:
    """Narrow the float64 geometry coordinate back to the run dtype for storage.

    The cube masks are built from the float64 coordinate (see
    ``from_csN_grid``), but the coordinate the grid *stores* is storage like
    every other grid tensor, so it carries the run dtype.  A float64 run gets
    the object back unchanged, so that path is bit-identical.

    Note the same caveat ``grid.py``'s ``_GEOMETRY_DTYPE_NOTE`` records for the
    lat-lon line: on a float32 run the stored coordinate can no longer
    reproduce the stored masks, because the masks came from the float64 twin.
    Consumers must read the masks, not rebuild them.

    Float32 metric note: cube metrics derive ``depth`` from
    ``interface_depth[-1]`` (cubed_sphere_metric_builder) as a single rounding
    of the f64 cumsum, not a native-f32 cumsum, so a float32 run's
    cell_volume/face_area_* can differ by ~1 ULP from a native-f32 build.
    The shipped float32 cases (cube_cs1024_prod / _ah125_prod / _smoke)
    take this path.
    """
    if vertical.layer_thickness.dtype is dtype:
        return vertical
    return ZLevelCoordinate(layer_thickness=vertical.layer_thickness.to(dtype=dtype))


def _vertical_coordinate(
    nz: int,
    depth: float,
    *,
    vertical_layer_thicknesses=None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> ZLevelCoordinate:
    if vertical_layer_thicknesses is None:
        return ZLevelCoordinate.uniform(nz, depth, device=device, dtype=dtype)
    vertical = ZLevelCoordinate.from_delR(
        vertical_layer_thicknesses,
        device=device,
        dtype=dtype,
    )
    if vertical.nz != nz:
        raise ValueError("vertical_layer_thicknesses length must equal nz")
    return vertical


def _placeholder_metrics(
    *,
    face_count: int,
    nx: int,
    ny: int,
    lx: float,
    ly: float,
    depth: float,
    device: str,
    dtype: torch.dtype,
) -> MetricSet:
    # Uniform geometry uses the same stagger-aware metric type as file grids.
    from .cubed_sphere_metric_builder import CubedSphereMetricSet

    dx = lx / nx
    dy = ly / ny
    center_shape = (face_count, ny, nx)
    u_shape = (face_count, ny, nx + 1)
    v_shape = (face_count, ny + 1, nx)
    corner_shape = (face_count, ny + 1, nx + 1)

    def full(shape, value):
        return torch.full(shape, value, dtype=dtype, device=device)

    geometry = {
        "xC": full(center_shape, 0.), "yC": full(center_shape, 0.),
        "dxF": full(center_shape, dx), "dyF": full(center_shape, dy),
        "rA": full(center_shape, dx * dy),
        "xG": full(corner_shape, 0.), "yG": full(corner_shape, 0.),
        "dxV": full(corner_shape, dx), "dyU": full(corner_shape, dy),
        "rAz": full(corner_shape, dx * dy),
        "dxC": full(u_shape, dx), "dyC": full(v_shape, dy),
        "rAw": full(u_shape, dx * dy), "rAs": full(v_shape, dx * dy),
        "dxG": full(v_shape, dx), "dyG": full(u_shape, dy),
        "angleCS": full(center_shape, 1.), "angleSN": full(center_shape, 0.),
    }
    return CubedSphereMetricSet(
        dx_scalar=float(dx),
        dy_scalar=float(dy),
        dx_center=geometry["dxF"],
        dy_center=geometry["dyF"],
        cell_area=geometry["rA"],
        cell_volume=full(center_shape, dx * dy * depth),
        depth_center=full(center_shape, depth),
        face_area_u=full(u_shape, depth * dy),
        face_area_v=full(v_shape, depth * dx),
        dxG_row=geometry["dxG"][:, :ny, :],
        dxC_row=geometry["dxC"][:, :, :nx],
        dyG_scalar_1d=float(dy),
        dxG_row_ext=geometry["dxG"],
        rAz_row=geometry["rAz"],
        face_geometry=geometry,
    )


def _placeholder_masks(
    *,
    face_count: int,
    nx: int,
    ny: int,
    nz: int,
    device: str,
    dtype: torch.dtype,
) -> MaskSet:
    center_shape = (face_count, ny, nx)
    u_shape = (face_count, ny, nx + 1)
    v_shape = (face_count, ny + 1, nx)
    from .cubed_sphere_bathy import CubedSphereMaskSet

    return CubedSphereMaskSet(
        center=torch.ones((face_count, nz, ny, nx), dtype=torch.bool, device=device),
        u=torch.ones((face_count, nz, ny, nx + 1), dtype=torch.bool, device=device),
        v=torch.ones((face_count, nz, ny + 1, nx), dtype=torch.bool, device=device),
        hFacC=torch.ones((face_count, nz, ny, nx), dtype=dtype, device=device),
        hFacW=torch.ones((face_count, nz, ny, nx + 1), dtype=dtype, device=device),
        hFacS=torch.ones((face_count, nz, ny + 1, nx), dtype=dtype, device=device),
        maskInC=torch.ones(center_shape, dtype=dtype, device=device),
        maskInW=torch.ones(u_shape, dtype=dtype, device=device),
        maskInS=torch.ones(v_shape, dtype=dtype, device=device),
    )


def _reconcile_cube_seam_hfac(grid: "CubedSphereGrid") -> "CubedSphereGrid":
    """Re-derive cube ``hFacW``/``hFacS`` so seam faces use the cross-seam ``min``.

    ``build_cubed_sphere_mask_set`` builds each face independently with
    ``boundary_topology=None``, so the four seam u/v-faces (west ``i=0``, east
    ``i=nx``, south ``j=0``, north ``j=ny``) fall back to the ``masks.py``
    else-branch local copy (``hFacW[...,0]=hFacC[...,0]`` etc.) and never see the
    cross-seam neighbour's depth.  At a "deep local column / shallow cross-seam
    neighbour" edge this over-estimates the open fraction — and therefore the
    cg2d barotropic depth ``H_u=(dz*hFacW).sum`` that every barotropic consumer
    reads — by up to ~30x, which seeds the seam free-surface instability.

    MITgcm (``ini_masks_etc.F:244-254`` + ``EXCH_UV``) defines
    ``hFacW(i,j,k)=min(hFacC(i-1,j,k),hFacC(i,j,k))`` with the neighbour supplied
    by the halo exchange.  We reproduce that exactly by rebuilding both faces
    from the cross-face halo-gathered ``hFacC``: interior faces are
    bitwise-unchanged (halo cells equal local cells there) and only the four seam
    edges pick up the cross-seam ``min``.  The reconstruction reads only the
    EDGE halo strips (interior rows/cols), never the valence-3 corner blocks
    (which ``_exchange_cubed_sphere_scalar`` now fills via its corner-fill
    post-pass), so it is corner-safe regardless of the corner values.  This
    keeps cg2d AND the transport-divergence / exactConserv / viscosity consumers
    (which all read the single ``grid.masks.hFacW``) mutually consistent — a
    cg2d-only seam fix desyncs them and blows up sooner.

    The cross-seam minimum recovers MIT ``aW2d`` to floating-point precision.
    """
    from dataclasses import replace

    from .exchange import HaloExtentV1, _exchange_cubed_sphere_scalar

    masks = grid.masks
    hFacC = masks.hFacC
    n = grid.nx
    # T-point (centre scalar) cross-face halo gather; hFac carries no sign.
    hc = _exchange_cubed_sphere_scalar(
        hFacC, grid=grid, location="T", extent=HaloExtentV1.uniform(1)
    )  # [6, nz, n+2, n+2]; interior at [1:n+1, 1:n+1], edge strips filled, corners filled (unread here).
    # hFacW[i] = min(hFacC(i-1), hFacC(i)); halo col i maps to hc col i, cell i to
    # col i+1.  i=0/i=nx now read the cross-seam neighbour strip instead of a copy.
    hFacW = torch.minimum(hc[:, :, 1 : n + 1, 0 : n + 1], hc[:, :, 1 : n + 1, 1 : n + 2])
    hFacS = torch.minimum(hc[:, :, 0 : n + 1, 1 : n + 1], hc[:, :, 1 : n + 2, 1 : n + 1])
    hFacW = hFacW.to(dtype=masks.hFacW.dtype)
    hFacS = hFacS.to(dtype=masks.hFacS.dtype)
    new_masks = replace(
        masks,
        hFacW=hFacW,
        hFacS=hFacS,
        u=hFacW > 0.0,
        v=hFacS > 0.0,
    )
    return replace(grid, masks=new_masks)


@dataclass(frozen=True)
class CubedSphereGrid:
    """Cubed-sphere `Grid` skeleton with step3 geometry placeholders.

    `tile_connectivity` and `cube_corner_indices` intentionally default to
    empty containers.  Step3 fills those tables from the cs32 grid fixture;
    until then, `face_seam_rotation` raises instead of returning a silent
    placeholder.
    """

    nx: int
    ny: int
    metrics: MetricSet
    masks: MaskSet
    vertical_coordinate: ZLevelCoordinate
    boundary_topology: BoundaryTopology = field(default_factory=_default_boundary_topology)
    face_count: int = _CUBED_SPHERE_FACE_COUNT
    tile_connectivity: Mapping[tuple[int, str], tuple[int, str, int]] = field(default_factory=dict)
    cube_corner_indices: tuple[tuple[int, int, int], ...] = ()
    distributed_neighbors: Mapping[str, object] = field(default_factory=dict)
    parallel_layout: object | None = None
    # Cube-only horizontal-advection topology DECLARATION.  This is the single
    # source of truth the GAD dispatch reads (``dynamics/gad_advection.py``); it
    # replaced the former runtime layout probe, which silently
    # fell back to six_face whenever any of its ~10 conditions missed.  ``None``
    # is not a usable value — it exists only because a dataclass field placed
    # after defaulted fields cannot itself be mandatory; ``__post_init__`` below
    # hard-rejects it, so construction without an explicit declaration raises.
    advection_topology: str | None = None
    # MIT SIZE.h block declaration; required for ``mit_12tile``, optional (and
    # normally absent) for ``six_face_approx``, which has no MIT counterpart.
    mit_tiling: MitTiling | None = None

    def __post_init__(self) -> None:
        # Match StructuredCGrid: guard consumed constants, not every memo key.
        object.__setattr__(self, "_runtime_cache", {})
        if self.face_count != _CUBED_SPHERE_FACE_COUNT:
            raise ValueError("CubedSphereGrid face_count is locked to 6")
        if self.nx < 1 or self.ny < 1:
            raise ValueError("CubedSphereGrid needs nx>=1 and ny>=1")
        self._validate_advection_topology()

    def _validate_advection_topology(self) -> None:
        """Fail-closed validation of the advection-topology declaration.

        Every rejection here raises: a cube grid that cannot prove it satisfies
        the declared topology must NOT silently degrade to ``six_face_approx``
        (a silent fallback would make a broken layout table indistinguishable
        from a deliberate six-face run).
        """
        if self.advection_topology is None:
            raise ValueError(
                "CubedSphereGrid requires explicit advection_topology "
                f"(one of {list(ADVECTION_TOPOLOGIES)}); got None. Cube grids "
                "declare their horizontal-advection topology at construction — "
                "there is no default."
            )
        if self.advection_topology not in ADVECTION_TOPOLOGIES:
            raise ValueError(
                "CubedSphereGrid advection_topology must be one of "
                f"{list(ADVECTION_TOPOLOGIES)}, got {self.advection_topology!r}"
            )
        if self.advection_topology != "mit_12tile":
            # six_face_approx: no MIT SIZE.h counterpart, so mit_tiling may be
            # None (we do not invent a tiling that no reference model defines).
            # A tiling that IS declared must still be geometrically true of this
            # grid, even though six_face never consumes it — an unchecked
            # declaration is indistinguishable from a wrong one.
            if self.mit_tiling is not None and (
                self.nx % self.mit_tiling.tile_nx != 0
                or self.ny % self.mit_tiling.tile_ny != 0
            ):
                raise ValueError(
                    f"CubedSphereGrid mit_tiling {self.mit_tiling!r} does not "
                    f"tile the {self.nx}x{self.ny} face exactly"
                )
            return

        if self.mit_tiling is None:
            raise ValueError(
                "CubedSphereGrid advection_topology='mit_12tile' requires "
                "mit_tiling (the MIT SIZE.h sNx/sNy/OLx block declaration)"
            )
        # Derivation check: the tile schedule computes every index from this
        # declaration and the face size (``CubeTileLayout``), so the layout is
        # admitted iff that derivation is consistent -- exact tiling, the MIT
        # multi-dim minimum overlap, a halo the face exchange can supply.  It
        # is derived HERE, at construction, so a declaration the schedule
        # cannot run surfaces before prep/IC are burned, not at the first GAD
        # call.  Only ``MitTiling(32, 16, 4)`` on the 32x32 face has been
        # audited against MIT dumps; any other admitted layout runs the same
        # transcription unaudited.
        try:
            CubeTileLayout.from_face(self.nx, self.ny, self.mit_tiling)
        except ValueError as exc:
            raise ValueError(
                "CubedSphereGrid advection_topology='mit_12tile' cannot derive "
                f"a tile layout: {exc}"
            ) from exc

        # Layout-data checks: the 12-tile schedule
        # reads the seam topology, the 24-entry connectivity table, the 8 cube
        # corners, and the per-face rA/dxG/dyG metrics.  A grid missing any of
        # them cannot run the declared path.
        if (
            self.boundary_topology.west_east != BoundaryType.SEAM
            or self.boundary_topology.south_north != BoundaryType.SEAM
        ):
            raise ValueError(
                "CubedSphereGrid advection_topology='mit_12tile' requires SEAM "
                "boundary_topology on both axes, got "
                f"west_east={self.boundary_topology.west_east}, "
                f"south_north={self.boundary_topology.south_north}"
            )
        connectivity = self.tile_connectivity or {}
        expected_keys = {
            (face, side)
            for face in range(_CUBED_SPHERE_FACE_COUNT)
            for side in _CUBE_SIDES
        }
        if len(connectivity) != len(expected_keys) or not expected_keys.issubset(
            connectivity
        ):
            raise ValueError(
                "CubedSphereGrid advection_topology='mit_12tile' requires a "
                f"complete {len(expected_keys)}-entry tile_connectivity table "
                f"(6 faces x {len(_CUBE_SIDES)} sides); got {len(connectivity)} entries"
            )
        if len(self.cube_corner_indices or ()) != 8:
            raise ValueError(
                "CubedSphereGrid advection_topology='mit_12tile' requires 8 "
                f"cube_corner_indices, got {len(self.cube_corner_indices or ())}"
            )
        face_geometry = getattr(self.metrics, "face_geometry", None) or {}
        missing = {"rA", "dxG", "dyG"}.difference(face_geometry)
        if missing:
            raise ValueError(
                "CubedSphereGrid advection_topology='mit_12tile' requires "
                f"metrics.face_geometry to carry {sorted({'rA', 'dxG', 'dyG'})}; "
                f"missing {sorted(missing)}"
            )

    @classmethod
    def from_cs_face_count(
        cls,
        *,
        face_count: int = _CUBED_SPHERE_FACE_COUNT,
        nx: int = 32,
        ny: int = 32,
        nz: int = 15,
        lx: float = 32.0,
        ly: float = 32.0,
        depth: float = 15.0,
        vertical_layer_thicknesses=None,
        boundary_topology: BoundaryTopology | None = None,
        advection_topology: str,
        mit_tiling: MitTiling | None = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> "CubedSphereGrid":
        """Build an all-wet, uniform synthetic cube with explicit seam topology.

        Metrics retain the file-grid stagger shapes and masks retain the depth
        dimension. Coordinates and rotation fields are uniform synthetic values;
        physical cube geometry is supplied by the file-based factories.
        """
        from .cubed_sphere_corner import build_cube_corner_indices
        from .cubed_sphere_metric_builder import build_tile_connectivity

        if face_count != _CUBED_SPHERE_FACE_COUNT:
            raise ValueError("CubedSphereGrid face_count is locked to 6")
        if nx < 1 or ny < 1:
            raise ValueError("CubedSphereGrid needs nx>=1 and ny>=1")
        if nz < 1:
            raise ValueError("CubedSphereGrid needs nz>=1")
        vertical = _vertical_coordinate(
            nz,
            depth,
            vertical_layer_thicknesses=vertical_layer_thicknesses,
            device=device,
            dtype=dtype,
        )
        return cls(
            nx=nx,
            ny=ny,
            metrics=_placeholder_metrics(
                face_count=face_count,
                nx=nx,
                ny=ny,
                lx=lx,
                ly=ly,
                depth=depth,
                device=device,
                dtype=dtype,
            ),
            masks=_placeholder_masks(
                face_count=face_count,
                nx=nx,
                ny=ny,
                nz=vertical.nz,
                device=device,
                dtype=dtype,
            ),
            vertical_coordinate=vertical,
            boundary_topology=boundary_topology or _default_boundary_topology(),
            face_count=face_count,
            tile_connectivity=build_tile_connectivity(),
            cube_corner_indices=tuple(
                (face, i * nx, j * ny)
                for face, i, j in build_cube_corner_indices(1)
            ),
            advection_topology=advection_topology,
            mit_tiling=mit_tiling,
        )

    @classmethod
    def from_csN_grid(
        cls,
        grid_dir,
        n: int,
        vertical_coord: ZLevelCoordinate | None = None,
        *,
        stem: str = "grid_cs32",
        bathy_path,
        advection_topology: str,
        mit_tiling: MitTiling | None = None,
        dtype: torch.dtype = torch.float64,
        hfac_min: float = 0.1,
        hfac_min_dr: float = 20.0,
    ) -> "CubedSphereGrid":
        """Build a cubed-sphere grid of any face size ``n`` from MITgcm files.

        ``n`` is the per-face cell count (32 for cs32, 192 for cs192).  The
        face geometry files are ``<stem>.face00{1..6}.bin`` (18 records of
        ``(n+1, n+1)`` big-endian float64).  ``bathy_path`` is required and must
        be the MIT compact bathymetry for this resolution (negative ocean); it
        is ingested via ``project_bathy_to_cubed_sphere``.  The metric builder,
        corner indices, and seam-hFac reconciliation are all resolution-generic,
        so the cs32 path (``from_cs32_grid``) is a thin wrapper that pins the
        cs32 defaults; nothing on the ``n=32`` path changes.

        ``advection_topology`` is mandatory: this factory serves every cube
        resolution.  ``mit_12tile`` needs a ``mit_tiling`` the face can carry
        (``CubeTileLayout.from_face``); only cs32's ``MitTiling(32, 16, 4)`` is
        audited against MIT, other layouts run the derived schedule unaudited.
        ``six_face_approx`` needs no tiling.
        """
        from .cubed_sphere_bathy import (
            build_cubed_sphere_mask_set,
            project_bathy_to_cubed_sphere,
        )
        from .cubed_sphere_corner import build_cube_corner_indices
        from .cubed_sphere_metric_builder import (
            build_face_metric_set,
            build_tile_connectivity,
        )
        from locomgcm.io.cs32x15_loader import load_face_geometry, load_tile_geometry

        if n < 1:
            raise ValueError(f"face size n must be positive, got {n}")
        # ``vertical_geom`` and the bathymetry below are geometry *inputs*: the
        # hFac closure test that decides wet/dry compares them with no
        # tolerance, so both stay float64 whatever the run dtype is (grid.py
        # ``_GEOMETRY_DTYPE_NOTE``).  Widening after the fact would not do --
        # the interface depths are a cumulative sum and the compact bathymetry
        # is read as float64, so a float32 twin has already lost the low bits
        # that decide which side of a threshold a column falls on.
        # The float64 invariant must hold for caller-supplied
        # coordinates too.  A from_delR(..., float32) coordinate has already
        # rounded per element, so widening here cannot recover the low bits —
        # refuse instead of silently rebuilding the topology leak this path
        # was fixed to close.
        if (
            vertical_coord is not None
            and vertical_coord.layer_thickness.dtype is not _GEOMETRY_DTYPE
        ):
            raise ValueError(
                "from_csN_grid requires a float64 vertical coordinate: the hFac "
                "closure test that decides wet/dry has no tolerance, and a "
                f"{vertical_coord.layer_thickness.dtype} coordinate has already "
                "lost the low bits that place a column on one side of a "
                "threshold; build it with dtype=torch.float64 (the run-dtype "
                "narrowing happens after the masks are settled)"
            )
        vertical_geom = vertical_coord or ZLevelCoordinate.uniform(
            15, 15.0, dtype=_GEOMETRY_DTYPE
        )
        vertical_device = vertical_geom.interface_depth.device
        # ``stem`` is MITgcm's horizGridFile: empty selects the 16-record
        # tileNNN.mitgrid files with computed angles (ini_curvilinear_grid.F:278-284).
        raw_geometry = (
            load_tile_geometry(grid_dir, n)
            if stem == ""
            else load_face_geometry(grid_dir, n, stem=stem)
        )
        geometry_dict = {
            name: tensor.to(dtype=dtype, device=vertical_device)
            for name, tensor in raw_geometry.items()
        }
        metrics = build_face_metric_set(geometry_dict, vertical_geom)
        bathy_faces = project_bathy_to_cubed_sphere(
            bathy_path, n=n, dtype=_GEOMETRY_DTYPE
        ).to(device=vertical_device)
        masks = build_cubed_sphere_mask_set(
            bathy_faces,
            vertical_geom,
            hfac_min=hfac_min,
            hfac_min_dr=hfac_min_dr,
        )
        # The masks are settled; the stored coordinate is storage, so it carries
        # the run dtype like every other tensor the grid hands to the steppers.
        vertical = _run_dtype_vertical_coordinate(vertical_geom, dtype)

        grid = cls(
            nx=n,
            ny=n,
            metrics=metrics,
            masks=masks,
            vertical_coordinate=vertical,
            boundary_topology=_default_boundary_topology(),
            face_count=_CUBED_SPHERE_FACE_COUNT,
            tile_connectivity=build_tile_connectivity(),
            cube_corner_indices=build_cube_corner_indices(n),
            advection_topology=advection_topology,
            mit_tiling=mit_tiling,
        )
        # Per-face mask builder leaves seam u/v-faces at a local hFacC copy; the
        # cross-face exchange (needs the assembled grid) reconciles them to the
        # MITgcm cross-seam min so cg2d and every barotropic consumer stay
        # consistent.  See _reconcile_cube_seam_hfac.
        return _reconcile_cube_seam_hfac(grid)

    @classmethod
    def from_cs32_grid(
        cls,
        grid_dir,
        vertical_coord: ZLevelCoordinate | None = None,
        *,
        bathy_path=None,
        advection_topology: str = "mit_12tile",
        mit_tiling: MitTiling | None = None,
        dtype: torch.dtype = torch.float64,
        hfac_min: float = 0.1,
        hfac_min_dr: float = 20.0,
    ) -> "CubedSphereGrid":
        """Build a cs32 cubed-sphere grid from MITgcm face geometry files.

        Thin wrapper over ``from_csN_grid`` that pins the cs32 defaults
        (``n=32``, ``stem="grid_cs32"``, and the cs32x15 bathymetry fallback).
        Byte-for-byte identical to the pre-N-generic cs32 path.

        This loader IS the audited MIT cs32 layout loader, so the ``mit_12tile``
        declaration and its SIZE.h tiling live here, at the same address as the
        construction they describe — not as a library-wide soft default.  A
        caller may override ``advection_topology`` (e.g. to build a
        ``six_face_approx`` reference from the same fixture); overriding to
        ``six_face_approx`` also drops the implied tiling unless one is passed.
        When ``bathy_path`` is omitted, ``LOCOMGCM_CS32_INPUT_DIR`` can explicitly
        select the directory containing ``bathy_Hmin50.bin``. Without that
        setting, configure cs32_input_dir through LOCOMGCM_PATHS; no checkout fallback is used.
        """
        if mit_tiling is None and advection_topology == "mit_12tile":
            # MIT verification/global_ocean.cs32x15 SIZE.h: sNx=32, sNy=16, OLx=OLy=4.
            mit_tiling = MitTiling(32, 16, 4)

        if bathy_path is None:
            bathy_path = resolve("cs32_input_dir") / "bathy_Hmin50.bin"
        return cls.from_csN_grid(
            grid_dir,
            32,
            vertical_coord,
            stem="grid_cs32",
            bathy_path=bathy_path,
            advection_topology=advection_topology,
            mit_tiling=mit_tiling,
            dtype=dtype,
            hfac_min=hfac_min,
            hfac_min_dr=hfac_min_dr,
        )

    def to_runtime(
        self,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> "CubedSphereGrid":
        """Return a copy with every tensor field cast to ``device`` / ``dtype``.

        Mirrors the device/dtype migration ``model._grid_to_runtime`` performs
        for ``StructuredCGrid``, but preserves the cube identity
        (``face_count=6``, ``tile_connectivity``, ``cube_corner_indices``) instead
        of rebuilding as a single patch.  ``bool`` masks keep their dtype and are
        only moved to ``device``; floating tensors are cast to ``dtype``.  The
        per-face ``CubedSphereMetricSet.face_geometry`` mapping is carried through
        cast so the 2-D accessors (``dyG_for`` / ``dyC_for``) stay face-aware.
        """
        from dataclasses import fields, replace

        def _cast_tensor(t: torch.Tensor | None) -> torch.Tensor | None:
            if t is None:
                return None
            if t.is_floating_point():
                return t.to(device=device, dtype=dtype)
            return t.to(device=device)

        metric_updates: dict[str, torch.Tensor] = {}
        for f in fields(self.metrics):
            if not f.init:
                continue
            value = getattr(self.metrics, f.name)
            if isinstance(value, torch.Tensor):
                metric_updates[f.name] = _cast_tensor(value)
            elif f.name == "face_geometry":
                metric_updates[f.name] = {
                    k: _cast_tensor(v) for k, v in value.items()
                }
        metrics = replace(self.metrics, **metric_updates)

        mask_updates = {
            f.name: _cast_tensor(getattr(self.masks, f.name))
            for f in fields(self.masks)
            if f.init and isinstance(getattr(self.masks, f.name), torch.Tensor)
        }
        masks = replace(self.masks, **mask_updates)

        vertical = ZLevelCoordinate(
            layer_thickness=self.vertical_coordinate.layer_thickness.to(
                device=device, dtype=dtype
            ),
        )

        return CubedSphereGrid(
            nx=self.nx,
            ny=self.ny,
            metrics=metrics,
            masks=masks,
            vertical_coordinate=vertical,
            boundary_topology=self.boundary_topology,
            face_count=self.face_count,
            tile_connectivity=self.tile_connectivity,
            cube_corner_indices=self.cube_corner_indices,
            distributed_neighbors=self.distributed_neighbors,
            parallel_layout=self.parallel_layout,
            # Hand-written construction (not ``replace``): the topology
            # declaration MUST be carried explicitly or the device migration
            # would drop it and re-raise on the mandatory-field check.
            advection_topology=self.advection_topology,
            mit_tiling=self.mit_tiling,
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
    def center_shape(self) -> tuple[int, ...]:
        if self.face_count == 1:
            return (self.ny, self.nx)
        return (self.face_count, self.ny, self.nx)

    @property
    def u_shape(self) -> tuple[int, ...]:
        if self.face_count == 1:
            return (self.ny, self.nx + 1)
        return (self.face_count, self.ny, self.nx + 1)

    @property
    def v_shape(self) -> tuple[int, ...]:
        if self.face_count == 1:
            return (self.ny + 1, self.nx)
        return (self.face_count, self.ny + 1, self.nx)

    @property
    def tracer_shape(self) -> tuple[int, ...]:
        if self.face_count == 1:
            return (self.nz, self.ny, self.nx)
        return (self.face_count, self.nz, self.ny, self.nx)

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
        """Device-resident, memoized ``hFacC`` (mirrors :meth:`hfac_w_for`).

        The cube operators (vector-invariant momentum, horizontal viscosity,
        vertical shear) read ``masks.hFacC`` — a CPU-resident tensor — and
        ``.to(device=...)`` it on *every* step (~285 MB per cube call at
        cs384).  Memoizing the device copy once per (dtype, device) eliminates
        the repeated host→device transfer; the cached tensor is bit-identical
        to the per-step ``.to()`` (float64 H2D is exact).
        """
        return self._cached_runtime_tensor(
            "hfac_c",
            dtype=dtype,
            device=device,
            factory=lambda: self._build_hfac_c(dtype=dtype, device=device),
        )

    def face_geometry_for(
        self, name: str, dtype: torch.dtype, device: torch.device | str
    ) -> torch.Tensor:
        """Device-resident, memoized ``metrics.face_geometry[name]``.

        ``_curl_cube`` (``operators/c_grid.py``) reads per-face metric arrays
        (``dxC``/``dyC``/``rAz``) from the CPU-resident ``face_geometry`` dict
        and ``.to(device=...)`` them every call.  Memoizing the device copy
        once per (name, dtype, device) removes the per-step transfer; the cached
        tensor is bit-identical to the per-step ``.to()``.
        """
        return self._cached_runtime_tensor(
            f"face_geometry::{name}",
            dtype=dtype,
            device=device,
            factory=lambda: self.metrics.face_geometry[name].to(dtype=dtype, device=device),
        )

    def _build_hfac_c(self, *, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        if self.masks.hFacC is not None:
            hfac_c = self.masks.hFacC.to(dtype=dtype, device=device)
            if self.face_count == 1 and hfac_c.ndim == 4 and hfac_c.shape[0] == 1:
                return hfac_c.squeeze(0)
            return hfac_c
        if self.face_count == 1:
            return torch.ones(
                (self.nz, self.ny, self.nx),
                dtype=dtype,
                device=device,
            )
        return torch.ones(
            (self.face_count, self.nz, self.ny, self.nx),
            dtype=dtype,
            device=device,
        )

    def _build_hfac_w(self, *, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        if self.masks.hFacW is not None:
            hfac_w = self.masks.hFacW.to(dtype=dtype, device=device)
            if self.face_count == 1 and hfac_w.ndim == 4 and hfac_w.shape[0] == 1:
                return hfac_w.squeeze(0)
            return hfac_w
        if self.face_count == 1:
            return torch.ones(
                (self.nz, self.ny, self.nx + 1),
                dtype=dtype,
                device=device,
            )
        return torch.ones(
            (self.face_count, self.nz, self.ny, self.nx + 1),
            dtype=dtype,
            device=device,
        )

    def _build_hfac_s(self, *, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        if self.masks.hFacS is not None:
            hfac_s = self.masks.hFacS.to(dtype=dtype, device=device)
            if self.face_count == 1 and hfac_s.ndim == 4 and hfac_s.shape[0] == 1:
                return hfac_s.squeeze(0)
            return hfac_s
        if self.face_count == 1:
            return torch.ones(
                (self.nz, self.ny + 1, self.nx),
                dtype=dtype,
                device=device,
            )
        return torch.ones(
            (self.face_count, self.nz, self.ny + 1, self.nx),
            dtype=dtype,
            device=device,
        )

    def _boundary_for_axis(self, axis: Literal["x", "y"]) -> BoundaryType:
        if axis == "x":
            return self.boundary_topology.west_east
        if axis == "y":
            return self.boundary_topology.south_north
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")

    def edge_face_couples(self, axis: Literal["x", "y"]) -> bool:
        boundary = self._boundary_for_axis(axis)
        return boundary in (BoundaryType.PERIODIC, BoundaryType.SEAM)

    def physical_wall(self, axis: Literal["x", "y"]) -> bool:
        return self._boundary_for_axis(axis) == BoundaryType.WALL

    def open_boundary(self, axis: Literal["x", "y"]) -> bool:
        return self._boundary_for_axis(axis) == BoundaryType.OPEN

    def w_face_thickness_edge_couples(self, axis: Literal["x", "y"]) -> bool:
        # Every axis but a wall puts a real neighbour column in the ghost
        # slot, so the domain-edge W-cell face pairs the two end columns.
        # See ``StructuredCGrid.w_face_thickness_edge_couples``.
        boundary = self._boundary_for_axis(axis)
        return boundary in (
            BoundaryType.OPEN,
            BoundaryType.PERIODIC,
            BoundaryType.SEAM,
        )

    def face_seam_rotation(self, face: int, side: Literal["west", "east", "south", "north"]) -> int:
        """Return the seam rotation from `tile_connectivity`."""
        if not self.tile_connectivity:
            raise NotImplementedError(
                "CubedSphereGrid has no tile_connectivity yet "
                f"(face={face}, side={side}); call from_cs32_grid() for real values"
            )
        return self.tile_connectivity[(face, side)][2]


__all__ = ["ADVECTION_TOPOLOGIES", "CubedSphereGrid", "MitTiling"]
