"""Universal grid factory for yaml-driven runner cases.

Dispatches on ``run_cfg.grid.kind``: ``latlon`` (default) builds a
``StructuredCGrid`` from a bathymetry tensor; ``cubed_sphere`` builds a
``CubedSphereGrid`` from the MITgcm face-geometry / bathymetry files named by
the ``cube_*`` grid fields (see ``_build_cube_grid_from_run_cfg``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch

from defgcm.geometry import BoundaryTopology, BoundaryType, StructuredCGrid

if TYPE_CHECKING:  # annotation-only; the cube grid is imported lazily at call time
    from defgcm.geometry import CubedSphereGrid


DEG_TO_M_X = 111_320.0
DEG_TO_M_Y = 110_574.0


def _extras(run_cfg: Any) -> Mapping[str, Any]:
    return getattr(getattr(run_cfg, "case", None), "extras", {}) or {}


def _extra(run_cfg: Any, key: str, default: Any = None) -> Any:
    extras = _extras(run_cfg)
    for candidate in (key, key.replace("_", "-")):
        if candidate in extras and extras[candidate] is not None:
            return extras[candidate]
    return default


def _boundary_type(value: Any) -> BoundaryType:
    try:
        return BoundaryType(str(value))
    except ValueError as exc:
        raise ValueError(
            "boundary type must be one of "
            f"{[item.value for item in BoundaryType]}, got {value!r}"
        ) from exc


def obcs_fix_topo(run_cfg: Any) -> bool:
    """Resolve MITgcm ``OBCSfixTopo`` from ``boundary.obcs_fix_topo``.

    Default False = MITgcm's own default (``pkg/obcs/obcs_readparms.F:291``);
    a case opts in from its yaml, exactly as it would have to write
    ``OBCSfixTopo=.TRUE.,`` into ``data.obcs``.
    """
    return bool(getattr(getattr(run_cfg, "boundary", None), "obcs_fix_topo", False))


def _boundary_topology(run_cfg: Any) -> BoundaryTopology:
    west_east = _extra(run_cfg, "west_east_boundary", None)
    south_north = _extra(run_cfg, "south_north_boundary", None)
    if west_east is not None or south_north is not None:
        fallback = getattr(getattr(run_cfg, "boundary", None), "type", "periodic")
        return BoundaryTopology(
            west_east=_boundary_type(west_east if west_east is not None else fallback),
            south_north=_boundary_type(
                south_north if south_north is not None else fallback
            ),
        )

    boundary_cfg = getattr(run_cfg, "boundary", None)
    boundary = _boundary_type(getattr(boundary_cfg, "type", "periodic"))
    return BoundaryTopology(west_east=boundary, south_north=boundary)


def _extent(run_cfg: Any) -> tuple[float, float]:
    grid = run_cfg.grid
    lx = _extra(run_cfg, "lx", None)
    ly = _extra(run_cfg, "ly", None)
    if lx is None:
        lx = (
            int(grid.nx)
            * float(grid.resolution_deg)
            * DEG_TO_M_X
        )
    if ly is None:
        ly = (
            int(grid.ny)
            * float(grid.resolution_deg)
            * DEG_TO_M_Y
        )
    return float(lx), float(ly)


def build_grid_from_run_cfg(
    run_cfg: Any,
    *,
    bathymetry: torch.Tensor | None = None,
    del_x: Any = None,
    del_y: Any = None,
    device: str | torch.device,
    dtype: torch.dtype = torch.float64,
    hfac_min: float = 0.2,
    hfac_min_dz: float = 0.0,
) -> "StructuredCGrid | CubedSphereGrid":
    """Build the grid from ``RunConfig``, dispatching on ``grid.kind``.

    ``grid.kind='latlon'`` (default) builds a ``StructuredCGrid`` from the
    ``bathymetry`` tensor (required, shape ``[ny, nx]``, metres, LOCOMgcm
    sign: positive = ocean depth, zero or negative = land; a MITgcm
    ``bathyFile`` is negative in the ocean and must be negated first).
    ``StructuredCGrid.with_bathymetry`` rejects a tensor that looks like CF
    elevation and clamps land to 0 through ``obcs_fix_topo_and_clamp``.
    Dimensions come from ``run_cfg.grid``.
    Horizontal extent prefers ``case.extras.lx`` / ``case.extras.ly`` when
    present, otherwise it is estimated from ``grid.nx`` / ``grid.ny`` and
    ``resolution_deg``. Boundary topology comes from ``run_cfg.boundary`` unless
    ``case.extras`` supplies ``west_east_boundary`` or ``south_north_boundary``.

    ``grid.kind='cubed_sphere'`` builds a ``CubedSphereGrid`` from the MITgcm
    face-geometry / bathymetry files (``bathymetry`` is ignored on this path);
    see ``_build_cube_grid_from_run_cfg``.
    """

    if str(getattr(run_cfg.grid, "kind", "latlon")) == "cubed_sphere":
        return _build_cube_grid_from_run_cfg(
            run_cfg,
            device=device,
            dtype=dtype,
            hfac_min=hfac_min,
            hfac_min_dz=hfac_min_dz,
        )

    grid = run_cfg.grid
    if bathymetry is None:
        raise ValueError(
            "build_grid_from_run_cfg requires a bathymetry tensor for the "
            "latlon grid path (grid.kind='latlon')"
        )
    # float64 on purpose: the grid derives its land mask from an untoleranced
    # comparison, so narrowing the bathymetry here would make the topology
    # depend on the run precision (grid.py ``_GEOMETRY_DTYPE_NOTE``).
    bathy = torch.as_tensor(bathymetry, dtype=torch.float64, device=device)
    if tuple(bathy.shape) != (int(grid.ny), int(grid.nx)):
        raise ValueError(
            "bathymetry shape must be "
            f"({int(grid.ny)}, {int(grid.nx)}), got {tuple(bathy.shape)}"
        )
    vector_spacing = del_x is not None or del_y is not None
    if vector_spacing and (del_x is None or del_y is None):
        raise ValueError(
            "Cartesian vector spacing requires both del_x and del_y; "
            "resolution_deg fallback is disabled for file spacing"
        )
    if vector_spacing:
        lx = ly = None
    else:
        lx, ly = _extent(run_cfg)
    return StructuredCGrid.with_bathymetry(
        nx=int(grid.nx),
        ny=int(grid.ny),
        nz=int(grid.nz),
        lx=lx,
        ly=ly,
        del_x=del_x,
        del_y=del_y,
        depth=float(grid.depth_m),
        bathymetry=bathy,
        vertical_layer_thicknesses=tuple(
            float(v) for v in grid.vertical_layer_thicknesses
        ),
        vertical_center_spacing=tuple(grid.vertical_center_spacing),
        boundary_topology=_boundary_topology(run_cfg),
        hfac_min=hfac_min,
        hfac_min_dz=hfac_min_dz,
        obcs_fix_topo=obcs_fix_topo(run_cfg),
        device=str(device),
        dtype=dtype,
    )


def _build_cube_grid_from_run_cfg(
    run_cfg: Any,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
    hfac_min: float,
    hfac_min_dz: float,
) -> "CubedSphereGrid":
    """Build a ``CubedSphereGrid`` from a cube ``RunConfig`` (grid.kind='cubed_sphere').

    Mirrors the reference cs768 construction:
    the vertical coordinate is derived from ``grid.vertical_layer_thicknesses``
    and the horizontal face geometry / bathymetry from the ``cube_*`` grid
    fields (``cube_grid_dir`` / ``cube_grid_stem`` / ``cube_face_n`` /
    ``cube_bathy_path``). Those path fields are used verbatim (relative paths
    resolve against the current working directory, as on the latlon data path);
    no ``.resolve()`` is applied.

    The horizontal-advection topology is likewise declared in the yaml
    (``cube_advection_topology`` + the optional ``cube_tile_*`` SIZE.h triple)
    and passed straight through to the grid, which re-validates it. Nothing here
    infers a topology from the resolution.

    hFac naming note: this factory's ``hfac_min`` / ``hfac_min_dz`` map onto
    ``from_csN_grid``'s ``hfac_min`` / ``hfac_min_dr``. ``dz`` (lat-lon line) and
    ``dr`` (cube line) are two names for the same quantity — the minimum
    partial-cell thickness in metres — so the value is passed straight through.

    Device note: ``from_csN_grid`` takes no ``device`` argument; it builds on the
    vertical coordinate's device (CPU here, matching ``case_cs768.build``, which
    never moves the grid off its construction device). Moving the grid onto the
    compute device is the runtime's responsibility. ``device`` is accepted here
    for signature parity with the latlon path and is intentionally unused.
    """
    from defgcm.geometry import CubedSphereGrid, MitTiling
    from defgcm.geometry.vertical_coordinates import ZLevelCoordinate

    grid = run_cfg.grid
    if grid.vertical_center_spacing:
        raise ValueError(
            "grid.vertical_center_spacing is not carried on the cubed sphere; "
            "the cube's vertical coordinate puts the centres midway between interfaces"
        )
    # float64 on purpose, same rule as the latlon bathymetry above: the cube
    # masks derive their wet/dry topology from an untoleranced hFac closure test
    # against these layer interfaces, and the interfaces are a cumulative sum,
    # so narrowing the coordinate here would move the topology and could not be
    # undone downstream (grid.py ``_GEOMETRY_DTYPE_NOTE``).  ``from_csN_grid``
    # narrows the *stored* coordinate to ``dtype`` once the masks are built.
    vert = ZLevelCoordinate.from_delR(
        [float(v) for v in grid.vertical_layer_thicknesses], dtype=torch.float64
    )
    # Advection topology + MIT tiling come from the yaml (validated by
    # ``GridConfig.__post_init__``: the topology is mandatory on this path and
    # ``mit_12tile`` additionally requires all three tile fields).  The tiling is
    # assembled only when all three are present, so a ``six_face_approx`` case
    # that omits them (no MIT counterpart) passes ``mit_tiling=None`` rather than
    # a half-filled record.
    tile_fields = (grid.cube_tile_nx, grid.cube_tile_ny, grid.cube_tile_halo)
    mit_tiling = (
        MitTiling(int(tile_fields[0]), int(tile_fields[1]), int(tile_fields[2]))
        if all(f is not None for f in tile_fields)
        else None
    )
    return CubedSphereGrid.from_csN_grid(
        str(grid.cube_grid_dir),
        int(grid.cube_face_n),
        vert,
        stem=str(grid.cube_grid_stem),
        bathy_path=str(grid.cube_bathy_path),
        advection_topology=str(grid.cube_advection_topology),
        mit_tiling=mit_tiling,
        dtype=dtype,
        hfac_min=hfac_min,
        hfac_min_dr=hfac_min_dz,
    )
