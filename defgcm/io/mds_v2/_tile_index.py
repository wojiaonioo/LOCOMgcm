"""Discover multi-tile layout from a directory of MITgcm MDS output.

Algorithm:
  1. List all `<field>.<step>.<IPx:03d>.<IPy:03d>.meta` files for a given field
  2. Parse one to get (sNx, sNy) from local_end - local_start + 1
  3. Range over IPx/IPy to determine nPx/nPy
  4. Construct TileIndex
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ._meta_parser import parse_meta


# Regex for canonical MITgcm output filenames:
#   <fieldName>.<NNNNNNNNNN>.<IPx:03d>.<IPy:03d>.<data|meta>
# OR (single-tile time-invariant): <fieldName>.<IPx:03d>.<IPy:03d>.<data|meta>
# OR (single-tile time-variant, no tile suffix): <fieldName>.<NNNNNNNNNN>.<data|meta>
_TILE_RE_VARIANTS = [
    # (field, step, IPx, IPy)
    re.compile(r"^(?P<field>\w+)\.(?P<step>\d{10})\.(?P<ipx>\d{3})\.(?P<ipy>\d{3})\.meta$"),
    # Time-invariant tile-suffixed (e.g. hFacC.001.001.meta)
    re.compile(r"^(?P<field>\w+)\.(?P<ipx>\d{3})\.(?P<ipy>\d{3})\.meta$"),
    # Single-tile time-variant (no IPx.IPy suffix)
    re.compile(r"^(?P<field>\w+)\.(?P<step>\d{10})\.meta$"),
]


@dataclass(frozen=True)
class TileIndex:
    n_px: int
    n_py: int
    s_nx: int
    s_ny: int
    olx: int  # halo cells X (not always knowable from .meta; default 0 if absent)
    oly: int
    nx_global: int
    ny_global: int
    nz: int

    @property
    def is_single_tile(self) -> bool:
        return self.n_px == 1 and self.n_py == 1


def parse_filename(path: Path) -> dict[str, int | str] | None:
    """Parse a tile filename into components, or None if no pattern matches."""
    name = path.name
    for regex in _TILE_RE_VARIANTS:
        m = regex.match(name)
        if m:
            d = m.groupdict()
            if "step" in d:
                d["step"] = int(d["step"])
            if "ipx" in d:
                d["ipx"] = int(d["ipx"])
                d["ipy"] = int(d["ipy"])
            else:
                d["ipx"] = 1
                d["ipy"] = 1
            return d
    return None


def discover_tile_index(
    run_dir: Path,
    field_stem: str = "U",
    step: int | None = None,
) -> TileIndex:
    """Scan `<field_stem>.*.meta` files in `run_dir` to discover decomposition.

    If `step` is given, only files for that step are considered (time-variant
    output). Otherwise the first available step is used.
    """
    metas: list[tuple[Path, dict[str, int | str]]] = []
    for p in run_dir.glob(f"{field_stem}.*.meta"):
        parsed = parse_filename(p)
        if parsed is None:
            continue
        if step is not None and parsed.get("step", step) != step:
            continue
        metas.append((p, parsed))

    if not metas:
        raise ValueError(
            f"discover_tile_index: no '{field_stem}.*.meta' files in {run_dir}"
            f"{' for step ' + str(step) if step else ''}"
        )

    # Get the unique (IPx, IPy) pairs
    tile_keys = {(m[1]["ipx"], m[1]["ipy"]) for m in metas}
    n_px = max(k[0] for k in tile_keys)
    n_py = max(k[1] for k in tile_keys)

    # Parse one .meta to discover tile size + global size
    sample_path, _ = metas[0]
    parsed = parse_meta(sample_path)
    dim_list = parsed["dimList"]  # list of (gsize, lstart, lend)

    # MITgcm output layout (3D): dim 0 = X, dim 1 = Y, dim 2 = Z
    if len(dim_list) >= 3:
        x_gsize, x_lstart, x_lend = dim_list[0]
        y_gsize, y_lstart, y_lend = dim_list[1]
        z_gsize, _, _ = dim_list[2]
        nz = z_gsize
    elif len(dim_list) == 2:
        x_gsize, x_lstart, x_lend = dim_list[0]
        y_gsize, y_lstart, y_lend = dim_list[1]
        nz = 1
    else:
        raise ValueError(f"discover_tile_index: unexpected nDims={len(dim_list)} in {sample_path}")

    s_nx = x_lend - x_lstart + 1
    s_ny = y_lend - y_lstart + 1

    # OL not available in .meta typically; default 0 (caller may override)
    return TileIndex(
        n_px=n_px, n_py=n_py,
        s_nx=s_nx, s_ny=s_ny,
        olx=0, oly=0,
        nx_global=x_gsize,
        ny_global=y_gsize,
        nz=nz,
    )
