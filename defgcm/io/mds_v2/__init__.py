"""MITgcm MDS multi-tile reader."""

from ._meta_parser import parse_meta
from ._reader import MdsField, read_multi_tile_field
from ._tile_index import TileIndex, discover_tile_index, parse_filename

__all__ = [
    "MdsField",
    "TileIndex",
    "discover_tile_index",
    "parse_filename",
    "parse_meta",
    "read_multi_tile_field",
]
