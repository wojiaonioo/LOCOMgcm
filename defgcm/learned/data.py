"""Reader of the MITgcm truth runs used to train the learned closures.

The truth runs write MDS files: a binary ``<prefix>.<iter>.data`` and a text
``.meta`` naming the fields (``fldList``), the record count (``nrecords``), the
precision (``dataprec``) and the grid (``dimList``).  Files are big-endian; both
the global name ``pickup.0000000003.data`` and the single-tile name
``pickup.0000000003.001.001.data`` are read, one tile covering the whole grid.

Time levels: a pickup at iteration n holds the state at the end of step n.  A
``kpp3d``/``kpp2d`` snapshot at iteration n holds the KPP fields MITgcm computed
from that same state, the ones that mix during step n+1: snapshots are written
with the suffix myIter-1 (diagnostics_write.F:62-71).  Measured on case_smoke:
STATEKPP of pickup n reproduces KPPdbloc/KPPdbsfc of kpp n exactly.
A snapshot list without
``timePhase`` is sampled at phase |frequency|/2 (diagnostics_readparms.F:291-292),
so when the output interval spans several steps the kpp and pickup iterations
differ: case_1 has pickups at 3, 6, 9, ... and kpp snapshots at 1, 4, 7, ....

Fields are returned as MITgcm stores them, in MITgcm sign conventions, as
float64 tensors on the CPU: ``Uvel`` is on the west face of each cell (nx faces,
no far face), and diagnostics keep their ``missingValue`` where MITgcm wrote it.
"""
from __future__ import annotations

import bisect
import math
import re
from pathlib import Path

import numpy as np
import torch

_ITER_FILE = re.compile(r"^(?P<prefix>\w+)\.(?P<iter>\d{10})(?:\.\d{3}\.\d{3})?\.data$")
_META_ENTRY = re.compile(r"(\w+)\s*=\s*[\[{](.*?)[\]}]\s*;", re.DOTALL)
_BIG_ENDIAN = {"float64": ">f8", "float32": ">f4"}
_PREFIXES = ("pickup", "kpp3d", "kpp2d")
# pickup field name -> key returned by MitTruth.pickup
_PICKUP_KEYS = {
    "Uvel": "u",
    "Vvel": "v",
    "Theta": "temp",
    "Salt": "salt",
    "GuNm1": "gu_nm1",
    "GvNm1": "gv_nm1",
    "EtaN": "eta",
}
# write_pickup.F:328-332 writes every 3-D field first and starts the 2-D block
# with EtaN, so EtaN's position in fldList is the number of 3-D fields.
_PICKUP_FIRST_2D = "EtaN"


def _parse_meta(path: Path) -> dict[str, list]:
    meta: dict[str, list] = {}
    for key, body in _META_ENTRY.findall(path.read_text()):
        strings = re.findall(r"'([^']*)'", body)
        if strings:
            meta[key] = [s.strip() for s in strings]
        else:
            meta[key] = [
                float(tok) if any(c in tok for c in ".eE") else int(tok)
                for tok in body.replace(",", " ").split()
            ]
    return meta


class MitTruth:
    """Pickups and KPP diagnostics of one MITgcm run directory."""

    def __init__(self, run_dir, nx: int, ny: int, nz: int):
        self.run_dir = Path(run_dir)
        if not self.run_dir.is_dir():
            raise FileNotFoundError(f"MIT run directory not found: {self.run_dir}")
        self.nx, self.ny, self.nz = int(nx), int(ny), int(nz)
        self._files: dict[str, dict[int, Path]] = {prefix: {} for prefix in _PREFIXES}
        for path in self.run_dir.iterdir():
            match = _ITER_FILE.match(path.name)
            if match is None or match["prefix"] not in self._files:
                continue
            files = self._files[match["prefix"]]
            it = int(match["iter"])
            if it in files:
                raise ValueError(f"two files for {match['prefix']} iteration {it}: "
                                 f"{files[it].name}, {path.name}")
            files[it] = path
        self._pickup_iters = sorted(self._files["pickup"])

    def iters(self, prefix: str = "pickup") -> list[int]:
        """Iterations with a ``prefix`` file (pickup, kpp3d or kpp2d), ascending."""
        return sorted(self._files[prefix])

    def window(self, it0: int, n_out: int) -> list[int]:
        """``n_out`` consecutive pickup iterations starting with ``it0`` itself.

        Raises if ``it0`` has no pickup, if fewer than ``n_out`` pickups follow
        from it, or if their spacing is not uniform (a missing pickup).
        """
        if n_out < 1:
            raise ValueError(f"n_out must be >= 1, got {n_out}")
        start = bisect.bisect_left(self._pickup_iters, it0)
        if start == len(self._pickup_iters) or self._pickup_iters[start] != it0:
            raise KeyError(f"no pickup at iteration {it0} in {self.run_dir}")
        its = self._pickup_iters[start:start + n_out]
        if len(its) < n_out:
            raise ValueError(f"only {len(its)} pickups from iteration {it0}, need {n_out}")
        steps = {b - a for a, b in zip(its, its[1:])}
        if len(steps) > 1:
            raise ValueError(f"pickups from iteration {it0} are not evenly spaced: {its}")
        return its

    def pickup(self, it: int) -> dict[str, torch.Tensor | None]:
        """Model state of a pickup: keys u, v, temp, salt, gu_nm1, gv_nm1 (nz, ny, nx)
        and eta (ny, nx); a field the pickup does not contain is None."""
        fields = self.fields("pickup", it)
        return {key: fields.get(name) for name, key in _PICKUP_KEYS.items()}

    def kpp(self, it: int) -> dict[str, torch.Tensor]:
        """KPP diagnostics of iteration ``it`` under their MITgcm names: the kpp3d
        fields (KPPviscA, KPPdiffT, KPPdiffS, KPPghatK, KPPdbloc, KPPdbsfc,
        KPPdVsq, KPPshsq, KPPRi; (nz, ny, nx)) and the kpp2d fields (KPPhbl,
        KPPbfsfc, KPPfrac; (ny, nx))."""
        out = self.fields("kpp3d", it)
        out.update(self.fields("kpp2d", it))
        return out

    def fields(self, prefix: str, it: int) -> dict[str, torch.Tensor]:
        """Every field of the ``prefix`` file at iteration ``it``, keyed by its
        fldList name (for example ``PhiHyd`` of a pickup)."""
        path = self._files[prefix].get(int(it))
        if path is None:
            raise KeyError(f"no {prefix} file at iteration {it} in {self.run_dir}")
        meta = _parse_meta(path.with_suffix(".meta"))
        names = meta["fldList"]
        n_dims = meta["nDims"][0]
        n_records = meta["nrecords"][0]
        precision = meta["dataprec"][0]
        if precision not in _BIG_ENDIAN:
            raise ValueError(f"{path.name}: unsupported dataprec {precision!r}")
        dims = meta["dimList"]
        expected_dims = (self.nx, self.ny, self.nz)[:n_dims]
        for axis, size in enumerate(expected_dims):
            global_size, first, last = dims[3 * axis:3 * axis + 3]
            if (global_size, first, last) != (size, 1, size):
                raise ValueError(f"{path.name}: dimension {axis} is {dims[3 * axis:3 * axis + 3]}, "
                                 f"expected one tile of size {size}")

        plane = (self.ny, self.nx)
        volume = (self.nz, self.ny, self.nx)
        if n_dims == 3:
            if n_records != len(names):
                raise ValueError(f"{path.name}: {n_records} records for {len(names)} fields")
            shapes = [volume] * len(names)
        elif n_dims == 2:
            n_3d = names.index(_PICKUP_FIRST_2D) if _PICKUP_FIRST_2D in names else 0
            if n_3d * self.nz + (len(names) - n_3d) != n_records:
                raise ValueError(f"{path.name}: {n_records} records do not fit {n_3d} fields of "
                                 f"{self.nz} levels and {len(names) - n_3d} single-level fields")
            shapes = [volume] * n_3d + [plane] * (len(names) - n_3d)
        else:
            raise ValueError(f"{path.name}: unsupported nDims {n_dims}")

        raw = np.fromfile(path, dtype=_BIG_ENDIAN[precision])
        if raw.size != sum(math.prod(shape) for shape in shapes):
            raise ValueError(f"{path.name}: {raw.size} values, meta describes "
                             f"{sum(math.prod(shape) for shape in shapes)}")
        out: dict[str, torch.Tensor] = {}
        offset = 0
        for name, shape in zip(names, shapes):
            count = math.prod(shape)
            block = raw[offset:offset + count].reshape(shape).astype(np.float64)
            out[name] = torch.from_numpy(block)
            offset += count
        return out
