"""MITgcm pickup files read the way model/src/read_pickup.F reads a field-list pickup.

Arrays keep MITgcm's layout and signs: a 3-D field is [Nr, Ny, Nx] float64 with index 0 the
surface level, a surface field [Ny, Nx]; uVel(i) sits on cell i's west face, vVel(j) on its
south face, theta, salt and etaN at cell centres; no halo.  Every field a pickup holds has the
same sign in LOCOMgcm (U east, V north, eta up; a pickup holds no w), so nothing is converted.
Whether a run starts from a pickup is the caller's decision; this module only reads files.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from defgcm.io.mds_v2 import parse_meta

#: Fields read_pickup.F reads with one level (READ_MFLDS_3D_RL with nNz=1, read_pickup.F:443-463).
SURFACE_FIELDS = ("EtaN", "dEtaHdt", "EtaH", "dPhiNH", "Phi_rLow")
#: AB2 history fields and the start flag they set: a field-list pickup sets each start-AB
#: iteration to nIter0 (check_pickup.F:57-65) and a missing field zeroes it (:179-198).
AB_HISTORY_FIELDS = (("mom", ("GuNm1", "GvNm1")), ("temp", ("GtNm1",)), ("salt", ("GsNm1",)))
#: The records READ_PICKUP starts each prognostic field from, in place of INI_VEL, INI_THETA,
#: INI_SALT and INI_PSURF (ini_fields.F:27-38), by LOCOMgcm state field.
PROGNOSTIC_RECORDS = (("u", "Uvel"), ("v", "Vvel"), ("temp", "Theta"), ("salt", "Salt"), ("eta", "EtaN"))
#: The AB2 history record of each field's tendency: GuNm1 is gU before the extrapolation of the
#: previous step (adams_bashforth2.F:83-85), by LOCOMgcm AB history field.
AB_HISTORY_RECORDS = (("u", "GuNm1"), ("v", "GvNm1"), ("temp", "GtNm1"), ("salt", "GsNm1"))
#: pickup_cd records by CdState field: (field, record, grid point of the values named by the
#: state field there).  uVelD sits on V points and vVelD on U points (CD_CODE_VARS.h);
#: uNM1, vNM1 are C-grid velocities, etaNm1 a centre field (cd_code_read_pickup.F:68-75).
CD_PICKUP_RECORDS = (("u_d", "uVelD", "v"), ("v_d", "vVelD", "u"), ("u_nm1", "uNM1", "u"),
                     ("v_nm1", "vNM1", "v"), ("eta_nm1", "etaNm1", "eta"))


def pickup_field_list(meta_path: str | Path) -> tuple[str, ...]:
    """The ``fldList`` of a pickup meta file, names stripped of their blank padding."""
    text = Path(meta_path).read_text()
    found = re.search(r"fldList\s*=\s*\{(.*?)\}\s*;", text, flags=re.S)
    if found is None:
        raise ValueError(
            f"{meta_path}: no fldList; a positional pickup (usePickupBeforeC54, "
            "read_pickup.F:160-250) is not ported")
    return tuple(name.strip() for name in re.findall(r"'([^']*)'", found.group(1)))


def read_mit_pickup(meta_path: str | Path, data_path: str | Path, *, nx: int, ny: int,
                    nz: int) -> dict[str, np.ndarray]:
    """``{field name: array}`` of a single global float64 pickup file.

    Record numbers follow READ_MFLDS_SET / READ_MFLDS_3D_RL (pkg/rw/read_mflds.F:162-183,
    302-325): the first ``nFl3D = (nRecords - nFlds)/(Nr - 1)`` fields of the list are 3-D and
    take ``Nr`` records each, the rest one record each.
    """
    meta = parse_meta(Path(meta_path))
    names = pickup_field_list(meta_path)
    if meta["dataprec"] != "float64":
        raise ValueError(f"{meta_path}: pickup precision {meta['dataprec']}, read_pickup.F:108-117 "
                         "expects float64")
    if [tuple(d) for d in meta["dimList"]] != [(nx, 1, nx), (ny, 1, ny)]:
        raise ValueError(f"{meta_path}: dimList {meta['dimList']} is not one global {nx}x{ny} file")
    volume = tuple(name for name in names if name not in SURFACE_FIELDS)
    if names[:len(volume)] != volume:
        raise ValueError(f"{meta_path}: 3-D fields must precede the surface fields (read_mflds.F:302-325)")
    n_records = meta["nrecords"]
    if len(volume) * nz + len(names) - len(volume) != n_records:
        raise ValueError(f"{meta_path}: {n_records} records do not hold {len(volume)} 3-D fields of "
                         f"{nz} levels and {len(names) - len(volume)} surface fields")
    plane = ny * nx
    raw = np.fromfile(data_path, dtype=">f8")
    if raw.size != n_records * plane:
        raise ValueError(f"{data_path}: {raw.size} values, the meta file declares {n_records * plane}")
    fields: dict[str, np.ndarray] = {}
    for j, name in enumerate(names):
        if j < len(volume):
            start = j * nz * plane
            fields[name] = raw[start:start + nz * plane].reshape(nz, ny, nx).astype(np.float64)
        else:
            start = (len(volume) * nz + j - len(volume)) * plane
            fields[name] = raw[start:start + plane].reshape(ny, nx).astype(np.float64)
    return fields


def pickup_ab_start(fields: dict[str, np.ndarray], *, n_iter0: int) -> dict[str, bool]:
    """Whether the first step extrapolates each AB2 history.

    adams_bashforth2.F:61-65 takes abFac=0 at ``myIter == nIter0`` when the start-AB iteration
    is 0; it is nIter0 with the history present and 0 without (check_pickup.F:57-65, 179-198), so
    a pickup read at nIter0=0 (a non-blank pickupSuff, ini_fields.F:27-38) does not extrapolate.
    """
    return {kind: n_iter0 != 0 and all(name in fields for name in names)
            for kind, names in AB_HISTORY_FIELDS}


def read_mit_pickup_cd(data_path: str | Path, *, nx: int, ny: int, nz: int) -> dict[str, np.ndarray]:
    """``{record name: array}`` of a global float64 ``pickup_cd`` file.

    CD_CODE_READ_PICKUP reads fixed records (cd_code_read_pickup.F:45, 68-75): uVelD, vVelD,
    uNM1, vNM1 as records 1-4 of ``Nr`` levels, etaNm1 as the one-level record ``4*Nr+1``
    (the ``usePickupBeforeC54`` position ``6*Nr+1`` is not ported).
    """
    plane = ny * nx
    raw = np.fromfile(data_path, dtype=">f8")
    if raw.size != (4 * nz + 1) * plane:
        raise ValueError(f"{data_path}: {raw.size} values are not 4 fields of {nz} levels and "
                         f"one surface field on {nx}x{ny}")
    fields = {name: raw[r * nz * plane:(r + 1) * nz * plane].reshape(nz, ny, nx).astype(np.float64)
              for r, name in enumerate(("uVelD", "vVelD", "uNM1", "vNM1"))}
    fields["etaNm1"] = raw[4 * nz * plane:].reshape(ny, nx).astype(np.float64)
    return fields
