"""Parse MITgcm .meta companion files (Fortran namelist subset).

Format follows the upstream MITgcm MDS namelist convention:

    nDims = [ <N> ];
    dimList = [
       <dim1_global_size>, <dim1_local_start_1based>, <dim1_local_end_1based>,
       ...
    ];
    dataprec = [ '<float32|float64>' ];
    nrecords = [ <N> ];
    timeStepNumber = [ <N> ];

This parser tolerates whitespace and optional fields like `nFlds`, `fldList`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def parse_meta(meta_path: Path) -> dict[str, Any]:
    """Parse a MITgcm .meta file into a dict.

    Returns:
      {
        "nDims": int,
        "dimList": list[tuple[int, int, int]],   # per dim: (global, lstart, lend)
        "dataprec": "float32" | "float64",
        "nrecords": int,
        "timeStepNumber": int,  # 0 when omitted for a static grid field
      }

    Raises ValueError on malformed input.
    """
    text = meta_path.read_text()

    def _block(name: str) -> str:
        m = re.search(rf"{name}\s*=\s*\[(.*?)\]\s*;", text, flags=re.S)
        if m is None:
            raise ValueError(f"{meta_path}: missing field {name!r}")
        return m.group(1)

    # nDims
    n_dims_raw = _block("nDims").strip()
    try:
        n_dims = int(n_dims_raw)
    except ValueError as e:
        raise ValueError(f"{meta_path}: nDims not parseable: {n_dims_raw!r}") from e

    # dimList: list of comma-separated ints, 3 per dim
    dim_list_raw = _block("dimList")
    ints = [int(x) for x in re.findall(r"-?\d+", dim_list_raw)]
    if len(ints) != 3 * n_dims:
        raise ValueError(
            f"{meta_path}: dimList has {len(ints)} ints; expected {3 * n_dims} "
            f"(3 per dim × {n_dims} dims)"
        )
    dim_list = [tuple(ints[i * 3 : i * 3 + 3]) for i in range(n_dims)]

    # dataprec (quoted string)
    dataprec_raw = _block("dataprec")
    m_dp = re.search(r"'([^']+)'", dataprec_raw)
    if m_dp is None:
        raise ValueError(f"{meta_path}: dataprec has no quoted string: {dataprec_raw!r}")
    dataprec = m_dp.group(1)
    if dataprec not in {"float32", "float64"}:
        raise ValueError(f"{meta_path}: unsupported dataprec {dataprec!r}")

    # nrecords + timeStepNumber
    nrec = int(re.search(r"-?\d+", _block("nrecords")).group(0))  # type: ignore[union-attr]
    # MDSWRITEMETA omits timeStepNumber for static grids (myIter < 0).
    tstep_block = _block("timeStepNumber") if re.search(r"\btimeStepNumber\b", text) else ""
    tstep_match = re.search(r"-?\d+", tstep_block)
    tstep = int(tstep_match.group(0)) if tstep_match else 0

    return {
        "nDims": n_dims,
        "dimList": dim_list,
        "dataprec": dataprec,
        "nrecords": nrec,
        "timeStepNumber": tstep,
    }
