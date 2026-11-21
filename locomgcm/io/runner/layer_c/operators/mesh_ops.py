"""Mesh-side operators: extract_mesh_dims, compute_mpi_topology.

These operate on MESH-kind bundles or on RunConfig metadata and produce
small "side-info" bundles that downstream Layer D consumers serialize as
SIZE.h / data namelist / topology yaml.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
from typing import Any

import numpy as np
import xarray as xr

from locomgcm.io.runner.layer_a import (
    BundleMeta,
    ConventionTag,
    DataBundle,
    LocationKind,
    ProvenanceStep,
    VariableSpec,
)

from .._registry import register_operator
from .._types import OperatorCapabilities


# ============================================================
# extract_mesh_dims
# ============================================================


@register_operator("extract_mesh_dims")
class ExtractMeshDimsOperator:
    """MESH bundle → summary bundle with mesh dims in extras.

    Output bundle preserves MESH location_kind but stamps mesh dims into
    `bundle.meta.extras['mesh_dims']` for downstream consumers (mitgcm_size_h,
    mitgcm_topology). `mesh_dims` always carries `nx`/`ny`; `nz` is included
    only when the upstream source stamped `extras['nz']` (otherwise omitted).
    """

    capabilities = OperatorCapabilities(
        accepts=(LocationKind.MESH,),
        produces=(LocationKind.MESH,),
        convention_in=None,
        convention_out=None,
        preserves_variables=True,
    )

    def __init__(self, **_: Any) -> None:
        pass

    def __call__(self, bundle: DataBundle) -> DataBundle:
        ny = bundle.data.sizes.get("latitude", 0)
        nx = bundle.data.sizes.get("longitude", 0)
        nz_from_extras = bundle.meta.extras.get("nz")
        mesh_dims = {"nx": nx, "ny": ny}
        if nz_from_extras is not None:
            mesh_dims["nz"] = nz_from_extras
        new_meta = dataclasses.replace(
            bundle.meta,
            extras={**dict(bundle.meta.extras), "mesh_dims": mesh_dims},
        )
        return DataBundle(data=bundle.data, meta=new_meta)


# ============================================================
# compute_mpi_topology
# ============================================================


@register_operator("compute_mpi_topology")
class ComputeMpiTopologyOperator:
    """Given mesh dims + target nProcs, compute optimal nPx × nPy.

    Picks the (nPx, nPy) divisor pair of nProcs that:
      - integer-divides nx and ny respectively
      - minimizes halo overhead (proxy: minimizes max(sNx, sNy) / min(sNx, sNy))

    Stamps decision into `bundle.meta.extras['mpi_topology']` for consumers
    (mitgcm_topology, mitgcm_size_h).
    """

    capabilities = OperatorCapabilities(
        accepts=(LocationKind.MESH,),
        produces=(LocationKind.MESH,),
        convention_in=None,
        convention_out=None,
        preserves_variables=True,
    )

    def __init__(self, *, target_nProcs: int = 1, **_: Any) -> None:
        self._target_nprocs = int(target_nProcs)

    def __call__(self, bundle: DataBundle) -> DataBundle:
        mesh = bundle.meta.extras.get("mesh_dims")
        if mesh is None:
            ny = bundle.data.sizes.get("latitude", 0)
            nx = bundle.data.sizes.get("longitude", 0)
        else:
            nx = mesh["nx"]
            ny = mesh["ny"]
        nprocs = self._target_nprocs

        # Find all (nPx, nPy) pairs that nPx*nPy == nprocs AND nx % nPx == 0 AND ny % nPy == 0
        candidates: list[tuple[int, int, float]] = []
        for nPx in range(1, nprocs + 1):
            if nprocs % nPx != 0:
                continue
            nPy = nprocs // nPx
            if nx % nPx != 0 or ny % nPy != 0:
                continue
            sNx = nx // nPx
            sNy = ny // nPy
            # Halo overhead proxy: maximize min-tile-dim relative to halo (assume halo=4)
            halo = 4
            cost = (sNx + 2 * halo) * (sNy + 2 * halo) / (sNx * sNy)
            candidates.append((nPx, nPy, cost))

        if not candidates:
            raise ValueError(
                f"compute_mpi_topology: no valid (nPx, nPy) for nProcs={nprocs}, "
                f"nx={nx}, ny={ny}"
            )
        # Best = lowest halo overhead
        candidates.sort(key=lambda c: c[2])
        best_nPx, best_nPy, _ = candidates[0]
        sNx = nx // best_nPx
        sNy = ny // best_nPy

        topology = {
            "nPx": best_nPx,
            "nPy": best_nPy,
            "sNx": sNx,
            "sNy": sNy,
            "n_procs": nprocs,
            "halo": 4,
        }
        new_meta = dataclasses.replace(
            bundle.meta,
            extras={**dict(bundle.meta.extras), "mpi_topology": topology},
        )
        return DataBundle(data=bundle.data, meta=new_meta)
