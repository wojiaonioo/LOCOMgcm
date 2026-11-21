"""Content-keyed cache for expensive external-data regrid pipelines.

Only the stage1a external-source pipelines (HYCOM / ERA5 / GEBCO → regridded
zarr) are cached. Their output is a *pure function* of
``{source files, grid, time window, operator chain, consumer}`` so the key
built here is provably complete. Config-dependent pipelines (e.g. stage1c MIT
namelists, whose output depends on dt / physics / sponge) are intentionally
NOT cached — caching them with this key would under-key and silently serve a
stale namelist.

Layout: ``<cache.root>/pipeline_cas/<consumer>/<entry>/`` holds the artifact,
``key.json`` (the full key) and ``meta.json``. Entry directories are named
``<pipeline>_<nx>x<ny>x<nz>_<time window>`` (``.2``, ``.3`` … on a name
clash); older entries carry a 64-hex name. Lookup
ignores the name: it parses every ``<consumer>/*/key.json`` and compares its
canonical text with the current key's in full, so both layouts hit.

Safety properties (per the alignment protocol's no-silent-corruption rule —
a wrong cache hit is far worse than a miss):

  * the key includes EVERY input that affects the output — crucially the
    source files' ``size + mtime_ns`` (not just their path, so regenerating
    a .nc in place busts the cache) plus a bumpable schema version (so a code
    change to the regrid invalidates old entries);
  * publish is ATOMIC — the entry is built under  and
    ``os.replace``d into place, so a crash mid-write never leaves a
    half-written entry that a later run treats as valid;
  * HIT / MISS / PUBLISH are logged with the entry directory name;
  * opt-in ``cache.verify`` recomputes and bitwise-asserts the cached artifact
    matches — the second independent source that proves the key is complete;
  * escape hatches: ``cache.rebuild`` forces republish; delete ``cache.root``.

Materialization of a hit is a SYMLINK into the (immutable) CAS, so reuse is
instant and shares storage; the originating run keeps its own real artifact.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from locomgcm.io.runner.layer_d import ArtifactKind, ArtifactRef

# Bump when the regrid/operator code changes in a way that alters output bytes.
_CACHE_SCHEMA_VERSION = "v2-2026-06-09-hycom-xy-before-z"

# Pipelines whose primary_source is one of these raw external datasets are the
# only ones whose output is provably keyed by {source + grid + time + ops}.
_REGRID_SOURCE_TYPES = frozenset(
    {
        "hycom_netcdf",
        "hycom_raw_by_variable",
        "era5_netcdf",
        "gebco_netcdf",
        "tpxo9_netcdf",
    }
)

# Sources that ignore the query time window (static climatologies). Their CAS
# key must NOT include start_date/time_range so the artifact is reusable across
# simulation start dates. For the TPXO9 tidal atlas this is essential: the
# Greenwich amplitude/phase output is genuinely start-date independent (V0 is
# applied at model-load time, not in the lane).
_TIME_INVARIANT_SOURCE_TYPES = frozenset({"tpxo9_netcdf"})

_CAS_SUBDIR = "pipeline_cas"

# Entry names are built from the pipeline name; anything else becomes ``_``.
_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9._-]")

# Rounds of "look up by content, then rename into a free name" before publish
# gives up; a round is only repeated when a concurrent run took the name.
_PUBLISH_ATTEMPTS = 5


def _source_spec(run_cfg, name: str):
    for s in run_cfg.sources:
        if s.name == name:
            return s
    return None


def _fingerprint_source(src: Any, source_spec) -> dict:
    """Identity of the source's inputs: type + config + per-file size+mtime.

    Path alone is insufficient — regenerating a .nc in place at the same path
    must bust the cache, so we record (path, size, mtime_ns) of every candidate
    file. Over-keying (an unrelated extra file in the dir) only causes a miss,
    never a wrong hit.
    """
    fp: dict[str, Any] = {
        "type": source_spec.type if source_spec is not None else "?",
        "config": dict(source_spec.config) if source_spec is not None else {},
    }
    files = getattr(src, "_files", None)
    paths: list[Path] = []
    if files:
        paths = [Path(f) for f in files]
    else:
        p = getattr(src, "_path", None)
        if p is not None and Path(p).exists():
            paths = [Path(p)]
    stamped = []
    for f in sorted(paths):
        try:
            st = f.stat()
            stamped.append([str(f), st.st_size, st.st_mtime_ns])
        except OSError:
            stamped.append([str(f), -1, -1])
    fp["files"] = stamped
    return fp


def _key_text(key: Any) -> str:
    """Canonical text of a key; ``lookup`` compares it in full.

    The key first takes the form it reads back from key.json (tuples become
    lists, other non-JSON values go through ``str``), then is dumped with
    sorted keys. Apart from the order of non-string dict keys (sorted here
    after conversion to strings), this is the text older entries were keyed
    on, so value types count as they did then: 200 and 200.0,
    true and 1, -0.0 and 0.0 stay distinct, and NaN matches NaN.
    """
    return json.dumps(json.loads(json.dumps(key, default=str)), sort_keys=True)


def _entry_base_name(payload: dict) -> str:
    """Readable entry directory name: pipeline, grid size, time window.

    Only for people browsing the cache; lookup never parses it. Characters
    outside ``[A-Za-z0-9._-]`` become ``_`` and a leading dot is replaced, so a
    pipeline name such as ``a/b`` or ``../x`` still yields one visible path
    component directly under the consumer directory, where lookup globs.
    """
    g = payload["grid"]
    tr = payload["query"]["time_range"]
    window = f"{tr[0][:10]}_{tr[1][:10]}" if tr else "static"
    name = _UNSAFE_NAME_CHARS.sub(
        "_", f"{payload['pipeline_name']}_{g['nx']}x{g['ny']}x{g['nz']}_{window}"
    )
    return "_" + name[1:] if name.startswith(".") else name


def _arr_equal(a, b) -> bool:
    import numpy as np

    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if np.issubdtype(a.dtype, np.floating):
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


class PipelineCache:
    """Per-pipeline content-keyed cache, wired into run_pipeline_spec."""

    def __init__(self, run_cfg, logger) -> None:
        cc = run_cfg.cache
        self.run_cfg = run_cfg
        self.logger = logger
        self.active = bool(getattr(cc, "reuse", False))
        self.verify = bool(getattr(cc, "verify", False))
        self.rebuild = bool(getattr(cc, "rebuild", False))
        self.root = Path(cc.root)

    # -- scoping -------------------------------------------------------
    def is_cacheable(self, pipeline) -> bool:
        spec = _source_spec(self.run_cfg, pipeline.primary_source)
        return spec is not None and spec.type in _REGRID_SOURCE_TYPES

    # -- key -----------------------------------------------------------
    def compute_key(self, pipeline, src, query) -> dict:
        """The full cache key; an entry hits when its key.json has the same ``_key_text``."""
        g = self.run_cfg.grid
        spec = _source_spec(self.run_cfg, pipeline.primary_source)
        tr = query.time_range
        # Time-invariant sources (TPXO9 atlas) ignore the query time window —
        # keep start_date out of the key for cross-date artifact reuse.
        if spec is not None and spec.type in _TIME_INVARIANT_SOURCE_TYPES:
            tr = None
        payload = {
            "v": _CACHE_SCHEMA_VERSION,
            "pipeline_name": pipeline.name,
            "primary_source": pipeline.primary_source,
            "operators": [[op.op, dict(op.config)] for op in pipeline.operators],
            "consumer": pipeline.consumer,
            "consumer_config": dict(pipeline.consumer_config),
            "grid": {
                "nx": g.nx, "ny": g.ny, "nz": g.nz,
                "center_lat": g.center_lat, "center_lon": g.center_lon,
                "resolution_deg": g.resolution_deg, "depth_m": g.depth_m,
                "vertical_layer_thicknesses": list(g.vertical_layer_thicknesses),
            },
            "query": {
                "time_range": [str(tr[0]), str(tr[1])] if tr else None,
                "first_time_only": bool((query.extras or {}).get("first_time_only", False)),
                "variables": sorted(query.variables),
                "location_kind": str(query.location_kind),
            },
            "source": _fingerprint_source(src, spec),
        }
        return payload

    # -- store layout --------------------------------------------------
    def lookup(self, consumer: str, payload: dict) -> Path | None:
        """Entry whose key.json holds ``payload``, or None.

        Reads every ``<consumer>/*/key.json`` regardless of the directory
        name and compares its ``_key_text`` with that of ``payload``, so
        entries published under the old 64-hex names still hit.
        """
        wanted = _key_text(payload)
        for key_file in sorted((self.root / _CAS_SUBDIR / consumer).glob("*/key.json")):
            entry = key_file.parent
            if not (entry / "meta.json").exists():
                continue
            try:
                stored = _key_text(json.loads(key_file.read_text()))
            except (OSError, ValueError):
                continue
            if stored == wanted:
                return entry
        return None

    def _free_entry_dir(self, consumer: str, payload: dict) -> Path:
        parent = self.root / _CAS_SUBDIR / consumer
        base = _entry_base_name(payload)
        candidate, n = parent / base, 1
        while candidate.exists():
            n += 1
            candidate = parent / f"{base}.{n}"
        return candidate

    # -- hit -----------------------------------------------------------
    def materialize(self, entry: Path, ctx) -> ArtifactRef:
        meta = json.loads((entry / "meta.json").read_text())
        basename = meta["basename"]
        cas_artifact = (entry / basename).resolve()
        target = ctx.consumer_context.output_dir / basename
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)
        target.symlink_to(cas_artifact)
        return ArtifactRef(
            name=meta["name"],
            path=target,
            artifact_kind=ArtifactKind[meta["artifact_kind"]],
            bundle_hash=meta["bundle_hash"],
            schema_version=meta["schema_version"],
            extras=dict(meta.get("extras", {})),
        )

    # -- miss ----------------------------------------------------------
    def publish(self, consumer: str, artifact: ArtifactRef, payload: dict,
                *, overwrite: bool = False) -> Path:
        """Store ``artifact`` under ``payload``; return the entry directory.

        Without ``overwrite``, an entry with an equal key.json already in place
        (a concurrent run published first) is returned as is. With
        ``overwrite`` (cache.rebuild), that entry is replaced at the same path,
        so symlinks earlier runs made into it stay valid.

        A run whose last lookup precedes a concurrent run's rename, but whose
        name choice follows it, still publishes a second equal entry under a
        ``.N`` name. Both were built from the same key; lookup returns the
        first by name, so the extra copy only costs disk space.
        """
        cas_root = self.root / _CAS_SUBDIR
        tmp = cas_root / ".tmp" / uuid4().hex
        tmp.mkdir(parents=True, exist_ok=True)
        basename = Path(artifact.path).name
        try:
            shutil.copytree(artifact.path, tmp / basename)
            (tmp / "key.json").write_text(json.dumps(payload, indent=2, default=str))
            (tmp / "meta.json").write_text(json.dumps({
                "name": artifact.name,
                "basename": basename,
                "artifact_kind": artifact.artifact_kind.name,
                "bundle_hash": artifact.bundle_hash,
                "schema_version": artifact.schema_version,
                "extras": dict(artifact.extras or {}),
                "created_at": datetime.utcnow().isoformat(),
            }, indent=2, default=str))
            for _ in range(_PUBLISH_ATTEMPTS):
                # Look up by content right before the rename: a concurrent run
                # may have published an equal entry since our miss.
                final = self.lookup(consumer, payload)
                if final is not None and not overwrite:
                    return final
                if final is not None:
                    shutil.rmtree(final, ignore_errors=True)
                else:
                    final = self._free_entry_dir(consumer, payload)
                final.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.replace(tmp, final)  # atomic within cas_root FS
                except OSError:
                    if final.exists():
                        continue  # a concurrent run took this name; look again
                    raise
                tmp = None  # consumed by replace
                return final
            raise RuntimeError(
                f"[cache] {consumer}: entry still not in place after {_PUBLISH_ATTEMPTS} tries"
            )
        finally:
            if tmp is not None and tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)

    # -- verify (opt-in) ----------------------------------------------
    def assert_bitwise_equal(self, entry: Path, artifact: ArtifactRef, pipeline) -> None:
        import zarr

        meta = json.loads((entry / "meta.json").read_text())
        cached = str((entry / meta["basename"]).resolve())
        fresh = str(Path(artifact.path).resolve())
        ga = zarr.open_group(cached, mode="r")
        gb = zarr.open_group(fresh, mode="r")
        ka, kb = set(ga.array_keys()), set(gb.array_keys())
        if ka != kb:
            raise RuntimeError(
                f"[cache VERIFY FAIL] {pipeline.name}: array set differs "
                f"cached={sorted(ka)} fresh={sorted(kb)} — cache key is INCOMPLETE"
            )
        for k in sorted(ka):
            if not _arr_equal(ga[k], gb[k]):
                raise RuntimeError(
                    f"[cache VERIFY FAIL] {pipeline.name}/{k}: cached != recomputed "
                    f"— cache key is INCOMPLETE (a field affecting output is not in the key)"
                )
        self.logger.info(f"[cache VERIFY PASS] {pipeline.name}")
