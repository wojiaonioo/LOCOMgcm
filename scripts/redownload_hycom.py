#!/usr/bin/env python3
"""Re-download corrupt HYCOM 3-hourly UVTS files identified by quality scan.

Two download modes (selectable via --mode):

    split  (default)   — Per bad date, fire 32 small NCSS sub-requests
                         (4 variables × 8 timesteps), then merge them
                         locally with xarray into one daily file. Each
                         sub-request is ~30 MB, so individual stalls and
                         server-side packaging waits are bounded.

    bulk               — Use the installed ocean_downloader CLI,
                         which fires one big NCSS request per
                         day combining all variables + times. ~893 MB per
                         request; can wait many minutes for server
                         packaging.

In both modes you control which files to (re-)fetch via the quality
report JSON. The script first deletes corrupt files (so they appear
"missing" to whatever downloader runs) unless --no-delete is set, and
all destructive actions are gated by --execute (default: dry-run).

Granularity flags (--variables / --hours) and --area let you override
the split-mode request shape from the CLI without touching the YAML
ocean_downloader config (which is only consulted in --mode=bulk).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "outputs" / "hycom_quality_report.json"
DEFAULT_CONFIG = ROOT / "scripts" / "configs" / "hycom_redownload.yaml"
DEFAULT_DATA_ROOT = ROOT / "data" / "hycom_uvts_2019-2023_3hour"

DATE_RE = re.compile(r"hycom_uvts_(\d{4})(\d{2})(\d{2})\.nc$")

# HYCOM GLBy0.08/expt_93.0 NCSS endpoint (covers 2018-01-01 → 2024-08-10).
NCSS_BASE = "https://ncss.hycom.org/thredds/ncss/grid/GLBy0.08/expt_93.0"

# Geospatial domain — copied verbatim from existing files so the redownloaded
# files line up cell-for-cell. Original: lat 0..45 (n=1126, dlat=0.04),
# lon 94.9599609375..170.0 (n=939, dlon≈0.078125).
DEFAULT_AREA = dict(
    west=94.9599609375,
    east=170.0,
    south=0.0,
    north=45.0,
)

DEFAULT_VARIABLES = ("water_u", "water_v", "water_temp", "salinity")
DEFAULT_HOURS = (0, 3, 6, 9, 12, 15, 18, 21)

# Encoding template lifted from existing well-formed file
# (see `hycom_uvts_20190101.nc`: int16 + zlib level 5 + chunks (1,20,563,470))
HYCOM_ENCODING_TEMPLATE = dict(
    zlib=True,
    complevel=5,
    chunksizes=(1, 20, 563, 470),
)


# ------------------------------------------------------------------
# Quality-report → file-list bridge
# ------------------------------------------------------------------

def collect_corrupt_files(report_path: Path, data_root: Path = DEFAULT_DATA_ROOT) -> list[dict]:
    """Return [{path, year, month, file, reasons, date}] for files needing re-fetch."""
    if not report_path.exists():
        sys.exit(
            f"Quality report not found: {report_path}\n"
            "Run the scan first."
        )
    with report_path.open() as f:
        report = json.load(f)

    by_file: dict[str, dict] = {}
    for issue in report["all_issues"]:
        sev, cat = issue["severity"], issue["category"]
        if sev != "CRITICAL" and not (sev == "WARN" and cat == "time_len"):
            continue
        fname = issue["file"]
        m = DATE_RE.search(fname)
        if not m:
            continue
        year, month, day = m.groups()
        full = str(data_root / year / f"{year}-{month}" / fname)
        rec = by_file.setdefault(
            full,
            {
                "path": full,
                "year": year,
                "month": month,
                "file": fname,
                "date": f"{year}-{month}-{day}",
                "reasons": [],
            },
        )
        rec["reasons"].append(f"[{sev}] {cat}: {issue['detail']}")
    return sorted(by_file.values(), key=lambda r: r["path"])


def delete_corrupt_files(files: list[dict], execute: bool) -> int:
    deleted = 0
    for r in files:
        p = Path(r["path"])
        if not p.exists():
            print(f"  [skip] not on disk: {p}")
            continue
        if execute:
            p.unlink()
            print(f"  [del]  {p}")
            deleted += 1
        else:
            print(f"  [dry]  would delete {p}")
    return deleted


# ------------------------------------------------------------------
# NCSS download (split mode)
# ------------------------------------------------------------------

def _build_ncss_url(*, variable: str, time_iso: str, area: dict) -> str:
    q = [
        ("var", variable),
        ("north", str(area["north"])),
        ("south", str(area["south"])),
        ("east", str(area["east"])),
        ("west", str(area["west"])),
        ("horizStride", "1"),
        ("time_start", time_iso),
        ("time_end", time_iso),
        ("timeStride", "1"),
        ("accept", "netcdf4"),
    ]
    return NCSS_BASE + "?" + urllib.parse.urlencode(q)


def _fetch_one(url: str, out: Path, *, timeout: float, retries: int, backoff: float = 5.0) -> None:
    """Download URL → out atomically with retries and exponential backoff."""
    out.parent.mkdir(parents=True, exist_ok=True)
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        tmp = out.with_suffix(out.suffix + ".part")
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "redownload_hycom/1.0"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                with tmp.open("wb") as f:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
            tmp.replace(out)
            return
        except Exception as e:  # noqa: BLE001
            last_exc = e
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            if attempt < retries:
                wait = backoff * (2 ** (attempt - 1))
                print(
                    f"      attempt {attempt}/{retries} failed: {e!s} — retry in {wait:.0f}s",
                    flush=True,
                )
                time.sleep(wait)
    raise RuntimeError(f"failed after {retries} attempts: {last_exc}")


def _merge_subfiles(
    sub_files: dict[tuple[str, int], Path],
    variables: Iterable[str],
    hours: Iterable[int],
    output: Path,
) -> None:
    """Concat per-(var,hour) subfiles into one daily NetCDF4."""
    import xarray as xr

    per_var = {}
    for v in variables:
        files = [sub_files[(v, h)] for h in hours]
        # decode_cf=False keeps int16 + scale_factor/add_offset intact, so
        # we only round-trip raw on-disk packed values rather than decode and
        # re-encode through float32.
        ds = xr.open_mfdataset(
            [str(p) for p in files],
            combine="by_coords",
            decode_cf=False,
            engine="netcdf4",
        )
        per_var[v] = ds

    merged = xr.merge(list(per_var.values()), combine_attrs="drop_conflicts")

    # Build per-variable encoding mirroring original. NCSS may return slightly
    # different chunk sizes; we override to original spec for byte-level parity.
    #
    # NB: with decode_cf=False, NCSS-emitted scale_factor / add_offset /
    # _FillValue / missing_value sit on .attrs *and* the file's CF encoding.
    # to_netcdf refuses if they're in BOTH attrs and explicit encoding (it
    # treats the duplication as ambiguous). Move them from attrs into
    # encoding here so we end up with one source of truth.
    ENCODING_KEYS = ("dtype", "scale_factor", "add_offset", "_FillValue", "missing_value")
    encoding = {}
    for v in variables:
        enc = dict(HYCOM_ENCODING_TEMPLATE)
        for key in ENCODING_KEYS:
            val = merged[v].encoding.get(key)
            if val is None:
                val = merged[v].attrs.get(key)
            if val is not None:
                enc[key] = val
            # Ensure attrs is clean — to_netcdf would otherwise complain about
            # the duplicate key.
            merged[v].attrs.pop(key, None)
        encoding[v] = enc

    output.parent.mkdir(parents=True, exist_ok=True)
    # Write to temporary local storage before moving the completed file.
    with tempfile.TemporaryDirectory(prefix="hycom-merge-") as scratch:
        local_tmp = Path(scratch) / output.name
        merged.to_netcdf(local_tmp, engine="netcdf4", encoding=encoding)
        for ds in per_var.values():
            ds.close()
        shutil.move(str(local_tmp), str(output))


def download_one_split(
    *,
    rec: dict,
    variables: Iterable[str],
    hours: Iterable[int],
    area: dict,
    request_timeout: float,
    retries: int,
    overwrite: bool,
) -> str:
    """Per-day driver: 32 sub-requests + merge, in --mode=split."""
    target = Path(rec["path"])
    if target.exists() and not overwrite:
        return "skip-existing"

    date = rec["date"]  # e.g. 2019-01-18
    print(f"  -> {date} ({target.name}) — {len(list(variables))*len(list(hours))} sub-requests")

    with tempfile.TemporaryDirectory(prefix="hycom_split_", dir=ROOT / "tmp") as td:
        td = Path(td)
        sub_files: dict[tuple[str, int], Path] = {}
        t0 = time.time()
        for v in variables:
            for h in hours:
                ts = f"{date}T{h:02d}:00:00Z"
                url = _build_ncss_url(variable=v, time_iso=ts, area=area)
                sp = td / f"{v}_{h:02d}.nc"
                t_one = time.time()
                _fetch_one(url, sp, timeout=request_timeout, retries=retries)
                size_mb = sp.stat().st_size / 1e6
                print(
                    f"     {v:<11} t={h:02d}h  {size_mb:6.1f} MB  {time.time()-t_one:5.1f}s",
                    flush=True,
                )
                sub_files[(v, h)] = sp
        elapsed_dl = time.time() - t0

        print(f"     merging {len(sub_files)} subfiles ...", flush=True)
        t_merge = time.time()
        _merge_subfiles(sub_files, list(variables), list(hours), target)
        out_mb = target.stat().st_size / 1e6
        print(
            f"     wrote {target.name}  ({out_mb:.1f} MB; "
            f"download {elapsed_dl:.0f}s, merge {time.time()-t_merge:.0f}s)",
            flush=True,
        )

    return "ok"


def run_split_mode(
    files: list[dict],
    *,
    variables: list[str],
    hours: list[int],
    area: dict,
    request_timeout: float,
    retries: int,
    execute: bool,
    overwrite: bool,
) -> int:
    if not execute:
        n = len(files) * len(variables) * len(hours)
        print(
            f"  [dry] split-mode would issue {n} NCSS sub-requests "
            f"({len(files)} dates × {len(variables)} vars × {len(hours)} hours)"
        )
        return 0

    ok = skipped = failed = 0
    for i, rec in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {rec['date']}")
        try:
            result = download_one_split(
                rec=rec,
                variables=variables,
                hours=hours,
                area=area,
                request_timeout=request_timeout,
                retries=retries,
                overwrite=overwrite,
            )
            if result == "skip-existing":
                skipped += 1
                print(f"  -> skip (file already on disk)")
            else:
                ok += 1
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  -> FAILED: {e}", flush=True)

    print(
        f"\nsplit-mode summary: ok={ok}  skipped={skipped}  failed={failed}",
        flush=True,
    )
    return 0 if failed == 0 else 1


# ------------------------------------------------------------------
# bulk mode (delegate to ocean_downloader)
# ------------------------------------------------------------------

def run_bulk_mode(config_path: Path, execute: bool) -> int:
    if importlib.util.find_spec("ocean_downloader") is None:
        sys.exit(
            "Install ocean_downloader in the active environment or use --mode split."
        )
    cmd = [
        sys.executable,
        "-m",
        "ocean_downloader.cli",
        "--config",
        str(config_path),
        "download",
    ]
    env = os.environ.copy()
    print(f"\nInvoking: {' '.join(cmd)}")
    if not execute:
        print("  [dry] not running (pass --execute to run)")
        return 0
    proc = subprocess.run(cmd, cwd=ROOT, env=env)
    return proc.returncode


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_csv_int(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_csv_str(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT,
                    help=f"Quality scan JSON (default: {DEFAULT_REPORT.relative_to(ROOT)})")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                    help=f"ocean_downloader config (only used in bulk mode; default: {DEFAULT_CONFIG.relative_to(ROOT)})")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                    help="HYCOM data directory; match the bulk-mode YAML output directory.")
    ap.add_argument("--mode", choices=["split", "bulk"], default="split",
                    help="split = many small NCSS requests + local merge (default); bulk = ocean_downloader, one big request per day")
    ap.add_argument("--variables", type=parse_csv_str,
                    default=list(DEFAULT_VARIABLES),
                    help="Comma-separated list (split mode only). Default: " + ",".join(DEFAULT_VARIABLES))
    ap.add_argument("--hours", type=parse_csv_int,
                    default=list(DEFAULT_HOURS),
                    help="Comma-separated hours of day (split mode only). Default: 0,3,6,9,12,15,18,21")
    ap.add_argument("--area", type=parse_csv_str,
                    default=None,
                    help="west,east,south,north (degrees). Default matches existing files.")
    ap.add_argument("--request-timeout", type=float, default=900.0,
                    help="Per-sub-request socket timeout in seconds (split mode). Default: 900")
    ap.add_argument("--retries", type=int, default=5,
                    help="Per-sub-request retry attempts (split mode). Default: 5")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-download even if final daily file already exists.")
    ap.add_argument("--execute", action="store_true",
                    help="Actually delete + download. Without this flag, dry-run only.")
    ap.add_argument("--no-delete", action="store_true",
                    help="Skip delete step (useful to resume an interrupted run).")
    ap.add_argument("--no-download", action="store_true",
                    help="Only delete (or print would-deletes); no fetch.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N bad files (for testing).")
    args = ap.parse_args()

    files = collect_corrupt_files(args.report, args.data_root)
    if args.limit:
        files = files[: args.limit]
    print(f"Files to re-download: {len(files)}")
    by_year: dict[str, int] = {}
    for r in files:
        by_year[r["year"]] = by_year.get(r["year"], 0) + 1
    for yr in sorted(by_year):
        print(f"  {yr}: {by_year[yr]}")
    print()

    if not args.no_delete:
        print("=== Delete step ===")
        n = delete_corrupt_files(files, args.execute)
        print(f"  {'Deleted' if args.execute else 'Would delete'} {n} files\n")

    if args.no_download:
        return 0

    print(f"=== Download step ({args.mode}) ===")
    if args.mode == "split":
        area = dict(DEFAULT_AREA)
        if args.area:
            if len(args.area) != 4:
                sys.exit("--area must be west,east,south,north")
            area = dict(zip(("west", "east", "south", "north"),
                            [float(x) for x in args.area]))
        print(f"  area: west={area['west']} east={area['east']} south={area['south']} north={area['north']}")
        print(f"  variables: {args.variables}")
        print(f"  hours: {args.hours}")
        print(f"  request_timeout={args.request_timeout}s  retries={args.retries}\n")
        return run_split_mode(
            files,
            variables=args.variables,
            hours=args.hours,
            area=area,
            request_timeout=args.request_timeout,
            retries=args.retries,
            execute=args.execute,
            overwrite=args.overwrite,
        )

    return run_bulk_mode(args.config, args.execute)


if __name__ == "__main__":
    raise SystemExit(main())
