"""Write synthetic raw inputs for the built-in ``pureocean`` case.

The ``pureocean`` case (``experiments/configs/cases/pureocean.yaml``) reads
HYCOM ocean state, ERA5 10 m winds and GEBCO bathymetry in stage 1a.  This
script writes small files in the same formats so that the pipeline and the
examples run without downloading anything.  The fields are idealized and have
no physical meaning beyond exercising the code:

* flat 1000 m bottom, no land;
* temperature and salinity profiles that vary with depth, plus a warm
  Gaussian anomaly in the domain centre that spins up a geostrophic eddy;
* zero initial velocity and sea surface height;
* a uniform easterly wind with a weak daily modulation.

Usage (from the repository root)::

    export LOCOMGCM_SYNTHETIC_PUREOCEAN_DIR="$PWD/data/synthetic_pureocean"
    python examples/make_synthetic_inputs.py --out "$LOCOMGCM_SYNTHETIC_PUREOCEAN_DIR"

Then select the overlay with named input references::

    --overlay examples/pureocean_cpu.yaml
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# Source grid: 0.25 degree, covering the pureocean domain
# (centre 11.75N 111.75E, 36 x 1/12 degree = 3 degrees wide) plus the 1 degree
# margin that the HYCOM and ERA5 readers cut around the model box.
LAT = np.arange(8.0, 15.5 + 1e-9, 0.25)
LON = np.arange(108.0, 115.5 + 1e-9, 0.25)
# A subset of the HYCOM GLBy0.08 standard depths, reaching below the 1000 m
# model bottom so that vertical interpolation never extrapolates.
DEPTH = np.array(
    [0, 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500,
     600, 700, 800, 900, 1000, 1250, 1500],
    dtype="float64",
)
CENTER_LAT, CENTER_LON = 11.75, 111.75


def _temperature_salinity() -> tuple[np.ndarray, np.ndarray]:
    z = DEPTH[:, None, None]
    lat = LAT[None, :, None]
    lon = LON[None, None, :]
    background = 5.0 + 23.0 * np.exp(-z / 300.0)
    r2 = ((lat - CENTER_LAT) ** 2 + (lon - CENTER_LON) ** 2) / 0.5**2
    anomaly = 1.0 * np.exp(-r2) * np.exp(-z / 200.0)
    temp = background + anomaly
    salt = 34.0 + 0.6 * (1.0 - np.exp(-z / 300.0)) + 0.0 * lat * lon
    return temp, salt


def write_hycom(out: Path, start: datetime, days: int) -> None:
    """Daily files ``hycom_uvts_YYYYMMDD.nc`` with eight 3-hourly records."""
    out.mkdir(parents=True, exist_ok=True)
    temp, salt = _temperature_salinity()
    shape3 = (8, DEPTH.size, LAT.size, LON.size)
    for d in range(days):
        day = start + timedelta(days=d)
        times = pd.date_range(day, periods=8, freq="3h")
        ds = xr.Dataset(
            {
                "water_temp": (("time", "depth", "lat", "lon"),
                               np.broadcast_to(temp, shape3).astype("float32")),
                "salinity": (("time", "depth", "lat", "lon"),
                             np.broadcast_to(salt, shape3).astype("float32")),
                "water_u": (("time", "depth", "lat", "lon"), np.zeros(shape3, "float32")),
                "water_v": (("time", "depth", "lat", "lon"), np.zeros(shape3, "float32")),
                "surf_el": (("time", "lat", "lon"),
                            np.zeros((8, LAT.size, LON.size), "float32")),
            },
            coords={"time": times, "depth": DEPTH, "lat": LAT, "lon": LON},
        )
        ds.to_netcdf(out / f"hycom_uvts_{day:%Y%m%d}.nc")


def write_era5(out: Path, start: datetime, days: int) -> None:
    """Hourly ``<out>/<year>/era5_{u10,v10}_<year>.nc`` (latitude descending)."""
    times = pd.date_range(start, periods=24 * days + 1, freq="1h")
    hours = np.arange(times.size, dtype="float64")
    modulation = np.sin(2.0 * np.pi * hours / 24.0)[:, None, None]
    lat_desc = LAT[::-1]
    shape = (times.size, lat_desc.size, LON.size)
    winds = {
        "u10": -6.0 + 1.0 * modulation + np.zeros(shape),
        "v10": 2.0 + 0.5 * modulation + np.zeros(shape),
    }
    for year in sorted(set(times.year)):
        sel = times.year == year
        ydir = out / str(year)
        ydir.mkdir(parents=True, exist_ok=True)
        for name, values in winds.items():
            ds = xr.Dataset(
                {name: (("valid_time", "latitude", "longitude"),
                        values[sel].astype("float32"))},
                coords={"valid_time": times[sel], "latitude": lat_desc,
                        "longitude": LON},
            )
            ds[name].attrs["units"] = "m s**-1"
            ds.to_netcdf(ydir / f"era5_{name}_{year}.nc")


def write_gebco(path: Path, depth_m: float = 1000.0) -> None:
    """GEBCO-style ``elevation`` (negative below sea level) on a 1/24 degree grid."""
    lat = np.arange(8.0, 15.5 + 1e-9, 1.0 / 24.0)
    lon = np.arange(108.0, 115.5 + 1e-9, 1.0 / 24.0)
    elevation = np.full((lat.size, lon.size), -depth_m, dtype="float32")
    ds = xr.Dataset(
        {"elevation": (("lat", "lon"), elevation)},
        coords={"lat": lat, "lon": lon},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/synthetic_pureocean"))
    parser.add_argument("--start", default="2019-01-01",
                        help="first day; must equal run.start_date of the case")
    parser.add_argument("--days", type=int, default=10,
                        help="days of HYCOM/ERA5 records (pureocean needs 9 for "
                             "its 7-day time.days window)")
    args = parser.parse_args(argv)
    start = datetime.fromisoformat(args.start)
    write_hycom(args.out / "hycom", start, args.days)
    write_era5(args.out / "era5", start, args.days)
    write_gebco(args.out / "gebco.nc")
    print(f"wrote synthetic pureocean inputs under {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
