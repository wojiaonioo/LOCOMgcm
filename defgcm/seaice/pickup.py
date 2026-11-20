"""Native big-endian MIT pickup codec and explicit restart reconstruction.

Codec roundtrips preserve metadata and signed zeros. Reconstruction deliberately
uses MIT restart semantics for unrecorded histories; it is a separate operation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import re
import struct

import torch

from .records import Clock, DynamicsCarry, OfflineCarry, SeaiceCarry, ThsiceState, THSICE_FIELDS
from .space import SeaiceSpace

OCEAN_NAMES = ("Uvel", "Vvel", "Theta", "Salt", "GtNm1", "EtaN", "dEtaHdt", "EtaH")
SEAICE_NAMES = ("siTICE", "siAREA", "siHEFF", "siHSNOW", "siUICE", "siVICE")


@dataclass(frozen=True)
class NativeFile:
    stem: str
    names: tuple[str, ...]
    values: tuple[torch.Tensor, ...]
    iteration: int
    time: float | None
    meta: str


@dataclass(frozen=True)
class NativePickup:
    ocean: NativeFile
    seaice: NativeFile | None
    thsice: NativeFile | None
    chain: str


def read_binary(path: Path, shape: tuple[int, ...], *, device="cpu") -> torch.Tensor:
    """I/O boundary only; preserve IEEE finite values, including negative zero."""
    data = Path(path).read_bytes()
    count = 1
    for size in shape:
        count *= size
    if len(data) != count * 8:
        raise ValueError(f"{path}: expected {count * 8} bytes, got {len(data)}")
    return torch.tensor(struct.unpack(f">{count}d", data), dtype=torch.float64, device=device).reshape(shape)


def write_binary(path: Path, values: torch.Tensor) -> None:
    # Observation boundary; no NumPy/detach in scientific calculations.
    if values.dtype != torch.float64:
        raise ValueError("native binary requires float64")
    flat = values.cpu().reshape(-1).tolist()
    Path(path).write_bytes(struct.pack(f">{len(flat)}d", *flat))


def _number(meta: str, key: str, *, required=True):
    match = re.search(rf"\b{key}\s*=\s*\[\s*([^]]+)\]", meta)
    if match is None:
        if required:
            raise ValueError(f"pickup metadata lacks {key}")
        return None
    return match.group(1).strip()


def read_native_file(directory: Path, stem: str, names: tuple[str, ...], *, suffix="ckptA", device="cpu") -> NativeFile:
    prefix = Path(directory) / f"{stem}.{suffix}"
    meta = Path(str(prefix) + ".meta").read_text()
    if int(_number(meta, "nDims")) != 2 or "float64" not in _number(meta, "dataprec"):
        raise ValueError("pickup requires two-dimensional float64 native records")
    dims = tuple(int(v) for v in re.findall(r"\d+", _number(meta, "dimList")))
    if dims != (80, 1, 80, 42, 1, 42):
        raise ValueError(f"incompatible pickup layout {dims}")
    count = int(_number(meta, "nrecords"))
    match = re.search(r"fldList\s*=\s*\{([^}]+)\}", meta)
    if stem == "pickup_ic":
        if match or _number(meta, "timeInterval", required=False) is not None:
            raise ValueError("native ic metadata has neither fldList nor timeInterval")
    elif match is None or tuple(v.strip() for v in re.findall(r"'([^']*)'", match.group(1))) != names:
        raise ValueError(f"{stem}: incompatible native field order")
    if count != len(names):
        raise ValueError(f"{stem}: expected {len(names)} records, got {count}")
    values = read_binary(Path(str(prefix) + ".data"), (count, 42, 80), device=device)
    interval = _number(meta, "timeInterval", required=False)
    return NativeFile(stem, names, tuple(values.unbind()), int(_number(meta, "timeStepNumber")),
                      None if interval is None else float(interval.replace("D", "E")), meta)


def read_pickup(directory: Path, *, chain: str, nIter0: int, deltaT: float,
                startTime: float, pickupSuff="ckptA", device="cpu") -> NativePickup:
    """Validate common generation, then leave native records unmodified."""
    if chain not in ("T", "L", "H", "B"):
        raise ValueError(chain)
    if chain == "B" and nIter0 % 2:
        raise ValueError("base native restarts require an even dynamics entry")
    onames = tuple(n for n in OCEAN_NAMES if chain != "L" or n != "GtNm1")
    ocean = read_native_file(directory, "pickup", onames, suffix=pickupSuff, device=device)
    si = None if chain == "H" else read_native_file(directory, "pickup_seaice", ("siUICE", "siVICE") if chain == "B" else SEAICE_NAMES, suffix=pickupSuff, device=device)
    th = read_native_file(directory, "pickup_ic", THSICE_FIELDS, suffix=pickupSuff, device=device) if chain in "HB" else None
    if any(f.iteration != nIter0 for f in (ocean, si, th) if f is not None):
        raise ValueError("pickup files do not share nIter0")
    if ocean.time != nIter0 * deltaT or startTime != ocean.time:
        raise ValueError("pickup ocean timeInterval must equal startTime and nIter0*deltaT")
    if si is not None and si.time != ocean.time:
        raise ValueError("seaice timeInterval differs from ocean")
    if "GtNm1" in ocean.names and bool((ocean.values[ocean.names.index("GtNm1")] != 0).any()):
        raise ValueError("this host only restores zero GtNm1 history")
    return NativePickup(ocean, si, th, chain)


def write_pickup(directory: Path, pickup: NativePickup, *, pickupSuff="ckptA") -> None:
    """Write an unmodified decoded record byte-for-byte, or generated records."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for native in (pickup.ocean, pickup.seaice, pickup.thsice):
        if native is None:
            continue
        prefix = directory / f"{native.stem}.{pickupSuff}"
        write_binary(Path(str(prefix) + ".data"), torch.stack(native.values))
        Path(str(prefix) + ".meta").write_text(native.meta)


def metadata(names: tuple[str, ...], iteration: int, time: float, *, ic=False) -> str:
    result = (" nDims = [   2 ];\n dimList = [\n    80,    1,   80,\n    42,    1,   42\n ];\n"
              f" dataprec = [ 'float64' ];\n nrecords = [ {len(names):10d} ];\n timeStepNumber = [ {iteration:10d} ];\n")
    if not ic:
        result += f" timeInterval = [ {time:19.12E} ];\n nFlds = [ {len(names):4d} ];\n fldList = {{\n"
        result += " " + " ".join(f"'{name:8s}'" for name in names) + "\n };\n"
    return result


def encode_carry(offline: OfflineCarry, *, chain: str, space: SeaiceSpace) -> NativePickup:
    """write_pickup.F; seaice_write_pickup.F; thsice_write_pickup.F:54-62."""
    carry = offline.ice
    if offline.ocean.clock != carry.clock:
        raise ValueError("ocean and ice clocks differ")
    ocean, si, th = offline.ocean, carry.seaice, carry.thsice
    values = (ocean.uVel, ocean.vVel, ocean.theta, ocean.salt, torch.zeros_like(ocean.gtNm1), ocean.etaN, ocean.dEtaHdt, ocean.etaHnm1)
    pairs = tuple((n, v) for n, v in zip(OCEAN_NAMES, values) if chain != "L" or n != "GtNm1")

    def make(stem, pairs):
        names, fields = zip(*pairs)
        arrays = tuple(space.to_global(field).reshape(42, 80) for field in fields)
        return NativeFile(stem, tuple(names), arrays, carry.clock.myIter,
                          None if stem == "pickup_ic" else carry.clock.myTime,
                          metadata(tuple(names), carry.clock.myIter, carry.clock.myTime, ic=stem == "pickup_ic"))

    ocean_file = make("pickup", pairs)
    ice_pairs = tuple(zip(SEAICE_NAMES, (si.TICES, si.AREA, si.HEFF, si.HSNOW, si.UICE, si.VICE)))
    seaice_file = None if chain == "H" else make("pickup_seaice", ice_pairs[-2:] if chain == "B" else ice_pairs)
    thsice_file = make("pickup_ic", tuple((name, getattr(th, name)) for name in THSICE_FIELDS)) if chain in "HB" else None
    return NativePickup(ocean_file, seaice_file, thsice_file, chain)


def restore_carry(pickup: NativePickup, offline: OfflineCarry, *, space: SeaiceSpace,
                  eccentricity: float | torch.Tensor = 2., rho_ice=900., rho_snow=330.) -> OfflineCarry:
    """Apply R/I3; reset DWATN and reconstruct viscosity from restored HEFF.

    seaice_read_pickup.F:318-322; thsice_ini_vars.F:160-176;
    seaice_init_varia.F:79,435-443. EtaH caches reset on native restart.
    """
    initialized = offline.ice
    names = {"Uvel": "uVel", "Vvel": "vVel", "Theta": "theta", "Salt": "salt", "GtNm1": "gtNm1", "EtaN": "etaN"}
    ocean = {names[n]: space.from_global(v) for n, v in zip(pickup.ocean.names, pickup.ocean.values) if n in names}
    ocean = space.exchange("R", ocean, chain=pickup.chain)
    zero = torch.zeros_like(initialized.sIceLoad)
    clock = Clock(pickup.ocean.iteration, pickup.ocean.time, initialized.clock.generation + 1)
    ocean.update(gtNm1=zero.clone(), etaH=zero.clone(), etaHnm1=zero.clone(), dEtaHdt=zero.clone(), clock=clock)
    restored_ocean = replace(offline.ocean, **ocean)
    si = initialized.seaice
    if pickup.seaice is not None:
        snames = dict(zip(SEAICE_NAMES, ("TICES", "AREA", "HEFF", "HSNOW", "UICE", "VICE")))
        data = {snames[n]: space.from_global(v.unsqueeze(0) if n == "siTICE" else v) for n, v in zip(pickup.seaice.names, pickup.seaice.values)}
        si = replace(si, **space.exchange("R", data, chain=pickup.chain))
    th = initialized.thsice
    if pickup.thsice is not None:
        data = {n: space.from_global(v) for n, v in zip(THSICE_FIELDS, pickup.thsice.values)}
        th = ThsiceState(**space.exchange("I3", data, chain=pickup.chain))
    zeta = si.HEFF * 1.e11
    ecc = torch.as_tensor(eccentricity, dtype=torch.float64, device=zeta.device)
    denominator = torch.where(ecc != 0, ecc * ecc, torch.ones_like(ecc))
    eta = torch.where(ecc != 0, zeta / denominator, torch.zeros_like(zeta))
    dynamics = DynamicsCarry(zero.clone(), eta, zeta)
    load = (th.snowHeight * rho_snow + th.iceHeight * rho_ice) * th.iceMask if pickup.chain == "B" else zero.clone()
    return OfflineCarry(SeaiceCarry(si, th, dynamics, load, clock), restored_ocean)
