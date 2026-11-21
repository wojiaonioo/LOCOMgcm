"""Per-step time-varying ERA5 bulk surface forcing for the cube global run.

Wraps the projected ``[T,6,N,N]`` ERA5 bulk series and,
each step, runs ``exf_bulk_fluxes`` with the cube model SST to produce the MITgcm-
convention surface forcing held in ``CubeSurfaceForcing``:

    tau_x / tau_y   (face-local wind stress, set_cube_wind_stress)
    qnet / qsw / empmr   (set_cube_surface_fluxes, MIT convention)
    penetrating_sw / rho_const_fresh   (shortwave mode and freshwater density)
    bulk   (the produced SurfaceForcing bundle)

Single- and multi-rank unified
------------------------------
The cube stepper expects GLOBAL ``[6,N,N]`` forcing in ``CubeSurfaceForcing`` and
scatters it to the rank-local faces when ``cube_ctx`` is set (``_apply_cube_wind_stress``
/ ``_compute_cube_kpp_fields``).  So this driver always computes the bulk on the
GLOBAL grid:

* multi-rank: ``gather_faces`` all-gathers the surface temp to global ``[6,N,N]``
  (true global SST, one ~36 MB collective/step), computes the bulk globally, sets
  the record's global fields; the stepper scatters.
* single-rank: the state is already global; no gather.

Sign convention: the bulk output is MITgcm upward-positive and feeds
``set_cube_surface_fluxes`` with NO flip.  The geographic ustress/vstress are rotated
to face-local with angleCS/angleSN (``tau_x = aCS*tau_E + aSN*tau_N`` ;
``tau_y = -aSN*tau_E + aCS*tau_N``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from locomgcm.dynamics.bulk_formulae import (
    EXF_BULK_DEFAULTS,
    exf_bulk_fluxes,
    exf_longwave,
    exf_shortwave,
)
from locomgcm.dynamics.surface_flux import surface_forcing_from_bulk
from locomgcm.io.era5 import ERA5BulkInputs, ERA5BulkSource

OUT_FIELDS = ("atemp", "aqh", "uwind", "vwind", "swdown", "lwdown", "precip")


class LazyZarrBulkSource:
    """Lazy zarr-backed cube ERA5 bulk source (drop-in for :class:`ERA5BulkSource`).

    Holds resident float64 ``times`` and seven Zarr array handles. Only the
    two time frames needed for interpolation are loaded, rather than the
    complete ``[n_t,6,N,N]`` series.

    :meth:`sample` reproduces ``ERA5BulkSource.sample`` (``locomgcm/io/era5.py``)
    **verbatim** -- identical ``searchsorted``/``clamp``/weight and the identical
    ``(1-w)*fld[lower] + w*fld[upper]`` expression -- differing only in that the two
    bracketing frames are pulled lazily from zarr (as f32, byte-identical to the npz
    fields) and cached per bracket, so a 6-hour window (hundreds of steps) reads each
    frame pair exactly once.
    """

    _FIELDS = OUT_FIELDS

    def __init__(self, times: torch.Tensor, arrays: dict) -> None:
        self.times = times                      # [n_times] f64, resident (small)
        self._arrays = arrays                   # name -> zarr.Array (lazy handle)
        self.period_seconds = None              # cube bulk forcing is not periodic
        self._bracket_key: tuple[int, int] | None = None
        self._bracket_frames: dict | None = None

    def _bracket(self, li: int, ui: int) -> dict:
        """Cached ``(frame[li], frame[ui])`` torch f32 tensors for every field."""
        if self._bracket_key == (li, ui) and self._bracket_frames is not None:
            return self._bracket_frames
        frames = {}
        for name in self._FIELDS:
            z = self._arrays[name]
            flo = torch.from_numpy(np.ascontiguousarray(z[li]))          # [6,N,N] f32
            fhi = flo if ui == li else torch.from_numpy(np.ascontiguousarray(z[ui]))
            frames[name] = (flo, fhi)
        self._bracket_key = (li, ui)
        self._bracket_frames = frames
        return frames

    def sample(self, model_time_seconds: float | torch.Tensor) -> ERA5BulkInputs:
        """Linearly interpolate every bulk field to *model_time_seconds*.

        Bit-for-bit identical to ``ERA5BulkSource.sample``; only the two bracketing
        frames are read lazily from zarr instead of from a resident tensor.
        """
        t = torch.as_tensor(
            model_time_seconds, dtype=self.times.dtype, device=self.times.device
        )
        if self.period_seconds is not None and self.period_seconds > 0.0:
            t = t % self.period_seconds
        t = torch.clamp(t, self.times[0], self.times[-1])
        upper = torch.clamp(
            torch.searchsorted(self.times, t.unsqueeze(0), right=False).squeeze(),
            0, self.times.shape[0] - 1,
        )
        lower = torch.clamp(upper - 1, 0, self.times.shape[0] - 1)
        t_low, t_high = self.times[lower], self.times[upper]
        diff = torch.where(t_high == t_low, torch.ones_like(t_high), t_high - t_low)
        w = torch.where(t_high == t_low, torch.zeros_like(diff), (t - t_low) / diff)
        frames = self._bracket(int(lower), int(upper))
        out = {}
        for name in self._FIELDS:
            flo, fhi = frames[name]
            out[name] = (1.0 - w) * flo + w * fhi
        zero = torch.zeros_like(out["precip"])
        return ERA5BulkInputs(**out, runoff=zero, lwflux=zero, swflux=zero)


class CubeBulkForcing:
    """Time-varying ERA5 bulk forcing installed in CubeSurfaceForcing each step."""

    def __init__(
        self,
        source: "ERA5BulkSource | LazyZarrBulkSource",
        angle_cs: torch.Tensor,
        angle_sn: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
        rho_const_fresh: float = 1035.0,
    ) -> None:
        self.source = source                 # cube-shaped fields, kept on CPU
        self.angle_cs = angle_cs.to(device=device, dtype=dtype)  # [6,N,N]
        self.angle_sn = angle_sn.to(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
        self.rho_const_fresh = float(rho_const_fresh)

    @classmethod
    def from_npz(
        cls,
        npz_path: str | Path,
        *,
        device: torch.device,
        dtype: torch.dtype,
        rho_const_fresh: float = 1035.0,
    ) -> "CubeBulkForcing":
        z = np.load(npz_path)
        # Source fields stay on CPU (f32) to keep them off the GPU; sampled frames
        # are upcast + moved per step.  sample() is shape-agnostic (time axis only).
        times = torch.from_numpy(z["times"].astype(np.float64))
        fields = {f: torch.from_numpy(z[f]) for f in OUT_FIELDS}  # [T,6,N,N] f32 CPU
        dummy = torch.zeros(1, dtype=torch.float64)
        source = ERA5BulkSource(
            times=times, time_values=tuple(str(int(t)) for t in z["times"]),
            lat=dummy, lon=dummy, period_seconds=None, **fields,
        )
        angle_cs = torch.from_numpy(z["angleCS"].astype(np.float64))
        angle_sn = torch.from_numpy(z["angleSN"].astype(np.float64))
        return cls(source, angle_cs, angle_sn,
                   device=device, dtype=dtype, rho_const_fresh=rho_const_fresh)

    @classmethod
    def from_zarr(
        cls,
        zarr_path: str | Path,
        *,
        device: torch.device,
        dtype: torch.dtype,
        rho_const_fresh: float = 1035.0,
    ) -> "CubeBulkForcing":
        """Streaming counterpart of :meth:`from_npz`.

        Opens a chunked Zarr group with ``times``, ``angleCS`` and ``angleSN``
        loaded into memory. The seven forcing fields remain lazy Zarr arrays
        in :class:`LazyZarrBulkSource`, which loads one time bracket at a time.
        Both input formats use the same ``apply`` method.
        """
        import zarr

        root = zarr.open_group(str(zarr_path), mode="r")
        # times/angle small -> resident (mirrors from_npz dtypes exactly).
        times = torch.from_numpy(np.asarray(root["times"][:]).astype(np.float64))
        arrays = {f: root[f] for f in OUT_FIELDS}  # lazy zarr.Array handles (not loaded)
        source = LazyZarrBulkSource(times=times, arrays=arrays)
        angle_cs = torch.from_numpy(np.asarray(root["angleCS"][:]).astype(np.float64))
        angle_sn = torch.from_numpy(np.asarray(root["angleSN"][:]).astype(np.float64))
        return cls(source, angle_cs, angle_sn,
                   device=device, dtype=dtype, rho_const_fresh=rho_const_fresh)

    def _global_sst(self, state, cube_ctx) -> torch.Tensor:
        """Surface temp [6,N,N] on the global face order (gather when multi-rank)."""
        sst_local = state.temp.data[:, 0].contiguous()  # [n_local,N,N] (or [6,..] single)
        if cube_ctx is None:
            return sst_local
        from locomgcm.geometry.cube_decomposition import gather_faces
        return gather_faces(sst_local, cube_ctx)

    def _global_surface_salinity(self, state, cube_ctx) -> torch.Tensor:
        """Surface salinity [6,N,N] on the global face order."""
        salinity_local = state.salt.data[:, 0].contiguous()
        if cube_ctx is None:
            return salinity_local
        from locomgcm.geometry.cube_decomposition import gather_faces
        return gather_faces(salinity_local, cube_ctx)

    def apply(self, stepper, state, time_seconds: float, cube_ctx=None) -> None:
        """Compute bulk fluxes at ``time_seconds`` and replace CubeSurfaceForcing inputs."""
        sst = self._global_sst(state, cube_ctx).to(dtype=self.dtype)  # [6,N,N] degC

        atmo = self.source.sample(float(time_seconds))  # cube [6,N,N] fields, CPU f32

        def dev(x: torch.Tensor) -> torch.Tensor:
            return x.to(device=self.device, dtype=self.dtype)

        flx = exf_bulk_fluxes(
            dev(atmo.atemp), dev(atmo.aqh), dev(atmo.uwind), dev(atmo.vwind), dev(atmo.precip),
            sst_degc=sst,
            lwflux=exf_longwave(sst + EXF_BULK_DEFAULTS.cen2kel, dev(atmo.lwdown),
                                constants=EXF_BULK_DEFAULTS),
            swflux=exf_shortwave(dev(atmo.swdown), constants=EXF_BULK_DEFAULTS),
            runoff=dev(atmo.runoff), constants=EXF_BULK_DEFAULTS,
            rho_const_fresh=self.rho_const_fresh,
        )

        # Geographic (east/north) stress -> face-local (same as make_cs192_prep wind).
        tau_x = self.angle_cs * flx.ustress + self.angle_sn * flx.vstress
        tau_y = -self.angle_sn * flx.ustress + self.angle_cs * flx.vstress
        stepper.set_cube_wind_stress(tau_x, tau_y)
        rho_const = float(stepper.cube_static_state.rho_const)
        forcing = surface_forcing_from_bulk(
            flx.qnet,
            flx.qsw,
            flx.empmr,
            rho_const=rho_const,
            rho_const_fresh=self.rho_const_fresh,
            convert_fw2salt=-1.0,
            surface_salinity=self._global_surface_salinity(state, cube_ctx),
        )
        stepper.set_cube_surface_fluxes(forcing)
