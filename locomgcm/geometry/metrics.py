from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

from locomgcm.accel.constant_cache import _build_eager


#: Dtype every *geometry decision* is evaluated in, whatever the run precision
#: is.  Mirrors ``grid.py``'s ``_GEOMETRY_DTYPE`` (defined separately because
#: ``grid`` imports this module, not the other way round); the rationale is in
#: ``grid.py``'s ``_GEOMETRY_DTYPE_NOTE``.
_DECISION_DTYPE = torch.float64


def _cache_device_key(device: torch.device | str) -> str:
    return str(torch.device(device))


def spherical_centre_latitudes(
    ygOrigin_deg: float, delY_deg: float, ny: int
) -> tuple[float, ...]:
    """Cell-centre latitudes ``yC(j) = ygOrigin + (j + 1/2) * delY`` [deg].

    Host float64 arithmetic, spelled as the ``$coriolis_profile_c`` sentinel
    spells it (``io/runner/_canonical_helpers.py`` ``runtime_kwargs``), so the
    two give the same bits for the same origin and spacing.  MITgcm accumulates
    yG tile by tile before averaging (``ini_local_grid.F:135-163``); that order
    differs from this one by rounding only.
    """
    origin, spacing = float(ygOrigin_deg), float(delY_deg)
    return tuple(origin + (j + 0.5) * spacing for j in range(int(ny)))


def spherical_corner_latitudes(
    ygOrigin_deg: float, delY_deg: float, ny: int
) -> tuple[float, ...]:
    """Corner latitudes ``yG(j) = ygOrigin + j * delY`` [deg], j = 0..ny.

    Spelled as the ``$coriolis_profile_g`` sentinel spells it, for the same
    reason as :func:`spherical_centre_latitudes`.
    """
    origin, spacing = float(ygOrigin_deg), float(delY_deg)
    return tuple(origin + j * spacing for j in range(int(ny) + 1))


@dataclass(frozen=True)
class MetricSet:
    """Grid metric tensors.

    Legacy scalar API (``dx_scalar`` / ``dy_scalar`` and the ``dx_center``/
    ``dy_center``/``cell_area``/``face_area_u``/``face_area_v`` 2-D tensors) is
    retained so existing Cartesian / rectangular test-cases keep running
    unchanged. Callers that need **latitude-varying** metrics for
    spherical-polar alignment with MITgcm must use
    :meth:`spherical_polar` and read the row-wise accessors:

    * ``dxG_row_for`` — ``rSphere · cos(yG) · dlon`` (length, varies with j)
    * ``dyG_scalar``  — ``rSphere · dlat`` (length, constant for uniform grid)
    * ``cell_area``    — ``rA`` = exact spherical surface-integral area per cell

    MITgcm ``GAD_ADVECTION`` uses the following bitwise formulation for
    tracer horizontal transport (``gad_advection.F:297-504``):

        xA = dyG · hFacW · drF         (u-face volume cross-section area)
        yA = dxG · hFacS · drF         (v-face volume cross-section area)
        uTrans = u · xA                (m^3/s through u-face)
        vTrans = v · yA                (m^3/s through v-face)
        dT/dt  = -(uTrans_e - uTrans_w)·recip_rA·recip_hFacC - ...

    To match MITgcm bit-for-bit on a spherical polar grid, flux-form
    operators must divide the flux-difference by ``rA`` (``cell_area``) rather
    than by a scalar ``dx``, and multiply v-face fluxes by the per-row
    ``dxG``.
    """

    dx_scalar: float
    dy_scalar: float
    dx_center: torch.Tensor     # [ny, nx]   — dxF, zonal cell width through center.
                                #              For spherical_polar this is lat-varying.
    dy_center: torch.Tensor     # [ny, nx]   — dyF, meridional cell width through center.
    cell_area: torch.Tensor     # [ny, nx]   — rA, the exact spherical cell area
                                #              (lat-varying under spherical_polar).
    cell_volume: torch.Tensor   # [ny, nx]   — rA · depth (for bathymetry) or dx*dy*depth
    depth_center: torch.Tensor  # [ny, nx]   — ocean depth at each column
    face_area_u: torch.Tensor   # [ny, nx+1] — legacy Cartesian "u-face volume area"
                                #              (depth*dy). Kept for backwards compat;
                                #              not used by flux-form operators.
    face_area_v: torch.Tensor   # [ny+1, nx] — same (depth*dx).
    # New row-wise (ny,) tensors for spherical polar alignment.  For rectangular
    # grids these are filled with constants; for spherical_polar they carry the
    # true MITgcm per-latitude values.
    dxG_row: torch.Tensor       # [ny]  — MITgcm dxG(i,j) for non-rotated grid,
                                #         at the south edge of cells 0..ny-1
                                #         (= ny-subset of the v-face rows).
                                #         Shape matches cell-row consumers.
    dxC_row: torch.Tensor       # [ny]  — MITgcm dxC(i,j) for non-rotated grid
    dyG_scalar_1d: float        # constant — MITgcm dyG value (independent of j for uniform)
    # (ny+1,)-length fields cover every v-face and corner row, including the
    # north OB row at yG[ny]. curl_uv_to_corner uses circulation / rAz;
    # _vorticity_flux_u weights both v-face rows of each u-face by dxG.
    # The staggered metric definitions below determine each denominator.
    dxG_row_ext: torch.Tensor   # [ny+1]  — MITgcm dxG at every v-face row yG[0..ny]
    rAz_row: torch.Tensor       # [ny+1]  — MITgcm rAz at every corner row yG[0..ny]
    # Exact Cartesian-vector inputs and staggered backing tensors. Legacy scalar and
    # spherical constructors leave these unset and retain their original code paths.
    del_x: torch.Tensor | None = None       # [nx], exact global Cartesian delX
    del_y: torch.Tensor | None = None       # [ny], exact global Cartesian delY
    dxF: torch.Tensor | None = None         # [ny, nx]
    dyF: torch.Tensor | None = None         # [ny, nx]
    dxG: torch.Tensor | None = None         # [ny+1, nx]
    dyG: torch.Tensor | None = None         # [ny, nx+1]
    dxC: torch.Tensor | None = None         # [ny, nx+1]
    dyC: torch.Tensor | None = None         # [ny+1, nx]
    dxV: torch.Tensor | None = None         # [ny+1, nx+1]
    dyU: torch.Tensor | None = None         # [ny+1, nx+1]
    rA: torch.Tensor | None = None          # [ny, nx]
    rAw: torch.Tensor | None = None         # [ny, nx+1]
    rAs: torch.Tensor | None = None         # [ny+1, nx]
    rAz: torch.Tensor | None = None         # [ny+1, nx+1]
    # Cell-centre and corner latitudes of each row [deg] on a spherical-polar
    # grid, as host float64 scalars (never narrowed to the run dtype); None on
    # other grids.  KPP's Ekman depth takes fCori at the centre latitude
    # (kpp_calc.F:555, ini_cori.F:84-95) when the momentum profile is the
    # spherical map at the corner latitudes.  See
    # :func:`spherical_centre_latitudes` and :func:`spherical_corner_latitudes`.
    yC_row: tuple[float, ...] | None = None  # [ny]
    yG_row: tuple[float, ...] | None = None  # [ny+1]
    _is_uniform_x: bool = field(init=False, repr=False, compare=False)
    _is_uniform_y: bool = field(init=False, repr=False, compare=False)
    _has_uniform_row_geometry: bool = field(init=False, repr=False, compare=False)
    west_dx: float = field(init=False, repr=False, compare=False)
    _meridional_lengths: dict[torch.dtype, float] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Execution memo, excluded from frozen-dataclass field expansion.
        # Its consumers still guard the entries they actually read.
        object.__setattr__(self, "_runtime_cache", {})
        # Fixed geometry decisions belong to construction, not traced tensor consumers.
        for axis in ("x", "y"):
            spacing = getattr(self, f"del_{axis}")
            uniform = spacing is None or torch.equal(spacing, spacing[:1].expand_as(spacing))
            object.__setattr__(self, f"_is_uniform_{axis}", uniform)
        rows = (self.dxC_row, self.dxG_row_ext, self.rAz_row)
        # Non-row-representable metrics retain the existing cube routing verdict.
        verdict = any(t.dim() != 1 for t in rows) or all(
            bool(torch.equal(t, t[:1].expand_as(t))) for t in rows
        )
        object.__setattr__(self, "_has_uniform_row_geometry", verdict)
        object.__setattr__(
            self, "west_dx",
            float(self.del_x[0].item()) if self.del_x is not None else self.dx_scalar,
        )
        lengths = {}
        if self.del_y is not None and not self._is_uniform_y:
            for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                lengths[dtype] = float(self.del_y.to(dtype=dtype).sum().item())
        object.__setattr__(self, "_meridional_lengths", lengths)

    def meridional_length_value(self, *, dtype: torch.dtype, device: torch.device) -> float:
        """Preserve the original dtype/device reduction and Python-scalar arithmetic.

        Runtime grids are colocated with their state. Standalone consumers on a
        different device retain the original reduction on that device.
        """
        if self.del_y is None:
            raise ValueError("meridional_length_value requires del_y")
        if device == self.del_y.device and dtype in self._meridional_lengths:
            return self._meridional_lengths[dtype]
        return float(self.del_y.to(dtype=dtype, device=device).sum().item())

    @staticmethod
    def _cartesian_spacing_vector(
        *,
        values: torch.Tensor | Sequence[float] | None,
        extent: float | None,
        count: int,
        axis: str,
        device: str,
    ) -> torch.Tensor:
        if values is not None and extent is not None:
            raise ValueError(f"del_{axis} and l{axis} are mutually exclusive")
        if values is None and extent is None:
            raise ValueError(f"exactly one of del_{axis} or l{axis} is required")
        if count <= 0:
            raise ValueError(f"n{axis} must be positive, got {count}")

        if values is None:
            spacing = torch.full(
                (count,),
                float(extent) / count,
                dtype=torch.float64,
                device=device,
            )
        else:
            try:
                spacing = torch.as_tensor(
                    values,
                    dtype=torch.float64,
                    device=device,
                ).detach().clone()
            except (TypeError, ValueError, RuntimeError) as exc:
                raise ValueError(f"del_{axis} must be a 1-D float64 tensor or sequence") from exc

        if spacing.ndim != 1:
            raise ValueError(f"del_{axis} must be 1-D, got shape {tuple(spacing.shape)}")
        if spacing.numel() != count:
            raise ValueError(
                f"del_{axis} length must equal n{axis}={count}, got {spacing.numel()}"
            )
        if not bool(torch.isfinite(spacing).all().item()):
            raise ValueError(f"del_{axis} values must all be finite")
        if not bool((spacing > 0.0).all().item()):
            raise ValueError(f"del_{axis} values must all be strictly positive")
        return spacing

    @classmethod
    def _cartesian_vector(
        cls,
        *,
        nx: int,
        ny: int,
        lx: float | None,
        ly: float | None,
        del_x: torch.Tensor | Sequence[float] | None,
        del_y: torch.Tensor | Sequence[float] | None,
        depth: float | None,
        bathymetry: torch.Tensor | None,
        device: str,
        dtype: torch.dtype,
    ) -> "MetricSet":
        del_x_1d = cls._cartesian_spacing_vector(
            values=del_x,
            extent=lx,
            count=nx,
            axis="x",
            device=device,
        )
        del_y_1d = cls._cartesian_spacing_vector(
            values=del_y,
            extent=ly,
            count=ny,
            axis="y",
            device=device,
        )

        # ini_cartesian_grid.F:88-102: face-through-center and boundary lengths.
        dxF = del_x_1d.view(1, nx).expand(ny, nx).contiguous()
        dyF = del_y_1d.view(ny, 1).expand(ny, nx).contiguous()
        dxG = del_x_1d.view(1, nx).expand(ny + 1, nx).contiguous()
        dyG = del_y_1d.view(ny, 1).expand(ny, nx + 1).contiguous()

        # ini_local_grid.F:144-150 supplies the preceding global item by MOD.
        half = torch.tensor(0.5, dtype=torch.float64, device=device)
        dxC_mod = half * (del_x_1d + torch.roll(del_x_1d, shifts=1, dims=0))
        dxC_mod = torch.cat((dxC_mod, dxC_mod[:1]), dim=0)
        dyC_mod = half * (del_y_1d + torch.roll(del_y_1d, shifts=1, dims=0))
        dyC_mod = torch.cat((dyC_mod, dyC_mod[:1]), dim=0)

        # ini_cartesian_grid.F:108-132: centre and velocity-point distances.
        dxC = dxC_mod.view(1, nx + 1).expand(ny, nx + 1).contiguous()
        dyC = dyC_mod.view(ny + 1, 1).expand(ny + 1, nx).contiguous()
        dxV = dxC_mod.view(1, nx + 1).expand(ny + 1, nx + 1).contiguous()
        dyU = dyC_mod.view(ny + 1, 1).expand(ny + 1, nx + 1).contiguous()

        # ini_cartesian_grid.F:134-140: preserve the four explicit products.
        rA = dxF * dyF
        rAw = dxC * dyG
        rAs = dxG * dyC
        rAz = dxV * dyU

        if bathymetry is None:
            if depth is None:
                raise ValueError("depth is required when bathymetry is not provided")
            bathy = torch.full((ny, nx), float(depth), dtype=torch.float64, device=device)
            cell_volume = rA * float(depth)
            max_depth = float(depth)
        else:
            bathy = bathymetry.to(dtype=torch.float64, device=device)
            cell_volume = bathy * dxF
            cell_volume = cell_volume * dyF
            max_depth = float(bathy.max().item())

        face_area_u = max_depth * dyG
        face_area_v = max_depth * dxG

        # Construct geometry in float64, then narrow the stored products just
        # as the other grid constructors do. All casts are no-ops for float64.
        return cls(
            dx_scalar=float(del_x_1d[0].item()),
            dy_scalar=float(del_y_1d[0].item()),
            dx_center=dxF.to(dtype=dtype),
            dy_center=dyF.to(dtype=dtype),
            cell_area=rA.to(dtype=dtype),
            cell_volume=cell_volume.to(dtype=dtype),
            depth_center=bathy.to(dtype=dtype),
            face_area_u=face_area_u.to(dtype=dtype),
            face_area_v=face_area_v.to(dtype=dtype),
            dxG_row=dxG[:ny, 0].contiguous().to(dtype=dtype),
            dxC_row=dxC[:, 0].contiguous().to(dtype=dtype),
            dyG_scalar_1d=float(del_y_1d[0].item()),
            dxG_row_ext=dxG[:, 0].contiguous().to(dtype=dtype),
            rAz_row=rAz[:, 0].contiguous().to(dtype=dtype),
            # Source spacing must survive artifact reconstruction in float64.
            del_x=del_x_1d,
            del_y=del_y_1d,
            dxF=dxF.to(dtype=dtype),
            dyF=dyF.to(dtype=dtype),
            dxG=dxG.to(dtype=dtype),
            dyG=dyG.to(dtype=dtype),
            dxC=dxC.to(dtype=dtype),
            dyC=dyC.to(dtype=dtype),
            dxV=dxV.to(dtype=dtype),
            dyU=dyU.to(dtype=dtype),
            rA=rA.to(dtype=dtype),
            rAw=rAw.to(dtype=dtype),
            rAs=rAs.to(dtype=dtype),
            rAz=rAz.to(dtype=dtype),
        )

    @classmethod
    def rectangular(
        cls,
        *,
        nx: int,
        ny: int,
        lx: float | None = None,
        ly: float | None = None,
        depth: float,
        del_x: torch.Tensor | Sequence[float] | None = None,
        del_y: torch.Tensor | Sequence[float] | None = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> "MetricSet":
        if del_x is not None or del_y is not None:
            return cls._cartesian_vector(
                nx=nx,
                ny=ny,
                lx=lx,
                ly=ly,
                del_x=del_x,
                del_y=del_y,
                depth=depth,
                bathymetry=None,
                device=device,
                dtype=dtype,
            )
        dx = lx / nx
        dy = ly / ny
        return cls(
            dx_scalar=float(dx),
            dy_scalar=float(dy),
            dx_center=torch.full((ny, nx), dx, dtype=dtype, device=device),
            dy_center=torch.full((ny, nx), dy, dtype=dtype, device=device),
            cell_area=torch.full((ny, nx), dx * dy, dtype=dtype, device=device),
            cell_volume=torch.full(
                (ny, nx),
                dx * dy * depth,
                dtype=dtype,
                device=device,
            ),
            depth_center=torch.full((ny, nx), depth, dtype=dtype, device=device),
            face_area_u=torch.full((ny, nx + 1), depth * dy, dtype=dtype, device=device),
            face_area_v=torch.full((ny + 1, nx), depth * dx, dtype=dtype, device=device),
            dxG_row=torch.full((ny,), dx, dtype=dtype, device=device),
            dxC_row=torch.full((ny,), dx, dtype=dtype, device=device),
            dyG_scalar_1d=float(dy),
            dxG_row_ext=torch.full((ny + 1,), dx, dtype=dtype, device=device),
            rAz_row=torch.full((ny + 1,), dx * dy, dtype=dtype, device=device),
        )

    @classmethod
    def with_bathymetry(
        cls,
        *,
        nx: int,
        ny: int,
        lx: float | None = None,
        ly: float | None = None,
        bathymetry: torch.Tensor,
        del_x: torch.Tensor | Sequence[float] | None = None,
        del_y: torch.Tensor | Sequence[float] | None = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> "MetricSet":
        """Like ``rectangular`` but with variable ``depth_center`` from bathymetry.

        Parameters
        ----------
        bathymetry : Tensor [ny, nx]
            Ocean depth at each cell (positive = ocean).
        """
        if del_x is not None or del_y is not None:
            return cls._cartesian_vector(
                nx=nx,
                ny=ny,
                lx=lx,
                ly=ly,
                del_x=del_x,
                del_y=del_y,
                depth=None,
                bathymetry=bathymetry,
                device=device,
                dtype=dtype,
            )
        dx = lx / nx
        dy = ly / ny
        bathy = bathymetry.to(dtype=dtype, device=device)
        max_depth = float(bathy.max().item())
        return cls(
            dx_scalar=float(dx),
            dy_scalar=float(dy),
            dx_center=torch.full((ny, nx), dx, dtype=dtype, device=device),
            dy_center=torch.full((ny, nx), dy, dtype=dtype, device=device),
            cell_area=torch.full((ny, nx), dx * dy, dtype=dtype, device=device),
            cell_volume=bathy * dx * dy,
            depth_center=bathy,
            face_area_u=torch.full((ny, nx + 1), max_depth * dy, dtype=dtype, device=device),
            face_area_v=torch.full((ny + 1, nx), max_depth * dx, dtype=dtype, device=device),
            dxG_row=torch.full((ny,), dx, dtype=dtype, device=device),
            dxC_row=torch.full((ny,), dx, dtype=dtype, device=device),
            dyG_scalar_1d=float(dy),
            dxG_row_ext=torch.full((ny + 1,), dx, dtype=dtype, device=device),
            rAz_row=torch.full((ny + 1,), dx * dy, dtype=dtype, device=device),
        )

    @classmethod
    def spherical_polar(
        cls,
        *,
        nx: int,
        ny: int,
        ygOrigin_deg: float,
        delY_deg: float,
        delX_deg: float,
        bathymetry: torch.Tensor | None = None,
        depth: float = 0.0,
        r_sphere: float = 6_370_000.0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> "MetricSet":
        """MITgcm-compatible spherical polar grid metrics.

        Reproduces ``model/src/ini_spherical_polar_grid.F`` bit-for-bit for the
        uniform non-rotated case used by the regional MITgcm comparison runs.
        All formulas use ``rSphere = 6,370,000 m`` by default, matching MITgcm's
        ``ini_parms.F:1355`` default.

        Parameters
        ----------
        nx, ny : int
            Grid dimensions.
        ygOrigin_deg : float
            Latitude of the south-west **corner** (``yG(i=1, j=1)`` in Fortran
            1-indexed, ``yG[0,0]`` in 0-indexed Python).
        delY_deg : float
            Uniform cell size in latitude (degrees).
        delX_deg : float
            Uniform cell size in longitude (degrees).
        bathymetry : Tensor [ny, nx] or None
            Positive ocean depth at each column. If ``None``, uses a flat
            ``depth`` everywhere.
        depth : float
            Reference depth for the flat case. Ignored if ``bathymetry`` is set.
        r_sphere : float
            Earth radius in metres. MITgcm default is ``6_370_000.0``.
        device, dtype :
            Tensor placement.

        MITgcm formulas (``ini_spherical_polar_grid.F``):

        ::

            dxF(i,j) = rSphere · cos(yC(i,j)·deg2rad) · dlon·deg2rad
            dxG(i,j) = rSphere · cos(yG_mid(j)·deg2rad) · dlon·deg2rad
                where yG_mid(j) = 0.5·(yG(i,j)+yG(i+1,j)) → yG(j) for non-rotated
            dxC(i,j) = 0.5 · (dxF(i,j) + dxF(i-1,j))  → dxF for non-rotated uniform
            dyF,dyG,dyC,dyU = rSphere · dlat·deg2rad  (constant)

            rA(i,j) = rSphere² · dlon·deg2rad ·
                      |sin(yG(j+1)·deg2rad) - sin(yG(j)·deg2rad)|
        """
        deg2rad = math.pi / 180.0

        # Corner latitudes yG[0..ny] (0-indexed Python).  MITgcm Fortran 1-indexed
        # yG(i=1,j=1) = ygOrigin, so Python yG[0] = ygOrigin_deg.
        j_idx = torch.arange(ny + 1, dtype=dtype, device=device)
        yG_deg = ygOrigin_deg + j_idx * delY_deg          # [ny+1]

        # Reject unsupported polar metrics: MITgcm zeroes rAs/rAz
        # whose sine band crosses a pole and clamps sub-metre polar dxG
        # (ini_spherical_polar_grid.F:222-224,108-120; zero reciprocals in
        # ini_grid.F).  LOCOMgcm implements none of those clamps, so a grid
        # whose corner bands would cross +/-90 deg must be rejected rather than
        # silently mis-metered.  Band endpoints sit at yG +/- delY/2; exact
        # +/-90 endpoints (tgol-style halo rows) are allowed (MIT tests GT.90).
        #
        # The reach is a *decision*, not storage, so it is computed in float64
        # whatever the run dtype is -- same rule as the wet/dry topology (see
        # grid.py ``_GEOMETRY_DTYPE_NOTE``).  Evaluated in float32 the guard has
        # a false negative: at ygOrigin_deg=89.999986, delY_deg=1e-5, ny=1 the
        # band truly reaches 90.00000100000001 but float32 rounds the corner
        # latitudes back to a reach of 89.99999737, so a polar-crossing grid was
        # silently accepted and metered without the MIT clamps.  The float64
        # path reuses ``yG_deg`` itself, so nothing moves there.
        _yG_guard = (
            yG_deg
            if dtype is _DECISION_DTYPE
            else ygOrigin_deg
            + torch.arange(ny + 1, dtype=_DECISION_DTYPE, device=device) * delY_deg
        )
        _band_reach = float(torch.max(torch.abs(_yG_guard)).item()) + 0.5 * float(delY_deg)
        if _band_reach > 90.0 + 1e-9:
            raise NotImplementedError(
                "spherical_polar grid has corner bands crossing the pole "
                f"(max |yG|+delY/2 = {_band_reach} deg > 90); MITgcm clamps "
                "rAs/rAz/dxG to zero there and LOCOMgcm does not implement "
                "those clamps -- narrow the domain or add MIT polar clamps"
            )

        # Cell centre latitudes yC[0..ny-1].
        # Fortran: yC(i,j) = 0.25·(yG(i,j)+yG(i+1,j)+yG(i,j+1)+yG(i+1,j+1))
        #                 = 0.5·(yG(j)+yG(j+1)) for a non-rotated grid.
        yC_deg = 0.5 * (yG_deg[:-1] + yG_deg[1:])          # [ny]

        dlon_rad = delX_deg * deg2rad
        dlat_rad = delY_deg * deg2rad

        # dxG at the south face of cell j — uses yG_deg[j] (non-rotated simplification
        # of 0.5·(yGloc(i,j)+yGloc(i+1,j)), which reduces to yGloc(j)).
        dxG_row = r_sphere * torch.cos(yG_deg[:ny] * deg2rad) * dlon_rad  # [ny]
        # dxF at cell centre — uses yC_deg[j].
        dxF_row = r_sphere * torch.cos(yC_deg * deg2rad) * dlon_rad       # [ny]
        # dxC at u-face between cells i-1 and i — MITgcm averages dxF(i) and
        # dxF(i-1). For non-rotated uniform grid both are the same per-row value,
        # so dxC_row = dxF_row.
        dxC_row = dxF_row.clone()                                         # [ny]

        # Meridional distances are all equal to ``r_sphere · dlat`` regardless
        # of latitude for a uniform-spaced grid.
        dy_const = r_sphere * dlat_rad                                    # scalar
        dyG_scalar = float(dy_const)

        # Exact spherical cell area (MITgcm ini_spherical_polar_grid.F:187-188):
        #   rA = rSphere² · dlon·deg2rad · |sin((lat+dlat)·deg2rad) - sin(lat·deg2rad)|
        # where lat = 0.5·(yGloc(i,j)+yGloc(i+1,j)) = yG_deg[j] for non-rotated.
        sin_yG = torch.sin(yG_deg * deg2rad)                              # [ny+1]
        rA_row = (r_sphere ** 2) * dlon_rad * torch.abs(
            sin_yG[1:] - sin_yG[:-1]
        )                                                                  # [ny]

        # dxG on every v-face row including the north OB row at yG[ny].
        # Same formula as ``dxG_row`` but evaluated at all ny+1 yG latitudes.
        # Needed by the MITgcm-native ``_vorticity_flux_u`` (``mom_vi_u_coriolis.F``
        # scheme 1), which weights the south v-face row j by ``dxG(j)`` and the
        # north v-face row j+1 by ``dxG(j+1)`` — when j = ny-1 the latter is
        # ``dxG(ny)`` which is not covered by the (ny,)-long ``dxG_row``.
        dxG_row_ext = r_sphere * torch.cos(yG_deg * deg2rad) * dlon_rad   # [ny+1]

        # vorticity point area rAz(i,j) at corner rows yG[0..ny].
        # MITgcm ``ini_spherical_polar_grid.F:228-239``:
        #     lat  = 0.5·(yGloc(i,j)+yGloc(i,j+1))
        #     dlat = 0.5·(delYloc(j)+delYloc(j-1))
        #     rAz(i,j) = rSphere² · dlon·deg2rad ·
        #                |sin(lat·deg2rad) - sin((lat-dlat)·deg2rad)|
        #
        # For the non-rotated uniform grid used here, ``delYloc`` is a constant
        # ``delY_deg`` everywhere and ``yGloc`` depends only on j, so the
        # integration band reduces to
        #     [yG_deg[j] - delY/2,  yG_deg[j] + delY/2]
        # i.e. the band is centred on yG[j] with half-width ``delY/2``.
        # rAz therefore only varies along j — we store it as a 1-D
        # ``rAz_row`` of length ny+1 (broadcast along i by the consumer).
        lat_top_z = yG_deg + 0.5 * delY_deg                               # [ny+1]
        lat_bot_z = yG_deg - 0.5 * delY_deg                               # [ny+1]
        rAz_row = (r_sphere ** 2) * dlon_rad * torch.abs(
            torch.sin(lat_top_z * deg2rad) - torch.sin(lat_bot_z * deg2rad)
        )                                                                  # [ny+1]

        # Broadcast row-wise metrics to (ny, nx) 2-D tensors.
        dxC_2d = dxC_row.view(ny, 1).expand(ny, nx).contiguous()
        dyC_2d = torch.full((ny, nx), dy_const, dtype=dtype, device=device)
        rA_2d = rA_row.view(ny, 1).expand(ny, nx).contiguous()

        # Depth handling.
        if bathymetry is not None:
            bathy = bathymetry.to(dtype=dtype, device=device)
        else:
            bathy = torch.full((ny, nx), float(depth), dtype=dtype, device=device)

        cell_volume = bathy * rA_2d
        depth_center = bathy

        # Legacy face_area_u / face_area_v fields (depth · perpendicular length,
        # unused by any operator but kept for backwards compat).
        max_depth = float(bathy.max().item()) if bathy.numel() > 0 else 0.0
        face_area_u = torch.full((ny, nx + 1), max_depth * dy_const, dtype=dtype, device=device)
        face_area_v_col = dxG_row * max_depth                              # [ny]
        # Need shape (ny+1, nx): append a north-halo row (periodic copy of row 0)
        face_area_v = torch.zeros((ny + 1, nx), dtype=dtype, device=device)
        for j in range(ny):
            face_area_v[j, :] = face_area_v_col[j]
        face_area_v[ny, :] = face_area_v_col[0]  # periodic halo (matches EXCH_XY_RL)

        # Representative scalars: use the domain-centre row so legacy code paths
        # that read ``dx_scalar`` pick a reasonable single number. Flux-form
        # operators should NOT rely on these for bitwise alignment; they should
        # consume dxG_row / cell_area instead.
        dx_scalar_repr = float(dxC_row[ny // 2].item())

        return cls(
            dx_scalar=dx_scalar_repr,
            dy_scalar=dyG_scalar,
            dx_center=dxC_2d,
            dy_center=dyC_2d,
            cell_area=rA_2d,
            cell_volume=cell_volume,
            depth_center=depth_center,
            face_area_u=face_area_u,
            face_area_v=face_area_v,
            dxG_row=dxG_row,
            dxC_row=dxC_row,
            dyG_scalar_1d=dyG_scalar,
            dxG_row_ext=dxG_row_ext,
            rAz_row=rAz_row,
            yC_row=spherical_centre_latitudes(ygOrigin_deg, delY_deg, ny),
            yG_row=spherical_corner_latitudes(ygOrigin_deg, delY_deg, ny),
        )

    @property
    def is_uniform_x(self) -> bool:
        return self._is_uniform_x

    @property
    def is_uniform_y(self) -> bool:
        return self._is_uniform_y

    @property
    def has_uniform_row_geometry(self) -> bool:
        """True iff the per-row metric rows carry no j-dependence.

        ``is_uniform_x/y`` only inspect the OPTIONAL Cartesian spacing vectors,
        so a spherical-polar grid (which leaves them ``None``) reports uniform
        even though ``dxC/dxG/rAz`` vary with latitude — that misrouted tracer
        diffusion onto the Cartesian row shortcut.
        Row-shortcut consumers must ALSO require this predicate.  Uniform
        rectangular constructors store constant rows (True); spherical-polar
        and native 2-D (cube) metric sets vary (False).
        """
        return self._has_uniform_row_geometry

    @property
    def dx(self) -> float:
        if not self.is_uniform_x:
            raise RuntimeError(
                "MetricSet.dx is undefined for non-uniform x spacing; "
                "use del_x or a staggered metric accessor"
            )
        return self.dx_scalar

    @property
    def dy(self) -> float:
        if not self.is_uniform_y:
            raise RuntimeError(
                "MetricSet.dy is undefined for non-uniform y spacing; "
                "use del_y or a staggered metric accessor"
            )
        return self.dy_scalar

    def _runtime_memo(
        self,
        key: tuple[str, torch.dtype, str],
        build: Callable[[], torch.Tensor],
    ) -> torch.Tensor:
        """Route cold builds outside traced CUDAGraph ownership.
        The eager builder owns the tensor before it enters the runtime memo.
        Warm hits remain a plain dictionary lookup and therefore stay traceable.
        The unchanged key and build expression keep the result bitwise neutral.
        """
        cached = self._runtime_cache.get(key)
        if cached is None:
            cached = _build_eager(build)
            self._runtime_cache[key] = cached
        return cached

    def _cached_tensor(
        self,
        name: str,
        tensor: torch.Tensor | Callable[[], torch.Tensor],
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> torch.Tensor:
        key = (name, dtype, _cache_device_key(device))

        def _build() -> torch.Tensor:
            # A callable source is evaluated INSIDE the eager cold build: passing
            # ``self._dxV_tensor()`` (already evaluated, i.e. traced) as the argument
            # would put that evaluation inside the traced step.
            source = tensor() if callable(tensor) else tensor
            return source.to(dtype=dtype, device=device)

        return self._runtime_memo(key, _build)

    def dx_center_for(self, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        tensor = self.dxF if self.dxF is not None else self.dx_center
        return self._cached_tensor("dx_center", tensor, dtype=dtype, device=device)

    def dxF_for(self, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        """Return ``dxF`` at cell centers using its unambiguous MITgcm name."""
        return self.dx_center_for(dtype, device)

    def dy_center_for(self, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        tensor = self.dyF if self.dyF is not None else self.dy_center
        return self._cached_tensor("dy_center", tensor, dtype=dtype, device=device)

    def dyF_for(self, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        """Return ``dyF`` at cell centers using its unambiguous MITgcm name."""
        return self.dy_center_for(dtype, device)

    def dxG_row_for(self, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        """Return ``dxG`` (length [ny]) on the requested device/dtype.

        Matches MITgcm's ``dxG(i,j)`` at the south edge of each cell row,
        constant along i for a non-rotated uniform grid.
        """
        return self._cached_tensor("dxG_row", self.dxG_row, dtype=dtype, device=device)

    def dxC_row_for(self, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        """Return ``dxC`` per row (length [ny]) on the requested device/dtype."""
        return self._cached_tensor("dxC_row", self.dxC_row, dtype=dtype, device=device)

    def dxG_row_ext_for(self, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        """Return ``dxG`` at every v-face row (length [ny+1]) on requested device/dtype.

        Superset of :meth:`dxG_row_for`: the first ``ny`` entries are identical
        to ``dxG_row``, and the trailing entry is ``dxG`` at the north OB
        v-face row ``yG[ny]``.  Consumers that need to look up ``dxG(j+1)``
        for ``j ∈ [0, ny-1]`` without extrapolation should read this field.
        """
        return self._cached_tensor("dxG_row_ext", self.dxG_row_ext, dtype=dtype, device=device)

    def rAz_row_for(self, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        """Return ``rAz`` per corner row (length [ny+1]) on requested device/dtype.

        MITgcm's vorticity-point area ``rAz(i,j)`` at the corner
        ``(xG[i], yG[j])``.  For a non-rotated uniform spherical polar grid
        ``rAz`` depends only on ``j`` (latitude), so we store a 1-D
        ``rAz_row[j]`` of length ``ny+1`` that the consumer broadcasts along
        ``i``.
        """
        return self._cached_tensor("rAz_row", self.rAz_row, dtype=dtype, device=device)

    def cell_area_for(self, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        """Return ``rA`` (cell area, shape ``[ny, nx]``) on requested device/dtype."""
        tensor = self.rA if self.rA is not None else self.cell_area
        return self._cached_tensor("cell_area", tensor, dtype=dtype, device=device)

    def rA_for(self, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        """Return tracer-cell area ``rA`` using its MITgcm name."""
        return self.cell_area_for(dtype, device)

    # ---- dxC (zonal centre-to-centre distance, u-face stagger) ----

    def _dxC_tensor(self) -> torch.Tensor:
        """Source tensor for :meth:`dxC_for` — base grid: u-face per-row broadcast.

        MITgcm ``dxC(i,j)`` lives at the u-point (western cell edge), shape
        ``[ny, nx+1]``.  For a non-rotated uniform spherical-polar grid it
        varies only with latitude (constant in ``i``), so we broadcast the
        1-D ``dxC_row[j]`` along ``i``.
        :class:`~locomgcm.geometry.cubed_sphere_metric_builder.CubedSphereMetricSet`
        overrides this to return the genuinely 2-D per-face ``dxC``.
        """
        if self.dxC is not None:
            return self.dxC
        ny = int(self.cell_area.shape[-2])
        nx = int(self.cell_area.shape[-1])
        return self.dxC_row.view(ny, 1).expand(ny, nx + 1).contiguous()

    def dxC_for(self, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        """Return ``dxC`` at the u-face (``[ny, nx+1]``; ``[face, ny, nx+1]`` on a
        cubed sphere) on the requested device/dtype.

        The 2-D-capable zonal centre-to-centre distance for x-gradient
        denominators.  On a uniform spherical-polar grid it is a constant
        tensor bit-equal to ``dxC_row`` broadcast along ``i``; on a
        cubed-sphere grid it carries the true per-cell value.
        """
        key = ("dxC", dtype, _cache_device_key(device))
        return self._runtime_memo(
            key, lambda: self._dxC_tensor().to(dtype=dtype, device=device)
        )

    # ---- dxG (zonal cell-edge width, v-face stagger) ----

    def _dxG_tensor(self) -> torch.Tensor:
        """Source tensor for :meth:`dxG_for` — base grid: v-face per-row broadcast.

        MITgcm ``dxG(i,j)`` lives at the v-point (southern cell edge), shape
        ``[ny+1, nx]``.  For a non-rotated uniform spherical-polar grid it
        varies only with latitude, so we broadcast ``dxG_row_ext[j]`` along
        ``i``.  The cubed-sphere subclass overrides this.
        """
        if self.dxG is not None:
            return self.dxG
        ny = int(self.cell_area.shape[-2])
        nx = int(self.cell_area.shape[-1])
        return self.dxG_row_ext.view(ny + 1, 1).expand(ny + 1, nx).contiguous()

    def dxG_for(self, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        """Return ``dxG`` at the v-face (``[ny+1, nx]``; ``[face, ny+1, nx]`` on a
        cubed sphere) on the requested device/dtype.
        """
        key = ("dxG", dtype, _cache_device_key(device))
        return self._runtime_memo(
            key, lambda: self._dxG_tensor().to(dtype=dtype, device=device)
        )

    # ---- rAz (corner-point cell area) ----

    def _rAz_tensor(self) -> torch.Tensor:
        """Source tensor for :meth:`rAz_for` — base grid: corner per-row broadcast.

        MITgcm ``rAz(i,j)`` lives at corner points, shape ``[ny+1, nx+1]``.
        For a non-rotated uniform spherical-polar grid it varies only with
        latitude, so we broadcast ``rAz_row[j]`` along ``i``.  The
        cubed-sphere subclass overrides this.
        """
        if self.rAz is not None:
            return self.rAz
        ny = int(self.cell_area.shape[-2])
        nx = int(self.cell_area.shape[-1])
        return self.rAz_row.view(ny + 1, 1).expand(ny + 1, nx + 1).contiguous()

    def rAz_for(self, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        """Return ``rAz`` at corners (``[ny+1, nx+1]``; ``[face, ny+1, nx+1]`` on a
        cubed sphere) on the requested device/dtype.
        """
        key = ("rAz", dtype, _cache_device_key(device))
        return self._runtime_memo(
            key, lambda: self._rAz_tensor().to(dtype=dtype, device=device)
        )

    def _dyG_tensor(self) -> torch.Tensor:
        """Source tensor for :meth:`dyG_for` — base grid: u-face constant.

        MITgcm ``dyG(i,j)`` lives at the u-point (western cell edge), shape
        ``[ny, nx+1]``.  For a non-rotated uniform spherical-polar grid it is
        the constant ``dyG_scalar_1d`` (``= rSphere·dlat``) everywhere, so we
        broadcast that scalar to the full u-face shape.
        :class:`~locomgcm.geometry.cubed_sphere_metric_builder.CubedSphereMetricSet`
        overrides this to return the genuinely 2-D per-cell face ``dyG``.
        """
        if self.dyG is not None:
            return self.dyG
        ny = int(self.cell_area.shape[-2])
        nx = int(self.cell_area.shape[-1])
        return torch.full(
            (ny, nx + 1),
            self.dyG_scalar_1d,
            dtype=self.cell_area.dtype,
            device=self.cell_area.device,
        )

    def dyG_for(self, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        """Return ``dyG`` at the u-face (``[ny, nx+1]``; ``[face, ny, nx+1]`` on a
        cubed sphere) on the requested device/dtype.

        The 2-D-capable accessor that supersedes direct reads of the scalar
        ``dyG_scalar_1d``.  On a uniform
        spherical-polar grid it is a constant tensor **bit-equal** to
        ``dyG_scalar_1d``, so consumers that broadcast against u-face fields can
        swap the scalar for this accessor with no change on existing uniform
        cases; on a cubed-sphere / variable-spacing grid it carries the true
        per-cell value (where the scalar would be a lossy collapse).
        """
        key = ("dyG", dtype, _cache_device_key(device))
        return self._runtime_memo(
            key, lambda: self._dyG_tensor().to(dtype=dtype, device=device)
        )

    def _dyC_tensor(self) -> torch.Tensor:
        """Source tensor for :meth:`dyC_for` — base grid: v-face constant.

        MITgcm ``dyC(i,j)`` is the meridional CENTRE-to-CENTRE distance, living
        at the v-point (southern cell edge), shape ``[ny+1, nx]``.  It is the
        denominator of every meridional gradient (``recip_dyC``), distinct from
        the u-face edge length ``dyG`` (:meth:`_dyG_tensor`) and the cell
        meridional width ``dyF`` (carried by ``dy_center`` on the cube).  For a
        non-rotated uniform spherical-polar grid ``dyC == dyG == dyF`` so it is
        the constant ``dyG_scalar_1d`` everywhere; the cubed-sphere subclass
        overrides this to return the genuinely 2-D per-face ``dyC``.
        """
        if self.dyC is not None:
            return self.dyC
        ny = int(self.cell_area.shape[-2])
        nx = int(self.cell_area.shape[-1])
        return torch.full(
            (ny + 1, nx),
            self.dyG_scalar_1d,
            dtype=self.cell_area.dtype,
            device=self.cell_area.device,
        )

    def dyC_for(self, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        """Return ``dyC`` at the v-face (``[ny+1, nx]``; ``[face, ny+1, nx]`` on a
        cubed sphere) on the requested device/dtype.

        The 2-D-capable meridional centre-to-centre distance that supersedes
        direct reads of the scalar ``dyG_scalar_1d`` in y-gradient denominators
        (``/dy``, ``/dy²``).  On a uniform
        spherical-polar grid it is a constant tensor **bit-equal** to
        ``dyG_scalar_1d``, so consumers that divide a v-face difference by the
        scalar can swap it for this accessor with no change on existing uniform
        cases; on a cubed-sphere / variable-spacing grid it carries the true
        per-cell value (where the scalar would be a lossy collapse).
        """
        key = ("dyC", dtype, _cache_device_key(device))
        return self._runtime_memo(
            key, lambda: self._dyC_tensor().to(dtype=dtype, device=device)
        )

    def _dxV_tensor(self) -> torch.Tensor:
        if self.dxV is not None:
            return self.dxV
        face_geometry = getattr(self, "face_geometry", None)
        if face_geometry is not None:
            return face_geometry["dxV"]
        dxG = self._dxG_tensor()
        return torch.cat((dxG, dxG[..., :1]), dim=-1)

    def dxV_for(self, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        """Return ``dxV`` at corners (``[ny+1, nx+1]`` regionally)."""
        return self._cached_tensor("dxV", self._dxV_tensor, dtype=dtype, device=device)

    def _dyU_tensor(self) -> torch.Tensor:
        if self.dyU is not None:
            return self.dyU
        face_geometry = getattr(self, "face_geometry", None)
        if face_geometry is not None:
            return face_geometry["dyU"]
        dyG = self._dyG_tensor()
        return torch.cat((dyG, dyG[..., :1, :]), dim=-2)

    def dyU_for(self, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        """Return ``dyU`` at corners (``[ny+1, nx+1]`` regionally)."""
        return self._cached_tensor("dyU", self._dyU_tensor, dtype=dtype, device=device)

    def _rAw_tensor(self) -> torch.Tensor:
        if self.rAw is not None:
            return self.rAw
        face_geometry = getattr(self, "face_geometry", None)
        if face_geometry is not None:
            return face_geometry["rAw"]
        rA = self.rA if self.rA is not None else self.cell_area
        return torch.cat((rA, rA[..., :1]), dim=-1)

    def rAw_for(self, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        """Return u-cell area ``rAw`` (``[ny, nx+1]`` regionally)."""
        return self._cached_tensor("rAw", self._rAw_tensor, dtype=dtype, device=device)

    def _rAs_tensor(self) -> torch.Tensor:
        if self.rAs is not None:
            return self.rAs
        face_geometry = getattr(self, "face_geometry", None)
        if face_geometry is not None:
            return face_geometry["rAs"]
        nx = int(self.cell_area.shape[-1])
        return self._rAz_tensor()[..., :nx]

    def rAs_for(self, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        """Return v-cell area ``rAs`` (``[ny+1, nx]`` regionally)."""
        return self._cached_tensor("rAs", self._rAs_tensor, dtype=dtype, device=device)
