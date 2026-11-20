from __future__ import annotations

from dataclasses import dataclass

import torch

from defgcm.geometry import StructuredCGrid
from defgcm.geometry.exchange import FillSpec, HaloExtentV1, exchange
from defgcm.operators import average_u_to_center, average_v_to_center, center_to_u, center_to_v
from defgcm.state import DiagnosticState, PrognosticState, StateTendency

from defgcm.accel.constant_cache import constant_cache
from ._host_constants import host_profile_to_tensor
from .base import TendencyModule, TendencyStepDomain
from .vector_invariant_momentum import _scheme_name
from defgcm.dynamics._registry import register_dynamics_module


#: selectCoriScheme of the flux-form term (mom_u_coriolis.F, mom_v_coriolis.F),
#: as (name, MITgcm code) pairs: 0 the original stencil, 1 the original stencil
#: times the Jamart wet-point factor, 2 the energy-conserving stencil, 3 the
#: energy-conserving stencil times the wet-point factor.  Codes >= 4 (averaged
#: transport) are not implemented.
FLUX_CORI_SCHEMES = (
    ("original", 0),
    ("original_wet_point", 1),
    ("energy", 2),
    ("energy_wet_point", 3),
)

_X_HALO = HaloExtentV1(west_left_extent=1, east_right_extent=1,
                       south_bottom_extent=0, north_top_extent=0)
_Y_HALO = HaloExtentV1(west_left_extent=0, east_right_extent=0,
                       south_bottom_extent=1, north_top_extent=1)


def wet_point_factor_u(u_tend: torch.Tensor, mask_s_ext: torch.Tensor) -> torch.Tensor:
    """Jamart wet-point factor on U faces (mom_u_coriolis.F:79-91).

    ``uC*4/max(1, maskS(i,j)+maskS(i,j+1)+maskS(i-1,j)+maskS(i-1,j+1))``;
    ``mask_s_ext`` is ``maskS`` with one x-halo column on each side.
    """
    wet = (
        (mask_s_ext[..., :-1, 1:] + mask_s_ext[..., 1:, 1:])
        + mask_s_ext[..., :-1, :-1]
    ) + mask_s_ext[..., 1:, :-1]
    return u_tend * 4.0 / torch.clamp(wet, min=1.0)


def wet_point_factor_v(v_tend: torch.Tensor, mask_w_ext: torch.Tensor) -> torch.Tensor:
    """Jamart wet-point factor on V faces (mom_v_coriolis.F:79-91).

    ``vC*4/max(1, maskW(i,j)+maskW(i+1,j)+maskW(i,j-1)+maskW(i+1,j-1))``;
    ``mask_w_ext`` is ``maskW`` with one y-halo row on each side.
    """
    wet = (
        (mask_w_ext[..., 1:, :-1] + mask_w_ext[..., 1:, 1:])
        + mask_w_ext[..., :-1, :-1]
    ) + mask_w_ext[..., :-1, 1:]
    return v_tend * 4.0 / torch.clamp(wet, min=1.0)


@dataclass(frozen=True)
@register_dynamics_module("f_coriolis")
class FCoriolis(TendencyModule):
    """Flux-form horizontal Coriolis tendency with final wet-face mask.

    Mirrors MITgcm flux-form raw Coriolis stencil
    (``mom_u_coriolis.F`` / ``mom_v_coriolis.F``, ``selectCoriScheme=0``
    "original discretization" branch) and applies the final ``maskW``/``maskS``
    step from ``mom_fluxform.F`` so that averaged Coriolis values do not leak
    into dry target faces:

        uCoriolis(i,j) =  f̄ⁱ(i,j) · ¼(v(i,j)+v(i,j+1)+v(i-1,j)+v(i-1,j+1))
        vCoriolis(i,j) = -f̄ʲ(i,j) · ¼(u(i,j)+u(i+1,j)+u(i,j-1)+u(i+1,j-1))

    Two modes:

    * **scalar f-plane** (``coriolis_profile_c is None``): constant ``f`` via the
      nested C-grid averaging operators (legacy behaviour, unchanged).
    * **latitude-varying** (``coriolis_profile_c`` provided): cell-centre
      Coriolis profile ``fCori(j) = 2Ω·sin(yC(j))`` (length ``ny``), averaged to
      the U-point (``f̄ⁱ = fCori(j)``, i-independent on spherical polar) and to
      the V-point (``f̄ʲ = ½(fCori(j)+fCori(j-1))``).  The 4-point velocity
      average uses MITgcm's **exact flat-sum order** ``¼(a+b+c+d)`` (NOT the
      nested ``½(½(·)+½(·))`` operator, which differs at FP-floor).  Verified
      bitwise vs the compiled-MITgcm ``MOM_U/V_CORIOLIS`` dump on scs-768-5d.

    Note: ``coriolis_profile_c`` is the **cell-centre** profile, distinct from
    the vector-invariant module's ``coriolis_profile_g`` (vorticity **corner**
    points ``yG``).  Flux-form Coriolis interpolates centre ``fCori``; VI uses
    corner ``fCoriG``.

    ``select_cori_scheme`` is MITgcm ``selectCoriScheme`` (``FLUX_CORI_SCHEMES``):
    0 the stencil above, 2 the energy-conserving stencil
    ``½(fCori(i)·v̄ʲ(i) + fCori(i-1)·v̄ʲ(i-1))``, and 1 and 3 the same two times
    the Jamart wet-point factor ``4/max(1, Σ neighbour masks)``.  Schemes other
    than 0 need ``coriolis_profile_c``; the scalar f-plane path implements 0.
    """

    coriolis_parameter: float = 0.0
    coriolis_profile_c: tuple[float, ...] | None = None
    select_cori_scheme: int = 0
    name: str = "f_plane_coriolis"
    stepper_domain: TendencyStepDomain = TendencyStepDomain.SLOW

    def __post_init__(self) -> None:
        scheme = _scheme_name(
            FLUX_CORI_SCHEMES, self.select_cori_scheme, "select_cori_scheme")
        if self.coriolis_profile_c is None and scheme != "original":
            raise NotImplementedError(
                f"FCoriolis select_cori_scheme={self.select_cori_scheme} needs "
                "coriolis_profile_c; the scalar f-plane path implements only 0"
            )

    def evaluate(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
    ) -> StateTendency:
        inputs = diagnostics.inputs
        if inputs is not None and inputs.coriolis_u is not None:
            # GGL90 useLANGMUIR: the term reads uRes/vRes (mom_fluxform.F:1083-1090)
            state = state.with_tensors(u=inputs.coriolis_u, v=inputs.coriolis_v)

        scheme = _scheme_name(
            FLUX_CORI_SCHEMES, self.select_cori_scheme, "select_cori_scheme")
        if self.coriolis_profile_c is None:
            # __post_init__ admits only "original" here.
            if self.coriolis_parameter == 0.0:
                return StateTendency.zeros_like(state)
            u_tend, v_tend = self._flux_coriolis_scalar(state, grid)
        elif scheme == "original":
            u_tend, v_tend = self._flux_coriolis_latvar(state, grid)
        elif scheme == "original_wet_point":
            u_tend, v_tend = self._wet_point_scale(
                *self._flux_coriolis_latvar(state, grid), grid)
        elif scheme == "energy":
            u_tend, v_tend = self._flux_coriolis_energy(state, grid)
        elif scheme == "energy_wet_point":
            u_tend, v_tend = self._wet_point_scale(
                *self._flux_coriolis_energy(state, grid), grid)

        if grid.masks.has_3d_masks:
            u_mask = grid.masks.u_float(dtype=state.u.data.dtype, device=state.u.data.device)
            v_mask = grid.masks.v_float(dtype=state.v.data.dtype, device=state.v.data.device)
            u_tend = u_tend * u_mask
            v_tend = v_tend * v_mask

        return StateTendency(
            eta=torch.zeros_like(state.eta.data),
            u=u_tend,
            v=v_tend,
            temp=torch.zeros_like(state.temp.data),
            salt=torch.zeros_like(state.salt.data),
        )

    def _flux_coriolis_scalar(
        self, state: PrognosticState, grid: StructuredCGrid
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Legacy constant-f path (nested C-grid averaging operators)."""
        f = state.u.data.new_tensor(self.coriolis_parameter)
        u_center = average_u_to_center(state.u.data)
        v_center = average_v_to_center(state.v.data)
        v_on_u = center_to_u(v_center, grid)
        u_on_v = center_to_v(u_center, grid)
        return f * v_on_u, -f * u_on_v

    def prepare_host_constants(self, grid, *, dtype, device) -> None:
        """Resolve host profiles before tracing, rebuilding each step if uncached."""
        if self.coriolis_profile_c is not None:
            object.__setattr__(self, "_prepared_profiles", (
                grid, dtype, torch.device(device), self._profiles_for(grid, dtype, device),
            ))

    def _profiles_for(self, grid, dtype, device):
        ny = grid.ny
        y_halo = _Y_HALO

        def _build_profiles() -> tuple[torch.Tensor, torch.Tensor]:
            fc = host_profile_to_tensor(
                self.coriolis_profile_c, dtype=dtype, device=device
            )  # (ny,)
            if fc.shape[0] != ny:
                raise ValueError(
                    f"coriolis_profile_c length {fc.shape[0]} != ny={ny}"
                )
            # f̄ʲ(j) = ½(fCori(j)+fCori(j-1)) needs fCori one cell beyond each
            # tile edge, and which value that is follows the topology through
            # the same exchange as the ``u`` halo below (numerics rule 11: the
            # seam value of a grid constant comes from edge_face_couples OR a
            # distributed neighbour):
            #   periodic y   -- the ghost is the wrap neighbour: fCori(-1) is
            #                   fCori(ny-1) and fCori(ny) is fCori(0), so the
            #                   single physical seam face gets ONE f̄ʲ from both
            #                   sides.  Replicating the edge value instead gave
            #                   the seam face the f of the adjacent cell CENTRE
            #                   (off by ±β·Δy/2 on a beta plane). MITgcm's
            #                   selectCoriMap=3 uses the exchanged profile
            #                   (ini_cori.F:178 EXCH).
            #   rank seam    -- the ghost is the neighbour rank's first/last
            #                   fCori, i.e. the next row of the global profile,
            #                   as MITgcm's halo f on an interior tile seam
            #                   (ini_cori.F:55-100 from yC, which
            #                   ini_local_grid.F:134-164 continues across
            #                   tiles).  A y split is then bitwise the single
            #                   rank; the global seam of a periodic y split
            #                   wraps through the neighbour table.
            #   wall / open  -- the edge face is masked (wall) or prescribed
            #                   (open), so its tendency is never consumed; keep
            #                   the historical replicate, which is bit-neutral
            #                   for every existing case.  MITgcm's own halo f
            #                   there is the coordinate-extrapolated value,
            #                   which a profile of ny centre values cannot
            #                   reproduce for a non-linear f.
            # A constant: exchanged once in the eager cold build, outside any
            # differentiation window (constant_cache / constant_transport).
            fc_ext = exchange(
                fc,
                grid=grid,
                location="T",
                extent=y_halo,
                fill=(
                    FillSpec.WRAP if grid.edge_face_couples("y") else FillSpec.REPLICATE
                ),
            )  # (ny+2,) fCori(-1 .. ny)
            f_v = 0.5 * (fc_ext[:-1] + fc_ext[1:])  # (ny+1,) ½(fCori(j-1)+fCori(j))
            return fc, f_v

        # Grid-constant profiles, eager cold build + traced warm reuse.
        # The v-face profile depends on the grid's y topology and neighbours as
        # well as on the module's profile: the grid owns it, and the key holds
        # the profile's values, so two Coriolis configurations evaluated on one
        # grid get their own entries.
        fc, f_v_row = constant_cache.get_or_build(
            ("coriolis_profile_c", tuple(self.coriolis_profile_c), dtype, str(device),
             grid.edge_face_couples("y")),
            _build_profiles,
            variant="host_profile", site="coriolis.coriolis_profile_c", owner=grid,
        )  # (ny,), (ny+1,)
        return fc, f_v_row

    def face_profiles(self, grid, *, dtype, device) -> tuple[torch.Tensor, torch.Tensor]:
        """``fCori(j)`` [ny] and ``(fCori(j-1)+fCori(j))/2`` [ny+1] for ``coriolis_profile_c``.

        The first is also the U-point average on a latitude-longitude grid,
        where fCori does not vary with i; the second is the V-point average
        with its topology-driven halo.  A compiled step reads the profiles
        ``prepare_host_constants`` resolved for this grid, dtype and device.
        """
        prepared = getattr(self, "_prepared_profiles", None)
        if (torch.compiler.is_compiling() and prepared is not None
                and prepared[0] is grid and prepared[1] == dtype
                and prepared[2] == device):
            return prepared[3]
        return self._profiles_for(grid, dtype, device)

    def _flux_coriolis_energy(
        self, state: PrognosticState, grid: StructuredCGrid
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Energy-conserving stencil (mom_u_coriolis.F:54-64, mom_v_coriolis.F:54-64).

        ``halfRL*(fCori(i)*halfRL*(v(i,j)+v(i,j+1)) + fCori(i-1)*halfRL*(...))``
        is ``center_to_u(fCori * average_v_to_center(v))``.  MIT halves
        ``fCori`` and the operator halves the velocity sum; a power-of-two
        scale commutes with the rounding, so both orders give the same bits.
        """
        u = state.u.data  # [B, nz, ny, nx+1]
        v = state.v.data  # [B, nz, ny+1, nx]
        fc, _ = self.face_profiles(grid, dtype=u.dtype, device=u.device)
        f_c = fc.view(1, 1, grid.ny, 1)
        u_tend = center_to_u(f_c * average_v_to_center(v), grid)
        v_tend = -center_to_v(f_c * average_u_to_center(u), grid)
        return u_tend, v_tend

    def _wet_point_scale(
        self,
        u_tend: torch.Tensor,
        v_tend: torch.Tensor,
        grid: StructuredCGrid,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Jamart wet-point factor of selectCoriScheme 1 and 3 (mom_u/v_coriolis.F:79-91)."""
        mask_s_ext = exchange(
            grid.masks.v_float(dtype=v_tend.dtype, device=v_tend.device),
            grid=grid, location="V", extent=_X_HALO, fill=FillSpec.ZERO,
            constant=True,
        )
        mask_w_ext = exchange(
            grid.masks.u_float(dtype=u_tend.dtype, device=u_tend.device),
            grid=grid, location="U", extent=_Y_HALO, fill=FillSpec.ZERO,
            constant=True,
        )
        return (
            wet_point_factor_u(u_tend, mask_s_ext),
            wet_point_factor_v(v_tend, mask_w_ext),
        )

    def _flux_coriolis_latvar(
        self, state: PrognosticState, grid: StructuredCGrid
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Latitude-varying flux-form Coriolis with MIT's flat-sum stencil."""
        u = state.u.data  # [B, nz, ny, nx+1]
        v = state.v.data  # [B, nz, ny+1, nx]
        nx, ny = grid.nx, grid.ny
        fc, f_v_row = self.face_profiles(grid, dtype=u.dtype, device=u.device)
        x_halo = _X_HALO
        y_halo = _Y_HALO

        # ── u-tendency: f̄ⁱ(j) · ¼(v(i,j)+v(i,j+1)+v(i-1,j)+v(i-1,j+1)) ──
        v_xpad = exchange(v, grid=grid, location="F", extent=x_halo,
                          fill=FillSpec.REPLICATE)          # [B,nz,ny+1,nx+2]
        v_i = v_xpad[..., 1:nx + 2]                          # i   : [B,nz,ny+1,nx+1]
        v_im1 = v_xpad[..., 0:nx + 1]                        # i-1 : [B,nz,ny+1,nx+1]
        v_i_j = v_i[..., 0:ny, :]
        v_i_jp1 = v_i[..., 1:ny + 1, :]
        v_im1_j = v_im1[..., 0:ny, :]
        v_im1_jp1 = v_im1[..., 1:ny + 1, :]
        # MIT order: (i,j)+(i,j+1)+(i-1,j)+(i-1,j+1) (mom_u_coriolis.F:50-52);
        # the CD scheme groups its corners differently (cd_code_scheme.F:129-130).
        sumv = ((v_i_j + v_i_jp1) + v_im1_j) + v_im1_jp1
        v_on_u = 0.25 * sumv                                 # [B,nz,ny,nx+1]
        f_u = fc.view(1, 1, ny, 1)                           # f̄ⁱ = fCori(j)
        u_tend = f_u * v_on_u

        # ── v-tendency: -f̄ʲ(j) · ¼(u(i,j)+u(i+1,j)+u(i,j-1)+u(i+1,j-1)) ──
        u_ypad = exchange(u, grid=grid, location="F", extent=y_halo,
                          fill=FillSpec.REPLICATE)           # [B,nz,ny+2,nx+1]
        u_j = u_ypad[..., 1:ny + 2, :]                        # j   : [B,nz,ny+1,nx+1]
        u_jm1 = u_ypad[..., 0:ny + 1, :]                      # j-1 : [B,nz,ny+1,nx+1]
        u_j_i = u_j[..., 0:nx]
        u_j_ip1 = u_j[..., 1:nx + 1]
        u_jm1_i = u_jm1[..., 0:nx]
        u_jm1_ip1 = u_jm1[..., 1:nx + 1]
        # MIT order: (i,j)+(i+1,j)+(i,j-1)+(i+1,j-1) (mom_v_coriolis.F:50-52);
        # the CD scheme groups its corners differently (cd_code_scheme.F:191-192).
        sumu = ((u_j_i + u_j_ip1) + u_jm1_i) + u_jm1_ip1
        u_on_v = 0.25 * sumu                                 # [B,nz,ny+1,nx]
        # f̄ʲ(j) = ½(fCori(j)+fCori(j-1)), built with its topology-driven halo in
        # _build_profiles above.
        f_v = f_v_row.view(1, 1, ny + 1, 1)
        v_tend = -f_v * u_on_v

        return u_tend, v_tend
