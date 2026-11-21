"""The ``cd_scheme`` module: MITgcm ``useCDscheme`` for the implicit free-surface stepper.

The module is not a tendency term of the AB2 assembly.  MITgcm calls
``CD_CODE_SCHEME`` from ``TIMESTEP`` after Adams-Bashforth and after the
forcing and dissipation kept outside it (``timestep.F:228-270``), and adds
its Coriolis terms outside AB.  The module therefore declares the PREDICTOR
domain, which no tendency assembly holds; the implicit free-surface stepper
calls ``CdScheme.advance`` at that point of its predictor, and a stepper that
does not list the domain refuses the module at assembly.  The momentum core
must then carry no C-grid Coriolis term
(``mom_fluxform.F:1081-1108``).

Inputs are LOCOMgcm's prognostic shapes (``locomgcm.dynamics.cd_scheme.kernels``);
tendencies are in m s-2, eta is up-positive in m.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from locomgcm.dynamics._registry import register_dynamics_module
from locomgcm.dynamics.base import MomentumTermClass, TendencyModule, TendencyStepDomain
from locomgcm.dynamics.beta_plane import BetaPlaneCoriolis
from locomgcm.dynamics.bottom_drag import BottomDrag
from locomgcm.dynamics.coriolis import FCoriolis
from locomgcm.dynamics.flux_form_momentum import FluxFormMomentum
from locomgcm.dynamics.vector_invariant_momentum import VectorInvariantMomentumAdvection
from locomgcm.geometry.exchange import FillSpec, HaloExtentV1, exchange
from locomgcm.geometry.metric_bundle import _positive_reciprocal

from . import kernels
from .state import CdState

#: CPP options of ``CD_CODE_OPTIONS.h`` this port implements, each either
#: defined or not (both undefined in pkg/cd_code, both defined in
#: verification/exp2/code).  CD_CODE_NO_AB_MOMENTUM steps the D grid with the
#: tendency before Adams-Bashforth (timestep.F:148-158, 237-260);
#: CD_CODE_NO_AB_CORIOLIS takes the Coriolis term on uVel, vVel alone and
#: rounds pf in another order (cd_code_scheme.F:97-105, 134-138, 196-200).
CD_CODE_OPTIONS = ("CD_CODE_NO_AB_MOMENTUM", "CD_CODE_NO_AB_CORIOLIS")
#: Time-stepping regimes this port implements.  Under staggerTimeStep MITgcm
#: adds ``pfFacMom*dPhiHyd`` inside the scheme (cd_code_scheme.F:77-84).
CD_STEPPING_MODES = ("synchronous", "staggered")
#: Momentum forms this port implements.  The vector-invariant form keeps its
#: vorticity advection while dropping MOM_VI_CORIOLIS (mom_vecinv.F:677-735),
#: which has no LOCOMgcm counterpart.
CD_MOMENTUM_FORMS = ("flux_form",)

_X1 = HaloExtentV1(west_left_extent=1, east_right_extent=1,
                   south_bottom_extent=0, north_top_extent=0)
_Y1 = HaloExtentV1(west_left_extent=0, east_right_extent=0,
                   south_bottom_extent=1, north_top_extent=1)


@dataclass(frozen=True)
@register_dynamics_module("cd_scheme")
class CdScheme(TendencyModule):
    """MITgcm CD scheme parameters (data PARM03) and the per-step adapter.

    Parameters
    ----------
    coriolis_profile_c : tuple[float, ...]
        Cell-centre ``fCori(j)`` in s-1, length ``ny`` (as ``FCoriolis``).
    tau_cd : float
        ``tauCD`` in s; 0 resolves to ``deltaTMom`` (ini_parms.F:1116).
    r_cd : float
        ``rCD``; negative resolves to ``1 - deltaTMom/tauCD`` (ini_parms.F:1117).
    eps_ab_cd : float | None
        ``epsAB_CD``; None is MITgcm's UNSET_RL and resolves to ``abEps``
        (ini_parms.F:1118).
    cpp_defines : tuple[str, ...]
        The members of ``CD_CODE_OPTIONS`` the reference build defines.
    """

    coriolis_profile_c: tuple[float, ...]
    tau_cd: float
    r_cd: float
    eps_ab_cd: float | None
    cpp_defines: tuple[str, ...]
    name: str = "cd_scheme"
    stepper_domain: TendencyStepDomain = TendencyStepDomain.PREDICTOR
    # Owns the centre and V-face fCori profiles through its constant_cache entry.
    _coriolis: FCoriolis = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        unknown = sorted(set(self.cpp_defines) - set(CD_CODE_OPTIONS))
        if unknown:
            raise ValueError(f"cd_scheme cpp_defines {unknown} are not among {CD_CODE_OPTIONS}")
        object.__setattr__(self, "_coriolis", FCoriolis(coriolis_profile_c=self.coriolis_profile_c))

    @property
    def steps_with_pre_ab_tendency(self) -> bool:
        """CD_CODE_NO_AB_MOMENTUM: gUtmp is gU before AB plus the terms kept outside it."""
        return "CD_CODE_NO_AB_MOMENTUM" in self.cpp_defines

    def prepare_host_constants(self, grid, *, dtype, device) -> None:
        """Resolve the fCori profiles before tracing (``FCoriolis.prepare_host_constants``)."""
        self._coriolis.prepare_host_constants(grid, dtype=dtype, device=device)

    def evaluate(self, state, diagnostics, grid):
        raise NotImplementedError(
            "cd_scheme is called by the implicit free-surface stepper inside its "
            "momentum predictor (MITgcm TIMESTEP); it is not an assembled tendency"
        )

    def advance(
        self,
        *,
        history: CdState | None,
        u: torch.Tensor,
        v: torch.Tensor,
        eta: torch.Tensor,
        g_u: torch.Tensor,
        g_v: torch.Tensor,
        hydrostatic_gradient: tuple[torch.Tensor, torch.Tensor] | None,
        grid,
        gravity: float,
        dt_mom: float,
        ab_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, CdState]:
        """One ``CD_CODE_SCHEME`` call over all levels: ``(guCor, gvCor, next state)``.

        ``u``, ``v``, ``eta`` are the step-entry fields (MITgcm uVel, vVel,
        etaN); ``g_u``, ``g_v`` are gUtmp, gVtmp: after AB, or before it under
        ``steps_with_pre_ab_tendency``, plus the terms kept outside AB.
        ``hydrostatic_gradient`` is the pressure gradient tendency pair
        ``-pfFacMom*(dPhiHydX, dPhiHydY)`` under staggerTimeStep, which gUtmp
        lacks there, and None where ``phxFac = phyFac = 0``.  ``history`` is the state the
        previous step returned, or None on the first step of a fresh run.
        ``gravity`` is Bo_surf, which is gBaro in z coordinates
        (ini_linear_phisurf.F:79-89).  The returned state stores etaNm1 = etaN,
        the value SOLVE_FOR_PRESSURE saves before the solve
        (solve_for_pressure.F:126-127), since no call between DYNAMICS and the
        solve changes etaN.
        """
        tau_cd = dt_mom if self.tau_cd == 0.0 else self.tau_cd
        r_cd = 1.0 - dt_mom / tau_cd if self.r_cd < 0.0 else self.r_cd
        eps_ab_cd = ab_eps if self.eps_ab_cd is None else self.eps_ab_cd
        if history is None:
            # cd_code_ini_vars.F:136-160 zeroes the state; myIter=0 weights
            # (cd_code_scheme.F:69-71).
            ab15, ab05 = 1.0, -0.0
            history = CdState(
                u_d=torch.zeros_like(v), v_d=torch.zeros_like(u),
                eta_nm1=torch.zeros_like(eta),
                u_nm1=torch.zeros_like(u), v_nm1=torch.zeros_like(v),
            )
        else:
            ab15, ab05 = 1.5 + eps_ab_cd, -0.5 - eps_ab_cd

        dtype, device = u.dtype, u.device
        f_c, f_v = self._coriolis.face_profiles(grid, dtype=dtype, device=device)
        mask_w = grid.masks.u_float(dtype=dtype, device=device)
        mask_s = grid.masks.v_float(dtype=dtype, device=device)

        if "CD_CODE_NO_AB_CORIOLIS" in self.cpp_defines:
            pf = kernels.extrapolated_surface_pressure_no_ab_coriolis(
                eta, history.eta_nm1, bo_surf=gravity, ab15=ab15, ab05=ab05)
            u_cor, v_cor = u, v
        else:
            pf = kernels.extrapolated_surface_pressure(
                eta, history.eta_nm1, bo_surf=gravity, ab15=ab15, ab05=ab05)
            u_cor = ab15 * u + ab05 * history.u_nm1
            v_cor = ab15 * v + ab05 * history.v_nm1

        # D-grid v on U points: af on V points needs pf one row past each edge,
        # its corner average one column past each edge.
        pf_y = exchange(pf, grid=grid, location="T", extent=_Y1, fill=FillSpec.REPLICATE)
        recip_dyc = _positive_reciprocal(grid.metrics.dyC_for(dtype, device))
        if hydrostatic_gradient is None:
            af_v = kernels.v_point_acceleration(g_v, pf_y, recip_dyc=recip_dyc, mask_s=mask_s)
        else:
            af_v = kernels.v_point_acceleration_staggered(
                g_v, pf_y, hydrostatic_gradient[1], recip_dyc=recip_dyc, mask_s=mask_s)
        v_d, gu_cor = kernels.d_grid_v_step(
            exchange(af_v, grid=grid, location="V", extent=_X1, fill=FillSpec.REPLICATE),
            history.v_d, u_cor,
            exchange(v, grid=grid, location="V", extent=_X1, fill=FillSpec.REPLICATE),
            exchange(history.v_nm1, grid=grid, location="V", extent=_X1, fill=FillSpec.REPLICATE),
            f_u=f_c.view(-1, 1), mask_w=mask_w,
            ab15=ab15, ab05=ab05, dt_mom=dt_mom, r_cd=r_cd,
        )

        # D-grid u on V points: af on U points needs pf one column past each
        # edge, its corner average one row past each edge.
        pf_x = exchange(pf, grid=grid, location="T", extent=_X1, fill=FillSpec.REPLICATE)
        recip_dxc = _positive_reciprocal(grid.metrics.dxC_for(dtype, device))
        if hydrostatic_gradient is None:
            af_u = kernels.u_point_acceleration(g_u, pf_x, recip_dxc=recip_dxc, mask_w=mask_w)
        else:
            af_u = kernels.u_point_acceleration_staggered(
                g_u, pf_x, hydrostatic_gradient[0], recip_dxc=recip_dxc, mask_w=mask_w)
        u_d, gv_cor = kernels.d_grid_u_step(
            exchange(af_u, grid=grid, location="U", extent=_Y1, fill=FillSpec.REPLICATE),
            history.u_d, v_cor,
            exchange(u, grid=grid, location="U", extent=_Y1, fill=FillSpec.REPLICATE),
            exchange(history.u_nm1, grid=grid, location="U", extent=_Y1, fill=FillSpec.REPLICATE),
            f_v=f_v.view(-1, 1), mask_s=mask_s,
            ab15=ab15, ab05=ab05, dt_mom=dt_mom, r_cd=r_cd,
        )
        # cd_code_scheme.F:236-241 keeps uNM1 = uVel, vNM1 = vVel.
        return gu_cor, gv_cor, CdState(u_d=u_d, v_d=v_d, eta_nm1=eta, u_nm1=u, v_nm1=v)


def cd_scheme_of(modules) -> CdScheme | None:
    """The composition's CD scheme module, or None when MITgcm's useCDscheme is off."""
    return next((module for module in modules if isinstance(module, CdScheme)), None)


def admit_cd_scheme(
    modules,
    *,
    grid,
    stagger_tracer_stepping: bool,
    momentum_pressure_forcing: bool,
    momentum_implicit_vertical_advection: bool,
) -> None:
    """Refuse compositions the CD scheme does not cover (SPEC_cd section 1)."""
    found = [module for module in modules if isinstance(module, CdScheme)]
    if not found:
        return
    refused = []
    if len(found) > 1:
        refused.append(f"{len(found)} cd_scheme modules")
    if getattr(grid, "face_count", 1) == 6:
        refused.append("a cubed-sphere grid (config_check.F:1049-1053)")
    if grid.boundary_topology.has_open_boundaries:
        refused.append("open boundaries (obcs_check.F:361-368)")
    if momentum_implicit_vertical_advection:
        refused.append("momentum implicit vertical advection (config_check.F:1055-1059)")
    if any(isinstance(module, BottomDrag) and module.implicit for module in modules):
        refused.append("implicit bottom drag (config_check.F:1061-1065)")
    stepping = "staggered" if stagger_tracer_stepping else "synchronous"
    if stepping not in CD_STEPPING_MODES:
        refused.append(f"{stepping} stepping (not ported: cd_code_scheme.F:77-84)")
    hydrostatic = [module for module in modules
                   if module.stepper_domain == TendencyStepDomain.SLOW
                   and module.momentum_term_class == MomentumTermClass.PRESSURE_GRADIENT]
    if stepping == "staggered" and momentum_pressure_forcing and len(hydrostatic) != 1:
        refused.append(f"{len(hydrostatic)} hydrostatic pressure gradient modules under "
                       "staggered stepping (the scheme takes one dPhiHyd, cd_code_scheme.F:117-118)")
    form = ("vector_invariant"
            if any(isinstance(module, VectorInvariantMomentumAdvection) for module in modules)
            else "flux_form")
    if form not in CD_MOMENTUM_FORMS:
        refused.append(f"{form} momentum (not ported: mom_vecinv.F:677-735)")
    if any(isinstance(module, (FCoriolis, BetaPlaneCoriolis)) for module in modules) or any(
        isinstance(module, FluxFormMomentum) and module.coriolis_profile_c is not None
        for module in modules
    ):
        refused.append("a C-grid Coriolis term in the composition (mom_fluxform.F:1081-1108)")
    if refused:
        raise ValueError("cd_scheme does not cover " + "; ".join(refused))
