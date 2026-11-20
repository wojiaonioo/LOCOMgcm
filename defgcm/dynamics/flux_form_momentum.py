"""Flux-form momentum bundle — the swappable counterpart to vector-invariant.

Composes the FP-floor-aligned flux-form pieces into one ``TendencyModule`` with
the SAME ``evaluate()`` signature as ``VectorInvariantMomentumAdvection`` so the
stepper sums it identically without a separate runtime dispatch path:

    gU/gV = advection  (MomentumAdvection, mom_fluxform advective divergence)
          + Coriolis   (FCoriolis latitude-varying, mom_u/v_coriolis)
          + metric     (SphericalMetric, mom_u/v_metric_sphere)   [if profiles declared]
          + NH-3D-Cori (_u_coriolis_nh_scheme1, mom_u_coriolis_nh; gU only)  [if active]
    then × maskW / maskS  (single final mask, exactly as mom_fluxform.F).

Mirrors MITgcm's ``vectorInvariantMomentum=.FALSE.`` package selection.  Verified
bitwise/FP-floor vs the compiled-MITgcm per-term + final masked gU/gV dump on
scs-768-5d.

Sign conventions (docs/NUMERICS.md): the vertical-advection w is read from
``EvaluationInputs.w_interface`` (MIT upward-positive) and negated inside
``MomentumAdvection``; the NH-3D-Coriolis helper consumes the same cache as
MIT-upward-positive directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from defgcm.geometry import StructuredCGrid
from defgcm.state import DiagnosticState, PrognosticState, StateTendency
from defgcm.state.state import resolved_input

from .base import TendencyModule, TendencyStepDomain
from .momentum_advection import MomentumAdvection
from .coriolis import FCoriolis
from .spherical_metric import SphericalMetric
from .vector_invariant_momentum import CORI_3D_SCHEMES, _scheme_name, _u_coriolis_nh_scheme1
from defgcm.constants import ROTATION_PERIOD_EARTH, omega_from_rotation_period
from defgcm.dynamics._registry import register_dynamics_module

_RECIP_RSPHERE_DEFAULT = 1.0 / 6_370_000.0


@dataclass(frozen=True)
@register_dynamics_module("flux_form_momentum")
class FluxFormMomentum(TendencyModule):
    """Flux-form momentum: advection + Coriolis + spherical metric + NH-3D-Coriolis.

    Parameters
    ----------
    coriolis_profile_c : tuple[float, ...] | None
        Required (the explicit-input requirement).  Cell-centre Coriolis profile
        ``2Ω·sin(yC)`` (length ``ny``).  Drives the flux-form horizontal
        Coriolis stencil.  Distinct from VI's corner ``coriolis_profile_g``.
        ``None`` declares that the configuration has no Coriolis term (MITgcm
        ``useCoriolis=.FALSE.``); an f-plane is a constant profile.
    tan_phi_u, tan_phi_v : tuple[float, ...] | None
        Required (the explicit-input requirement).  ``tan(yC)`` (length ``ny``) / ``tan(yG)``
        (length ``ny+1``) for the spherical metric.  ``None`` for both declares
        ``selectMetricTerms=0`` (MITgcm forces it on Cartesian grids,
        ``set_parms.F:54-57``); one without the other raises.
    select_r_star : int
        MITgcm ``select_rStar`` of the run's stepper, written by the canonical
        assembly; the advection's linear free-surface correction is added only
        when it is 0 (``mom_u_adv_wu.F:93``).
    recip_rsphere : float
        ``1/a`` for the metric; default ``1/6_370_000``.
    select_3d_cori_scheme : int
        MITgcm code from ``CORI_3D_SCHEMES``: ``1`` enables the NH 3D-Coriolis
        ``-f'·w`` term on gU (requires the w-interface cache), ``0`` (default)
        omits it; other codes are refused.
    select_cori_scheme : int
        MITgcm ``selectCoriScheme`` of the horizontal Coriolis term, passed to
        ``FCoriolis`` (codes in ``coriolis.FLUX_CORI_SCHEMES``).
    rotation_period : float
        MITgcm ``rotationPeriod`` in seconds (default 86164); its ``omega``
        (``ini_parms.F:487-490``) sets the latitude form of ``fCoriCos`` in
        the NH 3D-Coriolis term.
    """

    coriolis_profile_c: tuple[float, ...] | None
    tan_phi_u: tuple[float, ...] | None
    tan_phi_v: tuple[float, ...] | None
    select_r_star: int
    recip_rsphere: float = _RECIP_RSPHERE_DEFAULT
    select_3d_cori_scheme: int = 0
    select_cori_scheme: int = 0
    rotation_period: float = ROTATION_PERIOD_EARTH
    name: str = "flux_form_momentum"
    stepper_domain: TendencyStepDomain = TendencyStepDomain.SLOW
    # The component terms, built once with this module and reused by every
    # evaluate.  FCoriolis and SphericalMetric keep their grid profiles in
    # constant_cache under their own object identity (id(self)), so components
    # rebuilt inside evaluate made every call a cold build: a graph break in
    # the compiled step and a Dynamo frame guarded on a new object each step,
    # until the recompile limit sent that frame back to eager execution.
    _advection: MomentumAdvection = field(init=False, repr=False, compare=False)
    _coriolis: FCoriolis = field(init=False, repr=False, compare=False)
    _metric: SphericalMetric | None = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _scheme_name(
            CORI_3D_SCHEMES, self.select_3d_cori_scheme, "select_3d_cori_scheme")
        if (self.tan_phi_u is None) != (self.tan_phi_v is None):
            raise ValueError(
                "FluxFormMomentum needs tan_phi_u and tan_phi_v together "
                "(spherical metric) or both None (selectMetricTerms=0); got "
                f"tan_phi_u={'None' if self.tan_phi_u is None else 'profile'}, "
                f"tan_phi_v={'None' if self.tan_phi_v is None else 'profile'}"
            )
        object.__setattr__(self, "_advection", MomentumAdvection(select_r_star=self.select_r_star))
        object.__setattr__(
            self,
            "_coriolis",
            FCoriolis(
                coriolis_profile_c=self.coriolis_profile_c,
                select_cori_scheme=self.select_cori_scheme,
            ),
        )
        object.__setattr__(
            self,
            "_metric",
            None
            if self.tan_phi_u is None
            else SphericalMetric(
                tan_phi_u=self.tan_phi_u,
                tan_phi_v=self.tan_phi_v,
                recip_rsphere=self.recip_rsphere,
            ),
        )

    def prepare_host_constants(self, grid, *, dtype, device) -> None:
        """Prepare the same persistent components consumed by evaluate."""
        self._coriolis.prepare_host_constants(grid, dtype=dtype, device=device)
        if self._metric is not None:
            self._metric.prepare_host_constants(grid, dtype=dtype, device=device)

    def evaluate(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
    ) -> StateTendency:
        # 1) advective flux divergence (reads & negates the MIT-up w-cache internally)
        adv = self._advection.evaluate(state, diagnostics, grid)
        tend_u = adv.u
        tend_v = adv.v

        # 2) latitude-varying flux-form horizontal Coriolis
        cori = self._coriolis.evaluate(state, diagnostics, grid)
        tend_u = tend_u + cori.u
        tend_v = tend_v + cori.v

        # 3) spherical-polar metric (curvature) terms
        if self._metric is not None:
            metr = self._metric.evaluate(state, diagnostics, grid)
            tend_u = tend_u + metr.u
            tend_v = tend_v + metr.v

        # 4) NH 3-D Coriolis (-f'·w) on gU only (V-NH needs curvilinear grid)
        inputs = diagnostics.inputs
        select_3d = int(resolved_input(
            inputs, "select_3d_cori_scheme", self.select_3d_cori_scheme))
        cori_3d_scheme = _scheme_name(CORI_3D_SCHEMES, select_3d, "select_3d_cori_scheme")
        if cori_3d_scheme == "none":
            pass
        elif cori_3d_scheme == "original":
            w_interface = None if inputs is None else inputs.w_interface
            if w_interface is None:
                # the explicit-input requirement: the term is configured, so a missing w is
                # an unwired input, not an absent term.
                raise RuntimeError(
                    f"select_3d_cori_scheme={select_3d} needs the MIT "
                    "upward-positive w in EvaluationInputs.w_interface "
                    "(MOM_U_CORIOLIS_NH); none was given"
                )
            tend_u = tend_u + _u_coriolis_nh_scheme1(
                w_interface, grid, state.u.data, f_cori_cos=inputs.f_cori_cos,
                omega=omega_from_rotation_period(self.rotation_period))

        # 5) single final wet-face mask (mom_fluxform.F: gU*=maskW, gV*=maskS).
        # Re-masking the already-masked adv/cori/metric is a no-op; it zeroes the
        # un-masked NH contribution on dry faces, matching MIT exactly.
        if grid.masks.has_3d_masks:
            u_mask = grid.masks.u_float(dtype=tend_u.dtype, device=tend_u.device)
            v_mask = grid.masks.v_float(dtype=tend_v.dtype, device=tend_v.device)
            tend_u = tend_u * u_mask
            tend_v = tend_v * v_mask

        return StateTendency(
            eta=torch.zeros_like(state.eta.data),
            u=tend_u,
            v=tend_v,
            temp=torch.zeros_like(state.temp.data),
            salt=torch.zeros_like(state.salt.data),
        )
