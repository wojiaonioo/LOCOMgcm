"""Implicit vertical viscosity columns and their mixing coefficients.

Fields use [..., nz, y, x] with surface-first levels at centers or U/V faces.
Layer thickness is positive in meters; diffusivity/viscosity is in m^2/s and
steps are seconds. Matrix reciprocals are dimensionless, zero on dry cells;
the caller selects their time level. Solves retain the field dtype/device.

``assemble_tracer_diffusivity`` and ``assemble_vertical_viscosity`` give
MITgcm's KappaRT/KappaRS (interfaces, [B, nz, ny, nx], row k the top
interface of layer k) and KappaRU/KappaRV (U faces [B, nz, ny, nx + 1], V
faces [B, nz, ny + 1, nx]) for the column solves, or the scalar background
where no package makes them fields.
The scheme diagnoses read the step-start state; densities are anomalies of
one equation-of-state module, both members of a pair at the lower layer's
reference level (MITgcm FIND_RHO_2D with kRef = k).  Background columns
(``viscArNr``, ``diffKrNrS``) are [nz]; face masks are the 0/1
``grid.masks.u_float``/``v_float``.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from defgcm.dynamics.cd_scheme import cd_scheme_of
from defgcm.dynamics.equation_of_state import linear_density_anomaly
from defgcm.dynamics.kpp_mixing import KppFields
from defgcm.dynamics.gmredi.transport import add_kwz_to_kappa
from defgcm.dynamics.vertical_mixing_schemes.coefficients import add_centre_viscosity, add_scheme_coefficient
from defgcm.dynamics.vertical_mixing_schemes import ggl90 as g90
from defgcm.dynamics.vertical_mixing_schemes.kl10 import kl10_coefficients
from defgcm.dynamics.vertical_mixing_schemes.ivdc import ivdc_convection_count, ivdc_tracer_diffusivity
from defgcm.dynamics.vertical_mixing_schemes.my82 import My82Parameters, my82_coefficients
from defgcm.dynamics.vertical_mixing_schemes.pp81 import Pp81Parameters, pp81_coefficients, pp81_ri_limit
from defgcm.dynamics.vertical_mixing_schemes.richardson import (
    negative_buoyancy_frequency, richardson_number, vertical_shear_squared,
)
from defgcm.dynamics.implicit_vertical_advection import tracer_hfac_factors
from defgcm.geometry.exchange import FillSpec, HaloExtentV1, exchange
from defgcm.solvers import build_implicit_diffusion_system, tridiag_solve

#: Vertical mixing packages besides KPP and ivdc_kappa (MITgcm usePP81,
#: useMY82, useGGL90, useKL10); at most one, and none with KPP.
VERTICAL_MIXING_SCHEMES = ("none", "pp81", "my82", "ggl90", "kl10")

#: Restart key of the GGL90 TKE (``defgcm.checkpoint.state.RESTART_STATE_KEYS_V2``).
GGL90_RESTART_KEY = "_ggl90_tke"

#: Equation-of-state modules whose kRef = k densities the mixing schemes can form.
MIXING_EQUATIONS_OF_STATE = (
    "linear_equation_of_state", "jmd95_equation_of_state", "mdjwf_equation_of_state",
)


@dataclass(frozen=True)
class VerticalMixingStep:
    """This step's vertical mixing, read by the momentum and the tracer column solves.

    ``kappa_ru``/``kappa_rv`` are CALC_VISCOSITY's result; ``tracer_kr`` and
    ``contributions`` are what CALC_3D_DIFFUSIVITY starts from at the tracer
    stage (``assemble_tracer_diffusivity``).
    """

    kappa_ru: torch.Tensor | float
    kappa_rv: torch.Tensor | float
    tracer_kr: torch.Tensor | float
    contributions: tuple


@dataclass(frozen=True)
class KwzContribution:
    """GM/Redi ``Kwz`` [nz, ny, nx] at W points (row k the top of layer k) and ``maskInC``."""

    kwz: torch.Tensor
    mask_in_c: torch.Tensor


@dataclass(frozen=True)
class CentreSchemeContribution:
    """PP81, MY82 or KL10 output, read by both coefficient assemblies.

    ``viscosity`` is PPviscAr/MYviscAr/KLviscAr at centres with one halo cell
    [B, nz, ny + 2, nx + 2]; ``diffusivity`` PPdiffKr/MYdiffKr/KLdiffKr [B, nz, ny, nx];
    ``visc_arnr``/``diffkr_nr_s`` the background columns [nz]; ``mask_w``,
    ``mask_s`` the U and V face masks.
    """

    viscosity: torch.Tensor
    diffusivity: torch.Tensor
    visc_arnr: torch.Tensor
    diffkr_nr_s: torch.Tensor
    mask_w: torch.Tensor
    mask_s: torch.Tensor


@dataclass(frozen=True)
class Ggl90Contribution:
    """GGL90 output, read by both coefficient assemblies.

    ``visc_u`` [B, nz, ny, nx + 1], ``visc_v`` [B, nz, ny + 1, nx] are
    GGL90viscArU/V, ``diffusivity`` GGL90diffKr [B, nz, ny, nx];
    ``visc_arnr``/``diffkr_nr_s`` the background columns [nz].
    """

    visc_u: torch.Tensor
    visc_v: torch.Tensor
    diffusivity: torch.Tensor
    visc_arnr: torch.Tensor
    diffkr_nr_s: torch.Tensor


def validate_vertical_mixing(
    *, ivdc_kappa: float, implicit_tracer_diffusion: bool, scheme: str, kpp_enabled: bool,
    implicit_vertical_viscosity: bool, nonhydrostatic: bool, momentum_implicit_vertical_advection: bool,
) -> None:
    """MITgcm's checks of the vertical mixing options (``ini_parms.F``, ``pp81_check.F``, ``my82_check.F``,
    ``ggl90_check.F``, ``kl10_check.F``)."""
    if ivdc_kappa != 0.0 and not implicit_tracer_diffusion:
        # ini_parms.F:738-744
        raise ValueError("ivdc_kappa requires implicit vertical tracer diffusion (implicitDiffusion)")
    if scheme not in VERTICAL_MIXING_SCHEMES:
        raise ValueError(f"vertical_mixing_scheme {scheme!r} is not one of {VERTICAL_MIXING_SCHEMES}")
    if scheme == "none":
        return
    # pp81_check.F:38-65, my82_check.F:38-72, ggl90_check.F:40-128; the
    # convection checks are in validate_convection
    if kpp_enabled and scheme == "kl10":
        # MITgcm runs KL10 after KPP (calc_viscosity.F:76-101); the centre
        # viscosity merge here starts from the background faces
        raise ValueError("kl10 with KPP is not ported")
    if kpp_enabled:
        raise ValueError(f"{scheme} and KPP cannot be turned on at the same time")
    if not implicit_tracer_diffusion:
        raise ValueError(f"{scheme} needs implicit vertical tracer diffusion (implicitDiffusion)")
    if not implicit_vertical_viscosity:
        raise ValueError(f"{scheme} needs implicit vertical viscosity (implicitViscosity)")
    # Paths that read kappaRU(Nr+1) without this step's scheme field: CALC_GW
    # (calc_gw.F:404-410) and the bottom drag of the implicit vertical
    # advection evaluator.
    if nonhydrostatic:
        raise ValueError(f"{scheme} is not wired into the nonhydrostatic CALC_GW viscosity")
    if momentum_implicit_vertical_advection:
        raise ValueError(f"{scheme} is not wired into momentum implicit vertical advection")


def validate_vertical_mixing_composition(*, scheme: str, parameters, nz: int, modules) -> None:
    """The scheme checks that read the grid's level count and the module composition."""
    if scheme == "kl10" and nz == 1:
        # kl10_calc.F:136-137 reads rhoS(2)
        raise ValueError("kl10 needs more than one level")
    if scheme == "ggl90":
        if nz == 1:
            # do_oceanic_phys.F:1005 skips GGL90_CALC for Nr = 1
            raise ValueError("ggl90 needs more than one level")
        if parameters.use_langmuir:
            # The Stokes drift reaches the flux-form Coriolis term (FCoriolis,
            # also inside flux_form_momentum) through EvaluationInputs; the
            # vector-invariant and beta-plane Coriolis do not read it, and
            # MITgcm stops for useCDscheme (ggl90_check.F:90-107).
            names = {getattr(m, "name", "") for m in modules}
            if (names & {"vector_invariant_momentum", "beta_plane_coriolis"}
                    or cd_scheme_of(modules) is not None):
                raise ValueError(
                    "GGL90 useLANGMUIR reaches only the flux-form Coriolis term; "
                    f"refused with {sorted(names & {'vector_invariant_momentum', 'beta_plane_coriolis'})} "
                    "or the CD scheme")


def validate_convection(*, c_adj_freq: float, ivdc_kappa: float, scheme: str, kpp_enabled: bool) -> None:
    """MITgcm's checks of convection, cAdjFreq or ivdc_kappa non-zero, beside the mixing packages.

    ``c_adj_freq`` is the convective adjustment module's interval and 0 without
    the module (MITgcm cAdjFreq = 0).
    """
    if ivdc_kappa != 0.0 and c_adj_freq != 0.0:
        # ini_parms.F:1107-1112
        raise ValueError(
            f"ivdc_kappa={ivdc_kappa} and convective_adjustment (cAdjFreq={c_adj_freq}) cannot both be enabled")
    if c_adj_freq == 0.0 and ivdc_kappa == 0.0:
        return
    # kpp_check.F:128-133, pp81_check.F:39-44, my82_check.F:39-44,
    # kl10_check.F:39-44; GGL90 only reports convection (ggl90_check.F:40-45)
    if kpp_enabled or scheme == "pp81" or scheme == "my82" or scheme == "kl10":
        package = "KPP" if kpp_enabled else scheme
        raise ValueError(
            f"{package} cannot run with convection enabled: convective_adjustment (cAdjFreq={c_adj_freq}) "
            f"or ivdc_kappa={ivdc_kappa} is non-zero")


def vertical_mixing_parameters(scheme: str, parameters: Mapping[str, Any]):
    """The scheme's parameter record from its configuration mapping, with MITgcm's parameter checks.

    KL10 reads one value and gets the KLviscMax scalar instead of a record.  A
    record or the scalar passes through: ``dataclasses.replace`` of the stepper
    runs the construction again on the converted field.
    """
    if scheme == "none":
        if parameters:
            raise ValueError(f"vertical_mixing_parameters {dict(parameters)} given without a scheme")
        return None
    elif scheme == "pp81":
        p = parameters if isinstance(parameters, Pp81Parameters) else Pp81Parameters(**parameters)
        # pp81_readparms.F:102-116; a negative PPnRi is outside the port
        if not p.visc_max > 0.0 or p.alpha == 0.0 or p.n_ri < 1:
            raise ValueError(f"PP81 needs PPviscMax > 0, PPalpha != 0 and PPnRi >= 1, got {p}")
        return p
    elif scheme == "my82":
        p = parameters if isinstance(parameters, My82Parameters) else My82Parameters(**parameters)
        # my82_readparms.F:95-105
        if not p.visc_max > 0.0 or not p.diff_max > 0.0:
            raise ValueError(f"MY82 needs MYviscMax > 0 and MYdiffMax > 0, got {p}")
        return p
    elif scheme == "kl10":
        # KLviscMax, the one KL10_PARM01 value the kernel reads (kl10_readparms.F:37-38)
        visc_max = parameters
        if isinstance(parameters, Mapping):
            if set(parameters) != {"visc_max"}:
                raise ValueError(f"KL10 vertical_mixing_parameters take exactly visc_max, got {dict(parameters)}")
            visc_max = parameters["visc_max"]
        # kl10_readparms.F:103-107
        if not visc_max > 0.0:
            raise ValueError(f"KL10 needs KLviscMax > 0, got visc_max={visc_max}")
        return visc_max
    elif scheme == "ggl90":
        p = parameters if isinstance(parameters, g90.Ggl90Parameters) else g90.Ggl90Parameters(**parameters)
        # ggl90_readparms.F:257-288
        if (not p.tke_min > 0.0 or p.tke_bottom < 0.0 or not p.mixing_length_min > 0.0
                or not p.visc_max > 0.0 or not p.diff_max > 0.0):
            raise ValueError("GGL90 needs GGL90TKEmin, GGL90mixingLengthMin, GGL90viscMax, "
                             f"GGL90diffMax > 0 and GGL90TKEbottom >= 0, got {p}")
        # ggl90_mixinglength.F:301-307, :352-358
        if p.mxl_max_flag not in g90.GGL90_MIXING_LENGTH_FLAGS:
            raise ValueError(f"GGL90 mxlMaxFlag {p.mxl_max_flag} is not one of {g90.GGL90_MIXING_LENGTH_FLAGS}")
        # ggl90_mixinglength.F:352-358
        if p.use_langmuir and p.mxl_max_flag not in g90.GGL90_LANGMUIR_MIXING_LENGTH_FLAGS:
            raise ValueError(f"GGL90 useLANGMUIR with mxlMaxFlag {p.mxl_max_flag} is not one of "
                             f"{g90.GGL90_LANGMUIR_MIXING_LENGTH_FLAGS}")
        return p
    raise ValueError(f"vertical_mixing_scheme {scheme!r} is not one of {VERTICAL_MIXING_SCHEMES}")


def mixing_equation_of_state(model):
    """The one equation-of-state module of ``model`` (``MIXING_EQUATIONS_OF_STATE``)."""
    modules = [m for m in getattr(model, "dynamics_modules", ()) or ()
               if getattr(m, "name", "") in MIXING_EQUATIONS_OF_STATE]
    if len(modules) != 1:
        raise ValueError(
            "vertical mixing needs exactly one equation-of-state module among "
            f"{MIXING_EQUATIONS_OF_STATE}, got {[m.name for m in modules]}")
    return modules[0]


def mixing_column_geometry(grid, *, dtype, device) -> tuple[torch.Tensor, torch.Tensor]:
    """``(maskC [nz, ny, nx], recip_drC [nz])`` for the column kernels.

    ``recip_drC(k) = 1/(0.5*(drF(k-1) + drF(k)))`` (``ini_vertical_grid.F:123-127,196``)
    at index k >= 1; index 0 is never read.
    """
    mask_c = grid.masks.center_float(dtype=dtype, device=device)
    if not grid.masks.has_3d_masks:
        mask_c = mask_c.expand(grid.nz, *mask_c.shape)
    spacing = grid.vertical_coordinate.interface_spacing.to(dtype=dtype, device=device)
    recip_drc = torch.cat((spacing.new_ones(1), 1.0 / spacing))
    return mask_c, recip_drc


def mixing_face_masks(grid, *, dtype, device) -> tuple[torch.Tensor, torch.Tensor]:
    """``(maskW [nz, ny, nx + 1], maskS [nz, ny + 1, nx])`` for the scheme merges."""
    mask_w = grid.masks.u_float(dtype=dtype, device=device)
    mask_s = grid.masks.v_float(dtype=dtype, device=device)
    if not grid.masks.has_3d_masks:
        mask_w = mask_w.expand(grid.nz, *mask_w.shape)
        mask_s = mask_s.expand(grid.nz, *mask_s.shape)
    return mask_w, mask_s


def mixing_densities(eos_module, *, temp, salt, pressure):
    """``(rhoInSitu, rhoKm1)`` of ``do_oceanic_phys.F:758-767, 811-823``.

    ``rhoInSitu`` [B, nz, ny, nx] is each layer at its own reference level;
    ``rhoKm1`` [B, nz - 1, ny, nx] is layer k - 1 at the reference level of
    layer k.  The linear EOS takes ``tRef(kRef)``, ``sRef(kRef)``
    (``find_rho.F:98-99``); JMD95 and MDJWF take ``pressure`` [B, nz, ny, nx]
    of layer k, the PRESSURE_FOR_EOS(kRef = k) the model EOS reads this step.
    """
    name = eos_module.name
    if name == "linear_equation_of_state":
        t_ref = torch.as_tensor(eos_module.reference_temperature, dtype=temp.dtype, device=temp.device)
        s_ref = torch.as_tensor(eos_module.reference_salinity, dtype=temp.dtype, device=temp.device)
        # The anomaly w.r.t. rhoNil: the module holds no rhoConst, and the
        # schemes use only differences of the two densities.
        coefficients = dict(reference_density=eos_module.reference_density,
                            rho_const=eos_module.reference_density,
                            thermal_expansion=eos_module.thermal_expansion,
                            haline_contraction=eos_module.haline_contraction)
        rho_in_situ = linear_density_anomaly(
            temp, salt, reference_temperature=t_ref, reference_salinity=s_ref, **coefficients)
        rho_km1 = linear_density_anomaly(
            temp[:, :-1], salt[:, :-1],
            reference_temperature=t_ref if t_ref.ndim == 0 else t_ref[1:],
            reference_salinity=s_ref if s_ref.ndim == 0 else s_ref[1:], **coefficients)
    elif name == "jmd95_equation_of_state" or name == "mdjwf_equation_of_state":
        rho_in_situ = eos_module.density_anomaly_at_pressure(temp, salt, pressure)
        rho_km1 = eos_module.density_anomaly_at_pressure(temp[:, :-1], salt[:, :-1], pressure[:, 1:])
    return rho_in_situ, rho_km1


def ivdc_tracer_diffusivity_at_step(*, sigma_r, ivdc_kappa: float, background_kr: float) -> torch.Tensor:
    """KappaRT/RS of a tracer without KPP under ivdc_kappa (``calc_3d_diffusivity.F:107-144``).

    ``IVDConvCount`` (``calc_ivdc.F:44-53``) of the step-start ``sigmaR``
    (``grad_sigma.F:84-98``, ``vertical_density_gradient``).
    """
    count = ivdc_convection_count(sigma_r)
    return ivdc_tracer_diffusivity(count, ivdc_kappa=ivdc_kappa, background_kr=background_kr)


def centre_scheme_contribution_at_step(
    *, scheme: str, parameters, grid, rho_in_situ, rho_km1, u, v, mask_c, recip_drc,
    gravity: float, mass2r_unit: float, background_ar: float, visc_arnr, diffkr_nr_s,
) -> CentreSchemeContribution:
    """PP81_CALC or MY82_CALC of the step-start state (``do_oceanic_phys.F:967-998``).

    The Richardson number of ``pp81_ri_number.F`` / ``my82_ri_number.F``
    from the kRef = k density pair and the centre velocity shear, the scheme
    coefficients on the interior, then one halo ring of the centre viscosity
    for the face averages (MITgcm forms it on ``2-OLx..sNx+OLx-1``,
    ``pp81_calc.F:61-64``).  A wall face has ``maskW = 0``, so its halo cell
    only has to be finite.  ``background_ar`` is ``viscArNr(1)`` as a host
    float for PP81's RiLimit (``pp81_readparms.F:117-123``).
    """
    minus_n2 = negative_buoyancy_frequency(
        rho_in_situ, rho_km1, recip_drc=recip_drc, gravity=gravity, mass2r_unit=mass2r_unit)
    shear = vertical_shear_squared(u, v, recip_drc=recip_drc)
    ri = richardson_number(minus_n2, shear)
    if scheme == "pp81":
        p = parameters
        visc, diff = pp81_coefficients(
            ri, mask_c=mask_c, alpha=p.alpha, nu0=p.nu0, n_ri=p.n_ri, visc_max=p.visc_max,
            ri_limit=pp81_ri_limit(nu0=p.nu0, visc_arnr_surface=background_ar,
                                   visc_max=p.visc_max, n_ri=p.n_ri, alpha=p.alpha),
            visc_arnr=visc_arnr, diffkr_nr_s=diffkr_nr_s)
    elif scheme == "my82":
        p = parameters
        interface_depth = grid.vertical_coordinate.interface_depth[:-1].to(
            dtype=ri.dtype, device=ri.device)
        visc, diff, _ = my82_coefficients(
            ri, minus_n2, shear, mask_c=mask_c, interface_depth=interface_depth,
            hbl_scale=p.hbl_scale, visc_max=p.visc_max, diff_max=p.diff_max,
            visc_arnr=visc_arnr, diffkr_nr_s=diffkr_nr_s)
    visc_halo = exchange(visc, grid=grid, location="T", extent=HaloExtentV1(1, 1, 1, 1),
                         fill=FillSpec.REPLICATE)
    mask_w, mask_s = mixing_face_masks(grid, dtype=visc.dtype, device=visc.device)
    return CentreSchemeContribution(
        viscosity=visc_halo, diffusivity=diff, visc_arnr=visc_arnr, diffkr_nr_s=diffkr_nr_s,
        mask_w=mask_w, mask_s=mask_s,
    )


def kl10_contribution_at_step(
    *, visc_max, grid, rho_in_situ, sigma_r, mask_c, recip_drc, gravity: float,
    mass2r_unit: float, visc_arnr, diffkr_nr_s,
) -> CentreSchemeContribution:
    """KL10_CALC of the step-start state (``do_oceanic_phys.F:978-986``).

    ``rho_in_situ`` is the shared kRef = k density (its layer 1 is rhoInSitu
    of ``kl10_calc.F:85``), ``sigma_r`` the shared sigmaR.  The centre
    viscosity gets one halo ring for the face averages, as PP81's (MITgcm
    forms it on ``2-OLx..sNx+OLx-1``, ``kl10_calc.F:79-82``).
    """
    dtype, device = sigma_r.dtype, sigma_r.device
    spacing = grid.vertical_coordinate.interface_spacing.to(dtype=dtype, device=device)
    _, center_depth = grid.vertical_coordinate.reference_centres_for(dtype, device)
    visc, diff = kl10_coefficients(
        rho_in_situ[:, 0], sigma_r, mask_c=mask_c, drc=torch.cat((spacing.new_ones(1), spacing)),
        recip_drc=recip_drc, center_depth=center_depth, gravity=gravity, mass2r_unit=mass2r_unit,
        visc_max=visc_max, visc_arnr=visc_arnr, diffkr_nr_s=diffkr_nr_s)
    visc_halo = exchange(visc, grid=grid, location="T", extent=HaloExtentV1(1, 1, 1, 1),
                         fill=FillSpec.REPLICATE)
    mask_w, mask_s = mixing_face_masks(grid, dtype=dtype, device=device)
    return CentreSchemeContribution(
        viscosity=visc_halo, diffusivity=diff, visc_arnr=visc_arnr, diffkr_nr_s=diffkr_nr_s,
        mask_w=mask_w, mask_s=mask_s,
    )


def ggl90_cold_start_tke(parameters, grid, *, like: torch.Tensor) -> torch.Tensor:
    """GGL90TKE of a fresh run, ``GGL90TKEmin*maskC`` (``ggl90_init_varia.F:42-56``), shaped like ``like``."""
    mask_c, _ = mixing_column_geometry(grid, dtype=like.dtype, device=like.device)
    return g90.ggl90_initial_tke(mask_c, tke_min=parameters.tke_min).expand_as(like).contiguous()


def encode_ggl90_tke(tke):
    return {} if tke is None else {GGL90_RESTART_KEY: tke}


def decode_ggl90_tke(payload):
    return payload.get(GGL90_RESTART_KEY)


def ggl90_step(
    *, parameters, grid, tke, sigma_r, u, v, surf_forc_u, surf_forc_v, mask_c, recip_drc,
    gravity: float, recip_rho_const: float, dt: float, visc_arnr, diffkr_nr_s,
) -> tuple[torch.Tensor, Ggl90Contribution]:
    """One GGL90_CALC (``do_oceanic_phys.F:1000-1014``): the new TKE and the coefficients.

    ``tke`` is last step's GGL90TKE; ``sigma_r`` the step-start sigmaR of
    the diagnostic loop (``do_oceanic_phys.F:803-887``); ``surf_forc_u/v``
    this step's surfaceForcingU/V [B, ny, nx + 1], [B, ny + 1, nx] in m^2/s^2
    (``external_forcing_surf.F:222``); ``dt`` is dTtracerLev(1)
    (``ggl90_calc.F:207``).  The kernel chain and its branches follow
    ``ggl90_calc.F`` and ``ggl90_mixinglength.F`` in statement order; the
    tridiagonal solve sits between the bottom condition and the floor
    (``:946-975``).  ``GGL90visctmp`` gets one halo ring for the face
    averages (``:1026-1070``); a wall face has ``maskW = 0``.
    """
    p = parameters
    dtype, device = tke.dtype, tke.device
    nz = grid.nz
    drf = grid.layer_thickness_for(dtype, device)
    lmin = p.mixing_length_min
    # kLowC: the deepest level with hFacC /= 0 (ini_masks_etc.F:180-183)
    levels = torch.arange(1, nz + 1, device=device).reshape(-1, 1, 1)
    k_low = torch.where(mask_c != 0, levels, torch.zeros_like(levels)).amax(dim=0)

    sqrt_tke, nsquare, ml = g90.ggl90_initial_state(
        tke, sigma_r, mask_c=mask_c, gravity=gravity, recip_rho_const=recip_rho_const,
        mixing_length_min=lmin)
    if p.mxl_surf_flag:
        ml = g90.ggl90_surface_mixing_length(ml, drf=drf)
    flag = p.mxl_max_flag
    if flag == 0 or flag == 1:
        # Ro_surf - R_low, the drF*hFacC sum of ini_masks_etc.F:132-141 on the
        # resting geometry: UPDATE_SURF_DR moves hFacC but not Ro_surf/R_low
        # (update_surf_dr.F:86-93); the matrix factors below read this step's.
        resting = grid.masks.resting_hfac
        hfac_c = grid.masks.hFacC if resting is None else resting[0]
        column_depth = torch.zeros_like(mask_c[0])
        for k in range(nz):
            column_depth = column_depth + drf[k] * (mask_c[k] if hfac_c is None
                                                    else hfac_c[k].to(dtype=dtype, device=device))
    if flag == 0:
        ml = g90.ggl90_limit_by_column_depth(ml, column_depth=column_depth)
    elif flag == 1:
        ml = g90.ggl90_limit_by_boundary_distance(
            ml, interface_depth=grid.vertical_coordinate.interface_depth[:-1].to(dtype=dtype, device=device),
            column_depth=column_depth)
    elif flag == 2 or flag == 3:
        ml, mx_length_dn = g90.ggl90_limit_by_sweeps(ml, drf=drf, mixing_length_min=lmin)
    if p.use_langmuir:
        # ggl90_mixinglength.F:309-376, on the limited length before its floor;
        # the reference of flag 1 is Ro_surf - rF(k), the interface depth
        if flag == 1:
            reference = grid.vertical_coordinate.interface_depth[:-1].to(
                dtype=dtype, device=device).reshape(1, -1, 1, 1)
        elif flag == 2 or flag == 3:
            reference = mx_length_dn
        kappa_length = g90.ggl90_langmuir_mixing_length(
            ml, reference, lc_gamma=p.lc_gamma, mixing_length_min=lmin)
        if flag == 1 or flag == 2:
            kappa_length = g90.ggl90_floor_langmuir_mixing_length(kappa_length, mixing_length_min=lmin)
    if flag == 0 or flag == 1 or flag == 2:
        ml, r_ml = g90.ggl90_floor_mixing_length(ml, mixing_length_min=lmin)
    elif flag == 3:
        r_ml = g90.ggl90_geometric_mixing_length(ml, mx_length_dn, mixing_length_min=lmin)
    if not p.use_langmuir:
        kappa_length = ml
    if p.calc_mean_vert_shear:
        shear = g90.ggl90_mean_vertical_shear(u, v, recip_drc=recip_drc)
    else:
        shear = vertical_shear_squared(u, v, recip_drc=recip_drc)
    # KappaM reads LCmixingLength under useLANGMUIR (ggl90_calc.F:443-455)
    tke, kappa_m, kappa_e, visc_tmp, prandtl = g90.ggl90_explicit_tke(
        tke, sqrt_tke, nsquare, kappa_length, shear, mask_c=mask_c, ck=p.ck, alpha=p.alpha, dt=dt,
        diffkr_nr_s=diffkr_nr_s, visc_arnr=visc_arnr)
    if p.use_langmuir:
        # ggl90_calc.F:510-561, 628-638
        stokes_args = dict(
            interface_depth=grid.vertical_coordinate.interface_depth[:-1].to(dtype=dtype, device=device),
            recip_drc=recip_drc, lc_num=p.lc_num, lc_lambda=p.lc_lambda)
        if p.calc_mean_vert_shear:
            stokes = g90.ggl90_langmuir_mean_stokes_shear(u, v, surf_forc_u, surf_forc_v, **stokes_args)
        else:
            stokes = g90.ggl90_langmuir_stokes_shear(u, v, surf_forc_u, surf_forc_v, **stokes_args)
        tke = g90.ggl90_langmuir_tke_source(tke, kappa_m, stokes, dt=dt)

    _, recip_hfac_c = tracer_hfac_factors(field=tke, hfac_c=grid.masks.hFacC)
    if p.missing_hfac_bug:
        recip_hfac_i = torch.ones_like(mask_c)
    else:
        recip_hfac_i = g90.ggl90_recip_hfac_interface(
            mask_c if grid.masks.hFacC is None else grid.masks.hFacC.to(dtype=dtype, device=device))
    a, c = g90.ggl90_tke_off_diagonals(
        kappa_e, mask_c=mask_c, recip_hfac_c=recip_hfac_c, recip_hfac_i=recip_hfac_i,
        recip_drf=1.0 / drf, recip_drc=recip_drc, k_low=k_low, dt=dt)
    if not p.dirichlet:
        c = g90.ggl90_neumann_bottom(c, k_low=k_low)
    b = g90.ggl90_tke_diagonal(a, c, sqrt_tke, r_ml, mask_c=mask_c, dt=dt, ceps=p.ceps)
    if p.calc_mean_vert_shear:
        friction_squared = g90.ggl90_mean_surface_friction_squared(surf_forc_u, surf_forc_v)
    else:
        friction_squared = g90.ggl90_surface_friction_squared(surf_forc_u, surf_forc_v)
    tke, a = g90.ggl90_surface_tke(tke, a, friction_squared, mask_c=mask_c, m2=p.m2,
                                   tke_surf_min=p.tke_surf_min)
    if p.dirichlet:
        tke, c = g90.ggl90_dirichlet_bottom(tke, c, k_low=k_low, tke_bottom=p.tke_bottom)
    tke = g90.ggl90_tke_floor(tridiag_solve(a, b, c, tke, mitgcm_order=True), mask_c=mask_c,
                              tke_min=p.tke_min)

    visc_tmp = exchange(visc_tmp, grid=grid, location="T", extent=HaloExtentV1(1, 1, 1, 1),
                        fill=FillSpec.REPLICATE)
    mask_w, mask_s = mixing_face_masks(grid, dtype=dtype, device=device)
    visc_u, visc_v, diff = g90.ggl90_coefficients(
        visc_tmp, prandtl, mask_w=mask_w, mask_s=mask_s, visc_max=p.visc_max, diff_max=p.diff_max,
        visc_arnr=visc_arnr, diffkr_nr_s=diffkr_nr_s)
    return tke, Ggl90Contribution(visc_u=visc_u, visc_v=visc_v, diffusivity=diff,
                                  visc_arnr=visc_arnr, diffkr_nr_s=diffkr_nr_s)


def ggl90_coriolis_velocity(parameters, grid, *, u, v, surf_forc_u, surf_forc_v):
    """``(uRes, vRes)`` of GGL90_ADD_STOKESDRIFT (``ggl90_add_stokesdrift.F:46-69``).

    The step-start velocity plus the Langmuir Stokes drift of this step's
    surfaceForcingU/V, which the flux-form Coriolis term of a useLANGMUIR
    run reads (``mom_fluxform.F:1083-1090``).
    """
    p = parameters
    mask_w, mask_s = mixing_face_masks(grid, dtype=u.dtype, device=u.device)
    # rC accumulated as ini_vertical_grid.F:144-146
    _, center_depth = grid.vertical_coordinate.reference_centres_for(u.dtype, u.device)
    return g90.ggl90_residual_velocity(
        u, v, surf_forc_u, surf_forc_v,
        center_depth=center_depth,
        mask_w=mask_w, mask_s=mask_s, lc_num=p.lc_num, lc_lambda=p.lc_lambda)


def assemble_vertical_viscosity(*, background_ar: float, contributions: tuple):
    """``(KappaRU, KappaRV)`` of MITgcm ``CALC_VISCOSITY`` (``calc_viscosity.F:63-131``).

    The background ``viscArNr``, raised by each enabled package's
    contribution: KPP (``KppFields``) returns the raised face fields
    (``KPP_CALC_VISC``); PP81, MY82 and KL10 (``CentreSchemeContribution``) raise
    the background faces with the centre-to-face maximum
    (``pp81_calc_visc.F:40-54``, ``my82_calc_visc.F:37-51``, ``kl10_calc_visc.F:40-53``); GGL90
    (``Ggl90Contribution``) adds ``GGL90viscArU - viscArNr(k)``
    (``ggl90_calc_visc.F:47-59``).  Interface Nr+1 is not formed here
    (``coefficients.bottom_interface_viscosity``).
    """
    kappa_ru = background_ar
    kappa_rv = background_ar
    for contribution in contributions:
        if isinstance(contribution, KppFields):
            kappa_ru = contribution.kappa_ru
            kappa_rv = contribution.kappa_rv
        elif isinstance(contribution, CentreSchemeContribution):
            c = contribution
            batch, nz, ny, nx = c.diffusivity.shape
            # calc_viscosity.F:66-72 sets viscArNr(k) on every face first
            background = c.visc_arnr.reshape(-1, 1, 1)
            kappa_ru, kappa_rv = add_centre_viscosity(
                background.expand(batch, nz, ny, nx + 1), background.expand(batch, nz, ny + 1, nx),
                c.viscosity, visc_arnr=c.visc_arnr, mask_w=c.mask_w, mask_s=c.mask_s)
        elif isinstance(contribution, Ggl90Contribution):
            c = contribution
            background = c.visc_arnr.reshape(-1, 1, 1)
            kappa_ru = add_scheme_coefficient(background.expand_as(c.visc_u), c.visc_u,
                                              background=c.visc_arnr)
            kappa_rv = add_scheme_coefficient(background.expand_as(c.visc_v), c.visc_v,
                                              background=c.visc_arnr)
        else:
            raise TypeError(f"no vertical viscosity merge for {type(contribution).__name__}")
    return kappa_ru, kappa_rv


def assemble_tracer_diffusivity(*, tracer_kr: float | torch.Tensor, contributions: tuple):
    """``(KappaRT, KappaRS)`` of MITgcm ``CALC_3D_DIFFUSIVITY`` (``calc_3d_diffusivity.F:83-245``).

    ``tracer_kr`` is the block of ``:83-168`` (``diffKrNrT/S``, LOCOMgcm has one
    scalar for both, plus ``ivdc_kappa`` on convecting interfaces).  Each
    enabled package contributes in MITgcm's order, whatever the tuple order:
    KPP (``KppFields``) replaces both (``:172-199``), then GM/Redi adds
    ``Kwz*maskInC`` to both (``:202-209``, ``gmredi_calc_diff.F:49-66``),
    then PP81, KL10 or MY82 (``CentreSchemeContribution``, ``kl10_calc_diff.F:45-77``) or GGL90
    (``Ggl90Contribution``) adds ``X - diffKrNrS(k)`` to both (``:211-245``,
    ``pp81_calc_diff.F:47-61``, ``my82_calc_diff.F:45-59``,
    ``ggl90_calc_diff.F:48-58``).  Packages that are off pass nothing.
    """
    by_kind = {type(c): c for c in contributions}
    unknown = set(by_kind) - {KppFields, KwzContribution, CentreSchemeContribution, Ggl90Contribution}
    if unknown:
        raise TypeError(f"no tracer diffusivity merge for {sorted(k.__name__ for k in unknown)}")
    kappa_rt = tracer_kr
    kappa_rs = tracer_kr
    if KppFields in by_kind:
        kappa_rt = by_kind[KppFields].diff_kzt
        kappa_rs = by_kind[KppFields].diff_kzs
    if KwzContribution in by_kind:
        gm = by_kind[KwzContribution]
        kappa_rt = add_kwz_to_kappa(kappa_rt, gm.kwz, maskInC=gm.mask_in_c)
        kappa_rs = add_kwz_to_kappa(kappa_rs, gm.kwz, maskInC=gm.mask_in_c)
    for kind in (CentreSchemeContribution, Ggl90Contribution):
        if kind in by_kind:
            c = by_kind[kind]
            kappa_rt = add_scheme_coefficient(kappa_rt, c.diffusivity, background=c.diffkr_nr_s)
            kappa_rs = add_scheme_coefficient(kappa_rs, c.diffusivity, background=c.diffkr_nr_s)
    return kappa_rt, kappa_rs


def solve_implicit_vertical_diffusion(
    *,
    field: torch.Tensor,
    layer_thickness: torch.Tensor,
    interface_spacing: torch.Tensor,
    dt: float,
    diffusivity: float | torch.Tensor,
    recip_hfac_matrix: torch.Tensor | None,
) -> torch.Tensor:
    """Apply implicit vertical diffusion via tridiagonal solve.

    MITgcm equivalent: ``IMPLDIFF`` called from ``dynamics.F`` after the
    Adams-Bashforth prediction step.  Solves

        (I - dt · K · ∂²/∂z²) field_new = field

    using the Thomas algorithm.

    Parameters
    ----------
    field : Tensor [B, nz, ny, nx] — velocity component or center tracer
    layer_thickness : Tensor [nz] — positive layer thickness in meters
    interface_spacing : Tensor [nz-1] — drC at the interior interfaces in meters
    dt : float — time step
    diffusivity : float | Tensor — vertical diffusivity or viscosity (m²/s)
    recip_hfac_matrix : Tensor [nz, ny, nx] | None — reciprocal partial-cell
        fraction at the time level selected by the caller (MITgcm:
        recip_hFacW for U, recip_hFacS for V, recip_hFacC for tracers).

    Returns
    -------
    field_new : Tensor [B, nz, ny, nx] — implicitly diffused field
    """
    a, b, c, d = build_implicit_diffusion_system(
        field, layer_thickness, interface_spacing=interface_spacing,
        diffusivity=diffusivity, dt=dt, recip_hFac=recip_hfac_matrix,
    )
    return tridiag_solve(a, b, c, d, mitgcm_order=True)
