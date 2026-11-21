"""KPP configuration, static tensors, forcing, and regional diagnostic inputs.

Gravity is in m/s^2, reference density in kg/m^3, viscosity in m^2/s,
and mixed-layer depths and positive layer thicknesses in m. Coriolis is
cell-centred in s^-1; static tensors use the requested dtype and device.
Static zgrid is negative with z upward.
Surface T/S fields are on C cells in K m/s and PSU m/s; U/V forcing is
tau*mass2rUnit on C-grid faces in m^2/s^2. Salt forcing is positive into the ocean;
qsw retains MIT upward-positive W/m^2. EOS pressure is sampled in Pa.
Static maskC is hFacC > 0; face reconstruction receives explicit masks.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from typing import TYPE_CHECKING, Any

import torch

from .coriolis import spherical_centre_coriolis

if TYPE_CHECKING:
    from locomgcm.dynamics.kpp_mixing import KppFields, KppStatic, RegionalKppInputs


KPP_EOS_TYPES = (
    ("jmd95_equation_of_state", "jmd95"),
    ("mdjwf_equation_of_state", "mdjwf"),
    ("linear_equation_of_state", "linear"),
)

#: Dynamics modules that force momentum at the surface (MITgcm fu/fv).
SURFACE_STRESS_MODULES = ("era5_wind_forcing", "surface_wind_stress")


@dataclass(frozen=True)
class KppEvaluation:
    """Step-start KPP inputs and their diagnosed mixing fields."""

    # EOS pressure is in Pa at C cells; forcing follows the surface conventions above.
    inputs: RegionalKppInputs
    # U/V viscosity and C diffusivity are in m^2/s; hbl is positive depth in m.
    # These fields retain the step-start state, masks and dtype.
    fields: KppFields


@dataclass(frozen=True)
class KppRunValues:
    """Run-scoped driver values shared by the regional and cube paths."""

    gravity: float
    implicit_viscosity: float
    min_hbl: float | None
    estimate_uref: bool
    smooth_shsq: bool
    smooth_dbloc: bool
    sign_zero: str
    surface_buoyancy: str
    dbloc_form: str
    ricr: float | None
    use_double_diff: bool


def kpp_eos_name(eos_module: Any) -> str:
    """Resolve the configured module name used by KPP EOS selection."""
    return str(getattr(eos_module, "name", ""))


def kpp_eos_module(model) -> Any:
    """The one equation-of-state module of ``model``, the EOS STATEKPP evaluates."""
    modules = getattr(model, "dynamics_modules", ()) or ()
    eos_modules = [
        module
        for module in modules
        if kpp_eos_name(module).endswith("_equation_of_state")
    ]
    if len(eos_modules) > 1:
        names = [kpp_eos_name(module) for module in eos_modules]
        raise ValueError(f"KPP requires exactly one equation-of-state module, got {names}")
    if not eos_modules:
        raise ValueError("KPP enabled but configured equation-of-state module was not found")
    return eos_modules[0]


@dataclass(frozen=True)
class KppStaticInputs:
    """Regional KPP resources and the latest harvested reference density."""

    static: KppStatic | None
    rho_const: float | None


@dataclass(frozen=True)
class KppStepForcing:
    """Surface inputs for one KPP interval, including explicit absent sources."""

    surf_forc_t: torch.Tensor | None
    surf_forc_s: torch.Tensor | None
    surf_forc_u: torch.Tensor | None
    surf_forc_v: torch.Tensor | None
    surf_qsw: torch.Tensor | None
    penetrating_sw: bool


def seed_kpp_forcing_inputs(
    forcing: KppStepForcing, *, fields, fill, penetrating_sw, template, ny, nx,
) -> KppStepForcing:
    """Fill the named fields still absent with ``fill``.

    NaN reserves the fields of the producers that will overwrite them; zero is
    the forcing of a model without a surface forcing module.
    """
    shapes = {
        "surf_forc_u": (1, ny, nx + 1),
        "surf_forc_v": (1, ny + 1, nx),
    }
    values = {}
    for name in fields:
        if name == "penetrating_sw":
            values[name] = penetrating_sw
        elif getattr(forcing, name) is None:
            values[name] = torch.full(
                shapes.get(name, (1, ny, nx)), fill,
                dtype=template.dtype, device=template.device,
            )
    return replace(forcing, **values)


def prepare_kpp_face_stress(
    tau_x_at_step_start, tau_y_at_step_start, grid, *, face_mask_w, face_mask_s,
    dtype, device,
):
    """Reconstruct prescribed centre stress with explicit surface face masks."""
    from locomgcm.operators.reconstructions import (
        center_to_u_centered,
        center_to_v_centered,
    )

    tx = tau_x_at_step_start.to(dtype=dtype, device=device).view(
        1, 1, grid.ny, grid.nx
    )
    ty = tau_y_at_step_start.to(dtype=dtype, device=device).view(
        1, 1, grid.ny, grid.nx
    )
    tau_x_on_u = center_to_u_centered(tx, grid, face_mask=face_mask_w)
    tau_y_on_v = center_to_v_centered(ty, grid, face_mask=face_mask_s)
    return tau_x_on_u, tau_y_on_v


def prepare_kpp_wind_forcing(
    tau_x_on_u_at_step_start, tau_y_on_v_at_step_start, mass2r_unit, *, ny, nx,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert face stress to KPP surfaceForcingU/V in m^2/s^2.

    ``surfaceForcingU = fu*mass2rUnit`` with ``mass2rUnit = recip_rhoConst``
    computed once (external_forcing_surf.F:222, ini_parms.F:763, :1573): a
    multiply by the precomputed reciprocal, which differs from ``tau/rhoConst``
    in the last bit.  ``mass2r_unit`` is the stepper's prepared factor.
    """
    sfU = (tau_x_on_u_at_step_start[:, 0] * mass2r_unit).reshape(1, ny, nx + 1)
    sfV = (tau_y_on_v_at_step_start[:, 0] * mass2r_unit).reshape(1, ny + 1, nx)
    return sfU, sfV


def prepare_kpp_surface_forcing(
    surf_forc_t_at_step_start, surf_forc_s_at_step_start, qsw_at_step_start, *,
    ny, nx, dtype, device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert and shape the already-produced surface fields for regional KPP."""
    # surface heat forcing -> sfT [K m/s] (+ net shortwave qsw for KPP bosol).
    # KPP's sfT IS surface_heat_forcing's output (surface_flux.py docstring);
    # the penetrating shortwave is fed separately via surf_qsw. All fields stay
    sfT = surf_forc_t_at_step_start.to(dtype=dtype, device=device)
    qs = qsw_at_step_start.to(dtype=dtype, device=device)
    sfT = sfT.reshape(1, ny, nx)
    qs = qs.reshape(1, ny, nx)
    # Harvest the bundle's salt forcing too; without it the boundary layer
    # would see no freshwater buoyancy / nonlocal salt transport.
    sfS = surf_forc_s_at_step_start.to(dtype=dtype, device=device)
    sfS = sfS.reshape(1, ny, nx)
    return sfT, sfS, qs


def prepare_kpp_heat_salt_forcing(
    surf_forc_t_at_step_start, salt_flux, *, ny, nx, dtype, device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shape prescribed surfaceForcingT/S without changing their signs."""
    sfT = surf_forc_t_at_step_start
    if isinstance(sfT, torch.Tensor):
        sfT = sfT.to(dtype=dtype, device=device).reshape(1, ny, nx)
    else:
        sfT = torch.full((1, ny, nx), sfT, dtype=dtype, device=device)
    # salt_flux is already surfaceForcingS [PSU m/s], positive
    # into the ocean, as consumed by the tracer tendency.
    sfS = torch.full((1, ny, nx), salt_flux, dtype=dtype, device=device)
    return sfT, sfS


def prepare_regional_kpp_surface_inputs(
    *, static, surf_forc_t, surf_forc_u, surf_forc_v, surf_forc_s,
    surf_qsw, allow_freezing, freezing_flux, dtype, device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Resolve the surface fields consumed by regional KPP at step start."""
    if static is None:
        raise RuntimeError(
            "KPP static tensors not prepared: call prepare_kpp_static(...) "
            "before computing regional KPP fields"
        )
    sfT, sfU, sfV = surf_forc_t, surf_forc_u, surf_forc_v
    if sfT is None or sfU is None or sfV is None:
        raise ValueError("KPP enabled but surf_forc_t/surf_forc_u/surf_forc_v not set")
    sfT = sfT.to(dtype=dtype, device=device)
    if allow_freezing:
        if freezing_flux is None:
            raise ValueError("KPP with allow_freezing requires the FREEZE_SURFACE flux")
        # Both KPP surface buoyancy and nonlocal heat transport consume
        # surfaceForcingT + adjustColdSST_diag (kpp_forcing_surf.F,
        # kpp_transport_t.F). The ordinary heat budget already saw the clamp.
        sfT = sfT + freezing_flux
    elif freezing_flux is not None:
        raise ValueError("KPP freezing flux supplied with allow_freezing disabled")
    sfU = sfU.to(dtype=dtype, device=device)
    sfV = sfV.to(dtype=dtype, device=device)
    # Salt surface forcing -> bo salinity buoyancy + nonlocal salt transport.
    # Harvested from the producer bundle; absent (e.g. a harness
    # that injects KppStepForcing directly, or no freshwater-capable module) => 0.
    sfS = surf_forc_s
    sfS = (sfS.to(dtype=dtype, device=device) if sfS is not None
           else torch.zeros_like(sfT))
    # Net shortwave -> bosol (penetrating-SW buoyancy); absent => bosol=0.
    qsw = surf_qsw
    qsw = qsw.to(dtype=dtype, device=device) if qsw is not None else None
    return sfT, sfS, sfU, sfV, qsw


def prepare_regional_kpp_inputs(
    *, static, rho_const, surf_forc_t, surf_forc_s, surf_forc_u,
    surf_forc_v, surf_qsw, eos_pressure_pa_at_call, eos_module,
    run_values: KppRunValues, forcing_penetrating_sw,
    tracer_vertical_diffusivity, convective_implicit,
) -> RegionalKppInputs:
    """Assemble frozen KPP inputs with pressure sampled at the caller's station."""
    from locomgcm.dynamics.kpp_mixing import RegionalKppInputs

    params = kpp_driver_params(
        run_values, rho_const=rho_const, eos_module=eos_module,
        min_hbl_default=static.min_hbl_default,
    )
    params["penetrating_sw"] = forcing_penetrating_sw
    return RegionalKppInputs(
        surf_forc_t=surf_forc_t, surf_forc_s=surf_forc_s,
        surf_forc_u=surf_forc_u, surf_forc_v=surf_forc_v,
        surf_qsw=surf_qsw, static=static,
        background_diffusivity_s=float(tracer_vertical_diffusivity),
        background_diffusivity_t=float(tracer_vertical_diffusivity),
        eos_pressure_pa=eos_pressure_pa_at_call, params=params,
        convective_implicit=convective_implicit,
    )


def kpp_coriolis_profile_c(profile_c, *, ny, nx, dtype, device):
    """Construct the explicitly configured cell-centre Coriolis field."""
    # KPPMIX takes fCori at the cell centre (kpp_calc.F:555).  An
    # explicit profile overrides the grid-derived one below.
    fc = torch.as_tensor(profile_c, dtype=dtype, device=device)
    if tuple(fc.shape) != (ny,):
        raise ValueError(
            f"kpp_coriolis_profile_c has shape {tuple(fc.shape)}, "
            f"expected ({ny},) cell-centre rows"
        )
    return fc.view(ny, 1).expand(ny, nx).contiguous()


def kpp_coriolis_rows_from_spherical(centre_rows, *, dtype, device):
    """Convert the selected spherical cell-centre Coriolis rows."""
    # The momentum profile is the spherical map 2*Omega*sin(yG)
    # of this grid, and KPPMIX takes fCori = 2*Omega*sin(yC)
    # (kpp_calc.F:555, ini_cori.F:84-95); the two-corner
    # average would be fCori*cos(dlat/2).
    fc = torch.as_tensor(centre_rows, dtype=dtype, device=device)
    source = "2*Omega*sin of the grid's cell-centre latitudes"
    return fc, source


def kpp_coriolis_rows_from_corners(profile_g, *, dtype, device):
    """Average the selected f-plane or beta-plane corner profile to cell centres."""
    # f-plane or beta plane: linear in y, so the two-corner
    # average is the centre value.
    fg = torch.as_tensor(profile_g, dtype=dtype, device=device)  # (ny+1,)
    fc = 0.5 * (fg[:-1] + fg[1:])  # (ny,)
    source = "the two-corner average of coriolis_profile_g"
    return fc, source


def kpp_coriolis_field(fc, source, *, ny, nx):
    """Broadcast resolved cell-centre rows and report their selected source."""
    logging.getLogger("locomgcm.steppers.implicit_free_surface").info(
        "KPP Coriolis parameter: %s", source
    )
    return fc.view(ny, 1).expand(ny, nx).contiguous()


def resolve_kpp_coriolis(*, modules, profile_c, metrics, ny, nx, dtype, device, omega):
    coriolis = None
    if profile_c is not None:
        coriolis = kpp_coriolis_profile_c(
            profile_c,
            ny=ny, nx=nx, dtype=dtype, device=device,
        )
    if coriolis is None:
        for m in modules:
            prof = getattr(m, "coriolis_profile_g", None)
            if prof is not None:
                centre_rows = spherical_centre_coriolis(
                    prof,
                    centre=getattr(metrics, "yC_row", None),
                    corner=getattr(metrics, "yG_row", None),
                    omega=omega,
                )
                if centre_rows is not None:
                    fc, source = kpp_coriolis_rows_from_spherical(
                        centre_rows, dtype=dtype, device=device,
                    )
                else:
                    fc, source = kpp_coriolis_rows_from_corners(
                        prof, dtype=dtype, device=device,
                    )
                coriolis = kpp_coriolis_field(fc, source, ny=ny, nx=nx)
                break
            fpar = getattr(m, "coriolis_parameter", None)
            if fpar:
                coriolis = torch.full(
                    (ny, nx), float(fpar), dtype=dtype, device=device,
                )
                break
    return coriolis


def prepare_kpp_static(
    grid, layer_thickness, coriolis, *, coriolis_override, rho_const, dtype, device, omega,
) -> KppStaticInputs:
    """Build regional KPP static tensors outside traced code."""
    from locomgcm.dynamics.kpp_mixing import build_kpp_static_from_grid

    lt = layer_thickness
    if lt is None:
        raise ValueError("KPP enabled but layer_thickness not set")
    cor = coriolis
    if coriolis_override is not None:
        cor = torch.full(
            (grid.ny, grid.nx),
            coriolis_override,
            dtype=dtype,
            device=device,
        )
    static = build_kpp_static_from_grid(
        grid, lt.to(dtype=dtype, device=device), dtype=dtype, device=device,
        coriolis=cor.to(dtype=dtype, device=device) if cor is not None else None,
        omega=omega,
    )
    return KppStaticInputs(
        static=static, rho_const=rho_const,
    )


def kpp_eos_type(eos_name: str) -> str:
    """KPP driver ``eos_type`` of the equation-of-state module in use."""
    eos_types = dict(KPP_EOS_TYPES)
    if eos_name not in eos_types:
        raise ValueError(f"KPP does not support configured EOS module {eos_name!r}")
    return eos_types[eos_name]


def kpp_driver_params(
    run_values: KppRunValues, *, rho_const: float, eos_module: Any, min_hbl_default: float,
) -> dict:
    """KPP driver parameters of this run, shared by the lat-lon and cube steps.

    The package constants come from ``KPP_STOCK_PARAMS``; the run-scoped
    data.kpp overrides and the background viscosity come from ``run_values``.
    ``eos_type`` is the KPP family of ``eos_module`` (``KPP_EOS_TYPES``); a
    linear module also hands STATEKPP its rhoNil, tAlpha, sBeta, tRef, sRef.
    ``minKPPhbl`` uses ``run_values.min_hbl`` when set, otherwise the supplied
    ``min_hbl_default``, half the top layer (MITgcm ``-rC(1)``,
    ``kpp_init_fixed.F:162-164``). ``visc_arnr`` is ``viscArNr(1)``, the
    configured ``run_values.implicit_viscosity`` (``kpp_routines.F:1193-1206``).
    """
    from locomgcm.dynamics.kpp import KppLinearEos
    from locomgcm.dynamics.kpp_mixing import KPP_STOCK_PARAMS

    eos_type = kpp_eos_type(kpp_eos_name(eos_module))
    linear_eos = None
    if eos_type == "linear":
        linear_eos = KppLinearEos(
            reference_density=eos_module.reference_density,
            thermal_expansion=eos_module.thermal_expansion,
            haline_contraction=eos_module.haline_contraction,
            reference_temperature=eos_module.reference_temperature,
            reference_salinity=eos_module.reference_salinity,
        )
    params = {
        **KPP_STOCK_PARAMS,
        "gravity": run_values.gravity,
        "rho_const": rho_const,
        "eos_type": eos_type,
        "linear_eos": linear_eos,
        "visc_arnr": float(run_values.implicit_viscosity),
        "minKPPhbl": (
            run_values.min_hbl
            if run_values.min_hbl is not None
            else min_hbl_default
        ),
        "kpp_estimate_uref": run_values.estimate_uref,
        "kpp_smooth_shsq": run_values.smooth_shsq,
        "kpp_smooth_dbloc": run_values.smooth_dbloc,
        "sign_zero": run_values.sign_zero,
        "surface_buoyancy": run_values.surface_buoyancy,
        "dbloc_form": run_values.dbloc_form,
        "use_double_diff": run_values.use_double_diff,
    }
    if run_values.ricr is not None:
        params["Ricr"] = run_values.ricr
    return params
