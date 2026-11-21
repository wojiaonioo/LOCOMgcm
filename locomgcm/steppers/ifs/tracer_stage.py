"""Regional tracer updates and staggered Adams-Bashforth retention.

Raw temperature/salinity tendencies occupy regional C-grid centres in
[B,nz,ny,nx] layout and keep their dtype. A reserved history preserves the
present-null restart slot separately from an absent or populated history.
Temperature and salinity updates use the input tensor promotion rules.
Column w uses MIT upward-positive metres per second and layer thickness
uses positive metres. Callers select the matrix and Euler-continuity hFac
time layers. Wet masks and the remaining column metrics use current geometry.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .carry import _carry


@dataclass(frozen=True)
class TracerABHistory:
    temp: torch.Tensor | None
    salt: torch.Tensor | None
    reserved: bool


def encode_tracer_ab_history(history: TracerABHistory | None):
    if history is None:
        return {}
    if history.reserved:
        return {"_ab2_prev_tracer_tendency": None}
    return {"_ab2_prev_tracer_tendency": {
        name: getattr(history, name) for name in ("temp", "salt")
        if getattr(history, name) is not None}}


def decode_tracer_ab_history(payload):
    if "_ab2_prev_tracer_tendency" not in payload:
        return None
    value = payload["_ab2_prev_tracer_tendency"]
    if value is None:
        return TracerABHistory(None, None, True)
    return TracerABHistory(value.get("temp"), value.get("salt"), False)


def retain_tracer_ab(tendency):
    return _carry(tendency).clone()


def advance_explicit(*, tracer, dt_tracer, tendency):
    return tracer + dt_tracer * tendency


def mix_and_solve_columns(
    *, temp_new, salt_new, temp_reference, salt_reference,
    kpp_enabled, kpp, layer_thickness, grid, dt_tracer, tracer_w,
    temperature_implicit_advection, salinity_implicit_advection,
    temperature_advection_scheme, salinity_advection_scheme, multi_dim_advection,
    nonlocal_tracer_names, continuity_names, convective_implicit,
    implicit_tracer_diffusion, kappa_rt, kappa_rs, nonlocal_temp_diffusivity,
    nonlocal_salt_diffusivity, recip_hfac, mask_c_current, recip_hfac_current_c,
    recip_hfac_continuity,
):
    if kpp_enabled:
        from locomgcm.dynamics.kpp_mixing import mix_kpp_tracers

        temp_new, salt_new = mix_kpp_tracers(
            temp_new, salt_new,
            temp_reference=temp_reference,
            salt_reference=salt_reference,
            temp_limiter_reference=temp_reference,
            salt_limiter_reference=salt_reference,
            kpp=kpp, temp_diffusivity=kappa_rt, salt_diffusivity=kappa_rs,
            nonlocal_temp_diffusivity=nonlocal_temp_diffusivity,
            nonlocal_salt_diffusivity=nonlocal_salt_diffusivity,
            layer_thickness=layer_thickness,
            grid=grid, dt=dt_tracer, tracer_w=tracer_w,
            temperature_implicit_advection=temperature_implicit_advection,
            salinity_implicit_advection=salinity_implicit_advection,
            temperature_advection_scheme=temperature_advection_scheme,
            salinity_advection_scheme=salinity_advection_scheme,
            multi_dim_advection=multi_dim_advection,
            nonlocal_tracer_names=nonlocal_tracer_names,
            continuity_tracer_names=continuity_names,
            convective_implicit=convective_implicit,
            recip_hfac=recip_hfac,
            mask_c_current=mask_c_current,
            recip_hfac_current_c=recip_hfac_current_c,
            recip_hfac_continuity=recip_hfac_continuity,
        )
    elif not (temperature_implicit_advection or salinity_implicit_advection) and implicit_tracer_diffusion:
        from locomgcm.dynamics.implicit_vertical_advection import solve_tracer_vertical_diffusion

        temp_new = solve_tracer_vertical_diffusion(
            temp_new, grid, dt_tracer, kappa_rt,
            recip_hFac=recip_hfac,
        )
        salt_new = solve_tracer_vertical_diffusion(
            salt_new, grid, dt_tracer, kappa_rs,
            recip_hFac=recip_hfac,
        )
    elif temperature_implicit_advection or salinity_implicit_advection:
        from locomgcm.dynamics.implicit_vertical_advection import (
            add_tracer_vertical_continuity,
            solve_tracer_implicit_vertical_advection,
            solve_tracer_vertical_diffusion,
        )

        if temperature_implicit_advection:
            if "temp" in continuity_names:
                tracer_n = temp_reference
                temp_new = add_tracer_vertical_continuity(
                    tracer_n=tracer_n, rhs=temp_new,
                    w=tracer_w, dt_lev=dt_tracer, grid=grid,
                    mask_c_current=mask_c_current,
                    recip_hfac_matrix_c=recip_hfac_continuity,
                )
            temp_new = solve_tracer_implicit_vertical_advection(
                rhs=temp_new,
                advection_scheme=temperature_advection_scheme,
                limiter_tracer=temp_new if multi_dim_advection else temp_reference,
                w=tracer_w,
                dt_lev=dt_tracer,
                diffusivity=kappa_rt,
                grid=grid,
                mask_c_current=mask_c_current,
                recip_hfac_matrix_c=recip_hfac,
                recip_hfac_current_c=recip_hfac_current_c,
            )
        elif implicit_tracer_diffusion:
            temp_new = solve_tracer_vertical_diffusion(
                temp_new, grid, dt_tracer, kappa_rt,
                recip_hFac=recip_hfac,
            )
        if salinity_implicit_advection:
            if "salt" in continuity_names:
                tracer_n = salt_reference
                salt_new = add_tracer_vertical_continuity(
                    tracer_n=tracer_n, rhs=salt_new,
                    w=tracer_w, dt_lev=dt_tracer, grid=grid,
                    mask_c_current=mask_c_current,
                    recip_hfac_matrix_c=recip_hfac_continuity,
                )
            salt_new = solve_tracer_implicit_vertical_advection(
                rhs=salt_new,
                advection_scheme=salinity_advection_scheme,
                limiter_tracer=salt_new if multi_dim_advection else salt_reference,
                w=tracer_w,
                dt_lev=dt_tracer,
                diffusivity=kappa_rs,
                grid=grid,
                mask_c_current=mask_c_current,
                recip_hfac_matrix_c=recip_hfac,
                recip_hfac_current_c=recip_hfac_current_c,
            )
        elif implicit_tracer_diffusion:
            salt_new = solve_tracer_vertical_diffusion(
                salt_new, grid, dt_tracer, kappa_rs,
                recip_hFac=recip_hfac,
            )
    return temp_new, salt_new


def mask_tracer_pair(temp, salt, *, mask_c):
    return temp * mask_c, salt * mask_c
