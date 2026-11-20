"""Exact PP81 closure for the hybrid stepper.

This module compares LOCOMgcm and MITgcm with PP81 mixing. MITgcm runs
pkg/pp81; LOCOMgcm runs its regional stepper with KPP enabled and every KPP
diagnosis replaced, through ``LearnedKppInjection``, by ``Pp81Closure``.  The
KPP reference, the tridiagonal solvers and the stepper stay untouched.

``Pp81Closure.features_from_inputs`` hands the closure the temperature,
salinity and velocity the stepper passes to KPP (``RegionalKppInputs.diagnose``)
together with drF, the wet mask, g and rhoConst; ``forward`` returns
``pp81_coefficients`` of those (pp81_ri_number.F, pp81_calc.F,
pp81_calc_diff.F) in the closure output layout of ``closure_kpp``.  PP81 has
no non-local flux and no boundary layer depth: ``ghat`` and ``gamma_iface``
are 0, and ``hbl`` is 0, which enters only the shortwave share of the
non-local forcing that ``ghat = 0`` multiplies away.

The viscosity reaches the u/v faces through ``to_kpp_fields``, the average of
pp81_calc_visc.F:40-53, so the implicit viscosity at interfaces 2..Nr is the
MITgcm one.  One MITgcm consumer is not reproduced: with PP81 calc_viscosity.F
copies KappaRU(Nr) into KappaRU(Nr+1) (lines 121-130), which the no-slip
bottom drag reads at the bottom level (mom_u_botdrag_coeff.F:75,122-128);
the stepper's bottom drag keeps the background viscArNr there, as MITgcm does
with KPP.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from defgcm.dynamics.equation_of_state import reference_pressure_from_layers
from defgcm.learned.hidden import Pp81Params, pp81_coefficients
from defgcm.learned.hybrid import _crop_center


@dataclass(frozen=True)
class Pp81Inputs:
    """What PP81 reads from one KPP diagnosis of the stepper.

    ``temp``/``salt`` (B, nz, ny, nx); ``u`` (B, nz, ny, nx+1) and ``v``
    (B, nz, ny+1, nx) with both outer faces; ``drf`` (nz,); ``mask_c``
    (nz, ny, nx); ``gravity`` and ``rho_const`` are the KPP driver's.
    """

    temp: torch.Tensor
    salt: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor
    drf: torch.Tensor
    mask_c: torch.Tensor
    gravity: float
    rho_const: float


class Pp81Closure(nn.Module):
    """MITgcm PP81 as a closure of ``LearnedKppInjection`` (no parameters).

    ``nu0``, ``alpha``, ``n_ri``, ``visc_max`` are PPnu0, PPalpha, PPnRi and
    PPviscMax (pp81_readparms.F:64-69 defaults).  The backgrounds come from the
    stepper at every call: ``visc_arnr`` is viscArNr (also the RiLimit
    background, pp81_readparms.F:117-123), ``background_diff_t``/``_s`` are
    diffKrNrT/diffKrNrS.  ``last_params`` keeps the ``Pp81Params`` of the latest
    call so a driver can record what the closure used.
    """

    def __init__(self, *, nu0: float = 1.0e-2, alpha: float = 5.0, n_ri: int = 2,
                 visc_max: float = 1.0):
        super().__init__()
        self.nu0 = float(nu0)
        self.alpha = float(alpha)
        self.n_ri = int(n_ri)
        self.visc_max = float(visc_max)
        self.last_params: Pp81Params | None = None

    def features_from_inputs(self, inputs, temp, salt, u, v, *, grid) -> Pp81Inputs:
        """The stepper's KPP inputs, checked against what pp81_coefficients models.

        MITgcm evaluates both densities of pp81_ri_number.F at pRef4EOS(K)
        when selectP_inEOS_Zc <= 1 (pressure_for_eos.F).  The stepper hands KPP
        the pressure of its EOS module; it must be that static pressure, which
        ``pp81_coefficients`` forms in the MITgcm order rhoConst*|rC|*g.
        """
        params = inputs.params
        eos = str(params["eos_type"]).lower()
        if eos != "mdjwf":
            raise NotImplementedError(f"Pp81Closure supports eos_type 'mdjwf', got {eos!r}")
        if temp.ndim != 4:
            raise ValueError(f"temp must be (B, nz, ny, nx), got {tuple(temp.shape)}")
        nz, ny, nx = temp.shape[-3:]
        drf = inputs.static.layer_thickness.to(dtype=temp.dtype, device=temp.device)
        rho_const = float(params["rho_const"])
        gravity = float(params["gravity"])
        pressure = inputs.eos_pressure_pa
        if pressure is not None:
            # ponytail: static EOS pressure only (selectP_inEOS_Zc <= 1); a
            # dynamic-pressure case needs pp81_coefficients to take the
            # stepper's pressure of layer K for both densities.
            static = reference_pressure_from_layers(
                drf, rho_const=rho_const, gravity=gravity,
            ).reshape(1, nz, 1, 1).expand_as(pressure)
            if not torch.allclose(pressure, static, rtol=1.0e-12, atol=0.0):
                raise NotImplementedError(
                    "Pp81Closure models the static EOS pressure rhoConst*|rC|*g; the "
                    "stepper passes a different pressure (dynamic EOS pressure?)"
                )
        mask_c = _crop_center((grid.masks.hFacC > 0).to(temp.dtype), ny, nx)
        return Pp81Inputs(temp=temp, salt=salt, u=u, v=v, drf=drf, mask_c=mask_c,
                          gravity=gravity, rho_const=rho_const)

    def forward(self, feats: Pp81Inputs, *, background_diff_t, background_diff_s, visc_arnr,
                min_hbl) -> dict[str, torch.Tensor]:
        del min_hbl  # PP81 has no boundary layer depth
        for name, value in (("background_diff_t", background_diff_t),
                            ("background_diff_s", background_diff_s),
                            ("visc_arnr", visc_arnr)):
            if isinstance(value, torch.Tensor):
                # ponytail: scalar backgrounds only (diffKrNrT/S, viscArNr of one
                # level set); a 3-D diffKr would need ALLOW_3D_DIFFKR in PP81 too.
                raise NotImplementedError(f"Pp81Closure needs a scalar {name}")
        params = Pp81Params(
            visc_background=float(visc_arnr),
            diff_background_t=float(background_diff_t),
            diff_background_s=float(background_diff_s),
            gravity=feats.gravity,
            rho_const=feats.rho_const,
            nu0=self.nu0, alpha=self.alpha, n_ri=self.n_ri, visc_max=self.visc_max,
        )
        self.last_params = params
        out = pp81_coefficients(feats.temp, feats.salt, feats.u, feats.v,
                                drf=feats.drf, mask_c=feats.mask_c, params=params)
        batch, nz, ny, nx = feats.temp.shape
        zeros = feats.temp.new_zeros
        return {
            "diff_kzt": out["diff_kzt"],
            "diff_kzs": out["diff_kzs"],
            "visc_cell": out["visc_cell"],
            "ri": out["ri"],
            "ghat": zeros(batch, nz, ny, nx),
            "hbl": zeros(batch, ny, nx),
            "gamma_iface": zeros(batch, nz + 1, ny, nx),
        }
