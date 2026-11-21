"""MITgcm convective adjustment (``cAdjFreq``, ``convective_adjustment.F``).

Tracers are [B, nz, ny, nx] at cell centres, index 0 the surface level; ``hfac_c`` is
[nz, ny, nx] (the 0/1 centre mask where the grid has no partial cells), ``drf`` the
positive layer thicknesses [nz] in m.  Densities are MITgcm's linear-EOS ``rhoLoc``
(kg m-3, ``find_rho.F:94-108``); every tensor keeps the tracer dtype.  The adjustment is
not a tendency: the implicit free-surface stepper calls it after the tracer update, where
MITgcm's TRACERS_CORRECTION_STEP does (``tracers_correction_step.F:114-118``).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from locomgcm.dynamics._registry import register_dynamics_module
from locomgcm.dynamics.base import TendencyModule, TendencyStepDomain
from locomgcm.dynamics.equation_of_state import linear_density_anomaly

#: Equation-of-state modules whose FIND_RHO_2D the adjustment forms; the cAdjFreq
#: experiments of MITgcm's verification all run eosType='LINEAR'.
ADJUSTMENT_EQUATIONS_OF_STATE = ("linear_equation_of_state",)
#: rkSign*gravitySign of a z-coordinate ocean (convective_adjustment.F:90-97 picks kTop=2,
#: kBottom=Nr, deltaK=-1 when it is positive; LOCOMgcm has no p-coordinate).
RK_SIGN_GRAVITY_SIGN = 1.0


def convective_adjustment(
    temp: torch.Tensor,
    salt: torch.Tensor,
    *,
    hfac_c: torch.Tensor,
    drf: torch.Tensor,
    rho_nil: float | torch.Tensor,
    rho_const: float,
    thermal_expansion: float,
    haline_contraction: float,
    reference_temperature: torch.Tensor,
    reference_salinity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mix each statically unstable pair of layers, interface by interface from the top.

    convective_adjustment.F:100-163: for MITgcm interface k = 2..Nr (index k = 1..nz-1 here)
    both layers' densities take the reference level kRef = k-1 of the upper layer
    (FIND_RHO_2D with k+deltaK), the weights come from CONVECTIVE_WEIGHTS and theta, then
    salt, are mixed with them (CONVECTIVELY_MIXTRACER); interface k sees the layers already
    mixed across interface k-1.  ``reference_temperature``/``reference_salinity`` are
    tRef/sRef per level [nz].
    """
    temp_levels = list(temp.unbind(-3))
    salt_levels = list(salt.unbind(-3))
    zero = temp.new_zeros(())
    for k in range(1, len(temp_levels)):
        coefficients = dict(
            reference_density=rho_nil, rho_const=rho_const, thermal_expansion=thermal_expansion,
            haline_contraction=haline_contraction,
            reference_temperature=reference_temperature[k - 1],
            reference_salinity=reference_salinity[k - 1])
        # find_rho.F:94-108 rhoLoc = rhoNil*(sBeta*(S-sRef(kRef)) - tAlpha*(T-tRef(kRef))) + dRho
        rho_km1 = linear_density_anomaly(temp_levels[k - 1], salt_levels[k - 1], **coefficients)
        rho_k = linear_density_anomaly(temp_levels[k], salt_levels[k], **coefficients)
        # convective_weights.F:55-72
        unstable = ((hfac_c[k - 1] * hfac_c[k] > 0)
                    & ((rho_k - rho_km1) * RK_SIGN_GRAVITY_SIGN < 0))
        d1 = hfac_c[k - 1] * drf[k - 1]
        d2 = hfac_c[k] * drf[k]
        ds = torch.where(unstable, d1 + d2, torch.ones_like(d1))
        weight_a = torch.where(unstable, d2 / ds, zero)
        weight_b = torch.where(unstable, d1 / ds, zero)
        # convectively_mixtracer.F:43-53, theta then salt
        for levels in (temp_levels, salt_levels):
            del_trac = levels[k] - levels[k - 1]
            levels[k - 1] = levels[k - 1] + weight_a * del_trac
            levels[k] = levels[k] - weight_b * del_trac
    return torch.stack(temp_levels, dim=-3), torch.stack(salt_levels, dim=-3)


@dataclass(frozen=True)
@register_dynamics_module("convective_adjustment")
class ConvectiveAdjustment(TendencyModule):
    """MITgcm ``cAdjFreq`` with INCLUDE_CONVECT_CALL.

    ``c_adj_freq`` is the resolved interval in s (ini_parms.F:1104-1106 turns a negative
    cAdjFreq into deltaTClock) and ``dt_clock`` MITgcm's deltaTClock.  DIFFERENT_MULTIPLE
    (different_multiple.F) is true on every step when ``0 < cAdjFreq <= deltaTClock``; a longer
    interval needs a per-step host decision that is not ported and is refused.
    ``initial_adjustment`` is MITgcm's INCLUDE_CONVECT_INI_CALL together with its run-time
    condition ``startTime .EQ. baseTime`` (initialise_varia.F:281-295), resolved by the
    configuration: ``initialize_fresh_step_state`` then adjusts the initial tracers once with
    the same loop (convective_adjustment_ini.F).
    """

    c_adj_freq: float
    dt_clock: float
    initial_adjustment: bool = False
    name: str = "convective_adjustment"
    stepper_domain: TendencyStepDomain = TendencyStepDomain.TRACER_CORRECTION

    def __post_init__(self) -> None:
        if not 0.0 < self.c_adj_freq <= self.dt_clock:
            raise ValueError(
                f"convective_adjustment c_adj_freq={self.c_adj_freq} with dt_clock={self.dt_clock}: "
                "only an adjustment on every step (0 < cAdjFreq <= deltaTClock) is ported")

    def evaluate(self, state, diagnostics, grid):
        raise NotImplementedError(
            "convective_adjustment is called by the implicit free-surface stepper after the "
            "tracer update (MITgcm TRACERS_CORRECTION_STEP); it is not an assembled tendency")

    def adjust(self, temp: torch.Tensor, salt: torch.Tensor, *, grid, modules) -> tuple[torch.Tensor, torch.Tensor]:
        """Adjust the updated tracers with the composition's linear EOS and rhoConst."""
        eos = [m for m in modules if getattr(m, "name", "") in ADJUSTMENT_EQUATIONS_OF_STATE]
        pressure_gradient = [m for m in modules if getattr(m, "name", "") == "hydrostatic_pressure_gradient"]
        if len(eos) != 1 or len(pressure_gradient) != 1:
            raise ValueError(
                "convective_adjustment needs one equation of state among "
                f"{ADJUSTMENT_EQUATIONS_OF_STATE} and one hydrostatic_pressure_gradient (rhoConst)")
        eos = eos[0]
        dtype, device = temp.dtype, temp.device
        hfac_c = grid.masks.hFacC
        if hfac_c is None:
            # Full cells: hFacC is the centre mask.
            hfac_c = grid.masks.center_float(dtype=dtype, device=device)
            if hfac_c.ndim == 2:
                hfac_c = hfac_c.expand(grid.nz, *hfac_c.shape)
        nz = grid.nz
        reference = [torch.as_tensor(value, dtype=dtype, device=device).expand(nz)
                     for value in (eos.reference_temperature, eos.reference_salinity)]
        return convective_adjustment(
            temp, salt,
            hfac_c=hfac_c.to(dtype=dtype, device=device),
            drf=grid.layer_thickness.to(dtype=dtype, device=device),
            rho_nil=eos.reference_density,
            rho_const=float(pressure_gradient[0].reference_density),
            thermal_expansion=float(eos.thermal_expansion),
            haline_contraction=float(eos.haline_contraction),
            reference_temperature=reference[0],
            reference_salinity=reference[1],
        )


def convective_adjustment_of(modules) -> ConvectiveAdjustment | None:
    """The composition's convective adjustment, or None when MITgcm's cAdjFreq is 0."""
    return next((m for m in modules if isinstance(m, ConvectiveAdjustment)), None)


def validate_convective_adjustment_composition(*, modules, face_count: int) -> None:
    """The cube step has no tracer-correction call; refuse the module there."""
    if face_count != 1 and convective_adjustment_of(modules) is not None:
        raise ValueError("convective_adjustment (MITgcm cAdjFreq) is not ported to the cubed sphere")
