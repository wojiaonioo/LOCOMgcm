"""The sea-ice component's carry in the ocean stepper and the composition it needs.

Kept apart from ``locomgcm.seaice.coupling`` so that the stepper, the model and
the checkpoint codec read it without importing the ice program and its EXF
inputs.  ``SeaiceStepState`` holds float64 native ice tensors
(``locomgcm.seaice.records``); the checkpoint stores them under one key.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

from .records import Clock, DynamicsCarry, SeaiceCarry, SeaiceState, ThsiceState

if TYPE_CHECKING:
    from .coupling import SeaiceCoupling
    from .forcing import ForcingSample

#: Restart key of the ice carry in the stepper's step state.
SEAICE_RESTART_KEY = "_seaice"
#: Surface-flux modules whose fluxes the ice component's own EXF replaces.
_OTHER_SURFACE_FLUX_MODULES = (
    "era5_bulk_forcing", "era5_wind_forcing", "surface_wind_stress",
    "surface_heat_salt_flux", "wpflux_surface_forcing",
)


@dataclass(frozen=True)
class SeaiceStepState:
    """The ice carry across ocean steps and the forcing sample its EXF reuses.

    ``previous_sample`` keeps the period-zero wind halo of the last step
    (``exf_set_fld.F:122-124``); a restart starts it again from ``None``, as
    the offline program does.
    """

    carry: SeaiceCarry
    previous_sample: ForcingSample | None


def encode_seaice_state(state: SeaiceStepState | None):
    if state is None:
        return {}
    carry = state.carry
    return {SEAICE_RESTART_KEY: {
        "seaice": {f.name: getattr(carry.seaice, f.name) for f in fields(SeaiceState)},
        "thsice": {f.name: getattr(carry.thsice, f.name) for f in fields(ThsiceState)},
        "dynamics": {f.name: getattr(carry.dynamics, f.name) for f in fields(DynamicsCarry)},
        "sIceLoad": carry.sIceLoad,
        "clock": {f.name: getattr(carry.clock, f.name) for f in fields(Clock)},
    }}


def decode_seaice_state(payload):
    if SEAICE_RESTART_KEY not in payload:
        return None
    value = payload[SEAICE_RESTART_KEY]
    carry = SeaiceCarry(SeaiceState(**value["seaice"]), ThsiceState(**value["thsice"]),
                        DynamicsCarry(**value["dynamics"]), value["sIceLoad"],
                        Clock(**value["clock"]))
    return SeaiceStepState(carry, None)


def seaice_module(model) -> "SeaiceCoupling | None":
    """The model's sea-ice component, the dynamics module named ``seaice``."""
    return next((m for m in getattr(model, "dynamics_modules", ()) or ()
                 if getattr(m, "name", "") == "seaice"), None)


def validate_seaice_composition(*, modules, stepper, grid) -> None:
    """Refuse the compositions the ocean side of the coupling does not cover.

    Ported: the tracer forcing of the published fluxes and AREA for restoring,
    with a linear free surface, on one rank of a doubly periodic grid the ice
    space covers.  Not ported: the momentum forcing fu, fv
    (``external_forcing_surf.F:222-224``), the KPP surface forcing, the real
    freshwater flux (EmPmR in continuity and sIceLoad in phi0surf,
    ``external_forcing_surf.F:149-166,367-373``) and a nonlinear free surface.
    The ice's own EXF produces the open-water fluxes, so no other surface-flux
    module may run beside it.
    """
    if not any(getattr(m, "name", "") == "seaice" for m in modules):
        return
    refused = [name for name, on in (
        ("mom_stepping=True (the fu, fv momentum forcing is not ported)",
         getattr(stepper, "mom_stepping", True)),
        ("KPP (its surface forcing harvest does not read the ice fluxes)",
         getattr(stepper, "kpp_enabled", False)),
        ("use_real_freshwater_flux (EmPmR in continuity and sIceLoad in phi0surf are not ported)",
         getattr(stepper, "use_real_freshwater_flux", False)),
        ("nonlin_free_surf>0", getattr(stepper, "nonlin_free_surf", 0) > 0),
        ("the cubed sphere", getattr(grid, "face_count", 1) != 1),
        ("several ranks", getattr(getattr(grid, "parallel_layout", None), "world_size", 1) > 1),
        ("a grid that is not doubly periodic (the ice space is)",
         not (grid.edge_face_couples("x") and grid.edge_face_couples("y"))),
    ) if on]
    others = sorted({getattr(m, "name", "") for m in modules} & set(_OTHER_SURFACE_FLUX_MODULES))
    if others:
        refused.append(f"the surface-flux modules {others} (the ice component's EXF produces "
                       "the surface fluxes)")
    if refused:
        raise ValueError(f"the seaice module does not support {'; '.join(refused)}")
