"""Parameters a case yaml may write twice.

``physics.<field>`` feeds the MITgcm export; ``model.dynamics[<module>].kwargs.<key>``
or ``model.stepper.kwargs.<key>`` feeds the LOCOMgcm build. The table below is
the single list of such pairs. ``legacy_bridge`` uses it to reject a case whose
two spellings disagree. The loader uses it to keep an override
consistent: when an overlay or ``--set`` item writes one spelling and the merged
yaml carries both, the other spelling follows, so one flag changes one parameter.

Background tracer diffusion aliases are resolved by ``vertical_diffusion``;
they deliberately do not participate in this table's override mirroring.
"""

from __future__ import annotations

import copy
from typing import Any

DOUBLE_TRACK_FIELDS: tuple[tuple[str, str, str, str], ...] = (
    # (physics_field, kwarg_key, model_location, module_name)
    # location="dynamics" looks up model.dynamics[<module_name>].kwargs[<kwarg_key>];
    # location="stepper"  looks up model.stepper.kwargs[<kwarg_key>] (module_name unused).
    ("viscosity", "viscosity", "dynamics", "horizontal_viscosity"),
    ("viscosity_scheme", "viscosity_scheme", "dynamics", "horizontal_viscosity"),
    ("visc_c2_leith", "visc_c2_leith", "dynamics", "horizontal_viscosity"),
    ("visc_c2_smag", "visc_c2_smag", "dynamics", "horizontal_viscosity"),
    ("diffusivity", "diffusivity", "dynamics", "horizontal_tracer_diffusion"),
    ("enable_vertical_shear", "enable_vertical_shear", "dynamics", "vector_invariant_momentum_advection"),
    ("split_coriolis", "split_coriolis", "dynamics", "vector_invariant_momentum_advection"),
    # physics.ke_scheme drives MIT data.PARM01.selectKEscheme,
    # which controls mom_vecinv (and calc_r_star), not bot_drag. Lock it to
    # vec_inv's select_ke_scheme so the two consumers cannot silently diverge.
    ("ke_scheme", "select_ke_scheme", "dynamics", "vector_invariant_momentum_advection"),
    ("nonlin_free_surf", "nonlin_free_surf", "stepper", ""),
    ("implicit_viscosity", "implicit_viscosity", "stepper", ""),
    ("implicit_vertical_viscosity", "implicit_vertical_viscosity", "stepper", ""),
)


def _model_kwargs(tree: dict, location: str, module_name: str) -> dict | None:
    """Return the kwargs mapping of the addressed stepper or dynamics entry, or None."""
    model = tree.get("model")
    if not isinstance(model, dict):
        return None
    if location == "stepper":
        stepper = model.get("stepper")
        kwargs = stepper.get("kwargs") if isinstance(stepper, dict) else None
        return kwargs if isinstance(kwargs, dict) else None
    for entry in model.get("dynamics") or []:
        if isinstance(entry, dict) and entry.get("name") == module_name:
            kwargs = entry.get("kwargs")
            return kwargs if isinstance(kwargs, dict) else None
    return None


def _layer_writes_model(layer: dict, location: str, module_name: str, kwarg_key: str) -> bool:
    kwargs = _model_kwargs(layer, location, module_name)
    if kwargs is not None and kwarg_key in kwargs:
        return True
    model = layer.get("model")
    if location == "dynamics" and isinstance(model, dict):
        # An overlay or --set item may address the entry as dynamics[<name>].
        entry = model.get(f"dynamics[{module_name}]")
        kwargs = entry.get("kwargs") if isinstance(entry, dict) else None
        return isinstance(kwargs, dict) and kwarg_key in kwargs
    return False


def _layer_writes_physics(layer: dict, physics_field: str) -> bool:
    physics = layer.get("physics")
    return isinstance(physics, dict) and physics_field in physics


def mirror_double_track(layer: dict, merged: dict) -> dict:
    """Copy an override of one spelling onto the other.

    ``layer`` is the overlay, ``--set`` or extra mapping that was just merged
    into ``merged``. For every pair in ``DOUBLE_TRACK_FIELDS`` that the layer
    writes on exactly one side while ``merged`` carries both sides, the side the
    layer did not write takes the written value. A layer that writes both sides,
    or a case file that carries only one, is left alone; disagreement between
    the two spellings is still reported by ``legacy_bridge``.
    """
    if not isinstance(layer, dict) or not isinstance(merged, dict):
        return merged
    out = merged
    copied = False
    for physics_field, kwarg_key, location, module_name in DOUBLE_TRACK_FIELDS:
        wrote_physics = _layer_writes_physics(layer, physics_field)
        wrote_model = _layer_writes_model(layer, location, module_name, kwarg_key)
        if wrote_physics == wrote_model:
            continue
        physics = out.get("physics")
        kwargs = _model_kwargs(out, location, module_name)
        has_physics = isinstance(physics, dict) and physics_field in physics
        has_model = kwargs is not None and kwarg_key in kwargs
        if not (has_physics and has_model):
            continue
        if not copied:
            out = copy.deepcopy(out)
            copied = True
            physics = out["physics"]
            kwargs = _model_kwargs(out, location, module_name)
        if wrote_model:
            physics[physics_field] = kwargs[kwarg_key]
        else:
            kwargs[kwarg_key] = physics[physics_field]
    return out


__all__ = ["DOUBLE_TRACK_FIELDS", "mirror_double_track"]
