"""Resolve background tracer diffusion once and bind the selected implementation.

No tensor or model is constructed here. Bindings are receipts for assignments,
not a separate catalogue of possible consumers.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._schema import DynamicsEntry, TracerConfig


DEFAULT_DIFFUSIVITY_KR = 0.0
DEFAULT_IMPLICIT_DIFFUSION = False


class VerticalDiffusionError(ValueError):
    """An ambiguous declaration or an unavailable diffusion contribution."""


@dataclass(frozen=True)
class VerticalDiffusion:
    diffusivity_explicit: bool
    implicit_explicit: bool
    stepper_diffusivity_explicit: bool
    stepper_implicit_explicit: bool
    module_diffusivity_explicit: bool
    diffusivity_kr: float
    implicit: bool
    diffusivity_sources: tuple[str, ...]
    implicit_sources: tuple[str, ...]
    consumer: str
    stepper_name: str
    accepts_implicit: bool
    module_index: int | None
    module_tracers: tuple[str, ...]

    @property
    def stepper_diffusivity_kr(self) -> float:
        # Module coefficients must not enter stepper carry or KPP cold start.
        return 0.0 if self.consumer == "vertical_mixing" else self.diffusivity_kr


@dataclass(frozen=True)
class Binding:
    option: str
    contribution: str
    implementation: str
    field: str
    side: str
    value: Any


@dataclass(frozen=True)
class DiffusionAssembly:
    stepper: Any
    dynamics: tuple[Any, ...]
    bindings: tuple[Binding, ...]


def resolve_vertical_diffusion(*, tracer: TracerConfig, stepper_name: str | None,
                               stepper_kwargs: Mapping[str, Any],
                               dynamics: Sequence[DynamicsEntry], grid_kind: str) -> VerticalDiffusion:
    """Resolve declaration fragments without loading run inputs or a model."""
    import defgcm.steppers  # noqa: F401 - register constructor capabilities
    from defgcm.steppers._registry import STEPPERS

    kw = stepper_kwargs
    try:
        cls = STEPPERS.get(stepper_name)
    except KeyError as exc:
        raise VerticalDiffusionError(f"stepper {stepper_name!r} not registered") from exc
    accepts = "implicit_tracer_diffusion" in inspect.signature(cls).parameters
    p_kr = "physics.tracer.diffusivity_kr"
    p_implicit = "physics.tracer.implicit_vertical_diffusion"
    s_kr = "model.stepper.kwargs.tracer_vertical_diffusivity_kr"
    s_implicit = "model.stepper.kwargs.implicit_tracer_diffusion"
    declarations = []
    implicit_declarations = []
    if tracer.diffusivity_kr is not None:
        declarations.append((p_kr, float(tracer.diffusivity_kr)))
    if kw.get("tracer_vertical_diffusivity_kr") is not None:
        declarations.append((s_kr, float(kw["tracer_vertical_diffusivity_kr"])))
    if tracer.implicit_vertical_diffusion is not None:
        implicit_declarations.append((p_implicit, tracer.implicit_vertical_diffusion))
    if kw.get("implicit_tracer_diffusion") is not None:
        implicit_declarations.append((s_implicit, kw["implicit_tracer_diffusion"]))
    implicit = bool(implicit_declarations[0][1]) if implicit_declarations else DEFAULT_IMPLICIT_DIFFUSION
    if any(value != implicit for _, value in implicit_declarations):
        raise VerticalDiffusionError(
            "implicit tracer diffusion dual-entry mismatch: "
            f"{p_implicit}={tracer.implicit_vertical_diffusion!r} but model.stepper.kwargs gives "
            f"{kw['implicit_tracer_diffusion']!r}"
        )
    if implicit and not accepts:
        raise VerticalDiffusionError(
            "physics.tracer.implicit_vertical_diffusion=True requires a "
            "stepper with an implicit_tracer_diffusion consumer"
        )
    stagger = bool(kw.get("stagger_tracer_stepping", False))
    modules = []
    inactive_tracer_modules = []
    for i, entry in enumerate(dynamics):
        if entry.name != "vertical_mixing":
            continue
        module_kw = entry.kwargs
        if (module_kw.get("mix_momentum", False)
                and module_kw.get("viscosity") is None
                and module_kw.get("diffusivity") is None):
            raise VerticalDiffusionError(
                f"model.dynamics[{i}].vertical_mixing mixes momentum without a coefficient: "
                "declare viscosity explicitly, or declare diffusivity on the module "
                "to use it for both momentum and tracers"
            )
        if not module_kw.get("tracer_names", ("temp", "salt")):
            continue
        if module_kw.get("diffusivity") is not None:
            declarations.append((f"model.dynamics[{i}].kwargs.diffusivity",
                                 float(module_kw["diffusivity"])))
        if stagger and module_kw.get("tendency_phase", "tracer") != "tracer":
            inactive_tracer_modules.append(i)
        else:
            modules.append((i, entry))
    if len(modules) > 1:
        raise VerticalDiffusionError("background tracer diffusion has multiple vertical_mixing consumers")
    kpp = bool(kw.get("kpp_enabled", False))
    if kpp and not accepts:
        raise VerticalDiffusionError("KPP background tracer diffusion requires a stepper consumer")
    if (modules or inactive_tracer_modules) and (implicit or kpp):
        raise VerticalDiffusionError(
            "double vertical mixing: vertical_mixing with tracers (including modules outside "
            "the tracer phase) and stepper implicit diffusion or KPP are both selected; "
            "remove tracer diffusion from the module, or disable implicit diffusion and KPP "
            "and put the tracer module in an active tracer phase"
        )
    module_index, module_tracers, module_explicit = None, (), False
    if modules:
        module_index, entry = modules[0]
        module_tracers = tuple(entry.kwargs.get("tracer_names", ("temp", "salt")))
        module_explicit = entry.kwargs.get("diffusivity") is not None
    kr = declarations[0][1] if declarations else DEFAULT_DIFFUSIVITY_KR
    for path, value in declarations[1:]:
        if not math.isclose(kr, value, rel_tol=1.0e-12, abs_tol=0.0):
            raise VerticalDiffusionError(
                "implicit tracer diffusivity dual-entry mismatch: "
                f"{declarations[0][0]}={kr!r} but {path} gives {value!r}"
            )
    cube = grid_kind == "cubed_sphere"
    if not cube and not dynamics and kr != 0.0:
        raise VerticalDiffusionError(
            "nonzero background tracer diffusion requires explicit model.dynamics: "
            "empty dynamics installs a hidden default VerticalMixing; declare the "
            "dynamics modules explicitly, or declare diffusivity_kr=0"
        )
    if modules:
        consumer = "vertical_mixing"
    elif kpp:
        consumer = "kpp_background"
    elif implicit:
        consumer = "ifs_implicit"
    elif cube and accepts:
        consumer = "cube_explicit"
    else:
        consumer = "off"
        if kr != 0.0:
            if inactive_tracer_modules:
                raise VerticalDiffusionError(
                    f"{declarations[0][0]}={kr!r} has no tracer consumer: "
                    f"vertical_mixing at model.dynamics indices {inactive_tracer_modules} "
                    "is outside the tracer phase under stagger_tracer_stepping; "
                    "put tracer diffusion in a tendency_phase='tracer' module, "
                    "or disable stagger_tracer_stepping"
                )
            raise VerticalDiffusionError(
                f"{p_kr}={kr!r} has no consumer: enable implicit_vertical_diffusion, "
                "add an explicit vertical_mixing tracer module, or declare diffusivity_kr=0"
            )
    return VerticalDiffusion(
        tracer.diffusivity_kr is not None, tracer.implicit_vertical_diffusion is not None,
        kw.get("tracer_vertical_diffusivity_kr") is not None,
        kw.get("implicit_tracer_diffusion") is not None, module_explicit,
        kr, implicit, tuple(path for path, _ in declarations) or ("default:0.0",),
        tuple(path for path, _ in implicit_declarations) or ("default:False",),
        consumer, stepper_name, accepts, module_index, module_tracers,
    )


def vertical_diffusion_from_mapping(config: Mapping[str, Any], *, error_type=None) -> VerticalDiffusion:
    """Adapt YAML declarations without resolving sources, grids, or input paths."""
    from ._schema import DynamicsEntry, TracerConfig

    physics = config.get("physics") or {}
    tracer = physics.get("tracer") or {}
    model = config.get("model") or {}
    stepper = model.get("stepper") or {}
    dynamics = tuple(
        DynamicsEntry(entry, {}) if isinstance(entry, str)
        else DynamicsEntry(entry.get("name"), dict(entry.get("kwargs") or {}))
        for entry in model.get("dynamics") or ()
    )
    try:
        return resolve_vertical_diffusion(
            tracer=TracerConfig(
                implicit_vertical_diffusion=(None if tracer.get("implicit_vertical_diffusion") is None
                                             else bool(tracer["implicit_vertical_diffusion"])),
                diffusivity_kr=(None if tracer.get("diffusivity_kr") is None
                                else float(tracer["diffusivity_kr"])),
            ),
            stepper_name=stepper.get("name"), stepper_kwargs=stepper.get("kwargs") or {},
            dynamics=dynamics, grid_kind=(config.get("grid") or {}).get("kind", "latlon"),
        )
    except VerticalDiffusionError as exc:
        if error_type is None:
            from defgcm.io.runner.builders.from_manifest import ManifestBuildError
            error_type = ManifestBuildError
        raise error_type(str(exc)) from exc


def resolved_vertical_diffusion(run_cfg, *, error_type=None) -> VerticalDiffusion:
    """Read the RunConfig's cached definition, using the boundary's error class."""
    try:
        return run_cfg.vertical_diffusion
    except VerticalDiffusionError as exc:
        if error_type is None:
            from defgcm.io.runner.builders.from_manifest import ManifestBuildError
            error_type = ManifestBuildError
        raise error_type(str(exc)) from exc


def _assign(target, key, value, bindings, *, option, contribution, implementation, side):
    target[key] = value
    bindings.append(Binding(option, contribution, implementation, key, side, value))


def assemble_vertical_diffusion(run_cfg) -> DiffusionAssembly:
    """Return constructor arguments and receipts for exactly those assignments."""
    resolved = resolved_vertical_diffusion(run_cfg)
    kwargs = dict(run_cfg.model.stepper.kwargs)
    # Aliases are declarations, including an explicit false/zero on a stepper
    # without these constructor fields. Only the selected consumer gets them.
    kwargs.pop("implicit_tracer_diffusion", None)
    kwargs.pop("tracer_vertical_diffusivity_kr", None)
    dynamics = list(run_cfg.model.dynamics)
    bindings = []
    if resolved.accepts_implicit:
        _assign(kwargs, "implicit_tracer_diffusion", resolved.implicit, bindings,
                option="implicit_vertical_diffusion", contribution="background_tracer_diffusion",
                implementation=resolved.consumer, side="TO")
        _assign(kwargs, "tracer_vertical_diffusivity_kr", resolved.stepper_diffusivity_kr, bindings,
                option="diffusivity_kr", contribution="background_tracer_diffusion",
                implementation=resolved.consumer if resolved.module_index is None else "stepper_background_off",
                side="TO")
    if resolved.module_index is not None:
        index = resolved.module_index
        entry = dynamics[index]
        module_kwargs = dict(entry.kwargs)
        _assign(module_kwargs, "diffusivity", resolved.diffusivity_kr, bindings,
                option="diffusivity_kr", contribution="background_tracer_diffusion",
                implementation=f"model.dynamics[{index}].vertical_mixing", side="TO")
        dynamics[index] = replace(entry, kwargs=module_kwargs)
    return DiffusionAssembly(replace(run_cfg.model.stepper, kwargs=kwargs), tuple(dynamics), tuple(bindings))


def bind_mit_vertical_diffusion(resolved: VerticalDiffusion, *, write=None,
                                fields=("diffKrT", "diffKrS", "implicitDiffusion")):
    """Write selected MIT fields and return receipts for those writes.

    ``write`` accepts a namelist field and its typed value. Translation writers
    render there; the inspection entry uses the same assignments in a dictionary.
    """
    values, bindings = {}, []
    for key in fields:
        if key not in ("diffKrT", "diffKrS", "implicitDiffusion"):
            raise ValueError(f"unknown vertical diffusion field: {key}")
        implicit = key == "implicitDiffusion"
        value = resolved.implicit if implicit else resolved.diffusivity_kr
        if write is not None:
            write(key, value)
        _assign(values, key, value, bindings,
                option="implicit_vertical_diffusion" if implicit else "diffusivity_kr",
                contribution="background_tracer_diffusion", implementation="data.PARM01", side="MIT")
    return values, tuple(bindings)


def describe_vertical_diffusion(run_cfg) -> dict:
    """Inspect declarations and construction without allocating model tensors."""
    from defgcm.io.runner.layer_d.consumers.mit_binary import vertical_diffusion_export_args

    assembly = assemble_vertical_diffusion(run_cfg)
    resolved = resolved_vertical_diffusion(run_cfg)
    mit, bindings = vertical_diffusion_export_args(run_cfg)
    return {"resolved": asdict(resolved), "stepper_kwargs": assembly.stepper.kwargs,
            "module_kwargs": [dict(e.kwargs) for e in assembly.dynamics if e.name == "vertical_mixing"],
            "mit_scope": "Regional export W arguments and bindings; casegen edits are in pjob plan.json",
            "mit": mit, "bindings": [asdict(b) for b in (*assembly.bindings, *bindings)]}


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    from defgcm.io.runner.config import load_run_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("yaml", type=Path)
    print(json.dumps(describe_vertical_diffusion(load_run_config(parser.parse_args().yaml)), indent=2))
