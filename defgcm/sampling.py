"""Component-owned clocks for eagerly sampled step inputs.

START is the step's input time; END is input time plus dt_clock. Boundary
application has a separate entry clock because stored faces enter at START.
Python timestamps and interpolation/cache metadata are never scientific inputs.
"""
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import StrEnum


class SampleClock(StrEnum):
    START = "start"
    END = "end"
    INDEPENDENT = "independent"


@dataclass(frozen=True)
class SamplingRule:
    clock: SampleClock
    inputs: tuple[str, ...] = ()
    entry_clock: SampleClock | None = None
    providers: tuple[str, ...] = ()

    def at_entry(self) -> SampleClock:
        return self.clock if self.entry_clock is None else self.entry_clock


def sampling_rule(component: object) -> SamplingRule:
    rule = getattr(component, "sampling_rule", None)
    name = getattr(component, "name", type(component).__name__)
    if not isinstance(rule, SamplingRule):
        raise ValueError(f"{name}: missing sampling_rule (SamplingRule clock contract)")
    if not isinstance(rule.clock, SampleClock) or (
        rule.entry_clock is not None and not isinstance(rule.entry_clock, SampleClock)
    ):
        raise ValueError(f"{name}: sampling_rule requires SampleClock declarations")
    return rule


def sampling_components(boundary: object, modules=()) -> tuple[object, ...]:
    """Validate assembly and visit only explicitly declared provider edges.

    The update hook detects undeclared participants; it never supplies a clock.
    Stateless sources also declare their clock, although their owning component
    stores the sampled tensors and calls their sample method.
    """
    result = []
    seen = set()

    def visit(component):
        if component is None or id(component) in seen:
            return
        seen.add(id(component))
        rule = sampling_rule(component)
        result.append(component)
        for attribute in rule.providers:
            if not hasattr(component, attribute):
                raise ValueError(
                    f"{getattr(component, 'name', type(component).__name__)}: "
                    f"missing declared sampling provider {attribute}"
                )
            provider = getattr(component, attribute)
            if provider is not None:
                provider_rule = sampling_rule(provider)
                if provider_rule.clock not in (rule.clock, SampleClock.INDEPENDENT):
                    raise ValueError(
                        f"{getattr(component, 'name', type(component).__name__)}.{attribute}: "
                        f"sampling clock {provider_rule.clock.name} conflicts with {rule.clock.name}"
                    )
            visit(provider)

    visit(boundary)
    for component in modules:
        if (callable(getattr(component, "update_model_time", None))
                or getattr(component, "sampling_rule", None) is not None):
            visit(component)
    return tuple(result)


def sampled_host_paths(value: object, path: str) -> tuple[str, ...]:
    """Find scientific host scalars, including those hidden inside containers.

    None is an explicit empty channel. Dtype/device tags in prepared target
    caches are run constants. Numerical scalar inputs must be tensors.
    """
    import torch

    seen = set()

    def visit(item, location):
        if item is None or isinstance(item, (torch.Tensor, torch.dtype, torch.device)):
            return []
        if id(item) in seen:
            return []
        seen.add(id(item))
        if isinstance(item, Mapping):
            children = ((f"[{key!r}]", child) for key, child in item.items())
        elif isinstance(item, (tuple, list)):
            children = ((f"[{index}]", child) for index, child in enumerate(item))
        elif is_dataclass(item) and not isinstance(item, type):
            children = ((f".{field.name}", getattr(item, field.name)) for field in fields(item))
        else:
            return [location]
        return [found for suffix, child in children for found in visit(child, location + suffix)]

    return tuple(visit(value, path))
