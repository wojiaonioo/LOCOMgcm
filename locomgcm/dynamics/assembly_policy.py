"""Time-step assembly policy for momentum and tracer tendencies."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Iterable, Mapping, Self

from .base import MomentumTermClass, TendencyModule


class AssemblyBucket(StrEnum):
    AB2 = "ab2"
    POST_AB2 = "post_ab2"


@dataclass(frozen=True)
class AssemblyPolicy:
    """Map physical momentum term classes to their integration position.

    The four primary inputs mirror MITgcm's run-time time-step regime.  The
    ``bottom_drag_in_ab`` input preserves the eight existing LOCOMgcm cases
    that historically placed only bottom drag after AB2 while leaving other
    dissipation inside AB2.

    Tracers use the declared ``tracForcingOutAB=1`` regime independently of
    these momentum inputs: GAD terms enter AB2, forcing enters afterwards.
    """

    mom_forcing_out_ab: bool | None = None
    mom_dissip_in_ab: bool = True
    stagger_tracer_stepping: bool = False
    implicit_int_grav_wave: bool = False
    bottom_drag_in_ab: bool = True

    def __post_init__(self) -> None:
        if self.mom_forcing_out_ab is None:
            raise ValueError(
                "AssemblyPolicy.mom_forcing_out_ab must be explicitly set to "
                "true or false; pass mom_forcing_out_ab=<bool>"
            )
        invalid = {
            name: type(getattr(self, name)).__name__
            for name in self.semantic_inputs()
            if type(getattr(self, name)) is not bool
        }
        if invalid:
            raise ValueError(f"AssemblyPolicy inputs must be bool: {invalid}")

    @staticmethod
    def semantic_inputs() -> tuple[str, ...]:
        return (
            "mom_forcing_out_ab",
            "mom_dissip_in_ab",
            "stagger_tracer_stepping",
            "implicit_int_grav_wave",
            "bottom_drag_in_ab",
        )

    @staticmethod
    def placement_side_channels() -> tuple[str, ...]:
        return (
            "ab2_eligible",
            "assembly_bucket",
            "assembly_policy_bucket",
            "is_forcing",
            "is_momentum_dissipation",
        )

    @classmethod
    def regional(cls, stepper: Any) -> Self:
        """Resolve the regional policy from the stepper's shared inputs."""
        return cls(
            mom_forcing_out_ab=stepper.mom_forcing_out_ab,
            mom_dissip_in_ab=stepper.mom_dissip_in_ab,
            stagger_tracer_stepping=stepper.stagger_tracer_stepping,
            implicit_int_grav_wave=stepper.implicit_int_grav_wave,
            bottom_drag_in_ab=stepper.bottom_drag_in_ab,
        )

    @classmethod
    def cube_neutral(cls, *, bottom_drag_in_ab: bool) -> Self:
        """Return the explicit inputs that reproduce historical cube placement."""
        return cls(
            mom_forcing_out_ab=False,
            mom_dissip_in_ab=True,
            stagger_tracer_stepping=False,
            implicit_int_grav_wave=False,
            bottom_drag_in_ab=bottom_drag_in_ab,
        )

    @classmethod
    def cube(cls, stepper: Any, *, bottom_drag_implicit: bool) -> Self:
        """Resolve the cube-neutral defaults and explicit cube overrides."""
        neutral = cls.cube_neutral(
            bottom_drag_in_ab=not bottom_drag_implicit
        ).with_overrides(
            {
                "mom_dissip_in_ab": stepper.mom_dissip_in_ab,
                "implicit_int_grav_wave": stepper.implicit_int_grav_wave,
            }
        )
        configured = stepper.cube_assembly_policy
        if configured is None:
            return neutral
        if isinstance(configured, cls):
            return configured
        return neutral.with_overrides(configured)

    def with_overrides(self, overrides: Mapping[str, bool]) -> Self:
        """Apply a validated partial configuration to this policy."""
        known = set(self.semantic_inputs())
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(f"unknown AssemblyPolicy inputs: {sorted(unknown)}")
        invalid = {
            name: type(value).__name__
            for name, value in overrides.items()
            if type(value) is not bool
        }
        if invalid:
            raise ValueError(f"AssemblyPolicy inputs must be bool: {invalid}")
        return replace(self, **overrides)

    def semantics(self) -> dict[str, bool]:
        """Return the canonical checkpoint/config representation."""
        return {name: getattr(self, name) for name in self.semantic_inputs()}

    @property
    def hpg_in_ab2(self) -> bool:
        return not self.stagger_tracer_stepping and not self.implicit_int_grav_wave

    def bucket_for(self, term_class: MomentumTermClass) -> AssemblyBucket:
        """Return the integration bucket for one momentum term class."""
        if not isinstance(term_class, MomentumTermClass):
            raise ValueError(
                "momentum module declaration conflicts with AssemblyPolicy: "
                "momentum_term_class must be a MomentumTermClass, got "
                f"{term_class!r}. Declare the physical term class with "
                "MomentumTermClass and leave placement to AssemblyPolicy."
            )
        if term_class == MomentumTermClass.FORCING:
            return (
                AssemblyBucket.POST_AB2
                if self.mom_forcing_out_ab
                else AssemblyBucket.AB2
            )
        if term_class == MomentumTermClass.PRESSURE_GRADIENT:
            return AssemblyBucket.AB2 if self.hpg_in_ab2 else AssemblyBucket.POST_AB2
        if term_class == MomentumTermClass.BOTTOM_DRAG and not self.bottom_drag_in_ab:
            return AssemblyBucket.POST_AB2
        if term_class in {
            MomentumTermClass.DISSIPATION,
            MomentumTermClass.BOTTOM_DRAG,
        }:
            return (
                AssemblyBucket.AB2
                if self.mom_dissip_in_ab
                else AssemblyBucket.POST_AB2
            )
        return AssemblyBucket.AB2

    def validate_modules(
        self,
        modules: Iterable[TendencyModule],
        *,
        source: str,
    ) -> None:
        """Reject invalid declarations and per-module placement side channels."""
        for module in modules:
            name = str(getattr(module, "name", type(module).__name__))
            for field_name in self.placement_side_channels():
                if not hasattr(module, field_name):
                    continue
                declaration = getattr(module, field_name)
                term_class = getattr(module, "momentum_term_class", None)
                bucket = self.bucket_for(term_class)
                raise ValueError(
                    f"{source}: assembly consistency conflict for module {name!r}: "
                    f"module declaration {field_name}={declaration!r} is a "
                    "per-module placement side channel, while AssemblyPolicy "
                    f"resolves {term_class.value!r} to {bucket.value!r}. Remove "
                    f"{name}.{field_name} and express placement only through "
                    "AssemblyPolicy/stepper policy inputs."
                )
            self.bucket_for(getattr(module, "momentum_term_class", None))

    def bucket_of(self, module: TendencyModule) -> AssemblyBucket:
        return self.bucket_for(module.momentum_term_class)

    @staticmethod
    def tracer_bucket_of(module: TendencyModule) -> AssemblyBucket:
        """Place tracer outputs independently of the momentum term class.

        ``ab2_eligible`` is the existing tracer declaration: surface fluxes,
        restoring and OBCS sponges declare False; GAD advection/diffusion
        defaults to True. In particular, VerticalMixing's tracer output is
        inside AB even when its momentum dissipation is outside AB.
        MIT temp/salt_integrate: GAD_CALC_RHS, AB2, then APPLY_FORCING output
        for the supported tracForcingOutAB=1 regime.
        """
        return (
            AssemblyBucket.AB2
            if bool(getattr(module, "ab2_eligible", True))
            else AssemblyBucket.POST_AB2
        )

    @classmethod
    def split_tracers(
        cls, modules: Iterable[TendencyModule],
    ) -> tuple[tuple[TendencyModule, ...], tuple[TendencyModule, ...]]:
        modules = tuple(modules)
        return (
            tuple(m for m in modules if cls.tracer_bucket_of(m) == AssemblyBucket.AB2),
            tuple(m for m in modules if cls.tracer_bucket_of(m) == AssemblyBucket.POST_AB2),
        )

    def split(
        self,
        modules: Iterable[TendencyModule],
    ) -> tuple[tuple[TendencyModule, ...], tuple[TendencyModule, ...]]:
        module_tuple = tuple(modules)
        # Memoised on the policy, keyed by the modules' class names: the bucket
        # assignment is a pure function of the frozen policy and each module's
        # class-level ``momentum_term_class``, so re-deriving it every step only
        # put an ``isinstance(StrEnum)`` check into the compiled step (one
        # graph break).  The key holds values only, so a compiled step's
        # guard on it can be evaluated for any model; the stored tuple is checked
        # element-wise with ``is`` (the modules are dataclasses whose ``__eq__``
        # compares tensor fields).  The memo ends with the policy.
        memo = getattr(self, "_split_memo", None)
        if memo is None:
            memo = {}
            object.__setattr__(self, "_split_memo", memo)
        memo_key = tuple(type(module).__name__ for module in module_tuple)
        cached = memo.get(memo_key)
        if (
            cached is not None
            and len(cached[0]) == len(module_tuple)
            and all(kept is module for kept, module in zip(cached[0], module_tuple))
        ):
            return cached[1], cached[2]
        ab2_modules = tuple(
            module for module in module_tuple if self.bucket_of(module) == AssemblyBucket.AB2
        )
        post_modules = tuple(
            module
            for module in module_tuple
            if self.bucket_of(module) == AssemblyBucket.POST_AB2
        )
        memo[memo_key] = (module_tuple, ab2_modules, post_modules)
        return ab2_modules, post_modules
