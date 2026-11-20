"""Rule-table primitives (Rule, Ctx, the map/skip/defer/gap helpers) and the grid family."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from experiments.mitcases.code_inventory import DEFER, MAP, SKIP, UNSUPPORTED

from .core import (
    ANALYTIC_OBCS,
    Deferred,
    Gap,
    Skipped,
    TranslationError,
    _ANALYTIC_OBCS_MISMATCH,
    _ASSEMBLY,
    _Assembly,
    _FORCING_SOURCE_NAME,
    _IC_SOURCE_NAME,
    _OBCS_CORE_FILE_KEYS,
    _OBCS_SIDE_NAMES,
    _RECORD_METADATA,
)

# --------------------------------------------------------------------------
# rule table primitives
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Ctx:
    """Everything a conversion function may look at besides its own value.

    Deliberately carries no case identity: a conversion function must not be
    able to branch on which case it is translating.  ``case_dir`` exists only
    so structural source overrides can be inspected by content.  ``verbatim``
    is the translation mode (:func:`translate_case`), for the rows whose key a
    :data:`CAMPAIGN_DISABLED` ruling disables outside it.
    """

    namelists: dict[str, dict[str, dict[str, Any]]]
    size_h: dict[str, int]
    grid_family: str
    case_dir: Path
    verbatim: bool = False

    def data(self, group: str, key: str, default: Any = None) -> Any:
        return self.namelists.get("data", {}).get(group, {}).get(key, default)

    @property
    def nx(self) -> int:
        return self.size_h["nx"]

    @property
    def ny(self) -> int:
        return self.size_h["ny"]

    @property
    def nz(self) -> int:
        return self.size_h["nz"]


Convert = Callable[[Any, Ctx], dict[str, Any]]


@dataclass(frozen=True)
class Rule:
    disposition: str
    target: str | None = None
    convert: str = "identity"
    reason: str | None = None
    fn: Convert | None = None


def _map(target: str, convert: str = "identity", fn: Convert | None = None) -> Rule:
    return Rule(MAP, target=target, convert=convert, fn=fn)


def _map_both(*targets: str, convert: str = "identity") -> Rule:
    """Write one namelist value to several LOCOMgcm targets.

    Used where a declaration site and the runtime consumer are different paths;
    both have to be written or the declaration is decorative.
    """

    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        return {target: value for target in targets}

    return Rule(MAP, target=None, convert=convert, fn=fn)


def _skip(reason: str) -> Rule:
    return Rule(SKIP, reason=reason)


def _defer(reason: str) -> Rule:
    return Rule(DEFER, reason=reason)


def _gap(reason: str) -> Rule:
    return Rule(UNSUPPORTED, reason=reason)


def _zero_or_gap(reason: str) -> Rule:
    """Numeric knob LOCOMgcm lacks: harmless at zero, a gap otherwise."""

    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        if _is_zero(value):
            raise Skipped(f"{reason} (value is zero: no-op)")
        raise Gap(reason)

    return Rule(MAP, target=None, convert="zero -> no-op, else capability gap", fn=fn)


def _false_or_gap(reason: str) -> Rule:
    """Logical switch LOCOMgcm lacks: harmless when .FALSE., a gap when .TRUE."""

    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        if value is False:
            raise Skipped(f"{reason} (switch is .FALSE.: no-op)")
        raise Gap(reason)

    return Rule(MAP, target=None, convert=".FALSE. -> no-op, else capability gap", fn=fn)


def _true_or_gap(reason: str) -> Rule:
    """Logical switch matching LOCOMgcm only when MITgcm leaves it enabled."""

    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        del ctx
        if value is True:
            raise Skipped(f"{reason} (switch is .TRUE.: matches LOCOMgcm)")
        raise Gap(reason)

    return Rule(MAP, target=None, convert=".TRUE. -> match, else capability gap", fn=fn)


def _unread_binary_input(role: str, basis: str) -> Rule:
    """PARM05 file name no LOCOMgcm source reads: a capability gap unless blank.

    A blank name is MITgcm's "no file" (ini_forcing.F:70-90 and
    external_fields_load.F guard each read with ``.NE. ' '``).  A named file
    MITgcm reads (``basis``) would be dropped by a defer, so it is refused until a
    reader takes it; its path and dtype stay under mapping.binary_inputs.
    """

    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        del ctx
        if value is None or not str(value).strip():
            raise Skipped(f"{role} file name is blank; MITgcm reads no file")
        raise Gap(f"no LOCOMgcm reader of the {role} file: {basis}")

    return Rule(MAP, target=None, convert="blank -> skip, else capability gap", fn=fn)


def _ic_binary_input(kwarg: str, role: str) -> Rule:
    """PARM05 file name the initial-condition source reads.

    The name lands on the stage1a ``state`` lane's source block; the dtype rides
    along beside it as ``read_binary_prec`` (see
    :func:`_wire_initial_condition_source`), so the reader never has to guess
    either one.
    """

    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        return {f"sources.{_IC_SOURCE_NAME}.{kwarg}": str(value)}

    return Rule(
        MAP,
        target=f"sources.{_IC_SOURCE_NAME}.{kwarg}",
        convert=f"MIT {role} file name, read at readBinaryPrec (big-endian)",
        fn=fn,
    )


def _forcing_binary_input(key: str, role: str) -> Rule:
    """PARM05 core surface-forcing file name.

    The name is staged rather than written straight into a source block: the
    consumer only exists when MITgcm steps the corresponding momentum/tracer,
    and :func:`_build_model_assembly` is where that is known.  A blank name is
    MITgcm's own "no file" spelling -- ``model/src/ini_forcing.F:70-90`` guards
    every read with ``.NE. ' '`` -- so it stays a skip, not an ingest.
    """

    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        if value is None or not str(value).strip():
            raise Skipped(
                f"{role} file name is blank; MITgcm reads no file "
                "(ini_forcing.F:70-90)"
            )
        return {f"{_ASSEMBLY}.{key}": str(value)}

    return Rule(
        MAP,
        target=f"sources.{_FORCING_SOURCE_NAME}.{key} <- {key}",
        convert=f"MIT {role} file name, read at readBinaryPrec (big-endian)",
        fn=fn,
    )


def _stage(key: str, convert: str) -> Rule:
    """Map a value into the model-assembly staging namespace.

    The staging namespace is not part of the emitted yaml: it is consumed by
    :func:`_build_model_assembly`, which turns it into ``model.dynamics``
    entries and records each one under ``mapping.derived``.
    """

    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        return {f"{_ASSEMBLY}.{key}": value}

    return Rule(MAP, target=f"model.dynamics <- {key}", convert=convert, fn=fn)


def _use_obcs_rule() -> Rule:
    """Map authorised analytic overrides or the prescribed-binary subset."""

    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        del value
        parms = ctx.namelists.get("data.obcs", {}).get("OBCS_PARM01", {})
        source = ctx.case_dir / "code" / "obcs_calc.F"
        if source.is_file():
            spec = _authorised_obcs(source)
            if spec is None:
                raise Gap(_ANALYTIC_OBCS_MISMATCH)
            return {
                f"{_ASSEMBLY}.use_obcs": True,
                f"{_ASSEMBLY}.analytic_obcs": dict(spec),
                _RECORD_METADATA: {
                    "authorized_source": spec["authorized_source"],
                    "ruling": spec["ruling"],
                    "overrides": spec["overrides"],
                },
            }

        # No case obcs_calc.F: MITgcm builds the package's own OBCS_CALC, which
        # sets zero velocity, tRef and sRef on every open side
        # (pkg/obcs/obcs_calc.F:120-137,170-187,204-221,238-255), lets
        # Orlanski replace a side that selects it (:97,147,197,231), then
        # overwrites with the prescribed files (useOBCSprescribe, :408-414)
        # and Stevens (:442-448).
        has_core_file = any(parms.get(key) for key in _OBCS_CORE_FILE_KEYS)
        orlanski = any(parms.get(f"useorlanski{name}") is True for name in _OBCS_SIDE_NAMES.values())
        if (not bool(parms.get("useobcsprescribe", False)) or not has_core_file) and not orlanski:
            raise Deferred(
                "open-boundary values from pkg/obcs/obcs_calc.F (zero velocity, "
                "tRef, sRef), not from core OB[NSEW][uvwts]File binaries under "
                "useOBCSprescribe: the translator wires the prescribed lane and "
                "Orlanski sides (useOrlanski* rows)"
            )
        return {f"{_ASSEMBLY}.use_obcs": True}

    return Rule(
        MAP,
        target="sources.mit_obcs + model.boundary_condition",
        convert=(
            "code/obcs_calc.F equal to an authorised copy -> registered boundary "
            "plugin; otherwise useOBCSprescribe with core OB files or a useOrlanski* "
            "side -> mit_obcs_binary lane with prescribed_3d value_schemes"
        ),
        fn=fn,
    )


def _authorised_obcs(source: Path) -> dict[str, Any] | None:
    """Return the :data:`ANALYTIC_OBCS` row whose copy equals ``source`` bytewise."""
    payload = source.read_bytes()
    repo = Path(__file__).resolve().parents[3]
    for spec in ANALYTIC_OBCS:
        if (repo / spec["authorized_source"]).read_bytes() == payload:
            return spec
    return None


def _is_zero(value: Any) -> bool:
    if isinstance(value, bool):
        return value is False
    if isinstance(value, (int, float)):
        return float(value) == 0.0
    if isinstance(value, list):
        return all(_is_zero(v) for v in value)
    return False


def _as_list(value: Any) -> list[float]:
    return [float(v) for v in (value if isinstance(value, list) else [value])]


def _uniform(values: list[float]) -> float | None:
    """Return the common value of a spacing array, or None when it varies."""
    if not values:
        return None
    first = values[0]
    return first if all(abs(v - first) <= 1e-12 * max(1.0, abs(first)) for v in values) else None


# --------------------------------------------------------------------------
# grid family (the ONLY structural branch; decided by using*Grid switches)
# --------------------------------------------------------------------------

_GRID_SWITCHES = {
    "usingcartesiangrid": "cartesian",
    "usingsphericalpolargrid": "spherical_polar",
    "usingcurvilineargrid": "curvilinear",
    "usingcylindricalgrid": "cylindrical",
}

# MITgcm default when no using*Grid switch is set: Cartesian.
# Source: references/MITgcm/model/src/ini_parms.F:1337-1348.
_GRID_FAMILY_DEFAULT = "cartesian"

#: MITgcm grid family -> LOCOMgcm's ``case.extras.grid_mode`` selector.
#:
#: Both grid builders read this key with ``"cartesian"`` as the *default* when
#: it is absent (``experiments/cases/_runtime.py:648``,
#: ``defgcm/io/runner/model_input.py:785``), and both branch on
#: ``!= "spherical_polar"``.  A spherical case that left the key out would
#: therefore be built on a Cartesian grid without a word -- the exact
#: silent fallback this declaration prevents. The family is a *family-level*
#: derivation from the ``using*Grid`` switches (:func:`grid_family`), never a
#: per-case lookup, and :func:`_assert_grid_mode_declared` refuses to emit a
#: case yaml that lacks it.
_TO_GRID_MODE = {
    "cartesian": "cartesian",
    "spherical_polar": "spherical_polar",
}

_GRID_FAMILY_GAPS = {
    "curvilinear": (
        "curvilinear / cubed-sphere geometry: TO needs an explicit cube_* "
        "declaration block (grid.kind=cubed_sphere), outside this translator"
    ),
    "cylindrical": "cylindrical geometry has no TO grid family",
}


def grid_family(parm04: dict[str, Any]) -> str:
    """Decide the grid family from the namelist switches alone."""
    on = sorted({fam for key, fam in _GRID_SWITCHES.items() if parm04.get(key) is True})
    if len(on) > 1:
        raise TranslationError(f"conflicting grid switches: {on}")
    return on[0] if on else _GRID_FAMILY_DEFAULT


def _assert_grid_mode_declared(
    yaml_tree: dict[str, Any], family: str, case: str
) -> None:
    """Refuse to emit a case yaml whose grid mode is not spelled out.

    Unreachable by construction -- a family outside :data:`_TO_GRID_MODE` has a
    ``using*Grid`` switch whose rule already raised :class:`Gap`, so the case is
    UNSUPPORTED and no yaml is written.  It is kept as a ratchet: the day
    someone adds a family to :data:`_GRID_FAMILY_GAPS` without adding it here,
    the failure is a loud refusal instead of a Cartesian grid built for a
    spherical case.
    """
    expected = _TO_GRID_MODE.get(family)
    actual = ((yaml_tree.get("case") or {}).get("extras") or {}).get("grid_mode")
    if expected is None or actual != expected:
        raise TranslationError(
            f"{case}: grid family {family!r} would be emitted with "
            f"case.extras.grid_mode={actual!r} (expected {expected!r}); the "
            "grid builders default the key to 'cartesian', so emitting it "
            "unset would silently build the wrong grid"
        )


def _other_axis_spacing(ctx: Ctx, axis: str) -> float | None:
    """The uniform spacing of the *other* horizontal axis, if it has one."""
    other = "y" if axis == "x" else "x"
    spacing = ctx.data("PARM04", f"del{other}")
    if spacing is None:
        spacing = ctx.data("PARM04", f"d{other}spacing")
    return None if spacing is None else _uniform(_as_list(spacing))


def _horizontal_spacing(value: Any, ctx: Ctx, axis: str) -> dict[str, Any]:
    """delX/delY/dXspacing/dYspacing -> resolution (sphere) or extras (cartesian)."""
    values = _as_list(value)
    has_spacing_file = any(
        bool(ctx.data("PARM04", f"del{candidate}file"))
        for candidate in ("x", "y")
    )
    if ctx.grid_family == "cartesian" and has_spacing_file:
        count = ctx.nx if axis == "x" else ctx.ny
        if len(values) == 1:
            values = values * count
        if len(values) != count:
            raise Gap(
                f"inline del{axis.upper()} spacing must contain 1 or {count} "
                f"values when file spacing is active; found {len(values)}"
            )
        if not all(math.isfinite(item) for item in values):
            raise Gap(f"inline del{axis.upper()} spacing values must all be finite")
        if not all(item > 0.0 for item in values):
            raise Gap(
                f"inline del{axis.upper()} spacing values must all be strictly positive"
            )
        return {
            f"case.extras.mitcase.mesh_spacing.del_{axis}": values
        }

    common = _uniform(values)
    if common is None:
        raise Gap(f"non-uniform {axis} spacing (TO carries a single resolution)")
    if ctx.grid_family == "spherical_polar":
        # A spherical case carries ONE grid.resolution_deg, and both consumers
        # feed it to delX_deg *and* delY_deg (experiments/cases/_runtime.py:678-680,
        # defgcm/io/runner/model_input.py:812-813).  delX != delY would let
        # whichever axis is translated last silently overwrite the other.
        other = _other_axis_spacing(ctx, axis)
        if other is not None and other != common:
            raise Gap(
                "anisotropic spherical spacing (delX != delY): TO carries a "
                "single grid.resolution_deg for both axes"
            )
        return {"grid.resolution_deg": common}
    if ctx.grid_family == "cartesian":
        n = ctx.nx if axis == "x" else ctx.ny
        return {f"case.extras.d{axis}": common, f"case.extras.l{axis}": common * n}
    raise Gap(_GRID_FAMILY_GAPS[ctx.grid_family])


def _spacing_file(value: Any, ctx: Ctx, axis: str) -> dict[str, Any]:
    """Map a Cartesian delXFile/delYFile after fail-closed path validation."""
    filename = str(value).strip()
    if not filename:
        raise Skipped(f"del{axis.upper()}File is blank; MITgcm reads no file")
    if ctx.grid_family == "spherical_polar":
        raise Deferred(
            f"spherical-polar del{axis.upper()}File requires angular vector metrics"
        )
    if ctx.grid_family != "cartesian":
        raise Gap(_GRID_FAMILY_GAPS[ctx.grid_family])

    raw_prec = ctx.data("PARM01", "readbinaryprec", 32)
    try:
        prec = int(raw_prec or 32)
    except (TypeError, ValueError):
        raise Gap(
            f"del{axis.upper()}File readBinaryPrec must be 32 or 64, got {raw_prec!r}"
        ) from None
    if prec not in (32, 64):
        raise Gap(
            f"del{axis.upper()}File readBinaryPrec must be 32 or 64, got {raw_prec!r}"
        )

    raw_path = Path(filename)
    if raw_path.is_absolute():
        raise Gap(
            f"del{axis.upper()}File must be relative to the verification input "
            f"directory, got absolute path {filename!r}"
        )
    input_dir = (ctx.case_dir / "input").resolve()
    resolved = (input_dir / raw_path).resolve()
    try:
        resolved.relative_to(input_dir)
    except ValueError:
        raise Gap(
            f"del{axis.upper()}File escapes the verification input directory: "
            f"{filename!r}"
        ) from None
    if not resolved.is_file():
        raise Gap(f"del{axis.upper()}File not found: {resolved}")

    other = "y" if axis == "x" else "x"
    other_file = ctx.data("PARM04", f"del{other}file")
    other_inline = ctx.data("PARM04", f"del{other}")
    if other_inline is None:
        other_inline = ctx.data("PARM04", f"d{other}spacing")
    if not other_file and other_inline is None:
        raise Gap(
            f"del{axis.upper()}File requires an explicit del{other.upper()}File, "
            f"del{other.upper()}, or d{other.upper()}spacing; "
            "resolution_deg fallback is disabled"
        )

    return {
        f"case.extras.mitcase.mesh_spacing.del_{axis}_file": filename
    }


def _origin(value: Any, ctx: Ctx, axis: str) -> dict[str, Any]:
    """xgOrigin/ygOrigin -> domain centre (sphere) or a recorded offset (cartesian)."""
    origin = float(value)
    if ctx.grid_family == "spherical_polar":
        spacing = ctx.data("PARM04", "delx" if axis == "x" else "dely")
        if spacing is None:
            spacing = ctx.data("PARM04", "dxspacing" if axis == "x" else "dyspacing")
        if spacing is None:
            raise Deferred(f"{axis}gOrigin needs a horizontal spacing to centre on")
        values = _as_list(spacing)
        common = _uniform(values)
        if common is None:
            raise Gap(f"non-uniform {axis} spacing (TO carries a single resolution)")
        n = ctx.nx if axis == "x" else ctx.ny
        centre = origin + 0.5 * common * n
        key = "grid.center_lon" if axis == "x" else "grid.center_lat"
        return {key: centre}
    if ctx.grid_family == "cartesian":
        return {f"case.extras.mitcase.{axis}g_origin_m": origin}
    raise Gap(_GRID_FAMILY_GAPS[ctx.grid_family])


def _grid_switch(value: Any, ctx: Ctx) -> dict[str, Any]:
    if value is not True:
        raise Skipped("grid switch is .FALSE.")
    family = ctx.grid_family
    if family in _GRID_FAMILY_GAPS:
        raise Gap(_GRID_FAMILY_GAPS[family])
    if family == "cartesian":
        return {
            "case.extras.mitcase.grid_family": family,
            "mit_comparison.cartesian": True,
        }
    return {"case.extras.mitcase.grid_family": family}




def _mit_cori_map(ctx: Ctx) -> int:
    """MITgcm's effective selectCoriMap: its own value, else 1 (beta plane) on a
    Cartesian or cylindrical grid and 2 (spherical) otherwise (set_parms.F:73-81)."""
    select_cori_map = int(ctx.data("PARM01", "selectcorimap", -1))
    if select_cori_map == -1:
        select_cori_map = (
            1 if ctx.grid_family in ("cartesian", "cylindrical") else 2
        )
    return select_cori_map


def _coriolis_profile(asm: _Assembly, ctx: Ctx, *, corner: bool) -> Any:
    """The Coriolis map MITgcm would build for this grid family.

    An unset ``selectCoriMap`` resolves to 1 (beta plane) on Cartesian /
    cylindrical grids and 2 (spherical) otherwise
    (``model/src/set_parms.F:73-81``).  Explicit maps use the same branches;
    combinations whose coordinate units cannot be represented fail loudly.

    * beta plane -- ``fCori = f0 + beta*yC`` with ``facGrid = 1`` in metres
      (``model/src/ini_cori.F:77-78``); ``yC(j) = ygOrigin + (j+0.5)*delY`` at
      centres, ``yG(j) = ygOrigin + j*delY`` at vorticity corners.
    * spherical -- ``fCori = 2 omega sin(yC)`` (``ini_cori.F:85-95``), which is
      exactly the runtime's ``$coriolis_profile_c`` / ``$coriolis_profile_g``
      sentinel over ``grid.center_lat`` / ``grid.resolution_deg``, resolved with
      omega from ``physics.rotation_period`` (runtime_kwargs).

    A profile is emitted even when ``beta`` is zero: the momentum bundles carry
    ``coriolis_profile_*`` only, and FCoriolis' scalar branch is a different
    (FP-floor-distinct) discretisation from the profile branch that was verified
    bitwise against MITgcm.

    Returns a tuple, a sentinel string, or ``None`` when the beta-plane profile
    cannot be built (no uniform ``delY``), which the caller reports as a defer.
    """
    select_cori_map = _mit_cori_map(ctx)
    if select_cori_map == 0:
        return (asm.number("f0"),) * (ctx.ny + 1 if corner else ctx.ny)
    if select_cori_map == 2:
        if ctx.grid_family in ("cartesian", "cylindrical"):
            raise TranslationError(
                "selectCoriMap=2 cannot use metre-valued Cartesian yC as degrees"
            )
        return "$coriolis_profile_g" if corner else "$coriolis_profile_c"
    if select_cori_map != 1:
        raise TranslationError(
            f"selectCoriMap={select_cori_map} requires an unsupported external map"
        )
    if ctx.grid_family not in ("cartesian", "cylindrical"):
        raise TranslationError(
            "selectCoriMap=1 on a spherical grid requires beta*yC*deg2rad*rSphere, "
            "which the current translation schema cannot represent"
        )
    f0 = asm.number("f0")
    beta = asm.number("beta")
    n = ctx.ny + 1 if corner else ctx.ny
    if beta == 0.0:
        return (f0,) * n
    dy = asm.yaml("case.extras.dy")
    if dy is None:
        return None
    origin = float(asm.yaml("case.extras.mitcase.yg_origin_m", 0.0) or 0.0)
    offsets = (float(j) for j in range(n)) if corner else (j + 0.5 for j in range(ctx.ny))
    return tuple(f0 + beta * (origin + off * float(dy)) for off in offsets)
