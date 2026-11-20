"""CPP option headers and SIZE.h of the case's code directory."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from experiments.mitcases.code_inventory import (
    DEFER,
    KPP_CPP_TARGETS,
    NEGATIVE_SALINITY_CHECK_TARGET,
    SKIP,
    compiled_packages,
    default_header,
    evaluate_header,
)

from .core import (
    Record,
    TranslationError,
    UnknownParameter,
    _record_gap,
    _set_dotted,
)

# --------------------------------------------------------------------------
# CPP headers / SIZE.h
# --------------------------------------------------------------------------

_CPP_MACRO = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*")

def _reference_mit_root(root: Path | None = None) -> Path:
    """Resolve the configured source and its declared version lazily."""
    from defgcm.external_paths import ExternalPathError, resolve
    try:
        return resolve("mitgcm_root", root)
    except ExternalPathError as error:
        raise TranslationError(str(error)) from error


_KPP_CPP_DEFER_REASONS = {
    "KPP_SMOOTH_DVSQ": "vertical shear-square smoothing has no K5 runtime key",
    "KPP_SMOOTH_DENS": "density-field smoothing has no K5 runtime key",
    "KPP_SMOOTH_VISC": "vertical-viscosity smoothing has no K5 runtime key",
    "KPP_SMOOTH_DIFF": "vertical-diffusivity smoothing has no K5 runtime key",
    "KPP_DO_NOT_MATCH_DIFFUSIVITIES": (
        "diffusivity matching selection has no K5 runtime key"
    ),
    "KPP_DO_NOT_MATCH_DERIVATIVES": (
        "diffusivity-derivative matching selection has no K5 runtime key"
    ),
    "KPP_SMOOTH_REGULARISATION": (
        "smooth regularisation selection has no K5 runtime key"
    ),
    "KPP_SCALE_SHEARMIXING": "small-shear scaling has no K5 runtime key",
    "EXCLUDE_KPP_SHEAR_MIX": (
        "interior shear-mixing exclusion has no K5 runtime key"
    ),
    "EXCLUDE_KPP_DOUBLEDIFF": (
        "double-diffusive mixing exclusion has no K5 runtime key"
    ),
    "ALLOW_KPP_VERTICALLY_SMOOTH": (
        "vertical Richardson-number smoothing has no K5 runtime key"
    ),
}

_KPP_CPP_SKIP_REASONS = {
    "KPP_AUTODIFF_EXCESSIVE_STORE": (
        "automatic-differentiation storage option; no forward-state effect"
    ),
}

#: Macros LOCOMgcm reproduces in one state only, with no runtime key: that state and the
#: behaviour; the other state has no LOCOMgcm branch.
_KPP_CPP_FIXED_STATES = {
    "KPP_GHAT": (
        True,
        "non-local transport -diffKz*ghat(k-1)*surface forcing into the T and S vertical "
        "fluxes (kpp_transport_t.F:54-104, kpp_transport_s.F:56-123, gad_calc_rhs.F:657-690); "
        "LOCOMgcm applies it to temp and salt whenever KPP is on (defgcm/dynamics/kpp_mixing.py "
        "kpp_ghat_tendency; implicit_free_surface.py kpp_ab_names and kpp_post_names)",
    ),
}


def parse_cpp_macro_states(
    path: Path, *, predefined: frozenset[str] = frozenset()
) -> dict[str, bool]:
    """Return macro states after evaluating a header's simple conditionals.

    The pinned KPP headers use only ``ifdef`` / ``ifndef`` conditionals.  An
    unsupported ``if`` expression fails closed instead of treating directives
    in a potentially inactive branch as live.
    """
    states = {name: True for name in predefined}
    stack: list[tuple[bool, bool, bool]] = []
    active = True
    for line_number, raw in enumerate(
        path.read_text(errors="replace").splitlines(), start=1
    ):
        stripped = raw.lstrip()
        if not stripped.startswith("#"):
            continue
        body = stripped[1:].lstrip()
        if not body:
            continue
        parts = body.split(None, 1)
        directive = parts[0]
        argument = parts[1] if len(parts) == 2 else ""
        match = _CPP_MACRO.match(argument)
        macro = match.group(0) if match else ""

        if directive in {"ifdef", "ifndef"}:
            if not macro:
                raise TranslationError(
                    f"{path}:{line_number}: malformed #{directive} directive"
                )
            condition = states.get(macro, False)
            if directive == "ifndef":
                condition = not condition
            stack.append((active, condition, False))
            active = active and condition
        elif directive == "else":
            if not stack:
                raise TranslationError(f"{path}:{line_number}: unmatched #else")
            parent_active, condition, seen_else = stack[-1]
            if seen_else:
                raise TranslationError(f"{path}:{line_number}: duplicate #else")
            stack[-1] = (parent_active, condition, True)
            active = parent_active and not condition
        elif directive == "endif":
            if not stack:
                raise TranslationError(f"{path}:{line_number}: unmatched #endif")
            parent_active, _, _ = stack.pop()
            active = parent_active
        elif directive in {"if", "elif"}:
            raise TranslationError(
                f"{path}:{line_number}: unsupported #{directive} expression"
            )
        elif directive in {"define", "undef"} and active:
            if not macro:
                raise TranslationError(
                    f"{path}:{line_number}: malformed #{directive} directive"
                )
            states[macro] = directive == "define"

    if stack:
        raise TranslationError(f"{path}: unterminated CPP conditional")
    return states


def _resolve_penetrating_sw(case_dir: Path, selection: Any = None) -> tuple[bool, Path]:
    """Effective oceanic shortwave mode (ini_parms.F:653-657, bldepth).

    CPP_OPTIONS.h is a replacement header, not an overlay. An enabled mode
    remains enabled at night and when no shortwave file supplies a nonzero flux.
    """
    header = case_dir / "code" / "CPP_OPTIONS.h"
    if not header.is_file():
        header = _reference_mit_root() / "model" / "inc" / "CPP_OPTIONS.h"
    compiled = parse_cpp_macro_states(header).get("SHORTWAVE_HEATING", False)
    selected = int(compiled) if selection is None else int(selection)
    return compiled and selected >= 1, header


def _wire_negative_salinity_check(yaml_tree: dict[str, Any], case_dir: Path, record: Record) -> None:
    """CHECK_SALINITY_FOR_NEGATIVE_VALUES of the CPP_OPTIONS.h the build compiles (the case's
    replacement header, else the reference one) -> the stepper's
    check_salinity_for_negative_values, written only when the macro is defined (undefined is
    the stepper default false, model/inc/CPP_OPTIONS.h:43 #undef).  The compositions its port
    does not reach are refused by the model build (admission.validate_negative_salinity_check).
    """
    header = case_dir / "code" / "CPP_OPTIONS.h"
    if not header.is_file():
        header = _reference_mit_root() / "model" / "inc" / "CPP_OPTIONS.h"
    if not parse_cpp_macro_states(header).get("CHECK_SALINITY_FOR_NEGATIVE_VALUES", False):
        return
    _set_dotted(yaml_tree, NEGATIVE_SALINITY_CHECK_TARGET, True)
    record.derived.append(
        {
            "target": NEGATIVE_SALINITY_CHECK_TARGET,
            "value": True,
            "source": str(header),
            "rationale": (
                "CHECK_SALINITY_FOR_NEGATIVE_VALUES defined: find_rho.F:85-89 resets negative "
                "salinity to 0 in the prognostic salt before every equation of state "
                "(:1230-1232, do_oceanic_phys.F:758-765)"
            ),
        }
    )


def _gmredi_cpp_states(case_dir: Path) -> tuple[dict[str, bool], str]:
    """Macro states of the GMREDI_OPTIONS.h the build compiles, and its source.

    A case's ``code/GMREDI_OPTIONS.h`` replaces the package header, as for
    KPP_OPTIONS.h (_translate_kpp_cpp_options); the header is evaluated with
    ALLOW_GMREDI defined (its body sits in ``#ifdef ALLOW_GMREDI``, GMREDI_OPTIONS.h:18).
    """
    header = case_dir / "code" / "GMREDI_OPTIONS.h"
    source = "code/GMREDI_OPTIONS.h"
    if not header.is_file():
        header = _reference_mit_root() / "pkg" / "gmredi" / "GMREDI_OPTIONS.h"
        source = str(header)
    return parse_cpp_macro_states(header, predefined=frozenset({"ALLOW_GMREDI"})), source


def _translate_kpp_cpp_options(
    case_dir: Path, *, record: Record, yaml_tree: dict[str, Any]
) -> None:
    """Resolve the KPP macros of the header the build compiles.

    A case's ``code/KPP_OPTIONS.h`` replaces the package header (genmake2 puts
    the ``-mods`` directory before the package directories; code_inventory's
    module conventions), so a macro the case header leaves out keeps the state
    of the headers included before it (``PACKAGES_CONFIG.h``,
    ``CPP_OPTIONS.h``), where no pinned case defines a KPP macro: undefined.
    """
    header = case_dir / "code" / "KPP_OPTIONS.h"
    source = "code/KPP_OPTIONS.h"
    if not header.is_file():
        header = _reference_mit_root() / "pkg" / "kpp" / "KPP_OPTIONS.h"
        source = str(header)
        if not header.is_file():
            raise TranslationError(f"missing KPP package header: {header}")
    states = parse_cpp_macro_states(header, predefined=frozenset({"ALLOW_KPP"}))
    known = (
        set(KPP_CPP_TARGETS)
        | set(_KPP_CPP_DEFER_REASONS)
        | set(_KPP_CPP_SKIP_REASONS)
        | set(_KPP_CPP_FIXED_STATES)
        | {"ALLOW_KPP", "KPP_OPTIONS_H"}
    )
    unknown = sorted(states.keys() - known)
    if unknown:
        raise UnknownParameter(unknown[0], source)

    def resolve(macro: str) -> tuple[bool, str]:
        return states.get(macro, False), source

    for macro, target in KPP_CPP_TARGETS.items():
        value, source = resolve(macro)
        _set_dotted(yaml_tree, target, value)
        record.derived.append(
            {
                "target": target,
                "value": value,
                "source": source,
                "rationale": (
                    f"{macro} compile-time state after evaluating nested CPP "
                    "guards with ALLOW_KPP defined, in code/KPP_OPTIONS.h when "
                    "the case has one (it replaces the package header), "
                    "otherwise in the package header"
                ),
            }
        )
    for macro, reason in _KPP_CPP_DEFER_REASONS.items():
        value, source = resolve(macro)
        record.add(
            source=source,
            parameter=f"CPP.{macro}",
            value=value,
            disposition=DEFER if value else SKIP,
            reason=(
                reason
                if value
                else f"{reason}; macro is undefined, so the branch is inactive"
            ),
        )
    for macro, reason in _KPP_CPP_SKIP_REASONS.items():
        value, source = resolve(macro)
        record.add(
            source=source,
            parameter=f"CPP.{macro}",
            value=value,
            disposition=SKIP,
            reason=reason,
        )
    for macro, (state, behaviour) in _KPP_CPP_FIXED_STATES.items():
        value, source = resolve(macro)
        row = {"source": source, "parameter": f"CPP.{macro}", "value": value}
        if value == state:
            record.add(**row, disposition=SKIP,
                       reason=f"{macro} {'defined' if state else 'undefined'}: {behaviour}")
        else:
            _record_gap(record, row, f"{macro} {'defined' if value else 'undefined'}: LOCOMgcm has "
                        f"no branch for this state; it always takes the other one, {behaviour}")


_SIZE_KEYS = ("sNx", "sNy", "OLx", "OLy", "nSx", "nSy", "nPx", "nPy", "Nr")


def parse_size_h(path: Path) -> dict[str, int]:
    """Extract the SIZE.h PARAMETER block (``sNx`` ... ``Nr``)."""
    values: dict[str, int] = {}
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if line.startswith(("C", "c", "#", "!")):
            continue
        body = line[1:] if line.startswith("&") else line
        for chunk in body.split(","):
            if "=" not in chunk:
                continue
            name, _, rhs = chunk.partition("=")
            name = name.strip()
            if name not in _SIZE_KEYS:
                continue
            rhs = rhs.strip().rstrip(")").strip()
            try:
                values[name] = int(rhs)
            except ValueError:
                continue
    missing = [k for k in _SIZE_KEYS if k not in values]
    if missing:
        raise TranslationError(f"{path}: SIZE.h is missing {missing}")
    return {
        "nx": values["sNx"] * values["nSx"] * values["nPx"],
        "ny": values["sNy"] * values["nSy"] * values["nPy"],
        "nz": values["Nr"],
        **values,
    }




def _exf_macros(case_dir: Path) -> frozenset[str]:
    """Macros defined after the build's EXF_OPTIONS.h, evaluated as code_inventory does.

    code/EXF_OPTIONS.h replaces pkg/exf/EXF_OPTIONS.h; it sees the compiled
    packages' ALLOW_<PKG> and the state after CPP_OPTIONS.h
    (code_inventory.inventory_case).
    """
    mit_root = _reference_mit_root()
    code = case_dir / "code"
    state = {f"ALLOW_{name.upper()}": "" for name in compiled_packages(code, mit_root).compiled}
    for name in ("CPP_OPTIONS.h", "EXF_OPTIONS.h"):
        header = code / name if (code / name).is_file() else default_header(name, mit_root)
        for macro, (value, _) in evaluate_header(header, state).items():
            if value is None:
                state.pop(macro, None)
            else:
                state[macro] = value
    return frozenset(state)
