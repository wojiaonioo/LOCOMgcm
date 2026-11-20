"""Public entry points: translate_case, Result, the verification root and pin."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml
from experiments.mitcases.g2_compare import ARTIFACT_SCHEMA, COMPARATOR_SCHEMA
from experiments.mitcases.code_inventory import DEFER, MAP, UNSUPPORTED

from .core import (
    InvalidParameter,
    REFERENCE_UPSTREAM,
    Record,
    STATUS_DEGRADED,
    STATUS_ERROR,
    STATUS_NO_INPUT,
    STATUS_OK,
    STATUS_UNSUPPORTED,
    TranslationError,
    UnknownParameter,
    _CLIMATOLOGY_SOURCE_NAME,
    _CLIMATOLOGY_SOURCE_TYPE,
    _FORCING_SOURCE_TYPE,
    _dtype_of,
    _lookup_key,
    _record_gap,
)
from .rules import (
    Ctx,
    _assert_grid_mode_declared,
    grid_family,
)
from .namelist import (
    _mit_namelist,
)
from .cpp import (
    _reference_mit_root,
    _translate_kpp_cpp_options,
    _wire_negative_salinity_check,
    parse_size_h,
)
from .convert import (
    _apply_mit_3d_coriolis_controls,
    _apply_mit_cg2d_defaults,
    _apply_mit_cg3d_defaults,
    _apply_mit_hfac_min_default,
    _apply_mit_time_defaults,
    _refuse_unclocked_inputs,
)
from .data_rules import (
    DATA_RULES,
    EEDATA_RULES,
)
from .vmix import (
    _VERTICAL_MIXING_PACKAGES,
)
from .gmredi import (
    GMREDI_RULES,
)
from .obcs import (
    OBCS_RULES,
)
from .exf import (
    EXF_RULES,
)
from .package_rules import (
    CAL_RULES,
    KPP_RULES,
)
from .engine import (
    _account_auxiliary_namelists,
    _account_code_directory,
    _translate_namelist,
    _translate_packages,
)
from .assembly import (
    _DERIVED_DECISIONS,
    _base_yaml,
    _build_model_assembly,
    _try_construct,
    _wire_initial_condition_source,
)

# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------


@dataclass
class Result:
    case: str
    status: str
    yaml_path: Path | None
    registry_path: Path | None
    counts: dict[str, int]
    gaps: list[dict[str, str]]
    error: str | None = None
    #: ``local_mit_rerun`` when an explicit case override disabled a switch, i.e. the
    #: upstream ``results/output.txt`` no longer describes this configuration.
    reference_source: str = REFERENCE_UPSTREAM

    @property
    def headline(self) -> str:
        if self.status == STATUS_ERROR:
            return self.error or "error"
        if self.gaps:
            return self.gaps[0]["gap"]
        return ""


def default_verification_root() -> Path:
    """Return the verification directory of the configured pinned source."""
    verification = _reference_mit_root() / "verification"
    if not verification.is_dir():
        raise TranslationError(f"missing MITgcm verification directory: {verification}")
    return verification


def upstream_pin(root: Path) -> str | None:
    source = root.parent
    if not (source / "PIN").is_file() and not (source / ".git").exists():
        return None  # A direct, self-contained synthetic case has no upstream identity.
    _reference_mit_root(source)
    from defgcm.external_paths import resolve
    return resolve("mitgcm_pin")


def translate_case(
    case_dir: Path, out_dir: Path | None = None, *, verbatim: bool = False
) -> Result:
    """Translate one MITgcm verification case directory.

    Args:
      case_dir: ``<verification>/<case>`` -- must hold ``input/`` and ``code/``.
      out_dir: where ``<case>.yaml`` and ``<case>.registry.yaml`` are written.
        ``None`` runs the translation without writing anything.
      verbatim: the case is to run as MITgcm ran it (``bin/casegen fixed``):
        every deferred row and every :data:`CAMPAIGN_DISABLED` switch is
        ``unsupported`` (:class:`Record`), so the yaml is written only when
        each key is mapped or skipped.

    Returns:
      A :class:`Result`; ``status`` is one of the module-level ``STATUS_*``.
    """
    case = case_dir.name
    input_dir = case_dir / "input"
    record = Record(verbatim=verbatim)

    def _fail(status: str, message: str) -> Result:
        return Result(case, status, None, None, record.counts(), record.gaps, message)

    if not (input_dir / "data").is_file():
        return _fail(STATUS_NO_INPUT, f"{case}: no input/data namelist")

    size_path = case_dir / "code" / "SIZE.h"
    if not size_path.is_file():
        return _fail(STATUS_ERROR, f"{case}: no code/SIZE.h")

    try:
        size_h = parse_size_h(size_path)
        namelists = {"data": _mit_namelist(input_dir / "data")}
        if (input_dir / "data.obcs").is_file():
            namelists["data.obcs"] = _mit_namelist(input_dir / "data.obcs")
        if (input_dir / "data.kpp").is_file():
            namelists["data.kpp"] = _mit_namelist(input_dir / "data.kpp")
        if (input_dir / "data.exf").is_file():
            namelists["data.exf"] = _mit_namelist(input_dir / "data.exf")
        if (input_dir / "data.gmredi").is_file():
            namelists["data.gmredi"] = _mit_namelist(input_dir / "data.gmredi")
        for _scheme, filename, _rules in _VERTICAL_MIXING_PACKAGES.values():
            if (input_dir / filename).is_file():
                namelists[filename] = _mit_namelist(input_dir / filename)
        if (input_dir / "data.pkg").is_file():
            namelists["data.pkg"] = _mit_namelist(input_dir / "data.pkg")
        family = grid_family(namelists["data"].get("PARM04", {}))
    except (TranslationError, ValueError) as exc:
        return _fail(STATUS_ERROR, str(exc))

    ctx = Ctx(
        namelists=namelists,
        size_h=size_h,
        grid_family=family,
        case_dir=case_dir,
        verbatim=verbatim,
    )
    yaml_tree = _base_yaml(case, size_h, family)
    record.derived.extend(_DERIVED_DECISIONS)

    try:
        _translate_namelist(
            input_dir / "data",
            DATA_RULES,
            ctx,
            label="input/data",
            record=record,
            yaml_tree=yaml_tree,
        )
        enabled: set[str] = set()
        if (input_dir / "data.pkg").is_file():
            enabled = _translate_packages(
                input_dir / "data.pkg", ctx, record=record, yaml_tree=yaml_tree
            )
        if (input_dir / "eedata").is_file():
            _translate_namelist(
                input_dir / "eedata",
                EEDATA_RULES,
                ctx,
                label="input/eedata",
                record=record,
                yaml_tree=yaml_tree,
            )
        if "useobcs" in enabled and (input_dir / "data.obcs").is_file():
            _translate_namelist(
                input_dir / "data.obcs",
                OBCS_RULES,
                ctx,
                label="input/data.obcs",
                record=record,
                yaml_tree=yaml_tree,
            )
            if bool(
                ctx.namelists["data.obcs"]
                .get("OBCS_PARM01", {})
                .get("useobcssponge", False)
            ):
                _record_gap(
                    record,
                    {
                        "source": "input/data.obcs:OBCS_PARM01.useOBCSsponge",
                        "parameter": "<derived:useOBCSsponge capability>",
                        "value": True,
                    },
                    "OBCS sponge is not implemented by the binary ingest",
                )
        if "usekpp" in enabled and (input_dir / "data.kpp").is_file():
            _translate_namelist(
                input_dir / "data.kpp",
                KPP_RULES,
                ctx,
                label="input/data.kpp",
                record=record,
                yaml_tree=yaml_tree,
            )
        if "usekpp" in enabled:
            _translate_kpp_cpp_options(
                case_dir, record=record, yaml_tree=yaml_tree
            )
        if "useexf" in enabled and (input_dir / "data.exf").is_file():
            _translate_namelist(
                input_dir / "data.exf",
                EXF_RULES,
                ctx,
                label="input/data.exf",
                record=record,
                yaml_tree=yaml_tree,
            )
        if "usecal" in enabled and (input_dir / "data.cal").is_file():
            _translate_namelist(
                input_dir / "data.cal",
                CAL_RULES,
                ctx,
                label="input/data.cal",
                record=record,
                yaml_tree=yaml_tree,
            )
        # outside the verbatim mode the case-scope rule disables the package, and MITgcm
        # never reads data.gmredi (_account_auxiliary_namelists records it disabled)
        if "usegmredi" in enabled and verbatim and (input_dir / "data.gmredi").is_file():
            _translate_namelist(
                input_dir / "data.gmredi",
                GMREDI_RULES,
                ctx,
                label="input/data.gmredi",
                record=record,
                yaml_tree=yaml_tree,
            )
        for flag, (_scheme, filename, rules) in _VERTICAL_MIXING_PACKAGES.items():
            if flag in enabled and (input_dir / filename).is_file():
                _translate_namelist(
                    input_dir / filename,
                    rules,
                    ctx,
                    label=f"input/{filename}",
                    record=record,
                    yaml_tree=yaml_tree,
                )
        _account_auxiliary_namelists(input_dir, enabled, record=record)
        _account_code_directory(case_dir, ctx, record=record)
    except (UnknownParameter, InvalidParameter):
        # A missing rule is a defect in this module's table, not a property of
        # the case, and an uninterpretable physical selector cannot be priced as
        # a capability status: both must reach the caller loud.
        raise
    except (TranslationError, ValueError) as exc:
        return _fail(STATUS_ERROR, str(exc))

    try:
        _collect_binary_inputs(ctx, record)
        _apply_mit_time_defaults(yaml_tree, ctx, record)
        _apply_mit_cg2d_defaults(yaml_tree, ctx, record)
        _apply_mit_hfac_min_default(yaml_tree, ctx, record)
        _apply_mit_cg3d_defaults(yaml_tree, ctx, record)
        _apply_mit_3d_coriolis_controls(yaml_tree, ctx, record)
        _wire_negative_salinity_check(yaml_tree, case_dir, record)
        _build_model_assembly(yaml_tree, case_dir, ctx, record)
        _refuse_unclocked_inputs(yaml_tree, ctx, record)
        _try_construct(yaml_tree, record)
        _wire_initial_condition_source(yaml_tree, case_dir, ctx, record)
    except (TranslationError, ValueError) as exc:
        return _fail(STATUS_ERROR, str(exc))

    counts = record.counts()
    if counts[UNSUPPORTED]:
        status = STATUS_UNSUPPORTED
    elif counts[DEFER]:
        status = STATUS_DEGRADED
    else:
        status = STATUS_OK

    registry = _build_registry(case, case_dir, status, family, record)
    yaml_path = registry_path = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        registry_path = out_dir / f"{case}.registry.yaml"
        _write_yaml(
            registry_path, _merge_preserved(registry_path, registry), _REGISTRY_HEADER
        )
        if status in (STATUS_OK, STATUS_DEGRADED):
            _assert_grid_mode_declared(yaml_tree, family, case)
            yaml_path = out_dir / f"{case}.yaml"
            _write_yaml(yaml_path, yaml_tree, _CASE_HEADER)

    return Result(
        case,
        status,
        yaml_path,
        registry_path,
        counts,
        record.gaps,
        reference_source=record.reference_source,
    )


_INGESTED_NOTE = "read by case.extras.ic_source (mit_input_binary_ic)"
_DECLARED_NOTE = "declared only; no LOCOMgcm ingest yet"

#: PARM05 roles read by a source other than the initial-condition one.
_INGESTED_NOTES = {
    "delxfile": "read by stage1a mesh source (mit_input_binary_mesh)",
    "delyfile": "read by stage1a mesh source (mit_input_binary_mesh)",
    "zonalwindfile": (
        f"read by case.extras.forcing_source ({_FORCING_SOURCE_TYPE})"
    ),
    "meridwindfile": (
        f"read by case.extras.forcing_source ({_FORCING_SOURCE_TYPE})"
    ),
    "surfqnetfile": (
        f"read by case.extras.forcing_source ({_FORCING_SOURCE_TYPE})"
    ),
    "empmrfile": (
        f"read by case.extras.forcing_source ({_FORCING_SOURCE_TYPE})"
    ),
    "thetaclimfile": (
        f"read by sources.{_CLIMATOLOGY_SOURCE_NAME} ({_CLIMATOLOGY_SOURCE_TYPE})"
    ),
    "saltclimfile": (
        f"read by sources.{_CLIMATOLOGY_SOURCE_NAME} ({_CLIMATOLOGY_SOURCE_TYPE})"
    ),
}


def _collect_binary_inputs(ctx: Ctx, record: Record) -> None:
    """The case's PARM05 file table -- one row per declared binary.

    ``dtype`` is the case's own ``readBinaryPrec`` rendered big-endian; this is
    the table the initial-condition source is wired against, so a row whose
    disposition is ``map`` is actually read, and a ``defer`` row is only
    recorded.
    """
    dtype = _dtype_of(ctx.data("PARM01", "readbinaryprec", 32))
    for row in record.parameters:
        parameter_key = _lookup_key(str(row["parameter"]).rsplit(".", 1)[-1])
        is_initial_or_forcing = row["source"].startswith("input/data:PARM05.")
        is_spacing = (
            row["source"].startswith("input/data:PARM04.")
            and parameter_key in {"delxfile", "delyfile"}
        )
        if not (is_initial_or_forcing or is_spacing):
            continue
        if not isinstance(row["value"], str) or not row["value"]:
            continue
        if row["disposition"] not in (MAP, DEFER, UNSUPPORTED):
            continue
        ingested = row["disposition"] == MAP
        record.binary_inputs.append(
            {
                "parameter": row["parameter"],
                "path": row["value"],
                "dtype": dtype,
                "ingested": ingested,
                "note": (
                    _INGESTED_NOTES.get(
                        parameter_key,
                        _INGESTED_NOTE,
                    )
                    if ingested
                    else _DECLARED_NOTE
                ),
            }
        )


def _build_registry(
    case: str,
    case_dir: Path,
    status: str,
    family: str,
    record: Record,
) -> dict[str, Any]:
    root = case_dir.parent
    files = [
        str(p.relative_to(case_dir))
        for p in (
            case_dir / "input" / "data",
            case_dir / "input" / "data.pkg",
            case_dir / "input" / "eedata",
            case_dir / "code" / "SIZE.h",
            case_dir / "code" / "KPP_OPTIONS.h",
            case_dir / "code" / "obcs_calc.F",
        )
        if p.is_file()
    ]
    return {
        "registry_schema": 1,
        "case": case,
        "variant": None,
        # block 1 -- upstream anchor
        "upstream": {
            "pin": upstream_pin(root),
            "verification_root": str(root),
            "files": files,
        },
        # block 2 -- per-parameter mapping record (generated; do not hand-edit)
        "mapping": {
            "status": status,
            "grid_family": family,
            "counts": record.counts(),
            # True when at least one switch was explicitly disabled: the emitted
            # yaml is NOT the upstream configuration any more.
            "reconfigured": bool(record.disabled),
            "campaign_disabled": record.disabled,
            "silent_divergences": record.silent_divergences,
            "derived": record.derived,
            "binary_inputs": record.binary_inputs,
            "parameters": record.parameters,
        },
        # block 3 -- gate status + identity-scoped ratchet floors
        "gates": {
            # Which MITgcm run owns this case's reference values.  Derived from
            # the mapping record, so it is regenerated even though the rest of
            # this block is preserved (_REGENERATED_IN_PRESERVED).
            "reference_source": record.reference_source,
            "g0_translation": {"status": "generated"},
            "g1_baseline": {"execution_identity": None, "fingerprint": None},
            "g2_monitor": {
                "comparator_schema": COMPARATOR_SCHEMA,
                "artifact_schema": ARTIFACT_SCHEMA,
                # First-freeze metadata and floors are added only after a
                # qualifying artifact has been reviewed.  The whole gate block
                # remains preserved by _merge_preserved on retranslation.
                "identities": {},
            },
        },
        # block 4 -- capability gaps
        "capability_gaps": record.gaps,
        # block 5 -- adjoint axis, reserved
        "adjoint": {"status": "not_assessed", "variants": []},
    }


_PRESERVED_BLOCKS = ("gates", "adjoint")

#: Keys *inside* a preserved block that are nonetheless regenerated.  A
#: ``reference_source`` carried over from an earlier translation would keep
#: claiming the upstream reference is comparable after an explicit case override
#: disabled a switch -- stale provenance is exactly the failure mode the block
#: exists to prevent.
_REGENERATED_IN_PRESERVED: dict[str, tuple[str, ...]] = {
    "gates": ("reference_source",),
}


def _merge_preserved(path: Path, generated: dict[str, Any]) -> dict[str, Any]:
    """Keep hand-curated blocks 3/5 when regenerating an existing registry file."""
    if not path.is_file():
        return generated
    try:
        existing = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError:
        return generated
    merged = dict(generated)
    for block in _PRESERVED_BLOCKS:
        if block not in existing:
            continue
        kept = dict(existing[block]) if isinstance(existing[block], dict) else existing[block]
        if isinstance(kept, dict):
            for key in _REGENERATED_IN_PRESERVED.get(block, ()):
                if key in generated.get(block, {}):
                    kept[key] = generated[block][key]
        merged[block] = kept
    return merged


_REGISTRY_HEADER = (
    "# GENERATED by experiments/mitcases/translate.py -- do not hand-edit the\n"
    "# 'upstream' / 'mapping' / 'capability_gaps' blocks; 'gates' and 'adjoint'\n"
    "# are preserved across regeneration.\n"
)

_CASE_HEADER = (
    "# GENERATED by experiments/mitcases/translate.py from an MITgcm\n"
    "# verification case; the per-parameter provenance lives in the sibling\n"
    "# <case>.registry.yaml mapping record.\n"
)


def _write_yaml(path: Path, payload: dict[str, Any], header: str) -> None:
    text = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False, width=100)
    path.write_text(header + text)
