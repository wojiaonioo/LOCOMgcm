"""The translation engine: namelist and package translation, code directory accounting."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from experiments.cases.scs_clm.mit_native_io import parse_namelist
from experiments.mitcases.code_inventory import (
    DEFER,
    DISABLED,
    MAP,
    REVIEW,
    SKIP,
    UNSUPPORTED,
    Run,
    inventory_case,
)

from .core import (
    CAMPAIGN_RULING,
    Deferred,
    Gap,
    Record,
    Skipped,
    UnknownParameter,
    _RECORD_METADATA,
    _lookup_key,
    _record_gap,
    _set_dotted,
    campaign_ruling,
)
from .rules import (
    Ctx,
    Rule,
)
from .namelist import (
    _MIT_UNSET,
    _mit_unset,
    _upper_groups,
)
from .cpp import (
    _reference_mit_root,
)
from .vmix import (
    _VERTICAL_MIXING_PACKAGES,
)
from .package_rules import (
    PACKAGE_RULES,
)

# --------------------------------------------------------------------------
# translation engine
# --------------------------------------------------------------------------


def _apply(
    rule: Rule,
    value: Any,
    ctx: Ctx,
    *,
    source: str,
    parameter: str,
    record: Record,
    yaml_tree: dict[str, Any],
) -> None:
    row: dict[str, Any] = {"source": source, "parameter": parameter, "value": value}
    if rule.disposition == SKIP:
        record.add(**row, disposition=SKIP, reason=rule.reason)
        return
    if rule.disposition == DEFER:
        record.add(**row, disposition=DEFER, reason=rule.reason)
        return
    if rule.disposition == UNSUPPORTED:
        _record_gap(record, row, rule.reason)
        return
    fn = rule.fn or (lambda v, _ctx: {rule.target: v})
    try:
        writes = dict(fn(value, ctx))
    except Skipped as exc:
        record.add(**row, disposition=SKIP, reason=exc.reason)
        return
    except Deferred as exc:
        record.add(**row, disposition=DEFER, reason=exc.reason)
        return
    except Gap as exc:
        _record_gap(record, row, exc.reason)
        return
    metadata = dict(writes.pop(_RECORD_METADATA, {}))
    disposition = metadata.pop("disposition", MAP)
    for target, resolved in writes.items():
        _set_dotted(yaml_tree, target, resolved)
    record.add(
        **row,
        disposition=disposition,
        target=sorted(writes),
        convert=rule.convert,
        **metadata,
    )


def _translate_namelist(
    path: Path,
    rules: dict[tuple[str, str], Rule],
    ctx: Ctx,
    *,
    label: str,
    record: Record,
    yaml_tree: dict[str, Any],
) -> None:
    for group, entries in parse_namelist(path).items():
        gname = group.upper()
        for key, value in entries.items():
            parameter = f"{gname}.{key}"
            source = f"{label}:{parameter}"
            if value is None:
                record.add(
                    source=source,
                    parameter=parameter,
                    value=None,
                    disposition=SKIP,
                    reason="empty assignment (MITgcm leaves the default in place)",
                )
                continue
            unset = _MIT_UNSET.get((path.name, gname, _lookup_key(key)))
            if unset is not None and _mit_unset(path.name, gname, key, value):
                record.add(
                    source=source,
                    parameter=parameter,
                    value=value,
                    disposition=SKIP,
                    reason=f"MITgcm's unset value {unset.value!r}, the same as no entry ({unset.source})",
                )
                continue
            if (unset is not None and not unset.per_element and isinstance(value, list)
                    and any(_mit_unset(path.name, gname, key, item) for item in value)):
                _record_gap(
                    record,
                    {"source": source, "parameter": parameter, "value": value},
                    f"entries at MITgcm's unset value {unset.value!r}, which MITgcm resolves "
                    f"entry by entry ({unset.source}) and the translator does not",
                )
                continue
            rule = rules.get((gname, _lookup_key(key)))
            if rule is None:
                raise UnknownParameter(parameter, source)
            _apply(
                rule,
                value,
                ctx,
                source=source,
                parameter=parameter,
                record=record,
                yaml_tree=yaml_tree,
            )


def _translate_packages(
    path: Path,
    ctx: Ctx,
    *,
    record: Record,
    yaml_tree: dict[str, Any],
) -> set[str]:
    """Return the set of package flags switched on."""
    enabled: set[str] = set()
    for group, entries in parse_namelist(path).items():
        gname = group.upper()
        for key, value in entries.items():
            parameter = f"{gname}.{key}"
            source = f"input/data.pkg:{parameter}"
            lkey = _lookup_key(key)
            if value is not True:
                record.add(
                    source=source,
                    parameter=parameter,
                    value=value,
                    disposition=SKIP,
                    reason="package switched off (MITgcm never reads its namelist)",
                )
                continue
            rule = PACKAGE_RULES.get(lkey)
            if rule is None:
                raise UnknownParameter(parameter, source)
            enabled.add(lkey)
            _apply(
                rule,
                value,
                ctx,
                source=source,
                parameter=parameter,
                record=record,
                yaml_tree=yaml_tree,
            )
    return enabled


#: The one auxiliary namelist file a package's readparms opens; another file the
#: case ships under the package's name is never read.
_PACKAGE_NAMELIST_FILES = {"useexf": ("data.exf", "pkg/exf/exf_readparms.F:931-934")}


def _account_auxiliary_namelists(
    input_dir: Path,
    enabled: set[str],
    *,
    record: Record,
) -> None:
    """Account every ``data.<pkg>`` file at file level.

    A package that is not switched on in ``data.pkg`` is never read by MITgcm
    (packages_boot.F:113-145), so its namelist is whitelisted wholesale.  A
    package that *is* on already contributed its own ``defer``/``unsupported``
    row, and its parameters inherit that disposition -- the translator does not carry
    per-package parameter tables.  A file whose owning package cannot be
    established from its name is deferred rather than whitelisted: claiming
    "MITgcm never reads this" without evidence would be fail-open.  A package
    in :data:`_PACKAGE_NAMELIST_FILES` opens one file only, so any other file
    under its name is a skip.
    """
    for path in sorted(input_dir.glob("data.*")):
        if (
            path.name == "data.pkg"
            or (path.name == "data.obcs" and "useobcs" in enabled)
            or (path.name == "data.kpp" and "usekpp" in enabled)
            or (path.name == "data.exf" and "useexf" in enabled)
            or (path.name == "data.cal" and "usecal" in enabled)
            or (path.name == "data.gmredi" and "usegmredi" in enabled and record.verbatim)
            or any(path.name == filename and flag in enabled
                   for flag, (_, filename, _) in _VERTICAL_MIXING_PACKAGES.items())
            or not path.is_file()
        ):
            continue
        flag = _owning_package_flag(path.name, enabled)
        anchor = ""
        if flag is None:
            disposition: str = DEFER
            reason = (
                f"{path.name} is not attributable to a data.pkg flag by name; "
                "file-level accounting only"
            )
        elif flag in _PACKAGE_NAMELIST_FILES and path.name != _PACKAGE_NAMELIST_FILES[flag][0]:
            opened, where = _PACKAGE_NAMELIST_FILES[flag]
            disposition = SKIP
            reason = f"the package opens {opened} only ({where}): MITgcm never reads {path.name}"
        elif flag in enabled:
            rule = PACKAGE_RULES[flag]
            ruling = None if record.verbatim else campaign_ruling(flag, rule.reason)
            if ruling is not None:
                # The flag itself is explicitly disabled, so the re-run MITgcm
                # never reads this namelist either (packages_boot.F:113-145).
                disposition = DISABLED
                anchor = ruling.anchor
                reason = (
                    f"{flag} is {CAMPAIGN_RULING}: {ruling.note}; with the "
                    "package off MITgcm never reads this namelist"
                )
            else:
                disposition = rule.disposition
                reason = (
                    f"{flag} is on: this namelist inherits its disposition "
                    f"({rule.disposition}: {rule.reason}); the translator carries no "
                    "per-package parameter table"
                )
        else:
            disposition = SKIP
            reason = f"{flag} is not .TRUE. in data.pkg: MITgcm never reads this file"
        row = {
            "source": f"input/{path.name}",
            "parameter": f"<file:{path.name}>",
            "value": None,
            "disposition": disposition,
        }
        if anchor:
            row["ruling"] = anchor
        row["gap" if disposition == UNSUPPORTED else "reason"] = reason
        record.add(**row)


def _account_code_directory(case_dir: Path, ctx: Ctx, *, record: Record) -> None:
    """One row per item of the case's ``code/`` (code_inventory.inventory_case).

    The inventory's dispositions are the record's, except ``review`` (no ruling
    table covers the item), which is recorded ``defer``: the case runs without a
    decision on it, and the verbatim mode refuses it (:meth:`Record.add`).  A
    conditional ruling reads the case's namelists, eedata included, its SIZE.h
    and its input file names.
    """
    input_dir = case_dir / "input"
    namelists = dict(ctx.namelists)
    if (input_dir / "eedata").is_file():
        namelists["eedata"] = _upper_groups(parse_namelist(input_dir / "eedata"))
    run = Run(namelists, ctx.size_h, frozenset(p.name for p in input_dir.iterdir()))
    for item in inventory_case(case_dir, _reference_mit_root(), run):
        name = f"{item.file}:{item.item}" if item.item else item.file
        row = {
            "source": f"code/{item.file}",
            "parameter": f"<code:{name}>",
            "value": item.change,
        }
        if item.basis:
            row["basis"] = item.basis
        if item.disposition == REVIEW:
            record.add(**row, disposition=DEFER,
                       reason=f"{item.kind} item without a code_inventory ruling")
        elif item.disposition == UNSUPPORTED:
            record.add(**row, disposition=UNSUPPORTED, gap=item.model)
        else:
            record.add(**row, disposition=item.disposition, reason=item.model)


def _owning_package_flag(filename: str, enabled: set[str]) -> str | None:
    """``data.exf_YearlyFields`` -> ``useexf``; unattributable names -> ``None``.

    MITgcm's auxiliary-namelist names are only loosely derived from package
    names.  An exact ``use<token>`` hit wins outright; a looser prefix hit
    (``data.shap`` -> ``useshap_filt``) is accepted only when it is unique *and*
    the package is actually switched on, so a wrong guess can never manufacture
    the fail-open claim "MITgcm never reads this file".
    """
    token = filename.split(".", 1)[1].lower()
    parts = [p for chunk in token.split(".") for p in chunk.split("_")]
    for stop in range(len(parts), 0, -1):
        for joiner in ("_", "."):
            flag = "use" + joiner.join(parts[:stop])
            if flag in PACKAGE_RULES:
                return flag
    for stop in range(len(parts), 0, -1):
        prefix = "use" + "_".join(parts[:stop])
        hits = sorted(f for f in enabled if f.startswith(prefix))
        if len(hits) == 1:
            return hits[0]
    return None
