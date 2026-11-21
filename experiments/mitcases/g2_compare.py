"""Pure MIT ``%MON`` parsing and G2 significant-digits comparison.

Schema contract
---------------
``compare_g2`` accepts a ``to-g2-history/v1`` artifact and a selected
``g2-monitor/v1`` gate.  The public constants in this module freeze the key
sets shared with the Tier-0 producer, history producer, and registry/CLI
packages.  The comparator itself performs no filesystem, environment, git,
clock, or subprocess access and does not mutate its inputs.  Filesystem-facing
callers share the rooted reference resolver defined in this module.

Intentional differences from upstream ``tr_cmpnum.c``
-----------------------------------------------------
* Non-finite inputs and non-finite arithmetic are ``INVALID`` instead of
  silently truncating comparison at a NaN.
* The upstream 999-line limit is not inherited; an in-memory sequence has no
  legacy fixed-buffer limit.
* Missing terminators and malformed text are schema/parser ``INVALID`` errors,
  not the overloaded upstream ``99`` score.
* Paired sequences must have equal, non-zero lengths; unmatched points are
  structural errors rather than line-reader sentinel states.

The numeric path otherwise follows ``tr_cmpnum.c``: each pair uses the average
absolute magnitude, rounds ``log10(relative_error)`` to the nearest even
integer under Python's standard rounding, and the sequence score is the worst
pair.  Exact non-zero equality scores 16 and a double zero scores 22, matching
the upstream sentinels rather than clipping to the 14 digits printed by MIT.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Mapping, Sequence


COMPARATOR_SCHEMA = "g2-monitor/v1"
ARTIFACT_SCHEMA = "to-g2-history/v1"

GRID_BASE_FIELDS = (
    "XC",
    "XG",
    "DXC",
    "DXF",
    "DXG",
    "DXV",
    "YC",
    "YG",
    "DYC",
    "DYF",
    "DYG",
    "DYU",
    "RA",
    "RAW",
    "RAS",
    "RAZ",
    "AngleCS",
    "AngleSN",
    "fCori",
    "fCoriG",
    "fCoriCos",
)
GRID_STAT_SUFFIXES = ("max", "min", "mean", "sd")
GRID_KEYS = tuple(
    f"{field}_{suffix}"
    for field in GRID_BASE_FIELDS
    for suffix in GRID_STAT_SUFFIXES
)
TIER0_KEYS = (*GRID_KEYS, "ke_vol")
TIER1_KEYS = (
    "advcfl_uvel_max",
    "advcfl_vvel_max",
    "advcfl_wvel_max",
)
TIME_KEYS = ("time_tsnumber", "time_secondsf")

EXECUTION_IDENTITY_REQUIRED_FIELDS = frozenset(
    {
        "device",
        "dtype",
        "distributed_world_size",
        "IVADV",
        "TRITON_KPP",
        "FUSED",
    }
)

PASS = "PASS"
FAIL = "FAIL"
WARN_STALE = "WARN_STALE"
INVALID = "INVALID"
VALID = "VALID"
NOT_EVALUATED = "NOT_EVALUATED"

_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_INTEGER_RE = re.compile(r"[+-]?\d+")
_MON_RE = re.compile(
    r"^(?:\(PID\.TID\s+\d+\.\d+\)\s+)?"
    r"%MON\s+([A-Za-z0-9_]+)\s*=\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][+-]?\d+)?)\s*$"
)
_EXPONENT_RE = re.compile(r"[DEde]([+-]?\d+)$")
_UPSTREAM_OUTPUT_RE = re.compile(r"output(?:\.[A-Za-z0-9_.-]+)?\.txt")
_REFERENCE_ROOTS = frozenset({"verification", "repo"})
_REPO_ROOT = Path(__file__).resolve().parents[2]


class G2Invalid(ValueError):
    """A fail-closed parser, schema, identity, or pairing error."""


@dataclass(frozen=True)
class MonValue:
    """One parsed decimal lexeme and its binary64 comparison value."""

    key: str
    lexeme: str
    decimal: Decimal
    value: float
    printed_exponent: int


@dataclass(frozen=True)
class MonBlock:
    """One dynamic monitor block keyed by MIT iteration and model time."""

    iteration: int
    time: MonValue
    values: Mapping[str, MonValue]
    extras: Mapping[str, MonValue]


@dataclass(frozen=True)
class MitMonReference:
    """Parsed initialization statistics and ordered dynamic monitor blocks."""

    grid: Mapping[str, MonValue]
    initialization_extras: Mapping[str, MonValue]
    blocks: tuple[MonBlock, ...]
    ke_vol: MonValue


@dataclass(frozen=True)
class SignificantDigitsResult:
    """Worst-pair tr_cmpnum score plus maximum absolute/relative errors."""

    samples: int
    abs_error: float
    rel_error: float
    digits: int
    digits_floor: int


@dataclass(frozen=True)
class G2KeyResult:
    """One frozen key's numeric result against its registry floor."""

    key: str
    tier: str
    samples: int
    abs_error: float
    rel_error: float
    digits: int
    digits_floor: int
    required_floor: int
    passed: bool


@dataclass(frozen=True)
class G2Verdict:
    """Complete fail-closed result returned by :func:`compare_g2`."""

    overall_status: str
    artifact_status: str
    reference_status: str
    tier0_status: str
    tier1_status: str
    eligible_pass: bool
    key_results: tuple[G2KeyResult, ...] = ()
    unpaired: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class _GateSpec:
    case: str
    variant: str | None
    evaluation_commit: str
    expected_identity: Mapping[str, object]
    evaluate_tier1: bool
    tier0_floors: Mapping[str, int]
    tier1_floors: Mapping[str, int]
    reference: Mapping[str, object]


@dataclass(frozen=True)
class _ArtifactRecord:
    local_step: int
    global_step: int
    model_time_seconds: float
    values: Mapping[str, float]


@dataclass(frozen=True)
class _ArtifactSpec:
    commit: str
    stale: bool
    start_step: int
    tier0_values: Mapping[str, float]
    records: tuple[_ArtifactRecord, ...]


def _as_text_bytes(text: str | bytes) -> tuple[str, bytes]:
    if isinstance(text, bytes):
        try:
            return text.decode("utf-8"), text
        except UnicodeDecodeError as exc:
            raise G2Invalid("MIT reference is not valid UTF-8 text") from exc
    if isinstance(text, str):
        return text, text.encode("utf-8")
    raise G2Invalid(f"MIT reference must be str or bytes, got {type(text).__name__}")


def _parse_mon_value(key: str, lexeme: str) -> MonValue:
    normalized = lexeme.replace("D", "E").replace("d", "E")
    try:
        decimal = Decimal(normalized)
        value = float(decimal)
    except (InvalidOperation, OverflowError, ValueError) as exc:
        raise G2Invalid(f"invalid numeric value for %MON {key}: {lexeme!r}") from exc
    if not decimal.is_finite() or not math.isfinite(value):
        raise G2Invalid(f"non-finite numeric value for %MON {key}: {lexeme!r}")
    exponent_match = _EXPONENT_RE.search(lexeme)
    printed_exponent = (
        int(exponent_match.group(1))
        if exponent_match is not None
        else (decimal.adjusted() if decimal else 0)
    )
    return MonValue(key, lexeme, decimal, value, printed_exponent)


def _mon_entries(text: str) -> list[tuple[str, MonValue]]:
    entries: list[tuple[str, MonValue]] = []
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.rstrip("\r")
        if "%MON" not in line:
            continue
        match = _MON_RE.fullmatch(line)
        if match is None:
            raise G2Invalid(f"malformed %MON line {line_number}: {line!r}")
        key, lexeme = match.groups()
        entries.append((key, _parse_mon_value(key, lexeme)))
    if not entries:
        raise G2Invalid("MIT reference contains no %MON records")
    return entries


def parse_mit_mon(text: str | bytes) -> MitMonReference:
    """Parse a raw or harvested MIT ``%MON`` stream without filesystem I/O.

    Padding around the fixed-column MIT output is intentionally accepted, but
    the complete line, key grammar, decimal grammar, block headers, key
    placement, and required-key cardinalities are validated fail-closed.
    """

    decoded, _ = _as_text_bytes(text)
    entries = _mon_entries(decoded)
    grid: dict[str, MonValue] = {}
    init_extras: dict[str, MonValue] = {}
    blocks: list[MonBlock] = []
    current_iteration: int | None = None
    current_time: MonValue | None = None
    current_values: dict[str, MonValue] = {}
    current_extras: dict[str, MonValue] = {}
    seen_iterations: set[int] = set()
    awaiting_time = False

    def finish_block() -> None:
        nonlocal current_iteration, current_time, current_values, current_extras
        if current_iteration is None:
            return
        if current_time is None:
            raise G2Invalid(
                f"iteration {current_iteration} is missing adjacent time_secondsf"
            )
        missing = [key for key in (*TIER1_KEYS, "ke_vol") if key not in current_values]
        if missing:
            raise G2Invalid(
                f"iteration {current_iteration} is missing required keys: {missing}"
            )
        blocks.append(
            MonBlock(
                current_iteration,
                current_time,
                dict(current_values),
                dict(current_extras),
            )
        )
        current_iteration = None
        current_time = None
        current_values = {}
        current_extras = {}

    for key, parsed in entries:
        if key == "time_tsnumber":
            finish_block()
            if not _INTEGER_RE.fullmatch(parsed.lexeme):
                raise G2Invalid("time_tsnumber must use a decimal integer lexeme")
            iteration = int(parsed.lexeme, 10)
            if iteration in seen_iterations:
                raise G2Invalid(f"duplicate time_tsnumber {iteration}")
            if blocks and iteration <= blocks[-1].iteration:
                raise G2Invalid("time_tsnumber values must be strictly increasing")
            seen_iterations.add(iteration)
            current_iteration = iteration
            awaiting_time = True
            continue

        if awaiting_time:
            if key != "time_secondsf":
                raise G2Invalid(
                    f"time_tsnumber {current_iteration} is not followed by time_secondsf"
                )
            current_time = parsed
            awaiting_time = False
            continue

        if current_iteration is None:
            if key == "time_secondsf":
                raise G2Invalid("time_secondsf appears outside a dynamic block")
            if key in TIER1_KEYS or key == "ke_vol":
                raise G2Invalid(f"dynamic key {key} appears before the first time block")
            target = grid if key in GRID_KEYS else init_extras
            if key in target:
                raise G2Invalid(f"duplicate initialization key {key}")
            target[key] = parsed
            continue

        if key == "time_secondsf":
            raise G2Invalid("duplicate or misplaced time_secondsf")
        if key in GRID_KEYS:
            raise G2Invalid(f"initialization grid key {key} appears in a dynamic block")
        target = current_values if key in TIER1_KEYS or key == "ke_vol" else current_extras
        if key in target or key in current_values or key in current_extras:
            raise G2Invalid(f"duplicate key {key} in iteration {current_iteration}")
        target[key] = parsed

    if awaiting_time:
        raise G2Invalid(f"iteration {current_iteration} is missing time_secondsf")
    finish_block()
    missing_grid = [key for key in GRID_KEYS if key not in grid]
    if missing_grid:
        raise G2Invalid(f"initialization grid block is missing keys: {missing_grid}")
    if len(grid) != len(GRID_KEYS):
        raise G2Invalid("initialization grid block cardinality is invalid")
    if not blocks:
        raise G2Invalid("MIT reference contains no dynamic monitor blocks")
    first_ke = blocks[0].values["ke_vol"]
    for block in blocks[1:]:
        if block.values["ke_vol"].decimal != first_ke.decimal:
            raise G2Invalid("ke_vol changes between dynamic monitor blocks")
    return MitMonReference(dict(grid), dict(init_extras), tuple(blocks), first_ke)


def compare_significant_digits(
    to_values: Sequence[float],
    mit_values: Sequence[float],
) -> SignificantDigitsResult:
    """Return the upstream-equivalent worst-pair significant-digits score."""

    if len(to_values) != len(mit_values) or not to_values:
        raise G2Invalid("numeric series must have equal, non-zero lengths")
    best: int | None = None
    max_abs_error = 0.0
    max_rel_error = 0.0
    for index, (to_raw, mit_raw) in enumerate(zip(to_values, mit_values, strict=True)):
        to_value = _finite_float(to_raw, f"LOCOMgcm numeric series[{index}]")
        mit_value = _finite_float(mit_raw, f"MIT numeric series[{index}]")

        # tr_cmpnum: abave = 0.5*(fabs(a)+fabs(b)).
        abave = 0.5 * (abs(to_value) + abs(mit_value))
        if not math.isfinite(abave):
            raise G2Invalid(f"non-finite average magnitude at numeric pair {index}")
        abs_error = abs(to_value - mit_value)
        if not math.isfinite(abs_error):
            raise G2Invalid(f"non-finite absolute error at numeric pair {index}")

        # tr_cmpnum: an average magnitude greater than zero uses relative error.
        if abave > 0.0:
            rel_error = abs_error / abave
            # tr_cmpnum: positive error rounds log10(error) with rint.
            if rel_error > 0.0:
                cmplin = int(round(math.log10(rel_error)))
            # tr_cmpnum: exact equality at non-zero magnitude uses -16.
            else:
                cmplin = -16
        # tr_cmpnum: a double zero uses -22.
        else:
            rel_error = 0.0
            cmplin = -22

        # tr_cmpnum: best stores the largest cmplin, i.e. the worst pair.
        best = cmplin if best is None else max(best, cmplin)
        max_abs_error = max(max_abs_error, abs_error)
        max_rel_error = max(max_rel_error, rel_error)

    assert best is not None
    digits = -best
    return SignificantDigitsResult(
        len(to_values), max_abs_error, max_rel_error, digits, math.floor(digits)
    )


def _canonical_json(value: Mapping[str, object]) -> str:
    """Serialize a JSON-safe mapping with sorted compact keys.

    Two identities are compared through this text rather than with ``==`` so
    that ``True``/``1`` and ``0``/``0.0`` stay distinct, as JSON keeps them.
    """

    return json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise G2Invalid("canonical mappings require string keys")
            result[key] = _json_value(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise G2Invalid("canonical mappings reject non-finite floats")
        return value
    raise G2Invalid(f"canonical mappings reject {type(value).__name__}")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise G2Invalid(f"{label} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise G2Invalid(f"{label} must use string keys")
    return value


def _reject_unknown_fields(
    value: Mapping[str, object], allowed: set[str], label: str
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise G2Invalid(f"{label} has unknown fields: {unknown}")


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise G2Invalid(f"{label} must be a non-empty trimmed string")
    return value


def parse_reference_path(value: object) -> tuple[str, PurePosixPath]:
    """Parse one portable rooted reference path without filesystem access."""

    label = "gate.reference.path"
    path_text = _nonempty_string(value, label)
    migration = (
        "use verification:<case>/results/output.txt or "
        "repo:<repo-relative-path>"
    )
    if (
        PurePosixPath(path_text).is_absolute()
        or PureWindowsPath(path_text).is_absolute()
    ):
        raise G2Invalid(f"{label} absolute paths are forbidden; {migration}")
    root_name, separator, relative_text = path_text.partition(":")
    if not separator or root_name not in _REFERENCE_ROOTS:
        supplied = root_name if separator else "<missing>"
        raise G2Invalid(
            f"{label} has unknown root prefix {supplied!r}; expected "
            "'verification:' or 'repo:'"
        )
    relative = PurePosixPath(relative_text)
    if (
        not relative_text
        or relative.is_absolute()
        or PureWindowsPath(relative_text).is_absolute()
    ):
        raise G2Invalid(f"{label} absolute paths are forbidden; {migration}")
    if ".." in relative.parts:
        raise G2Invalid(f"{label} must not contain '..'")
    return root_name, relative


def _reference_root(root_name: str) -> Path:
    if root_name != "verification":
        return _REPO_ROOT
    from experiments.mitcases import translate

    root = translate.default_verification_root()
    if root is None:
        raise G2Invalid(
            "verification: reference root is unavailable; set "
            "MITCASES_VERIFICATION_ROOT or configure mitgcm_root "
            "(docs/INPUTS.md)"
        )
    return root


def resolve_reference_path(value: object) -> Path:
    """Resolve one portable reference path against its registered root."""

    root_name, relative = parse_reference_path(value)
    return _reference_root(root_name).joinpath(*relative.parts)


def resolve_reference(reference: Mapping[str, object]) -> Path:
    """Resolve a gate reference, refusing an upstream tree at another pin.

    A ``verification:`` path names a file of whichever tree
    ``MITCASES_VERIFICATION_ROOT`` or its defaults select.  The tree's ``PIN``
    file must name the commit recorded in ``reference.identity.pin``, so a root
    pointing at another MITgcm version cannot silently swap the reference.
    ``repo:`` references are tracked files of this repository.
    """

    root_name, relative = parse_reference_path(reference.get("path"))
    root = _reference_root(root_name)
    if root_name == "verification":
        from experiments.mitcases import translate

        identity = reference.get("identity")
        expected = identity.get("pin") if isinstance(identity, Mapping) else None
        observed = translate.upstream_pin(root)
        if not isinstance(expected, str) or observed != expected:
            raise G2Invalid(
                f"verification root {root} is pinned at {observed!r}, but the "
                f"reference was registered at pin {expected!r}"
            )
    return root.joinpath(*relative.parts)


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise G2Invalid(f"{label} must be an integer")
    return value


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise G2Invalid(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise G2Invalid(f"{label} must be finite")
    return result


def _commit(value: object, label: str) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise G2Invalid(f"{label} must be a known lowercase 40-hex commit")
    return value


def _identity(value: object, label: str) -> Mapping[str, object]:
    identity = _mapping(value, label)
    missing = sorted(EXECUTION_IDENTITY_REQUIRED_FIELDS - identity.keys())
    if missing:
        raise G2Invalid(f"{label} is missing required fields: {missing}")
    _canonical_json(identity)
    return identity


def _floors(value: object, expected_keys: Sequence[str], label: str) -> Mapping[str, int]:
    floors = _mapping(value, label)
    expected = set(expected_keys)
    actual = set(floors)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise G2Invalid(f"{label} key mismatch: missing={missing}, extra={extra}")
    validated: dict[str, int] = {}
    for key in expected_keys:
        floor = _integer(floors[key], f"{label}.{key}")
        if not 0 <= floor <= 22:
            raise G2Invalid(f"{label}.{key} must be between 0 and 22")
        validated[key] = floor
    return validated


def _validate_attribution(gate: Mapping[str, object], floors: Mapping[str, int]) -> None:
    attribution = _mapping(gate.get("attribution", {}), "gate.attribution")
    allowed = {"definition_difference", "convention_difference", "model_difference"}
    for key, floor in floors.items():
        if floor >= 4:
            continue
        item = _mapping(attribution.get(key), f"gate.attribution.{key}")
        if item.get("category") not in allowed:
            raise G2Invalid(f"gate.attribution.{key}.category is invalid")
        _nonempty_string(item.get("note"), f"gate.attribution.{key}.note")


def validate_reference_route(
    reference: Mapping[str, object], case: str, variant: str | None
) -> None:
    source = reference.get("source")
    if source not in {"upstream_shipped", "local_mit_rerun"}:
        raise G2Invalid("gate.reference.source is invalid")
    _, path = parse_reference_path(reference.get("path"))
    is_upstream_result = (
        path.parent.name == "results"
        and _UPSTREAM_OUTPUT_RE.fullmatch(path.name) is not None
    )
    if source == "local_mit_rerun":
        if is_upstream_result:
            raise G2Invalid("local_mit_rerun must not route to results/output*.txt")
        return
    expected_name = "output.txt" if variant is None else f"output.{variant}.txt"
    if not is_upstream_result or path.name != expected_name:
        raise G2Invalid(
            f"upstream_shipped must route to results/{expected_name}"
        )
    if path.parent.parent.name != case:
        raise G2Invalid("upstream reference path does not match gate.case")


def _validate_gate(gate_raw: Mapping[str, object]) -> _GateSpec:
    gate = _mapping(gate_raw, "gate")
    if gate.get("comparator_schema") != COMPARATOR_SCHEMA:
        raise G2Invalid(f"gate.comparator_schema must be {COMPARATOR_SCHEMA!r}")
    if gate.get("artifact_schema") != ARTIFACT_SCHEMA:
        raise G2Invalid(f"gate.artifact_schema must be {ARTIFACT_SCHEMA!r}")
    case = _nonempty_string(gate.get("case"), "gate.case")
    variant_raw = gate.get("variant")
    variant = None if variant_raw is None else _nonempty_string(variant_raw, "gate.variant")
    evaluation_commit = _commit(gate.get("evaluation_commit"), "gate.evaluation_commit")
    expected_identity = _identity(
        gate.get("expected_execution_identity"),
        "gate.expected_execution_identity",
    )
    if not isinstance(gate.get("evaluate_tier1"), bool):
        raise G2Invalid("gate.evaluate_tier1 must be boolean")
    evaluate_tier1 = gate["evaluate_tier1"]
    floors = _mapping(gate.get("floors"), "gate.floors")
    tier0_floors = _floors(floors.get("tier0"), TIER0_KEYS, "gate.floors.tier0")
    tier1_expected = TIER1_KEYS if evaluate_tier1 else ()
    tier1_floors = _floors(
        floors.get("tier1"), tier1_expected, "gate.floors.tier1"
    )
    _validate_attribution(gate, {**tier0_floors, **tier1_floors})
    reference = _mapping(gate.get("reference"), "gate.reference")
    validate_reference_route(reference, case, variant)
    reference_identity = _mapping(
        reference.get("identity"), "gate.reference.identity"
    )
    if not reference_identity:
        raise G2Invalid("gate.reference.identity must not be empty")
    return _GateSpec(
        case,
        variant,
        evaluation_commit,
        expected_identity,
        evaluate_tier1,
        tier0_floors,
        tier1_floors,
        reference,
    )


def _same_time(left: float, right: float) -> bool:
    tolerance = max(8.0 * math.ulp(left), 8.0 * math.ulp(right))
    return abs(left - right) <= tolerance


def _validate_artifact(
    artifact_raw: Mapping[str, object], gate: _GateSpec
) -> _ArtifactSpec:
    artifact = _mapping(artifact_raw, "artifact")
    _reject_unknown_fields(
        artifact,
        {"schema", "case", "variant", "producer", "clock", "records", "tier0"},
        "artifact",
    )
    if artifact.get("schema") != ARTIFACT_SCHEMA:
        raise G2Invalid(f"artifact.schema must be {ARTIFACT_SCHEMA!r}")
    if artifact.get("case") != gate.case or artifact.get("variant") != gate.variant:
        raise G2Invalid("artifact case/variant does not match the selected gate")
    producer = _mapping(artifact.get("producer"), "artifact.producer")
    # ``identity_sha256`` is a legacy field of older
    # artifacts; it is accepted and ignored, the identity itself is compared.
    _reject_unknown_fields(
        producer,
        {"commit", "dirty", "execution_identity", "identity_sha256"},
        "artifact.producer",
    )
    commit = _commit(producer.get("commit"), "artifact.producer.commit")
    if producer.get("dirty") is not False:
        raise G2Invalid("artifact.producer.dirty must be false")
    identity = _identity(
        producer.get("execution_identity"),
        "artifact.producer.execution_identity",
    )
    if _canonical_json(identity) != _canonical_json(gate.expected_identity):
        raise G2Invalid("artifact execution identity does not match the selected gate")

    tier0 = _mapping(artifact.get("tier0"), "artifact.tier0")
    _reject_unknown_fields(tier0, {"values", "unavailable"}, "artifact.tier0")
    unavailable = _mapping(tier0.get("unavailable"), "artifact.tier0.unavailable")
    unavailable_required = sorted(set(unavailable) & set(TIER0_KEYS))
    if unavailable_required:
        raise G2Invalid(f"required Tier-0 values are unavailable: {unavailable_required}")
    tier0_raw = _mapping(tier0.get("values"), "artifact.tier0.values")
    if set(tier0_raw) != set(TIER0_KEYS):
        missing = sorted(set(TIER0_KEYS) - set(tier0_raw))
        extra = sorted(set(tier0_raw) - set(TIER0_KEYS))
        raise G2Invalid(
            f"artifact.tier0.values key mismatch: missing={missing}, extra={extra}"
        )
    tier0_values = {
        key: _finite_float(tier0_raw[key], f"artifact.tier0.values.{key}")
        for key in TIER0_KEYS
    }

    clock_raw = artifact.get("clock")
    has_clock = clock_raw is not None
    clock = _mapping(clock_raw, "artifact.clock") if has_clock else {}
    _reject_unknown_fields(
        clock,
        {
            "base_time_seconds",
            "start_time_seconds",
            "start_iteration",
            "start_step",
            "dt_clock",
        },
        "artifact.clock",
    )
    clock_fields = {"base_time_seconds", "start_step", "dt_clock"}
    complete_clock = clock_fields <= set(clock)
    if has_clock and not complete_clock:
        raise G2Invalid(
            "artifact.clock must contain base_time_seconds/start_step/dt_clock together"
        )
    base_time = 0.0
    clock_start = 0
    dt_clock = 0.0
    if complete_clock:
        base_time = _finite_float(
            clock["base_time_seconds"], "artifact.clock.base_time_seconds"
        )
        clock_start = _integer(clock["start_step"], "artifact.clock.start_step")
        dt_clock = _finite_float(clock["dt_clock"], "artifact.clock.dt_clock")
        if dt_clock <= 0.0:
            raise G2Invalid("artifact.clock.dt_clock must be positive")
        if "start_iteration" in clock:
            start_iteration = _integer(
                clock["start_iteration"], "artifact.clock.start_iteration"
            )
            if start_iteration != clock_start:
                raise G2Invalid("artifact.clock.start_iteration must equal start_step")
        if "start_time_seconds" in clock:
            start_time = _finite_float(
                clock["start_time_seconds"], "artifact.clock.start_time_seconds"
            )
            expected_start_time = base_time + clock_start * dt_clock
            if not _same_time(start_time, expected_start_time):
                raise G2Invalid("artifact.clock.start_time_seconds is inconsistent")

    records_raw = artifact.get("records")
    if not isinstance(records_raw, (list, tuple)) or not records_raw:
        raise G2Invalid("artifact.records must be a non-empty sequence")
    records: list[_ArtifactRecord] = []
    derived_start: int | None = None
    seen_global: set[int] = set()
    for index, record_raw in enumerate(records_raw):
        record = _mapping(record_raw, f"artifact.records[{index}]")
        _reject_unknown_fields(
            record,
            {
                "local_step",
                "global_step",
                "elapsed_seconds",
                "model_time_seconds",
                "values",
            },
            f"artifact.records[{index}]",
        )
        local_step = _integer(record.get("local_step"), f"artifact.records[{index}].local_step")
        global_step = _integer(
            record.get("global_step"), f"artifact.records[{index}].global_step"
        )
        if local_step < 0:
            raise G2Invalid(f"artifact.records[{index}].local_step must be non-negative")
        candidate_start = global_step - local_step
        if derived_start is None:
            derived_start = candidate_start
        elif candidate_start != derived_start:
            raise G2Invalid("artifact record global/local step origins are inconsistent")
        if global_step in seen_global:
            raise G2Invalid(f"duplicate artifact global_step {global_step}")
        if records and (
            local_step <= records[-1].local_step
            or global_step <= records[-1].global_step
        ):
            raise G2Invalid("artifact records must be strictly ordered by local/global step")
        seen_global.add(global_step)

        explicit_time = record.get("model_time_seconds")
        if complete_clock:
            model_time = base_time + global_step * dt_clock
            if not math.isfinite(model_time):
                raise G2Invalid(f"artifact.records[{index}] derived model time is non-finite")
            if explicit_time is not None:
                supplied_time = _finite_float(
                    explicit_time, f"artifact.records[{index}].model_time_seconds"
                )
                if not _same_time(supplied_time, model_time):
                    raise G2Invalid(
                        f"artifact.records[{index}].model_time_seconds is inconsistent"
                    )
        else:
            if explicit_time is None:
                raise G2Invalid(
                    "artifact lacks both a complete absolute clock and explicit model_time_seconds"
                )
            model_time = _finite_float(
                explicit_time, f"artifact.records[{index}].model_time_seconds"
            )
        if "elapsed_seconds" in record:
            elapsed = _finite_float(
                record["elapsed_seconds"], f"artifact.records[{index}].elapsed_seconds"
            )
            if complete_clock and not _same_time(elapsed, local_step * dt_clock):
                raise G2Invalid(
                    f"artifact.records[{index}].elapsed_seconds is inconsistent"
                )

        values_raw = _mapping(record.get("values"), f"artifact.records[{index}].values")
        if set(values_raw) != set(TIER1_KEYS):
            missing = sorted(set(TIER1_KEYS) - set(values_raw))
            extra = sorted(set(values_raw) - set(TIER1_KEYS))
            raise G2Invalid(
                f"artifact.records[{index}].values mismatch: missing={missing}, extra={extra}"
            )
        values = {
            key: _finite_float(
                values_raw[key], f"artifact.records[{index}].values.{key}"
            )
            for key in TIER1_KEYS
        }
        records.append(_ArtifactRecord(local_step, global_step, model_time, values))

    assert derived_start is not None
    if complete_clock and derived_start != clock_start:
        raise G2Invalid("artifact clock start_step disagrees with record step origin")
    return _ArtifactSpec(
        commit,
        commit != gate.evaluation_commit,
        derived_start,
        tier0_values,
        tuple(records),
    )


def _time_tolerance(model_time: float, mit_time: MonValue) -> float:
    decimal_rounding = 0.5 * 10.0 ** (mit_time.printed_exponent - 13)
    return max(8.0 * math.ulp(model_time), decimal_rounding)


def _pair_times(artifact: _ArtifactSpec, reference: MitMonReference) -> None:
    if artifact.start_step != reference.blocks[0].iteration:
        raise G2Invalid(
            "artifact start_step does not match the reference nIter0/first iteration"
        )
    artifact_steps = tuple(record.global_step for record in artifact.records)
    reference_steps = tuple(block.iteration for block in reference.blocks)
    if artifact_steps != reference_steps:
        missing_to = sorted(set(reference_steps) - set(artifact_steps))
        missing_mit = sorted(set(artifact_steps) - set(reference_steps))
        raise G2Invalid(
            f"unpaired iterations: missing_to={missing_to}, missing_mit={missing_mit}"
        )
    for record, block in zip(artifact.records, reference.blocks, strict=True):
        tolerance = _time_tolerance(record.model_time_seconds, block.time)
        if abs(record.model_time_seconds - block.time.value) > tolerance:
            raise G2Invalid(
                f"model time mismatch at iteration {record.global_step}: "
                f"LOCOMgcm={record.model_time_seconds}, MIT={block.time.lexeme}, atol={tolerance}"
            )


def _key_result(
    key: str,
    tier: str,
    to_values: Sequence[float],
    mit_values: Sequence[float],
    required_floor: int,
) -> G2KeyResult:
    numeric = compare_significant_digits(to_values, mit_values)
    return G2KeyResult(
        key,
        tier,
        numeric.samples,
        numeric.abs_error,
        numeric.rel_error,
        numeric.digits,
        numeric.digits_floor,
        required_floor,
        numeric.digits >= required_floor,
    )


def _invalid_verdict(
    reason: str,
    *,
    artifact_status: str = NOT_EVALUATED,
    reference_status: str = NOT_EVALUATED,
    unpaired: tuple[str, ...] = (),
) -> G2Verdict:
    return G2Verdict(
        INVALID,
        artifact_status,
        reference_status,
        INVALID,
        NOT_EVALUATED,
        False,
        unpaired=unpaired,
        reasons=(reason,),
    )


def compare_g2(
    to_artifact: Mapping[str, object],
    mit_reference_text: str | bytes,
    gate: Mapping[str, object],
) -> G2Verdict:
    """Compare one artifact and reference using only caller-supplied values."""

    try:
        gate_spec = _validate_gate(gate)
    except (G2Invalid, TypeError, ValueError) as exc:
        return _invalid_verdict(str(exc))
    try:
        artifact = _validate_artifact(to_artifact, gate_spec)
    except (G2Invalid, TypeError, ValueError) as exc:
        return _invalid_verdict(str(exc), artifact_status=INVALID)

    artifact_status = WARN_STALE if artifact.stale else VALID
    try:
        decoded, _ = _as_text_bytes(mit_reference_text)
        reference = parse_mit_mon(decoded)
    except (G2Invalid, TypeError, ValueError) as exc:
        return _invalid_verdict(
            str(exc), artifact_status=artifact_status, reference_status=INVALID
        )

    try:
        _pair_times(artifact, reference)
        key_results: list[G2KeyResult] = []
        for key in GRID_KEYS:
            key_results.append(
                _key_result(
                    key,
                    "tier0",
                    (artifact.tier0_values[key],),
                    (reference.grid[key].value,),
                    gate_spec.tier0_floors[key],
                )
            )
        key_results.append(
            _key_result(
                "ke_vol",
                "tier0",
                (artifact.tier0_values["ke_vol"],),
                (reference.ke_vol.value,),
                gate_spec.tier0_floors["ke_vol"],
            )
        )
        tier0_pass = all(result.passed for result in key_results)
        tier0_status = PASS if tier0_pass else FAIL
        tier1_status = NOT_EVALUATED
        if tier0_pass and gate_spec.evaluate_tier1:
            for key in TIER1_KEYS:
                key_results.append(
                    _key_result(
                        key,
                        "tier1",
                        tuple(record.values[key] for record in artifact.records),
                        tuple(block.values[key].value for block in reference.blocks),
                        gate_spec.tier1_floors[key],
                    )
                )
            tier1_pass = all(
                result.passed for result in key_results if result.tier == "tier1"
            )
            tier1_status = PASS if tier1_pass else FAIL
    except (G2Invalid, TypeError, ValueError) as exc:
        unpaired = (str(exc),) if "unpaired" in str(exc) else ()
        return _invalid_verdict(
            str(exc),
            artifact_status=artifact_status,
            reference_status=VALID,
            unpaired=unpaired,
        )

    numeric_pass = tier0_status == PASS and tier1_status in {PASS, NOT_EVALUATED}
    if artifact.stale:
        overall_status = WARN_STALE
        eligible_pass = False
        reasons = ("artifact commit differs from evaluation_commit",)
    else:
        overall_status = PASS if numeric_pass else FAIL
        eligible_pass = numeric_pass
        reasons = ()
    return G2Verdict(
        overall_status,
        artifact_status,
        VALID,
        tier0_status,
        tier1_status,
        eligible_pass,
        tuple(key_results),
        reasons=reasons,
    )
