"""Single source of truth for time-step count and total simulated seconds.

Background — a float-days literal can make the two models run different
step counts.  ``days: 0.0008680556`` is 75/86400 = 0.0008680555555... rounded
UP at the 10th decimal; times 86400 it gives 75.00000384 s, 3.84 microseconds
over a clean 5-step run at dt=15s.  ``int(round(75.00000384 / 15)) = 5`` but
``int(math.ceil(75.00000384 / 15)) = 6``: if the LOCOMgcm runtime rounds and the
MIT export takes the ceiling, MIT runs one extra step (endTime=90s), and a
comparison with a loose time tolerance pairs LOCOMgcm step 5 (t=75s) with MIT
step 6 (t=90s).  The resulting "SSH error" is a full real dynamics step, not
a numerical difference.

This module exists so that **cannot happen**:

* a single :func:`compute_n_steps` is the only place that maps
  ``(days, dt, steps_override)`` to an integer step count;
* :func:`exact_total_seconds` recovers the simulated horizon as exact
  ``n_steps * dt`` so MIT's ``endTime`` is always an integer multiple of
  ``dt``, never re-derived from a floating-point ``days`` literal;
* :func:`compute_n_steps` raises unless ``time.seconds`` or legacy
  ``time.days`` resolves to an integer number of steps within a tiny
  floating-point tolerance. The user is told to either fix the duration
  literal or use ``time.steps: N`` directly.

All LOCOMgcm step loops, all MIT-export step calculations, and all stage5
alignment checks must route through this module. Do not reintroduce
``math.ceil(days * 86400 / dt)`` anywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "StepCountTrapError",
    "StepResolution",
    "compute_n_steps",
    "exact_total_seconds",
    "resolve_steps",
]


SECONDS_PER_DAY: float = 86400.0
"""Whole-second alignment uses 86400 s/day exactly (no leap seconds)."""

_STEP_RATIO_ABS_TOL: float = 1.0e-10
"""Allowed absolute residual in ``requested_seconds / dt_seconds`` steps."""


class StepCountTrapError(ValueError):
    """Raised when requested duration is not an integer step count.

    The yaml literal expanded by ``days * SECONDS_PER_DAY`` or the direct
    ``time.seconds`` value must be an integer multiple of ``dt`` within
    ``_STEP_RATIO_ABS_TOL`` steps. This catches the step-mismatch state
    described in the module docstring and also rejects large nearest-step
    rounding, including exact half-step requests. The caller must respond by
    either widening/fixing the duration literal or — preferred when the step
    count is the intent — switching to ``time.steps: N``.
    """


@dataclass(frozen=True)
class StepResolution:
    """Resolved step bookkeeping derived from ``(days, dt, steps_override)``.

    Fields
    ------
    n_steps:
        Integer step count actually used by the runtime.
    dt_seconds:
        The ``dt`` actually used (mirrors the input, kept here to keep
        downstream call sites self-contained).
    total_seconds_exact:
        ``n_steps * dt_seconds`` — exact integer-arithmetic horizon.
        MIT's ``endTime`` must be set from this, not from
        ``days * 86400``.
    source:
        ``"steps_override"`` when ``time.steps`` decided the count,
        ``"seconds_round"`` when ``time.seconds`` did, and
        ``"days_round"`` when ``days`` did. Used by diagnostics.
    """

    n_steps: int
    dt_seconds: float
    total_seconds_exact: float
    source: str


def compute_n_steps(
    days: float | None = None,
    dt_seconds: float | None = None,
    steps_override: int | None = None,
    *,
    seconds_override: float | None = None,
    context: str = "",
) -> int:
    """Resolve the integer step count from yaml ``time`` config.

    Precedence (most precise first):

    1. ``steps_override`` (yaml ``time.steps``) — exact integer count.
    2. ``seconds_override`` (yaml ``time.seconds``) — total simulated
       seconds; divided by ``dt`` and checked against an integer step
       ratio. Strings like ``"5 * 86400"`` are pre-evaluated by
       ``TimeConfig.__post_init__`` so this function only ever sees a
       float.
    3. ``days`` (yaml ``time.days``) — legacy float-days input.

    The ``StepCountTrapError`` fires whenever the requested duration is not
    an integer number of steps within ``_STEP_RATIO_ABS_TOL``. Caller must
    respond by switching to ``steps`` or ``seconds`` (preferred) or widening
    the float literal to enough decimals that the integer-multiple check
    succeeds.

    Parameters
    ----------
    days:
        Legacy ``run.time.days`` value. Ignored when a higher-precision
        override is given. May be ``None``.
    dt_seconds:
        ``run.time.dt`` in seconds. Must be > 0.
    steps_override:
        ``run.time.steps`` if set — exact integer count, bypasses any
        divide. **Most preferred** for back-compat with yamls already
        using the field.
    seconds_override:
        ``run.time.seconds`` if set — total simulated seconds (already
        evaluated from string expression if any). **Preferred for new
        yamls** because it expresses intent in physical units and
        composes with the ULP-trap detector to catch silly mistakes.
    context:
        Free-text label included in trap error messages.
    """
    if (
        dt_seconds is None
        or not math.isfinite(float(dt_seconds))
        or dt_seconds <= 0.0
    ):
        raise ValueError(f"dt_seconds must be > 0, got {dt_seconds!r}")
    if steps_override is not None:
        if steps_override <= 0:
            raise ValueError(
                f"time.steps must be a positive int, got {steps_override!r}"
            )
        return int(steps_override)
    if seconds_override is not None:
        if (
            not math.isfinite(float(seconds_override))
            or seconds_override <= 0
        ):
            raise ValueError(
                f"time.seconds must be > 0, got {seconds_override!r}"
            )
        raw = float(seconds_override) / float(dt_seconds)
        source_label = "time.seconds"
        source_value = seconds_override
    elif days is not None:
        if not math.isfinite(float(days)) or days <= 0.0:
            raise ValueError(
                f"time.days must be > 0 when steps/seconds are unset, got {days!r}"
            )
        raw = float(days) * SECONDS_PER_DAY / float(dt_seconds)
        source_label = "time.days * 86400"
        source_value = days
    else:
        raise ValueError(
            "must provide one of {time.steps, time.seconds, time.days}"
        )
    n_round = int(round(raw))
    residual = raw - float(n_round)
    fix_hint = (
        "Fix by either:\n"
        "  (a) `time.steps: <int>` — exact integer count (most precise), or\n"
        "  (b) `time.seconds: <int|float|\"expr\">` — total seconds, "
        "supports arithmetic expressions like `\"5 * 86400\"` "
        "(preferred for new yamls), or\n"
        "  (c) widening the `time.days` literal to enough decimals that "
        "it resolves to an integer step count (e.g. "
        "`0.0008680555555556` instead of "
        "`0.0008680556`).\n"
    )
    if not math.isclose(
        raw,
        float(n_round),
        rel_tol=0.0,
        abs_tol=_STEP_RATIO_ABS_TOL,
    ):
        n_ceil = int(math.ceil(raw))
        n_floor = int(math.floor(raw))
        raise StepCountTrapError(
            f"yaml `{source_label}` is not an integer number of time "
            "steps — refusing nearest-step rounding.\n"
            f"  source value: {source_value!r}\n"
            f"  dt           : {dt_seconds!r} s\n"
            f"  raw          : {raw!r}\n"
            f"  round        : {n_round}\n"
            f"  ceil         : {n_ceil}\n"
            f"  floor        : {n_floor}\n"
            f"  residual     : {residual!r} steps\n"
            f"  residual_sec : {residual * float(dt_seconds)!r} s\n"
            f"  tolerance    : ±{_STEP_RATIO_ABS_TOL!r} steps\n"
            f"{fix_hint}"
            f"  context      : {context or '(unspecified)'}"
        )
    if n_round <= 0:
        raise ValueError(
            f"resolved n_steps must be > 0, got {n_round} "
            f"(source={source_label}={source_value!r}, "
            f"dt={dt_seconds!r}, context={context!r})"
        )
    return n_round


def exact_total_seconds(n_steps: int, dt_seconds: float) -> float:
    """Return ``n_steps * dt_seconds`` as the canonical run horizon.

    Used to set MIT's ``endTime`` so the value is an exact integer
    multiple of ``dt`` and never re-derived from a possibly-imprecise
    ``days`` literal.
    """
    if n_steps <= 0:
        raise ValueError(f"n_steps must be > 0, got {n_steps!r}")
    if dt_seconds <= 0.0:
        raise ValueError(f"dt_seconds must be > 0, got {dt_seconds!r}")
    return float(n_steps) * float(dt_seconds)


def resolve_steps(
    days: float | None = None,
    dt_seconds: float | None = None,
    steps_override: int | None = None,
    *,
    seconds_override: float | None = None,
    context: str = "",
) -> StepResolution:
    """Convenience wrapper returning the full :class:`StepResolution`."""
    n_steps = compute_n_steps(
        days=days,
        dt_seconds=dt_seconds,
        steps_override=steps_override,
        seconds_override=seconds_override,
        context=context,
    )
    if steps_override is not None:
        source = "steps_override"
    elif seconds_override is not None:
        source = "seconds_round"
    else:
        source = "days_round"
    return StepResolution(
        n_steps=n_steps,
        dt_seconds=float(dt_seconds),
        total_seconds_exact=exact_total_seconds(n_steps, dt_seconds),
        source=source,
    )
