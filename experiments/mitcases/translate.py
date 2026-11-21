"""Translate an MITgcm verification case directory into a LOCOMgcm case yaml.

Translation rules
-----------------------------------------------------
**One generic translator plus one shared mapping table.**  There is
deliberately no per-case branching anywhere in this module: no ``if case ==
...``, no case-name lookup, no per-case fixup.  Every case-to-case difference
is expressed as *data* -- a row in :data:`DATA_RULES` / :data:`EEDATA_RULES` /
:data:`PACKAGE_RULES`, an authorised source copy in :data:`ANALYTIC_OBCS`, a whitelist
reason, or an ``unsupported`` gap that lands
in the generated registry record.  The only structural branching allowed is at
*grid-family* granularity (:func:`grid_family`), and the family is decided by
the MITgcm ``using*Grid`` switches, never by a case name.

Fail-closed: a namelist parameter that has no rule raises
:class:`UnknownParameter` and no yaml is produced.  Wide admission is allowed
only through explicit rows -- ``skip`` (whitelisted no-op, with a reason),
``defer`` (LOCOMgcm can express it but the translation is unavailable) and ``unsupported`` (capability gap).  Every row lands in the mapping
record, so "wide" is never silent.

Dispositions
------------
``map``           translated into the case yaml
``skip``          whitelisted no-op, with a reason
``defer``         LOCOMgcm can express it; the translation is unavailable
``unsupported``   capability gap; the yaml is withheld
``disabled``      the configuration specifies running *both* sides with this
                  mechanism switched off (:data:`CAMPAIGN_DISABLED`).  The
                  translated configuration is then no longer the upstream one,
                  so the registry's gate block flips
                  ``reference_source`` to ``local_mit_rerun``: the shipped
                  upstream ``%MON`` values belong to the *original* config and
                  must never be compared against.

The verbatim mode (``translate_case(..., verbatim=True)``) translates the case
as MITgcm ran it: ``defer`` and ``disabled`` rows become ``unsupported``.

Every file of ``code/``, every changed CPP macro and every compiled package gets
a row with the disposition of :mod:`experiments.mitcases.code_inventory`; an
item no inventory table rules (``review``) is recorded ``defer``.

Statuses
--------
``OK``            every parameter mapped, whitelisted or explicitly disabled; yaml written
``DEGRADED``      some parameters have no translation; yaml written
``UNSUPPORTED``   at least one capability gap; yaml withheld
``ERROR``         unknown parameter (fail-closed) or malformed input
``NO_INPUT``      the case carries no ``input/`` directory (stub / adjoint-only)

Namelist parsing is *not* reimplemented here: it reuses
``experiments.cases.scs_clm.mit_native_io.parse_namelist``.

Each mapped row records its sign and unit conversion formula in the generated registry.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from experiments.cases.scs_clm.mit_native_io import parse_namelist
from experiments.mitcases.code_inventory import (
    DEFER,
    DISABLED,
    MAP,
    SKIP,
    UNSUPPORTED,
    inventory_case,
)
from locomgcm.io.runner._canonical_helpers import _EXF_BULK_FIELDS, _EXF_FLUX_FIELDS

# The translator lives in experiments.mitcases.translation (split by responsibility);
# these names are re-exported for the callers and records that name translate.<name>.
# They are aliases: each module resolves its
# globals in its own namespace, so assigning to an attribute here changes nothing.
from .translation.core import (
    Gap,
    InvalidParameter,
    Record,
    STATUS_DEGRADED,
    STATUS_ERROR,
    STATUS_OK,
    STATUS_UNSUPPORTED,
    Skipped,
    TranslationError,
    UnknownParameter,
    _ASSEMBLY,
)
from .translation.rules import (
    Ctx,
    _skip,
    _use_obcs_rule,
)
from .translation.namelist import (
    _exf_prefix,
    _upper_groups,
)
from .translation.cpp import (
    _exf_macros,
    _reference_mit_root,
    _resolve_penetrating_sw,
    _translate_kpp_cpp_options,
    _wire_negative_salinity_check,
    parse_size_h,
)
from .translation.data_rules import (
    DATA_RULES,
    EEDATA_RULES,
)
from .translation.vmix import (
    GGL90_RULES,
    KL10_RULES,
    MY82_RULES,
    PP81_RULES,
    _vertical_mixing_parameters,
)
from .translation.gmredi import (
    GMREDI_RULES,
    _gmredi_entry,
)
from .translation.obcs import (
    OBCS_RULES,
)
from .translation.exf import (
    EXF_RULES,
    _EXF_CONSTANT_NAMES,
    _wire_exf,
)
from .translation.package_rules import (
    CAL_RULES,
    KPP_RULES,
    PACKAGE_RULES,
)
from .translation.assembly import (
    _build_model_assembly,
    _cd_code_options,
    _convective_adjustment_entry,
    _try_construct,
)
from .translation.api import (
    Result,
    default_verification_root,
    translate_case,
    upstream_pin,
)

# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("case", nargs="?", help="case name under the verification root")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--all", action="store_true", help="translate every case")
    args = parser.parse_args(argv)

    try:
        root = (_reference_mit_root(args.root.expanduser().resolve().parent) / "verification"
                if args.root is not None else default_verification_root())
        if args.root is not None and args.root.expanduser().resolve() != root:
            raise TranslationError("--root must name the verified source's verification directory")
    except TranslationError as error:
        parser.error(str(error))
    if root is None or not root.is_dir():
        parser.error(
            "no MITgcm verification tree found; set MITCASES_VERIFICATION_ROOT "
            "or pass --root"
        )

    if args.all:
        cases = sorted(p for p in root.iterdir() if p.is_dir())
    elif args.case:
        cases = [root / args.case]
    else:
        parser.error("give a case name or --all")

    width = max(len(p.name) for p in cases)
    for case_dir in cases:
        try:
            result = translate_case(case_dir, args.out)
        except UnknownParameter as exc:
            result = Result(
                case_dir.name,
                STATUS_ERROR,
                None,
                None,
                {MAP: 0, SKIP: 0, DEFER: 0, UNSUPPORTED: 0, DISABLED: 0},
                [],
                str(exc),
            )
        counts = result.counts
        print(
            f"{result.case:<{width}}  {result.status:<12} "
            f"map={counts[MAP]:<4} skip={counts[SKIP]:<4} "
            f"defer={counts[DEFER]:<4} gap={counts[UNSUPPORTED]:<4} "
            f"off={counts[DISABLED]:<4} "
            f"{result.headline}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
