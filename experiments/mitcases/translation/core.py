"""Translator base: constants, exceptions, dispositions and statuses, the case overrides (CAMPAIGN_DISABLED), the mapping record and the assembly state."""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass, field, fields
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Callable

import yaml

from experiments.cases.scs_clm.mit_native_io import parse_namelist
from experiments.mitcases.g2_compare import ARTIFACT_SCHEMA, COMPARATOR_SCHEMA
from experiments.mitcases.code_inventory import (
    DEFER,
    DISABLED,
    INTERNAL_WAVE_OBCS_CALC,
    KPP_CPP_TARGETS,
    MAP,
    NEGATIVE_SALINITY_CHECK_TARGET,
    REVIEW,
    SKIP,
    UNSUPPORTED,
    Run,
    compiled_packages,
    default_header,
    evaluate_header,
    inventory_case,
)
from locomgcm.constants import ROTATION_PERIOD_EARTH
from locomgcm.dycore_parameters import resolve_3d_coriolis_controls
from locomgcm.dynamics.bulk_formulae import CEN2KEL, EXF_BULK_DEFAULTS
from locomgcm.dynamics.equation_of_state import EOS_TYPES
from locomgcm.dynamics.kpp_mixing import KPP_STOCK_PARAMS
from locomgcm.dynamics.gmredi.calc_tensor import GM_TAPER_SCHEMES
from locomgcm.dynamics.era5_wind_forcing import WINDSTRESSMAX_EXF_DEFAULT
from locomgcm.dynamics.implicit_vertical_advection import IMPLICIT_VERTICAL_TRACER_SCHEMES
from locomgcm.dynamics._registry import DYNAMICS_MODULES
from locomgcm.io.runner._canonical_helpers import ASSEMBLY_SOURCE_TYPES, _EXF_BULK_FIELDS, _EXF_FLUX_FIELDS
from locomgcm.io.exf_source import EXF_OPTIONS_UNDEFINED
from locomgcm.io.runner.layer_b import DATA_SOURCES
from locomgcm.dynamics.cd_scheme import CD_CODE_OPTIONS, CD_MOMENTUM_FORMS, CD_STEPPING_MODES
from locomgcm.dynamics.coriolis import FLUX_CORI_SCHEMES
from locomgcm.boundaries.open import OpenBoundaryValueScheme
from locomgcm.dynamics.vector_invariant_momentum import CORI_SCHEMES, KE_SCHEMES, VORT_SCHEMES
from locomgcm.steppers._registry import STEPPERS
from locomgcm.steppers.ifs.columns import VERTICAL_MIXING_SCHEMES
from locomgcm.dynamics.vertical_mixing_schemes.pp81 import PP81_CPP_OPTIONS
from locomgcm.dynamics.vertical_mixing_schemes.my82 import MY82_CPP_OPTIONS
from locomgcm.dynamics.vertical_mixing_schemes.kl10 import KL10_CPP_OPTIONS
from locomgcm.dynamics.vertical_mixing_schemes.ggl90 import GGL90_CPP_OPTIONS
from locomgcm.io.runner.config._schema import AdvectionConfig
from locomgcm.io.runner.builders.from_manifest import ManifestBuildError, _refuse_unread_center_spacing
from locomgcm.geometry.vertical_coordinates import layer_thickness_from_center_spacing
from locomgcm.steppers.ifs.admission import _refuse_stepping_switches_unimplemented
from locomgcm.operators.fluxes.tracer_flux import (
    HORIZONTAL_TRACER_SCHEMES,
    TRACER_ADVECTION_SCHEMES,
    TRACER_SCHEME_BY_NAME,
    VERTICAL_TRACER_SCHEMES,
)

#: Names of the generated stage1a sources that read the case's own binaries.
_IC_SOURCE_NAME = "mit_input"
_MESH_SOURCE_NAME = "mit_mesh"
_FORCING_SOURCE_NAME = "mit_forcing"
_OBCS_SOURCE_NAME = "mit_obcs"

#: The registered Layer B plugin behind ``_FORCING_SOURCE_NAME``.  The runtime
#: instantiates the forcing source by *plugin* name
#: (``experiments/cases/_runtime.py:609``), so this is also the value that goes
#: into ``case.extras.forcing_source``.
_FORCING_SOURCE_TYPE = "mit_input_binary_forcing"
_OBCS_SOURCE_TYPE = "mit_obcs_binary"

#: The thetaClimFile/saltClimFile source of surface_climate_relaxation, which
#: the run attaches by type (_canonical_helpers.surface_climatology_source).
_CLIMATOLOGY_SOURCE_NAME = "surface_climatology"
_CLIMATOLOGY_SOURCE_TYPE = "mit_surface_climatology"

#: The pkg/exf source: mit_exf under ALLOW_ATM_TEMP (bulk formulae), mit_exf_flux
#: without it (fluxes and stress read from files); the run attaches it by type
#: (_canonical_helpers.exf_bulk_source, exf_flux_sources).
_EXF_SOURCE_NAME = "exf"

#: stage1a's canonical builder always wants a boundary bundle
#: (model_input.py:263), so the lane is emitted for every case and carries the
#: initial state's own edge strip.  Which of those edges is actually *open* is
#: resolved from ``data.obcs`` only when the selected OBCS path is authorised.
_OBCS_SIDES = ("N", "E", "S", "W")
_OBCS_INDEX_KEYS = {
    "ob_jnorth": "N",
    "ob_ieast": "E",
    "ob_jsouth": "S",
    "ob_iwest": "W",
}
_OBCS_SIDE_NAMES = {"N": "north", "E": "east", "S": "south", "W": "west"}
_OBCS_FILE_ROLES = {
    "u": "u",
    "v": "v",
    "w": "w",
    "t": "temperature",
    "s": "salinity",
    "eta": "eta",
}
_OBCS_CORE_FILE_KEYS = {
    f"ob{side.lower()}{suffix}file": (side, role)
    for side in _OBCS_SIDES
    for suffix, role in _OBCS_FILE_ROLES.items()
}

#: Authorisations for case-private analytic OBCS overrides.  Each row names a
#: repository copy of the authorised upstream ``code/obcs_calc.F``; a case's own
#: file selects the row whose copy it equals byte for byte, never a case name.
#: The copy of ``internal_wave/code/obcs_calc.F`` equals the verification tree at
#: MITgcm pin 27256b2d6ae1005c94d4ba4253f5cff6a5df4e5c (the copy was taken at
#: 242228367a50, where the file is the same). Each row
#: declares the plugin and arguments used when that exact source is present.
ANALYTIC_OBCS: tuple[dict[str, Any], ...] = (
    {
        "authorized_source": INTERNAL_WAVE_OBCS_CALC,
        "plugin": "internal_wave_open",
        "kwargs": {"enforce_obcs_balance": False},
        "tref_kwarg": "Tref",
        # The boundary salinity of each side, as the authorised file leaves it: OBEs=sRef(K)
        # (obcs_calc.F:95); OBWs is never assigned, so it keeps OBCS_INIT_VARIABLES's 0
        # (pkg/obcs/obcs_init_variables.F:136) and OBCS_APPLY_TS writes that 0 into the
        # west boundary cells (obcs_apply_ts.F:206).
        "salt_kwargs": {"east_salt": "sRef", "west_salt": "zero"},
        "ruling": "Analytic OBCS source matches the registered MITgcm reference",
        "overrides": "case-private obcs_calc override",
    },
)

_ANALYTIC_OBCS_MISMATCH = (
    "case-private obcs_calc override (content differs from every authorised copy)"
)

# Conversion functions may return this internal key to annotate their mapping
# row without leaking bookkeeping into the emitted yaml.
_RECORD_METADATA = "<record-metadata>"

#: Staging namespace for the model assembly.  Rules write here,
#: :func:`_build_model_assembly` consumes it, and it is removed before the case
#: yaml is written -- it is a translator-internal channel, not a schema block.
_ASSEMBLY = "_assembly"

STATUS_OK = "OK"
STATUS_DEGRADED = "DEGRADED"
STATUS_UNSUPPORTED = "UNSUPPORTED"
STATUS_ERROR = "ERROR"
STATUS_NO_INPUT = "NO_INPUT"

#: Value of ``registry['gates']['reference_source']``.  ``upstream_shipped``
#: means the case's ``results/output.txt`` was produced by the very
#: configuration this translation describes; ``local_mit_rerun`` means at least
#: one switch was explicitly disabled, so the shipped reference belongs to a
#: *different* configuration and a local MITgcm re-run owes the reference.
REFERENCE_UPSTREAM = "upstream_shipped"
REFERENCE_LOCAL_RERUN = "local_mit_rerun"

#: Anchor recorded verbatim on every explicitly disabled row.
CAMPAIGN_RULING = "disabled_by_case_scope"


class TranslationError(RuntimeError):
    """Base class for fail-closed translation refusals."""


class InvalidParameter(TranslationError):
    """A known namelist parameter carries a value that cannot be interpreted."""


class UnknownParameter(TranslationError):
    """A namelist parameter has no rule: refuse to guess, refuse to emit."""

    def __init__(self, parameter: str, source: str) -> None:
        super().__init__(
            f"{source}: no mapping rule for {parameter!r}; add a row to "
            "experiments/mitcases/translate.py (map / skip / defer / unsupported) "
            "rather than letting it pass silently"
        )
        self.parameter = parameter
        self.source = source


class _Redirect(Exception):
    """Value-dependent disposition raised from a rule's conversion function."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class Skipped(_Redirect):
    """This value makes the parameter a no-op for LOCOMgcm."""


class Deferred(_Redirect):
    """LOCOMgcm can express this, but the mapping belongs to a later wave."""


class Gap(_Redirect):
    """LOCOMgcm cannot express this: a capability gap, priced not implemented."""


# --------------------------------------------------------------------------
# Explicit case overrides: switches disabled in both compared models
# --------------------------------------------------------------------------
#
# Each entry declares a mechanism that may be disabled in both LOCOMgcm and
# its MITgcm reference. A compiled package may remain unused by the selected
# case; a dycore mechanism such as useCDscheme is handled by the same rule.
# The table does not authorise changing the equation of state, restart state,
# grid definition or advection discretisation. Those choices define the case.
#
# An override applies only when the parameter token and exact capability-gap
# text match its entry. Applying it changes the reference configuration, so
# the generated registry records reference_source=local_mit_rerun. The shipped
# results/output.txt belongs to the original configuration and cannot supply
# reference values for the modified case.


@dataclass(frozen=True)
class Ruling:
    """One declared override that translates an MITgcm switch as *disabled*.

    ``gap`` is the exact capability-gap text this authorisation overrides.  It
    is matched verbatim: if the rule row's wording ever changes, the
    authorisation stops applying and the case falls back to ``unsupported``
    rather than silently disabling something else.
    """

    anchor: str
    gap: str
    note: str


#: Keyed by the lowercase namelist parameter token (never by case name).
CAMPAIGN_DISABLED: dict[str, Ruling] = {
    # ---- Compiled packages that the selected case explicitly disables -------
    "usegmredi": Ruling(
        "case scope: package off",
        "GM/Redi eddy parameterisation (out of scope)",
        "GM/Redi switched off in both models; the resolved-flow case remains",
    ),
    "useseaice": Ruling(
        "case scope: package off",
        "dynamic sea ice (out of scope)",
        "dynamic sea ice switched off in both models",
    ),
    "usethsice": Ruling(
        "case scope: package off",
        "thermodynamic sea ice (out of scope)",
        "thermodynamic sea ice switched off in both models",
    ),
    "useptracers": Ruling(
        "case scope: package off",
        "passive tracers (pkg/ptracers)",
        "passive tracers carried no feedback into the momentum/T/S state; the "
        "channel is dropped in both models instead of being renamed onto 'temp'",
    ),
    "useprofiles": Ruling(
        "case scope: package off",
        "profile cost (pkg/profiles)",
        "observation-cost package switched off in both models",
    ),
    "usesbo": Ruling(
        "case scope: package off",
        "solid body / ocean-bottom diagnostics (pkg/sbo)",
        "diagnostic package switched off in both models",
    ),
    # ---- Explicit override one level below the package layer ----------------
    "usecdscheme": Ruling(
        "case scope: explicit override",
        "C-D scheme momentum time stepping",
        "C-D grid Coriolis time stepping switched off in both models; the "
        "companion tauCD is already a no-op row governed by this switch",
    ),
    # ---- Explicitly disabled physical mechanisms ----------------------------
    "ivdc_kappa": Ruling(
        "case scope: mechanism off",
        "convective adjustment via implicit vertical diffusion (ivdc_kappa)",
        "implicit-diffusion convective adjustment switched off in both models",
    ),
    "cadjfreq": Ruling(
        "case scope: mechanism off",
        "explicit convective adjustment (cAdjFreq)",
        "explicit convective adjustment switched off in both models",
    ),
    "tauthetaclimrelax": Ruling(
        "case scope: mechanism off",
        "surface T climatology relaxation (tauThetaClimRelax)",
        "surface temperature restoring switched off in both models",
    ),
    "tausaltclimrelax": Ruling(
        "case scope: mechanism off",
        "surface S climatology relaxation (tauSaltClimRelax)",
        "surface salinity restoring switched off in both models",
    ),
    "usenhmterms": Ruling(
        "case scope: mechanism off",
        "non-hydrostatic metric terms",
        "non-hydrostatic metric terms switched off in both models",
    ),
    "quasihydrostatic": Ruling(
        "case scope: mechanism off",
        "quasi-hydrostatic approximation",
        "the quasi-hydrostatic Coriolis terms are switched off in both models",
    ),
}


_MISSING = object()


@dataclass
class Record:
    """The mapping record under construction (registry block 2).

    ``verbatim``: the translation must be the namelist MITgcm ran, so a deferred
    row is recorded as ``unsupported`` (the translation of that key is missing) and
    :data:`CAMPAIGN_DISABLED` does not apply (:func:`_record_gap`).
    """

    parameters: list[dict[str, Any]] = field(default_factory=list)
    binary_inputs: list[dict[str, Any]] = field(default_factory=list)
    derived: list[dict[str, Any]] = field(default_factory=list)
    silent_divergences: list[dict[str, Any]] = field(default_factory=list)
    gaps: list[dict[str, str]] = field(default_factory=list)
    disabled: list[dict[str, str]] = field(default_factory=list)
    verbatim: bool = False

    def add(self, **row: Any) -> None:
        if self.verbatim and row["disposition"] == DEFER:
            reason = row.pop("reason")
            row.update(disposition=UNSUPPORTED, gap=f"deferred translation: {reason}")
        self.parameters.append(row)
        if row["disposition"] == UNSUPPORTED:
            self.gaps.append({"gap": row.get("gap", ""), "source": row["source"]})
        elif row["disposition"] == DISABLED:
            self.disabled.append(
                {
                    "switch": row["parameter"],
                    "source": row["source"],
                    "ruling": row.get("ruling", ""),
                    "effect": row.get("reason", ""),
                }
            )

    def counts(self) -> dict[str, int]:
        out = {MAP: 0, SKIP: 0, DEFER: 0, UNSUPPORTED: 0, DISABLED: 0}
        for row in self.parameters:
            out[row["disposition"]] += 1
        return out

    @property
    def reference_source(self) -> str:
        """Which MITgcm run owns this case's reference values.

        A explicitly disabled switch means the translated configuration is not
        the upstream one, so the shipped ``results/output.txt`` describes a
        different model and therefore requires a local reference re-run.
        """
        return REFERENCE_LOCAL_RERUN if self.disabled else REFERENCE_UPSTREAM


def campaign_ruling(parameter: str, gap: str) -> Ruling | None:
    """The declared override for this parameter, if the gap text matches.

    ``parameter`` is the record's dotted name (``PARM01.useCDscheme``) or a bare
    ``data.pkg`` flag; the table is keyed by the lowercase token alone, so no
    case name can ever reach it.  The gap text is compared verbatim: a rule
    whose wording drifted away from the table keeps its ``unsupported``
    disposition instead of picking up an authorisation meant for something else.
    """
    ruling = CAMPAIGN_DISABLED.get(_lookup_key(parameter.rsplit(".", 1)[-1]))
    return ruling if ruling is not None and ruling.gap == gap else None


def _set_dotted(tree: dict[str, Any], dotted: str, value: Any) -> None:
    node = tree
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _dtype_of(prec: Any) -> str:
    """readBinaryPrec bits -> numpy big-endian dtype string (docs/VALIDATION.md)."""
    return ">f8" if int(prec or 32) == 64 else ">f4"


def _lookup_key(key: str) -> str:
    """Rule-table key: lowercase, array subscripts stripped."""
    key = key.lower()
    return key.split("(", 1)[0] if "(" in key else key


def _record_gap(record: Record, row: dict[str, Any], reason: str) -> None:
    """Record a capability gap unless a declared override disables its mechanism.

    This is the single place where :data:`CAMPAIGN_DISABLED` is consulted, so a
    ``disabled`` row can only ever be produced by a rule that *would otherwise
    have been a gap*: the authorisation can suppress a refusal, never invent a
    mapping.  A verbatim record keeps the gap.
    """
    ruling = None if record.verbatim else campaign_ruling(str(row["parameter"]), reason)
    if ruling is None:
        record.add(**row, disposition=UNSUPPORTED, gap=reason)
        return
    record.add(
        **row,
        disposition=DISABLED,
        ruling=ruling.anchor,
        reason=f"{CAMPAIGN_RULING}: {ruling.note}",
        overrides_gap=reason,
    )

#: MITgcm source-coded defaults the model assembly needs when the namelist is
#: silent.  Every entry carries its source line: the assembly must never invent
#: a value, and a reviewer reads the citation, not this module.
_MIT_ASSEMBLY_DEFAULTS: dict[str, tuple[Any, str]] = {
    "xg_origin": (
        0.0,
        "model/src/set_defaults.F:72 (UNSET_RL); "
        "model/src/ini_parms.F:1391 (resolved to 0.)",
    ),
    "yg_origin": (
        0.0,
        "model/src/set_defaults.F:73 (UNSET_RL); "
        "model/src/ini_parms.F:1392-1400 (resolved to 0.)",
    ),
    "mom_stepping": (True, "model/src/set_defaults.F:192"),
    "mom_advection": (True, "model/src/set_defaults.F:187"),
    "mom_viscosity": (True, "model/src/set_defaults.F:186"),
    "implicit_vertical_diffusion": (False, "model/src/set_defaults.F:209"),
    "implicit_vertical_viscosity": (False, "model/src/set_defaults.F:210"),
    "use_coriolis": (True, "model/src/set_defaults.F:190"),
    "vector_invariant": (False, "model/src/set_defaults.F:193"),
    "temp_stepping": (True, "model/src/set_defaults.F:194"),
    "temp_advection": (True, "model/src/set_defaults.F:195"),
    "salt_stepping": (True, "model/src/set_defaults.F:198"),
    "salt_advection": (True, "model/src/set_defaults.F:199"),
    "no_slip_bottom": (True, "model/src/set_defaults.F:133"),
    "no_slip_sides": (True, "model/src/set_defaults.F:132"),
    "forcing_in_ab": (True, "model/src/ini_parms.F:969"),
    "bottom_drag_linear": (0.0, "model/src/set_defaults.F:136"),
    "bottom_drag_quadratic": (0.0, "model/src/set_defaults.F:137"),
    "f0": (1.0e-4, "model/src/set_defaults.F:111"),
    "beta": (1.0e-11, "model/src/set_defaults.F:112"),
    "t_alpha": (2.0e-4, "model/src/ini_eos.F:85"),
    "s_beta": (7.4e-4, "model/src/ini_eos.F:86"),
    # rhoNil; an unset rhoConst (set_defaults.F:105) is rhoNil (ini_parms.F:483),
    # resolved in _build_model_assembly.
    "rho_nil": (999.8, "model/src/set_defaults.F:104"),
    "heat_capacity": (3994.0, "model/src/set_defaults.F:174"),
    "salt_evprn": (0.0, "model/src/set_defaults.F:265"),
    "periodic_external_forcing": (False, "model/src/set_defaults.F:331"),
    "extern_forcing_period": (0.0, "model/src/set_defaults.F:332"),
    "extern_forcing_cycle": (0.0, "model/src/set_defaults.F:333"),
    "use_obcs_prescribe": (False, "pkg/obcs/obcs_readparms.F:267"),
    "use_obcs_sponge": (False, "pkg/obcs/obcs_readparms.F:269"),
    "use_obcs_balance": (False, "pkg/obcs/obcs_readparms.F:283"),
    "obcs_u1_adv_t": (0, "pkg/obcs/obcs_readparms.F:289"),
    "obcs_u1_adv_s": (0, "pkg/obcs/obcs_readparms.F:290"),
    "implicit_surf_press": (1.0, "model/src/set_defaults.F:252"),
    "implicit_div_2d_flow": (1.0, "model/src/set_defaults.F:253"),
    "exact_conserv": (False, "model/src/set_defaults.F:254"),
    "upwind_vorticity": (False, "model/src/set_defaults.F:235"),
    "high_order_vorticity": (False, "model/src/set_defaults.F:236"),
    "sadourny_coriolis": (False, "model/src/ini_parms.F:344"),
    "ab_eps": (0.01, "model/src/set_defaults.F:317"),
    "check_ini_temp": (True, "model/src/set_defaults.F:368"),
    "allow_freezing": (False, "model/src/set_defaults.F:220"),
    "do_ab_on_gt_gs": (True, "model/src/set_defaults.F:316"),
    "kpp_ricr": (0.3, "pkg/kpp/kpp_readparms.F:112"),
    "kpp_min_hbl": (
        None,
        "pkg/kpp/kpp_readparms.F:88; derived by pkg/kpp/kpp_init_fixed.F:162-163",
    ),
    "gm_adv_form": (False, "pkg/gmredi/gmredi_readparms.F:110"),
    "gm_background_k": (0.0, "pkg/gmredi/gmredi_readparms.F:114"),
    "gm_max_slope": (1.0e-2, "pkg/gmredi/gmredi_readparms.F:115"),
    "gm_kmin_horiz": (0.0, "pkg/gmredi/gmredi_readparms.F:116"),
    "gm_small_number": (1.0e-20, "pkg/gmredi/gmredi_readparms.F:117"),
    "gm_slope_sq_cutoff": (1.0e48, "pkg/gmredi/gmredi_readparms.F:118"),
    # ' ', which Fortran compares equal to '' (GM_TAPER_SCHEMES' no-taper name)
    "gm_taper_scheme": ("", "pkg/gmredi/gmredi_readparms.F:119"),
    "gm_s_crit": (0.004, "pkg/gmredi/gmredi_readparms.F:123"),
    "gm_s_d": (0.001, "pkg/gmredi/gmredi_readparms.F:124"),
    "gm_iso_fac_calc_k": (1.0, "pkg/gmredi/gmredi_readparms.F:125"),
    "gm_visbeck_alpha": (0.0, "pkg/gmredi/gmredi_readparms.F:151"),
    "gm_visbeck_length": (200.0e3, "pkg/gmredi/gmredi_readparms.F:152"),
    "gm_visbeck_depth": (1000.0, "pkg/gmredi/gmredi_readparms.F:153"),
    "gm_visbeck_min_depth": (0.0, "pkg/gmredi/gmredi_readparms.F:154"),
    "gm_visbeck_min_val_k": (0.0, "pkg/gmredi/gmredi_readparms.F:156"),
    "gm_visbeck_max_val_k": (2500.0, "pkg/gmredi/gmredi_readparms.F:157"),
    "pp81_n_ri": (2, "pkg/pp81/pp81_readparms.F:64"),
    "pp81_visc_max": (1.0, "pkg/pp81/pp81_readparms.F:67"),
    "pp81_nu0": (1.0e-2, "pkg/pp81/pp81_readparms.F:68"),
    "pp81_alpha": (5.0, "pkg/pp81/pp81_readparms.F:69"),
    "my82_visc_max": (1.0, "pkg/my82/my82_readparms.F:61"),
    "my82_diff_max": (1.0, "pkg/my82/my82_readparms.F:62"),
    "my82_hbl_scale": (0.1, "pkg/my82/my82_readparms.F:63"),
    "kl10_visc_max": (300.0, "pkg/kl10/kl10_readparms.F:61"),
    "ggl90_ck": (0.1, "pkg/ggl90/ggl90_readparms.F:108"),
    "ggl90_ceps": (0.7, "pkg/ggl90/ggl90_readparms.F:109"),
    "ggl90_alpha": (1.0, "pkg/ggl90/ggl90_readparms.F:110"),
    "ggl90_m2": (3.75, "pkg/ggl90/ggl90_readparms.F:112"),
    "ggl90_tke_min": (1.0e-11, "pkg/ggl90/ggl90_readparms.F:113"),
    "ggl90_tke_surf_min": (1.0e-4, "pkg/ggl90/ggl90_readparms.F:115"),
    "ggl90_visc_max": (100.0, "pkg/ggl90/ggl90_readparms.F:117"),
    "ggl90_diff_max": (100.0, "pkg/ggl90/ggl90_readparms.F:118"),
    "ggl90_mixing_length_min": (1.0e-8, "pkg/ggl90/ggl90_readparms.F:120"),
    "ggl90_mxl_max_flag": (0, "pkg/ggl90/ggl90_readparms.F:121"),
    "ggl90_mxl_surf_flag": (False, "pkg/ggl90/ggl90_readparms.F:123"),
    "ggl90_dirichlet": (True, "pkg/ggl90/ggl90_readparms.F:125"),
    "ggl90_calc_mean_vert_shear": (False, "pkg/ggl90/ggl90_readparms.F:126"),
    "ggl90_use_langmuir": (False, "pkg/ggl90/ggl90_readparms.F:128"),
    "ggl90_lc_gamma": (10.0, "pkg/ggl90/ggl90_readparms.F:159"),
    "ggl90_lc_num": (0.32, "pkg/ggl90/ggl90_readparms.F:160"),
    "ggl90_lc_lambda": (40.0, "pkg/ggl90/ggl90_readparms.F:161"),
}


class _Assembly:
    """Read-side view over the staged values plus the cited MITgcm defaults."""

    def __init__(self, staged: dict[str, Any], yaml_tree: dict[str, Any]) -> None:
        self._staged = staged
        self._tree = yaml_tree
        self.defaults_used: dict[str, str] = {}

    def get(self, key: str) -> Any:
        if key in self._staged:
            return self._staged[key]
        value, source = _MIT_ASSEMBLY_DEFAULTS[key]
        self.defaults_used[key] = f"{value!r} ({source})"
        return value

    def staged(self, key: str, default: Any = None) -> Any:
        """A staged value with no MITgcm default behind it (e.g. a file name)."""
        return self._staged.get(key, default)

    def flag(self, key: str) -> bool:
        return bool(self.get(key))

    def number(self, key: str) -> float:
        return float(self.get(key))

    def yaml(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._tree
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def _entry(name: str, **kwargs: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {"name": name}
    if kwargs:
        entry["kwargs"] = kwargs
    return entry
