"""Inventory of the ``code/`` directory of an MITgcm verification case.

Every file of ``code/`` gets a row, and so does every CPP macro of a
``*OPTIONS.h`` header whose compile-time state differs from the pinned default
header, and every package that ``packages.conf`` compiles or excludes.  Each row
carries a disposition from the explicit tables :data:`FILE_RULINGS`,
:data:`MACRO_RULINGS` and :data:`PACKAGE_RULINGS`; an item with no table row is
``review``.  No ruling depends on a case name: cases differ only by which rows
their files match.

Conventions
-----------
* Paths in a ruling's ``basis`` are relative to the MITgcm root ``M`` (the
  ``mitgcm_root`` of :mod:`defgcm.external_paths`), except those starting with
  ``experiments/``, which are relative to this repository.
* A case header replaces the default header of the same name (genmake2 puts the
  ``-mods`` directories before the standard include directories), so a macro the
  case header leaves out keeps the state of the headers included before it, not
  the state written in the default header.
* Headers are evaluated with the ``ALLOW_<PKG>`` macros of the compiled package
  set defined (genmake2's ``PACKAGES_CONFIG.h``, ``tools/genmake2:2690-2726``).
  Package option headers also see the macros of the effective ``CPP_OPTIONS.h``
  and ``CPP_EEOPTIONS.h``, which they include.  ``#include`` lines are otherwise
  not followed: an included header is inventoried as its own file.
* A ruling may hold only under a run-time :class:`Condition` on the case's
  namelists, ``SIZE.h`` and input files.  :func:`inventory_case` given the
  :class:`Run` of the case records an item whose condition fails as
  ``unsupported``; without one (the command line) the condition is only
  printed.
* The compiled package set follows genmake2: ``packages.conf`` or
  ``default_pkg_list`` (``tools/genmake2:2427-2468``), group expansion from
  ``pkg/pkg_groups`` (``:212-237``), ``-pkg`` exclusions (``:2470-2491``), then
  the ``+`` and ``=`` rules of ``pkg/pkg_depend`` (``:2587-2671``).  NetCDF is
  taken as available, so ``mnc`` stays compiled (``:2511-2567``).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The dispositions of the translator's mapping record (``translate`` imports
#: them), plus ``review`` for an item no table row covers.
MAP = "map"
SKIP = "skip"
DEFER = "defer"
UNSUPPORTED = "unsupported"
DISABLED = "disabled"
REVIEW = "review"

KIND_SIZE = "SIZE.h"
KIND_PACKAGES = "packages.conf"
KIND_OPTIONS = "options"
KIND_HEADER = "header"
KIND_SOURCE = "source"
KIND_OTHER = "other"

#: genmake2 always adds these directories before the packages; pkg_depend rules
#: whose first name is one of them always fire (``tools/genmake2:2346``, ``:2606-2608``).
_STANDARD_DIRS = ("eesupp", "model")


@dataclass(frozen=True)
class Run:
    """What a ruling condition reads: the case's parsed namelists (file -> GROUP ->
    lowercase key -> value), ``SIZE.h`` (``sNx`` ... ``Nr``) and input file names."""

    namelists: dict[str, dict[str, dict[str, Any]]]
    size_h: dict[str, int]
    input_files: frozenset[str]

    def get(self, file: str, group: str, key: str, default: Any) -> Any:
        return self.namelists.get(file, {}).get(group, {}).get(key, default)


@dataclass(frozen=True)
class Condition:
    """The run-time condition a ruling needs, with its MITgcm evidence."""

    text: str
    holds: Callable[[Run], bool]


@dataclass(frozen=True)
class Ruling:
    """What an item corresponds to in LOCOMgcm, and the MITgcm evidence for it."""

    disposition: str
    model: str
    basis: str
    condition: Condition | None = None


@dataclass(frozen=True)
class Item:
    """One row of the inventory of a case's ``code/`` directory."""

    file: str
    item: str
    kind: str
    change: str
    disposition: str
    model: str
    basis: str


# --------------------------------------------------------------------------
# rulings
# --------------------------------------------------------------------------

#: Repository copy of the one authorised case-private ``obcs_calc.F``, the
#: upstream ``verification/internal_wave/code/obcs_calc.F``; ``translate.ANALYTIC_OBCS``
#: selects its boundary plugin.
INTERNAL_WAVE_OBCS_CALC = (
    "experiments/mitcases/references/analytic_obcs/internal_wave_open.obcs_calc.F"
)

#: ``KPP_OPTIONS.h`` macros with a LOCOMgcm stepper key: macro -> dotted yaml target
#: (``translate._translate_kpp_cpp_options`` writes the compile-time state there).
KPP_CPP_TARGETS = {
    "KPP_ESTIMATE_UREF": "model.stepper.kwargs.kpp_estimate_uref",
    "KPP_SMOOTH_SHSQ": "model.stepper.kwargs.kpp_smooth_shsq",
    "KPP_SMOOTH_DBLOC": "model.stepper.kwargs.kpp_smooth_dbloc",
}

#: The stepper key of CPP_OPTIONS.h CHECK_SALINITY_FOR_NEGATIVE_VALUES
#: (``translate._wire_negative_salinity_check`` writes it when the macro is defined).
NEGATIVE_SALINITY_CHECK_TARGET = "model.stepper.kwargs.check_salinity_for_negative_values"

#: ``(file name, authorised copy, ruling)``.  A row with an authorised copy
#: matches only a file whose bytes equal that repository file; the first matching
#: row wins.
FILE_RULINGS: tuple[tuple[str, str | None, Ruling], ...] = (
    ("SIZE.h", None, Ruling(
        MAP, "grid shape and overlaps (translate.parse_size_h reads sNx..Nr)",
        "model/inc/SIZE.h:51-62 PARAMETER block (Nx = sNx*nSx*nPx, :60)")),
    ("SIZE.h_mpi", None, Ruling(
        SKIP, "none: the harness builds single-rank from SIZE.h (casegen mitrun size_layout)",
        "verification/testreport:712-758 links *_mpi files only for MPI runs; SIZE.h_mpi feeds "
        "mk_mpi_size at :1645")),
    ("CPP_EEOPTIONS.h_mpi", None, Ruling(
        SKIP, "none: the harness builds single-rank (casegen mitrun size_layout)",
        "verification/testreport:712-758 links *_mpi files only for MPI runs")),
    ("SIZE.h_eddy", None, Ruling(
        SKIP, "none: never compiled",
        "tools/genmake2 compiles SIZE.h and verification/testreport:723 substitutes only *_mpi "
        "files, so no build includes SIZE.h_eddy")),
    ("DIAGNOSTICS_SIZE.h", None, Ruling(
        SKIP, "none: diagnostics output buffer sizes",
        "pkg/diagnostics/DIAGNOSTICS_SIZE.h:6-23 (numLevels, numDiags, diagSt_size size the "
        "diagnostics storage arrays)")),
    ("MDSIO_BUFF_3D.h", None, Ruling(
        SKIP, "none: binary I/O buffer size",
        "pkg/mdsio/MDSIO_BUFF_3D.h:18-34 (size3dBuf sizes the MDS read/write buffers)")),
    ("obcs_calc.F", INTERNAL_WAVE_OBCS_CALC,
     Ruling(
        MAP, "boundary plugin internal_wave_open (translate.ANALYTIC_OBCS)",
        "replaces pkg/obcs/obcs_calc.F; analytic western boundary of "
        "verification/internal_wave/code/obcs_calc.F:110-150, bytewise equal to the authorised "
        "copy")),
    ("obcs_calc.F", None, Ruling(
        UNSUPPORTED, "none: a private analytic boundary without an authorised copy",
        "replaces pkg/obcs/obcs_calc.F; translate._use_obcs_rule raises Gap for any copy not "
        "listed in translate.ANALYTIC_OBCS")),
    ("mom_vi_hfacz_diss.F", None, Ruling(
        SKIP, "none: never called",
        "the only call site, pkg/mom_vecinv/mom_vecinv.F:282, is commented out")),
    ("mom_vi_mask_vort3.F", None, Ruling(
        SKIP, "none: never called",
        "the only call site, pkg/mom_vecinv/mom_vecinv.F:663, is commented out")),
)

def _cos_factors_one_new_side_drag(run: Run) -> bool:
    spherical = run.get("data", "PARM04", "usingsphericalpolargrid", False) is True
    cos_power = float(run.get("data", "PARM01", "cospower", 0.0))
    return (not spherical or cos_power == 0.0) and float(
        run.get("data", "PARM01", "sidedragfactor", 2.0)) > 0.0


#: The cos(lat) viscosity scaling macros of MOM_COMMON_OPTIONS.h select which of
#: cosFacU/V, sqCosFacU/V multiplies a viscous term, and in the side drag's
#: sideDragFactor <= 0 branch whether the grid-limited Ahtmp/A4tmp is used.
_COS_SCALING = Condition(
    "cosFacU/V and sqCosFacU/V are 1 (no spherical polar grid, or cosPower = 0: "
    "model/src/ini_grid.F:105-108, ini_spherical_polar_grid.F:256-268) and sideDragFactor > 0 "
    "(default 2, model/src/set_defaults.F:135), so the side drag takes its MOM_CALC_VISC branch "
    "(pkg/mom_common/mom_u_sidedrag.F:58, mom_v_sidedrag.F:58)",
    _cos_factors_one_new_side_drag,
)
_COS_SCALING_BASIS = (
    "its uses multiply a viscous term by cosFacU/V or sqCosFacU/V or leave the factor out: "
    "pkg/mom_common/mom_u_sidedrag.F:117-139, mom_v_sidedrag.F:115-124, mom_w_sidedrag.F:83-123, "
    "pkg/mom_fluxform/mom_u_del2u.F:59-61, mom_u_xviscflux.F:62-66, mom_u_yviscflux.F:61-70, "
    "mom_v_del2v.F:58-60, mom_v_xviscflux.F:63-67, mom_v_yviscflux.F:60-69, "
    "model/src/calc_gw.F:315-400; in the sideDragFactor <= 0 branch they also choose "
    "viscAh/viscA4 over the grid-limited Ahtmp/A4tmp (mom_u_sidedrag.F:85-95, "
    "mom_v_sidedrag.F:88-92); LOCOMgcm has no cos(lat) scaling and no such side drag"
)


def _single_periodic_tile(run: Run) -> bool:
    size = run.size_h
    return (size["nSx"] * size["nSy"] * size["nPx"] * size["nPy"] == 1
            and run.get("eedata", "EEPARMS", "notusingxperiodicity", False) is not True
            and run.get("eedata", "EEPARMS", "notusingyperiodicity", False) is not True
            and run.get("data", "PARM01", "nonhydrostatic", False) is not True
            and run.get("data", "PARM01", "implicitintgravwave", False) is not True)


def _exch2_single_facet(run: Run) -> bool:
    return ("data.exch2" not in run.input_files
            and run.get("eedata", "EEPARMS", "usecubedsphereexchange", False) is not True)


def _no_external_forcing(run: Run) -> bool:
    packages = run.namelists.get("data.pkg", {}).get("PACKAGES", {})
    return (all(on is not True for key, on in packages.items() if key not in ("usediagnostics", "usemnc"))
            and run.get("eedata", "EEPARMS", "usecoupler", False) is not True
            and not str(run.get("data", "PARM05", "zonalwindfile", "")).strip()
            and not str(run.get("data", "PARM05", "meridwindfile", "")).strip()
            and run.get("data", "PARM01", "tempstepping", True) is False
            and run.get("data", "PARM01", "saltstepping", True) is False)


def _ptracers_off(run: Run) -> bool:
    return run.get("data.pkg", "PACKAGES", "useptracers", False) is not True


def _admits(keys: str, basis: str) -> Ruling:
    """A macro whose only role is to let run-time ``keys`` be switched on.

    Its target is the run-time key, whose own translator row decides.  A difference
    between "macro on, key off" and "macro off" shows up in the station comparison
    against the MITgcm truth frozen with the case's own build.
    """
    return Ruling(MAP, f"run-time {keys} (the translator row of the key decides)", basis)


def _gmredi_off_or(holds: Callable[[Run], bool]) -> Callable[[Run], bool]:
    """``holds``, or useGMRedi off, where no GMREDI_OPTIONS.h macro reaches the run."""
    return lambda run: run.get("data.pkg", "PACKAGES", "usegmredi", False) is not True or holds(run)


def _switched_off(switch: str, namelist: int, default: int, refusal: str) -> Condition:
    """The data.pkg ``switch`` off: core code calls a package only under its use switch, so a
    macro read only inside the package reaches no run; ``refusal`` names what refuses the macro
    when the package is on."""
    return Condition(
        f"{switch} off (model/src/packages_boot.F:{namelist}, default .FALSE. at :{default}; core "
        f"code calls the package only under it, doc/contributing/contributing.rst:461-467); with it "
        f"on, {refusal}",
        lambda run: run.get("data.pkg", "PACKAGES", switch.lower(), False) is not True)


_EXF_OFF = _switched_off(
    "useEXF", 60, 126, "the EXF sources refuse the macro (defgcm/io/exf_source.py "
    "EXF_OPTIONS_UNDEFINED, checked by _canonical_helpers._exf_config)")


#: The K arrays of GM_USE_K3D_REDI/GM (gmredi_calc_tensor.F:11-19) hold GM_isopycK and
#: GM_background_K times unit factors (:237-260) until a variable K is added to them, so
#: their two-point means 0.5*(K+K) (:664-675, :785-847, :982-1044) equal the constant K.
_GM_K3D_BASIS = (
    "the macro defines GM_USE_K3D_REDI/GM (pkg/gmredi/gmredi_calc_tensor.F:11-19), whose K "
    "arrays hold GM_isopycK, GM_background_K times unit factors (:237-260) and get a variable K "
    "only under the run-time switch, so their two-point means 0.5*(K+K) (:664-675, :785-847, "
    ":982-1044) equal the constant K of the gmredi module"
)


#: ``(header, macro) -> ruling`` for a macro whose state differs from the default
#: header.  The default state is fixed by the pin, so the key also fixes the
#: direction of the change.
MACRO_RULINGS: dict[tuple[str, str], Ruling] = {
    ("CPP_OPTIONS.h", "NONLIN_FRSURF"): _admits(
        "nonlinFreeSurf, select_rStar",
        "model/src/config_check.F:305-315 (nonlinFreeSurf) and :317-327 (select_rStar) stop when "
        "the key is set without the macro"),
    ("CPP_OPTIONS.h", "ALLOW_NONHYDROSTATIC"): _admits(
        "nonHydrostatic, implicitIntGravWave",
        "model/src/config_check.F:163-175 stops on use3Dsolver without the macro; "
        "model/src/set_parms.F:104 sets use3Dsolver = nonHydrostatic .OR. implicitIntGravWave"),
    ("CPP_OPTIONS.h", "ALLOW_ADDFLUID"): _admits(
        "selectAddFluid", "model/src/config_check.F:363-373 stops when selectAddFluid is nonzero"),
    ("CPP_OPTIONS.h", "ALLOW_ADAMSBASHFORTH_3"): _admits(
        "alph_AB, beta_AB", "model/src/config_check.F:189-199 stops when alph_AB or beta_AB is set"),
    ("CPP_OPTIONS.h", "ALLOW_FRICTION_HEATING"): _admits(
        "addFrictionHeating", "model/src/config_check.F:440-450 stops when addFrictionHeating"),
    ("CPP_OPTIONS.h", "ALLOW_SOLVE4_PS_AND_DRAG"): _admits(
        "selectImplicitDrag=2", "model/src/config_check.F:251-261 stops when selectImplicitDrag=2"),
    ("CPP_OPTIONS.h", "ALLOW_BALANCE_FLUXES"): _admits(
        "selectBalanceEmPmR, balanceQnet",
        "model/src/config_check.F:474-492, whose #else branch (:481-491) stops when either is set"),
    ("CPP_OPTIONS.h", "ALLOW_BALANCE_RELAX"): _admits(
        "balanceThetaClimRelax, balanceSaltClimRelax",
        "model/src/config_check.F:494-505 stops when either is set"),
    ("CPP_OPTIONS.h", "ALLOW_QHYD_STAGGER_TS"): _admits(
        "quasiHydrostatic together with staggerTimeStep",
        "model/src/config_check.F:177-187 stops when both are set"),
    ("CPP_OPTIONS.h", "ALLOW_SMAG_3D_DIFFUSIVITY"): _admits(
        "smag3D_diffCoeff", "model/src/config_check.F:151-161 stops when smag3D_diffCoeff is nonzero"),
    ("CPP_OPTIONS.h", "ALLOW_SRCG"): _admits(
        "useSRCGSolver (default .FALSE., model/src/set_defaults.F:297)",
        "model/src/config_check.F:520-531 stops when useSRCGSolver is set without the macro; its "
        "other uses are model/src/solve_for_pressure.F:290-296 inside IF (useSRCGSolver) and "
        "model/src/cg2d_sr.F:64-456"),
    ("MOM_COMMON_OPTIONS.h", "ALLOW_BOTTOMDRAG_ROUGHNESS"): _admits(
        "zRoughBot",
        "model/src/config_check.F:533-543 stops when zRoughBot > 0; read under the macro at "
        "pkg/mom_common/mom_init_fixed.F:178-226"),
    ("MOM_COMMON_OPTIONS.h", "ALLOW_LEITH_QG"): _admits(
        "viscC2LeithQG", "model/src/config_check.F:126-138 stops when viscC2LeithQG is nonzero"),
    ("MOM_COMMON_OPTIONS.h", "ALLOW_SMAG_3D"): _admits(
        "useSmag3D", "model/src/config_check.F:140-150 stops when useSmag3D"),
    ("GAD_OPTIONS.h", "GAD_ALLOW_TS_SOM_ADV"): _admits(
        "tempAdvScheme, saltAdvScheme = 80 or 81",
        "pkg/generic_advdiff/gad_check.F:65-78 stops when either scheme is ENUM_SOM_PRATHER or "
        "ENUM_SOM_LIMITER"),
    ("GMREDI_OPTIONS.h", "GM_VISBECK_VARIABLE_K"): Ruling(
        MAP, "gmredi.visbeck_alpha (translate GMREDI_RULES GM_Visbeck_alpha reads the header state)",
        "pkg/gmredi/GMREDI_OPTIONS.h:36 default #undef; defined, GM_Visbeck_alpha > 0 adds the "
        "Visbeck K (pkg/gmredi/gmredi_calc_tensor.F:151-160,416-503,579-597), which the gmredi "
        "module computes, and alpha = 0 leaves the constant K; gmredi_check.F:166-176 stops on "
        "GM_Visbeck_alpha without the macro"),
    ("GMREDI_OPTIONS.h", "GM_GEOM_VARIABLE_K"): Ruling(
        SKIP, "none while the condition holds: GM_skewflx*GEOM_K3d adds zero",
        "pkg/gmredi/GMREDI_OPTIONS.h:39 default #undef; it adds GM_skewflx*GEOM_K3d "
        "(pkg/gmredi/gmredi_calc_tensor.F:677-679, :831-836, :1028-1033); gmredi_check.F:192-228 "
        "stops on GM_useGEOM without it",
        Condition(
            "useGMRedi off or GM_useGEOM .FALSE., so GEOM_K3d stays zero "
            "(pkg/gmredi/gmredi_init_varia.F:128, gmredi_calc_tensor.F:365-376)",
            _gmredi_off_or(lambda run: run.get(
                "data.gmredi", "GM_PARM01", "gm_usegeom", False) is not True),
        )),
    ("GMREDI_OPTIONS.h", "GM_BOLUS_ADVEC"): Ruling(
        SKIP, "none while the condition holds: its code runs only under GM_AdvForm",
        "pkg/gmredi/GMREDI_OPTIONS.h:56 default #define; the bolus streamfunction and residual "
        "flow it compiles run only under GM_AdvForm (pkg/gmredi/gmredi_calc_tensor.F:609-610, "
        "gmredi_residual_flow.F:58)",
        Condition(
            "useGMRedi off or GM_AdvForm .FALSE. (pkg/gmredi/gmredi_check.F:230-241 stops "
            "otherwise)",
            _gmredi_off_or(lambda run: run.get(
                "data.gmredi", "GM_PARM01", "gm_advform", False) is not True),
        )),
    ("GMREDI_OPTIONS.h", "GM_BOLUS_BVP"): Ruling(
        SKIP, "none: undefined, MITgcm has no BVP streamfunction, as LOCOMgcm has none; the "
        "translator's GM_UseBVP row reads the macro, since MITgcm then ignores GM_UseBVP",
        "pkg/gmredi/GMREDI_OPTIONS.h:59 default #define; it compiles only the GM_UseBVP branch "
        "of the advective-form streamfunction (pkg/gmredi/gmredi_calc_tensor.F:609-631)"),
    ("GMREDI_OPTIONS.h", "GM_EXTRA_DIAGONAL"): Ruling(
        MAP, "gmredi.extra_diagonal (translate._gmredi_entry reads the header state)",
        "pkg/gmredi/GMREDI_OPTIONS.h:52 default #define; it compiles Kuz, Kvz "
        "(gmredi_calc_tensor.F:810-847, :1007-1044) and the off-diagonal x, y fluxes "
        "(gmredi_xtransport.F:142-192), all under GM_ExtraDiag; gmredi_check.F:307-318 stops "
        "when GM_ExtraDiag is set without it"),
    ("GMREDI_OPTIONS.h", "GM_NON_UNITY_DIAGONAL"): Ruling(
        MAP, "gmredi.non_unity_diagonal (translate._gmredi_entry reads the header state)",
        "pkg/gmredi/GMREDI_OPTIONS.h:48 default #define; defined, Kux and Kvy are "
        "GM_isopycK*taperFct floored at GM_Kmin_horiz (gmredi_calc_tensor.F:785-808, "
        ":982-1005), undefined, the constant GM_isopycK (:1093-1118); gmredi_check.F:320-355 "
        "stops when GM_iso2dFile or GM_iso1dFile is set without it"),
    ("GMREDI_OPTIONS.h", "ALLOW_GM_LEITH_QG"): Ruling(
        SKIP, "none while the condition holds: the K arrays are the constant K",
        "pkg/gmredi/GMREDI_OPTIONS.h:62 default #undef; " + _GM_K3D_BASIS
        + "; gmredi_check.F:357-367 stops on GM_useLeithQG without the macro",
        Condition(
            "useGMRedi off, or GM_useLeithQG off after pkg/gmredi/gmredi_readparms.F:234-240 "
            "resets it for viscC2LeithQG = 0, so GM_LeithQG_K is never added "
            "(gmredi_calc_tensor.F:344-357)",
            _gmredi_off_or(lambda run: run.get(
                "data.gmredi", "GM_PARM01", "gm_useleithqg", False) is not True
                or float(run.get("data", "PARM01", "viscc2leithqg", 0.0)) == 0.0),
        )),
    ("OBCS_OPTIONS.h", "ALLOW_OBCS_SOUTH"): _admits(
        "southern open boundary (OB_Jsouth)",
        "pkg/obcs/obcs_check.F:287-297 stops when a tile has a southern open boundary"),
    ("OBCS_OPTIONS.h", "ALLOW_OBCS_EAST"): _admits(
        "eastern open boundary (OB_Ieast)",
        "pkg/obcs/obcs_check.F:298-308 stops when a tile has an eastern open boundary"),
    ("OBCS_OPTIONS.h", "ALLOW_OBCS_WEST"): _admits(
        "western open boundary (OB_Iwest)",
        "pkg/obcs/obcs_check.F:309-319 stops when a tile has a western open boundary"),
    ("OBCS_OPTIONS.h", "ALLOW_OBCS_PRESCRIBE"): _admits(
        "useOBCSprescribe", "pkg/obcs/obcs_check.F:477-487 stops when useOBCSprescribe"),
    ("OBCS_OPTIONS.h", "ALLOW_OBCS_SPONGE"): _admits(
        "useOBCSsponge", "pkg/obcs/obcs_check.F:489-499 stops when useOBCSsponge"),
    ("OBCS_OPTIONS.h", "ALLOW_OBCS_SEAICE_SPONGE"): _admits(
        "useSeaiceSponge", "pkg/obcs/obcs_check.F:501-511 stops when useSeaiceSponge"),
    ("OBCS_OPTIONS.h", "ALLOW_OBCS_BALANCE"): _admits(
        "useOBCSbalance",
        "pkg/obcs/obcs_check.F:513-538, whose #else branch (:528-537) stops when useOBCSbalance"),
    ("OBCS_OPTIONS.h", "ALLOW_OBCS_TIDES"): _admits(
        "useOBCStides", "pkg/obcs/obcs_check.F:540-550 stops when useOBCStides"),
    ("OBCS_OPTIONS.h", "ALLOW_ORLANSKI"): _admits(
        "useOrlanskiNorth, useOrlanskiSouth, useOrlanskiEast, useOrlanskiWest",
        "pkg/obcs/OBCS_OPTIONS.h:19 default #define; pkg/obcs/obcs_check.F:370-384 stops when one "
        "of the keys is set without it, and pkg/obcs/obcs_readparms.F:507-518 reads OBCS_PARM02 "
        "under it only when one is set"),
    ("OBCS_OPTIONS.h", "ALLOW_OBCS_STEVENS"): _admits(
        "useStevensNorth, useStevensSouth, useStevensEast, useStevensWest",
        "pkg/obcs/obcs_calc.F:439-450 calls OBCS_CALC_STEVENS under the macro only when one of "
        "the keys is set; pkg/obcs/obcs_readparms.F:538-544 reads OBCS_PARM04 likewise"),
    ("GGL90_OPTIONS.h", "ALLOW_GGL90_LANGMUIR"): _admits(
        "useLANGMUIR", "pkg/ggl90/ggl90_check.F:65-73 stops when useLANGMUIR"),
    ("GGL90_OPTIONS.h", "ALLOW_GGL90_IDEMIX"): Ruling(
        SKIP, "none while useGGL90 is off",
        "pkg/ggl90/GGL90_OPTIONS.h:25 default #undef; pkg/ggl90/ggl90_readparms.F:187-206 and "
        ":240-248 act on useIDEMIX only under the macro, which also guards the IDEMIX namelist at "
        ":59-69",
        _switched_off("useGGL90", 54, 120, "LOCOMgcm's GGL90 rejects it: it is not in GGL90_CPP_OPTIONS "
                      "(defgcm/dynamics/vertical_mixing_schemes/ggl90.py:40-44), which "
                      "translate._vertical_mixing_parameters reads")),
    ("GGL90_OPTIONS.h", "GGL90_IDEMIX_CVMIX_VERSION"): Ruling(
        SKIP, "none while useGGL90 is off",
        "pkg/ggl90/GGL90_OPTIONS.h:29; read only in pkg/ggl90/ggl90_idemix.F:166, :194, :212, "
        "which pkg/ggl90/ggl90_calc.F:264 calls under useIDEMIX",
        _switched_off("useGGL90", 54, 120, "LOCOMgcm's GGL90 rejects ALLOW_GGL90_IDEMIX "
                      "(defgcm/dynamics/vertical_mixing_schemes/ggl90.py:40-43), so it has no "
                      "IDEMIX for the macro to select")),
    ("PTRACERS_OPTIONS.h", "PTRACERS_ALLOW_DYN_STATE"): _admits(
        "PTRACERS_advScheme = 80 or 81",
        "pkg/ptracers/ptracers_init_fixed.F:93-95 sets PTRACERS_SOM_Advection from the scheme; "
        "pkg/ptracers/ptracers_check.F:232-243 stops when it is set without the macro"),
    ("SEAICE_OPTIONS.h", "SEAICE_ZETA_SMOOTHREG"): Ruling(
        SKIP, "none while useSEAICE is off",
        "pkg/seaice/SEAICE_OPTIONS.h:149 default #define; read only in "
        "pkg/seaice/seaice_calc_viscosities.F:83-85, :365-379",
        _switched_off("useSEAICE", 79, 145, "LOCOMgcm's sea-ice dynamics compile it defined "
                      "(defgcm/seaice/dynamics/types.py:60), another rheology than a build "
                      "that leaves it out")),
    ("SEAICE_OPTIONS.h", "SEAICE_ALLOW_FREEDRIFT"): _admits(
        "SEAICEuseFREEDRIFT", "pkg/seaice/seaice_check.F:1060-1073 stops when SEAICEuseFREEDRIFT"),
    ("SEAICE_OPTIONS.h", "SEAICE_ALLOW_EVP"): _admits(
        "SEAICEuseEVP (SEAICE_deltaTevp)",
        "pkg/seaice/seaice_check.F:849-885, whose #else branch (:872-884) stops when SEAICEuseEVP"),
    ("SEAICE_OPTIONS.h", "SEAICE_ALLOW_JFNK"): _admits(
        "SEAICEuseJFNK",
        "pkg/seaice/seaice_check.F:1087-1126, whose #else branch stops when SEAICEuseJFNK"),
    ("SEAICE_OPTIONS.h", "SEAICE_ALLOW_KRYLOV"): _admits(
        "SEAICEuseKrylov", "pkg/seaice/seaice_check.F:1127-1137 stops when SEAICEuseKrylov"),
    ("SEAICE_OPTIONS.h", "SEAICE_ALLOW_LSR_FLEX"): _admits(
        "SEAICEuseLSRflex", "pkg/seaice/seaice_check.F:920-927 stops when SEAICEuseLSRflex"),
    ("SEAICE_OPTIONS.h", "SEAICE_ALLOW_CLIPVELS"): _admits(
        "SEAICE_clipVelocities",
        "pkg/seaice/seaice_check.F:944-954 stops when SEAICE_clipVelocities"),
    ("SEAICE_OPTIONS.h", "SEAICE_ALLOW_TEM"): _admits(
        "SEAICEuseTEM", "pkg/seaice/seaice_check.F:970-977 stops when SEAICEuseTEM"),
    ("SEAICE_OPTIONS.h", "SEAICE_ALLOW_MCE"): _admits(
        "SEAICEuseMCE", "pkg/seaice/seaice_check.F:978-985 stops when SEAICEuseMCE"),
    ("SEAICE_OPTIONS.h", "SEAICE_ALLOW_MCS"): _admits(
        "SEAICEuseMCS", "pkg/seaice/seaice_check.F:986-993 stops when SEAICEuseMCS"),
    ("SEAICE_OPTIONS.h", "SEAICE_ALLOW_TEARDROP"): _admits(
        "SEAICEuseTD", "pkg/seaice/seaice_check.F:994-1007 stops when SEAICEuseTD"),
    ("SEAICE_OPTIONS.h", "SEAICE_ALLOW_BOTTOMDRAG"): _admits(
        "SEAICEbasalDragK2", "pkg/seaice/seaice_check.F:1213-1223 stops when SEAICEbasalDragK2 > 0"),
    ("SEAICE_OPTIONS.h", "SEAICE_ALLOW_SIDEDRAG"): _admits(
        "SEAICEsideDrag, uCoastLineFile, vCoastLineFile",
        "pkg/seaice/seaice_check.F:1225-1249 stops when any of them is set"),
    ("EXF_OPTIONS.h", "EXF_ALLOW_TIDES"): _admits(
        "tidePotFile", "pkg/exf/exf_check.F:304-311 stops when tidePotFile is set"),
    ("EXF_OPTIONS.h", "ALLOW_RUNOFF"): _admits(
        "runofffile", "pkg/exf/exf_check.F:322-329 stops when runofffile is set"),
    ("EXF_OPTIONS.h", "ALLOW_RUNOFTEMP"): Ruling(
        SKIP, "none while useEXF is off",
        "pkg/exf/EXF_OPTIONS.h:189 default #undef; it adds the runoff temperature field and "
        "its heat flux (pkg/exf/exf_init_varia.F:368-379, exf_getffields.F, exf_mapfields.F); "
        "outside pkg/exf only pkg/autodiff/check_lev1_dir_forcing.h reads it", _EXF_OFF),
    ("EXF_OPTIONS.h", "ALLOW_SALTFLX"): _admits(
        "saltflxfile", "pkg/exf/exf_check.F:346-352 stops when saltflxfile is set"),
    ("EXF_OPTIONS.h", "ALLOW_DOWNWARD_RADIATION"): _admits(
        "swdownfile, lwdownfile",
        "pkg/exf/exf_check.F:263-293, whose #else branch (:280-292) stops when either is set"),
    ("EXF_OPTIONS.h", "ALLOW_ZENITHANGLE"): Ruling(
        SKIP, "none while useEXF is off",
        "pkg/exf/EXF_OPTIONS.h:200 default #undef; it compiles the zenith-angle albedo and "
        "incoming radiation of pkg/exf/exf_radiation.F:113-150 (exf_zenithangle.F); nothing "
        "outside pkg/exf reads it", _EXF_OFF),
    ("EXF_OPTIONS.h", "ALLOW_BULK_LARGEYEAGER04"): Ruling(
        SKIP, "none while useEXF is off",
        "pkg/exf/EXF_OPTIONS.h:176 default #undef; it replaces the transfer coefficients and "
        "stability iteration of pkg/exf/exf_bulkformulae.F by the Large and Yeager (2004) ones; "
        "outside pkg/exf only pkg/thsice/thsice_get_exf.F:326-385 reads it, which "
        "thsice_solve4temp.F:354, :407 call under useEXF", _EXF_OFF),
    ("EXF_OPTIONS.h", "USE_EXF_INTERPOLATION"): Ruling(
        UNSUPPORTED, "none: the EXF sources read model-grid records and refuse it "
        "(defgcm/io/exf_source.py EXF_OPTIONS_UNDEFINED, checked by _canonical_helpers._exf_config)",
        "pkg/exf/EXF_OPTIONS.h:221 default #undef; it reads EXF_NML_04 (pkg/exf/exf_readparms.F:"
        "989-995) and interpolates every field from its input lat-lon grid (exf_interp.F)"),
    ("EXF_OPTIONS.h", "ALLOW_ATM_TEMP"): Ruling(
        MAP, "translate._wire_exf: the mit_exf_flux source (hflux, sflux and the stress read from "
        "files) with wpflux_surface_forcing and era5_wind_forcing",
        "pkg/exf/EXF_OPTIONS.h:169 default #define; undefined, EXF computes no bulk formulae and "
        "takes hflux, sflux and the stress from their files (pkg/exf/exf_getforcing.F:268-307, "
        "exf_mapfields.F:100-284)"),
    ("EXF_OPTIONS.h", "ALLOW_ATM_WIND"): Ruling(
        MAP, "translate._wire_exf: mit_exf_flux when ALLOW_ATM_TEMP is undefined too; under "
        "ALLOW_ATM_TEMP the bulk kernel needs the wind, so _wire_exf records the gap",
        "pkg/exf/EXF_OPTIONS.h:170 default #define; undefined, EXF reads no wind and forms no "
        "stress from it (pkg/exf/exf_wind.F, exf_getforcing.F:266, exf_readparms.F:313-317)"),
    ("EXF_OPTIONS.h", "ALLOW_BULKFORMULAE"): Ruling(
        MAP, "translate._wire_exf: undefined without ALLOW_ATM_TEMP (mit_exf_flux); under "
        "ALLOW_ATM_TEMP the bulk kernel is these formulae, so _wire_exf records the gap",
        "pkg/exf/EXF_OPTIONS.h:172-185 defines it under ALLOW_ATM_TEMP; undefined there, EXF reads "
        "the turbulent fluxes (ALLOW_READ_TURBFLUXES) instead of exf_bulkformulae.F"),
    ("EXF_OPTIONS.h", "EXF_LWDOWN_WITH_EMISSIVITY"): Ruling(
        MAP, "translate._wire_exf: the bulk kernel's longwave flux multiplies lwdown by the "
        "emissivity, so under ALLOW_ATM_TEMP _wire_exf records the gap; without ALLOW_ATM_TEMP "
        "EXF forms no longwave flux",
        "pkg/exf/EXF_OPTIONS.h:205-207 defines it under ALLOW_DOWNWARD_RADIATION; it selects "
        "lwflux = emissivity*(sigma*T**4 - lwdown) in pkg/exf/exf_radiation.F:52-78, reached only "
        "under ALLOW_ATM_TEMP"),
    ("EXF_OPTIONS.h", "EXF_INTERP_USE_DYNALLOC"): Ruling(
        SKIP, "none: storage of the interpolation input buffer",
        "pkg/exf/EXF_OPTIONS.h:229 default #define; it allocates the input buffer of "
        "pkg/exf/exf_interp_read.F:52-98,136-145 on the stack instead of a static array checked "
        "by exf_check.F:714-735, read only under USE_EXF_INTERPOLATION; no value changes"),
    ("CPP_OPTIONS.h", "SHORTWAVE_HEATING"): Ruling(
        MAP, "penetrating shortwave selection (translate._resolve_penetrating_sw)",
        "model/inc/CPP_OPTIONS.h:21 default #undef; model/src/ini_parms.F:653-657 sets "
        "selectPenetratingSW=1 under it"),
    ("KPP_OPTIONS.h", "KPP_ESTIMATE_UREF"): Ruling(
        MAP, KPP_CPP_TARGETS["KPP_ESTIMATE_UREF"],
        "pkg/kpp/KPP_OPTIONS.h:39 default #undef; used at pkg/kpp/kpp_calc.F:438 and "
        "pkg/kpp/kpp_forcing_surf.F:114"),
    ("KPP_OPTIONS.h", "KPP_SMOOTH_SHSQ"): Ruling(
        MAP, KPP_CPP_TARGETS["KPP_SMOOTH_SHSQ"],
        "pkg/kpp/KPP_OPTIONS.h:18 default #define; used at pkg/kpp/kpp_calc.F:486"),
    ("KPP_OPTIONS.h", "KPP_SMOOTH_DBLOC"): Ruling(
        MAP, KPP_CPP_TARGETS["KPP_SMOOTH_DBLOC"],
        "pkg/kpp/KPP_OPTIONS.h:22 default #define; used at pkg/kpp/kpp_calc.F:285"),
    ("CPP_OPTIONS.h", "INCLUDE_CONVECT_INI_CALL"): Ruling(
        MAP, "translate._convective_adjustment_entry: convective_adjustment.initial_adjustment "
        "when cAdjFreq is nonzero and startTime = baseTime",
        "model/inc/CPP_OPTIONS.h:70 default #undef; model/src/initialise_varia.F:280-295 adjusts "
        "the initial tracers once under it (CONVECTIVE_ADJUSTMENT_INI) when startTime = baseTime "
        "and cAdjFreq is nonzero, and does nothing otherwise"),
    ("CPP_OPTIONS.h", "INCLUDE_SOUNDSPEED_CALC_CODE"): Ruling(
        SKIP, "none: sound-speed diagnostic",
        "model/inc/CPP_OPTIONS.h:61; model/src/dynamics.F:677-679 calls DIAGS_SOUND_SPEED, whose "
        "output cSound (model/inc/DYNVARS.h:149-154) is written at "
        "model/src/diags_sound_speed.F:86,119 and read only by DIAGNOSTICS_FILL at :128"),
    ("CPP_EEOPTIONS.h", "USE_FORTRAN_SCRATCH_FILES"): Ruling(
        SKIP, "none: how the scratch copy of a namelist file is opened",
        "eesupp/inc/CPP_EEOPTIONS.h:83; eesupp/src/eeset_parms.F:147-150 and "
        "eesupp/src/open_copy_data_file.F:86-100 open a Fortran scratch file instead of a named "
        "one"),
    ("DIAG_OPTIONS.h", "DIAGSTATS_REGION_MASK"): Ruling(
        SKIP, "none: regional statistics of the diagnostics package",
        "pkg/diagnostics/DIAG_OPTIONS.h:9-11; used only in "
        "pkg/diagnostics/diagstats_set_regions.F:37-225 and pkg/diagnostics/DIAGSTATS_REGIONS.h:22-28"),
    ("MOM_COMMON_OPTIONS.h", "ALLOW_MOM_TEND_EXTRA_DIAGS"): Ruling(
        SKIP, "none: extra momentum-tendency diagnostics",
        "pkg/mom_common/MOM_COMMON_OPTIONS.h:51; its blocks in pkg/mom_vecinv/mom_vecinv.F:249-260, "
        ":510-514, :535-545, :620-624, :645-655 only fill diagnostics (the scratch vrF written at "
        ":539 is recomputed at :554 before use); pkg/mom_common/mom_diagnostics_init.F:486 "
        "registers them"),
    ("PROFILES_OPTIONS.h", "PROFILES_USE_MDSFINDUNITS"): Ruling(
        SKIP, "none: which routine picks Fortran unit numbers for profile files",
        "pkg/profiles/profiles_init_fixed.F:972-1017 (MDSFINDUNIT instead of PROFILES_FINDUNIT)"),
    ("FLT_OPTIONS.h", "FLT_SECOND_ORDER_RUNGE_KUTTA"): Ruling(
        SKIP, "none: how float trajectories are integrated",
        "pkg/flt/FLT_OPTIONS.h:23 default #undef; its only use, pkg/flt/flt_main.F:107-119, calls "
        "FLT_RUNGA2 instead of FLT_RUNGA4; pkg/flt reads the state and writes back nothing "
        "(translate PACKAGE_RULES useflt: flt_up.F:130-136)"),
    ("GGL90_OPTIONS.h", "GGL90_MISSING_HFAC_BUG"): Ruling(
        MAP, "ggl90 vertical_mixing_parameters.missing_hfac_bug (translate._vertical_mixing_parameters "
        "reads the header state; GGL90_CPP_OPTIONS of "
        "defgcm/dynamics/vertical_mixing_schemes/ggl90.py holds it)",
        "pkg/ggl90/GGL90_OPTIONS.h:39 default #undef; its only use, pkg/ggl90/ggl90_calc.F:677-685, "
        "is inside GGL90_CALC, which model/src/do_oceanic_phys.F:1005-1011 calls only when useGGL90"),
    ("MOM_COMMON_OPTIONS.h", "COSINEMETH_III"): Ruling(
        SKIP, "none while the condition holds", _COS_SCALING_BASIS, _COS_SCALING),
    ("MOM_COMMON_OPTIONS.h", "ISOTROPIC_COS_SCALING"): Ruling(
        SKIP, "none while the condition holds", _COS_SCALING_BASIS, _COS_SCALING),
    ("CPP_EEOPTIONS.h", "DISCONNECTED_TILES"): Ruling(
        SKIP, "none while the condition holds: a single tile is the whole periodic domain",
        "eesupp/inc/CPP_EEOPTIONS.h:123 default #undef.  It replaces each exchange by "
        "FILL_HALO_LOCAL, a locally periodic halo fill of the tile (eesupp/src/exch0_rx.template:"
        "82, :101-237), which for the only tile equals its periodic exchange with itself, and "
        "CG2D by CG2D_EX0 (model/src/solve_for_pressure.F:280-286), which differs from "
        "model/src/cg2d.F only by tile sums and maxima in place of global sums and "
        "FILL_HALO_LOCAL in place of EXCH_S3D (cg2d_ex0.F:134-367); config_check.F:606-636 only "
        "drops an overlap check and eesupp/src/ini_procs.F:539-550 stops on several processes. "
        "CG3D_EX0 (solve_for_pressure.F:419-424) is not examined, hence no 3-D solver",
        Condition(
            "one tile (nSx*nSy*nPx*nPy = 1 in SIZE.h), periodic in x and y (notUsingXPeriodicity "
            "and notUsingYPeriodicity false, eesupp/src/eeset_parms.F:104-105) and no 3-D "
            "solver (nonHydrostatic and implicitIntGravWave false, model/src/set_parms.F:104)",
            _single_periodic_tile)),
    ("CPP_OPTIONS.h", "CHECK_SALINITY_FOR_NEGATIVE_VALUES"): Ruling(
        MAP, NEGATIVE_SALINITY_CHECK_TARGET,
        "model/inc/CPP_OPTIONS.h:43 default #undef; its only active use, "
        "model/src/find_rho.F:85-89, calls LOOK_FOR_NEG_SALINITY before every equation of "
        "state branch, which sets negative salinity to 0 in place (:1230-1232) in the "
        "prognostic salt that do_oceanic_phys.F:758-765 passes; the find_alpha.F:68-73 and "
        ":401-406 calls are commented out.  translate._wire_negative_salinity_check writes "
        "the compiled state; the stepper refuses the compositions its port does not reach "
        "(admission.validate_negative_salinity_check)"),
    ("CPP_OPTIONS.h", "USE_OLD_EXTERNAL_FORCING"): Ruling(
        SKIP, "none while no forcing term is added: the old calling sequence of the same terms",
        "model/inc/CPP_OPTIONS.h:154 default #undef; APPLY_FORCING_U/V then let "
        "EXTERNAL_FORCING_U/V add the surface stress and the package tendencies to gU, gV "
        "(model/src/external_forcing.F:66-146, :206-286) and add the difference to the tendency "
        "(apply_forcing.F:67-92, :257-283), the terms of the default branch (:96-195, :284-385) "
        "less momTidalForcing, which config_check.F:428-437 refuses under the macro; a zero "
        "increment leaves the tendency bitwise unchanged.  APPLY_FORCING_T/S run only inside "
        "TEMP_INTEGRATE, SALT_INTEGRATE (thermodynamics.F:317-335)",
        Condition(
            "T and S not stepped (tempStepping, saltStepping false), no zonalWindFile or "
            "meridWindFile, no coupler and no data.pkg package on besides diagnostics and mnc, so "
            "every term EXTERNAL_FORCING_U/V adds is zero (surfaceForcingU/V = fu, fv from the "
            "wind files or a package, model/src/external_forcing_surf.F:222)",
            _no_external_forcing)),
    ("CD_CODE_OPTIONS.h", "CD_CODE_NO_AB_MOMENTUM"): Ruling(
        MAP, "cd_scheme cpp_defines (translate._cd_code_options; "
        "defgcm/dynamics/cd_scheme/module.py CD_CODE_OPTIONS)",
        "pkg/cd_code/CD_CODE_OPTIONS.h:17 default #undef; used at model/src/timestep.F:148-158"),
    ("CD_CODE_OPTIONS.h", "CD_CODE_NO_AB_CORIOLIS"): Ruling(
        MAP, "cd_scheme cpp_defines (translate._cd_code_options; "
        "defgcm/dynamics/cd_scheme/module.py CD_CODE_OPTIONS)",
        "pkg/cd_code/CD_CODE_OPTIONS.h:18 default #undef; used at pkg/cd_code/cd_code_scheme.F:97-105, "
        ":134-138, :196-200"),
    ("W2_OPTIONS.h", "W2_CUMSUM_USE_MATRIX"): Ruling(
        SKIP, "none: how the barotropic streamfunction diagnostic sums across tiles",
        "pkg/exch2/W2_OPTIONS.h:22 default #undef; it selects the matrix form of the tile "
        "cumulative sum, set up by pkg/exch2/w2_set_map_cumsum.F:264-365 (from w2_e2setup.F:141) "
        "and used in w2_cumulsum_z_tile.F:162-230; that sum's only caller is "
        "eesupp/src/cumulsum_z_tile.F:71, reached only from pkg/diagnostics/diag_calc_psivel.F:237, "
        "which assigns no model state variable"),
    ("LAYERS_OPTIONS.h", "LAYERS_THERMODYNAMICS"): Ruling(
        SKIP, "none: layer-budget diagnostics",
        "pkg/layers/LAYERS_OPTIONS.h:19; referenced only inside pkg/layers (layers_fill.F:15-20, "
        "layers_calc.F:93-188, layers_diapycnal.F:66-324), which assigns no model state variable"),
}

_GATED_BASIS = (
    "model/src/packages_boot.F:{namelist} reads {switch} from data.pkg, default .FALSE. at "
    ":{default}; core code calls a package only under its use switch "
    "(doc/contributing/contributing.rst:461-467)"
)


def _gated(switch: str, namelist: int, default: int) -> Ruling:
    return Ruling(
        SKIP,
        f"none by itself: activation is the data.pkg switch {switch}, ruled by translate "
        "PACKAGE_RULES",
        _GATED_BASIS.format(switch=switch, namelist=namelist, default=default),
    )


def _absent(switch: str, namelist: int) -> Ruling:
    return Ruling(
        SKIP,
        f"none: a run that sets {switch} without the package stops at start-up",
        f"model/src/packages_boot.F:{namelist} reads {switch}; "
        "doc/examples/reentrant_channel/reentrant_channel.rst:1276-1278 (a use flag set for "
        "a package not in the build terminates the model)",
    )


#: Packages switched on at run time by a ``data.pkg`` flag: ``package -> (flag,
#: namelist line, default line)`` in ``model/src/packages_boot.F``.  The
#: adjoint packages whose flag defaults to ``.TRUE.`` once compiled
#: (``packages_boot.F:166-175``) are not listed.
_RUNTIME_SWITCHES: dict[str, tuple[str, int, int]] = {
    "obcs": ("useOBCS", 47, 113),
    "shap_filt": ("useSHAP_FILT", 48, 114),
    "zonal_filt": ("useZONAL_FILT", 49, 115),
    "opps": ("useOPPS", 50, 116),
    "pp81": ("usePP81", 51, 117),
    "kl10": ("useKL10", 52, 118),
    "my82": ("useMY82", 53, 119),
    "ggl90": ("useGGL90", 54, 120),
    "kpp": ("useKPP", 55, 121),
    "gmredi": ("useGMRedi", 56, 122),
    "down_slope": ("useDOWN_SLOPE", 57, 123),
    "bbl": ("useBBL", 58, 124),
    "cal": ("useCAL", 59, 125),
    "exf": ("useEXF", 60, 126),
    "bulk_force": ("useBulkForce", 61, 127),
    "ebm": ("useEBM", 62, 128),
    "cheapaml": ("useCheapAML", 63, 129),
    "profiles": ("usePROFILES", 67, 133),
    "obsfit": ("useOBSFIT", 68, 134),
    "sbo": ("useSBO", 71, 137),
    "flt": ("useFLT", 72, 138),
    "ptracers": ("usePTRACERS", 73, 139),
    "gchem": ("useGCHEM", 74, 140),
    "rbcs": ("useRBCS", 75, 141),
    "offline": ("useOffLine", 76, 142),
    "matrix": ("useMATRIX", 77, 143),
    "frazil": ("useFRAZIL", 78, 144),
    "seaice": ("useSEAICE", 79, 145),
    "salt_plume": ("useSALT_PLUME", 80, 146),
    "shelfice": ("useShelfIce", 81, 147),
    "steep_icecavity": ("useSTIC", 82, 148),
    "streamice": ("useStreamIce", 83, 149),
    "icefront": ("useICEFRONT", 84, 150),
    "thsice": ("useThSIce", 85, 151),
    "land": ("useLand", 86, 152),
    "aim_v23": ("useAIM", 88, 154),
    "atm_phys": ("useAtm_Phys", 89, 155),
    "fizhi": ("useFizhi", 90, 156),
    "gridalt": ("useGridAlt", 91, 157),
    "diagnostics": ("useDiagnostics", 92, 158),
    "layers": ("useLayers", 94, 160),
    "mnc": ("useMNC", 95, 161),
    "mypackage": ("useMYPACKAGE", 98, 164),
}

_CORE_SELECTION = (
    "none by itself: the default build the translator assumes; the form in use is selected by "
    "vectorInvariantMomentum, momStepping, tempStepping, saltStepping and *AdvScheme, ruled by "
    "translate DATA_RULES"
)

#: ``(package, compiled) -> ruling``.  ``compiled`` is False for a package that
#: ``packages.conf`` excludes with ``-``.
PACKAGE_RULINGS: dict[tuple[str, bool], Ruling] = {
    **{(name, True): _gated(*spec) for name, spec in _RUNTIME_SWITCHES.items()},
    **{(name, False): _absent(spec[0], spec[1]) for name, spec in _RUNTIME_SWITCHES.items()},
    ("mom_common", True): Ruling(SKIP, _CORE_SELECTION, "pkg/pkg_groups:15,17 default package list"),
    ("mom_fluxform", True): Ruling(SKIP, _CORE_SELECTION, "pkg/pkg_groups:15,17 default package list"),
    ("mom_vecinv", True): Ruling(SKIP, _CORE_SELECTION, "pkg/pkg_groups:15,17 default package list"),
    ("generic_advdiff", True): Ruling(
        SKIP, _CORE_SELECTION, "pkg/pkg_groups:15,17 default package list"),
    ("debug", True): Ruling(
        SKIP, "none: debug printing",
        "pkg/pkg_groups:17; core calls are IF (debugMode) CALL DEBUG_* "
        "(model/src/do_oceanic_phys.F:250-252), which print to STDOUT "
        "(pkg/debug/debug_enter.F:8)"),
    ("mdsio", True): Ruling(
        SKIP, "none: binary file I/O",
        "pkg/pkg_groups:17; pkg/mdsio/mdsio_read_field.F:4-21 low-level reader"),
    ("rw", True): Ruling(
        SKIP, "none: file I/O front end",
        "pkg/pkg_groups:17 and pkg/pkg_depend:16; pkg/rw/read_fld_xyz_rl.F:4 front end of the "
        "low-level I/O"),
    ("monitor", True): Ruling(
        SKIP, "none: statistics written to STDOUT",
        "pkg/pkg_groups:17; pkg/monitor/monitor.F:12-13"),
    ("mom_fluxform", False): Ruling(
        SKIP, "none: flux-form momentum cannot be selected without it",
        "model/src/packages_check.F:158-169 stops when momStepping and not "
        "vectorInvariantMomentum"),
    ("mom_vecinv", False): Ruling(
        SKIP, "none: vector-invariant momentum cannot be selected without it",
        "model/src/packages_check.F:145-156 stops when momStepping and vectorInvariantMomentum"),
    ("mom_common", False): Ruling(
        SKIP, "none: no momentum form can be compiled without it",
        "pkg/pkg_depend:23-25 (mom_fluxform, mom_vecinv, cd_code need it); "
        "model/src/packages_check.F:145-169 stops when momStepping without a momentum form"),
    ("generic_advdiff", False): Ruling(
        SKIP, "none: T and S cannot be stepped without it",
        "model/src/packages_check.F:172-183 stops when tempStepping or saltStepping"),
    ("exch2", True): Ruling(
        SKIP, "none while the condition holds: the same doubly periodic tile connectivity as the "
        "standard exchange",
        "pkg/exch2/w2_readparms.F:65-69 (preDefTopol = 1 without useCubedSphereExchange) and "
        ":129-134 (no data.exch2: single sub-domain), pkg/exch2/w2_set_single_facet.F:41-58 (one "
        "Nx x Ny facet whose N-S and E-W edges join each other); the model's exch2 branches read "
        "facet and corner flags only under useCubedSphereExchange "
        "(pkg/mom_common/mom_calc_relvort3.F:80, mom_calc_hfacz.F:229, "
        "pkg/mom_vecinv/mom_vi_u_coriolis_c4.F:74, pkg/generic_advdiff/gad_advection.F:252, "
        "model/src/ini_global_domain.F:145) or take the grid size of that facet "
        "(model/src/ini_parms.F:326-328, ini_local_grid.F:100-102)",
        Condition("exch2 runs its single-facet default: no input/data.exch2 and "
                  "useCubedSphereExchange false", _exch2_single_facet)),
    ("cd_code", True): Ruling(
        SKIP, "none by itself: activation is useCDscheme, ruled by translate DATA_RULES",
        "model/src/packages_check.F:192-195 (useCDscheme requires the package); pkg/pkg_depend:25"),
    ("longstep", True): Ruling(
        SKIP, "none while the condition holds: the package only re-times passive tracers",
        "no flag of its own, active with usePTRACERS (model/src/packages_boot.F:363-365); "
        "LONGSTEP_OPTIONS.h defines no macro; its calls sit under usePTRACERS "
        "(model/src/forward_step.F:701, :742, :1029, packages_check.F:297, set_parms.F:332, "
        "packages_init_fixed.F:416, packages_init_variables.F:329) or return at once without it "
        "(packages_readparms.F:247, pkg/longstep/longstep_readparms.F:46-55); under ALLOW_AUTODIFF "
        "LONGSTEP_INIT_VARIA runs regardless but writes only LS_* and LS_doTimeStep "
        "(longstep_init_varia.F:37-72), read at do_fields_blocking_exchanges.F:92 around a "
        "usePTRACERS call (:96) and at forward_step.F:1067 with useGCHEM, which needs usePTRACERS "
        "(pkg/gchem/gchem_check.F:108); the ALLOW_LONGSTEP branches outside the package act only "
        "on tracer ids >= GAD_TR1 (model/src/calc_3d_diffusivity.F:92, :186, "
        "pkg/gmredi/gmredi_calc_diff.F:54, :73, gmredi_xtransport.F:117, :175, :204, "
        "gmredi_ytransport.F:117, :175, :204, gmredi_rtransport.F:135, :165, "
        "pkg/generic_advdiff/gad_calc_rhs.F:675) or inside pkg/ptracers, "
        "and only PTRACERS_INTEGRATE steps those tracers (pkg/longstep/longstep_thermodynamics.F:189, "
        "reached from the usePTRACERS calls above; the macro only removes the thermodynamics.F "
        "call, :6-8, :339-354)",
        Condition("usePTRACERS false (data.pkg)", _ptracers_off)),
}


# --------------------------------------------------------------------------
# CPP evaluation
# --------------------------------------------------------------------------

_DIRECTIVE = re.compile(r"^\s*#\s*(\w+)\s*(.*)$")
_NAME = re.compile(r"[A-Za-z_]\w*")
_C_COMMENT = re.compile(r"/\*.*?\*/")
_COND_TOKEN = re.compile(r"\s*(defined|&&|\|\||!|\(|\)|[A-Za-z_]\w*|\d+)")
_COND_WORDS = {"&&": "and", "||": "or", "!": "not", "(": "(", ")": ")"}


def _directives(path: Path):
    """Yield ``(line, directive, argument)`` with continuation lines joined."""
    lines = path.read_text(errors="replace").splitlines()
    index = 0
    while index < len(lines):
        start, text = index, lines[index]
        while text.endswith("\\") and index + 1 < len(lines):
            index += 1
            text = text[:-1] + " " + lines[index]
        index += 1
        match = _DIRECTIVE.match(text)
        if match:
            yield start + 1, match.group(1), _C_COMMENT.sub(" ", match.group(2)).strip()


def _condition(expression: str, state: dict[str, str], where: str) -> bool:
    """Evaluate an ``#if`` made of ``defined``, ``!``, ``&&``, ``||`` and parentheses."""
    tokens = _COND_TOKEN.findall(expression)
    if "".join(tokens) != re.sub(r"\s+", "", expression):
        raise ValueError(f"{where}: unsupported #if expression: {expression}")
    words = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "defined":
            name = tokens[index + 2] if tokens[index + 1] == "(" else tokens[index + 1]
            words.append(str(name in state))
            index += 4 if tokens[index + 1] == "(" else 2
            continue
        if token in _COND_WORDS:
            words.append(_COND_WORDS[token])
        elif token.isdigit():
            words.append(token)
        else:
            raise ValueError(f"{where}: unsupported #if operand {token} in: {expression}")
        index += 1
    return bool(eval(" ".join(words), {"__builtins__": {}}))


def evaluate_header(path: Path, context: dict[str, str]) -> dict[str, tuple[str | None, int]]:
    """Macro states set by the active directives of ``path`` itself.

    ``context`` maps every macro defined before the header to its value (``""``
    for a macro defined without one).  The result maps a macro to ``(value,
    line)``, where ``value`` is ``None`` for ``#undef``; a later directive for the
    same macro overrides an earlier one.
    """
    state = dict(context)
    own: dict[str, tuple[str | None, int]] = {}
    stack: list[tuple[bool, bool]] = []  # (enclosing block active, a branch was taken)
    active = True
    for line, directive, argument in _directives(path):
        where = f"{path}:{line}"
        if directive in ("ifdef", "ifndef"):
            taken = (_NAME.match(argument).group(0) in state) == (directive == "ifdef")
            stack.append((active, taken))
            active = active and taken
        elif directive == "if":
            taken = active and _condition(argument, state, where)
            stack.append((active, taken))
            active = taken
        elif directive == "elif":
            enclosing, done = stack[-1]
            taken = enclosing and not done and _condition(argument, state, where)
            stack[-1] = (enclosing, done or taken)
            active = taken
        elif directive == "else":
            enclosing, done = stack[-1]
            stack[-1] = (enclosing, True)
            active = enclosing and not done
        elif directive == "endif":
            active = stack.pop()[0]
        elif active and directive == "define":
            name = _NAME.match(argument).group(0)
            value = argument[len(name):].strip()
            state[name] = value
            own[name] = (value, line)
        elif active and directive == "undef":
            name = _NAME.match(argument).group(0)
            state.pop(name, None)
            own[name] = (None, line)
    return own


def _apply(state: dict[str, str], own: dict[str, tuple[str | None, int]]) -> dict[str, str]:
    merged = dict(state)
    for name, (value, _) in own.items():
        if value is None:
            merged.pop(name, None)
        else:
            merged[name] = value
    return merged


def _state_text(value: str | None, where: str, line: int) -> str:
    state = "undef" if value is None else f"define {value}".rstrip()
    return f"{state} ({where}:{line})" if line else f"{state} (not set in {where})"


# --------------------------------------------------------------------------
# MITgcm defaults
# --------------------------------------------------------------------------

def _mit_original(name: str, mit_root: Path, standard: tuple[str, ...]) -> Path | None:
    """The file of this name genmake2 would use without the case: a standard
    directory first, otherwise the single package that ships it."""
    for base in standard:
        if (mit_root / base / name).is_file():
            return mit_root / base / name
    hits = sorted(mit_root.glob(f"pkg/*/{name}"))
    return hits[0] if len(hits) == 1 else None


def default_header(name: str, mit_root: Path) -> Path | None:
    return _mit_original(name, mit_root, ("model/inc", "eesupp/inc"))


def original_source(name: str, mit_root: Path) -> Path | None:
    return _mit_original(name, mit_root, ("model/src", "eesupp/src"))


def _words(path: Path) -> list[str]:
    return [w for line in path.read_text(errors="replace").splitlines()
            for w in line.split("#", 1)[0].split()]


def package_groups(mit_root: Path) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for line in (mit_root / "pkg" / "pkg_groups").read_text().splitlines():
        head, sep, members = line.split("#", 1)[0].partition(":")
        if sep and head.split():
            groups[head.strip()] = members.split()
    return groups


def package_dependencies(mit_root: Path) -> list[tuple[str, str, str]]:
    """``(package, rule, dependency)`` with rule one of ``+``, ``=``, ``-``."""
    rules = []
    for line in (mit_root / "pkg" / "pkg_depend").read_text().splitlines():
        words = line.split("#", 1)[0].split()
        rules.extend((words[0], dep[0], dep[1:]) for dep in words[1:])
    return rules


@dataclass(frozen=True)
class PackageSet:
    compiled: dict[str, str]  # package -> how it entered the build
    excluded: frozenset[str]  # names written with "-" in packages.conf


def compiled_packages(code_dir: Path, mit_root: Path) -> PackageSet:
    """The package set genmake2 compiles for ``code_dir``."""
    conf = code_dir / "packages.conf"
    if conf.is_file():
        return _expand_packages(_words(conf), "packages.conf", mit_root)
    return _expand_packages(["default_pkg_list"], "default_pkg_list (no packages.conf)", mit_root)


def _expand_packages(listed: list[str], origin_root: str, mit_root: Path) -> PackageSet:
    groups = package_groups(mit_root)
    origin = {name: origin_root for name in listed}
    names = list(listed)
    while any(name in groups for name in names):
        expanded = []
        for name in names:
            if name in groups:
                for member in groups[name]:
                    origin.setdefault(member, f"group {name}")
                expanded.extend(groups[name])
            else:
                expanded.append(name)
        names = expanded
    excluded = frozenset(name[1:] for name in names if name.startswith("-"))
    compiled = {name: origin[name] for name in names
                if not name.startswith("-") and name not in excluded}
    rules = package_dependencies(mit_root)
    changed = True
    while changed:
        changed = False
        for owner, rule, dependency in rules:
            active = owner in compiled or owner in _STANDARD_DIRS
            if active and rule in "+=" and dependency not in compiled and dependency not in excluded:
                compiled[dependency] = f"pkg_depend {owner} {rule}{dependency}"
                changed = True
    return PackageSet(compiled, excluded)


def _package_macros(packages: PackageSet) -> dict[str, str]:
    macros = {f"ALLOW_{name.upper()}": "" for name in packages.compiled}
    if "aim_v23" in packages.compiled:
        macros["ALLOW_AIM"] = ""  # tools/genmake2:2720-2723
    return macros


# --------------------------------------------------------------------------
# inventory
# --------------------------------------------------------------------------

def _relative(path: Path | None, mit_root: Path) -> str:
    return str(path.relative_to(mit_root)) if path is not None else "none"


def _replaces(original: Path | None, mit_root: Path) -> str:
    if original is None:
        return "new file: no MITgcm file of this name"
    return f"replaces {_relative(original, mit_root)}"


def _file_ruling(path: Path, repo: Path) -> Ruling | None:
    for name, copy, ruling in FILE_RULINGS:
        if name == path.name and (copy is None or (repo / copy).read_bytes() == path.read_bytes()):
            return ruling
    return None


def _ruled(file: str, item: str, kind: str, change: str, ruling: Ruling | None,
           run: Run | None) -> Item:
    if ruling is None:
        return Item(file, item, kind, change, REVIEW, "", "")
    if ruling.condition is None:
        return Item(file, item, kind, change, ruling.disposition, ruling.model, ruling.basis)
    if run is None:
        return Item(file, item, kind, change, ruling.disposition,
                    f"{ruling.model} (when {ruling.condition.text})", ruling.basis)
    if ruling.condition.holds(run):
        return Item(file, item, kind, change, ruling.disposition,
                    f"{ruling.model} ({ruling.condition.text})", ruling.basis)
    return Item(file, item, kind, change, UNSUPPORTED,
                f"the ruling needs {ruling.condition.text}, which this run does not satisfy",
                ruling.basis)


def _package_items(packages: PackageSet, defaults: PackageSet, run: Run | None) -> list[Item]:
    items = []
    for name in sorted(packages.compiled.keys() | packages.excluded):
        compiled = name in packages.compiled
        if compiled:
            status = "default" if name in defaults.compiled else "added"
            change = f"{status}; {packages.compiled[name]}"
        else:
            change = "removed" if name in defaults.compiled else "excluded"
        items.append(_ruled(KIND_PACKAGES, name, KIND_PACKAGES, change,
                            PACKAGE_RULINGS.get((name, compiled)), run))
    return items


def _option_items(header: Path, context: dict[str, str], mit_root: Path,
                  run: Run | None) -> list[Item]:
    default = default_header(header.name, mit_root)
    case_own = evaluate_header(header, context)
    default_own = evaluate_header(default, context) if default is not None else {}
    reference = _relative(default, mit_root)
    items = []
    for name in sorted(case_own.keys() | default_own.keys()):
        case_value, case_line = case_own.get(name, (context.get(name), 0))
        default_value, default_line = default_own.get(name, (context.get(name), 0))
        if case_value == default_value:
            continue
        change = (f"{_state_text(default_value, reference, default_line)} -> "
                  f"{_state_text(case_value, f'code/{header.name}', case_line)}")
        items.append(_ruled(header.name, name, KIND_OPTIONS, change,
                            MACRO_RULINGS.get((header.name, name)), run))
    if not items:
        items.append(Item(header.name, "", KIND_OPTIONS, f"no macro differs from {reference}",
                          SKIP, "none: same compile-time state as the default header",
                          f"{reference} evaluated with the same predefined macros"))
    return items


def inventory_case(case_dir: Path, mit_root: Path, run: Run | None = None) -> list[Item]:
    """Rows for every file of ``case_dir/code``; empty when there is no ``code/``.

    With ``run``, a conditional ruling whose condition fails gives ``unsupported``.
    """
    code = Path(case_dir) / "code"
    if not code.is_dir():
        return []
    repo = Path(__file__).resolve().parents[2]
    packages = compiled_packages(code, mit_root)
    defaults = _expand_packages(["default_pkg_list"], "default_pkg_list", mit_root)
    base = _package_macros(packages)
    top = dict(base)
    for name in ("CPP_OPTIONS.h", "CPP_EEOPTIONS.h"):
        effective = code / name if (code / name).is_file() else default_header(name, mit_root)
        top = _apply(top, evaluate_header(effective, base))
    items = _package_items(packages, defaults, run)
    for path in sorted(p for p in code.iterdir() if p.is_file()):
        name = path.name
        if name == "packages.conf":
            continue
        if name.endswith("OPTIONS.h"):
            context = base if name in ("CPP_OPTIONS.h", "CPP_EEOPTIONS.h") else top
            items.extend(_option_items(path, context, mit_root, run))
        elif name == "SIZE.h":
            items.append(_ruled(name, "", KIND_SIZE, "replaces model/inc/SIZE.h",
                                _file_ruling(path, repo), run))
        elif name.endswith(".h"):
            change = _replaces(default_header(name, mit_root), mit_root)
            items.append(_ruled(name, "", KIND_HEADER, change, _file_ruling(path, repo), run))
        elif name.endswith((".F", ".F90")):
            change = _replaces(original_source(name, mit_root), mit_root)
            items.append(_ruled(name, "", KIND_SOURCE, change, _file_ruling(path, repo), run))
        else:
            items.append(_ruled(name, "", KIND_OTHER, "not compiled under this name",
                                _file_ruling(path, repo), run))
    return items


# --------------------------------------------------------------------------
# scope and command line
# --------------------------------------------------------------------------

#: Main configurations in scope: the 38 ocean and sea-ice cases of the boundary survey
#: minus its exclusions.  Subcases share
#: their main case's ``code/``.
IN_SCOPE: tuple[str, ...] = (
    "tutorial_barotropic_gyre", "tutorial_baroclinic_gyre", "tutorial_deep_convection",
    "tutorial_global_oce_latlon", "tutorial_advection_in_gyre", "tutorial_reentrant_channel",
    "tutorial_plume_on_slope", "vermix", "internal_wave", "advect_xy", "advect_xz", "advect_cs",
    "MLAdjust", "front_relax", "ideal_2D_oce", "short_surf_wave", "inverted_barometer", "exp2",
    "exp4", "dome", "adjustment.cs-32x32x1", "offline_exf_seaice", "seaice_itd",
    "1D_ocean_ice_column", "lab_sea", "seaice_obcs", "global_ocean.cs32x15",
    "global_ocean.90x40x15", "global_oce_latlon",
)

#: Out of scope by section 6 of the boundary survey, with its reason.
OUT_OF_SCOPE: dict[str, str] = {
    "adjustment.128x64x1": "atmosphere",
    "aim.5l_cs": "atmosphere", "aim.5l_Equatorial_Channel": "atmosphere",
    "aim.5l_LatLon": "atmosphere", "atm_gray": "atmosphere", "cpl_aim+ocn": "atmosphere",
    "fizhi-cs-32x32x40": "atmosphere", "fizhi-cs-aqualev20": "atmosphere",
    "fizhi-gridalt-hs": "atmosphere", "hs94.128x64x5": "atmosphere",
    "hs94.1x64x5": "atmosphere", "hs94.cs-32x32x5": "atmosphere",
    "tutorial_held_suarez_cs": "atmosphere", "solid-body.cs-32x32x1": "atmosphere",
    "cfc_example": "biogeochemistry", "tutorial_cfc_offline": "biogeochemistry, offline",
    "global_oce_biogeo_bling": "biogeochemistry", "so_box_biogeo": "biogeochemistry",
    "tutorial_global_oce_biogeo": "biogeochemistry",
    "tutorial_dic_adjoffline": "biogeochemistry, offline",
    "isomip": "ice shelf", "shelfice_2d_remesh": "ice shelf", "halfpipe_streamice": "streamice",
    "global_ocean_ebm": "EBM", "cheapAML_box": "cheapAML", "matrix_example": "matrix",
    "tutorial_global_oce_in_p": "pressure coordinates",
    "tutorial_rotating_tank": "cylindrical grid, rigid lid",
    "deep_anelastic": "planetary-shell convection",
}


def scope(case: str) -> str:
    if case in IN_SCOPE:
        return "in"
    if case in OUT_OF_SCOPE:
        return f"out: {OUT_OF_SCOPE[case]}"
    return "unlisted"


_COLUMNS = ("case", "scope", "file", "item", "kind", "change", "disposition", "model", "basis")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="TSV written for all cases")
    parser.add_argument("--mit-root", type=Path, default=None,
                        help="MITgcm root (default: mitgcm_root of defgcm.external_paths)")
    args = parser.parse_args(argv)
    if args.mit_root is None:
        from defgcm.external_paths import resolve
        mit_root = Path(resolve("mitgcm_root"))
    else:
        mit_root = args.mit_root
    cases = sorted(p for p in (mit_root / "verification").iterdir() if (p / "code").is_dir())
    rows = [(case.name, scope(case.name), item) for case in cases
            for item in inventory_case(case, mit_root)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(_COLUMNS)
        for case, where, item in rows:
            writer.writerow((case, where, item.file, item.item, item.kind, item.change,
                             item.disposition, item.model, item.basis))

    with_sources = sorted({case for case, _, item in rows if item.kind == KIND_SOURCE})
    print(f"code/ directories: {len(cases)}; with .F/.F90 changes: {len(with_sources)}")
    print("  " + " ".join(with_sources))
    macro_cases = Counter((item.file, item.item) for _, _, item in rows if item.kind == KIND_OPTIONS
                          and item.item)
    print(f"changed macros: {len(macro_cases)} distinct (header, macro) pairs")
    for (header, macro), count in sorted(macro_cases.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {count:3d}  {header}  {macro}")
    review = [(case, where, item) for case, where, item in rows if item.disposition == REVIEW]
    in_scope_review = [r for r in review if r[1] == "in"]
    print(f"rows: {len(rows)}; review: {len(review)} ({len(in_scope_review)} in scope)")
    for kind, count in Counter(item.kind for _, _, item in review).most_common():
        in_scope = sum(1 for _, where, item in in_scope_review if item.kind == kind)
        print(f"  review {kind}: {count} ({in_scope} in scope)")
    print("dispositions: " + ", ".join(
        f"{d} {n}" for d, n in Counter(item.disposition for _, _, item in rows).most_common()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
