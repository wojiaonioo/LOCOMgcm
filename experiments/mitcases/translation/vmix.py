"""The shared mapping tables of the vertical mixing packages' data.<pkg>."""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import Any, Callable
from experiments.mitcases.code_inventory import MAP
from defgcm.steppers.ifs.columns import VERTICAL_MIXING_SCHEMES
from defgcm.dynamics.vertical_mixing_schemes.pp81 import PP81_CPP_OPTIONS
from defgcm.dynamics.vertical_mixing_schemes.my82 import MY82_CPP_OPTIONS
from defgcm.dynamics.vertical_mixing_schemes.kl10 import KL10_CPP_OPTIONS
from defgcm.dynamics.vertical_mixing_schemes.ggl90 import GGL90_CPP_OPTIONS

from .core import (
    Gap,
    Record,
    Skipped,
    _ASSEMBLY,
    _Assembly,
    _record_gap,
    _set_dotted,
)
from .rules import (
    Ctx,
    Rule,
    _false_or_gap,
    _gap,
    _skip,
    _zero_or_gap,
)
from .cpp import (
    _reference_mit_root,
    parse_cpp_macro_states,
)

def _vertical_mixing_package(scheme: str) -> Rule:
    """use<PKG> of a vertical mixing package -> the stepper's vertical_mixing_scheme ``scheme``.

    MITgcm then forms the package's viscosity and diffusivity from the step-start
    state in DO_OCEANIC_PHYS (do_oceanic_phys.F:967-1012) and merges them into
    KappaRU/RV (calc_viscosity.F:85-130) and KappaRT/RS (calc_3d_diffusivity.F:
    213-240); the stepper does the same for each scheme of
    columns.VERTICAL_MIXING_SCHEMES, one at a time.  The package's data.<pkg> is
    read key by key (_VERTICAL_MIXING_PACKAGES) and resolved into
    vertical_mixing_parameters (_vertical_mixing_parameters).  The stepper checks
    at construction the MITgcm stops of <pkg>_check.F on what it reads (ivdc_kappa,
    KPP, implicitDiffusion, implicitViscosity; _try_construct), so both implicit
    switches are written beside the scheme, as ivdc_kappa's row writes
    implicitDiffusion; the model checks the scheme against cAdjFreq, the
    convective_adjustment module, at construction (columns.validate_convection).
    """

    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        del value
        if scheme not in VERTICAL_MIXING_SCHEMES:
            raise Gap(f"vertical mixing scheme {scheme!r} is not one of the stepper's "
                      f"VERTICAL_MIXING_SCHEMES {VERTICAL_MIXING_SCHEMES}")
        packages = ctx.namelists.get("data.pkg", {}).get("PACKAGES", {})
        on = sorted(flag for flag in _VERTICAL_MIXING_PACKAGES if packages.get(flag) is True)
        if len(on) > 1:
            raise Gap(f"{on} on together; the stepper carries one vertical_mixing_scheme")
        return {
            "model.stepper.kwargs.vertical_mixing_scheme": scheme,
            "model.stepper.kwargs.implicit_tracer_diffusion": bool(
                ctx.data("PARM01", "implicitdiffusion", False)),
            "model.stepper.kwargs.implicit_vertical_viscosity": bool(
                ctx.data("PARM01", "implicitviscosity", False)),
        }

    return Rule(
        MAP, target="model.stepper.kwargs.vertical_mixing_scheme",
        convert=f".TRUE. -> {scheme!r} with implicitDiffusion and implicitViscosity beside it; "
                "its data.<pkg> resolves vertical_mixing_parameters",
        fn=fn,
    )
# --------------------------------------------------------------------------
# THE SHARED MAPPING TABLES -- the vertical mixing packages' ``data.<pkg>``
# --------------------------------------------------------------------------
#
# One table per package of columns.VERTICAL_MIXING_SCHEMES, keyed by the
# NAMELIST of its <pkg>_readparms.F.  A key the scheme's parameter record
# carries is staged as <scheme>_<field> and resolved after MITgcm's defaults
# (_MIT_ASSEMBLY_DEFAULTS) by _vertical_mixing_parameters; a field is in the
# same unit on both sides, so the conversion is the identity.  Output controls
# are skips.  A key only a branch outside the port reads is a skip while that
# branch is off and unsupported otherwise.


def _vm_field(scheme: str, field: str, cast: Callable[[Any], Any], default_at: str) -> Rule:
    """A data.<pkg> key the ``scheme`` parameter record carries as ``field``."""

    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        del ctx
        return {f"{_ASSEMBLY}.{scheme}_{field}": cast(value)}

    return Rule(
        MAP, target=f"model.stepper.kwargs.vertical_mixing_parameters.{field}",
        convert=f"identity (default {default_at})", fn=fn,
    )


def _package_cpp_states(case_dir: Path, package: str) -> tuple[dict[str, bool], str]:
    """Macro states of the <PKG>_OPTIONS.h the build compiles, and its source.

    A case's ``code/`` copy replaces the package header (_gmredi_cpp_states); the
    header is evaluated with ALLOW_<PKG> defined, its body sitting in that #ifdef.
    """
    name = f"{package.upper()}_OPTIONS.h"
    header = case_dir / "code" / name
    source = f"code/{name}"
    if not header.is_file():
        header = _reference_mit_root() / "pkg" / package / name
        source = str(header)
    states = parse_cpp_macro_states(header, predefined=frozenset({f"ALLOW_{package.upper()}"}))
    return states, source


def _pp81_lower_bound(key: str) -> Rule:
    """PPviscMin or PPdiffMin: read always (pp81_readparms.F:35-44), used only under ALLOW_PP81_LOWERBOUND."""

    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        del value
        if not _package_cpp_states(ctx.case_dir, "pp81")[0].get("ALLOW_PP81_LOWERBOUND", False):
            raise Skipped(f"{key} is read only by the ALLOW_PP81_LOWERBOUND block (pp81_calc.F:94-106), "
                          "which the PP81_OPTIONS.h the build compiles leaves undefined")
        raise Gap("PP81 lower bound of the level-2 viscosity and diffusivity (ALLOW_PP81_LOWERBOUND, "
                  "pp81_calc.F:94-106); PP81_CPP_OPTIONS holds no macro")

    return Rule(MAP, target=None, convert="ALLOW_PP81_LOWERBOUND undefined -> no-op, else capability gap",
                fn=fn)


# PP81_PARM01 of pp81_readparms.F:35-44, defaults :64-73.
PP81_RULES: dict[tuple[str, str], Rule] = {
    ("PP81_PARM01", "ppnri"): _vm_field("pp81", "n_ri", int, "pkg/pp81/pp81_readparms.F:64"),
    ("PP81_PARM01", "ppviscmax"): _vm_field("pp81", "visc_max", float, "pkg/pp81/pp81_readparms.F:67"),
    ("PP81_PARM01", "ppnu0"): _vm_field("pp81", "nu0", float, "pkg/pp81/pp81_readparms.F:68"),
    ("PP81_PARM01", "ppalpha"): _vm_field("pp81", "alpha", float, "pkg/pp81/pp81_readparms.F:69"),
    ("PP81_PARM01", "ppviscmin"): _pp81_lower_bound("PPviscMin"),
    ("PP81_PARM01", "ppdiffmin"): _pp81_lower_bound("PPdiffMin"),
    ("PP81_PARM01", "ppdumpfreq"): _skip("PP81 snapshot interval (pp81_output.F:70-71); output only"),
    ("PP81_PARM01", "ppwritestate"): _skip("PP81 snapshot files (pp81_output.F:74); output only"),
    ("PP81_PARM01", "ppmixingmaps"): _skip("stored in PP81.h (PP81.h:53-55); no routine reads it"),
}

# MY_PARM01 of my82_readparms.F:35-41, defaults :61-67.
MY82_RULES: dict[tuple[str, str], Rule] = {
    ("MY_PARM01", "myviscmax"): _vm_field("my82", "visc_max", float, "pkg/my82/my82_readparms.F:61"),
    ("MY_PARM01", "mydiffmax"): _vm_field("my82", "diff_max", float, "pkg/my82/my82_readparms.F:62"),
    ("MY_PARM01", "myhblscale"): _vm_field("my82", "hbl_scale", float, "pkg/my82/my82_readparms.F:63"),
    ("MY_PARM01", "mydumpfreq"): _skip("MY82 snapshot interval (my82_output.F:70-71); output only"),
    ("MY_PARM01", "mywritestate"): _skip("MY82 snapshot files (my82_output.F:74); output only"),
    ("MY_PARM01", "mymixingmaps"): _skip("stored in MY82.h (MY82.h:59-61); no routine reads it"),
}

# KL10_PARM01 of kl10_readparms.F:37-41, defaults :61-64.
KL10_RULES: dict[tuple[str, str], Rule] = {
    ("KL10_PARM01", "klviscmax"): _vm_field("kl10", "visc_max", float, "pkg/kl10/kl10_readparms.F:61"),
    ("KL10_PARM01", "kldumpfreq"): _skip("KL10 snapshot interval (kl10_output.F:73-74); output only"),
    ("KL10_PARM01", "klwritestate"): _skip("KL10 snapshot files (kl10_output.F:77); output only"),
    ("KL10_PARM01", "kltavefreq"): _gap("retired KLtaveFreq: any value stops MITgcm (kl10_readparms.F:89-98)"),
}


def _ggl90_macro(ctx: Ctx, macro: str) -> bool:
    return _package_cpp_states(ctx.case_dir, "ggl90")[0].get(macro, False)


def _ggl90_switch(ctx: Ctx, key: str) -> bool:
    return ctx.namelists.get("data.ggl90", {}).get("GGL90_PARM01", {}).get(key, False) is True


def _f_ggl90_use_langmuir(value: Any, ctx: Ctx) -> dict[str, Any]:
    """useLANGMUIR -> use_langmuir; without ALLOW_GGL90_LANGMUIR MITgcm stops (ggl90_check.F:65-73)."""
    if value is True and not _ggl90_macro(ctx, "ALLOW_GGL90_LANGMUIR"):
        raise Gap("useLANGMUIR without ALLOW_GGL90_LANGMUIR stops MITgcm (ggl90_check.F:65-73)")
    return {f"{_ASSEMBLY}.ggl90_use_langmuir": bool(value)}


def _ggl90_langmuir_field(field: str, default_at: str) -> Rule:
    """A GGL90_PARM03 key, read only with ALLOW_GGL90_LANGMUIR and useLANGMUIR (ggl90_readparms.F:208-227)."""

    def fn(value: Any, ctx: Ctx) -> dict[str, Any]:
        if not (_ggl90_macro(ctx, "ALLOW_GGL90_LANGMUIR") and _ggl90_switch(ctx, "uselangmuir")):
            raise Skipped("GGL90_PARM03 is read only with ALLOW_GGL90_LANGMUIR and useLANGMUIR "
                          "(ggl90_readparms.F:208-227)")
        return {f"{_ASSEMBLY}.ggl90_{field}": float(value)}

    return Rule(
        MAP, target=f"model.stepper.kwargs.vertical_mixing_parameters.{field}",
        convert=f"identity (default {default_at}); unread without ALLOW_GGL90_LANGMUIR and useLANGMUIR",
        fn=fn,
    )


_GGL90_IDEMIX_GAP = (
    "IDEMIX internal-wave energy (ggl90_idemix.F, called at ggl90_calc.F:264; "
    "ggl90_calc.F:565,617,678); the ggl90 port has none"
)


def _ggl90_idemix_parameter(value: Any, ctx: Ctx) -> dict[str, Any]:
    """A GGL90_PARM02 key, read only with ALLOW_GGL90_IDEMIX and useIDEMIX (ggl90_readparms.F:187-206)."""
    del value
    if not (_ggl90_macro(ctx, "ALLOW_GGL90_IDEMIX") and _ggl90_switch(ctx, "useidemix")):
        raise Skipped("GGL90_PARM02 is read only with ALLOW_GGL90_IDEMIX and useIDEMIX "
                      "(ggl90_readparms.F:187-206)")
    raise Gap(_GGL90_IDEMIX_GAP)


def _f_ggl90_tke_file(value: Any, ctx: Ctx) -> dict[str, Any]:
    """GGL90TKEFile: blank is MITgcm's cold start GGL90TKEmin*maskC (ggl90_init_varia.F:52)."""
    del ctx
    if not str(value).strip():
        raise Skipped("blank, the MIT default (ggl90_readparms.F:124): the cold start "
                      "GGL90TKEmin*maskC (ggl90_init_varia.F:52)")
    raise Gap("initial TKE from GGL90TKEFile (ggl90_init_varia.F:135-146); the stepper "
              "cold-starts at GGL90TKEmin*maskC (columns.ggl90_cold_start_tke)")


# GGL90_PARM01 of ggl90_readparms.F:48-57, defaults :104-128; GGL90_PARM02
# (IDEMIX, :63-68) and GGL90_PARM03 (Langmuir, :75-76), each read only under its
# macro and switch.
GGL90_RULES: dict[tuple[str, str], Rule] = {
    ("GGL90_PARM01", "ggl90ck"): _vm_field("ggl90", "ck", float, "pkg/ggl90/ggl90_readparms.F:108"),
    ("GGL90_PARM01", "ggl90ceps"): _vm_field("ggl90", "ceps", float, "pkg/ggl90/ggl90_readparms.F:109"),
    ("GGL90_PARM01", "ggl90alpha"): _vm_field("ggl90", "alpha", float, "pkg/ggl90/ggl90_readparms.F:110"),
    ("GGL90_PARM01", "ggl90m2"): _vm_field("ggl90", "m2", float, "pkg/ggl90/ggl90_readparms.F:112"),
    ("GGL90_PARM01", "ggl90tkemin"): _vm_field(
        "ggl90", "tke_min", float, "pkg/ggl90/ggl90_readparms.F:113"),
    ("GGL90_PARM01", "ggl90tkesurfmin"): _vm_field(
        "ggl90", "tke_surf_min", float, "pkg/ggl90/ggl90_readparms.F:115"),
    ("GGL90_PARM01", "ggl90tkebottom"): _vm_field(
        "ggl90", "tke_bottom", float, "pkg/ggl90/ggl90_readparms.F:116, unset -> GGL90TKEmin at :251-253"),
    ("GGL90_PARM01", "ggl90viscmax"): _vm_field(
        "ggl90", "visc_max", float, "pkg/ggl90/ggl90_readparms.F:117"),
    ("GGL90_PARM01", "ggl90diffmax"): _vm_field(
        "ggl90", "diff_max", float, "pkg/ggl90/ggl90_readparms.F:118"),
    ("GGL90_PARM01", "ggl90mixinglengthmin"): _vm_field(
        "ggl90", "mixing_length_min", float, "pkg/ggl90/ggl90_readparms.F:120"),
    ("GGL90_PARM01", "mxlmaxflag"): _vm_field(
        "ggl90", "mxl_max_flag", int, "pkg/ggl90/ggl90_readparms.F:121"),
    ("GGL90_PARM01", "mxlsurfflag"): _vm_field(
        "ggl90", "mxl_surf_flag", bool, "pkg/ggl90/ggl90_readparms.F:123"),
    ("GGL90_PARM01", "ggl90_dirichlet"): _vm_field(
        "ggl90", "dirichlet", bool, "pkg/ggl90/ggl90_readparms.F:125"),
    ("GGL90_PARM01", "calcmeanvertshear"): _vm_field(
        "ggl90", "calc_mean_vert_shear", bool, "pkg/ggl90/ggl90_readparms.F:126"),
    ("GGL90_PARM01", "uselangmuir"): Rule(
        MAP, target="model.stepper.kwargs.vertical_mixing_parameters.use_langmuir",
        convert="identity (default pkg/ggl90/ggl90_readparms.F:128); .TRUE. without "
                "ALLOW_GGL90_LANGMUIR -> capability gap (MITgcm stops)",
        fn=_f_ggl90_use_langmuir,
    ),
    ("GGL90_PARM01", "ggl90difftkeh"): _zero_or_gap(
        "horizontal TKE diffusion (GGL90diffTKEh, ggl90_calc.F:387-440, :640-650 under "
        "ALLOW_GGL90_HORIZDIFF; ggl90_check.F:47-64); the ggl90 port has none"
    ),
    ("GGL90_PARM01", "useidemix"): _false_or_gap(_GGL90_IDEMIX_GAP),
    ("GGL90_PARM01", "ggl90tkefile"): Rule(
        MAP, target=None, convert="blank -> cold start, else capability gap", fn=_f_ggl90_tke_file,
    ),
    ("GGL90_PARM01", "ggl90tavefreq"): _gap(
        "retired GGL90taveFreq: any value stops MITgcm (ggl90_readparms.F:292-310)"
    ),
    ("GGL90_PARM01", "admxlmaxflag"): _skip(
        "the adjoint mode's mixing-length flag (ggl90_mixinglength.F:111-114 under ALLOW_AUTODIFF "
        "and inAdMode; ggl90_check.F:131-150)"
    ),
    ("GGL90_PARM01", "ggl90dumpfreq"): _skip("GGL90 snapshot interval (ggl90_output.F:70-71); output only"),
    ("GGL90_PARM01", "ggl90writestate"): _skip("GGL90 snapshot files (ggl90_output.F:74); output only"),
    ("GGL90_PARM01", "ggl90mixingmaps"): _skip(
        "printed (ggl90_readparms.F:327-328); no routine reads it"
    ),
    ("GGL90_PARM03", "lc_gamma"): _ggl90_langmuir_field("lc_gamma", "pkg/ggl90/ggl90_readparms.F:159"),
    ("GGL90_PARM03", "lc_num"): _ggl90_langmuir_field("lc_num", "pkg/ggl90/ggl90_readparms.F:160"),
    ("GGL90_PARM03", "lc_lambda"): _ggl90_langmuir_field("lc_lambda", "pkg/ggl90/ggl90_readparms.F:161"),
}
for _key in ("idemix_tau_v", "idemix_tau_h", "idemix_gamma", "idemix_jstar", "idemix_mu0",
             "idemix_tidal_file", "idemix_wind_file", "idemix_mixing_efficiency",
             "idemix_diff_max", "idemix_diff_min", "idemix_frac_f_b", "idemix_frac_f_s",
             "idemix_include_gm", "idemix_include_gm_bottom"):
    GGL90_RULES[("GGL90_PARM02", _key)] = Rule(
        MAP, target=None, convert="unread without ALLOW_GGL90_IDEMIX and useIDEMIX, else capability gap",
        fn=_ggl90_idemix_parameter,
    )


#: data.pkg flag -> (the stepper's vertical_mixing_scheme, the one namelist file
#: its <pkg>_readparms.F opens, that file's table).
_VERTICAL_MIXING_PACKAGES: dict[str, tuple[str, str, dict[tuple[str, str], Rule]]] = {
    "usepp81": ("pp81", "data.pp81", PP81_RULES),  # pp81_readparms.F:80-84
    "usemy82": ("my82", "data.my82", MY82_RULES),  # my82_readparms.F:74-78
    "usekl10": ("kl10", "data.kl10", KL10_RULES),  # kl10_readparms.F:70-74
    "useggl90": ("ggl90", "data.ggl90", GGL90_RULES),  # ggl90_readparms.F:98-101
}


def _vertical_mixing_parameters(
    asm: _Assembly, yaml_tree: dict[str, Any], case_dir: Path, record: Record
) -> None:
    """vertical_mixing_parameters of the scheme use<PKG> selected, after MITgcm's defaults.

    The stepper replaces the mapping by the scheme's record at construction
    (columns.vertical_mixing_parameters), which requires every field and checks
    the values MITgcm's readparms stops on.  A macro the build's <PKG>_OPTIONS.h
    defines outside the port's <PKG>_CPP_OPTIONS is unsupported.
    """
    scheme = asm.yaml("model.stepper.kwargs.vertical_mixing_scheme")
    if scheme is None:
        return
    states, header = _package_cpp_states(case_dir, scheme)
    if scheme == "pp81":
        accepted = PP81_CPP_OPTIONS
        parameters: dict[str, Any] = {
            "nu0": asm.number("pp81_nu0"),
            "alpha": asm.number("pp81_alpha"),
            "n_ri": int(asm.get("pp81_n_ri")),
            "visc_max": asm.number("pp81_visc_max"),
        }
        rationale = "data.pp81 after pp81_readparms.F:64-69"
    elif scheme == "my82":
        accepted = MY82_CPP_OPTIONS
        parameters = {
            "hbl_scale": asm.number("my82_hbl_scale"),
            "visc_max": asm.number("my82_visc_max"),
            "diff_max": asm.number("my82_diff_max"),
        }
        rationale = "data.my82 after my82_readparms.F:61-63"
    elif scheme == "kl10":
        accepted = KL10_CPP_OPTIONS
        parameters = {"visc_max": asm.number("kl10_visc_max")}
        rationale = "data.kl10 after kl10_readparms.F:61"
    elif scheme == "ggl90":
        accepted = GGL90_CPP_OPTIONS
        tke_min = asm.number("ggl90_tke_min")
        tke_bottom = asm.staged("ggl90_tke_bottom")
        parameters = {
            "ck": asm.number("ggl90_ck"),
            "ceps": asm.number("ggl90_ceps"),
            "alpha": asm.number("ggl90_alpha"),
            "m2": asm.number("ggl90_m2"),
            "tke_min": tke_min,
            "tke_surf_min": asm.number("ggl90_tke_surf_min"),
            # ggl90_readparms.F:116, :251-253
            "tke_bottom": tke_min if tke_bottom is None else float(tke_bottom),
            "visc_max": asm.number("ggl90_visc_max"),
            "diff_max": asm.number("ggl90_diff_max"),
            "mixing_length_min": asm.number("ggl90_mixing_length_min"),
            "mxl_max_flag": int(asm.get("ggl90_mxl_max_flag")),
            "mxl_surf_flag": asm.flag("ggl90_mxl_surf_flag"),
            "dirichlet": asm.flag("ggl90_dirichlet"),
            "calc_mean_vert_shear": asm.flag("ggl90_calc_mean_vert_shear"),
            "use_langmuir": asm.flag("ggl90_use_langmuir"),
            "lc_gamma": asm.number("ggl90_lc_gamma"),
            "lc_num": asm.number("ggl90_lc_num"),
            "lc_lambda": asm.number("ggl90_lc_lambda"),
            # its one use, ggl90_calc.F:677-685
            "missing_hfac_bug": states.get("GGL90_MISSING_HFAC_BUG", False),
        }
        rationale = ("data.ggl90 after ggl90_readparms.F:104-161, GGL90TKEbottom unset -> GGL90TKEmin "
                     "(:251-253); missing_hfac_bug from the GGL90_OPTIONS.h the build compiles")
    outside = sorted(
        macro for macro, on in states.items()
        if on and macro not in accepted
        and macro not in (f"{scheme.upper()}_OPTIONS_H", f"ALLOW_{scheme.upper()}")
    )
    if outside:
        _record_gap(
            record,
            {"source": header, "parameter": f"<derived:{scheme.upper()}_OPTIONS.h>", "value": outside},
            f"{outside} defined; the {scheme} port accepts {scheme.upper()}_CPP_OPTIONS {accepted}",
        )
    _set_dotted(yaml_tree, "model.stepper.kwargs.vertical_mixing_parameters", parameters)
    record.derived.append({
        "source": f"input/data.pkg + input/data.{scheme}",
        "target": "model.stepper.kwargs.vertical_mixing_parameters",
        "value": dict(parameters),
        "rationale": rationale,
    })
