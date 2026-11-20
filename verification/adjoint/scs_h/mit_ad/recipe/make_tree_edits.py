"""Derive the edited MITgcm files of the regional adjoint build from a MITgcm tree, with exact-match checks.

Every edit names the exact source line it changes and refuses to run if that line is not found the
expected number of times, so a different tree fails loudly instead of building something else.

Edits and their numerical purpose:
  model/src/ini_pressure.F     drop the two `#ifndef ALLOW_AUTODIFF` guards (lines 41-50 and 76-183)
                               that compile the Picard iteration out of every AD build; without them
                               the AD build starts from totPhiHyd = 0 instead of the INI_PRESSURE seed.
  model/src/ini_fields.F       remove CALL INI_PRESSURE (line 33) ...
  model/src/initialise_varia.F ... and call it right after PACKAGES_INIT_VARIABLES (line 263), i.e.
                               after CTRL_INIT_VARIABLES added xx_theta/xx_salt and OBCS_INIT_VARIABLES
                               wrote the open-boundary ring, as LOCOMgcm seeds from the bounded state.
  pkg/ctrl/CTRL_OPTIONS.h      GENARR2D, GENTIM2D and the four OBCS controls off; GENARR3D stays on,
                               BOTTOMDRAG stays off.
  pkg/ctrl/CTRL_SIZE.h         maxCtrlArr3D 1 -> 2 (xx_theta and xx_salt).
  pkg/autodiff/AUTODIFF_OPTIONS.h  TAMC, TAMC checkpointing, adjoint monitor diagnostics and TAF
                               tape I/O off.

usage: python make_tree_edits.py <MITgcm tree> <output code dir>
Writes the edited files into <output code dir> and a unified diff of all edits to
<output code dir>/../tree_edits.diff.
"""
import difflib
import sys
from pathlib import Path

TREE = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
diffs = []


def edit(rel, fn):
    src = (TREE / rel).read_text()
    new = fn(src)
    assert new != src, rel
    (OUT / Path(rel).name).write_text(new)
    diffs.extend(difflib.unified_diff(src.splitlines(keepends=True), new.splitlines(keepends=True),
                                      fromfile=f"tree/{rel}", tofile=f"code/{Path(rel).name}"))


def replace_once(text, old, new, count=1):
    assert text.count(old) == count, (old, text.count(old))
    return text.replace(old, new)


def ini_pressure(text):
    lines = text.splitlines(keepends=True)
    guard = [i for i, line in enumerate(lines) if line.rstrip() == "#ifndef ALLOW_AUTODIFF"]
    assert len(guard) == 2, guard
    decl_end = next(i for i in range(guard[0] + 1, len(lines)) if lines[i].startswith("#"))
    assert lines[decl_end].rstrip() == "#endif", (decl_end, lines[decl_end])
    loop_end = [i for i, line in enumerate(lines) if line.rstrip() == "#endif /* ALLOW_AUTODIFF */"]
    assert len(loop_end) == 1 and loop_end[0] > guard[1], loop_end
    drop = {guard[0], decl_end, guard[1], loop_end[0]}
    # One comment line in place of the first guard, so the reason sits in the file.
    note = ("C     the ALLOW_AUTODIFF guards around the Picard\n"
            "C     iteration are removed, so the AD build seeds totPhiHyd as the\n"
            "C     forward build does; Tapenade tapes the iterations itself.\n")
    out = []
    for i, line in enumerate(lines):
        if i == guard[0]:
            out.append(note)
        if i not in drop:
            out.append(line)
    return "".join(out)


def ini_fields(text):
    return replace_once(
        text, "        CALL INI_PRESSURE( myThid )\n",
        "C       INI_PRESSURE moved to INITIALISE_VARIA,\n"
        "C       after PACKAGES_INIT_VARIABLES (control and OBCS initial values).\n")


def initialise_varia(text):
    anchor = "      CALL PACKAGES_INIT_VARIABLES( myThid )\n"
    return replace_once(
        text, anchor, anchor +
        "\n"
        "C--   step-zero hydrostatic seed, moved here from\n"
        "C     INI_FIELDS.  It now sees theta and salt after CTRL_INIT_VARIABLES added\n"
        "C     xx_theta, xx_salt and OBCS_INIT_VARIABLES wrote the open-boundary ring,\n"
        "C     as LOCOMgcm seeds from the bounded initial state.  Same condition\n"
        "C     as the call it replaces (ini_fields.F:27-28).\n"
        "      IF ( startTime .EQ. baseTime .AND.  nIter0 .EQ. 0\n"
        "     &     .AND. pickupSuff .EQ. ' ' ) THEN\n"
        "        CALL INI_PRESSURE( myThid )\n"
        "      ENDIF\n")


def ctrl_options(text):
    text = replace_once(text, "#define ALLOW_GENARR2D_CONTROL\n", "#undef ALLOW_GENARR2D_CONTROL\n")
    text = replace_once(text, "#define ALLOW_GENTIM2D_CONTROL\n", "#undef ALLOW_GENTIM2D_CONTROL\n")
    for side in "NSEW":
        text = replace_once(text, f"# define ALLOW_OBCS{side}_CONTROL\n", f"# undef ALLOW_OBCS{side}_CONTROL\n")
    assert text.count("#define ALLOW_GENARR3D_CONTROL\n") == 1
    assert text.count("#undef ALLOW_BOTTOMDRAG_CONTROL\n") == 1
    return text


def ctrl_size(text):
    return replace_once(text, "      PARAMETER ( maxCtrlArr3D = 1 )\n", "      PARAMETER ( maxCtrlArr3D = 2 )\n")


def autodiff_options(text):
    for name in ("ALLOW_AUTODIFF_TAMC", "ALLOW_TAMC_CHECKPOINTING", "ALLOW_AUTODIFF_MONITOR_DIAG",
                 "ALLOW_AUTODIFF_WHTAPEIO"):
        text = replace_once(text, f"#define {name}\n", f"#undef {name}\n")
    return text


edit("model/src/ini_pressure.F", ini_pressure)
edit("model/src/ini_fields.F", ini_fields)
edit("model/src/initialise_varia.F", initialise_varia)
edit("pkg/ctrl/CTRL_OPTIONS.h", ctrl_options)
edit("pkg/ctrl/CTRL_SIZE.h", ctrl_size)
edit("pkg/autodiff/AUTODIFF_OPTIONS.h", autodiff_options)
(OUT.parent / "tree_edits.diff").write_text("".join(diffs))
print(f"wrote {len(diffs)} diff lines to {OUT.parent / 'tree_edits.diff'}")
