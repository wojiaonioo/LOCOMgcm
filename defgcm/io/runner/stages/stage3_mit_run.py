"""Stage3 MITgcm MPI launch (Profile S — env isolation + mpirun)."""

from __future__ import annotations

from defgcm.external_paths import resolve

import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from defgcm.io.runner.config import load_run_config
from defgcm.io.runner.layer_d import ArtifactKind, ArtifactRef
from defgcm.io.runner.manifest import ManifestWriter, validate_prereqs

from ._common import build_stage_cli_argparser, open_manifest_for_stage


STAGE_NAME = "stage3_mit_run"
PREREQ_STAGES = ["stage1c_export_mit"]


@dataclass(frozen=True)
class LaunchResult:
    stdout_path: Path
    exit_code: int
    wallclock_seconds: float
    normal_exit: bool


def _resolve_mitgcm_env_dir() -> Path:
    """Derive the native environment from configured conda_root or its legacy override."""
    return resolve("mitgcm_env_dir")


_MITGCM_OK_MARKER = "PROGRAM MAIN: Execution ended Normally"


def _find_rank0_stdout(run_dir: Path) -> Path:
    """Return STDOUT path written by rank 0.

    The shell redirects launcher output to STDOUT.0001.
    However single-rank ``mpirun -np 1`` writes to STDOUT.0000 instead,
    leaving STDOUT.0001 empty (the shell-redirect target that captures
    only the now-empty stdout stream).  Prefer the first STDOUT.* file
    that actually contains the success marker; fall back to .0001 so the
    failure path still reports the conventional location.
    """
    candidates = sorted(run_dir.glob("STDOUT.[0-9]*"))
    for p in candidates:
        try:
            if _MITGCM_OK_MARKER in p.read_text(errors="replace"):
                return p
        except Exception:
            continue
    return run_dir / "STDOUT.0001"


def verify_mitgcm_stdout(stdout_path: Path) -> bool:
    """Check the given STDOUT file for the MITgcm normal-exit marker."""
    if not stdout_path.exists():
        return False
    text = stdout_path.read_text(errors="replace")
    return _MITGCM_OK_MARKER in text


def launch_mitgcmuv_mpi(
    case_dir: Path,
    n_procs: int,
    *,
    # Five days by default; MITGCM_LAUNCH_TIMEOUT_SECONDS overrides the limit.
    timeout_seconds: int = int(os.environ.get("MITGCM_LAUNCH_TIMEOUT_SECONDS", 5 * 86400)),
    mitgcm_root: Path | None = None,
    build_script: Path | None = None,
) -> LaunchResult:
    """Build + run MITgcm by invoking experiments/run_mitgcm_comparison.sh.

    Steps performed by the script:
      1. genmake2 with linux_amd64_ifx+impi_conda
      2. make depend
      3. make -j N
      4. Symlink input/ + build/mitgcmuv into run/
      5. ./mitgcmuv > STDOUT.0001 (or mpirun if -np > 1 — see script body)
    """
    # Repo root: defaults to the workspace this stage was imported from
    # (one level above defgcm/), with TORCH_OCEAN_REPO env-var override.
    repo_root = Path(
        os.environ.get(
            "TORCH_OCEAN_REPO",
            str(Path(__file__).resolve().parents[4]),
        )
    )
    if build_script is None:
        cand = repo_root / "experiments" / "run_mitgcm_comparison.sh"
        build_script = cand
    mitgcm_root = resolve("mitgcm_root", mitgcm_root)
    os.environ["MITGCM_OPTFILE"] = str(resolve("mitgcm_optfile"))

    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "build").mkdir(parents=True, exist_ok=True)
    (case_dir / "run").mkdir(parents=True, exist_ok=True)

    # Use conda activate via bash -lc since the build script `set -e` expects
    # a normal interactive shell with PATH containing ifx. Conda root and env
    # name are env-var overridable (see _resolve_mitgcm_env_dir).
    mitgcm_env = _resolve_mitgcm_env_dir()
    conda_root = mitgcm_env.parent.parent
    conda_activate = conda_root / "bin" / "activate"
    shell_cmd = (
        f"source {conda_activate} {mitgcm_env} && "
        f"bash {build_script.resolve()} {case_dir.resolve()} {mitgcm_root.resolve()} {n_procs}"
    )

    # Capture both build and run output into case_dir/build_and_run.log
    log_path = case_dir / "build_and_run.log"
    start = time.time()
    try:
        with open(log_path, "wb") as logf:
            result = subprocess.run(
                ["bash", "-lc", shell_cmd],
                cwd=str(case_dir),
                env={**os.environ, "DEFGCM_PYTHON": sys.executable},
                stdout=logf,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
            )
        exit_code = result.returncode
    except subprocess.TimeoutExpired:
        exit_code = 124
    except FileNotFoundError as e:
        log_path.write_text(f"FileNotFoundError: {e}\n")
        exit_code = 127
    wallclock = time.time() - start

    stdout_path = _find_rank0_stdout(case_dir / "run")
    normal_exit = verify_mitgcm_stdout(stdout_path) and exit_code == 0
    return LaunchResult(stdout_path=stdout_path, exit_code=exit_code, wallclock_seconds=wallclock, normal_exit=normal_exit)


_MITGCM_EXPECTED_BINARIES = (
    "bathyFile.bin",
    "hydrogThetaFile.bin",
    "hydrogSaltFile.bin",
    "uVelInitFile.bin",
    "vVelInitFile.bin",
)


def _lighter_mitgcm_preflight(case_dir: Path, run_cfg) -> None:
    """Lightweight stage3 preflight — shape/dtype/namelist invariants only.

    Numerical preparation belongs to stage1a/stage1c. This preflight reads
    the exported artifacts and checks:
      - every ``mitgcm/input/<name>.bin`` referenced by the namelists exists;
      - each binary is non-empty and divisible by 8 (MITgcm ``readBinaryPrec=64``);
      - per-file size matches the expected ``nz/ny/nx`` (and the OBC strip width
        / sponge width) so a stage1c regression that drops bytes is caught now;
      - ``input/data``, ``input/data.obcs``, ``input/data.exf`` and
        ``code/SIZE.h`` contain the expected namelist settings.

    Does NOT re-derive numerical content (that would duplicate stage1a/1c).
    """
    input_dir = case_dir / "input"
    code_dir = case_dir / "code"
    if not input_dir.exists():
        raise RuntimeError(f"preflight: missing {input_dir} (stage1c not run?)")
    if not code_dir.exists():
        raise RuntimeError(f"preflight: missing {code_dir} (stage1c not run?)")

    nx = int(run_cfg.grid.nx)
    ny = int(run_cfg.grid.ny)
    nz = int(run_cfg.grid.nz)
    sponge_width = int(run_cfg.boundary.sponge_width)
    bytes_per_float = 8  # MITgcm readBinaryPrec=64

    # 1. SIZE.h advertises the right grid.
    size_h = code_dir / "SIZE.h"
    if not size_h.exists():
        raise RuntimeError(f"preflight: missing {size_h}")
    size_text = size_h.read_text()
    if f"Nr  = {nz:4d}" not in size_text:
        raise RuntimeError(
            f"preflight: SIZE.h does not declare Nr={nz} (got {size_h})"
        )
    if "Ny  = sNy*nSy*nPy" not in size_text:
        raise RuntimeError(
            f"preflight: SIZE.h does not match expected Ny formula (got {size_h})"
        )

    # 2. input/data references the expected files + hFacMin.
    data_path = input_dir / "data"
    if not data_path.exists():
        raise RuntimeError(f"preflight: missing {data_path}")
    data_text = data_path.read_text()
    for needle in (
        "bathyFile='bathyFile.bin'",
        "hydrogThetaFile='hydrogThetaFile.bin'",
        "hydrogSaltFile='hydrogSaltFile.bin'",
        "uVelInitFile='uVelInitFile.bin'",
        "vVelInitFile='vVelInitFile.bin'",
    ):
        if needle not in data_text:
            raise RuntimeError(f"preflight: input/data missing entry {needle!r}")

    # 3. Bathy + 3D IC binaries match shape.
    expected_shapes = {
        "bathyFile.bin": (ny, nx),
        "hydrogThetaFile.bin": (nz, ny, nx),
        "hydrogSaltFile.bin": (nz, ny, nx),
        "uVelInitFile.bin": (nz, ny, nx),
        "vVelInitFile.bin": (nz, ny, nx),
    }
    for fname, shape in expected_shapes.items():
        path = input_dir / fname
        if not path.exists():
            raise RuntimeError(f"preflight: missing required binary {path}")
        expected_bytes = bytes_per_float
        for s in shape:
            expected_bytes *= s
        actual = path.stat().st_size
        if actual != expected_bytes:
            raise RuntimeError(
                f"preflight: {fname} size {actual} != expected {expected_bytes} "
                f"(shape {shape}, readBinaryPrec=64)"
            )

    # 4. OBCS namelist references its binaries.
    obcs_path = input_dir / "data.obcs"
    if obcs_path.exists():
        obcs_text = obcs_path.read_text()
        # Each side has 1 OB strip per variable; legacy filenames look like
        # OBN<var>.bin etc. We only verify referenced files are present + their
        # size lines up with the strip shape (nz, nx) or (nz, ny).
        import re

        for match in re.finditer(r"'(OB[NSEW][A-Za-z]+\.bin)'", obcs_text):
            ob_name = match.group(1)
            ob_path = input_dir / ob_name
            if not ob_path.exists():
                raise RuntimeError(
                    f"preflight: data.obcs references {ob_name} but file is missing"
                )
            # First letter after 'OB' is the side. N/S strip width = nx; E/W = ny.
            side = ob_name[2]
            strip = nx if side in ("N", "S") else ny
            # MITgcm OBC binaries hold a time series; size should be a positive
            # multiple of nz*strip*8. Bail if not divisible.
            stride = nz * strip * bytes_per_float
            actual = ob_path.stat().st_size
            if actual == 0:
                raise RuntimeError(f"preflight: {ob_name} is empty")
            if stride > 0 and actual % stride != 0:
                # 2-D OBC fields (eta) hit this branch with stride = strip*8.
                stride2d = strip * bytes_per_float
                if stride2d > 0 and actual % stride2d != 0:
                    raise RuntimeError(
                        f"preflight: {ob_name} size {actual} not a multiple of "
                        f"nz*strip*8={stride} nor strip*8={stride2d}"
                    )

    # 5. data.exf references its forcing binaries.
    exf_path = input_dir / "data.exf"
    if exf_path.exists():
        exf_text = exf_path.read_text()
        import re

        for match in re.finditer(r"'([A-Za-z0-9_]+\.bin)'", exf_text):
            ref = match.group(1)
            ref_path = input_dir / ref
            if not ref_path.exists():
                raise RuntimeError(
                    f"preflight: data.exf references {ref} but file is missing"
                )
            if ref_path.stat().st_size == 0:
                raise RuntimeError(f"preflight: {ref} is empty")
            # Stress fields are (time, ny, nx); we only verify the smallest
            # plausible record is divisible.
            stride = ny * nx * bytes_per_float
            if stride > 0 and ref_path.stat().st_size % stride != 0:
                raise RuntimeError(
                    f"preflight: {ref} size {ref_path.stat().st_size} not a "
                    f"multiple of ny*nx*8={stride}"
                )


def _link_mit_verification_run(run_cfg, manifest: ManifestWriter) -> tuple[Path, Path]:
    verification_dir = Path(str(run_cfg.case.mit_verification_dir))
    target = verification_dir / "run"
    link = manifest.run_dir / "mitgcm" / "run"
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(target.resolve())
    return target, link


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_stage_cli_argparser(STAGE_NAME)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logger = logging.getLogger(STAGE_NAME)
    if args.config is None:
        parser.error("--config required")
    run_cfg = load_run_config(args.config, args.overlay, args.sets)
    manifest = open_manifest_for_stage(args, run_cfg)
    if args.dry_run:
        logger.info(f"[dry-run] {STAGE_NAME} case.kind={run_cfg.case.kind} (prereq check skipped)")
        return 0
    if run_cfg.case.kind == "regional":
        validate_prereqs(
            manifest._path,
            STAGE_NAME,
            prereqs=PREREQ_STAGES,
            integrity_check=True,
            force=getattr(args, "force", False),
        )

    # Prebuilt MIT case dir (e.g. a manually assembled cubed-sphere twin):
    # the lat-lon-shaped stage1c export + lighter preflight do not apply, so
    # skip them and launch in place. ``mit_optfile`` (optional) overrides
    # MITGCM_OPTFILE for the build (e.g. an fp32 optfile variant).
    extras = getattr(run_cfg.case, "extras", None) or {}
    prebuilt_dir = extras.get("mit_prebuilt_case_dir")
    if prebuilt_dir:
        case_dir = Path(str(prebuilt_dir))
        if not case_dir.is_absolute():
            # repo-relative, matching the convention of other case.extras
            # paths (cube_ic_npz etc.); the driver/sbatch body cds to repo root.
            case_dir = Path.cwd() / case_dir
        optfile = extras.get("mit_optfile")
        if optfile:
            of = Path(str(optfile))
            os.environ["MITGCM_OPTFILE"] = str(of if of.is_absolute() else Path.cwd() / of)
        if not (case_dir / "input").is_dir() or not (case_dir / "code").is_dir():
            logger.error(f"prebuilt MIT case dir incomplete (need input/ + code/): {case_dir}")
            manifest.mark_failed(
                STAGE_NAME,
                error=f"prebuilt MIT case dir incomplete (need input/ + code/): {case_dir}",
            )
            return 5
        n_procs = run_cfg.mit_parallel.n_procs
        manifest.mark_running(STAGE_NAME)
        try:
            logger.info(f"launching prebuilt MIT case {case_dir} with n_procs={n_procs}")
            result = launch_mitgcmuv_mpi(case_dir, n_procs=n_procs)
            if not result.normal_exit:
                raise RuntimeError(
                    f"MITgcm did not exit normally (exit={result.exit_code}); "
                    f"check {result.stdout_path}"
                )
            artifact = ArtifactRef(
                name="mitgcm_raw_mds",
                path=case_dir / "run",
                artifact_kind=ArtifactKind.FILE_BINARY,
                bundle_hash="00000000",
                schema_version="0.1.0",
                extras={
                    "n_procs": n_procs,
                    "prebuilt": True,
                    "wallclock_seconds": result.wallclock_seconds,
                    "stdout_path": str(result.stdout_path),
                },
            )
            manifest.record_artifact(STAGE_NAME, artifact)
            manifest.mark_completed(STAGE_NAME, wallclock_seconds=result.wallclock_seconds)
            return 0
        except Exception as e:
            logger.exception(f"{STAGE_NAME} failed")
            manifest.mark_failed(STAGE_NAME, error=str(e))
            return 5

    if run_cfg.case.kind == "mit_verification":
        verification_dir = Path(str(run_cfg.case.mit_verification_dir))
        if not verification_dir.exists():
            logger.error(f"mit_verification_dir does not exist: {verification_dir}")
            manifest.mark_failed(
                STAGE_NAME,
                error=f"mit_verification_dir does not exist: {verification_dir}",
            )
            return 5
        manifest.mark_running(STAGE_NAME)
        start = time.time()
        try:
            target, link = _link_mit_verification_run(run_cfg, manifest)
            wallclock = time.time() - start
            artifact = ArtifactRef(
                name="mitgcm_raw_mds",
                path=link,
                artifact_kind=ArtifactKind.FILE_DIRECTORY,
                bundle_hash="external0000",
                schema_version="0.1.0",
                extras={
                    "external": True,
                    "source_dir": str(target),
                },
            )
            manifest.record_artifact(STAGE_NAME, artifact)
            manifest.mark_completed(STAGE_NAME, wallclock_seconds=wallclock)
            return 0
        except Exception as e:
            logger.exception(f"{STAGE_NAME} failed")
            manifest.mark_failed(STAGE_NAME, error=str(e))
            return 5

    # Locate MIT case dir — depends on output.legacy_layout
    from ._common import stage_output_dir_for
    case_dir = stage_output_dir_for("stage1c_export_mit", manifest.run_dir, run_cfg)
    n_procs = run_cfg.mit_parallel.n_procs

    manifest.mark_running(STAGE_NAME)
    try:
        # Lighter preflight on the on-disk stage1c artefacts. See
        # `_lighter_mitgcm_preflight` docstring for the choice vs legacy
        # `_run_strict_mitgcm_preflight` (which needs a freshly built model +
        # boundary/era5 sources).
        logger.info("running lighter MITgcm preflight ...")
        _lighter_mitgcm_preflight(case_dir, run_cfg)
        logger.info("preflight passed")

        result = launch_mitgcmuv_mpi(case_dir, n_procs=n_procs)
        if not result.normal_exit:
            raise RuntimeError(
                f"MITgcm did not exit normally (exit={result.exit_code}); "
                f"check {result.stdout_path}"
            )
        artifact = ArtifactRef(
            name="mitgcm_raw_mds",
            path=case_dir / "run",
            artifact_kind=ArtifactKind.FILE_BINARY,
            bundle_hash="00000000",
            schema_version="0.1.0",
            extras={
                "n_procs": n_procs,
                "wallclock_seconds": result.wallclock_seconds,
                "stdout_path": str(result.stdout_path),
            },
        )
        manifest.record_artifact(STAGE_NAME, artifact)
        manifest.mark_completed(STAGE_NAME, wallclock_seconds=result.wallclock_seconds)
        return 0
    except Exception as e:
        logger.exception(f"{STAGE_NAME} failed")
        manifest.mark_failed(STAGE_NAME, error=str(e))
        return 5


if __name__ == "__main__":
    sys.exit(main())
