"""Resolve explicitly configured external inputs; importing performs no I/O."""
from __future__ import annotations

import os
import glob
from pathlib import Path
import re
import subprocess
import tomllib


ENV_NAMES = {
    "mitgcm_root": ("MITGCM_ROOT", "MITCASES_MIT_ROOT", "RPAIR_MIT_ROOT"),
    "mitgcm_pin": ("MITGCM_PIN", "MITCASES_MIT_PIN", "RPAIR_MIT_PIN"),
    "mitgcm_ad_root": ("MITGCM_AD_ROOT", "CS32_MIT_ROOT"),
    "mitgcm_ad_pin": ("MITGCM_AD_PIN", "CS32_MIT_PIN"),
    "cs32_grid_dir": ("DEFGCM_CS32_GRID_DIR",),
    "cs32_input_dir": ("DEFGCM_CS32_INPUT_DIR",),
    "conda_root": ("MITGCM_CONDA_ROOT",),
    "mitgcm_optfile": ("MITGCM_OPTFILE", "MITCASES_OPTFILE", "RPAIR_OPTFILE"),
    "hycom_dir": ("DEFGCM_HYCOM_DIR",),
    "era5_dir": ("DEFGCM_ERA5_DIR",),
    "gebco_file": ("DEFGCM_GEBCO_FILE",),
    "gebco_scs_file": ("DEFGCM_GEBCO_SCS_FILE",),
    "topo_file": ("DEFGCM_TOPO_FILE",),
    "tpxo9_dir": ("DEFGCM_TPXO9_DIR",),
    "global_data_dir": ("DEFGCM_GLOBAL_DATA_DIR",),
    "global_forecast_dir": ("DEFGCM_GLOBAL_FORECAST_DIR",),
    "scs_clm_dir": ("DEFGCM_SCS_CLM_DIR",),
    "cube_cs192_dir": ("DEFGCM_CUBE_CS192_DIR",),
    "cube_cs1024_dir": ("DEFGCM_CUBE_CS1024_DIR",),
    "synthetic_pureocean_dir": ("DEFGCM_SYNTHETIC_PUREOCEAN_DIR",),
}
SOURCE_PINS = {"mitgcm_root": "mitgcm_pin", "mitgcm_ad_root": "mitgcm_ad_pin"}
DATA_KEYS = frozenset(key for key in ENV_NAMES if ENV_NAMES[key][0].startswith("DEFGCM_")
                      and not key.startswith("cs32_"))


class ExternalPathError(ValueError):
    """An external path or its declared source version is missing or invalid."""


def _configuration() -> dict[str, str]:
    filename = os.environ.get("DEFGCM_PATHS")
    if not filename:
        return {}
    path = Path(filename).expanduser()
    if not path.is_absolute():
        raise ExternalPathError("DEFGCM_PATHS must be an absolute path")
    try:
        with path.open("rb") as stream:
            config = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ExternalPathError(f"cannot read DEFGCM_PATHS={path}: {error}") from error
    if type(config.get("schema_version")) is not int or config["schema_version"] != 1 or set(config) - {"schema_version", "paths"}:
        raise ExternalPathError("DEFGCM_PATHS requires schema_version=1 and [paths]")
    values = config.get("paths", {})
    if not isinstance(values, dict) or set(values) - set(ENV_NAMES):
        raise ExternalPathError("unknown [paths] key; expected " + ", ".join(ENV_NAMES))
    for key, value in values.items():
        if not isinstance(value, str) or not value.strip():
            raise ExternalPathError(f"[paths].{key} must be a nonempty string")
        if not key.endswith("_pin") and not Path(value).expanduser().is_absolute():
            raise ExternalPathError(f"[paths].{key} must be an absolute path")
    for root, pin in SOURCE_PINS.items():
        if (root in values) != (pin in values):
            raise ExternalPathError(f"[paths].{root} and [paths].{pin} must be configured together")
    return values


def _environment(key: str) -> str | None:
    values = [(name, os.environ[name]) for name in ENV_NAMES[key] if os.environ.get(name)]
    if key == "mitgcm_root" and os.environ.get("MITCASES_VERIFICATION_ROOT"):
        verification = Path(os.environ["MITCASES_VERIFICATION_ROOT"]).expanduser().resolve()
        if verification.name != "verification":
            raise ExternalPathError("MITCASES_VERIFICATION_ROOT must name the source's verification directory")
        values.append(("MITCASES_VERIFICATION_ROOT", str(verification.parent)))
    normalized = {value.lower() if key.endswith("_pin") else str(Path(value).expanduser().resolve()) for _, value in values}
    if len(normalized) > 1:
        raise ExternalPathError(f"conflicting environment aliases for {key}: " + ", ".join(name for name, _ in values))
    return values[0][1] if values else None


def _verify_source(root: Path, expected: str, key: str) -> None:
    observed = []
    marker = root / "PIN"
    if marker.is_file():
        observed.append(("PIN", marker.read_text(encoding="ascii").strip().lower()))
    git = root / ".git"
    if git.exists() or git.is_symlink():
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel", "HEAD"],
            check=False, capture_output=True, text=True,
        )
        lines = result.stdout.splitlines()
        if result.returncode or len(lines) != 2 or Path(lines[0]).resolve() != root:
            raise ExternalPathError(f"cannot identify the Git checkout for {key}: {root}")
        observed.append(("Git HEAD", lines[1].lower()))
    if not observed:
        raise ExternalPathError(f"{key} needs a PIN file or its own Git checkout: {root}")
    for kind, actual in observed:
        if actual != expected:
            raise ExternalPathError(f"{key} {kind} {actual!r} does not match configured {SOURCE_PINS[key]}={expected}")


def resolve(key: str, value: str | Path | None = None, *, validate: bool = True) -> Path | str:
    """Resolve an explicit argument, legacy environment setting, or TOML key.

    Root/pin pairs are always required. validate=False is only for metadata-only
    plans; actual consumers validate existence and source identity. Environment
    variables take precedence over TOML. TOML paths must be absolute so freezing
    or relocating the configuration does not change their meaning.
    """
    # These are derived tool paths, not additional TOML keys.
    if key == "mitgcm_env_dir":
        direct = value or os.environ.get("MITCASES_MIT_ENV")
        if direct:
            path = Path(direct).expanduser().resolve()
        else:
            path = resolve("conda_root", validate=validate) / "envs" / os.environ.get("MITGCM_CONDA_ENV_NAME", "mitgcm-intel")
        if validate and not path.is_dir():
            raise ExternalPathError(f"missing mitgcm_env_dir: {path}; configure conda_root or MITCASES_MIT_ENV")
        return path
    if key == "conda_exe":
        direct = value or os.environ.get("CONDA_EXE")
        path = Path(direct).expanduser().resolve() if direct else resolve("conda_root", validate=validate) / "bin" / "conda"
        if validate and not path.is_file():
            raise ExternalPathError(f"missing conda executable: {path}; configure conda_root or CONDA_EXE")
        return path
    if key not in ENV_NAMES:
        raise ExternalPathError(f"unknown external path key {key!r}")
    raw = str(value) if value is not None else _environment(key)
    if raw is None:
        raw = _configuration().get(key)
    if raw is None:
        raise ExternalPathError(f"missing external path key {key}; set {', '.join(ENV_NAMES[key])} or [paths].{key} in DEFGCM_PATHS")
    if key.endswith("_pin"):
        if not re.fullmatch(r"[0-9a-fA-F]{40}", raw):
            raise ExternalPathError(f"{key} must be a full 40-character Git commit")
        return raw.lower()
    path = Path(raw).expanduser()
    if key in DATA_KEYS and not path.is_absolute():
        raise ExternalPathError(f"{key} must be an absolute path; set {', '.join(ENV_NAMES[key])} or [paths].{key} in DEFGCM_PATHS")
    path = path.resolve()
    expected = resolve(SOURCE_PINS[key], validate=validate) if key in SOURCE_PINS else None
    if validate:
        present = path.is_file() if key in {"mitgcm_optfile", "gebco_file", "gebco_scs_file", "topo_file"} else path.is_dir()
        if not present:
            raise ExternalPathError(f"configured {key} does not exist: {path}")
        if expected is not None:
            _verify_source(path, expected, key)
    return path


def resolve_input_path(value: str | Path) -> Path:
    """Resolve ``external:KEY[/child]`` or an absolute raw-input path.

    Relative paths are rejected, including relative environment overrides.
    File globs are allowed only on an absolute path or below a named directory.
    This performs only path/metadata checks, before a reader opens any data.
    """
    raw = str(value)
    if raw.startswith("external:"):
        key, _, child = raw.removeprefix("external:").partition("/")
        if key.endswith("_pin"):
            raise ExternalPathError(f"external input needs a path key, got {key!r}")
        root = resolve(key)
        if child and (Path(child).is_absolute() or ".." in Path(child).parts):
            raise ExternalPathError(f"external input must stay below {key}: {child!r}")
        if child and not root.is_dir():
            raise ExternalPathError(f"external input {key} is not a directory: {root}")
        path = root / child if child else root
    else:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            raise ExternalPathError(
                f"raw input path must be absolute or external:KEY[/child], got {raw!r}; "
                "configure the key in DEFGCM_PATHS or its environment variable"
            )
    if not path.exists() and not (glob.has_magic(str(path)) and glob.glob(str(path))):
        raise ExternalPathError(f"configured raw input does not exist: {path}")
    return path


def resolve_input_references(value):
    """Resolve named references after all configuration overlays are merged."""
    if isinstance(value, dict):
        return {key: resolve_input_references(item) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_input_references(item) for item in value]
    if isinstance(value, str) and value.startswith("external:"):
        return str(resolve_input_path(value))
    return value


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("key", choices=[*ENV_NAMES, "mitgcm_env_dir", "conda_exe"])
    parser.add_argument("--value", help="Explicit caller path; a source root still requires its configured pin")
    args = parser.parse_args(argv)
    try:
        print(resolve(args.key, args.value))
    except ExternalPathError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
