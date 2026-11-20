from .base import ExperimentAdapter, StandardGrid
from .exp4 import Exp4Adapter
from .hycom import HycomAdapter
from .internal_wave import InternalWaveAdapter

ADAPTER_REGISTRY: list[type[ExperimentAdapter]] = [
    HycomAdapter,
    InternalWaveAdapter,
    Exp4Adapter,
]


def detect_adapter(experiment_dir) -> ExperimentAdapter:
    import yaml
    from pathlib import Path

    meta_path = Path(experiment_dir) / "metadata.yaml"
    metadata = yaml.safe_load(meta_path.read_text()) if meta_path.exists() else {}
    for cls in ADAPTER_REGISTRY:
        if cls.can_handle(metadata):
            return cls(Path(experiment_dir), metadata)
    available = ", ".join(getattr(c, "__name__", str(c)) for c in ADAPTER_REGISTRY)
    raise ValueError(
        f"No adapter for type: {metadata.get('type')!r}. "
        f"Registered adapters: [{available}]"
    )


__all__ = [
    "StandardGrid",
    "ExperimentAdapter",
    "HycomAdapter",
    "InternalWaveAdapter",
    "Exp4Adapter",
    "ADAPTER_REGISTRY",
    "detect_adapter",
]

