"""LOCOMgcm: a differentiable ocean general circulation model in PyTorch."""

from ._legacy_env import reject_legacy_environment as _reject_legacy_environment

# Before any model module reads its settings: an obsolete environment name stops the import.
_reject_legacy_environment()

from .capabilities import ModelCapabilities  # noqa: E402
from .model import RegionalOceanConfig, TorchRegionalOceanModel  # noqa: E402
from .operators import TracerFluxScheme  # noqa: E402
from .runner import ModelRunner, RunnerCompileConfig, SimulationResult, StepTimer  # noqa: E402
from .steppers import SplitExplicitStepper  # noqa: E402

__all__ = [
    "ModelCapabilities",
    "ModelRunner",
    "RegionalOceanConfig",
    "RunnerCompileConfig",
    "SimulationResult",
    "StepTimer",
    "SplitExplicitStepper",
    "TracerFluxScheme",
    "TorchRegionalOceanModel",
]
