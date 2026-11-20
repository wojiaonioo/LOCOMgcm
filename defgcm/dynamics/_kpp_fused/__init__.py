"""Full regional KPP CUDA fused-scope dispatcher."""

from .dispatcher import (
    RI_TAP_NAMES,
    STAGE_NAMES,
    direct_decline_reason,
    load_extension,
    regional_kpp_fused,
    regional_kpp_fused_dispatch,
    regional_kpp_fused_stages,
)

__all__ = [
    "RI_TAP_NAMES",
    "STAGE_NAMES",
    "direct_decline_reason",
    "load_extension",
    "regional_kpp_fused",
    "regional_kpp_fused_dispatch",
    "regional_kpp_fused_stages",
]
