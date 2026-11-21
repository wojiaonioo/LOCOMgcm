"""Native CUDA client for vim coriolis u (gate: vim_coriolis_u).
"""

from ._fast import (
    compile_vim_coriolis_u_extension,
    vim_coriolis_u_launch_config,
    vim_coriolis_u_launch_count,
    u_face_coriolis_from_haloed_v_row_metrics_cuda_direct,
    reset_vim_coriolis_u_launch_count,
)
from .dispatcher import u_face_coriolis_from_haloed_v_row_metrics_cuda

__all__ = [
    "compile_vim_coriolis_u_extension",
    "vim_coriolis_u_launch_config",
    "vim_coriolis_u_launch_count",
    "u_face_coriolis_from_haloed_v_row_metrics_cuda",
    "u_face_coriolis_from_haloed_v_row_metrics_cuda_direct",
    "reset_vim_coriolis_u_launch_count",
]
