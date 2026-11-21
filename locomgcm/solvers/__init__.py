from .base import LinearOperator, LinearSolver
from .checked_root import SolveRejected, SolveStatus, checked_root_solve, evaluate_loss_and_grad
from .cg2d import pcg_solve
from .cg3d import CG3DResult, CG3DSolver
from .cg3d_operator import CG3DOperator, apply_cg3d_operator, build_cg3d_operator
from .cg3d_post import post_cg3d_separate_dphinh
from .cg2d_preconditioner import (
    apply_cg2d_jacobi_preconditioner,
    apply_cg2d_ssor_preconditioner,
    apply_cg2d_ssor_preconditioner_edge,
    build_cg2d_jacobi_inverse,
    build_cg2d_ssor_preconditioner,
    build_cg2d_ssor_preconditioner_edge,
)
from .cg3d_preconditioner import (
    CG3DPreconditioner,
    apply_cg3d_preconditioner,
    build_cg3d_preconditioner,
)
from .cg3d_rhs import assemble_cg3d_rhs
from .cg3d_timestep_w import timestep_wvel
from .implicit_vertical import (
    MomentumWNmHalfMIT,
    TracerVerticalMetrics,
    TracerWNpHalfMIT,
    TridiagonalSolveError,
    TridiagonalSystem,
    UVerticalMetrics,
    VVerticalMetrics,
    assemble_tracer_implicit_vertical_system,
    assemble_u_implicit_vertical_system,
    assemble_v_implicit_vertical_system,
    solve_tridiagonal_system,
    tracer_vertical_transport_correction,
)
from .implicit_vertical_fused import (
    configured_fused_vertical_backend,
    fused_vertical_backend_available,
    solve_tracer_band_thomas_fused,
    solve_u_band_thomas_fused,
    solve_v_band_thomas_fused,
)
from .tridiag import build_implicit_diffusion_system, tridiag_solve

__all__ = [
    "SolveRejected",
    "SolveStatus",
    "checked_root_solve",
    "evaluate_loss_and_grad",
    "CG3DOperator",
    "CG3DPreconditioner",
    "CG3DResult",
    "CG3DSolver",
    "LinearOperator",
    "LinearSolver",
    "MomentumWNmHalfMIT",
    "TracerVerticalMetrics",
    "TracerWNpHalfMIT",
    "TridiagonalSolveError",
    "TridiagonalSystem",
    "UVerticalMetrics",
    "VVerticalMetrics",
    "apply_cg3d_operator",
    "apply_cg2d_jacobi_preconditioner",
    "apply_cg2d_ssor_preconditioner",
    "apply_cg2d_ssor_preconditioner_edge",
    "apply_cg3d_preconditioner",
    "assemble_cg3d_rhs",
    "assemble_tracer_implicit_vertical_system",
    "assemble_u_implicit_vertical_system",
    "assemble_v_implicit_vertical_system",
    "build_cg2d_jacobi_inverse",
    "build_cg2d_ssor_preconditioner",
    "build_cg2d_ssor_preconditioner_edge",
    "build_implicit_diffusion_system",
    "build_cg3d_operator",
    "build_cg3d_preconditioner",
    "pcg_solve",
    "post_cg3d_separate_dphinh",
    "solve_tridiagonal_system",
    "solve_tracer_band_thomas_fused",
    "solve_u_band_thomas_fused",
    "solve_v_band_thomas_fused",
    "timestep_wvel",
    "tracer_vertical_transport_correction",
    "tridiag_solve",
    "configured_fused_vertical_backend",
    "fused_vertical_backend_available",
]
