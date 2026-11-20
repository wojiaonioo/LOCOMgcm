"""Convective KPP mixing solved implicitly with the mixed tracers.

Scheme ``convective_be_column_v1`` (stepper option ``kpp_convective_implicit``,
off by default). The lagged scheme diagnoses every KPP coefficient from the
step-start state and solves a linear column system. Here the convective part
of the coefficients, the raw buoyancy jump N^2 of the new T/S, goes through
RI_IWMIX, BLMIX, enhance and combine inside one nonlinear residual

    F(Y; X) = Y - R + dt D[K(Y, X)] - dt C = 0,   Y = (T, S) after mixing,

with the diffusive, advective and nonlocal fluxes and the continuity
correction of the lagged column solve. Shear, smoothed Ri, the boundary-layer
depth and index, pressure, w and the surface forcing stay at the diagnosis
time layer; they are the external arguments X. New-Y dependence does not enter
the horizontal smoothing, so the columns of F_Y are independent while F_X
keeps every horizontal coupling of the outer graph.

Conventions: w is MIT upward-positive (``TracerWNpHalfMIT``), layer thickness
is positive, ``zgrid``/``hwide`` are the MIT-indexed KPP tables (zgrid
negative below the surface), k=0 is the surface layer.

The root is found by full-column Newton with Armijo backtracking, started from
the lagged solution and followed by ``polish_steps`` refinements. The root
policy ``newton_rank_one_brent_v1`` selects every single-factor column's root
independently of Newton's outcome: T/S are eliminated linearly, exact endpoint
roots have priority (0 before 1), and otherwise Brent selects on [0, 1]. An
already converged representation within the root accuracy is retained;
endpoint representations must have their exact selected factor. The selected
state must satisfy the original residual. ``newton_factor_box_v1`` keeps that
path and recovers every other
unconverged column by the same elimination over its m convective factors,
c in [0, 1]^m: projected damped Newton on H(c) = c - P(N^2(Y(c))) from the
Picard prediction, the Newton iterate and the box corners, taking the first
start whose state satisfies the original residual. ``newton_factor_homotopy_v1``
keeps all of that and, where every box start fails, follows a piecewise-linear
fixed-point homotopy of G(c) = c - H(c) on the box from the Picard prediction
(Merrill's restart algorithm, mesh halved per level), checking the complete
residual at each level's zero and after each of at most 10 box Newton steps.
Positive convective feedback at the BL matching interface can stall local
searches near a nearly singular factor Jacobian; the homotopy path does not
descend |H|. In exact arithmetic, for G continuous on the box, every level
ends after finitely many pivots and the levels reach any positive residual;
the float64 implementation decides near-tied pivots in exact rational
arithmetic on its stored labels and stops at explicit pivot, evaluation and
mesh limits, so it is a deterministic fallback, not an unconditional solver.
Recovered columns record the singular values of their scaled F_z, and a
numerically rank-deficient one stops the pullback with SINGULAR_ROOT_NO_IFT.
The derivative is the implicit-function pullback F_Y^T lam = v,
dX = -F_X^T lam with the analytic column F_Y and the complete automatic F_X;
a root or adjoint residual above its tolerance raises ``RootFailure``, whose
message carries the failed column, the reason and its dependency set. Setting
``DEFGCM_KPP_ROOT_DUMP_DIR`` (diagnostic, unset by default) saves every failed
station with its external arguments to that directory.
Unknowns are scaled per level by ``residual_scale`` (a solver normalisation;
the equation does not depend on it); ``root_tolerance`` bounds the infinity
norm of the scaled residual.

The compiled execution compiles the residual and F_Y with Inductor at fixed
shape and requires TRITON_DEFAULT_FP_FUSION=0 in the environment.
Supported: one regional lat-lon tile, batch 1, JMD95 KPP buoyancy, implicit
vertical advection of T and S, nonlocal and continuity terms outside AB.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from defgcm.accel import dispatch_scope
from defgcm.accel.tensor_context import forward_mode_active
from defgcm.dynamics import kpp
from defgcm.dynamics import equation_of_state as eos

#: STATEKPP dbloc form of this scheme: its column coefficients (``coeff``) and the
#: analytic Jacobian evaluate the JMD95 divided difference, so every STATEKPP call
#: here uses it too and the stepper's KPP must be configured with the same form.
DBLOC_FORM = "divided_difference"


SCHEME = "convective_be_column_v1"
ROOT_POLICIES = ("newton", "newton_rank_one_brent_v1", "newton_factor_box_v1", "newton_factor_homotopy_v1")
RECOVERY_POLICIES = ("newton_rank_one_brent_v1", "newton_factor_box_v1", "newton_factor_homotopy_v1")
FACTOR_POLICIES = ("newton_factor_box_v1", "newton_factor_homotopy_v1")
EXECUTIONS = ("eager", "compiled")
LINEAR_SOLVERS = ("dense", "structured")
ADJOINT_TOLERANCE = 1e-11
AUX_NAMES = ("rhs", "tref", "pressure", "w", "sf", "ri_1", "ri_3", "ri_4", "ri_5",
             "bl_0", "bl_1", "bl_2", "bl_3", "bl_4")


@dataclass(frozen=True)
class ConvectiveImplicitConfig:
    """Resolved ``kpp_convective_implicit`` mapping of the stepper."""

    root_policy: str = "newton"
    root_tolerance: float = 1e-11
    polish_steps: int = 2
    residual_scale_temp: tuple[float, ...] | None = None
    residual_scale_salt: tuple[float, ...] | None = None
    execution: str = "eager"
    linear_solver: str = "structured"
    compact_columns: bool = True
    checkpoint_reuse: bool = True

    def __post_init__(self) -> None:
        if self.root_policy not in ROOT_POLICIES:
            raise ValueError(f"kpp_convective_implicit.root_policy must be one of {ROOT_POLICIES}")
        if self.execution not in EXECUTIONS:
            raise ValueError(f"kpp_convective_implicit.execution must be one of {EXECUTIONS}")
        if self.linear_solver not in LINEAR_SOLVERS:
            raise ValueError(f"kpp_convective_implicit.linear_solver must be one of {LINEAR_SOLVERS}")
        for name in ("compact_columns", "checkpoint_reuse"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"kpp_convective_implicit.{name} must be a boolean")
        if not (math.isfinite(self.root_tolerance) and self.root_tolerance > 0):
            raise ValueError("kpp_convective_implicit.root_tolerance must be positive")
        if type(self.polish_steps) is not int or self.polish_steps < 0:
            raise ValueError("kpp_convective_implicit.polish_steps must be a non-negative integer")
        for name in ("residual_scale_temp", "residual_scale_salt"):
            value = getattr(self, name)
            if value is not None and not all(math.isfinite(v) and v > 0 for v in value):
                raise ValueError(f"kpp_convective_implicit.{name} must be positive and finite")

    @classmethod
    def from_mapping(cls, value):
        """None (off), a resolved config, or the yaml mapping with an optional
        ``residual_scale: {temp: [...], salt: [...]}`` (one value per level)."""
        if value is None or isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("kpp_convective_implicit must be a mapping or null")
        known = {f.name for f in dataclasses.fields(cls)} - {"residual_scale_temp", "residual_scale_salt"}
        unknown = set(value) - known - {"residual_scale"}
        if unknown:
            raise ValueError(f"kpp_convective_implicit: unknown keys {sorted(unknown)}")
        kwargs = {k: v for k, v in value.items() if k != "residual_scale"}
        scale = value.get("residual_scale")
        if scale is not None:
            if set(scale) != {"temp", "salt"}:
                raise ValueError("kpp_convective_implicit.residual_scale needs exactly temp and salt")
            kwargs["residual_scale_temp"] = tuple(float(v) for v in scale["temp"])
            kwargs["residual_scale_salt"] = tuple(float(v) for v in scale["salt"])
        if "root_tolerance" in kwargs:
            kwargs["root_tolerance"] = float(kwargs["root_tolerance"])
        return cls(**kwargs)


@dataclass
class KppMixingInputs:
    """Diagnosis-time operands of RI_IWMIX and BLMIX kept for the implicit scheme.

    ``ri_args``/``bl_args`` are the positional arguments of the actual
    ``kpp_ri_iwmix``/``kpp_blmix`` calls on the haloed slab.
    """

    temp: torch.Tensor
    salt: torch.Tensor
    ri_args: tuple
    ri_kwargs: dict
    bl_args: tuple
    bl_kwargs: dict
    pressure: torch.Tensor
    params: dict
    config: ConvectiveImplicitConfig


# Diagnostic only, off by default: a directory that receives every failed
# station (all columns, the external arguments, the start and the last iterate).
ROOT_DUMP_ENV = "DEFGCM_KPP_ROOT_DUMP_DIR"


def _describe_failure(record, problem):
    """Status plus the recovery and Newton fields that locate a failure."""
    parts = [str(record.get("status"))]
    column = record.get("failed_column")
    if column is not None:
        nx = getattr(problem, "nx", None)
        indices = getattr(problem, "column_ids", None)
        if indices is not None:
            column = int(indices[column])
            nx = problem.source_nx
        parts.append(f"failed_column={column}" + (f" (j={column // nx}, i={column % nx})" if nx else ""))
    for key in ("dependency_size", "dependency_interfaces", "unconverged_columns", "singular_columns",
                "reason", "factor_reason", "homotopy_reason", "value", "tolerance", "dump", "dump_error"):
        if key in record:
            parts.append(f"{key}={record[key]}")
    primary = record.get("primary")
    if isinstance(primary, dict):
        parts.append(f"newton={primary.get('status')} last={(primary.get('history') or [None])[-1]}")
    return "; ".join(parts)


class RootFailure(RuntimeError):
    def __init__(self, record, *, problem=None, root=None, aux=None):
        super().__init__(f"Nonlinear convective residual did not converge: {_describe_failure(record, problem)}")
        self.record = record
        self.problem = problem
        self.root = root
        self.aux = aux


def _root_failure(record, problem, root, aux, start=None):
    """RootFailure; with ROOT_DUMP_ENV set, the complete station is saved first."""
    directory = os.environ.get(ROOT_DUMP_ENV)
    if directory:
        path = os.path.join(directory, f"kpp_root_failure_{os.getpid()}_{time.time_ns()}.pt")
        try:
            os.makedirs(directory, exist_ok=True)
            torch.save(dict(record=record, problem=_cpu_value(vars(problem)), root=_cpu_value(root),
                            aux=_cpu_value(aux), start=_cpu_value(start)), path)
            record = dict(record, dump=path)
        except Exception as error:  # the root failure stays the reported error
            record = dict(record, dump_error=repr(error))
    return RootFailure(record, problem=problem, root=root, aux=aux)


class ScalarRecoveryFailure(RuntimeError):
    pass


# Solve counts since the last reset; callers that record runs read them.
_STATISTICS: dict = {}


def reset_statistics() -> None:
    _STATISTICS.clear()
    _STATISTICS.update(stations=0, backward_stations=0, max_primal_residual=0.0,
                       max_adjoint_residual=0.0, recovered_columns=0, factor_recovered_columns=0,
                       homotopy_recovered_columns=0, homotopy_pivots=0, homotopy_labels=0,
                       homotopy_evaluations=0, recovery_wall_s=0.0, max_recovered_condition=0.0,
                       newton_stations=0, replayed_stations=0, newton_iterations=0,
                       max_newton_iterations=0, line_search_evaluations=0, max_line_search_steps=0,
                       active_column_iterations=0, polish_iterations=0, polish_improved_columns=0,
                       total_columns=0, solved_columns=0, dry_columns=0,
                       structured_linear_solves=0, dense_linear_solves=0,
                       dense_fallback_columns=0, max_linear_relative_residual=0.0)


def statistics() -> dict:
    return dict(_STATISTICS)


reset_statistics()


def _nrm(x):
    maxabs = x.abs().max()
    return maxabs * torch.linalg.vector_norm(x / maxabs) if float(maxabs) else maxabs


def inactive(x):
    """Detached copy of tensors inside dataclasses, dicts, tuples and lists."""
    if isinstance(x, torch.Tensor):
        return x.detach()
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return dataclasses.replace(x, **{
            f.name: inactive(getattr(x, f.name)) for f in dataclasses.fields(x) if f.init})
    if isinstance(x, dict):
        return {k: inactive(v) for k, v in x.items()}
    if isinstance(x, tuple):
        return tuple(inactive(v) for v in x)
    if isinstance(x, list):
        return [inactive(v) for v in x]
    return x


def _cpu_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_value(v) for v in value)
    return value


def install_integer_identity_compat() -> None:
    """Fold integer ``Identity`` values at construction (torch 2.9 Inductor).

    Inductor's stride estimator substitutes loop indices with exact integers;
    an Identity holding an integer cannot then be compared by SymPy Min/Max,
    which fails compiling the residual. Only exact integers are folded;
    symbolic barriers and all tensor computations are unchanged.
    """
    import sympy
    from torch.utils._sympy.functions import Identity

    if not str(torch.__version__).startswith("2.9."):
        raise RuntimeError("Integer Identity compatibility was verified only for torch 2.9")
    if getattr(Identity, "_defgcm_integer_identity", False):
        return
    original = Identity.eval

    def exact_integer_eval(cls, value):
        if isinstance(value, sympy.Integer):
            return value
        return original(value)

    Identity.eval = classmethod(exact_integer_eval)
    from sympy.core.cache import clear_cache
    clear_cache()
    Identity._defgcm_integer_identity = True
    torch.compiler.config.cache_key_tag = (
        (torch.compiler.config.cache_key_tag or "") + "\ndefgcm.integer_identity.v1")


_COMPILED_READY = False


def _require_compiled_execution() -> None:
    """Prerequisites of the no-FMA compiled residual programs."""
    global _COMPILED_READY
    if _COMPILED_READY:
        return
    if os.environ.get("TRITON_DEFAULT_FP_FUSION") != "0":
        raise RuntimeError(
            "kpp_convective_implicit.execution='compiled' requires "
            "TRITON_DEFAULT_FP_FUSION=0 in the process environment")
    from defgcm.accel.compile_compat import install_compile_compat
    install_compile_compat()
    install_integer_identity_compat()
    torch.compiler.config.cache_key_tag = (
        (torch.compiler.config.cache_key_tag or "") + "\nFP_FUSION=0")
    _COMPILED_READY = True


def _newton_summary_device(F, tolerance, column_ids, stalled=None):
    norm = F.abs().amax(-1)
    done = norm <= tolerance
    worst = norm.argmax()
    if column_ids is not None:
        worst = column_ids.gather(0, worst.reshape(1))[0]
    active = ~done if stalled is None else ~(done | stalled)
    summary = torch.stack((norm.max(), worst.to(F.dtype),
                           active.sum().to(F.dtype),
                           torch.isfinite(F).all().to(F.dtype)))
    return done, summary


def _line_accept_device(trial, Ft, old, scale, mask, factors, oldnorm,
                        accepted, nextz, nextF, trialnorm, tolerance, column_ids,
                        stalled):
    physical = old + trial.T.reshape(old.shape) * scale
    nz, nc = old.shape[1], trial.shape[0]
    salinity_ok = ((physical[1] >= 0) | (~mask)).reshape(nz, nc).all(0)
    good = (torch.isfinite(Ft).all(-1) & salinity_ok
            & (trialnorm <= (1. - 1e-4 * factors) * oldnorm))
    new = good & (~accepted)
    nextz = torch.where(new[:, None], trial, nextz)
    nextF = torch.where(new[:, None], Ft, nextF)
    accepted = accepted | good
    # This value is unused after an all-accepted iteration. Computing it here
    # lets the compiled program fuse the original next-iteration update.
    factors = torch.where(accepted, factors, .5 * factors)
    # If another trial is required this provisional summary is discarded. On
    # the last trial, its unaccepted columns become exactly the original stalls.
    next_stalled = ~accepted if stalled is None else stalled | ~accepted
    done, summary = _newton_summary_device(nextF, tolerance, column_ids, next_stalled)
    packet = torch.cat((accepted.all().to(Ft.dtype).reshape(1), summary))
    return nextz, nextF, accepted, packet, factors, done


def _polish_select_device(trials, Ft, old, scale, mask, z, oldnorm, tolerance, trialnorm):
    nc, nv = z.shape
    nz = old.shape[1]
    physical = old + trials.reshape(5 * nc, nv).T.reshape(old.shape) * scale
    salt_ok = ((physical[1] >= 0) | (~mask)).reshape(nz, 5, nc).all(0)
    finite = torch.isfinite(Ft).all(-1)
    bounded = Ft.abs().amax(-1) <= tolerance
    best, bestnorm = z.clone(), oldnorm.clone()
    improved = torch.zeros_like(oldnorm, dtype=torch.bool)
    for power in range(5):
        good = (finite[power] & salt_ok[power]
                & (trialnorm[power] < bestnorm) & bounded[power])
        best = torch.where(good[:, None], trials[power], best)
        bestnorm = torch.where(good, trialnorm[power], bestnorm)
        improved = improved | good
    return best, improved.sum()


# Data-dependent decisions and batched readbacks stay in the host controller. Norms
# used by strict comparisons keep their original ATen reductions outside these
# programs; moving them into Inductor can change equal-norm trial selection.
_newton_summary_program = torch.compile(
    _newton_summary_device, fullgraph=True, dynamic=False, backend="inductor")
_line_accept_program = torch.compile(
    _line_accept_device, fullgraph=True, dynamic=False, backend="inductor")
_polish_select_program = torch.compile(
    _polish_select_device, fullgraph=True, dynamic=False, backend="inductor")


class ConvectiveColumnProblem:
    """Residual, analytic F_Y and root solve of one mixing station.

    ``aux`` holds the external arguments X in the order ``AUX_NAMES``;
    ``ra``/``ba`` hold the remaining RI_IWMIX/BLMIX operands (indices, depth
    tables), detached, at the diagnosis time layer.
    """

    compiled = False
    linear_solver = "dense"
    checkpoint_reuse = False
    dry_rhs_finite = None

    def pack(self, x):
        return x.reshape(self.nv, self.nc).T.contiguous()

    def unpack(self, x):
        return x.T.reshape((2,) + self.shape)

    # -- residual -------------------------------------------------------------
    def coeff(self, Y, aux):
        # This station consumes only STATEKPP's adjacent-layer buoyancy jump.
        # Preserve its JMD95 divided differences and operation order without
        # evaluating the unused surface density and expansion coefficients.
        t, s = Y[0], Y[1]
        pressure = aux["pressure"][1:]
        rho0 = self.par["rho_const"]
        density = eos.jmd95_density_anomaly(
            t[1:], s[1:], pressure, reference_density=rho0) + rho0
        difference = eos.jmd95_density_difference(t[1:], s[1:], t[:-1], s[:-1], pressure)
        buoyancy = self.par["gravity"] * difference / density
        dbloc = torch.cat((buoyancy, torch.zeros_like(t[:1])), 0) * self.interface_mask
        return self.coeff_from_buoyancy(Y, dbloc, aux)

    def coeff_from_buoyancy(self, Y, dbloc, aux):
        """Canonical KPP composition with an explicit raw buoyancy jump."""
        ra = list(self.ra)
        for i in (1, 3, 4, 5):
            ra[i] = aux["ri_" + str(i)]
        ra[2] = dbloc
        vis, ks, kt = kpp.kpp_ri_iwmix(*ra, **self.ri_kwargs)
        interior = torch.stack((vis, ks, kt))
        below = (torch.arange(self.nz, device=Y.device)[:, None, None] + 1 >= ra[0][None])
        interior = torch.where(below[None], torch.zeros_like(interior), interior)
        pad = Y.new_zeros((3, 1, self.ny, self.nx))
        diffus = torch.cat((pad, interior, pad), 1)
        ba = list(self.ba)
        for i in (0, 1, 2, 3, 4):
            ba[i] = aux["bl_" + str(i)]
        ba[6] = diffus
        blmc, ghat, dkm1 = kpp.kpp_blmix(*ba, **self.bl_kwargs)
        blmc, ghat = kpp.kpp_enhance(dkm1, ba[2], ba[5], ba[4], diffus, blmc, ghat, ba[7])
        out, ghat = kpp.kpp_combine(blmc, diffus, ghat, ba[5], visc_arnr=self.par["visc_arnr"],
                                    bg_KzS_bottom=ra[4][-1], bg_KzT_bottom=ra[5][-1])
        km1 = torch.clamp(torch.arange(self.nz, device=Y.device) - 1, min=0)
        cmask = self.mask * self.mask.index_select(0, km1)
        K = torch.stack((out[2, :self.nz], out[1, :self.nz])) * cmask
        return K, ghat * cmask

    def residual_eager(self, z, aux):
        Y = self.old + self.unpack(z) * self.scale
        K, ghat = self.coeff(Y, aux)
        return self.residual_from_coefficients(Y, K, ghat, aux)

    def residual_from_coefficients(self, Y, K, ghat, aux):
        """The finite-volume column equation with explicit coefficient operands."""
        w = torch.cat((torch.zeros_like(aux["w"][:1]), aux["w"][1:-1] * self.pair,
                       torch.zeros_like(aux["w"][:1])), 0)
        # Downward diffusive + advective + nonlocal flux at each interior interface.
        linear = K[:, 1:] * (Y[:, :-1] - Y[:, 1:]) / self.dc[None, :, None, None]
        adv = -w[None, 1:-1] * torch.where(w[None, 1:-1] >= 0, Y[:, 1:], Y[:, :-1])
        nonlocal_flux = K[:, 1:] * ghat[None, :-1] * aux["sf"][:, None]
        pad = Y.new_zeros((2, 1, self.ny, self.nx))
        flux = torch.cat((pad, (linear + adv + nonlocal_flux) * self.pair[None], pad), 1)
        divergence = self.dt * (flux[:, 1:] - flux[:, :-1]) * self.inv_volume[None]
        correction = (self.dt * aux["tref"] * (w[:-1] - w[1:])[None]
                      * self.inv_volume[None] * self.interior)
        physical = Y - aux["rhs"] + divergence - correction
        return self.pack(physical / self.scale)

    def residual(self, z, aux=None):
        aux = self.aux if aux is None else aux
        with dispatch_scope("kpp_blmix_cuda"):
            if self.compiled:
                return _residual_program(self, z, aux)
            return self.residual_eager(z, aux)

    # -- Jacobian -------------------------------------------------------------
    def linear_jacobian(self, z, aux):
        Y = self.old + self.unpack(z) * self.scale
        K, _ = self.coeff(Y, aux)
        return self.linear_jacobian_from_coefficients(z, Y, K, aux)

    def linear_jacobian_from_coefficients(self, z, Y, K, aux):
        """Fixed-coefficient matrix, also used by the exact scalar elimination."""
        middle = (K[:, 1:] / self.dc[None, :, None, None]) * self.pair
        w = aux["w"][1:-1] * self.pair
        above = middle + (-w).clamp_min(0)[None]
        below = -middle - w.clamp_min(0)[None]
        zeros = Y.new_zeros((2, 1, self.ny, self.nx))
        dplus = torch.cat((above, zeros), 1)
        dminus = torch.cat((zeros, below), 1)
        diagonal = 1 + self.dt * (dplus - dminus) * self.inv_volume[None]
        lower = -self.dt * above * self.inv_volume[None, 1:] * (self.scale[:, :-1] / self.scale[:, 1:])
        upper = self.dt * below * self.inv_volume[None, :-1] * (self.scale[:, 1:] / self.scale[:, :-1])
        block = torch.diag_embed(diagonal.reshape(2, self.nz, self.nc).permute(2, 0, 1))
        block = block + torch.diag_embed(lower.reshape(2, self.nz - 1, self.nc).permute(2, 0, 1), offset=-1)
        block = block + torch.diag_embed(upper.reshape(2, self.nz - 1, self.nc).permute(2, 0, 1), offset=1)
        J = z.new_zeros((self.nc, self.nv, self.nv))
        J[:, :self.nz, :self.nz] = block[:, 0]
        J[:, self.nz:, self.nz:] = block[:, 1]
        return J

    def jacobian(self, z, aux=None):
        aux = self.aux if aux is None else aux
        with torch.no_grad(), dispatch_scope("kpp_blmix_cuda"):
            if self.compiled:
                return _jacobian_program(self, z, aux)
            return self.analytic_jacobian(z, aux)

    def analytic_jacobian(self, z, aux):
        """F_Y: transport plus the chain through N^2, the convective polynomial,
        BL matching, enhance and combine. abs'(0)=0, clamp'=1 and a maximum tie
        weighs 1/2, as the Torch residual differentiates."""
        nz, nc = self.nz, self.nc
        Y = self.old + self.unpack(z) * self.scale
        base = self.linear_jacobian(z, aux)
        t, s = Y[0], Y[1]
        p = aux["pressure"][1:]
        g, rho0 = self.par["gravity"], self.par["rho_const"]
        den = eos.jmd95_density_anomaly(t[1:], s[1:], p, reference_density=rho0) + rho0
        drho = eos.jmd95_density_difference(t[1:], s[1:], t[:-1], s[:-1], p)
        buoy = g * drho / den
        db = torch.cat((buoy, torch.zeros_like(t[:1])), 0) * self.interface_mask
        dz = (self.zgrid[1:nz + 1] - self.zgrid[2:nz + 2])[:, None, None]
        ns = db / dz
        at, bs = eos.jmd95_alpha_beta(t[:-1], s[:-1], p)
        ab, bb = eos.jmd95_alpha_beta(t[1:], s[1:], p)
        common = g / (dz[:-1] * den) * self.pair
        upper = torch.stack((-common * at, -common * bs)) * self.scale[:, :-1]
        lower = torch.stack((common * (1 - drho / den) * ab, common * (1 - drho / den) * bb)) * self.scale[:, 1:]
        zeros = Y.new_zeros((2, 1, self.ny, self.nx))
        diagonal = torch.cat((upper, zeros), dim=1).reshape(2, nz, nc).permute(2, 0, 1)
        off = lower.reshape(2, nz - 1, nc).permute(2, 0, 1)
        Jns = torch.diag_embed(diagonal) + torch.diag_embed(off, offset=1)
        Jns = torch.cat((Jns[:, 0], Jns[:, 1]), dim=-1)
        b = self.ri_kwargs["BVSQcon"]
        ratio = torch.clamp((b - torch.clamp(ns, min=b)) / b, max=1.)
        active = (ns > b) & (ns < 0) & self.interface_mask
        df = torch.where(active, 6 * ratio * (1 - ratio.square()).square() / b, torch.zeros_like(ns))
        Jfactor = df.reshape(nz, nc).T[..., None] * Jns
        amplitudes = Y.new_tensor([self.ri_kwargs[n] for n in ("difmcon", "difscon", "diftcon")])
        DD = Y.new_zeros((3, nc, nz + 2, 2 * nz))
        DD[:, :, 1:nz + 1] = amplitudes[:, None, None, None] * Jfactor[None]
        km = self.ra[0].reshape(nc)
        below = (torch.arange(nz, device=Y.device)[None, :] + 1) >= km[:, None]
        DD[:, :, 1:nz + 1] *= (~below)[None, :, :, None]
        ra = list(self.ra)
        for i in (1, 3, 4, 5):
            ra[i] = aux["ri_" + str(i)]
        ra[2] = db
        visc, ks, kt = kpp.kpp_ri_iwmix(*ra, **self.ri_kwargs)
        interior = torch.stack((visc, ks, kt))
        interior = torch.where(below.T.reshape(1, nz, self.ny, self.nx), torch.zeros_like(interior), interior)
        pad = Y.new_zeros((3, 1, self.ny, self.nx))
        diffus = torch.cat((pad, interior, pad), dim=1)
        ba = list(self.ba)
        for i in (0, 1, 2, 3, 4):
            ba[i] = aux["bl_" + str(i)]
        ba[6] = diffus
        blmc, ghat, dkm1 = kpp.kpp_blmix(*ba, **self.bl_kwargs)
        ustar, bfsfc, hbl, stable, casea, kbl = ba[:6]
        zgrid, hwide, wmt, wst = ba[7:11]
        kw = self.bl_kwargs
        phepsi = kw.get("phepsi", 1e-10)
        epsilon, conc1 = kw["epsilon"], kw["conc1"]
        extra = {k: v for k, v in kw.items() if k not in ("epsilon", "conc1", "cg", "phepsi", "sign_zero")}
        sigma = stable + (1 - stable) * epsilon
        wm, ws = kpp.kpp_wscale(sigma, hbl, ustar, bfsfc, wmt, wst, **extra)
        mode = kw.get("sign_zero", "signbit")
        wm = kpp._fortran_sign(1., wm, mode) * wm.abs().clamp_min(phepsi)
        ws = kpp._fortran_sign(1., ws, mode) * ws.abs().clamp_min(phepsi)
        w0 = torch.stack((wm, ws, ws)).reshape(3, nc)
        case_int = (casea + phepsi).long()
        kn = (case_int * (kbl - 1) + (1 - case_int) * kbl).reshape(nc).long()
        H = hbl.reshape(nc)
        ST = stable.reshape(nc)
        CA = casea.reshape(nc)
        KB = kbl.reshape(nc).long()

        def gather(value, index):
            if value.ndim == 1:
                return value[index.clamp(0, value.shape[0] - 1)]
            ix = index.clamp(0, value.shape[2] - 1)[None, :, None, None].expand(
                value.shape[0], -1, 1, value.shape[-1])
            return value.gather(2, ix).squeeze(2)

        values = diffus.reshape(3, nz + 2, nc).permute(0, 2, 1).unsqueeze(-1)
        hw = gather(hwide, kn)
        hwp = gather(hwide, kn + 1)
        delhat = .5 * hw - gather(zgrid, kn) - H
        R = 1 - delhat / hw
        dminus, dkn, dplus = [gather(values, ix).squeeze(-1) for ix in (kn - 1, kn, kn + 1)]
        up = (dminus - dkn) / hw
        dn = (dkn - dplus) / hwp
        pm = .5 * ((1 - R) * (up + up.abs()) + R * (dn + dn.abs()))
        dp = ((1 - R)[None, :, None] * (.5 * (1 + up.sign()))[..., None]
              * (gather(DD, kn - 1) - gather(DD, kn)) / hw[None, :, None]
              + R[None, :, None] * (.5 * (1 + dn.sign()))[..., None]
              * (gather(DD, kn) - gather(DD, kn + 1)) / hwp[None, :, None])
        hm = dkn + pm * delhat
        dh = gather(DD, kn) + dp * delhat[None, :, None]
        f1 = (stable * conc1 * bfsfc / (ustar ** 4).clamp_min(phepsi)).reshape(nc)
        dgat = dh / H[None, :, None] / w0[..., None]
        rawdat = -pm / w0 + f1 * hm
        ddat = (-dp / w0[..., None] + f1[None, :, None] * dh) * (rawdat <= 0)[..., None]
        lev = torch.arange(1, nz + 1, device=Y.device)
        sig = (-zgrid[lev] + .5 * hwide[lev])[:, None, None] / hbl
        sigma_i = stable * sig + (1 - stable) * sig.clamp(max=epsilon)
        wi_m, wi_s = kpp.kpp_wscale(sigma_i, hbl, ustar, bfsfc, wmt, wst, **extra)
        wi = torch.stack((wi_m, wi_s, wi_s)).reshape(3, nz, nc).permute(0, 2, 1)
        sig = sig.reshape(nz, nc).T
        dbmix = (H[None, :, None] * wi * sig[None].square())[..., None] * (
            (3 - 2 * sig)[None, :, :, None] * dgat[:, :, None, :]
            + (sig - 1)[None, :, :, None] * ddat[:, :, None, :])
        sigd = -gather(zgrid, KB - 1) / H
        sigma_d = (ST * sigd + (1 - ST) * sigd.clamp(max=epsilon)).reshape(self.ny, self.nx)
        wd_m, wd_s = kpp.kpp_wscale(sigma_d, hbl, ustar, bfsfc, wmt, wst, **extra)
        wd = torch.stack((wd_m, wd_s, wd_s)).reshape(3, nc)
        ddkm = (H[None] * wd * sigd[None].square())[..., None] * (
            (3 - 2 * sigd)[None, :, None] * dgat + (sigd - 1)[None, :, None] * ddat)
        ki = KB - 1
        valid = (ki >= 1) & (ki < nz)
        to = (ki - 1).clamp(0, nz - 1)
        delta = (H + gather(zgrid, ki)) / (gather(zgrid, ki) - gather(zgrid, ki + 1))
        dki = gather(DD, ki)
        dbki = gather(dbmix, to)
        new = kpp._enhance_diffusivity(delta[None, :, None], CA[None, :, None], dki, dbki, ddkm)
        new = torch.where(valid[None, :, None], new, dbki)
        dbmix = dbmix.scatter(2, to[None, :, None, None].expand(3, nc, 1, 2 * nz), new[:, :, None, :])
        post, ghat = kpp.kpp_enhance(dkm1, hbl, kbl, casea, diffus, blmc, ghat, zgrid)
        bg = torch.stack((torch.full_like(ra[4][-1], self.par["visc_arnr"]), ra[4][-1], ra[5][-1]))
        bg = bg.reshape(3, nc)
        postvalues = post.reshape(3, nz, nc).permute(0, 2, 1)
        weight = torch.where(postvalues > bg[..., None], 1., torch.where(postvalues < bg[..., None], 0., .5))
        within = torch.arange(1, nz + 1, device=Y.device)[None, :] < KB[:, None]
        combined = torch.where(within[None, :, :, None], weight[..., None] * dbmix, DD[:, :, 1:nz + 1])
        combined = torch.cat((combined.new_zeros((3, nc, 1, 2 * nz)), combined[:, :, :nz - 1]), dim=2)
        cmask = self.mask * torch.cat((self.mask[:1], self.mask[:-1]), 0)
        combined *= cmask.reshape(nz, nc).T[None, :, :, None]
        dK = torch.stack((combined[2], combined[1]))
        # ghat does not depend on the new Y in this residual.
        ghat = torch.where(within.T.reshape(nz, self.ny, self.nx), ghat, torch.zeros_like(ghat)) * cmask
        factor = (Y[:, :-1] - Y[:, 1:]) / self.dc[None, :, None, None] + ghat[None, :-1] * aux["sf"][:, None]
        factor = (factor * self.pair[None]).reshape(2, nz - 1, nc).permute(0, 2, 1)
        inner = factor[..., None] * dK[:, :, 1:]
        padj = inner.new_zeros((2, nc, 1, 2 * nz))
        flux = torch.cat((padj, inner, padj), dim=2)
        div = self.dt * (flux[:, :, 1:] - flux[:, :, :-1])
        div *= self.inv_volume.reshape(nz, nc).T[None, :, :, None]
        div /= self.scale.reshape(2, nz)[:, None, :, None]
        return base + div.permute(1, 0, 2, 3).reshape(nc, 2 * nz, 2 * nz)

    def solve_linear(self, z, rhs, aux, *, transpose=False):
        """Solve the complete scaled Jacobian, with a checked compact backend."""
        if self.linear_solver == "structured":
            from defgcm.dynamics._kpp_convective_linear import LinearSolveFailure, solve_checked

            try:
                solution, factors, info = solve_checked(
                    self, z, rhs, aux=aux, transpose=transpose,
                    backend="cuda" if z.is_cuda else "torch", tolerance=ADJOINT_TOLERANCE,
                )
            except LinearSolveFailure as error:
                failure = dict(error.record)
                failed = failure.get("failed_columns", ())
                failure.update(status="JACOBIAN_LINEAR_SOLVE",
                               reason=str(error), transpose=transpose)
                if failed:
                    failure["failed_column"] = failed[0]
                raise _root_failure(failure, self, z, aux) from error
            _STATISTICS["structured_linear_solves"] += 1
            _STATISTICS["dense_fallback_columns"] += info.get("dense_fallback_columns", 0)
            _STATISTICS["max_linear_relative_residual"] = max(
                _STATISTICS["max_linear_relative_residual"], info.get("relative_residual", 0.0))
            return solution, factors.linearization
        matrix = self.jacobian(z, aux)
        operator = matrix.transpose(-1, -2) if transpose else matrix
        solution = torch.linalg.solve(operator, rhs[..., None])[..., 0]
        _STATISTICS["dense_linear_solves"] += 1
        return solution, matrix

    @staticmethod
    def apply_linear(linearization, value, *, transpose=False):
        if isinstance(linearization, torch.Tensor):
            matrix = linearization.transpose(-1, -2) if transpose else linearization
            return (matrix @ value[..., None])[..., 0]
        from defgcm.dynamics._kpp_convective_linear import matvec

        return matvec(linearization, value, transpose=transpose)

    # -- root ------------------------------------------------------------------
    def subset(self, indices, aux=None):
        """The same problem restricted to the listed columns."""
        oldshape = (self.ny, self.nx)
        ncol = self.nc

        def take(v):
            if isinstance(v, torch.Tensor) and v.ndim >= 2 and tuple(v.shape[-2:]) == oldshape:
                return v.reshape(v.shape[:-2] + (ncol,)).index_select(-1, indices).unsqueeze(-2)
            return v
        result = copy.copy(self)
        for field in ("old", "saved", "hfac", "mask", "volume", "inv_volume", "pair",
                      "interface_mask", "interior"):
            setattr(result, field, take(getattr(self, field)))
        result.ra = [take(v) for v in self.ra]
        # The last four BL arguments are vertical coordinates and lookup
        # tables; their trailing dimensions do not denote horizontal columns.
        result.ba = [take(v) if i < 7 else v for i, v in enumerate(self.ba)]
        result.aux = {n: take(v) for n, v in (self.aux if aux is None else aux).items()}
        result.ny = 1
        result.nx = int(indices.numel())
        result.nc = result.nx
        result.shape = (result.nz, result.ny, result.nx)
        if hasattr(self, "column_ids"):
            result.column_ids = self.column_ids.index_select(0, indices)
        return result

    def polish(self, z, aux, tol):
        """Refine five trial states with canonical norms and strict selection."""
        records = []
        sub = None
        program = _polish_select_program if self.compiled else _polish_select_device
        for iteration in range(self.polish_steps):
            F = self.residual(z, aux)
            oldnorm = torch.linalg.vector_norm(F, dim=-1)
            correction, linearization = self.solve_linear(z, -F, aux)
            del linearization
            if sub is None:
                indices = torch.arange(self.nc, device=z.device).repeat(5)
                sub = self.subset(indices, aux)
                factors = z.new_tensor([1., .5, .25, .125, .0625])[:, None, None]
            trials = z[None] + factors * correction[None]
            Ft = sub.residual(trials.reshape(5 * self.nc, self.nv), sub.aux).reshape(5, self.nc, self.nv)
            trialnorm = torch.linalg.vector_norm(Ft, dim=-1)
            z, count = program(trials, Ft, sub.old, sub.scale, sub.mask, z, oldnorm, tol, trialnorm)
            z = z.detach()
            del trials, Ft
            records.append(dict(iteration=iteration, improved_columns=int(count)))
        return z, records

    def newton(self, z0, tol, aux, maxiter=40):
        """Damped Newton with canonical norms and compiled tensor decisions."""
        z = z0.detach().clone()
        history = []
        summary_fn = _newton_summary_program if self.compiled else _newton_summary_device
        line_fn = _line_accept_program if self.compiled else _line_accept_device
        columns = getattr(self, 'column_ids', None)
        source_nx = self.source_nx if columns is not None else self.nx
        stalled = None
        next_summary = None
        with torch.no_grad():
            F = self.residual(z, aux)
            for iteration in range(maxiter + 1):
                if next_summary is None:
                    done, device_summary = summary_fn(F, tol, columns, stalled)
                    summary = device_summary.tolist()
                else:
                    done, summary = next_done, next_summary
                worst, active_count = int(summary[1]), int(summary[2])
                if stalled is not None:
                    active = ~(done | stalled)
                row = dict(iteration=iteration, residual_inf=summary[0],
                           worst_column=[worst // source_nx, worst % source_nx],
                           active_columns=active_count)
                history.append(row)
                if not active_count:
                    return z, dict(status='CONVERGED' if stalled is None else 'LINE_SEARCH_FAILED', history=history)
                if not summary[3]:
                    return z, dict(status='NONFINITE', history=history)
                if iteration == maxiter:
                    return z, dict(status='ITERATION_LIMIT' if stalled is None else 'LINE_SEARCH_FAILED', history=history)
                if stalled is None:
                    step, linearization = self.solve_linear(z, -F, aux)
                else:
                    indices = torch.nonzero(active).flatten()
                    sub = self.subset(indices, aux)
                    # Active-set sizes change after a stall; keep their small
                    # linearization eager instead of compiling each new shape.
                    sub.compiled = False
                    if not hasattr(sub, 'column_ids'):
                        sub.column_ids = indices
                        sub.source_nx = self.nx
                    correction, linearization = sub.solve_linear(
                        z.index_select(0, indices), -F.index_select(0, indices), sub.aux)
                    step = torch.zeros_like(z).index_copy_(0, indices, correction)
                del linearization
                step = torch.where(done[:, None], torch.zeros_like(step), step)
                factors = torch.ones(self.nc, dtype=z.dtype, device=z.device)
                accepted = done.clone() if stalled is None else done | stalled
                nextz, nextF = z.clone(), F.clone()
                oldnorm = torch.linalg.vector_norm(F, dim=-1)
                for line in range(25):
                    trial = z + factors[:, None] * step
                    Ft = self.residual(trial, aux)
                    trialnorm = torch.linalg.vector_norm(Ft, dim=-1)
                    nextz, nextF, accepted, device_packet, factors, next_done = line_fn(
                        trial, Ft, self.old, self.scale, self.mask,
                        factors, oldnorm, accepted, nextz, nextF, trialnorm,
                        tol, columns, stalled)
                    packet = device_packet.tolist()
                    all_accepted, next_summary = bool(packet[0]), packet[1:]
                    if all_accepted:
                        break
                row['line_search_steps'] = line + 1
                if not all_accepted:
                    row['failed_columns'] = [int(i) for i in torch.nonzero(~accepted).flatten().cpu()]
                    # Failed columns are independent: retain the accepted work
                    # and spend the remaining original budget on the others.
                    stalled = ~accepted if stalled is None else stalled | ~accepted
                z, F = nextz.detach(), nextF.detach()
        raise AssertionError('unreachable')

    def solve(self, z0, tol, aux=None, maxiter=40):
        aux = self.aux if aux is None else aux
        z, record = self.newton(z0, tol, aux, maxiter)
        if record["status"] == "CONVERGED":
            with torch.no_grad():
                z, polishing = self.polish(z, aux, tol)
                error = float(self.residual(z, aux).abs().max())
            record.update(polishing=polishing, final_residual_inf=error)
            if error > tol:
                record["status"] = "POLISHED_RESIDUAL_FAILED"
        if record["status"] != "CONVERGED" and self.root_policy in RECOVERY_POLICIES:
            z, record = self.recover(z, tol, aux, record)
        if record["status"] == "CONVERGED" and self.root_policy in RECOVERY_POLICIES:
            z, record = self.select_scalar_roots(z, tol, aux, record)
        return z, record

    def select_scalar_roots(self, z, tol, aux, record):
        """Apply the scalar policy to every rank-one column, including Newton roots."""
        start = time.perf_counter()
        with torch.no_grad():
            kmt = self.ra[0].reshape(self.nc).long()
            kbl = self.ba[5].reshape(self.nc).long()
            case = (aux["bl_4"].reshape(self.nc) + self.bl_kwargs.get("phepsi", 1e-10)).long()
            # The original support union is the consecutive interval
            # max(1, kbl-1-case), ..., kmt-1 for either matching case.
            size = (kmt - (kbl - 1 - case).clamp_min(1)).clamp_min(0)
            selected = torch.nonzero((size == 1) & ((case == 0) | (case == 1))).flatten()
            if not selected.numel():
                return z, record
            sub = self.subset(selected, aux)
            interfaces = (kmt.index_select(0, selected) - 1).cpu().tolist()
            elimination = _BatchedFactorElimination(sub, [[ki] for ki in interfaces])
            roots, rows, errors = _solve_scalar_batch(
                elimination, tol, iterate=z.index_select(0, selected))
            indices = selected.cpu().tolist()
            for index, error in zip(indices, errors, strict=True):
                if error is not None:
                    return z, dict(status="SCALAR_ROOT_SELECTION_FAILED", primary=record,
                                   failed_column=index, reason=error)
            changed = [i for i, row in enumerate(rows) if not row["retained_root"]]
            if not changed:
                return z, record
            take = torch.tensor(changed, device=z.device)
            changed_indices = selected.index_select(0, take)
            z.index_copy_(0, changed_indices, roots.index_select(0, take))
            replacements = {}
            for i in changed:
                row = rows[i]
                row.update(column=indices[i], effective_interfaces=[interfaces[i]],
                           root_selection="scalar_endpoint_brent")
                replacements[indices[i]] = row
            del elimination, sub, roots
            if self.root_policy == "newton_factor_homotopy_v1":
                sv = _root_singular_values(self, z, aux, changed_indices)
                for i, values in zip(changed, sv, strict=True):
                    top, bottom = float(values[0]), float(values[-1])
                    rows[i].update(F_z_sigma_max=top, F_z_sigma_min=bottom,
                                   F_z_condition=top / bottom if bottom else math.inf,
                                   F_z_rank_deficient=bottom <= self.nv * torch.finfo(values.dtype).eps * top)
            recovered = [replacements.pop(row["column"], row)
                         for row in record.get("recovered_columns", ())]
            recovered.extend(replacements.values())
            final = float(self.residual(z, aux).abs().max())
            status = "CONVERGED" if math.isfinite(final) and final <= tol else "SCALAR_ROOT_SELECTION_RESIDUAL_FAILED"
        return z, dict(record, status=status, recovered_columns=recovered, final_residual_inf=final,
                       recovery_wall_s=record.get("recovery_wall_s", 0.) + time.perf_counter() - start)

    def recover(self, z, tol, aux, primary):
        """Exact elimination of failed columns: Brent for one factor
        (``newton_rank_one_brent_v1``); ``newton_factor_box_v1`` solves the
        other columns, and those where Brent fails, over all their factors;
        ``newton_factor_homotopy_v1`` follows the PL homotopy where that fails."""
        start = time.perf_counter()
        with torch.no_grad():
            F = self.residual(z, aux)
            if not torch.isfinite(F).all():
                return z, primary
            failed = torch.nonzero(F.abs().amax(-1) > tol).flatten()
            indices = failed.tolist()
            sub = self.subset(failed, aux)
            roots, records, scalar_errors, factor_errors, interfaces = _recover_columns_device(
                sub, z.index_select(0, failed), tol=tol, policy=self.root_policy)
            remaining = [i for i, record in enumerate(records) if record is None]
            cpu = None
            if remaining and self.root_policy == "newton_factor_homotopy_v1":
                pending = sub.subset(torch.tensor(remaining, device=z.device))
                cpu = ConvectiveColumnProblem.__new__(ConvectiveColumnProblem)
                cpu.__dict__.update(_cpu_value(vars(pending)))
                cpu.device = torch.device("cpu")
                cpu.compiled = False
            pending_positions = {position: i for i, position in enumerate(remaining)}
            recovery = []
            for position, index in enumerate(indices):
                record = records[position]
                if record is None:
                    # Preserve the original prefix on failure even though the
                    # independent device solves have already evaluated later columns.
                    if position:
                        z.index_copy_(0, failed[:position], roots[:position])
                    dependency = interfaces[position]
                    failure = dict(status="RANK_ONE_RECOVERY_FAILED", primary=primary,
                                   failed_column=index, reason=scalar_errors[position],
                                   dependency_interfaces=dependency,
                                   dependency_size=len(dependency) if isinstance(dependency, list) else None,
                                   unconverged_columns=indices,
                                   recovered_columns=recovery)
                    if self.root_policy not in FACTOR_POLICIES:
                        return z, failure
                    failure.update(status="FACTOR_RECOVERY_FAILED",
                                   factor_reason=factor_errors[position])
                    if self.root_policy != "newton_factor_homotopy_v1":
                        return z, failure
                    column = cpu.subset(torch.tensor([pending_positions[position]]))
                    try:
                        root, record = _solve_factor_homotopy(column, column.aux, tol=tol)
                    except (ScalarRecoveryFailure, ValueError, RuntimeError) as homotopy_error:
                        return z, dict(failure, status=getattr(homotopy_error, "status", "HOMOTOPY_ERROR"),
                                       homotopy_reason=str(homotopy_error))
                    roots[position:position + 1] = root.to(z)
                    record["factor_recovery"] = factor_errors[position]
                    record["scalar_recovery"] = scalar_errors[position]
                record["column"] = index
                recovery.append(record)
            z.index_copy_(0, failed, roots)
            del sub, roots
            # The recovered state must satisfy the residual on the device.
            z, polishing = self.polish(z, aux, tol)
            Y = self.old + self.unpack(z) * self.scale
            actual = self.pack((Y - self.old) / self.scale)
            final = float(self.residual(actual, aux).abs().max())
            status = "CONVERGED" if math.isfinite(final) and final <= tol else "DEVICE_RECOVERY_RESIDUAL_FAILED"
            if self.root_policy == "newton_factor_homotopy_v1" and recovery:
                # Conditioning of the scaled F_z that the pullback solves, per recovered
                # column; numerical rank deficiency at sigma_min <= nv eps sigma_max.
                index = torch.tensor([r["column"] for r in recovery], device=z.device)
                sv = _root_singular_values(self, actual, aux, index)
                for r, values in zip(recovery, sv, strict=True):
                    top, bottom = float(values[0]), float(values[-1])
                    r.update(F_z_sigma_max=top, F_z_sigma_min=bottom,
                             F_z_condition=top / bottom if bottom else math.inf,
                             F_z_rank_deficient=bottom <= self.nv * torch.finfo(values.dtype).eps * top)
        return actual, dict(status=status, primary=primary, recovered_columns=recovery,
                            polishing=polishing, final_residual_inf=final,
                            recovery_wall_s=time.perf_counter() - start)


@torch.compile(fullgraph=True, backend="inductor", dynamic=False)
def _residual_program(problem, z, aux):
    return ConvectiveColumnProblem.residual_eager(problem, z, aux)


@torch.compile(fullgraph=True, backend="inductor", dynamic=False)
def _jacobian_program(problem, z, aux):
    return ConvectiveColumnProblem.analytic_jacobian(problem, z, aux)


@torch.compile(fullgraph=True, backend="inductor", dynamic=None)
def _conditioning_jacobian_program(problem, z, aux):
    return ConvectiveColumnProblem.analytic_jacobian(problem, z, aux)


def _root_singular_values(problem, z, aux, indices):
    """Condition only the recovered roots, retaining the original analytic matrix."""
    sub = problem.subset(indices, aux)
    sub.compiled = False
    with torch.no_grad(), dispatch_scope("kpp_blmix_cuda"):
        selected = z.index_select(0, indices)
        matrix = (_conditioning_jacobian_program(sub, selected, sub.aux) if problem.compiled
                  else sub.analytic_jacobian(selected, sub.aux))
        return torch.linalg.svdvals(matrix).cpu()


def _effective_interfaces(problem, aux):
    """Structural support of one column's new-state convective coefficients:
    BL matching, enhance and the retained interior; surface and seabed are zero."""
    if problem.nc != 1:
        raise ScalarRecoveryFailure("Scalar elimination requires one independent column")
    kmt = int(problem.ra[0].item())
    kbl = int(problem.ba[5].item())
    case = int((aux["bl_4"] + problem.bl_kwargs.get("phepsi", 1e-10)).item())
    if case not in (0, 1):
        raise ScalarRecoveryFailure("Unexpected KPP caseA index")
    kn = case * (kbl - 1) + (1 - case) * kbl
    active = ({kn - 1, kn, kn + 1, kbl - 1} | set(range(max(1, kbl), kmt))) & set(range(1, kmt))
    return sorted(active)


def _fixed_factor_map(problem, buoy):
    """Eliminate T/S at fixed buoyancy, independent of the factor count."""
    p, a = problem, problem.aux
    K, gh = p.coeff_from_buoyancy(p.old, buoy, a)
    valid_K = (torch.isfinite(K) & ((K >= 0) | ~p.mask)).flatten(0, 2).all(0)
    F0 = p.residual_from_coefficients(p.old, K, gh, a)
    A = p.linear_jacobian_from_coefficients(torch.zeros_like(F0), p.old, K, a)
    scales = p.pack(p.scale.expand_as(p.old))
    physical = scales[:, :, None] * A / scales[:, None, :]
    wet = p.pack(p.mask.expand_as(p.old)).bool()
    mass = p.pack(p.volume.expand_as(p.old))
    block = torch.where(wet[:, :, None] & wet[:, None, :], physical, 0.)
    off = block - torch.diag_embed(block.diagonal(dim1=-2, dim2=-1))
    column_error = (mass[:, None, :] @ block)[:, 0] - mass
    bound = 100 * torch.finfo(A.dtype).eps * (mass[:, :, None] * block.abs()).sum(1)
    valid_A = (torch.isfinite(A).flatten(1).all(1) & ~(off > 0).flatten(1).any(1)
               & ((block.diagonal(dim1=-2, dim2=-1) > 0) | ~wet).all(1)
               & (column_error.abs() <= bound).all(1))
    valid = valid_K & valid_A
    eye = torch.eye(p.nv, device=A.device, dtype=A.dtype)
    zz, info = torch.linalg.solve_ex(torch.where(valid[:, None, None], A, eye), -F0[..., None])
    zz = zz[..., 0]
    Y = p.old + p.unpack(zz) * p.scale
    valid_Y = ((info == 0) & torch.isfinite(Y).flatten(0, 2).all(0)
               & ((Y[1] >= 0) | ~p.mask).flatten(0, 1).all(0))
    reason = torch.where(~valid_K, 1, torch.where(~valid_A, 2, torch.where(~valid_Y, 3, 0)))
    return zz, Y, reason


@torch.compile(fullgraph=True, backend="inductor", dynamic=False)
def _fixed_factor_map_program(problem, buoy):
    return _fixed_factor_map(problem, buoy)


class _BatchedFactorElimination:
    """The same fixed-factor elimination for independent device columns.

    A group has m effective interfaces per column, with different interface
    indices allowed. Candidate rows are ordered (candidate, column, factor).
    The host only controls group-wide iteration; column decisions stay tensors.
    """

    def __init__(self, problem, interfaces):
        self.compiled = problem.compiled and problem.old.is_cuda
        self.problem = copy.copy(problem)
        self.problem.compiled = False
        self.nc, self.m = problem.nc, len(interfaces[0])
        self.interfaces = torch.tensor(interfaces, device=problem.old.device, dtype=torch.long)
        self.spacing = problem.zgrid[self.interfaces] - problem.zgrid[self.interfaces + 1]
        self.b = float(problem.ri_kwargs["BVSQcon"])
        self.copies = {}
        self.evaluations = torch.zeros(self.nc, device=problem.old.device, dtype=torch.long)
        # Bound the dense fixed-factor matrices and repeated KPP inputs on the
        # 8 GB card. More candidates are evaluated in consecutive chunks.
        self.candidate_chunk = max(1, min(8, 4096 // self.nc))

    def _copy(self, count):
        if count not in self.copies:
            p = self.problem
            indices = torch.arange(self.nc, device=p.old.device).repeat(count)
            q = p if count == 1 else p.subset(indices)
            buoy = kpp.kpp_statekpp(q.old[0], q.old[1], q.aux["pressure"],
                                      gravity=q.par["gravity"], rho_const=q.par["rho_const"],
                                      eos_type=q.par["eos_type"], dbloc_form=DBLOC_FORM)[0] * q.interface_mask
            self.copies[count] = q, buoy
        return self.copies[count]

    def factors_of(self, Y, count=1):
        p, _ = self._copy(count)
        db = kpp.kpp_statekpp(Y[0], Y[1], p.aux["pressure"], gravity=p.par["gravity"],
                                rho_const=p.par["rho_const"], eos_type=p.par["eos_type"], dbloc_form=DBLOC_FORM)[0]
        ns = db.reshape(p.nz, p.nc).T.gather(1, self.interfaces.repeat(count, 1) - 1)
        ns = ns / self.spacing.repeat(count, 1)
        ratio = torch.clamp((self.b - torch.clamp(ns, min=self.b)) / self.b, max=1.)
        return ((1 - ratio * ratio) ** 3).reshape(count, self.nc, self.m)

    def evaluate(self, factors):
        """H, eliminated z and failure code, without any device readback."""
        count = factors.shape[0]
        p, original_buoy = self._copy(count)
        factors = factors.reshape(p.nc, self.m)
        ns = self.b * (1 - torch.sqrt(torch.clamp(1 - factors ** (1 / 3), min=0.)))
        buoy = original_buoy.reshape(p.nz, p.nc).T.scatter(
            1, self.interfaces.repeat(count, 1) - 1, ns * self.spacing.repeat(count, 1))
        buoy = buoy.T.reshape(p.nz, 1, p.nc)
        with dispatch_scope("kpp_blmix_cuda"):
            program = _fixed_factor_map_program if self.compiled else _fixed_factor_map
            zz, Y, reason = program(p, buoy)
        H = factors - self.factors_of(Y, count).reshape(p.nc, self.m)
        valid_H = torch.isfinite(H).all(1)
        reason = torch.where((reason == 0) & ~valid_H, 4, reason)
        valid = reason == 0
        return (torch.where(valid[:, None], H, torch.nan).reshape(count, self.nc, self.m),
                zz.reshape(count, self.nc, p.nv), reason.reshape(count, self.nc))

    def residual_of(self, zz):
        p = self.problem
        Y = p.old + p.unpack(zz) * p.scale
        actual = p.pack((Y - p.old) / p.scale)
        return actual, p.residual(actual, p.aux).abs().amax(-1)

    def newton(self, c, eligible, maxiter=50):
        """Original box Newton, independently masked through every stop."""
        H, zz, reason = self.evaluate(c[None])
        self.evaluations += eligible
        H, zz = H[0], zz[0]
        has_state = eligible & (reason[0] == 0)
        active = has_state.clone()
        # 0 iteration limit, 1 converged, 2 invalid start, 3 invalid difference,
        # 4 line search failed. Values are decoded together after the solve.
        status = torch.where(has_state, 0, 2)
        iterations = torch.zeros_like(status)
        powers = 2. ** -torch.arange(31, device=c.device, dtype=c.dtype)
        for iteration in range(maxiter + 1):
            iterations = torch.where(active, iteration, iterations)
            converged = active & (H.abs().amax(-1) <= 2. ** -50) if self.m else active
            status = torch.where(converged, 1, status)
            active = active & ~converged
            if iteration == maxiter or not bool(active.any()):
                break
            h = 2. ** -26
            sign = torch.where(c + h <= 1, 1., -1.).to(c.dtype)
            differences, difference_ok = [], []
            for first in range(0, self.m, self.candidate_chunk):
                last = min(self.m, first + self.candidate_chunk)
                perturb = torch.eye(self.m, device=c.device, dtype=c.dtype)[first:last]
                Hp, _, why = self.evaluate(c[None] + perturb[:, None] * sign[None] * h)
                differences.append(Hp)
                difference_ok.append(why == 0)
            self.evaluations += self.m * active
            Hp = torch.cat(differences)
            difference_valid = torch.cat(difference_ok).all(0)
            status = torch.where(active & ~difference_valid, 3, status)
            active &= difference_valid
            if not bool(active.any()):
                break
            J = (Hp - H[None]).permute(1, 2, 0) / (sign * h)[:, None]
            identity = torch.eye(self.m, device=c.device, dtype=c.dtype)
            safe_J = torch.where(active[:, None, None], J, identity)
            d, info = torch.linalg.solve_ex(safe_J, -torch.where(active[:, None], H, 0.))
            d_ok = (info == 0) & torch.isfinite(d).all(1)
            gradient = (-J.transpose(1, 2) @ H[:, :, None])[:, :, 0]
            norm = torch.linalg.vector_norm(H, dim=1)
            accepted = torch.zeros_like(active)
            nextc, nextH, nextz = c.clone(), H.clone(), zz.clone()
            for direction, usable in ((d, d_ok), (gradient, torch.ones_like(d_ok))):
                searching = active & ~accepted & usable
                if not bool(searching.any()):
                    continue
                self.evaluations += 31 * searching
                for first in range(0, 31, self.candidate_chunk):
                    steps = powers[first:first + self.candidate_chunk]
                    trials = torch.clamp(c[None] + steps[:, None, None] * direction[None], 0., 1.)
                    Ht, zt, why = self.evaluate(trials)
                    good = ((why == 0) & searching[None] & ~accepted[None]
                            & (torch.linalg.vector_norm(Ht, dim=-1)
                               <= (1 - 1e-4 * steps[:, None]) * norm[None]))
                    found = good.any(0)
                    pick = good.to(torch.int64).argmax(0)
                    column = torch.arange(self.nc, device=c.device)
                    nextc = torch.where(found[:, None], trials[pick, column], nextc)
                    nextH = torch.where(found[:, None], Ht[pick, column], nextH)
                    nextz = torch.where(found[:, None], zt[pick, column], nextz)
                    accepted |= found
            status = torch.where(active & ~accepted, 4, status)
            active &= accepted
            c, H, zz = nextc, nextH, nextz
        return c, zz, status, iterations, H.abs().amax(-1) if self.m else c.new_zeros(self.nc), has_state


def _select_scalar_state(e, iterate, chosen, xcur, tol):
    """Retain the existing precision representation of the canonical scalar root."""
    p = e.problem
    current = e.factors_of(p.old + p.unpack(iterate) * p.scale)[0, :, 0]
    endpoint = (xcur == 0) | (xcur == 1)
    # At an endpoint the fixed-factor physical equation has its unique
    # M-matrix solution. For an interior root retain the existing precision
    # representation only after comparing its complete scaled state with
    # the independently selected Brent state at the requested root accuracy.
    # A small residual alone cannot identify which root was selected.
    physical = p.old + p.unpack(chosen) * p.scale
    canonical = p.pack((physical - p.old) / p.scale)
    retained = torch.where(endpoint, current == xcur,
                           (iterate - canonical).abs().amax(-1) <= tol)
    chosen = torch.where(retained[:, None], iterate, chosen)
    actual, error = e.residual_of(chosen)
    actual = torch.where(retained[:, None], iterate, actual)
    return actual, error, retained


@torch.compile(fullgraph=True, backend="inductor", dynamic=True)
def _select_scalar_state_program(e, iterate, chosen, xcur, tol):
    return _select_scalar_state(e, iterate, chosen, xcur, tol)


@torch.compile(fullgraph=True, backend="inductor", dynamic=True)
def _scalar_factor_evaluation_program(elimination, factors):
    return _BatchedFactorElimination.evaluate(elimination, factors)


def _brent_head_device(active, iterations, xpre, fpre, xcur, fcur, xblk, fblk,
                       spre, scur, iteration_value):
    iterations = torch.where(active, iteration_value, iterations)
    opposite = active & (fpre != 0) & (fcur != 0) & (torch.signbit(fpre) != torch.signbit(fcur))
    xblk, fblk = torch.where(opposite, xpre, xblk), torch.where(opposite, fpre, fblk)
    spre, scur = torch.where(opposite, xcur - xpre, spre), torch.where(opposite, xcur - xpre, scur)
    exchange = active & (fblk.abs() < fcur.abs())
    oldcur, oldfcur = xcur, fcur
    xcur, fcur = torch.where(exchange, xblk, xcur), torch.where(exchange, fblk, fcur)
    xpre, fpre = torch.where(exchange, oldcur, xpre), torch.where(exchange, oldfcur, fpre)
    xblk, fblk = torch.where(exchange, oldcur, xblk), torch.where(exchange, oldfcur, fblk)
    delta = (1e-14 + 4 * torch.finfo(xcur.dtype).eps * xcur.abs()) / 2
    bisect = (xblk - xcur) / 2
    active &= (fcur != 0) & (bisect.abs() >= delta)
    return iterations, active, xpre, fpre, xcur, fcur, xblk, fblk, spre, scur, delta, bisect


def _brent_proposal_device(active, xpre, fpre, xcur, fcur, xblk, fblk,
                           spre, scur, delta, bisect, negative_fcur=None):
    interpolate = (spre.abs() > delta) & (fcur.abs() < fpre.abs())
    dpre = (fpre - fcur) / (xpre - xcur)
    dblk = (fblk - fcur) / (xblk - xcur)
    trial = torch.where(xpre == xblk,
                        (-fcur if negative_fcur is None else negative_fcur) * (xcur - xpre) / (fcur - fpre),
                        (-fcur if negative_fcur is None else negative_fcur) * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre)))
    use_trial = interpolate & (2 * trial.abs() < torch.minimum(spre.abs(), 3 * bisect.abs() - delta))
    spre, scur = torch.where(use_trial, scur, bisect), torch.where(use_trial, trial, bisect)
    xpre, fpre = xcur, fcur
    step = torch.where(scur.abs() > delta, scur, torch.where(bisect > 0, delta, -delta))
    xcur = torch.where(active, xcur + step, xcur)
    return spre, scur, xpre, fpre, xcur


def _brent_commit_device(active, fcur, zz, reasons, H, state, why):
    fcur = torch.where(active, H[0, :, 0], fcur)
    zz = torch.where(active[:, None], state[0], zz)
    reasons = torch.where(active & (why[0] != 0), why[0], reasons)
    active &= why[0] == 0
    return fcur, zz, reasons, active


_brent_head_program = torch.compile(_brent_head_device, fullgraph=True, backend="inductor", dynamic=True)
_brent_proposal_program = torch.compile(_brent_proposal_device, fullgraph=True, backend="inductor", dynamic=True)
_brent_commit_program = torch.compile(_brent_commit_device, fullgraph=True, backend="inductor", dynamic=True)


def _solve_scalar_batch(elimination, tol, *, iterate=None):
    """SciPy brentq's safeguarded interpolation, with one state per column."""
    e = elimination
    start = time.perf_counter()
    p = e.problem
    evaluate = e.evaluate
    if iterate is not None and e.compiled:
        e._copy(1)

        def evaluate(factors):
            return _scalar_factor_evaluation_program(e, factors)

    xpre, xcur = p.old.new_zeros(e.nc), p.old.new_ones(e.nc)
    fpre, zpre, why0 = evaluate(xpre[None, :, None])
    fcur, zz, why1 = evaluate(xcur[None, :, None])
    fpre, fcur, zz = fpre[0, :, 0], fcur[0, :, 0], zz[0]
    reasons = torch.where(why0[0] != 0, why0[0], why1[0])
    bracket = (fpre <= 0) & (fcur >= 0)
    reasons = torch.where((reasons == 0) & ~bracket, 5, reasons)
    at_zero = (reasons == 0) & (fpre == 0)
    xcur = torch.where(at_zero, 0., xcur)
    fcur = torch.where(at_zero, fpre, fcur)
    active = (reasons == 0) & ~at_zero & (fcur != 0)
    if iterate is not None and e.compiled:
        # Endpoint factors stay fixed throughout Brent. Keep their exact
        # states and evaluate only the original nonendpoint columns.
        positions = torch.nonzero(active).flatten()
        cached_H = fcur[None, :, None]
        cached_z = torch.where(at_zero[None, :, None], zpre, zz[None])
        cached_why = torch.where(at_zero[None], why0, why1)
        pending = None
        if positions.numel():
            sub = p.subset(positions)
            sub.compiled = e.compiled
            pending = _BatchedFactorElimination(
                sub, e.interfaces.index_select(0, positions).cpu().tolist())
            pending.copies[1] = (pending.problem, e._copy(1)[1].index_select(-1, positions))

        def evaluate(factors):
            if pending is None:
                return cached_H, cached_z, cached_why
            H, state, why = _scalar_factor_evaluation_program(
                pending, factors.index_select(1, positions))
            return (cached_H.index_copy(1, positions, H),
                    cached_z.index_copy(1, positions, state),
                    cached_why.index_copy(1, positions, why))

    iterations = torch.zeros_like(reasons)
    xblk, fblk = torch.zeros_like(xcur), torch.zeros_like(fcur)
    spre, scur = torch.zeros_like(xcur), torch.zeros_like(xcur)
    head = _brent_head_program if e.compiled else _brent_head_device
    proposal = _brent_proposal_program if e.compiled else _brent_proposal_device
    commit = _brent_commit_program if e.compiled else _brent_commit_device
    iteration_values = None
    for iteration in range(100):
        if not e.compiled and not bool(active.any()):
            break
        if e.compiled and iteration_values is None:
            # Prepare diagnostic integers once without a host activity read.
            # A 0-D tensor avoids specializing on each Python iteration.
            iteration_values = torch.arange(1, 101, device=xcur.device, dtype=iterations.dtype)
        iteration_value = iteration_values[iteration] if e.compiled else iteration + 1
        iterations, active, xpre, fpre, xcur, fcur, xblk, fblk, spre, scur, delta, bisect = head(
            active, iterations, xpre, fpre, xcur, fcur, xblk, fblk, spre, scur, iteration_value)
        # An inactive head preserves every selected value and iteration count.
        # One post-head read also handles inactivity from the preceding commit.
        if not bool(active.any()):
            break
        negative_fcur = None
        if e.compiled:
            # ponyTail: Triton's unary minus currently lowers to +0 - x,
            # losing -0 for x=+0. Keep the original negation outside this
            # program; fuse it again when lowering preserves signed zero.
            negative_fcur = -fcur
        spre, scur, xpre, fpre, xcur = proposal(
            active, xpre, fpre, xcur, fcur, xblk, fblk, spre, scur, delta, bisect, negative_fcur)
        H, state, why = evaluate(xcur[None, :, None])
        fcur, zz, reasons, active = commit(active, fcur, zz, reasons, H, state, why)
    reasons = torch.where(active, 6, reasons)
    # Exchange/end-point decisions can select a cached factor. Evaluate the
    # final factor together, as the scalar implementation's cache lookup does.
    H, zz, why = evaluate(xcur[None, :, None])
    retained = None
    if iterate is None:
        actual, error = e.residual_of(zz[0])
    else:
        program = _select_scalar_state_program if e.compiled else _select_scalar_state
        actual, error, retained = program(e, iterate, zz[0], xcur, tol)
    reasons = torch.where((reasons == 0) & (why[0] != 0), why[0], reasons)
    reasons = torch.where((reasons == 0) & (~torch.isfinite(error) | (error > tol)), 7, reasons)
    values = torch.stack((xcur, H[0, :, 0], iterations.to(xcur.dtype), error,
                          reasons.to(xcur.dtype)), 1).cpu().tolist()
    elapsed = time.perf_counter() - start
    messages = {1: "Invalid diffusivity in scalar elimination",
                2: "Fixed-factor matrix failed the conservative M-matrix check",
                3: "Invalid physical salinity in eliminated state", 4: "Nonfinite scalar residual",
                5: "Original factor interval does not bracket its residual",
                6: "Failed to converge after 100 iterations."}
    records, errors = [], []
    retained_values = retained.cpu().tolist() if retained is not None else None
    for column, (factor, scalar, iteration, residual, reason) in enumerate(values):
        if not reason:
            records.append(dict(factor=factor, scalar_residual=scalar, brent_converged=True,
                                brent_iterations=int(iteration), actual_physical_residual_inf=residual,
                                wall_s=elapsed))
            if retained_values is not None:
                records[-1]["retained_root"] = retained_values[column]
            errors.append(None)
        else:
            records.append(None)
            errors.append(messages[int(reason)] if reason != 7 else
                          f"Scalar root failed the original physical residual: {residual:g} > {tol:g}")
    return actual, records, errors


def _solve_factor_batch(elimination, iterate, tol):
    """Fixed start order and exact duplicate suppression, independently per column."""
    start = time.perf_counter()
    e, p = elimination, elimination.problem
    m = e.m
    initial = [("picard", e.factors_of(p.saved)[0]),
               ("newton_iterate", e.factors_of(p.old + p.unpack(iterate) * p.scale)[0]),
               ("zero", p.old.new_zeros((e.nc, m))), ("one", p.old.new_ones((e.nc, m)))]
    if m <= 4:
        initial += [(f"corner{k}", p.old.new_tensor([float((k >> j) & 1) for j in range(m)])
                     .expand(e.nc, m)) for k in range(1, 2 ** m - 1)]
    success = torch.zeros(e.nc, device=p.old.device, dtype=torch.bool)
    selected = torch.full((e.nc,), -1, device=p.old.device, dtype=torch.long)
    roots = torch.zeros_like(iterate)
    factors = p.old.new_zeros((e.nc, m))
    row_values, previous = [], []
    for number, (name, c0) in enumerate(initial):
        duplicate = torch.zeros_like(success)
        for other in previous:
            duplicate |= (c0 == other).all(1)
        previous.append(c0)
        eligible = ~success & ~duplicate
        if not bool(eligible.any()):
            continue
        c, zz, status, iterations, H_inf, has_state = e.newton(c0.clone(), eligible)
        actual, error = e.residual_of(zz)
        good = eligible & has_state & torch.isfinite(error) & (error <= tol)
        roots = torch.where(good[:, None], actual, roots)
        factors = torch.where(good[:, None], c, factors)
        selected = torch.where(good, number, selected)
        success |= good
        row_values.append((name, torch.stack((eligible.to(error.dtype), status.to(error.dtype),
                                             iterations.to(error.dtype), H_inf, error), 1)))
        if bool(success.all()):
            break
    selected_cpu = selected.cpu().tolist()
    factors_cpu = factors.cpu().tolist()
    counts_cpu = e.evaluations.cpu().tolist()
    rows_cpu = [(name, values.cpu().tolist()) for name, values in row_values]
    status_names = ("ITERATION_LIMIT", "CONVERGED", "INVALID_START", "INVALID_DIFFERENCE", "LINE_SEARCH_FAILED")
    elapsed = time.perf_counter() - start
    records, errors = [], []
    for column, choice in enumerate(selected_cpu):
        rows = []
        for name, values in rows_cpu:
            used, status, iteration, H_inf, error = values[column]
            if used:
                row = dict(status=status_names[int(status)], iterations=int(iteration), start=name)
                if int(status) != 2:
                    row.update(H_inf=H_inf, actual_physical_residual_inf=error)
                rows.append(row)
        if choice >= 0:
            row = rows[-1]
            records.append(dict(method="factor_box", factors=factors_cpu[column],
                                selected_start=initial[choice][0], starts=rows, H_inf=row["H_inf"],
                                actual_physical_residual_inf=row["actual_physical_residual_inf"],
                                evaluations=counts_cpu[column], wall_s=elapsed))
            errors.append(None)
        else:
            records.append(None)
            errors.append(rows)
    return roots, records, errors


def _recover_columns_device(problem, iterate, *, tol, policy):
    """Scalar and factor-box recovery; one bulk structural readback per station."""
    nc = problem.nc
    records, scalar_errors, factor_errors, interfaces = ([None] * nc for _ in range(4))
    roots = torch.zeros_like(iterate)
    if not nc:
        return roots, records, scalar_errors, factor_errors, interfaces
    structure = torch.stack((problem.ra[0].reshape(nc).long(), problem.ba[5].reshape(nc).long(),
                             (problem.aux["bl_4"] + problem.bl_kwargs.get("phepsi", 1e-10))
                             .reshape(nc).long()), 1).cpu().tolist()
    groups = {}
    for column, (kmt, kbl, case) in enumerate(structure):
        if case not in (0, 1):
            interfaces[column] = repr(ScalarRecoveryFailure("Unexpected KPP caseA index"))
            scalar_errors[column] = factor_errors[column] = "Unexpected KPP caseA index"
            continue
        kn = case * (kbl - 1) + (1 - case) * kbl
        support = sorted(({kn - 1, kn, kn + 1, kbl - 1} | set(range(max(1, kbl), kmt)))
                         & set(range(1, kmt)))
        interfaces[column] = support
        groups.setdefault(len(support), []).append(column)
        if len(support) != 1:
            scalar_errors[column] = f"Unsupported nonlinear coefficient rank: {len(support)}, interfaces={support}"
    for m, columns in groups.items():
        selected = torch.tensor(columns, device=iterate.device)
        p = problem.subset(selected)
        dependencies = [interfaces[column] for column in columns]
        e = _BatchedFactorElimination(p, dependencies)
        spacing_ok = bool((e.spacing > 0).all()) and e.b < 0
        if not spacing_ok:
            for column in columns:
                scalar_errors[column] = factor_errors[column] = "Invalid physical convective scales"
            continue
        pending = list(range(len(columns)))
        if m == 1:
            zz, result, errors = _solve_scalar_batch(e, tol)
            roots.index_copy_(0, selected, zz)
            pending = []
            for row, column in enumerate(columns):
                records[column], scalar_errors[column] = result[row], errors[row]
                if result[row] is None:
                    pending.append(row)
                else:
                    result[row]["effective_interfaces"] = interfaces[column]
        if not pending or policy not in FACTOR_POLICIES:
            continue
        if len(pending) != len(columns):
            take = torch.tensor(pending, device=iterate.device)
            selected = selected.index_select(0, take)
            columns = [columns[row] for row in pending]
            dependencies = [interfaces[column] for column in columns]
            p = p.subset(take)
            e = _BatchedFactorElimination(p, dependencies)
        zz, result, errors = _solve_factor_batch(e, iterate.index_select(0, selected), tol)
        roots.index_copy_(0, selected, zz)
        for row, column in enumerate(columns):
            record = result[row]
            records[column] = record
            if record is not None:
                record.update(effective_interfaces=interfaces[column], scalar_recovery=scalar_errors[column])
            else:
                factor_errors[column] = (f"No factor start reached the original residual: m={m}, "
                                         f"interfaces={interfaces[column]}, starts={errors[row]}")
    return roots, records, scalar_errors, factor_errors, interfaces


class _FactorElimination:
    """Exact elimination over the convective factors of one CPU column.

    c holds the factors P(N^2) at the column's m dependency interfaces. For
    fixed c the canonical KPP composition gives the coefficients and the column
    equation is linear; Y(c) is its solution, under the validity checks of the
    scalar elimination, and H(c) = c - P(N^2(Y(c))).
    """

    def __init__(self, problem, aux):
        if problem.old.device.type != "cpu":
            raise ScalarRecoveryFailure("Factor recovery must use its explicit CPU copy")
        self.problem, self.aux = problem, aux
        self.interfaces = _effective_interfaces(problem, aux)
        self.m = len(self.interfaces)
        self.spacings = [float(problem.zgrid[ki] - problem.zgrid[ki + 1]) for ki in self.interfaces]
        self.b = float(problem.ri_kwargs["BVSQcon"])
        if not all(s > 0 for s in self.spacings) or not self.b < 0:
            raise ScalarRecoveryFailure("Invalid physical convective scales")
        self.copies = {}
        self.evaluations = 0

    def factors_of(self, Y):
        """P(N^2) of state Y at the dependency interfaces: (columns, m)."""
        p = self.problem
        db = kpp.kpp_statekpp(Y[0], Y[1], self.aux["pressure"], gravity=p.par["gravity"],
                              rho_const=p.par["rho_const"], eos_type=p.par["eos_type"], dbloc_form=DBLOC_FORM)[0]
        if not self.m:
            return db.new_zeros((db[0].numel(), 0))
        ns = torch.stack([db[ki - 1].reshape(-1) / spacing
                          for ki, spacing in zip(self.interfaces, self.spacings, strict=True)], -1)
        ratio = torch.clamp((self.b - torch.clamp(ns, min=self.b)) / self.b, max=1.)
        return (1 - ratio * ratio) ** 3

    def evaluate(self, factors):
        """H, the packed eliminated state and validity for a batch of factor rows; invalid H is NaN."""
        n = factors.shape[0]
        self.evaluations += n
        if n not in self.copies:
            self.copies[n] = self.problem.subset(torch.zeros(n, dtype=torch.long), self.aux)
        p = self.copies[n]
        a = p.aux
        buoy = kpp.kpp_statekpp(p.old[0], p.old[1], a["pressure"], gravity=p.par["gravity"],
                                rho_const=p.par["rho_const"], eos_type=p.par["eos_type"], dbloc_form=DBLOC_FORM)[0] * p.interface_mask
        for j, (ki, spacing) in enumerate(zip(self.interfaces, self.spacings, strict=True)):
            # The inverse convective polynomial only feeds the canonical composition.
            buoy[ki - 1, 0] = self.b * (1 - torch.sqrt(torch.clamp(1 - factors[:, j] ** (1 / 3), min=0.))) * spacing
        K, gh = p.coeff_from_buoyancy(p.old, buoy, a)
        valid = (torch.isfinite(K) & ((K >= 0) | ~p.mask)).flatten(0, 2).all(0)
        F0 = p.residual_from_coefficients(p.old, K, gh, a)
        A = p.linear_jacobian_from_coefficients(F0.new_zeros(F0.shape), p.old, K, a)
        scales = p.pack(p.scale.expand_as(p.old))
        physical = scales[:, :, None] * A / scales[:, None, :]
        wet = p.pack(p.mask.expand_as(p.old)).bool()[0]
        mass = p.pack(p.volume.expand_as(p.old))[0][wet]
        block = physical[:, wet][:, :, wet]
        off = block - torch.diag_embed(block.diagonal(dim1=-2, dim2=-1))
        column_error = (mass[None, None] @ block)[:, 0] - mass
        bound = 100 * torch.finfo(A.dtype).eps * (mass[None, :, None] * block.abs()).sum(1)
        valid &= (torch.isfinite(A).flatten(1).all(1) & ~(off > 0).flatten(1).any(1)
                  & (block.diagonal(dim1=-2, dim2=-1) > 0).all(1) & (column_error.abs() <= bound).all(1))
        eye = torch.eye(p.nv, dtype=A.dtype).expand_as(A)
        zz, info = torch.linalg.solve_ex(torch.where(valid[:, None, None], A, eye), -F0[..., None])
        zz = zz[..., 0]
        valid &= (info == 0) & torch.isfinite(zz).all(1)
        Y = p.old + p.unpack(zz) * p.scale
        valid &= ((Y[1] >= 0) | ~p.mask).flatten(0, 1).all(0)
        H = factors - self.factors_of(Y)
        valid &= torch.isfinite(H).all(1)
        return torch.where(valid[:, None], H, torch.nan), zz, valid

    def residual_of(self, zz):
        """The returned packed state of one eliminated row and its complete original residual."""
        p = self.problem
        Y = p.old + p.unpack(zz[None]) * p.scale
        actual = p.pack((Y - p.old) / p.scale)
        return actual, float(p.residual(actual, self.aux).abs().max())

    def newton(self, c, maxiter=50, accept=None):
        """Projected damped Newton on the box from c; one-sided difference Jacobian.
        ``accept(c, packed_state)``, when given, is asked after every accepted step
        and stops the iteration with status ROOT_ACCEPTED when it returns True."""
        H, zz, valid = self.evaluate(c[None])
        if not bool(valid[0]):
            return c, None, dict(status="INVALID_START", iterations=0)
        H, zz = H[0], zz[0]
        status, iteration = "ITERATION_LIMIT", 0
        steps = 2. ** -torch.arange(31, dtype=c.dtype)
        for iteration in range(maxiter + 1):
            if not self.m or float(H.abs().max()) <= 2. ** -50:
                status = "CONVERGED"
                break
            if iteration == maxiter:
                break
            h = 2. ** -26
            sign = torch.where(c + h <= 1, 1., -1.).to(c.dtype)
            Hp, _, ok = self.evaluate(c[None] + torch.diag(sign * h))
            if not bool(ok.all()):
                status = "INVALID_DIFFERENCE"
                break
            J = (Hp - H[None]).T / (sign * h)[None]
            directions = []
            d, info = torch.linalg.solve_ex(J, -H)
            if int(info) == 0 and bool(torch.isfinite(d).all()):
                directions.append(d)
            directions.append(-J.T @ H)
            norm = torch.linalg.vector_norm(H)
            accepted = False
            for d in directions:
                trials = torch.clamp(c[None] + steps[:, None] * d[None], 0., 1.)
                Ht, zt, ok = self.evaluate(trials)
                good = ok & (torch.linalg.vector_norm(Ht, dim=1) <= (1 - 1e-4 * steps) * norm)
                if bool(good.any()):
                    k = int(torch.nonzero(good)[0])
                    c, H, zz = trials[k], Ht[k], zt[k]
                    accepted = True
                    break
            if not accepted:
                status = "LINE_SEARCH_FAILED"
                break
            if accept is not None and accept(c, zz):
                status = "ROOT_ACCEPTED"
                break
        return c, zz, dict(status=status, iterations=iteration, H_inf=float(H.abs().max()) if self.m else 0.)

    def starts(self, iterate):
        """Fixed start order: Picard prediction, Newton iterate, c=0, c=1, other corners (m <= 4)."""
        p, m = self.problem, self.m
        rows = [("picard", self.factors_of(p.saved)[0]),
                ("newton_iterate", self.factors_of(p.old + p.unpack(iterate) * p.scale)[0]),
                ("zero", p.old.new_zeros(m)), ("one", p.old.new_ones(m))]
        if m <= 4:
            rows += [(f"corner{k}", p.old.new_tensor([float((k >> j) & 1) for j in range(m)]))
                     for k in range(1, 2 ** m - 1)]
        unique = []
        for name, c in rows:
            if not any(torch.equal(c, u) for _, u in unique):
                unique.append((name, c))
        return unique


def _solve_factor_column(problem, aux, *, tol, iterate):
    """``newton_factor_box_v1``: the first start, in the fixed order, whose
    projected Newton root satisfies the original residual. H is continuous on
    [0, 1]^m and P maps into [0, 1], so a root exists; when several exist this
    selects the one Newton reaches from the earliest start (Picard first)."""
    start = time.perf_counter()
    rows = []
    with torch.no_grad(), dispatch_scope("kpp_blmix_cuda"):
        e = _FactorElimination(problem, aux)
        for name, c0 in e.starts(iterate):
            c, zz, row = e.newton(c0.clone())
            row["start"] = name
            rows.append(row)
            if zz is None:
                continue
            actual, error = e.residual_of(zz)
            row["actual_physical_residual_inf"] = error
            if math.isfinite(error) and error <= tol:
                return actual, dict(method="factor_box", effective_interfaces=e.interfaces,
                                    factors=c.tolist(), selected_start=name, starts=rows,
                                    H_inf=row["H_inf"], actual_physical_residual_inf=error,
                                    evaluations=e.evaluations, wall_s=time.perf_counter() - start)
    raise ScalarRecoveryFailure(
        f"No factor start reached the original residual: m={e.m}, interfaces={e.interfaces}, starts={rows}")


# Resource safeguards of the homotopy recovery, not convergence parameters:
# the first and the finest mesh, the pivots and the factor-map rows evaluated
# for one column, and the local Newton steps after each level. Running out of
# one raises HOMOTOPY_RESOURCE_LIMIT.
HOMOTOPY_FIRST_MESH = 0.5
HOMOTOPY_FINEST_MESH = 2. ** -40
HOMOTOPY_MAX_PIVOTS = 200000
HOMOTOPY_MAX_EVALUATIONS = 2000000
HOMOTOPY_POLISH_ITERATIONS = 10
HOMOTOPY_STATUSES = ("HOMOTOPY_INVALID_MAP", "HOMOTOPY_PIVOT_UNDECIDED", "HOMOTOPY_RESOURCE_LIMIT")


class HomotopyFailure(ScalarRecoveryFailure):
    """A stop of the homotopy recovery; ``status`` is one of HOMOTOPY_STATUSES."""

    def __init__(self, status, message):
        if status not in HOMOTOPY_STATUSES:
            raise ValueError(status)
        super().__init__(message)
        self.status = status


def _exact_solve(B, R):
    """X with B X = R in exact rational arithmetic on the binary64 entries
    (nested lists of floats in, of Fractions out); None when B is singular."""
    from fractions import Fraction

    n = len(B)
    A = [[Fraction(v) for v in b] + [Fraction(v) for v in r] for b, r in zip(B, R, strict=True)]
    for col in range(n):
        pivot = next((r for r in range(col, n) if A[r][col]), None)
        if pivot is None:
            return None
        A[col], A[pivot] = A[pivot], A[col]
        head = A[col][col]
        A[col] = [v / head for v in A[col]]
        for r in range(n):
            factor = A[r][col]
            if r != col and factor:
                A[r] = [a - factor * b for a, b in zip(A[r], A[col], strict=True)]
    return [row[n:] for row in A]


def _inverse_with_bound(B):
    """Float64 inverse X of the label basis and a componentwise bound on |X - B^-1|.

    X - B^-1 = B^-1 (I - B X); the residual is bounded by its computed value
    plus the rounding of its evaluation, n eps |B||X| + eps, and B^-1 by |X|
    plus the bound itself, taken to first order with a factor 8 of margin.
    (None, None) when the inverse is not finite or its residual exceeds 1e-8.
    """
    n = B.shape[0]
    eps = torch.finfo(B.dtype).eps
    try:
        X = torch.linalg.inv(B)
    except RuntimeError:
        return None, None
    residual = B @ X - torch.eye(n, dtype=B.dtype)
    if not bool(torch.isfinite(X).all()) or not float(residual.abs().max()) <= 1e-8:
        return None, None
    return X, 8 * X.abs() @ (residual.abs() + n * eps * (B.abs() @ X.abs()) + eps)


def _leaving_column(B, y):
    """Basis column that leaves when label y enters: the exact lexicographic
    minimum ratio rule over rows of B^-1 with positive entering coefficient.

    Float64 decides when its componentwise error bounds separate the first
    ratios of the certainly positive coefficients, and every coefficient of
    uncertain sign has a row whose ratio would exceed that minimum even at the
    coefficient's upper bound; otherwise the rule is evaluated exactly in
    rational arithmetic on the stored binary64 labels. Returns (column, decided_exactly)."""
    n = B.shape[0]
    eps = torch.finfo(B.dtype).eps
    X, bound_x = _inverse_with_bound(B)
    if X is not None:
        alpha = X @ y
        r = B @ alpha - y
        bound_a = 8 * (X.abs() + bound_x) @ (r.abs() + n * eps * (B.abs() @ alpha.abs() + y.abs()))
        rows = torch.nonzero(alpha > bound_a).flatten()
        if bool(torch.isfinite(alpha).all()) and rows.numel():
            a = alpha[rows]
            first = X[rows, 0] / a
            error = (bound_x[rows, 0] + first.abs() * bound_a[rows]) / (a - bound_a[rows])
            k = int(first.argmin())
            gap = first - first[k] - error - error[k]
            gap[k] = 1.
            uncertain = alpha.abs() <= bound_a
            lower = (X[:, 0] - bound_x[:, 0]) / (alpha.abs() + bound_a)
            excluded = (X[:, 0] > bound_x[:, 0]) & (lower > first[k] + error[k])
            if bool((gap > 0).all()) and bool((excluded | ~uncertain).all()):
                return int(rows[k]), False
    exact = _exact_solve(B.tolist(), [[float(i == j) for j in range(n)] + [float(y[i])] for i in range(n)])
    if exact is None:
        raise HomotopyFailure("HOMOTOPY_PIVOT_UNDECIDED", "Singular label basis")
    rows = [i for i in range(n) if exact[i][n] > 0]
    if not rows:
        raise HomotopyFailure("HOMOTOPY_PIVOT_UNDECIDED", "No leaving column for the entering label")
    return min(rows, key=lambda i: [v / exact[i][n] for v in exact[i][:n]]), True


def _facet_weights(B):
    """Barycentric weights w of the zero on a completely labelled facet,
    B w = e1: checked nonnegative with unit sum, exactly when float64 is not conclusive.
    Returns (w, decided_exactly)."""
    n = B.shape[0]
    X, bound_x = _inverse_with_bound(B)
    if X is not None:
        w, bound_w = X[:, 0], bound_x[:, 0]
        if bool((w > bound_w).all()) and abs(float(w.sum()) - 1) <= float(bound_w.sum()) + n * torch.finfo(B.dtype).eps:
            return w, False
    exact = _exact_solve(B.tolist(), [[float(i == 0)] for i in range(n)])
    if exact is None:
        raise HomotopyFailure("HOMOTOPY_PIVOT_UNDECIDED", "Singular terminal facet")
    w = [row[0] for row in exact]
    if any(v < 0 for v in w) or sum(w) != 1:
        raise HomotopyFailure("HOMOTOPY_PIVOT_UNDECIDED", "Terminal facet weights are not a convex combination")
    return B.new_tensor([float(v) for v in w]), True


class _MerrillHomotopy:
    """Piecewise-linear fixed-point homotopy for a map G of [0,1]^m into itself.

    One level triangulates R^m x [0, 1] by the Freudenthal triangulation with
    x-mesh ``delta`` and one t-layer. Vertices at t=0 carry the label x - a and
    vertices at t=1 the label x - G(proj x), proj the projection on the box
    (Merrill's restart algorithm; Allgower and Georg 1990, Numerical
    Continuation Methods, ch. 12-13). The path starts at the t=0 facet that
    contains the anchor a in its interior, moves by door-in door-out steps under
    the lexicographic minimum ratio rule, and ends at a completely labelled t=1
    facet; the zero of the piecewise-linear interpolant there is returned. Any
    zero of the interpolant is a convex combination of a and of values of G, so
    the path stays near the box; with exact pivot decisions it ends after
    finitely many pivots, and |x - G(x)| at the returned zero is bounded by the
    variation of G over one simplex. ``fixed_map(c)`` returns G(c) for c in the
    box or raises HomotopyFailure; ``on_pivot(basis, B, exact)``, when set, sees
    every completely labelled facet.
    """

    def __init__(self, fixed_map, m, max_pivots, on_pivot=None):
        self.fixed_map = fixed_map
        self.m = m
        self.max_pivots = max_pivots
        self.on_pivot = on_pivot
        self.pivots = 0
        self.exact_pivots = 0
        self.labels = 0

    def level(self, a, delta):
        """The zero on the terminal facet of one mesh level and the level's pivot count."""
        m, dtype = self.m, a.dtype
        # a lies inside the first simplex, at weights m/(m+1), ..., 1/(m+1).
        origin = a - delta * torch.arange(m, 0, -1, dtype=dtype) / (m + 1)
        cache = {}

        def point(v):
            return origin + delta * torch.tensor(v[:m], dtype=dtype)

        def label(v):
            if v not in cache:
                x = point(v)
                if v[m] == 0:
                    h = x - a
                else:
                    self.labels += 1
                    h = x - self.fixed_map(x.clamp(0., 1.))
                if not bool(torch.isfinite(h).all()):
                    raise HomotopyFailure("HOMOTOPY_INVALID_MAP", "Nonfinite homotopy label")
                cache[v] = torch.cat((h.new_ones(1), h))
            return cache[v]

        def vertices(base, perm):
            out = [base]
            for d in perm:
                v = list(out[-1])
                v[d] += 1
                out.append(tuple(v))
            return out

        base, perm = (0,) * (m + 1), list(range(m + 1))  # coordinate m is t, stepped last
        V = vertices(base, perm)
        basis = V[:m + 1]
        B = torch.stack([label(v) for v in basis], 1)
        _facet_weights(B)  # the start facet is completely labelled
        enter = V[m + 1]
        pivots = 0
        while True:
            pivots += 1
            if self.pivots + pivots > self.max_pivots:
                raise HomotopyFailure("HOMOTOPY_RESOURCE_LIMIT", f"Pivot limit {self.max_pivots}")
            y = label(enter)
            out, exact = _leaving_column(B, y)
            self.exact_pivots += exact
            leave = basis[out]
            basis[out] = enter
            B[:, out] = y
            if self.on_pivot is not None:
                self.on_pivot(list(basis), B.clone(), exact)
            if all(v[m] == 1 for v in basis):
                w, _ = _facet_weights(B)
                self.pivots += pivots
                return w @ torch.stack([point(v) for v in basis]), pivots
            old = vertices(base, perm)
            k = old.index(leave)
            if k == 0:
                base = old[1]
                perm = perm[1:] + perm[:1]
                enter = vertices(base, perm)[m + 1]
            elif k == m + 1:
                if perm[-1] == m:
                    raise HomotopyFailure("HOMOTOPY_PIVOT_UNDECIDED", "Homotopy path returned to t=0")
                step = list(base)
                step[perm[-1]] -= 1
                base, perm = tuple(step), perm[-1:] + perm[:-1]
                enter = base
            else:
                perm = perm[:k - 1] + [perm[k], perm[k - 1]] + perm[k + 1:]
                enter = vertices(base, perm)[k]
            if set(vertices(base, perm)) != set(basis) | {enter}:
                raise HomotopyFailure("HOMOTOPY_PIVOT_UNDECIDED", "Simplex pivot lost the shared facet")


def _solve_factor_homotopy(problem, aux, *, tol, on_pivot=None):
    """``newton_factor_homotopy_v1`` after every box start failed.

    Merrill levels from the Picard prediction with the mesh halved per level.
    The projected zero of each level is checked against the complete original
    residual of its eliminated state and returned when it meets tol; otherwise
    at most HOMOTOPY_POLISH_ITERATIONS box Newton steps follow, each step
    checked the same way, and a failed polish leaves the level's zero as the
    next anchor. Every stop other than an accepted root raises HomotopyFailure.
    """
    start = time.perf_counter()
    levels = []
    with torch.no_grad(), dispatch_scope("kpp_blmix_cuda"):
        e = _FactorElimination(problem, aux)
        if not e.m:
            raise HomotopyFailure("HOMOTOPY_INVALID_MAP", "No convective factor to recover")

        def spend():
            if e.evaluations > HOMOTOPY_MAX_EVALUATIONS:
                raise HomotopyFailure("HOMOTOPY_RESOURCE_LIMIT",
                                      f"Factor-map evaluation limit {HOMOTOPY_MAX_EVALUATIONS}")

        def fixed_map(c):
            spend()
            H, _, valid = e.evaluate(c[None])
            if not bool(valid[0]):
                raise HomotopyFailure("HOMOTOPY_INVALID_MAP", f"Factor map invalid at c={c.tolist()}")
            return c - H[0]

        path = _MerrillHomotopy(fixed_map, e.m, HOMOTOPY_MAX_PIVOTS, on_pivot)
        x = e.factors_of(problem.saved)[0]
        delta = HOMOTOPY_FIRST_MESH
        while delta >= HOMOTOPY_FINEST_MESH:
            x, pivots = path.level(x, delta)
            c = x.clamp(0., 1.)
            row = dict(mesh=delta, pivots=pivots)
            levels.append(row)
            H, zz, valid = e.evaluate(c[None])
            if not bool(valid[0]):
                raise HomotopyFailure("HOMOTOPY_INVALID_MAP", f"Factor map invalid at c={c.tolist()}")
            actual, error = e.residual_of(zz[0])
            row.update(H_inf=float(H.abs().max()), endpoint_residual_inf=error)
            found = {}
            if math.isfinite(error) and error <= tol:
                found.update(c=c, actual=actual, error=error, at="level_zero")
            else:
                def accept(cc, packed):
                    state, err = e.residual_of(packed)
                    if math.isfinite(err) and err <= tol:
                        found.update(c=cc, actual=state, error=err, at="newton")
                        return True
                    return False
                _, _, polish = e.newton(c.clone(), maxiter=HOMOTOPY_POLISH_ITERATIONS, accept=accept)
                row["newton"] = polish
            spend()
            if found:
                return found["actual"], dict(
                    method="factor_homotopy", status="ROOT_ACCEPTED", accepted_at=found["at"],
                    effective_interfaces=e.interfaces, factors=found["c"].tolist(), levels=levels,
                    actual_physical_residual_inf=found["error"], pivots=path.pivots,
                    exact_pivots=path.exact_pivots, labels=path.labels, evaluations=e.evaluations,
                    wall_s=time.perf_counter() - start)
            delta /= 2
    raise HomotopyFailure("HOMOTOPY_RESOURCE_LIMIT",
                          f"Finest mesh {HOMOTOPY_FINEST_MESH} without the original residual: m={e.m}, "
                          f"interfaces={e.interfaces}, last levels={levels[-3:]}")


def _solve_scalar_column(problem, aux, *, tol):
    """Solve H(c) = c - P(N2(Y(c))) on the physical interval [0, 1].

    Fixed-c linear solves eliminate T/S inside this root algorithm only; the
    returned root is checked against the complete nonlinear residual.
    """
    from scipy.optimize import brentq

    interfaces = _effective_interfaces(problem, aux)
    if len(interfaces) != 1:
        raise ScalarRecoveryFailure(
            f"Unsupported nonlinear coefficient rank: {len(interfaces)}, interfaces={interfaces}")
    if problem.old.device.type != "cpu":
        raise ScalarRecoveryFailure("Bounded scalar recovery must use its explicit CPU copy")
    ki = interfaces[0]
    spacing = float(problem.zgrid[ki] - problem.zgrid[ki + 1])
    b = float(problem.ri_kwargs["BVSQcon"])
    if not spacing > 0 or not b < 0:
        raise ScalarRecoveryFailure("Invalid physical convective scales")
    zero = torch.zeros((1, problem.nv), dtype=problem.old.dtype)
    history = []
    cache = {}
    start = time.perf_counter()
    with torch.no_grad(), dispatch_scope("kpp_blmix_cuda"):
        original_buoy = kpp.kpp_statekpp(problem.old[0], problem.old[1], aux["pressure"],
                                         gravity=problem.par["gravity"],
                                         rho_const=problem.par["rho_const"],
                                         eos_type=problem.par["eos_type"], dbloc_form=DBLOC_FORM)[0] * problem.interface_mask

        def evaluate(c):
            c = float(c)
            if c in cache:
                return cache[c]
            if not 0 <= c <= 1:
                raise ScalarRecoveryFailure("Convective factor left [0,1]")
            # Invert the convective polynomial only to supply the canonical
            # composition with a buoyancy jump; this value is not a state.
            ns = b * (1 - math.sqrt(max(0., 1 - c ** (1 / 3))))
            buoy = original_buoy.clone()
            buoy[ki - 1] = ns * spacing
            K, gh = problem.coeff_from_buoyancy(problem.old, buoy, aux)
            if not torch.isfinite(K).all() or bool((K[:, problem.mask] < 0).any()):
                raise ScalarRecoveryFailure("Invalid diffusivity in scalar elimination")
            F0 = problem.residual_from_coefficients(problem.old, K, gh, aux)
            A = problem.linear_jacobian_from_coefficients(zero, problem.old, K, aux)
            scales = problem.pack(problem.scale.expand_as(problem.old))[0]
            physical_A = scales[:, None] * A[0] / scales[None, :]
            wet = problem.pack(problem.mask.expand_as(problem.old)).bool()[0]
            mass = problem.pack(problem.volume.expand_as(problem.old))[0][wet]
            block = physical_A[wet][:, wet]
            off = block - torch.diag(block.diagonal())
            column_error = mass @ block - mass
            bound = 100 * torch.finfo(A.dtype).eps * (mass[:, None] * block.abs()).sum(0)
            if (not torch.isfinite(A).all() or bool((off > 0).any()) or bool((block.diagonal() <= 0).any())
                    or bool((column_error.abs() > bound).any())):
                raise ScalarRecoveryFailure("Fixed-factor matrix failed the conservative M-matrix check")
            zz = torch.linalg.solve(A, -F0[..., None])[..., 0]
            Y = problem.old + problem.unpack(zz) * problem.scale
            if not torch.isfinite(Y).all() or bool((Y[1][problem.mask] < 0).any()):
                raise ScalarRecoveryFailure("Invalid physical salinity in eliminated state")
            db = kpp.kpp_statekpp(Y[0], Y[1], aux["pressure"], gravity=problem.par["gravity"],
                                  rho_const=problem.par["rho_const"], eos_type=problem.par["eos_type"], dbloc_form=DBLOC_FORM)[0]
            actual_ns = float(db[ki - 1, 0, 0]) / spacing
            ratio = min((b - max(actual_ns, b)) / b, 1.)
            polynomial = (1 - ratio * ratio) ** 3
            H = c - polynomial
            if not math.isfinite(H):
                raise ScalarRecoveryFailure("Nonfinite scalar residual")
            actual = problem.pack((Y - problem.old) / problem.scale)
            history.append(dict(factor=c, residual=H, N2=actual_ns))
            cache[c] = (H, actual, Y)
            return cache[c]

        f0 = evaluate(0.)[0]
        f1 = evaluate(1.)[0]
        if not f0 <= 0 <= f1:
            raise ScalarRecoveryFailure("Original factor interval does not bracket its residual")
        if f0 == 0. or f1 == 0.:
            factor = 0. if f0 == 0. else 1.
            iterations = 0
            converged = True
        else:
            factor, bracket = brentq(lambda c: evaluate(c)[0], 0., 1., xtol=1e-14,
                                     rtol=4 * torch.finfo(problem.old.dtype).eps, maxiter=100,
                                     full_output=True, disp=True)
            iterations = int(bracket.iterations)
            converged = bool(bracket.converged)
        H, z, Y = evaluate(factor)
        F = problem.residual(z, aux)
        actual_error = float(F.abs().max())
        if not math.isfinite(actual_error) or actual_error > tol:
            raise ScalarRecoveryFailure(
                f"Scalar root failed the original physical residual: {actual_error:g} > {tol:g}")
    return z, dict(effective_interfaces=interfaces, factor=factor, scalar_residual=H,
                   brent_converged=converged, brent_iterations=iterations,
                   actual_physical_residual_inf=actual_error,
                   wall_s=time.perf_counter() - start)


class _ImplicitColumn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, problem, record, *args):
        aux = dict(zip(problem.aux_names, (a.detach() for a in args), strict=True))
        from defgcm.dynamics._kpp_convective_replay import replay_root, retain_root

        tolerance = problem.root_tolerance
        reused = replay_root(problem.old, identity=problem.replay_identity,
                             enabled=problem.checkpoint_reuse)
        if reused is None:
            if problem.saved is None:
                raise RuntimeError("implicit KPP needs a lagged start outside root replay")
            z0 = problem.pack((problem.saved - problem.old) / problem.scale)
            z, solve = problem.solve(z0, tolerance, aux=aux)
            if solve["status"] != "CONVERGED":
                raise _root_failure(solve, problem, z, aux, z0)
            Y = problem.old + problem.unpack(z) * problem.scale
        else:
            Y, solve = reused
            z0 = None
        record["forward"] = solve
        record["replayed"] = reused is not None
        # Accept the residual of the returned rounded physical state.
        actual_z = problem.pack((Y - problem.old) / problem.scale)
        residual_value = problem.residual(actual_z, aux=aux).abs().max()
        if problem.dry_rhs_finite is None:
            residual = float(residual_value)
        else:
            residual, dry_finite = torch.stack(
                (residual_value, problem.dry_rhs_finite.to(residual_value.dtype))).tolist()
            if not dry_finite:
                raise _root_failure(dict(status="NONFINITE_DRY_RHS"), problem, actual_z, aux, z0)
        record["returned_state_residual_inf"] = residual
        if not math.isfinite(residual) or residual > tolerance:
            raise _root_failure(dict(status="RETURNED_STATE_RESIDUAL", value=residual, tolerance=tolerance),
                                problem, actual_z, aux, z0)
        if reused is None:
            retain_root(Y, solve, identity=problem.replay_identity,
                        enabled=problem.checkpoint_reuse)
        ctx.problem = problem
        ctx.record = record
        ctx.flags = tuple(a.requires_grad for a in args)
        ctx.save_for_backward(Y, *args)
        return Y

    @staticmethod
    def backward(ctx, seed):
        problem = ctx.problem
        record = ctx.record
        Y, *saved = ctx.saved_tensors
        z = problem.pack((Y.detach() - problem.old) / problem.scale)
        args = [a.detach().requires_grad_(flag) for a, flag in zip(saved, ctx.flags, strict=True)]
        aux = dict(zip(problem.aux_names, args, strict=True))
        with torch.enable_grad():
            residual = problem.residual(z, aux=aux)
            primal = float(residual.detach().abs().max())
            tolerance = problem.root_tolerance
            if not math.isfinite(primal) or primal > tolerance:
                raise _root_failure(dict(status="RECOMPUTED_RESIDUAL", value=primal, tolerance=tolerance),
                                    problem, z, aux)
            singular = [r["column"] for r in record["forward"].get("recovered_columns", ())
                        if r.get("F_z_rank_deficient")]
            if singular:
                raise _root_failure(dict(status="SINGULAR_ROOT_NO_IFT", failed_column=singular[0],
                                         singular_columns=singular), problem, z, aux)
            scaled_seed = problem.pack(seed * problem.scale)
            magnitude = scaled_seed.abs().max()
            if not torch.isfinite(magnitude):
                raise FloatingPointError("Nonfinite implicit seed")
            if float(magnitude):
                rhs = scaled_seed / magnitude
                lam, linearization = problem.solve_linear(z, rhs, aux, transpose=True)
                discrepancy = problem.apply_linear(linearization, lam, transpose=True) - rhs
                del linearization
                relative = float(_nrm(discrepancy) / _nrm(rhs))
                if not math.isfinite(relative) or relative > ADJOINT_TOLERANCE:
                    raise _root_failure(dict(status="ADJOINT_RESIDUAL", value=relative,
                                             tolerance=ADJOINT_TOLERANCE),
                                        problem, z, aux)
                active = [a for a in args if a.requires_grad]
                grads = torch.autograd.grad(residual, active, -lam * magnitude, allow_unused=True) if active else []
            else:
                relative = 0.
                grads = [torch.zeros_like(a) for a in args if a.requires_grad]
        iterator = iter(grads)
        result = [next(iterator) if flag else None for flag in ctx.flags]
        if any(g is not None and not torch.isfinite(g).all() for g in result):
            raise FloatingPointError("Nonfinite full external implicit pullback")
        record["backward"] = dict(relative_residual=relative, primal_residual_inf=primal)
        _STATISTICS["backward_stations"] += 1
        _STATISTICS["max_adjoint_residual"] = max(_STATISTICS["max_adjoint_residual"], relative)
        return (None, None, *result)


def _crop(value, ny, nx, halo):
    if (isinstance(value, torch.Tensor) and value.ndim >= 2
            and tuple(value.shape[-2:]) == (ny + 2 * halo, nx + 2 * halo)):
        value = value[..., halo:halo + ny, halo:halo + nx]
    return value


def mix_convective_implicit(temp_rhs, salt_rhs, *, initial, temp_reference, salt_reference,
                            kpp_fields, inputs: KppMixingInputs, layer_thickness, grid, dt,
                            tracer_w, halo):
    """(T, S) after the implicit convective mixing station.

    ``initial`` is the lagged-scheme solution, the Newton start, or None when
    the checkpoint scope supplies the accepted root. The other arguments are
    those of :func:`kpp_mixing.mix_kpp_tracers`; ``halo`` is the width of the
    haloed KPP slab.
    """
    if forward_mode_active():
        # Its Newton programs are Inductor-compiled and its column solve is a
        # Function with a backward only: a tangent would be dropped or refused
        # deep inside; say so at the entry.
        raise NotImplementedError(
            "kpp_convective_implicit has no forward-mode derivative; run forward-mode "
            "derivatives without the implicit convective KPP")
    config = inputs.config
    if config.execution == "compiled":
        _require_compiled_execution()
    params = inputs.params
    if kpp._canonical_kpp_eos(str(params["eos_type"])) != "jmd95":
        raise ValueError("kpp_convective_implicit supports the JMD95 KPP buoyancy only")
    if params["dbloc_form"] != DBLOC_FORM:
        raise ValueError(f"kpp_convective_implicit requires KPP dbloc_form={DBLOC_FORM!r}: its "
                         "column coefficients and Jacobian use the JMD95 divided difference")
    if temp_rhs.ndim != 4 or temp_rhs.shape[0] != 1:
        raise ValueError("kpp_convective_implicit supports a single regional batch")
    if temp_rhs.dtype != torch.float64:
        # The root tolerance and the adjoint residual bound are float64 quantities.
        raise ValueError("kpp_convective_implicit requires float64 tracers")
    from defgcm.dynamics.implicit_vertical_advection import (
        tracer_hfac_factors, build_tracer_vertical_metrics,
    )

    nz, ny, nx = temp_rhs.shape[-3:]
    shape = (nz, ny, nx)
    device = temp_rhs.device
    b = ConvectiveColumnProblem()
    b.device = device
    b.shape = shape
    b.nz, b.ny, b.nx = nz, ny, nx
    b.nc = ny * nx
    b.nv = 2 * nz
    b.dt = float(dt)
    b.par = params
    b.root_policy = config.root_policy
    b.root_tolerance = config.root_tolerance
    b.polish_steps = config.polish_steps
    b.compiled = config.execution == "compiled"
    b.linear_solver = config.linear_solver
    b.checkpoint_reuse = config.checkpoint_reuse
    b.source_nx, b.source_ny = nx, ny
    b.ri_kwargs = inputs.ri_kwargs
    b.bl_kwargs = inputs.bl_kwargs
    b.ra = [inactive(_crop(x, ny, nx, halo)) for x in inputs.ri_args]
    b.ba = [inactive(_crop(x, ny, nx, halo) if i < 7 else x)
            for i, x in enumerate(inputs.bl_args)]
    b.zgrid = b.ra[6]
    b.dz = layer_thickness.detach().to(device=device, dtype=torch.float64)
    b.dc = .5 * (b.dz[:-1] + b.dz[1:])
    b.hfac = grid.masks.hFacC.to(dtype=torch.float64, device=device).reshape(shape)
    b.mask = b.hfac > 0
    b.volume = b.hfac * b.dz[:, None, None]
    b.inv_volume = torch.where(b.mask, 1 / b.volume.clamp_min(1e-30), 0.)
    b.pair = b.mask[:-1] & b.mask[1:]
    b.interface_mask = torch.cat((b.pair, torch.zeros_like(b.mask[:1])), 0)
    # Boundary-owned cells take no continuity correction, as in
    # add_tracer_vertical_continuity.
    mask_c, recip_hfac_current = tracer_hfac_factors(
        field=temp_rhs, hfac_c=getattr(grid.masks, "hFacC", None),
    )
    b.interior = build_tracer_vertical_metrics(
        temp_rhs, grid, mask_c_current=mask_c,
        recip_hfac_matrix_c=recip_hfac_current,
        recip_hfac_current_c=recip_hfac_current,
    ).mask_in_c.to(
        dtype=torch.float64, device=device).reshape(ny, nx)
    scale = []
    for values in (config.residual_scale_temp, config.residual_scale_salt):
        if values is None:
            scale.append(torch.ones(nz, dtype=torch.float64, device=device))
        elif len(values) != nz:
            raise ValueError("kpp_convective_implicit.residual_scale needs one value per level")
        else:
            scale.append(torch.tensor(values, dtype=torch.float64, device=device))
    b.scale = torch.stack(scale)[:, :, None, None]
    b.old = torch.stack([v.detach().reshape(shape) for v in (inputs.temp, inputs.salt)])
    b.saved = (None if initial is None else
               torch.stack([v.detach().reshape(shape) for v in initial]))
    aux = dict(
        rhs=torch.stack([temp_rhs.reshape(shape), salt_rhs.reshape(shape)]),
        tref=torch.stack([temp_reference.reshape(shape), salt_reference.reshape(shape)]),
        pressure=inputs.pressure.reshape(shape),
        w=tracer_w.data.reshape(nz + 1, ny, nx),
        sf=torch.stack([kpp_fields.surf_forc_t_nl.reshape(ny, nx), kpp_fields.surf_forc_s.reshape(ny, nx)]))
    for i in (1, 3, 4, 5):
        aux["ri_" + str(i)] = _crop(inputs.ri_args[i], ny, nx, halo)
    for i in (0, 1, 2, 3, 4):
        aux["bl_" + str(i)] = _crop(inputs.bl_args[i], ny, nx, halo)
    if tuple(aux) != AUX_NAMES:
        raise RuntimeError("Implicit external argument layout changed")
    b.aux_names = AUX_NAMES
    full_rhs = aux["rhs"]
    column_ids = None
    if config.compact_columns:
        # The columns of F_Y are independent. Dry columns solve Y=rhs exactly;
        # gather/scatter remain outside the IFT boundary to preserve F_X.
        wet_columns = b.mask.reshape(nz, b.nc).any(0)
        column_ids = torch.nonzero(wet_columns).flatten()
        if column_ids.numel() < b.nc:
            # The closed-form dry equation still requires a finite RHS. This
            # scalar shares the returned-root readback with the wet residual.
            b.dry_rhs_finite = (torch.isfinite(full_rhs.detach()).reshape(2, nz, b.nc)
                                | wet_columns[None, None, :]).all()
            b.column_ids = torch.arange(b.nc, device=device)
            b = b.subset(column_ids, aux)
            aux = b.aux
    b.aux = {n: v.detach() for n, v in aux.items()}
    from defgcm.dynamics._kpp_convective_replay import segment_active

    static_identity = None
    if segment_active(enabled=b.checkpoint_reuse):
        # Replay belongs to this static grid, including its wet-column layout.
        # Use the persistent owner, not the temporary cast/reshape above.
        source_mask = grid.masks.hFacC
        if torch.is_inference(source_mask):
            raise ValueError("implicit KPP checkpoint reuse requires an ordinary grid mask; "
                             "prepare the grid outside inference mode or set checkpoint_reuse=False")
        static_identity = (id(source_mask), source_mask._version)
    b.replay_identity = (
        SCHEME, shape, b.shape, static_identity,
        b.dt, config.root_policy, config.root_tolerance,
        config.polish_steps, config.execution, config.linear_solver, config.compact_columns,
        config.residual_scale_temp, config.residual_scale_salt,
    )
    record = dict(station=_STATISTICS["stations"])
    if b.nc:
        Y = _ImplicitColumn.apply(b, record, *[aux[n] for n in AUX_NAMES])
        if column_ids is not None and b.nc < ny * nx:
            Y = full_rhs.reshape(2, nz, ny * nx).index_copy(
                -1, column_ids, Y.reshape(2, nz, b.nc)).reshape((2,) + shape)
    else:
        if not bool(b.dry_rhs_finite):
            raise _root_failure(dict(status="NONFINITE_DRY_RHS"), b, None, {"rhs": full_rhs})
        Y = full_rhs
        record.update(forward=dict(status="CONVERGED", history=[], polishing=[]),
                      returned_state_residual_inf=0.0, replayed=False)
    _STATISTICS["total_columns"] += ny * nx
    _STATISTICS["solved_columns"] += b.nc
    _STATISTICS["dry_columns"] += ny * nx - b.nc
    _STATISTICS["stations"] += 1
    if record["replayed"]:
        _STATISTICS["replayed_stations"] += 1
    elif b.nc:
        _STATISTICS["newton_stations"] += 1
        primary = record["forward"].get("primary", record["forward"])
        rows = [row for row in primary.get("history", ()) if "line_search_steps" in row]
        _STATISTICS["newton_iterations"] += len(rows)
        _STATISTICS["max_newton_iterations"] = max(_STATISTICS["max_newton_iterations"], len(rows))
        _STATISTICS["line_search_evaluations"] += sum(row["line_search_steps"] for row in rows)
        _STATISTICS["max_line_search_steps"] = max(
            _STATISTICS["max_line_search_steps"], max((row["line_search_steps"] for row in rows), default=0))
        _STATISTICS["active_column_iterations"] += sum(row["active_columns"] for row in rows)
        polishing = record["forward"].get("polishing", ())
        _STATISTICS["polish_iterations"] += len(polishing)
        _STATISTICS["polish_improved_columns"] += sum(row["improved_columns"] for row in polishing)
    _STATISTICS["max_primal_residual"] = max(_STATISTICS["max_primal_residual"],
                                             record["returned_state_residual_inf"])
    recovered = () if record["replayed"] else record["forward"].get("recovered_columns", ())
    _STATISTICS["recovered_columns"] += len(recovered)
    _STATISTICS["factor_recovered_columns"] += sum(r.get("method") == "factor_box" for r in recovered)
    for r in recovered:
        if r.get("method") == "factor_homotopy":
            _STATISTICS["homotopy_recovered_columns"] += 1
            for key in ("pivots", "labels", "evaluations"):
                _STATISTICS["homotopy_" + key] += r[key]
        if "F_z_condition" in r:
            _STATISTICS["max_recovered_condition"] = max(_STATISTICS["max_recovered_condition"],
                                                          r["F_z_condition"])
    if not record["replayed"]:
        _STATISTICS["recovery_wall_s"] += record["forward"].get("recovery_wall_s", 0.)
    return Y[0].reshape_as(temp_rhs), Y[1].reshape_as(salt_rhs)
