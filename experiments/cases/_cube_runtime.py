"""Cube-global yaml-driven runtime.

Config-driven promotion of the validated cube runner: the step loop /
snapshot / checkpoint machinery of ``experiments/cube_global/legacy/run_cs192_multi.py``
and the grid / IC / stepper build of ``experiments/cube_global/legacy/case_cs768.py``
(which drove
5 months of cube production).  The numerical semantics are ported verbatim; only the
plumbing is config-driven — ``RunConfig`` instead of argparse, torchrun (via
``init_distributed``) instead of ``mp.spawn``, ``case.extras`` instead of hard-coded
paths.  ``run_simulation`` (experiments/cases/_runtime.py) dispatches here when
``grid.kind == 'cubed_sphere'``.

Grid-device contract
--------------------
``build_grid_from_run_cfg``'s cube branch calls ``CubedSphereGrid.from_csN_grid``,
which takes NO ``device`` argument and builds every grid tensor on CPU at the run
dtype.  The runtime constructs the regular ``TorchRegionalOceanModel``,
whose ``_grid_to_runtime`` runs ``grid.to_runtime(device, dtype)`` eagerly.  The
historical "grid.to(device) must NOT be added" docstring guarded against a dtype
downgrade of an f64 master grid; a prerequisite check settled that this is
numerically LAZY (mechanism grep, code read, dtype probe and a cs32
full-step bitwise check) —
the production cube grid is already built at the run dtype, so ``to_runtime`` is a
pure device move (exact for float/bool tensors).  The stepper-side lazy migration
(``_prepare_cube_step_entry`` / ``ensure_grid_device_resident``) still runs and is
an idempotent no-op on an already device-resident grid.

IC forward-compat: the ``cube_ic_npz`` must carry ``temp`` / ``salt``
``[6,nz,n,n]``; ``u`` / ``v`` / ``eta`` are loaded when present, else zero-filled
(cold start, the case_cs768 baseline).  This leaves the slot for a full-state
IC without a schema change.

Checkpoints: the cube loop drives the regular ``CheckpointManager``
(``checkpoint:`` yaml block — schedule / keep_last / resume / resume_dir), the
same ``capture.export/import_step_state`` machinery the earlier hand-rolled per-rank
scheme used underneath; the ``cube_checkpoint_interval_steps`` /
``cube_resume_ckpt`` extras are retired (fail-loud, see ``_RETIRED_EXTRAS``).
Old fixed-name ``ckpt_r{rank}.pt`` files are NOT readable by the new stack (no
meta/fingerprint).  The process group is NOT destroyed here — its lifecycle
belongs to torchrun / stage2, matching the regional line
(experiments/cases/_runtime.py never calls ``destroy_process_group``).
"""
from __future__ import annotations

import difflib
import json
import logging
import math
import os
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from defgcm.step.contract import StepDiagnostics

from defgcm.io.runner._canonical_helpers import (
    implicit_vertical_advection_identity,
    implicit_vertical_advection_stepper_kwargs,
)

from experiments.cases._runtime import (
    _extra,
    _extra_bool,
    _required_extra,
    _resolve_device,
    _steps_from_config,
)

_REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

# physics.<field> <-> cube_cache_knobs['<knob>'] pairs that carry the SAME physical
# quantity through two config surfaces.  When both are set to conflicting values a
# run silently uses one and ignores the other — exactly the sponge-τ drift bug.
# The 4th column is the physics-side schema default (PhysicsConfig / TracerConfig in
# defgcm/io/runner/config/_schema.py): a physics value that differs from it is a
# deliberate setting, so the twin knob must be present too (absent-hole check below).
# The viscAh row (physics.viscosity <-> _cube_viscAh) and the
# viscAr row (physics.implicit_viscosity <-> _cube_viscAr) are gone: their
# normalized surfaces are single sources of truth (physics.viscosity feeds the
# injected horizontal_viscosity module; stepper.implicit_viscosity feeds both the
# implicit vertical viscosity and the bottom_drag module), so there is no second
# surface left to drift against.
# The last two rows (physics.tracer.diffusivity_kr <-> _cube_tracer_diffkr,
# physics.diffusivity <-> _cube_tracer_diffKh) are gone too: the cube step
# reads the stepper fields (tracer_vertical_diffusivity_kr / tracer_diffusivity_kh)
# which the runtime fills from physics.*, so those cache shadows are gone and there is
# no twin left to drift against.  The table is empty; the consistency check
# below is a no-op until a dual-entry pair reappears.
_D7_DUAL_ENTRY = ()


def _check_dual_entry_consistency(physics: Any, cache_knobs: dict[str, Any]) -> None:
    """Raise if a physics field and its cube_cache_knobs twin disagree.

    Two failure modes, both silent-drift traps the cube path used to swallow:

    * **mismatch** (both surfaces set): the knob is present, the physics value is
      non-default (non-zero) and the two differ beyond ``rel_tol=1e-12``.  A physics
      default of 0 means "unset here — the cube knob owns it", so it is not a conflict.
    * **absent hole** (physics set, knob missing): the physics value differs from its
      schema default but the twin knob is absent.  The cube step reads only its knob
      (falling back to the *stepper* default), so the deliberate physics value is
      silently ignored — the same class of bug in reverse.  Fixed by requiring the knob.

    Background tracer diffusion now preserves explicit declarations and binds
    through config.vertical_diffusion, independently of this retired knob table.
    """
    for phys_name, getter, knob_key, schema_default in _D7_DUAL_ENTRY:
        phys_val = getter(physics)
        if knob_key not in cache_knobs:
            # absent hole: knob missing but physics is a deliberate non-default value.
            if not math.isclose(phys_val, schema_default, rel_tol=1e-12, abs_tol=0.0):
                raise ValueError(
                    "cube dual-entry absent: "
                    f"{phys_name}={phys_val!r} is non-default (schema default "
                    f"{schema_default!r}) but its twin "
                    f"case.extras.cube_cache_knobs['{knob_key}'] is missing — the cube "
                    "step reads only the knob (falling back to the stepper default), so "
                    "the physics value would be silently ignored. Add "
                    f"cube_cache_knobs['{knob_key}']={phys_val!r} (sponge-τ lesson, "
                    "docs/NUMERICS.md)."
                )
            continue
        # both present: original mismatch semantics, unchanged.
        if phys_val == 0.0:
            continue
        knob_val = float(cache_knobs[knob_key])
        if not math.isclose(phys_val, knob_val, rel_tol=1e-12, abs_tol=0.0):
            raise ValueError(
                "cube dual-entry mismatch: "
                f"{phys_name}={phys_val!r} but "
                f"case.extras.cube_cache_knobs['{knob_key}']={knob_val!r} "
                "(rel_tol 1e-12). Set them equal or remove one — two configs for one "
                "physical quantity drift apart silently (sponge-τ lesson, docs/NUMERICS.md)."
            )


# ---------------------------------------------------------------------------
# cube_cache_knobs whitelist + unsupported-config-block rejection
# ---------------------------------------------------------------------------

# Config-intent cube knobs use external YAML names and are resolved into
# CubeStaticState. There are no
# momentum/viscosity/bottom_drag knobs (``_cube_viscAh``/``_cube_viscAr``/
# ``_cube_no_slip_sides``/``_cube_side_drag_factor``/``_cube_bottom_drag_linear``/
# ``_cube_bottom_drag_quadratic``/``_cube_bottom_drag_implicit``/``_cube_no_slip_bottom``):
# those 3 modules are built via the dynamics registry from the regular
# config surface (``build_cube_model``), so their parameters live in
# ``physics.*`` / ``stepper.kwargs`` / ``model.dynamics[<name>].kwargs``, not here.
# CubeStaticState owns module injections, corner Coriolis and KPP geometry.
# The stepper's cube_w_interface_readout is the host continuity diagnostic.
# CubeSurfaceForcing owns tau_x, tau_y, qnet, qsw, empmr, penetrating_sw,
# rho_const_fresh and bulk. set_cube_wind_stress and set_cube_surface_fluxes
# replace this current-step record. These fields are absent from cube_cache_knobs;
# rho_const_fresh is supplied through extras.cube_rho_const_fresh.
# There are no tracer diffusivity knobs (``_cube_tracer_diffKh``/
# ``_cube_tracer_diffkr``): the step reads the stepper fields
# (``tracer_diffusivity_kh`` / ``tracer_vertical_diffusivity_kr``) which the cube
# runtime fills from ``physics.diffusivity`` / ``physics.tracer.diffusivity_kr`` (single
# source of truth), so there is no second config surface left.
# The two gate switches (``_cube_enable_hpg`` /
# ``_cube_eos_dynamic_pressure``) are stepper dataclass fields
# (``cube_enable_hpg`` / ``cube_eos_dynamic_pressure`` via model.stepper.kwargs),
# leaving ``_cube_rho_const`` as the only whitelist entry — and that one is just
# the injection slot of the ``physics.reference_density`` source.
_CUBE_KNOB_WHITELIST = frozenset(
    {
        "_cube_rho_const",
    }
)

# Retired gate knobs must fail loud with the replacement spelling —
# silently accepting (or silently dropping) a retired key hides a misconfiguration.
_RETIRED_KNOBS = {
    "_cube_enable_hpg": "model.stepper.kwargs.cube_enable_hpg (default true)",
    "_cube_eos_dynamic_pressure": (
        "model.stepper.kwargs.cube_eos_dynamic_pressure (default false; true is "
        "refused by validate_dynamic_pressure_composition — cube JMD95P is not implemented)"
    ),
}


def _apply_reference_density_knob(run_cfg: Any, cache_knobs: dict[str, Any]) -> None:
    """``physics.reference_density`` feeds ``_cube_rho_const``.

    The runner resolves this external key into CubeStaticState.rho_const for
    the cube step, using the canonical ``physics.reference_density`` value.
    A yaml writing BOTH surfaces with different values raises (dual-entry
    discipline); the bare knob keeps working for yamls that predate the face.
    """
    phys = getattr(run_cfg.physics, "reference_density", None)
    if phys is None:
        return
    knob = cache_knobs.get("_cube_rho_const")
    if knob is not None and not math.isclose(
        float(phys), float(knob), rel_tol=1e-12, abs_tol=0.0
    ):
        raise ValueError(
            f"reference_density dual-entry mismatch: physics.reference_density="
            f"{float(phys)!r} but cube_cache_knobs['_cube_rho_const']={float(knob)!r}. "
            "physics.reference_density is the single source — "
            "remove the knob or set them equal."
        )
    cache_knobs["_cube_rho_const"] = float(phys)


def _check_cube_cache_knobs(cache_knobs: dict[str, Any]) -> None:
    """Raise on any ``cube_cache_knobs`` key outside the whitelist.

    Reject unknown keys with a nearest-match hint.
    """
    for key in cache_knobs:
        if key in _CUBE_KNOB_WHITELIST:
            continue
        if key in _RETIRED_KNOBS:
            raise ValueError(
                f"cube_cache_knobs key {key!r} was retired: set "
                f"{_RETIRED_KNOBS[key]} instead."
            )
        suggestion = difflib.get_close_matches(key, _CUBE_KNOB_WHITELIST, n=1)
        hint = f" — did you mean {suggestion[0]!r}?" if suggestion else ""
        raise ValueError(
            f"unknown cube_cache_knobs key {key!r}{hint}. Allowed config-intent "
            f"knobs: {sorted(_CUBE_KNOB_WHITELIST)}. Runtime-internal cube cache "
            "keys must not be set from yaml."
        )


# Extras keys retired by the checkpoint unification.  The cube path honors
# the regular ``checkpoint:`` yaml block (CheckpointManager); the old shadow keys
# must fail loud with the replacement spelling, because silently ignoring a retired key
# hides a misconfiguration.
_RETIRED_EXTRAS = {
    "cube_checkpoint_interval_steps": (
        "checkpoint: {enabled: true, schedule: [{type: interval, every_steps: N, "
        "include_first: false}], keep_last: 1}"
    ),
    "cube_resume_ckpt": (
        "checkpoint.resume: latest (+ checkpoint.resume_dir: <other run dir> to "
        "resume from another run)"
    ),
}


def _check_retired_extras(run_cfg: Any) -> None:
    """Raise on extras keys retired by the checkpoint.* unification."""
    extras = dict(getattr(run_cfg.case, "extras", {}) or {})
    for key, replacement in _RETIRED_EXTRAS.items():
        if key in extras:
            raise ValueError(
                f"case.extras.{key} was retired: the cube "
                f"path now honors the regular checkpoint config. Use {replacement}"
            )


# The cube-capable dynamics modules the cube path assembles via the registry
# from the regular config surface (see build_cube_model).  A cube yaml MAY declare these
# in model.dynamics; anything else is still dropped-on-the-floor and raises.
# Derived from the grid_support capability matrix (each class's declaration,
# defgcm/grid_support.py) instead of a hard-coded name set — adding a cube
# leaf extends this whitelist by declaring ``grid_support`` on the class.
def _cube_capable_dynamics() -> frozenset[str]:
    from defgcm.grid_support import cube_dynamics_names

    return cube_dynamics_names()


def _reject_unsupported_config_blocks(run_cfg: Any) -> None:
    """Fail loud on yaml blocks the cube path silently ignores.

    ``run_simulation`` dispatches cube cases straight to ``run_cube_simulation``,
    bypassing the regular config pipeline, so ``model.boundary_condition`` /
    ``physics.nonhydrostatic`` and a non-default ``boundary`` block are dropped
    on the floor.  Until they are wired in, writing them is a configuration
    error, not a no-op.  ``checkpoint.*`` is honored (CheckpointManager
    runs inside the cube loop), so it no longer raises here.

    Exception: ``model.dynamics`` may declare cube-capable modules
    (``grid_support`` capability matrix, ``_cube_capable_dynamics``) — the cube
    runtime assembles them via the registry (``build_cube_model``).  Any other
    dynamics name still raises.
    """
    from defgcm.io.runner.config._schema import BoundaryConfig

    model = run_cfg.model
    if model.dynamics:
        cube_capable = _cube_capable_dynamics()
        bad = [e.name for e in model.dynamics if e.name not in cube_capable]
        if bad:
            raise ValueError(
                "model.dynamics on the cube path only supports "
                f"{sorted(cube_capable)}; got unsupported "
                f"{bad}. Other dynamics modules are hard-coded inside the cube stepper "
                "until the manifest is wired in."
            )
    if model.boundary_condition.name is not None:
        raise ValueError(
            "model.boundary_condition is not supported on the cube path (got "
            f"{model.boundary_condition.name!r}): the cube is a closed global domain "
            "with no open-boundary system."
        )
    if model.stepper.name not in (None, "implicit_free_surface"):
        raise ValueError(
            f"model.stepper.name={model.stepper.name!r}: the cube path always uses "
            "implicit_free_surface — the cube runtime constructs "
            "ImplicitFreeSurfaceStepper unconditionally, so any other stepper name is "
            "a silent misconfiguration."
        )
    if run_cfg.physics.nonhydrostatic.enabled:
        raise ValueError(
            "nonhydrostatic is not supported on the cube path: the cube stepper is "
            "hydrostatic-only (cube+NH raises rather than silently running "
            "hydrostatic)."
        )
    if run_cfg.boundary != BoundaryConfig():
        raise ValueError(
            "a non-default boundary block is not supported on the cube path: the cube "
            "is a closed global cube with no OBCS. A default BoundaryConfig equals "
            "'yaml wrote no boundary block', so only an explicit non-default boundary "
            "block trips this."
        )


# ---------------------------------------------------------------------------
# IC assembly (port of case_cs768.build:156-186; decoupled from the real grid so
# it is unit-testable with synthetic hFacC / layer_thickness tensors)
# ---------------------------------------------------------------------------


def _field_or_zero(
    arr_np: np.ndarray | None,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    """Forward-compat: use the loaded field when present (shape-checked), else zero."""
    if arr_np is None:
        return torch.zeros(*shape, dtype=dtype)
    t = torch.from_numpy(np.ascontiguousarray(arr_np)).to(dtype)
    if tuple(t.shape) != tuple(shape):
        raise ValueError(
            f"cube IC field {name!r} shape {tuple(t.shape)} != expected {tuple(shape)}"
        )
    return t


def _load_cube_ic(
    npz_path: str | Path,
    *,
    n: int,
    nz: int,
    hfac_c: torch.Tensor,
    layer_thickness: torch.Tensor,
    land_temp: float,
    land_salt: float,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Any:
    """Assemble the GLOBAL cube ``PrognosticState`` from an IC npz (case_cs768 parity).

    ``temp`` / ``salt`` ``[6,nz,n,n]`` are required.  ``u`` / ``v`` / ``eta`` load
    when present, else zero-fill.  Land cells (``hFacC == 0``) are filled with
    ``land_temp`` / ``land_salt`` then wet-masked (``* cm``).  When the npz carries a
    ``delR40`` key it must match ``layer_thickness`` (fail-loud on a vertical-grid
    mismatch — a config error, not a silent regrid).

    ``layer_thickness`` is the grid's built vertical-thickness tensor
    (``CubedSphereGrid.layer_thickness``); the grid exposes no
    ``vertical_layer_thicknesses`` attribute, so that is the delR40 comparison target.
    """
    from defgcm.fields import Field, FieldLocation
    from defgcm.state import PrognosticState

    with np.load(str(npz_path)) as z:
        files = set(z.files)
        t0 = torch.from_numpy(np.ascontiguousarray(z["temp"])).to(dtype)
        s0 = torch.from_numpy(np.ascontiguousarray(z["salt"])).to(dtype)
        delR = np.asarray(z["delR40"], dtype=np.float64) if "delR40" in files else None
        eta_np = np.asarray(z["eta"]) if "eta" in files else None
        u_np = np.asarray(z["u"]) if "u" in files else None
        v_np = np.asarray(z["v"]) if "v" in files else None

    if delR is not None:
        want = layer_thickness.detach().cpu().numpy().astype(np.float64)
        if delR.shape != want.shape or not np.allclose(delR, want):
            raise ValueError(
                "cube IC delR40 does not match grid layer_thickness "
                f"(npz={delR.tolist()} vs grid={want.tolist()}); vertical-grid "
                "mismatch is a configuration error"
            )

    cm = (hfac_c.to(dtype) > 0).to(dtype)
    t0 = torch.where(cm > 0, t0, torch.full_like(t0, float(land_temp))) * cm
    s0 = torch.where(cm > 0, s0, torch.full_like(s0, float(land_salt))) * cm

    eta0 = _field_or_zero(eta_np, (6, n, n), dtype, "eta")
    u0 = _field_or_zero(u_np, (6, nz, n, n + 1), dtype, "u")
    v0 = _field_or_zero(v_np, (6, nz, n + 1, n), dtype, "v")

    def mk(x: torch.Tensor) -> torch.Tensor:
        return x.to(device)

    return PrognosticState(
        eta=Field("eta", mk(eta0), FieldLocation.CENTER, "m"),
        u=Field("u", mk(u0), FieldLocation.U_FACE, "m s-1"),
        v=Field("v", mk(v0), FieldLocation.V_FACE, "m s-1"),
        temp=Field("temp", mk(t0.clone()), FieldLocation.CENTER, "degC"),
        salt=Field("salt", mk(s0.clone()), FieldLocation.CENTER, "psu"),
    )


def _scatter_state(state: Any, cube_ctx: Any) -> Any:
    """Slice a GLOBAL [6,...] PrognosticState to this rank's owned faces (run_cs192_multi:519-525)."""
    from defgcm.fields import Field, FieldLocation
    from defgcm.geometry.cube_decomposition import scatter_faces
    from defgcm.state import PrognosticState

    def sc(t: torch.Tensor) -> torch.Tensor:
        return scatter_faces(t, cube_ctx)

    return PrognosticState(
        eta=Field("eta", sc(state.eta.data), FieldLocation.CENTER, "m"),
        u=Field("u", sc(state.u.data), FieldLocation.U_FACE, "m s-1"),
        v=Field("v", sc(state.v.data), FieldLocation.V_FACE, "m s-1"),
        temp=Field("temp", sc(state.temp.data), FieldLocation.CENTER, "degC"),
        salt=Field("salt", sc(state.salt.data), FieldLocation.CENTER, "psu"),
    )


# ---------------------------------------------------------------------------
# Model assembly (the regular TorchRegionalOceanModel)
# ---------------------------------------------------------------------------


def build_cube_model(
    run_cfg: Any,
    grid: Any,
    stepper: Any,
    *,
    device: Any,
    dtype: torch.dtype,
    cube_ctx: Any = None,
) -> Any:
    """Assemble the regular ``TorchRegionalOceanModel`` for the cube path.

    The 3 dynamics modules (momentum / viscosity / bottom_drag) are built
    ONCE via the dynamics registry from the regular config surface and used twice:

      (a) passed to the model constructor, so ``model.dynamics_modules`` /
          ``model.capabilities`` are real for the cube (the
          capability-matrix data source);
      (b) injected into ``stepper.cube_static_state`` as momentum, viscosity
          and bottom_drag, which the cube step body consumes.

    Single sources of truth:
      * horizontal_viscosity.viscosity  <- ``physics.viscosity``
      * bottom_drag.vertical_viscosity  <- ``stepper.implicit_viscosity``
    Structural parameters come from ``model.dynamics[<name>].kwargs``.

    The model constructor runs ``grid.to_runtime(device, dtype)``: numerically
    lazy (the production cube grid is already built
    at the run dtype, so this is a pure device move; exact for float/bool) and
    ``ensure_grid_device_resident`` in the step then no-ops.  Boundary
    condition is ``Passthrough`` (the cube is a closed global domain; the
    Structured SEAM policy would raise) and the diagnostics suite is
    explicitly EMPTY — cube conservation goes through
    ``CubeConservationMonitor``, not ``model.compute_diagnostics`` (the
    regional default suite mis-broadcasts on ``[6, ...]``).

    ``RegionalOceanConfig`` carries only the fields the cube path consumes
    (grid dims / dt / dtype / device / gravity / depth); the linear-EOS
    parameters and ``reference_density`` stay at schema defaults — the cube
    reads rhoConst from CubeStaticState.rho_const, populated by
    ``_apply_reference_density_knob``.
    """
    from defgcm.boundaries import PassthroughBoundaryCondition
    from defgcm.diagnostics import DiagnosticSuite
    from defgcm.io.runner.builders.from_manifest import _instantiate_dynamics
    from defgcm.io.runner.config._schema import DynamicsEntry
    from defgcm.model import RegionalOceanConfig, TorchRegionalOceanModel

    phys = run_cfg.physics
    dyn = {e.name: dict(e.kwargs) for e in run_cfg.model.dynamics}

    # momentum: all structural params from yaml kwargs (defaults == old cube
    # literals); no single-source-of-truth override.
    mom_kw = dict(dyn.get("vector_invariant_momentum_advection", {}))
    if "rotation_period" in mom_kw:
        raise ValueError(
            "model.dynamics[vector_invariant_momentum_advection].kwargs.rotation_period: "
            "set physics.rotation_period, the one rotation period of the run"
        )
    mom_kw["rotation_period"] = float(phys.rotation_period)
    momentum = _instantiate_dynamics(
        DynamicsEntry(name="vector_invariant_momentum_advection", kwargs=mom_kw)
    )

    # viscosity: viscosity <- physics.viscosity (single source); scheme /
    # no_slip_sides / side_drag_factor from yaml kwargs.
    visc_kw = dict(dyn.get("horizontal_viscosity", {}))
    visc_kw["viscosity"] = float(phys.viscosity)
    viscosity = _instantiate_dynamics(
        DynamicsEntry(name="horizontal_viscosity", kwargs=visc_kw)
    )

    # bottom_drag: vertical_viscosity <- stepper.implicit_viscosity (single
    # source); linear / quadratic / implicit / no_slip_bottom from yaml
    # kwargs; dt is the momentum step dt, read only when implicit.
    bd_kw = dict(dyn.get("bottom_drag", {}))
    bd_kw["vertical_viscosity"] = float(stepper.implicit_viscosity)
    bd_kw.setdefault("dt", float(run_cfg.time.dt))
    bottom_drag = _instantiate_dynamics(
        DynamicsEntry(name="bottom_drag", kwargs=bd_kw)
    )

    object.__setattr__(
        stepper, "cube_static_state",
        replace(
            stepper.cube_static_state,
            momentum=momentum,
            viscosity=viscosity,
            bottom_drag=bottom_drag,
        ),
    )

    config = RegionalOceanConfig(
        nx=int(grid.nx),
        ny=int(grid.ny),
        nz=int(grid.nz),
        depth=float(grid.layer_thickness.sum()),
        gravity=float(stepper.gravity),
        default_dt=float(run_cfg.time.dt),
        dtype=dtype,
        device=str(device),
    )
    return TorchRegionalOceanModel(
        config,
        grid=grid,
        dynamics_modules=[momentum, viscosity, bottom_drag],
        stepper=stepper,
        boundary_condition=PassthroughBoundaryCondition(),
        diagnostics=DiagnosticSuite(modules=[]),
        cube_ctx=cube_ctx,
    )


# ---------------------------------------------------------------------------
# Snapshot helpers (verbatim port of run_cs192_multi.py).  Exact checkpoints
# go through the regular CheckpointManager (checkpoint.* yaml block), same
# capture.export/import_step_state underneath.
# ---------------------------------------------------------------------------


def _load_state_from_dump(path, device, dtype):
    """Global PrognosticState + start step from a dump npz (forecast or full)."""
    from defgcm.fields import Field, FieldLocation
    from defgcm.state import PrognosticState

    z = np.load(path)

    def fld(name, loc, unit):
        return Field(
            name,
            torch.from_numpy(np.ascontiguousarray(z[name])).to(device=device, dtype=dtype),
            loc,
            unit,
        )

    state = PrognosticState(
        eta=fld("eta", FieldLocation.CENTER, "m"),
        u=fld("u", FieldLocation.U_FACE, "m s-1"),
        v=fld("v", FieldLocation.V_FACE, "m s-1"),
        temp=fld("temp", FieldLocation.CENTER, "degC"),
        salt=fld("salt", FieldLocation.CENTER, "psu"),
    )
    return state, int(z["step"])


def _snapshot(out_dir: Path, step: int, state, stepper, grid, nz: int, diagnostics, forecast: bool = False) -> dict:
    if forecast:
        # Dashboard-bound forecast frame: f32 prognostics only (no AB2/etaH/w).
        def f32(t):
            return t.detach().cpu().to(torch.float32).numpy()

        payload = {
            "eta": f32(state.eta.data),
            "temp": f32(state.temp.data),
            "salt": f32(state.salt.data),
            "u": f32(state.u.data),
            "v": f32(state.v.data),
            "step": np.array(step),
        }
        np.savez(out_dir / f"step_{step:05d}.npz", **payload)
        return payload

    ab = stepper._momentum_ab
    tracer_ab = stepper._tracer_ab
    etaH = stepper._continuity.eta_h
    # Fall back to the legacy grid
    # attribute for backward compat with any run that did not go through step_cube.
    w = stepper.cube_w_interface_readout
    if w is None:
        w = getattr(grid, "_cached_w_interface", None)

    def cached(name: str) -> np.ndarray:
        if ab is not None and getattr(ab, name) is not None:
            return getattr(ab, name).detach().cpu().numpy()
        if (tracer_ab is not None
                and getattr(tracer_ab, name) is not None):
            return getattr(tracer_ab, name).detach().cpu().numpy()
        return np.zeros(0)

    payload = {
        "eta": state.eta.data.detach().cpu().numpy(),
        "u": state.u.data.detach().cpu().numpy(),
        "v": state.v.data.detach().cpu().numpy(),
        "temp": state.temp.data.detach().cpu().numpy(),
        "salt": state.salt.data.detach().cpu().numpy(),
        "ab2_u": cached("u"),
        "ab2_v": cached("v"),
        "ab2_temp": cached("temp"),
        "ab2_salt": cached("salt"),
        "etaH": (etaH.detach().cpu().numpy() if etaH is not None else np.zeros(0)),
        "w_interface": (w[:, :nz].detach().cpu().numpy() if w is not None else np.zeros(0)),
        "cg2d_iters": np.array((-1 if diagnostics.cg2d_iters is None else int(diagnostics.cg2d_iters))),
        "cg2d_residual": np.array((float("nan") if diagnostics.cg2d_residual is None else float(diagnostics.cg2d_residual))),
        "step": np.array(step),
    }
    np.savez(out_dir / f"step_{step:05d}.npz", **payload)
    return payload


def _gather_snapshot(out_dir: Path, step: int, local_state, stepper, grid, nz: int, ctx, diagnostics, forecast: bool = False) -> None:
    """Gather local state + caches to global face order and write npz (rank 0 only)."""
    from defgcm.fields import Field, FieldLocation
    from defgcm.geometry.cube_decomposition import gather_faces
    from defgcm.state import PrognosticState

    eta_g = gather_faces(local_state.eta.data, ctx)
    u_g = gather_faces(local_state.u.data, ctx)
    v_g = gather_faces(local_state.v.data, ctx)
    temp_g = gather_faces(local_state.temp.data, ctx)
    salt_g = gather_faces(local_state.salt.data, ctx)

    if forecast:
        if ctx.rank != 0:
            return
        global_state = PrognosticState(
            eta=Field("eta", eta_g, FieldLocation.CENTER, "m"),
            u=Field("u", u_g, FieldLocation.U_FACE, "m s-1"),
            v=Field("v", v_g, FieldLocation.V_FACE, "m s-1"),
            temp=Field("temp", temp_g, FieldLocation.CENTER, "degC"),
            salt=Field("salt", salt_g, FieldLocation.CENTER, "psu"),
        )
        _snapshot(out_dir, step, global_state, stepper, grid, nz, diagnostics, forecast=True)
        return

    ab = stepper._momentum_ab
    etaH_local = stepper._continuity.eta_h
    w_local = stepper.cube_w_interface_readout

    etaH_g = gather_faces(etaH_local, ctx) if etaH_local is not None else None
    w_g = gather_faces(w_local, ctx) if w_local is not None else None
    ab_u_g = gather_faces(ab.u, ctx) if ab is not None else None
    ab_v_g = gather_faces(ab.v, ctx) if ab is not None else None

    if ctx.rank != 0:
        return  # only rank 0 writes

    global_state = PrognosticState(
        eta=Field("eta", eta_g, FieldLocation.CENTER, "m"),
        u=Field("u", u_g, FieldLocation.U_FACE, "m s-1"),
        v=Field("v", v_g, FieldLocation.V_FACE, "m s-1"),
        temp=Field("temp", temp_g, FieldLocation.CENTER, "degC"),
        salt=Field("salt", salt_g, FieldLocation.CENTER, "psu"),
    )

    original_continuity = stepper._continuity
    orig_w = stepper.cube_w_interface_readout
    original_momentum_ab = stepper._momentum_ab

    if etaH_g is not None:
        object.__setattr__(stepper, "_continuity", stepper._continuity.with_eta_h(etaH_g))
    if w_g is not None:
        object.__setattr__(
            stepper, "cube_w_interface_readout", w_g,
        )
    if ab_u_g is not None and ab_v_g is not None:
        object.__setattr__(stepper, "_momentum_ab", stepper._momentum_ab.with_velocities(ab_u_g, ab_v_g))

    try:
        _snapshot(out_dir, step, global_state, stepper, grid, nz, diagnostics)
    finally:
        object.__setattr__(stepper, "_continuity", original_continuity)
        if w_g is not None:
            object.__setattr__(stepper, "cube_w_interface_readout", orig_w)
        object.__setattr__(stepper, "_momentum_ab", original_momentum_ab)


def _initialize_fresh_cube_step_state(model: Any, state: Any, resume_step: int) -> Any:
    """Initialize a cube loop seed only at the fresh step-zero boundary."""
    if type(resume_step) is not int:
        raise TypeError("resume_step must be an exact int")
    if resume_step != 0:
        return state
    from defgcm.runner import initialize_fresh_step_state

    return initialize_fresh_step_state(model, state)


# ---------------------------------------------------------------------------
# Run-level helpers
# ---------------------------------------------------------------------------


def _peak_mem_str(device) -> str:
    """Peak per-GPU memory (capacity probe).  Empty on CPU."""
    if getattr(device, "type", None) == "cuda":
        a = torch.cuda.max_memory_allocated() / 2 ** 30
        r = torch.cuda.max_memory_reserved() / 2 ** 30
        return f" peakGiB_alloc={a:.2f} peakGiB_resv={r:.2f}"
    return ""


def _resolve_dump_steps(run_cfg: Any, n_steps: int, dt: float) -> set[int]:
    """Dump-step set from output.snapshot cadence ∪ {0, n_steps}."""
    snap_s = float(run_cfg.output.resolved_snapshot_seconds())
    interval = max(1, int(round(snap_s / dt)))
    steps = set(range(0, n_steps + 1, interval))
    steps.add(0)
    steps.add(n_steps)
    return steps


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO).decode().strip()
    except Exception:
        return "unknown"


def _write_run_meta(
    output_dir: Path,
    run_cfg: Any,
    *,
    commit: str,
    world_size: int,
    dt: float,
    n_steps: int,
    dump_steps: set[int],
    cache_knobs: dict[str, Any],
    stepper_kwargs: dict[str, Any],
    precision: str,
) -> None:
    meta = {
        "case": str(run_cfg.run.experiment_name),
        "case_type": str(_extra(run_cfg, "case_type", "")),
        "commit": commit,
        "ranks": int(world_size),
        "grid": {
            "kind": str(run_cfg.grid.kind),
            "n": int(run_cfg.grid.cube_face_n),
            "nz": int(run_cfg.grid.nz),
        },
        "dt": float(dt),
        "steps": int(n_steps),
        "precision": str(precision),
        "physics": {
            "implicit_vertical_advection": (
                implicit_vertical_advection_identity(run_cfg)
            ),
        },
        "cache_knobs": {k: cache_knobs[k] for k in sorted(cache_knobs)},
        "stepper_kwargs": {k: stepper_kwargs[k] for k in sorted(stepper_kwargs)},
        "dump_steps": sorted(dump_steps),
    }
    (output_dir / "cube_run_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n"
    )


def _apply_cube_precond_env(run_cfg: Any) -> None:
    """Set DEFGCM_CUBE_CG2D_PRECOND (default ssor) unless already set (case_cs768.py:68).

    ssor is required for cs384+ corner-cell cg2d convergence (Jacobi can stall).
    ``setdefault`` lets an explicit environment override win; ``cube_cg2d_precond``
    in case.extras overrides the ssor default when the env is unset.
    """
    os.environ.setdefault(
        "DEFGCM_CUBE_CG2D_PRECOND", str(_extra(run_cfg, "cube_cg2d_precond", "ssor"))
    )


def _build_forcing(run_cfg: Any, *, device, dtype):
    """Optional per-step ERA5 bulk forcing.

    Streaming default: ``extras.cube_forcing_zarr`` (or a ``cube_forcing_npz`` path
    ending in ``.zarr``) is read lazily via ``CubeBulkForcing.from_zarr`` (~1 GB/rank
    at cs1024).  Back-compat: a plain ``.npz`` still loads fully via ``from_npz``.
    """
    zarr_path = _extra(run_cfg, "cube_forcing_zarr", None)
    npz = _extra(run_cfg, "cube_forcing_npz", None)
    if zarr_path is None and npz is None:
        return None
    from experiments.cube_global.cube_forcing import CubeBulkForcing

    rho_const_fresh = float(_extra(run_cfg, "cube_rho_const_fresh", 1035.0))
    if zarr_path is not None or (npz is not None and str(npz).endswith(".zarr")):
        path = zarr_path if zarr_path is not None else npz
        return CubeBulkForcing.from_zarr(
            str(path), device=device, dtype=dtype, rho_const_fresh=rho_const_fresh
        )
    return CubeBulkForcing.from_npz(
        str(npz), device=device, dtype=dtype, rho_const_fresh=rho_const_fresh
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def cube_stepper_kwargs(run_cfg: Any) -> dict[str, Any]:
    """The production and casegen cube constructors consume identical arguments."""
    from defgcm.io.runner.config.vertical_diffusion import assemble_vertical_diffusion

    kwargs = dict(assemble_vertical_diffusion(run_cfg).stepper.kwargs)
    kwargs.update(implicit_vertical_advection_stepper_kwargs(run_cfg))
    kwargs.setdefault("tracer_diffusivity_kh", float(run_cfg.physics.diffusivity))
    if "rotation_period" in kwargs:
        raise ValueError(
            "model.stepper.kwargs.rotation_period: set physics.rotation_period, "
            "the one rotation period of the run"
        )
    kwargs["rotation_period"] = float(run_cfg.physics.rotation_period)
    return kwargs


def run_cube_simulation(run_cfg: Any, manifest_run_dir: Path, logger: logging.Logger) -> int:
    """Run a cube-global yaml simulation. Returns 0 (BOUNDED), 1 (blow-up), 5 (error).

    Single- and multi-rank unified: ``init_distributed()`` returns ``None`` for the
    single-process path (rank 0, world_size 1, no cube_ctx) and a
    ``DistributedContext`` under torchrun (world_size ∈ {2,3,6}); the per-rank
    ``CubeRankContext`` routes the in-step face exchanges.
    """
    # -- fail loud on yaml blocks the cube path silently ignores
    # and on extras keys retired by the checkpoint unification.
    _reject_unsupported_config_blocks(run_cfg)
    _check_retired_extras(run_cfg)

    # -- precond env (case_cs768.py:68 — ssor cg2d preconditioner is required for
    # cs384+ corner-cell convergence; setdefault lets an explicit env override win).
    _apply_cube_precond_env(run_cfg)

    # -- compute precision (mirror experiments/cases/_runtime.py:1310-1318).
    precision = getattr(run_cfg.run, "precision", "float64")
    if precision == "float64":
        dtype = torch.float64
    elif precision == "float32":
        dtype = torch.float32
    else:
        raise ValueError(
            f"unsupported run.precision {precision!r}; expected 'float64' or 'float32'"
        )
    # Bare-scalar safety net (run_cs192_multi keeps set_default_dtype at float64
    # regardless of the state/grid precision).
    torch.set_default_dtype(torch.float64)

    n_steps = _steps_from_config(run_cfg)
    dt = float(run_cfg.time.dt)

    # -- distributed (torchrun env -> DistributedContext; None = single rank).
    from defgcm.distributed import init_distributed

    dctx = init_distributed()
    if dctx is None:
        rank, world_size, cube_ctx = 0, 1, None
        device = torch.device(_resolve_device(run_cfg))
    else:
        from defgcm.geometry.cube_decomposition import (
            SUPPORTED_RANKS,
            CubeDecomposition,
            CubeRankContext,
        )

        world_size = dctx.world_size
        if world_size not in SUPPORTED_RANKS:
            raise ValueError(
                f"cube world_size={world_size} not in supported {SUPPORTED_RANKS} "
                "(face-aligned only)"
            )
        decomp = CubeDecomposition.for_ranks(world_size, n=int(run_cfg.grid.nx))
        cube_ctx = CubeRankContext.build(decomp, dctx.rank)
        rank = dctx.rank
        device = dctx.device
    is_rank0 = rank == 0

    output_dir = Path(manifest_run_dir)

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print("=" * 98)
        print(f"LOCOMgcm cube-global run: {run_cfg.run.experiment_name}")
        if dctx is not None:
            print(
                f"Distributed cube: rank={rank}/{world_size} "
                f"local_faces={cube_ctx.local_face_ids} backend={dctx.backend} device={device}",
                flush=True,
            )
        print("=" * 98)

        # -- grid (CPU-built; migrates to device inside the cube stepper — see module
        # docstring).  bathymetry=None: the cube branch reads cube_bathy_path itself.
        from defgcm.io.runner.builders import build_grid_from_run_cfg

        grid = build_grid_from_run_cfg(
            run_cfg,
            bathymetry=None,
            device=device,
            dtype=dtype,
            hfac_min=float(_extra(run_cfg, "hfac_min", 0.2)),
            hfac_min_dz=float(_extra(run_cfg, "hfac_min_dz", 0.0)),
        )
        nz = grid.nz

        # -- IC (GLOBAL [6,...]); optional lossy dump-resume replaces it BEFORE scatter.
        state = _load_cube_ic(
            _required_extra(run_cfg, "cube_ic_npz"),
            n=int(grid.nx),
            nz=nz,
            hfac_c=grid.masks.hFacC,
            layer_thickness=grid.layer_thickness,
            land_temp=float(_extra(run_cfg, "cube_land_temp", 5.0)),
            land_salt=float(_extra(run_cfg, "cube_land_salt", 35.0)),
            device=device,
            dtype=dtype,
        )
        from defgcm.io.runner.builders.from_manifest import apply_initial_theta_freezing

        state = state.with_tensors(
            temp=apply_initial_theta_freezing(state.temp.data, run_cfg=run_cfg),
        )
        resume_step = 0
        resume_dump = _extra(run_cfg, "cube_resume", None)
        if resume_dump is not None:
            state, resume_step = _load_state_from_dump(str(resume_dump), device, dtype)
            if is_rank0:
                print(f"  RESUME (lossy) from {resume_dump} at step {resume_step}", flush=True)

        # -- stepper + cube cache knobs + dual-entry gate.
        from defgcm.steppers.base import TimeStepBundle
        from defgcm.steppers.implicit_free_surface import ImplicitFreeSurfaceStepper

        stepper_kwargs = cube_stepper_kwargs(run_cfg)
        stepper = ImplicitFreeSurfaceStepper(**stepper_kwargs)
        cache_knobs = dict(_extra(run_cfg, "cube_cache_knobs", {}) or {})
        _check_cube_cache_knobs(cache_knobs)
        _check_dual_entry_consistency(run_cfg.physics, cache_knobs)
        _apply_reference_density_knob(run_cfg, cache_knobs)
        if "_cube_rho_const" in cache_knobs:
            object.__setattr__(
                stepper, "cube_static_state",
                replace(
                    stepper.cube_static_state,
                    rho_const=cache_knobs["_cube_rho_const"],
                ),
            )

        # -- assemble the regular TorchRegionalOceanModel (registry-built
        # dynamics modules into BOTH model.dynamics_modules and CubeStaticState).
        # After reference-density resolution and before checkpoint resume (which restores the
        # cross-step cache).  The constructor's grid.to_runtime is numerically lazy
        # — rebind ``grid`` to the runtime copy so every later
        # consumer (snapshot fallback, monitor) sees the grid the step actually
        # uses.
        model = build_cube_model(
            run_cfg, grid, stepper, device=device, dtype=dtype, cube_ctx=cube_ctx
        )
        grid = model.grid
        state.validate_against(grid)  # global [6, ...] contract check before scatter

        # -- scatter to rank-local faces (no-op for single rank).
        if cube_ctx is not None:
            state = _scatter_state(state, cube_ctx)

        # -- EXACT checkpoint resume via the regular CheckpointManager (the
        # cube_resume_ckpt extras path is retired).  Per-rank (rank=... routes each
        # process to checkpoints/rankXXXXX/), so it happens AFTER scatter; an exact
        # resume overrides a lossy cube_resume dump seed, matching the old priority.
        from defgcm.checkpoint import CheckpointManager

        ckpt_rank = rank if dctx is not None else None
        ckpt_mgr = CheckpointManager.from_run_config(run_cfg, output_dir, rank=ckpt_rank)
        if ckpt_mgr is not None:
            plan = ckpt_mgr.resolve_resume(model, n_steps)
            if plan is not None:
                state = ckpt_mgr.apply_resume(model, plan)
                resume_step = plan.start_step
                if is_rank0:
                    print(
                        f"  RESUME-CKPT (exact) from {plan.source_path} at step {resume_step}",
                        flush=True,
                    )

        forcing = _build_forcing(run_cfg, device=device, dtype=dtype)
        bundle = TimeStepBundle(dt=dt, dt_mom=dt, dt_tracer=dt, dt_freesurf=dt, dt_clock=dt)

        dump_steps = _resolve_dump_steps(run_cfg, n_steps, dt)
        monitor_interval = max(1, int(run_cfg.output.monitor_interval_steps))
        forecast = _extra_bool(run_cfg, "cube_forecast_dump", False)

        # -- cube-aware conservation watch (the regional DiagnosticSuite
        # mis-broadcasts on [6,...], so the cube loop gets its own observer).
        # Multi-rank: the state is rank-local [n_local,...], so the monitor gets
        # the rank-local grid (memoized localize_cube_grid — the same object the
        # step body uses) and the packed all_reduce restores global sums/extrema.
        from defgcm.diagnostics import CubeConservationMonitor

        if cube_ctx is None:
            conserv_grid = grid
        else:
            from defgcm.geometry.cube_localize import localize_cube_grid

            conserv_grid = localize_cube_grid(grid, cube_ctx)
        conserv = CubeConservationMonitor(
            conserv_grid,
            gravity=float(stepper.gravity),
            out_path=(output_dir / "cube_conservation.jsonl") if is_rank0 else None,
            is_rank0=is_rank0,
            interval=monitor_interval,
        )

        snap_dir = output_dir / "cube_snapshots"
        if is_rank0:
            snap_dir.mkdir(parents=True, exist_ok=True)

        commit = _git_commit()
        if is_rank0:
            _write_run_meta(
                output_dir, run_cfg,
                commit=commit, world_size=world_size, dt=dt, n_steps=n_steps,
                dump_steps=dump_steps, cache_knobs=cache_knobs,
                stepper_kwargs=stepper_kwargs, precision=precision,
            )
            print(
                f"run_cube {run_cfg.run.experiment_name} device={device} steps={n_steps} "
                f"ranks={world_size} dt={dt} nz={nz} dump={sorted(dump_steps)} "
                f"commit={commit[:8]}",
                flush=True,
            )

        # -- step loop (verbatim port of run_cs192_multi.py:577-608, config-driven
        # cadences).  cube_ctx=None routes to _snapshot; else _gather_snapshot.
        diagnostics = StepDiagnostics.absent()
        def _do_snapshot(s: int, st) -> None:
            if cube_ctx is None:
                _snapshot(snap_dir, s, st, stepper, grid, nz, diagnostics, forecast=forecast)
            else:
                _gather_snapshot(snap_dir, s, st, stepper, grid, nz, cube_ctx, diagnostics, forecast=forecast)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        blew = False
        step = resume_step

        # The cube driver owns its loop and therefore does not pass through
        # ModelRunner's fresh-state initializer.  Initialize after scatter and
        # exact resume resolution, but before a possible step-0 snapshot.  The
        # cube implicit-advection capability gate remains authoritative; this
        # call only establishes the common IFS step clock while all flags are
        # off, and would build a wrapper only after a future complete cube path
        # is admitted by the validator.
        state = _initialize_fresh_cube_step_state(model, state, resume_step)

        if resume_step == 0 and 0 in dump_steps:
            _do_snapshot(0, state)

        try:
            for k in range(resume_step, n_steps):
                if forcing is not None:
                    forcing.apply(stepper, state, k * dt, cube_ctx)
                outcome = model.step_outcome(state, bundle, time_seconds=k * dt)
                state = outcome.state
                diagnostics = outcome.diagnostics
                step = k + 1
                em = float(state.eta.data.abs().max())
                if not np.isfinite(em):
                    if is_rank0:
                        print(f"  step {step}: NaN/Inf -> BLOW UP", flush=True)
                    blew = True
                    break
                # All ranks call (may issue collectives on interval steps).
                conserv.maybe_record(step, state)
                if step in dump_steps:
                    _do_snapshot(step, state)
                if ckpt_mgr is not None:
                    ckpt_mgr.maybe_capture(
                        model, state,
                        global_step=step, model_time=step * dt, dt=dt, n_steps=n_steps,
                    )
                if is_rank0 and (step <= 3 or step % monitor_interval == 0 or step == n_steps):
                    cg_it = (-1 if diagnostics.cg2d_iters is None else int(diagnostics.cg2d_iters))
                    print(
                        f"  step {step:5d}: |eta|max={em:.5f} "
                        f"|u|max={float(state.u.data.abs().max()):.5f} cg_it={cg_it}",
                        flush=True,
                    )
        finally:
            if ckpt_mgr is not None:
                ckpt_mgr.close()

        elapsed = time.time() - t0
        if is_rank0:
            (output_dir / "DONE").write_text("BLEW\n" if blew else "BOUNDED\n")
            print(
                f"FINAL step={step} {'BLEW' if blew else 'BOUNDED'} "
                f"wall={elapsed:.1f}s{_peak_mem_str(device)} -> {output_dir}",
                flush=True,
            )
        # Process group lifecycle belongs to torchrun / stage2 (regional-line parity);
        # do NOT destroy it here.
        return 1 if blew else 0
    except Exception as exc:  # noqa: BLE001 — fail-loud + non-zero exit for stage2
        logger.exception("cube framework run failed")
        print(f"[cube] failed: {exc}", flush=True)
        return 5
