"""Assemble the LOCOMgcm twin of the learned-KPP MITgcm truth case.

The MITgcm
truth (``experiments/learned_kpp/truth/gen_case.py``) is vermix widened to a
4x4x26 doubly periodic box with column-dependent heat flux and wind stress.
This module builds the LOCOMgcm model that runs the *reference* KPP on the
same inputs, so the forward can be compared step by step with MITgcm and later
reused as the host model of the learned closure.

Assembly path.  The case yaml (produced by ``experiments/mitcases/translate.py``
from the MIT case directory, see the yaml header) is assembled through the
production L3 builder, ``defgcm.io.runner.model_input``, phase by phase:
stage-1a bundles in memory -> IC tensors -> grid -> canonical IC transforms ->
runtime forcing sources -> ``build_model_from_manifest``.  That is
``build_model_state`` verbatim except that the model is kept instead of being
reduced to a ``ModelInput``; no zarr is written.  The fresh-run preparation
(``initialize_fresh_step_state``) is the one ``ModelRunner.run`` performs
before its step loop (``defgcm/runner.py:697-705``).

Reference KPP only.  The four KPP accelerator kernels (statekpp, bldepth,
blmix, fused) must resolve to "off"; the builder checks the resolved gate of
each instead of trusting the environment, and refuses a process that exports
``DEFGCM_KPP_*`` / ``N2FUSED`` / a nonzero ``STATEKPP`` or ``TRITON_KPP``.

Forcing.  ``sample_forcing(t)`` evaluates the *same* source objects the model's
forcing modules hold, so a caller (feature builder, loss) sees exactly the
fields the step consumes.  Conventions are MITgcm's: ``qnet`` W/m2 positive =
ocean loses heat, ``tau_x``/``tau_y`` N/m2 on the U/V points (MIT ``fu``/``fv``),
record n centred at (n - 1/2) * externForcingPeriod, linear in time, cyclic
(``defgcm/io/runner/layer_b/sources/mit_input_binary.py:883-936``).

Usage (library)::

    from experiments.learned_kpp.twin.build_to import build_model
    twin = build_model("<case>.yaml")
    state = twin.state
    for n in range(nsteps):
        state = twin.model.step(state, dt=twin.time_step, time_seconds=n * twin.dt)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

#: KPP accelerator kernels that would replace the reference torch path.
KPP_KERNEL_GATES = (
    "kpp_statekpp_cuda",
    "kpp_bldepth_cuda",
    "kpp_blmix_cuda",
    "kpp_fused_cuda",
)


def _refuse_kpp_accelerator_environment() -> None:
    """Fail before import-time gate resolution sees an enabling spelling."""
    offenders = sorted(
        name
        for name, value in os.environ.items()
        if name.startswith("DEFGCM_KPP_")
        or name == "N2FUSED"
        or (name in {"STATEKPP", "TRITON_KPP"} and value.strip() not in {"", "0"})
    )
    if offenders:
        raise RuntimeError(
            "the twin runs the reference KPP only; unset these variables: "
            + ", ".join(offenders)
        )


def _assert_reference_kpp_gates() -> dict[str, bool]:
    from defgcm.accel import gate

    resolved = {key: bool(gate(key).effective_value) for key in KPP_KERNEL_GATES}
    enabled = [key for key, on in resolved.items() if on]
    if enabled:
        raise RuntimeError(f"KPP accelerator gates resolved on: {enabled}")
    return resolved


def _module(model: Any, name: str) -> Any:
    found = [m for m in model.dynamics_modules if getattr(m, "name", "") == name]
    if len(found) != 1:
        raise RuntimeError(f"expected exactly one {name!r} module, found {len(found)}")
    return found[0]


@dataclass
class TwinModel:
    """The assembled twin.

    ``state`` is the cold-start state after the ``ModelRunner`` preparation
    (boundary pass, step index 0), i.e. the state MITgcm writes as iteration 0.
    ``sample_forcing(t)`` returns ``{"qnet", "tau_x", "tau_y", "empmr"}`` as
    (ny, nx) tensors at model time ``t`` seconds.
    """

    model: Any
    state: Any
    run_cfg: Any
    dt: float
    time_step: Any
    sample_forcing: Callable[[float], dict[str, torch.Tensor]]
    kpp_gates: dict[str, bool]


def build_model(
    config: str | Path,
    *,
    sets: Sequence[str] = (),
    initialize: bool = True,
) -> TwinModel:
    """Build the twin from a case yaml (``--set``-style ``sets`` overrides allowed).

    ``initialize=False`` returns the state straight from ``create_state``
    without the fresh-run preparation, for callers that restore cross-step
    state (AB history, lagged hydrostatic potential) from a pickup themselves.
    """
    _refuse_kpp_accelerator_environment()

    from experiments.cases._runtime import _production_time_step_bundle
    from defgcm.io.runner import model_input as mi
    from defgcm.io.runner.config import load_run_config
    from defgcm.io.runner.config._loader import merge_yaml_layers
    from defgcm.runner import initialize_fresh_step_state
    from defgcm.step.carry import supports_explicit_history

    config = Path(config)
    run_cfg = load_run_config(config, (), tuple(sets))
    raw_yaml = merge_yaml_layers(config, (), tuple(sets))
    # CUDA as well (all computation can run on the GPU); the reference KPP gates below stay
    # off on either device.
    if run_cfg.run.device not in ("cpu", "cuda") or run_cfg.run.precision != "float64":
        raise ValueError(
            "the twin is a float64 run on cpu or cuda; the yaml (or a run.device set) must "
            f"give run.device cpu|cuda and run.precision=float64, got "
            f"{run_cfg.run.device}/{run_cfg.run.precision}"
        )
    if not run_cfg.model.stepper.kwargs.get("kpp_enabled", False):
        raise ValueError("the twin needs model.stepper.kwargs.kpp_enabled=true")
    device, dtype = run_cfg.run.device, torch.float64

    # build_model_state (model_input.py:75-178), keeping the model.
    # yaml check: the MITgcm translation with the comparison on, else the
    # double-track check (stage 1a's run_params path).
    mi._model_input_run_params(run_cfg, raw_yaml)
    target_grid = mi._target_grid(run_cfg)
    bundles = mi._stage1a_bundles(run_cfg, target_grid=target_grid, logger=mi._NullLogger())
    ic_tensors = mi._ic_tensors_from_bundles(bundles, run_cfg, device=device, dtype=dtype)
    grid = mi._build_grid_from_tensors(
        run_cfg,
        ic_tensors["bathymetry"],
        del_x=ic_tensors.get("del_x"),
        del_y=ic_tensors.get("del_y"),
        device=device,
        dtype=dtype,
    )
    mi._apply_canonical_transforms(ic_tensors, grid, dtype=dtype, device=device)
    obcs_source, wind_source, flux_source, bulk_source = mi._build_runtime_sources(
        bundles, run_cfg, device=device, dtype=dtype
    )
    model, state = mi._assemble_model(
        run_cfg,
        ic_bundle=bundles["state"],
        ic_tensors=ic_tensors,
        grid=grid,
        obcs_source=obcs_source,
        wind_source=wind_source,
        surface_flux_source=flux_source,
        bulk_source=bulk_source,
        device=device,
        tidal_bundle=bundles.get("tidal"),
    )

    kpp_gates = _assert_reference_kpp_gates()
    if supports_explicit_history(model):
        # ModelRunner would then take the explicit-carry initialisation, which
        # this builder does not reproduce.  KPP runs are excluded from it
        # (step/carry.py:108), so reaching here means the yaml lost KPP.
        raise RuntimeError("unexpected explicit-history admission for a KPP run")

    dt = float(run_cfg.time.dt)
    time_step = _production_time_step_bundle(model, dt)
    if initialize:
        model.reset_runtime_budget()
        model.prepare_step_time(0.0)
        state = initialize_fresh_step_state(model, state)

    wind = _module(model, "era5_wind_forcing").source
    flux = _module(model, "wpflux_surface_forcing").source

    def sample_forcing(t: float) -> dict[str, torch.Tensor]:
        tau_x, tau_y = wind.sample(t)
        heat_flux, freshwater_flux = flux.sample(t)
        # heat_flux is TO-convention (positive warms); MIT Qnet is its negation.
        return {"qnet": -heat_flux, "tau_x": tau_x, "tau_y": tau_y,
                "empmr": freshwater_flux}

    return TwinModel(
        model=model,
        state=state,
        run_cfg=run_cfg,
        dt=dt,
        time_step=time_step,
        sample_forcing=sample_forcing,
        kpp_gates=kpp_gates,
    )
