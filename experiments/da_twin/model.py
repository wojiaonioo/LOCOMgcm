"""Current LOCOMgcm model construction and checkpointed cost evaluation."""
from __future__ import annotations

import contextlib
import logging
from pathlib import Path
import time
from types import SimpleNamespace, MethodType

import numpy as np
import torch

from .control import ControlTransform, FIELDS, load_b_std
from .features import Grid, frame_step, load_rows
from .observations import Observations


def native_fields(path):
    """Read MITgcm west/south faces, appending the redundant exterior faces."""
    with np.load(path, allow_pickle=False) as z:
        fields = {n: torch.as_tensor(np.asarray(z[n], np.float64)) for n in FIELDS}
    fields['u'] = torch.cat((fields['u'], fields['u'][..., -1:]), -1)
    fields['v'] = torch.cat((fields['v'], fields['v'][..., -1:, :]), -2)
    return fields


class Problem:
    def __init__(self, spec, bstd, *, kpp='convective', segment=1,
                 device_bytes=32 * 1024**3, compiled=True, momentum=None):
        from defgcm.io.runner.config import load_run_config
        from experiments.cases._runtime import (
            _load_canonical_and_build_model, _production_time_step_bundle,
            _run_cfg_with_device,
        )
        from defgcm.checkpoint import capture
        from defgcm.runner import initialize_fresh_step_state
        from defgcm.accel import preload_requested_extensions

        self.spec, self.segment, self.device_bytes = spec, segment, device_bytes
        preload_requested_extensions(device='cuda')
        self.C = ControlTransform(spec['grid'], load_b_std(bstd))
        overlay = None
        if kpp == 'convective':
            std = load_b_std(bstd)
            overlay = {'model': {'stepper': {'kwargs': {
                'kpp_surface_buoyancy': 'divided_difference',
                'kpp_convective_implicit': {
                    'root_policy': 'newton_factor_homotopy_v1',
                    'root_tolerance': 1e-11, 'polish_steps': 2,
                    'execution': 'compiled' if compiled else 'eager',
                    'residual_scale': {n: std[n] for n in ('temp', 'salt')},
                },
            }}}}
        self.cfg = _run_cfg_with_device(load_run_config(Path(spec['case']), extra_yaml=overlay), 'cuda')
        self.dt = float(self.cfg.time.dt)
        if self.dt != 600:
            raise ValueError('The twin protocol requires a 600 s step')
        self.spd = round(86400 / self.dt)
        self.model, state, ctx = _load_canonical_and_build_model(
            self.cfg, output_dir=Path(spec['run_dir']), mitgcm_run_dir=None,
            n_steps=8 * self.spd, dt=self.dt, device='cuda', logger=logging.getLogger(__name__))
        self.bundle = _production_time_step_bundle(self.model, self.dt)
        self.origin = float(ctx['base_time_seconds'])
        self.shapes = {n: tuple(getattr(state, n).data.shape) for n in FIELDS}
        self.xb = {n: getattr(state, n).data.detach().cpu().reshape(self.C.shapes[n]).clone() for n in FIELDS}
        expected = native_fields(spec['background'])
        for n in FIELDS:
            if not torch.equal(self.xb[n][self.C.support[n]], expected[n][self.C.support[n]]):
                raise ValueError(f'{n}: prepared initial state differs from the adjusted background on control support')
        self.wet = {n: torch.as_tensor(m) for n, m in Grid(spec['grid']).mask.items()}
        self.bindings = ()
        if momentum is not None:
            from .momentum import NetworkMomentumTendency
            term = NetworkMomentumTendency(momentum, self.model)
            self.model.dynamics_modules = (*self.model.dynamics_modules, term)
            self.model.fast_dynamics_modules, self.model.slow_dynamics_modules = self.model._partition_dynamics_modules(self.model.dynamics_modules)
            self.model.slow_momentum_modules, self.model.slow_tracer_modules = self.model._partition_by_phase(self.model.slow_dynamics_modules)
            self.model.assembly_policy.validate_modules((term,), source='twin momentum network')
            self.bindings = ((momentum, 'theta', momentum.theta),)
        self.fingerprint = 'south-china-sea-twin'
        self.model.prepare_step_time(self.origin)
        state = initialize_fresh_step_state(self.model, state)
        self.seed = capture(self.model, state, global_step=0, model_time_seconds=self.origin,
                            dt=self.dt, config_fingerprint=self.fingerprint)
        self.template = state.with_tensors(**{n: self.xb[n].reshape(self.shapes[n]) for n in FIELDS})
        self.stack = contextlib.ExitStack()
        if compiled:
            self._compile()
        self.obs = {}

    def _compile(self):
        from defgcm.runner import RunnerCompileConfig, _configure_step_execution
        from defgcm.step import build_step_plan, finalize_step
        fn, meta = self.stack.enter_context(_configure_step_execution(
            self.model, RunnerCompileConfig(target='model_step', pin_schedule=False)))
        if not meta['enabled']:
            raise RuntimeError(f'Model compilation was not enabled: {meta}')

        def step(model, state, dt=None, *, time_seconds=None):
            actual = model.current_time_seconds if time_seconds is None else float(time_seconds)
            host = build_step_plan(model, state, dt, time_seconds=actual)
            return finalize_step(model, fn(state, host.plan), host)
        self.model.step = MethodType(step, self.model)

    def close(self):
        self.stack.close()

    def observations(self, steps):
        if steps not in self.obs:
            self.obs[steps] = Observations(self.spec['obs'], self.C.shapes, self.wet, steps)
        return self.obs[steps]

    def fields(self, v):
        inc = self.C(torch.as_tensor(v, dtype=torch.float64))
        return {n: self.xb[n] + inc[n] for n in FIELDS}, inc

    def restart(self, fields, grad=False):
        from defgcm.checkpoint import restore
        from defgcm.runner import initialize_fresh_step_state
        restore(self.model, self.seed, config_fingerprint=self.fingerprint)
        leaves = {n: fields[n].reshape(self.shapes[n]).detach().to('cuda', copy=True).requires_grad_(grad) for n in FIELDS}
        self.model.prepare_step_time(self.origin)
        state = initialize_fresh_step_state(self.model, self.template.with_tensors(**leaves))
        return state, leaves

    def forward(self, fields, steps, observe):
        with torch.no_grad():
            state, _ = self.restart(fields)
            values = []
            for index in range(steps):
                state = self.model.step(state, dt=self.bundle,
                    time_seconds=self.origin + index * self.bundle.dt_clock)
                values.append(observe(self.model, state))
        return float(torch.stack(values).sum())

    def evaluate(self, v, steps, grad=False, hx=None):
        from defgcm.checkpoint.window import WindowOwner
        from defgcm.checkpoint import CheckpointStore
        fields, _ = self.fields(v)
        obs = self.observations(steps)
        j0, _, g0 = obs.t0(fields)
        if hx is not None:
            for kd, name, idx, _, _ in obs.groups.get(0, []):
                hx[(0, kd)] = fields[name].reshape(-1)[idx].clone()
        observe = obs.objective(SimpleNamespace(origin=self.origin, bundle=self.bundle), steps, {}, hx)
        started = time.perf_counter()
        if grad:
            state, leaves = self.restart(fields, True)
            owner = WindowOwner.capture(self.model, [state], global_step=0,
                model_time_seconds=self.origin, dt=self.dt, config_fingerprint=self.fingerprint,
                store=CheckpointStore(device_bytes=self.device_bytes))
            for _, _, p in self.bindings:
                p.grad = None
            loss = owner.run(steps=steps, dt=self.bundle, observe=observe,
                segment_steps=self.segment, parameter_bindings=self.bindings).sum()
            owner.backward(loss)
            jo = float(loss.detach())
            gx = {n: leaves[n].grad.detach().cpu().reshape(self.C.shapes[n]) + g0[n] for n in FIELDS}
            g = np.asarray(v) + self.C.adjoint(gx).numpy()
        else:
            jo = self.forward(fields, steps, observe)
        value = 0.5 * float(np.dot(v, v)) + j0 + jo
        torch.cuda.synchronize()
        row = dict(cost=value, obs_count=obs.count, wall_s=time.perf_counter() - started)
        if not np.isfinite(value) or (grad and not np.isfinite(g).all()):
            raise FloatingPointError('Non-finite cost or control gradient')
        return (value, g, row) if grad else (value, row)

    def value_grad(self, v, steps, hx=None):
        return self.evaluate(v, steps, True, hx)

    def value(self, v, steps):
        return self.evaluate(v, steps)


class Window:
    def __init__(self, spec, bstd, **kwargs):
        self.spec, self.name, self.split = spec, spec['name'], spec['split']
        self.prob = Problem(spec, bstd, **kwargs)
        self.C, self.grid_npz = self.prob.C, spec['grid']
        self.dt, self.steps = self.prob.dt, 432
        self.frame_step = frame_step(self.dt)
        self.rows = load_rows(spec['obs'], self.steps, self.dt)
        self.m = len(self.rows['step'])

    def background(self, cache_dir=None):
        if not hasattr(self, 'hx'):
            rec = {}
            self.prob.evaluate(np.zeros(self.C.size), self.steps, hx=rec)
            self._background_rows(rec)
        return dict(J0=self.j0)

    def background_and_g0(self):
        rec = {}
        result = self.prob.value_grad(np.zeros(self.C.size), self.steps, hx=rec)
        self._background_rows(rec)
        return result

    def _background_rows(self, rec):
        self.hx = np.full(self.m, np.nan)
        for (step, kind), values in rec.items():
            mask = (self.rows['step'] == step) & (self.rows['kind'] == kind)
            self.hx[mask] = values.detach().cpu().numpy()
        if not np.isfinite(self.hx).all():
            raise ValueError('Background trajectory missed observation rows')
        self.j0 = float(0.5 * np.square(self.innovation / self.rows['sigma']).sum())

    @property
    def innovation(self):
        return self.rows['value'] - self.hx
