"""Wire the learned KPP closure into the regional stepper.

The regional stepper diagnoses KPP once per step through
``RegionalKppInputs.diagnose`` (defgcm/dynamics/kpp_mixing.py) and then
consumes the returned ``KppFields`` unchanged: implicit vertical viscosity,
explicit non-local tracer tendency, implicit tracer diffusion. The first
version of the hybrid model replaces the result of that one call and leaves
the stepper and the consumers untouched.

``learned_diagnose`` is the drop-in: same arguments, same return type as the
reference ``diagnose``.  ``LearnedKppInjection`` installs it on the class for
the duration of a ``with`` block and keeps the last closure output so a
training loop can add auxiliary terms (for example ``log_k_loss`` on
``diff_kzt``) without a second forward pass.
"""
from __future__ import annotations

from typing import Any

import torch

from defgcm.dynamics.kpp_mixing import (
    KPP_STOCK_PARAMS,
    KppFields,
    RegionalKppInputs,
    _nonlocal_temp_forcing,
)
from defgcm.learned.closure_kpp import LearnedKppClosure, to_kpp_fields
from defgcm.learned.features import ColumnFeatures, column_features


def _crop_center(t: torch.Tensor, ny: int, nx: int) -> torch.Tensor:
    """Drop a symmetric halo if ``t`` is wider than the interior (ny, nx)."""
    hy, hx = t.shape[-2] - ny, t.shape[-1] - nx
    if hy == 0 and hx == 0:
        return t
    if hy < 0 or hx < 0 or hy % 2 or hx % 2:
        raise ValueError(f"cannot crop {tuple(t.shape)} to interior ({ny}, {nx})")
    return t[..., hy // 2: hy // 2 + ny, hx // 2: hx // 2 + nx]


def features_from_inputs(
    inputs: RegionalKppInputs,
    temp: torch.Tensor,
    salt: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    *,
    grid,
) -> ColumnFeatures:
    """Column features from the same inputs the reference KPP driver receives."""
    params: dict[str, Any] = inputs.params
    ny, nx = temp.shape[-2], temp.shape[-1]
    mask_c = (grid.masks.hFacC > 0).to(temp.dtype)
    mask_c = _crop_center(mask_c, ny, nx)
    coriolis = _crop_center(inputs.static.coriolis.to(temp.dtype), ny, nx)
    return column_features(
        temp, salt, u, v,
        inputs.surf_forc_t, inputs.surf_forc_s,
        inputs.surf_forc_u, inputs.surf_forc_v, inputs.surf_qsw,
        eos=str(params["eos_type"]),
        eos_pressure_pa=inputs.eos_pressure_pa,
        drf=inputs.static.layer_thickness.to(temp.dtype),
        mask_c=mask_c,
        coriolis=coriolis,
        rho0=float(params["rho_const"]),
        gravity=float(params["gravity"]),
        cp=float(params.get("Cp", KPP_STOCK_PARAMS.get("Cp", 3994.0))),
    )


def learned_diagnose(
    inputs: RegionalKppInputs,
    closure: LearnedKppClosure,
    temp: torch.Tensor,
    salt: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    *,
    grid,
) -> tuple[KppFields, dict[str, torch.Tensor]]:
    """Learned replacement of ``RegionalKppInputs.diagnose``.

    Returns the ``KppFields`` for the stepper and the raw closure output
    (``diff_kzt``, ``diff_kzs``, ``visc_cell``, ``ghat``, ``hbl``,
    ``gamma_iface``) for diagnostics and auxiliary losses.

    A closure with its own ``features_from_inputs`` (same arguments as the
    module function) builds its input from the stepper's KPP inputs itself;
    ``pp81_closure.Pp81Closure`` does, to read the raw state.
    """
    params = inputs.params
    build = getattr(closure, "features_from_inputs", None)
    if build is None:
        build = features_from_inputs
    feats = build(inputs, temp, salt, u, v, grid=grid)
    out = closure(
        feats,
        background_diff_t=inputs.background_diffusivity_t,
        background_diff_s=inputs.background_diffusivity_s,
        visc_arnr=float(params["visc_arnr"]),
        min_hbl=float(params["minKPPhbl"]),
    )
    sf_t_nl = _nonlocal_temp_forcing(inputs.surf_forc_t, inputs.surf_qsw, out["hbl"], params)
    fields = to_kpp_fields(
        out, inputs.surf_forc_t, inputs.surf_forc_s, sf_t_nl,
        grid=grid, visc_arnr=float(params["visc_arnr"]),
    )
    return fields, out


class LearnedKppInjection:
    """Route every ``RegionalKppInputs.diagnose`` call to ``closure`` inside a
    ``with`` block.

    The patch is process wide, like the reference monkeypatch probes in
    ; nest no two injections.  ``last_out`` is the closure output
    of the most recent call, ``calls`` counts them.
    """

    _active: "LearnedKppInjection | None" = None   # class-wide: one injection at a time

    def __init__(self, closure: LearnedKppClosure):
        self.closure = closure
        self.last_out: dict[str, torch.Tensor] | None = None
        self.calls = 0
        self._original = None

    def __enter__(self) -> "LearnedKppInjection":
        if LearnedKppInjection._active is not None:
            raise RuntimeError("a LearnedKppInjection is already active")
        LearnedKppInjection._active = self
        self._original = RegionalKppInputs.diagnose
        injection = self

        def diagnose(inputs_self, temp, salt, u, v, *, grid):
            fields, out = learned_diagnose(inputs_self, injection.closure, temp, salt, u, v, grid=grid)
            injection.last_out = out
            injection.calls += 1
            return fields

        RegionalKppInputs.diagnose = diagnose  # type: ignore[method-assign]
        return self

    def __exit__(self, *exc) -> None:
        RegionalKppInputs.diagnose = self._original  # type: ignore[method-assign]
        self._original = None
        LearnedKppInjection._active = None
