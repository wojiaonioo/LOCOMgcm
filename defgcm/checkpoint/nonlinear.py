"""Restart codec for the two nonlinear free-surface geometry time levels."""

from __future__ import annotations

import math

import torch

from .state import _copy_tree, _snapshot_copy


_HFAC_NAMES = ("hFacC", "hFacW", "hFacS")
_FIELDS = {"hfac", "surface", "resting_surface", "time_varying", "coefficients"}


def _payload(model):
    stepper, grid = model.stepper, model.grid
    if getattr(stepper, "nonlin_free_surf", 0) < 1:
        return None
    if not stepper._nonlin_free_surf_active(grid):
        raise ValueError("nonlinear checkpoint requires C/W/S partial-cell geometry")
    if stepper.select_r_star != 0:
        raise NotImplementedError(
            "select_r_star=2: the r* factors and every-level hFac have no restart codec"
        )
    resting_geometry = stepper._surface_geometry.resting
    resting = (tuple(value[:1] for value in resting_geometry)
               if resting_geometry is not None else None)
    coefficients = []
    if stepper.nonlin_free_surf in (1, 2):
        # These coefficients keep the geometry of their FIRST assembly. The
        # resumed step's hFac cannot reconstruct them. Regional keys have no
        # cube layout; dtype/device are restored from the saved tensors/model.
        for key, value in grid._runtime_cache.items():
            if isinstance(key, tuple) and key and key[0] == "_helmholtz_coefficients":
                if key[-1] is not None:
                    raise ValueError("nonlinear checkpoint requires regional pressure coefficients")
                coefficients.append({"parameters": (key[1], key[2], key[3], key[6]),
                                     "values": value})
    return dict(hfac=tuple(getattr(grid.masks, name)[:1] for name in _HFAC_NAMES),
                surface=stepper._surface_geometry.surface, resting_surface=resting,
                time_varying=grid.masks.hfac_time_varying, coefficients=coefficients)


def encode_nonlinear_geometry(model, *, detach, device_resident=False):
    payload = _payload(model)
    return None if payload is None else _copy_tree(
        payload, copy=_snapshot_copy(detach=detach, device_resident=device_resident),
        path="nonlinear_geometry")


def head_nonlinear_geometry(model, *, snapshot=True):
    payload = _payload(model)
    return None if payload is None else _copy_tree(
        payload, copy=torch.Tensor.clone if snapshot else lambda value: value,
        path="nonlinear_geometry")


def validate_nonlinear_geometry(ckpt):
    level = ckpt.meta.physics_flags.get("nonlin_free_surf", 0)
    payload = ckpt.step_state.get("nonlinear_geometry")
    if level < 1:
        if payload is not None:
            raise ValueError("linear free surface cannot restore nonlinear_geometry")
        return
    if type(payload) is not dict or set(payload) != _FIELDS:
        raise ValueError("nonlinear continuation requires the complete nonlinear_geometry payload")
    ny, nx = ckpt.prognostic["eta"].shape[-2:]
    shapes = ((1, ny, nx), (1, ny, nx + 1), (1, ny + 1, nx))

    def triple(values, expected, name):
        if type(values) is not tuple or len(values) != 3:
            raise ValueError(f"nonlinear_geometry.{name} must contain C/W/S tensors")
        for value, shape in zip(values, expected, strict=True):
            if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                    or not value.is_floating_point() or not bool(torch.isfinite(value).all())):
                raise ValueError(f"nonlinear_geometry.{name} has invalid shape/dtype/values")

    triple(payload["hfac"], shapes, "hfac")
    initialized = payload["surface"] is not None
    if initialized != (payload["resting_surface"] is not None):
        raise ValueError("nonlinear_geometry surface and resting_surface must be saved together")
    if initialized:
        triple(payload["surface"], shapes, "surface")
        triple(payload["resting_surface"], shapes, "resting_surface")
    if type(payload["time_varying"]) is not bool:
        raise TypeError("nonlinear_geometry.time_varying must be a bool")
    if ckpt.meta.global_step > 0 and (not initialized or not payload["time_varying"]):
        raise ValueError("nonlinear continuation is missing initialized geometry time levels")
    coefficients = payload["coefficients"]
    if type(coefficients) is not list or (level > 2 and coefficients):
        raise ValueError("nonlinear_geometry has invalid retained pressure coefficients")
    seen = set()
    for entry in coefficients:
        if type(entry) is not dict or set(entry) != {"parameters", "values"}:
            raise ValueError("nonlinear_geometry pressure coefficient entry is invalid")
        params = entry["parameters"]
        if (type(params) is not tuple or len(params) != 4
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in params)):
            raise ValueError("nonlinear_geometry pressure coefficient parameters are invalid")
        triple(entry["values"], ((ny, nx + 1), (ny + 1, nx), (ny, nx)), "coefficients")
        identity = (params, entry["values"][0].dtype)
        if identity in seen:
            raise ValueError("nonlinear_geometry contains duplicate pressure coefficient keys")
        seen.add(identity)
    if level in (1, 2) and ckpt.meta.global_step > 0 and not coefficients:
        raise ValueError("nonlinear continuation is missing its first-assembly pressure coefficients")


def install_nonlinear_geometry(model, payload):
    """Install saved layers before prepare or any step-start physics reads hFac.

    With autograd enabled (a gradient window injecting a segment head) the
    layers enter by tensor replacement (``MaskSet.replace_surface_hfac``) and
    the resting geometry is the saved surface over the fixed deeper levels:
    the head tensors keep their graph, and no tensor another segment's graph
    saved is written or reached; the grid's ``hfac_*_for`` entries are rebuilt
    from the new tensors (``StructuredCGrid.refresh_hfac_runtime``).
    """
    if payload is None:
        return
    from defgcm.steppers.ifs.nlfs_geometry import SurfaceGeometry

    grid = model.grid
    # Only the surface changes during stepping. Reuse the configured deeper
    # geometry and replace the saved resting surface, never the lagged one.
    resting = payload["resting_surface"]
    if torch.is_grad_enabled():
        resting_geometry = None if resting is None else tuple(
            grid.masks.with_surface_level(name, value)
            for name, value in zip(_HFAC_NAMES, resting, strict=True))
        if payload["time_varying"]:
            grid.masks.mark_hfac_time_varying(resting_geometry)
        grid.masks.replace_surface_hfac(payload["hfac"])
        grid.refresh_hfac_runtime()
    else:
        resting_geometry = []
        for i, name in enumerate(_HFAC_NAMES):
            target = getattr(grid.masks, name)
            if resting is not None:
                reference = target.clone()
                reference[:1].copy_(resting[i].to(device=target.device, dtype=target.dtype))
                resting_geometry.append(reference)
        resting_geometry = tuple(resting_geometry) if resting is not None else None
        if payload["time_varying"]:
            grid.masks.mark_hfac_time_varying(resting_geometry)
        for name, value in zip(_HFAC_NAMES, payload["hfac"], strict=True):
            target = getattr(grid.masks, name)
            target[:1].copy_(value.to(device=target.device, dtype=target.dtype))
    surface = (None if payload["surface"] is None else
               tuple(value.to(device=model.config.device) for value in payload["surface"]))
    object.__setattr__(model.stepper, "_surface_geometry", SurfaceGeometry(resting_geometry, surface))
    for key in tuple(grid._runtime_cache):
        if isinstance(key, tuple) and key and key[0] == "_helmholtz_coefficients":
            del grid._runtime_cache[key]
    for entry in payload["coefficients"]:
        gravity, dt_mom, dt_freesurf, alpha_beta = entry["parameters"]
        values = tuple(value.to(device=model.config.device) for value in entry["values"])
        key = ("_helmholtz_coefficients", gravity, dt_mom, dt_freesurf,
               values[0].dtype, values[0].device, alpha_beta, None)
        grid._runtime_cache[key] = values
