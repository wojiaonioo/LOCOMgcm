"""Reconstruct the constant-cache state of a continued compiled C-grid step."""
import torch

from defgcm.geometry.grid import ensure_grid_device_resident


PREVIOUS_CONVENTION = "compiled_geometry_resources_v1"
CONVENTION = "compiled_geometry_resources_v2"
_METRICS = (
    "cell_area", "dx_center", "dy_center", "dxC", "dyC", "dxG", "dyG",
    "dxV", "dyU", "rAw", "rAs", "rAz", "dxC_row", "dxG_row",
    "dxG_row_ext", "rAz_row",
)
_HFAC = ("hfac_c", "hfac_w", "hfac_s")
_MASKS = ("center", "u", "v")
_TYPES = {str(dtype).split(".")[-1]: dtype for dtype in
          (torch.float64, torch.float32, torch.bfloat16, torch.float16)}
_KNOWN = {**dict.fromkeys(_METRICS, "metric"), **dict.fromkeys((*_HFAC, "layer_thickness"), "grid"),
          **dict.fromkeys(_MASKS, "mask"), "leith": "viscosity", "regional_background": "kpp"}


def _cube_face_data(grid):
    if hasattr(grid, "face_geometry_for"):
        return getattr(grid.metrics, "face_geometry", {})
    return {}


def _grid_resources(grid, device, prefix=""):
    requested = set()

    def add(family, name, dtype, dev):
        resource_device = torch.device(dev)
        if resource_device.type == "cuda" and resource_device.index is None:
            resource_device = torch.device("cuda", torch.cuda.current_device())
        if resource_device == device:
            dt = str(dtype).removeprefix("torch.")
            if dt in _TYPES:
                requested.add(f"{dt}:{prefix}{family}:{name}")

    for owner, family in ((grid, "grid"), (grid.metrics, "metric")):
        # The grid holds its hFac copies apart, in _hfac_runtime.
        for key in (*getattr(owner, "_runtime_cache", {}), *getattr(owner, "_hfac_runtime", {})):
            if not isinstance(key, tuple) or len(key) != 3:
                continue
            name, dtype, dev = key
            if _KNOWN.get(name) == family:
                add(family, name, dtype, dev)
            elif family == "grid" and isinstance(name, str) and name.startswith("face_geometry::"):
                field = name.removeprefix("face_geometry::")
                if field in _cube_face_data(grid):
                    add("face", field, dtype, dev)
    cache = getattr(grid.masks, "_float_cache", None)
    if cache is not None:
        for _, (name, dtype, dev) in cache._store:
            if name in _MASKS:
                add("mask", name, dtype, dev)
    if grid.face_count == 1:
        # The variable-viscosity and KPP background producers are regional.
        for variant, key in getattr(grid, "_constant_memo", {}):
            if variant == "leith_length_scales":
                dtype, dev, _, _ = key[1]
                add("viscosity", "leith", dtype, dev)
            elif variant == "kpp_background_diffusivity_s_t":
                dtype, dev = key[1][:2]
                add("kpp", "regional_background", dtype, dev)
    return requested


def observation(model, step, device):
    """Encode only resources actually built on the prognostic tensor's device."""
    pending = getattr(model, "_restored_compiled_geometry", None)
    if pending is not None and pending[0] == step:
        # Re-saving an unadvanced pickup keeps the donor's resources. The
        # receiving workspace hasn't yet run the step that rebuilds them.
        return {} if pending[1] is None else {pending[2]: pending[1]}
    # Capture can run from a step callback or after ModelRunner.run returns.
    # The runner's own model-step entry is present in both lifetimes.
    compiled = getattr(model, "_model_runner_compile_cache", {})
    if not any(key[0] == "model_step" for key in compiled):
        return {}
    if step == 0:
        return {CONVENTION: ""}
    grid = model.grid
    requested = _grid_resources(grid, device)
    ctx = getattr(model, "cube_ctx", None)
    if ctx is not None and grid.face_count == 6:
        from defgcm.geometry.cube_localize import cached_local_cube_grid

        local = cached_local_cube_grid(grid, ctx)
        if local is not None:
            requested.update(_grid_resources(local, device, "local_"))
    return {CONVENTION: " ".join(sorted(requested))}


def _legacy(model, state):
    """Older files have no receipt. Build the available C-grid geometry."""
    grid = model.grid
    dtype = state.u.data.dtype
    fields = [("metric", name, dtype) for name in _METRICS]
    for name, field in zip(_HFAC, (grid.masks.hFacC, grid.masks.hFacW, grid.masks.hFacS)):
        if field is not None:
            fields.append(("grid", name, dtype))
    fields.extend(("mask", name, dtype) for name in _MASKS)
    fields.append(("viscosity", "leith", dtype))
    return fields


def _decode(hint, grid, has_local):
    if not isinstance(hint, str):
        return None
    fields = []
    for token in hint.split():
        parts = token.split(":")
        if len(parts) != 3:
            return None
        dt, family, name = parts
        if dt not in _TYPES:
            return None
        owner_family = family.removeprefix("local_")
        if owner_family != family and (not has_local or owner_family not in ("face", "grid", "metric", "mask")):
            return None
        if owner_family == "face":
            if name not in _cube_face_data(grid):
                return None
        elif _KNOWN.get(name) != owner_family:
            return None
        fields.append((family, name, _TYPES[dt]))
    return fields


def prepare(model, state, hint, *, convention=PREVIOUS_CONVENTION):
    from defgcm.accel import gate

    grid = model.grid
    ctx = getattr(model, "cube_ctx", None)
    has_local = (ctx is not None and grid.face_count == 6 and gate("cube_local_grid_cache").effective_value)
    fields = _decode(hint, grid, has_local)
    exact = fields is not None and convention == CONVENTION
    if fields is None:
        fields = _legacy(model, state)
    if not exact:
        # Older receipts observed metric, mask and hFac types but did not
        # report the coordinate casts or the separately owned cube face data.
        dtypes = dict.fromkeys([state.u.data.dtype, *(dtype for _, _, dtype in fields)])
        face_data = _cube_face_data(grid)
        if has_local:
            # V1 saw only model.grid. The actual cube momentum/tracer getter
            # may live on a different three-/two-/one-face grid for this rank.
            fields.extend(("local_" + family, name, dtype)
                          for family, name, dtype in _legacy(model, state)
                          if family in ("metric", "grid", "mask"))
        if grid.face_count == 1 and getattr(model.stepper, "kpp_enabled", False):
            fields.append(("kpp", "regional_background", state.temp.data.dtype))
        for dtype in dtypes:
            for prefix in (("", "local_") if has_local else ("",)):
                fields.append((prefix + "grid", "layer_thickness", dtype))
                for name in ("dxC", "dyC", "rAz"):
                    if name in face_data:
                        fields.append((prefix + "face", name, dtype))
    if not fields:
        return
    device = state.u.data.device
    ensure_grid_device_resident(grid, device)
    local = None
    if any(family.startswith("local_") for family, _, _ in fields):
        from defgcm.geometry.cube_localize import localize_cube_grid

        local = localize_cube_grid(grid, ctx)
        ensure_grid_device_resident(local, device)
    cached = gate("cache_constants").effective_value
    for family, name, dtype in fields:
        target = local if family.startswith("local_") else grid
        family = family.removeprefix("local_")
        # Names are restricted by the fixed resource table above. A file
        # cannot select an arbitrary attribute or activate an absent module.
        if family == "metric":
            getattr(target.metrics, name + "_for")(dtype, device)
        elif family == "grid":
            getattr(target, name + "_for")(dtype, device)
        elif family == "face":
            target.face_geometry_for(name, dtype, device)
        elif family == "kpp" and cached:
            from defgcm.steppers.regional_cold_start import prepare_regional_kpp_background

            prepare_regional_kpp_background(model.stepper, grid, dtype, device)
        elif family == "mask" and cached:
            getattr(target.masks, name + "_float")(dtype=dtype, device=device)
        elif family == "viscosity" and cached:
            for module in model.dynamics_modules:
                prewarm = getattr(module, "prepare_compiled_resume_constants", None)
                if prewarm is not None:
                    prewarm(grid, dtype=dtype, device=device)
