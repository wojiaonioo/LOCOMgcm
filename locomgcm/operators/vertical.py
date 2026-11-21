from __future__ import annotations

from math import inf

import torch

from locomgcm.operators.fluxes.tracer_flux import (
    VERTICAL_CFL_SCHEMES,
    VERTICAL_VELOCITY_FLUX_SCHEMES,
    vertical_tracer_flux,
)


def _broadcast_layer_thickness(layer_thickness: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return layer_thickness.to(dtype=reference.dtype, device=reference.device).view(1, -1, 1, 1)


def _broadcast_interface_spacing(interface_spacing: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return interface_spacing.to(dtype=reference.dtype, device=reference.device).view(1, -1, 1, 1)


def _center_geopotentials(upper: torch.Tensor, lower: torch.Tensor) -> torch.Tensor:
    """Accumulate each interface-to-center and center-to-interface increment in level order."""
    nz = upper.shape[-3]
    if nz == 0:
        return upper
    # Preserve the widened accumulator of the CPU prefix sum. Round only
    # the stored centers, so a partial sum does not lose precision twice.
    dtype = upper.dtype
    accumulator_dtype = torch.float64 if dtype == torch.float32 else dtype
    if dtype in (torch.float16, torch.bfloat16):
        accumulator_dtype = torch.float32
    # A single unbind per column also assembles its layer cotangents once.
    # Independent selects give each layer a full-column scatter in backward.
    upper_levels = upper.unbind(dim=-3)
    lower_levels = lower.unbind(dim=-3) if nz > 1 else ()
    first = upper_levels[0]
    center = torch.zeros_like(first, dtype=accumulator_dtype) + first
    centers = [center.to(dtype)]
    for k in range(1, nz):
        interface = center + lower_levels[k - 1]
        center = interface + upper_levels[k]
        centers.append(center.to(dtype))
    return torch.stack(centers, dim=-3)


# Keep the column recurrence independent of its horizontal consumer. A compiler
# may otherwise use different parallel scan trees on the two sides of a halo.
if hasattr(torch.compiler, "nested_compile_region"):
    _center_geopotentials = torch.compiler.nested_compile_region(_center_geopotentials)


def hydrostatic_pressure_potential(
    density_anomaly: torch.Tensor,
    center_spacing: torch.Tensor,
    bottom_half_thickness: torch.Tensor,
    *,
    gravity: float,
    reference_density: float,
    hFacC: torch.Tensor | None = None,
) -> torch.Tensor:
    """Form MITgcm's half-level geopotential increments and return their center prefixes.

    CALC_PHI_HYD forms an upper-interface-to-center increment and a
    center-to-lower-interface increment (integr_GeoPot=2). Differences of
    the stored centers then give the momentum gradient in CALC_GRAD_PHI_HYD.
    Each column follows the same level order on both sides of a tile boundary.
    ``center_spacing`` is drC(1..Nr+1) and ``bottom_half_thickness`` the rC(Nr)
    to rF(Nr+1) distance (ZLevelCoordinate).
    """
    rho, upper, lower = _hydrostatic_layers(density_anomaly, center_spacing, bottom_half_thickness, hFacC)
    inv_rho = 1.0 / reference_density
    up = (upper * gravity) * rho * inv_rho
    down = (lower * gravity) * rho * inv_rho
    return _center_geopotentials(up, down)


def _hydrostatic_layers(
    density_anomaly: torch.Tensor,
    center_spacing: torch.Tensor,
    bottom_half_thickness: torch.Tensor,
    hFacC: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # calc_phi_hyd.F:251-256: rF(1) to rC(1) is drC(1), rF(k) to rC(k) and rC(k-1)
    # to rF(k) are drC(k)/2 below, rC(Nr) to rF(Nr+1) is bottom_half_thickness.
    drc = center_spacing.to(density_anomaly)
    if hFacC is not None:
        density_anomaly = density_anomaly * (hFacC > 0).to(density_anomaly)
    half_spacing = 0.5 * drc[1:-1]
    upper = torch.cat((drc[:1], half_spacing)).view(1, -1, 1, 1)
    lower = torch.cat((half_spacing, bottom_half_thickness.to(density_anomaly))).view(1, -1, 1, 1)
    return density_anomaly, upper, lower


def hydrostatic_pressure_integral(
    density_anomaly: torch.Tensor,
    center_spacing: torch.Tensor,
    bottom_half_thickness: torch.Tensor,
    *,
    hFacC: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the column mass anomaly at cell centers (kg m-2); geometry as hydrostatic_pressure_potential."""
    rho, upper, lower = _hydrostatic_layers(density_anomaly, center_spacing, bottom_half_thickness, hFacC)
    hydrostatic_load = rho * (upper + lower)
    return torch.cumsum(hydrostatic_load, dim=-3) - rho * lower


def vertical_velocity_from_divergence(
    horizontal_divergence: torch.Tensor,
    layer_thickness: torch.Tensor,
    *,
    top_boundary_value: float = 0.0,
) -> torch.Tensor:
    thickness = _broadcast_layer_thickness(layer_thickness, horizontal_divergence)
    top = torch.full(
        (
            horizontal_divergence.shape[0],
            1,
            horizontal_divergence.shape[-2],
            horizontal_divergence.shape[-1],
        ),
        fill_value=top_boundary_value,
        dtype=horizontal_divergence.dtype,
        device=horizontal_divergence.device,
    )
    interface_increments = torch.cumsum(-horizontal_divergence * thickness, dim=-3)
    return torch.cat((top, top + interface_increments), dim=-3)


def vertical_velocity_from_transport_divergence(
    transport_divergence: torch.Tensor,
    *,
    top_boundary_value: float = 0.0,
    bottom_up: bool = False,
) -> torch.Tensor:
    """Compute vertical velocity at layer interfaces from transport divergence.

    Unlike :func:`vertical_velocity_from_divergence`, this function expects
    the input to already be the transport divergence in units of m/s,
    including ``dz * hFac`` weighting.

    Two integration directions are supported:

    * ``bottom_up=False`` (default, backward-compatible): cumulative sum from
      surface downward.  ``w[0] = top_boundary_value`` (typically 0); the
      bottom ``w[nz]`` absorbs the column-integrated transport divergence.
      Consistent with rigid-lid surface boundary condition.

    * ``bottom_up=True`` (MITgcm-compatible): cumulative sum from bottom
      upward with ``w[nz] = 0`` (no-through-flow at seafloor, standard for
      free-surface z-coordinate). The surface ``w[0]`` then naturally
      equals the column-integrated transport divergence, which by continuity
      is ``-∂η/∂t`` in the positive-down convention (equivalent to MITgcm's
      upward-positive ``wVel_surface = ∂η/∂t``).  Required for tracer
      advection in multi-dim R-pass to avoid surface/bottom w mismatch
      with MITgcm, see ``docs/NUMERICS.md`` (w/depth sign conventions).

    Parameters
    ----------
    transport_divergence : Tensor [batch, nz, ny, nx]
        Horizontal transport divergence per layer, already weighted by
        ``dz * hFac``.  Units: m/s (velocity * length = transport/area).
    top_boundary_value : float
        Vertical velocity at the top interface (k=0), only used when
        ``bottom_up=False``.  Default 0.
    bottom_up : bool
        If True, integrate from bottom upward (MITgcm convention).
        Default False for backward compatibility.

    Returns
    -------
    w : Tensor [batch, nz+1, ny, nx]
        Vertical velocity at layer interfaces.
    """
    batch = transport_divergence.shape[0]
    ny = transport_divergence.shape[-2]
    nx = transport_divergence.shape[-1]
    dtype = transport_divergence.dtype
    device = transport_divergence.device

    if bottom_up:
        # Integrate bottom→top: w[nz] = 0, w[k] = w[k+1] + transport_div[k]
        # (rearranged from continuity w[k+1] - w[k] = -transport_div[k])
        bottom = torch.zeros((batch, 1, ny, nx), dtype=dtype, device=device)
        # reverse cumsum of transport_div along z: partial_sums[k] = sum(td[k:])
        # Then w[k] = partial_sums[k] (= -(-(w[k+1]-w[k]+...))).
        reversed_td = torch.flip(transport_divergence, dims=(-3,))
        reversed_cumsum = torch.cumsum(reversed_td, dim=-3)
        # partial_sums[k] = sum(transport_div[k:]) — re-flip back
        partial_sums = torch.flip(reversed_cumsum, dims=(-3,))
        # w[k] for k in [0, nz-1] = partial_sums[k]; w[nz] = 0
        return torch.cat((partial_sums, bottom), dim=-3)

    top = torch.full(
        (batch, 1, ny, nx),
        fill_value=top_boundary_value,
        dtype=dtype,
        device=device,
    )
    interface_increments = torch.cumsum(-transport_divergence, dim=-3)
    return torch.cat((top, top + interface_increments), dim=-3)


def vertical_velocity_continuity(
    horizontal_volume_divergence: torch.Tensor,
    *,
    cell_area: torch.Tensor | None = None,
    wet_mask: torch.Tensor | None = None,
    obcs_mask: torch.Tensor | None = None,
    form: str = "velocity",
    convention: str = "upward",
) -> torch.Tensor:
    """Geometry-neutral interface vertical velocity / volume flux from continuity.

    Single numerical core shared by the regional ``diagnose_vertical_velocity``,
    the cubed-sphere momentum ``w`` cache, and the cubed-sphere tracer R-pass
    volume flux, so the three ``w`` producers stop maintaining separate
    recurrences.  It reproduces MITgcm's ``INTEGRATE_FOR_W`` free-surface
    bottom-up recurrence (``integrate_for_w.F:186-196``)::

        wFld(k)  = wFld(k+1) + conv2d(k) * recip_rA      (conv2d = -hdiv)
        wFld(Nr) = conv2d(Nr) * recip_rA                 (seafloor w = 0)

    Leading dimensions are geometry-neutral (batch | cubed-sphere face | any
    independent column index): the kernel never reshapes or squeezes them.

    Conventions (docs/NUMERICS.md — sign bugs here surface ~150 steps later as a
    vertical-shear instability):

    * Integration is from the seafloor upward (``w[nz] = 0``), matching MITgcm's
      free-surface ``INTEGR_CONTINUITY`` (surface ``w`` ≠ 0, = ∂η/∂t).
    * ``convention='upward'`` returns MITgcm UPWARD-positive ``wVel`` (the native
      ``INTEGRATE_FOR_W`` sign), required by the cube momentum ``w`` cache and the
      cube tracer ``wflux`` consumers (which negate to LOCOMgcm-down at *their* boundary).
    * ``convention='downward'`` returns LOCOMgcm DOWNWARD-positive ``w`` (a
      positive value is downward flow), required by the regional
      ``diagnose_vertical_velocity`` output (``positive='down'`` metadata).
    * Negating ``w_down`` to obtain ``w_up`` is exact in IEEE-754 (sign-bit flip),
      so the regional ``convention='downward'`` path is byte-identical to the
      pre-refactor ``vertical_velocity_from_transport_divergence(bottom_up=True)``.

    Parameters
    ----------
    horizontal_volume_divergence : Tensor [*, nz, ny, nx]
        Per-layer horizontal volume-transport divergence ``dU + dV`` in m³/s
        (``horizontal_volume_divergence_from_transports``).  May already carry a
        wet-mask multiply (cube ``hdiv_vol``); pass ``wet_mask=None`` then.
    cell_area : Tensor or None
        MITgcm ``rA``, broadcastable to ``[*, nz, ny, nx]``.  Required for
        ``form='velocity'`` (the per-layer ``/rA`` that matches MIT's
        ``conv2d * recip_rA`` ordering — tighter than dividing the column sum).
        Ignored for ``form='volume_flux'``.
    wet_mask : optional Tensor
        Center wet mask (0/1), broadcastable to the divergence.  Multiplied onto
        the (post-``/rA``) center divergence; because it is 0/1 its multiply order
        relative to ``/rA`` is FP-irrelevant.  Pass ``None`` if the divergence is
        already masked.
    obcs_mask : optional Tensor
        MITgcm ``maskInC`` (OBCS interior mask), multiplied onto the final ``w``
        (``integr_continuity.F:293`` ``wVel*maskInC``).  Regional OBCS passes it;
        cube passes ``None`` (no open boundaries).
    form : str
        ``'velocity'`` → divide by ``cell_area`` (m/s; momentum / regional);
        ``'volume_flux'`` → no divide (m³/s; cube tracer R-pass ``wflux``).
    convention : str
        ``'upward'`` (MITgcm) or ``'downward'`` (LOCOMgcm).

    Returns
    -------
    w : Tensor [*, nz+1, ny, nx]
        Interface vertical velocity (``form='velocity'``) or volume flux
        (``form='volume_flux'``); ``w[nz] = 0`` (seafloor), surface ``w[0]`` ≠ 0.
    """
    if form == "velocity":
        if cell_area is None:
            raise ValueError("form='velocity' requires cell_area (rA)")
        center_divergence = horizontal_volume_divergence / cell_area
    elif form == "volume_flux":
        center_divergence = horizontal_volume_divergence
    else:
        raise ValueError(
            f"unsupported form: {form!r} (use 'velocity' | 'volume_flux')"
        )

    if wet_mask is not None:
        center_divergence = center_divergence * wet_mask

    # Bottom-up cumulative sum (w[nz]=0) producing LOCOMgcm downward-positive w.
    w_down = vertical_velocity_from_transport_divergence(
        center_divergence,
        bottom_up=True,
    )

    if convention == "upward":
        w = -w_down
    elif convention == "downward":
        w = w_down
    else:
        raise ValueError(
            f"unsupported convention: {convention!r} (use 'upward' | 'downward')"
        )

    if obcs_mask is not None:
        w = w * obcs_mask
    return w


def vertical_diffusive_flux(
    tracer: torch.Tensor,
    interface_spacing: torch.Tensor,
    *,
    diffusivity: float | torch.Tensor,
    surface_flux: torch.Tensor | None = None,
    bottom_flux: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute vertical diffusive flux on layer interfaces.

    Parameters
    ----------
    interface_spacing : tensor [nz-1]
        drC at the interior interfaces (ZLevelCoordinate.interface_spacing).
    surface_flux : optional tensor [batch, ny, nx]
        Prescribed flux at the top interface (k=0).  The sign convention is
        that *positive* surface_flux produces a *positive* tendency in the top
        layer (since ``tendency = -(F[1] - F[0]) / dz ≈ +F[0] / dz``).
        For wind-stress driven momentum mixing, pass ``+tau / rho_0``.
    bottom_flux : optional tensor [batch, ny, nx]
        Prescribed flux at the bottom interface (k=nz).  Same sign convention.
    mask : optional tensor [nz, ny, nx] or [batch, nz, ny, nx]
        Wet/dry mask (1=wet, 0=dry).  When provided, the flux at each
        interior interface is multiplied by ``mask[k] * mask[k-1]`` so that
        flux across a wet/dry boundary is exactly zero.  This matches
        MITgcm ``mom_u_rviscflux.F`` which uses ``_maskW(k)*_maskW(k-1)``.
    """
    flux = torch.zeros(
        (
            tracer.shape[0],
            tracer.shape[1] + 1,
            tracer.shape[2],
            tracer.shape[3],
        ),
        dtype=tracer.dtype,
        device=tracer.device,
    )
    if tracer.shape[1] < 2:
        if surface_flux is not None:
            flux[..., 0, :, :] = surface_flux
        if bottom_flux is not None:
            flux[..., -1, :, :] = bottom_flux
        return flux

    spacing = _broadcast_interface_spacing(interface_spacing, tracer)
    if isinstance(diffusivity, torch.Tensor):
        diffusivity_tensor = diffusivity.to(dtype=tracer.dtype, device=tracer.device)
        if tuple(diffusivity_tensor.shape) == tuple(tracer.shape):
            interface_diffusivity = 0.5 * (
                diffusivity_tensor[..., :-1, :, :]
                + diffusivity_tensor[..., 1:, :, :]
            )
        else:
            interface_diffusivity = diffusivity_tensor
    else:
        interface_diffusivity = tracer.new_tensor(diffusivity)

    # Zero flux at wet/dry interfaces (MITgcm: maskW(k)*maskW(k-1))
    if mask is not None:
        m = mask.to(dtype=tracer.dtype, device=tracer.device)
        if m.ndim == 3:
            m = m.unsqueeze(0)  # [1, nz, ny, nx]
        interface_mask = m[..., 1:, :, :] * m[..., :-1, :, :]
        interface_wet = interface_mask > 0
        tracer_below = torch.where(
            interface_wet,
            tracer[..., 1:, :, :],
            torch.zeros_like(tracer[..., 1:, :, :]),
        )
        tracer_above = torch.where(
            interface_wet,
            tracer[..., :-1, :, :],
            torch.zeros_like(tracer[..., :-1, :, :]),
        )
        tracer_gradient = (tracer_below - tracer_above) / spacing
        interior_flux = -interface_diffusivity * tracer_gradient
        masked_flux = interior_flux * interface_mask
        flux[..., 1:-1, :, :] = torch.where(
            interface_wet,
            masked_flux,
            torch.zeros_like(masked_flux),
        )
    else:
        tracer_gradient = (
            tracer[..., 1:, :, :] - tracer[..., :-1, :, :]
        ) / spacing
        flux[..., 1:-1, :, :] = -interface_diffusivity * tracer_gradient

    if surface_flux is not None:
        flux[..., 0, :, :] = surface_flux
    if bottom_flux is not None:
        flux[..., -1, :, :] = bottom_flux

    return flux


def vertical_flux_divergence(
    interface_flux: torch.Tensor,
    layer_thickness: torch.Tensor,
) -> torch.Tensor:
    thickness = _broadcast_layer_thickness(layer_thickness, interface_flux)
    return -(interface_flux[..., 1:, :, :] - interface_flux[..., :-1, :, :]) / thickness


def vertical_tracer_advection_transport_space(
    tracer: torch.Tensor,
    r_transport: torch.Tensor,
    *,
    velocity: torch.Tensor | None,
    cell_area: torch.Tensor,
    layer_thickness: torch.Tensor,
    scheme: str = "upwind",
    mask: torch.Tensor | None = None,
    interior_mask: torch.Tensor | None = None,
    tracer_for_correction: torch.Tensor | None = None,
    recip_drC: torch.Tensor | None = None,
    dt: float | None = None,
) -> torch.Tensor:
    """Geometry-neutral vertical tracer advection tendency (transport-space).

    Single numerical core shared by the regional ``VerticalTracerTransport``
    module and the cubed-sphere ``gad_advection_first_order_cube`` vertical pass,
    so the two geometries stop maintaining two separate vertical advection
    kernels.  It reproduces MITgcm's non-compressible multi-dim R-pass update
    (``gad_advection.F:567-573``) / single-RHS form (``gad_calc_rhs.F:769-781``)::

        gTracer = recip_rA * recip_drF *
                  ( -(fVerT[kDown] - fVerT[kUp])
                    - tracer * -(rTrans[kDown] - rTrans[kUp]) )

    expressed as ``vertical_flux_divergence(flux)/rA - tracer*vertical_flux_divergence(rTrans)/rA``.
   The returned tendency does NOT include the ``recip_hFacC`` factor — the caller
    applies it (regional ``VerticalTracerTransport.evaluate`` multiplies
    ``recip_hFacC``; the cube pass folds it into ``recip_volume``).  Leading
    dimensions are geometry-neutral (batch, cubed-sphere face, or any independent
    column index): the kernel never reshapes or squeezes them.

    Conventions (docs/NUMERICS.md, sign bugs here blow up ~150 steps later):
    ``r_transport`` is the SIGNED interface VOLUME transport ``w * rA`` in
    LOCOMgcm DOWNWARD-positive convention — a positive value is downward flow,
    whose donor cell is ABOVE the interface (matching the upwind kernel).
    Producers carrying MITgcm UPWARD-positive volume flux (``wflux``, e.g. the
    cube ``upward_vertical_volume_flux_from_horizontal_divergence``) MUST negate at
    the injection boundary: ``r_transport = -wflux``. The top (k=0, surface
    rigid lid) and bottom (k=nz, no-through-flow seafloor) interface transports
    are expected to be zero; callers zero them before calling.

    Parameters
    ----------
    tracer : Tensor [*, nz, ny, nx]
        Donor tracer for the advective flux.  In MIT's multi-dim sequential
        update this is the RUNNING tracer after the horizontal passes
        (``localTij``); the regional additive module passes the prognostic
        tracer (running == original there).
    r_transport : Tensor [*, nz+1, ny, nx]
        Signed interface volume transport (downward-positive), ends zeroed.
    velocity : Tensor [*, nz+1, ny, nx] or None
        Interface velocity (DOWNWARD-positive, MITgcm ``-wFld``) of the Courant
        number of a scheme in ``VERTICAL_CFL_SCHEMES`` outside
        ``VERTICAL_VELOCITY_FLUX_SCHEMES``; not read by the other schemes.
    cell_area : Tensor
        ``rA`` broadcastable to ``[*, nz, ny, nx]``.  Regional passes the scalar
        ``dx*dy`` (0-d tensor); cube passes the per-face 2-D area.  Used for the
        two ``/rA`` divides that fix MITgcm's transport-space FP ordering.
    layer_thickness : Tensor [nz]
        ``drF``.
    scheme : str
        ``"upwind"``/``"centered"`` select the donor-cell / centered kernel
        (transport-space, volume-weighted).  ``"dst3"`` selects MITgcm scheme 30
        (:func:`vertical_advective_flux_dst3`) and ``"dst3_fl"`` scheme 33
        (:func:`vertical_advective_flux_dst3fl`); both are velocity-space and
        require ``recip_drC`` and ``dt`` — the velocity is derived as
        ``r_transport/cell_area`` and its divergence is NOT divided by
        ``cell_area`` (already per-area).
    recip_drC : optional Tensor [nz-1]
        ``1/(center_depth[i]-center_depth[i-1])`` at interior interfaces; required
        for ``scheme="dst3"``/``"dst3_fl"`` (the DST3 Courant number).
    dt : optional float
        Tracer time step [s]; required for ``scheme="dst3"``/``"dst3_fl"``.
    mask : optional Tensor [nz,ny,nx] or [*,nz,ny,nx]
        Wet mask; zeroes flux across wet/dry interfaces (``maskC[k]*maskC[k-1]``).
    interior_mask : optional Tensor
        MITgcm ``maskInC`` (OBCS interior mask) applied to the flux and the
        transport divergence (``gad_calc_rhs.F:580``).  Regional OBCS passes it;
        cube passes ``None``.
    tracer_for_correction : optional Tensor [*, nz, ny, nx]
        Tracer for the ``-T * div(rTrans)`` correction.  MITgcm uses the ORIGINAL
        ``tracer(i,j,k)`` here even inside the multi-dim sequential update
        (``gad_advection.F:572``), so the cube pass must pass the original tracer
        while ``tracer`` (above) is the post-horizontal running field.  Defaults
        to ``tracer`` for the regional additive case.
    """
    t_corr = tracer if tracer_for_correction is None else tracer_for_correction

    if scheme in VERTICAL_CFL_SCHEMES and (recip_drC is None or dt is None):
        raise ValueError(
            f"{scheme} vertical advection requires recip_drC and dt"
        )
    if scheme in VERTICAL_VELOCITY_FLUX_SCHEMES:
        # MITgcm scheme 30 / 33: high-order DST3 VERTICAL advection, without and
        # with the Sweby flux limiter.  Both kernels are VELOCITY-space (their
        # CFL needs w in m/s and their flux body is velocity-weighted), so derive
        # the interface velocity from the signed VOLUME transport (rA cancels by
        # construction -> exactly consistent with the upwind transport-space
        # path) and DROP the ``/cell_area`` on the divergence (the velocity flux
        # is already per-area).  The ``-T*div(rTrans)`` correction below is
        # scheme-independent and stays in volume space.
        w_velocity = r_transport / cell_area
        flux = vertical_tracer_flux(
            scheme, tracer, None, velocity=w_velocity, recip_drC=recip_drC, dt=dt, mask=mask,
            layer_thickness=None, recip_center_spacing=None,
        )
        if interior_mask is not None:
            flux = flux * interior_mask
        flux_div = vertical_flux_divergence(flux, layer_thickness)
    else:
        flux_transport = vertical_tracer_flux(
            scheme, tracer, r_transport, velocity=velocity, recip_drC=recip_drC, dt=dt, mask=mask,
            layer_thickness=None, recip_center_spacing=None,
        )
        if interior_mask is not None:
            flux_transport = flux_transport * interior_mask
        flux_div = vertical_flux_divergence(flux_transport, layer_thickness) / cell_area

    div_r_transport = vertical_flux_divergence(r_transport, layer_thickness)
    if interior_mask is not None:
        div_r_transport = div_r_transport * interior_mask
    correction = t_corr * div_r_transport / cell_area

    return flux_div - correction


def explicit_vertical_diffusion_stability_limit(
    layer_thickness: torch.Tensor,
    *,
    diffusivity: float | torch.Tensor,
) -> torch.Tensor:
    if isinstance(diffusivity, torch.Tensor):
        # Keep as 1D to prevent torch.compile from promoting to SymFloat and crashing C++
        max_diffusivity = diffusivity.abs().max().view(1)
    else:
        max_diffusivity = layer_thickness.new_tensor([abs(float(diffusivity))])
    min_thickness = layer_thickness.min().view(1)
    limit = 0.5 * min_thickness * min_thickness / max_diffusivity
    inf_tensor = layer_thickness.new_tensor([float('inf')])
    return torch.where(max_diffusivity == 0.0, inf_tensor, limit).squeeze(0)
