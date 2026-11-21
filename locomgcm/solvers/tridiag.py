"""Tridiagonal solver (Thomas algorithm / TDMA).

Provides a batched tridiagonal solve for vertically implicit problems
in ocean modeling (e.g. vertical diffusion, semi-implicit free surface).

Design reference
----------------
- MOM6 ``regrid_solvers.F90:solve_tridiagonal_system`` — adapted to
  PyTorch batched tensors along the vertical (``dim=-3``) axis.
- Solver and physics are separated; the backend is replaceable.
"""

from __future__ import annotations


import torch


def _impldiff_build_batch_enabled() -> bool:
    """Toggle: vectorize the per-level ``for k`` diagonal-assembly loop in
    :func:`build_implicit_diffusion_system` into batched ops over the vertical
    axis.  Each level's (lower, diag, upper) coefficients depend only on
    level-``k`` inputs (drF, dz_interface, kappa at k/k+1, recip_hFac at k-1/k/k+1,
    dt) and NOT on a running k-accumulation, so the batched build is BITWISE vs
    the loop (arithmetic order preserved exactly; boundary rows k=0 / k=nz-1
    handled by slicing).  Default ON; ``LOCOMGCM_IMPLDIFF_BUILD_BATCH=0`` forces the
    per-level loop (eager fallback == HEAD, for OFF==ON bitwise verification).
    The Thomas SWEEP in :func:`tridiag_solve` is a genuine serial-in-k recurrence
    and is left UNTOUCHED.  The enable decision is owned by the unified control
    plane (``locomgcm.accel``), not by this module."""
    from locomgcm.accel import gate

    return gate("impldiff_build_batch").effective_value


def _tridiag_thomas(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
) -> torch.Tensor:
    """Thomas recurrence, verbatim as it stood before the adjoint was added.

    This is the general direct-division sweep. The
    gradient path (:class:`_TridiagSolveAdjoint`) does not re-implement it: it
    calls this same function a second time with permuted diagonals to solve the
    transposed system, so the forward and the adjoint can never drift apart.

    The serial slice assignments below are what autograd rejects once the
    coefficients carry a gradient ("modified by an inplace operation ... is at
    version nz"), which is why the gradient path runs this function inside an
    ``autograd.Function`` with grad mode off instead of taping it.
    """
    nz = d.shape[-3]

    # Work on clones to avoid mutating inputs.
    cp = c.clone()
    dp = d.clone()

    # --- forward sweep (elimination of lower diagonal) ---
    cp[..., 0, :, :] = cp[..., 0, :, :] / b[..., 0, :, :]
    dp[..., 0, :, :] = dp[..., 0, :, :] / b[..., 0, :, :]

    for k in range(1, nz):
        # The pivot is bound once instead of being spelled three times: the two
        # surviving uses are the same operands under the same operators in the
        # same order, so binding it is value-identical, not an algebraic
        # rewrite.
        a_k = a[..., k, :, :]
        pivot = b[..., k, :, :] - a_k * cp[..., k - 1, :, :]
        cp[..., k, :, :] = c[..., k, :, :] / pivot
        dp[..., k, :, :] = (dp[..., k, :, :] - a_k * dp[..., k - 1, :, :]) / pivot

    # --- backward sweep ---
    x = torch.empty_like(d)
    x[..., nz - 1, :, :] = dp[..., nz - 1, :, :]
    for k in range(nz - 2, -1, -1):
        x[..., k, :, :] = dp[..., k, :, :] - cp[..., k, :, :] * x[..., k + 1, :, :]

    return x


def forward_mode_requested(*tensors: torch.Tensor) -> bool:
    """True when any operand carries a forward-mode tangent.

    The routing predicate for the two tridiagonal solves: a tangent selects
    this module's implicit JVP, or the band solver's ordinary Torch recurrence.
    A dual tensor must not enter the reverse-only analytic-adjoint
    ``autograd.Function`` directly -- a
    ``Function`` in the traced region carries only ``backward``, because
    torch 2.9's Dynamo refuses one that declares ``jvp`` and breaks the graph
    at every tridiagonal solve (``_dynamo/variables/misc.py``).

    ``unpack_dual`` is the reading rather than
    ``torch._C._are_functorch_transforms_active()`` or
    ``torch._C._functorch.is_batchedtensor``: those two answer for reverse-mode
    transforms as well (``torch.func.grad``/``vjp``), where the recurrence's
    in-place sweep is exactly what autograd rejects, and neither is traceable
    under ``fullgraph=True``.  This one is true under ``torch.func.jvp`` and
    ``forward_ad.dual_level`` and false under plain taping and under a
    reverse-mode transform applied on its own (``torch.func.grad``/``vjp``,
    ``create_graph`` second order), which is the distinction the routing needs,
    and Dynamo folds it.

    Domain boundary : forward mode *composed with* reverse
    mode is outside it.  ``torch.func.jvp(torch.func.grad(f))`` hides the
    forward level under the grad level, the predicate reads ``False`` and the
    ``Function`` raises ``NotImplementedError`` for the missing ``jvp``; a
    band solver's ``dual_level`` tangent on a ``requires_grad`` leaf can still
    reach an in-place recurrence that reverse-mode autograd refuses. This
    predicate alone establishes no mixed-transform support. Reverse-mode
    ``vmap`` of the analytic boundary uses ``generate_vmap_rule``; batched
    forward tangents additionally require the selected backend's batching rule.
    """
    unpack = torch.autograd.forward_ad.unpack_dual
    return any(unpack(value).tangent is not None for value in tensors)


def _shift_down(t: torch.Tensor) -> torch.Tensor:
    """Move level ``k`` into slot ``k+1`` along the vertical axis; slot 0 zeroed
    (``out[k] = t[k-1]``)."""
    zero = torch.zeros_like(t[..., :1, :, :])
    return torch.cat([zero, t[..., :-1, :, :]], dim=-3)


def _shift_up(t: torch.Tensor) -> torch.Tensor:
    """Move level ``k`` into slot ``k-1`` along the vertical axis; last slot zeroed
    (``out[k] = t[k+1]``)."""
    zero = torch.zeros_like(t[..., :1, :, :])
    return torch.cat([t[..., 1:, :, :], zero], dim=-3)


class _TridiagSolveAdjoint(torch.autograd.Function):
    r"""Analytic adjoint of ``x = A^{-1} d`` for the tridiagonal ``A``.

    With ``A[k, k-1] = a[k]``, ``A[k, k] = b[k]``, ``A[k, k+1] = c[k]`` and a
    scalar loss ``L``, the reverse mode of a linear solve is

    .. math::
        \bar d = A^{-T}\,\bar x, \qquad \bar A = -\bar d\,x^{T},

    and because ``A`` only stores three diagonals the second identity collapses
    to three elementwise products::

        b_bar[k] = -d_bar[k] * x[k]
        a_bar[k] = -d_bar[k] * x[k-1]
        c_bar[k] = -d_bar[k] * x[k+1]

    ``A^T`` is again tridiagonal — its sub-diagonal is ``c`` shifted down and
    its super-diagonal is ``a`` shifted up — so the adjoint solve is the SAME
    forward recurrence with permuted diagonals (precedent:
    ``solvers/_cg3d_thomas/_fast.py`` ``_Cg3dThomasApply``; the coefficient
    gradients that routine returns as ``None`` are required here).

    Tape footprint is ``{a, b, c, x}``: four tensors, independent of ``nz``,
    against the ``~2*nz`` intermediates a taped recurrence would keep.  ``d`` is
    not saved — its gradient is ``d_bar`` itself.

    The transpose solve uses this same differentiable boundary recursively,
    so its coefficient and right-hand-side dependence remains available to
    higher derivatives.

    **Reverse mode only.**  This class declares ``backward`` and no ``jvp``:
    torch 2.9's Dynamo refuses an ``autograd.Function`` whose class declares
    one (``_dynamo/variables/misc.py``, ``"Unsupported custom jvp"``) and broke
    the graph at every tridiagonal solve of the compiled reverse pass -- nine
    breaks in the CS32 execution profile, with the lat-lon compiled backward on
    the same route. Forward mode is served by :func:`_tridiag_solve_tangent`,
    which unpacks the duals and solves ``A dx = dd - dA x`` through the public
    primal dispatcher. ``vmap`` without forward tangents still arrives here and is
    batched by ``generate_vmap_rule`` over ``forward``.
    """

    generate_vmap_rule = True

    @staticmethod
    def forward(a, b, c, d):  # type: ignore[override]
        return _tridiag_thomas(a, b, c, d)

    @staticmethod
    def setup_context(ctx, inputs, output):
        a, b, c, _ = inputs
        ctx.save_for_backward(a, b, c, output)

    @staticmethod
    def backward(ctx, grad_x):  # type: ignore[override]
        a, b, c, x = ctx.saved_tensors
        # The nested boundary retains the derivative of the transpose solve.
        d_bar = _TridiagSolveAdjoint.apply(
            _shift_down(c), b, _shift_up(a), grad_x.contiguous()
        )
        grad_a = grad_b = grad_c = grad_d = None
        if ctx.needs_input_grad[0]:
            grad_a = -d_bar * _shift_down(x)
        if ctx.needs_input_grad[1]:
            grad_b = -d_bar * x
        if ctx.needs_input_grad[2]:
            grad_c = -d_bar * _shift_up(x)
        if ctx.needs_input_grad[3]:
            grad_d = d_bar
        return grad_a, grad_b, grad_c, grad_d


def tridiag_solve(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    *,
    mitgcm_order: bool = False,
) -> torch.Tensor:
    """Solve a tridiagonal linear system along the vertical axis (dim=-3).

    Solves  A x = d  where A is tridiagonal with diagonals *a* (lower),
    *b* (main), *c* (upper) using the Thomas algorithm (TDMA).

    Parameters
    ----------
    a : Tensor [*, nz, ny, nx]
        Lower diagonal.  ``a[..., 0, :, :]`` is unused.
    b : Tensor [*, nz, ny, nx]
        Main diagonal.  Must be non-zero on all rows.
    c : Tensor [*, nz, ny, nx]
        Upper diagonal.  ``c[..., -1, :, :]`` is unused.
    d : Tensor [*, nz, ny, nx]
        Right-hand side.
    mitgcm_order : bool
        IMPLDIFF uses a reciprocal pivot followed by multiplication. The
        general solver keeps direct division, including for subnormal pivots.

    Returns
    -------
    x : Tensor [*, nz, ny, nx]
        Solution vector.

    Notes
    -----
    When any coefficient carries a gradient (KPP diagnoses ``KappaRX`` from the
    state, so ``a``/``b``/``c`` are functions of the prognostic variables) the
    call is routed through :class:`_TridiagSolveAdjoint`, which runs this exact
    recurrence with grad mode off and supplies the analytic adjoint.  The
    selected primal backend executes its existing Thomas recurrence.

    Forward mode solves ``A dx = dd - dA x`` through this same public primal
    solver. Splitting dual tensors before dispatch permits the native kernel to
    execute both solves, without differentiating its implementation or adding
    a custom ``Function.jvp`` to the compiled reverse path. The reference
    primal recurrence is unchanged; no matrix-coefficient tangent is frozen.
    """
    from locomgcm.accel import consumer_snapshot, in_accelerated_dispatch, requested

    # Split dual tensors before selecting a primal-only native kernel. The
    # implicit tangent uses the same A and retains all four input derivatives.
    if forward_mode_requested(a, b, c, d):
        return _tridiag_solve_tangent(a, b, c, d, mitgcm_order=mitgcm_order)
    if not in_accelerated_dispatch():
        with consumer_snapshot():
            if requested(("tridiag_cuda",)):
                from locomgcm.solvers._tridiag_cuda import dispatcher

                return dispatcher.tridiag_solve(a, b, c, d, mitgcm_order=mitgcm_order)
    needs_gradient = torch.is_grad_enabled() and (
        a.requires_grad or b.requires_grad or c.requires_grad or d.requires_grad
    )
    if mitgcm_order:
        from .implicit_vertical import _MitgcmOrderThomasAdjoint, _mitgcm_order_rows

        if needs_gradient:
            return _MitgcmOrderThomasAdjoint.apply(a, b, c, d, True)
        return _mitgcm_order_rows(a, b, c, d)
    if needs_gradient:
        return _TridiagSolveAdjoint.apply(a, b, c, d)
    return _tridiag_thomas(a, b, c, d)


def _tridiag_solve_tangent(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    *,
    mitgcm_order: bool,
) -> torch.Tensor:
    """Implicit JVP of a tridiagonal solve, including every coefficient.

    A tangent is a second solve of the SAME matrix, not a shorter physical
    time step. Normal tensors re-enter the public primal dispatcher, so both
    solves may use its native backend. Unpacking retains reverse dependencies;
    it is not tensor detachment.
    """
    duals = tuple(torch.autograd.forward_ad.unpack_dual(t) for t in (a, b, c, d))
    pa, pb, pc, pd = (t.primal for t in duals)
    da, db, dc, dd = (t.tangent for t in duals)
    x = tridiag_solve(pa, pb, pc, pd, mitgcm_order=mitgcm_order)
    rhs = torch.zeros_like(x) if dd is None else dd
    if da is not None:
        rhs = rhs - da * _shift_down(x)
    if db is not None:
        rhs = rhs - db * x
    if dc is not None:
        rhs = rhs - dc * _shift_up(x)
    tangent = tridiag_solve(pa, pb, pc, rhs, mitgcm_order=mitgcm_order)
    return torch.autograd.forward_ad.make_dual(x, tangent)


def build_implicit_diffusion_system(
    tracer: torch.Tensor,
    layer_thickness: torch.Tensor,
    *,
    interface_spacing: torch.Tensor,
    diffusivity: float | torch.Tensor,
    dt: float,
    recip_hFac: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Build the tridiagonal system for backward-Euler implicit vertical diffusion.

    Discretises  ``T^{n+1} = T^n + dt * ∂/∂z (κ ∂T^{n+1}/∂z)``
    as the tridiagonal system  ``(I - dt·L) T^{n+1} = T^n``
    where *L* is the second-order centered diffusion operator in *z*.

    Parameters
    ----------
    tracer : Tensor [B, nz, ny, nx]
    layer_thickness : Tensor [nz]
    interface_spacing : Tensor [nz-1]
        drC at the interior interfaces (ZLevelCoordinate.interface_spacing).
    diffusivity : float | Tensor
        Uniform or spatially varying diffusivity ``κ``.
    dt : float
        Time step.
    recip_hFac : Tensor [nz, ny, nx], optional
        Reciprocal of partial-cell fraction at each grid point.
        MITgcm ``impldiff.F:126-148``: coefficients are scaled by
        ``recip_hFac(k)`` and zeroed when the neighbour layer has
        ``recip_hFac = 0`` (solid wall boundary).

    Returns
    -------
    a, b, c, d : tuple of Tensor [B, nz, ny, nx]
        Lower, main, upper diagonals and RHS for :func:`tridiag_solve`.
    """
    from locomgcm.accel import consumer_snapshot, in_accelerated_dispatch, requested

    if not in_accelerated_dispatch():
        with consumer_snapshot():
            if requested(("tridiag_build_cuda",)):
                from locomgcm.solvers._tridiag_build_cuda import dispatcher

                return dispatcher.build_implicit_diffusion_system(
                    tracer,
                    layer_thickness,
                    interface_spacing=interface_spacing,
                    diffusivity=diffusivity,
                    dt=dt,
                    recip_hFac=recip_hFac,
                )
    nz = tracer.shape[-3]
    dz = layer_thickness.to(dtype=tracer.dtype, device=tracer.device).view(1, nz, 1, 1)

    # Interface spacing: distance between adjacent cell centres (MITgcm drC).
    dz_interface = interface_spacing.to(dtype=tracer.dtype, device=tracer.device).view(1, nz - 1, 1, 1)

    if isinstance(diffusivity, torch.Tensor):
        kappa = diffusivity.to(dtype=tracer.dtype, device=tracer.device)
    else:
        kappa = tracer.new_tensor(diffusivity)
    # Field diffusivity: a per-interface KappaR field (e.g. KPP KPPdiffKzT/KzS/viscAz)
    # where ``kappa[..., k, :, :]`` is the diffusivity at the TOP interface of layer k
    # (MITgcm KappaRX(k); the (k-1)<->k coupling).  impldiff.F uses KappaRX(k) for the
    # below-coupling of layer k and KappaRX(k+1) for the above-coupling.  A scalar /
    # non-vertical tensor keeps the original uniform broadcast (baseline path).
    kappa_is_field = kappa.ndim >= 3 and kappa.shape[-3] == nz

    def _kap(k: int) -> torch.Tensor:
        return kappa[..., k, :, :] if kappa_is_field else kappa

    # Diffusion coefficient at each interior interface:  κ / (dz_cell * dz_interface)
    # alpha_k = dt * κ / ( dz[k] * dz_interface[k] )   flux from layer k to k+1
    # beta_k  = dt * κ / ( dz[k] * dz_interface[k-1] ) flux from layer k to k-1

    # Broadcast recip_hFac for use in coefficient computation
    # MITgcm impldiff.F:126-148: a(k) = -dt * recip_hFac(k) * recip_drF(k)
    #                                    * KappaRX(k) * recip_drC(k)
    rhf: torch.Tensor | None = None
    if recip_hFac is not None:
        rhf = recip_hFac.to(dtype=tracer.dtype, device=tracer.device)
        if rhf.ndim == 3:
            rhf = rhf.unsqueeze(0)  # [1, nz, ny, nx]

    # IMPLDIFF constructs the row metric before multiplying by interface KappaRX.
    recip_drF = torch.reciprocal(dz)
    recip_drC = torch.reciprocal(dz_interface)
    row_scale = -dt * recip_drF if rhf is None else (-dt * rhf) * recip_drF
    a = torch.zeros_like(tracer)
    c_diag = torch.zeros_like(tracer)

    if _impldiff_build_batch_enabled():
        kappa_interface = kappa[..., 1:, :, :] if kappa_is_field else kappa
        lower = row_scale[..., 1:, :, :] * kappa_interface * recip_drC
        upper = row_scale[..., :-1, :, :] * kappa_interface * recip_drC
        if rhf is not None:
            lower = torch.where(rhf[..., :-1, :, :] != 0, lower, 0.0)
            upper = torch.where(rhf[..., 1:, :, :] != 0, upper, 0.0)
        a[..., 1:, :, :] = lower
        c_diag[..., :-1, :, :] = upper

        # The batched assembly above ran -- record the hit here, after the
        # accelerated work, never at the gate predicate.
        from locomgcm.accel import current_rank, record_dispatch

        record_dispatch(
            "impldiff_build_batch",
            outcome="hit",
            variant="batched_k_assembly",
            site="tridiag.build_implicit_diffusion_system",
            backend="torch",
            rank=current_rank(),
        )
    else:
        for k in range(1, nz):
            kappa_k = _kap(k)
            recip_face = recip_drC[..., k - 1, :, :]
            lower = row_scale[..., k, :, :] * kappa_k * recip_face
            upper = row_scale[..., k - 1, :, :] * kappa_k * recip_face
            if rhf is not None:
                lower = torch.where(rhf[..., k - 1, :, :] != 0, lower, 0.0)
                upper = torch.where(rhf[..., k, :, :] != 0, upper, 0.0)
            a[..., k, :, :] = lower
            c_diag[..., k - 1, :, :] = upper

    # impldiff.F:169 groups the two off-diagonal terms before subtracting.
    b = 1.0 - (a + c_diag)
    return a, b, c_diag, tracer.clone()
