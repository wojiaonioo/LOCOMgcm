"""Compact linearization of the implicit convective KPP column equation.

The exact Jacobian in scaled tracer coordinates is B + U V^T. B is a
2 by 2 block tridiagonal transport matrix, including the local convective
derivative below the boundary layer. The boundary-layer matching uses only
the three interior factors at kn-1, kn and kn+1; enhance reads kbl-1, which
belongs to that same set for both caseA values. Their full influence on all
boundary-layer interfaces forms the rank-at-most-three update.

The assembly follows the analytic residual derivative, including masks,
abs'(0)=0, the inclusive clamp derivative and half-weight maximum ties.
The external-input pullback remains the complete residual autograd program.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from locomgcm.accel import dispatch_scope
from locomgcm.dynamics import equation_of_state as eos
from locomgcm.dynamics import kpp
from locomgcm.solvers import _block_tridiag as blocks


@dataclass
class ColumnLinearization:
    lower: torch.Tensor
    diagonal: torch.Tensor
    upper: torch.Tensor
    u: torch.Tensor
    vt: torch.Tensor
    dense_overrides: list = field(default_factory=list)

    @property
    def nz(self) -> int:
        return self.diagonal.shape[0]

    @property
    def nc(self) -> int:
        return self.diagonal.shape[-1]


@dataclass
class ColumnFactors:
    linearization: ColumnLinearization
    block: blocks.BlockFactors
    w: torch.Tensor
    small_lu: torch.Tensor
    small_pivots: torch.Tensor
    info: torch.Tensor
    dense_factors: list = field(default_factory=list)


class LinearSolveFailure(RuntimeError):
    def __init__(self, record):
        super().__init__(f"Implicit column linear solve failed: {record}")
        self.record = record


def _diagonal_blocks(value):
    # (2, nz, nc) -> (nz, 2, 2, nc), without any nz by nz allocation.
    zero = torch.zeros_like(value[0])
    return torch.stack((torch.stack((value[0], zero), dim=1),
                        torch.stack((zero, value[1]), dim=1)), dim=1)


def _linearize_eager(problem, z, aux):
    p = problem
    nz, nc = p.nz, p.nc
    Y = p.old + p.unpack(z) * p.scale
    t, s = Y[0], Y[1]
    pressure = aux["pressure"][1:]
    gravity, rho0 = p.par["gravity"], p.par["rho_const"]
    den = eos.jmd95_density_anomaly(t[1:], s[1:], pressure, reference_density=rho0) + rho0
    drho = eos.jmd95_density_difference(t[1:], s[1:], t[:-1], s[:-1], pressure)
    buoyancy = gravity * drho / den
    db = torch.cat((buoyancy, torch.zeros_like(t[:1])), 0) * p.interface_mask
    dz = (p.zgrid[1:nz + 1] - p.zgrid[2:nz + 2])[:, None, None]
    ns = db / dz
    at, bs = eos.jmd95_alpha_beta(t[:-1], s[:-1], pressure)
    ab, bb = eos.jmd95_alpha_beta(t[1:], s[1:], pressure)
    common = gravity / (dz[:-1] * den) * p.pair
    top = torch.stack((-common * at, -common * bs)) * p.scale[:, :-1]
    bottom = torch.stack((common * (1 - drho / den) * ab,
                          common * (1 - drho / den) * bb)) * p.scale[:, 1:]
    b = p.ri_kwargs["BVSQcon"]
    ratio = torch.clamp((b - torch.clamp(ns, min=b)) / b, max=1.)
    active = (ns > b) & (ns < 0) & p.interface_mask
    df = torch.where(active, 6 * ratio * (1 - ratio.square()).square() / b, torch.zeros_like(ns))
    top = (df[:-1][None] * top).reshape(2, nz - 1, nc)
    bottom = (df[:-1][None] * bottom).reshape(2, nz - 1, nc)

    ra = list(p.ra)
    for index in (1, 3, 4, 5):
        ra[index] = aux["ri_" + str(index)]
    ra[2] = db
    viscosity, salt_diffusivity, temp_diffusivity = kpp.kpp_ri_iwmix(*ra, **p.ri_kwargs)
    interior = torch.stack((viscosity, salt_diffusivity, temp_diffusivity))
    km = p.ra[0].reshape(nc)
    below_bottom = (torch.arange(nz, device=Y.device)[None, :] + 1) >= km[:, None]
    interior = torch.where(below_bottom.T.reshape(1, nz, p.ny, p.nx),
                           torch.zeros_like(interior), interior)
    pad = Y.new_zeros((3, 1, p.ny, p.nx))
    diffus = torch.cat((pad, interior, pad), dim=1)
    ba = list(p.ba)
    for index in (0, 1, 2, 3, 4):
        ba[index] = aux["bl_" + str(index)]
    ba[6] = diffus
    blmc, ghat, dkm1 = kpp.kpp_blmix(*ba, **p.bl_kwargs)
    ustar, bfsfc, hbl, stable, casea, kbl = ba[:6]
    zgrid, hwide, wmt, wst = ba[7:11]
    kw = p.bl_kwargs
    phepsi = kw.get("phepsi", 1e-10)
    epsilon, conc1 = kw["epsilon"], kw["conc1"]
    extra = {key: value for key, value in kw.items()
             if key not in ("epsilon", "conc1", "cg", "phepsi", "sign_zero")}
    sigma = stable + (1 - stable) * epsilon
    wm, ws = kpp.kpp_wscale(sigma, hbl, ustar, bfsfc, wmt, wst, **extra)
    mode = kw.get("sign_zero", "signbit")
    wm = kpp._fortran_sign(1., wm, mode) * wm.abs().clamp_min(phepsi)
    ws = kpp._fortran_sign(1., ws, mode) * ws.abs().clamp_min(phepsi)
    w0 = torch.stack((wm, ws, ws)).reshape(3, nc)
    case_int = (casea + phepsi).long()
    kn = (case_int * (kbl - 1) + (1 - case_int) * kbl).reshape(nc).long()
    H, ST, CA, KB = hbl.reshape(nc), stable.reshape(nc), casea.reshape(nc), kbl.reshape(nc).long()

    # A derivative axis of length three replaces the former 2*nz axis. Its
    # coordinates are the three scalar convective factors, shared by T and S.
    matching = torch.stack((kn - 1, kn, kn + 1), dim=1)
    interface = torch.arange(nz + 2, device=Y.device)[None, :, None]
    selector = (interface == matching[:, None, :])
    selector = selector & (interface >= 1) & (interface < km[:, None, None]) & (interface <= nz)
    amplitudes = Y.new_tensor([p.ri_kwargs[name] for name in ("difmcon", "difscon", "diftcon")])
    DD = amplitudes[:, None, None, None] * selector[None].to(Y.dtype)

    # V^T maps scaled state perturbations to those three factors. It is stored
    # as three short dense rows; its only nonzeros join the two adjacent levels.
    level = torch.arange(nz, device=Y.device)[None, :, None]
    basis = (level == (matching - 1).T[:, None, :]).to(Y.dtype)
    top_full = torch.cat((top, top.new_zeros((2, 1, nc))), dim=1)
    bottom_full = torch.cat((bottom, bottom.new_zeros((2, 1, nc))), dim=1)
    vt_top = basis[:, None] * top_full[None]
    vt_bottom = basis[:, None] * bottom_full[None]
    vt_bottom = torch.cat((vt_bottom.new_zeros((3, 2, 1, nc)), vt_bottom[:, :, :-1]), dim=2)
    vt = (vt_top + vt_bottom).permute(3, 0, 1, 2).reshape(nc, 3, 2 * nz)

    def gather(value, index):
        if value.ndim == 1:
            return value[index.clamp(0, value.shape[0] - 1)]
        ix = index.clamp(0, value.shape[2] - 1)[None, :, None, None].expand(
            value.shape[0], -1, 1, value.shape[-1])
        return value.gather(2, ix).squeeze(2)

    values = diffus.reshape(3, nz + 2, nc).permute(0, 2, 1).unsqueeze(-1)
    hw, hwp = gather(hwide, kn), gather(hwide, kn + 1)
    delhat = .5 * hw - gather(zgrid, kn) - H
    R = 1 - delhat / hw
    dminus, dkn, dplus = [gather(values, index).squeeze(-1) for index in (kn - 1, kn, kn + 1)]
    up, dn = (dminus - dkn) / hw, (dkn - dplus) / hwp
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
    sigma_d = (ST * sigd + (1 - ST) * sigd.clamp(max=epsilon)).reshape(p.ny, p.nx)
    wd_m, wd_s = kpp.kpp_wscale(sigma_d, hbl, ustar, bfsfc, wmt, wst, **extra)
    wd = torch.stack((wd_m, wd_s, wd_s)).reshape(3, nc)
    ddkm = (H[None] * wd * sigd[None].square())[..., None] * (
        (3 - 2 * sigd)[None, :, None] * dgat + (sigd - 1)[None, :, None] * ddat)
    ki = KB - 1
    valid = (ki >= 1) & (ki < nz)
    to = (ki - 1).clamp(0, nz - 1)
    delta = (H + gather(zgrid, ki)) / (gather(zgrid, ki) - gather(zgrid, ki + 1))
    dki, dbki = gather(DD, ki), gather(dbmix, to)
    new = (1 - delta)[None, :, None] * dki + delta[None, :, None] * (
        (1 - delta)[None, :, None].square() * ddkm + delta[None, :, None].square() * (
            CA[None, :, None] * dki + (1 - CA)[None, :, None] * dbki))
    new = torch.where(valid[None, :, None], new, dbki)
    dbmix = dbmix.scatter(2, to[None, :, None, None].expand(3, nc, 1, 3), new[:, :, None, :])
    post, ghat = kpp.kpp_enhance(dkm1, hbl, kbl, casea, diffus, blmc, ghat, zgrid)
    # The value and derivative share this RI/BLMIX composition. Construct the
    # transport coefficients with the same combine and mask as the residual.
    out, ghat = kpp.kpp_combine(
        post, diffus, ghat, kbl, visc_arnr=p.par["visc_arnr"],
        bg_KzS_bottom=ra[4][-1], bg_KzT_bottom=ra[5][-1])
    km1 = torch.clamp(torch.arange(nz, device=Y.device) - 1, min=0)
    cmask = p.mask * p.mask.index_select(0, km1)
    K = torch.stack((out[2, :nz], out[1, :nz])) * cmask
    middle = (K[:, 1:] / p.dc[None, :, None, None]) * p.pair
    velocity = aux["w"][1:-1] * p.pair
    above = middle + (-velocity).clamp_min(0)[None]
    below = -middle - velocity.clamp_min(0)[None]
    zeros = Y.new_zeros((2, 1, p.ny, p.nx))
    dplus, dminus = torch.cat((above, zeros), 1), torch.cat((zeros, below), 1)
    diagonal = 1 + p.dt * (dplus - dminus) * p.inv_volume[None]
    lower = -p.dt * above * p.inv_volume[None, 1:] * (p.scale[:, :-1] / p.scale[:, 1:])
    upper = p.dt * below * p.inv_volume[None, :-1] * (p.scale[:, 1:] / p.scale[:, :-1])
    diagonal = _diagonal_blocks(diagonal.reshape(2, nz, nc))
    lower = _diagonal_blocks(torch.cat((zeros, lower), 1).reshape(2, nz, nc))
    upper = _diagonal_blocks(torch.cat((upper, zeros), 1).reshape(2, nz, nc))

    bg = torch.stack((torch.full_like(ra[4][-1], p.par["visc_arnr"]), ra[4][-1], ra[5][-1])).reshape(3, nc)
    postvalues = post.reshape(3, nz, nc).permute(0, 2, 1)
    weight = torch.where(postvalues > bg[..., None], 1., torch.where(postvalues < bg[..., None], 0., .5))
    within = torch.arange(1, nz + 1, device=Y.device)[None, :] < KB[:, None]
    combined = torch.where(within[None, :, :, None], weight[..., None] * dbmix, torch.zeros_like(dbmix))
    combined = torch.cat((combined.new_zeros((3, nc, 1, 3)), combined[:, :, :nz - 1]), dim=2)
    combined = combined * cmask.reshape(nz, nc).T[None, :, :, None]
    dK = torch.stack((combined[2], combined[1]))
    ghat = ghat * cmask
    flux_factor = (Y[:, :-1] - Y[:, 1:]) / p.dc[None, :, None, None] + ghat[None, :-1] * aux["sf"][:, None]
    flux_factor = (flux_factor * p.pair[None]).reshape(2, nz - 1, nc).permute(0, 2, 1)
    inner = flux_factor[..., None] * dK[:, :, 1:]
    padj = inner.new_zeros((2, nc, 1, 3))
    flux = torch.cat((padj, inner, padj), dim=2)
    div = p.dt * (flux[:, :, 1:] - flux[:, :, :-1])
    div = div * p.inv_volume.reshape(nz, nc).T[None, :, :, None]
    div = div / p.scale.reshape(2, nz)[:, None, :, None]
    u = div.permute(1, 0, 2, 3).reshape(nc, 2 * nz, 3)

    # Outside the BL, one interface coefficient reads only its own two levels.
    # Keep all four tracer-to-tracer entries of this local flux derivative.
    interface_level = torch.arange(1, nz, device=Y.device)[None, :]
    local = (interface_level >= KB[:, None]) & (interface_level < km[:, None])
    tracer_amplitude = amplitudes[[2, 1]]
    ff = flux_factor.permute(0, 2, 1) * tracer_amplitude[:, None, None] * local.T[None]
    flux_top, flux_bottom = ff[:, None] * top[None], ff[:, None] * bottom[None]
    zero_block = ff.new_zeros((2, 2, 1, nc))
    diag_extra = torch.cat((flux_top, zero_block), dim=2) - torch.cat((zero_block, flux_bottom), dim=2)
    lower_extra = torch.cat((zero_block, -flux_top), dim=2)
    upper_extra = torch.cat((flux_bottom, zero_block), dim=2)
    volume = p.inv_volume.reshape(nz, nc)[None, None]
    scaling = p.scale.reshape(2, nz)[:, None, :, None]
    diagonal = diagonal + ((p.dt * diag_extra) * volume / scaling).permute(2, 0, 1, 3)
    lower = lower + ((p.dt * lower_extra) * volume / scaling).permute(2, 0, 1, 3)
    upper = upper + ((p.dt * upper_extra) * volume / scaling).permute(2, 0, 1, 3)
    return lower.contiguous(), diagonal.contiguous(), upper.contiguous(), u.contiguous(), vt.contiguous()


@torch.compile(fullgraph=True, backend="inductor", dynamic=False)
def _linearize_program(problem, z, aux):
    return _linearize_eager(problem, z, aux)


def linearize(problem, z, aux=None) -> ColumnLinearization:
    """Assemble the complete scaled F_z with O(nz) storage per column."""
    aux = problem.aux if aux is None else aux
    with torch.no_grad(), dispatch_scope("kpp_blmix_cuda"):
        values = (_linearize_program(problem, z, aux) if problem.compiled
                  else _linearize_eager(problem, z, aux))
    return ColumnLinearization(*values)


def _as_rhs(value, nz, nc):
    vector = value.ndim == 2
    if vector:
        value = value[..., None]
    if value.ndim != 3 or value.shape[:2] != (nc, 2 * nz):
        raise ValueError("Packed column operands require (ncolumn, 2*nz[, nrhs])")
    return value, vector


def _to_blocks(value, nz):
    nc, _, nrhs = value.shape
    return value.reshape(nc, 2, nz, nrhs).permute(2, 1, 3, 0).contiguous()


def _from_blocks(value):
    nz, _, nrhs, nc = value.shape
    return value.permute(3, 1, 0, 2).reshape(nc, 2 * nz, nrhs).contiguous()


def factor(linearization: ColumnLinearization, *, backend: str = "torch") -> ColumnFactors:
    """Factor B, compute W=B^-1 U, and factor I+V^T W once."""
    a = linearization
    with torch.no_grad():
        bf = blocks.factor(a.lower, a.diagonal, a.upper, backend=backend)
        w = _from_blocks(blocks.apply_factors(bf, _to_blocks(a.u, a.nz)))
        small = torch.eye(3, dtype=a.u.dtype, device=a.u.device)[None] + a.vt @ w
        valid = (bf.info == 0) & torch.isfinite(w).all(dim=(1, 2)) & torch.isfinite(small).all(dim=(1, 2))
        identity = torch.eye(3, dtype=small.dtype, device=small.device)[None]
        lu, pivots, info = torch.linalg.lu_factor_ex(torch.where(valid[:, None, None], small, identity),
                                                    check_errors=False)
        info = torch.where(valid, info, torch.full_like(info, -1))
    return ColumnFactors(a, bf, w, lu, pivots, info)


def solve(factors: ColumnFactors, rhs, *, transpose: bool = False):
    """Apply the same factors to one or many packed RHS.

    The transposed Woodbury form needs only one block transpose solve:
    B^-T [r - V (I+V^T W)^-T W^T r].
    """
    a = factors.linearization
    r, vector = _as_rhs(rhs, a.nz, a.nc)
    with torch.no_grad():
        if transpose:
            small_rhs = factors.w.transpose(1, 2) @ r
            small = torch.linalg.lu_solve(factors.small_lu, factors.small_pivots, small_rhs, adjoint=True)
            corrected = r - a.vt.transpose(1, 2) @ small
            result = _from_blocks(blocks.apply_factors(factors.block, _to_blocks(corrected, a.nz), transpose=True))
        else:
            base = _from_blocks(blocks.apply_factors(factors.block, _to_blocks(r, a.nz)))
            small = torch.linalg.lu_solve(factors.small_lu, factors.small_pivots, a.vt @ base)
            result = base - factors.w @ small
        for indices, lu, pivots in factors.dense_factors:
            selected = torch.linalg.lu_solve(lu, pivots, r.index_select(0, indices), adjoint=transpose)
            result = result.index_copy(0, indices, selected)
    return result[..., 0] if vector else result


def matvec(linearization: ColumnLinearization, value, *, transpose: bool = False):
    a = linearization
    x, vector = _as_rhs(value, a.nz, a.nc)
    xb = _to_blocks(x, a.nz)
    result = _from_blocks(blocks.matvec(a.lower, a.diagonal, a.upper, xb, transpose=transpose))
    if transpose:
        result = result + a.vt.transpose(1, 2) @ (a.u.transpose(1, 2) @ x)
    else:
        result = result + a.u @ (a.vt @ x)
    for indices, dense in a.dense_overrides:
        matrix = dense.transpose(1, 2) if transpose else dense
        result = result.index_copy(0, indices, matrix @ x.index_select(0, indices))
    return result[..., 0] if vector else result


def _relative_residual(error, rhs):
    magnitude = rhs.abs().amax(dim=(1, 2))
    absolute = error.abs().amax(dim=(1, 2))
    relative = absolute / torch.where(magnitude > 0, magnitude, 1.)
    return torch.where((magnitude == 0) & (absolute != 0), torch.full_like(relative, torch.inf), relative)


def _solve_checked_device(a, rhs, transpose, backend, tolerance):
    """Factor, solve, and check the complete equation without host decisions."""
    f = factor(a, backend=backend)
    x = solve(f, rhs, transpose=transpose)
    r, _ = _as_rhs(rhs, a.nz, a.nc)
    xx, _ = _as_rhs(x, a.nz, a.nc)
    error = matvec(a, xx, transpose=transpose) - r
    relative = _relative_residual(error, r)
    failed = (f.info != 0) | ~torch.isfinite(xx).all(dim=(1, 2)) | ~torch.isfinite(relative) | (relative > tolerance)
    summary = torch.stack((failed.sum().to(relative.dtype),
                           torch.nan_to_num(relative, nan=torch.inf, posinf=torch.inf).amax()))
    return x, f, failed, summary


@torch.compile(fullgraph=True, backend="inductor", dynamic=False)
def _solve_checked_program(a, rhs, transpose, backend, tolerance):
    return _solve_checked_device(a, rhs, transpose, backend, tolerance)


def solve_checked(problem, z, rhs, aux=None, *, transpose: bool = False,
                  backend: str = "torch", tolerance: float = 1e-11):
    """Solve the exact compact equation and check every physical column.

    Compiled problems also compile the full device solve and equation check.
    The normal path reads one two-number summary on the host. Only failed
    columns materialize the original dense Jacobian, in bounded chunks; those
    matrices and LU factors belong solely to this returned factor object.
    """
    aux = problem.aux if aux is None else aux
    a = linearize(problem, z, aux)
    if getattr(problem, "compiled", False):
        if backend == "cuda":
            # Register native custom operators and their Fake implementations
            # before entering the fixed compiled program.
            from locomgcm.solvers import _block_tridiag_triton  # noqa: F401
        x, f, failed, summary = _solve_checked_program(a, rhs, transpose, backend, tolerance)
    else:
        x, f, failed, summary = _solve_checked_device(a, rhs, transpose, backend, tolerance)
    summary = summary.cpu().tolist()
    record = dict(backend=backend, columns=a.nc, rank_bound=3, dense_fallback_columns=int(summary[0]),
                  max_relative_residual=float(summary[1]), relative_residual=float(summary[1]))
    if not int(summary[0]):
        return x, f, record
    r, vector = _as_rhs(rhs, a.nz, a.nc)
    xx, _ = _as_rhs(x, a.nz, a.nc)
    indices = torch.nonzero(failed).flatten()
    with torch.no_grad(), dispatch_scope("kpp_blmix_cuda"):
        for selected in indices.split(64):
            sub = problem.subset(selected, aux)
            dense = sub.analytic_jacobian(z.index_select(0, selected), sub.aux)
            lu, pivots, info = torch.linalg.lu_factor_ex(dense, check_errors=False)
            sub_rhs = r.index_select(0, selected)
            solved = torch.linalg.lu_solve(lu, pivots, sub_rhs, adjoint=transpose)
            operator = dense.transpose(1, 2) if transpose else dense
            residual = _relative_residual(operator @ solved - sub_rhs, sub_rhs)
            # The dense fallback is the pre-compact solve and keeps its acceptance:
            # Newton and polish steps are governed by the nonlinear root residual,
            # the pullback by the global ADJOINT_RESIDUAL check of the custom
            # Function. A per-column bound at ``tolerance`` here rejected finite LU
            # solutions of ill-conditioned columns (2e-11 to 5e-11 at 4D-Var trial
            # states) that the dense-only solver had always accepted.
            record["dense_max_relative_residual"] = max(
                record.get("dense_max_relative_residual", 0.0),
                float(torch.nan_to_num(residual, nan=torch.inf).amax()))
            bad = (info != 0) | ~torch.isfinite(solved).all(dim=(1, 2)) | ~torch.isfinite(residual)
            if bool(bad.any()):
                columns = selected[bad].cpu().tolist()
                raise LinearSolveFailure(dict(record, status="DENSE_RESIDUAL_FAILED", failed_columns=columns,
                                              tolerance=tolerance))
            xx = xx.index_copy(0, selected, solved)
            a.dense_overrides.append((selected, dense))
            f.dense_factors.append((selected, lu, pivots))
    final_error = matvec(a, xx, transpose=transpose) - r
    record["max_relative_residual"] = float(_relative_residual(final_error, r).amax())
    record["relative_residual"] = record["max_relative_residual"]
    return (xx[..., 0] if vector else xx), f, record
