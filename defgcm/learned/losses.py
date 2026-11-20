"""Training losses of the learned KPP closure."""
from __future__ import annotations

from collections.abc import Mapping

import torch

_LOG_OFFSET = 1.0e-8  # m^2/s, added to both diffusivities before the log


def trajectory_loss(
    pred: Mapping[str, torch.Tensor],
    truth: Mapping[str, torch.Tensor],
    *,
    volume_weight: torch.Tensor | Mapping[str, torch.Tensor],
    sigma: Mapping[str, float | torch.Tensor],
    var_weight: Mapping[str, float],
) -> torch.Tensor:
    """Volume-weighted squared error of the model trajectory against the truth.

    ``L = sum_v w_v * sum((pred_v - truth_v)^2 * vol_v) / (sigma_v^2 * sum(vol_v))``

    over the variables named in ``var_weight`` (keys ``temp``, ``salt``, ``u``,
    ``v`` of ``pred`` and ``truth``).  Each ``pred_v``/``truth_v`` may carry any
    leading dimensions (time, batch) in front of its (nz, ny, nx[+1]) grid;
    both sums run over all of them, so each term is the volume-weighted mean
    squared error divided by sigma_v^2.  ``volume_weight`` is the wet-cell
    volume (nz, ny, nx), or a mapping from variable name to its own volume,
    which u (nz, ny, nx+1) and v (nz, ny+1, nx) need because they live on faces.
    """
    if not var_weight:
        raise ValueError("var_weight names no variable")
    total = None
    for name, weight in var_weight.items():
        err = pred[name] - truth[name]
        vol = volume_weight[name] if isinstance(volume_weight, Mapping) else volume_weight
        vol = vol.to(dtype=err.dtype, device=err.device).expand_as(err)
        term = weight * (err * err * vol).sum() / (sigma[name] ** 2 * vol.sum())
        total = term if total is None else total + term
    return total


def log_k_loss(
    pred_diff: torch.Tensor,
    truth_diff: torch.Tensor,
    wet_iface_k: torch.Tensor,
) -> torch.Tensor:
    """Masked mean of ``(log(pred + 1e-8) - log(truth + 1e-8))^2`` on interior interfaces.

    ``pred_diff``/``truth_diff`` (B, nz, ny, nx) use the cell-centred interface
    convention of ``KppFields.diff_kzt`` (index k = top interface of layer k);
    ``wet_iface_k`` (nz, ny, nx) is the interface wet mask in the same
    convention.  Index 0 (sea surface) is excluded, so only interfaces
    1..nz-1 enter the mean.
    """
    pred = pred_diff[..., 1:, :, :]
    truth = truth_diff[..., 1:, :, :]
    mask = (wet_iface_k[..., 1:, :, :] > 0).expand_as(pred)
    one = torch.ones_like(pred)
    # Select before the log so masked entries never evaluate log of a
    # non-positive argument (forward or backward).
    log_pred = torch.log(torch.where(mask, pred + _LOG_OFFSET, one))
    log_truth = torch.log(torch.where(mask, truth + _LOG_OFFSET, one))
    diff = log_pred - log_truth
    count = mask.sum().clamp_min(1).to(pred.dtype)
    return (diff * diff).sum() / count
