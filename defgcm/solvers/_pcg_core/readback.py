"""Completion-owned host diagnostics for the graphed CG2D solve."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PcgReadback:
    """One solve's residual and integer iteration/status lanes, copied together.

    The source and pinned destination belong to this result, not the reusable
    graph context.  A subsequent solve cannot overwrite a pending result.
    """

    source: torch.Tensor
    host: torch.Tensor
    event: torch.cuda.Event

    def numbers(self) -> tuple[int, float, int]:
        self.event.synchronize()
        integers = self.host[1:].view(torch.int64)
        return int(integers[0]), float(self.host[0]), int(integers[1])


def read_pcg_stats(
    num_iters: int, residual: float | PcgReadback,
) -> tuple[int, float]:
    """Resolve graphed diagnostics; eager solves already return Python numbers."""
    if isinstance(residual, PcgReadback):
        num_iters, residual, _status = residual.numbers()
    return num_iters, residual


def copy_pcg_stats(
    residual: torch.Tensor, num_iters: int, status: int,
) -> PcgReadback:
    """Enqueue one D2H copy without waiting for the residual's final kernels."""
    # Form sqrt in the solve dtype before this exact widening to float64.
    # Integer lanes are bit views, so even large iteration counts stay exact.
    source = torch.empty(3, dtype=torch.float64, device=residual.device)
    source[0].copy_(residual)
    integers = source[1:].view(torch.int64)
    integers[0].fill_(num_iters)
    integers[1].fill_(status)
    host = torch.empty(3, dtype=torch.float64, device="cpu", pin_memory=True)
    host.copy_(source, non_blocking=True)
    event = torch.cuda.Event()
    event.record(torch.cuda.current_stream(residual.device))
    return PcgReadback(source, host, event)
