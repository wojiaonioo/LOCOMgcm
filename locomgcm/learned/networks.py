"""Column networks of the learned closures.

Two per-column networks:

* ``ColumnMLP`` (first version): the interface features and column scalars of
  one water column enter as one flat vector and a fully connected SiLU network
  maps it to the closure's raw outputs.  It acts on the last dimension only, so
  every leading index (batch, j, i) is an independent column; its input and
  output sizes fix the number of interfaces.
* ``ColumnConv`` (second version): a one-dimensional convolution along the
  interfaces of each column.  No parameter depends on the number of
  interfaces, so one set of weights applies to any vertical grid.
"""
from __future__ import annotations

import torch
from torch import nn

# Scale applied to the default initialisation of the output weights so that the
# untrained network returns values close to its output bias; the closure sets
# that bias to the pre-activation of its background state.
_OUT_WEIGHT_SCALE = 1.0e-2


class ColumnMLP(nn.Module):
    """``depth`` hidden layers ``Linear(., hidden) + SiLU`` and a linear output layer.

    Parameters are float64.  The output layer's default weights are multiplied
    by ``1e-2`` and its bias is zero at construction, so the initial output is
    close to zero for inputs of order one.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, depth: int = 3):
        super().__init__()
        if depth < 1:
            raise ValueError(f"ColumnMLP needs depth >= 1 hidden layer, got {depth}")
        layers: list[nn.Module] = []
        width = int(in_dim)
        for _ in range(depth):
            layers.append(nn.Linear(width, hidden, dtype=torch.float64))
            layers.append(nn.SiLU())
            width = hidden
        self.hidden = nn.Sequential(*layers)
        self.out = nn.Linear(width, int(out_dim), dtype=torch.float64)
        with torch.no_grad():
            self.out.weight.mul_(_OUT_WEIGHT_SCALE)
            self.out.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.hidden(x))


class ColumnConv(nn.Module):
    """Vertical convolution over the interfaces of every column, plus a pooled scalar head.

    ``forward(x)`` takes ``x`` (N, in_channels, n): N independent columns, n
    interfaces each, any n >= 1.  ``depth`` hidden layers
    ``Conv1d(., hidden, kernel_size, dilation=dilation_base**i) + SiLU`` keep
    the length n (zero padding of ``dilation * (kernel_size - 1) // 2`` on each
    side); a pointwise ``Conv1d(hidden, out_channels, 1)`` then gives the
    per-interface output (N, out_channels, n).  The scalar head is
    ``Linear(hidden, scalar_dim)`` of the last hidden layer averaged over the
    column with weights ``pool_weight`` (N, n), non-negative; ``None`` weighs
    every kept interface equally.  A column whose weights are all 0 pools to 0.

    ``mask`` (N, n) bool marks the interfaces that exist (wet).  The input and
    every hidden layer are set to 0 at the other positions, so they act like
    the zero padding past the end of a column and the output at kept
    interfaces does not depend on how many masked positions follow them.

    Parameters are float64.  As in ``ColumnMLP``, the weights of the output
    convolution and of the scalar head are the default initialisation times
    ``1e-2`` and their biases are zero at construction.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scalar_dim: int,
        hidden: int = 64,
        depth: int = 4,
        kernel_size: int = 5,
        dilation_base: int = 1,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"ColumnConv needs depth >= 1 hidden layer, got {depth}")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"ColumnConv needs an odd kernel_size, got {kernel_size}")
        if dilation_base < 1:
            raise ValueError(f"ColumnConv needs dilation_base >= 1, got {dilation_base}")
        layers = []
        width = int(in_channels)
        for i in range(depth):
            dilation = int(dilation_base) ** i
            layers.append(nn.Conv1d(width, hidden, kernel_size, dilation=dilation,
                                    padding=dilation * (kernel_size - 1) // 2,
                                    dtype=torch.float64))
            width = hidden
        self.convs = nn.ModuleList(layers)
        self.act = nn.SiLU()
        self.out = nn.Conv1d(width, int(out_channels), 1, dtype=torch.float64)
        self.head = nn.Linear(width, int(scalar_dim), dtype=torch.float64)
        with torch.no_grad():
            for layer in (self.out, self.head):
                layer.weight.mul_(_OUT_WEIGHT_SCALE)
                layer.bias.zero_()

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        pool_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keep = None if mask is None else mask.unsqueeze(1)
        h = x if keep is None else torch.where(keep, x, torch.zeros_like(x))
        for conv in self.convs:
            h = self.act(conv(h))
            if keep is not None:
                h = torch.where(keep, h, torch.zeros_like(h))
        if pool_weight is None:
            pool_weight = (torch.ones_like(h[:, 0]) if mask is None
                           else mask.to(h.dtype))
        total = pool_weight.sum(dim=-1, keepdim=True)
        # The guard only matters for a column without any weight, whose
        # weighted sum is 0 as well.
        pooled = (h * pool_weight.unsqueeze(1)).sum(dim=-1) / total.clamp_min(1.0e-300)
        return self.out(h), self.head(pooled)
