"""Differentiable diagnostic reductions with an explicit compilation boundary."""

import torch


def _tracer_integrals(temp, salt, layer_thickness, cell_area, hfac):
    cell_volume = layer_thickness * cell_area
    if hfac is not None:
        cell_volume = cell_volume * hfac
    return (temp * cell_volume).sum(), (salt * cell_volume).sum()


_integral_region = torch.compiler.nested_compile_region(_tracer_integrals)


def tracer_integrals(
    temp: torch.Tensor,
    salt: torch.Tensor,
    layer_thickness: torch.Tensor,
    cell_area: torch.Tensor,
    hfac: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate local tracer fields with broadcastable geometry factors.

    The sum includes batch dimensions and performs no inter-rank reduction.
    The caller supplies wet-cell fractions. Multiplication order, promotion
    and derivatives remain those of the native PyTorch expression.

    Volume construction and both sums form one compiled region. This keeps
    diagnostic reductions out of the surrounding physics kernels without
    materializing a full three-dimensional volume at the region boundary.

    PyTorch 2.9 cannot transform invoke_subgraph with torch.func. Apply those
    transforms to the same plain expression; ordinary Inductor forward and
    autograd use the region. Dynamo understands this capability predicate and
    resolves it while tracing, before generating any device code.
    For an interpretable compiled backward use the aot_eager backend; the
    Dynamo-only eager backend cannot differentiate this PyTorch 2.9 operator.
    """
    if torch._C._are_functorch_transforms_active():
        return _tracer_integrals(temp, salt, layer_thickness, cell_area, hfac)
    return _integral_region(temp, salt, layer_thickness, cell_area, hfac)
