"""Round each physical product before it enters a compiled sum."""
import torch


@torch.library.custom_op("defgcm::mdjwf_rounded_product", mutates_args=())
def _product(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return left * right


@_product.register_fake
def _product_fake(left, right):
    return left * right


@torch.compiler.allow_in_graph
class _RoundedProduct(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(left, right):
        return _product(left, right)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)
        ctx.save_for_forward(*inputs)

    @staticmethod
    def backward(ctx, grad):
        left, right = ctx.saved_tensors
        return grad * right, grad * left

    @staticmethod
    def jvp(ctx, grad_left, grad_right):
        left, right = ctx.saved_tensors
        return grad_left * right + grad_right * left


def rounded_multiply(left, right):
    if not torch.compiler.is_compiling():
        return left * right
    if not isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor):
            return left * right
        left = right.new_tensor(left)
    if not isinstance(right, torch.Tensor):
        right = left.new_tensor(right)
    # torch.mul accepts a zero-dimensional CPU tensor beside an accelerator
    # tensor. Preserve its gradient path before the custom op is broadcast.
    if left.device != right.device:
        if left.device.type == "cpu" and left.ndim == 0:
            left = left.to(right.device)
        elif right.device.type == "cpu" and right.ndim == 0:
            right = right.to(left.device)
        else:
            return left * right
    return _RoundedProduct.apply(left, right)


def install_product_lowering():
    """Keep the scalar instruction fusible while preserving its IEEE rounding."""
    from torch._inductor import config
    from torch._inductor.lowering import fallback_handler, make_pointwise, register_lowering
    from torch._inductor.virtualized import ops

    operator = torch.ops.defgcm.mdjwf_rounded_product.default
    separate_product = fallback_handler(operator, add_to_fallback_set=False)

    @register_lowering(operator, broadcast=True, type_promotion_kind=None)
    def lower(left, right):
        device = left.get_device().type
        dtype = left.get_dtype()
        if (device == "cuda" and torch.version.hip is None
                and dtype == right.get_dtype() and dtype in (torch.float32, torch.float64)):
            opcode, registers = (("mul.rn.f32", "=f,f,f") if dtype == torch.float32
                                 else ("mul.rn.f64", "=d,d,d"))

            def multiply_rn(x, y):
                return ops.inline_asm_elementwise(
                    x, y, asm=f"{opcode} $0, $1, $2;", constraints=registers, dtype=dtype,
                )
            return make_pointwise(multiply_rn)(left, right)
        if device == "cpu" and config.cpp.enable_floating_point_contract_flag == "off":
            return make_pointwise(ops.mul)(left, right)
        return separate_product(left, right)
