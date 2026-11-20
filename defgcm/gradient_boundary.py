"""The cotangent contract shared by window, communication and codec edges."""

import torch


class GradientBoundary:
    """An absent consumer supplies None, never an allocated zero field."""

    @staticmethod
    def configure(ctx):
        ctx.set_materialize_grads(False)

    @staticmethod
    def roots(outputs, cotangents):
        if len(outputs) != len(cotangents):
            raise ValueError("gradient boundary output and cotangent layouts differ")
        return [(value, seed.to(value.device)) for value, seed in zip(outputs, cotangents)
                if isinstance(value, torch.Tensor) and value.requires_grad
                and seed is not None]

    @classmethod
    def backward(cls, outputs, cotangents, *, links=()):
        roots = cls.roots(outputs, cotangents) + list(links)
        if roots:
            torch.autograd.backward([value for value, _ in roots],
                                    [seed for _, seed in roots])
