"""Vertical mixing schemes of MITgcm as tensor-only kernels.

One module per scheme; each names the options it accepts in a top-level
tuple.  The kernels take tensors and scalars only; the caller selects the
time level, computes the densities with its equation of state and decides
which kernels run.
"""
