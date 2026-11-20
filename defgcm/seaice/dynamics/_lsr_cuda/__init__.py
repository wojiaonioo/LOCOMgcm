"""CUDA kernels for the sea-ice LSR line recurrence and the ordered residual sum.

``dispatcher`` owns the accelerator gates (``seaice_lsr_cuda``,
``seaice_ordered_sum_cuda``) and their ledger records; ``_fast`` loads the
extension and launches it without any gate or fallback.
"""
