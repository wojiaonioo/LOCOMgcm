"""Host-side constant sequences -> device tensors, opaque to ``torch.compile``.

``Tensor.new_tensor(<sequence of Python floats>)`` is the natural way to lift a
per-latitude profile onto the device, and it is what several dynamics modules
did inline.  It is also untraceable: under Dynamo the ny-or-ny+1 Python floats
are lifted to 0-d fake tensors first, so ``new_tensor`` receives a *sequence of
tensors* and dies with

    RuntimeError: The tensor has a non-zero number of elements, but its data is
    not allocated yet.

which is precisely the single blocker that made ``compile=model_step`` fail on
step 1.  PyTorch's own error text prescribes
the remedy used here: make the construction opaque to the tracer.

One helper rather than one fix per site, because the three call sites are the
same bug and a second copy of the rule is a copy that drifts.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


@torch._dynamo.disable
def host_profile_to_tensor(
    profile: Sequence[float],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build a 1-D tensor from a host-side float sequence.

    Exactly ``ref.new_tensor(profile)`` for a reference tensor of the given
    dtype/device -- same values through the same conversion, no arithmetic --
    so it is bitwise-identical to the inline form it replaces.
    """
    return torch.tensor(profile, dtype=dtype, device=device)
