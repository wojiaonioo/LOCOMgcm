"""Orlanski radiation open boundary, MITgcm ``pkg/obcs/orlanski_{east,west,north,south}.F``.

One call is one ``ORLANSKI_<side>`` of one ``OBCS_CALC``: from the step-start
fields at the boundary point and its two interior neighbours, and the storage
the previous call left, it returns the boundary values for the future time and
the new storage.  The four prognostic fields (u, v, theta, salt) are stacked
on the leading axis and share every operation, so each element follows the
scalar MITgcm expression.

Tensor conventions (``n`` is the along-boundary length, ``B`` the batch):

* ``boundary``, ``first``, ``second``: [4, B, nz, n], the field at the boundary
  point ``xb`` and at the first and second interior neighbours ``x1``, ``x2``
  (``SPEC.md`` table: e.g. east u is u(I_obc), u(I_obc-1), u(I_obc-2)).
* ``spacing``: [4, 1, 1, n], the grid spacing that turns the nondimensional
  phase speed CL into CVEL [m]; ``inverse_spacing``: [4, 1, 1, n], the
  reciprocal spacing of the update [1/m].
* ``storage``: [5, 4, B, nz, n], ``STORE_1``, ``STORE_2``, ``STORE_3``,
  ``STORE_4`` and ``CVEL`` [m/s] of ``ORLANSKI.h``; zeros is ``ORLANSKI_INIT``.

Values are in the units of their field; no mask is applied here (MITgcm masks
when it applies the values, ``obcs_apply_uv.F``/``obcs_apply_ts.F``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

#: Field order of the leading axis of the stacked inputs and of ``storage[:, i]``.
ORLANSKI_FIELDS = ("u", "v", "temp", "salt")


@dataclass(frozen=True)
class OrlanskiParameters:
    """``OBCS_PARM02`` (``obcs_readparms.F:189``) plus the two PARAMS it reads.

    ``ab_eps`` is PARM03 abEps and ``delta_t`` PARM03 deltaT, both read by
    ``orlanski_*.F:91-98``; ``use_fixed_c_east``/``use_fixed_c_west`` replace
    CVEL by ``cfix`` [m/s] on that side only (north and south have no fixed
    speed in MITgcm).
    """

    cmax: float
    cvel_time_scale: float
    cfix: float
    use_fixed_c_east: bool
    use_fixed_c_west: bool
    ab_eps: float
    delta_t: float


def orlanski_side_update(
    *,
    side: str,
    boundary: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
    spacing: torch.Tensor,
    inverse_spacing: torch.Tensor,
    storage: torch.Tensor,
    params: OrlanskiParameters,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(values, storage)`` after one ``ORLANSKI_<side>`` call.

    ``side`` is ``"east"``, ``"west"``, ``"north"`` or ``"south"``; each branch
    is its MITgcm file's expression with its own operand grouping.
    """
    ab1 = 1.5 + params.ab_eps
    ab2 = -0.5 - params.ab_eps
    frac_cvel = params.delta_t / params.cvel_time_scale
    f1 = frac_cvel
    f2 = 1.0 - frac_cvel
    delta_t = params.delta_t
    store1, store2, store3, store4, cvel = storage.unbind(0)

    denom = ab1 * store2 + ab2 * store3
    nonzero = denom != 0.0
    safe_denom = torch.where(nonzero, denom, torch.ones_like(denom))
    if side in ("east", "north"):
        # orlanski_east.F:106-114, orlanski_north.F:98-106
        cl = -(first - store1) / safe_denom
    elif side in ("west", "south"):
        # orlanski_west.F:108-116, orlanski_south.F:98-106
        cl = (first - store1) / safe_denom
    else:
        raise ValueError(f"unknown Orlanski side {side!r}")
    cl = torch.where(nonzero, torch.clamp(cl, min=0.0, max=params.cmax), torch.zeros_like(cl))

    fixed = ((side == "east" and params.use_fixed_c_east)
             or (side == "west" and params.use_fixed_c_west))
    if fixed:
        # orlanski_east.F:115-118, orlanski_west.F:117-120
        cvel = torch.full_like(cvel, params.cfix)
    else:
        cvel = f1 * (cl * spacing / delta_t) + f2 * cvel

    if side == "east":
        # orlanski_east.F:124-127
        values = boundary - cvel * (delta_t * inverse_spacing) * (
            ab1 * (boundary - first) + ab2 * (store4 - store1))
        gradient = first - second
    elif side == "west":
        # orlanski_west.F:126-129
        values = boundary + cvel * (delta_t * inverse_spacing) * (
            ab1 * (first - boundary) + ab2 * (store1 - store4))
        gradient = second - first
    elif side == "north":
        # orlanski_north.F:110-113
        values = boundary - cvel * delta_t * inverse_spacing * (
            ab1 * (boundary - first) + ab2 * (store4 - store1))
        gradient = first - second
    else:
        # orlanski_south.F:110-113
        values = boundary + cvel * delta_t * inverse_spacing * (
            ab1 * (first - boundary) + ab2 * (store1 - store4))
        gradient = second - first

    # orlanski_east.F:232-261 (and the same block of the other three files):
    # STORE_3 <- STORE_2, STORE_2 <- interior difference, STORE_1 <- x1, STORE_4 <- xb.
    new_storage = torch.stack((first, gradient, store2, boundary, cvel), dim=0)
    return values, new_storage
