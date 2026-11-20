"""GM/Redi isoneutral mixing (MITgcm ``pkg/gmredi``), ported as tensor kernels.

Conventions shared by every module of this package:

- Horizontal arrays carry the same halo of width ``ol`` on all four sides, as
  MITgcm's ``(1-OLx:sNx+OLx, 1-OLy:sNy+OLy)``: array index ``p`` along x holds
  MIT index ``i = p + 1 - ol``.  U and V points have the shape of C points;
  U point ``i`` is the west face of cell ``i``, V point ``j`` its south face.
- Level ``k`` (MIT, 1-based) is array row ``k-1``.  A W-point array row
  ``k-1`` is the interface above cell ``k``; row 0 is the surface.
- Vertical 1-D metrics are passed as ``[Nr, 1, 1]`` columns.
- Signs are MITgcm's z-coordinate ones (``rkSign = gravitySign = -1``):
  ``sigmaR`` is negative in stable stratification, vertical fluxes and the
  eddy-induced ``w`` are positive upward (+r).  LOCOMgcm's downward-positive
  ``w`` and depth fields are converted by the caller.
- Masks are 0/1 tensors of the field dtype.  No kernel exchanges halos,
  reads configuration or keeps state; the caller supplies valid halos.
- Where MITgcm divides a constant by a field, the kernels divide by the
  constant as a tensor (``field.new_tensor(c) / field``): torch evaluates
  ``float / Tensor`` as ``reciprocal() * float``, which differs from the
  division in the last bit.
"""
