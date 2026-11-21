"""A field that carries its halo in place, refreshed without reallocating.

:func:`~locomgcm.geometry.exchange.exchange` returns a NEW padded tensor
every call: it builds each axis with ``torch.cat``, so a caller that exchanges
the same field every iteration pays, per iteration, two concatenations plus the
copy that put the interior there.  In a captured CUDA graph replaying a
Krylov iteration that is pure overhead -- the buffer's address is fixed for the
life of the graph, the interior is written by a kernel that could just as
easily write into the middle of a larger allocation, and the only thing that
actually has to happen each iteration is that the border cells take their new
values.

:class:`PaddedField` is that layout: one allocation ``[..., ny+2h, nx+2h]``,
an ``interior`` view for producers and consumers, and :meth:`refresh` to bring
the border up to date in place.

**It is not a second exchange implementation.**  ``refresh`` calls the same
spec builder, the same packed transport and the same receive-store that
``exchange`` calls (``_distributed_halo_specs`` / ``_packed_halo_round`` /
``_store_received_halos``), and then writes the resulting strips into border
slots instead of handing them to ``_assemble_padded_axis``'s ``torch.cat``.
Every rule about which cells a border takes -- physical-edge fills, WRAP, the
C-grid seam skip, bool marshalling, peer resolution -- therefore lives in one
place.

**Axis order is a data dependency, not a convention.**  x is refreshed first
and the y strips are then cut from the ALREADY-x-padded rows, so a corner cell
reaches this rank through its y-neighbour carrying the x-ghost values that
neighbour had already received.  ``distributed/halo_epoch.py``'s "Stages"
section is the long form of why that cannot be reordered or fused; the same
reason makes ``refresh`` write x into the buffer before it plans y.

Nothing here is specific to one solver: the interface names a field, a grid, a
location and a fill, which is what ``exchange`` names.  CG3D's captured graph
is the first caller; the CG2D graph has the same shape of problem.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from locomgcm.distributed.comm_adjoint import session_for_grid
from locomgcm.distributed.halo_p2p import halo_p2p
from locomgcm.geometry.exchange import (
    EXCHANGE_PACKED,
    CGridLocation,
    FillSpec,
    ObcEdges,
    _canonical_axis,
    _distributed_halo_specs,
    _distributed_p2p_rounds,
    _ghost_cells,
    _has_distributed_transport,
    _packed_halo_round,
    _store_received_halos,
)
from locomgcm.geometry.grid import BoundaryType, Grid


@dataclass(frozen=True)
class PaddedField:
    """A tensor plus its halo, in one allocation whose address never moves.

    ``buffer`` is the whole thing, ``[..., ny + 2*halo, nx + 2*halo]``.
    ``interior`` is the view a producer writes and a consumer reads as the
    unhaloed field; it is a genuine view, so writing it writes the buffer.

    Frozen because the identity of the buffer IS the point -- a graph captures
    its address.  The tensors it holds are mutated in place; the dataclass just
    refuses to let the binding be swapped for a different allocation.
    """

    buffer: torch.Tensor
    halo: int

    @property
    def interior(self) -> torch.Tensor:
        h = self.halo
        return self.buffer[..., h:-h, h:-h] if h else self.buffer

    @classmethod
    def alloc(cls, shape, *, halo: int = 1, dtype=None, device=None) -> "PaddedField":
        """Allocate a zeroed padded buffer for a field of ``shape``.

        Zero, not empty, so the border is defined before the first
        :meth:`refresh` -- a reader between allocation and refresh sees zeros
        rather than whatever the allocator held.  Refresh itself writes every
        border it produces a strip for, physical edges included.
        """
        if halo < 0:
            raise ValueError(f"halo must be non-negative, got {halo}")
        shape = tuple(int(extent) for extent in shape)
        if len(shape) < 2:
            raise ValueError("a padded field needs at least two spatial axes")
        padded = (*shape[:-2], shape[-2] + 2 * halo, shape[-1] + 2 * halo)
        return cls(torch.zeros(padded, dtype=dtype, device=device), halo)

    @classmethod
    def like(cls, field: torch.Tensor, *, halo: int = 1) -> "PaddedField":
        """Allocate for ``field`` and copy it into the interior."""
        padded = cls.alloc(field.shape, halo=halo, dtype=field.dtype,
                           device=field.device)
        padded.interior.copy_(field)
        return padded

    def refresh(
        self,
        *,
        grid: Grid,
        location: CGridLocation = "T",
        fill: FillSpec = FillSpec.ZERO,
        obc: ObcEdges | None = None,
    ) -> "PaddedField":
        """Bring the border up to date from the current interior, in place.

        Returns ``self`` so a caller can chain; the buffer is the same object.
        Every border for which a strip is produced is written -- including
        physical edges, which produce a strip like any other side.

        Refuses a grid carrying a communication session for the same reason
        ``_pad_axis_distributed`` does (``exchange.py:1426-1429``): the packed
        byte views erase the floating-point graph at the sender, so a
        differentiable exchange needs the typed transport and cannot be served
        here.  Refusing is the honest answer; running anyway would silently
        detach.
        """
        if self.halo == 0:
            return self
        if session_for_grid(grid, None) is not None:
            raise NotImplementedError(
                "differentiable exchange requires the typed transport; "
                "PaddedField.refresh cannot serve a grid with a comm session"
            )
        h = self.halo
        # x first, then y from the x-padded rows: see the module docstring.
        self._refresh_axis(
            axis=-1, grid=grid, location=location, fill=fill, obc=obc,
            source=self.interior,
            before_side="west", after_side="east",
            boundary_type=grid.boundary_topology.west_east,
            rows=slice(h, -h),
        )
        self._refresh_axis(
            axis=-2, grid=grid, location=location, fill=fill, obc=obc,
            source=self.buffer[..., h:-h, :],
            before_side="south", after_side="north",
            boundary_type=grid.boundary_topology.south_north,
            rows=None,
            x_extent=(h, h),
            x_boundary_type=grid.boundary_topology.west_east,
        )
        return self

    # -- one axis ----------------------------------------------------------

    @staticmethod
    def _typed_halo_round(specs) -> list[torch.Tensor]:
        """The per-round transport, i.e. what ``exchange`` runs by default.

        ``EXCHANGE_PACKED`` is off in production, and the two transports do not
        merely differ in speed: the per-round path issues one ``halo_p2p`` per
        ``(wrap, position)`` so a tagless transport cannot pair a peer's two
        messages the wrong way, while the packed path concatenates them and
        removes the ambiguity instead.  A captured graph must contain the
        communication sequence the shipping path would run, so ``refresh``
        follows the same switch.

        The rounds, their order and their contents are taken from
        ``_distributed_p2p_rounds`` -- the same function ``exchange`` uses --
        so the two cannot disagree about how many rounds there are.
        """
        received = [recv for _position, recv, _tag, _peer, _is_wrap
                    in specs.recv_specs]
        for wrap_round, position_round in _distributed_p2p_rounds(
            specs.recv_specs, specs.send_specs
        ):
            indices = [
                index
                for index, (position, _recv, _tag, _peer, is_wrap)
                in enumerate(specs.recv_specs)
                if is_wrap == wrap_round and position == position_round
            ]
            sends = [spec for spec in specs.send_specs
                     if spec[3] == wrap_round and spec[4] == position_round]
            got = halo_p2p(
                [send for _peer, send, _tag, _is_wrap, _position in sends],
                [specs.recv_specs[index][1] for index in indices],
                [peer for peer, _send, _tag, _is_wrap, _position in sends],
                [specs.recv_specs[index][3] for index in indices],
                [tag for _peer, _send, tag, _is_wrap, _position in sends],
                [specs.recv_specs[index][2] for index in indices],
            )
            for index, recv in zip(indices, got):
                received[index] = recv
        return received

    def _slots(self, axis: int, rows):
        """The ``(before, after)`` border views this axis writes."""
        h = self.halo
        if axis == -1:
            base = self.buffer if rows is None else self.buffer[..., rows, :]
            return base[..., :h], base[..., -h:]
        base = self.buffer if rows is None else self.buffer[..., rows, :]
        return base[..., :h, :], base[..., -h:, :]

    def _refresh_axis(
        self, *, axis: int, grid, location, fill, obc, source,
        before_side, after_side, boundary_type, rows,
        x_extent: tuple[int, int] = (0, 0),
        x_boundary_type: BoundaryType | None = None,
    ) -> None:
        h = self.halo
        before_slot, after_slot = self._slots(axis, rows)
        # ``exchange`` canonicalises the axis at every entry point
        # (``exchange.py:1365/1448/1855``) and some fills depend on it:
        # ``_obc_ghost`` decides whether to widen a strip by comparing
        # ``axis == tensor.ndim - 2``, which a negative axis never satisfies,
        # so ``FillSpec.OBC_VALUE`` on y raises rather than filling.  This
        # class advertises the same fills ``exchange`` takes, so it has to
        # normalise the same way.
        axis = _canonical_axis(axis, source.ndim)
        if _has_distributed_transport(grid, (before_side, after_side)):
            specs = _distributed_halo_specs(
                source, grid=grid, location=location, axis=axis,
                before=h, after=h,
                before_side=before_side, after_side=after_side,
                boundary_type=boundary_type, fill=fill, obc=obc,
                x_extent=x_extent, x_boundary_type=x_boundary_type,
            )
            if specs.send_specs or specs.recv_specs:
                if EXCHANGE_PACKED:
                    sends = [(peer, (tag,), strip)
                             for peer, strip, tag, _is_wrap, _pos
                             in specs.send_specs]
                    recvs = [(peer, (tag,), template)
                             for _pos, template, tag, peer, _is_wrap
                             in specs.recv_specs]
                    received = _packed_halo_round(sends, recvs)
                else:
                    received = self._typed_halo_round(specs)
                _store_received_halos(specs, received)
            parts = specs.local_parts
        else:
            # ``_pad_axis`` is ``_ghost_cells(before) + tensor +
            # _ghost_cells(after)`` concatenated.  Calling ``_ghost_cells``
            # directly takes the same strips from the same authority without
            # building the concatenation this class exists to avoid.
            parts = {
                position: _ghost_cells(
                    source, grid=grid, location=location,
                    axis=axis, width=h, side=position,
                    boundary_side=side, boundary_type=boundary_type,
                    fill=fill, obc=obc, x_extent=x_extent,
                    x_boundary_type=x_boundary_type,
                )
                for position, side in (("before", before_side),
                                       ("after", after_side))
            }

        # Every side that produced a strip is written, physical edges included:
        # a ZERO edge produces a zero strip and gets it copied like any other.
        before_part = parts.get("before")
        if before_part is not None:
            before_slot.copy_(before_part)
        after_part = parts.get("after")
        if after_part is not None:
            after_slot.copy_(after_part)


__all__ = ["PaddedField"]
