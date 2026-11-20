"""MITgcm's per-column open-boundary guard of the sponge layers.

OBCS_SPONGE_U, _V, _T and _S relax a boundary column only where the side's OB
index table has an entry, ``IF ( OB_J*(i,bi,bj) .NE. OB_indexNone )``
(``pkg/obcs/obcs_sponge.F:67, 102, 137, 172`` for u, ``:267, 302, 337, 372`` for v,
``:467, 504, 541, 578`` for theta and ``:675, 712, 749, 786`` for salt).  Both
LOCOMgcm sponge modules take the table from the case's ``boundary.ob_index_mode`` and
``boundary.sides``, the fields the ``data.obcs`` export writes the tables from
(:func:`defgcm.boundaries.open.resolve_ob_along_masks`).
"""

from __future__ import annotations

import torch

from defgcm.accel.constant_cache import ConstantCache
from defgcm.boundaries.open import BoundarySide, ObcsAlongTables, resolve_ob_along_masks
from defgcm.geometry import StructuredCGrid


class ObIndexGate:
    """Mixin for a sponge dataclass that declares ``ob_index_mode``, ``ob_sides``,
    ``_ob_tables``, ``_ob_tables_grid`` and ``_ob_gate_cache``."""

    @staticmethod
    def _gate_axis(side: BoundarySide) -> int:
        """Tensor axis running ALONG the side (the one MIT's OB table indexes)."""
        return -2 if side in (BoundarySide.WEST, BoundarySide.EAST) else -1

    def _ob_along_tables(self, grid: StructuredCGrid) -> ObcsAlongTables:
        """The case's along-boundary tables for ``grid``, resolved once and cached.

        Both inputs to the resolver — the policy :attr:`ob_index_mode` and the side
        set :attr:`ob_sides` — come from the case config, never from ``grid``.  The
        grid supplies only ``masks.center``, and only under ``wet_only``.

        The cache is keyed on grid *identity*: a different grid object re-resolves,
        so a module reused across geometries cannot serve a stale table.

        🔴 Contract: ``grid.masks`` is treated as **immutable**.  Mutating
        ``grid.masks.center`` in place leaves the identity unchanged and this
        cache will keep serving the table built before the mutation.  That is a
        constructible defect, not an observed one — no production code path
        mutates ``masks.center`` after grid construction — and it is deliberately
        not defended against with a content version key (YAGNI); the contract is
        stated here instead so a future in-place mutator is a documented contract
        break rather than a silent wrong answer.
        """
        if self._ob_tables is None or self._ob_tables_grid is not grid:
            sides = (
                tuple(BoundarySide) if self.ob_sides is None else tuple(self.ob_sides)
            )
            self._ob_tables = resolve_ob_along_masks(
                self.ob_index_mode, grid, sides,
            )
            self._ob_tables_grid = grid
            self._ob_gate_cache = ConstantCache()
        return self._ob_tables

    def _ob_gate(
        self,
        side: BoundarySide,
        grid: StructuredCGrid,
        *,
        length: int,
        device: torch.device | str,
    ) -> torch.Tensor | None:
        """Cached along-boundary gate for one (side, axis length, device).

        🔴 :meth:`_ob_along_tables` is called **first and unconditionally**.  It
        owns the grid-identity check and clears this cache when the grid changed,
        so probing a key that deliberately does not carry the grid must never
        happen before it runs — doing so returns a gate built from the previous
        grid, which is what a module reused across two same-shaped geometries hits.
        """
        tables = self._ob_along_tables(grid)
        key = (side, int(length), str(device))
        return self._ob_gate_cache.get_or_build(
            key,
            lambda: tables.gate_for_axis(
                side, length=length, device=device,
                seam_entry=self._seam_entry(side, grid, tables, length=length),
            ),
            variant="ob_along_gate",
            site=f"{type(self).__name__}._ob_gate",
        )

    @staticmethod
    def _seam_entry(
        side: BoundarySide,
        grid: StructuredCGrid,
        tables: ObcsAlongTables,
        *,
        length: int,
    ) -> torch.Tensor | None:
        """The table entry of the column after this tile along ``side``, for the
        staggered face's last slot (see ``ObcsAlongTables.gate_for_axis``).

        ``None`` when the axis has no such slot, the side is ungated, the axis is
        not split across ranks (one tile: the slot takes the table's own first
        entry), or no rank follows this tile on that axis (the slot is the domain
        edge face).  With a distributed neighbour after this tile (east for the
        south and north tables, north for the west and east tables, the periodic
        wrap included) the entry is the neighbour's first one, received by a halo
        exchange of the table (a constant: masks carry no gradient).  Every rank
        on a split axis joins the exchange, the last one included, because it
        sends its first entry to the rank before it; the ranks along one side
        evaluate that side's sponge together, so they exchange together.
        """
        table = tables.get(side)
        if table is None or length != int(table.shape[0]) + 1:
            return None
        from defgcm.geometry.exchange import (
            FillSpec, HaloExtentV1, _has_distributed_transport, exchange,
        )

        along_x = side in (BoundarySide.SOUTH, BoundarySide.NORTH)
        axis_sides = ("west", "east") if along_x else ("south", "north")
        if not _has_distributed_transport(grid, axis_sides):
            return None
        extent = (HaloExtentV1(west_left_extent=0, east_right_extent=1,
                               south_bottom_extent=0, north_top_extent=0) if along_x
                  else HaloExtentV1(west_left_extent=0, east_right_extent=0,
                                    south_bottom_extent=0, north_top_extent=1))
        # A 1-D row is padded along the axis with non-zero extent; the table sits
        # on tracer-centre columns, hence "T".
        padded = exchange(table.to(dtype=torch.float64), grid=grid, location="T",
                          extent=extent, fill=FillSpec.REPLICATE, constant=True)
        neighbours = getattr(grid, "distributed_neighbors", None) or {}
        if neighbours.get(axis_sides[1]) is None:
            return None
        return padded[-1:] > 0.5

    def _gate_along_open_boundary(
        self,
        tendency: torch.Tensor,
        side: BoundarySide,
        grid: StructuredCGrid,
    ) -> torch.Tensor:
        """Zero the band where the case's OB index table has no open boundary.

        MITgcm guards the whole sponge body with
        ``IF ( OB_J*(i,bi,bj) .NE. OB_indexNone )`` (``obcs_sponge.F:67 / :102 /
        :137`` for u, ``:267 / :302 / :337`` for v) and gates ``gU_arr(i,j)`` and
        ``gV_arr(i,j)`` on the *same* integer ``i`` that indexes the table, so
        both faces share one along-boundary mask.  Where that table comes from is
        a case policy, not a geometric fact — see
        :func:`~defgcm.boundaries.open.resolve_ob_along_masks` and
        :attr:`ob_index_mode`.  With the ``wet_only`` policy and no gate,
        LOCOMgcm relaxes the band on columns whose boundary cell is land — a
        body force ``-w * lambda^-1 * (u - u_OB)`` that MITgcm never applies,
        injected afresh every step and proportional to the local velocity.

       The 3-D wet mask applied in :meth:`_relax_field` does NOT subsume this:
        it only asks whether the *band* face is wet, and a band face ten cells
        in from a land boundary cell usually is.

        Two structural properties this method must keep:

        * an ungated side returns ``tendency`` **unchanged, without entering**
          :func:`torch.where`.  That is what makes ``full_edge`` — the default,
          and every existing case — bit-for-bit the pre-gate behaviour by control
          flow rather than by trusting an all-true selection;
        * a gated side uses selection, never multiplication, so a non-finite
          value on the closed side cannot leak through a ``0 * NaN``
          (the explicit reconstruction-mask convention). The whole tensor is selected rather than just
          the band because the two ``_*_side_tendency`` helpers lay their band
          out differently (``_band_slice`` vs ``_legacy_edge_band_slice``);
          everything outside the band is already exactly zero, so selecting over
          it changes nothing.
        """
        dim = tendency.ndim + self._gate_axis(side)
        keep = self._ob_gate(
            side, grid, length=tendency.shape[dim], device=tendency.device,
        )
        if keep is None:
            return tendency
        view_shape = [1] * tendency.ndim
        view_shape[dim] = keep.shape[0]
        zero = torch.zeros((), dtype=tendency.dtype, device=tendency.device)
        return torch.where(keep.view(view_shape), tendency, zero)
