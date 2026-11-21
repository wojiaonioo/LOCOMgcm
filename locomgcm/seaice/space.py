"""Independent MIT tiles of one doubly periodic face, sized from SIZE.h.

Halo refreshes are explicit scientific events. No spatial operation silently
exchanges a prognostic field. The face is ``nx`` by ``ny`` cells cut into
``tile_nx`` by ``tile_ny`` tiles (MIT sNx, sNy), stored [bj, bi, j, i] with a
native halo of ``HALO`` cells on every side whatever MIT's OLx: every native
domain trim is counted from the storage edge for this halo, and interior cells
depend only on halo cells within the ported stencils' reach. Whether the face
edges couple comes from the grid's ``edge_face_couples``; only doubly periodic
faces are implemented (MIT's single-face exchange without OBCS).
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import torch

from locomgcm.accel.tensor_context import forward_mode_active
from locomgcm.geometry.tiling import CubeTileLayout, MitTiling
from .records import Domain, INTERIOR, Geometry, THSICE_FIELDS


@dataclass(frozen=True)
class ExchangeEvent:
    fields: tuple[str, ...]
    cpp_gate: str
    runtime_gate: str
    chains: str
    source: str


EXCHANGE_EVENTS = {
    "X0": ExchangeEvent(("climsst", "climsss"), "ALLOW_CLIMSST_RELAXATION / ALLOW_CLIMSSS_RELAXATION", "always", "TLHB", "exf_getclim.F:96,113"),
    "X1": ExchangeEvent(("ustress", "vstress"), "none", "not stressIsOnCgrid", "TLHB", "exf_getforcing.F:331"),
    "X2": ExchangeEvent(("Qnet", "EmPmR", "fu", "fv", "Qnet_ow", "EmPmR_ow", "fu_ow", "fv_ow"), "none", "always", "TLHB", "exf_mapfields.F:355-357"),
    "X3": ExchangeEvent(("SST", "SSS", "pLoad"), "ALLOW_CLIMSST_RELAXATION / ALLOW_CLIMSSS_RELAXATION / ATMOSPHERIC_LOADING", "always", "TLHB", "exf_mapfields.F:364-370"),
    "X4": ExchangeEvent(("uwind", "vwind"), "ALLOW_EXF", "useEXF", "TLB", "seaice_model.F:127"),
    "X5": ExchangeEvent(("uIce_fd", "vIce_fd"), "SEAICE_ALLOW_FREEDRIFT", "LSR_mixIniGuess>=1; diagnostic free drift omitted", "", "seaice_lsr.F:637"),
    "X6": ExchangeEvent(("UICE", "VICE", "uIce", "vIce"), "none", "scan executed", "LB", "seaice_lsr.F:987"),
    "X7": ExchangeEvent(("fu", "fv"), "none", "SEAICEupdateOceanStress", "TLB", "seaice_ocean_stress.F:142"),
    "X8": ExchangeEvent(("HEFF", "AREA", "HSNOW", "EmPmR", "saltFlux", "Qnet"), "ALLOW_THSICE", "not useThSice", "TL", "seaice_model.F:317-335"),
    "X9": ExchangeEvent(("iceMask", "iceHeight", "snowHeight", "Qice1", "Qice2"), "none", "useSEAICE or thSIceAdvScheme>0; stressReduction=0", "B", "thsice_do_exch.F:32-38"),
    "X10": ExchangeEvent(("theta", "salt"), "none", "not useOffLine and not implicitIntGravWave", "TLHB", "do_fields_blocking_exchanges.F:60-62"),
    "I1": ExchangeEvent(("hFacW", "hFacS", "uVel", "vVel", "etaN"), "none", "geometry or cold-start file", "TLHB", "ini_masks_etc.F:402-404; ini_vel.F:63; ini_psurf.F:62"),
    "I2": ExchangeEvent(("seaiceMaskU", "seaiceMaskV", "HEFF", "AREA", "HSNOW"), "SEAICE_CGRID", "masks; cold-start inventory files", "TLB", "seaice_init_varia.F:228,294,319,347"),
    "I3": ExchangeEvent(THSICE_FIELDS, "ALLOW_THSICE", "initialization including restart", "HB", "thsice_ini_vars.F:160-168"),
    "R": ExchangeEvent(("uVel", "vVel", "theta", "salt", "etaN", "gtNm1", "etaH", "etaHnm1", "dEtaHdt", "UICE", "VICE", "HEFF", "AREA", "TICES", "HSNOW"), "none", "native pickup restore", "TLHB", "read_pickup.F:538-562; seaice_read_pickup.F:318-322"),
}


def _ordered_sum(block: torch.Tensor) -> torch.Tensor:
    """Left fold from +0.0 over the logical (bj, bi, j, i) order of ``block``."""
    total = block.new_zeros(())
    for value in block.reshape(-1).unbind():
        total = total + value
    return total


# CPU TorchScript has no fuser, so the scripted fold runs the same additions.
_ordered_sum_script = torch.jit.script(_ordered_sum)


@torch.library.custom_op("locomgcm_seaice::ordered_sum", mutates_args=())
def _ordered_sum_op(block: torch.Tensor) -> torch.Tensor:
    """Opaque ordered reduction: a compiled step keeps one node and no reorder.

    The gate ``seaice_ordered_sum_cuda`` selects the single-thread CUDA fold.
    """
    from .dynamics._lsr_cuda.dispatcher import ordered_sum

    return ordered_sum(block, lambda: (_ordered_sum_script if block.device.type == "cpu" else _ordered_sum)(block))


@_ordered_sum_op.register_fake
def _ordered_sum_fake(block: torch.Tensor) -> torch.Tensor:
    return block.new_empty(())


@torch.library.custom_op("locomgcm_seaice::ordered_sum_pair", mutates_args=())
def _ordered_sum_pair_op(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Both residual means of seaice_lsr.F:1237-1284 as one opaque reduction.

    Each fold keeps its own accumulator and order; the CUDA kernel of
    ``seaice_ordered_sum_cuda`` runs the two folds in one launch.
    """
    from .dynamics._lsr_cuda.dispatcher import ordered_sum_pair

    def reference() -> torch.Tensor:
        fold = _ordered_sum_script if first.device.type == "cpu" else _ordered_sum
        return torch.stack((fold(first), fold(second)))

    return ordered_sum_pair(first, second, reference)


@_ordered_sum_pair_op.register_fake
def _ordered_sum_pair_fake(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return first.new_empty((2,))


#: offline_exf_seaice Cartesian grid (PARM04): uniform 5 km cells, southern
#: edge at ygOrigin=-110 km.
CARTESIAN_SPACING = 5000.
Y_ORIGIN = -110000.


#: Native halo width; see the module docstring.
HALO = 3


class SeaiceSpace:
    """Fixed topology and field operations; contains no model history."""

    def __init__(self, chain: str, *, nx: int, ny: int, tile_nx: int, tile_ny: int,
                 couples_x: bool, couples_y: bool) -> None:
        if chain not in ("T", "L", "H", "B"):
            raise ValueError(f"unknown chain {chain}")
        if not (couples_x and couples_y):
            raise ValueError("sea-ice halos are implemented for doubly periodic faces only; "
                             f"edge_face_couples x={couples_x}, y={couples_y}")
        if nx % tile_nx or ny % tile_ny:
            raise ValueError(f"tiles {tile_nx}x{tile_ny} do not cut the {nx}x{ny} face exactly")
        self.chain = chain
        self.nx, self.ny = nx, ny
        # Constructed directly: from_face's exch2 rule (halo within the tile)
        # does not bind the periodic copy below, which wraps modulo the face
        # like MIT's single-tile FILL_HALO_LOCAL (1D_ocean_ice_column: sNx=1).
        self.layout = CubeTileLayout(nx=nx, ny=ny, tile_nx=tile_nx, tile_ny=tile_ny, halo=HALO)
        self.shape = (self.layout.tiles_y, self.layout.tiles_x, self.layout.local_ny, self.layout.local_nx)
        self.category_shape = (*self.shape[:2], 1, *self.shape[2:])

    def validate(self, field: torch.Tensor) -> None:
        if tuple(field.shape) not in (self.shape, self.category_shape) or field.dtype != torch.float64:
            raise ValueError(f"expected float64 native {self.shape} or {self.category_shape}, got {field.shape}, {field.dtype}")
        # Reject expand/as_strided layouts in which distinct tile cells alias.
        if not field.is_contiguous():
            raise ValueError("native fields must have disjoint contiguous tile storage")

    def full(self, value=0.0, *, device=None, categories=False) -> torch.Tensor:
        return torch.full(self.category_shape if categories else self.shape, value, dtype=torch.float64, device=device)

    @staticmethod
    def interior(field: torch.Tensor) -> torch.Tensor:
        return field[..., HALO:-HALO, HALO:-HALO]

    def tiles(self, field: torch.Tensor) -> tuple[torch.Tensor, ...]:
        self.validate(field)
        return tuple(field[bj, bi].clone() for bj in range(self.shape[0]) for bi in range(self.shape[1]))

    def from_tiles(self, tiles: tuple[torch.Tensor, ...]) -> torch.Tensor:
        if len(tiles) != self.layout.tiles_per_face or any(t.shape[-2:] != self.shape[-2:] for t in tiles):
            raise ValueError(f"{self.layout.tiles_per_face} tiles of {self.shape[-2:]} required")
        result = torch.stack(tiles).reshape(*self.shape[:2], *tiles[0].shape)
        self.validate(result)
        return result

    def from_global(self, field: torch.Tensor) -> torch.Tensor:
        """Input adaptation only: copy inner cells, initialize halo to +0.0."""
        if field.dtype != torch.float64 or tuple(field.shape) not in ((self.ny, self.nx), (1, self.ny, self.nx)):
            raise ValueError(f"global input must be float64 [{self.ny},{self.nx}] or [1,{self.ny},{self.nx}]")
        result = self.full(device=field.device, categories=field.ndim == 3)
        for d in range(self.layout.tiles_per_face):
            tx, ty = self.layout.tile_origin(d)
            rows, cols = self.layout.interior_window(d)
            result[ty, tx, ..., HALO:-HALO, HALO:-HALO] = field[..., rows, cols]
        return result

    def _halo_indices(self, device) -> tuple[torch.Tensor, torch.Tensor]:
        j = torch.arange(-HALO, self.ny + HALO, device=device) % self.ny
        i = torch.arange(-HALO, self.nx + HALO, device=device) % self.nx
        return j, i

    def to_global(self, field: torch.Tensor, *, halo=False) -> torch.Tensor:
        """Output view: tile interiors own cells; optional global halo is periodic."""
        self.validate(field)
        rows = [torch.cat([self.interior(field[bj, bi]) for bi in range(self.shape[1])], dim=-1)
                for bj in range(self.shape[0])]
        result = torch.cat(rows, dim=-2)
        if halo:
            j, i = self._halo_indices(field.device)
            result = result.index_select(-2, j).index_select(-1, i)
        return result

    def _periodic_copies(self, global_field: torch.Tensor) -> torch.Tensor:
        j, i = self._halo_indices(global_field.device)
        padded = global_field.index_select(-2, j).index_select(-1, i)
        # face_window gives coordinates, stack allocates nonoverlapping copies.
        tiles = []
        for d in range(self.layout.tiles_per_face):
            rows, cols = self.layout.face_window(d)
            tiles.append(padded[..., rows, cols].clone())
        return self.from_tiles(tuple(tiles))

    def from_global_periodic(self, field: torch.Tensor) -> torch.Tensor:
        """Static geometry construction; never use for prognostic updates."""
        self.from_global(field)  # Validate input dimensions and precision.
        return self._periodic_copies(field)

    def exchange(self, event: str, fields: Mapping[str, torch.Tensor], *, chain: str, enabled: bool = True) -> dict[str, torch.Tensor]:
        """Copy neighboring interiors into all three halo layers, without rotation.

        The owner supplies the runtime gate. The fixed chain and CPP constraints
        are checked here, including prohibition of X5 in these four programs.
        """
        spec = EXCHANGE_EVENTS[event]
        illegal = set(fields) - set(spec.fields)
        if illegal:
            raise ValueError(f"{event} does not exchange {sorted(illegal)}")
        if chain not in "TLHB" or len(chain) != 1:
            raise ValueError(f"unknown chain {chain}")
        if enabled and chain not in spec.chains:
            raise ValueError(f"{event} is not enabled for {chain}: {spec.runtime_gate}")
        for field in fields.values():
            self.validate(field)
        return {name: self._periodic_copies(self.to_global(field)) if enabled else field.clone() for name, field in fields.items()}

    @staticmethod
    def replace_domain(old: torch.Tensor, value: torch.Tensor, domain: Domain = INTERIOR) -> torch.Tensor:
        """Write the specified mathematical domain and preserve the old band."""
        rows, cols = domain.slices
        result = old.clone()
        result[..., rows, cols] = value[..., rows, cols] if value.shape == old.shape else value
        return result

    @staticmethod
    def neighbor(field: torch.Tensor, dj: int, di: int, *, domain: Domain = INTERIOR) -> torch.Tensor:
        """Read this tile's own neighbor, including its retained halo, never roll."""
        # The trims are nonnegative storage-edge offsets (Domain), so the window
        # is plain integer arithmetic; slice.indices is not traceable by Dynamo,
        # and a graph break here made every caller's (dj, di, trim) its own frame.
        # min/max keep slice.indices' clamp of a trim larger than the storage.
        ny, nx = field.shape[-2], field.shape[-1]
        r0, r1 = min(domain.south, ny) + dj, max(ny - domain.north, 0) + dj
        c0, c1 = min(domain.west, nx) + di, max(nx - domain.east, 0) + di
        if r0 < 0 or r1 > ny or c0 < 0 or c1 > nx:
            raise ValueError("neighbor request leaves native tile storage")
        return field[..., r0:r1, c0:c1]

    def read(self, field: torch.Tensor, trim: tuple[int, int, int, int], dj=0, di=0) -> torch.Tensor:
        return self.neighbor(field, dj, di, domain=Domain(*trim))

    def write(self, field: torch.Tensor, values: torch.Tensor, trim: tuple[int, int, int, int]) -> torch.Tensor:
        return self.replace_domain(field, values, Domain(*trim))

    def tile(self, field: torch.Tensor, d: int) -> torch.Tensor:
        tx, ty = self.layout.tile_origin(d)
        return field[ty, tx].clone()

    def stack_tiles(self, fields: list[torch.Tensor]) -> torch.Tensor:
        return self.from_tiles(tuple(fields))

    def exchange_vector(self, u: torch.Tensor, v: torch.Tensor, *, event: str) -> tuple[torch.Tensor, torch.Tensor]:
        names = {"X1": ("ustress", "vstress"), "X4": ("uwind", "vwind"), "X6": ("UICE", "VICE"), "X7": ("fu", "fv")}[event]
        fields = self.exchange(event, dict(zip(names, (u, v))), chain=self.chain)
        return fields[names[0]], fields[names[1]]

    def max_abs(self, field: torch.Tensor, trim: tuple[int, int, int, int]) -> torch.Tensor:
        return self.read(field, trim).abs().amax()

    def owned(self, *, device=None) -> torch.Tensor:
        return self.replace_domain(self.full(device=device), self.full(1., device=device)).bool()

    def domain_mask(self, domain: Domain, *, device=None) -> torch.Tensor:
        return self.replace_domain(self.full(device=device), self.full(1., device=device), domain).bool()

    def neighbor_field(self, field: torch.Tensor, dj: int, di: int, *, domain: Domain) -> torch.Tensor:
        return self.replace_domain(torch.zeros_like(field), self.neighbor(field, dj, di, domain=domain), domain)

    def y_center(self, *, device=None) -> torch.Tensor:
        """Cartesian YC from ygOrigin=-110 km, delY=5 km (PARM04)."""
        y = Y_ORIGIN + (torch.arange(self.ny, dtype=torch.float64, device=device) + .5) * CARTESIAN_SPACING
        return self.from_global_periodic(y[:, None].expand(self.ny, self.nx))

    def ordered_sum(self, field: torch.Tensor, trim: tuple[int, int, int, int] = (3, 3, 3, 3)) -> torch.Tensor:
        """One accumulator in d,j,i order (seaice_lsr.F:1237-1284)."""
        self.validate(field)
        if field.ndim != 4:
            raise ValueError("residual reduction requires G")
        block = self.read(field, trim)
        if forward_mode_active():
            # The opaque operator has no autograd and would drop the tangent: the
            # same additions in Python, the decline recorded as its dispatcher
            # would, ahead of the requires_grad branch so it is always recorded.
            from .dynamics._lsr_cuda.dispatcher import SUM_KEY, decline_for_forward_mode

            decline_for_forward_mode(SUM_KEY, "ordered_sum", (block,), contiguous=False)
            return _ordered_sum(block)
        if torch.is_grad_enabled() and block.requires_grad:
            return _ordered_sum(block)
        return torch.ops.locomgcm_seaice.ordered_sum(block)

    def ordered_sum_pair(self, first: torch.Tensor, second: torch.Tensor,
                         trim: tuple[int, int, int, int] = (3, 3, 3, 3)) -> tuple[torch.Tensor, torch.Tensor]:
        """:meth:`ordered_sum` of two G fields, folded independently in one reduction."""
        for field in (first, second):
            self.validate(field)
            if field.ndim != 4:
                raise ValueError("residual reduction requires G")
        blocks = self.read(first, trim), self.read(second, trim)
        if forward_mode_active():
            from .dynamics._lsr_cuda.dispatcher import SUM_KEY, decline_for_forward_mode

            decline_for_forward_mode(SUM_KEY, "ordered_sum_pair", blocks, contiguous=False)
            return _ordered_sum(blocks[0]), _ordered_sum(blocks[1])
        if torch.is_grad_enabled() and any(block.requires_grad for block in blocks):
            return _ordered_sum(blocks[0]), _ordered_sum(blocks[1])
        both = torch.ops.locomgcm_seaice.ordered_sum_pair(*blocks)
        return both[0], both[1]

    def global_area(self, area: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Tile partial sums then d-order sum (ini_global_domain.F:77-116)."""
        weighted = self.interior(area * mask)
        total = area.new_zeros(())
        for bj in range(self.shape[0]):
            for bi in range(self.shape[1]):
                partial = area.new_zeros(())
                for value in weighted[bj, bi].reshape(-1).unbind():
                    partial = partial + value
                total = total + partial
        return total

    @staticmethod
    def global_max(field: torch.Tensor) -> torch.Tensor:
        return SeaiceSpace.interior(field).amax()

    def geometry(self, bathy: torch.Tensor) -> Geometry:
        """MIT bathy input is negative in the ocean; no partial cells here.

        ini_masks_etc.F:195-196,417-420,460-465; Cartesian PARM04.
        """
        if bathy.shape != (self.ny, self.nx) or bathy.dtype != torch.float64 or not bool(torch.isfinite(bathy).all()):
            raise ValueError(f"bathy must be finite float64 [{self.ny},{self.nx}], negative ocean")
        wet = (bathy < 0).to(dtype=torch.float64)
        c = self.from_global_periodic(wet)
        u = self.from_global_periodic(wet * torch.roll(wet, 1, -1))
        v = self.from_global_periodic(wet * torch.roll(wet, 1, -2))
        length = self.full(CARTESIAN_SPACING, device=bathy.device)
        area = self.full(CARTESIAN_SPACING * CARTESIAN_SPACING, device=bathy.device)
        zero = self.full(device=bathy.device)
        return Geometry(
            *(length.clone() for _ in range(8)), *(area.clone() for _ in range(4)),
            *(c.clone() for _ in range(5)), *(u.clone() for _ in range(5)), *(v.clone() for _ in range(5)),
            zero.clone(), zero.clone(), self.full(9.81, device=bathy.device),
            bathy.new_tensor(10.0), self.global_area(area, c),
        )
