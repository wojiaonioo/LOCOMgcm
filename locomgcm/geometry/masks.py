from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from locomgcm.accel.constant_cache import ConstantCache

if TYPE_CHECKING:
    from locomgcm.geometry.grid import BoundaryTopology


def _prune_isolated_wet_columns(b: torch.Tensor) -> torch.Tensor:
    """Zero wet columns that have no wet horizontal neighbour.

    The test uses four face neighbours and treats cells outside the domain as dry.
    It repeats until no isolated wet column remains. Apply this optional topology
    operation before constructing hFac so masks and exported bathymetry agree.
    """
    bb = b.clone()
    while True:
        wet = bb > 0
        if not wet.any():
            break
        # 4-neighbour wet status (treat outside-domain as dry for the test;
        # boundary cells with one wet neighbour interior to the domain are
        # kept).
        west = torch.zeros_like(wet)
        east = torch.zeros_like(wet)
        south = torch.zeros_like(wet)
        north = torch.zeros_like(wet)
        west[:, 1:] = wet[:, :-1]
        east[:, :-1] = wet[:, 1:]
        south[1:, :] = wet[:-1, :]
        north[:-1, :] = wet[1:, :]
        any_neighbour_wet = west | east | south | north
        isolated = wet & ~any_neighbour_wet
        if not isolated.any():
            break
        bb = torch.where(isolated, torch.zeros_like(bb), bb)
    return bb


def obcs_fix_topo_and_clamp(
    bathymetry: torch.Tensor,
    *,
    column_floor: float,
    boundary_topology: "BoundaryTopology | None",
    obcs_fix_topo: bool = False,
    prune_isolated: bool = False,
) -> torch.Tensor:
    """Apply the selected boundary topology correction and clamp ocean depth.

    Depth is positive in the ocean and zero on land. The operations are:

    1. With ``obcs_fix_topo=True``, shallow each open-boundary strip column to
       the minimum of its own depth and its interior neighbour's depth. This
       follows MITgcm ``pkg/obcs/obcs_check_depths.F``. The default is False,
       matching ``OBCSfixTopo=.FALSE.`` in ``pkg/obcs/obcs_readparms.F``.
    2. Clamp depth to ``[0, column_floor]``, where ``column_floor`` is the model's
       total vertical extent. This prevents deeper bathymetry or negative land
       depths from entering metric and volume calculations.
    3. With ``prune_isolated=True``, remove wet columns with four dry horizontal
       neighbours. The default is False, retaining MITgcm's isolated wet cells.

    Use the same ``obcs_fix_topo`` setting and corrected bathymetry for
    ``MaskSet.from_bathymetry_3d`` and the ``StructuredCGrid`` metric constructors.
    The hFac fractions, depth metrics, cell volumes and MITgcm bathymetry export
    must describe the same geometry.
    """
    b = bathymetry.clone()
    if obcs_fix_topo and boundary_topology is not None:
        if boundary_topology.open_y:
            b[0, :] = torch.minimum(b[0, :], b[1, :])
            b[-1, :] = torch.minimum(b[-1, :], b[-2, :])
        if boundary_topology.open_x:
            b[:, 0] = torch.minimum(b[:, 0], b[:, 1])
            b[:, -1] = torch.minimum(b[:, -1], b[:, -2])
    # Clamp land to zero. Loaders returning -elevation leave land cells
    # as negative depth; without this
    # clamp `MetricSet.depth_center` and `cell_volume` retain those negative
    # values and silently corrupt depth-dependent boundary code (e.g. Flather
    # `sqrt(gravity / depth)`).  Documented contract is positive=ocean,
    # zero=land — enforce it here.
    b = torch.clamp(b, min=0.0, max=column_floor)
    if prune_isolated:
        b = _prune_isolated_wet_columns(b)
    return b


def _combine_face_hfac(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    smooth: bool,
) -> torch.Tensor:
    """MITgcm's centre-pair reduction for ``hFacW`` / ``hFacS``.

    ``smooth=False`` is ``ini_masks_etc.F:244-254``'s ``MIN(a, b)``.

    ``smooth=True`` is the ``USE_SMOOTH_MIN`` form of
    ``update_masks_etc.F:220-235``::

        hFacW(i,j,k) = maskW(i,j,k) * SMOOTHMIN_RS( hFacC(i,j,k), hFacC(i-1,j,k) )

    with ``SMOOTHMIN_RS(a,b) = .5*( a + b - SMOOTHABS_RS(a-b) )`` (``:364``)
    and ``SMOOTHABS_RS ≡ 0`` for a negative ``smoothAbsFuncRange``
    (``:398-401``) — i.e. the arithmetic mean of the two centres.

    The ``maskW`` / ``maskS`` factor is not decorative: unlike ``MIN``, the
    mean does not close a face on its own, so without the factor a face next
    to a dry centre would stay open.  The mask is a boolean predicate, so no
    gradient flows through it; the adjoint of a wet face is ``0.5`` onto each
    centre, matching MITgcm's own ``maskW`` constant.

    The predicate used here is ``a > 0 and b > 0`` on the **smoothed** centres,
    which is where MITgcm is NOT reproduced exactly.  MITgcm's ``maskW`` is
    frozen at ``ini_masks_etc.F`` time and therefore derived from the
    *truncated* hFacC, whose closure test is ``hFac_loc < hFacMnSz/2``
    (``ini_masks_etc.F:113-117``, masks at ``:459-476``).  The smoothed hFacC
    closes only at ``hFacCtmp <= 0`` (``update_masks_etc.F:100``; the
    half-threshold is commented out one line above, at ``:101``).  On the band
    ``0 < raw < hFacMnSz/2`` MITgcm therefore has ``hFacC > 0`` yet
    ``maskW = 0`` and a closed face, while this function opens it.  See the
    note next to ``center`` in :meth:`from_bathymetry_3d` for why reproducing
    the frozen mask needs a two-bathymetry API.
    """
    if not smooth:
        return torch.minimum(a, b)
    wet = (a > 0.0) & (b > 0.0)
    return torch.where(wet, 0.5 * (a + b), torch.zeros_like(a))


@dataclass(frozen=True)
class MaskSet:
    """Binary masks and optional continuous hFac fractions.

    Masks can be 2D ``[ny, nx]`` (column-uniform) or 3D ``[nz, ny, nx]``
    (depth-dependent, for bathymetry).  Use :pyattr:`has_3d_masks` to check.

    The optional ``hFacC/hFacW/hFacS`` tensors hold continuous partial-cell
    fractions (0–1) following MITgcm convention.  When ``None``, callers
    should treat every wet cell as fully wet (hFac = 1).

    The ``maskInC`` / ``maskInW`` / ``maskInS`` fields mirror MITgcm's
    ``obcs_init_fixed.F`` interior-only masks: ``1`` for wet interior cells,
    ``0`` for OB-strip cells and land cells.  They are consumed by operations
    that need to skip the boundary strip entirely (``obcs_balance_flow``, KPP-
    style vertical mixing, diagnostic integrals).
    """

    center: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor

    # Continuous partial-cell fractions (None for flat-bottom / 2D masks)
    hFacC: torch.Tensor | None = None   # [nz, ny, nx]
    hFacW: torch.Tensor | None = None   # [nz, ny, nx+1]
    hFacS: torch.Tensor | None = None   # [nz, ny+1, nx]

    # MITgcm-style OB-strip-aware interior masks (None for legacy tests)
    maskInC: torch.Tensor | None = None   # [ny, nx]    — tracer cell strip mask
    maskInW: torch.Tensor | None = None   # [ny, nx+1]  — u-face strip mask
    maskInS: torch.Tensor | None = None   # [ny+1, nx]  — v-face strip mask

    # MITgcm ``smoothAbsFuncRange`` (PARAMS.h:1007) — the depth-control
    # geometry mode.  ``None`` = the build has no ``USE_SMOOTH_MIN``, i.e.
    # every existing case: hFacC keeps the ``hFacMin`` truncation and the
    # faces keep ``MIN``.  A negative value is the ``ALLOW_DEPTH_CONTROL``
    # build of ``verification/bottom_ctrl_5x5``; see
    # :meth:`from_bathymetry_3d` for the two formulas it selects.  Carried on
    # the mask set (not on the grid) because the only downstream consumer,
    # ``horizontal_diffusion._compute_hFacZ_4neighbor``, already receives the
    # grid and reads ``grid.masks``.
    smooth_abs_func_range: float | None = None

    @property
    def has_3d_masks(self) -> bool:
        return self.center.ndim == 3

    # ------------------------------------------------------------------
    # nonlinFreeSurf: the surface level of hFac changes while the run steps,
    # in place without autograd, by tensor replacement with it
    # ------------------------------------------------------------------
    @property
    def hfac_time_varying(self) -> bool:
        """True once a producer declared ``hFac*`` is rewritten during the run.

        Consumers that memoize an hFac-DERIVED tensor under a key built from
        grid/mask *identity* must consult this before caching: the write that
        makes their key stale does not change the identity (the in-place
        write keeps the tensors, :meth:`replace_surface_hfac` keeps the mask
        set).  See :meth:`mark_hfac_time_varying`.

        The three bool masks (``center``/``u``/``v``) are NOT covered by this
        flag and never need it -- ``UPDATE_SURF_DR`` preserves wet/dry (the
        ``hFacInf`` floor keeps every wet surface cell wet), so
        :meth:`_float_for` stays cacheable either way.
        """
        return getattr(self, "_hfac_time_varying", False)

    @property
    def resting_hfac(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """MITgcm ``h0FacC``/``h0FacW``/``h0FacS`` of a grid whose hFac moves.

        ``None`` while hFac is static, where the resting and the current
        fractions are the same tensors.  Once :meth:`mark_hfac_time_varying`
        has run, the resting (initial, fixed in time) fractions its producer
        declared: MITgcm reads them where a term must not follow the free
        surface, e.g. the closed fraction of the no-slip side drag
        (``mom_u_sidedrag.F:107-108``, ``mom_vecinv.F:311-323``).
        """
        return getattr(self, "_resting_hfac", None)

    def mark_hfac_time_varying(
        self,
        resting: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """Declare that ``hFacC``/``hFacW``/``hFacS`` are rewritten during the run.

        ``resting`` is the producer's ``(h0FacC, h0FacW, h0FacS)``, the
        geometry before the first rewrite, served by :attr:`resting_hfac`.

        MITgcm's ``UPDATE_SURF_DR`` (``nonlinFreeSurf>=1``) overwrites the
        surface level of the three hFac fields every step, and LOCOMgcm does
        the same at ``steppers/implicit_free_surface.py`` ``_apply_surf_dr``.
        The write keeps ``id(grid)`` and ``id(masks)`` fixed, so a memo keyed on
        grid identity that holds an hFac-derived quantity (``recip_hFac*``,
        corner ``hFacZ``, ...) would keep serving the pre-write geometry --
        MITgcm itself recomputes both every step (``update_surf_dr.F`` for
        ``recip_hFac*``, ``mom_calc_hfacz.F`` per ``MOM_*`` call).

        Producers call this AT the mutation site; consumers read
        :attr:`hfac_time_varying`.  Idempotent, and a one-way per-run latch: a
        grid whose hFac has moved never becomes static again.  That matters
        beyond tidiness -- the value a consumer reads has to be constant across
        steps, because a per-step-varying Python value on a memo key would
        guard-fail under ``torch.compile`` on every single step and exhaust
        ``recompile_limit``.

        Stored as an undeclared instance attribute (``MaskSet`` is a frozen
        dataclass) for the same reason ``_float_cache`` is: ``dataclasses``
        ``replace`` must not carry it onto a rebuilt mask set whose hFac
        tensors are different objects.
        """
        object.__setattr__(self, "_hfac_time_varying", True)
        object.__setattr__(self, "_resting_hfac", tuple(resting))

    def fixed_hfac(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The ``hFacC``/``hFacW``/``hFacS`` whose levels below the surface are the run's.

        Only those levels are read: ``UPDATE_SURF_DR`` changes the surface
        level alone.  They are the tensors held before the first
        :meth:`replace_surface_hfac`, latched again whenever the held tensors
        are not the ones it installed last, i.e. someone else rebound them (a
        device migration, a binding of the hFac as a control): the deeper
        levels then follow the rebound tensors and their graph.  Undeclared
        attributes, for the reason given in :meth:`mark_hfac_time_varying`.
        """
        held = (self.hFacC, self.hFacW, self.hFacS)
        installed = getattr(self, "_installed_hfac", None)
        fixed = getattr(self, "_fixed_hfac", None)
        if fixed is None or installed is None or any(
                a is not b for a, b in zip(held, installed, strict=True)):
            fixed = held
            object.__setattr__(self, "_fixed_hfac", fixed)
        return fixed

    def with_surface_level(self, name: str, layer: torch.Tensor) -> torch.Tensor:
        """A new ``name`` tensor: ``layer`` on the surface level, :meth:`fixed_hfac` below.

        ``layer`` has a leading level axis of length 1; dtype and device follow
        the current ``name`` tensor.
        """
        current = getattr(self, name)
        fixed = self.fixed_hfac()[("hFacC", "hFacW", "hFacS").index(name)]
        return torch.cat(
            (layer[0:1].to(dtype=current.dtype, device=current.device),
             fixed[1:].to(device=current.device)),
            dim=0,
        )

    def replace_surface_hfac(
        self,
        layers: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """Install new surface levels of ``hFacC``/``hFacW``/``hFacS`` as NEW tensors.

        The reverse-mode form of the in-place surface write (``UPDATE_SURF_DR``,
        ``steppers/ifs/nlfs_geometry.py`` ``apply_surf_dr``): an operation
        that read hFac before the write may have saved it for backward, so
        the old tensors keep their storage and version and new ones take
        their place.  ``layers`` are the C/W/S surface levels with a leading
        level axis of length 1.  The levels below the surface come from
        :meth:`fixed_hfac`, so a new tensor's graph never reaches an earlier
        surface level, nor an earlier checkpoint segment.  The caller then
        rebuilds the grid's ``hfac_*_for`` entries
        (``StructuredCGrid.refresh_hfac_runtime``).
        """
        fixed = self.fixed_hfac()
        new = tuple(self.with_surface_level(name, layer)
                    for name, layer in zip(("hFacC", "hFacW", "hFacS"), layers, strict=True))
        for name, value in zip(("hFacC", "hFacW", "hFacS"), new, strict=True):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_installed_hfac", new)
        object.__setattr__(self, "_fixed_hfac", fixed)

    # ------------------------------------------------------------------
    # r* coordinate (select_rStar > 0): every level of hFac moves
    # ------------------------------------------------------------------
    @property
    def vertical_coordinate(self) -> str:
        """``"r_star"`` once :meth:`install_r_star_hfac` has run, else ``"z"``.

        Under r* the continuity w carries the column-stretching term of
        ``integrate_for_w.F:123-145``, whose rate :attr:`r_star_thickness_rate`
        holds; the w producers branch on this value.
        """
        return getattr(self, "_vertical_coordinate", "z")

    @property
    def r_star_thickness_rate(self) -> torch.Tensor:
        """MITgcm ``rStarDhDt`` of the latest INTEGR_CONTINUITY, ``[*, ny, nx]`` in s-1.

        ``dEtaHdt*recip_Rcol`` (``integr_continuity.F:245-256``); set by the r*
        stepper with :meth:`set_r_star_thickness_rate` before any w is
        diagnosed from the velocities of that continuity call.
        """
        rate = getattr(self, "_r_star_thickness_rate", None)
        if rate is None:
            raise RuntimeError(
                "r* geometry is installed but no INTEGR_CONTINUITY rate "
                "(rStarDhDt) has been set for the w it requires"
            )
        return rate

    def set_r_star_thickness_rate(self, rate: torch.Tensor) -> None:
        object.__setattr__(self, "_r_star_thickness_rate", rate)

    def install_r_star_hfac(
        self,
        hfac: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        resting: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """MITgcm ``UPDATE_R_STAR``: install ``hFacC``/``hFacW``/``hFacS`` at every level.

        ``hfac`` is ``h0Fac*rStarFac`` (``update_r_star.F:50-59``).  With
        autograd enabled the three fields become NEW tensors, for the reason
        given in :meth:`replace_surface_hfac`, and the caller rebuilds the
        grid's ``hfac_*_for`` entries; without it they are written in place.
        r* keeps every wet cell wet (CALC_R_STAR stops below ``hFacInf``), so
        the bool masks are unchanged.
        """
        self.mark_hfac_time_varying(resting)
        object.__setattr__(self, "_vertical_coordinate", "r_star")
        for name, value in zip(("hFacC", "hFacW", "hFacS"), hfac, strict=True):
            current = getattr(self, name)
            value = value.to(dtype=current.dtype, device=current.device)
            if torch.is_grad_enabled():
                object.__setattr__(self, name, value)
            else:
                current.copy_(value)

    def _float_for(
        self,
        attr: str,
        src: torch.Tensor,
        dtype: torch.dtype | None,
        device: torch.device | str | None,
    ) -> torch.Tensor:
        """Materialise ``src`` as a float mask, memoized per (attr, dtype, device).

        Production masks are ``torch.bool`` on every path that builds them from
        bathymetry (``from_bathymetry_3d`` -> ``center = hFacC > 0.0`` etc.), so
        each of the ~40 call sites allocated and converted a whole field on
        every call.  The result is a pure function of an immutable source, so it
        is memoized once per (attribute, dtype, device).

        Why this is safe to cache -- the three source tensors are never written
        in place:

        * The nonlinFreeSurf per-step surface refresh writes ``hFacC[0]`` /
          ``hFacW[0]`` / ``hFacS[0]`` only (``steppers/implicit_free_surface.py``
          ``_apply_surf_dr``); the bool masks are not touched.  This is what
          blocks an hFac cache keyed on grid identity and does NOT block this one.
        * ``geometry/grid.py`` ``ensure_grid_device_resident`` rebinds
          ``center``/``u``/``v`` through ``object.__setattr__`` to bit-identical
          tensors on the compute device.  The key carries the RESOLVED device
          (``device=None`` resolves against the *current* ``src.device``), so a
          post-migration call lands on its own entry rather than reusing the
          host one.

        Why the cache is an undeclared attribute rather than a dataclass field:
        ``geometry/cubed_sphere_grid.py`` rebuilds a seam-reconciled MaskSet via
        ``dataclasses.replace(masks, ..., u=hFacW > 0.0, v=hFacS > 0.0)``.
        ``replace`` copies declared fields forward, so a field-held cache would
        follow the old ``u``/``v`` onto an object whose ``u``/``v`` differ.  A
        plain instance attribute is not copied by ``replace``.

        The returned tensor is now shared by every caller, so a caller that
        needs to modify it must clone first.  Both that do already clone
        (``dynamics/barotropic.py`` and ``boundaries/open.py`` zero the
        open-boundary strip in place); those clones are load-bearing.

        Shares the ``LOCOMGCM_CACHE_CONSTANTS`` gate with the per-grid constant memos
        in ``dynamics/horizontal_diffusion.py`` -- same class of object, one
        switch turns them all off and back to the per-call materialisation.
        """
        want_dtype = dtype or torch.float64
        if device is None:
            want_device = src.device
        else:
            want_device = torch.device(device)
            if (
                want_device.type == "cuda"
                and want_device.index is None
                and torch.cuda.is_available()
            ):
                # Normalise "cuda" to "cuda:N" so the bare and indexed spellings
                # of the same device share one entry instead of two.
                want_device = torch.device("cuda", torch.cuda.current_device())

        def _materialise() -> torch.Tensor:
            t = src.to(dtype=want_dtype, device=want_device)
            return t if t.dtype != torch.bool else t.float()

        cache = getattr(self, "_float_cache", None)
        if cache is None:
            cache = ConstantCache()
            object.__setattr__(self, "_float_cache", cache)
        key = (attr, want_dtype, str(want_device))
        return cache.get_or_build(
            key,
            _materialise,
            variant=f"mask_{attr}_float",
            site="MaskSet._float_for",
            tensor_only=True,
        )

    def center_float(self, *, dtype: torch.dtype | None = None, device: torch.device | str | None = None) -> torch.Tensor:
        return self._float_for("center", self.center, dtype, device)

    def u_float(self, *, dtype: torch.dtype | None = None, device: torch.device | str | None = None) -> torch.Tensor:
        return self._float_for("u", self.u, dtype, device)

    def v_float(self, *, dtype: torch.dtype | None = None, device: torch.device | str | None = None) -> torch.Tensor:
        return self._float_for("v", self.v, dtype, device)

    @classmethod
    def all_wet(cls, ny: int, nx: int, *, device: str = "cpu") -> "MaskSet":
        return cls(
            center=torch.ones((ny, nx), dtype=torch.bool, device=device),
            u=torch.ones((ny, nx + 1), dtype=torch.bool, device=device),
            v=torch.ones((ny + 1, nx), dtype=torch.bool, device=device),
        )

    @classmethod
    def from_bathymetry(
        cls,
        depth_2d: torch.Tensor,
        *,
        device: str = "cpu",
    ) -> "MaskSet":
        """Create a binary mask from a 2D depth field.

        Parameters
        ----------
        depth_2d : torch.Tensor
            Shape ``[ny, nx]``.  Positive values = ocean, zero/negative = land.

        Returns
        -------
        MaskSet
            Binary masks where ``True`` = wet (ocean).
            - center: ``[ny, nx]`` — wet where ``depth > 0``.
            - u: ``[ny, nx+1]`` — wet where at least one neighbor center is wet.
            - v: ``[ny+1, nx]`` — wet where at least one neighbor center is wet.
        """
        if depth_2d.ndim != 2:
            raise ValueError(f"expected 2D depth tensor, got {depth_2d.ndim}D")
        ny, nx = depth_2d.shape
        center = (depth_2d > 0.0).to(device=device)

        # u-face [ny, nx+1]: wet if either left or right center is wet
        u_mask = torch.zeros((ny, nx + 1), dtype=torch.bool, device=device)
        u_mask[:, 1:-1] = center[:, :-1] | center[:, 1:]
        u_mask[:, 0] = center[:, 0]
        u_mask[:, -1] = center[:, -1]

        # v-face [ny+1, nx]: wet if either south or north center is wet
        v_mask = torch.zeros((ny + 1, nx), dtype=torch.bool, device=device)
        v_mask[1:-1, :] = center[:-1, :] | center[1:, :]
        v_mask[0, :] = center[0, :]
        v_mask[-1, :] = center[-1, :]

        return cls(center=center, u=u_mask, v=v_mask)

    @classmethod
    def from_bathymetry_3d(
        cls,
        bathymetry: torch.Tensor,
        interface_depth: torch.Tensor,
        *,
        device: str = "cpu",
        hfac_min: float = 0.3,
        hfac_min_dz: float = 5.0,
        boundary_topology: "BoundaryTopology | None" = None,
        obcs_fix_topo: bool = False,
        smooth_abs_func_range: float | None = None,
    ) -> "MaskSet":
        """Create depth-dependent 3D masks with continuous hFac fractions.

        Parameters
        ----------
        bathymetry : Tensor [ny, nx]
            Ocean depth (positive = ocean).  Zero or negative = land.
        interface_depth : Tensor [nz+1]
            Depth of layer interfaces (positive downward, starting from 0).
        hfac_min : float
            Minimum non-zero hFac fraction (MITgcm default 0.2).  Cells with
            ``0 < hFac < hfac_min`` are rounded up to ``hfac_min``.
        hfac_min_dz : float
            Minimum dimensional cell thickness in metres (MITgcm default 5.0).
            For each level k, the effective minimum fraction is
            ``max(hfac_min, min(hfac_min_dz / dz[k], 1.0))``.
            Set to 0.0 to disable.
        boundary_topology : BoundaryTopology or None
            Open/periodic/wall topology per axis.  Selects the hFacW / hFacS
            outer-edge treatment below, and names which edges ``obcs_fix_topo``
            shallows when it is enabled.
        obcs_fix_topo : bool
            MITgcm ``OBCSfixTopo``.  When True, open-boundary strip columns are
            shallowed to min(self, interior neighbour) before hFac is computed.
            Default **False** = MITgcm's own default
            (``pkg/obcs/obcs_readparms.F:291``).  The caller that also builds a
            MetricSet from the same bathymetry MUST pass the identical value
            there — see ``obcs_fix_topo_and_clamp``.
        smooth_abs_func_range : float or None
            MITgcm ``smoothAbsFuncRange`` (``PARAMS.h:1007``), and at the same
            time the switch for the ``USE_SMOOTH_MIN`` geometry.  Default
            ``None`` reproduces every build without ``ALLOW_DEPTH_CONTROL``:
            ``ini_masks_etc.F`` truncation for hFacC, ``MIN`` for the faces.

            A **negative** value selects the depth-control geometry of
            ``verification/bottom_ctrl_5x5`` (``code_ad/CTRL_OPTIONS.h:53-57``
            defines ``ALLOW_DEPTH_CONTROL`` and therefore ``USE_SMOOTH_MIN``;
            ``input_ad/data:30`` sets ``-1.e-2``).  Two changes, both from
            ``model/src/update_masks_etc.F``:

            * hFacC (``:100-107``) loses the ``hFacMin`` truncation — that
              block is commented out at ``:111-122`` — and gains an
              exponential taper::

                  hFacC = t + m*( exp(-t/m) - exp(-1/m) ),  0 < t <= 1

              with ``t`` the raw fraction and ``m`` the same per-level
              ``hFacMnSz`` the default path uses (``:86``).  ``t <= 0`` still
              gives 0 and ``t > 1`` still gives 1.
            * hFacW / hFacS (``:220-235``) replace ``MIN(a, b)`` by
              ``SMOOTHMIN_RS`` (``:364``) ``= .5*(a + b - SMOOTHABS_RS(a-b))``.
              ``SMOOTHABS_RS`` (``:398-401``) returns **0** whenever
              ``smoothAbsFuncRange < 0``, so the face fraction is the plain
              arithmetic mean of the two neighbouring centres, still gated by
              MITgcm's ``maskW``/``maskS``.  The motive is in MIT's own note at
              ``:200-202``: ``MIN(x, y)`` has no continuous derivative at
              ``y = x``, which is fatal when the bathymetry is the control.

            Positive and zero values raise ``NotImplementedError`` — see the
            check below for why each is out of scope.

        Returns
        -------
        MaskSet with 3D boolean masks AND continuous hFac tensors:
            - center / u / v: boolean ``[nz, ny, nx(+1)]``
            - hFacC / hFacW / hFacS: float64 ``[nz, ny, nx(+1)]``
        """
        if bathymetry.ndim != 2:
            raise ValueError(f"expected 2D bathymetry, got {bathymetry.ndim}D")
        if interface_depth.ndim != 1:
            raise ValueError(f"expected 1D interface_depth, got {interface_depth.ndim}D")

        smooth_geometry = smooth_abs_func_range is not None
        if smooth_geometry and smooth_abs_func_range >= 0.0:
            # Both non-negative cases are real MITgcm configurations and both
            # would be silently wrong if folded into the negative branch, so
            # neither is accepted.
            #   > 0 : ``SMOOTHABS_RS`` (update_masks_etc.F:398-417) evaluates
            #         log(cosh(x*sf))/sf with sf = 10/smoothAbsFuncRange inside
            #         the interval and |x| outside — a third face formula that
            #         is neither MIN nor the arithmetic mean.
            #   = 0 : the MITgcm default (set_defaults.F:179).  SMOOTHABS then
            #         degenerates to |x|, so the faces are exactly MIN while
            #         hFacC still takes the exponential taper — ``USE_SMOOTH_MIN``
            #         is a compile-time flag independent of this parameter.
            raise NotImplementedError(
                "MaskSet.from_bathymetry_3d: smooth_abs_func_range="
                f"{smooth_abs_func_range!r} is not supported.  Only the "
                "negative branch (SMOOTHABS_RS == 0 ⇒ arithmetic-mean faces, "
                "update_masks_etc.F:398-401) is implemented; pass None for the "
                "default no-USE_SMOOTH_MIN geometry."
            )
        if smooth_geometry and not (hfac_min > 0.0 or hfac_min_dz > 0.0):
            # hFacMnSz appears in a denominator below (exp(-t/m)); MITgcm would
            # divide by zero here too, so refuse instead of emitting NaN.
            raise ValueError(
                "MaskSet.from_bathymetry_3d: smooth_abs_func_range requires a "
                "positive hFacMnSz, i.e. hfac_min > 0 or hfac_min_dz > 0 "
                "(update_masks_etc.F:86)."
            )

        ny, nx = bathymetry.shape
        nz = interface_depth.shape[0] - 1
        bathy_raw = bathymetry.to(dtype=torch.float64, device=device)
        iface = interface_depth.to(dtype=torch.float64, device=device)

        # Apply OBCSfixTopo and the column-floor clamp so that hFacC sees the
        # same bathy the caller's MetricSet sees.  ``obcs_fix_topo_and_clamp``
        # is shared between this method and ``StructuredCGrid.spherical_polar``
        # so both views stay consistent at OB-strip cells.
        column_floor = float(iface[-1].item())
        bathy = obcs_fix_topo_and_clamp(
            bathy_raw,
            column_floor=column_floor,
            boundary_topology=boundary_topology,
            obcs_fix_topo=obcs_fix_topo,
        )

        # --- hFacC: continuous fraction for center cells ---
        # hFacC[k,j,i] = clamp((bathy[j,i] - layer_top[k]) * recip_dz[k], 0, 1)
        # CRITICAL: MIT ``ini_masks_etc.F:112`` precomputes ``recip_drF(k) =
        # 1.0/drF(k)`` and uses MULTIPLICATION (``hFac_loc = (rF(k)-R_low) *
        # recip_drF(k)``).  When (bathy-top) == drF/N for small integer N
        # near the closure threshold (= hFacMin*0.5), the order of operations
        # matters at 1 ULP.  Example: bathy=3151, top=3070, drF=810 →
        #   div  : 81/810           = 0x1.999999999999ap-4 = 0.1   (>= 0.1)
        #   mul  : 81*(1.0/810)     = 0x1.9999999999999p-4 = 0.0999...  (< 0.1)
        # LOCOMgcm's division kept these cells partial (hFacC=0.2); MIT closed them
        # (hFacC=0).  Switching to ``* recip_dz`` reproduces MIT's evaluation
        # order bitwise.  Verified at 4 cells of scs-taiwan-500m-tidal-d4.
        layer_top = iface[:-1].view(nz, 1, 1)   # [nz, 1, 1]
        layer_bot = iface[1:].view(nz, 1, 1)     # [nz, 1, 1]
        dz = (layer_bot - layer_top)              # [nz, 1, 1]
        recip_dz = 1.0 / dz                       # [nz, 1, 1] — match MIT recip_drF

        hfac_raw = (bathy.unsqueeze(0) - layer_top) * recip_dz   # [nz, ny, nx]
        hFacC = hfac_raw.clamp(0.0, 1.0)  # [nz, ny, nx]

        # Effective per-level minimum fraction (MITgcm ini_masks_etc.F:2964):
        #   hFacMnSz = max(hFacMin, min(hFacMinDr * recip_drF(k), 1.0))
        # hfac_min_dz sets a dimensional floor; for thin levels it dominates.
        # ``update_masks_etc.F:86`` repeats this expression verbatim, so the
        # depth-control path below reuses the same hFacMnSz.
        if hfac_min_dz > 0.0:
            hfac_min_eff = torch.clamp(hfac_min_dz * recip_dz, max=1.0)  # [nz,1,1]
            hfac_min_eff = torch.clamp(hfac_min_eff, min=hfac_min)
        else:
            hfac_min_eff = hfac_min  # scalar fallback

        if smooth_geometry:
            # Depth-control hFacC (``update_masks_etc.F:100-107``).  The
            # truncation of the default path is absent here: MITgcm commented
            # it out at :111-122 precisely because the two thresholds are
            # discontinuous in the control.  The three Fortran branches map
            # onto one ``torch.where`` because the taper evaluated at the
            # clamped argument already returns 1 for t > 1:
            #   1 + m*( exp(-1/m) - exp(-1/m) ) == 1 + m*0 == 1  (exactly).
            # The clamp is also what keeps exp(-t/m) finite for the deep-land
            # cells that the ``where`` discards — without it those entries
            # overflow to +inf and poison the backward pass with inf*0 = NaN.
            m = hfac_min_eff
            t = hFacC   # == hfac_raw.clamp(0, 1); clamp passes grad on [0, 1]
            if isinstance(m, torch.Tensor):
                taper = m * (torch.exp(-t / m) - torch.exp(-1.0 / m))
            else:
                taper = m * (torch.exp(-t / m) - math.exp(-1.0 / m))
            hFacC = torch.where(hfac_raw > 0.0, t + taper, torch.zeros_like(t))
        else:
            # hFacMin truncation (MITgcm ini_masks_etc.F:2972-2977):
            # cells with raw hFac < hFacMnSz/2 are CLOSED (set to 0);
            # cells with hFacMnSz/2 <= hFac < hFacMnSz are rounded UP to hFacMnSz.
            too_thin = (hFacC > 0.0) & (hFacC < hfac_min_eff * 0.5)
            hFacC = torch.where(too_thin, torch.zeros_like(hFacC), hFacC)
            partial = (hFacC > 0.0) & (hFacC < hfac_min_eff)
            if isinstance(hfac_min_eff, torch.Tensor):
                hFacC = torch.where(partial, hfac_min_eff.expand_as(hFacC), hFacC)
            else:
                hFacC = torch.where(partial, torch.tensor(hfac_min_eff, dtype=torch.float64, device=device), hFacC)

        # Boolean masks from hFac (preserving existing semantics)
        #
        # Depth-control note: MITgcm does NOT recompute maskC/maskW/maskS in
        # ``update_masks_etc.F`` — every assignment to them there is commented
        # out (:291-312), so the masks stay frozen at the ``ini_masks_etc.F``
        # values built from the *first-guess* bathymetry.  LOCOMgcm's
        # constructor is handed one bathymetry, so it re-derives them from the
        # hFac it just built.  The two disagree on the band
        # ``0 < raw < hFacMnSz/2`` — ALREADY AT THE BASE POINT, not only after
        # the control flips a cell: ``ini_masks_etc.F:113-117`` closes a cell
        # there (so maskC/W/S = 0) while ``update_masks_etc.F:100`` keeps
        # hFacC > 0, its half-threshold being commented out at ``:101``.
        # MITgcm thus carries ``hFacC > 0`` with ``maskC = 0`` on that band;
        # this constructor carries ``maskC = 1``.  Reproducing the frozen masks
        # needs a two-bathymetry API (first guess for the masks, current one
        # for the fractions) and is out of scope here.  Whether the band is
        # populated at all is a property of the bathymetry file: for
        # bottom_ctrl_5x5 (hFacMin=0.05) it means a cell floor within 2.5 m
        # (dz=100) .. 125 m (dz=5000) of its layer top.
        center = hFacC > 0.0  # [nz, ny, nx]

        # --- hFacW: u-face fraction = min(hFacC_left, hFacC_right) ---
        # MITgcm ini_masks_etc.F:244-254:
        #   hFacW(i,j,k) = min(hFacC(i,j,k), hFacC(i-1,j,k))
        # At the domain's first interior column the left operand is the HALO
        # column hFacC(0), which ini_masks_etc.F:110-118 derives in place from
        # the halo ``R_low`` that ini_depths.F:135 filled by EXCH.  That
        # exchange runs on MITgcm's process topology, which is DOUBLY PERIODIC
        # in the default build — independent of tile count and of OBCS:
        #   ini_procs.F:124-125     mpiPeriodicity(1)=(2)=_mpiTRUE_
        #   eeset_parms.F:104-105   notUsingX/YPeriodicity default .FALSE.; they
        #       are absent from NAMELIST /EEPARMS/ (same file :67-73), so
        #       ``eedata`` cannot turn them off
        #   ini_communication_patterns.F:110-120  the tile-level wrap
        #       ``IF (pxW.LT.1) pxW=nPx`` carries no periodicity guard
        # OBCS only overwrites boundary VALUES later in the run; it never
        # re-wires the exchange topology, and hFac is built before it runs.
        # For our interior-only Python tensors this means:
        #   interior i=1..nx-1 : min of left/right wet columns  (unchanged)
        #   i=0 / i=nx on periodic-x OR open-x: the halo column is the opposite
        #       interior column ⇒ min(hFacC[:, :, 0], hFacC[:, :, -1])
        #   i=0 / i=nx on a WALL axis: 0.  A wall has no neighbour column and
        #       passes no flux.  MIT agrees by both of its routes to a wall:
        #       walled-by-land gives hFacC=0 on the outer column so the min is
        #       0; a build with periodicity prevented leaves the halo R_low at
        #       its ini_depths.F:70-76 zero, which yields hFacC(halo)=0 (the
        #       ``R_low.GE.Ro_surf`` closure at ini_masks_etc.F:116-118).
        #       MIT's own "no left neighbour" cell is likewise zeroRS
        #       (ini_masks_etc.F:249).  Copying the adjacent column instead
        #       would incorrectly open a face on a wall boundary.
        #   boundary_topology None / SEAM / DISTRIBUTED: copy the adjacent
        #       column.  The cube builder deliberately relies on this fallback
        #       and then rebuilds both faces cross-seam in
        #       ``_reconcile_cube_seam_hfac``.
        # ``smooth_abs_func_range < 0`` swaps the MIN for the arithmetic mean
        # at the two pair-reduction sites (interior faces and the periodic
        # wrap) via ``_combine_face_hfac``.  The wall branch is 0 either way,
        # and the adjacent-copy branch pairs a column with itself, where
        # ``0.5*(a+a) == a`` holds bitwise — so neither needs a second form.
        hFacW = torch.zeros((nz, ny, nx + 1), dtype=torch.float64, device=device)
        hFacW[:, :, 1:-1] = _combine_face_hfac(
            hFacC[:, :, :-1], hFacC[:, :, 1:], smooth=smooth_geometry
        )
        if boundary_topology is not None and (boundary_topology.periodic_x or boundary_topology.open_x):
            periodic_x_min = _combine_face_hfac(
                hFacC[:, :, 0], hFacC[:, :, -1], smooth=smooth_geometry
            )
            hFacW[:, :, 0] = periodic_x_min
            hFacW[:, :, -1] = periodic_x_min
        elif boundary_topology is not None and boundary_topology.wall_x:
            # solid wall: no neighbour column, no flux
            hFacW[:, :, 0] = 0.0
            hFacW[:, :, -1] = 0.0
        else:
            # no declared topology / seam / distributed: copy adjacent cell
            hFacW[:, :, 0] = hFacC[:, :, 0]
            hFacW[:, :, -1] = hFacC[:, :, -1]

        # --- hFacS: v-face fraction = min(hFacC_south, hFacC_north) ---
        hFacS = torch.zeros((nz, ny + 1, nx), dtype=torch.float64, device=device)
        hFacS[:, 1:-1, :] = _combine_face_hfac(
            hFacC[:, :-1, :], hFacC[:, 1:, :], smooth=smooth_geometry
        )
        if boundary_topology is not None and (boundary_topology.periodic_y or boundary_topology.open_y):
            periodic_y_min = _combine_face_hfac(
                hFacC[:, 0, :], hFacC[:, -1, :], smooth=smooth_geometry
            )
            hFacS[:, 0, :] = periodic_y_min
            hFacS[:, -1, :] = periodic_y_min
        elif boundary_topology is not None and boundary_topology.wall_y:
            hFacS[:, 0, :] = 0.0
            hFacS[:, -1, :] = 0.0
        else:
            hFacS[:, 0, :] = hFacC[:, 0, :]
            hFacS[:, -1, :] = hFacC[:, -1, :]

        # Boolean face masks (wet if hFac > 0).  Under the depth-control
        # geometry ``_combine_face_hfac`` already applies MITgcm's
        # ``maskW``/``maskS`` factor, so the mean never opens a face that the
        # MIN of the SAME (smoothed) centres would have closed.  It does not
        # follow that the face matches MITgcm's frozen ``maskW``: that one is
        # built from the truncated hFacC, and the two thresholds disagree on
        # ``0 < raw < hFacMnSz/2`` — see ``_combine_face_hfac``.
        u_mask = hFacW > 0.0
        v_mask = hFacS > 0.0

        # ------------------------------------------------------------------
        # maskInC / maskInW / maskInS  (MITgcm obcs_init_fixed.F)
        # ------------------------------------------------------------------
        # Start from "wet 2-D mask" (1 if any vertical level is wet, 0 land):
        wet_c_2d = (hFacC.sum(dim=0) > 0.0).to(dtype=torch.float64)       # [ny, nx]
        wet_u_2d = (hFacW.sum(dim=0) > 0.0).to(dtype=torch.float64)       # [ny, nx+1]
        wet_v_2d = (hFacS.sum(dim=0) > 0.0).to(dtype=torch.float64)       # [ny+1, nx]

        maskInC_2d = wet_c_2d.clone()
        maskInW_2d = wet_u_2d.clone()
        maskInS_2d = wet_v_2d.clone()

        # Zero the OB strip rows/cols — ``obcs_init_fixed.F:296-370`` applies
        # one-cell-wide clears along each open-boundary edge.  For OB_Ie=sNx
        # (Python interior i=-1) / OB_Iw=1 (i=0) / OB_Jn=sNy (j=-1) / OB_Js=1
        # (j=0) the effective pattern on the interior slice is:
        #   * maskInC: zero the outermost row/col on each open side
        #   * maskInW: zero entire rows j=0 / j=-1 when open_y, entire cols
        #              i=0 / i=-1 when open_x
        #   * maskInS: symmetric with maskInW (swap i/j)
        if boundary_topology is not None:
            if boundary_topology.open_x:
                maskInC_2d[:, 0] = 0.0
                maskInC_2d[:, -1] = 0.0
                maskInW_2d[:, 0] = 0.0      # west u-face is on the OB strip
                maskInW_2d[:, -1] = 0.0     # east halo u-face (periodic copy)
                maskInS_2d[:, 0] = 0.0
                maskInS_2d[:, -1] = 0.0
            if boundary_topology.open_y:
                maskInC_2d[0, :] = 0.0
                maskInC_2d[-1, :] = 0.0
                maskInW_2d[0, :] = 0.0
                maskInW_2d[-1, :] = 0.0
                maskInS_2d[0, :] = 0.0      # south v-face is on the OB strip
                maskInS_2d[-1, :] = 0.0     # north halo v-face

        return cls(
            center=center, u=u_mask, v=v_mask,
            hFacC=hFacC, hFacW=hFacW, hFacS=hFacS,
            maskInC=maskInC_2d, maskInW=maskInW_2d, maskInS=maskInS_2d,
            smooth_abs_func_range=smooth_abs_func_range,
        )
