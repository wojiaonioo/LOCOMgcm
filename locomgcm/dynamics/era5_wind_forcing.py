"""Time-varying gridded wind stress forcing as a TendencyModule.

The consumer accepts any source that samples ``(tau_x, tau_y)`` on the model grid.
``stress_location`` says where on the C grid the sampled stress already lives
and which MITgcm reader it follows, which is what differs between producers and
must not be guessed -- see :class:`WindStressLocation`.
"""
from __future__ import annotations

from locomgcm.sampling import SampleClock, SamplingRule

import weakref
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

import torch

from locomgcm.geometry import StructuredCGrid
from locomgcm.geometry.metric_bundle import _positive_reciprocal
from locomgcm.geometry.exchange import FillSpec, HaloExtentV1, exchange
from locomgcm.operators.reconstructions import center_to_u_centered, center_to_v_centered
from locomgcm.state import DiagnosticState, PrognosticState, StateTendency
from locomgcm.constants import RHO_CONST_DEFAULT

from .base import MomentumTermClass, TendencyModule, TendencyStepDomain
from locomgcm.dynamics._registry import register_dynamics_module
from locomgcm.dynamics.wind_stress import (
    local_surface_wind_tendency_from_face_stress,
)

if TYPE_CHECKING:
    from locomgcm.io.era5 import ERA5WindSource
    from .era5_bulk_forcing import ERA5BulkForcing


class WindStressLocation(StrEnum):
    """Where the sampled stress already sits on the C grid, and which MITgcm
    reader it follows.  Each member is one MITgcm path; a face field has two.

    ``CENTER`` -- the source hands over cell-centre stress, which has to be
    reconstructed onto the U/V faces.  This is MITgcm ``pkg/exf`` with
    ``readStressOnAgrid = .TRUE.``: ``exf_mapfields.F:246`` / ``:287`` average
    the two neighbouring centres and multiply the completed average by the face
    mask, with the dry-centre selection done first (``EXF_FILTER_RL``).

    ``C_GRID_FACE`` -- the source hands over stress that is *already* on the
    U/V faces, so it must be placed there verbatim.  This is MITgcm's core
    ``zonalWindFile`` / ``meridWindFile`` path: ``fu`` and ``fv`` are declared
    at the "Southwest C-grid U/V point" (``model/inc/FFIELDS.h:21`` / ``:27``)
    and ``model/src/ini_forcing.F:70-75`` reads the file straight into them --
    no averaging, and no mask (the dry-face zeroing is the ``_recip_hFacW`` /
    ``_recip_hFacS`` factor in ``model/src/apply_forcing.F:143-147``, which the
    tendency below applies).  Averaging such a field would smear it across two
    faces and is a discretisation error, not a style choice.

    ``C_GRID_FACE_EXF`` -- face stress as ``C_GRID_FACE``, read by MITgcm
    ``pkg/exf`` with ``readStressOnCgrid = .TRUE.`` (``stressIsOnCgrid``,
    ``exf_readparms.F:1045``).  EXF then sets the stress masks to ``'w'`` and
    ``'s'`` (``exf_init_fixed.F:63-65``), so ``EXF_FILTER_RL`` zeroes the stress
    on every face whose surface ``maskW`` / ``maskS`` is 0
    (``exf_filter_rl.F:66-81``, called on both time records by
    ``exf_set_fld.F:237,294``), and ``exf_mapfields.F:216-239`` / ``:254-277``
    clip each component to ``windstressmax`` before copying it into ``fu`` /
    ``fv`` unaveraged.  The tendency cannot tell the two face paths apart (a dry
    face has zero ``recip_hFac``), but KPP reads ``surfaceForcingU/V = fu, fv``
    (``external_forcing_surf.F:222``) unmasked and forms ``ustar`` from both faces
    of each column (``kpp_forcing_surf.F:187-211``), so a coastal column sees the
    dry-face stress on the core path and zero on the EXF path.
    """

    CENTER = "center"
    C_GRID_FACE = "c_grid_face"
    C_GRID_FACE_EXF = "c_grid_face_exf"


#: MITgcm ``pkg/exf`` default of ``windstressmax`` [N m-2]
#: (``exf_readparms.F:601``).  ``exf_mapfields.F:216-233`` / ``:254-271`` clip
#: each component of the A-grid stress to ``[-windstressmax, windstressmax]``
#: before the centre-to-face average.  The MIT export writes the module's value
#: to ``data.exf`` (``layer_d/_mit_switches.py`` ``windstressmax_value``), which
#: keeps its own copy of this default and a test ties the two together.
WINDSTRESSMAX_EXF_DEFAULT: float = 2.0


#: Halo extents used to lift a cell-shaped face field to its full face count.
_X_FACE_HALO = HaloExtentV1(
    west_left_extent=1,
    east_right_extent=1,
    south_bottom_extent=0,
    north_top_extent=0,
)
_Y_FACE_HALO = HaloExtentV1(
    west_left_extent=0,
    east_right_extent=0,
    south_bottom_extent=1,
    north_top_extent=1,
)


def _surface_face_masks(
    grid: StructuredCGrid,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return top-level binary maskW/maskS in broadcastable face shapes."""

    def _top_mask(hfac: torch.Tensor | None, binary: torch.Tensor) -> torch.Tensor:
        source = binary if hfac is None else hfac
        if source.ndim == 3:
            source = source[0]
        if source.ndim != 2:
            raise ValueError(f"surface face mask must be 2-D or 3-D, got {source.ndim}-D")
        return (source > 0).to(dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)

    return (
        _top_mask(grid.masks.hFacW, grid.masks.u),
        _top_mask(grid.masks.hFacS, grid.masks.v),
    )


def _face_stress_cache_is_usable(
    tau_x_on_u: torch.Tensor | None,
    tau_y_on_v: torch.Tensor | None,
    cached_grid_ref: weakref.ReferenceType[StructuredCGrid] | None,
    grid: StructuredCGrid,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> bool:
    return (
        tau_x_on_u is not None
        and tau_y_on_v is not None
        and cached_grid_ref is not None
        and cached_grid_ref() is grid
        and tau_x_on_u.shape == (1, 1, grid.ny, grid.nx + 1)
        and tau_y_on_v.shape == (1, 1, grid.ny + 1, grid.nx)
        and tau_x_on_u.dtype == dtype
        and tau_y_on_v.dtype == dtype
        and tau_x_on_u.device == device
        and tau_y_on_v.device == device
    )


@dataclass
@register_dynamics_module("era5_wind_forcing")
class ERA5WindForcing(TendencyModule):
    """Apply time-varying ERA5 wind stress to the top model layer.

    The tendency is:
        du/dt = τˣ(x, y, t) / (ρ₀ · Δz_top)
        dv/dt = τʸ(x, y, t) / (ρ₀ · Δz_top)

    The wind stress fields are linearly interpolated from the
    :class:`ERA5WindSource` at the current model time.

    ``stress_location`` declares where the sampled stress already sits on the
    C grid and which MITgcm reader it follows; the default keeps the historical
    centre-reconstruction path unchanged.  See :class:`WindStressLocation`.

    ``windstressmax`` is MITgcm EXF's clip of the stress, applied to each
    component before the average onto the faces (``CENTER``) or the copy into
    ``fu`` / ``fv`` (``C_GRID_FACE_EXF``) (``exf_mapfields.F:216-233`` /
    ``:254-271``); the default is EXF's 2 N m-2.  ``inf`` (yaml ``.inf``)
    turns the clip off; MITgcm has no such switch, so the MIT export then writes
    the largest double.  The ``C_GRID_FACE`` path is MITgcm's core
    ``zonalWindFile`` path, which does not clip, so the value does not apply
    there.
    """

    sampling_rule = SamplingRule(
        SampleClock.START,
        ("_current_tau_x", "_current_tau_y", "_current_tau_x_on_u", "_current_tau_y_on_v"),
        providers=("source", "bulk_forcing"),
    )

    source: ERA5WindSource | None
    use_bulk_stress: bool = field(default=False, kw_only=True)
    bulk_forcing: ERA5BulkForcing | None = field(default=None, init=False, repr=False)
    reference_density: float = RHO_CONST_DEFAULT
    stress_location: WindStressLocation = WindStressLocation.CENTER
    name: str = "era5_wind_forcing"
    stepper_domain: TendencyStepDomain = TendencyStepDomain.SLOW
    windstressmax: float = WINDSTRESSMAX_EXF_DEFAULT
    momentum_term_class = MomentumTermClass.FORCING
    current_time_seconds: float = field(default=0.0, init=False, repr=False)
    _current_tau_x: torch.Tensor | None = field(default=None, init=False, repr=False)
    _current_tau_y: torch.Tensor | None = field(default=None, init=False, repr=False)
    _current_tau_x_on_u: torch.Tensor | None = field(default=None, init=False, repr=False)
    _current_tau_y_on_v: torch.Tensor | None = field(default=None, init=False, repr=False)
    _face_stress_grid_ref: weakref.ReferenceType[StructuredCGrid] | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        # A non-positive bound has no MITgcm counterpart that matches a clamp:
        # exf_mapfields.F applies the upper test before the lower one, which
        # for windstressmax < 0 maps every value to -windstressmax.
        if isinstance(self.windstressmax, bool) or not isinstance(
            self.windstressmax, (int, float)
        ) or not float(self.windstressmax) > 0.0:
            raise ValueError(
                f"era5_wind_forcing windstressmax={self.windstressmax!r}: expected a "
                "positive stress bound [N m-2], or inf to turn the clip off"
            )
        self.windstressmax = float(self.windstressmax)
        # An unknown spelling raises here instead of running the centre path.
        self.stress_location = WindStressLocation(self.stress_location)
        # Declare the memo's grid reference before tracing so the first
        # face-stress calculation does not change the instance dictionary.
        self._face_stress_grid_ref = None

    def update_model_time(self, time_seconds: float) -> None:
        self.current_time_seconds = float(time_seconds)
        if self.use_bulk_stress:
            if self.bulk_forcing is None:
                raise ValueError("wind-derived stress requires a bound ERA5 bulk producer")
            tau_x_2d, tau_y_2d = None, None
        else:
            tau_x_2d, tau_y_2d = self.source.sample(time_seconds)
        self._current_tau_x = tau_x_2d
        self._current_tau_y = tau_y_2d
        self._current_tau_x_on_u = None
        self._current_tau_y_on_v = None

    def to_runtime(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> "ERA5WindForcing":
        if hasattr(self.source, "to_runtime"):
            self.source = self.source.to_runtime(device, dtype)
        elif hasattr(self.source, "to"):
            self.source = self.source.to(device, dtype)
        kwargs: dict[str, object] = {"device": device}
        if dtype is not None:
            kwargs["dtype"] = dtype
        self._current_tau_x = None if self._current_tau_x is None else self._current_tau_x.to(**kwargs)
        self._current_tau_y = None if self._current_tau_y is None else self._current_tau_y.to(**kwargs)
        self.update_model_time(self.current_time_seconds)
        return self

    def _ensure_face_stress(
        self,
        grid: StructuredCGrid,
        *,
        dtype: torch.dtype,
        device: torch.device,
        state: PrognosticState | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached face stress, producing it only at its declared produce point.

        ``CENTER``: surface ``maskC`` uses selection to match MIT
        ``EXF_FILTER_RL`` conditional dry-cell zeroing, then
        each component is clipped to ``windstressmax`` as ``EXF_MAPFIELDS``
        does before its average, while reconstruction ``face_mask`` remains the
        multiplicative ``EXF_MAPFIELDS`` mask (#1).

        ``C_GRID_FACE``: neither filter nor the clip applies.  ``ini_forcing.F:70-75`` copies
        the file into ``fu``/``fv`` unmasked and unaveraged, so the only step
        left is to widen the ``[ny, nx]`` cell-indexed field to the ``[ny, nx+1]``
        face count.  The extra face is the one MITgcm fills by ``EXCH``, so it is
        taken from this grid's own halo exchange -- periodic axes wrap (which is
        MITgcm's single-tile default, ``eesupp/src/ini_procs.F:124-125``) and
        closed axes replicate.  The tendency ignores that face where
        ``hFacW/S = 0``, but KPP's ``ustar`` of the last column or row reads it;
        on a non-periodic axis MITgcm's single-tile ``EXCH`` still wraps it from
        the first face, so that one column or row differs from MITgcm (on an
        open-boundary edge OBCS then prescribes its state).

        ``C_GRID_FACE_EXF``: before that widening, the faces ``0..nx-1`` /
        ``0..ny-1`` (MITgcm's tile faces ``1..sNx`` / ``1..sNy`` that
        ``EXF_FILTER_RL`` filters) whose surface ``maskW`` / ``maskS`` is 0 are set
        to zero by selection (MIT_SEMANTICS #2), then each component is clipped
        to ``windstressmax``; the halo face is the exchanged filtered value, as
        MITgcm's ``EXCH_UV_XY_RL`` of the filtered stress
        (``exf_getforcing.F:201-203,328-329``).
        """
        if self.use_bulk_stress:
            if state is None:
                raise ValueError("bulk stress requires the step's surface state")
            self.bulk_forcing.compute_surface_fluxes(state)
            self._current_tau_x = self.bulk_forcing._current_flux.ustress
            self._current_tau_y = self.bulk_forcing._current_flux.vstress
        if not _face_stress_cache_is_usable(
            self._current_tau_x_on_u,
            self._current_tau_y_on_v,
            self._face_stress_grid_ref,
            grid,
            dtype=dtype,
            device=device,
        ):
            tau_x_2d = self._current_tau_x
            tau_y_2d = self._current_tau_y
            if tau_x_2d is None or tau_y_2d is None:
                # the explicit-input requirement: no sample means the model clock never
                # reached this module (``update_model_time``, which the model
                # constructor and ``prepare_step_time`` call).  Zero stress
                # would remove the wind forcing without a signal.
                raise RuntimeError(
                    "ERA5WindForcing has no sampled wind stress; call "
                    "update_model_time() (TorchRegionalOceanModel.prepare_step_time) "
                    "before evaluating it"
                )
            tau_x_2d = tau_x_2d.to(dtype=dtype, device=device)
            tau_y_2d = tau_y_2d.to(dtype=dtype, device=device)

            if self.stress_location in (
                WindStressLocation.C_GRID_FACE, WindStressLocation.C_GRID_FACE_EXF
            ):
                tau_x_4d = tau_x_2d.view(1, 1, grid.ny, grid.nx)
                tau_y_4d = tau_y_2d.view(1, 1, grid.ny, grid.nx)
                if self.stress_location == WindStressLocation.C_GRID_FACE_EXF:
                    # EXF_FILTER_RL 'w'/'s' (exf_filter_rl.F:66-81): a selection on the
                    # tile's own faces, so NaN on a dry face does not pass; then the
                    # exf_mapfields.F:216-233 / :254-271 clip, fu = ustress (:234-239).
                    mask_w, mask_s = _surface_face_masks(grid, dtype=dtype, device=device)
                    tau_x_4d = torch.where(
                        mask_w[..., : grid.nx] > 0, tau_x_4d, tau_x_4d.new_zeros(())
                    )
                    tau_y_4d = torch.where(
                        mask_s[..., : grid.ny, :] > 0, tau_y_4d, tau_y_4d.new_zeros(())
                    )
                    bound = float(self.windstressmax)
                    tau_x_4d = tau_x_4d.clamp(min=-bound, max=bound)
                    tau_y_4d = tau_y_4d.clamp(min=-bound, max=bound)
                tau_x_on_u = exchange(
                    tau_x_4d,
                    grid=grid,
                    location="T",
                    extent=_X_FACE_HALO,
                    fill=FillSpec.OBC_VALUE,
                    constant=not tau_x_4d.requires_grad,
                )[..., 1:]
                tau_y_on_v = exchange(
                    tau_y_4d,
                    grid=grid,
                    location="T",
                    extent=_Y_FACE_HALO,
                    fill=FillSpec.OBC_VALUE,
                    constant=not tau_y_4d.requires_grad,
                )[..., 1:, :]
                self._current_tau_x_on_u = tau_x_on_u
                self._current_tau_y_on_v = tau_y_on_v
            else:
                mask_c = grid.masks.center_float(dtype=dtype, device=device)
                if mask_c.ndim == 3:
                    mask_c = mask_c[0]
                if mask_c.ndim != 2:
                    raise ValueError(
                        f"surface center mask must be 2-D or 3-D, got {mask_c.ndim}-D"
                    )
                mask_bool = mask_c.view(1, 1, grid.ny, grid.nx) > 0
                tau_x_4d = torch.where(
                    mask_bool,
                    tau_x_2d.view(1, 1, grid.ny, grid.nx),
                    tau_x_2d.new_zeros(()),
                )
                tau_y_4d = torch.where(
                    mask_bool,
                    tau_y_2d.view(1, 1, grid.ny, grid.nx),
                    tau_y_2d.new_zeros(()),
                )
                # exf_mapfields.F:216-233 / :254-271 clip each component of the
                # (already dry-filtered) A-grid stress before the average at
                # :241-247 / :279-286.  A value selection, so a stress inside the
                # bound keeps its bits, NaN passes, and an infinite bound is the
                # identity.
                bound = float(self.windstressmax)
                tau_x_4d = tau_x_4d.clamp(min=-bound, max=bound)
                tau_y_4d = tau_y_4d.clamp(min=-bound, max=bound)
                mask_w, mask_s = _surface_face_masks(grid, dtype=dtype, device=device)
                # Sampled stress is a constant unless it is a control (rank-uniform).
                self._current_tau_x_on_u = center_to_u_centered(
                    tau_x_4d, grid, face_mask=mask_w, constant=not tau_x_4d.requires_grad
                )
                self._current_tau_y_on_v = center_to_v_centered(
                    tau_y_4d, grid, face_mask=mask_s, constant=not tau_y_4d.requires_grad
                )
            self._face_stress_grid_ref = weakref.ref(grid)

        tau_x_on_u = self._current_tau_x_on_u
        tau_y_on_v = self._current_tau_y_on_v
        assert tau_x_on_u is not None and tau_y_on_v is not None
        return tau_x_on_u, tau_y_on_v

    def evaluate(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
    ) -> StateTendency:
        del diagnostics
        dtype = state.u.data.dtype
        device = state.u.data.device

        # Scale by reference density and top layer thickness
        # MITgcm: gU += surfaceForcingU * recip_drF(1) * recip_hFacW(i,j,1)
        #   where surfaceForcingU = taux / rhoConst
        # The hFac correction ensures partial cells receive the correct
        # forcing per unit volume (thinner cell → stronger tendency).
        top_thickness = grid.layer_thickness[0]
        base_scale = 1.0 / (float(self.reference_density) * top_thickness)

        # Retrieve cached face stress or produce it from center stress.
        tau_x_on_u, tau_y_on_v = self._ensure_face_stress(
            grid, dtype=dtype, device=device, state=state
        )

        u_tendency = torch.zeros_like(state.u.data)
        v_tendency = torch.zeros_like(state.v.data)

        # Apply hFac correction for partial cells (MITgcm: recip_hFacW/S)
        # The reciprocal selects its denominator before dividing (double where):
        # a bare 1/hFac inside a single where leaves 0*inf = NaN in the hFac
        # pullback on dry faces (MITgcm apply_forcing.F:146,336 takes 0 there).
        hFacW = getattr(grid.masks, "hFacW", None)
        hFacS = getattr(grid.masks, "hFacS", None)
        if hFacW is not None:
            hfw = hFacW[0:1] if hFacW.ndim == 3 else hFacW[:, 0:1]
            hfw = hfw.to(dtype=dtype, device=device)
            recip_hfw = _positive_reciprocal(hfw)
            u_tendency[:, 0:1, :, :] = base_scale * tau_x_on_u * recip_hfw
        else:
            u_tendency[:, 0:1, :, :] = base_scale * tau_x_on_u

        if hFacS is not None:
            hfs = hFacS[0:1] if hFacS.ndim == 3 else hFacS[:, 0:1]
            hfs = hfs.to(dtype=dtype, device=device)
            recip_hfs = _positive_reciprocal(hfs)
            v_tendency[:, 0:1, :, :] = base_scale * tau_y_on_v * recip_hfs
        else:
            v_tendency[:, 0:1, :, :] = base_scale * tau_y_on_v

        return StateTendency(
            eta=torch.zeros_like(state.eta.data),
            u=u_tendency,
            v=v_tendency,
            temp=torch.zeros_like(state.temp.data),
            salt=torch.zeros_like(state.salt.data),
        )

    def evaluate_local_surface(
        self,
        state: PrognosticState,
        diagnostics: DiagnosticState,
        grid: StructuredCGrid,
    ) -> StateTendency:
        """Enabled-path forcing placed at each face's first wet model level."""
        del diagnostics
        dtype = state.u.data.dtype
        device = state.u.data.device
        tau_x_on_u, tau_y_on_v = self._ensure_face_stress(
            grid, dtype=dtype, device=device, state=state
        )
        tau_x_on_u = tau_x_on_u[:, 0]
        tau_y_on_v = tau_y_on_v[:, 0]
        hfac_w = grid.masks.hFacW
        hfac_s = grid.masks.hFacS
        hfac_w = (
            torch.ones_like(state.u.data)
            if hfac_w is None
            else hfac_w.to(dtype=dtype, device=device)
        )
        hfac_s = (
            torch.ones_like(state.v.data)
            if hfac_s is None
            else hfac_s.to(dtype=dtype, device=device)
        )
        if hfac_w.ndim == state.u.data.ndim - 1:
            hfac_w = hfac_w.unsqueeze(0)
        if hfac_s.ndim == state.v.data.ndim - 1:
            hfac_s = hfac_s.unsqueeze(0)
        u_tendency, v_tendency = local_surface_wind_tendency_from_face_stress(
            tau_x_on_u,
            tau_y_on_v,
            hfac_w=hfac_w,
            hfac_s=hfac_s,
            layer_thickness=grid.layer_thickness.to(dtype=dtype, device=device),
            reference_density=self.reference_density,
        )
        return StateTendency(
            eta=torch.zeros_like(state.eta.data),
            u=u_tendency.expand_as(state.u.data),
            v=v_tendency.expand_as(state.v.data),
            temp=torch.zeros_like(state.temp.data),
            salt=torch.zeros_like(state.salt.data),
        )
