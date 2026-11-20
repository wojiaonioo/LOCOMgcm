"""Per-column input features of the learned KPP closure.

The features are the inputs of KPP (Large, McWilliams & Doney 1994) evaluated
from one water column, its surface forcing and its geometry, made dimensionless
with fixed reference scales rather than data statistics.

Conventions (docs/NUMERICS.md): layer k = 0 is the surface layer and depths
are positive down; interface k is the top of layer k, so interface 0 is the sea
surface and interface nz the bottom of the last layer.  Surface forcing is in
the MITgcm convention consumed by the reference KPP (``RegionalKppInputs``,
kpp_mixing.py:361-391): ``sf_t > 0`` warms the ocean, ``sf_s > 0`` adds salt,
``sf_u``/``sf_v`` are tau/rho0 on the u/v faces, ``qsw`` is the MITgcm
upward-positive net shortwave.  No w or bathymetry enters.

Buoyancy differences are those of STATEKPP in the reference implementation
(:func:`defgcm.dynamics.kpp.kpp_statekpp`), with its pressure choice
(kpp.py:1597-1601): the jump across interface k compares layer k with layer k-1
moved to the pressure of layer k, so the compressibility of sea water does not
enter N^2; the in-situ density divides, as in KPP.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from defgcm.dynamics.kpp import (
    jmd95z_layer_pressure,
    kpp_statekpp,
    kpp_surface_buoyancy_forcing,
)
from defgcm.operators import average_u_to_center, average_v_to_center

# Channels of one interface, in their order inside ``ColumnFeatures.x``.
INTERFACE_CHANNELS = (
    "n2",       # asinh(N^2 / 1e-5 s^-2)
    "s2",       # asinh(S^2 / 1e-5 s^-2), shear of the cell-centred velocity
    "ri_g",     # atan(N^2 / (S^2 + 1e-10))
    "ri_b",     # atan(bulk Richardson number against the surface layer)
    "depth",    # interface depth / 100 m
    "dz_c",     # distance between the two layer centres / 10 m
    "dz_f",     # thickness of the layer below the interface / 10 m
    "zeta",     # atan(depth / L), L = u*^3 / (kappa B_f)
    "ekman",    # asinh(depth |f| / u*)
    "wet",      # maskC(k) * maskC(k-1)
)
# Column scalars, appended once after the interface block.
COLUMN_CHANNELS = (
    "ustar",         # asinh(u* / 1e-2 m/s)
    "b_f",           # asinh(B_f / 1e-7 m^2/s^3), turbulent surface buoyancy flux
    "b_sol",         # asinh(B_sol / 1e-7 m^2/s^3), shortwave buoyancy flux
    "unstable",      # [B_f < 0]
    "coriolis",      # f / 1e-4 s^-1
    "wet_fraction",  # number of wet layers / nz
)

_N2_SCALE = 1.0e-5          # s^-2
_S2_SCALE = 1.0e-5          # s^-2
_DEPTH_SCALE = 100.0        # m
_DZ_SCALE = 10.0            # m
_USTAR_SCALE = 1.0e-2       # m/s
_BUOYANCY_FLUX_SCALE = 1.0e-7  # m^2/s^3
_CORIOLIS_SCALE = 1.0e-4    # 1/s
_DIV_GUARD = 1.0e-10
# Added to (tau/rho0)^2 under the fourth root: u* >= 1e-5 m/s, so u*^-3 stays
# finite and the root is differentiable at zero stress.  The reference floors
# u* at sqrt(0.5 * phepsi * drF(1)) instead (kpp.py:308-319), 2.2e-5 m/s for
# a 10 m top layer.
_USTAR4_FLOOR = 1.0e-20


@dataclass(frozen=True)
class ColumnFeatures:
    """Network input of every column and the masks the closure applies.

    ``x`` holds, per column, the interface block (interfaces 1..nz-1 in order,
    ``INTERFACE_CHANNELS`` for each) followed by ``COLUMN_CHANNELS``.
    """

    x: torch.Tensor          # (B, ny, nx, D)
    wet_iface: torch.Tensor  # (B, nz+1, ny, nx), maskC(k)*maskC(k-1); 0 at interfaces 0 and nz
    unstable: torch.Tensor   # (B, ny, nx), 1 where B_f < 0
    n_iface: int             # interior interfaces, nz - 1
    dim: int                 # D = (nz - 1) * len(INTERFACE_CHANNELS) + len(COLUMN_CHANNELS)


def _require_shape(name: str, tensor: torch.Tensor, shape: tuple[int, ...]) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(tensor.shape)}, expected {shape}")


def column_features(
    temp: torch.Tensor,
    salt: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    sf_t: torch.Tensor,
    sf_s: torch.Tensor,
    sf_u: torch.Tensor,
    sf_v: torch.Tensor,
    qsw: torch.Tensor | None,
    *,
    eos: str,
    eos_pressure_pa: torch.Tensor | None,
    drf: torch.Tensor,
    mask_c: torch.Tensor,
    coriolis: torch.Tensor,
    rho0: float,
    gravity: float,
    cp: float,
    kappa: float = 0.4,
) -> ColumnFeatures:
    """Dimensionless KPP inputs of every column.

    Shapes: ``temp``/``salt`` (B, nz, ny, nx); ``u`` (B, nz, ny, nx+1) and
    ``v`` (B, nz, ny+1, nx) with both outer faces, as in the LOCOMgcm state
    (on a periodic axis the far face is the image of the near one); ``sf_t``,
    ``sf_s``, ``qsw`` (B, ny, nx); ``sf_u`` (B, ny, nx+1); ``sf_v``
    (B, ny+1, nx); ``drf`` (nz,) positive layer thickness; ``mask_c``
    (nz, ny, nx) or (B, nz, ny, nx) wet-cell mask; ``coriolis`` (ny, nx) or
    (B, ny, nx) in 1/s.

    ``eos`` selects the density and expansion-coefficient kernels
    (``"mdjwf"`` or ``"jmd95"``/``"jmd95z"``/``"jmd95p"``).
    ``eos_pressure_pa`` (B, nz, ny, nx) in Pa is the pressure the reference KPP
    receives from the stepper; ``None`` selects rho0 * g * z_c, as the reference
    driver does (kpp.py:1597-1601).  ``qsw=None`` means no shortwave, B_sol = 0,
    as in kpp.py:659-666.

    Definitions at interface k = 1..nz-1 (layer centres z_c, interface depth d):

    * N^2 = dbloc[k-1] / (z_c[k] - z_c[k-1]), dbloc from ``kpp_statekpp``;
    * S^2 = |V[k-1] - V[k]|^2 / (z_c[k] - z_c[k-1])^2, V the u/v averaged to
      the cell centre;
    * Ri_b = (z_c[k] - z_c[0]) * dbsfc[k] / (|V[0] - V[k]|^2 + 1e-10): the
      reference layer is the surface layer and the depth factor is the Ritop
      scale zgrid(1) - zgrid(k) of the reference KPP (kpp.py:1636-1638),
      because dbsfc and V are layer-centre values; the unresolved shear
      Vt^2 of KPP is not included;
    * zeta = d * kappa * B_f / u*^3 and the Ekman ratio d |f| / u*, with the
      interface depth d; |f| because the Ekman depth 0.7 u*/|f| of KPP uses
      |f| (kpp.py:1016), the sign of f is in the ``coriolis`` column scalar.

    u* = ((tau_u/rho0)^2 + (tau_v/rho0)^2 + 1e-20)^(1/4) from the stresses
    averaged to the cell centre, which equals the reference expression
    (kpp.py:296-319) apart from its floor.  B_f and B_sol are ``bo`` and
    ``bosol`` of ``kpp_surface_buoyancy_forcing`` (kpp.py:641-671) with the
    surface-layer alpha, beta and density of STATEKPP; B_f < 0 is destabilising.

    Every feature of a dry interface is set to 0 by selection (``torch.where``).
    All operations are differentiable; ``unstable`` is a 0/1 gate and carries
    no gradient.
    """
    if temp.ndim != 4:
        raise ValueError(f"temp must be (B, nz, ny, nx), got shape {tuple(temp.shape)}")
    batch, nz, ny, nx = temp.shape
    if nz < 2:
        raise ValueError("column_features needs nz >= 2 (at least one interior interface)")
    _require_shape("salt", salt, (batch, nz, ny, nx))
    _require_shape("u", u, (batch, nz, ny, nx + 1))
    _require_shape("v", v, (batch, nz, ny + 1, nx))
    _require_shape("sf_t", sf_t, (batch, ny, nx))
    _require_shape("sf_s", sf_s, (batch, ny, nx))
    _require_shape("sf_u", sf_u, (batch, ny, nx + 1))
    _require_shape("sf_v", sf_v, (batch, ny + 1, nx))
    if qsw is not None:
        _require_shape("qsw", qsw, (batch, ny, nx))
    _require_shape("drf", drf, (nz,))
    dtype, device = temp.dtype, temp.device
    drf = drf.to(dtype=dtype, device=device)
    mask = mask_c.to(dtype=dtype, device=device).expand(batch, nz, ny, nx)
    f = coriolis.to(dtype=dtype, device=device).expand(batch, ny, nx)

    # Buoyancy differences and surface-layer expansion coefficients (STATEKPP).
    pressure = (
        eos_pressure_pa
        if eos_pressure_pa is not None
        else jmd95z_layer_pressure(drf, gravity=gravity, rho_const=rho0)
    )
    dbloc, dbsfc, rho_surf, alpha, beta = kpp_statekpp(
        temp, salt, pressure, gravity=gravity, rho_const=rho0, eos_type=eos,
        # the form these features are computed with
        dbloc_form="divided_difference",
    )
    b_f, b_sol = kpp_surface_buoyancy_forcing(
        alpha[:, 0], beta[:, 0], sf_t, sf_s, rho_surf,
        gravity=gravity, qsw=qsw, heat_capacity_cp=cp, rho_const=rho0,
    )

    # Vertical geometry, positive down; interior interfaces 1..nz-1.
    bottom = torch.cumsum(drf, 0)
    centre = bottom - 0.5 * drf
    iface_shape = (1, nz - 1, 1, 1)
    depth = (bottom - drf)[1:].reshape(iface_shape)
    dz_c = (centre[1:] - centre[:-1]).reshape(iface_shape)
    dz_f = drf[1:].reshape(iface_shape)
    ritop_scale = (centre[1:] - centre[0]).reshape(iface_shape)

    # Stratification and shear.
    u_c = average_u_to_center(u)
    v_c = average_v_to_center(v)
    n2 = dbloc[:, :-1] / dz_c
    du = u_c[:, :-1] - u_c[:, 1:]
    dv = v_c[:, :-1] - v_c[:, 1:]
    s2 = (du * du + dv * dv) / (dz_c * dz_c)
    ri_g = n2 / (s2 + _DIV_GUARD)
    du0 = u_c[:, :1] - u_c[:, 1:]
    dv0 = v_c[:, :1] - v_c[:, 1:]
    ri_b = ritop_scale * dbsfc[:, 1:] / (du0 * du0 + dv0 * dv0 + _DIV_GUARD)

    # Surface similarity scales.
    tau_u = average_u_to_center(sf_u)
    tau_v = average_v_to_center(sf_v)
    ustar = (tau_u * tau_u + tau_v * tau_v + _USTAR4_FLOOR) ** 0.25
    zeta = depth * (kappa * b_f / (ustar * ustar * ustar)).unsqueeze(1)
    ekman = depth * (f.abs() / ustar).unsqueeze(1)

    wet_inner = mask[:, 1:] * mask[:, :-1]
    iface_full = (batch, nz - 1, ny, nx)
    physical = torch.stack(
        (
            torch.asinh(n2 / _N2_SCALE),
            torch.asinh(s2 / _S2_SCALE),
            torch.atan(ri_g),
            torch.atan(ri_b),
            (depth / _DEPTH_SCALE).expand(iface_full),
            (dz_c / _DZ_SCALE).expand(iface_full),
            (dz_f / _DZ_SCALE).expand(iface_full),
            torch.atan(zeta),
            torch.asinh(ekman),
        ),
        dim=-1,
    )
    wet = (wet_inner > 0).unsqueeze(-1)
    physical = torch.where(wet, physical, torch.zeros_like(physical))
    iface = torch.cat((physical, wet_inner.unsqueeze(-1)), dim=-1)
    iface = iface.permute(0, 2, 3, 1, 4).reshape(
        batch, ny, nx, (nz - 1) * len(INTERFACE_CHANNELS)
    )

    unstable = (b_f < 0).to(dtype)
    scalars = torch.stack(
        (
            torch.asinh(ustar / _USTAR_SCALE),
            torch.asinh(b_f / _BUOYANCY_FLUX_SCALE),
            torch.asinh(b_sol / _BUOYANCY_FLUX_SCALE),
            unstable,
            f / _CORIOLIS_SCALE,
            mask.sum(dim=1) / nz,
        ),
        dim=-1,
    )
    x = torch.cat((iface, scalars), dim=-1)

    end = wet_inner.new_zeros(batch, 1, ny, nx)
    wet_iface = torch.cat((end, wet_inner, end), dim=1)
    return ColumnFeatures(
        x=x, wet_iface=wet_iface, unstable=unstable, n_iface=nz - 1, dim=x.shape[-1],
    )
