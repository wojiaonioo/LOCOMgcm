"""The four offline sea-ice programs, with atomic candidate publication.

THSICE thermodynamics precedes SEAICE dynamics; temperature follows the clock
increment. No program stage infers tile indices or slices horizontal arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace

import torch

from . import dynamics as dyn
from . import thermo_pw79 as pw
from . import thermo_thsice as th
from . import transport as tr
from .forcing import ExfConfig, ExfResult, ForcingSample, exf_step
from .host_single_layer_ocean import (HostParameters, SurfaceForcing, TemperatureStep, initialize_ocean,
                                      ocean_surface, surface_forcing, temperature_step)
from .pickup import NativePickup, restore_carry
from .ports import OceanSurface, PublishedFluxes, open_water_version, publish_fluxes
from .records import (Atmosphere, Clock, DynamicsCarry, Geometry, OceanFluxes, OfflineCarry,
                      Regularization, SeaiceCarry, SeaiceState, ThsiceState,
                      SOUTH_WEST, THSICE_ADVECTED, THSICE_FIELDS)
from .space import SeaiceSpace

# Keep the actual Fortran NINT/two-sided algorithm, shared with dynamics.
different_multiple = dyn.different_multiple


@dataclass(frozen=True)
class ProgramParameters:
    chain: str
    deltaT: float
    host: HostParameters
    dynamics: dyn.DynamicsParameters | None
    pw79: pw.Parameters | None
    thsice: th.Parameters | None
    regularization: tr.RegularizationParameters | None
    thsice_transport: tr.ThsiceParameters | None
    advection_scheme: int
    monitor_frequency: float
    debug_level: int
    debug_threshold: int
    t_ref: float
    s_ref: float

    def __post_init__(self):
        if self.chain not in ("T", "L", "H", "B"):
            raise ValueError(f"unsupported chain {self.chain}")
        required = ((self.chain != "H", self.dynamics, "dynamics"),
                    (self.chain == "T", self.pw79, "PW79"),
                    (self.chain in "HB", self.thsice, "THSICE"),
                    (self.chain in "TL", self.regularization, "regularization"),
                    (self.chain == "B", self.thsice_transport, "THSICE transport"))
        for active, value, name in required:
            if active and value is None:
                raise ValueError(f"{self.chain} requires {name} parameters")
        if self.deltaT <= 0 or self.deltaT != self.host.deltaT:
            raise ValueError("host and ice outer clocks must agree")
        if self.dynamics is not None and self.dynamics.dt_therm != self.deltaT:
            raise ValueError("SEAICE thermodynamic step must equal outer step")
        if self.thsice is not None and any(dt != self.deltaT for dt in (self.thsice.dt, self.thsice.dt_ocean, self.thsice.dt_temperature)):
            raise ValueError("this program requires one THSICE temperature step per outer step")


@dataclass(frozen=True)
class InitialFields:
    """Explicit native inputs with zero halo before I1/I2/I3."""

    uVel: torch.Tensor
    vVel: torch.Tensor
    etaN: torch.Tensor
    HEFF: torch.Tensor
    AREA: torch.Tensor
    HSNOW: torch.Tensor
    iceMask: torch.Tensor
    iceHeight: torch.Tensor
    snowHeight: torch.Tensor
    snowAge: torch.Tensor
    Tsrf: torch.Tensor


@dataclass(frozen=True)
class InitializationResult:
    carry: OfflineCarry
    geometry: Geometry


@dataclass(frozen=True)
class MappedThsice:
    seaice: SeaiceState
    thsice: ThsiceState


@dataclass(frozen=True)
class SeaiceBranch:
    seaice: SeaiceState
    fluxes: OceanFluxes
    regularization: Regularization


@dataclass(frozen=True)
class IceStepResult:
    """One ice step from an ``OceanSurface``; the ocean reads only ``fluxes``."""

    entry_clock: Clock
    candidate: SeaiceCarry
    forcing: ExfResult
    atmosphere_after_exchange: Atmosphere
    thermal: th.ThermalStep | None
    mapped: MappedThsice | None
    dynamics: dyn.DynamicsResult | None
    advected: tr.SeaiceInventory | None
    regularized: tr.RegularizationResult | None
    growth: pw.GrowthResult | None
    seaice_branch: SeaiceBranch | None
    thsice_transport: tr.ThsiceTransportResult | None
    fluxes: PublishedFluxes
    next_forcing_sample: ForcingSample


@dataclass(frozen=True)
class StepResult:
    """An offline step: the ice step, then the host consuming its fluxes."""

    ice: IceStepResult
    surface: SurfaceForcing
    temperature: TemperatureStep
    candidate: OfflineCarry


def initialize(initial: InitialFields, geometry: Geometry, parameters: ProgramParameters,
               *, space: SeaiceSpace) -> InitializationResult:
    """T0/L0/H0/B0: complete initialization, including prescribed exchanges."""
    p = parameters
    if space.chain != p.chain:
        raise ValueError("space and program chain differ")
    for field in fields(initial):
        space.validate(getattr(initial, field.name))
    ocean = initialize_ocean(geometry, initial.uVel, initial.vVel, initial.etaN,
                             space=space, t_ref=p.t_ref, s_ref=p.s_ref)
    zero = torch.zeros_like(initial.HEFF)
    heff, area, snow = zero.clone(), zero.clone(), zero.clone()
    if p.chain in "TL":
        # seaice_init_varia.F:293-352: exchange, clip, then mutual clearing.
        heff = space.exchange("I2", {"HEFF": initial.HEFF}, chain=p.chain)["HEFF"].clamp_min(0.)
        area = space.exchange("I2", {"AREA": initial.AREA}, chain=p.chain)["AREA"].clamp(0., 1.)
        heff = torch.where(area == 0., zero, heff)
        area = torch.where(heff == 0., zero, area)
        snow = space.exchange("I2", {"HSNOW": initial.HSNOW}, chain=p.chain)["HSNOW"].clamp_min(0.)
    si = SeaiceState(zero.clone(), zero.clone(), heff, area, snow, space.full(273., device=zero.device, categories=True))
    ti = ThsiceState(*(zero.clone() for _ in THSICE_FIELDS))
    if p.chain in "HB":
        tp = p.thsice
        # thsice_ini_vars.F:139-151: enthalpy at Tf=-1.70, nonzero area only.
        tf, tm = -1.70, -tp.mu * tp.salinity_internal
        q1 = -tp.cp_water * tm + tp.cp_ice * (tm - tf) + tp.latent * (1. - tm / tf)
        q2 = -tp.cp_ice * tf + tp.latent
        ti = ThsiceState(initial.iceMask, initial.iceHeight, initial.snowHeight,
                         initial.Tsrf, zero.clone(), zero.clone(),
                         torch.where(initial.iceMask != 0, q1, zero),
                         torch.where(initial.iceMask != 0, q2, zero), initial.snowAge)
        ti = ThsiceState(**space.exchange("I3", {f: getattr(ti, f) for f in THSICE_FIELDS}, chain=p.chain))
    ecc = p.dynamics.eccentricity if p.dynamics is not None else 2.
    ecc = torch.as_tensor(ecc, dtype=torch.float64, device=zero.device)
    viscosity = heff * 1.e11
    denominator = torch.where(ecc != 0, ecc * ecc, torch.ones_like(ecc))
    eta = torch.where(ecc != 0, viscosity / denominator, torch.zeros_like(viscosity))
    history = DynamicsCarry(zero.clone(), eta, viscosity)
    load = (ti.snowHeight * p.thsice.rho_snow + ti.iceHeight * p.thsice.rho_ice) * ti.iceMask if p.chain == "B" else zero.clone()
    return InitializationResult(OfflineCarry(SeaiceCarry(si, ti, history, load, Clock(0, 0.)), ocean), geometry)


def restore(pickup: NativePickup, initialized: OfflineCarry, parameters: ProgramParameters,
            *, space: SeaiceSpace) -> OfflineCarry:
    """T9/L11/H9/B10 restart entry, after native R and I3 exchanges."""
    if pickup.chain != parameters.chain:
        raise ValueError("pickup and program chain differ")
    eccentricity = parameters.dynamics.eccentricity if parameters.dynamics is not None else 2.
    tp = parameters.thsice
    return restore_carry(pickup, initialized, space=space, eccentricity=eccentricity,
                         rho_ice=tp.rho_ice if tp is not None else 900., rho_snow=tp.rho_snow if tp is not None else 330.)


def map_thsice(seaice: SeaiceState, thsice: ThsiceState, *, space: SeaiceSpace,
               add_snow_mass: bool) -> MappedThsice:
    """B3: X9 then seaice_model.F:98 / seaice_map_thsice.F:57-71."""
    exchanged = space.exchange("X9", {name: getattr(thsice, name) for name in THSICE_ADVECTED}, chain="B")
    ti = replace(thsice, **exchanged)
    si = replace(seaice, AREA=ti.iceMask.clone(), HEFF=ti.iceMask * ti.iceHeight,
                 HSNOW=ti.iceMask * ti.snowHeight if add_snow_mass else seaice.HSNOW.clone())
    return MappedThsice(si, ti)


def finish_seaice(si: SeaiceState, fluxes: OceanFluxes, regularization: Regularization,
                  parameters: ProgramParameters, *, space: SeaiceSpace) -> SeaiceBranch:
    """T6/L10: optional load, then only the six X8 fields (seaice_model.F:285-335)."""
    if parameters.chain == "L":
        load = si.HEFF * parameters.dynamics.rho_ice + si.HSNOW * parameters.dynamics.rho_snow
        fluxes = replace(fluxes, sIceLoad=space.replace_domain(fluxes.sIceLoad, load))
    values = space.exchange("X8", {"HEFF": si.HEFF, "AREA": si.AREA, "HSNOW": si.HSNOW,
                                   "Qnet": fluxes.Qnet, "EmPmR": fluxes.EmPmR, "saltFlux": fluxes.saltFlux}, chain=parameters.chain)
    si = replace(si, **{name: values[name] for name in ("HEFF", "AREA", "HSNOW")})
    fluxes = replace(fluxes, Qnet=values["Qnet"], EmPmR=values["EmPmR"], saltFlux=values["saltFlux"], AREA=si.AREA)
    return SeaiceBranch(si, fluxes, regularization)


def _dynamics_geometry(g: Geometry, space: SeaiceSpace) -> dyn.DynamicsGeometry:
    zero = torch.zeros_like(g.HEFFM)
    # Cartesian metric terms/tensile strength are zero (seaice_init_fixed.F).
    return dyn.DynamicsGeometry(g.dxF, g.dyF, g.dxC, g.dyC, g.dxV, g.dyU,
                                 g.rA, g.rAw, g.rAs, g.rAz, g.HEFFM, g.maskInC, g.SIMaskU, g.SIMaskV,
                                 g.seaiceMaskU, g.seaiceMaskV, g.maskInW, g.maskInS,
                                 space.y_center(device=zero.device), g.fCori, g.Bo_surf,
                                 *(zero.clone() for _ in range(9)), g.globalArea)


def _thermal_ocean(o: OceanSurface, si: SeaiceState, geometry: Geometry, space: SeaiceSpace) -> th.OceanSample:
    east = lambda x: space.neighbor_field(x, 0, 1, domain=SOUTH_WEST)
    north = lambda x: space.neighbor_field(x, 1, 0, domain=SOUTH_WEST)
    return th.get_ocean(hfac=geometry.hFacC, depth=geometry.drF, theta=o.theta, salt=o.salt,
                         u=o.uVel, u_east=east(o.uVel), v=o.vVel, v_north=north(o.vVel),
                         uice=si.UICE, uice_east=east(si.UICE), vice=si.VICE, vice_north=north(si.VICE),
                         velocity_domain=space.domain_mask(SOUTH_WEST, device=o.theta.device), use_seaice=space.chain == "B")


def advance_ice(carry: SeaiceCarry, ocean: OceanSurface, sample: ForcingSample, exf_config: ExfConfig,
                geometry: Geometry, parameters: ProgramParameters, *, space: SeaiceSpace,
                capture_scans: frozenset[tuple[int, int, int]] = frozenset({(0, 1, 1), (0, 1, 2)})) -> IceStepResult:
    """EXF and every ice stage of one outer step; reads the ocean only through ``ocean``."""
    p, clock = parameters, carry.clock
    if exf_config.chain != p.chain or space.chain != p.chain:
        raise ValueError("inconsistent program/forcing identity")
    if clock.myTime != clock.myIter * p.deltaT:
        raise ValueError("baseTime=0 outer clock mismatch")
    forcing = exf_step(ocean.theta, sample, exf_config, geometry, space, time=clock.myTime, iteration=clock.myIter)
    atmosphere, si, ti = forcing.atmosphere, carry.seaice, carry.thsice
    flux = open_water_version(forcing.open_water, carry.sIceLoad, si.AREA)
    thermal = mapped = dynamic = advected = regularized = growth = branch = transported = None
    owned = space.owned(device=si.HEFF.device)
    if p.chain in "HB":
        atmosphere_th = th.Atmosphere(*(getattr(atmosphere, f.name) for f in fields(th.Atmosphere)))
        thflux = th.OceanFlux(*(getattr(flux, f.name) for f in fields(th.OceanFlux)))
        thermal = th.step_thermodynamics(th.IceState(*(getattr(ti, name) for name in THSICE_FIELDS)),
                                         atmosphere_th, _thermal_ocean(ocean, si, geometry, space), thflux,
                                         p.thsice, owned, advects_ice=p.chain == "B")
        ti = ThsiceState(*(getattr(thermal.ice, name) for name in THSICE_FIELDS))
        flux = replace(flux, **{f.name: getattr(thermal.flux, f.name) for f in fields(thermal.flux)})
        if p.chain == "B":
            mapped = map_thsice(si, ti, space=space, add_snow_mass=p.dynamics.add_snow_mass)
            si, ti = mapped.seaice, mapped.thsice
    history = carry.dynamics
    if p.chain != "H":
        winds = space.exchange("X4", {"uwind": atmosphere.uwind, "vwind": atmosphere.vwind}, chain=p.chain)
        atmosphere = replace(atmosphere, **winds)
        din = dyn.DynamicsForcing(atmosphere.uwind, atmosphere.vwind, flux.fu, flux.fv,
                                  ocean.uVel, ocean.vVel, ocean.etaN, forcing.open_water.pLoad)
        # Only DWATN and the center viscosity bands cross calls. Other fields
        # are overwritten in their mathematical domains before any consumer.
        zero = torch.zeros_like(si.HEFF)
        dhistory = dyn.DynamicsCarry(history.DWATN, history.eta, history.zeta, *(zero.clone() for _ in range(7)))
        dynamic = dyn.dynamics_step(dyn.IceState(si.HEFF, si.AREA, si.HSNOW, si.UICE, si.VICE),
                                    dhistory, din, _dynamics_geometry(geometry, space), p.dynamics, space,
                                    my_iter=clock.myIter, my_time=clock.myTime,
                                    capture_scans=capture_scans,
                                    print_residual=p.debug_level >= p.debug_threshold and different_multiple(p.monitor_frequency, clock.myTime, p.deltaT))
        si = replace(si, UICE=dynamic.UICE, VICE=dynamic.VICE)
        history = DynamicsCarry(dynamic.carry.DWATN, dynamic.carry.eta, dynamic.carry.zeta)
        flux = replace(flux, fu=dynamic.fu, fv=dynamic.fv, AREA=si.AREA)
        if p.chain in "TL":
            advected = tr.seaice_advdiff(tr.SeaiceInventory(si.HEFF, si.AREA, si.HSNOW), si.UICE, si.VICE,
                                         geometry, deltaT=p.deltaT, scheme=p.advection_scheme)
            regularized = tr.seaice_reg_ridge(advected, si.TICES, p.regularization)
            si = replace(si, HEFF=regularized.inventory.HEFF, AREA=regularized.inventory.AREA,
                         HSNOW=regularized.inventory.HSNOW, TICES=regularized.TICES)
            reg = Regularization(si, regularized.d_HEFFbyNEG, regularized.d_HSNWbyNEG)
            if p.chain == "T":
                pin = pw.Inputs(atmosphere.atemp, atmosphere.aqh, atmosphere.lwdown, atmosphere.swdown,
                                atmosphere.wspeed, atmosphere.precip, atmosphere.snowPrecip, atmosphere.evap,
                                atmosphere.runoff, ocean.theta, ocean.salt, geometry.HEFFM,
                                space.y_center(device=si.HEFF.device), flux.Qnet, flux.Qsw, flux.EmPmR, flux.saltFlux,
                                flux.sIceLoad, reg.d_HEFFbyNEG, reg.d_HSNWbyNEG)
                growth = pw.growth(pw.IceState(si.HEFF, si.HSNOW, si.AREA, si.TICES), pin, p.pw79, owned,
                                   salinity=None, swfrac_below_surface=None)
                si = replace(si, HEFF=growth.ice.HEFF, HSNOW=growth.ice.HSNOW, AREA=growth.ice.AREA, TICES=growth.ice.TICES)
                flux = replace(flux, **{name: getattr(growth, name) for name in ("Qnet", "Qsw", "EmPmR", "saltFlux", "sIceLoad")})
            branch = finish_seaice(si, flux, reg, p, space=space)
            si, flux = branch.seaice, branch.fluxes
        else:
            inp = tr.ThsiceFields(*(getattr(ti, f.name) for f in fields(tr.ThsiceFields)))
            ocean = tr.OceanTransportFields(flux.Qnet, flux.EmPmR, flux.saltFlux, flux.sIceLoad)
            transported = tr.thsice_do_advect(inp, si.UICE, si.VICE, geometry.drF * geometry.hFacC,
                                              geometry, p.thsice_transport, ocean, cells=space.nx * space.ny)
            ti = replace(ti, **{f.name: getattr(transported.adjusted.fields, f.name) for f in fields(transported.adjusted.fields)})
            flux = replace(flux, **{f.name: getattr(transported.ocean, f.name) for f in fields(transported.ocean)})
    # forward_step.F:807-808 increments before staggered thermodynamics.
    end = Clock(clock.myIter + 1, (clock.myIter + 1) * p.deltaT, clock.generation)
    published_flux = publish_fluxes(flux, entry_iter=clock.myIter, start=clock.myTime, end=end.myTime, chain=p.chain,
                                    space=space)
    # exf_set_fld.F:122-124: period-zero wind retains the program's X4 halo.
    next_sample = replace(sample, uwind=atmosphere.uwind, vwind=atmosphere.vwind)
    candidate = SeaiceCarry(si, ti, history, flux.sIceLoad, end)
    return IceStepResult(clock, candidate, forcing, atmosphere, thermal, mapped, dynamic, advected,
                         regularized, growth, branch, transported, published_flux, next_sample)


def advance(carry: OfflineCarry, sample: ForcingSample, exf_config: ExfConfig,
            geometry: Geometry, parameters: ProgramParameters, *, space: SeaiceSpace,
            capture_scans: frozenset[tuple[int, int, int]] = frozenset({(0, 1, 1), (0, 1, 2)})) -> StepResult:
    """Evaluate a full offline candidate. ``publish_step`` checks it before publication."""
    if carry.ocean.clock != carry.ice.clock:
        raise ValueError("inconsistent program/host identity")
    ice = advance_ice(carry.ice, ocean_surface(carry.ocean), sample, exf_config, geometry, parameters,
                      space=space, capture_scans=capture_scans)
    surface = surface_forcing(carry.ocean, ice.forcing.open_water.SST, ice.fluxes.values, geometry, parameters.host)
    temperature = temperature_step(carry.ocean, surface, geometry, parameters.host, clock=ice.candidate.clock, space=space)
    return StepResult(ice, surface, temperature, OfflineCarry(ice.candidate, temperature.ocean))


def _published_tensors(value, path, found):
    """Every tensor of a step result with its path, in the order the check reports."""
    if isinstance(value, torch.Tensor):
        found.append((path, value))
    elif is_dataclass(value):
        for field in fields(value):
            _published_tensors(getattr(value, field.name), path + "." + field.name, found)
    elif isinstance(value, (tuple, list)):
        for i, item in enumerate(value):
            _published_tensors(item, f"{path}[{i}]", found)


def _finite_flags(tensors):
    """``isfinite(t).all()`` of every tensor with one host read per device.

    ``t * 0`` is zero exactly where ``t`` is finite and NaN elsewhere, and a sum
    keeps any NaN, so a zero L1 norm of the products is the same test. Integer
    and boolean tensors are always finite.
    """
    flags = [True] * len(tensors)
    groups = {}
    for index, tensor in enumerate(tensors):
        if tensor.is_floating_point() or tensor.is_complex():
            groups.setdefault(tensor.device, {}).setdefault(tensor.dtype, []).append(index)
    for by_dtype in groups.values():
        indices = [i for members in by_dtype.values() for i in members]
        finite = torch.cat([torch.stack(torch._foreach_norm(torch._foreach_mul([tensors[i] for i in members], 0.0), 1)) == 0
                            for members in by_dtype.values()])
        for index, value in zip(indices, finite.tolist()):
            flags[index] = value
    return flags


def _check_finite(value, path="candidate"):
    """Raise for the first nonfinite tensor in traversal order.

    The flags of all tensors are formed on their device and read once, rather
    than one synchronizing read per tensor.
    """
    found = []
    _published_tensors(value, path, found)
    for (name, _), finite in zip(found, _finite_flags([tensor for _, tensor in found])):
        if not finite:
            raise ValueError(f"nonfinite {name}; previous state remains published")


def publish_step(previous: OfflineCarry, result: StepResult) -> OfflineCarry:
    """Publish all components together; LSR iteration caps remain valid results."""
    if previous.ice.clock != result.ice.entry_clock or result.candidate.ocean.clock != result.candidate.ice.clock:
        raise ValueError("stale candidate or mixed component clocks")
    _check_finite(result)
    return result.candidate
