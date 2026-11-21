"""The dynamics gate and wind/LSR/ocean-stress sequence."""

from dataclasses import fields, replace
import math

import torch

from .lsr import solve_lsr
from .physics import ocean_stress, prepare_dynamics, wind_stress
from .types import (
    DynamicsCarry, DynamicsForcing, DynamicsGeometry, DynamicsParameters,
    DynamicsResult, DynamicsSpace, IceState, Record, WindStress, record,
)


@torch.compiler.disable(reason="host clock test; a traced NINT would specialize each step")
def different_multiple(frequency: float, value: float, step: float) -> bool:
    """different_multiple.F:41-61, including Fortran NINT ties away from zero.

    Compiled entries carry the clock as a CPU scalar tensor; the test reads it
    as the same float, so both execution modes evaluate identical arithmetic.
    """
    frequency, value, step = float(frequency), float(value), float(step)
    if frequency == 0.0:
        return False
    if abs(step) > frequency:
        return True
    ratio = value / frequency
    nearest = math.floor(ratio + 0.5) if ratio >= 0.0 else math.ceil(ratio - 0.5)
    multiple = nearest * frequency
    d1, d2, d3 = value - multiple, (value - step) - multiple, (value + step) - multiple
    return abs(d1) < abs(d2) and abs(d1) <= abs(d3)


@torch.compiler.disable(recursive=False, reason="host dynamics gate; wind stress remains compilable")
def wind_and_gate(ice: IceState, forcing: DynamicsForcing, g: DynamicsGeometry,
                  p: DynamicsParameters, space: DynamicsSpace, *, my_iter: int,
                  my_time: float, gate_sequence: int = 1) -> tuple[WindStress, Record]:
    """B4 boundary: seaice_dynsolver.F:130-137, before entering dynamics."""
    wind = wind_stress(ice, forcing, g, p, space)
    gate = p.use_dynamics and different_multiple(float(p.dt_dyn), my_time, p.dt_therm)
    return wind, record((gate_sequence, my_iter, int(gate), int(gate)), (my_time,), ice.UICE)


@torch.compiler.disable(recursive=False, reason="host dynamics sequence; called stages remain compilable")
def dynamics_step(ice: IceState, carry: DynamicsCarry, forcing: DynamicsForcing,
                   geometry: DynamicsGeometry, parameters: DynamicsParameters,
                   space: DynamicsSpace, *, my_iter: int, my_time: float,
                   print_residual: bool, gate_sequence: int = 1,
                   capture_scans: frozenset[tuple[int, int, int]] = frozenset({(0, 1, 1), (0, 1, 2)})) -> DynamicsResult:
    """T2/L8/B6 candidate boundary, seaice_dynsolver.F:130-386.

    Wind must already have undergone X4. No inventories or caller-owned
    arrays are modified. print_residual is the frozen clock/debug gate, not
    a logging preference. The final DWATN is never recomputed from final U/V.
    """
    for obj in (ice, carry, forcing, geometry):
        for item in fields(obj):
            value = getattr(obj, item.name)
            if not isinstance(value, torch.Tensor) or value.dtype != torch.float64:
                raise ValueError(f"{item.name} must be a float64 tensor")
            expected = () if item.name == "globalArea" else space.shape
            if tuple(value.shape) != expected:
                raise ValueError(f"{item.name} must have shape {expected}")
    wind, gate = wind_and_gate(ice, forcing, geometry, parameters, space,
                               my_iter=my_iter, my_time=my_time, gate_sequence=gate_sequence)
    prepared = lsr = None
    candidate = ice
    if bool(gate.integers[3]):
        prepared = prepare_dynamics(ice, forcing, wind, geometry, parameters, space)
        lsr = solve_lsr(ice, carry, prepared, forcing, geometry, parameters, space,
                        my_iter=my_iter, print_residual=print_residual, capture_scans=capture_scans)
        carry = lsr.carry
        candidate = replace(ice, UICE=lsr.UICE, VICE=lsr.VICE)
    fu, fv = ocean_stress(candidate, forcing, carry, geometry, parameters, space)
    return DynamicsResult(wind, gate, prepared, lsr, candidate.UICE, candidate.VICE, fu, fv, carry)
