"""Eager program admission, before loading kernels or detaching initial state."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Mapping

import torch

from .registry import REGISTRY
from .runtime import _REASONS
from .schema import GateView
from .table import CACHE_ENTRIES, COMMUNICATION_IMPLEMENTATIONS, CUDA_COMMUNICATION_IMPLEMENTATIONS, IMPLEMENTATION_TABLE, QUALIFIED_DOMAINS, ImplementationRow


@dataclass(frozen=True)
class ProgramRequirements:
    derivative_mode: Literal["none", "vjp", "jvp"]
    compile_mode: str | None
    world_size: int
    decomposition: str | None
    dtype: torch.dtype
    device_type: str
    grid_family: Literal["latlon", "cube", "seaice_native"]
    control_families: frozenset[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "control_families", frozenset(self.control_families))


@dataclass(frozen=True)
class AdmissionRecord:
    requirements: ProgramRequirements
    requested: Mapping[str, str | None]
    admitted: frozenset[str]
    selected: Mapping[str, ImplementationRow]
    rejections: Mapping[str, str]
    policy: Literal["explicit", "auto"]
    # Keep the complete request provenance as well as its raw spelling. A
    # later environment edit cannot alter a previously admitted program.
    gates: Mapping[str, GateView] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("requested", "selected", "rejections", "gates"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        object.__setattr__(self, "admitted", frozenset(self.admitted))


class AdmissionRejected(RuntimeError):
    def __init__(self, selector: str, missing: str, requirement: str,
                 row: ImplementationRow | None, *, reason: str, detail: str = "") -> None:
        # The dispatch ledger's table (runtime.Reason): a rejection reason feeds the same
        # summaries, and an unknown one is refused where it is made, on every path.
        if reason not in _REASONS:
            raise ValueError(f"unknown admission reason {reason!r}")
        self.selector = selector
        self.missing = missing
        self.requirement = requirement
        self.row = row
        self.reason = reason
        self.detail = detail
        reference = next((candidate.entry for candidate in IMPLEMENTATION_TABLE.get(selector, ())
                          if candidate.backend == "torch_reference"), "a qualified reference implementation")
        super().__init__(f"{reason}: selector={selector}, missing={missing}, "
                         f"requirement={requirement}; reference alternative={reference}. {detail}")


def _reject(row: ImplementationRow, missing: str, requirement: str, reason: str) -> None:
    raise AdmissionRejected(row.selector, missing, requirement, row, reason=reason)


def admit_communication(session, *, dtype, device_type, derivative_mode="vjp"):
    """Agree on FP64 first derivatives using Gloo/CPU or NCCL/CUDA transport."""
    import torch.distributed as dist

    reason = None
    if session is None:
        reason = "an explicit CommSession is required for distributed gradients"
    elif not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
        reason = "distributed derivative admission requires enabled gradients"
    elif derivative_mode != "vjp":
        reason = "communication is qualified for first reverse derivatives"
    elif dtype != torch.float64 or (device_type, dist.get_backend(session.group)) not in (
        ("cpu", "gloo"), ("cuda", "nccl")
    ):
        reason = "communication qualification covers FP64 Gloo/CPU and NCCL/CUDA"
    elif session.group is not dist.group.WORLD:
        reason = "runner communication requires the model's complete participant group"
    # The session's control domain owns the agreement, so a rejection reaches
    # exactly its members; without a session the participants are the job.
    local = (reason, str(dtype), device_type, derivative_mode)
    if session is None or session.control_group is None:
        records = [None] * dist.get_world_size()
        dist.all_gather_object(records, local)
    else:
        from defgcm.distributed.transport_failure import transport_operation

        records = [None] * dist.get_world_size(session.control_group)
        with transport_operation(session.control_group_name):
            dist.all_gather_object(records, local, group=session.control_group)
    if any(record[0] is not None or record != records[0] for record in records):
        raise AdmissionRejected("comm_session", "qualified_transport", "distributed_derivative", None,
                                reason="UNQUALIFIED_CAPABILITY", detail=repr(records))
    return (COMMUNICATION_IMPLEMENTATIONS if device_type == "cpu"
            else CUDA_COMMUNICATION_IMPLEMENTATIONS)


def _check(row: ImplementationRow, req: ProgramRequirements) -> None:
    spec = REGISTRY[row.selector]
    if req.grid_family not in spec.grid_families:
        _reject(row, "layout", "grid_family", "UNSUPPORTED_LAYOUT")
    mode = "multi" if req.world_size > 1 else "single"
    if mode not in spec.distributed:
        _reject(row, "topology", "world_size/decomposition", "UNSUPPORTED_TOPOLOGY")
    if row.backend != "torch_reference" and req.device_type != "cuda":
        _reject(row, "device", "device_type", "UNSUPPORTED_DEVICE")
    if req.derivative_mode != "none":
        if row.tier == "B" or row.backward_source == "none":
            # spelt out so that the contract scan can check both against the reason table
            _reject(row, req.derivative_mode, "derivative_mode",
                    "UNSUPPORTED_VJP" if req.derivative_mode == "vjp" else "UNSUPPORTED_JVP")
        # A partial RHS backward does not supply coefficient cotangents. The
        # ivadv native row additionally needs a nonsymmetric band ROOT entry;
        # the existing SPD checked_root is not a substitute.
        if row.tier == "C" and "physical" in req.control_families:
            _reject(row, "coefficient_adjoint", "control_families", "UNSUPPORTED_CONTROL_SET")
        if req.control_families - {"static"} and any(
            entry.admission == "reject_controls" for entry in CACHE_ENTRIES.get(row.selector, ())
        ):
            _reject(row, "control_payload", "control_families", "UNSUPPORTED_CONTROL_SET")
    axes = [("forward", "device_type/dtype/grid_family")]
    if req.derivative_mode != "none":
        axes.append((req.derivative_mode, "derivative_mode"))
    if req.compile_mode is not None:
        axes.append(("compile", "compile_mode"))
    if req.world_size > 1:
        axes.append(("multi_rank", "world_size/decomposition"))
    for axis, requirement in axes:
        if getattr(row, axis) != "PASS" or not row.evidence.get(axis):
            _reject(row, axis, requirement, "UNQUALIFIED_CAPABILITY")
    domain = QUALIFIED_DOMAINS.get(row.selector)
    if domain is None:
        _reject(row, "qualification_domain", "device_type/dtype/grid_family", "UNQUALIFIED_CAPABILITY")
    devices, dtype, layout, controls = domain
    if req.device_type not in devices:
        _reject(row, "device", "device_type", "UNSUPPORTED_DEVICE")
    if str(req.dtype) != dtype:
        _reject(row, "dtype", "dtype", "UNSUPPORTED_DTYPE")
    if req.grid_family != layout:
        _reject(row, "layout", "grid_family", "UNSUPPORTED_LAYOUT")
    if req.derivative_mode != "none" and not req.control_families <= controls:
        _reject(row, "controls", "control_families", "UNSUPPORTED_CONTROL_SET")


def _local_admission(req: ProgramRequirements, gates: Mapping[str, GateView],
                     policy: Literal["explicit", "auto"]) -> AdmissionRecord:
    selected: dict[str, ImplementationRow] = {}
    rejections: dict[str, str] = {}
    active = [key for key, view in gates.items() if view.effective_value]
    if req.derivative_mode not in ("none", "vjp", "jvp"):
        raise ValueError("derivative_mode must be none, vjp or jvp")
    if req.derivative_mode != "none" and (not torch.is_grad_enabled() or torch.is_inference_mode_enabled()):
        key = active[0] if active else "program"
        row = IMPLEMENTATION_TABLE[key][-1] if active else None
        raise AdmissionRejected(key, req.derivative_mode, "derivative_mode/outer_grad_mode", row,
                                reason="INVALID_GRAD_MODE")
    for key in active:
        rows = IMPLEMENTATION_TABLE[key]
        candidate = rows[-1]
        if gates[key].raw_spelling is None and req.grid_family not in REGISTRY[key].grid_families:
            # A library default is not a request: a gate declared for another grid
            # family stays out of this program instead of rejecting it.
            rejections[key] = "UNSUPPORTED_LAYOUT"
            continue
        try:
            _check(candidate, req)
        except AdmissionRejected as exc:
            if policy == "explicit" or candidate.backend == "torch_reference":
                raise
            reference = next(row for row in rows if row.backend == "torch_reference")
            _check(reference, req)
            rejections[key] = exc.reason
            candidate = reference
        selected[key] = candidate
    return AdmissionRecord(req, {key: view.raw_spelling for key, view in gates.items()},
                           frozenset(selected), selected, rejections, policy, gates)


def _signature(record: AdmissionRecord) -> tuple:
    req = record.requirements
    requirements = (req.derivative_mode, req.compile_mode, req.world_size, req.decomposition,
                    str(req.dtype), req.device_type, req.grid_family, tuple(sorted(req.control_families)))
    choices = tuple(sorted((key, row.operator, row.entry, row.tier, row.backend,
                            row.backward_source) for key, row in record.selected.items()))
    return requirements, record.policy, choices, tuple(sorted(record.rejections.items()))


def admit(requirements: ProgramRequirements, gates: Mapping[str, GateView], *,
          policy: Literal["explicit", "auto"] = "explicit", group=None) -> AdmissionRecord:
    """Resolve one immutable program; only auto may select a qualified reference.

    Multi-rank validation exchanges control-plane records before any model halo
    or solver communication. Local rejection is gathered too, so one rank does
    not throw while a peer starts its first scientific collective.  ``group``
    is the control domain of the participants (a session's control group);
    None means the whole job.
    """
    if policy not in ("explicit", "auto"):
        raise ValueError("policy must be explicit or auto")
    record = None
    rejection = None
    try:
        record = _local_admission(requirements, gates, policy)
    except AdmissionRejected as exc:
        rejection = exc
    import torch.distributed as dist

    group_size = dist.get_world_size(group) if dist.is_initialized() else 1
    if requirements.world_size != group_size:
        rejection = AdmissionRejected("program", "process_group", "world_size", None,
                                      reason="UNSUPPORTED_TOPOLOGY")
    # The initialized group, rather than a rank-local declared size, owns the
    # agreement collective. A wrong declaration must travel to every peer too.
    if group_size > 1:
        local = ("rejected", str(rejection)) if rejection is not None else ("selected", _signature(record))
        records = [None] * group_size
        dist.all_gather_object(records, local, group=group)
        if any(item[0] == "rejected" for item in records) or any(item != records[0] for item in records[1:]):
            raise AdmissionRejected("program", "uniform_selection", "world_size/decomposition", None,
                                    reason="UNSUPPORTED_TOPOLOGY", detail=repr(records))
    if rejection is not None:
        raise rejection
    assert record is not None
    return record
