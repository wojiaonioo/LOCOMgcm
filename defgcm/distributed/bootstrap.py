"""Runtime bootstrap for LOCOMgcm distributed execution.

The single-process path intentionally does nothing: callers can invoke
``init_distributed`` unconditionally and receive ``None`` when no torchrun or
SLURM rank environment is present.
"""

from __future__ import annotations

import logging

from dataclasses import dataclass
from typing import Mapping

import os

import torch
import torch.distributed as dist

from defgcm.accel import cuda_graphs


@dataclass(frozen=True)
class LaunchEnvironment:
    """Rank metadata detected from torchrun or SLURM environment variables."""

    rank: int
    local_rank: int
    world_size: int
    source: str
    master_addr: str | None = None
    master_port: str | None = None


@dataclass(frozen=True)
class DistributedContext:
    """Initialized distributed process context."""

    rank: int
    local_rank: int
    world_size: int
    backend: str
    device: torch.device
    source: str
    owns_process_group: bool = False
    nccl_user_buffer_registration: str = "not-applicable"

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1


def detect_launch_environment(
    environ: Mapping[str, str] | None = None,
) -> LaunchEnvironment | None:
    """Detect torchrun or direct SLURM launch metadata.

    Returns ``None`` for the ordinary single-process path.  The parser is pure
    and side-effect free so it can be unit-tested without initializing
    ``torch.distributed``.
    """

    env = os.environ if environ is None else environ
    if "RANK" in env and "WORLD_SIZE" in env:
        world_size = _read_int(env, "WORLD_SIZE")
        if world_size <= 1:
            return None
        rank = _read_int(env, "RANK")
        local_rank = _read_int(env, "LOCAL_RANK", default=rank)
        return LaunchEnvironment(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            source="torchrun",
            master_addr=env.get("MASTER_ADDR"),
            master_port=env.get("MASTER_PORT"),
        )

    if "SLURM_PROCID" in env and "SLURM_NTASKS" in env:
        world_size = _read_int(env, "SLURM_NTASKS")
        if world_size <= 1:
            return None
        rank = _read_int(env, "SLURM_PROCID")
        local_rank = _read_int(env, "SLURM_LOCALID", default=rank)
        return LaunchEnvironment(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            source="slurm",
            master_addr=env.get("MASTER_ADDR") or env.get("SLURM_LAUNCH_NODE_IPADDR"),
            master_port=env.get("MASTER_PORT"),
        )

    return None


def init_distributed(
    *,
    backend: str | None = None,
    init_method: str = "env://",
) -> DistributedContext | None:
    """Initialize ``torch.distributed`` when rank environment is present.

    Defaults to NCCL on CUDA hosts and Gloo otherwise.  Production multi-GPU
    runs should use NCCL; the Gloo fallback keeps CPU smoke tests possible.
    """

    launch = detect_launch_environment()
    if launch is None:
        return None

    backend_name = backend or ("nccl" if torch.cuda.is_available() else "gloo")
    _seed_env_for_init(launch)

    if backend_name == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL backend requested but CUDA is not available")
        if launch.local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={launch.local_rank} exceeds visible CUDA devices "
                f"({torch.cuda.device_count()})"
            )
        torch.cuda.set_device(launch.local_rank)
        device = torch.device("cuda", launch.local_rank)
    else:
        device = torch.device("cpu")

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build")
    process_group_initialized = dist.is_initialized()
    if process_group_initialized:
        backend_name = dist.get_backend()
    nccl_user_buffer_registration = _configure_nccl_user_buffer_registration(
        backend_name, process_group_initialized=process_group_initialized
    )
    owns_process_group = False
    if not process_group_initialized:
        dist.init_process_group(
            backend=backend_name,
            init_method=init_method,
            rank=launch.rank,
            world_size=launch.world_size,
        )
        owns_process_group = True

    rank = dist.get_rank()
    if rank == 0:
        logging.getLogger(__name__).info(
            "init_distributed nccl_user_buffer_registration=%s "
            "NCCL_GRAPH_REGISTER=%s NCCL_LOCAL_REGISTER=%s",
            nccl_user_buffer_registration,
            os.environ.get("NCCL_GRAPH_REGISTER", "<unset>"),
            os.environ.get("NCCL_LOCAL_REGISTER", "<unset>"),
        )
    return DistributedContext(
        rank=rank,
        local_rank=launch.local_rank,
        world_size=dist.get_world_size(),
        backend=backend_name,
        device=device,
        source=launch.source,
        owns_process_group=owns_process_group,
        nccl_user_buffer_registration=nccl_user_buffer_registration,
    )


def shutdown_distributed(context: DistributedContext | None) -> None:
    """Shut down an owned process group in the order that releases captured collectives before communicator teardown.
    Captured CUDA graphs must be released before the communicator is destroyed
    to avoid teardown hangs.
    The operation is idempotent and leaves externally owned process groups untouched.
    """

    if (
        context is None
        or not context.owns_process_group
        or not dist.is_initialized()
    ):
        return


    # Log each shutdown phase to identify incomplete teardown.
    log = logging.getLogger(__name__)
    rank = context.rank

    log.info("shutdown_distributed rank=%s phase=synchronize", rank)
    if torch.cuda.is_available():
        torch.cuda.synchronize()


    # A communicator cannot be destroyed while its captured graphs are alive.
    # Reset registered graphs first and log the number released.
    log.info("shutdown_distributed rank=%s phase=release_cuda_graphs", rank)
    released = cuda_graphs.release_all()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    log.info(
        "shutdown_distributed rank=%s phase=release_cuda_graphs_done released=%d",
        rank,
        released,
    )

    log.info("shutdown_distributed rank=%s phase=reset_cudagraph_trees", rank)
    inductor = getattr(torch, "_inductor", None)
    cudagraph_trees = getattr(inductor, "cudagraph_trees", None)
    reset_cudagraph_trees = getattr(cudagraph_trees, "reset_cudagraph_trees", None)
    if callable(reset_cudagraph_trees):
        reset_cudagraph_trees()

    log.info("shutdown_distributed rank=%s phase=dynamo_reset", rank)
    dynamo = getattr(torch, "_dynamo", None)
    reset_dynamo = getattr(dynamo, "reset", None)
    if callable(reset_dynamo):
        reset_dynamo()

    log.info("shutdown_distributed rank=%s phase=barrier", rank)
    dist.barrier()

    # Release captured graphs before destroying the communicator.
    # DEFGCM_DIST_DESTROY_PROCESS_GROUP=0 defers destruction to interpreter exit.
    if os.environ.get("DEFGCM_DIST_DESTROY_PROCESS_GROUP", "1") == "1":
        log.info("shutdown_distributed rank=%s phase=destroy_process_group", rank)
        dist.destroy_process_group()
    else:
        log.info("shutdown_distributed rank=%s phase=skip_destroy_process_group", rank)
    log.info("shutdown_distributed rank=%s phase=done", rank)


def _seed_env_for_init(launch: LaunchEnvironment) -> None:
    os.environ.setdefault("RANK", str(launch.rank))
    os.environ.setdefault("WORLD_SIZE", str(launch.world_size))
    os.environ.setdefault("LOCAL_RANK", str(launch.local_rank))
    if launch.master_addr is not None:
        os.environ.setdefault("MASTER_ADDR", launch.master_addr)
    elif launch.world_size == 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    else:
        raise RuntimeError(
            "MASTER_ADDR is required for multi-rank SLURM initialization; "
            "launch with torchrun or export MASTER_ADDR/MASTER_PORT"
        )
    os.environ.setdefault("MASTER_PORT", launch.master_port or "29500")


def _configure_nccl_user_buffer_registration(
    backend: str, *, process_group_initialized: bool
) -> str:
    """Enforce no NCCL user buffer registration of allocator-backed tensors.

    NCCL v2.27.5-1 src/transport/p2p.cc:836-913 exports only the first
    cuMem handle; src/include/param.h:19-27 caches parameters on first access.
    PyTorch v2.9.0 c10/cuda/CUDACachingAllocator.cpp:138-144,2642-2789
    backs expandable tensors with independently mapped 2 MiB or 20 MiB chunks.
    Registering only the first handle can cause illegal memory access when
    a collective touches later chunks. Disable registration before creating
    the process group; see docs/DEVELOPMENT.md, "NCCL user-buffer registration".
    """
    if backend != "nccl":
        return "not-applicable"

    variables = ("NCCL_GRAPH_REGISTER", "NCCL_LOCAL_REGISTER")
    for name in variables:
        value = os.environ.get(name)
        if value != "0" and (value is not None or process_group_initialized):
            instruction = (
                " Export NCCL_GRAPH_REGISTER=0 and NCCL_LOCAL_REGISTER=0 "
                "before initializing the process group."
                if process_group_initialized
                else ""
            )
            shown = "unset" if value is None else repr(value)
            raise RuntimeError(
                f"LOCOMgcm refuses NCCL user buffer registration with {name}={shown}: "
                "allocator-backed tensors may span several cuMem handles and NCCL "
                f"imports only the first.{instruction}"
            )
    for name in variables:
        os.environ.setdefault(name, "0")
    return "disabled"


def _read_int(
    env: Mapping[str, str],
    name: str,
    *,
    default: int | None = None,
) -> int:
    value = env.get(name)
    if value is None:
        if default is None:
            raise ValueError(f"missing required integer environment variable {name}")
        return default
    try:
        result = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if result < 0:
        raise ValueError(f"{name} must be non-negative, got {result}")
    return result
