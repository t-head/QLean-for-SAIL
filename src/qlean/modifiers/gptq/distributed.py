# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import torch
import torch.distributed as dist
from compressed_tensors.distributed import greedy_bin_packing, is_source_process, wait_for_comms
from compressed_tensors.offload import get_execution_device
from compressed_tensors.offload.dist_utils import as_broadcastable

__all__ = [
    "is_distributed",
    "greedy_bin_packing",
    "wait_for_comms",
    "broadcast_qparams_and_cleanup",
    "source_rank",
    "raise_for_distributed_errors",
]


def is_distributed() -> bool:
    """Return whether multiple distributed workers are active."""
    return dist.is_initialized() and dist.get_world_size() > 1


def source_rank(group=None) -> int:
    if not is_distributed():
        return 0
    candidate = dist.get_rank() if is_source_process() else -1
    candidates = [None] * dist.get_world_size(group)
    dist.all_gather_object(candidates, candidate, group=group)
    sources = [rank for rank in candidates if rank >= 0]
    if len(sources) != 1:
        raise RuntimeError(f"Expected exactly one checkpoint source rank, found {sources}")
    return sources[0]


def raise_for_distributed_errors(error, phase: str, group=None) -> None:
    if not is_distributed():
        if error is not None:
            raise error
        return
    rank = dist.get_rank()
    status = None if error is None else (rank, type(error).__name__, str(error))
    statuses = [None] * dist.get_world_size(group)
    dist.all_gather_object(statuses, status, group=group)
    failures = [status for status in statuses if status is not None]
    if failures:
        details = "; ".join(
            f"rank {rank} {kind}: {message}" for rank, kind, message in failures
        )
        failure = RuntimeError(f"Distributed GPTQ {phase} failed: {details}")
        if error is not None:
            raise failure from error
        raise failure


def broadcast_qparams_and_cleanup(
    module_list: list[torch.nn.Module],
    module_to_rank: dict[torch.nn.Module, int],
    qparam_names: Sequence[str],
    skip_cpu: bool = True,
) -> None:
    """Broadcast owner-rank qparams and release observer statistics."""
    pending_comms = []
    for module in module_list:
        should_broadcast = not skip_cpu or (
            get_execution_device(module) != torch.device("cpu")
        )
        if should_broadcast:
            for name in qparam_names:
                param = getattr(module, name, None)
                if param is not None:
                    pending_comms.append(
                        dist.broadcast(
                            as_broadcastable(param),
                            src=module_to_rank[module],
                            async_op=True,
                        )
                    )

        observer = getattr(module, "weight_observer", None)
        if observer is not None and observer.has_statistics:
            observer.delete_statistics()

    wait_for_comms(pending_comms)
