# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Re-exec qlean under torchrun for multi-process GPTQ.

Only the CLI entry point uses this. A script already launched by torchrun never
reaches it, and a worker spawned here sees TORCHELASTIC_RUN_ID and does not launch
again, so the two entry points cannot nest.
"""

from __future__ import annotations

import logging
import subprocess
import sys

from qlean.args import OtherArguments
from qlean.utils.distributed import is_torchrun_worker

logger = logging.getLogger(__name__)

__all__ = ["needs_launch", "launch_torchrun"]


def needs_launch(other_args: OtherArguments) -> bool:
    """Return whether this process should spawn torchrun workers."""
    return other_args.nproc_per_node > 1 and not is_torchrun_worker()


def launch_torchrun(other_args: OtherArguments, qlean_argv: list[str]) -> int:
    """Run qlean under torch.distributed.run and propagate its exit code."""
    command = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node", str(other_args.nproc_per_node),
        "--nnodes", str(other_args.nnodes),
        "--max_restarts", str(other_args.max_restarts),
    ]
    if other_args.rdzv_endpoint:
        command += ["--rdzv_backend", "c10d", "--rdzv_endpoint", other_args.rdzv_endpoint]
    command += ["-m", "qlean.cli", *qlean_argv]

    logger.info("Launching distributed GPTQ: %s", " ".join(command))
    return subprocess.call(command)