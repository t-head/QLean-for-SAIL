# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Distributed helpers shared by the torchrun-script and qlean-CLI GPTQ paths."""

from __future__ import annotations

import os

import torch.distributed as dist

__all__ = ["is_torchrun_worker", "maybe_init_dist"]


def is_torchrun_worker() -> bool:
    """Return whether this process was spawned by torchrun / torch.distributed.run."""
    return "TORCHELASTIC_RUN_ID" in os.environ


def maybe_init_dist() -> bool:
    """Initialize the default process group once, and only under torchrun.

    Returns True when this call performed the initialization.

    Scripts launched by torchrun typically call ``compressed_tensors``'
    ``init_dist()`` themselves before loading the model. That helper has no
    already-initialized guard, so calling it a second time raises
    "trying to initialize the default process group twice". Skipping when the
    group exists keeps both entry points usable with the same code.
    """
    if dist.is_initialized():
        return False
    if not is_torchrun_worker():
        return False

    from compressed_tensors.offload import init_dist

    init_dist()
    return True
