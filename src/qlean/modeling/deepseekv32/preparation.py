# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist
from compressed_tensors.distributed import is_source_process
from compressed_tensors.offload import update_offload_parameter
from compressed_tensors.offload.cache import (
    DistributedCPUCache,
    DistributedDiskCache,
    OffloadCache,
)


logger = logging.getLogger(__name__)
_INFERENCE_CONFIG_PATH = Path("inference/config_671B_v3.2.json")


def prepare_deepseek_v32_checkpoint(checkpoint_dir: str | Path) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    config_path = checkpoint_dir / "config.json"
    inference_config_path = checkpoint_dir / _INFERENCE_CONFIG_PATH

    with config_path.open(encoding="utf-8") as config_file:
        root_config = json.load(config_file)
    if not inference_config_path.exists():
        raise FileNotFoundError(
            f"Inference config not found at {inference_config_path}. "
            f"Expected the DeepSeek-V3.2-671B checkpoint structure "
            f"({_INFERENCE_CONFIG_PATH})."
        )
    with inference_config_path.open(encoding="utf-8") as config_file:
        inference_config = json.load(config_file)

    inference_config.pop("dtype", None)
    # This rewrites a shared source checkpoint: a truncated config.json would
    # permanently corrupt it, so stage the new content and replace atomically.
    tmp_path = config_path.parent / (config_path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as config_file:
        json.dump(root_config | inference_config, config_file)
    os.replace(tmp_path, config_path)


def restore_deepseek_v32_freqs_cis(model: torch.nn.Module) -> None:
    from qlean.modeling.deepseekv32.model import precompute_freqs_cis

    transformer = model.model
    buffers = transformer._buffers
    current = (
        buffers.offloaded_values.get("freqs_cis")
        if isinstance(buffers, OffloadCache)
        else buffers.get("freqs_cis")
    )
    if not isinstance(current, torch.Tensor):
        raise RuntimeError("DeepSeek-V3.2 model has no freqs_cis buffer to restore")

    restored = precompute_freqs_cis(model.config)
    if current.shape != restored.shape:
        raise RuntimeError(
            "DeepSeek-V3.2 freqs_cis shape does not match its configuration: "
            f"loaded={tuple(current.shape)}, expected={tuple(restored.shape)}"
        )

    shared_cache = isinstance(buffers, (DistributedCPUCache, DistributedDiskCache))
    if current.device.type == "meta" and not isinstance(buffers, OffloadCache):
        # precompute_freqs_cis builds on the default device (CPU); place the buffer
        # where the module actually executes so rotary multiplication does not mix
        # devices. The offload branch below keeps CPU tensors on purpose.
        device = next(
            (p.device for p in transformer.parameters() if p.device.type != "meta"),
            torch.device("cpu"),
        )
        transformer._buffers["freqs_cis"] = restored.to(device)
    elif not shared_cache or is_source_process():
        update_offload_parameter(transformer, "freqs_cis", restored)

    if shared_cache and dist.is_initialized():
        dist.barrier()

    logger.info("Restored DeepSeek-V3.2 freqs_cis after low-memory model loading")
