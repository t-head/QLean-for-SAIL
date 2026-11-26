# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared GPTQ model loading for the torchrun-script and qlean-CLI entry points.

Mirrors the reference scripts: apply model quirks, resolve the loading class from
the model identity, load inside ``load_context`` with disk offloading, then load a
processor/tokenizer for calibration. Both entry points call this same function so
their loading behaviour cannot drift apart.
"""

from __future__ import annotations

import logging

from qlean.args import ModelArguments

logger = logging.getLogger(__name__)

__all__ = ["load_gptq_model", "load_processor_or_tokenizer", "tokenizer_of"]


def load_processor_or_tokenizer(model_path: str, trust_remote_code: bool = True):
    """Prefer a processor (multimodal models), fall back to a plain tokenizer."""
    from transformers import AutoProcessor, AutoTokenizer

    try:
        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
    except Exception as error:
        logger.info("AutoProcessor unavailable (%s); using AutoTokenizer", error)
        processor = None

    if processor is not None and hasattr(processor, "apply_chat_template"):
        return processor
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)


def tokenizer_of(processor):
    """Return the callable tokenizer behind a processor or a bare tokenizer."""
    return getattr(processor, "tokenizer", processor)


def _run_preparations(preparations, model_args) -> None:
    """Run checkpoint preparations on the source rank, then synchronize.

    Preparations may rewrite the source checkpoint (DeepSeek-V3.2 merges its
    inference config into config.json), so only one rank may do it and the others
    must wait before loading. Errors are gathered across ranks so a failing
    source rank cannot leave the others blocked on a barrier.
    """
    import torch.distributed as dist

    from qlean.modifiers.gptq.distributed import raise_for_distributed_errors

    distributed = dist.is_initialized()
    for prepare in preparations:
        error = None
        if not distributed or dist.get_rank() == 0:
            try:
                prepare(model_args)
            except BaseException as exc:
                error = exc
        raise_for_distributed_errors(error, "checkpoint preparation")


def load_gptq_model(model_args: ModelArguments):
    """Load ``(model, processor)`` for GPTQ, or raise with actionable guidance."""
    from qlean.models.pretrained.model_registry import (
        apply_model_quirks,
        resolve_model_cls,
        resolve_model_profile,
    )
    from qlean.utils.dev import load_context
    from qlean.utils.optional import require_gptq

    require_gptq()

    if not model_args.offload_folder:
        raise ValueError(
            "GPTQ loads the model with disk offloading and needs a writable "
            "--offload_folder that stays available through calibration and export."
        )

    apply_model_quirks(model_args.model_name, model_args.model_path)
    profile = resolve_model_profile(model_args.model_name, model_args.model_path)
    model_cls = resolve_model_cls(model_args.model_name, model_args.model_path)

    kwargs = {
        "device_map": model_args.device_map,
        "trust_remote_code": model_args.trust_remote_code,
        "offload_folder": model_args.offload_folder,
    }
    if model_args.cpu_offload_gb is not None:
        kwargs["max_memory"] = {"cpu": model_args.cpu_offload_gb * 1e9}

    _run_preparations(profile["preparation"], model_args)

    logger.info("Loading %s with %s", model_args.model_path, model_cls.__name__)
    with load_context(model_cls):
        model = model_cls.from_pretrained(model_args.model_path, **kwargs)

    processor = load_processor_or_tokenizer(
        model_args.model_path, model_args.trust_remote_code
    )
    for quirk in profile["tokenizer_quirk"]:
        quirk(processor)
    return model, processor
