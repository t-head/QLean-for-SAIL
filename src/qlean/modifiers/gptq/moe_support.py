# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import List, Mapping

import torch.nn as nn
from loguru import logger

from qlean.core.routes import GPTQ_EXPERT_TARGET
from qlean.modeling.moe import get_non_linearized_moes, linearize_moe

def is_moe_model(model: nn.Module) -> bool:
    """Return ``True`` if *model* contains fused MoE expert layers."""
    return len(get_non_linearized_moes(model)) > 0


def get_moe_layers(model: nn.Module) -> List[str]:
    """Return dotted names of all fused (non-linearized) MoE expert layers."""
    return [name for name, _ in get_non_linearized_moes(model)]


def routed_expert_linears(model: nn.Module) -> set[nn.Module]:
    experts = set()
    for module in model.modules():
        container = getattr(module, "experts", None)
        if isinstance(container, nn.ModuleList):
            experts.update(
                child for child in container.modules() if isinstance(child, nn.Linear)
            )
    return experts


def validate_int4_expert_modules(
    model: nn.Module, modules: Mapping[str, nn.Module]
) -> None:
    if not modules:
        raise ValueError(
            "GPTQ channel INT4 requires at least one routed MoE expert Linear target"
        )
    expert_modules = routed_expert_linears(model)
    offenders = sorted(
        name for name, module in modules.items() if module not in expert_modules
    )
    if not offenders:
        return
    limit = 8
    examples = ", ".join(repr(name) for name in offenders[:limit])
    remainder = len(offenders) - limit
    if remainder > 0:
        examples += f", and {remainder} more"
    raise ValueError(
        "GPTQ channel INT4 supports only routed MoE expert Linear modules; "
        f"{len(offenders)} dense or non-expert module(s) remain selected: {examples}. "
        f"Use targets=[GPTQ_EXPERT_TARGET] ({GPTQ_EXPERT_TARGET!r}) or add every "
        "dense Linear selected by targets to GPTQModifier.ignore."
    )


def linearize_moe_layers(model: nn.Module) -> None:
    """In-place linearization of fused MoE expert layers.

    Delegates to the llm-compressor-style :func:`linearize_moe` implementation
    which supports any architecture using the transformers
    ``@use_experts_implementation`` decorator, plus registered specializations
    for Granite/Llama4 MoE models.
    """
    linearize_moe(model)
    logger.info("MoE linearization complete")
