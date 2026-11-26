# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
# Top-level MoE linearization ported from llmcompressor/modeling/moe/linearize.py.

from __future__ import annotations

import contextlib
from functools import wraps
from typing import Type

import torch
import tqdm
from compressed_tensors.utils import patch_attr
from loguru import logger
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers import conversion_mapping
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.monkey_patching import (
    get_patch_mapping,
    register_patch_mapping,
    unregister_patch_mapping,
)

from .conversion_mappings import (
    get_linearize_load_mappings,
    has_linearize_load_mappings,
    set_save_conversion_mapping,
)
from .helpers import FusedExpertsProtocol
from .linear_experts import LinearExperts2D

__all__ = ["linearize_moe", "get_non_linearized_moes", "load_quantizable_moe"]


@contextlib.contextmanager
def load_quantizable_moe(model_cls: Type[PreTrainedModel] = AutoModelForCausalLM):
    """Context manager for loading MoE models with linearized experts.

    Patches ``model_cls.from_pretrained`` so that fused experts modules are
    automatically replaced by :class:`LinearExperts2D` during model loading.
    If a checkpoint conversion mapping exists for the model type, weights are
    loaded directly in linearized format; otherwise the model is loaded normally
    and linearized afterwards.

    :param model_cls: model class to patch, defaults to ``AutoModelForCausalLM``
    """
    original_from_pretrained = model_cls.from_pretrained
    patched_fn_called = False

    @classmethod
    @wraps(original_from_pretrained)
    def patched(cls, *args, **kwargs):
        nonlocal patched_fn_called
        patched_fn_called = True

        config = AutoConfig.from_pretrained(*args, **kwargs)
        model_type = config.model_type

        if not has_linearize_load_mappings(model_type):
            loaded_model = original_from_pretrained(*args, **kwargs)
            model = loaded_model[0] if isinstance(loaded_model, tuple) else loaded_model
            linearize_moe(model)
            return loaded_model

        # TF has no unregister API; the public lookup also conflates absent/None entries.
        get_checkpoint_conversion_mapping(model_type)
        mapping_cache = conversion_mapping._checkpoint_conversion_mapping_cache
        mapping_present = model_type in mapping_cache
        original_mapping = mapping_cache.get(model_type)
        user_registered = model_type in conversion_mapping.USER_REGISTERED_MAPPINGS

        try:
            experts_cls, load_map, save_map = get_linearize_load_mappings(model_type)
            patch_key = experts_cls.__name__
            original_patch = get_patch_mapping().get(patch_key)
            try:
                linear_experts_2d_cls = LinearExperts2D.get_linear_experts_cls(experts_cls)
                register_patch_mapping(
                    {patch_key: linear_experts_2d_cls}, overwrite=True
                )
                register_checkpoint_conversion_mapping(model_type, load_map, overwrite=True)
                loaded_model = original_from_pretrained(*args, **kwargs)
            finally:
                if original_patch is not None:
                    register_patch_mapping({patch_key: original_patch}, overwrite=True)
                elif patch_key in get_patch_mapping():
                    unregister_patch_mapping([patch_key])
        finally:
            if mapping_present:
                mapping_cache[model_type] = original_mapping
            else:
                mapping_cache.pop(model_type, None)
            if user_registered:
                conversion_mapping.USER_REGISTERED_MAPPINGS.add(model_type)
            else:
                conversion_mapping.USER_REGISTERED_MAPPINGS.discard(model_type)

        model: PreTrainedModel = (
            loaded_model[0] if isinstance(loaded_model, tuple) else loaded_model
        )
        set_save_conversion_mapping(model, save_map)

        return loaded_model

    with patch_attr(model_cls, "from_pretrained", patched):
        try:
            yield
        finally:
            if not patched_fn_called:
                logger.warning(
                    f"`{model_cls.__name__}.from_pretrained` was never called. If "
                    f"you are loading with a model class other than "
                    f"{model_cls.__name__}, please pass it as an argument to "
                    "`load_quantizable_moe`"
                )


def linearize_moe(model: PreTrainedModel) -> PreTrainedModel:
    """Linearize all fused MoE layers in *model* in-place.

    Each detected experts module is replaced by a :class:`LinearExperts2D` container
    holding per-expert ``nn.Linear`` layers. This allows per-expert quantization and
    matches the dispatch forward signature expected by the parent MoE block.

    :param model: model containing fused MoE layers
    :return: the modified model
    """
    non_linearized_moe_names = _get_non_linearized_moe_names(model)

    if len(non_linearized_moe_names) <= 0:
        return model

    logger.info(
        "Linearizing MoE experts for quantization. If the checkpoint is already "
        "linearized this will perform an extra 3D -> 2D conversion."
    )

    for name in tqdm.tqdm(non_linearized_moe_names, desc="Linearizing experts"):
        module = model.get_submodule(name)
        config = getattr(module, "config", model.config)
        linear_experts_cls = LinearExperts2D.get_linear_experts_cls(module.__class__)
        linear_moe = linear_experts_cls.from_experts_module(module, config)
        model.set_submodule(name, linear_moe)
        del module

    return model


def get_non_linearized_moes(
    model: torch.nn.Module,
) -> list[tuple[str, torch.nn.Module]]:
    """Return fused experts modules that have not yet been linearized.

    A module is recognized as an experts layer if it conforms to
    :class:`FusedExpertsProtocol` or has a registered :class:`LinearExperts2D`
    replacement class.

    :param model: model to inspect
    :return: list of ``(name, module)`` tuples
    """
    return [(name, model.get_submodule(name)) for name in _get_non_linearized_moe_names(model)]


def _get_non_linearized_moe_names(model: torch.nn.Module) -> list[str]:
    """Return fused expert module names without retaining their weight tensors."""
    candidates = []
    rejected = []
    for name, module in model.named_modules():
        is_fused = isinstance(module, FusedExpertsProtocol)
        is_registered = LinearExperts2D.get_registration(module.__class__) is not None
        if is_fused or is_registered:
            candidates.append(name)
        elif hasattr(module, "down_proj") and hasattr(module, "num_experts"):
            # Looks like a fused experts module but did not pass detection. Log
            # the class name so we can diagnose why.
            rejected.append((name, module.__class__.__name__))

    if rejected:
        logger.warning(
            "Found {} module(s) that look like fused experts but were not detected: {}",
            len(rejected),
            ", ".join(f"{name}({cls})" for name, cls in rejected[:10]),
        )
    candidate_names = ", ".join(candidates)
    logger.info(
        "MoE detection: {} fused experts module(s) found{}",
        len(candidates),
        f" ({candidate_names})" if candidate_names else "",
    )
    return candidates
