# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Model-identity-driven loading decisions: class selection and per-model hacks.

Both tables are keyed by substrings of the model name/path, so supporting a new
model means adding one entry here rather than editing the loader. Quirks run
before ``from_pretrained``, matching what the reference torchrun scripts do at
import time.
"""

from __future__ import annotations

import importlib
import logging
from typing import Callable

logger = logging.getLogger(__name__)

__all__ = [
    "register_model_class",
    "register_model_quirk",
    "register_model_profile",
    "resolve_model_cls",
    "apply_model_quirks",
    "resolve_model_profile",
    "resolve_model_defaults",
]

# (name/path substrings, fully qualified transformers class)
_MODEL_CLASSES: list[tuple[tuple[str, ...], str]] = []
# (name/path substrings, zero-argument patch callable)
_MODEL_QUIRKS: list[tuple[tuple[str, ...], Callable[[], None]]] = []
# (name/path substrings, {"preparation": [...], "tokenizer_quirk": [...], "defaults": {...}})
_MODEL_PROFILES: list[tuple[tuple[str, ...], dict]] = []


def _identity(model_name: str | None, model_path: str | None) -> str:
    return f"{model_name or ''} {model_path or ''}".lower()


def register_model_class(*patterns: str, cls: str) -> None:
    """Map model-name/path substrings to a concrete transformers class."""
    _MODEL_CLASSES.append((tuple(p.lower() for p in patterns), cls))


def register_model_quirk(*patterns: str):
    """Register a loading workaround for models whose name/path matches."""

    def decorator(fn: Callable[[], None]):
        _MODEL_QUIRKS.append((tuple(p.lower() for p in patterns), fn))
        return fn

    return decorator


def resolve_model_cls(model_name: str | None, model_path: str | None):
    """Pick the loading class from the model identity, defaulting to the Auto class."""
    identity = _identity(model_name, model_path)
    for patterns, cls_path in _MODEL_CLASSES:
        if any(p in identity for p in patterns):
            module, _, name = cls_path.rpartition(".")
            logger.info("Resolved model class %s for %s", cls_path, model_name or model_path)
            return getattr(importlib.import_module(module), name)

    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM


def apply_model_quirks(model_name: str | None, model_path: str | None) -> list[str]:
    """Apply every matching workaround; returns the names of those applied."""
    identity = _identity(model_name, model_path)
    applied = []
    for patterns, fn in _MODEL_QUIRKS:
        if any(p in identity for p in patterns):
            fn()
            applied.append(fn.__name__)
            logger.info("Applied model quirk %s", fn.__name__)
    return applied


def register_model_profile(
    *patterns: str,
    preparation: Callable | None = None,
    tokenizer_quirk: Callable | None = None,
    **defaults,
) -> None:
    """Register per-model loading steps and CLI parameter defaults.

    ``preparation(model_args)`` runs before ``from_pretrained`` on the source rank
    only, with a barrier afterwards, for models whose checkpoint must be adjusted
    in place first. ``tokenizer_quirk(processor)`` runs on the loaded
    processor/tokenizer. ``defaults`` supplies values for CLI options the user did
    not pass explicitly, so one command reproduces the reference torchrun script.
    """
    _MODEL_PROFILES.append((
        tuple(p.lower() for p in patterns),
        {"preparation": preparation, "tokenizer_quirk": tokenizer_quirk, "defaults": defaults},
    ))


def resolve_model_profile(model_name: str | None, model_path: str | None) -> dict:
    """Merge every matching profile; later registrations win on defaults."""
    identity = _identity(model_name, model_path)
    merged = {"preparation": [], "tokenizer_quirk": [], "defaults": {}}
    for patterns, profile in _MODEL_PROFILES:
        if not any(p in identity for p in patterns):
            continue
        for key in ("preparation", "tokenizer_quirk"):
            if profile[key] is not None:
                merged[key].append(profile[key])
        merged["defaults"].update(profile["defaults"])
    return merged


def resolve_model_defaults(model_name: str | None, model_path: str | None) -> dict:
    """Return the CLI option defaults registered for this model identity."""
    return resolve_model_profile(model_name, model_path)["defaults"]


@register_model_quirk("qwen3.5", "qwen3_5")
def _qwen3_5_transformers_compat() -> None:
    """Workarounds the Qwen3.5 reference script applies before loading.

    transformers' fx probe rejects the Qwen3.5 graph, and the slow GPT-2 tokenizer
    module lost ``bytes_to_unicode`` in v5 while Qwen3.5 still imports it from there.
    """
    import transformers.utils.import_utils as import_utils

    import_utils.is_torch_fx_available = lambda: True

    import transformers.models.gpt2.tokenization_gpt2 as gpt2_tokenization

    if not hasattr(gpt2_tokenization, "bytes_to_unicode"):
        from transformers.convert_slow_tokenizer import bytes_to_unicode

        gpt2_tokenization.bytes_to_unicode = bytes_to_unicode


register_model_class("qwen3.5", "qwen3_5", cls="transformers.Qwen3_5MoeForConditionalGeneration")
# Qwen3.8 dense/multimodal checkpoints load through the Qwen3_5 conditional-generation
# class, matching the reference torchrun script for Qwen3.8-27B.
register_model_class("qwen3.8", "qwen3_8", cls="transformers.Qwen3_5ForConditionalGeneration")
register_model_class("glm5.3", "glm-5.3", "glm5_next", cls="transformers.Glm5NextForConditionalGeneration")
# DeepSeek-V3.2 loads through qlean's own port of the DeepSeek inference model, not
# through transformers, matching the reference torchrun script.
register_model_class(
    "deepseek-v3.2", "deepseek_v32", "deepseekv32",
    cls="qlean.modeling.deepseekv32.model.DeepseekV32ForCausalLM",
)

# The 397B reference script offloads up to 500GB to CPU.
register_model_profile("qwen3.5-397b", cpu_offload_gb=500.0)
# GLM-5.3-Flash reference script: 500GB CPU offload, two calibration rounds, and
# eight export workers for the grouped INT8 checkpoint.
register_model_profile(
    "glm5.3", "glm-5.3", "glm5_next",
    cpu_offload_gb=500.0, num_calibration_rounds=2, gptq_export_workers=8,
)

# From the reference torchrun script: generic HF Jinja fallback, since DeepSeek-V3.2 ships no chat_template.
_DEEPSEEK_CHAT_TEMPLATE = (
    "{% for message in messages %}\n"
    "{% if message['role'] == 'user' %}\n"
    "{{ '<|user|>\n' + message['content'] + eos_token }}\n"
    "{% elif message['role'] == 'system' %}\n"
    "{{ '<|system|>\n' + message['content'] + eos_token }}\n"
    "{% elif message['role'] == 'assistant' %}\n"
    "{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n"
    "{% endif %}\n"
    "{% if loop.last and add_generation_prompt %}\n"
    "{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
)


def _deepseek_v32_prepare(model_args) -> None:
    """Merge the inference config into config.json before loading.

    The DeepSeek-V3.2 checkpoint ships model dimensions in
    ``inference/config_671B_v3.2.json`` rather than in config.json, so the reference
    script merges them in place on rank 0 before ``from_pretrained``.
    """
    from qlean.modeling.deepseekv32.preparation import prepare_deepseek_v32_checkpoint

    prepare_deepseek_v32_checkpoint(model_args.model_path)


def _deepseek_v32_chat_template(processor) -> None:
    tokenizer = getattr(processor, "tokenizer", processor)
    if getattr(tokenizer, "chat_template", None) is None:
        tokenizer.chat_template = _DEEPSEEK_CHAT_TEMPLATE


register_model_profile(
    "deepseek-v3.2", "deepseek_v32", "deepseekv32",
    preparation=_deepseek_v32_prepare,
    tokenizer_quirk=_deepseek_v32_chat_template,
    cpu_offload_gb=500.0, num_calibration_rounds=2, calibration_padding="max_length",
)
