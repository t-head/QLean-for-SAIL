# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
from copy import deepcopy
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def validate_output_directory(model, source, save_directory, *, require_fresh=False):
    output = Path(save_directory).resolve()
    revision = getattr(getattr(model, "config", None), "_commit_hash", None)
    for candidate in dict.fromkeys((getattr(model, "name_or_path", None), source)):
        if not candidate:
            continue
        if Path(candidate).is_dir():
            source_path = Path(candidate).resolve()
        elif revision is not None:
            from transformers.utils import cached_file

            config_path = cached_file(
                candidate, "config.json", revision=revision, local_files_only=True,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
                _raise_exceptions_for_gated_repo=False,
            )
            if config_path is None:
                continue
            source_path = Path(config_path).parent.resolve()
        else:
            continue
        if output == source_path or output in source_path.parents or source_path in output.parents:
            raise ValueError("Source and output checkpoint directories must not overlap")

    if require_fresh and output.is_dir() and any(
        name.endswith((".safetensors", ".bin", ".index.json"))
        for name in os.listdir(output)
    ):
        raise ValueError("Export requires a fresh checkpoint output directory")


def prepare_generation_config_for_save(model):
    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        return None
    generation_config = deepcopy(generation_config)
    if generation_config.do_sample is not True:
        sampling_defaults = {
            "temperature": 1.0,
            "top_p": 1.0,
            "min_p": None,
            "top_h": None,
            "typical_p": 1.0,
            "top_k": 50,
            "epsilon_cutoff": 0.0,
            "eta_cutoff": 0.0,
        }
        for name, default in sampling_defaults.items():
            value = getattr(generation_config, name, None)
            if value is not None and value != default:
                setattr(generation_config, name, None)
    generation_config.validate(strict=True)
    return generation_config


def compress_model_for_save(model, *, quantization_format):
    if not hasattr(model, "save_pretrained"):
        raise RuntimeError(
            "Checkpoint compression requires a Hugging Face model with save_pretrained()"
        )
    from compressed_tensors import ModelCompressor

    try:
        save_compressor = ModelCompressor.from_pretrained_model(
            model, quantization_format=quantization_format
        )
        save_compressor.compress_model(model, skip_compressed=True)
    except Exception as error:
        raise RuntimeError(
            "Checkpoint compression failed; refusing to save an unpacked dense checkpoint."
        ) from error
    logger.info("Packed quantized weights into compressed-tensors format")
    return save_compressor


def validate_checkpoint_format(save_directory, expected_format):
    try:
        with (Path(save_directory) / "config.json").open(encoding="utf-8") as file:
            config = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "Checkpoint save did not produce a readable config.json"
        ) from error
    quantization_config = config.get("quantization_config")
    actual_format = (
        quantization_config.get("format")
        if isinstance(quantization_config, dict) else None
    )
    if actual_format != expected_format:
        raise RuntimeError(
            "Checkpoint save produced an invalid compressed-tensors "
            f"format: expected '{expected_format}', got '{actual_format}'."
        )


def save_checkpoint(model, save_directory, max_shard_size, save_compressor, expected_format):
    generation_config = prepare_generation_config_for_save(model)
    original_generation_config = getattr(model, "generation_config", None)
    os.makedirs(save_directory, exist_ok=True)
    if generation_config is not None:
        saved_config = generation_config.to_dict()
        cleared_fields = [
            name for name, value in original_generation_config.to_dict().items()
            if value is not None and saved_config.get(name) is None
        ]
        if cleared_fields:
            logger.warning(
                "Unsetting inactive sampling parameters in saved generation_config "
                "(do_sample=%r): %s",
                generation_config.do_sample, ", ".join(cleared_fields),
            )
    logger.info("Saving quantized model to %s", save_directory)
    try:
        if generation_config is not None:
            model.generation_config = generation_config
        model.save_pretrained(save_directory, max_shard_size=max_shard_size)
    finally:
        if generation_config is not None:
            model.generation_config = original_generation_config
    try:
        save_compressor.update_config(save_directory)
    except Exception as error:
        raise RuntimeError(
            "Checkpoint metadata update failed; refusing to report an incomplete "
            "checkpoint as successful."
        ) from error
    logger.info("Wrote quantization_config into config.json")
    validate_checkpoint_format(save_directory, expected_format)


def expert_key_patterns(model):
    patterns = []
    for name, module in model.named_modules():
        experts = getattr(module, "experts", None)
        if isinstance(experts, torch.nn.ModuleList) and len(experts) > 0:
            patterns.append(f"{name}.experts.")
    return patterns


def validate_expert_keys_in_checkpoint(model, save_directory, patterns):
    if not patterns:
        return
    from safetensors import safe_open
    from transformers.core_model_loading import WeightConverter, WeightRenaming, rename_source_key

    saved_keys = []
    for file_name in os.listdir(save_directory):
        if not file_name.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(save_directory, file_name), framework="pt", device="cpu") as file:
            saved_keys.extend(file.keys())
    if not saved_keys:
        return
    mappings = getattr(model, "_weight_conversions", None) or ()
    renamings = [mapping for mapping in mappings if isinstance(mapping, WeightRenaming)]
    converters = [mapping for mapping in mappings if isinstance(mapping, WeightConverter)]
    expert_keys = []
    for key in saved_keys:
        model_key = key
        if renamings or converters:
            model_key, _ = rename_source_key(key, renamings, converters)
        if any(pattern in key or pattern in model_key for pattern in patterns):
            expert_keys.append(key)
    logger.info(
        "Saved checkpoint contains %d expert weight key(s) across %d expected expert prefix(es)",
        len(expert_keys), len(patterns),
    )
    if not expert_keys:
        raise RuntimeError(
            "MoE expert keys appear to have collapsed after save. Expected keys "
            f"matching one of {patterns}, but none were found in the saved "
            "safetensors. This usually means a fused/non-fused MoE block was "
            "replaced by a single dense MLP during quantization."
        )


def rename_packed_weights(save_directory):
    """Rename ``*_packed`` checkpoint keys in place.

    Deliberately retained as the shared-path boundary that GPTQ's isolated staged
    exporters must not use: the save/isolation tests patch this name to forbid or
    to observe calls. It has no production caller — do not remove it.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    index_path = os.path.join(save_directory, "model.safetensors.index.json")
    index_data = None
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as file:
            index_data = json.load(file)
    renamed_any = False
    for file_name in sorted(os.listdir(save_directory)):
        if not file_name.endswith(".safetensors"):
            continue
        file_path = os.path.join(save_directory, file_name)
        with safe_open(file_path, framework="pt", device="cpu") as file:
            keys = list(file.keys())
            tensors = {key: file.get_tensor(key) for key in keys}
            metadata = file.metadata()
        renamed = {}
        file_renamed = False
        for key in keys:
            if key.endswith("_packed"):
                renamed[key.removesuffix("_packed")] = tensors[key]
                file_renamed = True
            else:
                renamed[key] = tensors[key]
        if file_renamed:
            save_file(renamed, file_path, metadata=metadata)
            renamed_any = True
            if index_data is not None and "weight_map" in index_data:
                weight_map = index_data["weight_map"]
                for old_key in list(weight_map):
                    if old_key.endswith("_packed"):
                        weight_map[old_key.removesuffix("_packed")] = weight_map.pop(old_key)
    if renamed_any:
        logger.info("Renamed *_packed keys to * in saved safetensors")
        if index_data is not None:
            with open(index_path, "w", encoding="utf-8") as file:
                json.dump(index_data, file, indent=2)
