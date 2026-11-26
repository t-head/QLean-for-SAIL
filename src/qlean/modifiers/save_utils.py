# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import re
from copy import deepcopy
from pathlib import Path, PurePosixPath

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from compressed_tensors import __version__ as ct_version
from compressed_tensors.base import (
    COMPRESSION_VERSION_NAME,
    QUANTIZATION_CONFIG_NAME,
    QUANTIZATION_METHOD_NAME,
    SPARSITY_CONFIG_NAME,
    TRANSFORM_CONFIG_NAME,
)
from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
)
from transformers.file_utils import CONFIG_NAME
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "update_config",
    "WA_for_sglang_config",
    "validate_acext_a8w4_shape",
    "preprocess_for_acext_a8w4",
]


def validate_acext_a8w4_shape(shape) -> None:
    if len(shape) != 2:
        raise ValueError("A8W4 weights must be 2D [N, K]")
    n, k = shape
    if n == 0 or k == 0:
        raise ValueError("A8W4 weights must be nonempty [N, K]")
    # The local row permutation and subbyte transpose require 64-element tiles.
    if n % 64 or k % 64:
        raise ValueError("PyTorch A8W4 packing requires N and K multiples of 64")


def preprocess_for_acext_a8w4(weight: torch.Tensor) -> torch.Tensor:
    if not isinstance(weight, torch.Tensor) or weight.dtype != torch.int8:
        raise TypeError("A8W4 weights must be a signed torch.int8 tensor")
    validate_acext_a8w4_shape(weight.shape)
    from qlean.utils.pack_acext_int4 import preprocess_weight_for_a8w4_torch

    n, k = weight.shape
    return preprocess_weight_for_a8w4_torch(weight).view(n, k // 2)


def WA_for_sglang_config(config_file, fp8_channelwise_layers=None, fp8_blockwise_layers=None):

    config = json.load(open(config_file))
    if "quantization_config" in config:
        qc = config["quantization_config"]

        if "config_groups" in qc:
            del qc["config_groups"]
        if "format" in qc:
            del qc["format"]
        if "global_compression_ratio" in qc:
            del qc["global_compression_ratio"]

        if "kv_cache_scheme" in qc:
            del qc["kv_cache_scheme"]
        if "quant_method" in qc:
            qc["quant_method"] = "mxfp4"
        if "quantization_status" in qc:
            del qc["quantization_status"]

        if "sparsity_config" in qc:
            del qc["sparsity_config"]
        if "transform_config" in qc:
            del qc["transform_config"]
        if "version" in qc:
            del qc["version"]
        if fp8_channelwise_layers is not None:
            qc["fp8_channelwise_layers"] = fp8_channelwise_layers
        if fp8_blockwise_layers is not None:
            qc["fp8_blockwise_layers"] = fp8_blockwise_layers

    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False, sort_keys=True)

def update_config(
    config_path: str | os.PathLike,
    save_directory: str | os.PathLike,
    scheme_name: str,
    scheme: QuantizationScheme,
    ignore: list[str],
):
    from qlean.utils.ct_compat import match_name

    for i in range(len(ignore) - 1, -1, -1):
        if match_name("model.embed_tokens", ignore[i]):
            logger.debug(f"ignore list pop {ignore[i]}")
            ignore.pop(i)

    # W8A8_BLOCK: use simple custom quantization_config format
    if scheme_name == "W8A8_BLOCK":
        qconfig_data = {
            "activation_scheme": "dynamic",
            "quant_method": "blockwise_int8",
            "weight_block_size": [128, 128],
        }
    else:
        # construct quantization config (compressed-tensors format)
        qconfig = QuantizationConfig.model_validate(
            {
                "config_groups": {scheme_name: scheme},
                "ignore": ignore,
                "quantization_status": QuantizationStatus.COMPRESSED,
            }
        )

        # construct compression (quantization) config
        qconfig_data = qconfig.model_dump(exclude=[QUANTIZATION_METHOD_NAME, "format"])
        qconfig_data = {
            COMPRESSION_VERSION_NAME: ct_version,
            QUANTIZATION_METHOD_NAME: "compressed-tensors",
            SPARSITY_CONFIG_NAME: {},
            TRANSFORM_CONFIG_NAME: {},
            "format": scheme.format,
            **qconfig_data,
        }

    # write results to config.json file
    logger.info(f"Try find config in {config_path}")
    config_file_path = _find_config_path(config_path)
    if config_file_path is not None:
        config_file_name = os.path.basename(config_file_path)
        with open(config_file_path, "r") as file:
            config_data = json.load(file)

        config_data[QUANTIZATION_CONFIG_NAME] = qconfig_data

        with open(os.path.join(save_directory, config_file_name), "w") as file:
            json.dump(config_data, file, indent=2, sort_keys=True)
            logger.info(
                f"New config file saved to {save_directory}, ori in {config_file_path}"
            )

    else:
        logger.warning(
            f"Could not find config file in {config_path}. "
            f"Please {json.dumps(qconfig_data, indent=2, sort_keys=True)}"
        )


def _find_config_path(save_directory: str | os.PathLike) -> str | None:
    for file_name in os.listdir(save_directory):
        if file_name in (CONFIG_NAME, "params.json"):
            return os.path.join(save_directory, file_name)

    return None


def _checkpoint_file(source, filename, revision):
    from transformers.utils import cached_file

    path = PurePosixPath(filename)
    if path.is_absolute() or ".." in path.parts or "\\" in filename:
        raise ValueError(f"Invalid checkpoint filename: {filename}")
    return cached_file(
        source,
        filename,
        revision=revision,
        _raise_exceptions_for_missing_entries=False,
    )


def _checkpoint_weight_map(directory):
    from compressed_tensors.utils.safetensors_load import get_weight_mappings

    index_path = os.path.join(directory, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as file:
            index = json.load(file)
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict):
            raise ValueError(
                f"Checkpoint index {index_path} must contain a 'weight_map' object"
            )
        return weight_map
    return {
        name: os.path.basename(path)
        for name, path in get_weight_mappings(directory).items()
    }


def _semantic_config(config):
    metadata = {
        "architectures", "_name_or_path", "_commit_hash", "transformers_version",
        "dtype", "torch_dtype", "quantization_config", "_pre_quantization_dtype",
    }
    if isinstance(config, dict):
        return {
            key: _semantic_config(value)
            for key, value in config.items()
            if key not in metadata
        }
    if isinstance(config, list):
        return [_semantic_config(value) for value in config]
    return config


_MTP_COUNT_FIELDS = ("num_mtp_layers", "mtp_num_hidden_layers", "num_nextn_predict_layers")


def _mtp_layer_count(config):
    counts = set()
    for field in _MTP_COUNT_FIELDS:
        value = config.get(field, 0)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Source {field} must be a nonnegative integer")
        if value:
            counts.add(value)
    if len(counts) > 1:
        raise ValueError("Source checkpoint has conflicting MTP layer counts")
    return next(iter(counts), 0)


def _source_mtp_layout(config, weight_map):
    text_config = config.get("text_config", config)
    if not isinstance(text_config, dict):
        raise ValueError("Source text_config must be an object")
    count = _mtp_layer_count(text_config)
    if text_config is not config:
        root_count = _mtp_layer_count(config)
        if count and root_count and count != root_count:
            raise ValueError("Source checkpoint has conflicting MTP layer counts")
        count = count or root_count
    dedicated = {
        prefix: {name for name in weight_map if name.startswith(f"{prefix}.")}
        for prefix in ("mtp", "model.mtp", "model.language_model.mtp")
    }
    dedicated = {prefix: keys for prefix, keys in dedicated.items() if keys}
    if len(dedicated) > 1:
        raise ValueError("Source checkpoint has ambiguous MTP namespaces")

    appended = {}
    if count:
        base = text_config.get("num_hidden_layers")
        if base is not None:
            if isinstance(base, bool) or not isinstance(base, int) or base <= 0:
                raise ValueError("Source num_hidden_layers must be a positive integer")
            for owner in ("model.layers", "model.language_model.layers"):
                prefixes = [f"{owner}.{index}" for index in range(base, base + count)]
                keys = {
                    name for name in weight_map
                    if any(name.startswith(f"{prefix}.") for prefix in prefixes)
                }
                if keys:
                    if any(not any(name.startswith(f"{prefix}.") for name in keys) for prefix in prefixes):
                        raise ValueError("Source checkpoint is missing declared MTP layers")
                    appended[owner] = keys
    if len(appended) > 1 or (dedicated and appended):
        raise ValueError("Source checkpoint has ambiguous MTP layer layouts")
    if dedicated:
        prefix, keys = next(iter(dedicated.items()))
        indices = {
            int(match.group(1)) for name in keys
            if (match := re.match(rf"^{re.escape(prefix)}\.layers\.(\d+)\.", name))
        }
        if count and indices and indices != set(range(count)):
            raise ValueError("Source checkpoint is missing declared MTP layers")
        return keys
    if appended:
        return next(iter(appended.values()))
    if count:
        raise ValueError("Source checkpoint declares MTP layers but has no MTP weights")
    return set()


def prepare_checkpoint_preservation(model, source, save_directory, preserve_config):
    from compressed_tensors.utils.safetensors_load import get_safetensors_header

    source = os.fspath(source)
    is_local = os.path.isdir(source)
    if is_local and os.path.realpath(source) == os.path.realpath(save_directory):
        raise ValueError("Checkpoint output must not overwrite the source model")
    revision = getattr(model.config, "_commit_hash", None)
    if not is_local and revision is None:
        raise ValueError("Checkpoint preservation requires the loaded Hub commit hash")
    config_path = _checkpoint_file(source, "config.json", revision)
    if config_path is None:
        raise ValueError("Checkpoint preservation requires the source config.json")
    if os.path.realpath(os.path.dirname(config_path)) == os.path.realpath(save_directory):
        raise ValueError("Checkpoint output must not overwrite the resolved source model")
    with open(config_path, encoding="utf-8") as file:
        source_config = json.load(file)
    if not isinstance(source_config, dict):
        raise ValueError("Source config.json must be an object")

    if preserve_config:
        if source_config.get("model_type") == "qwen3_5_moe":
            from transformers import Qwen3_5MoeForConditionalGeneration

            root = getattr(model, "model", None)
            if (
                not isinstance(model, Qwen3_5MoeForConditionalGeneration)
                or getattr(root, "visual", None) is None
                or getattr(root, "language_model", None) is None
            ):
                raise ValueError(
                    "A full Qwen3.5 checkpoint requires "
                    "Qwen3_5MoeForConditionalGeneration with vision weights; "
                    "reload the source using load_context with that class"
                )
        if source_config.get("model_type") != model.config.model_type:
            raise ValueError("Source and runtime checkpoint model_type differ")
        expected = type(model.config).from_dict(deepcopy(source_config))
        if _semantic_config(expected.to_dict()) != _semantic_config(model.config.to_dict()):
            raise ValueError("Source and runtime checkpoint configurations are incompatible")
        architectures = source_config.get("architectures", [])
        if architectures and type(model).__name__ not in architectures:
            raise ValueError("Source and runtime checkpoint architectures differ")

    index_path = _checkpoint_file(source, "model.safetensors.index.json", revision)
    if index_path is not None:
        with open(index_path, encoding="utf-8") as file:
            index = json.load(file)
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
    elif is_local:
        weight_map = _checkpoint_weight_map(source)
    else:
        weight_path = _checkpoint_file(source, "model.safetensors", revision)
        if weight_path is None:
            raise ValueError("Checkpoint preservation requires source safetensors weights")
        with safe_open(weight_path, framework="pt", device="cpu") as file:
            weight_map = {name: "model.safetensors" for name in file.keys()}
    if not isinstance(weight_map, dict) or any(
        not isinstance(name, str) or not isinstance(filename, str)
        for name, filename in weight_map.items()
    ):
        raise ValueError("Source checkpoint weight_map must map tensor names to filenames")
    for filename in set(weight_map.values()):
        path = PurePosixPath(filename)
        if not filename or path.is_absolute() or ".." in path.parts or "\\" in filename:
            raise ValueError(f"Invalid checkpoint filename: {filename}")

    source_text_config = source_config.get("text_config", source_config)
    mtp_keys = _source_mtp_layout(source_config, weight_map)
    mtp_metadata = {}
    if mtp_keys:
        source_quantization = (
            source_config.get("quantization_config")
            or source_text_config.get("quantization_config")
        )
        if source_quantization and source_quantization.get("quant_method"):
            raise ValueError(
                "Restoring quantized/FP8 MTP is not supported with the new quantization "
                "config; use a BF16 source checkpoint"
            )
        if any(Path(save_directory).glob("model_mtp*.safetensors")):
            raise ValueError("MTP output shard already exists; use a fresh output directory")
        by_source = {}
        for name in sorted(mtp_keys):
            by_source.setdefault(weight_map[name], set()).add(name)
        for filename, keys in by_source.items():
            path = _checkpoint_file(source, filename, revision)
            if path is None:
                raise ValueError(f"Missing source MTP shard: {filename}")
            header = get_safetensors_header(path)
            _validate_mtp_header(header, keys)
            mtp_metadata.update({name: header[name] for name in keys})

    visual_keys = {
        name for name in weight_map
        if name.startswith(("model.visual.", "visual."))
    } if preserve_config and source_config.get("vision_config") else set()
    if preserve_config and source_config.get("vision_config") and not visual_keys:
        raise ValueError("Source conditional checkpoint has no vision weights")
    # finalize_checkpoint_preservation reads model.model.visual.state_dict(); catch
    # a mismatched loading class here, before calibration, for any vision model.
    if visual_keys and getattr(getattr(model, "model", None), "visual", None) is None:
        raise ValueError(
            "Source checkpoint has vision weights but the loaded model has no "
            "model.visual sub-module; reload the source using load_context with "
            "the matching conditional-generation class"
        )
    if not preserve_config and not mtp_keys:
        return None
    return {
        "source": source,
        "revision": revision,
        "config": source_config,
        "weight_map": weight_map,
        "mtp_keys": mtp_keys,
        "mtp_metadata": mtp_metadata,
        # Explicit module names for every MTP matrix weight; 1D norms/buffers are not layers.
        "mtp_ignore": sorted(
            name.removesuffix(".weight") for name, item in mtp_metadata.items()
            if name.endswith(".weight") and len(item["shape"]) >= 2
        ),
        "visual_keys": visual_keys,
        "preserve_config": preserve_config,
    }


def _validate_mtp_header(header, keys):
    for name in keys:
        if name not in header:
            raise ValueError(f"Missing source MTP tensor: {name}")
        dtype = header[name]["dtype"]
        if (
            dtype not in {"BF16", "F16", "F32", "F64", "I64", "I32", "BOOL"}
            or (name.endswith(".weight") and dtype not in {"BF16", "F16", "F32", "F64"})
        ):
            raise ValueError(
                f"Quantized/FP8 MTP tensor {name} cannot be restored without its "
                "source quantization metadata; use a BF16 source checkpoint"
            )


def finalize_checkpoint_preservation(model, checkpoint, save_directory, max_shard_size):
    from compressed_tensors.offload import disable_onloading
    from compressed_tensors.utils.safetensors_load import (
        get_safetensors_header, str_to_torch_dtype,
    )
    from huggingface_hub.serialization import split_state_dict_into_shards_factory
    from transformers.core_model_loading import WeightConverter, WeightRenaming, rename_source_key

    saved_map = _checkpoint_weight_map(save_directory)
    headers = {}
    for filename in set(saved_map.values()):
        # These shards were just written locally, so resolve them directly rather
        # than through cached_file, which is for Hub/source lookups.
        path = os.path.join(save_directory, filename)
        if not os.path.isfile(path):
            raise ValueError(f"Missing saved checkpoint shard: {filename}")
        headers[filename] = get_safetensors_header(path)
    for name, filename in saved_map.items():
        if name not in headers[filename]:
            raise ValueError(f"Checkpoint index references a missing tensor: {name}")

    visual_keys = checkpoint["visual_keys"]
    if visual_keys:
        conversions = getattr(model, "_weight_conversions", None) or ()
        renamings = [item for item in conversions if isinstance(item, WeightRenaming)]
        converters = [item for item in conversions if isinstance(item, WeightConverter)]
        normalized = {
            rename_source_key(name, renamings, converters)[0]: name
            for name in saved_map
        }
        with disable_onloading():
            visual_metadata = {
                f"model.visual.{name}": (list(tensor.shape), tensor.dtype)
                for name, tensor in model.model.visual.state_dict().items()
            }
        source_visual_keys = {
            rename_source_key(name, renamings, converters)[0] for name in visual_keys
        }
        if source_visual_keys != set(visual_metadata):
            raise ValueError("Source conditional checkpoint has an incomplete vision tensor set")
        for model_key, (shape, dtype) in visual_metadata.items():
            saved_key = normalized.get(model_key)
            if saved_key is None:
                raise ValueError(f"Saved conditional checkpoint is missing vision tensor: {model_key}")
            item = headers[saved_map[saved_key]][saved_key]
            if (
                item["shape"] != shape
                or item["dtype"] not in {"BF16", "F16", "F32", "F64"}
                or str_to_torch_dtype[item["dtype"]] != dtype
            ):
                raise ValueError(f"Saved vision tensor was changed or quantized: {model_key}")
        text_config = model.config.get_text_config()
        required_language_keys = {"model.language_model.embed_tokens.weight"}
        if not text_config.tie_word_embeddings:
            required_language_keys.add("lm_head.weight")
        source_language_keys = {
            rename_source_key(name, renamings, converters)[0]
            for name in checkpoint["weight_map"]
            if name.endswith(("embed_tokens.weight", "lm_head.weight"))
        }
        for model_key in required_language_keys:
            if model_key not in source_language_keys:
                raise ValueError(f"Source checkpoint is missing required language tensor: {model_key}")
            saved_key = normalized.get(model_key)
            if saved_key is None:
                raise ValueError(f"Saved checkpoint is missing required language tensor: {model_key}")
            item = headers[saved_map[saved_key]][saved_key]
            if f"{saved_key}_scale" not in saved_map and (
                item["shape"] != [text_config.vocab_size, text_config.hidden_size]
                or item["dtype"] not in {"BF16", "F16", "F32", "F64"}
            ):
                raise ValueError(f"Saved language tensor has invalid metadata: {model_key}")

    mtp_keys = checkpoint["mtp_keys"]
    if mtp_keys:
        if mtp_keys.intersection(saved_map):
            raise ValueError("Saved checkpoint already contains conflicting MTP weights")
        if any(Path(save_directory).glob("model_mtp*.safetensors")):
            raise ValueError("MTP output shard already exists; refusing to overwrite it")
        source_paths = {}
        by_source = {}
        for name in sorted(mtp_keys):
            by_source.setdefault(checkpoint["weight_map"][name], set()).add(name)
        for filename, keys in by_source.items():
            path = _checkpoint_file(checkpoint["source"], filename, checkpoint["revision"])
            if path is None:
                raise ValueError(f"Missing source MTP shard: {filename}")
            header = get_safetensors_header(path)
            _validate_mtp_header(header, keys)
            if any(header[name] != checkpoint["mtp_metadata"][name] for name in keys):
                raise ValueError("Source MTP metadata changed after preparation")
            source_paths[filename] = path
        sizes = {
            name: item["data_offsets"][1] - item["data_offsets"][0]
            for name, item in sorted(checkpoint["mtp_metadata"].items())
        }
        split = split_state_dict_into_shards_factory(
            sizes, get_storage_size=lambda size: size,
            filename_pattern="model_mtp{suffix}.safetensors", max_shard_size=max_shard_size,
        )
        for filename, names in split.filename_to_tensors.items():
            tensors = {}
            shard_sources = {}
            for name in names:
                shard_sources.setdefault(checkpoint["weight_map"][name], []).append(name)
            for source_filename, keys in shard_sources.items():
                with safe_open(source_paths[source_filename], framework="pt", device="cpu") as file:
                    for name in keys:
                        tensors[name] = file.get_tensor(name).contiguous()
            path = os.path.join(save_directory, filename)
            save_file(tensors, path, metadata={"format": "pt"})
            del tensors
            header = get_safetensors_header(path)
            if set(header) - {"__metadata__"} != set(names):
                raise ValueError("MTP restoration did not preserve the source tensor keys")
            for name in names:
                original = checkpoint["mtp_metadata"][name]
                if header[name]["shape"] != original["shape"] or header[name]["dtype"] != original["dtype"]:
                    raise ValueError(f"MTP restoration changed tensor metadata: {name}")
            headers[filename] = header
        combined_map = {**saved_map, **split.tensor_to_filename}
        index_path = os.path.join(save_directory, "model.safetensors.index.json")
        if os.path.isfile(index_path):
            with open(index_path, encoding="utf-8") as file:
                index = json.load(file)
        else:
            index = {"metadata": {}}
        index["weight_map"] = combined_map
        index.setdefault("metadata", {})["total_size"] = sum(
            headers[filename][name]["data_offsets"][1] - headers[filename][name]["data_offsets"][0]
            for name, filename in combined_map.items()
        )
        with open(index_path, "w", encoding="utf-8") as file:
            json.dump(index, file, indent=2, sort_keys=True)

    config_path = os.path.join(save_directory, "config.json")
    with open(config_path, encoding="utf-8") as file:
        final_config = json.load(file)
    if checkpoint["preserve_config"]:
        original_config = deepcopy(checkpoint["config"])
        original_config["quantization_config"] = final_config["quantization_config"]
        final_config = original_config
    elif mtp_keys:
        for location in (None, "text_config"):
            original = checkpoint["config"] if location is None else checkpoint["config"].get(location, {})
            fields = {
                name: value for name, value in original.items()
                if name in (*_MTP_COUNT_FIELDS, "index_share_for_mtp_iteration")
                or name.startswith("mtp_")
            }
            target = final_config if location is None else final_config.get(location)
            if target is not None:
                for name in list(target):
                    if name in (*_MTP_COUNT_FIELDS, "index_share_for_mtp_iteration") or name.startswith("mtp_"):
                        del target[name]
            if fields:
                target = final_config if location is None else final_config.setdefault(location, {})
                target.update(deepcopy(fields))
    if mtp_keys:
        ignore = final_config["quantization_config"].setdefault("ignore", [])
        existing = set(ignore)
        ignore.extend(name for name in checkpoint["mtp_ignore"] if name not in existing)
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(final_config, file, indent=2, ensure_ascii=False, sort_keys=True)
