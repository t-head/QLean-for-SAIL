# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import math
import os
import re
import shutil
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp

import torch
import torch.distributed as dist
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
from compressed_tensors.quantization import QuantizationStrategy
from compressed_tensors.utils.safetensors_load import get_safetensors_header
from huggingface_hub.serialization import split_state_dict_into_shards_factory
from safetensors import safe_open
from safetensors.torch import save_file
from transformers.core_model_loading import WeightRenaming, rename_source_key

from qlean.modifiers.gptq.distributed import (
    greedy_bin_packing,
    is_distributed,
    raise_for_distributed_errors,
    source_rank,
)
from qlean.modifiers.gptq.moe_support import validate_int4_expert_modules
from qlean.modifiers.save_utils import (
    _checkpoint_weight_map,
    _mtp_layer_count,
    WA_for_sglang_config,
    finalize_checkpoint_preservation,
    prepare_checkpoint_preservation,
    preprocess_for_acext_a8w4,
    validate_acext_a8w4_shape,
)


_FLOAT_DTYPES = {"F16", "BF16", "F32", "F64"}
_SCALE_DTYPES = {"F16", "BF16", "F32"}
_INTEGER_DTYPES = {"I8", "U8", "I16", "I32", "I64"}
_CONVERTED_QPARAM_SUFFIXES = (
    "_packed", "_shape", "_zero_point", "_g_idx", "_global_scale",
    "_zero_point_packed", "_scale_packed",
)
_INDEX_NAME = "model.safetensors.index.json"

logger = logging.getLogger(__name__)


class _Int4ExportDevice:
    def __init__(self, value):
        self._warned = False
        if value == "auto":
            from qlean.utils.dev import get_main_device

            try:
                self.device = get_main_device()
            except (RuntimeError, NotImplementedError) as error:
                self.device = torch.device("cpu")
                self._disable(error)
        else:
            try:
                self.device = torch.device(value)
            except (RuntimeError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid GPTQ INT4 export device {value!r}") from error
            if self.device.type != "cpu" and self.device.index is None and dist.is_initialized():
                self.device = torch.device(
                    self.device.type, int(os.environ.get("LOCAL_RANK", dist.get_rank()))
                )

    def _disable(self, error):
        if not self._warned:
            logger.warning(
                "GPTQ INT4 export failed on %s; retrying on CPU and disabling "
                "accelerator packing for this rank: %s",
                self.device,
                error,
            )
            self._warned = True
        module = getattr(torch, self.device.type, None)
        empty_cache = getattr(module, "empty_cache", None)
        self.device = torch.device("cpu")
        if empty_cache is not None:
            try:
                empty_cache()
            except (RuntimeError, NotImplementedError):
                pass

    @staticmethod
    def _pack(tensor, bits, shape):
        tensor = unpack_from_int32(tensor, bits, torch.Size(shape))
        return preprocess_for_acext_a8w4(tensor) if bits == 4 else tensor

    def pack(self, tensor, bits, shape):
        if bits != 4 or self.device.type == "cpu":
            return self._pack(tensor, bits, shape)
        try:
            return self._pack(tensor.to(self.device), bits, shape).cpu()
        except (RuntimeError, NotImplementedError) as error:
            self._disable(error)
            return self._pack(tensor, bits, shape)


def _attached_weight_modules(model):
    return {
        name: module
        for name, module in model.named_modules()
        if getattr(getattr(module, "quantization_scheme", None), "weights", None)
        is not None
    }


def _validate_no_activation_or_kv(model, label):
    # Scan the whole model, including attention and activation-only companions.
    from compressed_tensors.modeling.kvcache import QuantizedKVCache

    for name, module in model.named_modules():
        scheme = getattr(module, "quantization_scheme", None)
        if scheme is not None and (
            getattr(scheme, "input_activations", None) is not None
            or getattr(scheme, "output_activations", None) is not None
        ):
            raise ValueError(f"{label} does not support activation quantization on '{name}'")
        config = getattr(module, "config", None)
        quantization_config = getattr(config, "quantization_config", None)
        configured_kv = (
            quantization_config.get("kv_cache_scheme") if isinstance(quantization_config, dict)
            else getattr(quantization_config, "kv_cache_scheme", None)
        )
        if (
            getattr(module, "kv_cache_scheme", None) is not None
            or getattr(scheme, "kv_cache_scheme", None) is not None
            or getattr(config, "kv_cache_scheme", None) is not None
            or configured_kv is not None
            or isinstance(module, QuantizedKVCache)
            or isinstance(getattr(module, "kv_cache", None), QuantizedKVCache)
        ):
            raise ValueError(f"{label} does not support KV cache quantization on '{name}'")


def _validate_channel_scheme(name, module, bits, label):
    weights = module.quantization_scheme.weights
    if (
        weights.type != "int"
        or weights.num_bits not in bits
        or not weights.symmetric
        or weights.strategy != "channel"
        or weights.group_size is not None
        or weights.block_structure is not None
        or weights.scale_dtype not in (None, torch.float16, torch.bfloat16, torch.float32)
        or weights.dynamic
    ):
        bit_name = "/".join(f"INT{bit}" for bit in bits)
        raise ValueError(
            f"{label} requires symmetric per-channel {bit_name} weights "
            f"without activation quantization; unsupported scheme on '{name}'"
        )
    if not isinstance(module, torch.nn.Linear):
        raise ValueError(f"{label} requires a Linear weight on '{name}'")
    if module.out_features <= 0 or module.in_features <= 0:
        raise ValueError(f"{label} requires a nonempty 2D Linear weight on '{name}'")


def a8w4_quantized_modules(model, *, require_int4=False):
    modules = _attached_weight_modules(model)
    int4_modules = {
        name: module
        for name, module in modules.items()
        if module.quantization_scheme.weights.type == "int"
        and module.quantization_scheme.weights.num_bits == 4
        and module.quantization_scheme.weights.strategy == "channel"
    }
    for name, module in modules.items():
        weights = module.quantization_scheme.weights
        if (
            weights.type == "int"
            and weights.num_bits == 4
            and weights.strategy not in (
                QuantizationStrategy.CHANNEL,
                QuantizationStrategy.BLOCK,
            )
            and not (
                weights.strategy == QuantizationStrategy.GROUP
                and weights.group_size == 128
            )
        ):
            raise ValueError(
                "ACExt A8W4 requires symmetric per-channel INT4/INT8 weights "
                f"without activation quantization; unsupported scheme on '{name}'"
            )
    if not int4_modules:
        if require_int4:
            raise ValueError("ACExt A8W4 requires at least one INT4 Linear target")
        return {}
    validate_int4_expert_modules(model, int4_modules)
    _validate_no_activation_or_kv(model, "ACExt A8W4")
    for name, module in modules.items():
        _validate_channel_scheme(name, module, (4, 8), "ACExt A8W4")
        if module.quantization_scheme.weights.num_bits == 4:
            try:
                validate_acext_a8w4_shape((module.out_features, module.in_features))
            except ValueError as error:
                raise ValueError(f"Invalid A8W4 weight dimensions on '{name}': {error}") from error
    return modules


def group_quantized_modules(model):
    """Return int4/int8 group-128 modules saved as standard compressed-tensors."""
    return {
        name: module
        for name, module in _attached_weight_modules(model).items()
        if module.quantization_scheme.weights.type == "int"
        and module.quantization_scheme.weights.strategy == "group"
        and module.quantization_scheme.weights.group_size == 128
        and module.quantization_scheme.weights.num_bits in (4, 8)
    }


def block_quantized_modules(model):
    """Return int4/int8 block-strategy modules saved as standard compressed-tensors."""
    return {
        name: module
        for name, module in _attached_weight_modules(model).items()
        if module.quantization_scheme.weights.type == "int"
        and module.quantization_scheme.weights.strategy == "block"
        and module.quantization_scheme.weights.num_bits in (4, 8)
    }


def int8_quantized_modules(model, *, require_int8=False):
    """Resolve only pure static channel INT8; any integer INT4 takes A8W4 priority."""
    modules = _attached_weight_modules(model)
    if any(
        module.quantization_scheme.weights.type == "int"
        and module.quantization_scheme.weights.num_bits == 4
        for module in modules.values()
    ):
        return {}
    has_channel_int8 = any(
        module.quantization_scheme.weights.type == "int"
        and module.quantization_scheme.weights.num_bits == 8
        and module.quantization_scheme.weights.strategy == "channel"
        for module in modules.values()
    )
    if not has_channel_int8:
        if require_int8:
            raise ValueError("INT8 export requires at least one channel INT8 target")
        return {}

    _validate_no_activation_or_kv(model, "INT8 export")
    for name, module in modules.items():
        _validate_channel_scheme(name, module, (8,), "INT8 export")
    return modules


def _weight_module_name(name):
    return "" if name == "weight" else name.removesuffix(".weight")


def _is_weight(name):
    return name == "weight" or name.endswith(".weight")


def _pure_renamings(model):
    # Stale fused loading converters do not describe the individual Linear weights.
    renamings = [
        mapping for mapping in (getattr(model, "_weight_conversions", None) or ())
        if isinstance(mapping, WeightRenaming)
    ]
    return renamings, [mapping.reverse_transform() for mapping in reversed(renamings)]


def _checkpoint_headers(directory, label, *, check_total_size=False):
    weight_map = _checkpoint_weight_map(directory)
    if not weight_map or not isinstance(weight_map, dict):
        raise ValueError(f"{label} checkpoint has no indexed tensors")
    for name, filename in weight_map.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"Invalid {label} checkpoint tensor name: {name}")
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or "\\" in filename
            or not filename.endswith(".safetensors")
        ):
            raise ValueError(f"Invalid {label} checkpoint filename: {filename}")
    filenames = sorted(set(weight_map.values()))
    if {path.name for path in directory.glob("*.safetensors")} != set(filenames):
        raise ValueError(f"{label} checkpoint index does not match its shard files")
    headers = {}
    for filename in filenames:
        # Opening validates the actual safetensors header/offsets without loading weights.
        with safe_open(directory / filename, framework="pt", device="cpu") as file:
            keys = set(file.keys())
        shard_header = get_safetensors_header(directory / filename)
        if keys != set(shard_header) - {"__metadata__"}:
            raise ValueError(f"Invalid {label} checkpoint header in '{filename}'")
        for name in sorted(keys):
            if name in headers or weight_map.get(name) != filename:
                raise ValueError(f"Invalid {label} checkpoint index for '{name}'")
            headers[name] = shard_header[name]
    if set(headers) != set(weight_map):
        raise ValueError(f"{label} checkpoint has missing indexed tensors")
    total_size = sum(item["data_offsets"][1] - item["data_offsets"][0] for item in headers.values())
    if check_total_size:
        index_path = directory / _INDEX_NAME
        if not index_path.is_file():
            raise ValueError(f"{label} checkpoint requires a safetensors index")
        with index_path.open(encoding="utf-8") as file:
            metadata = json.load(file).get("metadata", {})
        if metadata.get("total_size") != total_size:
            raise ValueError(f"Invalid {label} checkpoint index total_size")
    return weight_map, headers, total_size


def _read_tensor(directory, weight_map, name):
    with safe_open(directory / weight_map[name], framework="pt", device="cpu") as file:
        return file.get_tensor(name)


def _consume_qparams(directory, weight_map, headers, weight_name, shape, label, *, packed):
    consumed = set()
    unsupported = ("_g_idx", "_global_scale", "_zero_point_packed", "_scale_packed")
    if not packed:
        unsupported += ("_packed",)
    for suffix in unsupported:
        if f"{weight_name}{suffix}" in headers:
            raise ValueError(f"Unsupported {label} parameter: '{weight_name}{suffix}'")
    shape_name = f"{weight_name}_shape"
    if shape_name in headers:
        item = headers[shape_name]
        if (
            item["dtype"] not in _INTEGER_DTYPES
            or item["shape"] != [2]
            or tuple(_read_tensor(directory, weight_map, shape_name).tolist()) != shape
        ):
            raise ValueError(f"Invalid {label} original weight shape for '{weight_name}'")
        consumed.add(shape_name)
    elif packed:
        raise ValueError(f"Missing {label} shape or scale for '{weight_name}'")
    zero_name = f"{weight_name}_zero_point"
    if zero_name in headers:
        item = headers[zero_name]
        if item["shape"] != [shape[0], 1] or item["dtype"] not in (_INTEGER_DTYPES | _FLOAT_DTYPES):
            raise ValueError(f"Invalid {label} zero point for '{weight_name}'")
        if torch.any(_read_tensor(directory, weight_map, zero_name) != 0):
            raise ValueError(f"Nonzero {label} zero point for '{weight_name}'")
        consumed.add(zero_name)
    return consumed


def _ignore_covers(name, pattern):
    if pattern.startswith("re:"):
        try:
            return re.match(pattern[3:], name) is not None
        except re.error as error:
            raise ValueError(f"Invalid checkpoint ignore regex: '{pattern}'") from error
    # Protect both subtree-aware consumers and exact/suffix consumers.
    return name == pattern or name.startswith(f"{pattern}.") or name.endswith(f".{pattern}")


def _saved_ignored_modules(model, headers, saved_quantized, *, checkpoint=None):
    renamings, _ = _pure_renamings(model)
    actual_modules = dict(model.named_modules())
    ignored = set()
    for weight_name, item in headers.items():
        if not _is_weight(weight_name) or _weight_module_name(weight_name) in saved_quantized:
            continue
        if item["dtype"] not in _FLOAT_DTYPES:
            raise ValueError(f"Unsupported unquantized saved weight dtype for '{weight_name}'")
        if len(item["shape"]) != 2 or not all(item["shape"]):
            continue
        model_key, _ = rename_source_key(weight_name, renamings, [])
        module = actual_modules.get(_weight_module_name(model_key))
        # Do not use CT's class-name heuristic (Gate/Gating also matches norms/containers).
        # Unknown saved 2D weights include source-only MTP weights absent from the model.
        if module is None or (
            isinstance(module, torch.nn.Linear)
            and getattr(getattr(module, "quantization_scheme", None), "weights", None) is None
            and item["shape"] == [module.out_features, module.in_features]
        ):
            ignored.add(_weight_module_name(weight_name))
    if checkpoint is not None and checkpoint["mtp_keys"]:
        ignored.update(checkpoint["mtp_ignore"])
    for pattern in list(ignored):
        if any(_ignore_covers(name, pattern) for name in saved_quantized):
            if pattern.startswith("re:"):
                raise ValueError(f"Checkpoint ignore '{pattern}' covers a quantized module")
            # An actual unquantized weight can have quantized descendants. Exclude only
            # this saved bare module name, never the whole ancestor subtree.
            ignored.remove(pattern)
            ignored.add(f"re:^{re.escape(pattern)}$")
    return sorted(ignored)


def _saved_quantized_weights(model, modules, headers, label):
    renamings, _ = _pure_renamings(model)
    saved = {}
    for weight_name in headers:
        if not _is_weight(weight_name):
            continue
        model_key, _ = rename_source_key(weight_name, renamings, [])
        module_name = _weight_module_name(model_key)
        if module_name not in modules:
            continue
        if module_name in saved:
            raise ValueError(f"Duplicate {label} saved weight for '{module_name}'")
        saved[module_name] = weight_name
    if set(saved) != set(modules):
        missing = sorted(set(modules) - set(saved))
        raise ValueError(f"{label} checkpoint is missing quantized weights: {missing[:5]}")
    return saved


def _validate_custom_checkpoint(save_directory, model, modules, *, a8w4):
    directory = Path(save_directory)
    label = "A8W4" if a8w4 else "INT8"
    weight_map, headers, _ = _checkpoint_headers(directory, label, check_total_size=True)
    with (directory / "config.json").open(encoding="utf-8") as file:
        config = json.load(file)
    qc = config.get("quantization_config")
    method = "mixed_precision_w4" if a8w4 else "w8a8_int8"
    if (
        not isinstance(qc, dict)
        or qc.get("quant_method") != method
        or any(key in qc for key in (
            "config_groups", "format", "kv_cache_scheme", "quantization_status",
            "global_compression_ratio", "sparsity_config", "transform_config", "version",
        ))
        or qc.get("activation_scheme", "dynamic") != "dynamic"
    ):
        raise ValueError(f"Invalid final {label} quantization_config")
    ignored = qc.get("ignore")
    if not isinstance(ignored, list) or any(not isinstance(name, str) or not name for name in ignored):
        raise ValueError(f"Invalid final {label} ignore list")
    saved = _saved_quantized_weights(model, modules, headers, label)
    allowed_scales = set()
    for module_name, weight_name in saved.items():
        module = modules[module_name]
        bits = module.quantization_scheme.weights.num_bits
        n, k = module.out_features, module.in_features
        item = headers[weight_name]
        if item["dtype"] != "I8" or item["shape"] != [n, k // (2 if bits == 4 else 1)]:
            raise ValueError(f"Invalid final {label} weight dtype or shape for '{weight_name}'")
        scale_names = [f"{weight_name}_scale"]
        if a8w4:
            scale_names.append(f"{_weight_module_name(weight_name)}.scale")
        present_scales = [name for name in scale_names if name in headers]
        if len(present_scales) != 1:
            raise ValueError(f"Missing or ambiguous final {label} scale for '{weight_name}'")
        scale_name = present_scales[0]
        allowed_scales.add(scale_name)
        if headers[scale_name]["dtype"] != "F32" or headers[scale_name]["shape"] != [n, 1]:
            raise ValueError(f"Invalid final {label} scale dtype or shape for '{scale_name}'")
        scale = _read_tensor(directory, weight_map, scale_name)
        if not torch.isfinite(scale).all() or torch.any(scale <= 0):
            raise ValueError(f"Invalid final {label} scale for '{scale_name}'")
        saved_name = _weight_module_name(weight_name)
        if any(_ignore_covers(saved_name, pattern) for pattern in ignored):
            raise ValueError(f"Final {label} ignore covers quantized module '{saved_name}'")
    renamings, _ = _pure_renamings(model)
    for name in headers:
        if name in allowed_scales:
            continue
        for suffix in (*_CONVERTED_QPARAM_SUFFIXES, "_scale", ".scale"):
            if not name.endswith(suffix):
                continue
            bare_weight = name.removesuffix(suffix)
            if suffix == ".scale":
                bare_weight = f"{bare_weight}.weight"
            if not _is_weight(bare_weight):
                continue
            model_key, _ = rename_source_key(bare_weight, renamings, [])
            if _weight_module_name(model_key) in modules:
                raise ValueError(f"Unconverted {label} parameter: '{name}'")
    required_ignore = _saved_ignored_modules(
        model, headers, {_weight_module_name(name) for name in saved.values()}
    )
    if any(
        name not in ignored and not any(_ignore_covers(name, pattern) for pattern in ignored)
        for name in required_ignore
    ):
        raise ValueError(f"Final {label} ignore is missing an unquantized saved Linear weight")


def validate_int8_checkpoint(save_directory, model):
    """Validate the final W8A8 INT8 config, exact index/headers and per-channel scales."""
    modules = int8_quantized_modules(model, require_int8=True)
    if not modules:
        raise ValueError("INT8 checkpoint validation cannot be used for an INT4 model")
    _validate_custom_checkpoint(save_directory, model, modules, a8w4=False)


def _staged_shard_manifest(sizes, max_shard_size):
    split = split_state_dict_into_shards_factory(
        sizes,
        get_storage_size=lambda size: size,
        filename_pattern="model{suffix}.safetensors",
        max_shard_size=max_shard_size,
    )
    return {
        "filename_to_tensors": {
            filename: list(names) for filename, names in split.filename_to_tensors.items()
        },
        "metadata": split.metadata,
        "weight_map": split.tensor_to_filename,
    }


def validate_export_workers(workers):
    if type(workers) is not int or workers < 1:
        raise ValueError("gptq_export_workers must be a positive integer")
    return workers


def _write_staged_shard(directory, staging, weight_map, sources, weights, scale_keys,
                        manifest, filename, *, label, pack_device=None):
    tensors = {}
    by_source = {}
    for name in manifest["filename_to_tensors"][filename]:
        by_source.setdefault(weight_map[sources[name]], []).append(name)
    for source_filename, output_names in by_source.items():
        with safe_open(directory / source_filename, framework="pt", device="cpu") as file:
            for name in output_names:
                tensor = file.get_tensor(sources[name])
                if name in weights:
                    bits, shape = weights[name]
                    if pack_device is not None:
                        tensor = pack_device.pack(tensor, bits, shape)
                    # INT8 is already quantized by CT; never RTN/requantize here.
                    expected_shape = (shape[0], shape[1] // (2 if bits == 4 else 1))
                    if tensor.dtype != torch.int8 or tuple(tensor.shape) != expected_shape:
                        raise ValueError(f"Invalid {label} exported weight for '{name}'")
                elif name in scale_keys:
                    tensor = tensor.float()
                    if not torch.isfinite(tensor).all() or torch.any(tensor <= 0):
                        raise ValueError(f"Invalid {label} scale for '{name}'")
                tensors[name] = tensor.contiguous()
    save_file(tensors, staging / filename, metadata={"format": "pt"})
    return filename


def _write_staged_shards(directory, staging, weight_map, sources, weights, scale_keys,
                         manifest, filenames, *, a8w4, export_device="auto", workers=1,
                         label=None):
    label = label or ("A8W4" if a8w4 else "INT8")
    pack_device = _Int4ExportDevice(export_device) if a8w4 else None
    if a8w4:
        workers = 1
    else:
        validate_export_workers(workers)
    rank = dist.get_rank() if dist.is_initialized() else 0
    write = partial(
        _write_staged_shard, directory, staging, weight_map, sources, weights, scale_keys,
        manifest, label=label, pack_device=pack_device,
    )
    completed = 0

    def log_shard(filename):
        nonlocal completed
        completed += 1
        logger.info(
            "GPTQ %s export rank %d wrote shard %d/%d: %s",
            label, rank, completed, len(filenames), filename,
        )

    if workers == 1 or len(filenames) < 2:
        for filename in filenames:
            log_shard(write(filename))
        return
    remaining = iter(filenames)
    with ThreadPoolExecutor(max_workers=min(workers, len(filenames))) as pool:
        pending = {pool.submit(write, next(remaining)) for _ in range(min(workers, len(filenames)))}
        try:
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    log_shard(future.result())
                for _ in done:
                    filename = next(remaining, None)
                    if filename is not None:
                        pending.add(pool.submit(write, filename))
        except BaseException:
            for future in pending:
                future.cancel()
            # Executor shutdown drains active writers before staging can be removed.
            raise


def _write_staged_index(staging, manifest):
    with (staging / _INDEX_NAME).open("w", encoding="utf-8") as file:
        json.dump(
            {"metadata": manifest["metadata"], "weight_map": manifest["weight_map"]},
            file, indent=2, sort_keys=True,
        )


def _finalize_staging(staging, model, modules, config, checkpoint, max_shard_size, *, a8w4,
                      expert_patterns):
    from qlean.core.checkpoint import validate_expert_keys_in_checkpoint

    config["quantization_config"] = {
        "quant_method": "mixed_precision_w4" if a8w4 else "w8a8_int8",
        **({"activation_scheme": "dynamic"} if a8w4 else {}),
        "ignore": [],
    }
    with (staging / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, sort_keys=True)
    if checkpoint is not None:
        finalize_checkpoint_preservation(model, checkpoint, staging, max_shard_size)
    # Preservation can add shards and restore source configuration fields.
    weight_map, headers, total_size = _checkpoint_headers(staging, "custom export")
    saved = _saved_quantized_weights(model, modules, headers, "custom export")
    with (staging / "config.json").open(encoding="utf-8") as file:
        config = json.load(file)
    config["quantization_config"]["ignore"] = _saved_ignored_modules(
        model, headers, {_weight_module_name(name) for name in saved.values()}, checkpoint=checkpoint
    )
    with (staging / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False, sort_keys=True)
    with (staging / _INDEX_NAME).open(encoding="utf-8") as file:
        index = json.load(file)
    index.setdefault("metadata", {})["total_size"] = total_size
    index["weight_map"] = weight_map
    with (staging / _INDEX_NAME).open("w", encoding="utf-8") as file:
        json.dump(index, file, indent=2, sort_keys=True)
    _validate_custom_checkpoint(staging, model, modules, a8w4=a8w4)
    validate_expert_keys_in_checkpoint(model, staging, expert_patterns)
    return [*sorted(set(weight_map.values())), _INDEX_NAME, "config.json"]


def _publish_staged_checkpoint(directory, staging, filenames, publication_names, *, label):
    original_names = [*filenames, "config.json"]
    if (directory / _INDEX_NAME).exists():
        original_names.append(_INDEX_NAME)
    for filename in set(publication_names) - set(original_names):
        if (directory / filename).exists():
            raise ValueError(f"{label} publication would overwrite existing file '{filename}'")
    # Canonical recovery hardlinks must survive until the *entire* rollback completes.
    recovery = Path(mkdtemp(prefix=f".qlean-{label.lower()}-recovery-", dir=directory))
    published = []
    try:
        for filename in original_names:
            os.link(directory / filename, recovery / filename)
        for filename in publication_names:
            os.replace(staging / filename, directory / filename)
            published.append(filename)
        for filename in set(filenames) - set(publication_names):
            (directory / filename).unlink()
    except BaseException:
        try:
            for filename in original_names:
                if (recovery / filename).exists():
                    restore_path = recovery / f".restore-{filename}"
                    os.link(recovery / filename, restore_path)
                    os.replace(restore_path, directory / filename)
            for filename in set(published) - set(original_names):
                (directory / filename).unlink()
        except BaseException as error:
            raise RuntimeError(
                f"{label} publication rollback failed; original files remain in {recovery}"
            ) from error
        shutil.rmtree(recovery)
        raise
    shutil.rmtree(recovery)


def _prepare_export_checkpoint(save_directory, model, *, a8w4, expert_patterns):
    from qlean.core.checkpoint import expert_key_patterns

    if expert_patterns is None:
        expert_patterns = expert_key_patterns(model)
    modules = (
        a8w4_quantized_modules(model, require_int4=True) if a8w4
        else int8_quantized_modules(model, require_int8=True)
    )
    if not modules:
        raise ValueError("INT8 export cannot be used for an INT4 model; use A8W4")
    label = "A8W4" if a8w4 else "INT8"
    directory = Path(save_directory)
    weight_map, headers, _ = _checkpoint_headers(directory, f"{label} input")
    filenames = sorted(set(weight_map.values()))
    uses_v4_scale_names = a8w4 and any(
        ".ffn.experts." in f".{name}" and name.endswith(".weight_packed")
        for name in weight_map
    )
    renamings, save_renamings = _pure_renamings(model)
    sources, sizes, weights = {}, {}, {}
    scale_keys, consumed, exported_modules = set(), set(), set()
    for source_name, item in headers.items():
        if a8w4:
            if not source_name.endswith(".weight_packed"):
                continue
            weight_name = source_name.removesuffix("_packed")
        else:
            if not _is_weight(source_name):
                continue
            weight_name = source_name
        model_key, _ = rename_source_key(weight_name, renamings, [])
        module_name = _weight_module_name(model_key)
        if module_name not in modules:
            if a8w4:
                raise ValueError(f"Unsupported A8W4 saved weight mapping: '{source_name}'")
            continue
        if module_name in exported_modules:
            raise ValueError(f"Duplicate {label} export for '{module_name}'")
        module = modules[module_name]
        bits = module.quantization_scheme.weights.num_bits
        shape = (module.out_features, module.in_features)
        source_shape = [shape[0], math.ceil(shape[1] * bits / 32)] if a8w4 else list(shape)
        source_scale = f"{weight_name}_scale"
        if source_scale not in headers:
            raise ValueError(f"Missing {label} shape or scale for '{weight_name}'")
        if (
            item["dtype"] != ("I32" if a8w4 else "I8")
            or item["shape"] != source_shape
            or headers[source_scale]["shape"] != [shape[0], 1]
            or headers[source_scale]["dtype"] not in _SCALE_DTYPES
        ):
            raise ValueError(f"Invalid {label} quantized weight or scale for '{weight_name}'")
        consumed.update(_consume_qparams(
            directory, weight_map, headers, weight_name, shape, label, packed=a8w4
        ))
        output_weight, _ = rename_source_key(model_key, save_renamings, [], reverse=True)
        output_scale = (
            f"{_weight_module_name(output_weight)}.scale" if uses_v4_scale_names
            else f"{output_weight}_scale"
        )
        if output_weight in sources or output_scale in sources:
            raise ValueError(f"Duplicate {label} output key for '{output_weight}'")
        sources[output_weight] = source_name
        sizes[output_weight] = math.prod(shape) // (2 if bits == 4 else 1)
        weights[output_weight] = (bits, shape)
        sources[output_scale] = source_scale
        sizes[output_scale] = shape[0] * 4
        scale_keys.add(output_scale)
        consumed.update((source_name, source_scale))
        exported_modules.add(module_name)
    if exported_modules != set(modules):
        missing = sorted(set(modules) - exported_modules)
        raise ValueError(f"{label} checkpoint is missing quantized weights: {missing[:5]}")
    for name, item in headers.items():
        if name in consumed:
            continue
        if name in sources:
            raise ValueError(f"{label} output key would overwrite '{name}'")
        sources[name] = name
        start, end = item["data_offsets"]
        sizes[name] = end - start
    with (directory / "config.json").open(encoding="utf-8") as file:
        config = json.load(file)
    if (config.get("quantization_config") or {}).get("kv_cache_scheme") is not None:
        raise ValueError(f"{label} export does not support KV cache quantization")
    return {
        "directory": directory,
        "modules": modules,
        "filenames": filenames,
        "weight_map": weight_map,
        "sources": sources,
        "sizes": sizes,
        "weights": weights,
        "scale_keys": scale_keys,
        "config": config,
        "expert_patterns": expert_patterns,
    }


def _export_checkpoint(save_directory, model, max_shard_size, checkpoint, *, a8w4,
                       expert_patterns, export_device="auto", workers=2):
    if not a8w4:
        validate_export_workers(workers)
    context = _prepare_export_checkpoint(
        save_directory, model, a8w4=a8w4, expert_patterns=expert_patterns
    )
    label = "A8W4" if a8w4 else "INT8"
    manifest = _staged_shard_manifest(context["sizes"], max_shard_size)
    with TemporaryDirectory(
        prefix=f".qlean-{label.lower()}-", dir=context["directory"]
    ) as staging_directory:
        staging = Path(staging_directory)
        _write_staged_shards(
            context["directory"], staging, context["weight_map"], context["sources"],
            context["weights"], context["scale_keys"], manifest,
            list(manifest["filename_to_tensors"]), a8w4=a8w4,
            export_device=export_device, workers=workers,
        )
        _write_staged_index(staging, manifest)
        publication_names = _finalize_staging(
            staging, model, context["modules"], context["config"], checkpoint,
            max_shard_size, a8w4=a8w4, expert_patterns=context["expert_patterns"],
        )
        _publish_staged_checkpoint(
            context["directory"], staging, context["filenames"], publication_names,
            label=label,
        )


def _collect_distributed_failure(error, phase, group):
    try:
        raise_for_distributed_errors(error, phase, group)
    except BaseException as failure:
        return failure
    return None


def _cleanup_distributed_staging(staging, source, group):
    error = None
    if dist.get_rank() == source and staging is not None:
        try:
            shutil.rmtree(staging)
        except FileNotFoundError:
            pass
        except BaseException as caught:
            error = caught
    return _collect_distributed_failure(error, "staging cleanup", group)


def _export_a8w4_distributed(save_directory, model, max_shard_size, checkpoint,
                             expert_patterns, export_device, group):
    source = source_rank(group)
    rank = dist.get_rank()
    staging = None
    setup = [None]
    setup_error = None
    if rank == source:
        try:
            staging = Path(mkdtemp(prefix=".qlean-a8w4-", dir=save_directory))
            marker = os.urandom(16).hex()
            (staging / ".qlean-shared-marker").write_text(marker, encoding="utf-8")
            setup[0] = (str(staging), marker)
        except BaseException as error:
            setup_error = error
    failure = _collect_distributed_failure(setup_error, "staging creation", group)
    if failure is not None:
        cleanup_failure = _cleanup_distributed_staging(staging, source, group)
        if cleanup_failure is not None:
            raise cleanup_failure from failure
        raise failure
    dist.broadcast_object_list(setup, src=source, group=group)
    staging = Path(setup[0][0])
    marker = setup[0][1]
    try:
        visible = (staging / ".qlean-shared-marker").read_text(encoding="utf-8") == marker
    except OSError:
        visible = False
    visibility = [None] * dist.get_world_size(group)
    dist.all_gather_object(visibility, visible, group=group)
    parallel = all(visibility)
    if rank == source and not parallel:
        logger.warning(
            "GPTQ A8W4 staging is not shared across all ranks; falling back to source-only export"
        )

    context = None
    prepare_error = None
    if parallel or rank == source:
        try:
            context = _prepare_export_checkpoint(
                save_directory, model, a8w4=True, expert_patterns=expert_patterns
            )
        except BaseException as error:
            prepare_error = error
    failure = _collect_distributed_failure(prepare_error, "export planning", group)
    if failure is not None:
        cleanup_failure = _cleanup_distributed_staging(staging, source, group)
        if cleanup_failure is not None:
            raise cleanup_failure from failure
        raise failure

    plan = [None]
    plan_error = None
    if rank == source:
        try:
            manifest = _staged_shard_manifest(context["sizes"], max_shard_size)
            filenames = list(manifest["filename_to_tensors"])
            if parallel:
                _, assignments, _ = greedy_bin_packing(
                    filenames,
                    dist.get_world_size(group),
                    lambda filename: sum(
                        context["sizes"][name]
                        for name in manifest["filename_to_tensors"][filename]
                    ),
                )
            else:
                assignments = [[] for _ in range(dist.get_world_size(group))]
                assignments[source] = filenames
            plan[0] = (manifest, assignments)
        except BaseException as error:
            plan_error = error
    failure = _collect_distributed_failure(plan_error, "shard planning", group)
    if failure is not None:
        cleanup_failure = _cleanup_distributed_staging(staging, source, group)
        if cleanup_failure is not None:
            raise cleanup_failure from failure
        raise failure
    dist.broadcast_object_list(plan, src=source, group=group)
    manifest, assignments = plan[0]

    worker_error = None
    if assignments[rank]:
        try:
            _write_staged_shards(
                context["directory"], staging, context["weight_map"], context["sources"],
                context["weights"], context["scale_keys"], manifest, assignments[rank],
                a8w4=True, export_device=export_device,
            )
        except BaseException as error:
            worker_error = error
    failure = _collect_distributed_failure(worker_error, "shard export", group)
    if failure is not None:
        cleanup_failure = _cleanup_distributed_staging(staging, source, group)
        if cleanup_failure is not None:
            raise cleanup_failure from failure
        raise failure

    finalize_error = None
    if rank == source:
        try:
            _write_staged_index(staging, manifest)
            publication_names = _finalize_staging(
                staging, model, context["modules"], context["config"], checkpoint,
                max_shard_size, a8w4=True, expert_patterns=context["expert_patterns"],
            )
            _publish_staged_checkpoint(
                context["directory"], staging, context["filenames"], publication_names,
                label="A8W4",
            )
        except BaseException as error:
            finalize_error = error
    failure = _collect_distributed_failure(finalize_error, "finalization", group)
    cleanup_failure = _cleanup_distributed_staging(staging, source, group)
    if failure is not None:
        if cleanup_failure is not None:
            raise cleanup_failure from failure
        raise failure
    if cleanup_failure is not None:
        raise cleanup_failure


def export_a8w4_checkpoint(save_directory, model, max_shard_size, *, checkpoint=None,
                          expert_patterns=None, export_device="auto", process_group=None):
    """Export CT packed INT4/INT8 to ACExt A8W4, preserving source data before publication."""
    if process_group is not None and is_distributed():
        _export_a8w4_distributed(
            save_directory, model, max_shard_size, checkpoint, expert_patterns,
            export_device, process_group,
        )
        return
    _export_checkpoint(
        save_directory, model, max_shard_size, checkpoint, a8w4=True,
        expert_patterns=expert_patterns, export_device=export_device,
    )


def export_int8_checkpoint(save_directory, model, max_shard_size, *, checkpoint=None,
                          expert_patterns=None, workers=2):
    """Export CT signed INT8 codes unchanged, with FP32 scales and a W8A8 INT8 config."""
    _export_checkpoint(
        save_directory, model, max_shard_size, checkpoint, a8w4=False,
        expert_patterns=expert_patterns, workers=workers,
    )


def export_mxfp4_checkpoint(save_directory, model, max_shard_size, *, checkpoint=None,
                           expert_patterns=None, workers=2):
    from qlean.core.checkpoint import expert_key_patterns, validate_expert_keys_in_checkpoint

    validate_export_workers(workers)
    directory = Path(save_directory)
    index_path = directory / _INDEX_NAME
    weight_map, headers, total_size = _checkpoint_headers(
        directory, "MXFP4 input",
    )
    filenames = sorted(set(weight_map.values()))
    sources, output_map, by_shard = {}, {}, {filename: [] for filename in filenames}
    changed = set()
    for source_name, filename in weight_map.items():
        name = source_name.removesuffix("_packed")
        if name in sources:
            raise ValueError(f"MXFP4 rename would overwrite tensor '{name}'")
        sources[name] = source_name
        output_map[name] = filename
        by_shard[filename].append(name)
        if name != source_name:
            changed.add(filename)
    index = None
    if index_path.exists():
        with index_path.open(encoding="utf-8") as file:
            index = json.load(file)
        index["weight_map"] = output_map
        index.setdefault("metadata", {})["total_size"] = total_size
    if expert_patterns is None:
        expert_patterns = expert_key_patterns(model)
    with TemporaryDirectory(prefix=".qlean-mxfp4-", dir=directory) as staging_directory:
        staging = Path(staging_directory)
        for filename in set(filenames) - changed:
            os.link(directory / filename, staging / filename)
        _write_staged_shards(
            directory, staging, weight_map, sources, {}, set(),
            {"filename_to_tensors": by_shard}, sorted(changed),
            a8w4=False, workers=workers, label="MXFP4",
        )
        if index is not None:
            with (staging / _INDEX_NAME).open("w", encoding="utf-8") as file:
                json.dump(index, file, indent=2, sort_keys=True)
        shutil.copyfile(directory / "config.json", staging / "config.json")
        WA_for_sglang_config(staging / "config.json")
        if checkpoint is not None:
            finalize_checkpoint_preservation(model, checkpoint, staging, max_shard_size)
        final_map, final_headers, _ = _checkpoint_headers(
            staging, "MXFP4 output", check_total_size=(staging / _INDEX_NAME).exists(),
        )
        expected_keys = set(sources) | (checkpoint["mtp_keys"] if checkpoint is not None else set())
        if set(final_headers) != expected_keys:
            raise ValueError("MXFP4 export changed the checkpoint tensor set")
        for name, source_name in sources.items():
            original, saved = headers[source_name], final_headers[name]
            if any(original[field] != saved[field] for field in ("dtype", "shape")) or (
                original["data_offsets"][1] - original["data_offsets"][0]
                != saved["data_offsets"][1] - saved["data_offsets"][0]
            ):
                raise ValueError(f"MXFP4 export changed tensor metadata: '{name}'")
        validate_expert_keys_in_checkpoint(model, staging, expert_patterns)
        publication_names = [*sorted(set(final_map.values())), "config.json"]
        if (staging / _INDEX_NAME).exists():
            publication_names.append(_INDEX_NAME)
        _publish_staged_checkpoint(
            directory, staging, filenames, publication_names, label="MXFP4",
        )


def expected_save_format(model):
    if a8w4_quantized_modules(model):
        return "pack-quantized"
    if int8_quantized_modules(model):
        return "int-quantized"
    if block_quantized_modules(model):
        return "pack-quantized"
    if group_quantized_modules(model):
        return "pack-quantized"
    weights = [module.quantization_scheme.weights for module in _attached_weight_modules(model).values()]
    if weights and all(
        args.type == "float" and args.num_bits == 4 and args.group_size == 32
        for args in weights
    ):
        return "mxfp4-pack-quantized"
    return "pack-quantized"


def validate_gptq_output_directory(model, source, save_directory, modifiers):
    from qlean.core.checkpoint import validate_output_directory

    require_fresh = False
    for modifier in modifiers:
        config = modifier._resolved_config or modifier.resolve_quantization_config()
        for group in config.config_groups.values():
            weights = group.weights
            if weights is not None and weights.type == "int" and (
                weights.strategy == "channel" and weights.num_bits in (4, 8)
            ):
                require_fresh = True
    validate_output_directory(model, source, save_directory, require_fresh=require_fresh)


def prepare_source_checkpoint(model, source, save_directory, modifiers):
    config = getattr(model, "config", None)
    if config is None:
        return None
    text_config = config.get_text_config()
    preserve_config = config.model_type in {
        "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text",
    }
    has_mtp = _mtp_layer_count(text_config.to_dict()) or _mtp_layer_count(config.to_dict())
    source = getattr(model, "name_or_path", None) or source
    if source:
        source = os.fspath(source)
    missing_local_source = source and not os.path.isdir(source) and (
        os.path.isabs(source) or source.startswith(("./", "../", "~/"))
        or ".." in Path(source).parts
    )
    if not source or missing_local_source:
        if preserve_config or has_mtp:
            raise ValueError("Checkpoint preservation requires a valid source checkpoint")
        return None
    checkpoint = prepare_checkpoint_preservation(model, source, save_directory, preserve_config)
    if checkpoint is None:
        return None
    if checkpoint["visual_keys"]:
        from compressed_tensors.utils import match_named_modules

        for modifier in modifiers:
            quantization = modifier._resolved_config or modifier.resolve_quantization_config()
            for group in quantization.config_groups.values():
                if all(value is None for value in (
                    group.weights, group.input_activations, group.output_activations,
                )):
                    continue
                for name, _ in match_named_modules(model, group.targets, modifier.ignore):
                    if name.startswith(("model.visual.", "visual.")):
                        raise ValueError(
                            "Full Qwen3.5 GPTQ requires excluding vision modules "
                            "from quantization; add re:model.visual.* to ignore"
                        )
    return checkpoint


def finalize_gptq_checkpoint(
    model, save_directory, max_shard_size, expected_format, expert_patterns, *, checkpoint=None,
    export_device="auto", process_group=None, workers=2,
):
    from qlean.core.checkpoint import validate_expert_keys_in_checkpoint

    uses_a8w4 = bool(a8w4_quantized_modules(model))
    uses_int8 = expected_format == "int-quantized"
    if uses_a8w4:
        export_a8w4_checkpoint(
            save_directory, model, max_shard_size, checkpoint=checkpoint,
            expert_patterns=expert_patterns, export_device=export_device,
            process_group=process_group,
        )
        return {"output_format": "GPTQ_INT4", "scheme": "INT4_CHANNEL"}
    if uses_int8:
        export_int8_checkpoint(
            save_directory, model, max_shard_size, checkpoint=checkpoint,
            expert_patterns=expert_patterns, workers=workers,
        )
        return {"output_format": "GPTQ_INT8", "scheme": "INT8_CHANNEL"}
    if expected_format == "mxfp4-pack-quantized":
        export_mxfp4_checkpoint(
            save_directory, model, max_shard_size, checkpoint=checkpoint,
            expert_patterns=expert_patterns, workers=workers,
        )
        return {"output_format": "GPTQ_MXFP4", "scheme": "MXFP4"}
    validate_expert_keys_in_checkpoint(model, save_directory, expert_patterns)
    if checkpoint is not None:
        finalize_checkpoint_preservation(model, checkpoint, save_directory, max_shard_size)
    block_modules = block_quantized_modules(model)
    if block_modules:
        bits = {
            module.quantization_scheme.weights.num_bits
            for module in block_modules.values()
        }
        if len(bits) != 1:
            raise ValueError("Block INT4 and INT8 weights cannot be mixed in one checkpoint")
        bit = bits.pop()
        return {
            "output_format": f"GPTQ_INT{bit}_BLOCK",
            "scheme": f"INT{bit}_BLOCK",
        }
    group_modules = group_quantized_modules(model)
    if group_modules:
        bits = {
            module.quantization_scheme.weights.num_bits
            for module in group_modules.values()
        }
        if len(bits) != 1:
            raise ValueError("Group INT4 and INT8 weights cannot be mixed in one checkpoint")
        bit = bits.pop()
        return {
            "output_format": f"GPTQ_INT{bit}_GROUP",
            "scheme": f"INT{bit}_GROUP",
        }
    return {}
