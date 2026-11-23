# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import shutil
from pathlib import Path
from glob import glob
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from typing import Tuple, Dict, List

from compressed_tensors.quantization import QuantizationScheme, preset_name_to_scheme
import triton
import triton.language as tl
from compressed_tensors.utils import getattr_chain
from qlean.utils.downcast_to_mxfp import downcast_to_mxfp

import logging

logger = logging.getLogger(__name__)

__all__ = [
    "weight_quant",
    "validate_scheme",
    "copy_necessary_files",
    "FP8_SCALE_SUFFIXES",
    "find_scale_name",
    "load_file_to_cuda",
    "safe_save_file",
]

FP8_SCALE_SUFFIXES = ["_scale_inv", ".scale"]

def load_file_to_cuda(path: str) -> dict:
    """Load a safetensors file on CPU, then move every tensor to the default CUDA device."""
    state_dict = load_file(path, device="cpu")
    return {name: tensor.cuda() for name, tensor in state_dict.items()}


def _tensors_to_cpu(tensors: dict) -> dict:
    return {k: (v.cpu()).contiguous() for k, v in tensors.items()}


def safe_save_file(tensors: dict, path: str, metadata=None) -> None:
    """Save contiguous CPU tensors to a temporary file, then atomically replace path."""
    tmp_path = f"{path}.qlean_tmp"
    save_file(_tensors_to_cpu(tensors), tmp_path, metadata=metadata)
    os.replace(tmp_path, path)

def find_scale_name(weight_name: str, weight_map: dict) -> str | None:
    """Find the scale tensor name of an FP8 weight by trying candidate suffixes in order; None if absent."""
    for suffix in FP8_SCALE_SUFFIXES:
        candidate = f"{weight_name}{suffix}"
        if candidate in weight_map:
            return candidate
    if weight_name.endswith(".weight"):
        candidate = weight_name[:-len(".weight")] + ".scale"
        if candidate in weight_map:
            return candidate
    return None

def ensure_scales_in_weight_map(weight_map: dict, model_path: str) -> None:
    """If weight_map has no scale entries, scan the safetensors files under model_path to fill them in."""
    SCALE_SUFFIXES = ("input_scale", "weight_scale", "weight_scale_2", "scale_inv", "_global_scale")
    has_scale = any(name.endswith(SCALE_SUFFIXES) for name in weight_map)
    if has_scale:
        return

    logger.info("Scanning weight files for scales")
    safetensor_files = sorted(glob(os.path.join(model_path, "*.safetensors")))
    for st_file in safetensor_files:
        file_name = os.path.basename(st_file)
        with safe_open(st_file, framework="pt", device="cpu") as sf:
            for key in sf.keys():
                if key.endswith(SCALE_SUFFIXES):
                    weight_map[key] = file_name

def int8_channel_weight_quant(tensor: torch.Tensor):
    """Return INT8 weights and FP32 per-row scales for a 2D tensor."""
    if tensor.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {tensor.dim()}D")
    if tensor.dtype != torch.float32:
        tensor = tensor.float()
    qmax = 127.0
    eps = torch.finfo(torch.float32).eps
    abs_max = torch.abs(tensor).max(dim=1, keepdim=True)[0]  # [rows, 1]
    scale = abs_max / qmax  # [rows, 1]
    scale = torch.clamp(scale, min=eps)
    assert scale.shape == (tensor.shape[0], 1)
    quantized = torch.round(tensor / scale)
    quantized = torch.clamp(quantized, -qmax, qmax)
    return quantized.to(torch.int8), scale.to(torch.float32)

def fp8_channel_weight_quant(tensor: torch.Tensor):
    """Return E4M3 FP8 weights and FP32 per-row scales for a 2D tensor."""
    if tensor.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {tensor.dim()}D")
    if tensor.dtype != torch.float32:
        tensor = tensor.float()
    qtype = torch.float8_e4m3fn
    qmin, qmax = torch.finfo(qtype).min, torch.finfo(qtype).max
    abs_max = torch.abs(tensor).max(dim=1, keepdim=True)[0]  # [rows, 1]
    scale = abs_max / qmax  # [rows, 1]
    eps = torch.finfo(torch.float32).eps
    scale = torch.clamp(scale, min=eps)
    assert scale.shape == (tensor.shape[0], 1)
    quantized = tensor / scale
    quantized = torch.clamp(quantized, qmin, qmax)
    return quantized.to(torch.float8_e4m3fn), scale.to(torch.float32)

def fp8_block_weight_quant(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return E4M3 weights and FP32 scales for complete 128x128 blocks.

    Both dimensions must be multiples of 128; partial blocks are not handled.
    """
    if weight.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {weight.dim()}D")
    rows, cols = weight.shape
    num_row_blocks = rows  // 128
    num_col_blocks = cols  // 128

    scales = torch.zeros((num_row_blocks, num_col_blocks), dtype=torch.float32, device=weight.device)
    for i in range(num_row_blocks):
        row_start = i * 128
        row_end = row_start + 128

        for j in range(num_col_blocks):
            col_start = j * 128
            col_end = col_start + 128

            block = weight[row_start:row_end, col_start:col_end]
            absmax = block.abs().max()

            # Avoid division by zero
            if absmax > 1e-12:
                scales[i, j] = absmax / 448.0
            else:
                scales[i, j] = 1.0

    quantized = torch.empty_like(weight, dtype=torch.float8_e4m3fn)

    for i in range(num_row_blocks):
        row_start = i * 128
        row_end = min(row_start + 128, rows)

        for j in range(num_col_blocks):
            col_start = j * 128
            col_end = col_start + 128

            block = weight[row_start:row_end, col_start:col_end]
            quant_block = torch.clamp(block / scales[i,j], -448.0, 448.0)

            quantized[row_start:row_end, col_start:col_end] = quant_block.to(torch.float8_e4m3fn)

    return quantized, scales

@triton.jit
def int8_block_weight_quant_kernel(x_ptr, y_ptr, s_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """Write INT8 weights and one absmax scale per square block."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.max(tl.abs(x)) / 127.
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y, mask=mask)
    tl.store(s_ptr + pid_m * n + pid_n, s)

def int8_block_weight_quant(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-wise INT8 quantization (128×128 blocks), Triton kernel."""
    if weight.dtype != torch.float32:
        weight = weight.float()
    block_size = 128
    assert weight.is_contiguous()
    assert weight.dim() == 2
    M, N = weight.size()
    y = torch.empty_like(weight, dtype=torch.int8)
    sM, sN = torch.tensor(1.0*M/block_size).ceil().int(), torch.tensor(1.0*N/block_size).ceil().int()
    s = weight.new_empty(sM, sN, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']), triton.cdiv(N, meta['BLOCK_SIZE']))
    int8_block_weight_quant_kernel[grid](weight, y, s, M, N, BLOCK_SIZE=block_size)
    return y, s

def weight_quant_2D(tensor: torch.Tensor, scheme_name: str):
    """Quantize a 2D tensor by scheme name and return weights and scales."""
    if scheme_name == "W8A8":
        quant_weight, scale_inv = int8_channel_weight_quant(tensor)
    elif scheme_name == "W8A8_BLOCK":
        quant_weight, scale_inv = int8_block_weight_quant(tensor)
    elif scheme_name == "FP8_DYNAMIC":
        quant_weight, scale_inv = fp8_channel_weight_quant(tensor)
    elif scheme_name == "FP8_BLOCK":
        quant_weight, scale_inv = fp8_block_weight_quant(tensor)
    elif scheme_name == "MXFP4":
        quant_weight, scale_inv = downcast_to_mxfp(tensor, torch.uint8, axis=1)
    else:
        raise ValueError(f"Unsupported quantization scheme: {scheme_name}")

    return quant_weight, scale_inv

def weight_quant(tensor: torch.Tensor, scheme_name: str):
    """Return quantized weights and scales for a 2D tensor or a nonempty 3D batch."""

    if tensor.dim() == 2:
        return weight_quant_2D(tensor, scheme_name)
    elif tensor.dim() == 3:
        slice_w = []
        slice_s = []
        for idx in range(tensor.shape[0]):
            weight = tensor[idx]
            w, s = weight_quant_2D(weight, scheme_name)
            slice_w.append(w)
            slice_s.append(s)
        stack_w = torch.stack(slice_w, dim=0)
        stack_s = torch.stack(slice_s, dim=0)
        return stack_w,stack_s
    else:
        raise ValueError(f"Unsupported tensor dimension: {tensor.dim()}")

def validate_scheme(scheme: QuantizationScheme) -> tuple[str, QuantizationScheme]:
    """Validate a preset or scheme for data-free PTQ and return its name and normalized scheme.

    A supplied scheme object is modified in place.
    """
    # treat strings as preset schemes
    if isinstance(scheme, str):
        scheme_name = scheme
        if scheme == "W8A8_BLOCK":
            scheme = preset_name_to_scheme("W8A8", [])
        else:
            scheme = preset_name_to_scheme(scheme, [])
    else:
        scheme_name = "config_group_0"

    # weight quantization must be provided
    if scheme.weights is None:
        raise ValueError(
            "Must provide a weights quantization scheme to perform weights-only PTQ"
        )

    # activation quantization must be dynamic
    input_dynamic = getattr_chain(scheme, "input_activations.dynamic", True)
    output_dynamic = getattr_chain(scheme, "output_activations.dynamic", True)
    if input_dynamic is not True or output_dynamic is not True:
        raise ValueError("Model Free PTQ cannot calibrate activations. ")

    # override with static observers
    # Remove after https://github.com/vllm-project/compressed-tensors/pull/489
    if scheme.weights.observer in ("minmax", "mse"):
        new_observer = f"static_{scheme.weights.observer}"
        logger.info(
            f"Scheme uses {scheme.weights.observer} weight observer. "
            f"Using {new_observer} instead"
        )
        scheme.weights.observer = new_observer

    # target all modules; filter by ignore list
    # technically this should be "re:.*", but vllm's
    # ct moe layer has a hard coded check for "Linear"
    scheme.targets = ["Linear"]

    if scheme_name == "W8A8":
        scheme.format = "int-quantized"
    elif scheme_name == "W8A8_BLOCK":
        scheme.format = "int-quantized"
    elif scheme_name == "FP8_DYNAMIC":
        scheme.format = "float-quantized"
    elif scheme_name == "FP8_BLOCK":
        scheme.format = "float-quantized"
    elif scheme_name == "MXFP4":
        scheme.format = "mxfp4-pack-quantized"

    return scheme_name, scheme

def generate_model_index_file(src_dir: str, dst_dir: str):
    """Write a safetensors index in dst_dir from source shard headers; reject duplicate keys."""
    from safetensors import safe_open
    import json

    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    if not src_dir.is_dir():
        raise ValueError(f"Source directory {src_dir} does not exist.")

    safetensor_files = sorted([f for f in src_dir.glob("*.safetensors")])
    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors files found in {src_dir}")

    weight_map = {}
    for file_path in safetensor_files:
        filename = file_path.name
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in weight_map:
                    raise ValueError(f"Duplicate weight key '{key}' found in {filename} and {weight_map[key]}")
                weight_map[key] = filename

    index = {
        "metadata": {
            "total_size": sum(os.path.getsize(f) for f in safetensor_files)
        },
        "weight_map": weight_map
    }

    output_path = dst_dir / "model.safetensors.index.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    logger.info(f"Generated index file: {output_path}. Total tensors: {len(weight_map)}. Shard files: {len(safetensor_files)}")

def copy_necessary_files(src_dir: str, dst_dir: str):
    """Copy selected non-weight assets and return output config and index paths.

    Require config.json and generate an index when the source has none.
    README.md, LICENSE, safetensors-md5sum.txt, and hf_quant_config.json are excluded.
    """

    logger.debug(f"copy files {src_dir} -> {dst_dir}")
    os.makedirs(dst_dir, exist_ok=True)

    # check for model index file and config file
    config_file = os.path.join(src_dir, "config.json")
    model_index_file = os.path.join(src_dir, "model.safetensors.index.json")
    if not os.path.exists(config_file):
        raise FileNotFoundError(
            f"Config file not found in {src_dir}"
        )

    if not os.path.exists(model_index_file):
        logger.warning(
            f"Model index file not found in {src_dir}. Use safetensor files to generate one in {dst_dir}."
        )
        generate_model_index_file(src_dir, dst_dir)

    def get_files_except_safetensors(dir: str):
        path = Path(dir)
        exclude_names = {"README.md", "LICENSE", "safetensors-md5sum.txt", "hf_quant_config.json"}
        files = []
        for root, dirs, filenames in os.walk(path):
            root_path = Path(root)
            rel_dir = root_path.relative_to(path)
            if rel_dir != Path("."):
                os.makedirs(Path(dst_dir) / rel_dir, exist_ok=True)
            for filename in filenames:
                if filename.endswith(".safetensors") or filename in exclude_names:
                    continue
                files.append(root_path / filename)
        return files

    src_root = Path(src_dir)
    dst_root = Path(dst_dir)
    src_files = get_files_except_safetensors(src_dir)
    for f in src_files:
        rel_path = f.relative_to(src_root)
        dst_file = dst_root / rel_path
        os.makedirs(dst_file.parent, exist_ok=True)
        shutil.copy2(f, dst_file)
    logger.info(f"non-weight files copied to {dst_dir}")

    new_config_file = os.path.join(dst_dir, "config.json")
    new_model_index_file = os.path.join(dst_dir, "model.safetensors.index.json")

    return new_config_file, new_model_index_file
