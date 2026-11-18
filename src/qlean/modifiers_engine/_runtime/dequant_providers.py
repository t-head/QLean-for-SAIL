# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import logging
import math
from typing import Optional

import torch

from qlean.modifiers_engine._runtime.dequant import (
    MXFP8_GROUP_SIZE, dequantize_fp8_block,
    dequantize_mxfp4, dequantize_mxfp8, dequantize_nvfp4, dequantize_int4,
)
from qlean.modifiers_engine._runtime.scale_utils import (  # noqa: F401
    SCALE_RULES_BY_KIND as _SCALE_RULES_BY_KIND,
    find_scale_name,
)

logger = logging.getLogger(__name__)

_DTYPE_MAP = {"bf16": torch.bfloat16, "fp32": torch.float32}


def _target_dtype(ctx) -> torch.dtype:
    return _DTYPE_MAP.get(ctx.transform_cfg.get("dequant_dtype", "fp32"), torch.float32)


def validate_scale_shape(scale: torch.Tensor, weight_shape: tuple,
                         kind: str, block_size: int = 128, **kwargs) -> None:
    if kind == "fp8":
        # Block scales have shape [..., ceil(N/bs), ceil(K/bs)].
        if len(weight_shape) < 2:
            raise ValueError(
                f"[scale validation] fp8 block weight must be at least 2D, got {len(weight_shape)}D "
                f"(shape={list(weight_shape)})"
            )
        if scale.dim() != len(weight_shape):
            raise ValueError(
                f"[scale validation] fp8 block scale rank must match weight rank, "
                f"weight={len(weight_shape)}D, scale={scale.dim()}D "
                f"(scale shape={list(scale.shape)})"
            )
        *leading, n, k = weight_shape
        *scale_leading, expect_n, expect_k = scale.shape
        if list(scale_leading) != list(leading):
            raise ValueError(
                f"[scale validation] fp8 block scale leading dimensions do not match weight: "
                f"weight={list(weight_shape)}, scale={list(scale.shape)}"
            )
        expected_n = math.ceil(n / block_size)
        expected_k = math.ceil(k / block_size)
        if expect_n != expected_n or expect_k != expected_k:
            raise ValueError(
                f"[scale validation] fp8 block scale shape mismatch: weight={list(weight_shape)}, "
                f"block_size={block_size}, expected last two scale dimensions=({expected_n}, {expected_k}), "
                f"got scale={list(scale.shape)}"
            )

    elif kind == "mxfp4":
        # Two fp4 values per byte: ceil(K_logical/32) = ceil(K_packed/16).
        k_packed = weight_shape[-1]
        expect_k = math.ceil(k_packed / 16)
        if scale.shape[-1] != expect_k:
            raise ValueError(
                f"[scale validation] mxfp4 scale last dimension mismatch: weight K_packed={k_packed}, "
                f"expected last scale dimension={expect_k} (= ⌈K_packed/16⌉), "
                f"got={scale.shape[-1]} (scale shape={list(scale.shape)})"
            )

    elif kind == "mxfp8":
        # MX scales are per-row [..., M, ceil(K/32)], not 2D block scales.
        if len(weight_shape) < 2:
            raise ValueError(
                f"[scale validation] mxfp8 weight must be at least 2D, got {len(weight_shape)}D "
                f"(shape={list(weight_shape)})"
            )
        if scale.dim() != len(weight_shape):
            raise ValueError(
                f"[scale validation] mxfp8 scale rank must match weight rank, "
                f"weight={len(weight_shape)}D, scale={scale.dim()}D "
                f"(scale shape={list(scale.shape)})"
            )
        *leading, m, k = weight_shape
        expect_shape = list(leading) + [m, math.ceil(k / MXFP8_GROUP_SIZE)]
        if list(scale.shape) != expect_shape:
            raise ValueError(
                f"[scale validation] mxfp8 scale shape mismatch: weight={list(weight_shape)}, "
                f"group={MXFP8_GROUP_SIZE}, expected scale={expect_shape}, "
                f"got={list(scale.shape)}. "
                f"If the source is fp8 2D-block (scale=(⌈M/bs⌉, ⌈K/bs⌉)), "
                f"set this region's src to fp8_block."
            )

    elif kind == "nvfp4":
        group_size = kwargs.get("group_size", 16)
        if len(weight_shape) != 2:
            raise ValueError(
                f"[scale validation] nvfp4 weight must be 2D, got {len(weight_shape)}D "
                f"(shape={list(weight_shape)})"
            )
        if scale.dim() != 2:
            raise ValueError(
                f"[scale validation] nvfp4 local scale must be 2D, got {scale.dim()}D "
                f"(shape={list(scale.shape)})"
            )
        m, k_packed = weight_shape[0], weight_shape[1]
        expect_k = math.ceil((k_packed * 2) / group_size)
        if scale.shape[0] != m or scale.shape[1] != expect_k:
            raise ValueError(
                f"[scale validation] nvfp4 local scale shape mismatch: "
                f"weight={list(weight_shape)}, group_size={group_size}, "
                f"expected scale=({m}, {expect_k}), got={list(scale.shape)}"
            )

    elif kind == "int4":
        group_size = kwargs.get("group_size", 32)
        if not (isinstance(group_size, int) and not isinstance(group_size, bool)
                and group_size > 0):
            raise ValueError(
                f"[scale validation] int4 group_size must be a positive integer, got {group_size!r}"
            )
        if len(weight_shape) != 2:
            raise ValueError(
                f"[scale validation] int4 weight must be 2D, got {len(weight_shape)}D "
                f"(shape={list(weight_shape)})"
            )
        if scale.dim() != 2:
            raise ValueError(
                f"[scale validation] int4 scale must be 2D, got {scale.dim()}D "
                f"(shape={list(scale.shape)})"
            )
        m, k_packed = weight_shape
        k = k_packed * 8
        if k % group_size:
            raise ValueError(
                f"[scale validation] int4 logical K={k} must be divisible by group_size={group_size}; "
                f"partial groups are not supported (weight={list(weight_shape)})"
            )
        expect_k = k // group_size
        if list(scale.shape) != [m, expect_k]:
            raise ValueError(
                f"[scale validation] int4 scale shape mismatch: "
                f"weight={list(weight_shape)}, group_size={group_size}, "
                f"expected scale=({m}, {expect_k}), got={list(scale.shape)}"
            )

    else:
        raise NotImplementedError(
            f"[scale validation] Shape validation is not implemented for kind='{kind}', "
            f"supported: fp8, mxfp4, nvfp4, int4"
        )


PROVIDERS: dict = {}


def uniform_fp8_block(ctx, name: str, raw: torch.Tensor) -> torch.Tensor:
    if raw.dtype == torch.float8_e4m3fn:
        scale_name = find_scale_name(name, ctx.weight_map, kind="fp8")
        if scale_name is None:
            # Passing unscaled fp8 downstream would silently corrupt quantization.
            raise ValueError(
                f"[dequant] {name} is fp8 but no scale was found (tried all fp8 naming rules). "
                f"Check that the source index is complete or whether this region's src should use another format."
            )
        scale = ctx.get_tensor(scale_name)
        block_size = ctx.transform_cfg.get("dequant_block_size", 128)
        validate_scale_shape(scale, tuple(raw.shape), kind="fp8", block_size=block_size)
        return dequantize_fp8_block(raw, scale.float(), block_size=block_size,
                                    target_dtype=_target_dtype(ctx))
    if raw.dtype in (torch.int8, torch.uint8, torch.int32, torch.float8_e5m2):
        raise ValueError(
            f"[dequant] {name} declares src=fp8_block but received quantized dtype "
            f"{raw.dtype}; expected float8_e4m3fn. "
            f"For mixed checkpoints, declare region src for this region in the recipe"
        )
    return raw


def mxfp4(ctx, name: str, raw: torch.Tensor) -> torch.Tensor:
    if raw.dtype in (torch.int8, torch.uint8):
        scale_name = find_scale_name(name, ctx.weight_map, kind="mxfp4")
        if scale_name is None:
            raise ValueError(
                f"[dequant] {name} is mxfp4(int8/uint8) but no scale was found"
                f" (tried all mxfp4 naming rules)"
            )
        scale = ctx.get_tensor(scale_name)
        validate_scale_shape(scale, tuple(raw.shape), kind="mxfp4")
        return dequantize_mxfp4(raw, scale, target_dtype=_target_dtype(ctx))
    if raw.dtype in (torch.int32, torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError(
            f"[dequant] {name} declares src=mxfp4 but received quantized dtype {raw.dtype}. "
            f"For mixed checkpoints, declare region src for this region in the recipe"
        )
    return raw


def mxfp8(ctx, name: str, raw: torch.Tensor) -> torch.Tensor:
    if raw.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        scale_name = find_scale_name(name, ctx.weight_map, kind="mxfp8")
        if scale_name is None:
            raise ValueError(
                f"[dequant] {name} is mxfp8 but no scale was found"
                f" (tried all mxfp8 naming rules). "
                f"Check that the source index is complete or whether this region's src should use another format."
            )
        scale = ctx.get_tensor(scale_name)
        validate_scale_shape(scale, tuple(raw.shape), kind="mxfp8")
        return dequantize_mxfp8(raw, scale, target_dtype=_target_dtype(ctx))
    if raw.dtype in (torch.int8, torch.uint8, torch.int32):
        raise ValueError(
            f"[dequant] {name} declares src=mxfp8 but received quantized dtype "
            f"{raw.dtype} (possibly an mxfp4/nvfp4/int4 source). "
            f"For mixed checkpoints, declare region src for this region in the recipe"
        )
    return raw


def nvfp4(ctx, name: str, raw: torch.Tensor) -> torch.Tensor:
    if raw.dtype != torch.uint8:
        if raw.dtype in (torch.int8, torch.int32, torch.float8_e4m3fn, torch.float8_e5m2):
            raise ValueError(
                f"[dequant] {name} declares src=nvfp4 but received quantized dtype "
                f"{raw.dtype}; expected uint8 packed weights"
            )
        return raw
    scale_name = find_scale_name(name, ctx.weight_map, kind="nvfp4")
    if scale_name is None:
        raise ValueError(f"[dequant] {name} is nvfp4(uint8) but no local scale was found")
    base = scale_name[:-len("_scale")]
    scale2_name = next(
        (c for c in (f"{scale_name}_2", f"{base}_global_scale") if c in ctx.weight_map),
        None,
    )
    if scale2_name is None:
        raise ValueError(
            f"[dequant] {name} found local scale '{scale_name}' but is missing global scale "
            f"(tried '{scale_name}_2', '{base}_global_scale')"
        )
    scale_local = ctx.get_tensor(scale_name)
    scale_global = ctx.get_tensor(scale2_name)
    group_size = ctx.transform_cfg.get("nvfp4_group_size", 16)
    validate_scale_shape(scale_local, tuple(raw.shape), kind="nvfp4", group_size=group_size)
    return dequantize_nvfp4(raw, scale_local, scale_global, group_size=group_size,
                            target_dtype=_target_dtype(ctx))


def int4_groupwise(ctx, name: str, raw: torch.Tensor) -> torch.Tensor:
    if raw.dtype.is_floating_point:
        return raw
    if raw.dtype != torch.int32:
        raise ValueError(f"[dequant] {name} declares src=int4 but received {raw.dtype}; "
                         f"weight_packed must be int32 [M, K/8]")
    scale_name = find_scale_name(name, ctx.weight_map, kind="int4")
    if scale_name is None:
        raise ValueError(
            f"[dequant] {name} is int4 packed (int32) but no scale was found"
            f" (tried all int4 naming rules). Check that the source index is complete, "
            f"or guard this region with when: has_source_scale"
        )
    scale = ctx.get_tensor(scale_name)
    group_size = ctx.transform_cfg.get("int4_group_size", 32)
    validate_scale_shape(scale, tuple(raw.shape), kind="int4",
                         group_size=group_size)
    return dequantize_int4(raw, scale, group_size=group_size,
                           target_dtype=_target_dtype(ctx))


PROVIDERS.update({
    "fp8_block": uniform_fp8_block,
    "mxfp4": mxfp4,
    "mxfp8": mxfp8,
    "nvfp4": nvfp4,
    "int4": int4_groupwise,
})
