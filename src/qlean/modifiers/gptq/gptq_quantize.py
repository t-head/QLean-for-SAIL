# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
# Ported from: llmcompressor/modifiers/gptq/gptq_quantize.py
# Hessian accumulation is rewritten as the mathematically equivalent
# addmm_(inp, inp.t(), beta=1, alpha=2) form (upstream uses sqrt(2)*inp + matmul),
# and copy_hessian is a qlean addition.

from copy import copy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from compressed_tensors.quantization import (
    ActivationOrdering,
    QuantizationArgs,
    QuantizationStrategy,
    fake_quantize,
)
from loguru import logger

GPTQ_PRECISION = torch.float32

__all__ = ["make_empty_hessian", "accumulate_hessian", "quantize_weight"]


def make_empty_hessian(
    module: nn.Module, device: Optional[torch.device] = None
) -> torch.Tensor:
    weight = module.weight
    device = device if device is not None else weight.device
    return torch.zeros(
        (weight.shape[1], weight.shape[1]), device=device, dtype=GPTQ_PRECISION
    )


def accumulate_hessian(
    inp: torch.Tensor,
    module: nn.Module,
    hessian: torch.Tensor,
    num_samples: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if inp.dim() < 2:
        raise ValueError(
            "GPTQ Hessian accumulation expects an input with at least 2 dimensions, "
            f"got shape {tuple(inp.shape)}"
        )

    inp = inp.to(device=hessian.device)
    if inp.dim() == 2:
        inp = inp.unsqueeze(0)
    elif inp.dim() > 3:
        inp = inp.reshape(inp.shape[0], -1, inp.shape[-1])

    num_added = inp.shape[0]
    if isinstance(module, nn.Linear):
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        inp = inp.t()

    num_samples += num_added
    inp = inp.to(dtype=GPTQ_PRECISION)
    hessian.addmm_(inp, inp.t(), beta=1, alpha=2)
    return hessian, num_samples


def quantize_weight(
    module: nn.Module,
    quant_args: QuantizationArgs,
    hessian: torch.Tensor,
    blocksize: int = 128,
    percdamp: float = 0.01,
    copy_hessian: bool = True,
) -> Tuple[float, Dict[str, torch.Tensor]]:
    if blocksize <= 0:
        raise ValueError(f"GPTQ blocksize must be positive, got {blocksize}")

    strategy = quant_args.strategy
    actorder = getattr(quant_args, "actorder", None)
    final_shape = module.weight.shape
    final_dtype = module.weight.dtype
    weight = module.weight.detach().clone().to(dtype=GPTQ_PRECISION)
    if copy_hessian:
        hessian = hessian.clone()
    observer = module.weight_observer
    num_rows, num_columns = weight.shape
    column_quant_args = quant_args
    if strategy in (QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP):
        column_quant_args = copy(quant_args)
        column_quant_args.strategy = QuantizationStrategy.CHANNEL

    if actorder == ActivationOrdering.GROUP and strategy not in (
        QuantizationStrategy.GROUP,
        QuantizationStrategy.TENSOR_GROUP,
    ):
        logger.warning(
            "ActivationOrdering.GROUP requires grouped quantization; "
            "falling back to unordered GPTQ for this module."
        )
        actorder = None

    if actorder:
        weight, hessian, permutation = _apply_activation_ordering(weight, hessian)

    g_idx = None
    if strategy in (
        QuantizationStrategy.GROUP,
        QuantizationStrategy.TENSOR_GROUP,
        QuantizationStrategy.BLOCK,
    ):
        divisor = (
            quant_args.group_size
            if strategy != QuantizationStrategy.BLOCK
            else quant_args.block_structure[1]
        )
        g_idx = torch.arange(num_columns, device=weight.device, dtype=torch.int) // divisor
        if actorder == ActivationOrdering.WEIGHT:
            g_idx = g_idx[permutation]

    if actorder == ActivationOrdering.GROUP:
        observer.delete_statistics()
        observer(weight)

    qparams = observer.get_qparams()
    scale = qparams["scale"]
    zero_point = qparams["zero_point"]
    global_scale = qparams["global_scale"]

    losses = torch.zeros(num_rows, device=weight.device)
    dead = torch.diag(hessian) == 0
    hessian[dead, dead] = 1
    weight[:, dead] = 0

    try:
        damp = percdamp * torch.mean(torch.diag(hessian))
        diagonal = torch.arange(hessian.shape[0], device=hessian.device)
        hessian[diagonal, diagonal] += damp
        hessian = torch.linalg.cholesky(hessian)
        hessian = torch.cholesky_inverse(hessian)
        hinv = torch.linalg.cholesky(hessian, upper=True)
    except torch.linalg.LinAlgError:
        logger.warning(
            "Failed to invert the GPTQ Hessian due to numerical instability; "
            "falling back to round-to-nearest for this module. Increase "
            "dampening_frac, collect more calibration tokens, or shuffle the "
            "calibration dataset."
        )
        hinv = torch.eye(num_columns, dtype=hessian.dtype, device=hessian.device)

    for start in range(0, num_columns, blocksize):
        end = min(start + blocksize, num_columns)
        count = end - start
        weight_block = weight[:, start:end]
        quantized_block = torch.zeros_like(weight_block)
        error_block = torch.zeros_like(weight_block)
        loss_block = torch.zeros_like(weight_block)
        hinv_block = hinv[start:end, start:end]

        for index in range(count):
            column = weight_block[:, index]
            diagonal = hinv_block[index, index]
            quantized = _fake_quantize_column(
                column=column,
                quant_args=column_quant_args,
                strategy=strategy,
                scale=scale,
                zero_point=zero_point,
                global_scale=global_scale,
                column_index=start + index,
                g_idx=g_idx,
            )

            quantized_block[:, index] = quantized
            loss_block[:, index] = (column - quantized).square() / diagonal.square()
            column_error = (column - quantized) / diagonal
            weight_block[:, index:] -= column_error.unsqueeze(1).matmul(
                hinv_block[index, index:].unsqueeze(0)
            )
            error_block[:, index] = column_error

        weight[:, start:end] = quantized_block
        losses += loss_block.sum(dim=1) / 2
        weight[:, end:] -= error_block.matmul(hinv[start:end, end:])

    if actorder:
        inverse_permutation = torch.argsort(permutation)
        weight = weight[:, inverse_permutation]

    # Keep qparams in floating point and let the compressor encode them (MXFP4 uses
    # E8M0 via compress_mx_scale). Casting to quant_args.scale_dtype here would
    # truncate sub-1.0 scales to 0 before that encoding. Matches llmcompressor
    # modifiers/gptq/helpers.py:update_batch_qparams.
    q_param_dict = {
        "weight": weight.reshape(final_shape).to(dtype=final_dtype),
        "weight_scale": scale.to(dtype=final_dtype),
    }
    if zero_point is not None and not quant_args.symmetric:
        q_param_dict["weight_zero_point"] = zero_point.to(dtype=quant_args.zp_dtype)
    if global_scale is not None:
        q_param_dict["weight_global_scale"] = global_scale.to(dtype=torch.float32)
    if actorder == ActivationOrdering.GROUP:
        q_param_dict["weight_g_idx"] = g_idx[inverse_permutation]

    return losses.sum().item(), q_param_dict


def _fake_quantize_column(
    column: torch.Tensor,
    quant_args: QuantizationArgs,
    strategy: QuantizationStrategy,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    global_scale: Optional[torch.Tensor],
    column_index: int,
    g_idx: Optional[torch.Tensor],
) -> torch.Tensor:
    if strategy == QuantizationStrategy.TENSOR:
        return fake_quantize(
            column, scale, zero_point, quant_args, global_scale=global_scale
        )

    if strategy == QuantizationStrategy.CHANNEL:
        return fake_quantize(
            column,
            scale[:, 0],
            zero_point[:, 0],
            quant_args,
            global_scale=global_scale,
        )

    if strategy in (QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP):
        group_index = g_idx[column_index]
        return fake_quantize(
            column,
            scale[:, group_index],
            zero_point[:, group_index],
            quant_args,
            global_scale=global_scale,
        )

    if strategy == QuantizationStrategy.BLOCK:
        block_column_index = g_idx[column_index]
        return fake_quantize(
            column.unsqueeze(1),
            scale[:, block_column_index : block_column_index + 1],
            zero_point[:, block_column_index : block_column_index + 1],
            quant_args,
            global_scale=global_scale,
        ).squeeze(1)

    raise ValueError(f"Quantization strategy is not supported for GPTQ: {strategy}")


def _apply_activation_ordering(
    weight: torch.Tensor, hessian: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    permutation = torch.argsort(torch.diag(hessian), descending=True)
    return weight[:, permutation], hessian[permutation][:, permutation], permutation
