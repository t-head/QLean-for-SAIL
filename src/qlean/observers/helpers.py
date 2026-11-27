# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy

__all__ = ["flatten_for_calibration"]


def flatten_for_calibration(
    value: torch.Tensor, base_name: str, args: QuantizationArgs
) -> torch.Tensor:
    if base_name != "weight":
        raise ValueError(f"Unsupported observer base name: {base_name}")

    if args.strategy == QuantizationStrategy.TENSOR:
        return value.reshape(1, 1, -1)

    if args.strategy == QuantizationStrategy.CHANNEL:
        return value.unsqueeze(0).unsqueeze(-2)

    if args.strategy in (
        QuantizationStrategy.GROUP,
        QuantizationStrategy.TENSOR_GROUP,
    ):
        return value.unflatten(-1, (-1, args.group_size)).unsqueeze(0)

    if args.strategy == QuantizationStrategy.BLOCK:
        block_height, block_width = args.block_structure
        rows, columns = value.shape
        if rows % block_height or columns % block_width:
            raise ValueError(
                "GPTQ block quantization requires weights divisible by "
                f"block_structure={args.block_structure}, got {tuple(value.shape)}"
            )
        return (
            value.reshape(rows // block_height, block_height, columns // block_width, block_width)
            .transpose(1, 2)
            .flatten(-2, -1)
            .unsqueeze(0)
        )

    raise ValueError(f"Unsupported GPTQ weight strategy: {args.strategy}")
