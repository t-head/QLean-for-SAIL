# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Format route tables: output_format → recipe dir / scheme, plus modifier replacement chains. Plain literals, no third-party deps."""

# output_format → scheme value (None = keep the recipe default)
SCHEME_MAP = {
    "INT8_W8A8":   None,
    "FP8_BLOCK":   "FP8_BLOCK",
    "FP8_DYNAMIC": "FP8_DYNAMIC",
    "MXFP4":       "MXFP4",
    "BF16":        None,
}

RECIPE_DIR_MAP = {
    "INT8_W8A8":   "day0",
    "FP8_BLOCK":   "day0",
    "FP8_DYNAMIC": "day0",
    "MXFP4":       "mxfp4",
    "BF16":        "day0",
    "GPTQ_INT4":   "gptq",
    "GPTQ_INT8":   "gptq",
    "GPTQ_MXFP4":  "gptq",
    "GPTQ_INT4_GROUP": "gptq",
    "GPTQ_INT8_GROUP": "gptq",
}

# (input_format, output_format) → replacement rules for generalDay0Modifier
MODIFIER_REPLACEMENT = {
    ("BF16",  "INT8_W8A8"):   ["generalDay0Modifier"],
    ("BF16",  "FP8_BLOCK"):   ["generalDay0Modifier"],
    ("BF16",  "FP8_DYNAMIC"): ["generalDay0Modifier"],
    ("BF16",  "MXFP4"):       ["generalDay0Modifier"],
    ("BF16", "GPTQ_INT4"):    ["GPTQModifier"],
    ("BF16", "GPTQ_INT8"):    ["GPTQModifier"],
    ("BF16", "GPTQ_MXFP4"):   ["GPTQModifier"],
    ("BF16", "GPTQ_INT4_GROUP"): ["GPTQModifier"],
    ("BF16", "GPTQ_INT8_GROUP"): ["GPTQModifier"],
    ("FP8",   "BF16"):        ["dequantModifier"],
    ("FP8",   "INT8_W8A8"):   ["dequantModifier", "fp8BasedDay0Modifier"],
    ("FP8",   "FP8_BLOCK"):   ["dequantModifier", "fp8BasedDay0Modifier"],
    ("FP8",   "FP8_DYNAMIC"): ["dequantModifier", "fp8BasedDay0Modifier"],
    ("NVFP4", "BF16"):        ["nvfp4DequantModifier"],
    ("NVFP4", "MXFP4"):       ["nvfp4DequantModifier", "nvfp4BasedMxfp4Day0Modifier"],
}

GPTQ_OUTPUT_PRESETS = {
    "GPTQ_INT4": "INT4_CHANNEL",
    "GPTQ_INT8": "INT8_CHANNEL",
    "GPTQ_MXFP4": "MXFP4",
    "GPTQ_INT4_GROUP": "INT4_GROUP",
    "GPTQ_INT8_GROUP": "INT8_GROUP",
}

# INT4 channel is routed to experts only; all other presets target all Linear.
GPTQ_INT4_PRESETS = ("INT4_CHANNEL",)

GPTQ_EXPERT_TARGET = r"re:.*(?:mlp|ffn|feed_forward|block_sparse_moe)\.experts\..*"

GPTQ_WEIGHT_CONFIGS = {
    "INT4_CHANNEL": {
        "num_bits": 4,
        "type": "int",
        "symmetric": True,
        "strategy": "channel",
    },
    "INT8_CHANNEL": {
        "num_bits": 8,
        "type": "int",
        "symmetric": True,
        "strategy": "channel",
    },
    "INT4_GROUP": {
        "num_bits": 4,
        "type": "int",
        "symmetric": True,
        "strategy": "group",
        "group_size": 128,
    },
    "INT8_GROUP": {
        "num_bits": 8,
        "type": "int",
        "symmetric": True,
        "strategy": "group",
        "group_size": 128,
    },
    "MXFP4": {
        "num_bits": 4,
        "type": "float",
        "symmetric": True,
        "strategy": "group",
        "group_size": 32,
        "scale_dtype": "torch.uint8",
        "zp_dtype": "torch.uint8",
    },
}


def validate_gptq_output_format(output_format: str | None) -> None:
    """Reject non-canonical GPTQ output formats without quantization imports."""
    if (
        isinstance(output_format, str)
        and output_format.upper().startswith("GPTQ_")
        and output_format not in GPTQ_OUTPUT_PRESETS
    ):
        supported = ", ".join(GPTQ_OUTPUT_PRESETS)
        raise ValueError(
            f"Unsupported GPTQ output format {output_format!r}. "
            f"Supported GPTQ output formats: {supported}."
        )
