# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArguments:
    """Model identity, local paths, recipe, and conversion format options."""

    model_name: str
    model_path: str
    save_path: str
    recipe: Optional[str] = None  # Local YAML recipe path; None selects auto mode.
    mix_path: Optional[str] = None

    input_format: Optional[str] = None
    output_format: Optional[str] = "INT8_W8A8"
    max_shard_size: str | int = "10GB"
    gptq_int4_export_device: str = "auto"
    gptq_export_workers: int = 2

    # GPTQ model loading (used only when no model object is supplied).
    # offload_folder must stay writable through calibration and export.
    offload_folder: Optional[str] = None
    device_map: str = "auto_offload"
    trust_remote_code: bool = True
    cpu_offload_gb: float = 300.0

    verify: Optional[str] = None  # CLI --verify; overrides GeneralQuantEngineModifier.verify