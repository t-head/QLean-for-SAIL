# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class DatasetArguments:
    """Calibration data options.

    ``dataset`` carries an already-tokenized dataset object supplied by a caller;
    when it is None the loader builds one from ``dataset_name``/``dataset_path``.
    """

    dataset_name: Optional[str] = None
    dataset_path: Optional[str] = None
    dataset: Optional[Any] = None
    batch_size: int = 1

    dataset_config: Optional[str] = None
    dataset_split: str = "train_sft"
    num_calibration_samples: int = 256
    max_sequence_length: int = 2048
    # "none" keeps ragged lengths (batch size 1); "max_length" pads every sample to
    # max_sequence_length, which the DeepSeek-V3.2 reference script requires.
    calibration_padding: str = "none"
    seed: int = 42
    messages_column: str = "messages"
    text_column: str = "text"

    def __post_init__(self):
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")