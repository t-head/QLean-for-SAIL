# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional


@dataclass
class OtherArguments:
    """Additional arguments passed to the selected pipeline."""
    other_args: Optional[str] = None
    propagate_error: bool = True
    num_calibration_rounds: int = 1

    # torchrun launcher options; nproc_per_node > 1 re-execs qlean under torchrun.
    nproc_per_node: int = 1
    nnodes: int = 1
    rdzv_endpoint: Optional[str] = None
    max_restarts: int = 0

    def __post_init__(self):
        if (
            isinstance(self.num_calibration_rounds, bool)
            or not isinstance(self.num_calibration_rounds, int)
            or self.num_calibration_rounds <= 0
        ):
            raise ValueError("num_calibration_rounds must be a positive integer")
        for name in ("nproc_per_node", "nnodes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.max_restarts, bool) or not isinstance(self.max_restarts, int) or self.max_restarts < 0:
            raise ValueError("max_restarts must be a nonnegative integer")