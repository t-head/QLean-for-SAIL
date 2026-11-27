# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
# Global MoE calibration context ported from llmcompressor/modeling/moe/context.py.

from __future__ import annotations

import contextlib

_CALIBRATE_ALL_EXPERTS = False


def get_calibrate_all_experts_flag() -> bool:
    return _CALIBRATE_ALL_EXPERTS


@contextlib.contextmanager
def moe_calibration_context():
    """Context manager enabling calibration over *all* experts.

    When active, forward passes route the full input through every expert instead of
    only the top-k selected ones, ensuring each expert sees calibration data.
    """
    global _CALIBRATE_ALL_EXPERTS

    restore_value, _CALIBRATE_ALL_EXPERTS = _CALIBRATE_ALL_EXPERTS, True
    try:
        yield
    finally:
        _CALIBRATE_ALL_EXPERTS = restore_value
