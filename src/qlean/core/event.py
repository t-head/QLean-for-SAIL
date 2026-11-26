# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import Enum, unique
from typing import Optional

__all__ = ["EventType", "Event"]


@unique
class EventType(Enum):
    """Event names used by the modifier lifecycle."""

    INITIALIZE = "initialize"
    FINALIZE = "finalize"

    # calibration
    CALIBRATION_EPOCH_START = "calibration_epoch_start"
    SEQUENTIAL_EPOCH_START = "sequential_epoch_start"
    SEQUENTIAL_EPOCH_END = "sequential_epoch_end"
    CALIBRATION_EPOCH_END = "calibration_epoch_end"


@dataclass
class Event:
    """Carry an optional event type to modifier callbacks."""

    type_: Optional[EventType] = None
