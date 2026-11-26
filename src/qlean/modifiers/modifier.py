# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math
from abc import ABC, abstractmethod

from qlean.core.state import State
from qlean.core.event import Event, EventType

__all__ = ["Modifier"]


class Modifier(ABC):
    """Base hooks for initialization, start, events, end, and finalization."""

    initialized_: bool = False
    finalized_: bool = False
    started_: bool = False
    ended_: bool = False

    @abstractmethod
    def on_initialize(self, state: State):
        """Validate settings and initialize resources in a subclass."""
        raise NotImplementedError()

    def on_start(self, state: State):
        """Mark the modifier as started."""
        self.started_ = True

    def update_event(self, state: State, event: Event, **kwargs):
        self.on_event(state, event, **kwargs)
        callback = {
            EventType.CALIBRATION_EPOCH_START: self.on_calibration_start,
            EventType.SEQUENTIAL_EPOCH_START: self.on_sequential_epoch_start,
            EventType.SEQUENTIAL_EPOCH_END: self.on_sequential_epoch_end,
            EventType.CALIBRATION_EPOCH_END: self.on_calibration_end,
        }.get(event.type_)
        if callback is not None:
            callback(state, event, **kwargs)

    def on_event(self, state: State, event: Event, **kwargs):
        """Handle a pipeline event; the base implementation does nothing."""
        pass

    def on_calibration_start(self, state: State, event: Event, **kwargs):
        pass

    def on_sequential_epoch_start(self, state: State, event: Event, **kwargs):
        pass

    def on_sequential_epoch_end(self, state: State, event: Event, **kwargs):
        pass

    def on_calibration_end(self, state: State, event: Event, **kwargs):
        pass

    def on_end(self, state: State):
        """Mark the modifier as ended."""
        self.ended_ = True

    def on_finalize(self, state: State):
        """Mark the modifier as finalized."""
        self.finalized_ = True
