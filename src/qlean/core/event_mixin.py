# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import List
from abc import ABC

from qlean.core.event import EventType, Event
from qlean.core.event_lifecycle import EventLifecycle
from qlean.core.state import State
# Module-level so tests can monkeypatch event_mixin.is_gptq_modifier; registry is
# light (typing + custom_typing) and imports the GPTQ backend only lazily.
from qlean.pipelines.registry import is_gptq_modifier

import logging

logger = logging.getLogger(__name__)


class EventMixin(ABC):
    """Dispatch lifecycle calls and calibration events to registered modifiers."""

    from qlean.modifiers import Modifier

    state: State
    modifiers: List[Modifier]

    @EventLifecycle.initialize
    def initialize(self, *args, **kwargs):
        """Initialize every modifier with the shared conversion state."""
        for modifier in self.modifiers:
            modifier.on_initialize(self.state, *args, **kwargs)
            logger.debug(f"in initialize {self.state.model.path}")

    @EventLifecycle.finalize
    def finalize(self, **kwargs):
        """End unfinished modifiers and finalize their resources."""
        for modifier in self.modifiers:
            if not modifier.ended_:
                modifier.on_end(self.state)
            modifier.on_finalize(self.state, **kwargs)

    def calibration_epoch_start(self, **kwargs):
        """Send a calibration-start event to active modifiers."""
        logger.info("Calibration epoch start")
        event = Event(type_=EventType.CALIBRATION_EPOCH_START)
        self._handle_event(event, **kwargs)

    def sequential_epoch_start(self, **kwargs):
        event = Event(type_=EventType.SEQUENTIAL_EPOCH_START)
        self._handle_event(event, **kwargs)

    def sequential_epoch_end(self, **kwargs):
        event = Event(type_=EventType.SEQUENTIAL_EPOCH_END)
        self._handle_event(event, **kwargs)

    def calibration_epoch_end(self, **kwargs):
        """Send a calibration-end event to active modifiers."""
        logger.info("Calibration epoch end")
        event = Event(type_=EventType.CALIBRATION_EPOCH_END)
        self._handle_event(event, **kwargs)

    @EventLifecycle.event
    def _handle_event(self, event: Event, **kwargs):
        logger.info(f"Handling event {event.type_}")
        logger.debug(f"in event {self.state.model.path}")
        uses_gptq = any(is_gptq_modifier(modifier) for modifier in self.modifiers)
        for modifier in self.modifiers:
            if modifier.initialized_ and not modifier.ended_:
                if uses_gptq:
                    modifier.update_event(self.state, event, **kwargs)
                else:
                    modifier.on_event(self.state, event)
