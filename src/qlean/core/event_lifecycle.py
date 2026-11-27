# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from functools import wraps
from typing import List, Optional, Callable, Any, TYPE_CHECKING
from qlean.core.event import Event, EventType

if TYPE_CHECKING:
    from qlean.core.event_mixin import EventMixin

import logging

logger = logging.getLogger(__name__)


class EventLifecycle:
    """Decorators that track lifecycle state on the class before calling handlers."""

    initialized: bool = False
    finalized: bool = False

    @classmethod
    def initialize(cls, fn: Callable[[Any], Any]):
        """Mark the lifecycle as initialized before invoking the wrapped handler."""
        def validator(self: "EventMixin", *args, **kwargs):
            cls.initialized = True
            cls.finalized = False

        return cls._wrap_with_validation(fn, validator)

    @classmethod
    def finalize(cls, fn: Callable[[Any], Any]):
        """Validate and mark finalization before invoking the wrapped handler."""
        def validator(self: "EventMixin", **kwargs):
            if not cls.initialized:
                raise ValueError("Cannot finalize before initializing")
            if cls.finalized:
                raise ValueError("Cannot finalize twice")
            cls.finalized = True
            cls.initialized = False

        return cls._wrap_with_validation(fn, validator)

    @classmethod
    def global_step(cls, fn: Callable[[Any], Any]):
        """Unimplemented decorator placeholder; currently returns None."""
        pass

    @classmethod
    def event(cls, fn: Callable[[Any], Any]):
        """Wrap an event handler without additional validation."""
        def validator(self: "EventMixin", event: Event, **kwargs):
            pass

        return cls._wrap_with_validation(fn, validator)

    @classmethod
    def _wrap_with_validation(
        cls,
        fn: Callable[[Any], Any],
        validator: Callable[[Any], Any],
    ) -> Callable[[Any], Any]:

        @wraps(fn)
        def wrapped(*args, **kwargs):
            validator(*args, **kwargs)
            return fn(*args, **kwargs)

        return wrapped
