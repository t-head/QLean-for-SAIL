# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Availability checks for the optional GPTQ feature.

GPTQ needs compressed-tensors>=0.18 (the ``offload`` and ``distributed``
subpackages) and transformers>=5.15 (``conversion_mapping``). The data-free path
must stay importable without them, so availability is probed by capability rather
than by parsing version strings, and the probe only runs when GPTQ is reached.

Mirrors the ``is_*_available()`` / ``require_*()`` pattern used by transformers.
"""

from __future__ import annotations

import importlib
from functools import lru_cache

__all__ = [
    "GPTQ_INSTALL_HINT",
    "gptq_unavailable_reason",
    "is_gptq_available",
    "require_gptq",
]

GPTQ_INSTALL_HINT = (
    "GPTQ quantization is an optional feature. Install it with "
    "`pip install 'qlean[gptq]'`, which requires compressed-tensors>=0.18 "
    "(and therefore torch>=2.10) plus transformers>=5.15. "
    "The data-free conversions work without it."
)

# Probed in order; the first failure is reported to the user.
_REQUIRED_CAPABILITIES = (
    ("compressed-tensors>=0.18", "compressed_tensors.offload"),
    ("transformers>=5.15", "transformers.conversion_mapping"),
)


@lru_cache(maxsize=1)
def gptq_unavailable_reason() -> str | None:
    """Return None when GPTQ can run, otherwise a human-readable reason."""
    for requirement, module in _REQUIRED_CAPABILITIES:
        try:
            importlib.import_module(module)
        except Exception as error:
            return f"{requirement} is required but unavailable ({module}: {error})"
    return None


def is_gptq_available() -> bool:
    """Return whether the optional GPTQ dependencies are importable."""
    return gptq_unavailable_reason() is None


def require_gptq() -> None:
    """Raise ImportError with installation guidance when GPTQ cannot run."""
    reason = gptq_unavailable_reason()
    if reason is not None:
        raise ImportError(f"{reason}. {GPTQ_INSTALL_HINT}")
