# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Ignore the GPTQ suite when its backend cannot be imported.

Probes the same module that the tests/core GPTQ files pass to
``pytest.importorskip``, so both locations behave identically. Checking only
``is_gptq_available()`` is not enough: that covers the optional extra, while the
backend also needs core dependencies such as triton (absent on macOS).

Ignoring at collection time rather than skipping at run time matters because
``tests/run_checks.py`` treats any reported skip as an audit error.
"""

import importlib

from qlean.utils.optional import is_gptq_available


def _backend_importable() -> bool:
    if not is_gptq_available():
        return False
    try:
        importlib.import_module("qlean.modifiers.gptq.base")
    except ImportError:
        return False
    return True


collect_ignore_glob = [] if _backend_importable() else ["test_*.py"]