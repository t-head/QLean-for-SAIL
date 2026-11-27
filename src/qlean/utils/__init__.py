# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from qlean.utils.argparser import parse_to_dataclass
from qlean.utils.custom_typing import *
from qlean.utils.weight_file import *

# qlean.utils.dev pulls the GPTQ loading stack, so its names resolve on first use
# (PEP 562) instead of at import time. Existing `from qlean.utils import ...`
# call sites keep working unchanged.
_LAZY_DEV_NAMES = frozenset(
    {"get_rank_partition", "load_context", "temporary_offload_folder"}
)


def __getattr__(name: str):
    if name in _LAZY_DEV_NAMES:
        from qlean.utils import dev

        return getattr(dev, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _LAZY_DEV_NAMES)
