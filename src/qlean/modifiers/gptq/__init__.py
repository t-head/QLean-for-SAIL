# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Optional GPTQ modifier package.

Importing this package must stay safe on a core-only install, so the backend is
imported (and ``GPTQModifier`` registered) only when the optional dependencies are
present. Otherwise attribute access raises ImportError with installation guidance.
"""

from qlean.utils.optional import GPTQ_INSTALL_HINT, is_gptq_available

__all__ = [
    # Main modifier
    "GPTQModifier",
    # Core algorithm
    "make_empty_hessian",
    "accumulate_hessian",
    "quantize_weight",
    # MoE support
    "is_moe_model",
    "get_moe_layers",
    "linearize_moe_layers",
    # Distributed support
    "is_distributed",
    "greedy_bin_packing",
    "wait_for_comms",
]

if is_gptq_available():
    from qlean.modifiers.gptq.base import GPTQModifier
    from qlean.modifiers.gptq.gptq_quantize import (
        accumulate_hessian,
        make_empty_hessian,
        quantize_weight,
    )
    from qlean.modifiers.gptq.moe_support import (
        get_moe_layers,
        is_moe_model,
        linearize_moe_layers,
    )
    from qlean.modifiers.gptq.distributed import (
        greedy_bin_packing,
        is_distributed,
        wait_for_comms,
    )
else:

    def __getattr__(name: str):
        if name in __all__:
            raise ImportError(f"GPTQ support is unavailable. {GPTQ_INSTALL_HINT}")
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
