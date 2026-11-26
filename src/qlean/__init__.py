# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

__all__ = ["QLeanCompressor"]


def __getattr__(name: str):
    # PEP 562 lazy export: importing qlean must not pull torch/transformers/triton,
    # so CLI help and the data-free helpers stay usable on a core-only install.
    if name == "QLeanCompressor":
        from qlean.core.qleancompressor import QLeanCompressor

        return QLeanCompressor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
