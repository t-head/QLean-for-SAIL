# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Single resolution point for compressed-tensors API renames.

Core modifiers import ``match_name`` from here so the data-free path keeps working
on compressed-tensors 0.13 (``_match_name``) and on 0.18+ (``match_name``), which
the optional GPTQ extra requires. Both spellings have identical semantics.
"""

try:
    from compressed_tensors.utils.match import match_name
except ImportError:  # compressed-tensors < 0.18
    from compressed_tensors.utils.match import _match_name as match_name

__all__ = ["match_name"]
