# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# (strip_suffix, add_suffix), first match wins within each kind.
SCALE_RULES_BY_KIND = {
    "fp8": [
        ("", "_scale_inv"),
        (".weight", ".weight_scale"),
        ("", ".weight_scale_inv"),
        (".weight", ".scale"),
    ],
    "mxfp4": [
        (".weight", ".scale"),
        ("_packed", "_scale"),
        (".weight", ".weight_scale"),
    ],
    "mxfp8": [
        (".weight", ".scale"),
        (".weight", ".weight_scale"),
        ("", "_scale_inv"),
    ],
    "nvfp4": [
        ("_packed", "_scale"),
        ("", "_scale"),
    ],
    "int4": [
        ("_packed", "_scale"),
    ],
}


def _rules_for(kind: str) -> list:
    rules = SCALE_RULES_BY_KIND.get(kind)
    if rules is None:
        raise NotImplementedError(
            f"Unknown scale kind '{kind}', "
            f"registered: {sorted(SCALE_RULES_BY_KIND.keys())}"
        )
    return rules


def find_scale_name(weight_name: str, weight_map: dict,
                    kind: Optional[str] = None) -> Optional[str]:
    if kind is not None:
        return _find_in_rules(weight_name, weight_map, _rules_for(kind))

    hits = []
    for k, rules in SCALE_RULES_BY_KIND.items():
        result = _find_in_rules(weight_name, weight_map, rules)
        if result is not None:
            hits.append((k, result))
    if not hits:
        return None
    if len(hits) > 1:
        logger.warning(
            f"[find_scale_name] {weight_name} matched multiple rules: "
            f"{[(k, r) for k, r in hits]}; using the first ({hits[0][0]})"
        )
    return hits[0][1]


def _find_in_rules(weight_name: str, weight_map: dict, rules: list) -> Optional[str]:
    for strip, add in rules:
        if strip and not weight_name.endswith(strip):
            continue
        base = weight_name[:-len(strip)] if strip else weight_name
        candidate = base + add
        if candidate in weight_map:
            return candidate
    return None


def output_scale_name(weight_name: str, suffix: str) -> str:
    if not weight_name.endswith(".weight"):
        raise ValueError(
            f"output_scale_name expects a *.weight name, got {weight_name!r}"
        )
    return weight_name.removesuffix(".weight") + suffix


def output_weight_name(weight_name: str) -> str:
    """Normalize names only; packed storage is determined by the operator."""
    if weight_name.endswith("weight_packed"):
        return weight_name[:-len("_packed")]
    return weight_name
