# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Union

from qlean.utils.custom_typing import PipelineFn


def is_gptq_modifier(modifier: object) -> bool:
    """Recognize registered GPTQ instances without importing the GPTQ backend."""
    from qlean.modifiers.factory import ModifierFactory

    gptq_class = ModifierFactory._main_registry.get("GPTQModifier")
    return gptq_class is not None and isinstance(modifier, gptq_class)


def get_pipeline_from_modifiers(modifiers: List[Union[str, object]]) -> PipelineFn:
    """Select sequential for GPTQ and retain datafree for all other modifiers."""
    if any(
        mod == "GPTQModifier" if isinstance(mod, str) else is_gptq_modifier(mod)
        for mod in modifiers
    ):
        from qlean.pipelines.sequential import run_pipeline
    else:
        from qlean.pipelines.datafree import run_pipeline

    return run_pipeline
