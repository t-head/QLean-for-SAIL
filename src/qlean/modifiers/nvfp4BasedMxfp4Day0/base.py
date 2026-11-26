# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Dict, List, Union, Any
from pydantic import Field

import logging
import os
import json
from tqdm import tqdm

from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
)

import torch
from safetensors.torch import load_file, save_file

from qlean.utils.ct_compat import match_name

from qlean.modifiers import Modifier, ModifierFactory
from qlean.modifiers import QuantizationMixin
from qlean.core.event import Event, EventType
from qlean.modifiers.utils import validate_scheme, copy_necessary_files, ensure_scales_in_weight_map, load_file_to_cuda, safe_save_file
from qlean.modifiers.save_utils import update_config, WA_for_sglang_config
from qlean.utils.downcast_to_mxfp import downcast_to_mxfp

logger = logging.getLogger(__name__)


@ModifierFactory.register("nvfp4BasedMxfp4Day0Modifier")
class nvfp4BasedMxfp4Day0Modifier(QuantizationMixin, Modifier):

    config_groups: Optional[Dict[str, QuantizationScheme]] = None
    ignore: List[str] = Field(default_factory=list)
    targets: Union[str, List[str]] = Field(default_factory=lambda: ["Linear"])
    scheme: Optional[Union[str, Dict[str, Any]]] = None
    kv_cache_scheme: Optional[QuantizationArgs] = None
    num_calibration_steps: Optional[int] = None

    valid_scheme_name: str = None
    valid_scheme: QuantizationScheme = None

    def on_initialize(self, state, *args, **kwargs):
        logger.info("nvfp4-Based Mxfp4Day0: on_initialize")
        self.valid_scheme_name, self.valid_scheme = validate_scheme(self.scheme)
        logger.info(f"valid_scheme_name: {self.valid_scheme_name}; valid_scheme.format: {self.valid_scheme.format}")
        self.initialized_ = True

    def on_start(self, state, *args, **kwargs):
        logger.info("nvfp4-Based Mxfp4Day0: on_start")
        super().on_start(state)

        config_file, model_index_file = copy_necessary_files(
            state.model.path, state.save_path
        )

        # delete quantization_config
        config = json.load(open(config_file))
        config.pop("quantization_config", None)
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False, sort_keys=True)
        logger.info(f"config.json modified and saved to {config_file}")
        with open(model_index_file, "r") as f:
            model_index = json.load(f)
        weight_map = model_index["weight_map"]

        ensure_scales_in_weight_map(weight_map, state.model.path)

        # Cache for loaded safetensor files
        new_weight_map = {}
        module_ignored = []

        for file_name, resolved_path in tqdm(
            state.model.model_, desc="Quantizing weights"
        ):
            logger.debug(f"quantizing weight: {resolved_path}")
            state_dict = load_file_to_cuda(resolved_path)

            new_state_dict = {}
            for weight_name, weight in state_dict.items():
                # skip input_scale/weight_scale/weight_scale_2
                if "scale" in weight_name:
                    continue

                module_name, param_name = weight_name.rsplit(".", 1)
                is_linear_weight = param_name == "weight" and not module_name.endswith(
                    "norm"
                )
                is_ignored = any(match_name(module_name, ign) for ign in self.ignore)
                if is_ignored:
                    logger.debug(f"\nweight_name:{weight_name} is_ignored")
                    module_ignored.append(module_name)

                scale_name = f"{weight_name}_scale"
                if scale_name in weight_map:
                    assert weight.element_size() == 2
                    if not is_linear_weight or is_ignored:
                        new_state_dict[weight_name] = weight
                        new_weight_map[weight_name] = file_name
                    else:
                        quant, scale = downcast_to_mxfp(weight, torch.uint8, axis=1)
                        new_state_dict[weight_name] = quant
                        new_state_dict[scale_name] = scale
                        new_weight_map[weight_name] = file_name
                        new_weight_map[scale_name] = file_name
                        del quant, scale
                else:
                    new_state_dict[weight_name] = weight
                    new_weight_map[weight_name] = file_name

            new_safetensor_file = os.path.join(state.save_path, file_name)
            safe_save_file(new_state_dict, new_safetensor_file)
            del state_dict
            del new_state_dict
            torch.cuda.empty_cache()

        # Update model index
        new_model_index_file = os.path.join(state.save_path, "model.safetensors.index.json")
        with open(new_model_index_file, "w") as f:
            json.dump({"metadata": {}, "weight_map": new_weight_map}, f, indent=2, ensure_ascii=False, sort_keys=True)

        update_config(
            state.model.path,
            state.save_path,
            self.valid_scheme_name,
            self.valid_scheme,
            module_ignored,
        )

        if self.valid_scheme_name == "MXFP4":
            WA_for_sglang_config(config_file)

    def on_end(self, state):
        logger.info("nvfp4-Based Mxfp4Day0: on_end")
        super().on_end(state)

    def on_finalize(self, state):
        logger.info("nvfp4-Based Mxfp4Day0: on_finalize")
        super().on_finalize(state)

    def on_event(self, state, event: Event, **kwargs):
        logger.info("nvfp4-Based Mxfp4Day0: on_event")
        if event.type_ == EventType.CALIBRATION_EPOCH_START:
            if not self.started_:
                self.on_start(state)

        if event.type_ == EventType.CALIBRATION_EPOCH_END:
            if not self.ended_:
                self.on_end(state)
