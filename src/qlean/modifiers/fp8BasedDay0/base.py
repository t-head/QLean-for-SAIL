# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Dict, List, Union, Any
from pydantic import Field
import shutil

import torch, os
from safetensors.torch import load_file, save_file
import json
from tqdm import tqdm

from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
)
from qlean.utils.ct_compat import match_name

from qlean.modifiers import Modifier, ModifierFactory
from qlean.modifiers import QuantizationMixin
from qlean.core.event import Event, EventType

from qlean.modifiers.utils import validate_scheme, copy_necessary_files, ensure_scales_in_weight_map, weight_quant, FP8_SCALE_SUFFIXES, find_scale_name, load_file_to_cuda, safe_save_file
from qlean.modifiers.save_utils import update_config, WA_for_sglang_config

import logging

logger = logging.getLogger(__name__)


@ModifierFactory.register("fp8BasedDay0Modifier")
class fp8BasedDay0Modifier(QuantizationMixin, Modifier):

    config_groups: Optional[Dict[str, QuantizationScheme]] = None
    ignore: List[str] = Field(default_factory=list)
    targets: Union[str, List[str]] = Field(default_factory=lambda: ["Linear"])
    scheme: Optional[Union[str, Dict[str, Any]]] = None
    kv_cache_scheme: Optional[QuantizationArgs] = None
    num_calibration_steps: Optional[int] = None

    valid_scheme_name: str = None
    valid_scheme: QuantizationScheme = None

    def on_initialize(self, state, *args, **kwargs):
        logger.info("fp8-based day0: on_initialize")
        self.valid_scheme_name, self.valid_scheme = validate_scheme(self.scheme)
        logger.info(f"valid_scheme_name: {self.valid_scheme_name}; valid_scheme.format: {self.valid_scheme.format}")
        self.initialized_ = True

    def on_start(self, state, *args, **kwargs):
        logger.info("fp8-based day0: on_start")
        super().on_start(state)

        torch.set_default_dtype(torch.bfloat16)
        config_file, model_index_file = copy_necessary_files(
            state.model.path, state.save_path
        )

        with open(model_index_file, "r") as f:
            model_index = json.load(f)
        weight_map = model_index["weight_map"]

        ensure_scales_in_weight_map(weight_map, state.model.path)

        scale_count = len(
            [key for key in weight_map.keys() if any(key.endswith(s) for s in FP8_SCALE_SUFFIXES)]
        )
        logger.debug(f"scale_count: {scale_count}")
        if scale_count == 0:
            logger.error("model_index_file not FP8 format")

        # delete quantization_config
        config = json.load(open(config_file))
        config.pop("quantization_config", None)
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False, sort_keys=True)
        logger.info(f"config.json modified and saved to {config_file}")

        quant_count = 0
        skipped_scales = 0
        if self.ignore:
            logger.info(f"User-specified ignore patterns: {self.ignore}")
        new_weight_map = {}
        module_ignored = []
        for file_name, resolved_path in tqdm(
            state.model.model_, desc="Quantizing weights"
        ):
            logger.debug(f"quantizing weight: {resolved_path}")
            state_dict = load_file_to_cuda(resolved_path)
            new_state_dict = {}
            for weight_name, weight in state_dict.items():
                scale_inv_name = find_scale_name(weight_name, weight_map)
                has_scale = scale_inv_name in weight_map

                # Check user-specified ignore
                is_ignored = False
                if "." in weight_name:
                    module_name = weight_name.rsplit(".", 1)[0]
                    is_ignored = any(match_name(module_name, ign) for ign in self.ignore)
                    if is_ignored and has_scale:
                        logger.debug(f"\nweight_name:{weight_name} is skipping scale")
                        skipped_scales += 1

                if has_scale and not is_ignored:
                    assert weight.element_size() == 2
                    quant_count += 1
                    quant_weight, scale_inv = weight_quant(weight, self.valid_scheme_name)
                    new_state_dict[weight_name] = quant_weight
                    if scale_inv_name.endswith("_scale_inv"):
                        new_scale_name = scale_inv_name.replace("_scale_inv", "_scale")
                    else:
                        new_scale_name = scale_inv_name
                    new_state_dict[new_scale_name] = scale_inv

                    new_weight_map[weight_name] = file_name
                    new_weight_map[new_scale_name] = file_name
                else:
                    if "." in weight_name:
                        module_name, param_name = weight_name.rsplit(".", 1)
                        is_linear_weight = param_name == "weight" and not module_name.endswith("norm")
                        if is_linear_weight and module_name not in module_ignored:
                            module_ignored.append(module_name)
                    new_state_dict[weight_name] = weight
                    new_weight_map[weight_name] = file_name
            new_safetensor_file = os.path.join(state.save_path, file_name)
            safe_save_file(new_state_dict, new_safetensor_file)
            del state_dict
            del new_state_dict
            torch.cuda.empty_cache()
        logger.debug(f"{quant_count} weights are quantized.")
        if skipped_scales:
            logger.info(f"{skipped_scales} scales skipped due to user ignore patterns.")
        logger.info(f"{len(module_ignored)} modules ignored (no scale key or user ignore): {module_ignored[:10]}{'...' if len(module_ignored) > 10 else ''}")
        assert quant_count + skipped_scales == scale_count

        with open(model_index_file, "r") as f:
            model_index = json.load(f)
        model_index["weight_map"] = new_weight_map
        with open(model_index_file, "w", encoding="utf-8") as f:
            json.dump(model_index, f, indent=2, ensure_ascii=False, sort_keys=True)
        logger.info(
            f"model.safetensors.index.json modified and saved to {model_index_file}"
        )

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
        logger.info("fp8-based day0: on_end")
        super().on_end(state)

    def on_finalize(self, state):
        logger.info("fp8-based day0: on_finalize")
        super().on_finalize(state)

    def on_event(self, state, event: Event, **kwargs):
        logger.info("fp8-based day0: on_event")
        if event.type_ == EventType.CALIBRATION_EPOCH_START:
            if not self.started_:
                self.on_start(state)

        if event.type_ == EventType.CALIBRATION_EPOCH_END:
            if not self.ended_:
                self.on_end(state)
