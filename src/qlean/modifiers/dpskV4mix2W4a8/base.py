# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import List
from pydantic import Field
import torch, os
from safetensors.torch import load_file, save_file
import json
from tqdm import tqdm

from qlean.utils.ct_compat import match_name

from qlean.modifiers import Modifier, ModifierFactory
from qlean.modifiers import QuantizationMixin
from qlean.core.event import Event, EventType

from qlean.modifiers.utils import copy_necessary_files, load_file_to_cuda, safe_save_file
from qlean.utils.pack_acext_int4 import preprocess_weight_for_a8w4_torch

import logging

logger = logging.getLogger(__name__)

_FP8_SCALE_SUFFIXES = ["_scale_inv", ".scale"]

def _find_scale_name(weight_name: str, weight_map: dict) -> str | None:
    """Find the scale tensor name of an FP8 weight by trying candidate suffixes in order; None if absent."""
    for suffix in _FP8_SCALE_SUFFIXES:
        candidate = f"{weight_name}{suffix}"
        if candidate in weight_map:
            return candidate
    if weight_name.endswith(".weight"):
        candidate = weight_name[:-len(".weight")] + ".scale"
        if candidate in weight_map:
            return candidate
    return None
def quantize_bf16_to_int(weight, num_bits=4):
    if weight.dtype != torch.float32:
        weight = weight.float()

    weight_reshaped = weight.reshape(weight.shape[0], -1)

    abs_max = torch.max(torch.abs(weight_reshaped), dim=1).values  # [C_out]

    q_max = 2**(num_bits - 1) - 1  # 7
    q_min = -2**(num_bits - 1)     # -8

    scales = abs_max / q_max
    scales = scales.clamp(min=torch.finfo(torch.float32).eps)

    scales_expanded = scales.view(-1, *([1] * (weight.dim() - 1)))
    quantized = torch.clamp(
        torch.round(weight / scales_expanded),
        q_min, q_max
    )

    return quantized.to(torch.int8), scales.unsqueeze(1)

@ModifierFactory.register("dpskV4mix2W4a8Day0Modifier")
class dpskV4mix2W4a8Day0Modifier(QuantizationMixin, Modifier):

    ignore: List[str] = Field(default_factory=list)

    def on_initialize(self, state, *args, **kwargs):
        logger.info("dpskV4mix2W4a8Day0: on_initialize")
        self.initialized_ = True

    def on_start(self, state, *args, **kwargs):
        logger.info("dpskV4mix2W4a8Day0: on_start")
        super().on_start(state)

        torch.set_default_dtype(torch.bfloat16)
        config_file, model_index_file = copy_necessary_files(
            state.model.path, state.save_path
        )

        with open(model_index_file, "r") as f:
            model_index = json.load(f)
        weight_map = model_index["weight_map"]
        scale_count = len(
            [key for key in weight_map.keys() if any(key.endswith(s) for s in _FP8_SCALE_SUFFIXES)]
        )
        logger.debug(f"scale_count: {scale_count}")
        if scale_count == 0:
            logger.info("model_index_file not FP8 format")

        expert_quantized = 0
        channel_int8_quantized = 0
        new_weight_map = {}
        module_ignored = []
        for file_name, resolved_path in tqdm(
            state.model.model_, desc="Quantizing weights"
        ):
            state_dict = load_file_to_cuda(resolved_path)
            new_state_dict = {}
            for weight_name, weight in state_dict.items():
                if "." not in weight_name:
                    # Top-level key (no dot separator): not a module.param name, keep as-is
                    logger.warning(f"weight_name '{weight_name}' has no '.', kept as-is")
                    new_state_dict[weight_name] = weight
                    new_weight_map[weight_name] = file_name
                    continue
                module_name, param_name = weight_name.rsplit(".", 1)
                is_linear_weight = param_name == "weight" and not module_name.endswith(
                    "norm"
                )
                is_ignored = any(match_name(module_name, ign) for ign in self.ignore)
                if is_ignored:
                    logger.debug(f"\nweight_name:{weight_name} is_ignored")
                    module_ignored.append(module_name)
                if "ffn.experts" in weight_name and weight_name.endswith("weight"):
                    expert_quantized += 1
                    int8_weight, scale_tensor = quantize_bf16_to_int(weight, num_bits=4)

                    # Pack into INT4 format
                    packed_weight = preprocess_weight_for_a8w4_torch(
                        int8_weight
                    ).view(int8_weight.shape[0], int8_weight.shape[1] // 2)

                    scale_inv_name = weight_name.removesuffix(".weight") + ".scale"
                    if "ffn.experts.0.w1.weight" in weight_name:
                        logger.debug(f"Int4: {weight_name}, {scale_inv_name}")
                    new_state_dict[weight_name] = packed_weight
                    new_state_dict[scale_inv_name] = scale_tensor
                    new_weight_map[weight_name] = file_name
                    new_weight_map[scale_inv_name] = file_name

                elif not is_linear_weight or is_ignored:
                    new_state_dict[weight_name] = weight
                    new_weight_map[weight_name] = file_name

                elif "ffn.experts" not in weight_name and weight_name.endswith("weight"):
                    scale_inv_name =  _find_scale_name(weight_name, weight_map)
                    if scale_inv_name in weight_map:
                        assert weight.element_size() == 2
                        logger.debug(f"Int8: {weight_name}")
                        channel_int8_quantized += 1
                        # default perchannel_w8a8
                        int8_weight, scale_inv = quantize_bf16_to_int(weight, num_bits=8)
                        new_state_dict[weight_name] = int8_weight
                        if scale_inv_name.endswith("_scale_inv"):
                            new_scale_name = scale_inv_name.replace("_scale_inv", "_scale")
                        else:
                            new_scale_name = scale_inv_name
                        new_state_dict[new_scale_name] = scale_inv
                        new_weight_map[weight_name] = file_name
                        new_weight_map[new_scale_name] = file_name
                    else:
                        new_state_dict[weight_name] = weight
                        new_weight_map[weight_name] = file_name
                else:
                    logger.debug(f"bf16: {weight_name}")
                    new_state_dict[weight_name] = weight
                    new_weight_map[weight_name] = file_name

            new_safetensor_file = os.path.join(state.save_path, file_name)
            safe_save_file(new_state_dict, new_safetensor_file)
            del state_dict
            del new_state_dict
            torch.cuda.empty_cache()

        logger.info(f"Expert weights quantized: {expert_quantized}")
        logger.info(f"Channel INT8 weights quantized: {channel_int8_quantized}")

        # update model.safetensors.index.json
        with open(model_index_file, "r") as f:
            model_index = json.load(f)
        model_index["weight_map"] = new_weight_map
        with open(model_index_file, "w", encoding="utf-8") as f:
            json.dump(model_index, f, indent=2, ensure_ascii=False, sort_keys=True)
        logger.info(
            f"model.safetensors.index.json modified and saved to {model_index_file}"
        )

        # update config.json
        for i in range(len(module_ignored) - 1, -1, -1):
            if match_name("model.embed_tokens", module_ignored[i]):
                logger.debug(f"ignore list pop {module_ignored[i]}")
                module_ignored.pop(i)

        if os.path.exists(config_file):
            config = json.load(open(config_file))
            if "quantization_config" in config:
                quant_config = config["quantization_config"]
                quant_config.pop("fmt", None)
                quant_config.pop("scale_fmt", None)
                quant_config.pop("weight_block_size", None)
                quant_config["quant_method"] = "mixed_precision_w4"
                quant_config["activation_scheme"] = "dynamic"
                quant_config["ignore"] = module_ignored
            else:
                config["quantization_config"] = {
                    "activation_scheme": "dynamic",
                    "quant_method": "mixed_precision_w4",
                    "ignore": module_ignored
                }
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False, sort_keys=True)
            print(f"config.json modified and saved to {config_file}")

    def on_end(self, state):
        logger.info("dpskV4mix2W4a8Day0: on_end")
        super().on_end(state)

    def on_finalize(self, state):
        logger.info("dpskV4mix2W4a8Day0: on_finalize")
        super().on_finalize(state)

    def on_event(self, state, event: Event, **kwargs):
        logger.info("dpskV4mix2W4a8Day0: on_event")
        if event.type_ == EventType.CALIBRATION_EPOCH_START:
            if not self.started_:
                self.on_start(state)

        if event.type_ == EventType.CALIBRATION_EPOCH_END:
            if not self.ended_:
                self.on_end(state)
