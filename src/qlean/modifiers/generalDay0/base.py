# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Dict, List, Union, Any
from pydantic import Field

import re
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

from qlean.modifiers.utils import validate_scheme, copy_necessary_files, weight_quant, load_file_to_cuda, safe_save_file
from qlean.modifiers.save_utils import update_config, WA_for_sglang_config
from qlean.utils.hadamard import apply_had

import logging

logger = logging.getLogger(__name__)

@ModifierFactory.register("generalDay0Modifier")
class generalDay0Modifier(QuantizationMixin, Modifier):

    config_groups: Optional[Dict[str, QuantizationScheme]] = None
    ignore: List[str] = Field(default_factory=list)
    mxfp4_layers: List[str] = Field(default_factory=list)
    fp8_channelwise_layers: List[str] = Field(default_factory=list)
    fp8_blockwise_layers: List[str] = Field(default_factory=list)
    targets: Union[str, List[str]] = Field(default_factory=lambda: ["Linear"])
    scheme: Optional[Union[str, Dict[str, Any]]] = None
    kv_cache_scheme: Optional[QuantizationArgs] = None
    num_calibration_steps: Optional[int] = None
    rotation: Optional[str] = None

    valid_scheme_name: str = None
    valid_scheme: QuantizationScheme = None

    def on_initialize(self, state, *args, **kwargs):
        logger.info("general day0: on_initialize")
        self.valid_scheme_name, self.valid_scheme = validate_scheme(self.scheme)
        logger.info(f"valid_scheme_name: {self.valid_scheme_name}; valid_scheme.format: {self.valid_scheme.format}")
        self.initialized_ = True

    def on_start(self, state, *args, **kwargs):
        logger.info("general day0: on_start")
        super().on_start(state)

        torch.set_default_dtype(torch.bfloat16)
        config_file, model_index_file = copy_necessary_files(
            state.model.path, state.save_path
        )

        # delete quantization_config
        config = json.load(open(config_file))
        config.pop("quantization_config", None)
        if "text_config" in config:
            config["text_config"].pop("quantization_config", None)
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False, sort_keys=True)
        logger.info(f"config.json modified and saved to {config_file}")
        # get hidden_size
        hidden_size = config["text_config"]["hidden_size"] if "text_config" in config else config["hidden_size"]
        logger.info(f"hidden_size: {hidden_size}")
        # get model_index
        with open(model_index_file, "r") as f:
            model_index = json.load(f)

        quant_count = 0
        fp8_count = 0
        new_weight_map = {}
        module_ignored = []
        fp8_channelwise_found = []
        fp8_blockwise_found = []
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
                if module_name.endswith("experts"):
                    quant_count += 1
                    logger.debug(f"{weight_name} has dimension {weight.shape}")
                    for i in range(weight.shape[0]):
                        if param_name != "gate_up_proj":
                            curr_weight = weight[i].t().contiguous() if hidden_size == weight.shape[-1] else weight[i].contiguous()
                            quant_weight, scale_inv = weight_quant(curr_weight, self.valid_scheme_name)
                            curr_weight_name = f"{module_name}.{i}.{param_name}.weight"
                            new_state_dict[curr_weight_name] = quant_weight
                            new_scale_name = f"{curr_weight_name}_scale"
                            new_state_dict[new_scale_name] = scale_inv

                            new_weight_map[curr_weight_name] = file_name
                            new_weight_map[new_scale_name] = file_name
                        else:
                            curr_weight = weight[i].t() if hidden_size == weight.shape[1] else weight[i]
                            size = curr_weight.shape[0]
                            gate_weight = curr_weight[: size // 2, :].contiguous()
                            up_weight = curr_weight[size // 2 :, :].contiguous()
                            quant_gate_weight, gate_scale = weight_quant(gate_weight, self.valid_scheme_name)
                            quant_up_weight, up_scale = weight_quant(up_weight, self.valid_scheme_name)

                            gate_weight_name = f"{module_name}.{i}.gate_proj.weight"
                            gate_scale_name = f"{gate_weight_name}_scale"
                            up_weight_name = f"{module_name}.{i}.up_proj.weight"
                            up_scale_name = f"{up_weight_name}_scale"

                            new_state_dict[gate_weight_name] = quant_gate_weight
                            new_state_dict[gate_scale_name] = gate_scale
                            new_state_dict[up_weight_name] = quant_up_weight
                            new_state_dict[up_scale_name] = up_scale

                            new_weight_map[gate_weight_name] = file_name
                            new_weight_map[gate_scale_name] = file_name
                            new_weight_map[up_weight_name] = file_name
                            new_weight_map[up_scale_name] = file_name

                elif not is_linear_weight or is_ignored:
                    new_state_dict[weight_name] = weight
                    new_weight_map[weight_name] = file_name
                elif self.scheme == "MXFP4":
                    has_mxfp4 = bool(self.mxfp4_layers)

                     # determine quantization target
                    if any(match_name(module_name, p) for p in self.fp8_blockwise_layers):
                        target = "FP8_BLOCK"
                    elif any(match_name(module_name, p) for p in self.fp8_channelwise_layers):
                        target = "FP8_DYNAMIC"
                    elif any(match_name(module_name, p) for p in self.mxfp4_layers):
                        target = "MXFP4"
                    elif not has_mxfp4:
                        target = "MXFP4"
                    else:
                        target = "BF16"

                    if target == "MXFP4":
                        assert weight.element_size() == 2
                        if self.rotation == "hadamard":
                            weight = apply_had(weight)
                        quant_weight, scale_inv = weight_quant(weight, self.valid_scheme_name)
                        new_state_dict[weight_name] = quant_weight
                        new_state_dict[f"{weight_name}_scale"] = scale_inv
                        new_weight_map[weight_name] = file_name
                        new_weight_map[f"{weight_name}_scale"] = file_name
                        quant_count += 1
                    elif target == "FP8_DYNAMIC":
                        assert weight.element_size() == 2
                        quant_weight, scale_inv = weight_quant(weight, "FP8_DYNAMIC")
                        new_state_dict[weight_name] = quant_weight
                        new_state_dict[f"{weight_name}_scale"] = scale_inv
                        new_weight_map[weight_name] = file_name
                        new_weight_map[f"{weight_name}_scale"] = file_name
                        fp8_count += 1
                        fp8_channelwise_found.append(module_name)
                    elif target == "FP8_BLOCK":
                        assert weight.element_size() == 2
                        quant_weight, scale_inv = weight_quant(weight, "FP8_BLOCK")
                        new_state_dict[weight_name] = quant_weight
                        new_state_dict[f"{weight_name}_scale"] = scale_inv
                        new_weight_map[weight_name] = file_name
                        new_weight_map[f"{weight_name}_scale"] = file_name
                        fp8_count += 1
                        fp8_blockwise_found.append(module_name)
                    else:
                        new_state_dict[weight_name] = weight
                        new_weight_map[weight_name] = file_name
                else:
                    assert weight.element_size() == 2
                    quant_count += 1
                    if weight.dim() != 2:
                        logger.warning(
                            f"{weight_name} has dimension {weight.shape}."
                        )
                    quant_weight, scale_inv = weight_quant(weight, self.valid_scheme_name)
                    new_state_dict[weight_name] = quant_weight
                    new_scale_name = f"{weight_name}_scale"
                    new_state_dict[new_scale_name] = scale_inv

                    new_weight_map[weight_name] = file_name
                    new_weight_map[new_scale_name] = file_name

            new_safetensor_file = os.path.join(state.save_path, file_name)
            safe_save_file(new_state_dict, new_safetensor_file)
            del state_dict
            del new_state_dict
            torch.cuda.empty_cache()
        logger.info(f"{quant_count} weights are quantized.")
        if fp8_count:
            logger.info(f"{fp8_count} weights are quantized with FP8 (channelwise={len(fp8_channelwise_found)}, blockwise={len(fp8_blockwise_found)}).")

        with open(model_index_file, "r") as f:
            model_index = json.load(f)
        model_index["weight_map"] = new_weight_map
        with open(model_index_file, "w", encoding="utf-8") as f:
            json.dump(model_index, f, indent=2, ensure_ascii=False, sort_keys=True)
        logger.info(
            f"model.safetensors.index.json modified and saved to {model_index_file}"
        )

        update_config(
            state.save_path,
            state.save_path,
            self.valid_scheme_name,
            self.valid_scheme,
            module_ignored,
        )

        if self.valid_scheme_name == "MXFP4":
            WA_for_sglang_config(config_file, fp8_channelwise_found, fp8_blockwise_found)

    def on_end(self, state):
        logger.info("general day0: on_end")
        super().on_end(state)

    def on_finalize(self, state):
        logger.info("general day0: on_finalize")
        super().on_finalize(state)

    def on_event(self, state, event: Event, **kwargs):
        logger.info("general day0: on_event")
        if event.type_ == EventType.CALIBRATION_EPOCH_START:
            if not self.started_:
                self.on_start(state)

        if event.type_ == EventType.CALIBRATION_EPOCH_END:
            if not self.ended_:
                self.on_end(state)
