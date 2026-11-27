# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
import json
import logging
from copy import deepcopy
from functools import partial
from typing import Any, Dict, List, Optional, Union

import torch
from pydantic import Field

from qlean.utils.ct_compat import match_name

from qlean.modifiers import Modifier, ModifierFactory, QuantizationMixin
from qlean.modifiers.utils import validate_scheme, ensure_scales_in_weight_map
from qlean.modifiers.save_utils import update_config, WA_for_sglang_config
from qlean.modifiers_engine._runtime import (
    make_hooks, run_engine, load_source_metadata, EngineModifierMixin,
)
from qlean.modifiers_engine._runtime.region_router import (
    normalize_regions,
    validate_regions,
    validate_region_srcs,
    effective_src,
    is_passthrough,
    spec_lazy_input,
    spec_plan_fn,
    spec_on_plan,
    derive_dequant_map,
    derive_filter_source_key,
    derive_stream_map,
    report_region_hits,
)
from qlean.modifiers_engine._runtime.run_engine import copy_model_assets
from qlean.modifiers_engine._runtime.source_inspect import inspect_source_regions, read_source_index
from qlean.modifiers_engine._runtime.verify import verify_op, write_verify_summary

logger = logging.getLogger(__name__)


@ModifierFactory.register("GeneralQuantEngineModifier")
class GeneralQuantEngineModifier(EngineModifierMixin, QuantizationMixin, Modifier):
    """Convert checkpoint shards using ordered region rules and write output metadata."""

    regions: List[dict] = Field(default_factory=list)

    primary_scheme: Optional[Union[str, Dict[str, Any]]] = None
    hidden_size: Optional[int] = None
    config_writer: str = Field(default="compressed_tensors")
    sglang_wa: Union[str, bool] = Field(default="auto")
    extra_source_index: Optional[str] = None
    int4_group_size: int = Field(default=32, strict=True, gt=0)
    nvfp4_group_size: int = 16
    dequant_block_size: int = 128
    scale_naming: str = Field(
        default="underscore",
        description="Output scale naming for all operators (quantization / identity / streaming): "
                    "underscore=x.weight → x.weight_scale (default); "
                    "dot=x.weight → x.scale (module-level naming). Output weight keys normalize to "
                    "*.weight (weight_packed → weight)",
    )
    dequant_dtype: str = Field(
        default="fp32",
        description="Dequantization intermediate dtype: fp32 (default, high precision) | bf16 (dequantization-only output)",
    )
    verify: str = Field(
        default="off",
        description="Quantization output verification: off (default, no overhead, output and fingerprint unchanged byte-for-byte) | "
                    "bound (elementwise theoretical bound checks, written to "
                    "save_path/qlean_verify/<shard>.csv; numerical violations do not abort)",
    )
    scale_search: str = Field(
        default="absmax",
        description="acext_int4 scale: absmax (per-row absmax/7) | mse (per-row MSE search)",
    )
    mse_divisor_min: float = 4.0
    mse_divisor_max: float = 24.0
    mse_divisor_step: float = 0.25

    def on_initialize(self, state, *args, **kwargs):
        self.regions = normalize_regions(self.regions)
        validate_regions(self.regions)
        validate_region_srcs(self.regions)
        if self.dequant_dtype not in ("bf16", "fp32"):
            raise ValueError(
                f"Unknown dequant_dtype: {self.dequant_dtype!r}, allowed: 'bf16' | 'fp32'"
            )
        if self.verify not in ("off", "bound"):
            raise ValueError(
                f"Unknown verify: {self.verify!r}, allowed: 'off' | 'bound'"
            )
        if self.scale_naming not in ("underscore", "dot"):
            raise ValueError(
                f"Unknown scale_naming: {self.scale_naming!r}, "
                f"allowed: 'underscore' | 'dot'"
            )
        if self.scale_search not in ("absmax", "mse"):
            raise ValueError(
                f"Unknown scale_search: {self.scale_search!r}, "
                f"allowed: 'absmax' | 'mse'"
            )
        if self.scale_search == "mse":
            if (self.mse_divisor_min <= 0 or self.mse_divisor_max <= self.mse_divisor_min
                    or self.mse_divisor_step <= 0):
                raise ValueError(
                    "Invalid MSE int4 divisor grid: requires min > 0, max >= min, step > 0, "
                    f"got min={self.mse_divisor_min}, max={self.mse_divisor_max}, "
                    f"step={self.mse_divisor_step}"
                )
            if not any(r.get("dst") == "acext_int4" for r in self.regions):
                raise ValueError(
                    "scale_search=mse requires at least one region with dst=acext_int4"
                )
        if self.config_writer not in ("compressed_tensors",
                                      "mixed_precision_w4", "none"):
            raise ValueError(f"Unknown config_writer: {self.config_writer!r}")
        if self.config_writer == "compressed_tensors" and self.primary_scheme is None:
            raise ValueError(
                "config_writer=compressed_tensors requires an explicit primary_scheme"
            )
        if self.config_writer == "mixed_precision_w4":
            if not any(r.get("name") == "experts" for r in self.regions):
                raise ValueError(
                    "config_writer=mixed_precision_w4 requires a region named 'experts': "
                    "expert_dtype is written only for this name"
                )
            if not any(r.get("dst") == "acext_int4" for r in self.regions):
                raise ValueError(
                    "config_writer=mixed_precision_w4 requires at least one region with "
                    "dst=acext_int4"
                )
        self._validate_passthrough_export()
        self.initialized_ = True

    def _validate_passthrough_export(self) -> None:
        """Explicit passthrough leaves the export config unconstrained; callers own deployment compatibility."""

    def on_start(self, state, *args, **kwargs):
        super().on_start(state)
        torch.set_default_dtype(torch.bfloat16)

        config, weight_map = load_source_metadata(state.model.path, strip_quant_config=False)
        preserve_source_config = (
            self.config_writer == "none" and any(is_passthrough(r) for r in self.regions)
        )
        # Record field presence before stripping; text_config alone does not opt in to synchronization.
        text_config = config.get("text_config")
        update_text_quant_config = (
            isinstance(text_config, dict) and "quantization_config" in text_config
        )
        expert_dtype_paths = []
        for path, section in (((), config), (("text_config",), config.get("text_config"))):
            if not isinstance(section, dict):
                continue
            if "expert_dtype" in section:
                expert_dtype_paths.append((*path, "expert_dtype"))
            quant_config = section.get("quantization_config")
            if isinstance(quant_config, dict) and "expert_dtype" in quant_config:
                expert_dtype_paths.append((*path, "quantization_config", "expert_dtype"))
            if not preserve_source_config:
                section.pop("quantization_config", None)

        need_dequant = any(effective_src(r) != "bf16" for r in self.regions)
        if need_dequant:
            ensure_scales_in_weight_map(weight_map, state.model.path)
        if self.extra_source_index is not None:
            # Extra index shard names must match the source; the union drives has_source_scale.
            if not os.path.exists(self.extra_source_index):
                raise FileNotFoundError(
                    f"extra_source_index does not exist: {self.extra_source_index}"
                )
            extra_map = read_source_index(self.extra_source_index)
            weight_map = {**weight_map, **extra_map}

        hidden = self.hidden_size or (
            config["text_config"]["hidden_size"] if "text_config" in config
            else config.get("hidden_size"))
        transform_cfg = {
            "_regions": self.regions,
            "hidden_size": hidden,
            "int4_group_size": self.int4_group_size,
            "nvfp4_group_size": self.nvfp4_group_size,
            "dequant_block_size": self.dequant_block_size,
        }
        # Omit defaults to preserve existing recipe fingerprints.
        if self.scale_naming != "underscore":
            transform_cfg["scale_naming"] = self.scale_naming
        if self.dequant_dtype != "fp32":
            transform_cfg["dequant_dtype"] = self.dequant_dtype
        if self.scale_search == "mse":
            transform_cfg.update(
                scale_search=self.scale_search,
                mse_divisor_min=self.mse_divisor_min,
                mse_divisor_max=self.mse_divisor_max,
                mse_divisor_step=self.mse_divisor_step,
            )
            logger.info(
                "[int4] scale_search=mse divisor grid "
                "[%s, %s] / %s",
                self.mse_divisor_min,
                self.mse_divisor_max,
                self.mse_divisor_step,
            )
        elif any(r.get("dst") == "acext_int4" for r in self.regions):
            logger.info("[int4] scale_search=absmax (per-row absmax / 7)")

        # Dead-pattern checks also apply to unquantized sources.
        inspect_source_regions(state.model.path, weight_map, self.regions)

        # Output dtype and names cannot distinguish identity copies from requantization.
        self._log_passthrough_regions()
        source_files = list(state.model.model_)

        # partial is spawn-picklable; share verify_dir with resume sidecar checks.
        verify_fn = None
        verify_dir = None
        if self.verify == "bound":
            verify_dir = os.path.join(state.save_path, "qlean_verify")
            verify_fn = partial(verify_op, out_dir=verify_dir)
            logger.info(f"[verify] Bound checks enabled; details written to {verify_dir}")
        else:
            logger.info("[verify] Disabled (verify=off); output and fingerprint unchanged byte-for-byte")
        process_tensor, expected_output_keys = make_hooks(
            spec_plan_fn, on_plan=spec_on_plan, verify=verify_fn)
        stats = run_engine(
            state.model.path, state.save_path, source_files, weight_map,
            process_tensor=process_tensor,
            expected_output_keys=expected_output_keys,
            transform_cfg=transform_cfg,
            filter_source_key=derive_filter_source_key(self.regions),
            dequant_map=derive_dequant_map(self.regions),
            stream_map=derive_stream_map(self.regions),
            lazy_input=spec_lazy_input,
            num_gpus=self.num_gpus, resume=self.resume,
            force_rerun=self.force_rerun,
            verify_dir=verify_dir,
        )

        report_region_hits(self.regions, stats)
        copy_model_assets(state.model.path, state.save_path)
        config_file = os.path.join(state.save_path, "config.json")
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False, sort_keys=True)
        self._write_config(config_file, stats, expert_dtype_paths,
                           update_text_quant_config=update_text_quant_config)
        if verify_dir is not None:
            write_verify_summary(verify_dir, source_files, stats)

        state.model.model_ = [
            (fn, os.path.join(state.save_path, os.path.basename(fn)))
            for fn, _ in source_files
        ]

    def _log_passthrough_regions(self) -> None:
        for region in self.regions:
            if is_passthrough(region):
                src = effective_src(region)
                storage = ("reinterpret weights/scales as uint8, keep bytes and layout"
                           if src == "mxfp4" else "keep dtype and layout")
                logger.info(
                    "[passthrough] region=%s src=%s: copy weights and scales on CPU, "
                    "normalize key names, %s, no dequant or requant",
                    region["name"], src, storage,
                )

    def _write_config(self, config_file: str, stats: dict, expert_dtype_paths: list,
                      update_text_quant_config: bool = False) -> None:
        if self.config_writer == "none" and any(is_passthrough(r) for r in self.regions):
            logger.warning(
                "config_writer=none: source config kept as-is, no scheme/ignore/expert_dtype "
                "rewrite; caller must confirm it matches the normalized key names and other regions"
            )
            return
        # Ignore bf16 outputs so the config does not claim they were quantized.
        bf16_unq = sorted(set(stats.get("bf16_unquantized", [])))
        if bf16_unq:
            logger.info(
                f"[config] {len(bf16_unq)} modules kept as bf16 and added to ignore"
            )

        if self.config_writer == "compressed_tensors":
            scheme_name, scheme = validate_scheme(self.primary_scheme)
            save_path = os.path.dirname(config_file)
            update_config(save_path, save_path, scheme_name, scheme, bf16_unq)
        elif self.config_writer == "mixed_precision_w4":
            self._write_mixed_precision_w4(config_file, bf16_unq)
        else:
            logger.info("config_writer=none: skipping quantization_config write")

        if self._sglang_wa_enabled():
            # None avoids writing an empty FP8_DYNAMIC list to the config.
            WA_for_sglang_config(
                config_file, stats.get("fp8_channelwise_found") or None,
                sorted(set(stats.get("fp8_blockwise_found", []))) or None)

        if update_text_quant_config and self.config_writer in ("compressed_tensors", "mixed_precision_w4"):
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
            if isinstance(config.get("text_config"), dict):
                # Sync after WA but before restoring location-specific expert markers.
                config["text_config"]["quantization_config"] = deepcopy(config["quantization_config"])
                with open(config_file, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2, ensure_ascii=False, sort_keys=True)

        self._update_expert_dtype(config_file, expert_dtype_paths)

    def _write_mixed_precision_w4(self, config_file: str,
                                  bf16_unq: list = ()) -> None:
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)

        ignore_patterns = []
        for m in bf16_unq:
            if (not match_name("model.embed_tokens", m)
                    and m not in ignore_patterns):
                ignore_patterns.append(m)

        qconfig = {
            "activation_scheme": "dynamic",
            "quant_method": "mixed_precision_w4",
            "ignore": ignore_patterns,
        }
        config["quantization_config"] = qconfig

        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False, sort_keys=True)
        logger.info(
            f"quantization_config (mixed_precision_w4) saved to {config_file}"
            f" (ignore {len(ignore_patterns)} entries)"
        )

    def _sglang_wa_enabled(self) -> bool:
        if isinstance(self.sglang_wa, str):
            if self.sglang_wa == "auto":
                return any(
                    r.get("dst") == "MXFP4"
                    or (is_passthrough(r) and effective_src(r) == "mxfp4")
                    for r in self.regions
                )
            lowered = self.sglang_wa.lower()
            if lowered in ("true", "false"):
                return lowered == "true"
            raise ValueError(
                f"Invalid sglang_wa value: {self.sglang_wa!r}, "
                f"allowed: 'auto' | true | false"
            )
        return bool(self.sglang_wa)

    _DST_TO_EXPERT_DTYPE = {
        "W8A8": "int8",
        "FP8_DYNAMIC": "fp8",
        "FP8_BLOCK": "fp8",
        "acext_int4": "int4",
        "MXFP4": "fp4",
    }

    def _update_expert_dtype(self, config_file: str, paths: list) -> None:
        if not paths:
            return
        expert_region = next(
            (r for r in self.regions if r.get("name") == "experts"), None
        )
        if expert_region is None:
            return

        dst = expert_region.get("dst")
        if is_passthrough(expert_region):
            dst = {"mxfp4": "MXFP4", "bf16": "bf16"}.get(effective_src(expert_region))
        new_dtype = self._DST_TO_EXPERT_DTYPE.get(dst)
        if new_dtype is None and dst not in (None, "bf16"):
            return

        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)

        for path in paths:
            section = config
            for key in path[:-1]:
                section = section.get(key)
                if not isinstance(section, dict):
                    break
            else:
                if new_dtype is None:
                    # Remove stale markers rather than routing unquantized experts as quantized.
                    section.pop(path[-1], None)
                else:
                    section[path[-1]] = new_dtype

        if new_dtype is None:
            logger.info(f"[config] expert_dtype removed (experts dst={dst!r}, unquantized)")
        else:
            logger.info(f"[config] expert_dtype → {new_dtype!r} (dst={dst})")

        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False, sort_keys=True)
