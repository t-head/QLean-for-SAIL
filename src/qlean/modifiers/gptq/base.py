# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
# Ported from: llmcompressor/modifiers/gptq/base.py

from __future__ import annotations

import contextlib
import copy
import time
from typing import Dict, List, Optional, Set, Union

import torch
from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
    QuantizationStrategy,
)
from compressed_tensors.quantization.quant_args import ActivationOrdering
from compressed_tensors.utils import (
    align_module_device,
    get_execution_device,
    getattr_chain,
    match_named_modules,
    update_offload_parameter,
)
from loguru import logger
from pydantic import Field, PrivateAttr

from qlean.core.event import Event
from qlean.core.routes import GPTQ_EXPERT_TARGET, GPTQ_INT4_PRESETS, GPTQ_WEIGHT_CONFIGS
from qlean.core.state import State
from qlean.modifiers import Modifier, ModifierFactory
from qlean.modifiers.quantization import QuantizationCalibrationMixin
from qlean.modifiers.gptq.diagnostics import (
    CalibrationDiagnostics,
    _disable_onloading,
    diagnostics_enabled,
)
from qlean.modifiers.gptq.distributed import (
    broadcast_qparams_and_cleanup,
    greedy_bin_packing,
    is_distributed,
    wait_for_comms,
)
from qlean.modifiers.gptq.gptq_quantize import (
    accumulate_hessian,
    make_empty_hessian,
    quantize_weight,
)
from qlean.modeling.moe.context import moe_calibration_context
from qlean.modeling.moe.linear_experts import LinearExperts2D
from qlean.modifiers.gptq.moe_support import (
    linearize_moe_layers,
    validate_int4_expert_modules,
)

__all__ = ["GPTQModifier"]

# Quantization parameter names to broadcast in distributed mode
_GPTQ_Q_PARAMS = [
    "weight",
    "weight_scale",
    "weight_zero_point",
    "weight_global_scale",
    "weight_g_idx",
]


@ModifierFactory.register("GPTQModifier")
class GPTQModifier(Modifier, QuantizationCalibrationMixin):
    """GPTQ weight quantization with MXFP4, INT4/INT8 channel, or block schemes."""

    # GPTQ algorithm parameters
    block_size: int = Field(default=128, description="Block size for quantization")
    dampening_frac: Optional[float] = Field(
        default=0.01, description="Dampening fraction for Hessian"
    )
    actorder: Optional[Union[ActivationOrdering, str]] = Field(
        default="static", description="Activation ordering strategy"
    )
    offload_hessians: bool = Field(
        default=False, description="Offload Hessians to CPU memory"
    )
    moe_calibrate_all_experts: bool = Field(
        default=True,
        description=(
            "For MoE models, route calibration data through every expert so each "
            "expert accumulates statistics (matches llm-compressor's default). "
            "This is much slower than top-k-only routing (e.g. ~16x for 128 "
            "experts with top-8); disable for faster calibration when every "
            "expert is expected to see enough tokens through normal routing."
        ),
    )

    # Internal state (not serialized)
    _module_names: Dict[torch.nn.Module, str] = PrivateAttr(default_factory=dict)
    _hessians: Dict[torch.nn.Module, torch.Tensor] = PrivateAttr(default_factory=dict)
    _num_samples: Dict[torch.nn.Module, torch.Tensor] = PrivateAttr(default_factory=dict)
    _is_moe_model: bool = PrivateAttr(default=False)
    _moe_calibration_ctx: Optional[contextlib.AbstractContextManager] = PrivateAttr(
        default=None
    )
    _hook_call_counts: Dict[torch.nn.Module, int] = PrivateAttr(default_factory=dict)
    _nonfinite_reported: Set[torch.nn.Module] = PrivateAttr(default_factory=set)
    _skipped_modules: Dict[str, str] = PrivateAttr(default_factory=dict)
    _diagnostics: Optional[CalibrationDiagnostics] = PrivateAttr(default=None)
    _compressed_modules: Set[str] = PrivateAttr(default_factory=set)
    _processed_modules: Set[str] = PrivateAttr(default_factory=set)
    _active_modules: List[torch.nn.Module] = PrivateAttr(default_factory=list)
    _active_module_set: Set[torch.nn.Module] = PrivateAttr(default_factory=set)
    _sequential_calibration: bool = PrivateAttr(default=False)

    def resolve_quantization_config(self) -> QuantizationConfig:
        supported = ", ".join(GPTQ_WEIGHT_CONFIGS)
        fields_set = getattr(self, "model_fields_set", None)
        if fields_set is None:
            fields_set = getattr(self, "__fields_set__", set())
        if self.scheme is not None and self.config_groups is not None:
            raise ValueError("Specify only one of GPTQModifier.scheme and config_groups")
        if isinstance(self.scheme, str):
            if self.scheme not in GPTQ_WEIGHT_CONFIGS:
                raise ValueError(f"Unsupported GPTQ scheme '{self.scheme}'; supported schemes: {supported}")
            targets = [self.targets] if isinstance(self.targets, str) else self.targets
            if self.scheme in GPTQ_INT4_PRESETS and "targets" not in fields_set:
                targets = [GPTQ_EXPERT_TARGET]
            config = QuantizationConfig(
                config_groups={"preset": QuantizationScheme(
                    targets=targets,
                    weights=copy.deepcopy(GPTQ_WEIGHT_CONFIGS[self.scheme]),
                )},
                ignore=self.ignore,
                kv_cache_scheme=self.kv_cache_scheme,
                quantization_status=QuantizationStatus.INITIALIZED,
            )
            self._resolved_targets = set(targets)
        else:
            if isinstance(self.scheme, dict) and "weights" not in self.scheme:
                raise ValueError(f"Unsupported GPTQ scheme dictionary; supported schemes: {supported}")
            config = super().resolve_quantization_config().model_copy(deep=True)
        if not config.config_groups:
            raise ValueError(f"GPTQ requires a quantization configuration; supported schemes: {supported}")
        if config.kv_cache_scheme is not None:
            raise ValueError("GPTQ does not support KV-cache quantization")
        weight_types = set()
        int_strategies = set()
        block_bits = set()
        group_bits = set()
        for name, scheme in config.config_groups.items():
            if scheme.input_activations is not None or scheme.output_activations is not None:
                raise ValueError(f"GPTQ does not support activation quantization in '{name}'")
            weights = scheme.weights
            if weights is None:
                raise ValueError(f"GPTQ requires weights in '{name}'; supported schemes: {supported}")
            channel = (
                weights.type == "int" and weights.num_bits in (4, 8)
                and weights.strategy == "channel" and weights.group_size is None
                and weights.block_structure is None
                and weights.scale_dtype in (None, torch.float16, torch.bfloat16, torch.float32)
            )
            group = (
                weights.type == "int" and weights.num_bits in (4, 8)
                and weights.strategy == "group" and weights.group_size == 128
                and weights.block_structure is None
                and weights.scale_dtype in (None, torch.float16, torch.bfloat16, torch.float32)
            )
            block = (
                weights.type == "int" and weights.num_bits in (4, 8)
                and weights.strategy == "block" and weights.group_size is None
                and list(weights.block_structure or ()) == [128, 128]
                and weights.scale_dtype in (None, torch.float16, torch.bfloat16, torch.float32)
            )
            mxfp4 = (
                weights.type == "float" and weights.num_bits == 4
                and weights.strategy == "group" and weights.group_size == 32
                and weights.scale_dtype == torch.uint8 and weights.zp_dtype == torch.uint8
            )
            if (
                not (channel or group or block or mxfp4) or not weights.symmetric
                or weights.dynamic
            ):
                raise ValueError(f"Unsupported GPTQ weights in '{name}'; supported schemes: {supported}")
            weight_types.add(weights.type)
            if weights.type == "int":
                int_strategies.add(weights.strategy)
                if weights.strategy == "block":
                    block_bits.add(weights.num_bits)
                elif weights.strategy == "group":
                    group_bits.add(weights.num_bits)
        if len(weight_types) > 1:
            raise ValueError("GPTQ cannot mix MXFP4 with integer INT4/INT8 weights")
        if len(int_strategies) > 1:
            raise ValueError("GPTQ cannot mix block, group, and channel integer weights")
        if len(block_bits) > 1:
            raise ValueError("GPTQ cannot mix block INT4 and INT8 weights")
        if len(group_bits) > 1:
            raise ValueError("GPTQ cannot mix group INT4 and INT8 weights")
        actorder_is_explicit = "actorder" in fields_set

        def normalize_actorder(value):
            if value is None or isinstance(value, ActivationOrdering):
                return value
            return ActivationOrdering(value)

        modifier_actorder = normalize_actorder(self.actorder)
        grouped_strategies = (
            QuantizationStrategy.GROUP,
            QuantizationStrategy.TENSOR_GROUP,
        )
        supported_strategies = (
            QuantizationStrategy.TENSOR,
            QuantizationStrategy.CHANNEL,
            QuantizationStrategy.GROUP,
            QuantizationStrategy.TENSOR_GROUP,
            QuantizationStrategy.BLOCK,
        )

        for scheme in config.config_groups.values():
            weights = scheme.weights
            if weights is None or weights.strategy not in supported_strategies:
                continue

            configured_actorder = normalize_actorder(
                getattr(weights, "actorder", None)
            )
            if not actorder_is_explicit:
                resolved_actorder = (
                    configured_actorder
                    if configured_actorder is not None
                    else ActivationOrdering.STATIC
                )
            elif (
                configured_actorder is None
                or configured_actorder == modifier_actorder
            ):
                resolved_actorder = modifier_actorder
            else:
                raise ValueError(
                    "Cannot resolve activation ordering when both "
                    "GPTQModifier.actorder and QuantizationScheme.weights.actorder "
                    f"are provided and differ ({modifier_actorder}, "
                    f"{configured_actorder}). Set only one value or make them match."
                )

            if (
                resolved_actorder == ActivationOrdering.GROUP
                and weights.strategy not in grouped_strategies
            ):
                logger.warning(
                    "ActivationOrdering.GROUP is only supported for group and "
                    "tensor-group weights; disabling it for strategy={}. ",
                    weights.strategy,
                )
                resolved_actorder = None

            if (
                weights.strategy == QuantizationStrategy.BLOCK
                and resolved_actorder is not None
            ):
                logger.warning(
                    "Activation ordering changes block membership and is not "
                    "supported for block-wise GPTQ; disabling it for "
                    "strategy=block."
                )
                resolved_actorder = None

            weights.actorder = resolved_actorder

        return config

    def initialize_quantization(self, model: torch.nn.Module):
        config = self.resolve_quantization_config()
        int4_modules = {}
        has_int4 = False
        for scheme in config.config_groups.values():
            weights = scheme.weights
            if (
                weights is None or weights.type != "int" or weights.num_bits != 4
                or weights.strategy != "channel"
            ):
                continue
            has_int4 = True
            int4_modules.update(
                match_named_modules(model, scheme.targets, config.ignore)
            )
        if has_int4:
            validate_int4_expert_modules(model, int4_modules)
        return super().initialize_quantization(model)

    def on_initialize(self, state: State, **kwargs):
        """
        Initialize the modifier.

        - Linearize MoE layers if detected
        - Initialize quantization config (via QuantizationCalibrationMixin)
        """
        logger.info("GPTQModifier: on_initialize")

        model = state.model.model_

        # Restore the existing DeepSeek RoPE buffers only for this opt-in path.
        config = getattr(model, "config", None)
        if getattr(config, "model_type", None) == "deepseek_v32":
            from qlean.modeling.deepseekv32 import (
                DeepseekV32ForCausalLM,
                restore_deepseek_v32_freqs_cis,
            )
            if isinstance(model, DeepseekV32ForCausalLM):
                restore_deepseek_v32_freqs_cis(model)

        # Linearize fused MoE layers if detected. This replaces the fused experts
        # module (e.g. Qwen3MoeExperts at mlp.experts) with a LinearExperts2D
        # container of per-expert nn.Linear layers, which is exactly how
        # llm-compressor handles MoE for GPTQ. Linearization must happen before
        # quantization is initialized so the new Linear modules receive observers.
        linearize_moe_layers(model)

        # Treat the model as MoE if it was just linearized or already contains
        # linearized experts (e.g. loaded via load_quantizable_moe).
        self._is_moe_model = any(
            isinstance(module, LinearExperts2D) for module in model.modules()
        )
        logger.info(f"_is_moe_model after linearization: {self._is_moe_model}")

        # Initialize quantization (via QuantizationCalibrationMixin)
        if QuantizationCalibrationMixin.has_config(self):
            self.initialize_quantization(model)
            weights = [
                scheme.weights
                for scheme in self._resolved_config.config_groups.values()
                if scheme.weights is not None
            ]
            has_channel_int4 = any(
                args.type == "int" and args.num_bits == 4 and args.strategy == "channel"
                for args in weights
            )
            has_channel_int8 = any(
                args.type == "int" and args.num_bits == 8 and args.strategy == "channel"
                for args in weights
            )
            if has_channel_int4 or has_channel_int8:
                from qlean.modifiers.gptq.export import (
                    a8w4_quantized_modules,
                    int8_quantized_modules,
                )

                if self._resolved_config.kv_cache_scheme is not None:
                    raise ValueError("A8W4/channel INT8 does not support KV-cache quantization")
                if has_channel_int4:
                    a8w4_quantized_modules(model, require_int4=True)
                elif not a8w4_quantized_modules(model):
                    int8_quantized_modules(model, require_int8=True)

        self.initialized_ = True
        logger.info("GPTQModifier initialization complete")

    def on_start(self, state: State, **kwargs):
        """
        Start calibration: add hooks for Hessian accumulation.
        """
        logger.info("GPTQModifier: on_start (adding calibration hooks)")
        from qlean.modifiers.gptq.export import (
            a8w4_quantized_modules,
            int8_quantized_modules,
        )

        model = state.model.model_
        if not a8w4_quantized_modules(model):
            int8_quantized_modules(model)
        self._sequential_calibration = bool(kwargs.get("sequential", False))
        super().on_start(state)

        # For MoE models, route calibration data through every expert so each
        # expert accumulates its own Hessian.
        if self._is_moe_model and self.moe_calibrate_all_experts:
            self._moe_calibration_ctx = moe_calibration_context()
            self._moe_calibration_ctx.__enter__()
            logger.info(
                "Entered MoE all-experts calibration context (every expert "
                "processes every token; this is significantly slower than "
                "top-k-only routing. Set moe_calibrate_all_experts=False to "
                "disable)"
            )

        # Start calibration (via QuantizationCalibrationMixin)
        self.start_calibration(model)

        # Find all modules with a resolved weight quantization scheme.
        self._module_names = {}
        for name, module in match_named_modules(
            model, self._resolved_targets, self.ignore
        ):
            if (
                not isinstance(module, torch.nn.Embedding)
                and getattr_chain(module, "quantization_scheme.weights", None)
                is not None
            ):
                self._module_names[module] = name

        if not self._module_names:
            raise ValueError(
                "GPTQModifier was unable to find any weight modules to quantize. "
                "Check quantization config_groups and targets."
            )

        logger.info(f"Found {len(self._module_names)} modules to quantize")

        self._log_hessian_memory_estimate()

        # Initialize Hessian accumulation structures
        self._hessians = {}
        self._num_samples = {}
        self._compressed_modules.clear()
        self._processed_modules.clear()
        self._active_modules.clear()
        self._active_module_set.clear()
        self._skipped_modules.clear()

        # Register forward pre-hooks for Hessian accumulation
        total_modules = len(self._module_names)
        progress_interval = max(1, total_modules // 10)
        start = time.perf_counter()
        logger.info(
            "Registering full-model Hessian hooks for {} target modules; "
            "sequential calibration allocates Hessians only for the current round",
            total_modules,
        )
        for index, module in enumerate(self._module_names, start=1):
            self.register_hook(
                module,
                self._calibrate_hessian_hook,
                hook_type="forward_pre",
            )
            if index % progress_interval == 0 or index == total_modules:
                logger.info(
                    f"Hessian hook registration: {index}/{total_modules} "
                    f"({time.perf_counter() - start:.1f}s elapsed)"
                )

        if self._sequential_calibration:
            logger.info(
                "Full-model Hessian hooks registered; Hessian tensors are allocated "
                "only for the active sequential round"
            )
        else:
            logger.info(
                "Full-model Hessian hooks registered; Hessian tensors are allocated "
                "lazily when target modules execute"
            )

        if diagnostics_enabled():
            self._diagnostics = CalibrationDiagnostics(model, self._module_names)
            self._diagnostics.start(self.register_hook)

    def on_calibration_start(self, state: State, event: Event, **kwargs):
        if not self.started_:
            self.on_start(state, **kwargs)

    def on_sequential_epoch_start(
        self,
        state: State,
        event: Event,
        modules: List[torch.nn.Module],
        calibration_scope: str,
        calibration_round_index: int,
        num_calibration_rounds: int,
        is_last_calibration_round: bool,
        **kwargs,
    ):
        if not self._sequential_calibration:
            raise RuntimeError(
                "Sequential GPTQ round started outside sequential calibration"
            )
        if self._active_modules or self._hessians or self._num_samples:
            raise RuntimeError(
                "Sequential GPTQ started a new round before the previous round "
                "was compressed"
            )
        if num_calibration_rounds <= 0:
            raise ValueError("num_calibration_rounds must be positive")
        if not 0 <= calibration_round_index < num_calibration_rounds:
            raise ValueError(
                "calibration_round_index must be in "
                f"[0, {num_calibration_rounds}), got {calibration_round_index}"
            )
        expected_last_round = (
            calibration_round_index + 1 == num_calibration_rounds
        )
        if is_last_calibration_round != expected_last_round:
            raise ValueError(
                "is_last_calibration_round does not match calibration round "
                f"{calibration_round_index + 1}/{num_calibration_rounds}"
            )

        all_scope_modules = [
            module
            for module in dict.fromkeys(modules)
            if module in self._module_names
        ]
        base_size, remainder = divmod(
            len(all_scope_modules), num_calibration_rounds
        )
        start = (
            calibration_round_index * base_size
            + min(calibration_round_index, remainder)
        )
        size = base_size + (calibration_round_index < remainder)
        scope_modules = [
            module
            for module in all_scope_modules[start : start + size]
            if self._module_names[module] not in self._processed_modules
        ]

        self._assert_distributed_module_names(scope_modules)
        self._active_modules = scope_modules
        self._active_module_set = set(scope_modules)
        for module in scope_modules:
            self._initialize_hessian(module)

        processed = sum(
            self._module_names[module] in self._processed_modules
            for module in all_scope_modules
        )
        logger.info(
            "Sequential GPTQ {} round {}/{}: layer_total_targets={}, "
            "round_calibrate_quantize_targets={}, processed={}, "
            "remaining_before_round={}",
            calibration_scope,
            calibration_round_index + 1,
            num_calibration_rounds,
            len(all_scope_modules),
            len(scope_modules),
            processed,
            len(all_scope_modules) - processed,
        )

    def on_sequential_epoch_end(
        self,
        state: State,
        event: Event,
        modules: Optional[List[torch.nn.Module]] = None,
        calibration_scope: Optional[str] = None,
        calibration_round_index: Optional[int] = None,
        num_calibration_rounds: Optional[int] = None,
        is_last_calibration_round: bool = False,
        **kwargs,
    ):
        self.compress_modules()
        self._active_modules.clear()
        self._active_module_set.clear()

        if is_last_calibration_round and modules is not None:
            expected = {
                self._module_names[module]
                for module in modules
                if module in self._module_names
            }
            missing = sorted(expected - self._processed_modules)
            if missing:
                preview = ", ".join(missing[:10])
                suffix = " ..." if len(missing) > 10 else ""
                raise RuntimeError(
                    f"Sequential GPTQ exhausted its calibration rounds for "
                    f"'{calibration_scope}' with {len(missing)} target module(s) "
                    f"still unprocessed: {preview}{suffix}. Verify the calibration "
                    "round metadata and distributed model topology."
                )

    def on_calibration_end(self, state: State, event: Event, **kwargs):
        if not self.ended_:
            self.on_end(state)

    def on_end(self, state: State, **kwargs):
        """
        End calibration: remove hooks and freeze quantization.
        """
        logger.info("GPTQModifier: on_end (removing hooks)")
        super().on_end(state)

        model = state.model.model_

        # End calibration (via QuantizationCalibrationMixin)
        self.end_calibration(model)

        self._remove_calibration_hooks()

        if self._diagnostics is not None:
            self._diagnostics.log_summary()

        logger.info("GPTQModifier calibration complete")

    def on_finalize(self, state: State, **kwargs):
        """
        Finalize the modifier: clean up resources.
        """
        logger.info("GPTQModifier: on_finalize")

        if not self.ended_:
            self.on_end(state)

        finalization_error = None
        if self._num_samples:
            finalization_error = ValueError(
                f"Failed to compress {len(self._num_samples)} modules"
            )
        elif self._skipped_modules:
            names = sorted(self._skipped_modules)
            preview = ", ".join(names[:10])
            suffix = " ..." if len(names) > 10 else ""
            finalization_error = RuntimeError(
                f"{len(self._skipped_modules)} modules were skipped because "
                f"their Hessians were degenerate: {preview}{suffix}. The most "
                "common cause is non-finite activations during calibration "
                "(see the 'GPTQ diag' lines logged for every skipped module): "
                "load bfloat16 checkpoints in bfloat16, or check "
                "padding/attention_mask."
            )
            if self._diagnostics is not None:
                logger.error(self._diagnostics.format_skip_table())
            else:
                logger.error(
                    "Skipped modules: " + ", ".join(names)
                )

        self._cleanup_calibration()

        super().on_finalize(state)
        if finalization_error is not None:
            raise finalization_error
        logger.info("GPTQModifier finalized")

    def _remove_calibration_hooks(self):
        """Release hooks and the all-experts context on either exit path."""
        try:
            self.remove_hooks()
            self._calibration_hooks.clear()
        finally:
            ctx, self._moe_calibration_ctx = self._moe_calibration_ctx, None
            if ctx is not None:
                ctx.__exit__(None, None, None)
                logger.info("Exited MoE all-experts calibration context")

    def _cleanup_calibration(self, failed_model=None):
        """Resource teardown, without success freeze or completeness checks."""
        try:
            self._remove_calibration_hooks()
        finally:
            self._hessians.clear()
            self._num_samples.clear()
            self._module_names.clear()
            self._hook_call_counts.clear()
            self._nonfinite_reported.clear()
            self._skipped_modules.clear()
            self._compressed_modules.clear()
            self._processed_modules.clear()
            self._active_modules.clear()
            self._active_module_set.clear()
            self._sequential_calibration = False
            if self._diagnostics is not None:
                diagnostics, self._diagnostics = self._diagnostics, None
                diagnostics.close()
            if failed_model is not None:
                from compressed_tensors.quantization import disable_quantization
                failed_model.apply(disable_quantization)
                for module in failed_model.modules():
                    self._remove_module_observers(module)

    def _log_hessian_memory_estimate(self):
        """Log estimated total Hessian memory so users can anticipate OOM/swap."""
        if not self._module_names:
            return

        total_elems = 0
        total_modules = len(self._module_names)
        progress_interval = max(1, total_modules // 10)
        start = time.perf_counter()
        logger.info(f"Estimating Hessian memory for {total_modules} modules")
        for index, module in enumerate(self._module_names, start=1):
            # Avoid touching module.weight here: offloaded weights would be
            # onloaded just for this estimate, which is extremely slow for
            # MoE models with tens of thousands of experts.
            if isinstance(module, torch.nn.Linear):
                in_features = module.in_features
            else:
                with _disable_onloading():
                    weight = getattr(module, "weight", None)
                    if weight is None or weight.dim() != 2:
                        continue
                    in_features = weight.shape[1]
            total_elems += in_features * in_features
            if index % progress_interval == 0 or index == total_modules:
                logger.info(
                    f"Hessian memory estimate: {index}/{total_modules} "
                    f"({time.perf_counter() - start:.1f}s elapsed)"
                )

        total_bytes = total_elems * 4  # Hessians are float32
        location = "CPU" if self.offload_hessians else "GPU"
        if total_bytes > (1 << 30):
            logger.info(
                f"Estimated Hessian memory: {total_bytes / (1 << 30):.1f} GiB "
                f"({location}). If this exceeds available "
                f"{'RAM' if self.offload_hessians else 'GPU memory'}, "
                "the process will OOM or thrash; consider offload_hessians="
                f"{not self.offload_hessians} or excluding some targets."
            )
        else:
            logger.info(
                f"Estimated Hessian memory: {total_bytes / (1 << 20):.1f} MiB ({location})"
            )

    def _assert_distributed_module_names(
        self, modules: List[torch.nn.Module]
    ) -> None:
        if not is_distributed():
            return

        import torch.distributed as dist

        module_names = [self._module_names[module] for module in modules]
        names_by_rank = [None] * dist.get_world_size()
        dist.all_gather_object(names_by_rank, module_names)
        if any(names != module_names for names in names_by_rank):
            raise RuntimeError(
                "Distributed GPTQ ranks selected different module sets before "
                "calibration. Ensure every rank uses the same model topology."
            )

    def _initialize_hessian(self, module: torch.nn.Module) -> None:
        if module in self._num_samples:
            return
        init_device = "cpu" if self.offload_hessians else get_execution_device(module)
        self._hessians[module] = make_empty_hessian(module, device=init_device)
        self._num_samples[module] = torch.zeros(
            tuple(), device=get_execution_device(module)
        )

    def _calibrate_hessian_hook(self, module: torch.nn.Module, args: tuple, **kwargs):
        """
        Forward pre-hook to accumulate Hessian from input activations.
        """
        if self._sequential_calibration and module not in self._active_module_set:
            return

        # Assume first argument is the input
        inp = args[0]

        if module not in self._num_samples:
            module_name = self._module_names[module]
            if module_name in self._processed_modules:
                return
            self._initialize_hessian(module)

        # On the first few calls per module, verify the incoming activations are
        # finite. This pinpoints where NaN/Inf enters the network (cheap: only
        # runs a few times per module instead of on every batch).
        calls = self._hook_call_counts.get(module, 0) + 1
        self._hook_call_counts[module] = calls
        if self._diagnostics is not None:
            # Diagnostics check every call and keep per-module input history.
            self._diagnostics.observe_input(module, inp)
        elif calls <= 3:
            self._check_input_finite(module, inp, calls)

        # Accumulate Hessian with optional offloading
        if module in self._hessians:
            with self._maybe_onload_hessian(module):
                self._hessians[module], self._num_samples[module] = accumulate_hessian(
                    inp,
                    module,
                    self._hessians[module],
                    self._num_samples[module],
                )

    def _check_input_finite(
        self, module: torch.nn.Module, inp: torch.Tensor, call_idx: int
    ):
        """Report non-finite input activations with enough detail to diagnose."""
        if module in self._nonfinite_reported:
            return

        finite = torch.isfinite(inp)
        if bool(finite.all()):
            return

        self._nonfinite_reported.add(module)
        name = self._module_names.get(module, type(module).__name__)
        num_nan = int(torch.isnan(inp).sum())
        num_inf = int(torch.isinf(inp).sum())
        sample_vals = inp[~finite].flatten()[:5].tolist()
        logger.error(
            f"Non-finite input activations at '{name}' (forward call #{call_idx}): "
            f"{num_nan} NaN / {num_inf} Inf elements, sample values: {sample_vals}. "
            "Inf points to dtype overflow (e.g. Qwen models must be loaded in "
            "bfloat16, not float16); NaN points to an all-masked softmax row "
            "(padding / attention_mask) or a broken attention kernel (try "
            "attn_implementation='eager')."
        )

    def compress_modules(self):
        """
        Quantize all modules that have been calibrated.
        Handles both single-GPU and distributed scenarios.
        """
        module_list = (
            list(self._active_modules)
            if self._sequential_calibration
            else list(self._num_samples)
        )
        if not is_distributed():
            self._compress_module_list(module_list)
            return

        # Distributed: use greedy bin packing for load balancing
        import torch.distributed as dist

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        self._assert_distributed_module_names(module_list)
        if not module_list:
            return

        self._prepare_distributed_qparams(module_list)

        # Assign modules to ranks based on Hessian size (load balancing)
        module_list, rank_to_modules, module_to_rank = greedy_bin_packing(
            module_list,
            world_size,
            item_weight_fn=lambda mod: self._hessians[mod].numel(),
        )

        # Reduce Hessians to assigned ranks
        self._reduce_hessian_to_target_rank(module_list, module_to_rank)

        # Quantize assigned modules
        self._compress_module_list(rank_to_modules[rank])

        # Sync processed and successfully compressed modules
        local_processed = list(self._processed_modules)
        all_processed = [None] * world_size
        dist.all_gather_object(all_processed, local_processed)
        self._processed_modules = set().union(*all_processed)

        local_compressed = list(self._compressed_modules)
        all_compressed = [None] * world_size
        dist.all_gather_object(all_compressed, local_compressed)
        self._compressed_modules = set().union(*all_compressed)

        local_skipped = dict(self._skipped_modules)
        all_skipped = [None] * world_size
        dist.all_gather_object(all_skipped, local_skipped)
        self._skipped_modules = {
            name: reason
            for skipped in all_skipped
            for name, reason in skipped.items()
        }

        # Broadcast quantization parameters to all ranks
        self._broadcast_qparams(module_list, module_to_rank)

    def _compress_module_list(self, module_list: List[torch.nn.Module]):
        """
        Quantize a list of modules using their accumulated Hessians.
        """
        rank = torch.distributed.get_rank() if is_distributed() else 0
        for module in module_list:
            name = self._module_names[module]
            num_samples = self._num_samples[module]
            quant_args = getattr_chain(module, "quantization_scheme.weights")

            logger.info(
                f"[rank {rank}] Quantizing {name} using {int(num_samples)} samples"
            )

            with (
                torch.no_grad(),
                align_module_device(module),
                self._maybe_onload_hessian(module),
            ):
                hessian = self._hessians.pop(module)
                num_samples = self._num_samples.pop(module)
                self._processed_modules.add(name)
                if not torch.isfinite(hessian).all() or int(num_samples) == 0:
                    reason = (
                        "non-finite values" if int(num_samples) > 0 else "no data"
                    )
                    self._skipped_modules[name] = reason
                    logger.error(
                        f"Skipping {name}: Hessian has {reason} "
                        f"({int(num_samples)} samples seen). This module will be "
                        "left unquantized."
                    )
                    if self._diagnostics is not None:
                        self._diagnostics.log_skip_postmortem(
                            module, name, hessian, int(num_samples)
                        )
                    continue
                hessian.div_(num_samples)

                if not hasattr(module, "weight_observer"):
                    raise RuntimeError(
                        f"GPTQ weight observer is missing for module '{name}'. "
                        "Initialize quantization before starting calibration."
                    )
                module.weight_observer(module.weight)

                loss, q_param_dict = quantize_weight(
                    module=module,
                    quant_args=quant_args,
                    hessian=hessian,
                    blocksize=self.block_size,
                    percdamp=self.dampening_frac or 0.0,
                    copy_hessian=False,
                )
                self._compressed_modules.add(name)
                logger.info(
                    f"[rank {rank}] Quantization loss for {name}: {loss:.6f}"
                )

            for attr, val in q_param_dict.items():
                update_offload_parameter(module, attr, val)

    def _prepare_distributed_qparams(self, module_list: List[torch.nn.Module]):
        """Create matching qparam buffers before distributed broadcasts."""
        for module in module_list:
            quant_args = getattr_chain(module, "quantization_scheme.weights")
            observer = getattr(module, "weight_observer", None)
            if observer is None:
                name = self._module_names[module]
                raise RuntimeError(
                    f"GPTQ weight observer is missing for module '{name}'. "
                    "Initialize quantization before starting calibration."
                )

            with torch.no_grad(), align_module_device(module):
                observer(module.weight)
                for name, value in observer.get_qparams().items():
                    if value is None:
                        continue
                    if name == "zero_point" and quant_args.symmetric:
                        continue
                    update_offload_parameter(module, f"weight_{name}", value)

                if quant_args.actorder == ActivationOrdering.GROUP:
                    update_offload_parameter(
                        module,
                        "weight_g_idx",
                        torch.zeros(
                            module.weight.shape[1],
                            device=module.weight.device,
                            dtype=torch.int,
                        ),
                    )

    def _reduce_hessian_to_target_rank(
        self,
        module_list: List[torch.nn.Module],
        module_to_rank: Dict[torch.nn.Module, int],
    ):
        """
        Reduce Hessians to their assigned ranks in distributed mode.
        """
        import torch.distributed as dist

        rank = dist.get_rank()

        for module in module_list:
            target_rank = module_to_rank[module]
            with self._maybe_onload_hessian(module):
                pending_comms = [
                    dist.reduce(
                        self._hessians[module],
                        op=dist.ReduceOp.SUM,
                        dst=target_rank,
                        async_op=True,
                    ),
                    dist.reduce(
                        self._num_samples[module],
                        op=dist.ReduceOp.SUM,
                        dst=target_rank,
                        async_op=True,
                    ),
                ]
                wait_for_comms(pending_comms)
                if rank != target_rank:
                    self._hessians.pop(module, None)
                    self._num_samples.pop(module, None)

    def _broadcast_qparams(
        self,
        module_list: List[torch.nn.Module],
        module_to_rank: Dict[torch.nn.Module, int],
    ):
        """Broadcast quantization parameters from owning ranks to all ranks."""
        broadcast_qparams_and_cleanup(module_list, module_to_rank, _GPTQ_Q_PARAMS)

    @contextlib.contextmanager
    def _maybe_onload_hessian(self, module: torch.nn.Module):
        """
        Context manager for Hessian offloading.
        """
        if self.offload_hessians:
            device = get_execution_device(module)
            self._hessians[module] = self._hessians[module].to(device=device)

        yield

        if self.offload_hessians:
            if module in self._hessians:  # May have been deleted
                self._hessians[module] = self._hessians[module].to(device="cpu")

