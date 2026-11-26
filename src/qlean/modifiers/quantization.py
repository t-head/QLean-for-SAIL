# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
# Ported from: llmcompressor/modifiers/quantization/quantization/mixin.py

from typing import Any, Dict, List, Optional, Set, Union

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
    QuantizationStrategy,
    apply_quantization_config,
    disable_quantization,
    enable_quantization,
    is_preset_scheme,
    preset_name_to_scheme,
)
from pydantic import Field, PrivateAttr
from torch.utils.hooks import RemovableHandle

from qlean.utils.hooks import HooksMixin
from compressed_tensors.utils import match_named_modules

import logging

logger = logging.getLogger(__name__)

__all__ = ["QuantizationCalibrationMixin"]

class QuantizationCalibrationMixin(HooksMixin):
    """
    Mixin which enables a Modifier to act as a quantization config,
    attaching observers, calibration hooks, and compression wrappers.

    Lifecycle:
    - on_initialize: initialize_quantization()
        - Attach schemes to modules
        - Attach observers to modules
        - Disable quantization until calibration starts
    - on_start (calibration_epoch_start): start_calibration()
        - Attach calibration hooks
        - Apply calibration status
        - Enable quantization during calibration
    - on_end (calibration_epoch_end): end_calibration()
        - Remove calibration hooks
        - Apply freeze status
        - Keep quantization enabled for future steps

    :param config_groups: dictionary specifying quantization schemes
    :param targets: list of layer names to quantize
    :param ignore: list of modules to not quantize
    :param scheme: single quantization scheme or preset name
    :param kv_cache_scheme: optional KV cache quantization
    """

    # Quantization configuration fields
    config_groups: Optional[Dict[str, QuantizationScheme]] = Field(default=None)
    targets: Union[str, List[str]] = Field(default_factory=lambda: ["Linear"])
    ignore: List[str] = Field(default_factory=list)
    scheme: Optional[Union[str, Dict[str, Any]]] = Field(default=None)
    kv_cache_scheme: Optional[QuantizationArgs] = Field(default=None)
    bypass_divisibility_checks: bool = Field(default=False)

    # Resolved configuration
    _resolved_config: Optional[QuantizationConfig] = PrivateAttr(default=None)
    _resolved_targets: Set[str] = PrivateAttr(default_factory=set)
    _calibration_hooks: Set[RemovableHandle] = PrivateAttr(default_factory=set)

    @staticmethod
    def has_config(modifier: "QuantizationCalibrationMixin") -> bool:
        """Check if quantization configuration is provided."""
        return (
            modifier.config_groups is not None
            or modifier.scheme is not None
        )

    def initialize_quantization(self, model: torch.nn.Module):
        """
        Initialize quantization on the model.

        - Resolve quantization config
        - Apply config to model (attach schemes to modules)
        - Initialize observers
        - Disable quantization until calibration starts
        """
        if not QuantizationCalibrationMixin.has_config(self):
            raise ValueError(
                "Quantization configuration must be provided via "
                "config_groups or scheme"
            )

        # Resolve first: invalid configurations must not mutate any forwards.
        config = self.resolve_quantization_config()
        self._resolved_config = config

        # CT offload unwrapping can mutate a target forward even when it raises.
        forwards = []
        try:
            self._unwrap_partial_forwards(model, config, forwards)
            apply_quantization_config(model, config)
            self._validate_group_size_divisibility(model)

            for name, module in model.named_modules():
                if hasattr(module, "quantization_scheme"):
                    self._initialize_observers(module)

            model.apply(disable_quantization)
        except BaseException:
            for module, forward, offload_forward in reversed(forwards):
                module.forward = forward
                if offload_forward is not None:
                    module._original_forward_func = offload_forward
            raise

        logger.info(
            f"Initialized quantization with {len(self._resolved_targets)} target modules"
        )

    @staticmethod
    def _unwrap_partial_forwards(model, config, forwards):
        """Adapt only CT-supported targets, inside CT's offload wrapper."""
        import functools
        from compressed_tensors.offload import unwrap_offload_forward
        from compressed_tensors.linear.compressed_linear import CompressedLinear

        def make_forward(partial_forward):
            def forward(self, *args, **kwargs):
                return partial_forward(*args, **kwargs)
            return forward

        targets = {
            target for scheme in config.config_groups.values()
            for target in scheme.targets
        }
        for name, module in match_named_modules(model, targets, config.ignore):
            if not isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                continue
            if isinstance(module, CompressedLinear):
                continue
            forwards.append((
                module, module.forward,
                getattr(module, "_original_forward_func", None),
            ))
            with unwrap_offload_forward(module):
                if isinstance(module.forward, functools.partial):
                    module.forward = make_forward(module.forward).__get__(module)

    def start_calibration(self, model: torch.nn.Module):
        """
        Start calibration phase.

        - Attach calibration hooks
        - Apply calibration status

        Quantization stays DISABLED during calibration (matches
        llm-compressor): enabling the fake-quant forward here would run the
        model with uninitialized weight scales (GPTQ computes them only after
        calibration), corrupting activations with NaN/Inf and poisoning every
        downstream Hessian.
        """
        # Attach calibration hooks
        for name, module in model.named_modules():
            if hasattr(module, "quantization_scheme"):
                hooks = self._initialize_hooks(module)
                # Store hooks for later removal
                self._calibration_hooks.update(hooks)

        # Apply calibration status
        model.apply(self._apply_calibration_status)

        logger.info("Started calibration")

    def end_calibration(self, model: torch.nn.Module):
        """
        End calibration phase.

        - Remove calibration hooks
        - Apply freeze status
        - Enable quantization for future steps (after scales have been set)
        """
        # Remove calibration hooks
        self.remove_hooks(self._calibration_hooks)
        self._calibration_hooks.clear()

        # Apply freeze status
        model.apply(self._freeze_module_quantization)

        # Enable the quantized forward now that quantization parameters
        # (weight_scale / weight_zero_point) exist
        model.apply(enable_quantization)

        logger.info("Ended calibration")

    def resolve_quantization_config(self) -> QuantizationConfig:
        """
        Resolve quantization configuration from fields.

        Handles:
        - Preset schemes (e.g., "W8A8")
        - Scheme dictionaries
        - Config groups
        - KV cache scheme
        """
        config_groups = self.config_groups
        targets = [self.targets] if isinstance(self.targets, str) else self.targets

        # Handle preset scheme
        if self.scheme is not None:
            if isinstance(self.scheme, str) and is_preset_scheme(self.scheme):
                # Convert preset to scheme
                scheme = preset_name_to_scheme(self.scheme, targets)
                config_groups = {"preset": scheme}
            elif isinstance(self.scheme, str):
                raise ValueError(
                    f"Unknown quantization scheme {self.scheme!r}; not a recognized preset name"
                )
            elif isinstance(self.scheme, dict):
                if not self.scheme:
                    raise ValueError(
                        "Quantization scheme dict must not be empty; provide a "
                        "preset name (e.g. {'W8A8': ['Linear']}) or scheme fields"
                    )
                # Convert dict to QuantizationScheme
                if is_preset_scheme(list(self.scheme.keys())[0]):
                    # Format: {"W8A8": ["Linear"]}
                    preset_name = list(self.scheme.keys())[0]
                    preset_targets = self.scheme[preset_name]
                    if isinstance(preset_targets, str):
                        preset_targets = [preset_targets]
                    scheme = preset_name_to_scheme(preset_name, preset_targets)
                    config_groups = {"preset": scheme}
                else:
                    # Regular scheme dict
                    scheme = QuantizationScheme(**self.scheme)
                    config_groups = {"default": scheme}

        # Set targets if not in config_groups
        if config_groups is not None:
            for scheme in config_groups.values():
                if not hasattr(scheme, "targets") or scheme.targets is None:
                    scheme.targets = targets

        # Handle KV cache scheme
        kv_cache_scheme = self.kv_cache_scheme

        # Resolve targets
        self._resolved_targets = set()
        if config_groups:
            for scheme in config_groups.values():
                targets = getattr(scheme, "targets", [])
                if isinstance(targets, str):
                    targets = [targets]
                self._resolved_targets.update(targets)

        return QuantizationConfig(
            config_groups=config_groups or {},
            kv_cache_scheme=kv_cache_scheme,
            quantization_status=QuantizationStatus.INITIALIZED,
            ignore=self.ignore,
        )

    def _initialize_observers(self, module: torch.nn.Module):
        """Initialize observers for a module based on its quantization scheme."""
        if not hasattr(module, "quantization_scheme"):
            return

        scheme: QuantizationScheme = module.quantization_scheme

        # Weight observers
        if scheme.weights is not None:
            self._initialize_weight_observer(module, scheme.weights)

        # Input activation observers
        if scheme.input_activations is not None:
            if not scheme.input_activations.dynamic:
                self._initialize_activation_observer(module, "input", scheme.input_activations)

        # Output activation observers
        if scheme.output_activations is not None:
            if not scheme.output_activations.dynamic:
                self._initialize_activation_observer(module, "output", scheme.output_activations)

    def _validate_group_size_divisibility(self, model: torch.nn.Module) -> None:
        if self.bypass_divisibility_checks:
            return

        invalid_modules = []
        for name, module in match_named_modules(
            model, self._resolved_targets, self.ignore
        ):
            scheme = getattr(module, "quantization_scheme", None)
            weight_args = getattr(scheme, "weights", None)
            if (
                weight_args is None
                or weight_args.strategy
                not in (QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP)
                or weight_args.group_size is None
            ):
                continue

            if isinstance(module, torch.nn.Linear):
                columns = int(module.in_features)
            else:
                # Reading an offloaded weight would onload the whole tensor.
                from compressed_tensors.offload import disable_onloading

                with disable_onloading():
                    weight = getattr(module, "weight", None)
                    if weight is None:
                        continue
                    columns = int(weight.shape[-1])
            group_size = int(weight_args.group_size)
            if columns % group_size != 0:
                invalid_modules.append((name, columns, group_size))

        if not invalid_modules:
            return

        details = "\n".join(
            f"  - {name} (columns={columns}, group_size={group_size})"
            for name, columns, group_size in invalid_modules
        )
        raise ValueError(
            "The following layers have weight column counts not divisible by "
            "group_size. Group and tensor-group quantization require "
            "columns % group_size == 0. Add these layer names to `ignore`, "
            "or set bypass_divisibility_checks=True only when the target "
            f"runtime supports non-divisible groups.\n\n{details}"
        )

    def _initialize_weight_observer(
        self, module: torch.nn.Module, quant_args: QuantizationArgs
    ):
        """Initialize a qlean weight observer for static GPTQ qparams."""
        from qlean.observers import Observer

        observer_name = quant_args.observer or "memoryless_minmax"
        observer = Observer.load_from_registry(
            observer_name, base_name="weight", args=quant_args
        )
        module.register_module("weight_observer", observer)
        observer.attach(module)

    def _initialize_activation_observer(
        self, module: torch.nn.Module, base_name: str, quant_args: QuantizationArgs
    ):
        """Initialize activation observer for a module."""
        try:
            from compressed_tensors.quantization.observer import Observer
        except ModuleNotFoundError as error:
            raise ValueError(
                "Static activation calibration requires an available activation "
                "observer; use weight-only GPTQ when this dependency is unavailable"
            ) from error

        observer_name = quant_args.observer or "minmax"
        observer = Observer.load_from_registry(
            observer_name, base_name=base_name, args=quant_args
        )
        module.register_module(f"{base_name}_observer", observer)
        observer.attach(module)

    def _initialize_hooks(self, module: torch.nn.Module) -> Set[RemovableHandle]:
        """Initialize calibration hooks for a module."""
        hooks = set()

        if not hasattr(module, "quantization_scheme"):
            return hooks

        scheme: QuantizationScheme = module.quantization_scheme

        # Input calibration hook
        if scheme.input_activations is not None:
            if not scheme.input_activations.dynamic:
                handle = self.register_hook(
                    module, self._calibrate_input_hook, "forward_pre"
                )
                hooks.add(handle)

        # Output calibration hook
        if scheme.output_activations is not None:
            if not scheme.output_activations.dynamic:
                handle = self.register_hook(
                    module, self._calibrate_output_hook, "forward"
                )
                hooks.add(handle)

        return hooks

    @staticmethod
    def _calibrate_input_hook(module: torch.nn.Module, args: tuple):
        """Hook to calibrate input activations."""
        if hasattr(module, "input_observer"):
            inp = args[0] if isinstance(args, tuple) else args
            module.input_observer(inp)

    @staticmethod
    def _calibrate_output_hook(
        module: torch.nn.Module, args: tuple, output: torch.Tensor
    ):
        """Hook to calibrate output activations."""
        if hasattr(module, "output_observer"):
            module.output_observer(output)

    @staticmethod
    def _apply_calibration_status(module: torch.nn.Module):
        """Apply calibration status to a module."""
        if hasattr(module, "quantization_scheme"):
            module.quantization_status = QuantizationStatus.CALIBRATION

    @staticmethod
    def _freeze_module_quantization(module: torch.nn.Module):
        """Freeze quantization for a module by removing observers."""
        if not hasattr(module, "quantization_scheme"):
            return

        if getattr(module, "quantization_status", None) == QuantizationStatus.FROZEN:
            return

        QuantizationCalibrationMixin._remove_module_observers(module)
        module.quantization_status = QuantizationStatus.FROZEN

    @staticmethod
    def _remove_module_observers(module: torch.nn.Module):
        """Detach observers without changing quantization lifecycle status."""
        for name in ("input", "weight", "output"):
            obs_name = f"{name}_observer"
            if hasattr(module, obs_name):
                observer = getattr(module, obs_name)
                if hasattr(observer, "detach"):
                    observer.detach(module)
                delattr(module, obs_name)
