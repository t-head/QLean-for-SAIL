# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
# Linearized per-expert MLP modules ported from llmcompressor/modeling/moe/linear_experts.py.

from __future__ import annotations

from abc import ABC, abstractmethod
from importlib import import_module
from typing import Any, Callable, ClassVar

import torch
from compressed_tensors.offload import get_cache_init_kwargs, offload_module
from transformers import PreTrainedConfig
from transformers.activations import ACT2FN

from qlean.utils.dev import skip_weights_initialize

from .context import get_calibrate_all_experts_flag
from .helpers import (
    FusedExpertsProtocol,
    MoEConfig,
    get_use_experts_implementation_args,
)

try:
    from transformers.integrations.moe import _default_apply_gate
except ImportError:

    def _default_apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        # Injected as a class attribute, so it is retrieved as a bound method and
        # must take self, matching transformers' _default_apply_gate convention.
        gate, up = gate_up.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up


class ExpertMLP(torch.nn.Module, ABC):
    """Abstract base for a single de-fused expert MLP."""

    @abstractmethod
    def copy_from_experts_module(self, experts: FusedExpertsProtocol, index: int):
        raise NotImplementedError()


class ExpertMLPWithGate(ExpertMLP):
    """Single gated expert (e.g. SwiGLU) with separate gate/up/down projections."""

    up_proj: torch.nn.Linear
    gate_proj: torch.nn.Linear
    down_proj: torch.nn.Linear
    _apply_gate: Callable[[torch.Tensor], torch.Tensor]

    def __init__(
        self,
        hidden_dim: int,
        intermediate_size: int,
        mlp_bias: bool,
        _apply_gate: Callable[[torch.Tensor], torch.Tensor],
        dtype: torch.dtype,
    ):
        super().__init__()
        self.intermediate_size = intermediate_size
        self.up_proj = torch.nn.Linear(
            hidden_dim, intermediate_size, bias=mlp_bias, dtype=dtype
        )
        self.gate_proj = torch.nn.Linear(
            hidden_dim, intermediate_size, bias=mlp_bias, dtype=dtype
        )
        self.down_proj = torch.nn.Linear(
            intermediate_size, hidden_dim, bias=mlp_bias, dtype=dtype
        )
        self._apply_gate = _apply_gate

    def copy_from_experts_module(self, experts: FusedExpertsProtocol, index: int):
        # load weights
        if not experts.is_transposed:
            gate_weight = experts.gate_up_proj[index, : self.intermediate_size]
            up_weight = experts.gate_up_proj[index, self.intermediate_size :]
            down_weight = experts.down_proj[index]
        else:
            gate_weight = experts.gate_up_proj[index, :, : self.intermediate_size].T
            up_weight = experts.gate_up_proj[index, :, self.intermediate_size :].T
            down_weight = experts.down_proj[index].T

        self.gate_proj.weight.copy_(gate_weight)
        self.up_proj.weight.copy_(up_weight)
        self.down_proj.weight.copy_(down_weight)

        # load biases
        if experts.has_bias:
            gate_bias = experts.gate_up_proj_bias[index, : self.intermediate_size]
            up_bias = experts.gate_up_proj_bias[index, self.intermediate_size :]
            down_bias = experts.down_proj_bias[index]

            self.gate_proj.bias.copy_(gate_bias)
            self.up_proj.bias.copy_(up_bias)
            self.down_proj.bias.copy_(down_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self._apply_gate(
                torch.cat(
                    [self.gate_proj(hidden_states), self.up_proj(hidden_states)], dim=-1
                )
            )
        )


class ExpertMLPWithoutGate(ExpertMLP):
    """Single non-gated expert with up/down projections."""

    up_proj: torch.nn.Linear
    down_proj: torch.nn.Linear
    act_fn: torch.nn.Module

    def __init__(
        self,
        hidden_dim: int,
        intermediate_size: int,
        mlp_bias: bool,
        act_fn: torch.nn.Module,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.intermediate_size = intermediate_size
        self.up_proj = torch.nn.Linear(
            hidden_dim, intermediate_size, bias=mlp_bias, dtype=dtype
        )
        self.down_proj = torch.nn.Linear(
            intermediate_size, hidden_dim, bias=mlp_bias, dtype=dtype
        )
        self.act_fn = act_fn

    def copy_from_experts_module(self, experts: FusedExpertsProtocol, index: int):
        # load weights
        if not experts.is_transposed:
            up_weight = experts.up_proj[index]
            down_weight = experts.down_proj[index]
        else:
            up_weight = experts.up_proj[index].T
            down_weight = experts.down_proj[index].T

        self.up_proj.weight.copy_(up_weight)
        self.down_proj.weight.copy_(down_weight)

        # load biases
        if experts.has_bias:
            up_bias = experts.up_proj_bias[index]
            down_bias = experts.down_proj_bias[index]

            self.up_proj.bias.copy_(up_bias)
            self.down_proj.bias.copy_(down_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.up_proj(hidden_states)))


class LinearExperts2D(torch.nn.ModuleList):
    """Container that replaces a fused experts module with per-expert ``nn.Linear``s.

    The class can be constructed in two ways:

    1. Automatically from a ``FusedExpertsProtocol`` module via
       :meth:`from_experts_module`. This copies weights and offloading settings.
    2. Registered architecture-specific subclasses (e.g. Granite/Llama4) that know
       how to copy weights from their particular fused expert layout.
    """

    is_concatenated: ClassVar[bool]
    is_transposed: ClassVar[bool]
    has_bias: ClassVar[bool]
    has_gate: ClassVar[bool]
    _apply_gate: ClassVar[Callable[[torch.Tensor], torch.Tensor]]

    num_experts: int
    intermediate_size: int

    # custom model definitions
    _registry: ClassVar[dict[type[torch.nn.Module], type["LinearExperts2D"]]] = dict()

    @classmethod
    def get_registration(
        cls, key: type[torch.nn.Module], default: Any = None
    ) -> type["LinearExperts2D"]:
        if key in cls._registry:
            return cls._registry[key]

        if (
            key.__module__ == "transformers.models.granitemoe.modeling_granitemoe"
            and key.__name__ == "GraniteMoeParallelExperts"
        ):
            import_module(".granitemoe", __package__)
        elif (
            key.__module__ == "transformers.models.llama4.modeling_llama4"
            and key.__name__ == "Llama4TextExperts"
        ):
            import_module(".llama4", __package__)

        return cls._registry.get(key, default)

    @classmethod
    def get_linear_experts_cls(
        cls, experts_cls: type[FusedExpertsProtocol]
    ) -> type["LinearExperts2D"]:
        if linear_experts_cls := cls.get_registration(experts_cls):
            return linear_experts_cls

        experts_cls_args = get_use_experts_implementation_args(experts_cls)
        if experts_cls_args is None:
            raise ValueError(
                "Cannot create linear experts class from a class which does not have "
                "the `use_experts_implementation` argument."
            )

        experts_cls_args["_apply_gate"] = getattr(
            experts_cls, "_apply_gate", _default_apply_gate
        )

        # reuse existing classes to avoid creating excessive types
        linear_experts_cls = type("LinearExperts2D", (cls,), experts_cls_args)
        cls._registry[experts_cls] = linear_experts_cls
        return linear_experts_cls

    @classmethod
    @torch.no_grad()
    def from_experts_module(
        cls,
        experts: FusedExpertsProtocol,
        config: PreTrainedConfig,
    ):
        with skip_weights_initialize():
            self = cls(config, initialize_experts=False)

        # Apply the source offload policy before adding experts. Each expert can
        # then be copied and offloaded independently instead of retaining every
        # replacement weight until the whole MoE block is converted.
        offload_kwargs = get_cache_init_kwargs(experts)
        offload_module(self, **offload_kwargs)

        for index in range(self.num_experts):
            with skip_weights_initialize():
                expert: ExpertMLP = self._new_expert()
            expert.copy_from_experts_module(experts, index)
            self.append(expert)
            for module in expert.modules():
                offload_module(module, **offload_kwargs)

        self._finish_expert_initialization()
        return self

    def __init__(
        self,
        config: PreTrainedConfig,
        *args,
        initialize_experts: bool = True,
        **kwargs,
    ):
        moe_config = MoEConfig.from_config(config)

        # store num_experts before appending `act_fn` to module list
        self.num_experts = moe_config.num_experts
        self.intermediate_size = moe_config.intermediate_size
        act_fn: torch.nn.Module = ACT2FN[moe_config.hidden_act]

        self._expert_cls = ExpertMLPWithGate if self.has_gate else ExpertMLPWithoutGate
        post_up_fn = self._apply_gate if self.has_gate else act_fn.forward
        self._expert_args = (
            moe_config.hidden_dim,
            moe_config.intermediate_size,
            moe_config.use_bias,
            post_up_fn,
            moe_config.dtype,
        )
        super().__init__()

        object.__setattr__(self, "_pending_act_fn", act_fn)
        self.alpha = moe_config.alpha
        self.limit = moe_config.limit
        # borrowed `_apply_gate` implementations read this name, not `limit`
        self.swiglu_limit = moe_config.limit

        if initialize_experts:
            for _ in range(self.num_experts):
                self.append(self._new_expert())
            self._finish_expert_initialization()

        # Direct checkpoint loading provides every expert parameter.
        self._is_hf_initialized = True

    def _new_expert(self) -> ExpertMLP:
        return self._expert_cls(*self._expert_args)

    def _finish_expert_initialization(self):
        self.act_fn = self._pending_act_fn
        del self._pending_act_fn

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Mixtral/Qwen-style token dispatch forward.

        Args:
            hidden_states: ``(batch * seq_len, hidden_size)``
            top_k_index: ``(batch * seq_len, top_k)`` expert indices
            top_k_weights: ``(batch * seq_len, top_k)`` expert weights

        Returns:
            Accumulated expert outputs of shape ``hidden_states.shape``.
        """
        final_hidden_states = torch.zeros_like(hidden_states)

        # create tokens mask
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)

        for expert_index in range(self.num_experts):
            # select tokens for this expert
            top_k_pos, token_indices = torch.where(expert_mask[expert_index])

            # apply expert
            expert = self[expert_index]
            if get_calibrate_all_experts_flag():
                expert_output = expert(hidden_states)[token_indices]
            else:
                expert_output = expert(hidden_states[token_indices])

            # apply weighting to outputs
            expert_weights = top_k_weights[token_indices, top_k_pos, None]
            weighted_output = expert_output * expert_weights

            # accumulate using index_add_ to match eager implementation exactly
            final_hidden_states.index_add_(
                0, token_indices, weighted_output.to(final_hidden_states.dtype)
            )

        return final_hidden_states
