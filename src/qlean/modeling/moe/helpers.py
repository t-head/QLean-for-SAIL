# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
# MoE helper utilities ported from llmcompressor/modeling/moe/helpers.py.

from __future__ import annotations

import ast
import functools
import importlib
import inspect
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Protocol, _ProtocolMeta

import torch
from loguru import logger
from transformers import PreTrainedConfig


_MISSING = object()


class _TorchModuleProtocolMeta(_ProtocolMeta):
    """Metaclass enabling custom ``isinstance`` checks for module protocols."""

    def __instancecheck__(cls, obj: object) -> bool:
        return isinstance(obj, torch.nn.Module) and cls.__validate__(obj)


class TorchModuleProtocol(Protocol, metaclass=_TorchModuleProtocolMeta):
    """Base protocol with structural ``isinstance`` support for ``nn.Module``s."""

    @classmethod
    def __validate__(cls, obj: object) -> bool:
        return True


class FusedExpertsProtocol(TorchModuleProtocol):
    """Protocol describing a fused/batched experts module.

    A fused experts module stacks all expert weights along dimension 0 and exposes
    parameters such as ``gate_up_proj`` / ``up_proj`` / ``down_proj`` (and optional
    biases). The class-level ``__validate__`` method is used to detect such modules
    at runtime via structural ``isinstance`` checks.
    """

    config: PreTrainedConfig
    has_gate: bool
    has_bias: bool
    is_transposed: bool
    _apply_gate: ClassVar[Callable]

    down_proj: torch.nn.Parameter
    act_fn: torch.nn.Module

    # gate proj may or may not exist
    gate_up_proj: torch.nn.Parameter
    up_proj: torch.nn.Parameter

    # biases may or may not exist
    up_proj_bias: torch.nn.Parameter
    gate_up_proj_bias: torch.nn.Parameter
    down_proj_bias: torch.nn.Parameter

    @classmethod
    def __validate__(cls, object: object) -> bool:
        return (
            isinstance(getattr(object, "down_proj", None), torch.nn.Parameter)
            and (
                isinstance(getattr(object, "up_proj", None), torch.nn.Parameter)
                or isinstance(
                    getattr(object, "gate_up_proj", None), torch.nn.Parameter
                )
            )
            and get_use_experts_implementation_args(object.__class__) is not None
        )


def get_use_experts_implementation_args(
    experts_cls: type,
) -> dict[str, bool] | None:
    """Get keyword args passed to ``@use_experts_implementation`` on *experts_cls*.

    Transformers v5 MoE expert classes are decorated with ``@use_experts_implementation``
    which encodes the weight layout (concatenated, transposed, bias, gate). This
    function inspects the class source to recover those arguments.

    :param experts_cls: expert class decorated with ``@use_experts_implementation``
    :return: keyword arguments to the decorator, or ``None`` if not decorated
    """
    cached = _cached_use_experts_implementation_args(experts_cls)
    # Callers mutate the result (LinearExperts2D injects `_apply_gate`), so hand
    # out a fresh dict instead of the cached object.
    return None if cached is None else dict(cached)


@functools.lru_cache(maxsize=None)
def _cached_use_experts_implementation_args(
    experts_cls: type,
) -> tuple[tuple[str, bool], ...] | None:
    """Parse the decorator once per class; ``isinstance`` probes hit this per module."""
    default_args = {
        "is_concatenated": True,
        "is_transposed": False,
        "has_bias": False,
        "has_gate": True,
    }

    try:
        source = inspect.getsource(experts_cls)
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError) as error:
        logger.warning(
            "Could not inspect source of {} to find @use_experts_implementation: {}",
            experts_cls.__name__,
            error,
        )
        return None

    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        for decorator in node.decorator_list:
            # Handle @use_experts_implementation (no arguments)
            if (
                isinstance(decorator, ast.Name)
                and decorator.id == "use_experts_implementation"
            ):
                return tuple(default_args.items())

            # Handle @use_experts_implementation(...) with arguments
            if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
                if decorator.func.id == "use_experts_implementation":
                    args = default_args.copy()

                    for keyword in decorator.keywords:
                        if keyword.arg in args:
                            if isinstance(keyword.value, ast.Constant):
                                args[keyword.arg] = keyword.value.value
                            elif isinstance(keyword.value, ast.Name):
                                # Handle True/False as Name nodes (older Python AST).
                                # Any other Name falls through and keeps the default;
                                # removing this branch would turn that into an error.
                                if keyword.value.id in ("True", "False"):
                                    args[keyword.arg] = keyword.value.id == "True"
                            else:
                                raise ValueError(f"Invalid keyword {keyword.value}")

                    return tuple(args.items())

    logger.warning(
        "{} does not appear to be decorated with @use_experts_implementation; "
        "it will not be treated as a fused experts module",
        experts_cls.__name__,
    )
    return None


@dataclass
class MoEConfig:
    """Normalized MoE configuration extracted from a ``PreTrainedConfig``."""

    num_experts: int
    num_experts_per_tok: int
    hidden_dim: int
    intermediate_size: int
    use_bias: bool
    hidden_act: str
    limit: float
    alpha: float | None
    dtype: torch.dtype

    @classmethod
    def from_config(cls, config: PreTrainedConfig):
        ret = cls(
            num_experts=_getattr_fallbacks(
                config,
                [
                    "num_local_experts",
                    "moe_num_experts",
                    "num_experts",
                    "n_routed_experts",
                ],
            ),
            num_experts_per_tok=_getattr_fallbacks(
                config, ["top_k_experts", "num_experts_per_tok"]
            ),
            hidden_dim=_getattr_fallbacks(config, ["hidden_size", "hidden_dim"]),
            intermediate_size=_getattr_fallbacks(
                config,
                ["moe_intermediate_size", "expert_intermediate_size", "intermediate_size"],
            ),
            use_bias=_getattr_fallbacks(config, ["use_bias", "mlp_bias"], False),
            hidden_act=_getattr_fallbacks(
                config, ["hidden_act", "hidden_activation", "mlp_hidden_act"], None
            ),
            limit=_getattr_fallbacks(config, ["swiglu_limit"], None),
            alpha=None,
            dtype=_getattr_fallbacks(config, ["dtype"]),
        )

        model_type = getattr(config, "model_type", None)
        if model_type in ("gpt_oss", "openai_privacy_filter"):
            ret.use_bias = True
            ret.limit = 7.0
            ret.alpha = 1.702
            ret.hidden_act = "sigmoid"
        elif model_type == "lfm2_moe":
            ret.hidden_act = "silu"

        return ret


def import_or_none(paths: str | list[str]) -> object | None:
    """Import the first available attribute from a list of dotted paths."""
    if isinstance(paths, str):
        paths = [paths]

    for path in paths:
        try:
            module_path, attr_name = path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            return getattr(module, attr_name)
        except (ImportError, AttributeError):
            pass

    return None


def _getattr_fallbacks(
    target: object, attrs: list[str], default: Any = _MISSING
) -> Any:
    for attr in attrs:
        if hasattr(target, attr):
            return getattr(target, attr)

    if default is not _MISSING:
        return default

    raise AttributeError(f"{target} does not have any of {attrs} attributes")
