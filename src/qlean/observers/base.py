# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from copy import copy
from typing import Optional

import torch
from compressed_tensors import InternalModule
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from compressed_tensors.quantization.utils import calculate_qparams, generate_gparam
from compressed_tensors.registry.registry import RegistryMixin

from qlean.observers.helpers import flatten_for_calibration

__all__ = ["Observer"]


class Observer(InternalModule, RegistryMixin):
    _stats_attrs = ("min_vals", "max_vals")

    def __init__(self, base_name: str, args: QuantizationArgs, **observer_kwargs):
        super().__init__()
        self.base_name = base_name
        # args is the shared scheme.weights of a config group, so keep a private
        # copy. observer_kwargs needs its own copy too: a shallow copy of args
        # would still alias the caller's dict.
        self.args = copy(args)
        self.args.observer_kwargs = dict(self.args.observer_kwargs or {})
        self.args.observer_kwargs.update(observer_kwargs)

    @property
    def has_statistics(self) -> bool:
        return all(hasattr(self, attr) for attr in self._stats_attrs)

    @property
    def is_weight_observer(self) -> bool:
        return self.base_name == "weight"

    @abstractmethod
    def update_statistics_from_observed(self, observed: torch.Tensor) -> None:
        raise NotImplementedError()

    @torch.no_grad()
    def get_qparams(self) -> dict[str, Optional[torch.Tensor]]:
        if not self.has_statistics:
            raise RuntimeError("No statistics available. Call observer(value) first.")

        global_scale = None
        if self.args.strategy == QuantizationStrategy.TENSOR_GROUP:
            absmax = torch.maximum(self.max_vals.max(), -self.min_vals.min())
            global_scale = generate_gparam(-absmax.reshape(1), absmax.reshape(1))

        scale, zero_point = calculate_qparams(
            min_vals=self.min_vals,
            max_vals=self.max_vals,
            quantization_args=self.args,
            global_scale=global_scale,
        )
        self.delete_statistics()
        return {
            "scale": scale,
            "zero_point": zero_point,
            "global_scale": global_scale,
        }

    @torch.no_grad()
    def forward(self, observed: torch.Tensor) -> "Observer":
        if observed.numel() == 0:
            return self
        if self.is_weight_observer and self.has_statistics:
            return self

        flattened = flatten_for_calibration(observed, self.base_name, self.args)
        self.update_statistics_from_observed(flattened)
        return self

    def delete_statistics(self) -> None:
        for attr in self._stats_attrs:
            if hasattr(self, attr):
                delattr(self, attr)

    def attach(self, module: torch.nn.Module) -> None:
        pass

    def detach(self, module: torch.nn.Module) -> None:
        pass
