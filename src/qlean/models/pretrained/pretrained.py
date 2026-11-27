# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

from qlean.models import ModelFactory, Model
from qlean.args import ModelArguments

logger = logging.getLogger(__name__)


@ModelFactory.register("pretrained")
class PretrainedModel(Model):
    """
    Wrapper for a user-provided pretrained PyTorch model.

    Used for calibration-based quantization methods (e.g. GPTQ) that need
    to run forward passes on the actual nn.Module.
    """

    def __init__(self, model_args: ModelArguments, model: Any = None):
        super().__init__(model_args)
        self.model_ = model

    def load_model(self, model_args: ModelArguments):
        if self.model_ is None:
            logger.warning("PretrainedModel.load_model called but no model was provided")

    def forward(self, *args, **kwargs):
        if self.model_ is None:
            raise RuntimeError("PretrainedModel has no underlying model")
        return self.model_(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @property
    def device(self):
        if self.model_ is None:
            raise RuntimeError("PretrainedModel has no underlying model")
        return self.model_.device
