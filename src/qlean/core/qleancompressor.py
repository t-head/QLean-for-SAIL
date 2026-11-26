# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
from typing import List, Any, Optional

from qlean.utils.singleton import SingletonMixin
from qlean.args import ModelArguments, DatasetArguments, OtherArguments
from qlean.models import ModelFactory
from qlean.pipelines import get_pipeline_from_modifiers
from qlean.pipelines.registry import is_gptq_modifier

from qlean.core.event_mixin import EventMixin
from qlean.core.state import State
from qlean.core.utils import (
    get_modifiers_from_recipe,
    preprocess_model,
    install_dependencies,
    remove_unsupported_fields,
    save_qlean_metadata,
    validate_gptq_output_format,
)

import logging

logger = logging.getLogger(__name__)


class QLeanCompressor(EventMixin, SingletonMixin):
    """Load a model and run recipe modifiers to write a converted checkpoint."""

    from qlean.modifiers import Modifier

    state: State
    modifiers: List[Modifier]
    _model_args: ModelArguments
    _quant_meta: dict
    dataloader: Optional[Any] = None
    _processor: Optional[Any] = None

    def __init__(
        self,
        model_args: ModelArguments,
        model=None,
        modifiers: Optional[List[Any]] = None,
    ):
        """Install declared dependencies, load the model, and create modifiers."""
        if modifiers is not None and model_args.recipe is not None:
            raise ValueError(
                "Specify either model_args.recipe or programmatic modifiers, not both"
            )
        if modifiers is not None and not modifiers:
            raise ValueError("Programmatic modifiers must not be empty")
        validate_gptq_output_format(model_args.output_format)

        mode = install_dependencies(model_args.model_name)
        logger.info(f"Using mode: {mode}")

        if not ModelFactory._loaded:
            ModelFactory.load_from_package()

        if model is None:
            from qlean.core.gptq_workflow import is_gptq_output, load_model_and_processor

            if is_gptq_output(model_args):
                model, self._processor = load_model_and_processor(model_args)
                model_wrapper = ModelFactory.create("pretrained", model_args)
                model_wrapper.model_ = model
                recipe_mode = "day0"
            else:
                model_wrapper = ModelFactory.create(mode, model_args)
                model_wrapper.load_model(model_args)
                recipe_mode = mode
        else:
            model_wrapper = ModelFactory.create("pretrained", model_args)
            model_wrapper.model_ = model
            recipe_mode = "day0"

        model_wrapper = preprocess_model(model_wrapper)
        self.state = State(
            model=model_wrapper,
            save_path=model_args.save_path,
            mix_path=model_args.mix_path
        )
        self._model_args = model_args
        if modifiers is None:
            self.modifiers, self._quant_meta = get_modifiers_from_recipe(
                recipe_mode, model_args
            )
        else:
            self.modifiers = list(modifiers)
            serialized_modifiers = [
                {
                    "type": type(modifier).__name__,
                    "config": modifier.model_dump(mode="json"),
                }
                for modifier in self.modifiers
            ]
            self._quant_meta = {
                "input_format": model_args.input_format or "programmatic",
                "output_format": model_args.output_format or "programmatic",
                "recipe_path": "<programmatic>",
                "recipe_content": {"modifiers": serialized_modifiers},
                "modifier_names": [
                    type(modifier).__name__ for modifier in self.modifiers
                ],
                "scheme": getattr(self.modifiers[0], "scheme", None)
                if len(self.modifiers) == 1
                else None,
            }


    def set_calibration_dataset(self, dataset_args: DatasetArguments):
        from qlean.core.gptq_workflow import build_calibration_dataloader

        dataloader = build_calibration_dataloader(
            dataset_args, self._model_args, self._processor
        )
        if dataloader is None:
            return
        self.dataloader = dataloader
        logger.info(f"Calibration dataloader created with {len(self.dataloader)} batches")

    def post_train(self, other_args: OtherArguments):
        """Run the selected pipeline and write QLean conversion metadata."""
        gptq_modifiers = [modifier for modifier in self.modifiers if is_gptq_modifier(modifier)]
        if not gptq_modifiers:
            pipeline = get_pipeline_from_modifiers(self.modifiers)
            pipeline(self.state.model, self.dataloader, other_args)
            remove_unsupported_fields(self.state.save_path)
            save_qlean_metadata(
                save_path=self.state.save_path,
                model_args=self._model_args,
                **self._quant_meta,
            )
            return

        from qlean.core.gptq_workflow import run_gptq_post_train

        run_gptq_post_train(self, other_args, gptq_modifiers)
