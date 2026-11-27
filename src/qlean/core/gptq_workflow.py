# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GPTQ workflow extracted from QLeanCompressor.

QLeanCompressor stays the composition root and delegates here. Heavy imports live
inside functions so a core-only install never pulls the GPTQ backend.

The save path resolves ``get_pipeline_from_modifiers`` and ``save_qlean_metadata``
through the qleancompressor module at call time rather than importing them here:
those names are established monkeypatch points for the checkpoint tests, and
binding them locally would silently bypass the patches.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader

from qlean.args import DatasetArguments, ModelArguments, OtherArguments

logger = logging.getLogger(__name__)

__all__ = [
    "is_gptq_output",
    "load_model_and_processor",
    "build_calibration_dataloader",
    "run_gptq_post_train",
]


def is_gptq_output(model_args: ModelArguments) -> bool:
    """Return whether the requested output format routes to GPTQ."""
    from qlean.core.routes import GPTQ_OUTPUT_PRESETS

    return model_args.output_format in GPTQ_OUTPUT_PRESETS


def load_model_and_processor(model_args: ModelArguments):
    """Load ``(model, processor)`` when the caller supplied no model object.

    Uses the same loader the torchrun scripts reach when they omit ``model=``.
    ``init_dist`` must precede loading so rank/device binding matches the scripts.
    """
    from qlean.models.pretrained.loader import load_gptq_model
    from qlean.utils.distributed import maybe_init_dist

    maybe_init_dist()
    return load_gptq_model(model_args)


def _collate(batch):
    if not batch:
        return {}
    if isinstance(batch[0], dict):
        collated = {}
        for key in batch[0]:
            values = [example[key] for example in batch]
            if isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values)
            elif isinstance(values[0], str):
                continue
            else:
                try:
                    collated[key] = torch.tensor(values)
                except (TypeError, ValueError, RuntimeError) as error:
                    # torch.tensor raises RuntimeError for undeducible dtypes, so
                    # without it neither message below would ever be reached.
                    lengths = {
                        len(value) for value in values if hasattr(value, "__len__")
                    }
                    if len(lengths) > 1:
                        raise ValueError(
                            f"Calibration field '{key}' has ragged lengths "
                            f"{sorted(lengths)}; set --calibration_padding "
                            "max_length or keep --batch_size 1."
                        ) from error
                    raise ValueError(
                        f"Calibration field '{key}' must contain tensors or "
                        "numeric values. Tokenize the dataset and remove "
                        "unsupported metadata columns."
                    ) from error
        return collated
    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch)
    return torch.tensor(batch)


def build_calibration_dataloader(
    dataset_args: DatasetArguments,
    model_args: ModelArguments,
    processor: Optional[Any] = None,
) -> Optional[DataLoader]:
    """Build the calibration dataloader, or None for a data-free run without data.

    When no dataset object was supplied, GPTQ builds one through the same builder
    the torchrun scripts reach when they omit ``dataset=``.
    """
    dataset = dataset_args.dataset
    if dataset is None:
        if not is_gptq_output(model_args):
            logger.warning("No calibration dataset provided in dataset_args.dataset")
            return None

        from qlean.data.calibration import build_calibration_dataset
        from qlean.models.pretrained.loader import load_processor_or_tokenizer
        from qlean.utils.distributed import maybe_init_dist

        maybe_init_dist()
        if processor is None:
            processor = load_processor_or_tokenizer(
                model_args.model_path, model_args.trust_remote_code
            )
        dataset = build_calibration_dataset(dataset_args, processor)

    return DataLoader(
        dataset,
        batch_size=dataset_args.batch_size,
        collate_fn=_collate,
    )


def run_gptq_post_train(compressor, other_args: OtherArguments, gptq_modifiers: list) -> None:
    """Run the GPTQ pipeline and write the checkpoint, metadata and tokenizer."""
    from compressed_tensors.distributed import is_source_process
    from compressed_tensors.offload.convert.to_accelerate import to_accelerate

    from qlean.core import checkpoint
    from qlean.core import qleancompressor as compressor_module
    from qlean.modifiers.gptq import export
    from qlean.modifiers.gptq.distributed import (
        raise_for_distributed_errors,
        source_rank,
    )
    from qlean.utils.dev import suspend_distributed_timeout

    model_args = compressor._model_args
    quant_meta = compressor._quant_meta
    model = compressor.state.model.model_
    source = model_args.model_path
    save_path = compressor.state.save_path

    export.validate_gptq_output_directory(model, source, save_path, gptq_modifiers)
    # Fail-fast: validate(strict=True) before the expensive calibration. The return
    # value is intentionally discarded; save_checkpoint calls this again and uses it.
    checkpoint.prepare_generation_config_for_save(model)
    preservation = export.prepare_source_checkpoint(model, source, save_path, gptq_modifiers)
    pipeline = compressor_module.get_pipeline_from_modifiers(compressor.modifiers)
    pipeline(compressor.state.model, compressor.dataloader, other_args)

    export.validate_gptq_output_directory(model, source, save_path, gptq_modifiers)
    expected_format = export.expected_save_format(model)
    expert_patterns = checkpoint.expert_key_patterns(model)
    save_compressor = checkpoint.compress_model_for_save(
        model, quantization_format="int-quantized" if expected_format == "int-quantized" else None,
    )
    to_accelerate(model)
    with suspend_distributed_timeout() as save_group:
        source_rank(save_group)
        source_process = is_source_process()
        save_error = None
        if source_process:
            try:
                export.validate_gptq_output_directory(
                    model, source, save_path, gptq_modifiers
                )
                if expected_format in {"int-quantized", "mxfp4-pack-quantized"}:
                    export.validate_export_workers(model_args.gptq_export_workers)
                checkpoint.save_checkpoint(
                    model, save_path, model_args.max_shard_size,
                    save_compressor, expected_format,
                )
            except BaseException as error:
                save_error = error
        raise_for_distributed_errors(save_error, "checkpoint save", save_group)

        metadata = None
        finalize_error = None
        uses_a8w4 = bool(export.a8w4_quantized_modules(model))
        if uses_a8w4 and save_group is not None:
            metadata = export.finalize_gptq_checkpoint(
                model, save_path, model_args.max_shard_size, expected_format,
                expert_patterns, checkpoint=preservation,
                export_device=model_args.gptq_int4_export_device,
                process_group=save_group,
            )
        elif source_process:
            try:
                metadata = export.finalize_gptq_checkpoint(
                    model, save_path, model_args.max_shard_size, expected_format,
                    expert_patterns, checkpoint=preservation,
                    **({"export_device": model_args.gptq_int4_export_device}
                       if uses_a8w4 else {"workers": model_args.gptq_export_workers}),
                )
            except BaseException as error:
                finalize_error = error
        raise_for_distributed_errors(finalize_error, "checkpoint finalization", save_group)

        metadata_error = None
        if source_process:
            try:
                quant_meta.update(metadata)
                compressor_module.save_qlean_metadata(
                    save_path=save_path,
                    model_args=model_args,
                    **quant_meta,
                )
            except BaseException as error:
                metadata_error = error
        raise_for_distributed_errors(metadata_error, "metadata save", save_group)

        # _processor is set only when qlean loaded the model itself (the CLI path),
        # and is rank-uniform, so every rank enters or skips this stage together.
        # A caller that passed model= owns its auxiliary files, as the torchrun
        # scripts do. TORCHELASTIC_RUN_ID cannot distinguish the two: the CLI also
        # re-execs under torchrun when nproc_per_node > 1.
        if compressor._processor is not None:
            processor_error = None
            if source_process:
                try:
                    compressor._processor.save_pretrained(save_path)
                    logger.info("Saved tokenizer/processor to %s", save_path)
                except BaseException as error:
                    processor_error = error
            raise_for_distributed_errors(processor_error, "processor save", save_group)