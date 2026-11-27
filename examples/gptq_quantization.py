# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""BF16 -> GPTQ with MXFP4, INT4_CHANNEL, INT8_CHANNEL, INT4_GROUP, or INT8_GROUP.

Usage:
    python examples/gptq_quantization.py /path/to/model /path/to/fresh-output
    python examples/gptq_quantization.py /path/to/model /path/to/fresh-output --scheme INT8_CHANNEL
    python examples/gptq_quantization.py /path/to/model /path/to/fresh-output --scheme MXFP4 --model-name model-id

MXFP4 and the INT8 schemes target Linear weights except protected modules.
INT4_GROUP uses symmetric 128-wide groups along the input dimension with scales
in the model dtype, matching llmcompressor's W4A16 recipe, and targets all
Linear weights. INT4_CHANNEL uses QLean's original pure PyTorch packer,
requiring both expert dimensions aligned to 64 without an ACExt/native
extension; it targets routed MoE experts only, so dense models cannot use it.
INT8_CHANNEL uses symmetric per-channel weights with FP32 scales.

The GROUP schemes use standard compressed-tensors `pack-quantized` weights, so
standard compressed-tensors/Hugging Face weight-only loaders can read them.

Outputs map to GPTQ_MXFP4, GPTQ_INT4, GPTQ_INT8, GPTQ_INT4_GROUP, and
GPTQ_INT8_GROUP respectively. The channel INT4/INT8 outputs are not compatible
with standard compressed-tensors (CT) INT4/INT8 loaders or standard Hugging
Face reloading; use a fresh directory. Whole-model quantization has not been
validated. Loading defaults to Qwen3_5ForConditionalGeneration, which matches the
default Qwen3.8-27B checkpoint; the export guard compares the source and runtime
config model_type, so a text-only AutoModelForCausalLM class is rejected for that
composite checkpoint. Callers must pass an appropriate model_cls for other
architectures.
"""

import argparse

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

from qlean import QLeanCompressor
from qlean.args import DatasetArguments, ModelArguments, OtherArguments
from qlean.core.routes import (
    GPTQ_EXPERT_TARGET,
    GPTQ_INT4_PRESETS,
    GPTQ_OUTPUT_PRESETS,
    GPTQ_WEIGHT_CONFIGS,
)
from qlean.modifiers.gptq import GPTQModifier
from qlean.utils import get_rank_partition, load_context, temporary_offload_folder
from qlean.utils.distributed import maybe_init_dist


def quantize_model(
    model_path: str,
    save_path: str,
    scheme: str = "MXFP4",
    *,
    model_name: str = "Qwen/Qwen3.8-27B",
    model_cls=Qwen3_5ForConditionalGeneration,
    dataset: str = "HuggingFaceH4/ultrachat_200k",
    dataset_config: str | None = None,
    dataset_split: str = "train_sft",
    num_calibration_samples: int = 128,
    max_sequence_length: int = 2048,
    num_calibration_rounds: int = 1,
):
    if scheme not in GPTQ_WEIGHT_CONFIGS:
        raise ValueError(
            f"supported schemes: {', '.join(GPTQ_WEIGHT_CONFIGS)}; got {scheme!r}"
        )
    output_formats = {v: k for k, v in GPTQ_OUTPUT_PRESETS.items()}

    # This example loads the model and the dataset itself, so neither of the
    # compressor's maybe_init_dist() call sites runs. Initialize here so a torchrun
    # launch does not leave every rank calibrating and writing independently.
    maybe_init_dist()

    # Offloaded weights must remain available through calibration and export.
    with temporary_offload_folder() as offload_folder:
        with load_context(model_cls=model_cls):
            model = model_cls.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto_offload",
                offload_folder=offload_folder,
            )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Partition the split per rank before shuffling, as the reference torchrun
        # scripts and qlean's own calibration builder do; otherwise every rank
        # calibrates on the same samples and repeats the work world_size times.
        # Requires the process group initialized above.
        load_kwargs = {"split": get_rank_partition(dataset_split, num_calibration_samples)}
        if dataset_config:
            load_kwargs["name"] = dataset_config
        raw = load_dataset(dataset, **load_kwargs)
        if "messages" in raw.column_names:
            raw = raw.map(
                lambda row: {"text": tokenizer.apply_chat_template(
                    row["messages"], tokenize=False
                )}
            )
        elif "text" not in raw.column_names:
            raise ValueError(
                f"{dataset} has neither a 'messages' nor a 'text' column; "
                f"found {raw.column_names}"
            )
        raw = raw.filter(lambda row: bool(row["text"].strip()))
        dataset = raw.shuffle(seed=42)
        # Ragged lengths with batch size 1, matching the reference torchrun scripts
        # and qlean's own calibration builder. Fixed-length padding emits an int64
        # attention_mask that this checkpoint's SDPA path rejects, while GLM-5.2
        # quantizes fine with it.
        dataset = dataset.map(
            lambda batch: tokenizer(
                batch["text"], padding=False, truncation=True,
                max_length=max_sequence_length,
            ),
            batched=True,
            remove_columns=dataset.column_names,
        )

        ignore = [
            "lm_head",
            r"re:.*lm_head",
            r"re:.*embed_tokens",
            r"re:visual.*",
            r"re:model.visual.*",
            r"re:.*conv1d$",
            r"re:.*in_proj_a$",
            r"re:.*in_proj_b$",
            r"re:.*fc$",
            r"re:.*pre_fc_norm_embedding$",
            r"re:.*pre_fc_norm_hidden$",
            r"re:.*mtp.*",
            r"re:(?:.*\.)?gate$",
            r"re:(?:.*\.)?router$",
            r"re:(?:.*\.)?shared_expert_gate$",
        ]
        targets = [GPTQ_EXPERT_TARGET] if scheme in GPTQ_INT4_PRESETS else ["Linear"]
        modifiers = [
            GPTQModifier(
                scheme=scheme,
                targets=targets,
                ignore=ignore,
                block_size=128,
                dampening_frac=0.001,
                actorder="static",
                offload_hessians=False,
            ),
        ]
        model_args = ModelArguments(
            model_name=model_name,
            model_path=model_path,
            save_path=save_path,
            input_format="BF16",
            output_format=output_formats[scheme],
        )
        compressor = QLeanCompressor(model_args, model=model, modifiers=modifiers)
        compressor.set_calibration_dataset(
            DatasetArguments(
                dataset_name="HuggingFaceH4/ultrachat_200k", dataset=dataset, batch_size=1
            )
        )
        compressor.post_train(OtherArguments(num_calibration_rounds=num_calibration_rounds))
        # This example passes model=, so the compressor leaves auxiliary files to
        # the caller; save the tokenizer next to the weights.
        tokenizer.save_pretrained(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", help="Local BF16 checkpoint for the loading class")
    parser.add_argument("save_path", help="Fresh output weights directory")
    parser.add_argument(
        "--scheme",
        choices=[
            "MXFP4", "INT4_CHANNEL", "INT8_CHANNEL", "INT4_GROUP", "INT8_GROUP",
        ],
        default="MXFP4",
        help=(
            "GPTQ scheme; INT4_CHANNEL targets routed MoE experts only, "
            "all other schemes target Linear weights (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--model-name", default="Qwen/Qwen3.8-27B",
        help="QLean model name (default: %(default)s); does not change the loading class",
    )
    parser.add_argument(
        "--dataset", default="HuggingFaceH4/ultrachat_200k",
        help="Calibration dataset: Hub id or local directory (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset-config", default="",
        help="Dataset config/subset; empty for datasets without one (default: none)",
    )
    parser.add_argument("--dataset-split", default="train_sft", help="Split (default: %(default)s)")
    parser.add_argument(
        "--num-calibration-samples", type=int, default=128,
        help="Samples after shuffling (default: %(default)s)",
    )
    parser.add_argument(
        "--max-sequence-length", type=int, default=2048,
        help="Truncation cap; lengths stay ragged (default: %(default)s)",
    )
    parser.add_argument(
        "--num-calibration-rounds", type=int, default=1,
        help="Split each layer's targets into N rounds to bound Hessian memory "
             "(default: %(default)s; use 2 or more for large MoE checkpoints)",
    )
    args = parser.parse_args()
    quantize_model(
        args.model_path, args.save_path, args.scheme, model_name=args.model_name,
        dataset=args.dataset,
        dataset_config=args.dataset_config or None,
        dataset_split=args.dataset_split,
        num_calibration_samples=args.num_calibration_samples,
        max_sequence_length=args.max_sequence_length,
        num_calibration_rounds=args.num_calibration_rounds,
    )
