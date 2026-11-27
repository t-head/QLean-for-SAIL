# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build the GPTQ calibration dataset exactly as the reference torchrun scripts do.

Order matters and is kept identical to the scripts: partition the split per rank
first, then shuffle, then render chat templates, then tokenize. Tokenization
truncates to ``max_sequence_length`` and pads according to ``calibration_padding``
("none" keeps ragged lengths, which the default batch size of 1 makes safe).
"""

from __future__ import annotations

import logging

from qlean.args import DatasetArguments

logger = logging.getLogger(__name__)

__all__ = ["build_calibration_dataset"]


def build_calibration_dataset(dataset_args: DatasetArguments, processor):
    """Return a tokenized, rank-partitioned calibration dataset."""
    from datasets import load_dataset

    from qlean.models.pretrained.loader import tokenizer_of
    from qlean.utils.dev import get_rank_partition

    source = dataset_args.dataset_name or dataset_args.dataset_path
    if not source:
        raise ValueError(
            "GPTQ needs calibration data: pass --dataset_name (Hub id) or "
            "--dataset_path (local dataset), or supply dataset_args.dataset."
        )

    split = get_rank_partition(
        dataset_args.dataset_split, dataset_args.num_calibration_samples
    )
    load_kwargs = {"split": split}
    if dataset_args.dataset_config:
        load_kwargs["name"] = dataset_args.dataset_config

    logger.info("Loading calibration data %s (%s)", source, split)
    dataset = load_dataset(source, **load_kwargs)
    dataset = dataset.shuffle(seed=dataset_args.seed)

    tokenizer = tokenizer_of(processor)
    columns = dataset.column_names
    messages_column = dataset_args.messages_column
    text_column = dataset_args.text_column

    if messages_column in columns:
        def render(example):
            return {
                "text": processor.apply_chat_template(
                    example[messages_column], tokenize=False
                )
            }

        dataset = dataset.map(render)
        # render() always emits "text", regardless of the configured text_column.
        source_column = "text"
    elif text_column in columns:
        source_column = text_column
    else:
        raise ValueError(
            f"Calibration dataset has neither a '{messages_column}' nor a "
            f"'{text_column}' column; found {columns}. Set --messages_column or "
            "--text_column to match the dataset."
        )

    def tokenize(sample):
        return tokenizer(
            sample[source_column],
            padding="max_length" if dataset_args.calibration_padding == "max_length" else False,
            max_length=dataset_args.max_sequence_length,
            truncation=True,
            add_special_tokens=False,
        )

    return dataset.map(tokenize, remove_columns=dataset.column_names)