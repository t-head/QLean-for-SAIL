# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Hardening of configuration boundaries and shared-state isolation."""

import json
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("qlean.modifiers.gptq.base", reason="requires the qlean[gptq] extra")

from compressed_tensors.quantization import QuantizationArgs

from qlean.args import DatasetArguments
from qlean.core.utils import apply_gptq_output_format
from qlean.observers.min_max import MemorylessMinMaxObserver

pytestmark = [pytest.mark.gptq, pytest.mark.unit, pytest.mark.cpu]


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "2", None])
def test_batch_size_rejects_non_positive_integers(bad):
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        DatasetArguments(batch_size=bad)


@pytest.mark.parametrize("good", [1, 2, 8])
def test_batch_size_accepts_positive_integers(good):
    assert DatasetArguments(batch_size=good).batch_size == good


@pytest.mark.parametrize("output_format,preset", [
    ("GPTQ_MXFP4", "MXFP4"), ("GPTQ_INT8", "INT8_CHANNEL"),
    ("GPTQ_INT8_GROUP", "INT8_GROUP"),
])
def test_empty_gptq_modifier_body_is_materialized(output_format, preset):
    recipe = {"quant_stage": {"quant_modifiers": {"GPTQModifier": None}}}
    assert apply_gptq_output_format(recipe, output_format) == preset
    body = recipe["quant_stage"]["quant_modifiers"]["GPTQModifier"]
    assert isinstance(body, dict) and body["scheme"] == preset
    assert body["targets"] == ["Linear"]


def test_unknown_scheme_string_is_rejected_not_silently_ignored():
    from qlean.modifiers.quantization import QuantizationCalibrationMixin

    modifier = QuantizationCalibrationMixin(scheme="W8A88")
    with pytest.raises(ValueError, match="Unknown quantization scheme 'W8A88'"):
        modifier.resolve_quantization_config()


def test_observer_does_not_mutate_the_shared_scheme_args():
    args = QuantizationArgs(
        num_bits=8, type="int", strategy="channel", symmetric=True,
        observer_kwargs={"existing": 1},
    )
    first = MemorylessMinMaxObserver(base_name="weight", args=args, extra1=1)
    second = MemorylessMinMaxObserver(base_name="weight", args=args, extra2=2)
    assert first.args is not args and second.args is not args
    assert args.observer_kwargs == {"existing": 1}
    assert first.args.observer_kwargs == {"existing": 1, "extra1": 1}
    assert second.args.observer_kwargs == {"existing": 1, "extra2": 2}
    first.args.observer_kwargs["leak"] = True
    assert "leak" not in second.args.observer_kwargs
    assert "leak" not in args.observer_kwargs


def test_observer_handles_args_without_observer_kwargs():
    args = QuantizationArgs(num_bits=8, type="int", strategy="channel", symmetric=True)
    observer = MemorylessMinMaxObserver(base_name="weight", args=args, extra=1)
    assert observer.args.observer_kwargs == {"extra": 1}
    assert args.observer_kwargs in (None, {})


def test_deepseek_preparation_tolerates_missing_dtype(tmp_path):
    from types import SimpleNamespace

    from qlean.models.pretrained import model_registry

    (tmp_path / "inference").mkdir()
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "deepseek_v32"}))
    (tmp_path / "inference" / "config_671B_v3.2.json").write_text(
        json.dumps({"n_routed_experts": 256})
    )
    profile = model_registry.resolve_model_profile("DeepSeek-V3.2-BF16", str(tmp_path))
    profile["preparation"][0](SimpleNamespace(model_path=str(tmp_path)))
    merged = json.loads((tmp_path / "config.json").read_text())
    assert merged["n_routed_experts"] == 256 and "dtype" not in merged


def test_collate_distinguishes_ragged_lengths_from_bad_metadata():
    from qlean.core.gptq_workflow import _collate

    with pytest.raises(ValueError, match="ragged lengths"):
        _collate([{"input_ids": [1, 2]}, {"input_ids": [1, 2, 3]}])
    with pytest.raises(ValueError, match="must contain tensors or"):
        _collate([{"meta": object()}, {"meta": object()}])


def test_to_device_handles_non_dict_batches_and_skips_meta():
    from qlean.pipelines.sequential.pipeline import _to_device

    cpu = SimpleNamespace(device=torch.device("cpu"))
    tensor = torch.tensor([1, 2])
    assert torch.equal(_to_device(tensor, cpu), tensor)
    listed = _to_device([torch.tensor([1]), torch.tensor([2])], cpu)
    assert isinstance(listed, list) and len(listed) == 2
    tupled = _to_device((torch.tensor([1]),), cpu)
    assert isinstance(tupled, tuple)
    meta = SimpleNamespace(device=torch.device("meta"))
    assert _to_device(tensor, meta) is tensor
    assert _to_device({"input_ids": tensor}, meta) == {"input_ids": tensor}


def test_checkpoint_weight_map_rejects_index_without_weight_map(tmp_path):
    from qlean.modifiers.save_utils import _checkpoint_weight_map

    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}}))
    with pytest.raises(ValueError, match="weight_map"):
        _checkpoint_weight_map(str(tmp_path))