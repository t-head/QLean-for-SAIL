# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("qlean.modifiers.gptq.base", reason="requires the qlean[gptq] extra")

import torch
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32, unpack_from_int32
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from compressed_tensors.quantization.quant_config import get_vllm_module_type
from safetensors.torch import load_file, save_file
from transformers import LlamaConfig

import qlean.core.checkpoint as checkpoint_module
import qlean.core.qleancompressor as compressor_module
import qlean.modifiers.gptq.export as exporter
from qlean.args import ModelArguments
from qlean.core.qleancompressor import QLeanCompressor
from qlean.modifiers.gptq import GPTQModifier
from qlean.modifiers.save_utils import _checkpoint_weight_map
from qlean.utils.pack_acext_int4 import preprocess_weight_for_a8w4_torch


pytestmark = [pytest.mark.gptq, pytest.mark.integration, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def reset_compressor_singleton():
    QLeanCompressor._instances.pop(QLeanCompressor, None)
    yield
    QLeanCompressor._instances.pop(QLeanCompressor, None)


def _expert_scheme(targets):
    return QuantizationScheme(
        targets=targets,
        weights=QuantizationArgs(num_bits=4, type="int", strategy="channel", symmetric=True),
    )


def _qwen_source(tmp_path, sharded):
    from transformers import (
        Qwen3_5MoeConfig, Qwen3_5MoeTextConfig, Qwen3_5MoeVisionConfig,
        Qwen3_5MoeForConditionalGeneration,
    )

    torch.manual_seed(0)
    config = Qwen3_5MoeConfig(
        text_config=Qwen3_5MoeTextConfig(
            vocab_size=32, hidden_size=64, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            linear_key_head_dim=8, linear_value_head_dim=8,
            linear_num_key_heads=2, linear_num_value_heads=2,
            moe_intermediate_size=64, shared_expert_intermediate_size=64,
            num_experts=2, num_experts_per_tok=1, layer_types=["full_attention"],
            max_position_embeddings=32, num_mtp_layers=1,
            bos_token_id=1, eos_token_id=2, pad_token_id=0, tie_word_embeddings=False,
        ),
        vision_config=Qwen3_5MoeVisionConfig(
            depth=1, hidden_size=16, intermediate_size=16, num_heads=4,
            patch_size=2, spatial_merge_size=1, temporal_patch_size=1,
            out_hidden_size=64, num_position_embeddings=16,
        ),
        custom_setting={"preserve": ["vision", "mtp"]},
        tie_word_embeddings=False,
    )
    model = Qwen3_5MoeForConditionalGeneration(config).to(torch.bfloat16)
    source = tmp_path / "source"
    model.save_pretrained(source, max_shard_size="2KB" if sharded else "5GB")
    mtp = {
        "mtp.layers.0.self_attn.q_proj.weight": torch.arange(16).reshape(4, 4).to(torch.bfloat16),
        "mtp.layers.0.self_attn.q_proj.weight_scale": torch.ones(4, dtype=torch.bfloat16),
        "mtp.layers.0.norm.weight": torch.ones(4, dtype=torch.bfloat16),
    }
    if sharded:
        save_file(mtp, source / "source_mtp.safetensors")
        index_path = source / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        index["weight_map"].update({key: "source_mtp.safetensors" for key in mtp})
        index_path.write_text(json.dumps(index))
    else:
        save_file({**load_file(source / "model.safetensors"), **mtp}, source / "model.safetensors")
    # Calibrate a fresh BF16 reload, not the instance whose config was mutated by HF save.
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(source, dtype=torch.bfloat16)
    assert next(model.parameters()).dtype == torch.bfloat16
    model.name_or_path = str(source)
    return model, source, mtp


def _read_checkpoint(directory):
    weight_map = _checkpoint_weight_map(directory)
    tensors = {}
    for filename in set(weight_map.values()):
        shard = load_file(directory / filename)
        assert not tensors.keys() & shard.keys()
        assert all(weight_map[name] == filename for name in shard)
        tensors.update(shard)
    assert set(tensors) == set(weight_map)
    return tensors, weight_map


@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("export_device", ["cpu", "meta"])
@pytest.mark.parametrize("is_source,fail_copy", [
    (True, False), (False, False), (True, True),
], ids=["source", "non-source", "copy-failure"])
def test_qwen35_a8w4_source_only_save_preserves_vision_mtp_and_config(
    monkeypatch, tmp_path, sharded, is_source, fail_copy, export_device
):
    import compressed_tensors.distributed as distributed_module
    import qlean.modifiers.save_utils as save_utils
    import qlean.modifiers.gptq.export as exporter

    monkeypatch.setitem(sys.modules, "acext", None)
    model, source, mtp = _qwen_source(tmp_path, sharded)
    source_files = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    source_config = json.loads(source_files["config.json"])
    preserved = {
        name: tensor.clone() for name, tensor in model.state_dict().items()
        if ".experts." not in name
    }
    output = tmp_path / "output"
    snapshots, exports, completed = {}, [], []
    intermediate_files, prepared = {}, []
    original_prepare = exporter.prepare_checkpoint_preservation

    def prepare(*args, **kwargs):
        assert not prepared, "Prepare source preservation once, before calibration"
        checkpoint = original_prepare(*args, **kwargs)
        prepared.append(checkpoint)
        return checkpoint

    monkeypatch.setattr(exporter, "prepare_checkpoint_preservation", prepare)
    original_export = exporter.export_a8w4_checkpoint
    requested_device = export_device

    def observe_export(
        directory, source_model, max_shard_size, *, checkpoint=None,
        expert_patterns=None, export_device="auto", process_group=None,
    ):
        assert is_source
        assert source_model is model
        assert export_device == requested_device and process_group is None
        exports.append(directory)
        assert len(exports) == 1
        assert len(prepared) == 1 and checkpoint is prepared[0]
        directory = Path(directory)
        intermediate_files.update({p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()})
        quantization = json.loads((directory / "config.json").read_text())["quantization_config"]
        assert quantization["quant_method"] == "compressed-tensors"
        assert quantization["format"] == "pack-quantized"
        assert quantization["config_groups"]
        intermediate, _ = _read_checkpoint(directory)
        for key, packed in intermediate.items():
            if not key.endswith(".weight_packed"):
                continue
            name = key.removesuffix("_packed")
            shape = torch.Size(intermediate[f"{name}_shape"].tolist())
            snapshots[name] = (
                unpack_from_int32(packed, 4, shape), intermediate[f"{name}_scale"].clone(),
            )
        assert len(snapshots) == 6
        return original_export(
            directory, source_model, max_shard_size, checkpoint=checkpoint,
            expert_patterns=expert_patterns, export_device=export_device,
            process_group=process_group,
        )

    monkeypatch.setattr(exporter, "export_a8w4_checkpoint", observe_export)
    monkeypatch.setattr(distributed_module, "is_source_process", lambda: is_source)
    monkeypatch.setattr(compressor_module, "save_qlean_metadata", lambda **kwargs: completed.append(kwargs))
    if fail_copy:
        def fail_mtp_copy(*args, **kwargs):
            raise RuntimeError("MTP copy failed")
        monkeypatch.setattr(save_utils, "save_file", fail_mtp_copy)
    compressor = QLeanCompressor(
        ModelArguments(model_name="tiny-qwen3.5-moe", model_path=str(source),
                       save_path=str(output), output_format="GPTQ_INT4",
                       max_shard_size="8KB" if sharded else "10GB",
                       gptq_int4_export_device=export_device),
        model=model,
        modifiers=[GPTQModifier(config_groups={
            "experts": _expert_scheme([r"re:.*mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"]),
        }, block_size=32, ignore=[r"re:model\.visual.*", "lm_head", r"re:.*mlp\.gate$"])],
    )
    compressor.dataloader = [{
        "input_ids": torch.tensor([[1, 3, 4, 5]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }]
    assert not output.exists()
    if fail_copy:
        with pytest.raises(RuntimeError, match="MTP copy failed"):
            compressor.post_train(SimpleNamespace())
    else:
        compressor.post_train(SimpleNamespace())

    assert {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()} == source_files
    assert len(exports) == int(is_source)
    assert len(prepared) == 1
    assert len(completed) == int(is_source and not fail_copy)
    if not is_source:
        assert not output.exists()
        assert not snapshots
        return
    assert len(snapshots) == 6
    if fail_copy:
        assert not (output / "model_mtp.safetensors").exists()
        assert {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()} == intermediate_files
        assert not (output / "qlean_config.json").exists()
        return
    config = json.loads((output / "config.json").read_text())
    assert {key: value for key, value in config.items() if key != "quantization_config"} == source_config
    quantization = config["quantization_config"]
    assert quantization["quant_method"] == "mixed_precision_w4"
    assert quantization["activation_scheme"] == "dynamic"
    assert set(quantization) == {"quant_method", "activation_scheme", "ignore"}
    assert [item for item in quantization["ignore"] if "mtp" in item] == ["mtp.layers.0.self_attn.q_proj"]
    tensors, weight_map = _read_checkpoint(output)
    for name, (codes, scale) in snapshots.items():
        assert codes.shape == (64, 64)
        expected = preprocess_weight_for_a8w4_torch(codes).view(64, 32)
        assert tensors[name].dtype == torch.int8
        assert torch.equal(tensors[name], expected)
        assert tensors[f"{name}_scale"].dtype == torch.float32
        assert tensors[f"{name}_scale"].shape == (64, 1)
        assert torch.equal(tensors[f"{name}_scale"], scale.float())
        assert not any(f"{name}{suffix}" in tensors for suffix in (
            "_packed", "_shape", "_zero_point", "_g_idx", "_global_scale",
        ))
    assert any(name.startswith("model.visual.") for name in preserved)
    for name, tensor in {**preserved, **mtp}.items():
        assert torch.equal(tensors[name], tensor)
        assert tensors[name].dtype == tensor.dtype
    assert all(weight_map[name] == "model_mtp.safetensors" for name in mtp)
    assert not any(name.startswith("mtp") for name in model.state_dict())
    # This is the real pure-Torch producer contract, not a standard HF reload claim.


@pytest.fixture
def qwen35_linear_int4_source(tmp_path, sharded):
    from transformers import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig, Qwen3_5ForConditionalGeneration

    torch.manual_seed(0)
    config = Qwen3_5Config(
        text_config=Qwen3_5TextConfig(
            vocab_size=32, hidden_size=64, intermediate_size=128, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, head_dim=64,
            linear_key_head_dim=64, linear_value_head_dim=64,
            linear_num_key_heads=1, linear_num_value_heads=2, layer_types=["linear_attention"],
            max_position_embeddings=32, num_mtp_layers=0, mtp_num_hidden_layers=0,
            num_nextn_predict_layers=1, use_cache=False,
            bos_token_id=1, eos_token_id=2, pad_token_id=0, tie_word_embeddings=False,
        ),
        vision_config=Qwen3_5VisionConfig(
            depth=1, hidden_size=16, intermediate_size=16, num_heads=4,
            patch_size=2, spatial_merge_size=1, temporal_patch_size=1,
            out_hidden_size=64, num_position_embeddings=16,
        ),
        custom_setting={"preserve": ["vision", "mtp", "linear_attention"]},
        tie_word_embeddings=False,
    )
    model = Qwen3_5ForConditionalGeneration(config).to(torch.bfloat16)
    source = tmp_path / "source"
    model.save_pretrained(source, max_shard_size="2KB" if sharded else "10GB")
    mtp = {
        "mtp.layers.0.self_attn.q_proj.weight": torch.arange(16).reshape(4, 4).to(torch.bfloat16),
        "mtp.layers.0.self_attn.q_proj.weight_scale": torch.ones(4, dtype=torch.bfloat16),
        "mtp.layers.0.norm.weight": torch.ones(4, dtype=torch.bfloat16),
    }
    if sharded:
        save_file(mtp, source / "source_mtp.safetensors")
        path = source / "model.safetensors.index.json"
        index = json.loads(path.read_text())
        index["weight_map"].update({name: "source_mtp.safetensors" for name in mtp})
        index["metadata"]["total_size"] += sum(value.numel() * value.element_size() for value in mtp.values())
        path.write_text(json.dumps(index))
    else:
        path = source / "model.safetensors"
        save_file({**load_file(path), **mtp}, path)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(source, dtype=torch.bfloat16)
    assert next(model.parameters()).dtype == torch.bfloat16
    model.name_or_path = str(source)
    return model, source, mtp


@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("is_source,fail_copy", [
    (True, False), (False, False), (True, True),
], ids=["source", "non-source", "copy-failure"])
def test_qwen35_dense_int4_is_rejected_before_export(
    monkeypatch, tmp_path, sharded, is_source, fail_copy, qwen35_linear_int4_source
):
    import compressed_tensors.distributed as distributed_module
    import qlean.modifiers.save_utils as save_utils

    monkeypatch.setitem(sys.modules, "acext", None)
    model, source, mtp = qwen35_linear_int4_source
    prefix = "model.language_model.layers.0.linear_attn"
    modules = dict(model.named_modules())
    assert model.config.get_text_config().layer_types == ["linear_attention"]
    assert not any(isinstance(getattr(module, "experts", None), torch.nn.ModuleList) for module in modules.values())
    for name in (prefix, f"{prefix}.norm"):
        assert not isinstance(modules[name], torch.nn.Linear)
        assert get_vllm_module_type(type(modules[name]).__name__) == "Linear"
    ignored_linear = {
        name for name, module in modules.items()
        if isinstance(module, torch.nn.Linear)
        and (name.startswith("model.visual.") or name == "lm_head"
             or name in (f"{prefix}.in_proj_a", f"{prefix}.in_proj_b"))
    }
    selected = {
        f"{name}.weight": module for name, module in modules.items()
        if isinstance(module, torch.nn.Linear) and name not in ignored_linear
    }
    assert set(selected) == {
        f"{prefix}.{suffix}.weight" for suffix in ("out_proj", "in_proj_qkv", "in_proj_z")
    } | {
        f"model.language_model.layers.0.mlp.{suffix}.weight" for suffix in ("gate_proj", "up_proj", "down_proj")
    }
    assert {tuple(module.weight.shape) for module in selected.values()} == {(64, 128), (128, 64), (256, 64)}
    assert all(module.out_features % 64 == module.in_features % 64 == 0 for module in selected.values())
    for suffix in ("in_proj_a", "in_proj_b"):
        assert modules[f"{prefix}.{suffix}"].weight.shape == (2, 64)
    source_files = {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()}
    prepared, exports, completed = [], [], []
    snapshots, intermediate_files = {}, {}
    original_prepare = exporter.prepare_checkpoint_preservation

    def prepare(*args, **kwargs):
        assert not prepared, "Source preservation must be prepared once, before calibration"
        checkpoint = original_prepare(*args, **kwargs)
        prepared.append(checkpoint)
        return checkpoint

    monkeypatch.setattr(exporter, "prepare_checkpoint_preservation", prepare)
    original_pipeline = compressor_module.get_pipeline_from_modifiers

    def get_pipeline(modifiers):
        pipeline = original_pipeline(modifiers)

        def run(*args, **kwargs):
            assert len(prepared) == 1
            return pipeline(*args, **kwargs)
        return run

    monkeypatch.setattr(compressor_module, "get_pipeline_from_modifiers", get_pipeline)
    original_export = exporter.export_a8w4_checkpoint

    def observe_export(
        directory, source_model, max_shard_size, *, checkpoint=None,
        expert_patterns=None, export_device="auto", process_group=None,
    ):
        assert is_source and source_model is model
        assert export_device == "cpu" and process_group is None
        assert not exports
        exports.append(directory)
        assert len(prepared) == 1 and checkpoint is prepared[0]
        directory = Path(directory)
        intermediate_files.update({path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()})
        quantization = json.loads((directory / "config.json").read_text())["quantization_config"]
        assert quantization["quant_method"] == "compressed-tensors"
        assert quantization["format"] == "pack-quantized"
        assert quantization["config_groups"]
        assert prefix in quantization["ignore"] and f"{prefix}.norm" in quantization["ignore"]
        intermediate, _ = _read_checkpoint(directory)
        for key, packed in intermediate.items():
            if not key.endswith(".weight_packed"):
                continue
            name = key.removesuffix("_packed")
            module = selected[name]
            weights = module.quantization_scheme.weights
            assert weights.num_bits == 4 and weights.type == "int"
            assert weights.strategy == "channel" and weights.symmetric
            shape = torch.Size(intermediate[f"{name}_shape"].tolist())
            assert shape == (module.out_features, module.in_features)
            assert packed.dtype == torch.int32 and packed.shape == (shape[0], shape[1] // 8)
            codes = unpack_from_int32(packed, 4, shape)
            scale = intermediate[f"{name}_scale"]
            assert codes.dtype == torch.int8
            assert torch.all((codes >= -8) & (codes <= 7))
            assert scale.dtype == torch.bfloat16 and scale.shape == (shape[0], 1)
            snapshots[name] = (codes.clone(), scale.clone())
        assert set(snapshots) == set(selected)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        try:
            return original_export(
                directory, source_model, max_shard_size, checkpoint=checkpoint,
                expert_patterns=expert_patterns, export_device=export_device,
                process_group=process_group,
            )
        finally:
            after = model.state_dict()
            assert set(after) == set(before)
            for name, value in before.items():
                assert after[name].dtype == value.dtype
                assert torch.equal(after[name], value), name

    def fail_int8_export(*args, **kwargs):
        pytest.fail("Dense INT4 rejection must happen before any exporter")

    monkeypatch.setattr(exporter, "export_a8w4_checkpoint", observe_export)
    monkeypatch.setattr(exporter, "export_int8_checkpoint", fail_int8_export)
    monkeypatch.setattr(distributed_module, "is_source_process", lambda: is_source)
    original_metadata = compressor_module.save_qlean_metadata

    def complete(**kwargs):
        completed.append(kwargs)
        return original_metadata(**kwargs)

    monkeypatch.setattr(compressor_module, "save_qlean_metadata", complete)
    if fail_copy:
        def fail_mtp_copy(*args, **kwargs):
            raise RuntimeError("MTP copy failed")
        monkeypatch.setattr(save_utils, "save_file", fail_mtp_copy)
    output = tmp_path / "output"
    compressor = QLeanCompressor(
        ModelArguments(model_name="tiny-qwen3.5-dense-linear-int4", model_path=str(source),
                       save_path=str(output), output_format="GPTQ_INT4",
                       max_shard_size="8KB" if sharded else "10GB",
                       gptq_int4_export_device="cpu"), model=model,
        modifiers=[GPTQModifier(scheme="INT4_CHANNEL", targets=["Linear"], block_size=32,
                                 ignore=[r"re:model\.visual.*", "lm_head", r"re:.*linear_attn\.in_proj_[ab]$"])],
    )
    compressor.dataloader = [{
        "input_ids": torch.tensor([[1, 3, 4, 5]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "use_cache": False,
    }]
    assert not output.exists()
    with pytest.raises(ValueError, match="only routed MoE expert") as raised:
        compressor.post_train(SimpleNamespace())
    message = str(raised.value)
    assert "linear_attn.out_proj" in message
    assert "targets=[GPTQ_EXPERT_TARGET]" in message
    assert "GPTQModifier.ignore" in message
    assert {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()} == source_files
    assert len(prepared) == 1
    assert not exports and not completed and not snapshots
    assert not output.exists()


class _CheckpointConfigWriter:
    def update_config(self, directory):
        path = Path(directory) / "config.json"
        config = json.loads(path.read_text())
        config["quantization_config"] = {
            "quant_method": "compressed-tensors", "format": "pack-quantized", "ignore": ["lm_head"],
        }
        path.write_text(json.dumps(config))


@pytest.mark.parametrize("export_device", [None, "cpu"], ids=["default-auto", "explicit-cpu"])
def test_v4_save_scale_suffix_and_unexported_auxiliary_tensors(monkeypatch, tmp_path, export_device):
    monkeypatch.setitem(sys.modules, "acext", None)
    args = ModelArguments(model_name="defaults", model_path="unused", save_path=str(tmp_path))
    assert args.gptq_int4_export_device == "auto"
    assert args.gptq_export_workers == 2
    resolved = []
    original_resolver = exporter._Int4ExportDevice

    def resolve(requested, *args, **kwargs):
        resolved.append(requested)
        return original_resolver(requested, *args, **kwargs)

    monkeypatch.setattr(exporter, "_Int4ExportDevice", resolve)
    model = torch.nn.Module()
    model.config = LlamaConfig(num_mtp_layers=0)
    model.model = torch.nn.Module()
    layer = torch.nn.Module()
    layer.ffn = torch.nn.Module()
    expert = torch.nn.Module()
    expert.w1 = torch.nn.Linear(128, 64, bias=False)
    expert.w1.quantization_scheme = _expert_scheme(["Linear"])
    layer.ffn.experts = torch.nn.ModuleList([expert])
    model.model.layers = torch.nn.ModuleList([layer])
    name = "model.layers.0.ffn.experts.0.w1.weight"
    codes = (torch.arange(64 * 128).reshape(64, 128) % 16 - 8).to(torch.int8)
    scales = ((torch.arange(64).reshape(64, 1) + 1) / 128).to(torch.bfloat16)
    preserved = {
        "model.layers.0.ffn.shared_experts.w1.weight": torch.ones(64, 128, dtype=torch.bfloat16),
        "model.layers.0.ffn.shared_experts.w1.weight_scale": torch.full((64, 1), 0.5, dtype=torch.bfloat16),
        "model.layers.0.ffn.shared_experts.w1.weight_shape": torch.tensor([64, 128]),
        "model.layers.0.ffn.shared_experts.w1.weight_zero_point": torch.zeros(64, 1, dtype=torch.int8),
    }

    def save(directory, *, max_shard_size):
        assert max_shard_size == "10GB"
        model.config.save_pretrained(directory)
        save_file({
            **preserved, f"{name}_packed": pack_to_int32(codes, 4),
            f"{name}_shape": torch.tensor([64, 128]), f"{name}_scale": scales,
            f"{name}_zero_point": torch.zeros(64, 1, dtype=torch.int8),
        }, Path(directory) / "model.safetensors")

    model.save_pretrained = save
    modifiers = [GPTQModifier(config_groups={
        "experts": _expert_scheme([r"re:.*ffn\.experts.*"]),
    })]
    exporter.validate_gptq_output_directory(model, "unused", tmp_path, modifiers)
    checkpoint_module.save_checkpoint(model, tmp_path, "10GB", _CheckpointConfigWriter(), "pack-quantized")
    kwargs = {} if export_device is None else {"export_device": export_device}
    metadata = exporter.finalize_gptq_checkpoint(
        model, tmp_path, "10GB", "pack-quantized", ["model.layers.0.ffn.experts."], **kwargs,
    )
    assert resolved == ["auto" if export_device is None else "cpu"]
    assert metadata == {"output_format": "GPTQ_INT4", "scheme": "INT4_CHANNEL"}

    tensors, _ = _read_checkpoint(tmp_path)
    expected = preprocess_weight_for_a8w4_torch(codes).view(64, 64)
    assert torch.equal(tensors[name], expected)
    scale_key = "model.layers.0.ffn.experts.0.w1.scale"
    assert tensors[scale_key].dtype == torch.float32
    assert tensors[scale_key].shape == (64, 1)
    assert torch.equal(tensors[scale_key], scales.float())
    assert set(tensors) == set(preserved) | {name, scale_key}
    for key, tensor in preserved.items():
        assert torch.equal(tensors[key], tensor)
        assert tensors[key].dtype == tensor.dtype
    quantization = json.loads((tmp_path / "config.json").read_text())["quantization_config"]
    assert quantization["quant_method"] == "mixed_precision_w4"
    assert quantization["activation_scheme"] == "dynamic"
    assert set(quantization) == {"quant_method", "activation_scheme", "ignore"}
    assert "model.layers.0.ffn.shared_experts.w1" in quantization["ignore"]
    assert "model.layers.0.ffn.experts.0.w1" not in quantization["ignore"]


@pytest.mark.parametrize("existing", ["model.safetensors", "model.safetensors.index.json"])
def test_a8w4_save_requires_fresh_output_without_overwriting(existing, tmp_path):
    model = torch.nn.Module()
    model.config = LlamaConfig(num_mtp_layers=0)
    model.mlp = torch.nn.Module()
    model.mlp.experts = torch.nn.ModuleList([torch.nn.Linear(64, 64, bias=False)])
    model.mlp.experts[0].quantization_scheme = _expert_scheme(["Linear"])

    def must_not_save(*args, **kwargs):
        pytest.fail("An existing A8W4 checkpoint must be rejected before save_pretrained")

    model.save_pretrained = must_not_save
    modifiers = [GPTQModifier(config_groups={
        "experts": _expert_scheme([r"re:.*mlp\.experts.*"]),
    })]
    if existing.endswith(".safetensors"):
        save_file({"sentinel.weight": torch.ones(1)}, tmp_path / existing)
    else:
        (tmp_path / existing).write_text(json.dumps({"weight_map": {"sentinel.weight": "old.safetensors"}}))
    (tmp_path / "config.json").write_text(json.dumps({"sentinel": "preserve"}))
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}

    with pytest.raises(ValueError, match="fresh checkpoint output directory"):
        exporter.validate_gptq_output_directory(model, "unused", tmp_path, modifiers)
        checkpoint_module.save_checkpoint(model, tmp_path, "10GB", _CheckpointConfigWriter(), "pack-quantized")
        exporter.finalize_gptq_checkpoint(model, tmp_path, "10GB", "pack-quantized", ["mlp.experts."])

    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
