# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("qlean.modifiers.gptq.base", reason="requires the qlean[gptq] extra")

import torch
from compressed_tensors.quantization.quant_config import get_vllm_module_type
from safetensors.torch import load_file, save_file
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.core_model_loading import WeightRenaming, rename_source_key

import qlean.core.qleancompressor as compressor_module
import qlean.modifiers.gptq.export as exporter
from qlean.args import ModelArguments
from qlean.core.qleancompressor import QLeanCompressor
from qlean.modifiers.gptq import GPTQModifier
from qlean.modifiers.save_utils import _checkpoint_weight_map


pytestmark = [pytest.mark.gptq, pytest.mark.integration, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def reset_compressor_singleton():
    QLeanCompressor._instances.pop(QLeanCompressor, None)
    yield
    QLeanCompressor._instances.pop(QLeanCompressor, None)


def _batch():
    return {
        "input_ids": torch.tensor([[1, 3, 4, 5]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "use_cache": False,
    }


def _read_checkpoint(directory):
    directory = Path(directory)
    weight_map = _checkpoint_weight_map(directory)
    tensors = {}
    for filename in set(weight_map.values()):
        shard = load_file(directory / filename)
        assert not tensors.keys() & shard.keys()
        assert all(weight_map[name] == filename for name in shard)
        tensors.update(shard)
    assert set(weight_map) == set(tensors)
    assert set(weight_map.values()) == {p.name for p in directory.glob("*.safetensors")}
    index_path = directory / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        assert index["metadata"]["total_size"] == sum(
            value.numel() * value.element_size() for value in tensors.values()
        )
    return tensors, weight_map


def _files(directory):
    return {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}


def _covers(pattern, name):
    if pattern.startswith("re:"):
        return re.match(pattern[3:], name) is not None
    return name == pattern or name.startswith(pattern + ".")


def _observe_export(monkeypatch, model, *, is_source=True, prepared=None):
    record = SimpleNamespace(calls=0, params={}, intermediate=None)
    original = exporter.export_int8_checkpoint

    def observe(directory, source_model, max_shard_size, *, checkpoint=None, expert_patterns=None, workers=2):
        assert is_source and source_model is model
        assert workers == 2
        assert record.calls == 0
        record.calls += 1
        if prepared is not None:
            assert len(prepared) == 1
            assert checkpoint is prepared[0]
        directory = Path(directory)
        quantization = json.loads((directory / "config.json").read_text())["quantization_config"]
        assert quantization["quant_method"] == "compressed-tensors"
        assert quantization["format"] == "int-quantized"
        record.intermediate = _files(directory)
        tensors, _ = _read_checkpoint(directory)
        modules = exporter.int8_quantized_modules(model, require_int8=True)
        renamings = [
            conversion for conversion in getattr(model, "_weight_conversions", ()) or ()
            if isinstance(conversion, WeightRenaming)
        ]
        for name, tensor in tensors.items():
            if not name.endswith(".weight"):
                continue
            model_key, _ = rename_source_key(name, renamings, [])
            module = modules.get(model_key.removesuffix(".weight"))
            if module is None:
                continue
            assert tensor.dtype == torch.int8
            assert tensor.shape == (module.out_features, module.in_features)
            scale = tensors[f"{name}_scale"]
            assert scale.dtype == torch.bfloat16
            assert scale.shape == (module.out_features, 1)
            record.params[name] = (tensor.clone(), scale.clone())
        assert len(record.params) == len(modules) > 0
        before = {name: value.clone() for name, value in model.state_dict().items()}
        try:
            return original(
                directory, source_model, max_shard_size, checkpoint=checkpoint, expert_patterns=expert_patterns,
                workers=workers,
            )
        finally:
            # Export must not mutate the live model, even when finalization raises.
            assert set(model.state_dict()) == set(before)
            for name, value in before.items():
                assert model.state_dict()[name].dtype == value.dtype
                assert torch.equal(model.state_dict()[name], value), name

    monkeypatch.setattr(exporter, "export_int8_checkpoint", observe)
    return record


def _assert_int8_output(output, model, record, preserved, *, sharded):
    tensors, weight_map = _read_checkpoint(output)
    assert len(set(weight_map.values())) > 1 if sharded else len(set(weight_map.values())) == 1
    quantization = json.loads((output / "config.json").read_text())["quantization_config"]
    assert quantization["quant_method"] == "w8a8_int8"
    assert not set(quantization) & {
        "format", "config_groups", "kv_cache_scheme", "quantization_status",
        "sparsity_config", "transform_config", "version", "global_compression_ratio",
    }
    for name, (codes, scale) in record.params.items():
        assert tensors[name].dtype == torch.int8
        assert tensors[name].shape == codes.shape
        assert torch.equal(tensors[name], codes)
        output_scale = tensors[f"{name}_scale"]
        assert output_scale.dtype == torch.float32
        assert output_scale.shape == (codes.shape[0], 1)
        assert torch.equal(output_scale, scale.float())
        assert torch.isfinite(output_scale).all() and (output_scale > 0).all()
        assert not any(f"{name}{suffix}" in tensors for suffix in (
            "_packed", "_shape", "_zero_point", "_g_idx", "_global_scale",
        ))
        assert not any(_covers(pattern, name.removesuffix(".weight")) for pattern in quantization["ignore"])
    for name, value in preserved.items():
        assert tensors[name].dtype == value.dtype
        assert torch.equal(tensors[name], value), name
    exporter.validate_int8_checkpoint(output, model)
    metadata = json.loads((output / "qlean_config.json").read_text())
    assert metadata["scheme"] == "INT8_CHANNEL"
    assert metadata["output_format"] == "GPTQ_INT8"
    assert metadata["modifier_chain"] == ["GPTQModifier"]
    return tensors, weight_map, quantization


@pytest.mark.parametrize("sharded,is_source", [(False, True), (True, True), (True, False)])
def test_real_llama_channel_int8_codes_scales_and_source_only_save(monkeypatch, tmp_path, sharded, is_source):
    import compressed_tensors.distributed as distributed_module

    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=37, hidden_size=30, intermediate_size=46, num_hidden_layers=1,
        num_attention_heads=3, num_key_value_heads=1, head_dim=10,
        max_position_embeddings=32, use_cache=False, tie_word_embeddings=False,
    )).to(torch.bfloat16)
    source, output = tmp_path / "source", tmp_path / "output"
    model.save_pretrained(source)
    model = LlamaForCausalLM.from_pretrained(source, dtype=torch.bfloat16)
    before_source = _files(source)
    preserved = {
        name: value.clone() for name, value in model.state_dict().items()
        if not name.endswith("_proj.weight")
    }
    record = _observe_export(monkeypatch, model, is_source=is_source)
    completed = []
    monkeypatch.setattr(distributed_module, "is_source_process", lambda: is_source)
    original_metadata = compressor_module.save_qlean_metadata

    def complete(**kwargs):
        completed.append(kwargs)
        return original_metadata(**kwargs)

    monkeypatch.setattr(compressor_module, "save_qlean_metadata", complete)
    compressor = QLeanCompressor(
        ModelArguments(model_name="tiny-int8-llama", model_path=str(source), save_path=str(output),
                       output_format="GPTQ_INT8", max_shard_size="2KB" if sharded else "10GB"),
        model=model, modifiers=[GPTQModifier(scheme="INT8_CHANNEL",
                                             ignore=["lm_head"], block_size=16)],
    )
    compressor.dataloader = [_batch()]
    compressor.post_train(SimpleNamespace())
    assert _files(source) == before_source
    assert record.calls == len(completed) == int(is_source)
    if not is_source:
        assert not output.exists()
        return
    # Rectangular 30x46/46x30 and 10x30 demonstrate no INT4 64-tile restriction.
    assert {tuple(codes.shape) for codes, _ in record.params.values()} >= {(30, 46), (46, 30), (10, 30)}
    assert set(record.params) == {
        f"model.layers.0.{name}.weight" for name in (
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
        )
    }
    tensors, _, config = _assert_int8_output(output, model, record, preserved, sharded=sharded)
    assert set(tensors) == set(preserved) | set(record.params) | {f"{name}_scale" for name in record.params}
    assert tensors["lm_head.weight"].dtype == torch.bfloat16
    assert "lm_head" in config["ignore"]


def test_real_moe_channel_int8_without_int4_export(monkeypatch, tmp_path):
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    torch.manual_seed(0)
    model = Qwen3MoeForCausalLM(Qwen3MoeConfig(
        vocab_size=32, hidden_size=32, intermediate_size=48, moe_intermediate_size=48,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        num_experts=2, num_experts_per_tok=1, max_position_embeddings=32, use_cache=False,
    )).to(torch.bfloat16)
    source, output = tmp_path / "source", tmp_path / "output"
    model.save_pretrained(source)
    model = Qwen3MoeForCausalLM.from_pretrained(source, dtype=torch.bfloat16)
    record = _observe_export(monkeypatch, model)
    compressor = QLeanCompressor(
        ModelArguments(model_name="tiny-int8-moe", model_path=str(source), save_path=str(output),
                       output_format="GPTQ_INT8"), model=model,
        modifiers=[GPTQModifier(scheme="INT8_CHANNEL",
                                 ignore=["lm_head", r"re:.*mlp\.gate$"], block_size=16)],
    )
    compressor.dataloader = [_batch()]
    compressor.post_train(SimpleNamespace())
    assert sum(".experts." in name for name in record.params) == 6
    _assert_int8_output(output, model, record, {}, sharded=False)


def _qwen_linear_source(tmp_path, sharded):
    from transformers import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig, Qwen3_5ForConditionalGeneration

    torch.manual_seed(0)
    config = Qwen3_5Config(
        text_config=Qwen3_5TextConfig(
            vocab_size=32, hidden_size=32, intermediate_size=48, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            linear_key_head_dim=8, linear_value_head_dim=8,
            linear_num_key_heads=2, linear_num_value_heads=2, layer_types=["linear_attention"],
            max_position_embeddings=32, num_mtp_layers=0, mtp_num_hidden_layers=0,
            num_nextn_predict_layers=1, use_cache=False,
            bos_token_id=1, eos_token_id=2, pad_token_id=0, tie_word_embeddings=False,
        ),
        vision_config=Qwen3_5VisionConfig(
            depth=1, hidden_size=16, intermediate_size=16, num_heads=4,
            patch_size=2, spatial_merge_size=1, temporal_patch_size=1,
            out_hidden_size=32, num_position_embeddings=16,
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
    model.name_or_path = str(source)
    return model, source, mtp


@pytest.mark.parametrize("sharded,fail_copy", [(False, False), (True, False), (True, True)])
def test_qwen35_linear_attention_int8_ct_ignore_root_cause_and_staged_preservation(
    monkeypatch, tmp_path, sharded, fail_copy
):
    import qlean.modifiers.save_utils as save_utils

    model, source, mtp = _qwen_linear_source(tmp_path, sharded)
    prefix = "model.language_model.layers.0.linear_attn"
    modules = dict(model.named_modules())
    assert model.config.get_text_config().layer_types == ["linear_attention"]
    for name in (prefix, f"{prefix}.norm"):
        assert not isinstance(modules[name], torch.nn.Linear)
        assert get_vllm_module_type(type(modules[name]).__name__) == "Linear"
    source_files = _files(source)
    source_config = json.loads(source_files["config.json"])
    preserved = {
        name: value.clone() for name, value in model.state_dict().items()
        if not isinstance(modules.get(name.removesuffix(".weight")), torch.nn.Linear)
        or name.startswith("model.visual.") or name == "lm_head.weight"
        or name.removesuffix(".weight") in (f"{prefix}.in_proj_a", f"{prefix}.in_proj_b")
    }
    prepared = []
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
    record = _observe_export(monkeypatch, model, prepared=prepared)
    original_export = exporter.export_int8_checkpoint

    def observe_ignore(directory, source_model, max_shard_size, *, checkpoint=None, expert_patterns=None, workers=2):
        assert workers == 2
        ignore = json.loads((Path(directory) / "config.json").read_text())["quantization_config"]["ignore"]
        assert prefix in ignore and f"{prefix}.norm" in ignore
        for suffix in ("out_proj", "in_proj_qkv", "in_proj_z"):
            assert modules[f"{prefix}.{suffix}"].quantization_scheme.weights.num_bits == 8
        return original_export(
            directory, source_model, max_shard_size, checkpoint=checkpoint, expert_patterns=expert_patterns,
            workers=workers,
        )

    monkeypatch.setattr(exporter, "export_int8_checkpoint", observe_ignore)
    completed = []
    original_metadata = compressor_module.save_qlean_metadata

    def complete(**kwargs):
        completed.append(kwargs)
        return original_metadata(**kwargs)

    monkeypatch.setattr(compressor_module, "save_qlean_metadata", complete)
    if fail_copy:
        def fail(*args, **kwargs):
            raise RuntimeError("MTP copy failed")
        monkeypatch.setattr(save_utils, "save_file", fail)
    output = tmp_path / "output"
    compressor = QLeanCompressor(
        ModelArguments(model_name="tiny-qwen3.5-linear-attention", model_path=str(source),
                       save_path=str(output), output_format="GPTQ_INT8",
                       max_shard_size="4KB" if sharded else "10GB"), model=model,
        modifiers=[GPTQModifier(scheme="INT8_CHANNEL", block_size=16,
                                 ignore=[r"re:model\.visual.*", "lm_head", r"re:.*linear_attn\.in_proj_[ab]$"])],
    )
    compressor.dataloader = [_batch()]
    if fail_copy:
        with pytest.raises(RuntimeError, match="MTP copy failed"):
            compressor.post_train(SimpleNamespace())
    else:
        compressor.post_train(SimpleNamespace())
    assert _files(source) == source_files
    assert len(prepared) == record.calls == 1
    assert len(completed) == int(not fail_copy)
    if fail_copy:
        assert _files(output) == record.intermediate
        assert not (output / "qlean_config.json").exists()
        return
    tensors, weight_map, quantization = _assert_int8_output(
        output, model, record, {**preserved, **mtp}, sharded=True,
    )
    final_config = json.loads((output / "config.json").read_text())
    assert {name: value for name, value in final_config.items() if name != "quantization_config"} == source_config
    ignore = quantization["ignore"]
    assert prefix not in ignore and f"{prefix}.norm" not in ignore
    for suffix in ("in_proj_a", "in_proj_b"):
        assert f"{prefix}.{suffix}" in ignore
        assert tensors[f"{prefix}.{suffix}.weight"].dtype == torch.bfloat16
    assert "lm_head" in ignore
    for name, module in modules.items():
        if name.startswith("model.visual.") and isinstance(module, torch.nn.Linear):
            assert name in ignore
            assert tensors[f"{name}.weight"].dtype == torch.bfloat16
    for suffix in ("out_proj", "in_proj_qkv", "in_proj_z"):
        name = f"{prefix}.{suffix}"
        assert f"{name}.weight" in record.params
        assert tensors[f"{name}.weight"].dtype == torch.int8
        assert tensors[f"{name}.weight_scale"].dtype == torch.float32
        assert not any(_covers(pattern, name) for pattern in ignore)
    assert [item for item in ignore if "mtp" in item] == ["mtp.layers.0.self_attn.q_proj"]
    assert all(weight_map[name] == "model_mtp.safetensors" for name in mtp)
    assert not any(name.startswith("mtp") for name in model.state_dict())


def test_export_api_defaults_are_auto_and_two_workers():
    signatures = {
        name: inspect.signature(getattr(exporter, name)).parameters
        for name in (
            "export_a8w4_checkpoint", "export_int8_checkpoint",
            "export_mxfp4_checkpoint", "finalize_gptq_checkpoint", "_write_staged_shards",
        )
    }
    assert signatures["export_a8w4_checkpoint"]["export_device"].default == "auto"
    assert signatures["finalize_gptq_checkpoint"]["export_device"].default == "auto"
    for name in ("export_int8_checkpoint", "export_mxfp4_checkpoint", "finalize_gptq_checkpoint"):
        assert signatures[name]["workers"].default == 2
    assert signatures["_write_staged_shards"]["workers"].default == 1
    assert signatures["_write_staged_shards"]["label"].default is None


@pytest.mark.parametrize("workers", [0, -1, True, False, 1.5, "2", None])
def test_export_workers_require_positive_non_bool_integer(workers):
    with pytest.raises((TypeError, ValueError), match="worker|positive|integer"):
        exporter.validate_export_workers(workers)


@pytest.mark.parametrize("workers", [1, 2, 7])
def test_export_workers_accept_positive_integers(workers):
    assert exporter.validate_export_workers(workers) == workers


@pytest.mark.parametrize("relation", ["same", "child", "parent", "symlink-child"])
def test_gptq_source_output_overlap_is_rejected_before_calibration(monkeypatch, tmp_path, relation):
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=32, intermediate_size=48,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
    )).to(torch.bfloat16)
    source = tmp_path / "source"
    model.save_pretrained(source)
    model.name_or_path = str(source)
    if relation == "same":
        output = source
    elif relation == "child":
        output = source / "not-created" / "output"
    elif relation == "parent":
        output = tmp_path
    else:
        alias = tmp_path / "alias"
        alias.symlink_to(source, target_is_directory=True)
        output = alias / "not-created" / "output"
    before = _files(source)
    compressor = QLeanCompressor(
        ModelArguments(model_name="overlap", model_path=str(source), save_path=str(output), output_format="GPTQ_INT8"),
        model=model, modifiers=[GPTQModifier(scheme="INT8_CHANNEL")],
    )
    compressor.dataloader = [_batch()]

    def fail(*args, **kwargs):
        pytest.fail("Overlapping paths must fail before constructing/running calibration")

    monkeypatch.setattr(compressor_module, "get_pipeline_from_modifiers", fail)
    with pytest.raises(ValueError, match="[Oo]verlap|source|overwrite"):
        compressor.post_train(SimpleNamespace())
    assert _files(source) == before


@pytest.mark.parametrize("scheme", ["INT4_CHANNEL", "INT8_CHANNEL", "MXFP4"])
@pytest.mark.parametrize("relation", ["same", "child", "parent", "symlink-child", "disjoint"])
def test_loaded_hub_snapshot_output_guard_is_local_and_pinned(monkeypatch, tmp_path, scheme, relation):
    import transformers.utils as transformer_utils

    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=32, intermediate_size=48,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
    ))
    revision = "1" * 40
    model.config._commit_hash = revision
    model.name_or_path = "test/hub-source"
    source = tmp_path / "snapshots" / revision
    source.mkdir(parents=True)
    blob = tmp_path / "config-blob"
    blob.write_text("{}")
    config_path = source / "config.json"
    config_path.symlink_to(blob)
    if relation == "same":
        output = source
    elif relation == "child":
        output = source / "not-created" / "output"
    elif relation == "parent":
        output = source.parent
    elif relation == "symlink-child":
        alias = tmp_path / "alias"
        alias.symlink_to(source, target_is_directory=True)
        output = alias / "not-created" / "output"
    else:
        output = tmp_path / "output"
    calls = []

    def cached_file(repo, filename, **kwargs):
        assert repo == model.name_or_path and filename == "config.json"
        assert kwargs["revision"] == revision and kwargs["local_files_only"] is True
        calls.append(repo)
        return str(config_path)

    monkeypatch.setattr(transformer_utils, "cached_file", cached_file)
    modifier = GPTQModifier(scheme=scheme)
    compressor = QLeanCompressor(
        ModelArguments(model_name="hub-output-guard", model_path=model.name_or_path,
                       save_path=str(output), output_format="GPTQ_INT8"),
        model=model, modifiers=[modifier],
    )
    if relation == "disjoint":
        exporter.validate_gptq_output_directory(model, model.name_or_path, str(output), [modifier])
        assert not output.exists()
    else:
        def fail(*args, **kwargs):
            pytest.fail("Cached source overlap must fail before calibration")
        monkeypatch.setattr(compressor_module, "get_pipeline_from_modifiers", fail)
        with pytest.raises(ValueError, match="overlap"):
            compressor.post_train(SimpleNamespace())
    assert calls == [model.name_or_path]
    assert _files(source) == {"config.json": b"{}"}
    assert blob.read_bytes() == b"{}"


@pytest.mark.parametrize("artifact", ["model.safetensors", "model.safetensors.index.json"])
def test_channel_int8_fresh_output_guard_runs_before_calibration(monkeypatch, tmp_path, artifact):
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=32, intermediate_size=48,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
    )).to(torch.bfloat16)
    source, output = tmp_path / "source", tmp_path / "output"
    model.save_pretrained(source)
    output.mkdir()
    (output / artifact).write_bytes(b"existing-checkpoint-do-not-overwrite")
    (output / "config.json").write_text(json.dumps({"sentinel": True}))
    before = _files(output)
    compressor = QLeanCompressor(
        ModelArguments(model_name="fresh-output", model_path=str(source), save_path=str(output), output_format="GPTQ_INT8"),
        model=model, modifiers=[GPTQModifier(scheme="INT8_CHANNEL")],
    )
    compressor.dataloader = [_batch()]

    def fail(*args, **kwargs):
        pytest.fail("Existing checkpoints must fail before calibration")

    monkeypatch.setattr(compressor_module, "get_pipeline_from_modifiers", fail)
    with pytest.raises(ValueError, match="fresh checkpoint output directory"):
        compressor.post_train(SimpleNamespace())
    assert _files(output) == before
