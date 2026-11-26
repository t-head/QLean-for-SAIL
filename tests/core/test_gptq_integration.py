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
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

from qlean import QLeanCompressor
from qlean.args import DatasetArguments, ModelArguments, OtherArguments
from qlean.modifiers.gptq import GPTQModifier
from qlean.utils.pack_acext_int4 import preprocess_weight_for_a8w4_torch


pytestmark = [pytest.mark.gptq, pytest.mark.integration, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def reset_compressor_singleton():
    QLeanCompressor._instances.pop(QLeanCompressor, None)
    yield
    QLeanCompressor._instances.pop(QLeanCompressor, None)


def _batch():
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
        "attention_mask": torch.ones((1, 8), dtype=torch.long),
    }


def _read_checkpoint(directory):
    tensors = {}
    for shard in directory.glob("*.safetensors"):
        values = load_file(shard)
        assert not tensors.keys() & values.keys()
        tensors.update(values)
    return tensors


@pytest.mark.parametrize("scheme,output_format", [
    ("INT8_CHANNEL", "GPTQ_INT8"),
    ("MXFP4", "GPTQ_MXFP4"),
])
@pytest.mark.parametrize("with_mtp", [False, True])
def test_int8_and_mxfp4_checkpoint_payload(
    monkeypatch, tmp_path, scheme, output_format, with_mtp,
    assert_mxfp4_payload_reconstructs,
):
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=128, hidden_size=128, intermediate_size=128,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=32, use_cache=False,
    )).to(torch.bfloat16)
    source = tmp_path / "source"
    output = tmp_path / "output"
    if with_mtp:
        model.config.num_mtp_layers = 0
        model.config.mtp_num_hidden_layers = 1
    model.save_pretrained(source)
    mtp = {"mtp.layers.0.weight": torch.randn(8, 8, dtype=torch.bfloat16)} if with_mtp else {}
    if mtp:
        save_file({**load_file(source / "model.safetensors"), **mtp}, source / "model.safetensors")
    # save_pretrained may mutate config.dtype. Always use a fresh BF16 load for calibration.
    model = LlamaForCausalLM.from_pretrained(source, dtype=torch.bfloat16)
    assert next(model.parameters()).dtype == torch.bfloat16
    source_files = {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()}
    original_state = {name: value.clone() for name, value in model.state_dict().items()}
    dense_keys = {
        f"model.layers.0.{name}.weight" for name in (
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
        )
    }
    snapshots = {}
    if scheme == "INT8_CHANNEL":
        import qlean.modifiers.gptq.export as exporter

        original_export = exporter.export_int8_checkpoint

        def observe_export(directory, source_model, max_shard_size, *, checkpoint=None, expert_patterns=None, workers=2):
            assert source_model is model and not snapshots
            assert workers == 2
            intermediate = _read_checkpoint(Path(directory))
            config = json.loads((Path(directory) / "config.json").read_text())["quantization_config"]
            assert config["quant_method"] == "compressed-tensors"
            assert config["format"] == "int-quantized"
            for name in dense_keys:
                codes, scale = intermediate[name], intermediate[f"{name}_scale"]
                assert codes.dtype == torch.int8 and codes.shape == original_state[name].shape
                assert scale.dtype == torch.bfloat16 and scale.shape == (codes.shape[0], 1)
                snapshots[name] = (codes.clone(), scale.clone())
            return original_export(
                directory, source_model, max_shard_size, checkpoint=checkpoint,
                expert_patterns=expert_patterns, workers=workers,
            )

        monkeypatch.setattr(exporter, "export_int8_checkpoint", observe_export)
    else:
        import qlean.core.checkpoint as checkpoint_module
        import qlean.modifiers.gptq.export as exporter

        original_export = exporter.export_mxfp4_checkpoint

        def observe_export(directory, source_model, max_shard_size, *, checkpoint=None, expert_patterns=None, workers=2):
            assert source_model is model and not snapshots
            assert workers == 2
            directory = Path(directory)
            intermediate = _read_checkpoint(directory)
            index = json.loads((directory / "model.safetensors.index.json").read_text())
            affected = {index["weight_map"][f"{name}_packed"] for name in dense_keys}
            before_shards = {path.name: path.read_bytes() for path in directory.glob("*.safetensors")}
            unaffected = {name: payload for name, payload in before_shards.items() if name not in affected}
            assert affected and unaffected
            for name in dense_keys:
                snapshots[name] = (
                    intermediate[f"{name}_packed"].clone(),
                    intermediate[f"{name}_scale"].clone(),
                )
            result = original_export(
                directory, source_model, max_shard_size, checkpoint=checkpoint,
                expert_patterns=expert_patterns, workers=workers,
            )
            final_map = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
            assert all(final_map[name.removesuffix("_packed")] == filename
                       for name, filename in index["weight_map"].items())
            after_shards = {path.name: path.read_bytes() for path in directory.glob("*.safetensors")}
            assert {name for name, payload in before_shards.items() if after_shards[name] != payload} == affected
            for filename, payload in unaffected.items():
                assert after_shards[filename] == payload
            return result

        def forbidden_shared_rename(*args, **kwargs):
            pytest.fail("GPTQ MXFP4 must use its isolated staged exporter")

        monkeypatch.setattr(exporter, "export_mxfp4_checkpoint", observe_export)
        monkeypatch.setattr(checkpoint_module, "rename_packed_weights", forbidden_shared_rename)
    compressor = QLeanCompressor(
        ModelArguments(model_name="tiny-llama", model_path=str(source),
                       save_path=str(output), output_format=output_format, max_shard_size="8KB"),
        model=model, modifiers=[GPTQModifier(scheme=scheme, targets=["Linear"],
                                             ignore=["lm_head"], block_size=32)],
    )
    compressor.dataloader = [_batch()]
    compressor.post_train(SimpleNamespace())

    quantization = json.loads((output / "config.json").read_text())["quantization_config"]
    assert quantization["quant_method"] == ("mxfp4" if scheme == "MXFP4" else "w8a8_int8")
    assert not {"format", "config_groups", "kv_cache_scheme"} & quantization.keys()
    metadata = json.loads((output / "qlean_config.json").read_text())
    assert metadata["scheme"] == scheme
    assert metadata["output_format"] == output_format
    assert metadata["modifier_chain"] == ["GPTQModifier"]
    tensors = _read_checkpoint(output)
    preserved = {name: value for name, value in original_state.items() if name not in dense_keys}
    assert set(tensors) == set(preserved) | set(mtp) | dense_keys | {f"{name}_scale" for name in dense_keys}
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(tensors)
    assert index["metadata"]["total_size"] == sum(value.numel() * value.element_size() for value in tensors.values())
    assert set(index["weight_map"].values()) == {path.name for path in output.glob("*.safetensors")}
    for filename in set(index["weight_map"].values()):
        assert set(load_file(output / filename)) == {
            name for name, shard in index["weight_map"].items() if shard == filename
        }
    for name in dense_keys:
        weight, scale = tensors[name], tensors[f"{name}_scale"]
        if scheme == "INT8_CHANNEL":
            codes, original_scale = snapshots[name]
            assert weight.dtype == torch.int8 and weight.shape == original_state[name].shape
            assert torch.equal(weight, codes)
            assert scale.dtype == torch.float32 and scale.shape == (codes.shape[0], 1)
            assert torch.equal(scale, original_scale.float())
            assert torch.isfinite(scale).all() and (scale > 0).all()
        else:
            n, k = original_state[name].shape
            packed, original_scale = snapshots[name]
            assert weight.dtype == torch.uint8 and weight.shape == (n, k // 2)
            assert scale.dtype == torch.uint8 and scale.shape == (n, k // 32)
            assert torch.equal(weight, packed) and torch.equal(scale, original_scale)
            assert weight.shape[-1] == scale.shape[-1] * 16
        assert not any(f"{name}{suffix}" in tensors for suffix in (
            "_packed", "_shape", "_zero_point", "_g_idx", "_global_scale",
        ))
        assert name.removesuffix(".weight") not in quantization["ignore"]
    for name, value in preserved.items():
        assert tensors[name].dtype == value.dtype and torch.equal(tensors[name], value)
    assert {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()} == source_files
    if with_mtp:
        restored = load_file(output / "model_mtp.safetensors")
        assert restored.keys() == mtp.keys()
        index = json.loads((output / "model.safetensors.index.json").read_text())
        for name, value in mtp.items():
            assert torch.equal(restored[name], value)
            assert restored[name].dtype == value.dtype
            assert index["weight_map"][name] == "model_mtp.safetensors"
        assert "mtp.layers.0" in quantization["ignore"]
        assert not any(item.startswith("re:") and "mtp" in item for item in quantization["ignore"])
        assert not any(name.startswith("mtp") for name in model.state_dict())
    else:
        assert not (output / "model_mtp.safetensors").exists()
    if scheme == "MXFP4":
        # Like w8a8_int8, mxfp4 is a producer contract, not a standard CT/HF reload path.
        assert_mxfp4_payload_reconstructs(tensors, original_state, dense_keys)


@pytest.mark.parametrize("bits,scheme,output_format", [
    (4, "INT4_GROUP", "GPTQ_INT4_GROUP"),
    (8, "INT8_GROUP", "GPTQ_INT8_GROUP"),
])
def test_group_checkpoint_is_standard_pack_quantized(
    tmp_path, bits, scheme, output_format
):
    from compressed_tensors.compressors.pack_quantized import PackedQuantizationCompressor

    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=128, hidden_size=128, intermediate_size=128,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=32, use_cache=False,
    )).to(torch.bfloat16)
    source = tmp_path / "source"
    output = tmp_path / "output"
    model.save_pretrained(source)
    model = LlamaForCausalLM.from_pretrained(source, dtype=torch.bfloat16)
    original_state = {name: value.clone() for name, value in model.state_dict().items()}
    dense_keys = {
        f"model.layers.0.{name}.weight" for name in (
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
        )
    }
    compressor = QLeanCompressor(
        ModelArguments(model_name="tiny-llama", model_path=str(source),
                       save_path=str(output), output_format=output_format, max_shard_size="8KB"),
        model=model, modifiers=[GPTQModifier(scheme=scheme, targets=["Linear"],
                                             ignore=["lm_head"], block_size=32)],
    )
    compressor.dataloader = [_batch()]
    compressor.post_train(SimpleNamespace())

    quantization = json.loads((output / "config.json").read_text())["quantization_config"]
    assert quantization["quant_method"] == "compressed-tensors"
    assert quantization["format"] == "pack-quantized"
    group, = quantization["config_groups"].values()
    scheme_args = QuantizationScheme.model_validate(group)
    weights = scheme_args.weights
    assert weights.strategy == "group"
    assert weights.group_size == 128
    assert weights.block_structure is None
    assert weights.num_bits == bits and weights.type == "int" and weights.symmetric
    assert group["weights"].get("scale_dtype") is None
    metadata = json.loads((output / "qlean_config.json").read_text())
    assert metadata["scheme"] == scheme
    assert metadata["output_format"] == output_format

    tensors = _read_checkpoint(output)
    preserved = {name: value for name, value in original_state.items() if name not in dense_keys}
    packed_suffixes = (".weight_packed", ".weight_scale", ".weight_shape")
    assert {name for name in tensors if name.endswith(packed_suffixes)} == {
        f"{name.removesuffix('.weight')}{suffix}" for name in dense_keys for suffix in packed_suffixes
    }
    for name in preserved:
        assert torch.equal(tensors[name], preserved[name])
    for name in dense_keys:
        packed = tensors[f"{name}_packed"]
        scale = tensors[f"{name}_scale"]
        shape = tensors[f"{name}_shape"]
        assert packed.dtype == torch.int32
        assert packed.shape == (128, 128 * bits // 32)
        assert scale.dtype == torch.bfloat16 and scale.shape == (128, 1)
        assert torch.isfinite(scale).all() and (scale > 0).all()
        assert shape.tolist() == [128, 128]
        assert not any(f"{name}{suffix}" in tensors for suffix in (
            "_zero_point", "_g_idx", "_global_scale",
        ))
        module_name = name.removesuffix(".weight")
        local = {
            key.removeprefix(f"{module_name}."): value
            for key, value in tensors.items() if key.startswith(f"{module_name}.")
        }
        decompressed = PackedQuantizationCompressor.decompress(local, scheme_args)
        weight = decompressed["weight"]
        assert weight.shape == original_state[name].shape
        assert weight.dtype == torch.bfloat16
        assert torch.isfinite(weight).all()
        max_mean_error = 0.25 if bits == 4 else 0.05
        assert (weight.float() - original_state[name].float()).abs().mean() < max_mean_error


def _tiny_expert_model(expert_size=64):
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    torch.manual_seed(0)
    return Qwen3MoeForCausalLM(Qwen3MoeConfig(
        vocab_size=128, hidden_size=64, intermediate_size=64,
        moe_intermediate_size=expert_size, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=4, head_dim=16,
        num_experts=2, num_experts_per_tok=1, max_position_embeddings=32,
        use_cache=False,
    )).to(torch.bfloat16)


def _expert_groups(with_dense_int8=False):
    groups = {"experts": QuantizationScheme(
        targets=[r"re:.*mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"],
        weights=QuantizationArgs(num_bits=4, type="int", symmetric=True, strategy="channel"),
    )}
    if with_dense_int8:
        groups["attention"] = QuantizationScheme(
            targets=[r"re:.*self_attn\.q_proj$"],
            weights=QuantizationArgs(num_bits=8, type="int", symmetric=True, strategy="channel"),
        )
    return groups


def _observe_real_export(monkeypatch, model):
    import qlean.core.checkpoint as checkpoint_module
    import qlean.modifiers.gptq.export as exporter

    record = SimpleNamespace(calls=0, params={})
    original = exporter.export_a8w4_checkpoint

    def export(directory, source_model, max_shard_size, *, checkpoint=None, expert_patterns=None,
               export_device="auto", process_group=None):
        assert source_model is model
        assert export_device == "cpu" and process_group is None
        assert record.calls == 0
        record.calls += 1
        directory = Path(directory)
        config = json.loads((directory / "config.json").read_text())["quantization_config"]
        assert config["quant_method"] == "compressed-tensors"
        assert config["format"] == "pack-quantized"
        tensors = _read_checkpoint(directory)
        modules = exporter.a8w4_quantized_modules(model, require_int4=True)
        for key, packed in tensors.items():
            if not key.endswith(".weight_packed"):
                continue
            name = key.removesuffix("_packed")
            bits = modules[name.removesuffix(".weight")].quantization_scheme.weights.num_bits
            shape = torch.Size(tensors[f"{name}_shape"].tolist())
            record.params[name] = {
                "bits": bits, "codes": unpack_from_int32(packed, bits, shape),
                "scale": tensors[f"{name}_scale"].clone(),
            }
        assert record.params
        return original(
            directory, source_model, max_shard_size, checkpoint=checkpoint, expert_patterns=expert_patterns,
            export_device=export_device, process_group=process_group,
        )

    monkeypatch.setattr(exporter, "export_a8w4_checkpoint", export)
    original_rename = checkpoint_module.rename_packed_weights

    def rename(directory):
        assert record.calls == 1, "A8W4 must consume CT keys before generic packed-key renaming"
        return original_rename(directory)

    monkeypatch.setattr(checkpoint_module, "rename_packed_weights", rename)
    return record


@pytest.mark.parametrize("with_mtp", [False, True])
@pytest.mark.parametrize("with_dense_int8", [False, True])
def test_real_moe_int4_mixed_int8_mtp_export_without_native(
    monkeypatch, tmp_path, with_mtp, with_dense_int8
):
    monkeypatch.setitem(sys.modules, "acext", None)
    model = _tiny_expert_model(expert_size=128)
    source = tmp_path / "source"
    output = tmp_path / "output"
    if with_mtp:
        model.config.num_mtp_layers = 1
    model.save_pretrained(source)
    mtp = {
        "mtp.layers.0.weight": torch.arange(16).reshape(4, 4).to(torch.bfloat16),
        "mtp.layers.0.weight_scale": torch.ones(4, dtype=torch.bfloat16),
    } if with_mtp else {}
    if mtp:
        save_file({**load_file(source / "model.safetensors"), **mtp}, source / "model.safetensors")
    model = AutoModelForCausalLM.from_pretrained(source, dtype=torch.bfloat16)
    unchanged = {
        name: tensor.clone() for name, tensor in model.state_dict().items()
        if ".experts." not in name and not (with_dense_int8 and "self_attn.q_proj" in name)
    }
    record = _observe_real_export(monkeypatch, model)
    compressor = QLeanCompressor(
        ModelArguments(model_name="tiny-qwen3-moe", model_path=str(source),
                       save_path=str(output), output_format="GPTQ_INT4", max_shard_size="8KB",
                       gptq_int4_export_device="cpu"),
        model=model,
        modifiers=[GPTQModifier(config_groups=_expert_groups(with_dense_int8), block_size=32)],
    )
    compressor.dataloader = [_batch()]
    assert not output.exists()
    compressor.post_train(SimpleNamespace())

    assert record.calls == 1
    tensors = _read_checkpoint(output)
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(tensors)
    for filename in set(index["weight_map"].values()):
        assert all(index["weight_map"][key] == filename for key in load_file(output / filename))
    config = json.loads((output / "config.json").read_text())["quantization_config"]
    assert config["quant_method"] == "mixed_precision_w4"
    assert config["activation_scheme"] == "dynamic"
    assert set(config) == {"quant_method", "activation_scheme", "ignore"}
    metadata = json.loads((output / "qlean_config.json").read_text())
    assert metadata["output_format"] == "GPTQ_INT4"
    assert metadata["scheme"] == "INT4_CHANNEL"
    assert metadata["modifier_chain"] == ["GPTQModifier"]
    assert sum(params["bits"] == 4 for params in record.params.values()) == 6
    assert sum(params["bits"] == 8 for params in record.params.values()) == int(with_dense_int8)
    for name, params in record.params.items():
        codes = params["codes"]
        if params["bits"] == 4:
            assert codes.shape in [(128, 64), (64, 128)]
            expected = preprocess_weight_for_a8w4_torch(codes).view(codes.shape[0], codes.shape[1] // 2)
        else:
            expected = codes
        assert tensors[name].dtype == torch.int8
        assert torch.equal(tensors[name], expected)
        assert tensors[f"{name}_scale"].dtype == torch.float32
        assert torch.equal(tensors[f"{name}_scale"], params["scale"].float())
        assert torch.isfinite(tensors[f"{name}_scale"]).all()
        assert (tensors[f"{name}_scale"] > 0).all()
        assert name.removesuffix(".weight") not in config["ignore"]
        assert not any(f"{name}{suffix}" in tensors for suffix in (
            "_packed", "_shape", "_zero_point", "_g_idx", "_global_scale",
        ))
    for name, tensor in {**unchanged, **mtp}.items():
        assert torch.equal(tensors[name], tensor)
        assert tensors[name].dtype == tensor.dtype
    if with_mtp:
        assert "mtp.layers.0" in config["ignore"]
        assert not any(item.startswith("re:") and "mtp" in item for item in config["ignore"])
        assert load_file(output / "model_mtp.safetensors").keys() == mtp.keys()
        assert all(index["weight_map"][name] == "model_mtp.safetensors" for name in mtp)
        assert not any(name.startswith("mtp") for name in model.state_dict())
    else:
        assert not (output / "model_mtp.safetensors").exists()
    # Producer-only mixed_precision_w4 contract: no standard CT/HF reload claim.


@pytest.mark.parametrize("with_int8", [False, True])
@pytest.mark.parametrize("use_recipe", [False, True])
def test_dense_int4_public_save_is_rejected_before_calibration(
    monkeypatch, tmp_path, with_int8, use_recipe
):
    monkeypatch.setitem(sys.modules, "acext", None)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=32, use_cache=False,
    )).to(torch.bfloat16)
    source, output = tmp_path / "source", tmp_path / "output"
    model.save_pretrained(source)
    before = {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()}
    model = LlamaForCausalLM.from_pretrained(source, dtype=torch.bfloat16)
    groups = {"denseINT4": {
        "targets": [r"re:.*(?:self_attn|mlp).*"],
        "weights": {"num_bits": 4, "type": "int", "symmetric": True, "strategy": "channel"},
    }}
    if with_int8:
        groups["attentionINT8"] = {
            "targets": [r"re:.*self_attn\.q_proj$"],
            "weights": {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "channel"},
        }
    options = {"config_groups": groups, "ignore": ["lm_head"], "block_size": 32}
    recipe = tmp_path / "dense-int4.yaml"
    if use_recipe:
        recipe.write_text(json.dumps({"quant_stage": {"quant_modifiers": {"GPTQModifier": options}}}))
        monkeypatch.setattr("qlean.modifiers.ModifierFactory._loaded", True)
    compressor = QLeanCompressor(
        ModelArguments(model_name="tiny-dense", model_path=str(source), save_path=str(output),
                       output_format="GPTQ_INT4", recipe=str(recipe) if use_recipe else None),
        model=model, modifiers=None if use_recipe else [GPTQModifier(**options)],
    )
    compressor.set_calibration_dataset(DatasetArguments(
        dataset=[{name: value.squeeze(0) for name, value in _batch().items()}], batch_size=1,
    ))

    with pytest.raises(ValueError, match="only routed MoE expert") as raised:
        compressor.post_train(OtherArguments())

    message = str(raised.value)
    assert "dense or non-expert module(s) remain selected" in message
    assert "model.layers.0.self_attn" in message
    assert "targets=[GPTQ_EXPERT_TARGET]" in message
    assert "GPTQModifier.ignore" in message
    assert not output.exists()
    assert {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()} == before


def test_invalid_expert_shape_fails_before_calibration_hooks(monkeypatch):
    monkeypatch.setitem(sys.modules, "acext", None)
    model = _tiny_expert_model(expert_size=96)
    modifier = GPTQModifier(config_groups=_expert_groups(), block_size=32)
    with pytest.raises(ValueError, match="64"):
        modifier.on_initialize(SimpleNamespace(model=SimpleNamespace(model_=model)))
    assert not modifier._hessians
    assert not modifier._num_samples
    assert modifier._moe_calibration_ctx is None
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())


def test_valid_expert_initialization_needs_no_native(monkeypatch):
    monkeypatch.setitem(sys.modules, "acext", None)
    model = _tiny_expert_model()
    modifier = GPTQModifier(config_groups=_expert_groups(), block_size=32)
    modifier.on_initialize(SimpleNamespace(model=SimpleNamespace(model_=model)))
    assert not modifier._hessians
    assert not modifier._num_samples
    assert modifier._moe_calibration_ctx is None


@pytest.mark.parametrize("second_scheme", ["kv", "activation", "grouped_int8"])
def test_other_modifier_is_validated_before_calibration_hooks(monkeypatch, second_scheme):
    model = _tiny_expert_model()
    state = SimpleNamespace(model=SimpleNamespace(model_=model))
    modifier = GPTQModifier(config_groups=_expert_groups(), block_size=32)
    modifier.on_initialize(state)
    target = "model.layers.0.self_attn.q_proj"
    second = GPTQModifier(scheme="INT8_CHANNEL", targets=[target])
    second.on_initialize(state)
    module = dict(model.named_modules())[target]
    # Initialize a supported companion, then attach an unsupported scheme as if
    # another modifier changed it. The first modifier must revalidate at start.
    if second_scheme == "kv":
        module.kv_cache_scheme = QuantizationArgs(num_bits=8, type="int", strategy="tensor")
        match = "KV cache quantization"
    elif second_scheme == "activation":
        module.quantization_scheme = QuantizationScheme(
            targets=[target], input_activations=QuantizationArgs(
                num_bits=8, type="int", strategy="token", dynamic=True,
            ),
        )
        match = "activation quantization"
    else:
        module.quantization_scheme = QuantizationScheme(
            targets=[target], weights=QuantizationArgs(
                num_bits=8, type="int", strategy="group", group_size=32,
            ),
        )
        match = "symmetric per-channel"
    hooks = {
        module: (dict(module._forward_hooks), dict(module._forward_pre_hooks))
        for module in model.modules()
    }
    with pytest.raises(ValueError, match=match):
        modifier.on_start(state)
    assert not modifier._hessians
    assert not modifier._num_samples
    assert modifier._moe_calibration_ctx is None
    assert hooks == {
        module: (dict(module._forward_hooks), dict(module._forward_pre_hooks))
        for module in model.modules()
    }


def test_own_kv_scheme_is_rejected_before_calibration():
    model = _tiny_expert_model()
    modifier = GPTQModifier(
        config_groups=_expert_groups(),
        kv_cache_scheme=QuantizationArgs(num_bits=8, type="int", strategy="tensor"),
    )
    with pytest.raises(ValueError, match="GPTQ does not support KV-cache quantization"):
        modifier.on_initialize(SimpleNamespace(model=SimpleNamespace(model_=model)))
    assert not modifier._hessians and not modifier._num_samples
    assert modifier._moe_calibration_ctx is None
    assert not modifier.initialized_
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())


@pytest.mark.parametrize("companion", ["asymmetric", "grouped", "activation", "kv"])
def test_channel_int8_rejects_incompatible_companion_before_calibration(companion):
    model = torch.nn.Module()
    model.dense = torch.nn.Linear(32, 24, bias=False)
    model.other = torch.nn.Linear(32, 24, bias=False)
    weights = QuantizationArgs(num_bits=8, type="int", strategy="channel", symmetric=True)
    groups = {"dense": QuantizationScheme(targets=["dense"], weights=weights)}
    kwargs = {}
    if companion in ("asymmetric", "grouped"):
        arguments = {"num_bits": 8, "type": "int", "strategy": "channel", "symmetric": True}
        if companion == "asymmetric":
            arguments["symmetric"] = False
        else:
            arguments.update(strategy="group", group_size=16)
        groups["other"] = QuantizationScheme(targets=["other"], weights=QuantizationArgs(**arguments))
    elif companion == "activation":
        groups["other"] = QuantizationScheme(
            targets=["other"], input_activations=QuantizationArgs(
                num_bits=8, type="int", strategy="token", dynamic=True,
            ),
        )
    else:
        kwargs["kv_cache_scheme"] = QuantizationArgs(num_bits=8, type="int", strategy="tensor")
    modifier = GPTQModifier(config_groups=groups, **kwargs)
    state = SimpleNamespace(model=SimpleNamespace(model_=model))
    match = (
        "Unsupported GPTQ weights" if companion in ("asymmetric", "grouped")
        else "GPTQ does not support activation quantization" if companion == "activation"
        else "GPTQ does not support KV-cache quantization"
    )
    with pytest.raises(ValueError, match=match):
        modifier.on_initialize(state)
    assert not modifier._hessians and not modifier._num_samples
    assert modifier._moe_calibration_ctx is None
    assert not modifier.initialized_
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())


def test_channel_int8_without_matching_targets_cannot_report_completion(monkeypatch, tmp_path):
    import qlean.core.qleancompressor as compressor_module

    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=32, intermediate_size=48,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
    )).to(torch.bfloat16)
    source, output = tmp_path / "source", tmp_path / "output"
    model.save_pretrained(source)
    modifier = GPTQModifier(config_groups={"channel": QuantizationScheme(
        targets=[r"re:^no_such_layer$"],
        weights=QuantizationArgs(num_bits=8, type="int", strategy="channel", symmetric=True),
    )})
    completed = []
    monkeypatch.setattr(compressor_module, "save_qlean_metadata", lambda **kwargs: completed.append(kwargs))
    compressor = QLeanCompressor(
        ModelArguments(model_name="missing-int8-target", model_path=str(source),
                       save_path=str(output), output_format="GPTQ_INT8"),
        model=model, modifiers=[modifier],
    )
    compressor.dataloader = [_batch()]
    with pytest.raises((ValueError, RuntimeError), match="target|match|INT8|module"):
        compressor.post_train(SimpleNamespace())
    assert not completed
    assert not (output / "qlean_config.json").exists()
    assert not modifier._hessians and not modifier._num_samples


@pytest.mark.parametrize("scheme", ["W8A16", "MXFP4A16", "W4A8"])
def test_legacy_named_scheme_is_rejected_before_calibration(scheme):
    model = torch.nn.Sequential(torch.nn.Linear(128, 128, bias=False))
    modifier = GPTQModifier(scheme=scheme)
    with pytest.raises(ValueError, match="Unsupported GPTQ scheme.*supported schemes"):
        modifier.on_initialize(SimpleNamespace(model=SimpleNamespace(model_=model)))
    assert not modifier._hessians and not modifier._num_samples
    assert modifier._moe_calibration_ctx is None
    assert not modifier.initialized_
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())


@pytest.mark.parametrize("group_size", [16, 32, 64])
def test_non_128_group_int8_config_is_rejected_before_calibration(group_size):
    model = torch.nn.Sequential(torch.nn.Linear(128, 128, bias=False))
    modifier = GPTQModifier(config_groups={"grouped": QuantizationScheme(
        targets=["Linear"], weights=QuantizationArgs(
            num_bits=8, type="int", strategy="group", group_size=group_size, symmetric=True,
        ),
    )})
    with pytest.raises(ValueError, match="Unsupported GPTQ weights.*supported schemes"):
        modifier.on_initialize(SimpleNamespace(model=SimpleNamespace(model_=model)))
    assert not modifier._hessians and not modifier._num_samples
    assert modifier._moe_calibration_ctx is None
    assert not modifier.initialized_
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())


@pytest.mark.parametrize("failure", ["missing", "registry"])
@pytest.mark.parametrize("activation", ["input_activations", "output_activations"])
def test_static_activation_config_is_rejected_before_observer_lookup(monkeypatch, failure, activation):
    import types

    module_name = "compressed_tensors.quantization.observer"
    observer_calls = []
    if failure == "missing":
        monkeypatch.setitem(sys.modules, module_name, None)
    else:
        observer_module = types.ModuleType(module_name)

        class FailingObserver:
            @staticmethod
            def load_from_registry(*args, **kwargs):
                observer_calls.append((args, kwargs))
                raise RuntimeError("activation observer registry failed")

        observer_module.Observer = FailingObserver
        monkeypatch.setitem(sys.modules, module_name, observer_module)
    model = torch.nn.Sequential(torch.nn.Linear(128, 128, bias=False))
    # Static activations are unsupported regardless of observer availability.
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(num_bits=8, type="int", strategy="group", group_size=128),
        **{activation: QuantizationArgs(num_bits=8, type="int", strategy="tensor", dynamic=False)},
    )
    modifier = GPTQModifier(config_groups={"static_activation": scheme})
    with pytest.raises(ValueError, match="GPTQ does not support activation quantization"):
        modifier.on_initialize(SimpleNamespace(model=SimpleNamespace(model_=model)))
    assert not observer_calls
    assert not modifier._hessians and not modifier._num_samples
    assert modifier._moe_calibration_ctx is None
    assert not modifier.initialized_
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())


@pytest.mark.parametrize("companion", ["grouped", "activation", "kv"])
def test_channel_int8_revalidates_other_modifier_before_starting_hooks(monkeypatch, companion):
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=32, intermediate_size=48,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
    )).to(torch.bfloat16)
    state = SimpleNamespace(model=SimpleNamespace(model_=model))
    first = GPTQModifier(scheme="INT8_CHANNEL", targets=[r"re:.*self_attn\.q_proj$"])
    first.on_initialize(state)
    target = "model.layers.0.self_attn.v_proj"
    second = GPTQModifier(scheme="INT8_CHANNEL", targets=[target])
    second.on_initialize(state)
    module = dict(model.named_modules())[target]
    # Simulate a companion changing an attached scheme after valid initialization.
    if companion == "kv":
        module.kv_cache_scheme = QuantizationArgs(num_bits=8, type="int", strategy="tensor")
    elif companion == "grouped":
        module.quantization_scheme = QuantizationScheme(targets=[target], weights=QuantizationArgs(
            num_bits=8, type="int", strategy="group", group_size=16,
        ))
    else:
        module.quantization_scheme = QuantizationScheme(targets=[target], input_activations=QuantizationArgs(
            num_bits=8, type="int", strategy="token", dynamic=True,
        ))
    hooks = {
        module: (dict(module._forward_hooks), dict(module._forward_pre_hooks))
        for module in model.modules()
    }
    with pytest.raises(ValueError, match="channel|activation|KV"):
        first.on_start(state)
    assert not first._hessians and not first._num_samples
    assert first._moe_calibration_ctx is None
    assert hooks == {
        module: (dict(module._forward_hooks), dict(module._forward_pre_hooks))
        for module in model.modules()
    }


@pytest.mark.parametrize("do_sample", [False, True])
@pytest.mark.parametrize("failure", [None, "save", "metadata", "format"])
def test_checkpoint_generation_config_is_restored_on_success_and_failure(tmp_path, do_sample, failure):
    from transformers import GenerationConfig
    from qlean.core.checkpoint import save_checkpoint

    original = GenerationConfig(do_sample=do_sample, temperature=0.5, top_p=0.9)
    before = original.to_dict()
    model = SimpleNamespace(generation_config=original)
    calls = []

    def save(directory, *, max_shard_size):
        calls.append("save")
        assert max_shard_size == "8KB"
        assert model.generation_config is not original
        assert model.generation_config.temperature == (0.5 if do_sample else None)
        assert model.generation_config.top_p == (0.9 if do_sample else None)
        if failure == "save":
            raise RuntimeError("save failed")
        (Path(directory) / "config.json").write_text("{}")

    def update_config(directory):
        calls.append("metadata")
        assert model.generation_config is original
        if failure == "metadata":
            raise RuntimeError("metadata failed")
        quantization = {"quantization_config": {
            "format": "dense" if failure == "format" else "int-quantized",
        }}
        (Path(directory) / "config.json").write_text(json.dumps(quantization))

    model.save_pretrained = save
    compressor = SimpleNamespace(update_config=update_config)
    if failure is None:
        save_checkpoint(model, tmp_path, "8KB", compressor, "int-quantized")
    else:
        with pytest.raises(RuntimeError, match={
            "save": "save failed", "metadata": "metadata update failed", "format": "invalid.*format",
        }[failure]):
            save_checkpoint(model, tmp_path, "8KB", compressor, "int-quantized")
    assert model.generation_config is original
    assert original.to_dict() == before
    assert calls == (["save"] if failure == "save" else ["save", "metadata"])


@pytest.mark.parametrize("failure", ["construct", "compress"])
def test_checkpoint_compression_failure_has_no_dense_fallback(monkeypatch, failure):
    import compressed_tensors
    from qlean.core.checkpoint import compress_model_for_save

    error = ValueError("compression failed")
    model = SimpleNamespace(save_pretrained=lambda *args, **kwargs: pytest.fail("Unexpected save"))

    def compress(source, *, skip_compressed):
        assert source is model and skip_compressed is True
        raise error

    def construct(source, *, quantization_format):
        assert source is model and quantization_format == "int-quantized"
        if failure == "construct":
            raise error
        return SimpleNamespace(compress_model=compress)

    monkeypatch.setattr(compressed_tensors, "ModelCompressor", SimpleNamespace(from_pretrained_model=construct))
    with pytest.raises(RuntimeError, match="compression failed.*refusing") as raised:
        compress_model_for_save(model, quantization_format="int-quantized")
    assert raised.value.__cause__ is error


@pytest.mark.parametrize("qlean_loaded_processor", [False, True])
def test_processor_is_saved_only_when_qlean_loaded_the_model(tmp_path, qlean_loaded_processor):
    from tokenizers import Tokenizer, models
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=128, hidden_size=128, intermediate_size=128,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=32, use_cache=False,
    )).to(torch.bfloat16)
    source, output = tmp_path / "source", tmp_path / "output"
    model.save_pretrained(source)
    PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel(vocab={"[UNK]": 0, "hi": 1}, unk_token="[UNK]")),
        unk_token="[UNK]",
    ).save_pretrained(source)
    model = LlamaForCausalLM.from_pretrained(source, dtype=torch.bfloat16)
    compressor = QLeanCompressor(
        ModelArguments(model_name="tiny-llama", model_path=str(source),
                       save_path=str(output), output_format="GPTQ_MXFP4", max_shard_size="10GB"),
        model=model, modifiers=[GPTQModifier(scheme="MXFP4", targets=["Linear"],
                                             ignore=["lm_head"], block_size=32)],
    )
    compressor.dataloader = [_batch()]
    # A caller that supplies model= owns its auxiliary files; _processor stays None.
    if qlean_loaded_processor:
        compressor._processor = AutoTokenizer.from_pretrained(source)
    compressor.post_train(SimpleNamespace())

    assert (output / "tokenizer_config.json").exists() is qlean_loaded_processor
    quantization = json.loads((output / "config.json").read_text())["quantization_config"]
    assert quantization["quant_method"] == "mxfp4"
    if qlean_loaded_processor:
        assert (output / "tokenizer.json").exists()
        assert AutoTokenizer.from_pretrained(output).convert_tokens_to_ids("hi") == 1
