# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
import struct
from types import SimpleNamespace

import pytest

pytest.importorskip("qlean.modifiers.gptq.base", reason="requires the qlean[gptq] extra")

import torch
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from safetensors import SafetensorError
from safetensors.torch import load_file, save_file
from transformers import LlamaConfig

import qlean.modifiers.gptq.export as exporter
import qlean.modifiers.save_utils as preservation
from qlean.args import ModelArguments
from qlean.core.qleancompressor import QLeanCompressor
from qlean.modifiers.gptq import GPTQModifier
from qlean.utils.pack_acext_int4 import preprocess_weight_for_a8w4_torch


pytestmark = [pytest.mark.gptq, pytest.mark.integration, pytest.mark.cpu]


def _snapshot(directory):
    return {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}


def _payloads(directory):
    payloads = {}
    for path in directory.glob("*.safetensors"):
        raw = path.read_bytes()
        size = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8:8 + size])
        data = raw[8 + size:]
        for name, item in header.items():
            if name == "__metadata__":
                continue
            assert name not in payloads
            start, end = item["data_offsets"]
            payloads[name] = data[start:end]
    return payloads


def _source(tmp_path, base=78, count=1, *, nested=False, layout="appended", shards="single"):
    directory = tmp_path / "source"
    directory.mkdir()
    text = {
        "num_hidden_layers": base,
        "num_mtp_layers": 0,
        "num_nextn_predict_layers": count,
        "index_share_for_mtp_iteration": True,
        "mtp_loss_weight": 0.3,
    }
    config = {"model_type": "glm_moe_dsa" if base == 78 else "deepseek_v32", **text}
    if nested:
        config = {"model_type": "conditional", "text_config": text}
    owner = "model.language_model.layers" if nested else "model.layers"
    prefixes = [f"{owner}.{i}" for i in range(base, base + count)]
    if layout != "appended":
        prefixes = [f"{layout}.layers.{i}" for i in range(count)]
    mtp = {}
    for i, prefix in enumerate(prefixes):
        for j, suffix in enumerate((
            "eh_proj.weight", "enorm.weight", "hnorm.weight", "shared_head.norm.weight",
            "embed_tokens.weight", "shared_head.head.weight", "self_attn.q_proj.weight",
            "mlp.experts.0.gate_proj.weight",
        )):
            mtp[f"{prefix}.{suffix}"] = (
                torch.arange(12).reshape(3, 4) + 100 * i + 10 * j
            ).to(torch.bfloat16)
        mtp[f"{prefix}.iteration"] = torch.tensor(i, dtype=torch.int64)
        mtp[f"{prefix}.enabled"] = torch.tensor(True)
    primary = {
        f"{owner}.0.self_attn.q_proj.weight": torch.zeros(3, 4, dtype=torch.bfloat16),
        f"{owner}.{base * 10}.unrelated.weight": torch.ones(3, 4, dtype=torch.bfloat16),
        "mtpfoo.weight": torch.ones(2, dtype=torch.bfloat16),
    }
    groups = [{**primary, **mtp}]
    if shards == "separate":
        groups = [primary, mtp]
    elif shards == "mixed":
        values = list({**primary, **mtp}.items())
        groups = [dict(values[::2]), dict(values[1::2])]
    weight_map = {}
    for i, group in enumerate(groups):
        filename = "model.safetensors" if len(groups) == 1 else f"source-{i}.safetensors"
        save_file(group, directory / filename)
        weight_map.update({name: filename for name in group})
    if len(groups) > 1:
        (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (directory / "config.json").write_text(json.dumps(config))
    return directory, config, mtp, prefixes, weight_map


def _model(source, *, a8w4=False):
    model = torch.nn.Module()
    model.config = LlamaConfig(num_hidden_layers=1)
    model.name_or_path = str(source)
    model.dense = torch.nn.Linear(7, 3, bias=False)
    model.dense.quantization_scheme = QuantizationScheme(
        targets=["Linear"], weights=QuantizationArgs(num_bits=8, strategy="channel"),
    )
    if a8w4:
        model.mlp = torch.nn.Module()
        model.mlp.experts = torch.nn.ModuleList([torch.nn.Linear(64, 64, bias=False)])
        model.mlp.experts[0].quantization_scheme = QuantizationScheme(
            targets=["Linear"], weights=QuantizationArgs(num_bits=4, strategy="channel"),
        )
    return model


def _intermediate(directory, model):
    directory.mkdir()
    tensors, codes = {}, {}
    for name, module in model.named_modules():
        scheme = getattr(module, "quantization_scheme", None)
        if scheme is None:
            continue
        bits = scheme.weights.num_bits
        values = torch.arange(module.weight.numel()).reshape(module.weight.shape)
        code = (values % (2 ** bits) - 2 ** (bits - 1)).to(torch.int8)
        codes[name] = code
        if hasattr(model, "mlp"):
            tensors[f"{name}.weight_packed"] = pack_to_int32(code, bits).contiguous()
            tensors[f"{name}.weight_shape"] = torch.tensor(list(code.shape))
            tensors[f"{name}.weight_zero_point"] = torch.zeros(code.shape[0], 1, dtype=torch.int8)
        else:
            tensors[f"{name}.weight"] = code
        tensors[f"{name}.weight_scale"] = torch.ones(code.shape[0], 1, dtype=torch.bfloat16)
    save_file(tensors, directory / "model.safetensors")
    (directory / "config.json").write_text(json.dumps({
        "model_type": "runtime", "num_hidden_layers": 1, "keep": "runtime",
        "quantization_config": {"format": "pack-quantized" if len(codes) > 1 else "int-quantized"},
    }))
    return codes


def _assert_mtp(source, output, mtp, prefixes, max_shard_size):
    source_payloads, output_payloads = _payloads(source), _payloads(output)
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(output_payloads)
    assert index["metadata"]["total_size"] == sum(map(len, output_payloads.values()))
    assert {name for name in output_payloads if any(name.startswith(f"{p}.") for p in prefixes)} == set(mtp)
    for name, expected in mtp.items():
        shard = load_file(output / index["weight_map"][name])
        actual = shard[name]
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert torch.equal(actual, expected)
        assert output_payloads[name] == source_payloads[name]
    for filename in set(index["weight_map"].values()):
        shard = load_file(output / filename)
        assert set(shard) == {name for name, mapped in index["weight_map"].items() if mapped == filename}
        size = sum(value.numel() * value.element_size() for value in shard.values())
        assert size <= max_shard_size or len(shard) == 1
    ignore = json.loads((output / "config.json").read_text())["quantization_config"]["ignore"]
    expected = {
        name.removesuffix(".weight") for name, tensor in mtp.items()
        if name.endswith(".weight") and tensor.ndim >= 2
    }
    assert {item for item in ignore if any(item.startswith(f"{p}.") for p in prefixes)} == expected
    assert len(ignore) == len(set(ignore))
    assert not any(item.startswith("re:") for item in ignore)
    assert not any(item.endswith((".iteration", ".enabled")) for item in ignore)


@pytest.mark.parametrize("base,count,nested", [(78, 1, False), (61, 1, False), (61, 2, True)])
@pytest.mark.parametrize("shards", ["single", "separate", "mixed"])
@pytest.mark.parametrize("a8w4", [False, True])
@pytest.mark.parametrize("max_shard_size", [100000, 70])
def test_appended_source_mtp_exact_custom_export(
    tmp_path, base, count, nested, shards, a8w4, max_shard_size,
):
    source, source_config, mtp, prefixes, _ = _source(tmp_path, base, count, nested=nested, shards=shards)
    model = _model(source, a8w4=a8w4)
    output = tmp_path / "output"
    before = _snapshot(source)
    checkpoint = exporter.prepare_source_checkpoint(model, str(source), output, [])
    assert checkpoint["mtp_keys"] == set(mtp)
    codes = _intermediate(output, model)
    export = exporter.export_a8w4_checkpoint if a8w4 else exporter.export_int8_checkpoint
    export(output, model, max_shard_size, checkpoint=checkpoint)
    _assert_mtp(source, output, mtp, prefixes, max_shard_size)
    config = json.loads((output / "config.json").read_text())
    assert config["num_hidden_layers"] == 1 and config["keep"] == "runtime"
    location = config["text_config"] if nested else config
    original = source_config["text_config"] if nested else source_config
    for field in (*preservation._MTP_COUNT_FIELDS, "index_share_for_mtp_iteration", "mtp_loss_weight"):
        if field in original:
            assert location[field] == original[field]
    values = {name: value for path in output.glob("*.safetensors") for name, value in load_file(path).items()}
    assert torch.equal(values["dense.weight"], codes["dense"])
    assert values["dense.weight_scale"].dtype == torch.float32
    if a8w4:
        assert torch.equal(values["mlp.experts.0.weight"], preprocess_weight_for_a8w4_torch(codes["mlp.experts.0"]).view(64, 32))
    assert _snapshot(source) == before
    assert not list(output.glob(".qlean-*"))


@pytest.mark.parametrize("layout", ["mtp", "model.mtp", "model.language_model.mtp"])
def test_dedicated_mtp_without_count_is_preserved(tmp_path, layout):
    source, config, mtp, prefixes, _ = _source(tmp_path, layout=layout)
    for field in preservation._MTP_COUNT_FIELDS:
        config.pop(field, None)
    (source / "config.json").write_text(json.dumps(config))
    model = _model(source)
    output = tmp_path / "output"
    checkpoint = exporter.prepare_source_checkpoint(model, str(source), output, [])
    _intermediate(output, model)
    exporter.export_int8_checkpoint(output, model, 100000, checkpoint=checkpoint)
    _assert_mtp(source, output, mtp, [layout], 100000)
    assert (output / "model_mtp.safetensors").exists()


def test_undeclared_extra_decoder_layers_are_not_guessed_as_mtp(tmp_path):
    source, config, _, _, _ = _source(tmp_path)
    for field in preservation._MTP_COUNT_FIELDS:
        config.pop(field, None)
    (source / "config.json").write_text(json.dumps(config))
    assert exporter.prepare_source_checkpoint(_model(source), str(source), tmp_path / "output", []) is None


@pytest.mark.parametrize("bad", [True, -1, 1.5, "1", [1]])
def test_invalid_source_counts_fail_before_export(tmp_path, bad):
    source, config, _, _, _ = _source(tmp_path)
    config["num_nextn_predict_layers"] = bad
    (source / "config.json").write_text(json.dumps(config))
    before = _snapshot(source)
    with pytest.raises(ValueError, match="nonnegative integer"):
        exporter.prepare_source_checkpoint(_model(source), str(source), tmp_path / "output", [])
    assert _snapshot(source) == before


@pytest.mark.parametrize("failure", ["count", "missing_layer", "missing_shard", "missing_key", "path", "owners", "layouts", "fp8", "quantized"])
def test_source_mtp_errors_are_fail_closed(tmp_path, failure):
    source, config, mtp, _, weight_map = _source(tmp_path, shards="separate")
    shard_path = source / "source-1.safetensors"
    if failure == "count":
        config["num_mtp_layers"] = 2
    elif failure == "missing_layer":
        config["num_nextn_predict_layers"] = 2
    elif failure == "missing_shard":
        weight_map.update({name: "missing.safetensors" for name in mtp})
    elif failure == "missing_key":
        weight_map["model.layers.78.absent.weight"] = "source-1.safetensors"
    elif failure == "path":
        weight_map.update({name: "../escape.safetensors" for name in mtp})
    elif failure in {"owners", "layouts"}:
        extra = "model.language_model.layers.78.eh_proj.weight" if failure == "owners" else "mtp.layers.0.weight"
        tensors = {**mtp, extra: torch.zeros(2, dtype=torch.bfloat16)}
        save_file(tensors, shard_path)
        weight_map[extra] = shard_path.name
    elif failure == "fp8":
        tensors = dict(mtp)
        tensors[next(iter(tensors))] = torch.zeros(3, 4, dtype=torch.float8_e4m3fn)
        save_file(tensors, shard_path)
    else:
        config["quantization_config"] = {"quant_method": "fp8"}
    (source / "config.json").write_text(json.dumps(config))
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    before = _snapshot(source)
    with pytest.raises(ValueError):
        exporter.prepare_source_checkpoint(_model(source), str(source), tmp_path / "output", [])
    assert _snapshot(source) == before
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("failure", ["copy", "publication", "changed_header", "key_conflict", "shard_conflict"])
@pytest.mark.parametrize("a8w4", [False, True])
def test_custom_mtp_failure_retains_original_checkpoint(monkeypatch, tmp_path, failure, a8w4):
    source, _, mtp, _, _ = _source(tmp_path)
    model = _model(source, a8w4=a8w4)
    output = tmp_path / "output"
    checkpoint = exporter.prepare_source_checkpoint(model, str(source), output, [])
    _intermediate(output, model)
    if failure == "copy":
        monkeypatch.setattr(preservation, "save_file", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("copy failure")))
    elif failure == "publication":
        original_replace = exporter.os.replace
        calls = 0

        def replace(src, dst):
            nonlocal calls
            if ".qlean-" in str(src) and str(dst).endswith("model.safetensors.index.json"):
                calls += 1
                if calls == 1:
                    raise OSError("publication failure")
            return original_replace(src, dst)

        monkeypatch.setattr(exporter.os, "replace", replace)
    elif failure == "changed_header":
        tensors = load_file(source / "model.safetensors")
        tensors[next(iter(mtp))] = torch.zeros(5, dtype=torch.bfloat16)
        save_file(tensors, source / "model.safetensors")
    elif failure == "key_conflict":
        tensors = load_file(output / "model.safetensors")
        tensors[next(iter(mtp))] = mtp[next(iter(mtp))]
        save_file(tensors, output / "model.safetensors")
    else:
        save_file(mtp, output / "model_mtp.safetensors")
    before, source_before = _snapshot(output), _snapshot(source)
    export = exporter.export_a8w4_checkpoint if a8w4 else exporter.export_int8_checkpoint
    with pytest.raises((ValueError, RuntimeError, OSError)):
        export(output, model, 100000, checkpoint=checkpoint)
    assert _snapshot(output) == before
    assert _snapshot(source) == source_before
    assert not list(output.glob(".qlean-*"))
    assert not (output / "qlean_config.json").exists()


def test_pinned_hub_mtp_requests_keep_loaded_revision(monkeypatch, tmp_path):
    source, _, mtp, prefixes, weight_map = _source(tmp_path, shards="mixed")
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    model = _model("owner/model")
    model.config._commit_hash = "a" * 40
    requests = []

    def cached_file(repo, filename, revision, **kwargs):
        requests.append((repo, filename, revision))
        path = source / filename
        return str(path) if path.exists() else None

    monkeypatch.setattr("transformers.utils.cached_file", cached_file)
    output = tmp_path / "output"
    checkpoint = exporter.prepare_source_checkpoint(model, "owner/model", output, [])
    _intermediate(output, model)
    exporter.export_int8_checkpoint(output, model, 100000, checkpoint=checkpoint)
    _assert_mtp(source, output, mtp, prefixes, 100000)
    upstream = [entry for entry in requests if entry[0] == "owner/model"]
    assert upstream and all(revision == "a" * 40 for _, _, revision in upstream)
    assert {filename for _, filename, _ in upstream} >= {"config.json", "model.safetensors.index.json", *weight_map.values()}


def test_standard_preservation_keeps_primary_mapping_and_metadata(tmp_path):
    source, _, mtp, prefixes, _ = _source(tmp_path, base=61, count=2, shards="mixed")
    model = _model(source)
    output = tmp_path / "output"
    checkpoint = exporter.prepare_source_checkpoint(model, str(source), output, [])
    _intermediate(output, model)
    index = {"metadata": {"keep": True, "total_size": -1}, "weight_map": {
        name: "model.safetensors" for name in load_file(output / "model.safetensors")
    }}
    (output / "model.safetensors.index.json").write_text(json.dumps(index))
    primary_bytes = (output / "model.safetensors").read_bytes()
    preservation.finalize_checkpoint_preservation(model, checkpoint, output, 70)
    final = json.loads((output / "model.safetensors.index.json").read_text())
    assert final["metadata"]["keep"] is True
    assert all(final["weight_map"][name] == filename for name, filename in index["weight_map"].items())
    assert (output / "model.safetensors").read_bytes() == primary_bytes
    _assert_mtp(source, output, mtp, prefixes, 70)


@pytest.mark.parametrize("scheme,output_format", [("INT8_CHANNEL", "GPTQ_INT8"), ("MXFP4", "GPTQ_MXFP4")])
def test_real_cpu_gptq_preserves_appended_source_only_mtp(
    monkeypatch, tmp_path, scheme, output_format, assert_mxfp4_payload_reconstructs
):
    from transformers import LlamaForCausalLM

    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=128, hidden_size=128, intermediate_size=128, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=32, use_cache=False,
    )).to(torch.bfloat16)
    source, output = tmp_path / "source", tmp_path / "output"
    model.save_pretrained(source)
    config = json.loads((source / "config.json").read_text())
    config.update(num_nextn_predict_layers=1, index_share_for_mtp_iteration=True)
    (source / "config.json").write_text(json.dumps(config))
    mtp = {
        "model.layers.1.eh_proj.weight": torch.arange(12).reshape(3, 4).to(torch.bfloat16),
        "model.layers.1.shared_head.norm.weight": torch.ones(4, dtype=torch.bfloat16),
    }
    save_file({**load_file(source / "model.safetensors"), **mtp}, source / "model.safetensors")
    model = LlamaForCausalLM.from_pretrained(source, dtype=torch.bfloat16)
    del model.config.num_nextn_predict_layers
    del model.config.index_share_for_mtp_iteration
    before = _snapshot(source)
    original_state = {name: value.clone() for name, value in model.state_dict().items()}
    dense_keys = {
        f"model.layers.0.{name}.weight" for name in (
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
        )
    }
    snapshots = {}
    if scheme == "INT8_CHANNEL":
        original_export = exporter.export_int8_checkpoint

        def observe_export(directory, source_model, max_shard_size, *, checkpoint=None, expert_patterns=None, workers=2):
            assert source_model is model and not snapshots
            assert workers == 2
            directory = preservation.Path(directory)
            tensors = {name: value for path in directory.glob("*.safetensors")
                       for name, value in load_file(path).items()}
            config = json.loads((directory / "config.json").read_text())["quantization_config"]
            assert config["quant_method"] == "compressed-tensors" and config["format"] == "int-quantized"
            for name in dense_keys:
                codes, scale = tensors[name], tensors[f"{name}_scale"]
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

        original_export = exporter.export_mxfp4_checkpoint

        def observe_export(directory, source_model, max_shard_size, *, checkpoint=None, expert_patterns=None, workers=2):
            assert source_model is model and not snapshots
            assert workers == 2
            tensors = {name: value for path in preservation.Path(directory).glob("*.safetensors")
                       for name, value in load_file(path).items()}
            for name in dense_keys:
                snapshots[name] = (
                    tensors[f"{name}_packed"].clone(),
                    tensors[f"{name}_scale"].clone(),
                )
            return original_export(
                directory, source_model, max_shard_size, checkpoint=checkpoint,
                expert_patterns=expert_patterns, workers=workers,
            )

        def forbidden_shared_rename(*args, **kwargs):
            pytest.fail("GPTQ MXFP4 preservation must use isolated staging")

        monkeypatch.setattr(exporter, "export_mxfp4_checkpoint", observe_export)
        monkeypatch.setattr(checkpoint_module, "rename_packed_weights", forbidden_shared_rename)
    QLeanCompressor._instances.pop(QLeanCompressor, None)
    try:
        compressor = QLeanCompressor(
            ModelArguments(model_name="tiny-llama", model_path=str(source), save_path=str(output), output_format=output_format, max_shard_size=100000),
            model=model, modifiers=[GPTQModifier(scheme=scheme, targets=["Linear"],
                                                 ignore=["lm_head"], block_size=32)],
        )
        batch = {"input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])}
        compressor.dataloader = [batch]
        compressor.post_train(SimpleNamespace())
        _assert_mtp(source, output, mtp, ["model.layers.1"], 100000)
        assert _snapshot(source) == before
        final = json.loads((output / "config.json").read_text())
        assert final["num_hidden_layers"] == 1 and final["num_nextn_predict_layers"] == 1
        assert final["index_share_for_mtp_iteration"] is True
        quantization = final["quantization_config"]
        assert quantization["quant_method"] == ("w8a8_int8" if scheme == "INT8_CHANNEL" else "mxfp4")
        assert not {"format", "config_groups", "kv_cache_scheme"} & quantization.keys()
        metadata = json.loads((output / "qlean_config.json").read_text())
        assert metadata["scheme"] == scheme and metadata["output_format"] == output_format
        assert metadata["modifier_chain"] == ["GPTQModifier"]
        tensors = {name: value for path in output.glob("*.safetensors") for name, value in load_file(path).items()}
        preserved = {name: value for name, value in original_state.items() if name not in dense_keys}
        assert set(tensors) == set(preserved) | set(mtp) | dense_keys | {f"{name}_scale" for name in dense_keys}
        for name in dense_keys:
            weight, scale = tensors[name], tensors[f"{name}_scale"]
            n, k = original_state[name].shape
            if scheme == "INT8_CHANNEL":
                codes, original_scale = snapshots[name]
                assert weight.dtype == torch.int8 and weight.shape == (n, k)
                assert torch.equal(weight, codes)
                assert scale.dtype == torch.float32 and scale.shape == (n, 1)
                assert torch.equal(scale, original_scale.float())
                assert torch.isfinite(scale).all() and (scale > 0).all()
            else:
                packed, original_scale = snapshots[name]
                assert weight.dtype == torch.uint8 and weight.shape == (n, k // 2)
                assert scale.dtype == torch.uint8 and scale.shape == (n, k // 32)
                assert torch.equal(weight, packed) and torch.equal(scale, original_scale)
            assert not any(f"{name}{suffix}" in tensors for suffix in (
                "_packed", "_shape", "_zero_point", "_g_idx", "_global_scale",
            ))
        for name, value in preserved.items():
            assert tensors[name].dtype == value.dtype and torch.equal(tensors[name], value)
        assert not any(name.startswith("model.layers.1.") for name in model.state_dict())
        if scheme == "MXFP4":
            # Both custom producer contracts have no standard HF reload claim; verify
            # the saved MXFP4 payload decodes back to the source weight instead.
            assert_mxfp4_payload_reconstructs(tensors, original_state, dense_keys)
    finally:
        QLeanCompressor._instances.pop(QLeanCompressor, None)


@pytest.mark.parametrize("conflict", [False, True])
def test_nested_source_root_count_is_respected(tmp_path, conflict):
    source, config, mtp, _, _ = _source(tmp_path, nested=True)
    config["num_nextn_predict_layers"] = 1
    config["text_config"]["num_nextn_predict_layers"] = 2 if conflict else 0
    (source / "config.json").write_text(json.dumps(config))
    if conflict:
        with pytest.raises(ValueError, match="conflicting MTP"):
            exporter.prepare_source_checkpoint(_model(source), str(source), tmp_path / "output", [])
    else:
        checkpoint = exporter.prepare_source_checkpoint(_model(source), str(source), tmp_path / "output", [])
        assert checkpoint["mtp_keys"] == set(mtp)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64, torch.int32, torch.int64, torch.bool])
def test_mtp_auxiliary_dtype_and_payload_preserved(tmp_path, dtype):
    source, _, mtp, prefixes, _ = _source(tmp_path)
    tensors = load_file(source / "model.safetensors")
    key = "model.layers.78.auxiliary"
    mtp[key] = torch.tensor([0, 1, 2], dtype=dtype)
    save_file({**tensors, key: mtp[key]}, source / "model.safetensors")
    model = _model(source)
    output = tmp_path / "output"
    checkpoint = exporter.prepare_source_checkpoint(model, str(source), output, [])
    _intermediate(output, model)
    exporter.export_int8_checkpoint(output, model, 100000, checkpoint=checkpoint)
    _assert_mtp(source, output, mtp, prefixes, 100000)


@pytest.mark.parametrize("filename", ["/absolute.safetensors", "../escape.safetensors", "a/../../escape.safetensors", "a\\b.safetensors", ""])
def test_source_index_paths_rejected(tmp_path, filename):
    source, _, mtp, _, weight_map = _source(tmp_path, shards="separate")
    weight_map.update({name: filename for name in mtp})
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    with pytest.raises(ValueError, match="Invalid checkpoint filename"):
        exporter.prepare_source_checkpoint(_model(source), str(source), tmp_path / "output", [])


@pytest.mark.parametrize("bad", [[], None, {"key": 1}])
def test_source_weight_map_types_rejected(tmp_path, bad):
    source, _, _, _, _ = _source(tmp_path)
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": bad}))
    with pytest.raises(ValueError, match="weight_map"):
        exporter.prepare_source_checkpoint(_model(source), str(source), tmp_path / "output", [])


def test_mtp_shard_copy_opens_each_source_once_per_output_shard(monkeypatch, tmp_path):
    source, _, mtp, prefixes, weight_map = _source(tmp_path, count=2, shards="mixed")
    model = _model(source)
    output = tmp_path / "output"
    checkpoint = exporter.prepare_source_checkpoint(model, str(source), output, [])
    _intermediate(output, model)
    original = preservation.safe_open
    opens = []

    def safe_open(path, *args, **kwargs):
        opens.append(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(preservation, "safe_open", safe_open)
    preservation.finalize_checkpoint_preservation(model, checkpoint, output, 70)
    output_index = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
    expected = sum(len({weight_map[name] for name in mtp if output_index[name] == filename})
                   for filename in {output_index[name] for name in mtp})
    assert len(opens) == expected
    assert len(opens) < len(mtp)
    _assert_mtp(source, output, mtp, prefixes, 70)


@pytest.mark.parametrize("declared", [False, True])
def test_memory_model_without_source_is_optional_unless_declared(tmp_path, declared):
    model = _model("")
    if declared:
        model.config.num_nextn_predict_layers = 1
        with pytest.raises(ValueError, match="valid source checkpoint"):
            exporter.prepare_source_checkpoint(model, "", tmp_path / "output", [])
    else:
        assert exporter.prepare_source_checkpoint(model, "", tmp_path / "output", []) is None


@pytest.mark.parametrize("layout", ["none", "appended", "mtp", "model.mtp", "model.language_model.mtp"])
def test_pinned_single_file_hub_source(monkeypatch, tmp_path, layout):
    source, config, mtp, prefixes, weight_map = _source(
        tmp_path, layout="appended" if layout == "none" else layout,
    )
    if layout != "appended":
        for field in preservation._MTP_COUNT_FIELDS:
            config.pop(field, None)
        (source / "config.json").write_text(json.dumps(config))
    if layout == "none":
        tensors = load_file(source / "model.safetensors")
        save_file({name: value for name, value in tensors.items() if name not in mtp}, source / "model.safetensors")
    model = _model("owner/model")
    model.config._commit_hash = "b" * 40
    requests = []

    def cached_file(repo, filename, revision, **kwargs):
        requests.append((repo, filename, revision))
        path = source / filename if repo == "owner/model" else preservation.Path(repo) / filename
        return str(path) if path.exists() else None

    monkeypatch.setattr("transformers.utils.cached_file", cached_file)
    before = _snapshot(source)
    output = tmp_path / "output"
    checkpoint = exporter.prepare_source_checkpoint(model, "owner/model", output, [])
    if layout == "none":
        assert checkpoint is None
    else:
        assert checkpoint["weight_map"] == weight_map
        assert checkpoint["mtp_keys"] == set(mtp)
        _intermediate(output, model)
        exporter.export_int8_checkpoint(output, model, 100000, checkpoint=checkpoint)
        _assert_mtp(source, output, mtp, prefixes if layout == "appended" else [layout], 100000)
    upstream = [entry for entry in requests if entry[0] == "owner/model"]
    assert all(revision == "b" * 40 for _, _, revision in upstream)
    assert {filename for _, filename, _ in upstream} >= {
        "config.json", "model.safetensors.index.json", "model.safetensors",
    }
    assert _snapshot(source) == before


@pytest.mark.parametrize("failure", ["missing", "malformed"])
def test_pinned_single_file_hub_invalid_weights_fail_closed(monkeypatch, tmp_path, failure):
    source, _, _, _, _ = _source(tmp_path)
    model = _model("owner/model")
    model.config._commit_hash = "c" * 40
    if failure == "malformed":
        (source / "model.safetensors").write_bytes(b"invalid safetensors")
    requests = []

    def cached_file(repo, filename, revision, **kwargs):
        requests.append((repo, filename, revision))
        if filename == "model.safetensors" and failure == "missing":
            return None
        path = source / filename
        return str(path) if path.exists() else None

    monkeypatch.setattr("transformers.utils.cached_file", cached_file)
    before = _snapshot(source)
    output = tmp_path / "output"
    with pytest.raises((ValueError, OSError, SafetensorError)):
        exporter.prepare_source_checkpoint(model, "owner/model", output, [])
    assert _snapshot(source) == before and not output.exists()
    assert all(revision == "c" * 40 for _, _, revision in requests)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("stale_nested", [False, True])
@pytest.mark.parametrize("a8w4", [False, True])
def test_source_mtp_fields_replace_stale_runtime_aliases(tmp_path, nested, stale_nested, a8w4):
    source, source_config, mtp, _, _ = _source(tmp_path, nested=nested)
    original = source_config["text_config"] if nested else source_config
    original.pop("num_mtp_layers")
    (source / "config.json").write_text(json.dumps(source_config))
    model = _model(source, a8w4=a8w4)
    output = tmp_path / "output"
    checkpoint = exporter.prepare_source_checkpoint(model, str(source), output, [])
    _intermediate(output, model)
    config = json.loads((output / "config.json").read_text())
    config["num_hidden_layers"] = 78
    config["text_config"] = {"num_hidden_layers": 78, "keep_nested": "runtime"}
    stale = config["text_config"] if stale_nested else config
    stale.update(num_mtp_layers=2, mtp_num_hidden_layers=3, mtp_runtime_only=True,
                 index_share_for_mtp_iteration=False, mtp_loss_weight=99)
    (output / "config.json").write_text(json.dumps(config))
    export = exporter.export_a8w4_checkpoint if a8w4 else exporter.export_int8_checkpoint
    export(output, model, 100000, checkpoint=checkpoint)
    final = json.loads((output / "config.json").read_text())
    for location in (None, "text_config"):
        actual = final if location is None else final[location]
        expected = source_config if location is None else source_config.get(location, {})
        fields = lambda data: {name: value for name, value in data.items()
                               if name in (*preservation._MTP_COUNT_FIELDS, "index_share_for_mtp_iteration")
                               or name.startswith("mtp_")}
        assert fields(actual) == fields(expected)
        assert actual["num_hidden_layers"] == 78
    assert final["keep"] == "runtime" and final["text_config"]["keep_nested"] == "runtime"
    weight_map = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
    selected_keys = preservation._source_mtp_layout(final, weight_map)
    assert selected_keys == set(mtp)


@pytest.mark.parametrize("direct_export", [False, True])
def test_missing_unquantized_experts_fail_before_int8_publication(tmp_path, direct_export):
    source, _, _, _, _ = _source(tmp_path)
    model = _model(source)
    model.moe = torch.nn.Module()
    model.moe.experts = torch.nn.ModuleList([torch.nn.Linear(7, 3, bias=False)])
    output = tmp_path / "output"
    checkpoint = exporter.prepare_source_checkpoint(model, str(source), output, [])
    _intermediate(output, model)
    before, source_before = _snapshot(output), _snapshot(source)
    with pytest.raises(RuntimeError, match="expert keys appear to have collapsed"):
        if direct_export:
            exporter.export_int8_checkpoint(output, model, 100000, checkpoint=checkpoint)
        else:
            exporter.finalize_gptq_checkpoint(
                model, output, 100000, "int-quantized", ["moe.experts."], checkpoint=checkpoint,
            )
    assert _snapshot(output) == before and _snapshot(source) == source_before
    assert not list(output.glob(".qlean-*")) and not (output / "qlean_config.json").exists()


@pytest.mark.parametrize("a8w4", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_custom_expert_validation_is_once_before_publication(monkeypatch, tmp_path, a8w4, failure):
    import qlean.core.checkpoint as checkpoint_utils

    source, _, _, _, _ = _source(tmp_path)
    model = _model(source, a8w4=a8w4)
    output = tmp_path / "output"
    checkpoint = exporter.prepare_source_checkpoint(model, str(source), output, [])
    _intermediate(output, model)
    before, source_before = _snapshot(output), _snapshot(source)
    events = []
    patterns = ["captured-before-compression.experts."]
    publish = exporter._publish_staged_checkpoint

    def validate(model_arg, directory, supplied_patterns):
        assert model_arg is model and supplied_patterns == patterns
        assert preservation.Path(directory).parent == output
        assert _snapshot(output) == before
        assert any(preservation.Path(directory).glob("model_mtp*.safetensors"))
        events.append("validate")
        if failure:
            raise RuntimeError("expert validation failure")

    def publish_after_validation(*args, **kwargs):
        assert events == ["validate"]
        assert kwargs["label"] == ("A8W4" if a8w4 else "INT8")
        assert "a8w4" not in kwargs
        events.append("publish")
        return publish(*args, **kwargs)

    monkeypatch.setattr(checkpoint_utils, "validate_expert_keys_in_checkpoint", validate)
    monkeypatch.setattr(exporter, "_publish_staged_checkpoint", publish_after_validation)
    if failure:
        with pytest.raises(RuntimeError, match="expert validation failure"):
            exporter.finalize_gptq_checkpoint(
                model, output, 100000, "pack-quantized" if a8w4 else "int-quantized", patterns,
                checkpoint=checkpoint,
            )
        assert events == ["validate"] and _snapshot(output) == before
    else:
        exporter.finalize_gptq_checkpoint(
            model, output, 100000, "pack-quantized" if a8w4 else "int-quantized", patterns,
            checkpoint=checkpoint,
        )
        assert events == ["validate", "publish"]
    assert _snapshot(source) == source_before and not list(output.glob(".qlean-*"))
