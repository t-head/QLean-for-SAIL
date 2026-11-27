# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import json
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from safetensors.torch import load_file, save_file
from transformers.core_model_loading import WeightRenaming

import qlean.modifiers.gptq.export as exporter
from qlean.modifiers.gptq import GPTQModifier
from qlean.utils.pack_acext_int4 import preprocess_weight_for_a8w4_torch


pytestmark = [pytest.mark.gptq, pytest.mark.integration, pytest.mark.cpu]


def _scheme(bits=4, **overrides):
    args = {"num_bits": bits, "type": "int", "strategy": "channel", "symmetric": True}
    args.update(overrides)
    return QuantizationScheme(targets=["Linear"], weights=QuantizationArgs(**args))


def _model(*, v4=False, shape=(64, 64), dense_shape=(64, 64), dense_bits=8, experts=True):
    model = torch.nn.Module()
    if experts:
        parent = torch.nn.Module()
        parent.experts = torch.nn.ModuleList([torch.nn.Linear(shape[1], shape[0], bias=False)])
        setattr(model, "ffn" if v4 else "mlp", parent)
        parent.experts[0].quantization_scheme = _scheme()
    model.dense = torch.nn.Linear(dense_shape[1], dense_shape[0], bias=False)
    if dense_bits is not None:
        model.dense.quantization_scheme = _scheme(dense_bits)
    model.other = torch.nn.Linear(64, 64, bias=False)
    return model


def _checkpoint(directory, model, *, cross_shard=False, saved_prefix=None):
    codes, scales, tensors = {}, {}, {}
    for name, module in exporter.a8w4_quantized_modules(model).items():
        if saved_prefix and name.startswith("mlp."):
            name = name.replace("mlp.", f"{saved_prefix}.", 1)
        n, k = module.out_features, module.in_features
        bits = module.quantization_scheme.weights.num_bits
        values = torch.arange(n * k).reshape(n, k) + 3 * torch.arange(n).reshape(n, 1)
        codes[name] = (values % (2**bits) - 2 ** (bits - 1)).to(torch.int8)
        scales[name] = (torch.arange(1, n + 1).reshape(n, 1) / 64).to(torch.bfloat16)
        tensors[f"{name}.weight_packed"] = pack_to_int32(codes[name], bits).contiguous()
        tensors[f"{name}.weight_scale"] = scales[name]
        tensors[f"{name}.weight_shape"] = torch.tensor([n, k])
        tensors[f"{name}.weight_zero_point"] = torch.zeros(n, 1, dtype=torch.int8)
    untouched = torch.arange(64 * 64).reshape(64, 64).to(torch.bfloat16)
    tensors["other.weight"] = untouched
    if cross_shard:
        weights = {key: value for key, value in tensors.items() if key.endswith("_packed")}
        auxiliary = {key: value for key, value in tensors.items() if key not in weights}
        save_file(weights, directory / "model-00001-of-00002.safetensors")
        save_file(auxiliary, directory / "model-00002-of-00002.safetensors")
        weight_map = {
            **{key: "model-00001-of-00002.safetensors" for key in weights},
            **{key: "model-00002-of-00002.safetensors" for key in auxiliary},
        }
        (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    else:
        save_file(tensors, directory / "model.safetensors")
    (directory / "config.json").write_text(json.dumps({
        "model_type": "test", "custom_setting": {"keep": True},
        "quantization_config": {"format": "pack-quantized", "ignore": ["lm_head"]},
    }))
    return codes, scales, untouched


def _load_output(directory):
    tensors = {}
    for path in directory.glob("*.safetensors"):
        shard = load_file(path)
        assert not tensors.keys() & shard.keys()
        tensors.update(shard)
    return tensors


def _snapshot(directory):
    return {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}


@pytest.mark.parametrize("shape", [(64, 64), (64, 128), (128, 64)])
@pytest.mark.parametrize("cross_shard", [False, True])
@pytest.mark.parametrize("v4", [False, True])
def test_export_real_pure_torch_codes_exact_scales_and_fresh_output(
    monkeypatch, tmp_path, shape, cross_shard, v4
):
    monkeypatch.setitem(sys.modules, "acext", None)
    model = _model(v4=v4, shape=shape, dense_shape=shape, dense_bits=8)
    codes, scales, untouched = _checkpoint(tmp_path, model, cross_shard=cross_shard)
    original_state = {name: tensor.clone() for name, tensor in model.state_dict().items()}

    exporter.export_a8w4_checkpoint(tmp_path, model, max_shard_size=9000, export_device="cpu")

    tensors = _load_output(tmp_path)
    for name, value in original_state.items():
        assert torch.equal(model.state_dict()[name], value)
    expert_name = "ffn.experts.0" if v4 else "mlp.experts.0"
    n, k = shape
    expected = preprocess_weight_for_a8w4_torch(codes[expert_name]).view(n, k // 2)
    assert tensors[f"{expert_name}.weight"].dtype == torch.int8
    assert torch.equal(tensors[f"{expert_name}.weight"], expected)
    assert torch.equal(tensors["dense.weight"], codes["dense"])
    for name, scale in scales.items():
        key = f"{name}.scale" if v4 else f"{name}.weight_scale"
        assert tensors[key].dtype == torch.float32
        assert tensors[key].shape == scale.shape
        assert torch.equal(tensors[key], scale.float())
    assert torch.equal(tensors["other.weight"], untouched)
    assert tensors["other.weight"].dtype == torch.bfloat16
    assert not any(name.endswith(("_packed", "_shape", "_zero_point")) for name in tensors)
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(tensors)
    assert index["metadata"]["total_size"] == sum(t.numel() * t.element_size() for t in tensors.values())
    assert set(index["weight_map"].values()) == {p.name for p in tmp_path.glob("*.safetensors")}
    for filename in set(index["weight_map"].values()):
        shard = load_file(tmp_path / filename)
        assert all(index["weight_map"][name] == filename for name in shard)
        assert sum(t.numel() * t.element_size() for t in shard.values()) <= 9000
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["custom_setting"] == {"keep": True}
    assert config["quantization_config"] == {
        "quant_method": "mixed_precision_w4", "activation_scheme": "dynamic",
        # This fixture has no lm_head: stale CT ignores must not survive rebuilding.
        "ignore": ["other"],
    }
    assert not list(tmp_path.glob(".qlean-a8w4-*"))


@pytest.mark.parametrize("error_type", [RuntimeError, NotImplementedError, torch.OutOfMemoryError])
def test_accelerator_pack_failure_retries_cpu_once_and_preserves_bytes(
    monkeypatch, caplog, tmp_path, error_type
):
    model = _model(dense_bits=None)
    expert = torch.nn.Linear(64, 64, bias=False)
    expert.quantization_scheme = _scheme()
    model.mlp.experts.append(expert)
    codes, _, _ = _checkpoint(tmp_path, model)
    original_pack = exporter._Int4ExportDevice._pack
    attempts = []

    def pack(tensor, bits, shape):
        attempts.append(tensor.device.type)
        if tensor.device.type == "meta":
            raise error_type("accelerator unavailable")
        return original_pack(tensor, bits, shape)

    monkeypatch.setattr(exporter._Int4ExportDevice, "_pack", staticmethod(pack))
    exporter.export_a8w4_checkpoint(
        tmp_path, model, max_shard_size=100000, export_device="meta"
    )

    assert attempts == ["meta", "cpu", "cpu"]
    assert "retrying on CPU" in caplog.text
    tensors = _load_output(tmp_path)
    for name, value in codes.items():
        expected = preprocess_weight_for_a8w4_torch(value).view(64, 32)
        assert torch.equal(tensors[f"{name}.weight"], expected)


def test_failed_cpu_retry_does_not_publish(monkeypatch, tmp_path):
    model = _model(dense_bits=None)
    _checkpoint(tmp_path, model)
    before = _snapshot(tmp_path)
    attempts = []

    def fail(tensor, bits, shape):
        attempts.append(tensor.device.type)
        raise RuntimeError("pack failure")

    monkeypatch.setattr(exporter._Int4ExportDevice, "_pack", staticmethod(fail))
    with pytest.raises(RuntimeError, match="pack failure"):
        exporter.export_a8w4_checkpoint(tmp_path, model, 100000, export_device="meta")
    assert attempts == ["meta", "cpu"]
    assert _snapshot(tmp_path) == before
    assert not list(tmp_path.glob(".qlean-a8w4-*"))


def test_auto_unavailable_retries_cpu(monkeypatch, caplog, tmp_path):
    import qlean.utils.dev as dev

    model = _model(dense_bits=None)
    _checkpoint(tmp_path, model)

    def unavailable():
        raise RuntimeError("Parallelism has not been supported for accelerator")

    monkeypatch.setattr(dev, "get_main_device", unavailable)
    exporter.export_a8w4_checkpoint(tmp_path, model, 100000)
    assert caplog.text.count("retrying on CPU") == 1


def test_invalid_export_device_does_not_publish(monkeypatch, tmp_path):
    model = _model(dense_bits=None)
    _checkpoint(tmp_path, model)
    before = _snapshot(tmp_path)

    with pytest.raises(ValueError, match="Invalid GPTQ INT4 export device"):
        exporter.export_a8w4_checkpoint(
            tmp_path, model, max_shard_size=100000,
            export_device="not-a-torch-device",
        )

    assert _snapshot(tmp_path) == before
    assert not list(tmp_path.glob(".qlean-a8w4-*"))


@pytest.mark.parametrize("dense_shape", [(3, 7), (5, 10), (32, 96)])
def test_channel_int8_non64_dimensions_are_not_overrestricted(tmp_path, dense_shape):
    model = _model(shape=(64, 128), dense_shape=dense_shape)
    modules = exporter.a8w4_quantized_modules(model, require_int4=True)
    assert set(modules) == {"mlp.experts.0", "dense"}
    codes, scales, _ = _checkpoint(tmp_path, model, cross_shard=True)
    exporter.export_a8w4_checkpoint(tmp_path, model, 100000)
    tensors = _load_output(tmp_path)
    assert tensors["dense.weight"].shape == dense_shape
    assert tensors["dense.weight"].dtype == torch.int8
    assert torch.equal(tensors["dense.weight"], codes["dense"])
    assert torch.equal(tensors["dense.weight_scale"], scales["dense"].float())


@pytest.mark.parametrize("saved_prefix", ["saved", "ffn"])
def test_export_resolves_only_saved_weight_renamings(tmp_path, saved_prefix):
    model = _model(shape=(128, 64))
    model._weight_conversions = [WeightRenaming(
        source_patterns=[f"^{saved_prefix}\\."], target_patterns=["mlp."],
    )]
    codes, scales, _ = _checkpoint(tmp_path, model, saved_prefix=saved_prefix)
    exporter.export_a8w4_checkpoint(tmp_path, model, 10000)
    tensors = _load_output(tmp_path)
    expert_name = f"{saved_prefix}.experts.0"
    assert torch.equal(
        tensors[f"{expert_name}.weight"],
        preprocess_weight_for_a8w4_torch(codes[expert_name]).view(128, 32),
    )
    for name, scale in scales.items():
        suffix = "scale" if saved_prefix == "ffn" else "weight_scale"
        assert torch.equal(tensors[f"{name}.{suffix}"], scale.float())


@pytest.mark.parametrize("serialized_head", ["lm_head", "head"])
def test_v4_anchored_weight_renaming_preserves_native_head_name(tmp_path, serialized_head):
    model = _model(v4=True)
    model.lm_head = model.dense
    del model.dense
    model._weight_conversions = [WeightRenaming(
        source_patterns=r"^head\.weight$", target_patterns="lm_head.weight",
    )]
    codes, scales, _ = _checkpoint(tmp_path, model)
    if serialized_head == "head":
        path = tmp_path / "model.safetensors"
        tensors = load_file(path)
        save_file({
            key.replace("lm_head.", "head.", 1): tensor
            for key, tensor in tensors.items()
        }, path)

    exporter.export_a8w4_checkpoint(tmp_path, model, 100000)

    tensors = _load_output(tmp_path)
    assert torch.equal(tensors["head.weight"], codes["lm_head"])
    assert torch.equal(tensors["head.scale"], scales["lm_head"].float())
    assert not any(key.startswith("lm_head.") for key in tensors)
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(tensors)


@pytest.mark.parametrize("error_kind", ["scale", "zero_scale", "g_idx", "zero_point", "shape", "missing_scale"])
def test_validation_failure_leaves_input_unchanged(tmp_path, error_kind):
    model = _model()
    _checkpoint(tmp_path, model)
    path = tmp_path / "model.safetensors"
    tensors = load_file(path)
    name = "mlp.experts.0.weight"
    if error_kind in ("scale", "zero_scale"):
        tensors[f"{name}_scale"][0] = float("nan") if error_kind == "scale" else 0
    elif error_kind == "g_idx":
        tensors[f"{name}_g_idx"] = torch.zeros(64, dtype=torch.int32)
    elif error_kind == "zero_point":
        tensors[f"{name}_zero_point"][0] = 1
    elif error_kind == "shape":
        tensors[f"{name}_shape"] = torch.tensor([32, 128])
    else:
        tensors.pop(f"{name}_scale")
    save_file(tensors, path)
    before = _snapshot(tmp_path)
    with pytest.raises(ValueError):
        exporter.export_a8w4_checkpoint(tmp_path, model, 10000)
    assert _snapshot(tmp_path) == before
    assert not list(tmp_path.glob(".qlean-a8w4-*"))


def test_backend_packing_failure_leaves_input_unchanged(monkeypatch, tmp_path):
    model = _model()
    model.mlp.experts.append(torch.nn.Linear(64, 64, bias=False))
    model.mlp.experts[1].quantization_scheme = _scheme()
    _checkpoint(tmp_path, model)
    before = _snapshot(tmp_path)
    original_pack = exporter.preprocess_for_acext_a8w4
    calls = []

    def fail(codes):
        calls.append(codes.shape)
        if len(calls) == 2:
            raise RuntimeError("pure Torch packing failed")
        return original_pack(codes)

    monkeypatch.setattr(exporter, "preprocess_for_acext_a8w4", fail)
    with pytest.raises(RuntimeError, match="pure Torch packing failed"):
        exporter.export_a8w4_checkpoint(tmp_path, model, 3000, export_device="cpu")
    assert len(calls) == 2
    assert _snapshot(tmp_path) == before
    assert not list(tmp_path.glob(".qlean-a8w4-*"))


@pytest.mark.parametrize("cross_shard", [False, True])
@pytest.mark.parametrize("max_shard_size", [100000, 9000])
@pytest.mark.parametrize("failed_file", ["tensor", "model.safetensors.index.json", "config.json"])
def test_publication_failure_restores_original_checkpoint(
    monkeypatch, tmp_path, cross_shard, max_shard_size, failed_file
):
    model = _model()
    _checkpoint(tmp_path, model, cross_shard=cross_shard)
    before = _snapshot(tmp_path)
    original_replace = exporter.os.replace
    failures = []

    def fail(source, destination):
        source, destination = Path(source), Path(destination)
        staged = source.parent.name.startswith(".qlean-a8w4-") and not source.parent.name.startswith(".qlean-a8w4-recovery-")
        selected = destination.suffix == ".safetensors" if failed_file == "tensor" else destination.name == failed_file
        if staged and selected:
            failures.append(destination.name)
            raise OSError("publication failed")
        return original_replace(source, destination)

    monkeypatch.setattr(exporter.os, "replace", fail)
    with pytest.raises(OSError, match="publication failed"):
        exporter.export_a8w4_checkpoint(tmp_path, model, max_shard_size)
    assert failures
    assert _snapshot(tmp_path) == before
    assert not list(tmp_path.glob(".qlean-a8w4-*"))


@pytest.mark.parametrize("cross_shard", [False, True])
@pytest.mark.parametrize("restores_before_failure", [0, 1])
def test_partial_rollback_retains_all_original_hardlinks(
    monkeypatch, tmp_path, restores_before_failure, cross_shard
):
    model = _model()
    _checkpoint(tmp_path, model, cross_shard=cross_shard)
    before = _snapshot(tmp_path)
    original_inodes = {name: (tmp_path / name).stat().st_ino for name in before}
    original_replace = exporter.os.replace
    original_link = exporter.os.link
    restored = []
    linked = set()

    def record_link(source, destination, *args, **kwargs):
        result = original_link(source, destination, *args, **kwargs)
        destination = Path(destination)
        if destination.parent.name.startswith(".qlean-a8w4-recovery-") and destination.name in before:
            linked.add(destination.name)
            assert destination.stat().st_ino == original_inodes[destination.name]
        return result

    def fail(source, destination):
        source, destination = Path(source), Path(destination)
        recovery = source.parent.name.startswith(".qlean-a8w4-recovery-")
        if recovery:
            if len(restored) == restores_before_failure:
                raise OSError("rollback failed")
            result = original_replace(source, destination)
            restored.append(destination.name)
            return result
        if source.parent.name.startswith(".qlean-a8w4-") and destination.name == "config.json":
            raise OSError("publication failed")
        return original_replace(source, destination)

    monkeypatch.setattr(exporter.os, "link", record_link)
    monkeypatch.setattr(exporter.os, "replace", fail)
    with pytest.raises(RuntimeError, match="original files remain"):
        exporter.export_a8w4_checkpoint(tmp_path, model, 100000)
    assert len(restored) == restores_before_failure
    assert linked == set(before)
    recovery, = tmp_path.glob(".qlean-a8w4-recovery-*")
    # An interrupted restore may leave a temporary link as well as the originals.
    assert {name: (recovery / name).read_bytes() for name in before} == before
    assert {name: (recovery / name).stat().st_ino for name in before} == original_inodes
    for name in restored:
        assert (tmp_path / name).read_bytes() == before[name]


@pytest.mark.parametrize("shape", [(0, 64), (64, 0), (32, 64), (64, 32), (96, 128), (128, 96)])
def test_int4_shape_validation_before_hooks(shape):
    model = _model(shape=shape, dense_bits=None)
    modifier = GPTQModifier(config_groups={"int4": {
        "targets": [r"re:.*experts\.\d+$"],
        "weights": {"num_bits": 4, "type": "int", "strategy": "channel"},
    }})
    with pytest.raises(ValueError, match="nonempty|64"):
        modifier.on_initialize(SimpleNamespace(model=SimpleNamespace(model_=model)))
    assert not modifier._hessians
    assert not modifier._num_samples
    assert modifier._moe_calibration_ctx is None
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())


@pytest.mark.parametrize("overrides", [
    {"strategy": "group", "group_size": 32}, {"symmetric": False},
])
def test_rejects_incompatible_int4_contract(overrides):
    model = _model()
    model.mlp.experts[0].quantization_scheme = _scheme(**overrides)
    with pytest.raises(ValueError, match="symmetric per-channel"):
        exporter.a8w4_quantized_modules(model)


@pytest.mark.parametrize("shape", [(32, 64), (64, 64), (64, 128), (128, 64)])
def test_dense_int4_export_is_rejected_before_shape_validation(shape):
    model = _model(dense_shape=shape, dense_bits=4, experts=False)
    with pytest.raises(ValueError, match="only routed MoE expert") as raised:
        exporter.a8w4_quantized_modules(model, require_int4=True)
    assert "'dense'" in str(raised.value)
    assert "GPTQModifier.ignore" in str(raised.value)


def test_rejects_grouped_int8_in_mixed_a8w4():
    model = _model()
    model.dense.quantization_scheme = _scheme(8, strategy="group", group_size=32)
    with pytest.raises(ValueError, match="symmetric per-channel"):
        exporter.a8w4_quantized_modules(model)


def test_rejects_activation_only_quantization_in_a8w4():
    model = _model()
    model.other.quantization_scheme = QuantizationScheme(
        targets=["Linear"],
        input_activations=QuantizationArgs(num_bits=8, type="int", strategy="token", dynamic=True),
    )
    with pytest.raises(ValueError, match="activation quantization"):
        exporter.a8w4_quantized_modules(model)


def test_other_modifier_contract_is_checked_before_starting_hooks():
    model = _model()
    model.dense.quantization_scheme = _scheme(8, strategy="group", group_size=32)
    modifier = GPTQModifier()
    with pytest.raises(ValueError, match="symmetric per-channel"):
        modifier.on_start(SimpleNamespace(model=SimpleNamespace(model_=model)))
    assert not modifier._hessians
    assert not modifier._num_samples
    assert modifier._moe_calibration_ctx is None


def test_module_selection_has_no_native_prerequisite_or_check_backend_parameter(monkeypatch):
    monkeypatch.setitem(sys.modules, "acext", None)
    assert "check_backend" not in inspect.signature(exporter.a8w4_quantized_modules).parameters
    assert set(exporter.a8w4_quantized_modules(_model(), require_int4=True)) == {"mlp.experts.0", "dense"}
    model = torch.nn.Module()
    model.dense = torch.nn.Linear(7, 3, bias=False)
    model.dense.quantization_scheme = _scheme(8)
    assert exporter.a8w4_quantized_modules(model) == {}
    with pytest.raises(ValueError, match="at least one INT4 Linear"):
        exporter.a8w4_quantized_modules(model, require_int4=True)


def _int8_model(shape=(3, 7)):
    model = torch.nn.Module()
    model.dense = torch.nn.Linear(shape[1], shape[0], bias=False)
    model.dense.quantization_scheme = _scheme(8)
    model.other = torch.nn.Linear(7, 5, bias=False)
    return model


def _int8_checkpoint(directory, model, *, cross_shard=False, saved_name="dense"):
    n, k = model.dense.out_features, model.dense.in_features
    codes = (torch.arange(n * k).reshape(n, k) % 256 - 128).to(torch.int8)
    scale = (torch.arange(1, n + 1).reshape(n, 1) / 128).to(torch.bfloat16)
    untouched = torch.arange(35).reshape(5, 7).to(torch.bfloat16)
    tensors = {
        f"{saved_name}.weight": codes,
        f"{saved_name}.weight_scale": scale,
        f"{saved_name}.weight_zero_point": torch.zeros(n, 1, dtype=torch.int8),
        "other.weight": untouched,
    }
    if cross_shard:
        weights = {f"{saved_name}.weight": codes}
        params = {name: value for name, value in tensors.items() if name not in weights}
        save_file(weights, directory / "model-00001-of-00002.safetensors")
        save_file(params, directory / "model-00002-of-00002.safetensors")
        weight_map = {
            **{name: "model-00001-of-00002.safetensors" for name in weights},
            **{name: "model-00002-of-00002.safetensors" for name in params},
        }
        (directory / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": weight_map,
            "metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors.values())},
        }))
    else:
        save_file(tensors, directory / "model.safetensors")
    (directory / "config.json").write_text(json.dumps({
        "model_type": "test", "custom_setting": {"keep": True},
        "quantization_config": {
            "quant_method": "compressed-tensors", "format": "int-quantized",
            "config_groups": {"channel": _scheme(8).model_dump(mode="json")},
            "ignore": ["other"],
        },
    }))
    return codes, scale, untouched


def test_int8_export_never_resolves_int4_export_device(monkeypatch, tmp_path):
    model = _int8_model()
    _int8_checkpoint(tmp_path, model)

    def forbidden(*args, **kwargs):
        pytest.fail("INT8 export touched the INT4 device path")

    monkeypatch.setattr(exporter, "_Int4ExportDevice", forbidden)
    exporter.export_int8_checkpoint(tmp_path, model, 100000)


@pytest.mark.parametrize("shape", [(3, 7), (5, 10), (32, 96)])
@pytest.mark.parametrize("cross_shard", [False, True])
@pytest.mark.parametrize("max_shard_size", [100000, 150])
def test_int8_export_preserves_codes_fp32_scales_non64_shapes_and_index(
    tmp_path, shape, cross_shard, max_shard_size
):
    model = _int8_model(shape)
    codes, scales, untouched = _int8_checkpoint(tmp_path, model, cross_shard=cross_shard)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    exporter.export_int8_checkpoint(tmp_path, model, max_shard_size)
    tensors = _load_output(tmp_path)
    assert set(tensors) == {"dense.weight", "dense.weight_scale", "other.weight"}
    assert tensors["dense.weight"].dtype == torch.int8
    assert tensors["dense.weight"].shape == shape
    assert torch.equal(tensors["dense.weight"], codes)
    assert tensors["dense.weight_scale"].dtype == torch.float32
    assert tensors["dense.weight_scale"].shape == (shape[0], 1)
    assert torch.equal(tensors["dense.weight_scale"], scales.float())
    assert torch.equal(tensors["other.weight"], untouched)
    assert tensors["other.weight"].dtype == torch.bfloat16
    for name, value in before.items():
        assert model.state_dict()[name].dtype == value.dtype
        assert torch.equal(model.state_dict()[name], value)
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(tensors)
    assert index["metadata"]["total_size"] == sum(value.numel() * value.element_size() for value in tensors.values())
    assert set(index["weight_map"].values()) == {p.name for p in tmp_path.glob("*.safetensors")}
    for filename in set(index["weight_map"].values()):
        shard = load_file(tmp_path / filename)
        assert all(index["weight_map"][name] == filename for name in shard)
        size = sum(value.numel() * value.element_size() for value in shard.values())
        assert size <= max_shard_size or len(shard) == 1
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["custom_setting"] == {"keep": True}
    assert config["quantization_config"]["quant_method"] == "w8a8_int8"
    assert config["quantization_config"]["ignore"] == ["other"]
    assert not set(config["quantization_config"]) & {"format", "config_groups", "kv_cache_scheme"}
    exporter.validate_int8_checkpoint(tmp_path, model)
    assert not list(tmp_path.glob(".qlean-*"))


def test_int8_export_uses_pure_saved_weight_renamings_not_fused_converters(tmp_path):
    model = _int8_model()
    model._weight_conversions = [object(), WeightRenaming(
        source_patterns=r"^saved\.", target_patterns="dense.",
    )]
    codes, scales, _ = _int8_checkpoint(tmp_path, model, saved_name="saved")
    exporter.export_int8_checkpoint(tmp_path, model, 100000)
    tensors = _load_output(tmp_path)
    assert torch.equal(tensors["saved.weight"], codes)
    assert torch.equal(tensors["saved.weight_scale"], scales.float())
    assert not any(name.startswith("dense.") for name in tensors)
    exporter.validate_int8_checkpoint(tmp_path, model)


@pytest.mark.parametrize("overrides", [
    {"symmetric": False}, {"strategy": "group", "group_size": 4},
    {"strategy": "tensor"}, {"strategy": "tensor", "dynamic": True},
    {"type": "float"}, {"strategy": "block", "block_structure": [2, 2]},
])
def test_int8_module_validator_rejects_incompatible_companion(overrides):
    model = _int8_model()
    model.other.quantization_scheme = _scheme(8, **overrides)
    with pytest.raises(ValueError):
        exporter.int8_quantized_modules(model, require_int8=True)


@pytest.mark.parametrize("activation", ["input_activations", "output_activations"])
def test_int8_module_validator_rejects_activation_only_companion(activation):
    model = _int8_model()
    model.other.quantization_scheme = QuantizationScheme(targets=["Linear"], **{
        activation: QuantizationArgs(num_bits=8, type="int", strategy="token", dynamic=True),
    })
    with pytest.raises(ValueError, match="activation"):
        exporter.int8_quantized_modules(model, require_int8=True)


@pytest.mark.parametrize("shape", [(0, 7), (3, 0)])
def test_int8_module_validator_rejects_empty_linear(shape):
    with pytest.raises(ValueError, match="nonempty|dimensions"):
        exporter.int8_quantized_modules(_int8_model(shape), require_int8=True)


def test_int8_module_validator_rejects_non_linear_and_missing_target():
    model = _int8_model()
    model.other = torch.nn.Embedding(5, 7)
    model.other.quantization_scheme = _scheme(8)
    with pytest.raises(ValueError, match="Linear"):
        exporter.int8_quantized_modules(model, require_int8=True)
    empty = torch.nn.Module()
    assert exporter.int8_quantized_modules(empty) == {}
    with pytest.raises(ValueError, match="at least one|target|INT8"):
        exporter.int8_quantized_modules(empty, require_int8=True)


@pytest.mark.parametrize("error_kind", [
    "nan_scale", "zero_scale", "negative_scale", "infinite_scale", "scale_shape", "missing_scale",
    "nonzero_zp", "g_idx", "global_scale", "weight_shape", "int32_weight", "missing_weight", "nontarget_int8",
])
@pytest.mark.parametrize("cross_shard", [False, True])
def test_int8_input_validation_failure_retains_ct_checkpoint(tmp_path, error_kind, cross_shard):
    model = _int8_model()
    _int8_checkpoint(tmp_path, model, cross_shard=cross_shard)
    if cross_shard:
        index_path = tmp_path / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        shards = {name: load_file(tmp_path / name) for name in set(index["weight_map"].values())}
    else:
        shards = {"model.safetensors": load_file(tmp_path / "model.safetensors")}
    weight_file = next(name for name, values in shards.items() if "dense.weight" in values)
    scale_file = next(name for name, values in shards.items() if "dense.weight_scale" in values)
    weights, params = shards[weight_file], shards[scale_file]
    if error_kind in ("nan_scale", "zero_scale", "negative_scale", "infinite_scale"):
        params["dense.weight_scale"][0] = {
            "nan_scale": float("nan"), "zero_scale": 0,
            "negative_scale": -1, "infinite_scale": float("inf"),
        }[error_kind]
    elif error_kind == "scale_shape":
        params["dense.weight_scale"] = params["dense.weight_scale"].reshape(1, 3)
    elif error_kind == "missing_scale":
        params.pop("dense.weight_scale")
    elif error_kind == "nonzero_zp":
        params["dense.weight_zero_point"][0] = 1
    elif error_kind in ("g_idx", "global_scale"):
        params[f"dense.weight_{error_kind}"] = torch.ones(7)
    elif error_kind == "weight_shape":
        weights["dense.weight"] = torch.ones(7, 3, dtype=torch.int8)
    elif error_kind == "int32_weight":
        weights["dense.weight"] = weights["dense.weight"].int()
    elif error_kind == "missing_weight":
        weights.pop("dense.weight")
    else:
        params["other.weight"] = params["other.weight"].to(torch.int8)
    for filename, values in shards.items():
        save_file(values, tmp_path / filename)
    if cross_shard:
        index["weight_map"] = {key: filename for filename, values in shards.items() for key in values}
        index_path.write_text(json.dumps(index))
    before = _snapshot(tmp_path)
    with pytest.raises(ValueError):
        exporter.export_int8_checkpoint(tmp_path, model, 100000)
    assert _snapshot(tmp_path) == before
    assert not list(tmp_path.glob(".qlean-*"))


@pytest.mark.parametrize("error_kind", [
    "missing_indexed_scale", "wrong_shard", "duplicate", "scale_dtype", "zero_point", "packed", "shape",
    "ignore_ancestor", "method", "total_size",
])
def test_final_int8_validator_rejects_corruption_without_writes(tmp_path, error_kind):
    model = _int8_model()
    _int8_checkpoint(tmp_path, model, cross_shard=True)
    # Force codes and scales into distinct output shards before corrupting the index.
    exporter.export_int8_checkpoint(tmp_path, model, 25)
    index_path = tmp_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_path = tmp_path / index["weight_map"]["dense.weight"]
    scale_path = tmp_path / index["weight_map"]["dense.weight_scale"]
    if error_kind == "missing_indexed_scale":
        values = load_file(scale_path)
        values.pop("dense.weight_scale")
        save_file(values, scale_path)
    elif error_kind == "wrong_shard":
        index["weight_map"]["dense.weight_scale"] = weight_path.name
        index_path.write_text(json.dumps(index))
    elif error_kind == "duplicate":
        values = load_file(weight_path)
        values["dense.weight_scale"] = load_file(scale_path)["dense.weight_scale"]
        save_file(values, weight_path)
    elif error_kind == "scale_dtype":
        values = load_file(scale_path)
        values["dense.weight_scale"] = values["dense.weight_scale"].to(torch.bfloat16)
        save_file(values, scale_path)
    elif error_kind in ("zero_point", "packed", "shape"):
        values = load_file(weight_path)
        values[f"dense.weight_{error_kind}"] = torch.zeros(3, 1, dtype=torch.int32)
        save_file(values, weight_path)
        index["weight_map"][f"dense.weight_{error_kind}"] = weight_path.name
        index_path.write_text(json.dumps(index))
    elif error_kind in ("ignore_ancestor", "method"):
        path = tmp_path / "config.json"
        config = json.loads(path.read_text())
        if error_kind == "ignore_ancestor":
            config["quantization_config"]["ignore"].append("dense")
        else:
            config["quantization_config"]["quant_method"] = "compressed-tensors"
        path.write_text(json.dumps(config))
    else:
        index["metadata"]["total_size"] += 1
        index_path.write_text(json.dumps(index))
    before = _snapshot(tmp_path)
    with pytest.raises(ValueError):
        exporter.validate_int8_checkpoint(tmp_path, model)
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("failed_file", ["tensor", "model.safetensors.index.json", "config.json"])
def test_int8_publication_failure_restores_ct_input(monkeypatch, tmp_path, failed_file):
    model = _int8_model()
    _int8_checkpoint(tmp_path, model, cross_shard=True)
    before = _snapshot(tmp_path)
    original_replace = exporter.os.replace
    failures = []

    def fail(source, destination):
        source, destination = Path(source), Path(destination)
        staged = source.parent != tmp_path and "recovery" not in source.parent.name
        selected = destination.suffix == ".safetensors" if failed_file == "tensor" else destination.name == failed_file
        if staged and selected:
            failures.append(destination.name)
            raise OSError("INT8 publication failed")
        return original_replace(source, destination)

    monkeypatch.setattr(exporter.os, "replace", fail)
    with pytest.raises(OSError, match="INT8 publication failed"):
        exporter.export_int8_checkpoint(tmp_path, model, 150)
    assert failures
    assert _snapshot(tmp_path) == before
    assert not list(tmp_path.glob(".qlean-*"))


@pytest.mark.parametrize("restores_before_failure", [0, 1])
def test_int8_partial_rollback_keeps_all_original_hardlinks(monkeypatch, tmp_path, restores_before_failure):
    model = _int8_model()
    _int8_checkpoint(tmp_path, model, cross_shard=True)
    before = _snapshot(tmp_path)
    original_inodes = {name: (tmp_path / name).stat().st_ino for name in before}
    original_replace = exporter.os.replace
    restored = []

    def fail(source, destination):
        source, destination = Path(source), Path(destination)
        if "recovery" in source.parent.name:
            if len(restored) == restores_before_failure:
                raise OSError("INT8 rollback failed")
            result = original_replace(source, destination)
            restored.append(destination.name)
            return result
        if source.parent != tmp_path and destination.name == "config.json":
            raise OSError("INT8 publication failed")
        return original_replace(source, destination)

    monkeypatch.setattr(exporter.os, "replace", fail)
    with pytest.raises(RuntimeError, match="original files remain"):
        exporter.export_int8_checkpoint(tmp_path, model, 150)
    assert len(restored) == restores_before_failure
    recovery, = tmp_path.glob(".qlean-*recovery-*")
    assert {name: (recovery / name).read_bytes() for name in before} == before
    assert {name: (recovery / name).stat().st_ino for name in before} == original_inodes


def test_int8_validator_rejects_kv_on_unselected_module():
    model = _int8_model()
    model.other.kv_cache_scheme = QuantizationArgs(num_bits=8, type="int", strategy="tensor")
    with pytest.raises(ValueError, match="KV"):
        exporter.int8_quantized_modules(model, require_int8=True)


def test_int4_takes_priority_over_channel_int8_selection():
    model = _model()
    assert exporter.int8_quantized_modules(model) == {}
    assert set(exporter.a8w4_quantized_modules(model, require_int4=True)) == {"mlp.experts.0", "dense"}


def test_int8_export_preserves_non_target_auxiliary_qparams(tmp_path):
    model = _int8_model()
    _int8_checkpoint(tmp_path, model)
    path = tmp_path / "model.safetensors"
    tensors = load_file(path)
    auxiliary = {
        "other.weight_scale": torch.ones(5, 1, dtype=torch.bfloat16),
        "other.weight_shape": torch.tensor([5, 7]),
        "other.weight_zero_point": torch.zeros(5, 1, dtype=torch.int8),
        "other.weight_g_idx": torch.arange(7, dtype=torch.int32),
    }
    save_file({**tensors, **auxiliary}, path)
    exporter.export_int8_checkpoint(tmp_path, model, 100000)
    output = _load_output(tmp_path)
    for name, value in auxiliary.items():
        assert output[name].dtype == value.dtype
        assert torch.equal(output[name], value)
    exporter.validate_int8_checkpoint(tmp_path, model)


@pytest.mark.parametrize("invalid", ["zero_scale", "missing_weight"])
def test_int8_preservation_corruption_is_validated_in_staging(monkeypatch, tmp_path, invalid):
    model = _int8_model()
    _int8_checkpoint(tmp_path, model)
    before = _snapshot(tmp_path)
    checkpoint = {"sentinel": "already-prepared-source", "mtp_keys": []}
    calls = []

    def corrupt(source_model, prepared, directory, max_shard_size):
        assert source_model is model and prepared is checkpoint
        directory = Path(directory)
        assert directory != tmp_path
        calls.append(directory)
        index = json.loads((directory / "model.safetensors.index.json").read_text())
        key = "dense.weight_scale" if invalid == "zero_scale" else "dense.weight"
        path = directory / index["weight_map"][key]
        values = load_file(path)
        if invalid == "zero_scale":
            values[key][0] = 0
        else:
            values.pop(key)
        save_file(values, path)

    monkeypatch.setattr(exporter, "finalize_checkpoint_preservation", corrupt)
    with pytest.raises(ValueError):
        exporter.export_int8_checkpoint(tmp_path, model, 100000, checkpoint=checkpoint)
    assert len(calls) == 1
    assert _snapshot(tmp_path) == before
    assert not list(tmp_path.glob(".qlean-*"))


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="CPU gloo unavailable")
def test_real_gloo_parallel_export_fallback_and_failure(tmp_path):
    script = r'''
import importlib.machinery
import json
import os
import sys
import types
from datetime import timedelta
from pathlib import Path

import torch
import torch._dynamo
import compressed_tensors
import transformers
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from safetensors.torch import save_file

class UnavailableKernel:
    def __init__(self, function): self.__name__ = function.__name__
    def __call__(self, *args, **kwargs): raise RuntimeError("Triton execution unavailable")
    def __getitem__(self, grid): return self

triton = types.ModuleType("triton")
triton.__spec__ = importlib.machinery.ModuleSpec("triton", loader=None)
triton.jit = triton.Config = UnavailableKernel
language = types.ModuleType("triton.language")
language.__spec__ = importlib.machinery.ModuleSpec("triton.language", loader=None)
language.constexpr = lambda value: value
triton.language = language
sys.modules["triton"] = triton
sys.modules["triton.language"] = language
sys.path.insert(0, sys.argv[1])

import qlean.modifiers.gptq.distributed as distributed
import qlean.modifiers.gptq.export as exporter

rank = int(sys.argv[2])
root = Path(sys.argv[3])
source = 1
torch.distributed.init_process_group(
    "gloo", init_method=sys.argv[4], rank=rank, world_size=3,
    timeout=timedelta(seconds=30),
)
distributed.is_source_process = lambda: rank == source


def scheme(bits):
    return QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=bits, type="int", strategy="channel", symmetric=True,
        ),
    )


def make_model():
    model = torch.nn.Module()
    model.mlp = torch.nn.Module()
    model.mlp.experts = torch.nn.ModuleList([
        torch.nn.Linear(64, 64, bias=False),
    ])
    model.mlp.experts[0].quantization_scheme = scheme(4)
    model.dense = torch.nn.Linear(64, 64, bias=False)
    model.dense.quantization_scheme = scheme(8)
    model.other = torch.nn.Linear(64, 64, bias=False)
    return model


model = make_model()


def create_input(name):
    directory = root / name
    if rank == source:
        directory.mkdir()
        tensors = {}
        for module_name, module in exporter.a8w4_quantized_modules(model).items():
            bits = module.quantization_scheme.weights.num_bits
            n, k = module.out_features, module.in_features
            values = torch.arange(n * k).reshape(n, k) + 3 * torch.arange(n).reshape(n, 1)
            codes = (values % (2 ** bits) - 2 ** (bits - 1)).to(torch.int8)
            tensors[f"{module_name}.weight_packed"] = pack_to_int32(codes, bits).contiguous()
            tensors[f"{module_name}.weight_scale"] = (
                torch.arange(1, n + 1).reshape(n, 1) / 64
            ).to(torch.bfloat16)
            tensors[f"{module_name}.weight_shape"] = torch.tensor([n, k])
            tensors[f"{module_name}.weight_zero_point"] = torch.zeros(n, 1, dtype=torch.int8)
        tensors["other.weight"] = torch.arange(64 * 64).reshape(64, 64).to(torch.bfloat16)
        save_file(tensors, directory / "model.safetensors")
        (directory / "config.json").write_text(json.dumps({
            "model_type": "test",
            "quantization_config": {"format": "pack-quantized"},
        }))
    torch.distributed.barrier()
    return directory


def snapshot(directory):
    return {
        path.name: path.read_bytes()
        for path in directory.iterdir()
        if path.is_file()
    }


reference = create_input("reference")
success = create_input("success")
fallback = create_input("fallback")
failure = create_input("failure")
if rank == source:
    exporter.export_a8w4_checkpoint(reference, model, 9000, expert_patterns=[])
torch.distributed.barrier()

original_write = exporter._write_staged_shards
original_index = exporter._write_staged_index
original_publish = exporter._publish_staged_checkpoint
writes = []
index_calls = 0
publish_calls = 0


def observe_write(*args, **kwargs):
    writes.extend(args[7])
    return original_write(*args, **kwargs)


def observe_index(*args, **kwargs):
    global index_calls
    index_calls += 1
    assert rank == source
    return original_index(*args, **kwargs)


def observe_publish(*args, **kwargs):
    global publish_calls
    publish_calls += 1
    assert rank == source
    return original_publish(*args, **kwargs)


exporter._write_staged_shards = observe_write
exporter._write_staged_index = observe_index
exporter._publish_staged_checkpoint = observe_publish
exporter.export_a8w4_checkpoint(
    success, model, 9000, expert_patterns=[], process_group=torch.distributed.group.WORLD,
)
observed = [None] * 3
torch.distributed.all_gather_object(observed, (writes[:], index_calls, publish_calls))
with (success / "model.safetensors.index.json").open() as file:
    output_shards = set(json.load(file)["weight_map"].values())
assigned = [filename for rank_writes, _, _ in observed for filename in rank_writes]
assert set(assigned) == output_shards
assert len(assigned) == len(set(assigned))
assert sum(bool(rank_writes) for rank_writes, _, _ in observed) == 2
assert sum(not rank_writes for rank_writes, _, _ in observed) == 1
assert [index for _, index, _ in observed] == [0, 1, 0]
assert [publish for _, _, publish in observed] == [0, 1, 0]
assert snapshot(success) == snapshot(reference)

writes.clear()
index_calls = 0
publish_calls = 0
original_read_text = Path.read_text
if rank == 2:
    def hide_marker(path, *args, **kwargs):
        if path.name == ".qlean-shared-marker":
            raise OSError("rank-local staging")
        return original_read_text(path, *args, **kwargs)
    Path.read_text = hide_marker
try:
    exporter.export_a8w4_checkpoint(
        fallback, model, 9000, expert_patterns=[], process_group=torch.distributed.group.WORLD,
    )
finally:
    Path.read_text = original_read_text
observed = [None] * 3
torch.distributed.all_gather_object(observed, (writes[:], index_calls, publish_calls))
assert not observed[0][0] and observed[1][0] and not observed[2][0]
assert [index for _, index, _ in observed] == [0, 1, 0]
assert [publish for _, _, publish in observed] == [0, 1, 0]
assert snapshot(fallback) == snapshot(reference)

writes.clear()
index_calls = 0
publish_calls = 0
before_failure = snapshot(failure)


def fail_worker(*args, **kwargs):
    writes.extend(args[7])
    if rank != source:
        raise RuntimeError("worker sentinel")
    return original_write(*args, **kwargs)


exporter._write_staged_shards = fail_worker
caught = None
try:
    exporter.export_a8w4_checkpoint(
        failure, model, 9000, expert_patterns=[], process_group=torch.distributed.group.WORLD,
    )
except RuntimeError as error:
    caught = str(error)
errors = [None] * 3
torch.distributed.all_gather_object(errors, caught)
assert all(
    error is not None
    and "Distributed GPTQ shard export failed" in error
    and "worker sentinel" in error
    for error in errors
)
assert snapshot(failure) == before_failure
assert not list(failure.glob(".qlean-a8w4-*"))

exporter._write_staged_shards = observe_write
for target in ("mkdtemp", "_prepare_export_checkpoint", "_write_staged_index",
               "_publish_staged_checkpoint"):
    original = getattr(exporter, target)
    def fail_source(*args, **kwargs):
        if rank == source:
            raise RuntimeError("source sentinel " + target)
        return original(*args, **kwargs)
    setattr(exporter, target, fail_source)
    caught = None
    try:
        exporter.export_a8w4_checkpoint(
            failure, model, 9000, expert_patterns=[], process_group=torch.distributed.group.WORLD,
        )
    except RuntimeError as error:
        caught = str(error)
    finally:
        setattr(exporter, target, original)
    source_errors = [None] * 3
    torch.distributed.all_gather_object(source_errors, caught)
    assert len(set(source_errors)) == 1
    assert source_errors[0] is not None and "source sentinel " + target in source_errors[0]
    assert snapshot(failure) == before_failure
    assert not list(failure.glob(".qlean-a8w4-*"))

torch.distributed.barrier()
if rank == 0:
    print("DISTRIBUTED_EXPORT_OK=" + json.dumps({
        "parallel_assignments": [entry[0] for entry in observed],
        "failure": errors[0],
    }), flush=True)
torch.distributed.destroy_process_group()
'''
    source_root = Path(exporter.__file__).resolve().parents[3]
    rendezvous = "file://" + str(tmp_path / "gloo-export")
    workers = [subprocess.Popen(
        [sys.executable, "-c", script, str(source_root), str(rank), str(tmp_path), rendezvous],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ) for rank in range(3)]
    try:
        outputs = [worker.communicate(timeout=120) for worker in workers]
        for worker, (stdout, stderr) in zip(workers, outputs):
            assert worker.returncode == 0, stdout + stderr
        assert "DISTRIBUTED_EXPORT_OK=" in outputs[0][0]
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.communicate()


def _mxfp4_model(count=6):
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([
        torch.nn.Linear(64, 64, bias=False) for _ in range(count)
    ])
    for module in model.layers:
        module.quantization_scheme = QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                num_bits=4, type="float", strategy="group", group_size=32,
            ),
        )
    model.other = torch.nn.Linear(7, 5, bias=False)
    return model


def _mxfp4_checkpoint(directory, model, *, sharded=True):
    shards = {}
    count = len(model.layers) + 1
    for number, _ in enumerate(model.layers):
        prefix = f"layers.{number}.weight"
        filename = (
            f"model-{number + 1:05d}-of-{count:05d}.safetensors"
            if sharded else "model.safetensors"
        )
        shards.setdefault(filename, {}).update({
            f"{prefix}_packed": (torch.arange(2048).reshape(64, 32) % 256).to(torch.uint8),
            f"{prefix}_scale": (torch.arange(128).reshape(64, 2) % 255).to(torch.uint8),
            f"{prefix}_shape": torch.tensor([64, 64]),
            # MXFP4 export is a raw rename, not the INT8/INT4 qparam consumer.
            f"{prefix}_zero_point": torch.arange(128).reshape(64, 2).to(torch.int8),
        })
    untouched = f"model-{count:05d}-of-{count:05d}.safetensors" if sharded else None
    shards.setdefault(untouched or "model.safetensors", {}).update({
        "other.weight": torch.arange(35).reshape(5, 7).to(torch.bfloat16),
        "other.weight_scale": torch.ones(5, 1, dtype=torch.bfloat16),
        "other.weight_shape": torch.tensor([5, 7]),
        "other.weight_zero_point": torch.ones(5, 1, dtype=torch.int8),
        "other.weight_packed_backup": torch.arange(7, dtype=torch.int32),
    })
    for filename, values in shards.items():
        save_file(values, directory / filename, metadata={"format": "pt", "source": "CT"})
    weight_map = {key: filename for filename, values in shards.items() for key in values}
    tensors = {key: value for values in shards.values() for key, value in values.items()}
    if sharded:
        (directory / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": weight_map,
            "metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors.values())},
        }))
    config = {
        "model_type": "test", "custom_setting": {"keep": True},
        "quantization_config": {
            "quant_method": "compressed-tensors", "format": "mxfp4-pack-quantized",
            "config_groups": {"mxfp4": model.layers[0].quantization_scheme.model_dump(mode="json")},
            "ignore": ["other"], "custom_quant_setting": {"keep": True},
            "global_compression_ratio": 4, "kv_cache_scheme": None,
            "quantization_status": "compressed", "sparsity_config": {},
            "transform_config": {}, "version": "test",
        },
    }
    (directory / "config.json").write_text(json.dumps(config))
    return tensors, weight_map, untouched


def _non_int4_checkpoint(directory, kind):
    directory.mkdir()
    if kind == "mxfp4":
        model = _mxfp4_model()
        _mxfp4_checkpoint(directory, model)
        return model, 80
    model = _int8_model((16, 32))
    _int8_checkpoint(directory, model, cross_shard=True)
    path = directory / "model-00002-of-00002.safetensors"
    values = load_file(path)
    values.update({f"buffer{number}": torch.full((40,), number, dtype=torch.bfloat16) for number in range(6)})
    save_file(values, path)
    index_path = directory / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"].update({key: path.name for key in values})
    tensors = _load_output(directory)
    index["metadata"]["total_size"] = sum(t.numel() * t.element_size() for t in tensors.values())
    index_path.write_text(json.dumps(index))
    return model, 80


def _export_non_int4(kind, directory, model, max_shard_size, **kwargs):
    function = exporter.export_int8_checkpoint if kind == "int8" else exporter.export_mxfp4_checkpoint
    return function(directory, model, max_shard_size, **kwargs)


def _assert_exact_index(directory):
    tensors = _load_output(directory)
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(tensors)
    assert index["metadata"]["total_size"] == sum(t.numel() * t.element_size() for t in tensors.values())
    assert set(index["weight_map"].values()) == {path.name for path in directory.glob("*.safetensors")}
    for filename in set(index["weight_map"].values()):
        assert {key for key, owner in index["weight_map"].items() if owner == filename} == set(load_file(directory / filename))
    return index


@pytest.mark.parametrize("kind", ["int8", "mxfp4"])
@pytest.mark.parametrize("workers", [1, 2, 3])
def test_non_int4_real_shards_overlap_bounded_and_have_disjoint_ownership(
    monkeypatch, tmp_path, kind, workers
):
    directory = tmp_path / kind
    model, max_shard_size = _non_int4_checkpoint(directory, kind)
    coordinator = threading.get_ident()
    lock = threading.Lock()
    barrier = threading.Barrier(workers, timeout=15) if workers > 1 else None
    calls, active, peak = [], 0, 0
    original_write = exporter._write_staged_shard

    def observe(*args, **kwargs):
        nonlocal active, peak
        with lock:
            position = len(calls)
            calls.append((args[7], threading.get_ident(), kwargs.get("pack_device")))
            active += 1
            peak = max(peak, active)
        try:
            if barrier is not None and position < workers:
                barrier.wait()
            assert kwargs["label"] == kind.upper()
            result = original_write(*args, **kwargs)
            assert result == args[7]
            assert (Path(args[1]) / result).is_file()
            assert all(value.device.type == "cpu" for value in load_file(Path(args[1]) / result).values())
            return result
        finally:
            with lock:
                active -= 1

    def guard(function):
        def guarded(*args, **kwargs):
            assert threading.get_ident() == coordinator, "worker touched distributed/CUDA state"
            return function(*args, **kwargs)
        return guarded

    def forbidden(*args, **kwargs):
        pytest.fail("non-INT4 export resolved an INT4 packing device")

    monkeypatch.setattr(exporter, "_write_staged_shard", observe)
    monkeypatch.setattr(exporter, "_Int4ExportDevice", forbidden)
    for name in ("is_initialized", "get_rank", "get_world_size"):
        monkeypatch.setattr(exporter.dist, name, guard(getattr(exporter.dist, name)))
    for name in ("is_available", "is_initialized", "current_device", "device_count", "set_device"):
        monkeypatch.setattr(torch.cuda, name, guard(getattr(torch.cuda, name)))

    _export_non_int4(kind, directory, model, max_shard_size, workers=workers)

    index = _assert_exact_index(directory)
    filenames = [filename for filename, _, _ in calls]
    written = set(index["weight_map"].values())
    if kind == "mxfp4":
        written.remove("model-00007-of-00007.safetensors")
    assert set(filenames) == written
    assert len(filenames) == len(set(filenames))
    assert len(filenames) > workers
    assert peak == workers and active == 0
    ids = {ident for _, ident, _ in calls}
    if workers == 1:
        assert ids == {coordinator}
    else:
        assert coordinator not in ids
        assert len(ids) == workers
    assert all(device is None for _, _, device in calls)
    assert not list(directory.glob(".qlean-*"))


@pytest.mark.parametrize("kind", ["int8", "mxfp4"])
def test_non_int4_serial_and_threaded_payload_config_and_index_parity(tmp_path, kind):
    outputs = []
    for workers in (1, 3):
        directory = tmp_path / str(workers)
        model, max_shard_size = _non_int4_checkpoint(directory, kind)
        source = _load_output(directory)
        source_index = json.loads((directory / "model.safetensors.index.json").read_text())
        original_state = {key: value.clone() for key, value in model.state_dict().items()}
        untouched = directory / "model-00007-of-00007.safetensors" if kind == "mxfp4" else None
        untouched_inode = untouched.stat().st_ino if untouched else None
        untouched_bytes = untouched.read_bytes() if untouched else None
        _export_non_int4(kind, directory, model, max_shard_size, workers=workers)
        tensors = _load_output(directory)
        index = _assert_exact_index(directory)
        if kind == "int8":
            expected = {key: value for key, value in source.items() if key != "dense.weight_zero_point"}
            expected["dense.weight_scale"] = source["dense.weight_scale"].float()
            assert tensors["dense.weight"].dtype == torch.int8
            assert tensors["dense.weight_scale"].dtype == torch.float32
            exporter.validate_int8_checkpoint(directory, model)
        else:
            expected = {key.removesuffix("_packed"): value for key, value in source.items()}
            assert index["weight_map"] == {key.removesuffix("_packed"): filename for key, filename in source_index["weight_map"].items()}
            assert untouched.stat().st_ino == untouched_inode
            assert untouched.read_bytes() == untouched_bytes
        assert set(tensors) == set(expected)
        for key, value in expected.items():
            assert tensors[key].dtype == value.dtype
            assert tensors[key].shape == value.shape
            assert torch.equal(tensors[key].view(torch.uint8), value.contiguous().view(torch.uint8))
        for key, value in original_state.items():
            assert torch.equal(model.state_dict()[key], value)
        snapshot = _snapshot(directory)
        if untouched is not None:
            snapshot.pop(untouched.name)
        outputs.append(snapshot)
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("workers", [1, 2])
def test_mxfp4_raw_rename_preserves_layout_qparams_and_wa_config(
    monkeypatch, tmp_path, sharded, workers
):
    import qlean.core.checkpoint as checkpoint_utils

    model = _mxfp4_model(2)
    source, weight_map, untouched = _mxfp4_checkpoint(tmp_path, model, sharded=sharded)
    original_files = set(_snapshot(tmp_path))
    original_inode = (tmp_path / untouched).stat().st_ino if untouched else None
    original_wa = exporter.WA_for_sglang_config
    original_write = exporter._write_staged_shard
    wa_calls, staged_links = [], []

    def forbidden(*args, **kwargs):
        pytest.fail("MXFP4 export used the shared destructive packed-weight renamer")

    def observe_wa(path, *args, **kwargs):
        assert Path(path).parent != tmp_path
        wa_calls.append(Path(path))
        return original_wa(path, *args, **kwargs)

    def observe_write(*args, **kwargs):
        assert not args[4] and not args[5], "MXFP4 must not repack weights or convert scales"
        return original_write(*args, **kwargs)

    original_link = exporter.os.link

    def observe_link(source_path, destination, *args, **kwargs):
        result = original_link(source_path, destination, *args, **kwargs)
        destination = Path(destination)
        if untouched and destination.name == untouched and "recovery" not in destination.parent.name:
            assert destination.stat().st_ino == original_inode
            staged_links.append(destination)
        return result

    monkeypatch.setattr(checkpoint_utils, "rename_packed_weights", forbidden)
    monkeypatch.setattr(exporter, "WA_for_sglang_config", observe_wa)
    monkeypatch.setattr(exporter, "_write_staged_shard", observe_write)
    monkeypatch.setattr(exporter.os, "link", observe_link)
    exporter.export_mxfp4_checkpoint(tmp_path, model, 1, workers=workers)

    tensors = _load_output(tmp_path)
    expected = {key.removesuffix("_packed"): value for key, value in source.items()}
    assert set(tensors) == set(expected)
    for key, value in expected.items():
        assert tensors[key].dtype == value.dtype and tensors[key].shape == value.shape
        assert torch.equal(tensors[key].view(torch.uint8), value.contiguous().view(torch.uint8))
    assert set(_snapshot(tmp_path)) == original_files
    assert len(wa_calls) == 1
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["custom_setting"] == {"keep": True}
    assert config["quantization_config"] == {
        "quant_method": "mxfp4", "ignore": ["other"], "custom_quant_setting": {"keep": True},
    }
    if sharded:
        assert _assert_exact_index(tmp_path)["weight_map"] == {key.removesuffix("_packed"): filename for key, filename in weight_map.items()}
        assert staged_links
        assert (tmp_path / untouched).stat().st_ino == original_inode
    else:
        assert not (tmp_path / "model.safetensors.index.json").exists()
    assert not list(tmp_path.glob(".qlean-*"))


@pytest.mark.parametrize("collision", ["same_shard", "cross_shard"])
def test_mxfp4_rename_collision_is_rejected_before_staging(monkeypatch, tmp_path, collision):
    model = _mxfp4_model(2)
    _, weight_map, untouched = _mxfp4_checkpoint(tmp_path, model)
    filename = weight_map["layers.0.weight_packed"] if collision == "same_shard" else untouched
    path = tmp_path / filename
    values = load_file(path)
    values["layers.0.weight"] = torch.ones(64, 32, dtype=torch.uint8)
    save_file(values, path)
    index_path = tmp_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"]["layers.0.weight"] = filename
    index["metadata"]["total_size"] += 2048
    index_path.write_text(json.dumps(index))
    before = _snapshot(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("rename collision was not validated before staging")

    monkeypatch.setattr(exporter, "_write_staged_shard", forbidden)
    monkeypatch.setattr(exporter, "mkdtemp", forbidden)
    monkeypatch.setattr(exporter, "TemporaryDirectory", forbidden)
    with pytest.raises(ValueError, match="[Cc]ollision|[Dd]uplicate|overwrite"):
        exporter.export_mxfp4_checkpoint(tmp_path, model, 100000, workers=2)
    assert _snapshot(tmp_path) == before
    assert not list(tmp_path.glob(".qlean-*"))


@pytest.mark.parametrize("invalid", [0, -1, 1.5, "2", True, False, None])
def test_validate_export_workers_rejects_invalid_types_and_values(invalid):
    with pytest.raises((ValueError, TypeError), match="workers"):
        exporter.validate_export_workers(invalid)


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_validate_export_workers_accepts_positive_integers(workers):
    exporter.validate_export_workers(workers)


@pytest.mark.parametrize("entrypoint", ["int8", "mxfp4", "finalize_int8", "finalize_mxfp4"])
@pytest.mark.parametrize("invalid", [0, -2, 2.0, "2", True, False, None])
def test_invalid_workers_are_rejected_before_checkpoint_writes(monkeypatch, tmp_path, entrypoint, invalid):
    kind = entrypoint.removeprefix("finalize_")
    directory = tmp_path / kind
    model, max_shard_size = _non_int4_checkpoint(directory, kind)
    before = _snapshot(directory)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid workers reached a staging/write operation")

    for name in ("mkdtemp", "TemporaryDirectory", "save_file", "_write_staged_shard", "_publish_staged_checkpoint"):
        monkeypatch.setattr(exporter, name, forbidden)
    with pytest.raises((ValueError, TypeError), match="workers"):
        if entrypoint.startswith("finalize_"):
            exporter.finalize_gptq_checkpoint(
                model, directory, max_shard_size,
                "int-quantized" if kind == "int8" else "mxfp4-pack-quantized", [],
                workers=invalid,
            )
        else:
            _export_non_int4(kind, directory, model, max_shard_size, workers=invalid)
    assert _snapshot(directory) == before
    assert not list(directory.glob(".qlean-*"))


@pytest.mark.parametrize("kind", ["int8", "mxfp4"])
def test_non_int4_thread_failure_drains_before_cleanup_and_retains_ct_input(
    monkeypatch, tmp_path, kind
):
    directory = tmp_path / kind
    model, max_shard_size = _non_int4_checkpoint(directory, kind)
    before = _snapshot(directory)
    barrier = threading.Barrier(2, timeout=15)
    failure_started = threading.Event()
    lock = threading.Lock()
    active, started, finished, cleanup = set(), [], [], []
    original_write = exporter._write_staged_shard
    original_rmtree = exporter.shutil.rmtree

    def fail_one(*args, **kwargs):
        filename, ident = args[7], threading.get_ident()
        with lock:
            position = len(started)
            started.append((filename, ident))
            active.add(filename)
        try:
            if position < 2:
                barrier.wait()
            if position == 0:
                failure_started.set()
                raise RuntimeError("threaded shard sentinel")
            assert failure_started.wait(15)
            assert Path(args[1]).is_dir(), "staging was cleaned while a writer was active"
            return original_write(*args, **kwargs)
        finally:
            with lock:
                active.remove(filename)
                finished.append(filename)

    def observe_cleanup(path, *args, **kwargs):
        if Path(path).parent == directory and Path(path).name.startswith(".qlean-"):
            with lock:
                assert not active, "cleanup raced an active shard writer"
                cleanup.append(Path(path))
        return original_rmtree(path, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("failed threaded export attempted publication")

    monkeypatch.setattr(exporter, "_write_staged_shard", fail_one)
    monkeypatch.setattr(exporter.shutil, "rmtree", observe_cleanup)
    monkeypatch.setattr(exporter, "_publish_staged_checkpoint", forbidden)
    with pytest.raises(RuntimeError, match="threaded shard sentinel"):
        _export_non_int4(kind, directory, model, max_shard_size, workers=2)
    assert len({ident for _, ident in started}) == 2
    assert set(finished) == {filename for filename, _ in started}
    assert not active and cleanup
    assert _snapshot(directory) == before
    assert not list(directory.glob(".qlean-*"))


@pytest.mark.parametrize("failed_file", ["tensor", "model.safetensors.index.json", "config.json"])
def test_mxfp4_threaded_publication_failure_restores_ct_input(monkeypatch, tmp_path, failed_file):
    model = _mxfp4_model(3)
    _mxfp4_checkpoint(tmp_path, model)
    before = _snapshot(tmp_path)
    inodes = {name: (tmp_path / name).stat().st_ino for name in before}
    original_replace = exporter.os.replace
    failures = []

    def fail(source, destination):
        source, destination = Path(source), Path(destination)
        staged = source.parent != tmp_path and "recovery" not in source.parent.name
        selected = destination.suffix == ".safetensors" if failed_file == "tensor" else destination.name == failed_file
        if staged and selected:
            failures.append(destination.name)
            raise OSError("MXFP4 publication sentinel")
        return original_replace(source, destination)

    monkeypatch.setattr(exporter.os, "replace", fail)
    with pytest.raises(OSError, match="MXFP4 publication sentinel"):
        exporter.export_mxfp4_checkpoint(tmp_path, model, 100000, workers=2)
    assert failures
    assert _snapshot(tmp_path) == before
    assert {name: (tmp_path / name).stat().st_ino for name in before} == inodes
    assert not list(tmp_path.glob(".qlean-*"))


@pytest.mark.parametrize("corruption", ["wrong_shard", "duplicate", "missing_index_key"])
def test_mxfp4_rejects_header_index_corruption_without_writes(monkeypatch, tmp_path, corruption):
    model = _mxfp4_model(2)
    _, weight_map, untouched = _mxfp4_checkpoint(tmp_path, model)
    index_path = tmp_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    key = "layers.0.weight_packed"
    if corruption == "wrong_shard":
        index["weight_map"][key] = untouched
    elif corruption == "duplicate":
        path = tmp_path / untouched
        values = load_file(path)
        values[key] = load_file(tmp_path / weight_map[key])[key]
        save_file(values, path)
    elif corruption == "missing_index_key":
        index["weight_map"].pop(key)
    else:
        index["metadata"]["total_size"] += 1
    index_path.write_text(json.dumps(index))
    before = _snapshot(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid MXFP4 headers/index reached a writer")

    monkeypatch.setattr(exporter, "_write_staged_shard", forbidden)
    with pytest.raises(ValueError):
        exporter.export_mxfp4_checkpoint(tmp_path, model, 100000, workers=2)
    assert _snapshot(tmp_path) == before
    assert not list(tmp_path.glob(".qlean-*"))


def test_int4_default_auto_resolves_once_for_all_serial_shards(monkeypatch, tmp_path):
    import qlean.utils.dev as dev

    model = _model()
    codes, _, _ = _checkpoint(tmp_path, model)
    coordinator = threading.get_ident()
    resolved, pack_devices, filenames = [], [], []
    original_write = exporter._write_staged_shard

    def get_device():
        assert threading.get_ident() == coordinator
        resolved.append(coordinator)
        return torch.device("cpu")

    def observe(*args, **kwargs):
        assert threading.get_ident() == coordinator
        filenames.append(args[7])
        pack_devices.append(kwargs["pack_device"])
        return original_write(*args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("INT4 export must remain serial within each rank")

    monkeypatch.setattr(dev, "get_main_device", get_device)
    monkeypatch.setattr(exporter, "_write_staged_shard", observe)
    monkeypatch.setattr(exporter, "ThreadPoolExecutor", forbidden)
    exporter.export_a8w4_checkpoint(tmp_path, model, 3000)
    assert resolved == [coordinator]
    assert len(filenames) > 1 and len(filenames) == len(set(filenames))
    assert len({id(device) for device in pack_devices}) == 1
    assert pack_devices[0] is not None
    tensors = _load_output(tmp_path)
    assert torch.equal(tensors["mlp.experts.0.weight"], preprocess_weight_for_a8w4_torch(codes["mlp.experts.0"]).view(64, 32))


@pytest.mark.parametrize("bits", [4, 8])
def test_block_modules_use_standard_ct_save(bits):
    model = torch.nn.Module()
    model.dense = torch.nn.Linear(128, 128, bias=False)
    model.dense.quantization_scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=bits, type="int", symmetric=True, strategy="block",
            block_structure=[128, 128], scale_dtype=torch.float32,
        ),
    )
    assert exporter.block_quantized_modules(model) == {"dense": model.dense}
    assert exporter.a8w4_quantized_modules(model) == {}
    assert exporter.int8_quantized_modules(model) == {}
    assert exporter.expected_save_format(model) == "pack-quantized"


@pytest.mark.parametrize("bits", [4, 8])
def test_finalize_reports_block_scheme_metadata(tmp_path, bits):
    model = torch.nn.Module()
    model.dense = torch.nn.Linear(128, 128, bias=False)
    model.dense.quantization_scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=bits, type="int", symmetric=True, strategy="block",
            block_structure=[128, 128], scale_dtype=torch.float32,
        ),
    )
    result = exporter.finalize_gptq_checkpoint(
        model, tmp_path, 100000, "pack-quantized", [],
    )
    assert result == {
        "output_format": f"GPTQ_INT{bits}_BLOCK",
        "scheme": f"INT{bits}_BLOCK",
    }


@pytest.mark.parametrize("bits", [4, 8])
def test_group_modules_use_standard_ct_save(bits):
    model = torch.nn.Module()
    model.dense = torch.nn.Linear(128, 128, bias=False)
    model.dense.quantization_scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=bits, type="int", symmetric=True, strategy="group",
            group_size=128, scale_dtype=torch.float32,
        ),
    )
    assert exporter.group_quantized_modules(model) == {"dense": model.dense}
    assert exporter.block_quantized_modules(model) == {}
    assert exporter.a8w4_quantized_modules(model) == {}
    assert exporter.int8_quantized_modules(model) == {}
    assert exporter.expected_save_format(model) == "pack-quantized"


@pytest.mark.parametrize("bits", [4, 8])
def test_finalize_reports_group_scheme_metadata(tmp_path, bits):
    model = torch.nn.Module()
    model.dense = torch.nn.Linear(128, 128, bias=False)
    model.dense.quantization_scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=bits, type="int", symmetric=True, strategy="group",
            group_size=128, scale_dtype=torch.float32,
        ),
    )
    result = exporter.finalize_gptq_checkpoint(
        model, tmp_path, 100000, "pack-quantized", [],
    )
    assert result == {
        "output_format": f"GPTQ_INT{bits}_GROUP",
        "scheme": f"INT{bits}_GROUP",
    }


@pytest.mark.parametrize("group_size", [32, 64])
@pytest.mark.parametrize("bits", [4, 8])
def test_group_module_detection_requires_group_128(group_size, bits):
    model = torch.nn.Module()
    model.dense = torch.nn.Linear(128, 128, bias=False)
    model.dense.quantization_scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=bits, type="int", symmetric=True, strategy="group",
            group_size=group_size,
        ),
    )
    assert exporter.group_quantized_modules(model) == {}


def test_finalize_rejects_mixed_group_bits(tmp_path):
    model = torch.nn.Module()
    model.int4 = torch.nn.Linear(128, 128, bias=False)
    model.int4.quantization_scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4, type="int", symmetric=True, strategy="group", group_size=128,
        ),
    )
    model.int8 = torch.nn.Linear(128, 128, bias=False)
    model.int8.quantization_scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=8, type="int", symmetric=True, strategy="group", group_size=128,
        ),
    )
    with pytest.raises(ValueError, match="Group INT4 and INT8"):
        exporter.finalize_gptq_checkpoint(model, tmp_path, 100000, "pack-quantized", [])


@pytest.mark.parametrize("kind", ["int4", "int8", "mxfp4"])
def test_finalize_routes_to_exact_export_api_and_forwards_workers(monkeypatch, tmp_path, kind):
    import qlean.core.checkpoint as checkpoint_utils

    model = _model() if kind == "int4" else _int8_model() if kind == "int8" else _mxfp4_model(1)
    checkpoint = {"sentinel": "prepared"}
    patterns = ["sentinel"]
    group = object()
    calls = []
    method = {"int4": "export_a8w4_checkpoint", "int8": "export_int8_checkpoint", "mxfp4": "export_mxfp4_checkpoint"}[kind]

    def observe(*args, **kwargs):
        calls.append((args, kwargs))

    def forbidden(*args, **kwargs):
        pytest.fail("finalization entered another exporter or the shared packed renamer")

    for name in ("export_a8w4_checkpoint", "export_int8_checkpoint", "export_mxfp4_checkpoint"):
        monkeypatch.setattr(exporter, name, observe if name == method else forbidden)
    monkeypatch.setattr(checkpoint_utils, "rename_packed_weights", forbidden)
    expected_format = {"int4": "pack-quantized", "int8": "int-quantized", "mxfp4": "mxfp4-pack-quantized"}[kind]
    result = exporter.finalize_gptq_checkpoint(
        model, tmp_path, 100000, expected_format, patterns,
        checkpoint=checkpoint, process_group=group, workers=3,
    )
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (tmp_path, model, 100000)
    assert kwargs["checkpoint"] is checkpoint and kwargs["expert_patterns"] is patterns
    if kind == "int4":
        assert kwargs["export_device"] == "auto" and kwargs["process_group"] is group
    else:
        assert kwargs["workers"] == 3
        assert "process_group" not in kwargs and "export_device" not in kwargs
    assert result == {
        "output_format": f"GPTQ_{kind.upper()}",
        "scheme": {"int4": "INT4_CHANNEL", "int8": "INT8_CHANNEL", "mxfp4": "MXFP4"}[kind],
    }


def test_export_api_defaults_and_label_only_publication_contract():
    assert inspect.signature(exporter.export_a8w4_checkpoint).parameters["export_device"].default == "auto"
    finalize = inspect.signature(exporter.finalize_gptq_checkpoint).parameters
    assert finalize["export_device"].default == "auto" and finalize["workers"].default == 2
    for function in (exporter.export_int8_checkpoint, exporter.export_mxfp4_checkpoint):
        assert inspect.signature(function).parameters["workers"].default == 2
    writer = inspect.signature(exporter._write_staged_shards).parameters
    assert list(writer)[:8] == ["directory", "staging", "weight_map", "sources", "weights", "scale_keys", "manifest", "filenames"]
    assert writer["export_device"].default == "auto" and writer["workers"].default == 1
    publisher = inspect.signature(exporter._publish_staged_checkpoint).parameters
    assert "label" in publisher and "a8w4" not in publisher
