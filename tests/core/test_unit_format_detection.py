# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Input format detection with small configs and shards, without conversion."""

import json

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.cpu]


@pytest.mark.parametrize("config, expected", [
    ({"quantization_config": {"quant_method": "FP8"}}, "FP8"),
    ({"quant_algo": "NVFP4", "quant_method": "fp8"}, "NVFP4"),
    ({"quantization": {"quant_algo": "nvfp4"}}, "NVFP4"),
    ({"quantization": "fp8"}, "FP8"),
    ({"gguf": {"quantization_type": "fp8"}}, "FP8"),
    ({"text_config": {"quantization_config": {"quant_method": "fp8"}}}, "FP8"),
    ({"quant_method": "compressed-tensors"}, "COMPRESSED-TENSORS"),
    ({"quant_method": "gptq"}, "GPTQ"),
    ({"quant_method": "awq"}, "AWQ"),
    ({"quant_method": "bitsandbytes"}, "BITSANDBYTES"),
])
def test_config_format_precedes_index_and_shards(tmp_path, config, expected):
    from qlean.core.utils import detect_input_format

    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text("not-json", encoding="utf-8")
    (tmp_path / "a.safetensors").write_bytes(b"must-not-be-opened")
    assert detect_input_format(str(tmp_path)) == expected


@pytest.mark.parametrize("main_config, expected", [
    ({}, "NVFP4"),
    ({"quant_method": "fp8"}, "FP8"),
])
def test_main_config_precedes_alternative_config(tmp_path, main_config, expected):
    from qlean.core.utils import detect_input_format

    (tmp_path / "config.json").write_text(json.dumps(main_config), encoding="utf-8")
    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps({"quantization": {"quant_algo": "NVFP4"}}), encoding="utf-8",
    )
    assert detect_input_format(str(tmp_path)) == expected


@pytest.mark.parametrize("keys, expected", [
    (["x.weight.scale_inv"], "FP8"),
    (["x.weight_scale", "x.weight_scale_2", "x.input_scale"], "NVFP4"),
    (["x.weight.scale_inv", "x.weight_scale", "x.weight_scale_2", "x.input_scale"], "NVFP4"),
    (["x.weight_scale", "x.weight_global_scale", "x.input_scale"], "NVFP4"),
    (["x.weight.scale_inv", "x.weight_scale", "x.weight_global_scale", "x.input_scale"], "NVFP4"),
])
def test_index_format_precedes_shard_scan(tmp_path, keys, expected):
    from qlean.core.utils import detect_input_format

    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(keys, "a.safetensors")}), encoding="utf-8",
    )
    (tmp_path / "a.safetensors").write_bytes(b"must-not-be-opened")
    assert detect_input_format(str(tmp_path)) == expected


@pytest.mark.parametrize("keys, expected", [
    (["x.weight.scale_inv"], "FP8"),
    (["x.weight_scale", "x.weight_scale_2", "x.input_scale"], "NVFP4"),
    (["x.weight_scale", "x.weight_scale_2"], "BF16"),
    (["x.weight_scale", "x.weight_global_scale", "x.input_scale"], "NVFP4"),
    (["x.weight_scale", "x.weight_global_scale"], "BF16"),
    (["x.weight_global_scale", "x.input_scale"], "BF16"),
    (["x.weight_global_scale"], "BF16"),
    (["x.weight_scale", "x.input_global_scale", "x.input_scale"], "BF16"),
    (["x.weight.scale_inv", "x.weight_global_scale"], "FP8"),
    (["x.weight"], "BF16"),
])
def test_scan_later_shard_and_bf16_fallback(tmp_path, tensor_factory, keys, expected):
    from safetensors.torch import save_file
    from qlean.core.utils import detect_input_format

    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a.weight": "a.safetensors"}}), encoding="utf-8",
    )
    save_file({"a.weight": tensor_factory((1, 1))}, str(tmp_path / "a.safetensors"))
    save_file({key: tensor_factory((1, 1)) for key in keys}, str(tmp_path / "b.safetensors"))
    assert detect_input_format(str(tmp_path)) == expected


@pytest.mark.parametrize("global_key, existing_scale", [
    ("x.weight_scale_2", None),
    ("x.weight_global_scale", None),
    ("x.weight_packed_global_scale", None),
    ("x.weight_scale_2", "x.weight_scale"),
    ("x.weight_scale_2", "x.weight_global_scale"),
])
def test_scale_scan_fills_weight_map_only_when_scales_missing(
    tmp_path, tensor_factory, monkeypatch, global_key, existing_scale,
):
    from safetensors.torch import save_file
    from qlean.modifiers import utils

    save_file({"x.weight_packed": tensor_factory((1, 1))}, str(tmp_path / "a.safetensors"))
    scale_keys = ["x.weight_scale", global_key, "x.input_scale", "x.input_global_scale"]
    tensors = {key: tensor_factory((1, 1)) for key in [*scale_keys, "x.bias"]}
    save_file(tensors, str(tmp_path / "b.safetensors"))

    weight_map = {"x.weight_packed": "a.safetensors"}
    if existing_scale is None:
        expected = {**weight_map, **dict.fromkeys(scale_keys, "b.safetensors")}
    else:
        weight_map[existing_scale] = "b.safetensors"
        expected = dict(weight_map)

        def unexpected_scan(*args, **kwargs):
            pytest.fail("must not rescan an index that already has scales")

        monkeypatch.setattr(utils, "glob", unexpected_scan)

    utils.ensure_scales_in_weight_map(weight_map, str(tmp_path))
    assert weight_map == expected


def test_empty_directory_defaults_to_bf16(tmp_path):
    from qlean.core.utils import detect_input_format

    assert detect_input_format(str(tmp_path)) == "BF16"
