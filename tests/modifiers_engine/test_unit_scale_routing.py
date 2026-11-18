# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Source scale priorities, naming, and missing-scale checks."""

from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.cpu]


@pytest.mark.parametrize("offset", range(5))
def test_fp8_scale_candidates_follow_priority(offset):
    from qlean.modifiers_engine._runtime.scale_utils import find_scale_name

    candidates = ["x.weight_scale_inv", "x.weight_scale", "x.weight.weight_scale_inv", "x.scale"]
    mapping = dict.fromkeys(candidates[offset:], "scales.safetensors")
    expected = candidates[offset] if offset < len(candidates) else None
    assert find_scale_name("x.weight", mapping, kind="fp8") == expected


@pytest.mark.parametrize("kind, name, candidates", [
    ("mxfp4", "x.weight", ["x.scale", "x.weight_scale"]),
    ("mxfp4", "x.weight_packed", ["x.weight_scale"]),
    ("mxfp8", "x.weight", ["x.scale", "x.weight_scale", "x.weight_scale_inv"]),
    ("nvfp4", "x.weight", ["x.weight_scale"]),
    ("nvfp4", "x.weight_packed", ["x.weight_scale", "x.weight_packed_scale"]),
    ("int4", "x.weight_packed", ["x.weight_scale"]),
])
def test_source_scale_naming_and_missing_candidates(kind, name, candidates):
    from qlean.modifiers_engine._runtime.scale_utils import find_scale_name

    for index, expected in enumerate(candidates):
        mapping = dict.fromkeys(candidates[index:], "other-shard.safetensors")
        assert find_scale_name(name, mapping, kind=kind) == expected
    assert find_scale_name(name, {"unrelated.scale": "a.safetensors"}, kind=kind) is None


@pytest.mark.parametrize("sources, scale_keys", [
    (("mxfp8",), ["x.scale", "x.weight_scale", "x.weight_scale_inv"]),
    (("nvfp4",), ["x.weight_scale", "x.weight_scale_2", "x.weight_global_scale",
                   "x.weight_packed_scale", "x.weight_packed_scale_2",
                   "x.weight_packed_global_scale"]),
    (("mxfp8", "nvfp4"), ["x.scale", "x.weight_scale_inv", "x.weight_scale",
                             "x.weight_scale_2", "x.weight_global_scale"]),
])
def test_source_scale_aliases_are_filtered_before_routing(sources, scale_keys):
    from qlean.modifiers_engine._runtime.region_router import (
        derive_filter_source_key, validate_regions,
    )

    regions = [{"name": src, "match": f"{src}.*", "src": src, "dst": "W8A8"}
               for src in sources]
    regions.append({"name": "fallback", "match": "*"})
    validate_regions(regions)
    keep = derive_filter_source_key(regions)
    assert keep is not None
    for key in scale_keys:
        assert not keep(key), key
    for key in ("x.weight", "x.weight_packed", "x.bias", "x.norm.weight"):
        assert keep(key), key


def test_unspecified_kind_warns_on_ambiguity_and_uses_first(caplog):
    from qlean.modifiers_engine._runtime.scale_utils import find_scale_name

    mapping = {"x.weight_scale_inv": "a.safetensors", "x.scale": "b.safetensors"}
    assert find_scale_name("x.weight", mapping) == "x.weight_scale_inv"
    assert "matched multiple rules" in caplog.text
    assert "x.weight" in caplog.text
    caplog.clear()
    assert find_scale_name("x.weight", mapping, kind="mxfp8") == "x.scale"
    assert caplog.text == ""


def test_unknown_scale_kind_is_rejected():
    from qlean.modifiers_engine._runtime.scale_utils import find_scale_name

    with pytest.raises(NotImplementedError, match="Unknown scale kind 'unknown'"):
        find_scale_name("x.weight", {}, kind="unknown")


@pytest.mark.parametrize("src, packed_cols", [("int4", 16), ("mxfp4", 64), ("nvfp4", 64)])
@pytest.mark.parametrize("scheme", ["W8A8", "FP8_BLOCK", "FP8_DYNAMIC"])
@pytest.mark.parametrize("scale_naming, expected_scale", [
    ("underscore", "x.weight_scale"), ("dot", "x.scale"),
])
def test_packed_source_name_is_normalized_in_plan(scale_naming, expected_scale, scheme, src, packed_cols):
    from qlean.modifiers_engine._runtime import plan, region_router

    cfg = {"scale_naming": scale_naming, "_regions": [
        {"name": "packed", "match": "*.weight_packed", "src": src, "dst": scheme},
        {"name": "fallback", "match": "*"},
    ]}
    region_router.validate_regions(cfg["_regions"])
    assert region_router.derive_dequant_map(cfg["_regions"]) == {"*.weight_packed": src, "*": None}
    result = region_router.spec_plan_fn("x.weight_packed", [128, packed_cols], cfg,
                                      {"x.weight_scale": "scales.safetensors"})
    assert result.meta["src"] == src
    assert result.meta["dst"] == scheme
    assert result.ops == [("quant", "x.weight", scheme, expected_scale if scale_naming == "dot" else None)]
    assert plan.op_output_names(result.ops[0]) == ["x.weight", expected_scale]


@pytest.mark.parametrize("first_dst,second_src,second_dst,conflict", [
    ("W8A8", "mxfp4", "FP8_DYNAMIC", False),
    ("W8A8", "mxfp8", "W8A8", True),
    ("W8A8", "mxfp4", "passthrough", True),
    ("passthrough", "mxfp4", "W8A8", True),
])
def test_duplicate_pattern_dequant_mapping(first_dst, second_src, second_dst, conflict):
    from qlean.modifiers_engine._runtime import region_router

    regions = [
        {"name": "first", "match": "*.weight", "src": "mxfp4", "dst": first_dst},
        {"name": "second", "match": ["x.weight", "*.weight"], "src": second_src, "dst": second_dst},
        {"name": "fallback", "match": "*"},
    ]
    region_router.validate_regions(regions)
    if conflict:
        with pytest.raises(ValueError, match="conflicting dequantization providers") as error:
            region_router.derive_dequant_map(regions)
        assert "*.weight" in str(error.value)
    else:
        assert region_router.derive_dequant_map(regions) == {
            "*.weight": "mxfp4", "x.weight": "mxfp4", "*": None,
        }


@pytest.mark.parametrize("second_chunk_rows", [2, 3])
def test_duplicate_pattern_stream_mapping(second_chunk_rows):
    from qlean.modifiers_engine._runtime import region_router

    regions = [
        {"name": "first", "match": "*.weight", "src": "mxfp8", "dst": "W8A8",
         "chunked": True, "chunk_rows": 2},
        {"name": "second", "match": ["x.weight", "*.weight"], "src": "mxfp8", "dst": "FP8_DYNAMIC",
         "chunked": True, "chunk_rows": second_chunk_rows},
        {"name": "fallback", "match": "*"},
    ]
    region_router.validate_regions(regions)
    if second_chunk_rows != 2:
        with pytest.raises(ValueError, match="conflicting chunk_rows") as error:
            region_router.derive_stream_map(regions)
        assert "*.weight" in str(error.value)
    else:
        assert region_router.derive_stream_map(regions) == {"*.weight": 2, "x.weight": 2}


@pytest.mark.parametrize("src", ["fp8_block", "mxfp4", "mxfp8"])
def test_passthrough_requires_source_scale_in_plan(src):
    from qlean.modifiers_engine._runtime.region_router import spec_plan_fn

    cfg = {"_regions": [
        {"name": "keep", "match": "*.weight", "src": src, "dst": "passthrough"},
        {"name": "fallback", "match": "*"},
    ]}
    with pytest.raises(ValueError, match="x.weight: source scale missing"):
        spec_plan_fn("x.weight", [2, 4], cfg, {"x.weight": "a.safetensors"})


@pytest.mark.parametrize("provider, name, dtype", [
    ("uniform_fp8_block", "x.weight", "float8_e4m3fn"),
    ("mxfp4", "x.weight_packed", "uint8"),
    ("mxfp8", "x.weight", "float8_e4m3fn"),
    ("nvfp4", "x.weight", "uint8"),
    ("int4_groupwise", "x.weight_packed", "int32"),
])
def test_quantized_source_without_scale_fails_before_dequantization(
    tensor_factory, provider, name, dtype,
):
    import torch
    from qlean.modifiers_engine._runtime import dequant_providers

    def unexpected_read(key):
        pytest.fail(f"Missing scales must prevent tensor reads: {key}")

    ctx = SimpleNamespace(weight_map={}, transform_cfg={}, get_tensor=unexpected_read)
    raw = tensor_factory((2, 4), pattern="zeros", dtype=getattr(torch, dtype))
    with pytest.raises(ValueError, match="no .*scale") as error:
        getattr(dequant_providers, provider)(ctx, name, raw)
    assert name in str(error.value)
