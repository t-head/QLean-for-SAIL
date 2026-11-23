# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Dequantization provider numerics, broadcasting, dtypes, and input validation."""

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit
_FP4 = (0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6)


def _context(tensors, **cfg):
    reads = []

    def get(name):
        reads.append(name)
        return tensors[name]

    return SimpleNamespace(weight_map=dict.fromkeys(tensors, "scale.safetensors"),
                           transform_cfg=cfg, get_tensor=get, reads=reads)


@pytest.mark.gpu
@pytest.mark.parametrize("block_size", [32, 128])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("target", ["fp32", "bf16"])
def test_fp8_provider_blocks_tail_and_experts(block_size, batched, target):
    import torch
    from qlean.modifiers_engine._runtime.dequant_providers import uniform_fp8_block

    rows, cols = block_size + 1, block_size + 2
    values = (torch.arange(rows * cols, device="cuda") % 8 - 4).reshape(rows, cols).float()
    raw = values.to(torch.float8_e4m3fn)
    scale = torch.tensor([[0.5, 2], [4, 8]], dtype=torch.float32, device="cuda")
    expected = values * scale.repeat_interleave(block_size, 0).repeat_interleave(block_size, 1)[:rows, :cols]
    if batched:
        raw = torch.stack((raw, raw))
        scale = torch.stack((scale, scale * 2))
        expected = torch.stack((expected, expected * 2))
    ctx = _context({"x.weight_scale_inv": scale}, dequant_dtype=target,
                   dequant_block_size=block_size)
    before, scale_before = raw.view(torch.uint8).clone(), scale.clone()
    actual = uniform_fp8_block(ctx, "x.weight", raw)
    dtype = torch.float32 if target == "fp32" else torch.bfloat16
    torch.testing.assert_close(actual, expected.to(dtype), rtol=0, atol=0)
    assert ctx.reads == ["x.weight_scale_inv"]
    torch.testing.assert_close(raw.view(torch.uint8), before, rtol=0, atol=0)
    torch.testing.assert_close(scale, scale_before, rtol=0, atol=0)


@pytest.mark.parametrize("device, global_dtype, global_value", [
    pytest.param("cpu", "float32", 0.5, marks=pytest.mark.cpu),
    pytest.param("cpu", "float16", 3.0, marks=pytest.mark.cpu),
    pytest.param("cuda", "bfloat16", 3.0, marks=pytest.mark.gpu),
])
@pytest.mark.parametrize("group_size", [16, 32])
@pytest.mark.parametrize("target", ["fp32", "bf16"])
@pytest.mark.parametrize("name", ["x.weight", "x.weight_packed"])
@pytest.mark.parametrize("global_name", ["x.weight_scale_2", "x.weight_global_scale"])
def test_nvfp4_local_global_scale_and_tail(
    group_size, target, name, global_name, device, global_dtype, global_value,
):
    import torch
    from qlean.modifiers_engine._runtime.dequant_providers import nvfp4

    payload = [i | ((i + 1) << 4) for i in range(0, 16, 2)] * 2 + [0xF1]
    values = list(_FP4) * 2 + [0.5, -6]
    raw = torch.tensor([payload, payload], dtype=torch.uint8, device=device)
    groups = (34 + group_size - 1) // group_size
    local = torch.tensor([[0.5, 1, 2][:groups], [2, 4, 8][:groups]],
                         dtype=torch.float32, device=device)
    global_scale = torch.tensor(global_value, dtype=getattr(torch, global_dtype), device=device)
    ctx = _context({"x.weight_scale": local, global_name: global_scale},
                   dequant_dtype=target, nvfp4_group_size=group_size)
    multiplier = global_value if global_name.endswith("_scale_2") else 1.0 / global_value
    expected = [[v * float(local[r, c // group_size]) * multiplier for c, v in enumerate(values)]
                for r in range(2)]
    before, local_before, global_before = raw.clone(), local.clone(), global_scale.clone()
    actual = nvfp4(ctx, name, raw)
    dtype = torch.float32 if target == "fp32" else torch.bfloat16
    rtol = 2e-7 if target == "fp32" and global_value == 3.0 else 0
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=dtype, device=device),
                               rtol=rtol, atol=0)
    assert ctx.reads == ["x.weight_scale", global_name]
    torch.testing.assert_close(raw, before, rtol=0, atol=0)
    torch.testing.assert_close(local, local_before, rtol=0, atol=0)
    torch.testing.assert_close(global_scale, global_before, rtol=0, atol=0)


@pytest.mark.cpu
@pytest.mark.parametrize("name", ["x.weight", "x.weight_packed"])
def test_nvfp4_global_scale_legacy_name_takes_priority(name):
    import torch
    from qlean.modifiers_engine._runtime.dequant_providers import nvfp4

    tensors = {"x.weight_scale": torch.full((1, 1), 0.5),
               "x.weight_scale_2": torch.tensor(0.25),
               "x.weight_global_scale": torch.tensor(2.0)}
    ctx = _context(tensors, dequant_dtype="fp32", nvfp4_group_size=16)
    raw = torch.full((1, 8), 0xF1, dtype=torch.uint8)
    actual = nvfp4(ctx, name, raw)
    expected = torch.tensor([[0.0625, -0.75] * 8], dtype=torch.float32)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert ctx.reads == ["x.weight_scale", "x.weight_scale_2"]


@pytest.mark.cpu
@pytest.mark.parametrize("preferred", ["absent", "incomplete", "complete"])
@pytest.mark.parametrize("global_suffixes", [("_scale_2",), ("_global_scale",),
                                             ("_scale_2", "_global_scale")])
def test_nvfp4_selects_first_complete_scale_pair(preferred, global_suffixes):
    import torch
    from qlean.modifiers_engine._runtime.dequant_providers import nvfp4

    tensors = {"x.weight_packed_scale": torch.full((1, 1), 2.0)}
    for suffix in global_suffixes:
        tensors[f"x.weight_packed{suffix}"] = torch.tensor(4.0)
    if preferred != "absent":
        tensors["x.weight_scale"] = torch.full((1, 1), 0.5)
    if preferred == "complete":
        tensors["x.weight_global_scale"] = torch.tensor(2.0)
    ctx = _context(tensors, dequant_dtype="fp32", nvfp4_group_size=16)
    before = {key: value.clone() for key, value in tensors.items()}
    weight_map_before = dict(ctx.weight_map)
    raw = torch.full((1, 8), 0xF1, dtype=torch.uint8)
    raw_before = raw.clone()

    actual = nvfp4(ctx, "x.weight_packed", raw)

    if preferred == "complete":
        factor = 0.25
        expected_reads = ["x.weight_scale", "x.weight_global_scale"]
    else:
        factor = 8.0 if "_scale_2" in global_suffixes else 0.5
        expected_reads = ["x.weight_packed_scale", f"x.weight_packed{global_suffixes[0]}"]
    expected = torch.tensor([[0.5 * factor, -6 * factor] * 8])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert ctx.reads == expected_reads
    assert ctx.weight_map == weight_map_before
    torch.testing.assert_close(raw, raw_before, rtol=0, atol=0)
    for key, value in tensors.items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("kind, scale_name", [
    ("mxfp4", "x.scale"),
    ("mxfp8", "x.scale"),
    ("mxfp8", "x.weight_scale"),
    ("mxfp8", "x.weight_scale_inv"),
])
@pytest.mark.parametrize("storage", ["uint8", "int8"])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("target", ["fp32", "bf16"])
def test_mxfp_provider_row_scales_and_tail(kind, scale_name, storage, batched, target):
    import torch
    from qlean.modifiers_engine._runtime.dequant_providers import PROVIDERS

    if kind == "mxfp4":
        raw = torch.full((2, 17), 0xF1, dtype=torch.uint8, device="cuda").view(getattr(torch, storage))
        values = [0.5, -6] * 17
    else:
        values = [0.5, -4] * 17
        raw = torch.tensor([values] * 2, dtype=torch.float32, device="cuda").to(torch.float8_e4m3fn)
    scale = torch.tensor([[126, 128], [127, 129]], dtype=torch.uint8, device="cuda").view(getattr(torch, storage))
    expected = torch.tensor([[v * (0.5 if c < 32 else 2) for c, v in enumerate(values)],
                             [v * (1 if c < 32 else 4) for c, v in enumerate(values)]],
                            dtype=torch.float32, device="cuda")
    if batched:
        raw, scale, expected = (torch.stack((x, x)) for x in (raw, scale, expected))
    ctx = _context({scale_name: scale}, dequant_dtype=target)
    before, scale_before = raw.view(torch.uint8).clone(), scale.view(torch.uint8).clone()
    actual = PROVIDERS[kind](ctx, "x.weight", raw)
    dtype = torch.float32 if target == "fp32" else torch.bfloat16
    torch.testing.assert_close(actual, expected.to(dtype), rtol=0, atol=0)
    assert ctx.reads == [scale_name]
    torch.testing.assert_close(raw.view(torch.uint8), before, rtol=0, atol=0)
    torch.testing.assert_close(scale.view(torch.uint8), scale_before, rtol=0, atol=0)


@pytest.mark.cpu
@pytest.mark.parametrize("kind", ["fp8_block", "mxfp4", "mxfp8", "nvfp4", "int4"])
def test_float_source_is_returned_without_scale_read(kind):
    import torch
    from qlean.modifiers_engine._runtime.dequant_providers import PROVIDERS

    raw = torch.tensor([[1, -2]], dtype=torch.bfloat16)
    ctx = _context({})
    assert PROVIDERS[kind](ctx, "x.weight", raw) is raw
    assert ctx.reads == []


@pytest.mark.cpu
@pytest.mark.parametrize("kind,dtype_name", [
    ("fp8_block", "uint8"), ("mxfp4", "float8_e4m3fn"),
    ("mxfp8", "int32"), ("nvfp4", "int8"), ("int4", "uint8"),
])
def test_incompatible_quantized_dtype_fails_before_scale_read(kind, dtype_name):
    import torch
    from qlean.modifiers_engine._runtime.dequant_providers import PROVIDERS

    raw = torch.zeros((2, 4), dtype=getattr(torch, dtype_name))
    ctx = _context({})
    with pytest.raises(ValueError, match=f"declares src={kind}") as error:
        PROVIDERS[kind](ctx, "x.weight", raw)
    assert "x.weight" in str(error.value) and str(raw.dtype) in str(error.value)
    assert ctx.reads == []


@pytest.mark.cpu
@pytest.mark.parametrize("kind, dtype_name, shape, scale_shape", [
    ("fp8_block", "float8_e4m3fn", (129, 130), (1, 1)),
    ("fp8_block", "float8_e4m3fn", (2, 128, 128), (1, 1)),
    ("mxfp4", "uint8", (2, 17), (2, 1)),
    ("mxfp8", "float8_e4m3fn", (2, 34), (1, 2)),
    ("nvfp4", "uint8", (2, 17), (2, 1)),
    ("nvfp4", "uint8", (2, 2, 16), (2, 2, 2)),
    ("int4", "int32", (2, 8), (2, 1)),
])
def test_bad_scale_shape_fails_before_decode(monkeypatch, kind, dtype_name, shape, scale_shape):
    import torch
    from qlean.modifiers_engine._runtime import dequant_providers as providers

    def reject(*args, **kwargs):
        pytest.fail("Invalid scales must not reach the decoder")

    decoder = {"fp8_block": "dequantize_fp8_block", "int4": "dequantize_int4"}.get(kind, f"dequantize_{kind}")
    monkeypatch.setattr(providers, decoder, reject)
    raw = torch.zeros(shape, dtype=getattr(torch, dtype_name))
    scale_dtype = torch.uint8 if kind in ("mxfp4", "mxfp8") else torch.float32
    scale = torch.ones(scale_shape, dtype=scale_dtype)
    ctx = _context({"x.weight_scale": scale, "x.weight_scale_2": torch.tensor(1.0)})
    name = "x.weight_packed" if kind == "int4" else "x.weight"
    with pytest.raises(ValueError, match="scale validation"):
        providers.PROVIDERS[kind](ctx, name, raw)


@pytest.mark.cpu
@pytest.mark.parametrize("kind", ["mxfp4", "mxfp8"])
def test_mxfp_rejects_numeric_scale_bytes(kind):
    import torch
    from qlean.modifiers_engine._runtime.dequant_providers import PROVIDERS

    dtype, cols = (torch.uint8, 16) if kind == "mxfp4" else (torch.float8_e4m3fn, 32)
    ctx = _context({"x.scale": torch.ones((2, 1), dtype=torch.float32)})
    with pytest.raises(ValueError, match="scale dtype must be"):
        PROVIDERS[kind](ctx, "x.weight", torch.zeros((2, cols), dtype=dtype))


@pytest.mark.cpu
@pytest.mark.parametrize("name", ["x.weight", "x.weight_packed"])
@pytest.mark.parametrize("keys, message", [
    ([], "no local scale"),
    (["x.weight_packed_scale_2"], "no local scale"),
    (["x.weight_scale"], "missing global scale"),
    (["x.weight_scale", "x.weight_packed_scale"], "missing global scale"),
    (["x.weight_scale", "x.weight_packed_scale_2"], "missing global scale"),
])
def test_nvfp4_missing_scale_pair_fails_before_read(name, keys, message):
    import torch
    from qlean.modifiers_engine._runtime.dequant_providers import nvfp4

    ctx = _context({key: torch.ones((1, 1)) for key in keys})
    with pytest.raises(ValueError, match=message) as error:
        nvfp4(ctx, name, torch.zeros((1, 8), dtype=torch.uint8))
    if message == "missing global scale":
        assert "x.weight_scale_2" in str(error.value)
        assert "x.weight_global_scale" in str(error.value)
    assert ctx.reads == []
