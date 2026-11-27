# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys

import pytest
import torch

import qlean.modifiers.save_utils as save_utils
from qlean.utils.pack_acext_int4 import preprocess_weight_for_a8w4_torch


pytestmark = [pytest.mark.unit, pytest.mark.cpu]


@pytest.mark.parametrize("shape", [(64, 64), (64, 128), (128, 64), (128, 192)])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_wrapper_matches_existing_pure_torch_backend_without_native(
    monkeypatch, shape, noncontiguous
):
    monkeypatch.setitem(sys.modules, "acext", None)
    n, k = shape
    values = torch.arange(n * k).reshape(n, k) + 3 * torch.arange(n).reshape(n, 1)
    weight = (values % 16 - 8).to(torch.int8)
    if noncontiguous:
        weight = weight.t().contiguous().t()
        assert not weight.is_contiguous()
    original = weight.clone()
    assert {weight.min().item(), weight.max().item()} == {-8, 7}
    assert (weight == 0).any()
    expected = preprocess_weight_for_a8w4_torch(weight).view(n, k // 2)

    actual = save_utils.preprocess_for_acext_a8w4(weight)

    assert actual.shape == (n, k // 2)
    assert actual.dtype == torch.int8
    assert actual.device == weight.device
    assert actual.is_contiguous()
    assert torch.equal(actual, expected)
    assert torch.equal(weight, original)


@pytest.mark.parametrize("value", [-8, 0, 7])
@pytest.mark.parametrize("shape", [(64, 128), (128, 64)])
def test_boundary_code_bytes_match_existing_helper(shape, value):
    n, k = shape
    weight = torch.full(shape, value, dtype=torch.int8)
    actual = save_utils.preprocess_for_acext_a8w4(weight)
    expected = preprocess_weight_for_a8w4_torch(weight).view(n, k // 2)
    assert torch.equal(actual, expected)
    # A8W4 has no unsigned bias: both nibbles keep the signed INT4 code.
    nibble = value & 0xF
    assert torch.all(actual.view(torch.uint8) == (nibble | (nibble << 4)))


@pytest.mark.parametrize("shape", [(64, 64), (64, 128), (128, 64), torch.Size([128, 192])])
def test_shape_validator_accepts_nonempty_aligned_2d_shape(shape):
    save_utils.validate_acext_a8w4_shape(shape)


@pytest.mark.parametrize(
    "shape,match",
    [
        ((), "2D"), ((64,), "2D"), ((1, 64, 64), "2D"),
        ((0, 64), "nonempty"), ((64, 0), "nonempty"), ((0, 0), "nonempty"),
        ((32, 64), "64"), ((64, 32), "64"), ((96, 128), "64"),
        ((128, 96), "64"), ((63, 64), "64"), ((64, 65), "64"),
    ],
)
def test_invalid_shapes_rejected_by_validator_and_wrapper(shape, match):
    with pytest.raises(ValueError, match=match):
        save_utils.validate_acext_a8w4_shape(shape)
    with pytest.raises(ValueError, match=match):
        save_utils.preprocess_for_acext_a8w4(torch.zeros(shape, dtype=torch.int8))


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int16, torch.int32, torch.float32, torch.bfloat16])
def test_wrapper_requires_signed_int8(dtype):
    with pytest.raises(TypeError, match="torch.int8"):
        save_utils.preprocess_for_acext_a8w4(torch.zeros(64, 64, dtype=dtype))


def test_wrapper_rejects_non_tensor():
    with pytest.raises(TypeError, match="torch.int8"):
        save_utils.preprocess_for_acext_a8w4(None)


@pytest.mark.parametrize("value", [-9, 8])
def test_wrapper_retains_backend_int4_range_validation(value):
    weight = torch.zeros(64, 64, dtype=torch.int8)
    weight[0, 0] = value
    with pytest.raises(ValueError, match="int4 range"):
        save_utils.preprocess_for_acext_a8w4(weight)


def test_module_import_does_not_require_native(monkeypatch):
    monkeypatch.setitem(sys.modules, "acext", None)
    spec = importlib.util.spec_from_file_location("target_a8w4_save_utils", save_utils.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    weight = torch.zeros(64, 128, dtype=torch.int8)
    assert torch.equal(
        module.preprocess_for_acext_a8w4(weight),
        preprocess_weight_for_a8w4_torch(weight).view(64, 64),
    )
