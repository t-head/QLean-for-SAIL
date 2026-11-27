# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures with deferred production imports."""

from copy import deepcopy
from math import prod

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--require-gpu", action="store_true", default=False,
        help="Fail GPU tests if required devices are unavailable",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    layers = [name for name in ("unit", "integration", "e2e")
              if item.get_closest_marker(name)]
    cpu = item.get_closest_marker("cpu") is not None
    gpu = item.get_closest_marker("gpu") is not None
    multi_gpu = item.get_closest_marker("multi_gpu") is not None
    if layers and (len(layers) != 1 or cpu == gpu):
        pytest.fail("Tests must have one layer marker and one cpu/gpu marker")
    if multi_gpu and not gpu:
        pytest.fail("multi_gpu requires the gpu marker")
    if gpu:
        import torch

        required = 2 if multi_gpu else 1
        available = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if available < required:
            reason = f"Requires {required} CUDA GPUs; found {available}"
            if item.config.getoption("--require-gpu"):
                pytest.fail(reason)
            pytest.skip(reason)


@pytest.fixture
def isolated_state():
    """Opt-in isolation that restores singleton, lifecycle, registry, and dtype state."""
    import torch
    from qlean.core.event_lifecycle import EventLifecycle
    from qlean.utils.singleton import SingletonMixin
    from qlean.models import ModelFactory
    from qlean.modifiers import ModifierFactory

    default_dtype = torch.get_default_dtype()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(SingletonMixin, "_instances", {})
        patch.setattr(EventLifecycle, "initialized", False)
        patch.setattr(EventLifecycle, "finalized", False)
        for factory in (ModelFactory, ModifierFactory):
            patch.setattr(factory, "_main_registry", dict(factory._main_registry))
            patch.setattr(factory, "_errors", dict(factory._errors))
            patch.setattr(factory, "_loaded", factory._loaded)
        try:
            yield
        finally:
            torch.set_default_dtype(default_dtype)


@pytest.fixture
def recipe_factory():
    """Deep-copy a recipe or create a minimal single-Modifier recipe."""
    def make(template=None, *, modifier="GeneralQuantEngineModifier", **params):
        if template is not None:
            if params:
                raise ValueError("Modify parameters on the returned template copy")
            return deepcopy(template)
        if modifier == "GeneralQuantEngineModifier":
            defaults = {
                "primary_scheme": "W8A8",
                "config_writer": "compressed_tensors",
                "regions": [
                    {"name": "linear", "match": "*.weight", "dst": "W8A8"},
                    {"name": "fallback", "match": "*"},
                ],
            }
        else:
            defaults = {"scheme": "W8A8", "ignore": []}
        defaults.update(params)
        return deepcopy({"quant_stage": {"quant_modifiers": {modifier: defaults}}})

    return make


@pytest.fixture
def assert_tensor_equal():
    """Compare tensors exactly with quantization context and mismatch locations."""
    import torch

    def check(actual, expected, *, scheme, seed):
        context = f"scheme={scheme}, shape={tuple(actual.shape)}, dtype={actual.dtype}, seed={seed}"
        torch.testing.assert_close(actual, expected, rtol=0, atol=0,
                                   msg=lambda detail: f"{context}\n{detail}")

    return check


@pytest.fixture
def assert_mxfp4_payload_reconstructs():
    """Decode the GPTQ MXFP4 producer contract and bound its reconstruction error.

    MXFP4 export writes quant_method="mxfp4" with no config_groups/format, so the
    standard HF loader cannot attach a scheme and would read the packed bytes as a
    dense weight. Verify the payload with the project's own decoder instead. A
    degenerate E8M0 scale (every byte 127) reconstructs orders of magnitude off:
    the tiny-llama GPTQ fixture measures ~0.14 relative error, a degenerate scale ~113.
    """
    def check(tensors, original_state, dense_keys, *, max_rel_error=0.25):
        import torch
        from qlean.utils.bound_check import dequantize_mxfp4

        errors = {}
        for name in sorted(dense_keys):
            reference = original_state[name].float()
            rebuilt = dequantize_mxfp4(tensors[name], tensors[f"{name}_scale"])
            assert rebuilt.shape == reference.shape, name
            assert torch.isfinite(rebuilt).all(), name
            errors[name] = ((rebuilt - reference).norm()
                            / reference.norm().clamp_min(1e-12)).item()
        # GPTQ compensates accumulated error, so weights legitimately deviate beyond
        # the per-element RTN bound that compute_bound_check enforces; bound the
        # aggregate relative error instead.
        worst = max(errors.values())
        assert worst < max_rel_error, (
            f"MXFP4 relative reconstruction error {worst:.4f} >= {max_rel_error}: {errors}"
        )

    return check


@pytest.fixture
def tensor_factory():
    """Build tensors with local CPU RNG and strict size checks, then move devices."""
    import torch

    def positive_int(value):
        return isinstance(value, int) and not isinstance(value, bool) and value > 0

    def make(shape=(4, 8), *, seed=0, dtype=None, device="cpu",
             group_size=None, block_size=None, experts=None, pattern="normal"):
        shape = tuple(shape)
        if not shape or not all(positive_int(dim) for dim in shape):
            raise ValueError("shape must contain positive integers")
        if experts is not None:
            if not positive_int(experts) or len(shape) != 2:
                raise ValueError("experts must be a positive integer and the base shape must be 2D")
            shape = (experts, *shape)
        if group_size is not None:
            if (len(shape) < 2 or not isinstance(group_size, int)
                    or isinstance(group_size, bool)
                    or (group_size != -1 and group_size <= 0)):
                raise ValueError("group_size must be -1 or a positive integer; tensor must be at least 2D")
            if group_size != -1 and shape[-1] % group_size:
                raise ValueError("group_size must divide the last dimension")
        if block_size is not None:
            if isinstance(block_size, int):
                block = (block_size, block_size)
            elif isinstance(block_size, (tuple, list)):
                block = tuple(block_size)
            else:
                raise ValueError("block_size must be one or two positive integers")
            if len(shape) < 2 or len(block) != 2 or not all(positive_int(d) for d in block):
                raise ValueError("block_size must be one or two positive integers; tensor must be at least 2D")
            if any(dim % size for dim, size in zip(shape[-2:], block)):
                raise ValueError("block_size must divide the last two dimensions")
        if pattern == "normal":
            generator = torch.Generator(device="cpu").manual_seed(seed)
            tensor = torch.randn(shape, generator=generator, dtype=torch.float32, device="cpu")
        elif pattern == "arange":
            tensor = torch.arange(prod(shape), dtype=torch.float32, device="cpu").reshape(shape)
        elif pattern == "zeros":
            tensor = torch.zeros(shape, dtype=torch.float32, device="cpu")
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
        return tensor.to(device=device, dtype=dtype if dtype is not None else torch.float32)

    return make
