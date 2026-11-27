# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GPTQ-only forward adaptation and failure resource ownership."""

import contextlib
import functools
from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.quantization import QuantizationStatus

from qlean.core.state import State
from qlean.modifiers.gptq import GPTQModifier
from qlean.modifiers.quantization import QuantizationCalibrationMixin
import qlean.modifiers.gptq.base as gptq_base
import qlean.pipelines.sequential.pipeline as pipeline


pytestmark = [pytest.mark.gptq, pytest.mark.unit, pytest.mark.cpu]


def modifier(**kwargs):
    return GPTQModifier(config_groups={"weights": {
        "targets": ["re:.*"], "weights": {
            "num_bits": 8, "type": "int", "strategy": "channel", "symmetric": True,
        },
    }}, **kwargs)


def partial_model(offloaded=False):
    model = torch.nn.Module()
    model.target = torch.nn.Linear(64, 64)
    model.ignored = torch.nn.Linear(64, 64)
    model.other = torch.nn.ReLU()
    model.container = torch.nn.Sequential(torch.nn.ReLU())
    if offloaded:
        from compressed_tensors.offload import offload_module
        offload_module(model.target, onload_device="cpu", offload_device="cpu")
    for module in model.modules():
        module.forward = functools.partial(module.forward)
    return model


def int4_moe_model():
    model = torch.nn.Module()
    model.mlp = torch.nn.Module()
    model.mlp.experts = torch.nn.ModuleList([
        torch.nn.Linear(64, 64, bias=False),
    ])
    model.dense = torch.nn.Linear(64, 64, bias=False)
    return model


def test_int4_dense_selection_is_rejected_before_common_initialization(monkeypatch):
    model = int4_moe_model()
    modifier = GPTQModifier(scheme="INT4_CHANNEL", targets=["Linear"])
    monkeypatch.setattr(
        QuantizationCalibrationMixin,
        "initialize_quantization",
        lambda *args: pytest.fail("common initialization must not run"),
    )

    with pytest.raises(ValueError, match="only routed MoE expert") as raised:
        modifier.initialize_quantization(model)

    message = str(raised.value)
    assert "1 dense or non-expert" in message
    assert "'dense'" in message
    assert "targets=[GPTQ_EXPERT_TARGET]" in message
    assert "GPTQModifier.ignore" in message
    assert not hasattr(model.dense, "quantization_scheme")
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())
    assert not modifier._hessians and not modifier._num_samples


def test_int4_without_routed_experts_is_rejected_before_common_initialization(monkeypatch):
    model = partial_model()
    monkeypatch.setattr(
        QuantizationCalibrationMixin,
        "initialize_quantization",
        lambda *args: pytest.fail("common initialization must not run"),
    )

    with pytest.raises(ValueError, match="at least one routed MoE expert"):
        GPTQModifier(scheme="INT4_CHANNEL").initialize_quantization(model)

    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())


@pytest.mark.parametrize("kwargs", [
    {"scheme": "INT4_CHANNEL"},
    {"scheme": "INT4_CHANNEL", "targets": ["Linear"], "ignore": ["dense"]},
    {"scheme": "INT4_CHANNEL", "targets": [r"re:.*mlp\.experts\..*"]},
])
def test_int4_expert_only_selection_reaches_common_initialization(monkeypatch, kwargs):
    model = int4_moe_model()
    reached = []
    monkeypatch.setattr(
        QuantizationCalibrationMixin,
        "initialize_quantization",
        lambda self, selected_model: reached.append(selected_model),
    )

    GPTQModifier(**kwargs).initialize_quantization(model)

    assert reached == [model]


@pytest.mark.parametrize("offloaded", [False, True])
def test_supported_partial_targets_only_and_global_callable_identity(offloaded):
    import compressed_tensors.quantization.lifecycle.forward as ct_forward
    import compressed_tensors.quantization.lifecycle.initialize as ct_initialize
    original, cached = ct_forward.set_forward_quantized, ct_initialize.set_forward_quantized
    model = partial_model(offloaded)
    forwards = {name: module.forward for name, module in model.named_modules()}
    mod = modifier(ignore=["ignored"])
    mod.initialize_quantization(model)
    assert ct_forward.set_forward_quantized is original
    assert ct_initialize.set_forward_quantized is cached
    for name, forward in forwards.items():
        module = model.get_submodule(name)
        if name != "target":
            assert module.forward is forward
            assert not hasattr(module, "quantization_scheme")
    if offloaded:
        assert model.target.forward is forwards["target"]
        assert model.target._original_forward_func is not torch.nn.Linear.forward
        assert torch.isfinite(model.target(torch.randn(2, 64))).all()
        from compressed_tensors.offload import unwrap_offload_forward
        with unwrap_offload_forward(model.target):
            assert torch.isfinite(model.target(torch.randn(2, 64))).all()
    else:
        assert hasattr(model.target.forward, "__func__")
        assert torch.isfinite(model.target(torch.randn(2, 64))).all()
    mod._cleanup_calibration(model)


@pytest.mark.parametrize("offloaded", [False, True])
@pytest.mark.parametrize("failure_stage", ["apply", "observer", "config"])
def test_initialization_failure_restores_forward_identity(monkeypatch, offloaded, failure_stage):
    import compressed_tensors.quantization.lifecycle.initialize as ct_initialize
    model = partial_model(offloaded)
    forwards = {module: module.forward for module in model.modules()}
    mod = modifier(ignore=["ignored"])
    original = ct_initialize.set_forward_quantized
    failure = RuntimeError("original initialization error")

    def fail(*args, **kwargs):
        raise failure

    if failure_stage == "config":
        monkeypatch.setattr(GPTQModifier, "resolve_quantization_config", fail)
    elif failure_stage == "observer":
        monkeypatch.setattr(QuantizationCalibrationMixin, "_initialize_observers", fail)
    else:
        # Fail *inside* CT unwrap, whose exception path does not restore the
        # outer forward. QLean must restore its snapshot, not patch CT globally.
        monkeypatch.setattr(ct_initialize, "set_forward_quantized", fail)
    expected_global = ct_initialize.set_forward_quantized
    with pytest.raises(RuntimeError) as caught:
        mod.initialize_quantization(model)
    assert caught.value is failure
    assert ct_initialize.set_forward_quantized is expected_global
    for module, forward in forwards.items():
        assert module.forward is forward
    if offloaded:
        assert model.target._original_forward_func is torch.nn.Linear.forward
    mod._cleanup_calibration(model)
    assert original is not None


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64)
        self.failure = None

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        if self.failure is not None:
            raise self.failure
        return hidden_states


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([Decoder()])
        self.failure = None

    def forward(self, input_ids, use_cache=False):
        if self.failure is not None:
            raise self.failure
        return self.model.layers[0](input_ids)


@pytest.mark.parametrize("stage", [
    "initialize", "discovery", "capture", "start", "round", "forward", "compression",
])
def test_pipeline_failure_reuses_teardown_without_success_events(monkeypatch, stage):
    model, mod = TinyModel(), modifier()
    state = State(model=SimpleNamespace(model_=model, path="memory"))
    failure = RuntimeError("original " + stage + " failure")
    exited, summaries, closed = [], [], []

    @contextlib.contextmanager
    def moe_context():
        try:
            yield
        finally:
            exited.append(True)

    class Diagnostics:
        def __init__(self, *args):
            pass
        def start(self, register):
            register(model.model.layers[0], lambda *args: None, "forward_pre")
        def observe_input(self, *args):
            pass
        def log_summary(self):
            summaries.append(True)
        def close(self):
            closed.append(True)

    def fail(*args, **kwargs):
        raise failure

    def initialize():
        mod.on_initialize(state)
        mod._is_moe_model = True

    monkeypatch.setattr(gptq_base, "moe_calibration_context", moe_context)
    monkeypatch.setattr(gptq_base, "CalibrationDiagnostics", Diagnostics)
    monkeypatch.setattr(gptq_base, "diagnostics_enabled", lambda: True)
    monkeypatch.setattr(pipeline, "diagnostics_enabled", lambda: False)
    monkeypatch.setattr(pipeline, "set_onload_device", lambda *args: None)
    monkeypatch.setattr(pipeline, "get_main_device", lambda: torch.device("cpu"))
    if stage == "initialize":
        original = QuantizationCalibrationMixin._initialize_observers
        def observer_fail(self, module):
            original(self, module)
            if module is model.model.layers[0].linear:
                self.register_hook(module, lambda *args: None, "forward_pre")
                raise failure
        monkeypatch.setattr(QuantizationCalibrationMixin, "_initialize_observers", observer_fail)
    elif stage == "discovery":
        monkeypatch.setattr(pipeline, "_find_decoder_layers", fail)
    elif stage == "capture":
        model.failure = failure
    elif stage == "start":
        original = QuantizationCalibrationMixin.start_calibration
        def start_fail(self, model):
            original(self, model)
            raise failure
        monkeypatch.setattr(QuantizationCalibrationMixin, "start_calibration", start_fail)
    elif stage == "round":
        original = GPTQModifier._initialize_hessian
        def round_fail(self, module):
            original(self, module)
            raise failure
        monkeypatch.setattr(GPTQModifier, "_initialize_hessian", round_fail)
    elif stage == "forward":
        model.model.layers[0].failure = failure
    elif stage == "compression":
        monkeypatch.setattr(GPTQModifier, "compress_modules", fail)

    compressor = SimpleNamespace(
        modifiers=[mod], initialize=initialize,
        calibration_epoch_start=lambda **kw: mod.on_start(state, **kw),
        sequential_epoch_start=lambda **kw: mod.on_sequential_epoch_start(state, None, **kw),
        sequential_epoch_end=lambda **kw: mod.on_sequential_epoch_end(state, None, **kw),
        calibration_epoch_end=lambda: pytest.fail("Failure dispatched normal freeze"),
        finalize=lambda: pytest.fail("Failure dispatched normal finalize"),
    )
    monkeypatch.setattr(pipeline, "get_compressor", lambda: compressor)
    with pytest.raises(RuntimeError) as caught:
        pipeline.run_pipeline(model, [torch.randn(2, 64)], SimpleNamespace())
    assert caught.value is failure
    assert not mod._hooks and not mod._hessians and not mod._num_samples
    assert not mod._active_modules and not mod._active_module_set
    assert not mod._module_names and not mod._hook_call_counts
    assert not mod._skipped_modules and not mod._nonfinite_reported
    assert mod._diagnostics is None and mod._moe_calibration_ctx is None
    assert not summaries
    assert exited == ([] if stage == "initialize" else [True])
    assert closed == ([] if stage in ("initialize", "start") else [True])
    for module in model.modules():
        assert not module._forward_pre_hooks and not module._forward_hooks
        assert not hasattr(module, "weight_observer")
        assert getattr(module, "quantization_status", None) != QuantizationStatus.FROZEN
        assert not getattr(module, "quantization_enabled", False)
    # Both the failed model and later ordinary models remain callable without
    # calibration hooks or incomplete fake-quant qparams being executed.
    model.failure = model.model.layers[0].failure = None
    assert torch.isfinite(model(torch.randn(2, 64))).all()
    assert torch.isfinite(torch.nn.Linear(64, 64)(torch.randn(2, 64))).all()


def test_normal_finalize_retains_incomplete_quantization_error():
    mod = modifier()
    state = State(model=SimpleNamespace(model_=torch.nn.Linear(64, 64)))
    mod.ended_ = True
    mod._num_samples[state.model.model_] = torch.tensor(1)
    mod._hessians[state.model.model_] = torch.eye(64)
    with pytest.raises(ValueError, match="Failed to compress 1 modules"):
        mod.on_finalize(state)
    assert not mod._hessians and not mod._num_samples


def test_deepseek_restore_is_gptq_only_and_precedes_quantization(monkeypatch):
    import sys
    from types import ModuleType
    from qlean.core.utils import preprocess_model

    model = TinyModel()
    model.config = SimpleNamespace(model_type="deepseek_v32")
    wrapper = SimpleNamespace(model_=model)
    calls = []
    module = ModuleType("qlean.modeling.deepseekv32")
    module.DeepseekV32ForCausalLM = TinyModel
    module.restore_deepseek_v32_freqs_cis = lambda model: calls.append("restore")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert preprocess_model(wrapper) is wrapper
    assert not calls
    monkeypatch.setattr(gptq_base, "linearize_moe_layers", lambda model: calls.append("linearize"))
    monkeypatch.setattr(GPTQModifier, "initialize_quantization", lambda *args: calls.append("quantize"))
    # No config avoids export-specific validation; restoration is independent.
    mod = GPTQModifier()
    mod.on_initialize(State(model=wrapper))
    assert calls == ["restore", "linearize"]
    monkeypatch.setattr(QuantizationCalibrationMixin, "has_config", lambda *args: True)
    mod = GPTQModifier()
    # Empty resolved groups isolate ordering from export capability checks.
    mod._resolved_config = SimpleNamespace(config_groups={})
    calls.clear()
    mod.on_initialize(State(model=wrapper))
    assert calls == ["restore", "linearize", "quantize"]
