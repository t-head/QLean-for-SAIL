# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Keep modifier-owned datafree saves and legacy events isolated from GPTQ."""

import builtins
import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

import qlean.core.event_mixin as event_module
import qlean.core.qleancompressor as compressor_module
from qlean.args import ModelArguments
from qlean.core.event import EventType
from qlean.core.event_lifecycle import EventLifecycle
from qlean.core.event_mixin import EventMixin
from qlean.core.qleancompressor import QLeanCompressor
from qlean.core.state import State
from qlean.modifiers import Modifier, ModifierFactory
from qlean.models.day0.day0 import day0Model
from qlean.models.pretrained.pretrained import PretrainedModel
from qlean.pipelines.registry import get_pipeline_from_modifiers, is_gptq_modifier


EVENTS = (
    EventType.CALIBRATION_EPOCH_START,
    EventType.SEQUENTIAL_EPOCH_START,
    EventType.SEQUENTIAL_EPOCH_END,
    EventType.CALIBRATION_EPOCH_END,
)


pytestmark = [pytest.mark.gptq, pytest.mark.unit, pytest.mark.cpu]


def _forbidden(*args, **kwargs):
    pytest.fail("Non-GPTQ orchestration touched a GPTQ-only save/callback path")


class _ForbiddenHF:
    config = property(_forbidden)
    generation_config = property(_forbidden)
    named_modules = property(_forbidden)
    name_or_path = property(_forbidden)
    save_pretrained = _forbidden


class _ForbiddenDataloader:
    def __iter__(self):
        pytest.fail("The original datafree pipeline must not iterate calibration data")


class _LegacyModifier(Modifier):
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def on_initialize(self, state):
        self.initialized_ = True
        self.calls.append((self.name, "initialize"))

    # Deliberately retain the old signature: no kwargs and no new callbacks.
    def on_event(self, state, event):
        self.calls.append((self.name, event.type_))

    update_event = _forbidden
    on_calibration_start = _forbidden
    on_sequential_epoch_start = _forbidden
    on_sequential_epoch_end = _forbidden
    on_calibration_end = _forbidden


class _SavingModifier(_LegacyModifier):
    def on_event(self, state, event):
        super().on_event(state, event)
        if event.type_ == EventType.CALIBRATION_EPOCH_START:
            self.on_start(state)
            self.calls.append((self.name, "start"))
        elif event.type_ == EventType.CALIBRATION_EPOCH_END:
            output = Path(state.save_path)
            save_file({"layer.weight_packed": torch.tensor([[17, -3]], dtype=torch.int32)},
                      output / "model.safetensors")
            (output / "config.json").write_text(json.dumps({
                "modifier_owned": {"sentinel": "not an HF save", "num_mtp_layers": 2},
                "quantization_config": {
                    "scale_dtype": "legacy-scale", "zp_dtype": "legacy-zp",
                    "format": "modifier-owned",
                },
            }))
            (output / "generation_config.json").write_bytes(
                b'{"do_sample": false, "temperature": 0.23, "modifier_owned": true}\n'
            )
            self.saved_bytes = {
                name: (output / name).read_bytes()
                for name in ("model.safetensors", "generation_config.json")
            }
            self.calls.append((self.name, "save"))
            self.on_end(state)
            self.calls.append((self.name, "end"))


def _bare_compressor(wrapper, model_args, modifiers):
    compressor = object.__new__(QLeanCompressor)
    compressor.state = State(model=wrapper, save_path=model_args.save_path)
    compressor._model_args = model_args
    compressor.modifiers = modifiers
    compressor.dataloader = _ForbiddenDataloader()
    compressor._quant_meta = {
        "input_format": "BF16", "output_format": model_args.output_format,
        "recipe_path": "<legacy>", "recipe_content": {"modifier_owned": True},
        "modifier_names": [type(modifier).__name__ for modifier in modifiers],
        "scheme": "legacy",
    }
    return compressor


@pytest.fixture(autouse=True)
def preserve_lifecycle_state():
    initialized, finalized = EventLifecycle.initialized, EventLifecycle.finalized
    yield
    EventLifecycle.initialized, EventLifecycle.finalized = initialized, finalized


def _block_imports(monkeypatch, forbidden):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden):
            pytest.fail("Unexpected GPTQ-only import: " + name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


@pytest.mark.parametrize("wrapper_kind", ["day0", "hf"])
@pytest.mark.parametrize("is_source", [True, False], ids=["source", "non-source"])
@pytest.mark.parametrize("output_format", [
    "FP8_DYNAMIC", "GPTQ_INT4", "GPTQ_INT8", "GPTQ_MXFP4", "gptq_int4",
])
def test_original_datafree_save_is_modifier_owned_on_every_rank(
    monkeypatch, tmp_path, wrapper_kind, is_source, output_format,
):
    from compressed_tensors import ModelCompressor
    import compressed_tensors.distributed as distributed
    import qlean.utils.dev as dev
    import qlean.pipelines.datafree.pipeline as datafree

    # Legacy datafree permits reusing the input checkpoint directory.
    save_file({"source.weight": torch.ones(1)}, tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text('{"source_owned": true}')
    model_args = ModelArguments(
        model_name="legacy-model", model_path=str(tmp_path), save_path=str(tmp_path),
        output_format=output_format, gptq_int4_export_device="not-a-valid-torch-device",
        gptq_export_workers=True,
    )
    if wrapper_kind == "day0":
        wrapper = day0Model(model_args)
        wrapper.load_model(model_args)
        assert type(wrapper) is day0Model
        assert isinstance(wrapper.model_, list)
        assert wrapper.model_ == [("model.safetensors", str(tmp_path / "model.safetensors"))]
    else:
        wrapper = PretrainedModel(model_args, _ForbiddenHF())

    for name in ("config", "generation_config", "named_modules", "name_or_path"):
        monkeypatch.setattr(type(wrapper), name, property(_forbidden), raising=False)
    monkeypatch.setattr(type(wrapper), "save_pretrained", _forbidden, raising=False)
    assert callable(wrapper.save_pretrained)
    if wrapper_kind == "hf":
        assert callable(wrapper.model_.save_pretrained)

    calls, rank_calls = [], []
    modifier_type = (
        type("GPTQModifier", (_SavingModifier,), {}) if wrapper_kind == "hf"
        else _SavingModifier
    )
    modifier = modifier_type("legacy", calls)
    assert not is_gptq_modifier(modifier)
    compressor = _bare_compressor(wrapper, model_args, [modifier])
    monkeypatch.setattr(datafree, "get_compressor", lambda: compressor)

    def source_process():
        rank_calls.append(is_source)
        return is_source

    monkeypatch.setattr(distributed, "is_source_process", source_process)
    monkeypatch.setattr(ModelCompressor, "from_pretrained_model", _forbidden)
    monkeypatch.setattr(ModelCompressor, "compress_model", _forbidden)
    offload = importlib.import_module("compressed_tensors.offload.convert.to_accelerate")
    monkeypatch.setattr(offload, "to_accelerate", _forbidden)
    monkeypatch.setattr(dev, "suspend_distributed_timeout", _forbidden)
    monkeypatch.setattr(dev, "get_main_device", _forbidden)
    futures = importlib.import_module("concurrent.futures")
    thread = importlib.import_module("concurrent.futures.thread")
    monkeypatch.setattr(futures, "ThreadPoolExecutor", _forbidden)
    monkeypatch.setattr(thread, "ThreadPoolExecutor", _forbidden)
    _block_imports(monkeypatch, (
        "qlean.core.checkpoint", "qlean.modifiers.gptq", "qlean.pipelines.sequential",
        "compressed_tensors.offload.convert.to_accelerate", "concurrent.futures",
    ))

    original_cleanup = compressor_module.remove_unsupported_fields
    original_metadata = compressor_module.save_qlean_metadata

    def cleanup(path):
        assert modifier.ended_
        assert path == model_args.save_path
        calls.append(("orchestration", "cleanup"))
        original_cleanup(path)

    def metadata(**kwargs):
        assert calls[-1] == ("orchestration", "cleanup")
        assert kwargs == {
            "save_path": model_args.save_path, "model_args": model_args,
            **compressor._quant_meta,
        }
        calls.append(("orchestration", "metadata"))
        original_metadata(**kwargs)

    monkeypatch.setattr(compressor_module, "remove_unsupported_fields", cleanup)
    monkeypatch.setattr(compressor_module, "save_qlean_metadata", metadata)
    compressor.post_train(SimpleNamespace())

    assert calls == [
        ("legacy", "initialize"), ("legacy", EventType.CALIBRATION_EPOCH_START),
        ("legacy", "start"), ("legacy", EventType.CALIBRATION_EPOCH_END),
        ("legacy", "save"), ("legacy", "end"),
        ("orchestration", "cleanup"), ("orchestration", "metadata"),
    ]
    assert modifier.initialized_ and modifier.started_ and modifier.ended_
    assert not modifier.finalized_
    assert rank_calls == []  # Legacy cleanup/metadata were never source-only.
    for name, expected in modifier.saved_bytes.items():
        assert (tmp_path / name).read_bytes() == expected
    assert list(load_file(tmp_path / "model.safetensors")) == ["layer.weight_packed"]
    assert json.loads((tmp_path / "config.json").read_text()) == {
        "modifier_owned": {"sentinel": "not an HF save", "num_mtp_layers": 2},
        "quantization_config": {"format": "modifier-owned"},
    }
    metadata_result = json.loads((tmp_path / "qlean_config.json").read_text())
    assert metadata_result["scheme"] == "legacy"
    assert metadata_result["output_format"] == output_format
    assert metadata_result["recipe_content"] == {"modifier_owned": True}


@pytest.mark.parametrize("query", [[], ["GPTQ_INT4"], ["gptqmodifier"], ["OtherModifier"]])
def test_non_gptq_string_queries_do_not_import_sequential(monkeypatch, query):
    _block_imports(monkeypatch, ("qlean.pipelines.sequential", "qlean.modifiers.gptq"))
    assert get_pipeline_from_modifiers(query).__module__ == "qlean.pipelines.datafree.pipeline"


def test_detection_does_not_load_an_unregistered_backend(monkeypatch):
    monkeypatch.delitem(ModifierFactory._main_registry, "GPTQModifier", raising=False)
    monkeypatch.setattr(ModifierFactory, "load_from_package", _forbidden)
    _block_imports(monkeypatch, ("qlean.modifiers.gptq", "qlean.pipelines.sequential"))
    impostor = type("GPTQModifier", (), {})()
    for modifier in (impostor, "GPTQModifier", "GPTQ_INT4", object()):
        assert not is_gptq_modifier(modifier)
    assert get_pipeline_from_modifiers([impostor]).__module__ == (
        "qlean.pipelines.datafree.pipeline"
    )


def test_registered_gptq_subclasses_route_by_identity_not_name(monkeypatch):
    pytest.importorskip("qlean.modifiers.gptq.base", reason="requires the qlean[gptq] extra")
    from qlean.modifiers.gptq import GPTQModifier

    class RealSubclass(GPTQModifier):
        pass

    real = RealSubclass(scheme="INT8_CHANNEL")
    impostor = type("GPTQModifier", (), {})()
    assert ModifierFactory._main_registry["GPTQModifier"] is GPTQModifier
    assert is_gptq_modifier(real)
    assert is_gptq_modifier(GPTQModifier(scheme="INT8_CHANNEL"))
    assert not is_gptq_modifier(impostor)
    assert not is_gptq_modifier("GPTQModifier")
    assert not is_gptq_modifier("GPTQ_INT4")

    selected = ModuleType("qlean.pipelines.sequential")
    selected.run_pipeline = lambda *args: None
    monkeypatch.setitem(sys.modules, selected.__name__, selected)
    _block_imports(monkeypatch, ("qlean.modifiers.gptq", "qlean.pipelines.datafree"))
    for modifiers in ([real], [impostor, real], ["GPTQModifier"]):
        assert get_pipeline_from_modifiers(modifiers) is selected.run_pipeline


class _EventHarness(EventMixin):
    def __init__(self, modifiers):
        self.state = State(model=SimpleNamespace(path="legacy-source"))
        self.modifiers = modifiers


def test_non_gptq_events_keep_old_signature_order_and_active_filter(monkeypatch):
    calls = []
    modifiers = [_LegacyModifier(name, calls) for name in ("first", "new", "ended", "last")]
    for modifier in modifiers:
        modifier.initialized_ = modifier.name != "new"
        modifier.ended_ = modifier.name == "ended"
    harness = _EventHarness(modifiers)
    detections = []

    def detect(modifier):
        detections.append(modifier)
        return is_gptq_modifier(modifier)

    monkeypatch.setattr(event_module, "is_gptq_modifier", detect)
    for event_type in EVENTS:
        getattr(harness, event_type.value)(ignored="not forwarded", module=object())
    assert calls == [(name, event_type) for event_type in EVENTS for name in ("first", "last")]
    assert detections == modifiers * len(EVENTS)


def test_gptq_events_keep_callbacks_and_kwargs_for_the_whole_pipeline(monkeypatch):
    from qlean.modifiers.gptq import GPTQModifier

    class RealSubclass(GPTQModifier):
        pass

    calls = []

    def on_event(self, state, event, **kwargs):
        calls.append((type(self).__name__, "event", state, event.type_, kwargs))

    def callback(self, state, event, **kwargs):
        calls.append((type(self).__name__, "callback", state, event.type_, kwargs))

    class Companion(Modifier):
        def on_initialize(self, state):
            self.initialized_ = True

    for cls in (GPTQModifier, Companion):
        monkeypatch.setattr(cls, "on_event", on_event)
        for name in ("on_calibration_start", "on_sequential_epoch_start",
                     "on_sequential_epoch_end", "on_calibration_end"):
            monkeypatch.setattr(cls, name, callback)
    real, companion = RealSubclass(scheme="INT8_CHANNEL"), Companion()
    for modifier in (real, companion):
        modifier.initialized_ = True
        modifier.ended_ = False
    harness = _EventHarness([real, companion])
    detections = []

    def detect(modifier):
        detections.append(modifier)
        return is_gptq_modifier(modifier)

    monkeypatch.setattr(event_module, "is_gptq_modifier", detect)
    kwargs = {"module": object(), "layer_index": 3}
    for event_type in EVENTS:
        getattr(harness, event_type.value)(**kwargs)
    assert calls == [
        (name, kind, harness.state, event_type, kwargs)
        for event_type in EVENTS for name in ("RealSubclass", "Companion")
        for kind in ("event", "callback")
    ]
    assert detections == [real] * len(EVENTS)


def test_clean_process_non_gptq_orchestration_does_not_import_gptq_save_stack(tmp_path):
    # Block before qlean imports so loaded modules cannot hide eager imports.
    script = r'''
import importlib.abc
import importlib.machinery
import runpy
import sys
import types
from pathlib import Path

import torch
import torch._dynamo
import compressed_tensors
import transformers

class UnavailableKernel:
    def __init__(self, function):
        self.__name__ = function.__name__
    def __call__(self, *args, **kwargs):
        raise RuntimeError("Triton kernel execution unavailable in CPU isolation test")
    def __getitem__(self, grid):
        return self

triton = types.ModuleType("triton")
triton.__spec__ = importlib.machinery.ModuleSpec("triton", loader=None)
triton.jit = triton.Config = UnavailableKernel
language = types.ModuleType("triton.language")
language.__spec__ = importlib.machinery.ModuleSpec("triton.language", loader=None)
language.constexpr = lambda value: value
triton.language = language
sys.modules["triton"] = triton
sys.modules["triton.language"] = language

forbidden = ("qlean.core.checkpoint", "qlean.modifiers.gptq", "qlean.pipelines.sequential",
             "qlean.modifiers.quantization")
class RejectGPTQImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in forbidden):
            raise AssertionError("Unexpected GPTQ orchestration import: " + fullname)
sys.meta_path.insert(0, RejectGPTQImports())
sys.path.insert(0, sys.argv[1])
ct_forward = __import__("compressed_tensors.quantization.lifecycle.forward", fromlist=[""])
ct_initialize = __import__("compressed_tensors.quantization.lifecycle.initialize", fromlist=[""])
original = ct_forward.set_forward_quantized
cached_original = ct_initialize.set_forward_quantized
import qlean
assert ct_forward.set_forward_quantized is original
assert ct_initialize.set_forward_quantized is cached_original
assert qlean.__all__ == ["QLeanCompressor"]
assert Path(qlean.__file__).resolve() == Path(sys.argv[1]) / "qlean" / "__init__.py"
namespace = runpy.run_path(sys.argv[2])
ModifierFactory = namespace["ModifierFactory"]
def reject_scan(*args, **kwargs):
    raise AssertionError("Factory scans are outside this isolation regression")
ModifierFactory.load_from_package = reject_scan

output = Path(sys.argv[3])
namespace["save_file"]({"source.weight": torch.ones(1)}, output / "model.safetensors")
args = namespace["ModelArguments"](model_name="legacy", model_path=str(output),
                                   save_path=str(output), output_format="GPTQ_INT4",
                                   gptq_int4_export_device="not-a-valid-torch-device",
                                   gptq_export_workers=True)
wrapper = namespace["day0Model"](args)
wrapper.load_model(args)
assert isinstance(wrapper.model_, list)
calls = []
modifier = namespace["_SavingModifier"]("legacy", calls)
compressor = namespace["_bare_compressor"](wrapper, args, [modifier])
import qlean.pipelines.datafree.pipeline as datafree
datafree.get_compressor = lambda: compressor
import concurrent.futures
import concurrent.futures.thread
concurrent.futures.ThreadPoolExecutor = reject_scan
concurrent.futures.thread.ThreadPoolExecutor = reject_scan
import pytest
with pytest.MonkeyPatch.context() as patch:
    namespace["_block_imports"](patch, ("concurrent.futures",))
    compressor.post_train(types.SimpleNamespace())
assert modifier.ended_
assert (output / "qlean_config.json").is_file()
for name, expected in modifier.saved_bytes.items():
    assert (output / name).read_bytes() == expected
assert not any(name == prefix or name.startswith(prefix + ".")
               for name in sys.modules for prefix in forbidden)
print("clean-process non-GPTQ isolation verified")
'''
    source_root = Path(compressor_module.__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", script, str(source_root), str(Path(__file__).resolve()),
         str(tmp_path)], capture_output=True, text=True, check=False, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "clean-process non-GPTQ isolation verified" in result.stdout


LEGACY_CLASSES = (
    ("generalDay0", "generalDay0Modifier"),
    ("fp8BasedDay0", "fp8BasedDay0Modifier"),
    ("nvfp4BasedMxfp4Day0", "nvfp4BasedMxfp4Day0Modifier"),
    ("dpskV4mixGeneralDay0", "dpskV4mixGeneralDay0Modifier"),
    ("W4a8Day0", "W4a8Day0Modifier"),
    ("dpskV4mix2W4a8", "dpskV4mix2W4a8Day0Modifier"),
    ("dequant", "dequantModifier"),
)


@pytest.mark.parametrize("package,class_name", LEGACY_CLASSES)
def test_real_legacy_fields_and_dump_match_head(package, class_name, monkeypatch):
    import ast
    from qlean.modifiers import QuantizationMixin
    from qlean.modifiers.quantization import QuantizationCalibrationMixin

    module = importlib.import_module(f"qlean.modifiers.{package}.base")
    real_class = getattr(module, class_name)
    root = Path(compressor_module.__file__).resolve().parents[3]
    filename = f"src/qlean/modifiers/{package}/base.py"
    source = subprocess.check_output(
        ["git", "-c", "safe.directory=" + str(root), "-C", str(root),
         "show", f"HEAD:{filename}"], text=True,
    )
    # Execute only the original class body using the same dependency namespace;
    # no registration, source-checkout import, or file mutation is involved.
    tree = ast.parse(source)
    class_node = next(node for node in tree.body
                      if isinstance(node, ast.ClassDef) and node.name == class_name)
    class_node.decorator_list = []
    namespace = dict(vars(module))
    exec(compile(ast.Module(body=[class_node], type_ignores=[]), filename, "exec"), namespace)
    legacy_class = namespace[class_name]
    real, legacy = real_class(), legacy_class()
    assert real_class.model_fields.keys() == legacy_class.model_fields.keys()
    assert real.model_dump(mode="json") == legacy.model_dump(mode="json")
    assert set(QuantizationMixin.model_fields) == {"config_group"}
    assert QuantizationMixin().model_dump() == {"config_group": None}
    assert real._resolved_config is None
    assert not isinstance(real, QuantizationCalibrationMixin)
    assert not hasattr(real, "initialize_quantization")
    assert "bypass_divisibility_checks" not in real_class.model_fields


def test_real_non_gptq_constructor_does_not_inspect_deepseek(monkeypatch, tmp_path):
    from qlean.modifiers.W4a8Day0.base import W4a8Day0Modifier

    class DeepSeekSentinel(torch.nn.Module):
        @property
        def config(self):
            pytest.fail("Legacy constructor inspected DeepSeek configuration")

    monkeypatch.setattr(compressor_module, "install_dependencies", lambda name: "day0")
    monkeypatch.setattr(compressor_module.ModelFactory, "_loaded", True)
    monkeypatch.setattr(compressor_module.ModelFactory, "create",
                        lambda mode, args: SimpleNamespace(model_=None, path=args.model_path))
    _block_imports(monkeypatch, ("qlean.modeling.deepseekv32", "qlean.modifiers.quantization"))
    model = DeepSeekSentinel()
    args = ModelArguments(model_name="deepseek-sentinel", model_path=str(tmp_path),
                          save_path=str(tmp_path / "out"), output_format="FP8_DYNAMIC")
    compressor = object.__new__(QLeanCompressor)
    QLeanCompressor.__init__(compressor, args, model=model, modifiers=[W4a8Day0Modifier()])
    assert compressor.state.model.model_ is model
    assert compressor._quant_meta["recipe_content"]["modifiers"][0]["config"] == {
        "config_group": None, "ignore": [], "initialized_": False,
        "started_": False, "ended_": False, "finalized_": False,
    }


@pytest.mark.parametrize("package,class_name,expert,scale_suffix", [
    ("W4a8Day0", "W4a8Day0Modifier", "model.layers.0.mlp.experts.0.up_proj.weight", "_scale"),
    ("dpskV4mix2W4a8", "dpskV4mix2W4a8Day0Modifier", "model.layers.0.ffn.experts.0.w1.weight", ".scale"),
])
def test_original_pack_ast_and_real_datafree_save(
    monkeypatch, tmp_path, package, class_name, expert, scale_suffix,
):
    import ast
    from qlean.utils.pack_acext_int4 import preprocess_weight_for_a8w4_torch
    import qlean.pipelines.datafree.pipeline as datafree

    module = importlib.import_module(f"qlean.modifiers.{package}.base")
    root = Path(module.__file__).resolve().parents[4]
    filename = f"src/qlean/modifiers/{package}/base.py"
    original = subprocess.check_output(
        ["git", "-c", "safe.directory=" + str(root), "-C", str(root),
         "show", f"HEAD:{filename}"], text=True,
    )
    def pack_ast(source):
        return next(ast.dump(node.value, include_attributes=False)
                    for node in ast.walk(ast.parse(source))
                    if isinstance(node, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id == "packed_weight"
                        for target in node.targets))
    assert pack_ast(Path(module.__file__).read_text()) == pack_ast(original)
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    weight = torch.linspace(-2, 2, 64 * 64).reshape(64, 64).to(torch.bfloat16)
    ignored = "model.layers.0.attn.q_proj.weight"
    tensors = {expert: weight, ignored: weight.clone(), "model.layers.0.norm.weight": torch.ones(64)}
    save_file(tensors, source / "model.safetensors")
    (source / "config.json").write_text('{"hidden_size": 64}')
    (source / "generation_config.json").write_bytes(b'{"do_sample": false}\n')
    args = ModelArguments(model_name="legacy", model_path=str(source), save_path=str(output),
                          output_format="FP8_DYNAMIC")
    wrapper = day0Model(args)
    wrapper.load_model(args)
    modifier = getattr(module, class_name)(ignore=["re:.*attn.*"])
    compressor = _bare_compressor(wrapper, args, [modifier])
    monkeypatch.setattr(datafree, "get_compressor", lambda: compressor)
    # Only replace the GPU transfer boundary. Quantization, packer, lifecycle,
    # atomic safetensors save, index/config handling and metadata are real.
    monkeypatch.setattr(module, "load_file_to_cuda", lambda path: load_file(path, device="cpu"))
    _block_imports(monkeypatch, ("qlean.modifiers.gptq", "qlean.modifiers.quantization"))
    dtype = torch.get_default_dtype()
    try:
        compressor.post_train(SimpleNamespace())
    finally:
        torch.set_default_dtype(dtype)
    saved = load_file(output / "model.safetensors")
    codes, scales = module.quantize_bf16_to_int(weight)
    expected = preprocess_weight_for_a8w4_torch(codes).view(64, 32)
    assert torch.equal(saved[expert], expected)
    scale_name = expert + scale_suffix if scale_suffix == "_scale" else expert.removesuffix(".weight") + scale_suffix
    assert torch.equal(saved[scale_name], scales)
    assert torch.equal(saved[ignored], tensors[ignored])
    assert torch.equal(saved["model.layers.0.norm.weight"], tensors["model.layers.0.norm.weight"])
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert index["weight_map"] == {key: "model.safetensors" for key in saved}
    config = json.loads((output / "config.json").read_text())
    assert config["quantization_config"] == {
        "activation_scheme": "dynamic", "quant_method": "mixed_precision_w4",
        "ignore": [ignored.removesuffix(".weight")],
    }
    assert modifier.initialized_ and modifier.started_ and modifier.ended_
    assert (output / "generation_config.json").read_bytes() == (source / "generation_config.json").read_bytes()
    assert (output / "qlean_config.json").is_file()
    assert torch.equal(load_file(source / "model.safetensors")[expert], weight)


def test_real_general_datafree_fp8_save_keeps_legacy_quantization_and_ignore(monkeypatch, tmp_path):
    import qlean.modifiers.generalDay0.base as module
    import qlean.pipelines.datafree.pipeline as datafree

    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    tensors = {
        "model.layers.0.attn.q_proj.weight": torch.linspace(-2, 2, 64).reshape(8, 8).to(torch.bfloat16),
        "model.embed_tokens.weight": torch.ones(8, 8, dtype=torch.bfloat16),
        "model.layers.0.norm.weight": torch.ones(8, dtype=torch.bfloat16),
    }
    save_file(tensors, source / "model.safetensors")
    (source / "config.json").write_text(json.dumps({
        "hidden_size": 8, "quantization_config": {"source_only": True},
    }))
    args = ModelArguments(model_name="legacy", model_path=str(source), save_path=str(output),
                          output_format="FP8_DYNAMIC")
    wrapper = day0Model(args)
    wrapper.load_model(args)
    mod = module.generalDay0Modifier(scheme="FP8_DYNAMIC", ignore=["model.embed_tokens"])
    compressor = _bare_compressor(wrapper, args, [mod])
    monkeypatch.setattr(datafree, "get_compressor", lambda: compressor)
    monkeypatch.setattr(module, "load_file_to_cuda", lambda path: load_file(path, device="cpu"))
    _block_imports(monkeypatch, ("qlean.modifiers.gptq", "qlean.modifiers.quantization"))
    dtype = torch.get_default_dtype()
    try:
        compressor.post_train(SimpleNamespace())
    finally:
        torch.set_default_dtype(dtype)
    saved = load_file(output / "model.safetensors")
    quantized, scale = module.weight_quant(tensors["model.layers.0.attn.q_proj.weight"], "FP8_DYNAMIC")
    assert saved["model.layers.0.attn.q_proj.weight"].dtype == torch.float8_e4m3fn
    assert torch.equal(saved["model.layers.0.attn.q_proj.weight"].float(), quantized.float())
    assert torch.equal(saved["model.layers.0.attn.q_proj.weight_scale"], scale)
    assert torch.equal(saved["model.embed_tokens.weight"], tensors["model.embed_tokens.weight"])
    assert torch.equal(saved["model.layers.0.norm.weight"], tensors["model.layers.0.norm.weight"])
    config = json.loads((output / "config.json").read_text())["quantization_config"]
    assert config["quant_method"] == "compressed-tensors"
    assert config["format"] == "float-quantized"
    assert config["ignore"] == []  # Original embedding ignore removal contract.
    assert "source_only" not in config
    assert "bypass_divisibility_checks" not in compressor._quant_meta["recipe_content"]
    assert mod.initialized_ and mod.started_ and mod.ended_
    assert (output / "qlean_config.json").is_file()
