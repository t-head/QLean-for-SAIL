# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import copy
import importlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import PretrainedConfig, conversion_mapping, monkey_patching
from transformers.core_model_loading import WeightRenaming

linearize = importlib.import_module("qlean.modeling.moe.linearize")


pytestmark = [pytest.mark.gptq, pytest.mark.integration, pytest.mark.cpu]


class IsolationExperts(torch.nn.Module):
    pass


class IsolationLinearExperts(torch.nn.Module):
    pass


class CallerExperts(torch.nn.Module):
    pass


class LoadFailure(RuntimeError):
    pass


def _mapping(label):
    return [WeightRenaming(source_patterns=label, target_patterns=f"{label}_target")]


def _signature(mapping):
    if mapping is None:
        return None
    return [(item.source_patterns, item.target_patterns) for item in mapping]


@pytest.fixture
def tf_registries(monkeypatch):
    conversion_mapping.get_checkpoint_conversion_mapping("qlean_moe_loading_isolation")
    # Use isolated copies of real TF registries, not fake registration functions.
    monkeypatch.setattr(
        conversion_mapping, "_checkpoint_conversion_mapping_cache",
        conversion_mapping._checkpoint_conversion_mapping_cache.copy(),
    )
    monkeypatch.setattr(
        conversion_mapping, "USER_REGISTERED_MAPPINGS",
        conversion_mapping.USER_REGISTERED_MAPPINGS.copy(),
    )
    monkeypatch.setattr(
        monkey_patching, "_monkey_patch_mapping_cache",
        monkey_patching.get_patch_mapping(),
    )


@pytest.fixture
def loader(monkeypatch, tf_registries):
    model_type = "qlean_moe_loading_isolation"
    config = PretrainedConfig()
    config.model_type = model_type
    config.user_options = {"keep": [1, 2]}
    state = SimpleNamespace(
        model_type=model_type, config=config, load_map=_mapping("load"),
        save_map=_mapping("save"), calls=0,
    )

    def ordinary_load(*args, **kwargs):
        state.calls += 1
        model = torch.nn.Module()
        model.config = config
        return model

    state.load = ordinary_load

    class Loader:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return state.load(*args, **kwargs)

    state.cls = Loader
    state.original_method = Loader.from_pretrained.__func__
    monkeypatch.setattr(
        linearize.AutoConfig, "from_pretrained", lambda *args, **kwargs: config
    )
    monkeypatch.setattr(linearize, "has_linearize_load_mappings", lambda key: True)
    monkeypatch.setattr(
        linearize, "get_linearize_load_mappings",
        lambda key: (IsolationExperts, state.load_map, state.save_map),
    )
    monkeypatch.setattr(
        linearize.LinearExperts2D, "get_linear_experts_cls",
        classmethod(lambda cls, key: IsolationLinearExperts),
    )
    return state


def _seed_caller_state(loader, entry, registered, same_patch):
    cache = conversion_mapping._checkpoint_conversion_mapping_cache
    cache.pop(loader.model_type, None)
    if entry != "absent":
        cache[loader.model_type] = {
            "none": None, "empty": [], "mapping": _mapping("caller"),
        }[entry]
    if registered:
        conversion_mapping.USER_REGISTERED_MAPPINGS.add(loader.model_type)
    else:
        conversion_mapping.USER_REGISTERED_MAPPINGS.discard(loader.model_type)
    monkey_patching.register_patch_mapping({"UnrelatedIsolationModule": CallerExperts})
    if same_patch:
        monkey_patching.register_patch_mapping({"IsolationExperts": CallerExperts})
    return (
        cache.copy(), conversion_mapping.USER_REGISTERED_MAPPINGS.copy(),
        monkey_patching.get_patch_mapping(), copy.deepcopy(loader.config.to_dict()),
    )


def _assert_restored(loader, before):
    cache, users, patches, config = before
    current = conversion_mapping._checkpoint_conversion_mapping_cache
    assert current.keys() == cache.keys()
    assert all(current[key] is value for key, value in cache.items())
    assert conversion_mapping.USER_REGISTERED_MAPPINGS == users
    assert monkey_patching.get_patch_mapping() == patches
    assert loader.config.to_dict() == config
    assert loader.cls.from_pretrained.__func__ is loader.original_method
    # Both lookup behavior and key membership must be exact (missing != None).
    assert _signature(conversion_mapping.get_checkpoint_conversion_mapping(
        loader.model_type
    )) == _signature(cache.get(loader.model_type))


def _assert_temporary(loader):
    assert conversion_mapping._checkpoint_conversion_mapping_cache[
        loader.model_type
    ] is loader.load_map
    assert _signature(conversion_mapping.get_checkpoint_conversion_mapping(
        loader.model_type
    )) == _signature(loader.load_map)
    assert loader.model_type in conversion_mapping.USER_REGISTERED_MAPPINGS
    patches = monkey_patching.get_patch_mapping()
    assert patches["IsolationExperts"] is IsolationLinearExperts
    assert patches["UnrelatedIsolationModule"] is CallerExperts


@pytest.mark.parametrize("entry", ["absent", "none", "empty", "mapping"])
@pytest.mark.parametrize("registered", [False, True])
@pytest.mark.parametrize("same_patch", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_load_restores_exact_caller_state(loader, entry, registered, same_patch, fail):
    before = _seed_caller_state(loader, entry, registered, same_patch)
    model = torch.nn.Module()

    def load(*args, **kwargs):
        _assert_temporary(loader)
        if fail:
            raise LoadFailure("load sentinel")
        return (model, {"loading_info": True}) if same_patch else model

    loader.load = load
    with linearize.load_quantizable_moe(loader.cls):
        if fail:
            with pytest.raises(LoadFailure, match="load sentinel"):
                loader.cls.from_pretrained("fixture")
        else:
            result = loader.cls.from_pretrained("fixture")
            assert (result[0] if isinstance(result, tuple) else result) is model
            assert model._weight_conversions is loader.save_map
    _assert_restored(loader, before)

    def ordinary_load(*args, **kwargs):
        _assert_restored(loader, before)
        return "ordinary loader unaffected"

    loader.load = ordinary_load
    assert loader.cls.from_pretrained("ordinary") == "ordinary loader unaffected"


def test_unmapped_load_only_linearizes_locally(monkeypatch, loader):
    before = _seed_caller_state(loader, "mapping", True, True)
    model = torch.nn.Module()
    result = (model, {"loading_info": True})
    seen = []
    loader.load = lambda *args, **kwargs: result
    monkeypatch.setattr(linearize, "has_linearize_load_mappings", lambda key: False)
    monkeypatch.setattr(linearize, "linearize_moe", seen.append)
    with linearize.load_quantizable_moe(loader.cls):
        assert loader.cls.from_pretrained("unmapped") is result
    assert seen == [model]
    assert not hasattr(model, "_weight_conversions")
    _assert_restored(loader, before)


@pytest.mark.parametrize("entry", ["absent", "mapping"])
@pytest.mark.parametrize("same_patch", [False, True])
@pytest.mark.parametrize("stage", [
    "config", "mappings", "class", "patch_before", "patch_after",
    "conversion_before", "conversion_after", "save",
])
def test_failed_setup_or_save_restores_state(monkeypatch, loader, entry, same_patch, stage):
    before = _seed_caller_state(loader, entry, False, same_patch)

    def fail(*args, **kwargs):
        raise LoadFailure(f"{stage} sentinel")

    if stage == "config":
        monkeypatch.setattr(linearize.AutoConfig, "from_pretrained", fail)
    elif stage == "mappings":
        monkeypatch.setattr(linearize, "get_linearize_load_mappings", fail)
    elif stage == "class":
        monkeypatch.setattr(linearize.LinearExperts2D, "get_linear_experts_cls", fail)
    elif stage == "save":
        monkeypatch.setattr(linearize, "set_save_conversion_mapping", fail)
    else:
        name = (
            "register_patch_mapping" if stage.startswith("patch")
            else "register_checkpoint_conversion_mapping"
        )
        original = getattr(linearize, name)
        attempted = False

        def fail_once(*args, **kwargs):
            nonlocal attempted
            if attempted:
                return original(*args, **kwargs)
            attempted = True
            if stage.endswith("after"):
                original(*args, **kwargs)
            fail()

        monkeypatch.setattr(linearize, name, fail_once)

    with pytest.raises(LoadFailure, match=f"{stage} sentinel"):
        with linearize.load_quantizable_moe(loader.cls):
            loader.cls.from_pretrained("fixture")
    _assert_restored(loader, before)
    assert loader.calls == (1 if stage == "save" else 0)


@pytest.mark.parametrize("fail", [False, True])
def test_same_class_nested_contexts_restore_original(loader, fail):
    before = _seed_caller_state(loader, "mapping", True, True)
    model = torch.nn.Module()

    def load(*args, **kwargs):
        _assert_temporary(loader)
        if fail:
            raise LoadFailure("nested sentinel")
        return model

    loader.load = load
    with linearize.load_quantizable_moe(loader.cls):
        outer_method = loader.cls.from_pretrained.__func__
        with linearize.load_quantizable_moe(loader.cls):
            if fail:
                with pytest.raises(LoadFailure, match="nested sentinel"):
                    loader.cls.from_pretrained("fixture")
            else:
                assert loader.cls.from_pretrained("fixture") is model
        assert loader.cls.from_pretrained.__func__ is outer_method
        assert conversion_mapping.USER_REGISTERED_MAPPINGS == before[1]
        assert monkey_patching.get_patch_mapping() == before[2]
        # Use the still-active outer wrapper after the inner context exits.
        if fail:
            with pytest.raises(LoadFailure):
                loader.cls.from_pretrained("fixture")
        else:
            assert loader.cls.from_pretrained("fixture") is model
    _assert_restored(loader, before)


@pytest.mark.parametrize("inner_fail", [False, True])
@pytest.mark.parametrize("outer_fail", [False, True])
def test_reentrant_load_restores_outer_temporary_state(loader, inner_fail, outer_fail):
    before = _seed_caller_state(loader, "absent", False, True)

    class InnerLoader:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            _assert_temporary(loader)
            if inner_fail:
                raise LoadFailure("inner sentinel")
            return torch.nn.Module()

    original_inner = InnerLoader.from_pretrained.__func__

    def outer_load(*args, **kwargs):
        _assert_temporary(loader)
        with linearize.load_quantizable_moe(InnerLoader):
            if inner_fail:
                with pytest.raises(LoadFailure, match="inner sentinel"):
                    InnerLoader.from_pretrained("inner")
            else:
                assert InnerLoader.from_pretrained("inner")._weight_conversions is loader.save_map
        assert InnerLoader.from_pretrained.__func__ is original_inner
        _assert_temporary(loader)
        if outer_fail:
            raise LoadFailure("outer sentinel")
        return torch.nn.Module()

    loader.load = outer_load
    with linearize.load_quantizable_moe(loader.cls):
        if outer_fail:
            with pytest.raises(LoadFailure, match="outer sentinel"):
                loader.cls.from_pretrained("outer")
        else:
            loader.cls.from_pretrained("outer")
    _assert_restored(loader, before)


def test_unrelated_patch_added_during_load_survives(loader):
    before = _seed_caller_state(loader, "mapping", False, True)

    def load(*args, **kwargs):
        monkey_patching.register_patch_mapping({"AddedDuringLoad": CallerExperts})
        conversion_mapping.register_checkpoint_conversion_mapping(
            "unrelated_during_load", _mapping("unrelated")
        )
        return torch.nn.Module()

    loader.load = load
    with linearize.load_quantizable_moe(loader.cls):
        loader.cls.from_pretrained("fixture")
    before[0]["unrelated_during_load"] = (
        conversion_mapping._checkpoint_conversion_mapping_cache["unrelated_during_load"]
    )
    before[1].add("unrelated_during_load")
    before[2]["AddedDuringLoad"] = CallerExperts
    _assert_restored(loader, before)


def test_interrupt_restores_state(loader):
    before = _seed_caller_state(loader, "absent", False, False)

    def load(*args, **kwargs):
        raise KeyboardInterrupt("interrupt sentinel")

    loader.load = load
    with pytest.raises(KeyboardInterrupt, match="interrupt sentinel"):
        with linearize.load_quantizable_moe(loader.cls):
            loader.cls.from_pretrained("fixture")
    _assert_restored(loader, before)


@pytest.mark.parametrize("fail", [False, True])
def test_real_ordinary_loader_after_moe_context(loader, tmp_path, fail):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=32,
    )).eval()
    source = tmp_path / "tiny-dense"
    model.save_pretrained(source)
    config_bytes = (source / "config.json").read_bytes()
    before = (
        conversion_mapping._checkpoint_conversion_mapping_cache.copy(),
        conversion_mapping.USER_REGISTERED_MAPPINGS.copy(),
        monkey_patching.get_patch_mapping(), copy.deepcopy(loader.config.to_dict()),
    )
    if fail:
        def failed_load(*args, **kwargs):
            raise LoadFailure("MoE sentinel")
        loader.load = failed_load
    with linearize.load_quantizable_moe(loader.cls):
        if fail:
            with pytest.raises(LoadFailure, match="MoE sentinel"):
                loader.cls.from_pretrained("fixture")
        else:
            assert loader.cls.from_pretrained("fixture")._weight_conversions is loader.save_map
    _assert_restored(loader, before)
    # Use a real ordinary HF checkpoint loader after the mocked MoE transaction.
    ordinary = LlamaForCausalLM.from_pretrained(source).eval()
    inputs = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        torch.testing.assert_close(ordinary(inputs).logits, model(inputs).logits)
    assert (source / "config.json").read_bytes() == config_bytes
    _assert_restored(loader, before)


_LAZY_IMPORT_SCRIPT = r'''
import importlib.abc
import importlib.machinery
import sys
import types
from pathlib import Path

import torch
import torch._dynamo
import compressed_tensors
import transformers

# Import-only, fail-closed shim: no GPU kernel may silently execute in this test.
class UnavailableKernel:
    def __init__(self, function):
        self.__name__ = function.__name__
    def __call__(self, *args, **kwargs):
        raise RuntimeError("Triton execution unavailable in CPU isolation test")
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

mode = sys.argv[2]
architectures = ("granitemoe", "llama4")
forbidden = tuple(
    prefix
    for arch in architectures if arch != mode
    for prefix in (
        "qlean.modeling.moe." + arch,
        "transformers.models." + arch + ".configuration_" + arch,
        "transformers.models." + arch + ".modeling_" + arch,
    )
)
assert not any(name == prefix or name.startswith(prefix + ".")
               for name in sys.modules for prefix in forbidden)
class RejectArchitectureImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == prefix or fullname.startswith(prefix + ".") for prefix in forbidden):
            raise AssertionError("Eager architecture import: " + fullname)
sys.meta_path.insert(0, RejectArchitectureImports())
sys.path.insert(0, sys.argv[1])
import qlean
assert Path(qlean.__file__).resolve() == Path(sys.argv[1]) / "qlean" / "__init__.py"
from qlean.modeling.moe import LinearExperts2D, get_non_linearized_moes
assert all("qlean.modeling.moe." + arch not in sys.modules for arch in architectures)

if mode == "dense":
    from transformers import LlamaConfig, LlamaForCausalLM
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=16, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2,
    ))
    assert get_non_linearized_moes(model) == []
    # Neither a matching name alone nor a matching module alone is sufficient.
    default = object()
    for arch, name in (("granitemoe", "GraniteMoeParallelExperts"), ("llama4", "Llama4TextExperts")):
        wrong_module = type(name, (torch.nn.Module,), {"__module__": "user.models"})
        wrong_name = type("DenseProjection", (torch.nn.Module,), {
            "__module__": "transformers.models." + arch + ".modeling_" + arch,
        })
        assert LinearExperts2D.get_registration(wrong_module, default) is default
        assert LinearExperts2D.get_registration(wrong_name, default) is default
        LinearExperts2D._registry[wrong_module] = torch.nn.ModuleList
        assert LinearExperts2D.get_registration(wrong_module, default) is torch.nn.ModuleList
elif mode == "glm":
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaExperts
    experts = GlmMoeDsaExperts(GlmMoeDsaConfig(
        hidden_size=8, moe_intermediate_size=8, n_routed_experts=2,
    ))
    model = torch.nn.Module()
    model.add_module("experts", experts)
    assert get_non_linearized_moes(model) == [("experts", experts)]
else:
    if mode == "granitemoe":
        import transformers.models.granitemoe.modeling_granitemoe as architecture
        # New TF no longer exports the legacy class. Emulate only that class
        # export to exercise the preserved legacy registration implementation.
        if not hasattr(architecture, "GraniteMoeParallelExperts"):
            architecture.GraniteMoeParallelExperts = type(
                "GraniteMoeParallelExperts", (torch.nn.Module,),
                {"__module__": architecture.__name__},
            )
        key = architecture.GraniteMoeParallelExperts
        replacement_name = "GraniteMoeLinearExperts"
    else:
        from transformers.models.llama4.modeling_llama4 import Llama4TextExperts
        key = Llama4TextExperts
        replacement_name = "Llama4LinearExperts"
    assert "qlean.modeling.moe." + mode not in sys.modules
    # Explicit public registrations take priority and must not trigger imports.
    LinearExperts2D._registry[key] = torch.nn.ModuleList
    assert LinearExperts2D.get_registration(key) is torch.nn.ModuleList
    assert "qlean.modeling.moe." + mode not in sys.modules
    del LinearExperts2D._registry[key]
    replacement = LinearExperts2D.get_registration(key)
    assert replacement.__name__ == replacement_name
    assert replacement.__module__ == "qlean.modeling.moe." + mode
    assert LinearExperts2D.get_linear_experts_cls(key) is replacement
    assert LinearExperts2D.get_registration(key) is replacement

assert not any(name == prefix or name.startswith(prefix + ".")
               for name in sys.modules for prefix in forbidden)
print("lazy registration verified:", mode)
'''


@pytest.mark.parametrize("mode", ["dense", "glm", "granitemoe", "llama4"])
def test_clean_process_lazy_architecture_registration(mode):
    source_root = Path(linearize.__file__).resolve().parents[3]
    result = subprocess.run(
        [sys.executable, "-c", _LAZY_IMPORT_SCRIPT, str(source_root), mode],
        capture_output=True, text=True, check=False, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"lazy registration verified: {mode}" in result.stdout


def test_distributed_helpers_are_public_ct_functions(monkeypatch):
    import compressed_tensors.distributed as ct_distributed
    import qlean.modifiers.gptq.distributed as distributed

    assert distributed.greedy_bin_packing is ct_distributed.greedy_bin_packing
    assert distributed.wait_for_comms is ct_distributed.wait_for_comms
    monkeypatch.setattr(distributed.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(distributed.dist, "get_world_size", lambda: 2)
    assert not distributed.is_distributed()
    monkeypatch.setattr(distributed.dist, "is_initialized", lambda: True)
    assert distributed.is_distributed()
    monkeypatch.setattr(distributed.dist, "get_world_size", lambda: 1)
    assert not distributed.is_distributed()


@pytest.mark.parametrize("skip_cpu", [False, True])
def test_broadcast_retains_cpu_skip_and_observer_cleanup(monkeypatch, skip_cpu):
    import qlean.modifiers.gptq.distributed as distributed

    modules = [torch.nn.Module(), torch.nn.Module()]
    devices = dict(zip(modules, [torch.device("cpu"), torch.device("cuda")]))
    broadcasts, cleaned, waited = [], [], []
    for index, module in enumerate(modules):
        module.weight_scale = torch.tensor([float(index + 1)])
        module.weight_observer = SimpleNamespace(
            has_statistics=True, delete_statistics=lambda index=index: cleaned.append(index),
        )
    monkeypatch.setattr(distributed, "get_execution_device", devices.__getitem__)

    def broadcast(tensor, src, async_op):
        assert async_op
        broadcasts.append(src)
        return SimpleNamespace(wait=lambda: waited.append(src))

    monkeypatch.setattr(distributed.dist, "broadcast", broadcast)
    distributed.broadcast_qparams_and_cleanup(
        modules, dict(zip(modules, [0, 1])), ["weight_scale", "missing_param"],
        skip_cpu=skip_cpu,
    )
    assert broadcasts == ([1] if skip_cpu else [0, 1])
    assert waited == broadcasts
    assert cleaned == [0, 1]
