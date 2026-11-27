# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GPTQ routing contracts and 0.3.0 engine/CLI regression coverage."""

import argparse
import copy
import json
import runpy
import subprocess
import sys
from contextlib import nullcontext
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from compressed_tensors.utils.match import match_name, match_named_modules

import qlean.core.routes as routes
import qlean.core.utils as utils
from qlean.args import ModelArguments
from qlean.cli.__main__ import parse_output_format


pytestmark = [pytest.mark.gptq, pytest.mark.unit, pytest.mark.cpu]


def _gptq_recipe():
    return {
        "quant_stage": {
            "quant_modifiers": {
                "GPTQModifier": {
                    "block_size": 128,
                    "dampening_frac": 0.001,
                    "actorder": "static",
                    "config_groups": {
                        "group_0": {
                            "targets": ["Linear"],
                            "weights": {
                                "num_bits": 4, "type": "float", "symmetric": True,
                                "strategy": "group", "group_size": 32,
                                "scale_dtype": "torch.uint8", "zp_dtype": "torch.uint8",
                            },
                        },
                    },
                },
            },
        },
    }


@pytest.fixture
def capturing_factory(monkeypatch):
    class CapturingFactory:
        _loaded = True

        @staticmethod
        def create(modifier_type, **kwargs):
            return SimpleNamespace(type_name=modifier_type, **kwargs)

    import qlean.modifiers as modifiers

    monkeypatch.setattr(utils, "ModifierFactory", CapturingFactory)
    # get_modifiers_from_recipe imports the factory lazily from this package.
    monkeypatch.setattr(modifiers, "ModifierFactory", CapturingFactory)
    return CapturingFactory


_GPTQ_WEIGHTS = {
    "INT4_CHANNEL": {
        "num_bits": 4, "type": "int", "symmetric": True, "strategy": "channel",
    },
    "INT8_CHANNEL": {
        "num_bits": 8, "type": "int", "symmetric": True, "strategy": "channel",
    },
    "INT4_GROUP": {
        "num_bits": 4, "type": "int", "symmetric": True, "strategy": "group",
        "group_size": 128,
    },
    "INT8_GROUP": {
        "num_bits": 8, "type": "int", "symmetric": True, "strategy": "group",
        "group_size": 128,
    },
    "MXFP4": {
        "num_bits": 4, "type": "float", "symmetric": True, "strategy": "group",
        "group_size": 32, "scale_dtype": "torch.uint8", "zp_dtype": "torch.uint8",
    },
}
_GPTQ_OUTPUTS = [
    ("GPTQ_INT4", "INT4_CHANNEL"),
    ("GPTQ_INT8", "INT8_CHANNEL"),
    ("GPTQ_MXFP4", "MXFP4"),
    ("GPTQ_INT4_GROUP", "INT4_GROUP"),
    ("GPTQ_INT8_GROUP", "INT8_GROUP"),
]
_REMOVED_GPTQ_OUTPUTS = [
    "GPTQ_W4A8", "GPTQ_W8A16", "GPTQ_INT8A16",
    "GPTQ_MXFP4A16", "GPTQ_W4A16", "GPTQ_UNKNOWN",
    "GPTQ_INT4_BLOCK", "GPTQ_INT8_BLOCK",
]


def _assert_gptq_weights(weights, scheme):
    for key, value in _GPTQ_WEIGHTS[scheme].items():
        expected = getattr(torch, value.removeprefix("torch.")) if key in (
            "scale_dtype", "zp_dtype",
        ) else value
        assert getattr(weights, key) == expected
    assert weights.dynamic is False
    if scheme != "MXFP4":
        assert weights.group_size == (128 if scheme.endswith("_GROUP") else None)


@pytest.mark.parametrize("output_format,expected_preset", _GPTQ_OUTPUTS)
@pytest.mark.parametrize("scheme", ["W8A16", "MXFP4"])
def test_automatic_groups_preserve_targets_ignore_and_tuning(
    output_format, expected_preset, scheme,
):
    pytest.importorskip("qlean.modifiers.gptq.base", reason="requires the qlean[gptq] extra")
    from qlean.modifiers.gptq.base import GPTQModifier

    recipe = _gptq_recipe()
    modifier = recipe["quant_stage"]["quant_modifiers"]["GPTQModifier"]
    targets = [r"re:.*self_attn\..*"]
    ignore = ["lm_head", r"re:.*mlp\.shared_experts\..*"]
    modifier.update(scheme=scheme, targets=["Linear"], ignore=ignore)
    groups = modifier["config_groups"]
    groups["group_0"]["targets"] = targets
    groups["group_1"] = {
        "targets": [routes.GPTQ_EXPERT_TARGET],
        "weights": {"num_bits": 8, "type": "int", "strategy": "group", "group_size": 128},
    }
    original_targets = {name: group["targets"] for name, group in groups.items()}

    assert utils.apply_gptq_output_format(recipe, output_format) == expected_preset
    assert modifier["config_groups"] is groups
    assert "scheme" not in modifier
    assert modifier["targets"] == ["Linear"]
    assert modifier["ignore"] is ignore
    assert modifier["block_size"] == 128
    assert modifier["dampening_frac"] == 0.001
    assert modifier["actorder"] == "static"
    actual = GPTQModifier(**modifier)
    resolved = actual.resolve_quantization_config()
    assert resolved.ignore == ignore
    assert set(resolved.config_groups) == set(original_targets)
    assert actual._resolved_targets == {targets[0], routes.GPTQ_EXPERT_TARGET}
    for name, group in groups.items():
        assert group["targets"] is original_targets[name]
        assert group["weights"] == _GPTQ_WEIGHTS[expected_preset]
        resolved_group = resolved.config_groups[name]
        assert resolved_group.targets == original_targets[name]
        _assert_gptq_weights(resolved_group.weights, expected_preset)
        assert resolved_group.input_activations is None
        assert resolved_group.output_activations is None


@pytest.mark.parametrize("output_format,expected_preset", _GPTQ_OUTPUTS)
@pytest.mark.parametrize("targets", [None, [r"re:.*self_attn\..*"], r"re:.*mlp\..*"])
@pytest.mark.parametrize("empty_groups", [False, True])
@pytest.mark.parametrize("scheme", [None, "MXFP4A16"])
def test_automatic_scheme_routes_int4_to_experts_and_preserves_other_targets(
    output_format, expected_preset, targets, empty_groups, scheme,
):
    pytest.importorskip("qlean.modifiers.gptq.base", reason="requires the qlean[gptq] extra")
    from qlean.modifiers.gptq.base import GPTQModifier

    recipe = _gptq_recipe()
    modifier = recipe["quant_stage"]["quant_modifiers"]["GPTQModifier"]
    modifier.pop("config_groups")
    if scheme is not None:
        modifier["scheme"] = scheme
    ignore = ["lm_head", r"re:.*mlp\.shared_experts\..*"]
    modifier["ignore"] = ignore
    if targets is not None:
        modifier["targets"] = targets
    if empty_groups:
        modifier["config_groups"] = {}

    assert utils.apply_gptq_output_format(recipe, output_format) == expected_preset
    assert modifier["scheme"] == expected_preset
    assert not modifier.get("config_groups")
    if expected_preset in routes.GPTQ_INT4_PRESETS:
        expected_targets = [routes.GPTQ_EXPERT_TARGET]
    elif targets is None:
        expected_targets = ["Linear"]
    else:
        expected_targets = [targets] if isinstance(targets, str) else targets
    stored_targets = (
        expected_targets
        if expected_preset in routes.GPTQ_INT4_PRESETS or targets is None
        else targets
    )
    assert modifier["targets"] == stored_targets
    assert modifier["ignore"] is ignore
    actual = GPTQModifier(**modifier)
    resolved = actual.resolve_quantization_config()
    assert resolved.ignore == ignore
    assert actual._resolved_targets == set(expected_targets)
    group, = resolved.config_groups.values()
    assert group.targets == expected_targets
    _assert_gptq_weights(group.weights, expected_preset)
    assert group.input_activations is None
    assert group.output_activations is None


@pytest.mark.parametrize("output_format,scheme", _GPTQ_OUTPUTS)
@pytest.mark.parametrize("actorder", [None, "static", "weight", "group"])
def test_automatic_groups_preserve_explicit_weight_actorder(output_format, scheme, actorder):
    recipe = _gptq_recipe()
    modifier = recipe["quant_stage"]["quant_modifiers"]["GPTQModifier"]
    group = modifier["config_groups"]["group_0"]
    group["weights"]["actorder"] = actorder
    utils.apply_gptq_output_format(recipe, output_format)
    assert group["weights"] == {**_GPTQ_WEIGHTS[scheme], "actorder": actorder}
    assert modifier["actorder"] == "static"


@pytest.mark.parametrize("output_format,scheme", _GPTQ_OUTPUTS)
def test_automatic_group_weight_copies_are_isolated(output_format, scheme):
    recipe = _gptq_recipe()
    other_recipe = _gptq_recipe()
    modifier = recipe["quant_stage"]["quant_modifiers"]["GPTQModifier"]
    modifier["config_groups"]["other"] = copy.deepcopy(modifier["config_groups"]["group_0"])
    original_presets = copy.deepcopy(routes.GPTQ_WEIGHT_CONFIGS)
    utils.apply_gptq_output_format(recipe, output_format)
    utils.apply_gptq_output_format(other_recipe, output_format)
    modifier["config_groups"]["group_0"]["weights"]["num_bits"] = 16
    assert modifier["config_groups"]["other"]["weights"] == _GPTQ_WEIGHTS[scheme]
    other_modifier = other_recipe["quant_stage"]["quant_modifiers"]["GPTQModifier"]
    assert other_modifier["config_groups"]["group_0"]["weights"] == _GPTQ_WEIGHTS[scheme]
    assert routes.GPTQ_WEIGHT_CONFIGS == original_presets


@pytest.mark.parametrize(
    "scheme", ["MXFP4", "INT4_CHANNEL", "INT8_CHANNEL", "INT4_GROUP", "INT8_GROUP"],
)
@pytest.mark.parametrize("targets", [None, [r"re:.*self_attn\..*"], r"re:.*mlp\..*"])
@pytest.mark.parametrize("ignore", [None, ["lm_head", r"re:.*mlp\.gate$"]])
def test_named_gptq_schemes_resolve_locally_with_targets_and_ignore(
    scheme, targets, ignore, monkeypatch,
):
    pytest.importorskip("qlean.modifiers.gptq.base", reason="requires the qlean[gptq] extra")
    from qlean.modifiers.gptq.base import GPTQModifier
    import qlean.modifiers.quantization as quantization

    def forbid_ct_preset(*args, **kwargs):
        pytest.fail("GPTQ named schemes must use local weight configurations")

    monkeypatch.setattr(quantization, "is_preset_scheme", forbid_ct_preset)
    monkeypatch.setattr(quantization, "preset_name_to_scheme", forbid_ct_preset)
    kwargs = {"scheme": scheme}
    if targets is not None:
        kwargs["targets"] = targets
    if ignore is not None:
        kwargs["ignore"] = ignore
    modifier = GPTQModifier(**kwargs)
    resolved = modifier.resolve_quantization_config()
    expected_targets = (
        [routes.GPTQ_EXPERT_TARGET]
        if scheme in routes.GPTQ_INT4_PRESETS and targets is None
        else ["Linear"] if targets is None
        else [targets] if isinstance(targets, str)
        else targets
    )
    assert modifier.targets == (["Linear"] if targets is None else targets)
    assert modifier.ignore == ([] if ignore is None else ignore)
    assert resolved.ignore == modifier.ignore
    assert modifier._resolved_targets == set(expected_targets)
    group, = resolved.config_groups.values()
    assert group.targets == expected_targets
    _assert_gptq_weights(group.weights, scheme)
    assert group.weights.actorder == "static"
    assert group.input_activations is None
    assert group.output_activations is None
    assert resolved.kv_cache_scheme is None


@pytest.mark.parametrize(
    "scheme", ["MXFP4", "INT4_CHANNEL", "INT8_CHANNEL", "INT4_GROUP", "INT8_GROUP"],
)
@pytest.mark.parametrize("use_groups", [False, True])
def test_structured_gptq_configs_resolve_supported_weights(scheme, use_groups):
    pytest.importorskip("qlean.modifiers.gptq.base", reason="requires the qlean[gptq] extra")
    from qlean.modifiers.gptq.base import GPTQModifier

    targets = [r"re:.*self_attn\..*"]
    group_config = {"targets": targets, "weights": copy.deepcopy(_GPTQ_WEIGHTS[scheme])}
    kwargs = {"config_groups": {"custom": group_config}} if use_groups else {"scheme": group_config}
    modifier = GPTQModifier(**kwargs, ignore=["lm_head"])
    resolved = modifier.resolve_quantization_config()
    if use_groups:
        assert list(resolved.config_groups) == ["custom"]
    group, = resolved.config_groups.values()
    assert group.targets == targets
    _assert_gptq_weights(group.weights, scheme)
    assert modifier._resolved_targets == set(targets)
    assert resolved.ignore == ["lm_head"]


@pytest.mark.parametrize(
    "scheme", ["MXFP4", "INT4_CHANNEL", "INT8_CHANNEL", "INT4_GROUP", "INT8_GROUP"],
)
@pytest.mark.parametrize("use_groups", [False, True])
def test_gptq_resolution_copies_config_without_mutating_sources(scheme, use_groups):
    pytest.importorskip("qlean.modifiers.gptq.base", reason="requires the qlean[gptq] extra")
    from qlean.modifiers.gptq.base import GPTQModifier

    targets = [r"re:.*self_attn\..*"]
    kwargs = {"targets": targets, "ignore": ["lm_head"]}
    if use_groups:
        kwargs["config_groups"] = {
            "custom": {"targets": targets, "weights": copy.deepcopy(_GPTQ_WEIGHTS[scheme])},
        }
    else:
        kwargs["scheme"] = scheme
    original_kwargs = copy.deepcopy(kwargs)
    original_presets = copy.deepcopy(routes.GPTQ_WEIGHT_CONFIGS)
    modifier = GPTQModifier(**kwargs)
    original_modifier = copy.deepcopy(modifier.model_dump())
    resolved = modifier.resolve_quantization_config()
    assert modifier.model_dump() == original_modifier
    group, = resolved.config_groups.values()
    group.targets.append("Embedding")
    group.weights.num_bits = 16
    group.weights.actorder = None
    resolved.ignore.append("extra_ignore")
    assert modifier.model_dump() == original_modifier
    assert kwargs == original_kwargs
    assert routes.GPTQ_WEIGHT_CONFIGS == original_presets
    again = modifier.resolve_quantization_config()
    fresh_group, = again.config_groups.values()
    assert fresh_group.targets == targets
    assert fresh_group.weights.actorder == "static"
    _assert_gptq_weights(fresh_group.weights, scheme)
    assert again.ignore == ["lm_head"]
    assert modifier._resolved_targets == set(targets)


def _assert_gptq_resolver_rejects_without_mutation(kwargs, match=None):
    pytest.importorskip("qlean.modifiers.gptq.base", reason="requires the qlean[gptq] extra")
    from qlean.modifiers.gptq.base import GPTQModifier

    original = copy.deepcopy(kwargs)
    with pytest.raises(ValueError, match=match):
        modifier = GPTQModifier(**kwargs)
        snapshot = copy.deepcopy(modifier.model_dump())
        try:
            modifier.resolve_quantization_config()
        finally:
            assert modifier.model_dump() == snapshot
    assert kwargs == original


@pytest.mark.parametrize("scheme", [
    "unknown", "W4A8", "W4A16", "W8A16", "MXFP4A16", "W8A8",
    "W4A8_DYNAMIC", "FP8", "INT4", "INT8", "int4_channel",
])
def test_gptq_resolver_rejects_unknown_and_old_ct_scheme_names(scheme):
    _assert_gptq_resolver_rejects_without_mutation(
        {"scheme": scheme}, match="supported schemes",
    )


@pytest.mark.parametrize("kwargs", [{}, {"config_groups": None}, {"config_groups": {}}])
def test_gptq_resolver_requires_nonempty_config(kwargs):
    _assert_gptq_resolver_rejects_without_mutation(kwargs, match="scheme|config_groups")


@pytest.mark.parametrize(
    "scheme", ["MXFP4", "INT4_CHANNEL", "INT8_CHANNEL", "INT4_GROUP", "INT8_GROUP"],
)
@pytest.mark.parametrize("empty_groups", [False, True])
def test_gptq_resolver_rejects_scheme_and_groups_conflict(scheme, empty_groups):
    groups = {} if empty_groups else {
        "custom": {"targets": ["Linear"], "weights": copy.deepcopy(_GPTQ_WEIGHTS[scheme])},
    }
    _assert_gptq_resolver_rejects_without_mutation(
        {"scheme": scheme, "config_groups": groups}, match="scheme.*config_groups|config_groups.*scheme",
    )


@pytest.mark.parametrize("scheme,updates", [
    ("INT4_CHANNEL", {"num_bits": 3}),
    ("INT8_CHANNEL", {"num_bits": 16}),
    ("INT4_CHANNEL", {"symmetric": False}),
    ("INT8_CHANNEL", {"symmetric": False}),
    ("INT4_CHANNEL", {"dynamic": True}),
    ("INT8_CHANNEL", {"dynamic": True}),
    ("INT4_CHANNEL", {"strategy": "tensor"}),
    ("INT8_CHANNEL", {"strategy": "tensor"}),
    ("INT8_CHANNEL", {"type": "float"}),
    ("INT4_GROUP", {"group_size": 64}),
    ("INT4_GROUP", {"group_size": None}),
    ("INT8_GROUP", {"group_size": 32}),
    ("INT4_GROUP", {"num_bits": 2}),
    ("INT8_GROUP", {"num_bits": 16}),
    ("INT4_GROUP", {"symmetric": False}),
    ("INT8_GROUP", {"symmetric": False}),
    ("INT4_GROUP", {"dynamic": True}),
    ("INT8_GROUP", {"dynamic": True}),
    ("INT4_GROUP", {"strategy": "channel"}),
    ("INT8_GROUP", {"strategy": "channel"}),
    ("INT8_GROUP", {"strategy": "block"}),
    ("INT4_GROUP", {"scale_dtype": "torch.uint8"}),
    ("INT4_GROUP", {"block_structure": [128, 128]}),
    ("INT8_GROUP", {"type": "float"}),
    ("MXFP4", {"symmetric": False}),
    ("MXFP4", {"dynamic": True}),
    ("MXFP4", {"group_size": 64}),
    ("MXFP4", {"scale_dtype": "torch.float32"}),
    ("MXFP4", {"zp_dtype": "torch.int8"}),
])
def test_gptq_resolver_rejects_unsupported_structured_weights(scheme, updates):
    weights = {**_GPTQ_WEIGHTS[scheme], **updates}
    _assert_gptq_resolver_rejects_without_mutation({
        "config_groups": {"custom": {"targets": ["Linear"], "weights": weights}},
    })


@pytest.mark.parametrize("schemes", [
    ("INT4_GROUP", "INT8_CHANNEL"),
    ("INT4_CHANNEL", "INT8_GROUP"),
    ("INT4_GROUP", "INT8_GROUP"),
    ("INT4_GROUP", "MXFP4"),
    ("INT4_GROUP", "INT4_CHANNEL"),
])
def test_gptq_resolver_rejects_mixed_weight_families(schemes):
    groups = {
        f"group_{index}": {"targets": ["Linear"], "weights": copy.deepcopy(_GPTQ_WEIGHTS[scheme])}
        for index, scheme in enumerate(schemes)
    }
    _assert_gptq_resolver_rejects_without_mutation({"config_groups": groups})


@pytest.mark.parametrize("scheme", ["INT4_GROUP", "INT4_CHANNEL"])
def test_gptq_resolver_rejects_block_mixed_with_exposed_schemes(scheme):
    block_weights = {
        "num_bits": 4, "type": "int", "symmetric": True, "strategy": "block",
        "block_structure": [128, 128],
    }
    _assert_gptq_resolver_rejects_without_mutation({"config_groups": {
        "block": {"targets": ["Linear"], "weights": block_weights},
        "other": {"targets": ["Linear"], "weights": copy.deepcopy(_GPTQ_WEIGHTS[scheme])},
    }})


def test_gptq_resolver_rejects_group_without_weights():
    _assert_gptq_resolver_rejects_without_mutation({
        "config_groups": {"custom": {"targets": ["Linear"]}},
    })


@pytest.mark.parametrize(
    "scheme", ["MXFP4", "INT4_CHANNEL", "INT8_CHANNEL", "INT4_GROUP", "INT8_GROUP"],
)
@pytest.mark.parametrize("field", ["input_activations", "output_activations", "kv_cache_scheme"])
def test_gptq_resolver_rejects_activation_and_kv_quantization(scheme, field):
    group = {"targets": ["Linear"], "weights": copy.deepcopy(_GPTQ_WEIGHTS[scheme])}
    kwargs = {"config_groups": {"custom": group}}
    activation = {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "tensor"}
    if field == "kv_cache_scheme":
        kwargs[field] = activation
    else:
        group[field] = activation
    _assert_gptq_resolver_rejects_without_mutation(kwargs)


@pytest.mark.parametrize(
    "module_name,expected",
    [
        ("model.layers.0.mlp.experts.3.gate_proj", True),
        ("model.layers.0.ffn.experts.3.w1", True),
        ("model.layers.0.feed_forward.experts.3.up_proj", True),
        ("model.layers.0.block_sparse_moe.experts.3.w2", True),
        ("model.layers.0.self_attn.q_proj", False),
        ("model.layers.0.mlp.gate", False),
        ("model.layers.0.mlp.up_proj", False),
        ("model.layers.0.mlp.shared_experts.up_proj", False),
        ("model.layers.0.mlp.shared_expert.up_proj", False),
        ("model.layers.0.block_sparse_moe.gate", False),
        ("model.layers.0.mlpXexperts.3.up_proj", False),
        ("lm_head", False),
    ],
)
def test_int4_expert_target_uses_compressed_tensors_matching(module_name, expected):
    assert match_name(module_name, routes.GPTQ_EXPERT_TARGET) is expected


def test_int4_expert_target_preserves_ignore_with_real_modules():
    model = torch.nn.Module()
    model.mlp = torch.nn.Module()
    model.mlp.experts = torch.nn.ModuleList([
        torch.nn.Linear(4, 4), torch.nn.Linear(4, 4),
    ])
    model.mlp.shared_experts = torch.nn.Linear(4, 4)
    model.self_attn = torch.nn.Linear(4, 4)
    recipe = _gptq_recipe()
    modifier = recipe["quant_stage"]["quant_modifiers"]["GPTQModifier"]
    modifier["ignore"] = ["mlp.experts.1"]
    modifier["config_groups"]["group_0"]["targets"] = [routes.GPTQ_EXPERT_TARGET]
    utils.apply_gptq_output_format(recipe, "GPTQ_INT4")
    matched = list(match_named_modules(
        model, modifier["config_groups"]["group_0"]["targets"], modifier["ignore"],
    ))
    assert [name for name, _ in matched] == ["mlp.experts.0"]


def test_packaged_default_preserves_target_mxfp4_baseline():
    recipe = yaml.safe_load(
        files("qlean.configs").joinpath("gptq", "_arch", "default.yaml")
        .read_text(encoding="utf-8")
    )
    modifier = recipe["quant_stage"]["quant_modifiers"]["GPTQModifier"]
    assert modifier["block_size"] == 32
    assert modifier["dampening_frac"] == 0.001
    assert modifier["offload_hessians"] is False
    assert modifier["actorder"] == "static"
    assert modifier["scheme"] == "MXFP4"
    assert modifier["targets"] == ["Linear"]
    assert not modifier.get("config_groups")
    pytest.importorskip("qlean.modifiers.gptq.base", reason="requires the qlean[gptq] extra")
    from qlean.modifiers.gptq.base import GPTQModifier

    actual = GPTQModifier(**modifier)
    resolved = actual.resolve_quantization_config()
    group, = resolved.config_groups.values()
    assert group.targets == ["Linear"]
    _assert_gptq_weights(group.weights, "MXFP4")
    assert actual._resolved_targets == {"Linear"}


def test_datafree_mxfp4_does_not_rewrite_gptq_recipe():
    recipe = _gptq_recipe()
    original = copy.deepcopy(recipe)
    assert routes.RECIPE_DIR_MAP["MXFP4"] == "mxfp4"
    assert "MXFP4" not in routes.GPTQ_OUTPUT_PRESETS
    assert utils.apply_gptq_output_format(recipe, "MXFP4") is None
    assert recipe == original


@pytest.mark.parametrize("output_format", [name for name, _ in _GPTQ_OUTPUTS])
def test_gptq_output_requires_gptq_modifier(output_format):
    with pytest.raises(ValueError, match="GPTQModifier"):
        utils.apply_gptq_output_format({}, output_format)


@pytest.mark.parametrize("output_format", sorted(routes.GPTQ_OUTPUT_PRESETS))
def test_automatic_gptq_routing_uses_packaged_default(
    output_format, monkeypatch, tmp_path, capturing_factory,
):
    import qlean.core.auto_recipe as auto_recipe

    def forbid_engine(*args, **kwargs):
        pytest.fail("GPTQ must not use a GeneralQuantEngine-generated recipe")

    monkeypatch.setattr(auto_recipe, "build_auto_recipe", forbid_engine)
    model_args = ModelArguments(
        model_name="test-model", model_path=str(tmp_path),
        save_path=str(tmp_path / "out"), output_format=output_format,
    )
    parsed, metadata = utils.get_modifiers_from_recipe("day0", model_args)
    assert len(parsed) == 1
    modifier = parsed[0]
    assert modifier.type_name == "GPTQModifier"
    preset = routes.GPTQ_OUTPUT_PRESETS[output_format]
    expected_targets = (
        [routes.GPTQ_EXPERT_TARGET]
        if preset in routes.GPTQ_INT4_PRESETS else ["Linear"]
    )
    assert modifier.scheme == preset
    assert modifier.targets == expected_targets
    assert not getattr(modifier, "config_groups", None)
    pytest.importorskip("qlean.modifiers.gptq.base", reason="requires the qlean[gptq] extra")
    from qlean.modifiers.gptq.base import GPTQModifier

    actual = GPTQModifier(scheme=modifier.scheme, targets=modifier.targets)
    resolved = actual.resolve_quantization_config()
    group, = resolved.config_groups.values()
    assert group.targets == expected_targets
    _assert_gptq_weights(group.weights, preset)
    assert actual._resolved_targets == set(expected_targets)
    assert modifier.offload_hessians is False
    assert metadata["scheme"] == preset
    assert metadata["output_format"] == output_format
    recipe_path = Path(metadata["recipe_path"])
    assert recipe_path.name == "default.yaml"
    assert recipe_path.parent.parent.name == "gptq"


@pytest.mark.parametrize(
    "value,expected",
    [
        ("GPTQ_INT4", "GPTQ_INT4"), ("gptq_int4", "GPTQ_INT4"),
        ("GPTQ_INT8", "GPTQ_INT8"), ("gptq_int8", "GPTQ_INT8"),
        ("GPTQ_MXFP4", "GPTQ_MXFP4"), ("gptq_mxfp4", "GPTQ_MXFP4"),
        ("GPTQ_INT4_GROUP", "GPTQ_INT4_GROUP"), ("gptq_int4_group", "GPTQ_INT4_GROUP"),
        ("GPTQ_INT8_GROUP", "GPTQ_INT8_GROUP"), ("gptq_int8_group", "GPTQ_INT8_GROUP"),
        ("int8", "INT8_W8A8"), ("fp8-channel", "FP8_DYNAMIC"),
        ("mxfp4", "MXFP4"),
    ],
)
def test_cli_parses_formats_and_aliases(value, expected):
    assert parse_output_format(value) == expected


@pytest.mark.parametrize("value", _REMOVED_GPTQ_OUTPUTS + [name.lower() for name in _REMOVED_GPTQ_OUTPUTS])
def test_cli_rejects_noncanonical_gptq_output(value):
    with pytest.raises(argparse.ArgumentTypeError, match="Unsupported GPTQ output format") as exc:
        parse_output_format(value)
    for supported in ("GPTQ_INT4", "GPTQ_INT8", "GPTQ_MXFP4",
                      "GPTQ_INT4_GROUP", "GPTQ_INT8_GROUP"):
        assert supported in str(exc.value)


def test_public_routes_and_shared_validator_live_in_routes():
    for name in (
        "GPTQ_OUTPUT_PRESETS", "GPTQ_WEIGHT_CONFIGS",
        "RECIPE_DIR_MAP", "MODIFIER_REPLACEMENT", "SCHEME_MAP",
        "validate_gptq_output_format",
    ):
        assert getattr(utils, name) is getattr(routes, name)
    assert routes.GPTQ_OUTPUT_PRESETS == dict(_GPTQ_OUTPUTS)
    assert routes.GPTQ_WEIGHT_CONFIGS == _GPTQ_WEIGHTS
    for name, _ in _GPTQ_OUTPUTS:
        assert routes.RECIPE_DIR_MAP[name] == "gptq"
        assert routes.MODIFIER_REPLACEMENT[("BF16", name)] == ["GPTQModifier"]
    for name in _REMOVED_GPTQ_OUTPUTS:
        assert name not in routes.GPTQ_OUTPUT_PRESETS
        assert name not in routes.RECIPE_DIR_MAP
        assert ("BF16", name) not in routes.MODIFIER_REPLACEMENT
    assert "W4A16" not in routes.GPTQ_WEIGHT_CONFIGS


@pytest.mark.parametrize("value", [
    None, "GPTQ_INT4", "GPTQ_INT8", "GPTQ_MXFP4",
    "GPTQ_INT4_GROUP", "GPTQ_INT8_GROUP", "MXFP4",
])
def test_shared_validator_accepts_canonical_and_non_gptq_formats(value):
    assert utils.validate_gptq_output_format(value) is None


@pytest.mark.parametrize("value", _REMOVED_GPTQ_OUTPUTS + ["gptq_int8"])
def test_shared_validator_rejects_noncanonical_gptq_formats(value):
    with pytest.raises(ValueError, match="Unsupported GPTQ output format") as exc:
        utils.validate_gptq_output_format(value)
    assert "GPTQ_INT4, GPTQ_INT8, GPTQ_MXFP4" in str(exc.value)
    assert "GPTQ_INT4_GROUP" in str(exc.value) and "GPTQ_INT8_GROUP" in str(exc.value)


@pytest.mark.parametrize("output_format", _REMOVED_GPTQ_OUTPUTS)
def test_compressor_rejects_noncanonical_gptq_before_model_setup(
    output_format, monkeypatch, tmp_path,
):
    import qlean.core.qleancompressor as compressor_module

    monkeypatch.setattr(
        compressor_module, "install_dependencies",
        lambda _: pytest.fail("Invalid output format reached model setup"),
    )
    model_args = ModelArguments(
        model_name="test-model", model_path=str(tmp_path),
        save_path=str(tmp_path / "out"), output_format=output_format,
    )
    compressor = compressor_module.QLeanCompressor
    compressor._instances.pop(compressor, None)
    try:
        with pytest.raises(ValueError, match="Unsupported GPTQ output format"):
            compressor(model_args, model=object(), modifiers=[object()])
    finally:
        compressor._instances.pop(compressor, None)


def test_cli_lists_gptq_without_losing_engine_conversions(capsys):
    import qlean.cli.__main__ as cli
    from qlean.core.auto_recipe import AUTO_CONVERSIONS

    cli._print_formats()
    output = capsys.readouterr().out
    for name, _ in _GPTQ_OUTPUTS:
        assert name in output
    for name in _REMOVED_GPTQ_OUTPUTS:
        assert name not in output
    for inp in {inp for inp, _ in AUTO_CONVERSIONS}:
        row = next(line for line in output.splitlines() if line.strip().startswith(inp + " "))
        actual = {name.strip() for name in row.split(None, 1)[1].split(",")}
        expected = {out for source, out in AUTO_CONVERSIONS if source == inp}
        if inp == "BF16":
            expected |= {name for name, _ in _GPTQ_OUTPUTS}
        assert actual == expected


@pytest.mark.parametrize("output_format", [name for name, _ in _GPTQ_OUTPUTS])
def test_recipe_resolution_supports_canonical_gptq_outputs(output_format, tmp_path):
    recipe_path = utils.resolve_recipe(output_format, "test-model", str(tmp_path))
    assert recipe_path.name == "default.yaml"
    assert recipe_path.parent.parent.name == "gptq"


def test_recipe_resolution_prefers_exact_then_architecture(monkeypatch, tmp_path):
    (tmp_path / "gptq").mkdir()
    (tmp_path / "gptq" / "_arch").mkdir()
    exact = tmp_path / "gptq" / "test-model.yaml"
    arch = tmp_path / "gptq" / "_arch" / "TinyMoE.yaml"
    default = tmp_path / "gptq" / "_arch" / "default.yaml"
    for path in (exact, arch, default):
        path.write_text("{}")
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["TinyMoE"]}))
    monkeypatch.setattr(utils, "files", lambda _: tmp_path)
    assert utils.resolve_recipe("GPTQ_INT4", "test-model", str(tmp_path)) == exact
    assert utils.resolve_recipe("GPTQ_INT8", "other", str(tmp_path)) == arch
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["Unknown"]}))
    assert utils.resolve_recipe("GPTQ_MXFP4", "other", str(tmp_path)) == default


def test_missing_gptq_default_never_requests_engine_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(utils, "files", lambda _: tmp_path)
    with pytest.raises(ValueError, match="No GPTQ default recipe"):
        utils.resolve_recipe("GPTQ_INT4", "test-model", str(tmp_path))


@pytest.mark.parametrize("output_format", ["INT8_W8A8", "FP8_BLOCK", "FP8_DYNAMIC", "MXFP4", "BF16"])
def test_unmatched_non_gptq_recipe_still_requests_engine_fallback(output_format, tmp_path):
    assert utils.resolve_recipe(output_format, "unknown-model", str(tmp_path)) is None


@pytest.mark.parametrize("output_format", _REMOVED_GPTQ_OUTPUTS)
def test_noncanonical_gptq_output_rejected_before_recipe_mutation(output_format, tmp_path):
    recipe = _gptq_recipe()
    original = copy.deepcopy(recipe)
    with pytest.raises(ValueError, match="Unsupported GPTQ output format"):
        utils.apply_gptq_output_format(recipe, output_format)
    assert recipe == original
    with pytest.raises(ValueError, match="Unsupported GPTQ output format"):
        utils.resolve_recipe(output_format, "test-model", str(tmp_path))
    with pytest.raises(ValueError, match="Unsupported GPTQ output format"):
        utils.get_modifiers_from_dict_day0("BF16", output_format, recipe)


@pytest.mark.parametrize("name", ["day0", "gptq"])
@pytest.mark.parametrize("use_recipe", [False, True])
@pytest.mark.parametrize("output_format", _REMOVED_GPTQ_OUTPUTS + ["gptq_int4"])
def test_recipe_entrypoints_reject_noncanonical_gptq_output(
    name, use_recipe, output_format, tmp_path,
):
    model_args = ModelArguments(
        model_name="test-model", model_path=str(tmp_path), save_path=str(tmp_path / "out"),
        recipe=str(tmp_path / "unused.yaml") if use_recipe else None,
        output_format=output_format,
    )
    with pytest.raises(ValueError, match="Unsupported GPTQ output format"):
        utils.get_modifiers_from_recipe(name, model_args)


@pytest.mark.parametrize("name", ["day0", "gptq"])
def test_explicit_mixed_recipe_is_preserved(
    name, monkeypatch, tmp_path, capturing_factory,
):
    output_format = "GPTQ_INT4"
    recipe = _gptq_recipe()
    modifier = recipe["quant_stage"]["quant_modifiers"]["GPTQModifier"]
    modifier["ignore"] = ["lm_head", r"re:.*mlp\.shared_experts\..*"]
    modifier["config_groups"] = {
        "expertINT4": {
            "targets": [routes.GPTQ_EXPERT_TARGET],
            "weights": {"num_bits": 4, "type": "int", "symmetric": True, "strategy": "channel"},
        },
        "channelINT8": {
            "targets": [r"re:.*self_attn\..*"],
            "weights": {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "channel"},
        },
    }
    recipe_path = tmp_path / "mixed.yaml"
    recipe_path.write_text(yaml.safe_dump(recipe))

    def forbid_rewrite(*args):
        pytest.fail("Explicit recipes must not be automatically rewritten")

    monkeypatch.setattr(utils, "apply_gptq_output_format", forbid_rewrite)
    monkeypatch.setattr(utils, "resolve_recipe", forbid_rewrite)
    model_args = ModelArguments(
        model_name="test-model", model_path=str(tmp_path), save_path=str(tmp_path / "out"),
        recipe=str(recipe_path), output_format=output_format,
    )
    parsed, metadata = utils.get_modifiers_from_recipe(name, model_args)
    assert len(parsed) == 1
    assert parsed[0].config_groups == modifier["config_groups"]
    assert parsed[0].ignore == modifier["ignore"]
    assert metadata["recipe_content"] == recipe
    assert metadata["output_format"] == "user_specified"
    pytest.importorskip("qlean.modifiers.gptq.base", reason="requires the qlean[gptq] extra")
    from qlean.modifiers.gptq.base import GPTQModifier

    original_recipe = copy.deepcopy(recipe)
    actual = GPTQModifier(**modifier)
    resolved = actual.resolve_quantization_config()
    assert set(resolved.config_groups) == {"expertINT4", "channelINT8"}
    assert resolved.config_groups["expertINT4"].targets == [routes.GPTQ_EXPERT_TARGET]
    assert resolved.config_groups["channelINT8"].targets == [r"re:.*self_attn\..*"]
    _assert_gptq_weights(resolved.config_groups["expertINT4"].weights, "INT4_CHANNEL")
    _assert_gptq_weights(resolved.config_groups["channelINT8"].weights, "INT8_CHANNEL")
    assert resolved.ignore == modifier["ignore"]
    assert actual._resolved_targets == {routes.GPTQ_EXPERT_TARGET, r"re:.*self_attn\..*"}
    assert recipe == original_recipe


@pytest.mark.parametrize("name", ["day0", "gptq"])
@pytest.mark.parametrize("use_scheme", [False, True])
def test_explicit_grouped_int8_recipe_is_delivered_unchanged_but_resolver_rejects(
    name, use_scheme, monkeypatch, tmp_path, capturing_factory,
):
    output_format = "GPTQ_INT8"
    recipe = _gptq_recipe()
    modifier = recipe["quant_stage"]["quant_modifiers"]["GPTQModifier"]
    modifier["ignore"] = ["lm_head", r"re:.*mlp\.shared_experts\..*"]
    targets = [r"re:.*self_attn\..*"]
    if use_scheme:
        modifier.pop("config_groups")
        modifier.update(scheme="W8A16", targets=targets)
    else:
        modifier["config_groups"]["group_0"] = {
            "targets": targets,
            "weights": {
                "num_bits": 8, "type": "int", "symmetric": True,
                "strategy": "group", "group_size": 32,
            },
        }
    recipe_path = tmp_path / "grouped.yaml"
    recipe_path.write_text(yaml.safe_dump(recipe))

    def forbid_rewrite(*args):
        pytest.fail("Explicit grouped INT8 recipes must not be rewritten")

    monkeypatch.setattr(utils, "apply_gptq_output_format", forbid_rewrite)
    monkeypatch.setattr(utils, "resolve_recipe", forbid_rewrite)
    model_args = ModelArguments(
        model_name="test-model", model_path=str(tmp_path), save_path=str(tmp_path / "out"),
        recipe=str(recipe_path), output_format=output_format,
    )
    parsed, metadata = utils.get_modifiers_from_recipe(name, model_args)
    assert len(parsed) == 1
    assert metadata["recipe_content"] == recipe
    assert metadata["output_format"] == "user_specified"
    assert metadata["scheme"] is None
    assert parsed[0].ignore == modifier["ignore"]
    if use_scheme:
        assert parsed[0].scheme == "W8A16"
        assert parsed[0].targets == targets
        assert not hasattr(parsed[0], "config_groups")
    else:
        assert parsed[0].config_groups == modifier["config_groups"]
    original_recipe = copy.deepcopy(recipe)
    delivered = {
        key: value for key, value in vars(parsed[0]).items()
        if key not in ("type_name", "stage", "group")
    }
    assert delivered == modifier
    _assert_gptq_resolver_rejects_without_mutation(delivered)
    assert recipe == original_recipe
    assert metadata["recipe_content"] == original_recipe
    assert delivered == modifier


def test_generated_engine_recipe_keeps_verify_injection(tmp_path, capturing_factory):
    model_args = ModelArguments(
        model_name="unknown-model", model_path=str(tmp_path),
        save_path=str(tmp_path / "out"), input_format="BF16",
        output_format="FP8_DYNAMIC", verify="bound",
    )
    parsed, metadata = utils.get_modifiers_from_recipe("day0", model_args)
    assert parsed[0].type_name == "GeneralQuantEngineModifier"
    assert parsed[0].primary_scheme == "FP8_DYNAMIC"
    assert parsed[0].verify == "bound"
    assert metadata["recipe_path"] == "generated:GeneralQuantEngineModifier"
    assert metadata["scheme"] == "FP8_DYNAMIC"


@pytest.mark.parametrize("name", ["day0", "engine"])
def test_explicit_engine_recipe_keeps_verify_injection(name, tmp_path, capturing_factory):
    recipe = {
        "quant_stage": {"quant_modifiers": {
            "GeneralQuantEngineModifier": {"verify": "off"},
        }},
    }
    recipe_path = tmp_path / "engine.yaml"
    recipe_path.write_text(yaml.safe_dump(recipe))
    model_args = ModelArguments(
        model_name="unknown-model", model_path=str(tmp_path),
        save_path=str(tmp_path / "out"), recipe=str(recipe_path), verify="bound",
    )
    parsed, _ = utils.get_modifiers_from_recipe(name, model_args)
    assert parsed[0].verify == "bound"


def test_verify_on_gptq_warns_without_injecting_invalid_field(tmp_path, capturing_factory, caplog):
    model_args = ModelArguments(
        model_name="test-model", model_path=str(tmp_path),
        save_path=str(tmp_path / "out"), output_format="GPTQ_INT4", verify="bound",
    )
    parsed, _ = utils.get_modifiers_from_recipe("day0", model_args)
    assert not hasattr(parsed[0], "verify")
    assert "recipe has no GeneralQuantEngineModifier; ignoring" in caplog.text


@pytest.mark.parametrize("new_value", ["FP8_DYNAMIC", "MXFP4"])
def test_engine_scheme_update_keeps_primary_scheme_and_dst_rules(new_value):
    recipe = {
        "scheme": "W8A8", "primary_scheme": "W8A8",
        "regions": [
            {"dst": "W8A8"}, {"dst": "bf16"}, {"dst": "passthrough"},
        ],
    }
    utils.update_scheme(recipe, new_value)
    assert recipe["scheme"] == new_value
    assert recipe["primary_scheme"] == new_value
    assert recipe["regions"] == [
        {"dst": "W8A8" if new_value == "MXFP4" else new_value},
        {"dst": "bf16"}, {"dst": "passthrough"},
    ]


@pytest.mark.parametrize("value", ["GPTQ_INT4", "gptq_int4"])
def test_cli_passes_canonical_output_and_verify_lazily(monkeypatch, value):
    import qlean.cli.__main__ as cli
    import qlean.entrypoints as entrypoints

    captured = []
    monkeypatch.setattr(sys, "argv", [
        "qlean", "--model_name", "test-model", "--model_path", "source",
        "--save_path", "output", "--output-format", value, "--verify", "bound",
    ])
    monkeypatch.setattr(entrypoints, "post_train", lambda *args: captured.append(args))
    cli.main()
    assert len(captured) == 1
    assert captured[0][0].output_format == "GPTQ_INT4"
    assert captured[0][0].verify == "bound"
    assert captured[0][0].max_shard_size == "10GB"


def test_cli_help_and_parser_do_not_import_heavy_modules():
    import qlean.cli.__main__ as cli

    # Isolate the CLI module from the existing eager qlean package initializer.
    # This checks module-level laziness, not package-level __init__ behavior.
    script = r'''
import importlib.abc
import runpy
import sys
import types
from pathlib import Path

cli_path = Path(sys.argv[1])
package_path = cli_path.parents[1]
package = types.ModuleType("qlean")
package.__path__ = [str(package_path)]
package.__file__ = str(package_path / "__init__.py")
sys.modules["qlean"] = package

class RejectHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        forbidden = ("torch", "compressed_tensors", "safetensors", "qlean.core.utils",
                     "qlean.core.qleancompressor", "qlean.modifiers", "qlean.entrypoints")
        if any(fullname == name or fullname.startswith(name + ".") for name in forbidden):
            raise AssertionError("Unexpected heavy import: " + fullname)

sys.meta_path.insert(0, RejectHeavyImports())
namespace = runpy.run_path(str(cli_path))
assert namespace["parse_output_format"]("gptq_int4") == "GPTQ_INT4"
sys.argv = ["qlean", "--help"]
try:
    namespace["main"]()
except SystemExit as error:
    assert error.code == 0
else:
    raise AssertionError("--help must exit")
'''
    result = subprocess.run(
        [sys.executable, "-c", script, cli.__file__],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--verify" in result.stdout
    assert "--max-shard-size" not in result.stdout
    assert "--num-calibration-rounds" not in result.stdout


@pytest.mark.parametrize("scheme,output_format", [
    (None, "GPTQ_MXFP4"), ("INT4_CHANNEL", "GPTQ_INT4"),
    ("INT8_CHANNEL", "GPTQ_INT8"), ("MXFP4", "GPTQ_MXFP4"),
    ("INT4_GROUP", "GPTQ_INT4_GROUP"), ("INT8_GROUP", "GPTQ_INT8_GROUP"),
])
def test_gptq_example_preserves_defaults_and_named_modes(monkeypatch, scheme, output_format):
    example = Path(__file__).parents[2] / "examples" / "gptq_quantization.py"
    quantize = runpy.run_path(str(example))["quantize_model"]
    namespace = quantize.__globals__
    captured = {}
    model = object()

    class LoadingClass:
        @staticmethod
        def from_pretrained(path, **kwargs):
            captured["loading"] = (path, kwargs)
            return model

    def loading_context(*, model_cls):
        captured["model_cls"] = model_cls
        return nullcontext()

    def tokenize(text, **kwargs):
        captured["tokenization"] = kwargs
        return {"input_ids": [[1, 2]]}

    class Tokenizer:
        pad_token = None
        eos_token = "eos"

        def __call__(self, text, **kwargs):
            return tokenize(text, **kwargs)

        def save_pretrained(self, directory):
            captured["tokenizer_saved_to"] = directory

    tokenizer = Tokenizer()

    class Dataset:
        column_names = ["text"]

        def filter(self, predicate):
            assert predicate({"text": "sample"}) and not predicate({"text": " "})
            return self

        def shuffle(self, *, seed):
            assert seed == 42
            return self

        def map(self, tokenize_batch, *, batched, remove_columns):
            assert batched and remove_columns == ["text"]
            tokenize_batch({"text": ["sample"]})
            return self

    dataset = Dataset()

    class Compressor:
        def __init__(self, args, *, model, modifiers):
            captured["args"] = args
            captured["model"] = model
            captured["modifier"] = modifiers[0]

        def set_calibration_dataset(self, args):
            assert args.dataset is dataset
            captured["dataset_set"] = True

        def post_train(self, args):
            assert captured["dataset_set"]
            captured["post_train"] = True

    monkeypatch.setitem(namespace, "temporary_offload_folder", lambda: nullcontext("offload"))
    monkeypatch.setitem(namespace, "load_context", loading_context)
    monkeypatch.setitem(namespace, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda path: tokenizer))

    def loading_dataset(*args, **kwargs):
        captured["dataset_loading"] = (args, kwargs)
        return dataset

    monkeypatch.setitem(namespace, "load_dataset", loading_dataset)
    monkeypatch.setitem(namespace, "GPTQModifier", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setitem(namespace, "QLeanCompressor", Compressor)
    kwargs = {"model_cls": LoadingClass}
    if scheme is not None:
        kwargs["scheme"] = scheme
    if scheme == "INT8_CHANNEL":
        kwargs["model_name"] = "test-model"
    quantize("source", "output", **kwargs)

    assert captured["model_cls"] is LoadingClass and captured["model"] is model
    assert captured["loading"] == ("source", {
        "torch_dtype": torch.bfloat16, "device_map": "auto_offload", "offload_folder": "offload",
    })
    assert tokenizer.pad_token == tokenizer.eos_token
    assert captured["tokenization"] == {"padding": False, "truncation": True, "max_length": 2048}
    assert captured["dataset_loading"] == (
        ("HuggingFaceH4/ultrachat_200k",), {"split": "train_sft[0:128]"},
    )
    assert captured["args"].model_name == (
        "test-model" if scheme == "INT8_CHANNEL" else "Qwen/Qwen3.8-27B"
    )
    assert captured["args"].model_path == "source"
    assert captured["args"].save_path == "output"
    assert captured["args"].input_format == "BF16"
    assert captured["args"].output_format == output_format and captured["post_train"]
    modifier = captured["modifier"]
    assert modifier.scheme == (scheme or "MXFP4")
    assert modifier.targets == (
        [routes.GPTQ_EXPERT_TARGET]
        if modifier.scheme in routes.GPTQ_INT4_PRESETS else ["Linear"]
    )
    assert not getattr(modifier, "config_groups", None)
    assert modifier.block_size == 128
    assert modifier.dampening_frac == 0.001
    assert modifier.actorder == "static"
    assert modifier.offload_hessians is False
    assert modifier.ignore == [
        "lm_head", r"re:.*lm_head", r"re:.*embed_tokens", r"re:visual.*",
        r"re:model.visual.*", r"re:.*conv1d$", r"re:.*in_proj_a$", r"re:.*in_proj_b$",
        r"re:.*fc$", r"re:.*pre_fc_norm_embedding$", r"re:.*pre_fc_norm_hidden$",
        r"re:.*mtp.*", r"re:(?:.*\.)?gate$", r"re:(?:.*\.)?router$",
        r"re:(?:.*\.)?shared_expert_gate$",
    ]
    for name in ["model.visual.proj", "model.layers.0.linear_attn.in_proj_a",
                 "model.layers.0.linear_attn.in_proj_b", "model.layers.0.mlp.gate",
                 "model.layers.0.mlp.router", "model.layers.0.mlp.shared_expert_gate",
                 "mtp.layers.0.fc", "lm_head"]:
        assert any(match_name(name, pattern) for pattern in modifier.ignore), name
    for name in ["model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj",
                 "model.layers.0.self_attn.q_proj"]:
        assert not any(match_name(name, pattern) for pattern in modifier.ignore), name


@pytest.mark.parametrize("scheme", ["unknown", "W4A8", "W4A16", "W8A16", "MXFP4A16"])
def test_gptq_example_rejects_invalid_scheme_before_loading(monkeypatch, capsys, scheme):
    example = Path(__file__).parents[2] / "examples" / "gptq_quantization.py"
    quantize = runpy.run_path(str(example))["quantize_model"]
    monkeypatch.setitem(
        quantize.__globals__, "temporary_offload_folder", lambda: pytest.fail("Loading started"),
    )
    with pytest.raises(ValueError, match="supported schemes"):
        quantize("source", "output", scheme=scheme)
    monkeypatch.setattr(sys, "argv", [str(example), "source", "output", "--scheme", scheme])
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(example), run_name="__main__")
    assert raised.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_gptq_example_cli_help_lists_all_schemes_without_legacy_flags(monkeypatch, capsys):
    example = Path(__file__).parents[2] / "examples" / "gptq_quantization.py"
    monkeypatch.setattr(sys, "argv", [str(example), "--help"])
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(example), run_name="__main__")
    assert raised.value.code == 0
    output = capsys.readouterr().out
    assert "--scheme" in output
    for scheme in ("MXFP4", "INT4_CHANNEL", "INT8_CHANNEL", "INT4_GROUP", "INT8_GROUP"):
        assert scheme in output
    assert "--dense-int4" not in output
    assert "--channel-int8" not in output
