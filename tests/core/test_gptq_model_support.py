# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CLI support for the models covered by the reference GPTQ torchrun scripts."""

import json
from argparse import Namespace
from types import SimpleNamespace

import pytest

pytest.importorskip("qlean.modifiers.gptq.base", reason="requires the qlean[gptq] extra")

import yaml

from qlean.args import DatasetArguments
from qlean.core import utils
from qlean.models.pretrained import model_registry

pytestmark = [pytest.mark.gptq, pytest.mark.unit, pytest.mark.cpu]


# (model_name, output_format, recipe file, scheme, ignore list from the script)
REFERENCE_MODELS = [
    (
        "Qwen3.8-27B", "GPTQ_MXFP4", "Qwen3.8-27B.yaml", "MXFP4",
        [r"re:.*lm_head", r"re:.*conv1d$", r"re:.*in_proj_a$", r"re:.*in_proj_b$",
         r"re:.*mtp.*", r"re:model\.visual\..*", r"re:visual\..*"],
    ),
    (
        "Qwen3.5-397B-A17B", "GPTQ_INT8", "Qwen3.5-397B-A17B.yaml", "INT8_CHANNEL",
        ["re:.*lm_head", "re:visual.*", "re:model.visual.*", "re:.*mlp.gate$",
         "re:.*embed_tokens$", "re:.*shared_expert_gate$", "re:.*linear_attn.*"],
    ),
    (
        "GLM-5.2", "GPTQ_MXFP4", "GLM-5.2.yaml", "MXFP4",
        [r"re:^model\.layers\.[0-2]\..*", r"re:.*mlp\.gate.*",
         r"re:.*indexer\.weights_proj$", "lm_head", r"re:.*self_attn.*",
         r"re:.*eh_proj$"],
    ),
    (
        "GLM-5.3-Flash-BF16", "GPTQ_INT8_GROUP", "GLM-5.3-Flash-BF16.yaml", "INT8_GROUP",
        [r"re:.*visual.*", "lm_head", r"re:.*mlp\.gate$", r"re:.*shared_experts\..*",
         r"re:.*self_attn\..*", r"re:.*mlp\.(down|gate|up)_proj$",
         r"re:.*fused_qkv_a_proj_with_mqa"],
    ),
    (
        "DeepSeek-V3.2-BF16", "GPTQ_MXFP4", "DeepSeek-V3.2-BF16.yaml", "MXFP4",
        [r"re:.*lm_head$", r"re:.*block_sparse_moe\.gate$", r"re:.*embed_tokens.*",
         r"re:.*input_layernorm.*", r"re:.*post_attention_layernorm.*",
         r"re:.*mlp\.down_proj.*", r"re:.*mlp\.up_proj.*", r"re:.*mlp\.gate_proj.*",
         r"re:.*self_attn.*", r"re:.*shared_experts.*"],
    ),
]


@pytest.mark.parametrize("model_name,output_format,recipe_file,scheme,ignore", REFERENCE_MODELS)
@pytest.mark.parametrize("name_form", ["basename", "full_path"])
def test_reference_model_resolves_its_script_recipe(
    tmp_path, model_name, output_format, recipe_file, scheme, ignore, name_form,
):
    name = f"/checkpoints/{model_name}" if name_form == "full_path" else model_name
    path = utils.resolve_recipe(output_format, name, str(tmp_path))
    assert path is not None and path.name == recipe_file
    modifier = yaml.safe_load(path.read_text())["quant_stage"]["quant_modifiers"]["GPTQModifier"]
    assert modifier["scheme"] == scheme
    assert modifier["targets"] == ["Linear"]
    assert modifier["ignore"] == ignore
    assert modifier["dampening_frac"] == 0.001
    assert modifier["actorder"] == "static"
    assert modifier["offload_hessians"] is False


@pytest.mark.parametrize("model_name,expected", [
    ("GLM-5.3-Flash-BF16", "Glm5NextForConditionalGeneration"),
    ("GLM-5.3-Flash", "Glm5NextForConditionalGeneration"),
    ("DeepSeek-V3.2-BF16", "DeepseekV32ForCausalLM"),
    ("Qwen3.5-397B-A17B", "Qwen3_5MoeForConditionalGeneration"),
    ("Qwen3.8-27B", "Qwen3_5ForConditionalGeneration"),
])
def test_reference_model_resolves_its_loading_class(model_name, expected):
    cls = model_registry.resolve_model_cls(model_name, f"/checkpoints/{model_name}")
    assert cls.__name__ == expected
    if expected == "DeepseekV32ForCausalLM":
        assert cls.__module__ == "qlean.modeling.deepseekv32.model"
    else:
        assert cls.__module__.startswith("transformers")


@pytest.mark.parametrize("model_name,expected", [
    ("DeepSeek-V3.2-BF16", {"cpu_offload_gb": 500.0, "num_calibration_rounds": 2,
                            "calibration_padding": "max_length"}),
    ("GLM-5.3-Flash-BF16", {"cpu_offload_gb": 500.0, "num_calibration_rounds": 2,
                            "gptq_export_workers": 8}),
    ("Qwen3.5-397B-A17B", {"cpu_offload_gb": 500.0}),
    ("Qwen3.8-27B", {}),
    ("GLM-5.2", {}),
])
def test_reference_model_defaults_match_its_script(model_name, expected):
    assert model_registry.resolve_model_defaults(model_name, f"/checkpoints/{model_name}") == expected


def _cli_args(model_name, **overrides):
    """Reproduce the real CLI order: resolve defaults, then build the dataclasses."""
    from qlean.cli.__main__ import _apply_model_defaults
    from qlean.utils import parse_to_dataclass

    values = {
        "model_name": model_name, "model_path": f"/checkpoints/{model_name}",
        "save_path": "/tmp/out",
        "cpu_offload_gb": None, "gptq_export_workers": None,
        "num_calibration_rounds": None, "calibration_padding": None,
    }
    values.update(overrides)
    namespace = Namespace(**values)
    _apply_model_defaults(namespace)
    model_args, dataset_args, other_args = parse_to_dataclass(namespace)
    return namespace, model_args, dataset_args, other_args


@pytest.mark.parametrize("model_name,field,expected", [
    ("DeepSeek-V3.2-BF16", "calibration_padding", "max_length"),
    ("DeepSeek-V3.2-BF16", "num_calibration_rounds", 2),
    ("GLM-5.3-Flash-BF16", "gptq_export_workers", 8),
    ("Qwen3.8-27B", "calibration_padding", "none"),
    ("Qwen3.8-27B", "num_calibration_rounds", 1),
    ("Qwen3.8-27B", "cpu_offload_gb", 300.0),
    ("Qwen3.8-27B", "gptq_export_workers", 2),
])
def test_cli_fills_unset_options_from_profile_then_global_default(model_name, field, expected):
    namespace, model_args, dataset_args, other_args = _cli_args(model_name)
    assert getattr(namespace, field) == expected
    holders = [o for o in (model_args, dataset_args, other_args) if hasattr(o, field)]
    assert holders and all(getattr(o, field) == expected for o in holders)


@pytest.mark.parametrize("model_name", ["DeepSeek-V3.2-BF16", "GLM-5.3-Flash-BF16"])
def test_cli_explicit_values_override_model_defaults(model_name):
    namespace, model_args, dataset_args, other_args = _cli_args(
        model_name, cpu_offload_gb=120.0, gptq_export_workers=3,
        num_calibration_rounds=5, calibration_padding="none",
    )
    assert model_args.cpu_offload_gb == 120.0
    assert model_args.gptq_export_workers == 3
    assert other_args.num_calibration_rounds == 5
    assert dataset_args.calibration_padding == "none"


def test_deepseek_preparation_merges_inference_config(tmp_path):
    (tmp_path / "inference").mkdir()
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "deepseek_v32", "dim": 7168}))
    (tmp_path / "inference" / "config_671B_v3.2.json").write_text(
        json.dumps({"dtype": "bfloat16", "n_routed_experts": 256, "index_n_heads": 64})
    )
    profile = model_registry.resolve_model_profile("DeepSeek-V3.2-BF16", str(tmp_path))
    assert len(profile["preparation"]) == 1
    profile["preparation"][0](SimpleNamespace(model_path=str(tmp_path)))
    merged = json.loads((tmp_path / "config.json").read_text())
    assert merged["n_routed_experts"] == 256 and merged["index_n_heads"] == 64
    assert merged["dim"] == 7168
    assert "dtype" not in merged


def test_deepseek_chat_template_fallback_only_when_missing():
    profile = model_registry.resolve_model_profile("DeepSeek-V3.2-BF16", "/checkpoints/DeepSeek-V3.2-BF16")
    quirk, = profile["tokenizer_quirk"]
    bare = SimpleNamespace(chat_template=None)
    quirk(bare)
    assert "<|user|>" in bare.chat_template and "<|assistant|>" in bare.chat_template
    existing = SimpleNamespace(chat_template="keep me")
    quirk(SimpleNamespace(tokenizer=existing))
    assert existing.chat_template == "keep me"


class _Tokenizer:
    def __init__(self):
        self.calls = []
        self.received = []

    def __call__(self, text, **kwargs):
        self.calls.append(kwargs)
        self.received.append(text)
        return {"input_ids": [1, 2], "attention_mask": [1, 1]}


class _Dataset:
    column_names = ["messages"]

    def __init__(self):
        self.rows = [{"messages": [{"role": "user", "content": "hi"}]}]

    def shuffle(self, seed):
        return self

    def map(self, fn, **kwargs):
        self.rows = [{**row, **fn(row)} for row in self.rows]
        return self


@pytest.mark.parametrize("mode,expected", [("none", False), ("max_length", "max_length")])
def test_calibration_padding_mode_is_honoured(monkeypatch, mode, expected):
    import datasets

    from qlean.data.calibration import build_calibration_dataset

    tokenizer = _Tokenizer()
    processor = SimpleNamespace(tokenizer=tokenizer, apply_chat_template=lambda m, tokenize: "rendered")
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: _Dataset())
    args = DatasetArguments(dataset_name="local", calibration_padding=mode, max_sequence_length=512)
    build_calibration_dataset(args, processor)
    assert tokenizer.calls and tokenizer.calls[0]["padding"] is expected
    assert tokenizer.calls[0]["max_length"] == 512
    assert tokenizer.calls[0]["truncation"] is True
    assert tokenizer.calls[0]["add_special_tokens"] is False


class _TextDataset:
    """Dataset carrying a custom text column and no messages column."""

    column_names = ["content"]

    def __init__(self):
        self.rows = [{"content": "custom text"}]

    def shuffle(self, seed):
        return self

    def map(self, fn, **kwargs):
        self.rows = [{**row, **fn(row)} for row in self.rows]
        return self


def test_calibration_uses_the_configured_text_column(monkeypatch):
    import datasets

    from qlean.data.calibration import build_calibration_dataset

    tokenizer = _Tokenizer()
    processor = SimpleNamespace(tokenizer=tokenizer, apply_chat_template=lambda m, tokenize: "rendered")
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: _TextDataset())
    args = DatasetArguments(dataset_name="local", text_column="content")
    build_calibration_dataset(args, processor)
    assert tokenizer.received == ["custom text"]


def test_calibration_messages_branch_ignores_custom_text_column(monkeypatch):
    import datasets

    from qlean.data.calibration import build_calibration_dataset

    tokenizer = _Tokenizer()
    processor = SimpleNamespace(tokenizer=tokenizer, apply_chat_template=lambda m, tokenize: "rendered")
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: _Dataset())
    # render() emits "text"; a custom text_column must not redirect the lookup.
    args = DatasetArguments(dataset_name="local", text_column="content")
    build_calibration_dataset(args, processor)
    assert tokenizer.received == ["rendered"]


def test_calibration_rejects_dataset_with_neither_column(monkeypatch):
    import datasets

    from qlean.data.calibration import build_calibration_dataset

    class _Empty(_TextDataset):
        column_names = ["unrelated"]

        def __init__(self):
            self.rows = [{"unrelated": "x"}]

    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: _Empty())
    args = DatasetArguments(dataset_name="local", text_column="content")
    with pytest.raises(ValueError, match="neither a 'messages' nor a 'content' column"):
        build_calibration_dataset(args, SimpleNamespace(tokenizer=_Tokenizer()))


def test_preparation_error_propagates_without_distribution():
    from qlean.models.pretrained.loader import _run_preparations

    def boom(model_args):
        raise ValueError("prepare failed")

    with pytest.raises(ValueError, match="prepare failed"):
        _run_preparations([boom], SimpleNamespace(model_path="/tmp/x"))


@pytest.mark.parametrize("rank", [0, 1])
def test_preparation_runs_on_source_rank_and_reports_errors_collectively(monkeypatch, rank):
    import torch.distributed as dist

    from qlean.modifiers.gptq import distributed as gptq_dist
    from qlean.models.pretrained import loader

    calls = []
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda *a, **k: rank)
    monkeypatch.setattr(dist, "barrier", lambda *a, **k: pytest.fail("must not use a bare barrier"))

    def fake_raise(error, phase, group=None):
        calls.append((phase, error))
        # The real helper raises on every rank once any rank reported a failure.
        raise RuntimeError(f"Distributed GPTQ {phase} failed")

    monkeypatch.setattr(gptq_dist, "raise_for_distributed_errors", fake_raise)

    def prepare(model_args):
        calls.append(("prepare", rank))
        raise ValueError("prepare failed")

    with pytest.raises(RuntimeError, match="checkpoint preparation failed"):
        loader._run_preparations([prepare], SimpleNamespace(model_path="/tmp/x"))
    ran_prepare = ("prepare", rank) in calls
    assert ran_prepare is (rank == 0)
    assert calls[-1][0] == "checkpoint preparation"
    if rank == 0:
        assert isinstance(calls[-1][1], ValueError)
    else:
        assert calls[-1][1] is None