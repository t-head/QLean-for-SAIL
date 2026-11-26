# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Save synchronization regressions using real, timeout-bounded CPU Gloo ranks."""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("qlean.modifiers.gptq.base", reason="requires the qlean[gptq] extra")

import torch.distributed as dist


pytestmark = [pytest.mark.gptq, pytest.mark.integration, pytest.mark.cpu]


_GLOO_SAVE_SCRIPT = r'''
import importlib.machinery
import json
import sys
import threading
import types
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import torch
import torch._dynamo
import compressed_tensors
import transformers

class UnavailableKernel:
    def __init__(self, function):
        self.__name__ = function.__name__
    def __call__(self, *args, **kwargs):
        raise RuntimeError("Triton execution unavailable in CPU save tests")
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
sys.path.insert(0, sys.argv[1])

import importlib
import pytest
import torch.distributed as dist
import compressed_tensors.distributed as ct_distributed
accelerate_module = importlib.import_module("compressed_tensors.offload.convert.to_accelerate")
import qlean.core.qleancompressor as compressor_module
import qlean.modifiers.gptq.distributed as gptq_distributed
import qlean.modifiers.gptq.export as export
import qlean.utils.dev as dev
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from qlean.args import ModelArguments
from qlean.core import checkpoint
from qlean.core.qleancompressor import QLeanCompressor
from qlean.modifiers.gptq import GPTQModifier
from safetensors.torch import save_file
from types import SimpleNamespace

rank = int(sys.argv[2])
kind = sys.argv[4]
root = Path(sys.argv[5])
invalid_device = sys.argv[6] == "invalid-device"
invalid_workers = sys.argv[6] == "invalid-workers"
phases = ["success"] if invalid_device or invalid_workers else [
    "success", "save", "finalization", "metadata",
    *([] if kind == "INT4" else ["worker"]),
]
formats = {"INT4": "pack-quantized", "INT8": "int-quantized", "MXFP4": "mxfp4-pack-quantized"}
schemes = {"INT4": "INT4_CHANNEL", "INT8": "INT8_CHANNEL", "MXFP4": "MXFP4"}
results = []

dist.init_process_group(
    "gloo", init_method=sys.argv[3], rank=rank, world_size=2,
    timeout=timedelta(seconds=15),
)
try:
    for phase in phases:
        events = []
        groups = []
        sources = []
        error_phases = []
        lifecycle = []
        output = root / phase
        expected_format = formats[kind]
        marker = {"mtp_keys": set(), "mtp_ignore": []}
        shard_names = []
        shard_threads = []
        publish_labels = []

        def source_only(name, function, *, fail=False):
            def call(*args, **kwargs):
                events.append(name)
                assert rank == 1, "Non-source invoked " + name
                if fail:
                    raise ValueError("injected source " + phase)
                return function(*args, **kwargs)
            return call

        class SavedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Module()
                self.layer.experts = torch.nn.ModuleList()
                for index in range(2):
                    expert = torch.nn.Module()
                    expert.gate_proj = torch.nn.Linear(64, 64, bias=False)
                    if kind == "MXFP4":
                        weights = QuantizationArgs(num_bits=4, type="float", strategy="group", group_size=32)
                    else:
                        weights = QuantizationArgs(num_bits=4 if kind == "INT4" else 8,
                                                   type="int", strategy="channel", symmetric=True)
                    expert.gate_proj.quantization_scheme = QuantizationScheme(targets=["Linear"], weights=weights)
                    self.layer.experts.append(expert)

            def save_pretrained(self, directory, *, max_shard_size):
                events.append("save_pretrained")
                assert rank == 1, "Non-source wrote the CT checkpoint"
                assert max_shard_size == "2KB"
                if phase == "save":
                    raise ValueError("injected source save")
                directory = Path(directory)
                tensors = {"preserved.weight": torch.ones(4, dtype=torch.bfloat16)}
                for index in range(2):
                    name = f"layer.experts.{index}.gate_proj.weight"
                    codes = (torch.arange(4096).reshape(64, 64) % 16 - 8).to(torch.int8)
                    if kind == "INT4":
                        tensors[name + "_packed"] = pack_to_int32(codes, 4)
                        tensors[name + "_shape"] = torch.tensor([64, 64])
                        tensors[name + "_zero_point"] = torch.zeros(64, 1, dtype=torch.int8)
                    elif kind == "INT8":
                        tensors[name] = codes
                    else:
                        # Synthetic MXFP4 payload: this test exercises saving, not quantization.
                        tensors[name + "_packed"] = torch.zeros(64, 32, dtype=torch.uint8)
                    tensors[name + "_scale"] = torch.full((64, 1), 0.5, dtype=torch.bfloat16)
                if kind == "MXFP4":
                    weight_map = {}
                    for index in range(2):
                        filename = f"model-{index + 1:05d}-of-00003.safetensors"
                        prefix = f"layer.experts.{index}."
                        shard = {name: tensor for name, tensor in tensors.items() if name.startswith(prefix)}
                        save_file(shard, directory / filename)
                        weight_map.update({name: filename for name in shard})
                    filename = "model-00003-of-00003.safetensors"
                    save_file({"preserved.weight": tensors["preserved.weight"]}, directory / filename)
                    weight_map["preserved.weight"] = filename
                    (directory / "model.safetensors.index.json").write_text(json.dumps({
                        "weight_map": weight_map,
                        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors.values())},
                    }))
                else:
                    save_file(tensors, directory / "model.safetensors")
                (directory / "config.json").write_text(json.dumps({"model_type": "test-save"}))

        class SaveCompressor:
            def update_config(self, directory):
                events.append("update_config")
                assert rank == 1, "Non-source updated CT metadata"
                path = Path(directory) / "config.json"
                config = json.loads(path.read_text())
                config["quantization_config"] = {
                    "quant_method": "compressed-tensors", "format": expected_format,
                    "config_groups": {}, "ignore": [],
                }
                path.write_text(json.dumps(config))

        model = SavedModel()
        compressor = object.__new__(QLeanCompressor)
        compressor.modifiers = [GPTQModifier(scheme=schemes[kind])]
        compressor.state = SimpleNamespace(save_path=str(output), model=SimpleNamespace(model_=model))
        compressor.dataloader = None
        compressor._model_args = ModelArguments(
            model_name="gloo-save", model_path=str(root / "unused-source"), save_path=str(output),
            output_format="GPTQ_" + kind, max_shard_size="2KB",
            gptq_int4_export_device="not-a-valid-torch-device" if invalid_device else "cpu",
            gptq_export_workers=0 if invalid_workers else 2,
        )
        compressor._quant_meta = {
            "input_format": "BF16", "output_format": "before-finalization",
            "recipe_path": "<programmatic>", "recipe_content": {},
            "modifier_names": ["GPTQModifier"], "scheme": "before-finalization",
        }

        native_suspend = dev.suspend_distributed_timeout
        native_source_rank = gptq_distributed.source_rank
        native_errors = gptq_distributed.raise_for_distributed_errors
        native_finalize = export.finalize_gptq_checkpoint
        native_write_shards = export._write_staged_shards
        native_publish = export._publish_staged_checkpoint
        native_export_save_file = export.save_file

        @contextmanager
        def bounded_real_suspend():
            # The real context creates, synchronizes and destroys the temporary group.
            registered_before = set(dist.distributed_c10d._world.pg_map)
            try:
                with native_suspend(timeout=timedelta(seconds=15)) as group:
                    groups.append(group)
                    assert group is not dist.group.WORLD
                    assert dist.get_backend(group) == "gloo"
                    assert dist.get_world_size(group) == 2
                    assert group in dist.distributed_c10d._world.pg_map
                    yield group
            finally:
                assert groups[-1] not in dist.distributed_c10d._world.pg_map
                assert set(dist.distributed_c10d._world.pg_map) == registered_before
                assert dist.is_initialized()
                lifecycle.append("destroyed")

        def discover_source(group=None):
            assert group is groups[-1]
            source = native_source_rank(group)
            sources.append(source)
            return source

        def propagate(error, label, group=None):
            assert group is groups[-1]
            error_phases.append(label)
            return native_errors(error, label, group)

        def finalize(*args, **kwargs):
            events.append("finalize")
            assert kwargs["checkpoint"] is marker
            if kind == "INT4":
                assert kwargs["process_group"] is groups[-1]
                assert "workers" not in kwargs
            else:
                assert rank == 1, "Non-source finalized " + kind
                assert kwargs.get("process_group") is None
                assert kwargs["workers"] == 2
            return native_finalize(*args, **kwargs)

        def write_shards(*args, **kwargs):
            events.append("write_shards")
            if kind != "INT4":
                assert rank == 1, "Non-source exported threaded shards"
                assert kwargs.get("workers", 1) == 2
            else:
                assert kwargs.get("workers", 1) == 1
            assert kwargs["a8w4"] is (kind == "INT4")
            assert kwargs.get("label") in (
                None, {"INT4": "A8W4", "INT8": "INT8", "MXFP4": "MXFP4"}[kind],
            )
            shard_names.extend(args[7])
            return native_write_shards(*args, **kwargs)

        def track_export_save_file(*args, **kwargs):
            name = threading.current_thread().name
            shard_threads.append(name)
            if phase == "worker" and name != "MainThread":
                raise ValueError("injected source worker")
            return native_export_save_file(*args, **kwargs)

        def publish(*args, **kwargs):
            assert kwargs["label"] == {"INT4": "A8W4", "INT8": "INT8", "MXFP4": "MXFP4"}[kind]
            assert "a8w4" not in kwargs
            publish_labels.append(kwargs["label"])
            return source_only("_publish_staged_checkpoint", native_publish)(*args, **kwargs)

        def forbidden_int4(*args, **kwargs):
            events.append("forbidden_int4")
            raise AssertionError("INT8/MXFP4 resolved or entered INT4 export")

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(ct_distributed, "is_source_process", lambda: rank == 1)
            patch.setattr(gptq_distributed, "is_source_process", lambda: rank == 1)
            patch.setattr(compressor_module, "get_pipeline_from_modifiers", lambda _: lambda *args: events.append("pipeline"))
            patch.setattr(checkpoint, "compress_model_for_save", lambda *args, **kwargs: SaveCompressor())
            patch.setattr(accelerate_module, "to_accelerate", lambda model: None)
            patch.setattr(export, "prepare_source_checkpoint", lambda *args: marker)
            patch.setattr(dev, "suspend_distributed_timeout", bounded_real_suspend)
            patch.setattr(gptq_distributed, "source_rank", discover_source)
            patch.setattr(gptq_distributed, "raise_for_distributed_errors", propagate)
            patch.setattr(export, "finalize_gptq_checkpoint", finalize)
            patch.setattr(checkpoint, "save_checkpoint", source_only("save_checkpoint", checkpoint.save_checkpoint))
            patch.setattr(export, "_write_staged_shards", write_shards)
            patch.setattr(export, "save_file", track_export_save_file)
            for name in ("_write_staged_index", "_finalize_staging"):
                patch.setattr(export, name, source_only(
                    name, getattr(export, name),
                    fail=phase == "finalization" and name == "_finalize_staging",
                ))
            patch.setattr(export, "_publish_staged_checkpoint", publish)
            patch.setattr(export, "export_int8_checkpoint", source_only("export_int8", export.export_int8_checkpoint))
            patch.setattr(export, "export_mxfp4_checkpoint", source_only("export_mxfp4", export.export_mxfp4_checkpoint))
            if kind == "MXFP4":
                patch.setattr(checkpoint, "rename_packed_weights", lambda *args, **kwargs: (_ for _ in ()).throw(
                    AssertionError("MXFP4 called shared rename_packed_weights")
                ))
            else:
                patch.setattr(checkpoint, "rename_packed_weights", source_only("rename", checkpoint.rename_packed_weights))
            patch.setattr(export, "WA_for_sglang_config", source_only(
                "mxfp4_config", export.WA_for_sglang_config, fail=phase == "finalization",
            ))
            patch.setattr(checkpoint, "validate_expert_keys_in_checkpoint", source_only(
                "expert_validation", checkpoint.validate_expert_keys_in_checkpoint,
            ))
            patch.setattr(export, "finalize_checkpoint_preservation", source_only(
                "preservation", lambda *args, **kwargs: None,
            ))
            patch.setattr(compressor_module, "save_qlean_metadata", source_only(
                "metadata", compressor_module.save_qlean_metadata, fail=phase == "metadata",
            ))
            if kind != "INT4":
                patch.setattr(export, "_Int4ExportDevice", forbidden_int4)
                patch.setattr(export, "_export_a8w4_distributed", forbidden_int4)
                patch.setattr(export, "export_a8w4_checkpoint", forbidden_int4)

            try:
                compressor.post_train(SimpleNamespace())
            except Exception as error:
                outcome = type(error).__name__ + ": " + str(error)
            else:
                outcome = "ok"

        value = torch.tensor(rank + 1)
        dist.all_reduce(value)
        assert value.item() == 3
        results.append({
            "phase": phase, "outcome": outcome, "events": events, "sources": sources,
            "error_phases": error_phases, "lifecycle": lifecycle, "shards": shard_names,
            "shard_threads": shard_threads, "publish_labels": publish_labels,
            "metadata": compressor._quant_meta,
        })
    print("SAVE_RESULTS=" + json.dumps(results), flush=True)
finally:
    dist.destroy_process_group()
'''


def _run_gloo_save(tmp_path, kind, *, invalid_device=False, invalid_workers=False):
    source_root = Path(__file__).resolve().parents[2] / "src"
    rendezvous = "file://" + str(tmp_path / "gloo-save")
    workers = []
    deadline = time.monotonic() + 90
    try:
        for rank in range(2):
            workers.append(subprocess.Popen(
                [sys.executable, "-c", _GLOO_SAVE_SCRIPT, str(source_root), str(rank),
                 rendezvous, kind, str(tmp_path),
                 "invalid-device" if invalid_device else "invalid-workers" if invalid_workers else "valid"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            ))
        results = []
        for worker in workers:
            try:
                stdout, stderr = worker.communicate(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pytest.fail("CPU Gloo save workers exceeded the 90-second deadline")
            assert worker.returncode == 0, stdout + stderr
            line = next(line for line in stdout.splitlines() if line.startswith("SAVE_RESULTS="))
            results.append(json.loads(line.split("=", 1)[1]))
        return results
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
            worker.communicate()


def _assert_source_only_and_group_lifecycle(results, kind):
    source_helpers = {
        "save_checkpoint", "save_pretrained", "update_config", "metadata", "export_int8",
        "export_mxfp4", "rename", "mxfp4_config", "preservation", "expert_validation",
        "_write_staged_index", "_finalize_staging", "_publish_staged_checkpoint",
    }
    for non_source, source in zip(*results):
        assert non_source["outcome"] == source["outcome"]
        for result in (non_source, source):
            assert result["sources"] == [1]
            assert result["lifecycle"] == ["destroyed"]
            assert "forbidden_int4" not in result["events"]
            assert result["events"].count("pipeline") == 1
        assert not source_helpers.intersection(non_source["events"])
        assert source["events"].count("save_checkpoint") == 1
        assert source["events"].count("save_pretrained") == 1
        if kind != "INT4":
            assert "finalize" not in non_source["events"]
            assert "write_shards" not in non_source["events"]
            assert not non_source["shards"]
        if source["phase"] == "save":
            assert "finalize" not in source["events"]
            assert "finalize" not in non_source["events"]
        else:
            assert source["events"].count("update_config") == 1
            assert source["events"].count("finalize") == 1
            if kind == "INT4":
                assert non_source["events"].count("finalize") == 1
        assert source["events"].count("metadata") == int(source["phase"] in {"success", "metadata"})
        if source["phase"] == "success":
            assert source["metadata"]["output_format"] == "GPTQ_" + kind
            assert non_source["metadata"]["output_format"] == "before-finalization"
            assert source["publish_labels"] == [{"INT4": "A8W4", "INT8": "INT8", "MXFP4": "MXFP4"}[kind]]
            if kind == "INT4":
                assert all(name == "MainThread" for name in source["shard_threads"])
            else:
                assert any(name != "MainThread" for name in source["shard_threads"])


@pytest.mark.skipif(not dist.is_gloo_available(), reason="CPU Gloo unavailable")
@pytest.mark.parametrize("kind", ["INT4", "INT8", "MXFP4"])
def test_gptq_save_real_gloo_source_one_and_uniform_failures(tmp_path, kind):
    results = _run_gloo_save(tmp_path, kind)
    _assert_source_only_and_group_lifecycle(results, kind)
    for rank_results in results:
        assert [result["phase"] for result in rank_results] == [
            "success", "save", "finalization", "metadata", *([] if kind == "INT4" else ["worker"]),
        ]
        for result in rank_results:
            phase = result["phase"]
            if phase == "success":
                assert result["outcome"] == "ok"
            elif phase == "worker":
                assert "Distributed GPTQ checkpoint finalization failed" in result["outcome"]
                assert "injected source worker" in result["outcome"]
            else:
                label = {"save": "checkpoint save", "finalization": "checkpoint finalization",
                         "metadata": "metadata save"}[phase]
                if kind == "INT4" and phase == "finalization":
                    label = "finalization"  # The distributed exporter synchronizes this failure itself.
                assert result["outcome"] == (
                    f"RuntimeError: Distributed GPTQ {label} failed: rank 1 ValueError: injected source {phase}"
                )
            expected_phases = ["checkpoint save"]
            if phase != "save":
                if kind != "INT4" or phase != "finalization":
                    expected_phases.append("checkpoint finalization")
                if phase not in {"finalization", "worker"}:
                    expected_phases.append("metadata save")
            assert result["error_phases"] == expected_phases
    if kind == "INT4":
        non_source_shards, source_shards = (rank_results[0]["shards"] for rank_results in results)
        assert non_source_shards and source_shards
        assert not set(non_source_shards).intersection(source_shards)
        index = json.loads((tmp_path / "success" / "model.safetensors.index.json").read_text())
        assert set(index["weight_map"].values()) == set(non_source_shards + source_shards)
    for phase in ("save", "finalization", "metadata", *([] if kind == "INT4" else ["worker"])):
        assert not (tmp_path / phase / "qlean_config.json").exists()
        assert not list((tmp_path / phase).glob(".qlean-*"))
    metadata = json.loads((tmp_path / "success" / "qlean_config.json").read_text())
    assert metadata["output_format"] == "GPTQ_" + kind


@pytest.mark.skipif(not dist.is_gloo_available(), reason="CPU Gloo unavailable")
@pytest.mark.parametrize("kind", ["INT8", "MXFP4"])
def test_non_int4_save_ignores_invalid_int4_export_device(tmp_path, kind):
    results = _run_gloo_save(tmp_path, kind, invalid_device=True)
    _assert_source_only_and_group_lifecycle(results, kind)
    assert [rank_results[0]["outcome"] for rank_results in results] == ["ok", "ok"]
    config = json.loads((tmp_path / "success" / "config.json").read_text())
    assert config["quantization_config"]["quant_method"] == (
        "w8a8_int8" if kind == "INT8" else "mxfp4"
    )


@pytest.mark.skipif(not dist.is_gloo_available(), reason="CPU Gloo unavailable")
@pytest.mark.parametrize("kind", ["INT8", "MXFP4"])
def test_invalid_workers_fail_on_source_before_ct_save_without_deadlock(tmp_path, kind):
    results = _run_gloo_save(tmp_path, kind, invalid_workers=True)
    non_source, source = (rank_results[0] for rank_results in results)
    assert non_source["outcome"] == source["outcome"]
    assert "worker" in source["outcome"].lower()
    assert "positive" in source["outcome"].lower() or "integer" in source["outcome"].lower()
    assert source["sources"] == [1] and non_source["sources"] == [1]
    assert source["lifecycle"] == ["destroyed"] and non_source["lifecycle"] == ["destroyed"]
    assert "save_checkpoint" not in source["events"]
    assert "save_pretrained" not in source["events"]
    assert not (tmp_path / "success" / "model.safetensors").exists()
    assert not (tmp_path / "success" / "qlean_config.json").exists()
