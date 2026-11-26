# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sequence preflight uses the same collectives even on empty/unknown ranks."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import qlean.pipelines.sequential.pipeline as pipeline


pytestmark = [pytest.mark.gptq, pytest.mark.unit, pytest.mark.cpu]


@pytest.mark.parametrize("lengths,expected", [
    ([8, 16], 16), ([16, 8], 16), ([16, 16], None), ([], None),
])
def test_local_min_max_not_first_and_max(lengths, expected):
    batches = [{"input_ids": torch.zeros(2, length)} for length in lengths]
    assert pipeline._check_calibration_sequence_length(batches) == expected


def test_embedding_sequence_axis_and_id_priority():
    assert pipeline._extract_sequence_length({"inputs_embeds": torch.zeros(2, 17, 64)}) == 17
    assert pipeline._extract_sequence_length({"input_ids": torch.zeros(2, 17)}) == 17
    assert pipeline._extract_sequence_length({
        "input_ids": torch.zeros(2, 19), "inputs_embeds": torch.zeros(2, 17, 64),
    }) == 19
    assert pipeline._extract_sequence_length({"inputs_embeds": torch.zeros(64)}) is None
    assert pipeline._check_calibration_sequence_length([
        {"inputs_embeds": torch.zeros(2, 16, 64)},
        {"inputs_embeds": torch.zeros(2, 8, 64)},
    ]) == 16


def test_unknown_batches_keep_known_bounds_and_require_conservative_prewarm():
    assert pipeline._check_calibration_sequence_length([{}, {"input_ids": torch.zeros(2, 16)}]) == 16
    assert pipeline._check_calibration_sequence_length([{}]) is None
    assert pipeline._check_calibration_sequence_length(object()) is None


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="CPU gloo unavailable")
def test_two_rank_gloo_preflight_finishes_with_mismatched_splits(tmp_path):
    # Independent subprocesses use the same import-only, fail-closed Triton shim
    # as the supplied CPU runner. No mocked collectives or GPU claims.
    script = r'''
import importlib.machinery
import json
import sys
import types
from datetime import timedelta
import torch
import torch._dynamo
import compressed_tensors
import transformers
class UnavailableKernel:
    def __init__(self, function): self.__name__ = function.__name__
    def __call__(self, *args, **kwargs): raise RuntimeError("Triton execution unavailable")
    def __getitem__(self, grid): return self
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
import qlean.pipelines.sequential.pipeline as pipeline
pipeline.get_main_device = lambda: torch.device("cpu")
rank = int(sys.argv[2])
torch.distributed.init_process_group("gloo", init_method=sys.argv[3], rank=rank,
                                     world_size=2, timeout=timedelta(seconds=15))
# Include increasing vs fixed, decreasing vs fixed, different fixed lengths,
# empty/unknown partitions, no known length anywhere, and embedding batches.
scenarios = [
    ([8, 16], [16, 16]), ([16, 8], [16, 16]), ([8, 8], [16, 16]),
    ([], [16]), ([None], [16]), ([], []), ([None], [None]),
    ([None, 8], [16]),
]
results = []
try:
    for scenario in scenarios:
        batches = [{} if length is None else {"input_ids": torch.zeros(2, length)}
                   for length in scenario[rank]]
        results.append(pipeline._check_calibration_sequence_length(batches))
    embeddings = [16, 8] if rank == 0 else [16, 16]
    results.append(pipeline._check_calibration_sequence_length([
        {"inputs_embeds": torch.zeros(2, length, 64)} for length in embeddings]))
    print("PREFLIGHT_RESULTS=" + json.dumps(results), flush=True)
finally:
    torch.distributed.destroy_process_group()
'''
    source_root = Path(pipeline.__file__).resolve().parents[3]
    rendezvous = "file://" + str(tmp_path / "gloo-preflight")
    workers = [subprocess.Popen(
        [sys.executable, "-c", script, str(source_root), str(rank), rendezvous],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ) for rank in range(2)]
    try:
        outputs = [worker.communicate(timeout=60) for worker in workers]
        for worker, (stdout, stderr) in zip(workers, outputs):
            assert worker.returncode == 0, stdout + stderr
            line = next(line for line in stdout.splitlines() if line.startswith("PREFLIGHT_RESULTS="))
            assert json.loads(line.split("=", 1)[1]) == [16, 16, 16, None, 16, None, None, 16, 16]
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.communicate()
