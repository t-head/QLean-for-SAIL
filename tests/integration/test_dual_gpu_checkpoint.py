# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from collections import Counter
from functools import partial
import json
import multiprocessing as mp
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu, pytest.mark.multi_gpu]
AUDIT = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'run_checks.py'), run_name='dual_audit')


def _observed_process(ctx, file_name, name, weight, *, process):
    import torch

    cfg = ctx.transform_cfg
    trace = Path(cfg['_test_trace'])
    device = torch.cuda.current_device()
    assert weight.device.type == 'cuda' and weight.device.index == device
    marker = trace / f'device-{device}.ready'
    if not marker.exists():
        marker.write_text(str(os.getpid()))
        if cfg['_test_workers'] == 2:
            deadline = time.monotonic() + 60
            while not all((trace / f'device-{i}.ready').exists() for i in (0, 1)):
                if time.monotonic() >= deadline:
                    raise RuntimeError('Both workers must enter their tasks concurrently')
                time.sleep(0.05)
    produced = process(ctx, file_name, name, weight)
    torch.cuda.synchronize()
    event = dict(pid=os.getpid(), parent_pid=os.getppid(), device=device,
                 start_method=mp.get_start_method(), shard=file_name, key=name,
                 outputs=[key for key, _ in produced])
    with (trace / f'{os.getpid()}.jsonl').open('a') as handle:
        handle.write(json.dumps(event) + '\n')
    return produced


def _run_conversion(source, output, workers, qlean_root):
    sys.path.insert(0, str(qlean_root))
    import torch
    import qlean
    from qlean.modifiers_engine._runtime.plan import make_hooks
    from qlean.modifiers_engine._runtime.region_router import spec_plan_fn, spec_on_plan, validate_regions
    from qlean.modifiers_engine._runtime.run_engine import load_source_metadata, run_engine

    assert Path(qlean.__file__).resolve().parent.parent == qlean_root
    assert torch.cuda.is_available() and torch.cuda.device_count() >= workers
    assert not mp.active_children()
    trace = output.parent / (output.name + '-trace')
    trace.mkdir()
    _, weight_map = load_source_metadata(str(source))
    filenames = sorted(set(weight_map.values()))
    assert len(filenames) == 2
    regions = [
        {'name': 'quantized', 'match': ['model.proj.weight', 'model.other.weight'], 'dst': 'W8A8'},
        {'name': 'kept', 'match': '*'},
    ]
    validate_regions(regions)
    process, expected = make_hooks(spec_plan_fn, on_plan=spec_on_plan)
    cfg = {'_regions': regions, 'scale_naming': 'underscore',
           '_test_trace': str(trace), '_test_workers': workers}
    stats = run_engine(str(source), str(output), [(f, str(source / f)) for f in filenames], weight_map,
                       process_tensor=partial(_observed_process, process=process),
                       expected_output_keys=expected, transform_cfg=cfg,
                       num_gpus=workers, resume=False)
    assert not mp.active_children(), 'Engine left active worker processes behind'
    assert stats['processed_tensors'] == len(weight_map)
    assert stats['quant_count'] == 2
    result = dict(pid=os.getpid(), workers=workers, origin=qlean.__file__, stats=stats,
                  active_children=[], visible_devices=torch.cuda.device_count())
    (trace / 'summary.json').write_text(json.dumps(result, indent=2))


def _group_exists(pgid):
    while True:
        try:
            pid, _ = os.waitpid(-pgid, os.WNOHANG)
        except ChildProcessError:
            break
        if pid == 0:
            break
    return bool(AUDIT['owned_process_groups'](pgid, False))


def _launch(source, output, workers, qlean_root):
    command = [sys.executable, '-I', '-B', str(Path(__file__).resolve()),
               str(source), str(output), str(workers), str(qlean_root)]
    with (output.parent / (output.name + '.log')).open('w') as log:
        with AUDIT['managed_process'](command, session=False, stdout=log, stderr=subprocess.STDOUT) as proc:
            code = proc.wait(timeout=180)
            deadline = time.monotonic() + 10
            while _group_exists(proc.pid) and time.monotonic() < deadline:
                time.sleep(0.1)
            assert not _group_exists(proc.pid), 'Conversion left a live process group behind'
            assert code == 0, f'Conversion exited with {code}; see {output.name}.log'
    trace = output.parent / (output.name + '-trace')
    summary = json.loads((trace / 'summary.json').read_text())
    events = [json.loads(line) for path in sorted(trace.glob('*.jsonl'))
              for line in path.read_text().splitlines()]
    assert events
    return summary, events


def _read_outputs(directory, expected_map):
    from safetensors.torch import load_file

    index = json.loads((directory / 'model.safetensors.index.json').read_text())
    assert index == {'metadata': {}, 'weight_map': expected_map}
    shards = sorted(directory.glob('*.safetensors'))
    assert {p.name for p in shards} == set(expected_map.values())
    tensors, occurrences = {}, []
    for path in shards:
        payload = load_file(str(path))
        for name, tensor in payload.items():
            assert expected_map[name] == path.name
            occurrences.append(name)
            tensors[name] = tensor
    assert Counter(occurrences) == Counter(expected_map.keys()), 'Missing or duplicate output keys'
    return index, tensors


def test_two_workers_match_single_worker_bit_exact(checkpoint_factory, tmp_path):
    import torch
    import qlean

    assert torch.cuda.device_count() == 2, 'Allocate exactly two authorized visible devices'
    case = checkpoint_factory(source_format='BF16')
    qlean_root = Path(qlean.__file__).resolve().parent.parent
    single, dual = tmp_path / 'single-worker', tmp_path / 'dual-worker'
    one, one_events = _launch(case.path, single, 1, qlean_root)
    two, two_events = _launch(case.path, dual, 2, qlean_root)
    assert one['workers'] == 1 and two['workers'] == 2
    assert one['origin'] == two['origin'] == qlean.__file__
    expected_tasks = Counter((shard, key) for key, shard in case.weight_map.items())
    for events in (one_events, two_events):
        assert Counter((e['shard'], e['key']) for e in events) == expected_tasks
    assert {e['pid'] for e in one_events} == {one['pid']}
    assert {e['device'] for e in one_events} == {0}
    worker_pids = {e['pid'] for e in two_events}
    assert len(worker_pids) == 2 and two['pid'] not in worker_pids
    assert {e['parent_pid'] for e in two_events} == {two['pid']}
    assert {e['start_method'] for e in two_events} == {'spawn'}
    assert {e['device'] for e in two_events} == {0, 1}
    for pid in worker_pids:
        assigned = [e for e in two_events if e['pid'] == pid]
        assert len({e['device'] for e in assigned}) == 1
        assert len({e['shard'] for e in assigned}) == 1
    expected_map = dict(case.weight_map)
    for name in ('model.proj.weight', 'model.other.weight'):
        expected_map[name + '_scale'] = expected_map[name]
    for events in (one_events, two_events):
        assert Counter(key for e in events for key in e['outputs']) == Counter(expected_map.keys())
    index_one, tensors_one = _read_outputs(single, expected_map)
    index_two, tensors_two = _read_outputs(dual, expected_map)
    assert index_one == index_two
    index_name = 'model.safetensors.index.json'
    assert (single / index_name).read_bytes() == (dual / index_name).read_bytes()
    for name, left in tensors_one.items():
        right = tensors_two[name]
        assert left.dtype == right.dtype and left.shape == right.shape, name
        assert torch.equal(left.reshape(-1).view(torch.uint8), right.reshape(-1).view(torch.uint8)), name
    for name in ('model.proj.weight', 'model.other.weight'):
        assert tensors_one[name].dtype == torch.int8
        assert tensors_one[name].shape == case.tensors[name].shape
        assert tensors_one[name + '_scale'].shape == (case.tensors[name].shape[0], 1)
    for name in set(case.tensors) - {'model.proj.weight', 'model.other.weight'}:
        assert tensors_one[name].dtype == case.tensors[name].dtype
        assert torch.equal(tensors_one[name].reshape(-1).view(torch.uint8),
                           case.tensors[name].reshape(-1).view(torch.uint8)), name


if __name__ == '__main__':
    _run_conversion(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4]))
