# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from copy import deepcopy
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from textwrap import dedent
import time
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.cpu]
AUDIT = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'run_checks.py'), run_name='audit_under_test')
FAULTS = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'run_fault_injection.py'),
                       run_name='faults_under_test')


@pytest.fixture
def complete_record():
    nodes = ['test_a', 'test_b']
    return dict(selected=nodes, expected_selected=nodes.copy(), collected=nodes.copy(),
                deselected=[], expected_deselected=[], pytest_exitcode=0, collection_errors=[],
                reports=[dict(nodeid=node, when=phase, outcome='passed', wasxfail=None, detail='')
                         for node in nodes for phase in ('setup', 'call', 'teardown')])


@pytest.mark.skipif(not sys.platform.startswith('linux') or sys.version_info < (3, 11),
                    reason='Process cleanup requires Linux and Python >=3.11')
@pytest.mark.parametrize('session,failure', [
    pytest.param(False, 'parent-exited', id='parent-exited-conversion-group'),
    pytest.param(True, 'parent-exited', id='parent-exited-runner-session'),
    pytest.param(False, 'timeout', id='timeout-conversion-group'),
    pytest.param(True, 'timeout', id='timeout-runner-session'),
    pytest.param(False, 'exception', id='exception-conversion-group'),
    pytest.param(True, 'exception', id='exception-runner-session'),
    pytest.param(False, 'success', id='success-conversion-group'),
])
def test_cleanup_reaps_descendants_without_touching_other_groups(tmp_path, session, failure):
    ready = tmp_path / 'ready.json'
    helper = tmp_path / 'cleanup_process.py'
    helper.write_text(dedent('''\
        import json
        import os
        from pathlib import Path
        import signal
        import subprocess
        import sys
        import time

        role, ready_path, mode, new_group = sys.argv[1:]
        ready = Path(ready_path)
        if role == 'child':
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            temporary = ready.with_suffix('.tmp')
            temporary.write_text(json.dumps(dict(pid=os.getpid(), pgid=os.getpgrp(), sid=os.getsid(0))))
            temporary.replace(ready)
            time.sleep(120)
        else:
            options = {'process_group': 0} if new_group == '1' else {}
            subprocess.Popen([sys.executable, '-I', '-B', __file__, 'child', ready_path, mode, new_group],
                             **options)
            deadline = time.monotonic() + 30
            while not ready.exists():
                if time.monotonic() >= deadline:
                    raise RuntimeError('Child did not become ready')
                time.sleep(0.05)
            if mode not in ('parent-exited', 'success'):
                time.sleep(120)
        '''))
    command = [sys.executable, '-I', '-B', str(helper), 'parent', str(ready), failure, str(int(session))]
    sentinel = subprocess.Popen([sys.executable, '-I', '-B', '-c', 'import time; time.sleep(120)'],
                                start_new_session=True)
    expected = (nullcontext() if failure == 'success' else
                pytest.raises(subprocess.TimeoutExpired) if failure == 'timeout' else
                pytest.raises(RuntimeError, match='cleanup-regression'))
    try:
        with expected:
            # Allow loaded runners time to schedule and reap these real CPU processes.
            with AUDIT['managed_process'](command, session=session, grace=2) as proc:
                deadline = time.monotonic() + 30
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(0.05)
                assert ready.exists(), 'Descendant did not start'
                descendant = json.loads(ready.read_text())
                assert descendant['sid'] == (proc.pid if session else os.getsid(0))
                assert descendant['pgid'] == (descendant['pid'] if session else proc.pid)
                if failure == 'parent-exited':
                    assert proc.wait(timeout=30) == 0
                    os.kill(descendant['pid'], 0)
                    raise RuntimeError('cleanup-regression: parent already exited')
                elif failure == 'timeout':
                    proc.wait(timeout=0.1)
                elif failure == 'exception':
                    raise RuntimeError('cleanup-regression: parent still running')
                elif failure == 'success':
                    assert proc.wait(timeout=30) == 0
                    os.kill(descendant['pid'], 0)
        assert not Path(f"/proc/{descendant['pid']}").exists(), 'Descendant was not reaped'
        assert not AUDIT['owned_process_groups'](proc.pid, session)
        assert sentinel.poll() is None, 'Unrelated process was terminated'
    finally:
        sentinel.terminate()
        sentinel.wait(timeout=30)


def test_complete_execution_is_accepted(complete_record):
    assert AUDIT['execution_errors'](complete_record) == []


@pytest.mark.parametrize('matches', [True, False], ids=['fixed-environment', 'dependency-mismatch'])
def test_cpu_suite_validates_dependencies_and_selects_targets(tmp_path, monkeypatch, matches):
    root = AUDIT['ROOT']
    for name in list(sys.modules):
        if name == 'qlean' or name.startswith('qlean.'):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(root / 'src'))
    monkeypatch.setattr(AUDIT['metadata'], 'version', lambda name: '1.0')
    constraints = tmp_path / 'constraints.txt'
    constraints.write_text('pytest==1.0\n' if matches else 'pytest==2.0\n')
    calls = []

    def run_tests(options, plugins):
        calls.append(options)
        assert options[:options.index('-c')] == [str(root / 'tests')]
        audit, = plugins
        items = [SimpleNamespace(nodeid=device, get_closest_marker=lambda name, device=device:
                                 True if name in ('unit', device) else None)
                 for device in ('cpu', 'gpu')]
        config = SimpleNamespace(hook=SimpleNamespace(pytest_deselected=audit.pytest_deselected))
        session = SimpleNamespace(items=items)
        audit.pytest_collection_modifyitems(session, config, items)
        audit.pytest_collection_finish(session)
        assert [item.nodeid for item in items] == ['cpu']
        for phase in ('setup', 'call', 'teardown'):
            audit.pytest_runtest_logreport(SimpleNamespace(
                nodeid='cpu', when=phase, outcome='passed', duration=0))
        return 0

    monkeypatch.setattr(pytest, 'main', run_tests)
    args = SimpleNamespace(suite='cpu', installed=False, output=tmp_path, constraints=constraints)
    if not matches:
        with pytest.raises(RuntimeError, match='Dependency mismatch'):
            AUDIT['worker'](args)
        assert not calls
        return

    assert AUDIT['worker'](args) == 0
    assert len(calls) == 1
    result = json.loads((tmp_path / 'result.json').read_text())
    assert result['success']
    assert result['suite'] == 'cpu'
    assert not result['installed']
    assert Path(result['origin']).resolve().is_relative_to(root / 'src')
    assert result['dependency_versions'] == {'pytest': '1.0'}
    assert result['selected'] == ['cpu']
    assert result['deselected'] == ['gpu']


@pytest.mark.parametrize('count', [0, 1, 2])
def test_ppu_entry_requires_exactly_one_visible_card(count):
    if count == 1:
        AUDIT['require_single_card'](count)
    else:
        with pytest.raises(RuntimeError, match='exactly one visible device'):
            AUDIT['require_single_card'](count)


@pytest.mark.parametrize('count', [0, 1, 2])
def test_dual_ppu_entry_requires_exactly_two_visible_cards(count):
    if count == 2:
        AUDIT['require_dual_card'](count)
    else:
        with pytest.raises(RuntimeError, match='exactly two visible devices'):
            AUDIT['require_dual_card'](count)


@pytest.mark.parametrize('damage', ['missing_call', 'missing_teardown', 'duplicate', 'skip', 'xfail',
                                   'internal_error', 'empty', 'deselect', 'collection_error'])
def test_execution_audit_rejects_incomplete_or_hidden_results(complete_record, damage):
    data = deepcopy(complete_record)
    if damage in ('missing_call', 'missing_teardown'):
        data['reports'].pop(1 if damage == 'missing_call' else 2)
    elif damage == 'duplicate':
        data['reports'].append(data['reports'][1].copy())
    elif damage == 'skip':
        data['reports'][1]['outcome'] = 'skipped'
    elif damage == 'xfail':
        data['reports'][1]['wasxfail'] = 'known'
    elif damage == 'internal_error':
        data['pytest_exitcode'] = 3
    elif damage == 'empty':
        data.update(selected=[], expected_selected=[], collected=[], reports=[])
    elif damage == 'deselect':
        data['deselected'] = ['unexpected']
    else:
        data['collection_errors'] = ['import failed']
    assert AUDIT['execution_errors'](data)


@pytest.mark.parametrize('damage', [None, 'interrupted', 'wrong_node', 'wrong_phase', 'wrong_reason'])
def test_explicit_failure_requires_completed_execution_and_exact_identity(complete_record, damage):
    data = deepcopy(complete_record)
    data['pytest_exitcode'] = 1
    call = data['reports'][1]
    call.update(outcome='failed', detail='fp8e4nv not supported')
    allowed = {'test_a': 'fp8e4nv not supported'}
    if damage == 'interrupted':
        data['reports'] = data['reports'][:3]
        data['pytest_exitcode'] = 3
    elif damage == 'wrong_node':
        allowed = {'other_test': 'fp8e4nv not supported'}
    elif damage == 'wrong_phase':
        call['when'] = 'setup'
    elif damage == 'wrong_reason':
        call['detail'] = 'unknown op'
    errors = AUDIT['execution_errors'](data, allowed)
    assert bool(errors) == (damage is not None)


@pytest.fixture
def source_tree(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path / 'tests').mkdir()
    (tmp_path / 'src').mkdir()
    (tmp_path / 'tests/test_existing.py').write_text('def test_ok(): pass\n')
    (tmp_path / 'pytest.ini').write_text('[pytest]\n')
    (tmp_path / 'setup.py').write_text('version = 1\n')
    subprocess.run(['git', '-C', str(tmp_path), 'add', 'pytest.ini', 'setup.py', 'tests/test_existing.py'], check=True)
    return tmp_path


@pytest.mark.parametrize('change', ['add_test', 'remove_test', 'edit_config', 'edit_build', 'add_ci'])
def test_manifest_detects_set_and_configuration_changes(source_tree, change):
    before = AUDIT['source_manifest'](source_tree)
    if change == 'add_test':
        (source_tree / 'tests/test_new.py').write_text('def test_new(): pass\n')
    elif change == 'remove_test':
        (source_tree / 'tests/test_existing.py').rename(source_tree / 'removed.py')
    elif change == 'edit_config':
        (source_tree / 'pytest.ini').write_text('[pytest]\naddopts = -x\n')
    elif change == 'edit_build':
        (source_tree / 'setup.py').write_text('version = 2\n')
    else:
        (source_tree / 'ci').mkdir()
        (source_tree / 'ci/workflow.yml.in').write_text('jobs: {}\n')
    assert AUDIT['source_manifest'](source_tree) != before


def test_verify_record_binds_actual_installed_package(source_tree, complete_record, tmp_path_factory):
    package = tmp_path_factory.mktemp('installed_package')
    (package / '__init__.py').write_text('VERSION = 1\n')
    manifest = AUDIT['source_manifest'](source_tree)
    files = AUDIT['package_manifest'](package)
    data = dict(complete_record, schema_version=2, installed=True, success=True, audit_errors=[],
                source_manifest_before=manifest, source_manifest_after=manifest,
                package_before=files, package_after=files)
    assert AUDIT['verify_record'](data, source_tree, package) == []
    assert AUDIT['verify_record'](data, source_tree)
    (package / '__init__.py').write_text('VERSION = 2\n')
    assert AUDIT['verify_record'](data, source_tree, package)


@pytest.mark.parametrize('case_id', list(FAULTS['CASES']))
def test_fault_specs_match_source_and_existing_assertions(case_id):
    root = Path(__file__).resolve().parents[2]
    case = FAULTS['CASES'][case_id]
    text = (root / case['path']).read_text(encoding='utf-8')
    mutated = FAULTS['mutate_text'](text, case)
    assert mutated != text
    assert mutated == text.replace(case['before'], case['after'], 1)
    site = FAULTS['assertion_site'](root, case)
    assert site['file'] == case['node'].split('::')[0]
    assert site['function'] == case['function']
    assert 0 < site['start'] <= site['end']


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'unchanged', 'syntax'])
def test_fault_patch_rejects_unsafe_replacements(damage):
    text = 'def f():\n    return 1\n'
    case = dict(path='example.py', before='return 1', after='return 2')
    if damage == 'missing':
        case['before'] = 'return 3'
    elif damage == 'duplicate':
        text += 'def g():\n    return 1\n'
    elif damage == 'unchanged':
        case['after'] = case['before']
    else:
        case['after'] = 'return ('
    with pytest.raises(SyntaxError if damage == 'syntax' else ValueError):
        FAULTS['mutate_text'](text, case)


@pytest.mark.parametrize('scenario', [
    'baseline', 'baseline_failed', 'killed', 'survived', 'wrong_node', 'wrong_phase',
    'wrong_exception', 'wrong_file', 'wrong_function', 'wrong_line', 'missing_phase',
    'skip', 'xfail', 'collection_error', 'timeout', 'audit_error',
])
def test_fault_result_requires_the_expected_assertion(complete_record, scenario):
    data = deepcopy(complete_record)
    data.update(selected=['test_a'], expected_selected=['test_a'], collected=['test_a'],
                reports=data['reports'][:3], audit_errors=[], process_returncode=1, pytest_exitcode=1)
    site = dict(file='tests/test_a.py', function='test_a', start=10, end=12)
    call = data['reports'][1]
    call.update(outcome='failed', detail='AssertionError: wrong scale', exception='builtins.AssertionError',
                frames=[dict(file=site['file'], function=site['function'], line=11)])
    if scenario in ('baseline', 'survived'):
        call.update(outcome='passed', detail='', exception=None, frames=[])
        data.update(process_returncode=0, pytest_exitcode=0)
    elif scenario == 'wrong_node':
        call['nodeid'] = 'test_b'
    elif scenario == 'wrong_phase':
        call['when'] = 'setup'
    elif scenario == 'wrong_exception':
        call['exception'] = 'builtins.RuntimeError'
    elif scenario == 'wrong_file':
        call['frames'][0]['file'] = 'tests/other.py'
    elif scenario == 'wrong_function':
        call['frames'][0]['function'] = 'other_check'
    elif scenario == 'wrong_line':
        call['frames'][0]['line'] = 13
    elif scenario == 'missing_phase':
        data['reports'].pop()
    elif scenario == 'skip':
        call['outcome'] = 'skipped'
    elif scenario == 'xfail':
        call['wasxfail'] = 'known issue'
    elif scenario == 'collection_error':
        data['collection_errors'] = ['import failed']
    elif scenario == 'timeout':
        data['process_returncode'] = 124
    elif scenario == 'audit_error':
        data['audit_errors'] = ['wrong import origin']
    expected = {'baseline': 'BASELINE_PASSED', 'baseline_failed': 'BASELINE_FAILED',
                'killed': 'KILLED', 'survived': 'SURVIVED'}.get(scenario, 'ERROR')
    assert FAULTS['classify'](data, {'node': 'test_a'}, site,
                              mutated=scenario not in ('baseline', 'baseline_failed')) == expected


def test_fault_copy_does_not_modify_original_inputs(tmp_path):
    root, copied = tmp_path / 'original', tmp_path / 'copy'
    (root / 'src/qlean').mkdir(parents=True)
    (root / 'tests').mkdir()
    (root / 'pytest.ini').write_text('[pytest]\n', encoding='utf-8')
    relative = 'src/qlean/example.py'
    (root / relative).write_text('def f():\n    return 1\n', encoding='utf-8')
    (root / 'tests/test_example.py').write_text('def test_ok(): pass\n', encoding='utf-8')
    manifest = FAULTS['snapshot'](root)
    FAULTS['copy_inputs'](root, copied, manifest)
    case = dict(path=relative, before='return 1', after='return 2')
    target = copied / relative
    target.write_text(FAULTS['mutate_text'](target.read_text(encoding='utf-8'), case), encoding='utf-8')
    assert FAULTS['snapshot'](root) == manifest
    changed = FAULTS['snapshot'](copied)
    assert set(changed) == set(manifest)
    assert [name for name in manifest if changed[name] != manifest[name]] == [relative]
    with pytest.raises(FileExistsError):
        FAULTS['copy_inputs'](root, copied, manifest)


def test_fault_snapshot_rejects_symlinks(tmp_path):
    (tmp_path / 'src/qlean').mkdir(parents=True)
    (tmp_path / 'tests').mkdir()
    (tmp_path / 'pytest.ini').write_text('[pytest]\n', encoding='utf-8')
    (tmp_path / 'tests/linked.py').symlink_to(tmp_path / 'pytest.ini')
    with pytest.raises(ValueError):
        FAULTS['snapshot'](tmp_path)
