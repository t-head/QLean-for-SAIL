#!/usr/bin/env python3
# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
from collections import Counter
from contextlib import contextmanager
import ctypes
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PR_SET_CHILD_SUBREAPER = 36
PR_GET_CHILD_SUBREAPER = 37


def owned_process_groups(identity, session):
    """Use session/group identity because the original parent may have exited."""
    if not Path('/proc/self/stat').is_file():
        raise RuntimeError('Process cleanup requires a mounted Linux /proc')
    groups = set()
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text(errors='replace').rsplit(')', 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        pgid, sid = int(fields[2]), int(fields[3])
        if (sid if session else pgid) == identity:
            groups.add(pgid)
    return groups


def cleanup_processes(proc, *, session=True, grace=5):
    """Escalate based on remaining descendants, not only the parent exit."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        deadline = time.monotonic() + grace
        while True:
            proc.poll()
            groups = owned_process_groups(proc.pid, session)
            for pgid in groups:
                if pgid == os.getpgrp():
                    raise RuntimeError('Refusing to signal the caller process group')
                try:
                    os.killpg(pgid, sig)
                except ProcessLookupError:
                    pass
                while True:
                    try:
                        pid, _ = os.waitpid(-pgid, os.WNOHANG)
                    except ChildProcessError:
                        break
                    if pid == 0:
                        break
            if not owned_process_groups(proc.pid, session):
                proc.wait(timeout=grace)
                return
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)
    raise RuntimeError(f'Owned processes survived cleanup: {proc.pid}')


@contextmanager
def managed_process(command, *, session=True, grace=5, **kwargs):
    """Keep conversion groups in the outer session so its cleanup owns them."""
    if not sys.platform.startswith('linux'):
        raise RuntimeError('Managed test processes require Linux')
    if not Path('/proc/self/stat').is_file():
        raise RuntimeError('Process cleanup requires a mounted Linux /proc')
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    if (libc.prctl(PR_GET_CHILD_SUBREAPER, ctypes.byref(previous), 0, 0, 0)
            or libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)):
        raise OSError(ctypes.get_errno(), 'Cannot enable child subreaper')
    old_handler = signal.getsignal(signal.SIGTERM)
    def terminated(signum, frame):
        raise SystemExit(128 + signum)
    proc = None
    try:
        signal.signal(signal.SIGTERM, terminated)
        options = {'start_new_session': True} if session else {'process_group': 0}
        proc = subprocess.Popen(command, **options, **kwargs)
        yield proc
    finally:
        try:
            if proc is not None:
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                cleanup_processes(proc, session=session, grace=grace)
        finally:
            signal.signal(signal.SIGTERM, old_handler)
            if libc.prctl(PR_SET_CHILD_SUBREAPER, previous.value, 0, 0, 0):
                raise OSError(ctypes.get_errno(), 'Cannot restore child subreaper')


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def source_manifest(root):
    root = Path(root).resolve()
    git = ['git', '-c', 'safe.directory=' + str(root), '-C', str(root)]
    names = set(filter(None, subprocess.check_output(git + ['ls-files', '-z']).decode().split('\0')))
    for directory in ('src', 'tests', 'ci', '.github'):
        for p in (root / directory).rglob('*'):
            if (p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts
                    and not any(part.endswith('.egg-info') for part in p.parts)
                    and p.suffix not in ('.pyc', '.pyo')):
                names.add(p.relative_to(root).as_posix())
    for pattern in ('*.ini', '*.toml', '*.cfg', '*.py', 'MANIFEST.in', '*requirements*.txt', '*constraints*.txt'):
        names.update(p.name for p in root.glob(pattern) if p.is_file())
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in sorted(names) if (root / name).is_file()}


def require_single_card(count):
    if count != 1:
        raise RuntimeError(f'PPU suite requires exactly one visible device; found {count}. Allocate the device explicitly.')


def require_dual_card(count):
    if count != 2:
        raise RuntimeError(f'Dual-device PPU suite requires exactly two visible devices; found {count}.')


def package_manifest(package):
    package = Path(package)
    return {p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(package.rglob('*')) if p.is_file() and p.suffix in ('.py', '.yaml', '.yml')}


def execution_errors(data, allowed_failures=None):
    allowed = allowed_failures or {}
    errors = []
    selected = data.get('selected', [])
    if not selected or len(set(selected)) != len(selected):
        errors.append('Selection is empty or contains duplicate nodes')
    if selected != data.get('expected_selected'):
        errors.append('Selected nodes differ from the expected selection')
    if Counter(data.get('deselected', [])) != Counter(data.get('expected_deselected', [])):
        errors.append('Unexpected deselected nodes')
    if Counter(selected + data.get('deselected', [])) != Counter(data.get('collected', [])):
        errors.append('Selected and deselected nodes do not partition the collection')
    reports = data.get('reports', [])
    expected = Counter((node, phase) for node in selected for phase in ('setup', 'call', 'teardown'))
    if Counter((r['nodeid'], r['when']) for r in reports) != expected:
        errors.append('Execution phases are missing, duplicated, or refer to unselected nodes')
    failures = {}
    for report in reports:
        if report.get('wasxfail') is not None or report['outcome'] == 'skipped':
            errors.append('Unexpected skip or xfail')
        elif report['outcome'] == 'failed':
            node = report['nodeid']
            failures[node] = report
            if (report['when'] != 'call' or node not in allowed
                    or not allowed[node] or allowed[node] not in report.get('detail', '')):
                errors.append('Unexpected failure: ' + node)
        elif report['outcome'] != 'passed':
            errors.append('Unknown test outcome')
    if set(failures) != set(allowed):
        errors.append('Failed nodes differ from the explicitly allowed failures')
    if data.get('pytest_exitcode') != (1 if allowed else 0):
        errors.append('pytest exit code does not match the expected completed execution')
    if data.get('collection_errors'):
        errors.append('Test collection failed')
    return errors


def verify_record(data, root, package=None):
    errors = execution_errors(data)
    errors.extend(data.get('audit_errors', []))
    if data.get('schema_version') != 2:
        errors.append('Legacy report lacks the evidence required by the current audit schema')
    current = source_manifest(root)
    if current != data.get('source_manifest_before') or current != data.get('source_manifest_after'):
        errors.append('Source, test, or configuration manifest changed')
    if data.get('installed'):
        if package is None:
            errors.append('Installed report verification requires --package-root')
        else:
            actual = package_manifest(package)
            if actual != data.get('package_before') or actual != data.get('package_after'):
                errors.append('Installed package manifest changed')
    if not data.get('success'):
        errors.append('The original run was not successful')
    return errors


def check_constraints(text):
    from packaging.requirements import Requirement
    from packaging.version import Version

    versions = {}
    for line in text.splitlines():
        line = line.split(' #', 1)[0].strip()
        if not line or line.startswith('#'):
            continue
        requirement = Requirement(line)
        if requirement.url or requirement.extras or not requirement.specifier:
            raise ValueError('Constraints must contain versioned package requirements: ' + line)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        actual = metadata.version(requirement.name)
        version = Version(actual)
        for specifier in requirement.specifier:
            matches = specifier.contains(version, prereleases=True)
            if not matches:
                raise RuntimeError(f'Dependency mismatch: {requirement.name}={actual}; required {requirement.specifier}')
        versions[requirement.name] = actual
    if not versions:
        raise ValueError('Constraints contain no applicable package requirements')
    return versions


def worker(args):
    import faulthandler
    import pytest
    faulthandler.enable()
    if not args.installed:
        sys.path.insert(0, str(ROOT / 'src'))
    import qlean
    package = Path(qlean.__file__).resolve().parent
    expected = Path(sys.prefix).resolve() if args.installed else (ROOT / 'src').resolve()
    if not package.is_relative_to(expected):
        raise RuntimeError('Unexpected QLean import origin: ' + str(package))
    if args.installed and (sys.prefix == sys.base_prefix or os.environ.get('PYTHONPATH')):
        raise RuntimeError('Installed checks require an isolated venv and an unset PYTHONPATH')
    start = time.monotonic()
    git = ['git', '-c', 'safe.directory=' + str(ROOT), '-C', str(ROOT)]
    data = dict(schema_version=2, suite=args.suite, installed=args.installed,
                base_sha=subprocess.check_output(git + ['rev-parse', 'HEAD']).decode().strip(),
                origin=qlean.__file__, expected_origin=str(expected), python=sys.version, executable=sys.executable,
                versions={n: metadata.version(n) for n in ('torch', 'triton', 'numpy', 'pytest')},
                visibility={k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'PPU_VISIBLE_DEVICES')},
                source_manifest_before=source_manifest(ROOT), package_before=package_manifest(package),
                collected=[], selected=[], deselected=[], reports=[], collection_errors=[], audit_errors=[])

    class Audit:
        @pytest.hookimpl(tryfirst=True)
        def pytest_collection_modifyitems(self, session, config, items):
            data['collected'] = [item.nodeid for item in items]
            chosen, excluded = [], []
            for item in items:
                cpu, gpu = (item.get_closest_marker(m) is not None for m in ('cpu', 'gpu'))
                layers = sum(item.get_closest_marker(m) is not None for m in ('unit', 'integration', 'e2e'))
                if cpu == gpu or layers != 1:
                    data['audit_errors'].append('Missing or conflicting test markers: ' + item.nodeid)
                multi = item.get_closest_marker('multi_gpu') is not None
                if multi and not gpu:
                    data['audit_errors'].append('multi_gpu requires the gpu marker: ' + item.nodeid)
                choose = (args.suite == 'cpu' and cpu
                          or args.suite == 'ppu' and gpu and not multi
                          or args.suite == 'ppu-dual' and gpu and multi)
                (chosen if choose else excluded).append(item)
            data['expected_selected'] = [i.nodeid for i in chosen]
            data['expected_deselected'] = [i.nodeid for i in excluded]
            items[:] = chosen
            config.hook.pytest_deselected(items=excluded)

        def pytest_deselected(self, items):
            data['deselected'].extend(i.nodeid for i in items)

        def pytest_collection_finish(self, session):
            data['selected'] = [i.nodeid for i in session.items]

        def pytest_collectreport(self, report):
            if report.failed:
                data['collection_errors'].append(str(report.longrepr))

        def pytest_runtest_logreport(self, report):
            data['reports'].append(dict(nodeid=report.nodeid, when=report.when, outcome=report.outcome,
                                        duration=report.duration, wasxfail=getattr(report, 'wasxfail', None),
                                        detail=str(report.longrepr) if report.outcome != 'passed' else ''))

    options = [str(ROOT / 'tests')]
    options += ['-c', str(ROOT / 'pytest.ini'), '-p', 'no:cacheprovider', '-ra', '--tb=short',
                '--basetemp=' + str(args.output / 'tmp'), '--junitxml=' + str(args.output / 'junit.xml')]
    if args.constraints:
        data['constraints_path'] = str(args.constraints)
        data['constraints_text'] = args.constraints.read_text()
        data['dependency_versions'] = check_constraints(data['constraints_text'])
    if args.suite in ('ppu', 'ppu-dual'):
        if sys.version_info[:2] != (3, 12):
            raise RuntimeError('PPU suite requires the validated Python 3.12 environment')
        import torch
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        (require_dual_card if args.suite == 'ppu-dual' else require_single_card)(count)
        data['hardware'] = [str(torch.cuda.get_device_properties(i)) for i in range(count)]
        options.append('--require-gpu')
    data['pytest_exitcode'] = int(pytest.main(options, plugins=[Audit()]))
    data['audit_errors'].extend(execution_errors(data))
    data['source_manifest_after'] = source_manifest(ROOT)
    data['package_after'] = package_manifest(package)
    if data['source_manifest_before'] != data['source_manifest_after']:
        data['audit_errors'].append('Input manifest changed during execution')
    if data['package_before'] != data['package_after']:
        data['audit_errors'].append('QLean package changed during execution')
    imported = {n: m.__file__ for n, m in list(sys.modules.items())
                if (n == 'qlean' or n.startswith('qlean.')) and getattr(m, '__file__', None)}
    data['imported_modules'] = imported
    for name, path in imported.items():
        if not Path(path).resolve().is_relative_to(expected):
            data['audit_errors'].append('Unexpected import origin: ' + name)
    relevant = [r for r in data['reports'] if r['when'] == 'call' or r['outcome'] != 'passed']
    data['counts'] = {key: sum(r['outcome'] == key for r in relevant) for key in ('passed', 'failed', 'skipped')}
    data['counts'].update(collected=len(data['collected']), selected=len(data['selected']), deselected=len(data['deselected']))
    data['seconds'] = round(time.monotonic() - start, 3)
    data['success'] = data['pytest_exitcode'] == 0 and not data['audit_errors']
    save(args.output / 'result.json', data)
    return 0 if data['success'] else 1


def main():
    parser = argparse.ArgumentParser(description='Run audited full CPU, single-device PPU, or minimal dual-device PPU tests.')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--suite', choices=['cpu', 'ppu', 'ppu-dual'])
    mode.add_argument('--verify-report', type=Path)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--package-root', type=Path)
    parser.add_argument('--constraints', type=Path, help='Environment constraints; required for the PPU suites')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--installed', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.verify_report:
        errors = verify_record(json.loads(args.verify_report.read_text()), args.source_root, args.package_root)
        print(json.dumps({'success': not errors, 'errors': errors}, ensure_ascii=False, indent=2))
        return 1 if errors else 0
    if not sys.platform.startswith('linux') or sys.version_info < (3, 11):
        parser.error('Test runner requires Linux and Python >=3.11 (subprocess process_group)')
    if args.suite in ('ppu', 'ppu-dual') and args.constraints is None:
        parser.error('PPU suites require --constraints')
    if args.constraints:
        args.constraints = args.constraints.resolve()
        if not args.constraints.is_file():
            parser.error('--constraints must reference an existing file')
    if args.output is None:
        parser.error('--suite requires --output')
    args.output = args.output.resolve()
    if args.output.is_relative_to(ROOT):
        parser.error('Output must be outside the source repository')
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if args.worker:
        return worker(args)
    args.output.mkdir(parents=True, exist_ok=False)
    if args.constraints:
        contents = args.constraints.read_bytes()
        args.constraints = args.output / 'constraints.txt'
        args.constraints.write_bytes(contents)
    env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHON') and k != 'PYTEST_ADDOPTS'}
    env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               OMP_NUM_THREADS='1', TRITON_CACHE_DIR=str(args.output / 'triton_cache'),
               HF_HOME=str(args.output / 'hf_cache'), TORCH_HOME=str(args.output / 'torch_cache'))
    command = [sys.executable, '-I', '-B', str(Path(__file__).resolve()), '--worker', '--suite', args.suite,
               '--output', str(args.output)] + (['--installed'] if args.installed else [])
    if args.constraints:
        command += ['--constraints', str(args.constraints)]
    start = time.monotonic()
    with (args.output / 'pytest.log').open('w') as log:
        try:
            with managed_process(command, cwd=args.output, env=env, stdout=log, stderr=subprocess.STDOUT) as proc:
                code = proc.wait(timeout=900)
        except subprocess.TimeoutExpired:
            code = 124
    save(args.output / 'execution.json', dict(command=command, returncode=code, cwd=str(args.output),
                                            seconds=round(time.monotonic() - start, 3)))
    print('Results:', args.output, 'Exit code:', code)
    return code


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
