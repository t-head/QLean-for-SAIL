#!/usr/bin/env python3
# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import runpy
import subprocess
import sys
import tarfile
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
AUDIT = runpy.run_path(str(ROOT / 'tests/run_checks.py'), run_name='install_audit')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def installed_check(mode, output):
    import faulthandler
    import importlib
    import importlib.metadata as metadata
    import importlib.resources as resources
    faulthandler.enable()
    import qlean
    origin = Path(qlean.__file__).resolve()
    prefix = Path(sys.prefix).resolve()
    assert sys.prefix != sys.base_prefix and origin.is_relative_to(prefix), origin
    assert 'site-packages' in origin.parts and not os.environ.get('PYTHONPATH')
    assert 'include-system-site-packages = false' in (prefix / 'pyvenv.cfg').read_text()
    manifest = json.loads((output / 'manifest.json').read_text())
    dist = metadata.distribution('qlean')
    direct = json.loads(dist.read_text('direct_url.json'))
    assert not direct.get('dir_info', {}).get('editable')
    assert direct['archive_info']['hashes']['sha256'] == manifest['wheel_sha256']
    assert AUDIT['package_manifest'](origin.parent) == manifest['files'], 'Installed files differ from the wheel manifest'
    if mode == 'origin':
        print(json.dumps(dict(qlean=str(origin), python=sys.version, sys_path=sys.path, direct_url=direct), indent=2))
    elif mode == 'core':
        for name in ['torch', 'triton', 'qlean.args', 'qlean.core.auto_recipe', 'qlean.entrypoints',
                     'qlean.modifiers', 'qlean.modifiers_engine._runtime',
                     'qlean.modifiers_engine.generalQuantEngine', 'qlean.utils.bound_check']:
            mod = importlib.import_module(name)
            assert Path(mod.__file__).resolve().is_relative_to(prefix), mod.__file__
            print(name, mod.__file__, flush=True)
    elif mode == 'resources':
        import yaml
        found = {}
        def walk(node, path=''):
            for child in node.iterdir():
                name = path + child.name
                if child.is_dir():
                    walk(child, name + '/')
                elif child.name.endswith(('.yaml', '.yml')):
                    raw = child.read_bytes()
                    yaml.safe_load(raw.decode('utf-8'))
                    found[name] = sha(raw)
        walk(resources.files('qlean'))
        assert found and found == manifest['yaml'], 'Installed YAML paths or contents differ from the source manifest'
        print('Package resources read and parsed:', len(found))


def main():
    parser = argparse.ArgumentParser(description='Build and install a wheel in isolation, check CLI/YAML, and run CPU tests.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--constraints', type=Path)
    parser.add_argument('--worker', choices=['origin', 'core', 'resources'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    out = args.output.resolve()
    if args.worker:
        installed_check(args.worker, out)
        return 0
    if not sys.platform.startswith('linux') or sys.version_info < (3, 11):
        parser.error('Installation tests require Linux and Python >=3.11 (subprocess process_group)')
    if out.is_relative_to(ROOT):
        parser.error('Output must be outside the source repository')
    out.mkdir(parents=True, exist_ok=False)
    for name in ('outside', 'tmp', 'cache', 'wheel'):
        (out / name).mkdir()
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('PYTHON', 'PIP_')) and k not in ('VIRTUAL_ENV', 'CONDA_PREFIX', 'PYTEST_ADDOPTS')}
    for key in ('PIP_INDEX_URL', 'PIP_EXTRA_INDEX_URL', 'PIP_TRUSTED_HOST', 'PIP_CERT'):
        if key in os.environ:
            env[key] = os.environ[key]
    env.update(PIP_CONFIG_FILE=os.devnull, PIP_NO_CACHE_DIR='1', PIP_NO_INPUT='1',
               PIP_DISABLE_PIP_VERSION_CHECK='1', TMPDIR=str(out / 'tmp'),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1',
               HF_HOME=str(out / 'cache/hf'), TRITON_CACHE_DIR=str(out / 'cache/triton'),
               TORCH_HOME=str(out / 'cache/torch'), XDG_CACHE_HOME=str(out / 'cache'))
    report = dict(success=False, steps=[], visibility={k: env.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'PPU_VISIBLE_DEVICES')})
    start = time.monotonic()
    def step(name, cmd, timeout=180, marker=None):
        t = time.monotonic()
        log = out / (name + '.log')
        entry = dict(name=name, command=list(map(str, cmd)), log=str(log))
        try:
            with log.open('w') as handle:
                with AUDIT['managed_process'](
                        cmd, cwd=out / 'outside', env=env, stdout=handle,
                        stderr=subprocess.STDOUT, grace=15) as proc:
                    proc.wait(timeout=timeout)
            entry['returncode'] = proc.returncode
            entry['success'] = proc.returncode == 0 and (marker is None or marker in log.read_text())
        except Exception as exc:
            entry.update(success=False, error=repr(exc))
        entry['seconds'] = round(time.monotonic() - t, 3)
        report['steps'].append(entry)
        report['seconds'] = round(time.monotonic() - start, 3)
        write(out / 'report.json', report)
        print(name, entry['success'], entry['seconds'], flush=True)
        if not entry['success'] and log.exists():
            print(f'--- {name}.log tail ---', flush=True)
            print('\n'.join(log.read_text(errors='replace').splitlines()[-20:]), flush=True)
        return entry['success']
    try:
        git = ['git', '-c', 'safe.directory=' + str(ROOT), '-C', str(ROOT)]
        report['base_sha'] = subprocess.check_output(git + ['rev-parse', 'HEAD']).decode().strip()
        report['status'] = subprocess.check_output(git + ['status', '--porcelain=v1']).decode()
        report['source_manifest_before'] = AUDIT['source_manifest'](ROOT)
        with tarfile.open(out / 'source_snapshot.tar.gz', 'w:gz') as archive:
            for name in report['source_manifest_before']:
                archive.add(ROOT / name, arcname=name, recursive=False)
        report['snapshot_sha256'] = sha((out / 'source_snapshot.tar.gz').read_bytes())
        base = ROOT / 'src/qlean'
        files = AUDIT['package_manifest'](base)
        manifest = dict(files=files, yaml={k: v for k, v in files.items() if k.endswith(('.yaml', '.yml'))},
                        base_sha=report['base_sha'])
        constraints = None
        if args.constraints:
            report['constraints'] = args.constraints.resolve().read_text()
            constraints = out / 'constraints.txt'
            constraints.write_text(report['constraints'])
        if not step('build_wheel', [sys.executable, '-I', '-B', '-m', 'pip', 'wheel', '--no-deps',
                                   '--wheel-dir', str(out / 'wheel'), str(ROOT)], 600):
            return 1
        wheel, = (out / 'wheel').glob('qlean-*.whl')
        manifest['wheel_sha256'] = sha(wheel.read_bytes())
        report.update(wheel=str(wheel), wheel_sha256=manifest['wheel_sha256'])
        with zipfile.ZipFile(wheel) as archive:
            actual = {k[6:]: sha(archive.read(k)) for k in archive.namelist()
                      if k.startswith('qlean/') and k.endswith(('.py', '.yaml', '.yml'))}
        assert actual == files, 'Wheel Python or YAML manifest mismatch'
        assert report['source_manifest_before'] == AUDIT['source_manifest'](ROOT), 'Input manifest changed during wheel build'
        write(out / 'manifest.json', manifest)
        if not step('venv', [sys.executable, '-I', '-B', '-m', 'venv', str(out / 'venv')]):
            return 1
        python = str(out / 'venv/bin/python')
        command = [python, '-I', '-B', '-m', 'pip', 'install', '--report', str(out / 'pip_install_report.json'),
                   str(wheel), 'pytest', 'setuptools']
        if constraints:
            command += ['--constraint', str(constraints)]
        if not step('install', command, 1800):
            return 1
        step('pip_check', [python, '-I', '-B', '-m', 'pip', 'check'])
        step('freeze', [python, '-I', '-B', '-m', 'pip', 'freeze', '--all'])
        for mode in ('origin', 'core', 'resources'):
            step(mode, [python, '-I', '-B', str(Path(__file__).resolve()), '--output', str(out), '--worker', mode])
        for flag, marker in [('--help', 'Required Arguments'), ('--list-formats', 'Supported conversions')]:
            step(flag[2:], [str(out / 'venv/bin/qlean'), flag], marker=marker)
        cpu_command = [python, '-I', '-B', str(ROOT / 'tests/run_checks.py'), '--suite', 'cpu',
                       '--installed', '--output', str(out / 'cpu')]
        if constraints:
            cpu_command += ['--constraints', str(constraints)]
        step('cpu', cpu_command, 930)
        report['source_manifest_after'] = AUDIT['source_manifest'](ROOT)
        assert report['source_manifest_before'] == report['source_manifest_after'], 'Input manifest changed during installation checks'
        report['success'] = all(s['success'] for s in report['steps'])
    except Exception:
        report['error'] = traceback.format_exc()
        traceback.print_exc()
    finally:
        report['seconds'] = round(time.monotonic() - start, 3)
        write(out / 'report.json', report)
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
