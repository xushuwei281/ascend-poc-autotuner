"""Freeze runtime inputs on resume without recording environment secret values."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil


def _file_identity(info):
    # Access time changes during reads; it is not an input identity.
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


class _RuntimeSnapshot:
    """Share reads only within one check, then revalidate every observed path.

    This is deliberately not a process-wide digest cache. A subsequent integrity
    check rereads all hashed contents, including already imported controller
    source and command files used by new subprocesses.
    """

    def __init__(self):
        self.files = {}
        self.paths = {}
        self.packages = {}
        for name in ('vllm', 'vllm-ascend', 'torch', 'torch-npu'):
            try:
                self.packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                self.packages[name] = None
        self.python = platform.python_version()
        self.code_root = Path(__file__).parent
        self.code_paths = tuple(sorted(self.code_root.glob('*.py')))
        self.code = {path.name: self.file(path, hash_required=True)['sha256']
                     for path in self.code_paths}

    def file(self, path, *, hash_required=False):
        path = Path(path).absolute()
        resolved = path.resolve(strict=True)
        info = resolved.stat()
        identity = _file_identity(info)
        previous = self.paths.get(path)
        if previous is not None and previous != (resolved, identity):
            raise ValueError('Runtime input changed during integrity check')
        self.paths[path] = (resolved, identity)
        cached = self.files.get(resolved)
        if cached is not None and cached['identity'] != identity:
            raise ValueError('Runtime input changed during integrity check')
        if cached is None:
            cached = {'identity': identity}
            self.files[resolved] = cached
        if hash_required or info.st_size <= 32 * 1024 * 1024:
            if 'sha256' not in cached:
                cached['sha256'] = hashlib.sha256(resolved.read_bytes()).hexdigest()
                if (_file_identity(resolved.stat()) != identity
                        or path.resolve(strict=True) != resolved):
                    raise ValueError('Runtime input changed while hashing')
            return {'sha256': cached['sha256']}
        # Preserve the existing, explicitly labelled large-command-file policy.
        # Explicit frozen_files continue to use complete content verification.
        return {'size': info.st_size, 'mtime_ns': info.st_mtime_ns,
                'note': 'large-file metadata only'}

    def verify(self):
        if tuple(sorted(self.code_root.glob('*.py'))) != self.code_paths:
            raise ValueError('Controller source inventory changed during integrity check')
        for path, (resolved, identity) in self.paths.items():
            if (path.resolve(strict=True) != resolved
                    or _file_identity(resolved.stat()) != identity):
                raise ValueError('Runtime input changed during integrity check')


def runtime_signatures(configs: dict[str, dict]) -> dict[str, dict]:
    """Fingerprint profiles with one shared read per file in this batch only.

    Returned schemas match ``runtime_signature``. File identities, path targets,
    and the controller source inventory must still match at the end of the
    batch. No digest or package cache survives this call.
    """
    if not isinstance(configs, dict):
        raise TypeError('Runtime configurations must be a mapping')
    if not configs:
        return {}
    snapshot = _RuntimeSnapshot()
    result = {name: _runtime_signature(config, snapshot) for name, config in configs.items()}
    snapshot.verify()
    return result


def runtime_signature(config: dict) -> dict:
    """Fingerprint one configuration, without caching across integrity checks."""
    return runtime_signatures({'profile': config})['profile']


def _runtime_signature(config: dict, snapshot: _RuntimeSnapshot) -> dict:
    root = Path(config.get('_repo_dir', Path.cwd())).resolve()
    encoded = json.dumps({k: v for k, v in config.items() if not k.startswith('_')})
    env_names = sorted(set(re.findall(r'\{env:([A-Za-z_][A-Za-z0-9_]*)\}', encoded)))
    key_name = config['llm']['api_key_env']
    inherited = dict(os.environ)
    inherited.pop(key_name, None)
    # Overrides expand against the original inherited environment, not against
    # themselves. In particular X='{env:X}' must fingerprint inherited X rather
    # than the literal template. Explicit templates are frozen by config_hash.
    values = {name: hashlib.sha256(inherited[name].encode()).hexdigest() if name in inherited else None
              for name in env_names if name != key_name}

    def expand(argument, environment):
        # Match the runner's one-pass expansion: replacement text must not be
        # interpreted again. Trial directories do not exist at resume checks.
        def replace(match):
            name = match[1]
            if name == 'repo_dir':
                return str(root)
            if name.startswith('env:'):
                return environment.get(name[4:], '<unset>')
            if name.startswith('param.') and name[6:] in config.get('baseline', {}):
                value = config['baseline'][name[6:]]
                return str(value).lower() if isinstance(value, bool) else str(value)
            return match[0]
        return re.sub(r'\{([A-Za-z_][A-Za-z0-9_.:]*)\}', replace, argument)

    def environment(overrides):
        result = dict(inherited)
        result.update({name: expand(value, inherited) for name, value in overrides.items()})
        result.pop(key_name, None)
        return result

    server_env = environment(config['server'].get('env', {}))
    # Old single-profile contracts retain their original shared environment.
    # Workflows provide benchmark_env explicitly to freeze customer scripts.
    benchmark_env = environment(config['benchmark_env']) if 'benchmark_env' in config else server_env
    commands = [(config['server']['command'], server_env)]
    for scenario in config['scenarios']:
        commands.extend((scenario[k], benchmark_env) for k in ['command', 'warmup_command', 'before_each_command', 'quality_command'] if k in scenario)
    files = {}
    for argv, command_env in commands:
        for index, argument in enumerate(argv):
            value = expand(argument, command_env)
            if '{' in value or '<unset>' in value or value.startswith('-'):
                continue
            # Popen runs under root and the command's own PATH. Resolve relative
            # PATH entries against that cwd, rather than the controller's cwd.
            search_path = os.pathsep.join(str(Path(part) if Path(part).is_absolute() else root / part)
                                          for part in command_env.get('PATH', os.defpath).split(os.pathsep))
            executable = shutil.which(value, path=search_path) if index == 0 and os.sep not in value else None
            path = Path(executable or value)
            if not path.is_absolute():
                path = root / path
            if path.is_file():
                file_signature = snapshot.file(path)
                files[str(path.resolve(strict=True))] = file_signature
    return {'workdir': str(root), 'referenced_env_hashes': values, 'packages': dict(snapshot.packages),
            'python': snapshot.python, 'command_files': files, 'controller_code': dict(snapshot.code)}
