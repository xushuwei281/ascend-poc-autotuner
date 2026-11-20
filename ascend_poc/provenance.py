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


def runtime_signature(config: dict) -> dict:
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
    packages = {}
    for name in ['vllm', 'vllm-ascend', 'torch', 'torch-npu']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
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
                path = path.resolve()
                stat = path.stat()
                if stat.st_size <= 32 * 1024 * 1024:
                    files[str(path)] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                else:
                    files[str(path)] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'note': 'large-file metadata only'}
    # Prevent resuming an experiment with changed evaluator/controller code.
    code = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))}
    return {'workdir': str(root), 'referenced_env_hashes': values, 'packages': packages,
            'python': platform.python_version(), 'command_files': files, 'controller_code': code}
