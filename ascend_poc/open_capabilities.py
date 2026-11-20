"""Discover and render the installed engine's options without a fixed knob list.

Help establishes flag spelling and argument shape, not hardware compatibility.
The runtime's executable, connectivity and credentials are not Agent parameters.
Candidates remain argv data; no shell or placeholder expansion is introduced.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

from .runner import run_command
from .storage import atomic_json

_MAX_HELP_BYTES = 4 * 1024 * 1024
_MAX_VALUE_BYTES = 128 * 1024
_FLAG = re.compile(r'(?<![\w-])(--?[A-Za-z0-9][A-Za-z0-9_-]*)(?=[\s,=]|$)')
_CONTROL = re.compile(r'[\x00-\x1f\x7f]')
_PLACEHOLDER = re.compile(r'\{(?:env:|param\.|trial_dir\}|repo_dir\})')
_NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_-]*$')
_ENV_NAME = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
_SECRET_ENV = re.compile(r'(?:^|_)(?:API_KEY|TOKEN|SECRET|PASSWORD|CREDENTIALS?|AUTH)(?:_|$)', re.I)

# These establish a process/endpoint or load executable configuration; they are
# deliberately reported, rather than silently pretending they are tuning knobs.
_PROTECTED = {
    'host', 'port', 'uds', 'grpc', 'grpc-port', 'headless',
    'api-key', 'served-model-name', 'model', 'config', 'help', 'version',
    'ssl-keyfile', 'ssl-certfile', 'ssl-ca-certs', 'ssl-cert-reqs',
    'ssl-keyfile-password', 'enable-ssl-refresh', 'root-path',
    'trust-remote-code', 'worker-cls', 'worker-extension-cls', 'scheduler-cls',
    'logits-processors', 'middleware', 'io-processor-plugin',
    'tool-parser-plugin', 'reasoning-parser-plugin', 'otlp-traces-endpoint',
}
_PROTECTED_ENV = {
    'PATH', 'PYTHONPATH', 'PYTHONHOME', 'LD_PRELOAD', 'LD_LIBRARY_PATH',
    'HOME', 'BASH_ENV', 'ENV', 'SHELLOPTS', 'PYTHONSTARTUP', 'LD_AUDIT',
    'PYTHONBREAKPOINT', 'PYTHONINSPECT', 'VLLM_PLUGINS', 'VLLM_LOGGING_CONFIG_PATH',
    'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY',
    'REQUESTS_CA_BUNDLE', 'SSL_CERT_FILE', 'SSL_CERT_DIR',
    'ASCEND_RT_VISIBLE_DEVICES', 'ASCEND_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES',
    'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES',
}
_JSON_FLAGS = {
    'additional-config', 'compilation-config', 'speculative-config',
    'kv-transfer-config', 'kv-events-config', 'hf-overrides',
    'rope-scaling', 'mm-processor-kwargs', 'override-generation-config',
    'pooler-config', 'structured-outputs-config', 'attention-config',
    'kernel-config', 'load-config', 'model-loader-extra-config',
}


def _canonical(name):
    return name.lstrip('-').replace('_', '-')


def _opposite(positive, negative):
    """Recognize an observed alias pair; never invent an unobserved flag."""
    positive, negative = _canonical(positive), _canonical(negative)
    return (negative == 'no-' + positive
            or (positive.startswith('enable-')
                and negative == 'disable-' + positive[len('enable-'):]))


def _reason(name):
    if name in _PROTECTED:
        return 'Runtime connectivity, identity, credentials or executable configuration is controlled by the runtime adapter.'
    if name.endswith(('-plugin', '-plugin-path', '-worker-cls', '-scheduler-cls')):
        return 'Executable plugin/class loading is not an inference parameter.'
    return None


def _flag_spec(header, description):
    aliases = list(dict.fromkeys(_FLAG.findall(header)))
    longs = [a for a in aliases if a.startswith('--')]
    if not longs:
        return None
    primary = longs[0]
    negative = None
    for p in longs:
        for n in longs:
            if p != n and _opposite(p, n):
                primary, negative = p, n
                break
        if negative:
            break
    # Argument placeholders remain after removing all actual option names.
    residue = _FLAG.sub('', header).replace(',', '').strip()
    kind = 'value' if residue else 'boolean'
    if negative:
        kind = 'boolean'
    match = re.search(r'\{([^{}]+)\}', header)
    choices = [v.strip() for v in match.group(1).split(',')] if match else None
    name = _canonical(primary)
    nargs = None
    if kind == 'value' and '...' in residue:
        nargs = '*' if residue.startswith('[') else '+'
    encoding = 'json' if name in _JSON_FLAGS or 'json' in residue.lower() else ('argv_list' if nargs else 'value')
    return name, {
        'aliases': aliases, 'kind': kind, 'negative': negative,
        'choices': choices, 'help': description.strip(), 'metavar': residue,
        'nargs': nargs, 'encoding': encoding,
    }


def parse_help(text):
    """Parse ordinary argparse/vLLM help into dynamic flags and exclusions.

    Types and ranges not visible in help are deliberately not guessed. JSON
    objects are supported for every value flag; known JSON container arguments
    also serialize lists as one JSON value. Visible nargs lists use multiple argv
    items. The engine remains the final validator of unsupported combinations.
    """
    if not isinstance(text, str) or len(text.encode('utf-8')) > _MAX_HELP_BYTES:
        raise ValueError('Engine help must be text of at most 4 MiB')
    entries = []
    current = None
    in_usage = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith('usage:'):
            in_usage = True
            continue
        if in_usage:
            if not stripped or (not line[:1].isspace() and stripped.endswith(':')):
                in_usage = False
            else:
                continue
        if re.match(r'^--?[A-Za-z0-9]', stripped):
            parts = re.split(r'\s{2,}', stripped, maxsplit=1)
            header = parts[0]
            if _FLAG.match(header):
                if current and current[0].rstrip().endswith(','):
                    current[0] += ' ' + header
                    if len(parts) > 1:
                        current[1] += (' ' if current[1] else '') + parts[1]
                    continue
                if current:
                    entries.append(current)
                current = [header, parts[1] if len(parts) > 1 else '']
                continue
        if current:
            # A deeply indented continuation beginning with a flag and no
            # double-space description may be an alias wrapped by argparse.
            if line[:1].isspace() and stripped and not stripped.endswith(':'):
                current[1] += (' ' if current[1] else '') + stripped
            elif stripped:
                entries.append(current)
                current = None
    if current:
        entries.append(current)
    flags = {}
    for header, description in entries:
        result = _flag_spec(header, description)
        if result is None:
            continue
        name, spec = result
        if name in flags:
            raise ValueError(f'Engine help contains duplicate option definitions for {name}')
        flags[name] = spec
    if not flags:
        raise ValueError('No long option definitions found; capture vllm serve --help=all')
    # A few parsers document the opposite boolean action on a separate line.
    for name in list(flags):
        if name not in flags or flags[name]['kind'] != 'boolean':
            continue
        for other in list(flags):
            if (other != name and other in flags and flags[other]['kind'] == 'boolean'
                    and _opposite(name, other)):
                opposite = flags.pop(other)
                flags[name]['negative'] = next(a for a in opposite['aliases'] if a.startswith('--'))
                flags[name]['aliases'].extend(a for a in opposite['aliases'] if a not in flags[name]['aliases'])
    excluded = {}
    for name in list(flags):
        why = _reason(name)
        if why:
            spec = flags.pop(name)
            excluded[name] = {'reason': why, 'aliases': spec['aliases']}
    return {'flags': flags, 'excluded': excluded,
            'source_hash': hashlib.sha256(text.encode('utf-8')).hexdigest()}


def _json_value(value, depth=0):
    if depth > 16:
        raise ValueError('Engine option JSON exceeds nesting limit')
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError('Engine option numbers must be finite and representable')
        return value
    if isinstance(value, str):
        if _CONTROL.search(value) or _PLACEHOLDER.search(value):
            raise ValueError('Engine values cannot contain controls or controller placeholders')
        return value
    if isinstance(value, list):
        return [_json_value(v, depth + 1) for v in value]
    if isinstance(value, dict):
        clean = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError('Engine JSON keys must be strings')
            _json_value(key, depth + 1)
            clean[key] = _json_value(item, depth + 1)
        return clean
    raise ValueError('Engine options must be finite JSON values')


def _enabled(value):
    return value not in (None, False, 0, '', '0', 'false', 'False')


def _semantic_guard(value):
    """Block known fake inference paths, without inventing quality thresholds."""
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = _canonical(key)
            if normalized == 'enable-force-eplb' and _enabled(item):
                raise ValueError('enable_force_eplb replaces real model routing; profiling-only results cannot be inference scores')
            if (normalized == 'rejection-sample-method' and isinstance(item, str)
                    and item.lower() == 'synthetic'):
                raise ValueError('Synthetic speculative acceptance does not measure real model inference')
            if normalized in {'synthetic-acceptance-rates', 'synthetic-acceptance-length'} and item is not None:
                raise ValueError('Synthetic acceptance controls cannot be used for real inference scores')
            if normalized == 'trust-remote-code' and _enabled(item):
                raise ValueError('Executable remote-code configuration is not an engine tuning value')
            _semantic_guard(item)
    elif isinstance(value, list):
        for item in value:
            _semantic_guard(item)


def _as_text(value):
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return str(value)


def _alias_map(caps):
    if not isinstance(caps, dict) or not isinstance(caps.get('flags'), dict):
        raise ValueError('Invalid discovered capability catalog')
    aliases = {}
    for name, spec in caps['flags'].items():
        aliases[name] = (name, False)
        for alias in spec['aliases']:
            normalized = _canonical(alias)
            negative = alias == spec.get('negative')
            previous = aliases.get(normalized)
            if previous and previous != (name, negative):
                raise ValueError('Ambiguous engine flag aliases')
            aliases[normalized] = (name, negative)
    return aliases


def validate_engine(engine, caps):
    """Normalize a complete engine option object, preserving explicit nulls.

    Null means omit the option and use the backend's default. For a single-form
    boolean, false cannot express the opposite of an unknown backend default;
    use null to request omission instead. Lists follow discovered nargs shape.
    """
    if not isinstance(engine, dict):
        raise ValueError('engine must be an option object')
    aliases = _alias_map(caps)
    normalized = {}
    for raw, value in engine.items():
        if not isinstance(raw, str) or not _NAME.fullmatch(raw.lstrip('-')):
            raise ValueError('Engine option keys must be flag names, not command fragments')
        key = _canonical(raw)
        if key not in aliases:
            exclusion = caps.get('excluded', {}).get(key)
            reason = exclusion.get('reason') if exclusion else 'not discovered in the installed engine help'
            raise ValueError(f'Engine option {key} is unavailable: {reason}')
        name, inverted = aliases[key]
        if name in normalized:
            raise ValueError(f'Engine option {name} was supplied through multiple aliases')
        spec = caps['flags'][name]
        value = _json_value(value)
        if value is not None and spec['kind'] == 'boolean':
            if type(value) is not bool:
                raise ValueError(f'Engine option {name} requires a boolean or null')
            if inverted:
                value = not value
            if value is False and not spec.get('negative'):
                raise ValueError(f'Engine option {name} has no discovered negative alias; use null to restore the backend default')
        elif value is not None:
            choices = spec.get('choices')
            if choices:
                values = value if spec.get('encoding') == 'argv_list' and isinstance(value, list) else [value]
                if any(isinstance(v, (dict, list)) or _as_text(v) not in choices for v in values):
                    raise ValueError(f'Engine option {name} is outside the installed CLI choices')
            if spec.get('encoding') == 'argv_list' and isinstance(value, list):
                if spec.get('nargs') == '+' and not value:
                    raise ValueError(f'Engine option {name} requires at least one list item')
                for item in value:
                    if isinstance(item, (list, dict)) or item is None:
                        raise ValueError(f'Engine option {name} requires scalar argv list items')
                    if isinstance(item, str) and item.startswith('-'):
                        raise ValueError('Multi-value option strings cannot begin with a flag prefix')
        if name == 'load-format' and isinstance(value, str) and value.lower() == 'dummy':
            raise ValueError('Dummy weights are not valid customer model inference')
        if isinstance(value, str):
            # JSON supplied as text must not bypass the same semantic checks.
            try:
                decoded = json.loads(value)
            except (ValueError, TypeError):
                decoded = None
            _json_value(decoded)
            _semantic_guard(decoded)
        _semantic_guard({name: value})
        normalized[name] = value
    if len(json.dumps(normalized, ensure_ascii=False).encode('utf-8')) > _MAX_VALUE_BYTES:
        raise ValueError('Engine options exceed 128 KiB')
    return normalized


def build_engine_argv(engine, caps):
    """Return argv data, binding scalar/JSON values with '=' against injection."""
    clean = validate_engine(engine, caps)
    argv = []
    for name, value in clean.items():
        if value is None:
            continue
        spec = caps['flags'][name]
        positive = next(a for a in spec['aliases'] if a.startswith('--') and a != spec.get('negative'))
        if spec['kind'] == 'boolean':
            argv.append(positive if value else spec['negative'])
        elif spec.get('encoding') == 'argv_list' and isinstance(value, list):
            argv.append(positive)
            argv.extend(_as_text(v) for v in value)
        else:
            argv.append(positive + '=' + _as_text(value))
    return argv


def validate_env(values, allowed_names):
    """Normalize explicitly registered performance env vars; protect process IO."""
    if not isinstance(values, dict):
        raise ValueError('environment must be an object')
    if not isinstance(allowed_names, (list, tuple, set, frozenset)):
        raise ValueError('env_names must list supported performance environment variables')
    allowed = set()
    for name in allowed_names:
        if not isinstance(name, str) or not _ENV_NAME.fullmatch(name):
            raise ValueError('env_names contains an invalid environment name')
        if name in _PROTECTED_ENV or _SECRET_ENV.search(name):
            raise ValueError(f'env_names cannot expose protected environment variable {name}')
        allowed.add(name)
    clean = {}
    for name, value in values.items():
        if name not in allowed:
            raise ValueError(f'Environment variable {name} is not a registered tuning capability')
        value = _json_value(value)
        if value is None or isinstance(value, (dict, list)):
            raise ValueError('Environment tuning values must be scalar strings, booleans or numbers')
        clean[name] = ('1' if value else '0') if isinstance(value, bool) else str(value)
    return clean


def discover(config, output: Path, deadline: float, workdir: Path, *, exclude_env=()):
    """Capture local help through a trusted argv command or supplied help file.

    ``output`` is an experiment metadata directory. A command uses the existing
    process-group runner and the campaign deadline; captured files are bounded.
    A help file is a useful offline fixture, not proof of installed compatibility.
    ``exclude_env`` strips exact credential names supplied by the runtime, even
    when their names do not match common secret naming conventions.
    """
    if not isinstance(config, dict) or set(config) - {'command', 'help_file', 'timeout_s', 'env_names'}:
        raise ValueError('Discovery config permits only command/help_file, timeout_s and env_names')
    validate_env({}, config.get('env_names', []))
    if not isinstance(exclude_env, (list, tuple, set, frozenset)):
        raise ValueError('exclude_env must be a collection of exact environment variable names')
    excluded_env = set()
    for name in exclude_env:
        if not isinstance(name, str) or not _ENV_NAME.fullmatch(name):
            raise ValueError('exclude_env contains an invalid environment variable name')
        excluded_env.add(name)
    if ('command' in config) == ('help_file' in config):
        raise ValueError('Discovery requires exactly one command or help_file')
    if not isinstance(deadline, (int, float)) or isinstance(deadline, bool) or not math.isfinite(deadline) or deadline <= time.monotonic():
        raise ValueError('No valid time remains for capability discovery')
    output, workdir = Path(output).resolve(), Path(workdir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'capability-help.txt'
    if 'help_file' in config:
        source = config['help_file']
        if not isinstance(source, str) or not source or _CONTROL.search(source):
            raise ValueError('help_file must be a nonempty path')
        source = Path(source)
        if not source.is_absolute():
            source = workdir / source
        if source.stat().st_size > _MAX_HELP_BYTES:
            raise ValueError('Engine help exceeds 4 MiB')
        raw = source.read_bytes()
        method = 'help_file'
    else:
        command = config['command']
        if not isinstance(command, list) or not command or any(not isinstance(v, str) or not v or _CONTROL.search(v) for v in command):
            raise ValueError('Discovery command must be a nonempty argv array without controls')
        timeout = config.get('timeout_s', 60)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('Discovery timeout_s must be a positive finite number')
        env = {k: v for k, v in os.environ.items() if not _SECRET_ENV.search(k) and k not in excluded_env}
        run_command(command, cwd=workdir, env=env, log_path=target,
                    timeout_s=timeout, deadline=deadline, label='Capability discovery')
        if target.stat().st_size > _MAX_HELP_BYTES:
            raise ValueError('Engine help exceeds 4 MiB')
        raw = target.read_bytes()
        method = 'command'
    if len(raw) > _MAX_HELP_BYTES:
        raise ValueError('Engine help exceeds 4 MiB')
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError:
        raise ValueError('Engine help must be UTF-8') from None
    capabilities = parse_help(text)
    capabilities['source'] = method
    target.write_bytes(raw)
    atomic_json(output / 'capabilities.json', capabilities)
    return capabilities
