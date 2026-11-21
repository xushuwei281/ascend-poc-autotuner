"""Operator wiring for open-ended, request-driven inference experiments.

The runtime names actual resources and measurement adapters. It does not grant
baseline values the status of customer constraints. Candidate engine options are
validated against the installed CLI, while only the compiled request supplies
acceptance gates and parameter locks. Existing bounded-mode contracts are left
unchanged.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re

from .config import (_argv, _json_object, _keys, _number, _object,
                     _reject_constant, _requirements, _string, _url)
from .deepwiki import validate_endpoint
from .llm import _endpoint

_NAME = re.compile(r'^[A-Za-z][A-Za-z0-9_]*$')
_METRIC = re.compile(r'^[A-Za-z][A-Za-z0-9_.-]{0,127}$')
_ENV = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
_PLACEHOLDER = re.compile(r'\{([A-Za-z_][A-Za-z0-9_.:]*)\}')
_WORKLOAD = re.compile(r'\{workload\.([A-Za-z][A-Za-z0-9_]*)\}')
_FIXED_OPTIONS = frozenset({
    'host', 'port', 'served-model-name', 'root-path', 'api-key',
    'ssl-keyfile', 'ssl-certfile', 'ssl-ca-certs', 'ssl-cert-reqs',
    'enable-ssl-refresh', 'trust-remote-code',
})
_SCENARIO_COMMANDS = ('command', 'before_each_command', 'warmup_command', 'quality_command')
_SCENARIO_OUTPUTS = ('result_file', 'quality_result_file', 'telemetry_file')


def _finite_json(value, label='value', depth=0):
    if depth > 24:
        raise ValueError(f'{label}: excessive JSON nesting')
    if value is None or type(value) in (bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, str) and '\x00' not in value:
        return
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        for child in value.values():
            _finite_json(child, label, depth + 1)
        return
    if isinstance(value, list):
        for child in value:
            _finite_json(child, label, depth + 1)
        return
    raise ValueError(f'{label}: must contain only finite JSON values without NUL')


def _untrusted_string(value, label):
    _string(value, label, empty=True)
    if any(ord(c) < 32 for c in value) or _PLACEHOLDER.search(value):
        raise ValueError(f'{label}: control characters and executable placeholders are not allowed')
    return value


def _env_mapping(value, label):
    _object(value, label)
    for key, item in value.items():
        if not _ENV.fullmatch(key):
            raise ValueError(f'{label}: invalid environment variable name')
        _string(item, f'{label}.{key}', empty=True)
    return value


def _templates(command, label, *, workload=False, model=False):
    _argv(command, label)
    for index, token in enumerate(command):
        if workload and _WORKLOAD.search(token) and (
                index == 0 or command[index - 1] in {'-c', '--command', '-m'}):
            raise ValueError(f'{label}: workload values cannot choose executables, modules or code')
        for key in _PLACEHOLDER.findall(token):
            if key in {'repo_dir', 'trial_dir'} or (key.startswith('env:') and _ENV.fullmatch(key[4:])):
                continue
            if model and key == 'model':
                continue
            if workload and key.startswith('workload.') and _NAME.fullmatch(key[9:]):
                continue
            raise ValueError(f'{label}: unsupported placeholder {key!r}')


def _check_workload(values, specs):
    _object(values, 'candidate.workload')
    if set(values) != set(specs):
        raise ValueError('candidate.workload must contain every declared workload parameter exactly once')
    for key, value in values.items():
        spec = specs[key]
        kind = spec['type']
        if kind in {'integer', 'number'}:
            _number(value, f'candidate.workload.{key}', integer=kind == 'integer',
                    minimum=spec.get('minimum'), maximum=spec.get('maximum'))
        elif kind == 'boolean':
            if type(value) is not bool:
                raise ValueError(f'candidate.workload.{key}: must be boolean')
        else:
            _untrusted_string(value, f'candidate.workload.{key}')
            if value.startswith('-'):
                raise ValueError(f'candidate.workload.{key}: a string value cannot inject a command-line flag')
        if 'choices' in spec:
            if not any(type(value) is type(v) and value == v for v in spec['choices']):
                raise ValueError(f'candidate.workload.{key}: outside adapter choices')
    return copy.deepcopy(values)


def _candidate_shape(candidate):
    _object(candidate, 'candidate')
    if set(candidate) != {'model', 'engine', 'env', 'workload'}:
        raise ValueError('candidate must contain exactly model, engine, env and workload')
    _string(candidate['model'], 'candidate.model')
    for field in ('engine', 'env', 'workload'):
        _object(candidate[field], f'candidate.{field}')
    _finite_json(candidate, 'candidate')
    if len(json.dumps(candidate, ensure_ascii=False)) > 250_000:
        raise ValueError('candidate exceeds the size limit')


def load_runtime(path):
    """Load operator-authored wiring; no model API, device or help command runs."""
    source = Path(path).resolve()
    if source.stat().st_size > 8 * 1024 * 1024:
        raise ValueError('Runtime configuration exceeds 8 MiB')
    value = json.loads(source.read_text(encoding='utf-8'), object_pairs_hook=_json_object,
                       parse_constant=_reject_constant)
    result = validate_runtime(value, base_dir=source.parent)
    result['_source_path'] = str(source)
    return result


def validate_runtime(value, base_dir=None):
    """Normalize an in-memory runtime, also usable by the local synthetic demo."""
    config = copy.deepcopy(value)
    _object(config, 'runtime')
    _keys(config, {'schema_version', 'name', 'hardware', 'models', 'server', 'discovery',
                   'baseline', 'scenarios', 'workload_parameters', 'metric_names',
                   'benchmark_env', 'llm', 'research', 'budget', 'verification'}, 'runtime')
    _finite_json(config, 'runtime')
    if type(config.get('schema_version')) is not int or config['schema_version'] != 1:
        raise ValueError('runtime.schema_version must be 1')
    _string(config.get('name'), 'runtime.name')
    config['_source_dir'] = str(Path(base_dir or Path.cwd()).resolve())

    hardware = _object(config.get('hardware'), 'hardware')
    _keys(hardware, {'device_type', 'device_count', 'devices'}, 'hardware')
    _string(hardware.get('device_type'), 'hardware.device_type')
    _number(hardware.get('device_count'), 'hardware.device_count', minimum=1, integer=True)
    devices = hardware.get('devices')
    if not isinstance(devices, list) or len(devices) != hardware['device_count']:
        raise ValueError('hardware.devices must list each declared device')
    if any(not isinstance(d, str) or not re.fullmatch(r'[A-Za-z0-9_.:-]+', d) for d in devices):
        raise ValueError('hardware.devices contains an invalid device identifier')
    if len(set(devices)) != len(devices):
        raise ValueError('hardware.devices contains duplicates')
    resource = json.dumps(sorted(devices), separators=(',', ':')).encode()
    config['_resource_id'] = 'open-devices-' + hashlib.sha256(resource).hexdigest()[:20]

    models = config.get('models')
    if not isinstance(models, list) or not models:
        raise ValueError('models must contain available local model assets')
    seen = set()
    for model in models:
        _object(model, 'models item')
        _keys(model, {'id', 'path', 'family', 'precision'}, 'models item')
        for key in ('id', 'path', 'family'):
            _string(model.get(key), 'models.' + key)
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', model['id']) or model['id'] in seen:
            raise ValueError('model IDs must be unique simple identifiers')
        seen.add(model['id'])
        if 'precision' in model:
            _string(model['precision'], 'models.precision')
        # A runtime model path is a trusted local asset reference, not an Agent URL.
        for placeholder in _PLACEHOLDER.findall(model['path']):
            if not placeholder.startswith('env:') or not _ENV.fullmatch(placeholder[4:]):
                raise ValueError('models.path only supports trusted environment placeholders')
        if '://' in model['path']:
            raise ValueError('models.path must reference a local asset, not a download URL')

    server = _object(config.get('server'), 'server')
    _keys(server, {'command', 'fixed_options', 'env', 'health_url', 'startup_timeout_s', 'shutdown_timeout_s'}, 'server')
    _templates(server.get('command'), 'server.command', model=True)
    if sum(token.count('{model}') for token in server['command']) != 1 or '{model}' not in server['command']:
        raise ValueError('server.command must contain exactly one standalone {model} argument')
    _url(server.get('health_url'), 'server.health_url')
    options = _object(server.setdefault('fixed_options', {}), 'server.fixed_options')
    _keys(options, set(_FIXED_OPTIONS), 'server.fixed_options; inference options belong in baseline.engine')
    for key, item in options.items():
        if not isinstance(item, (str, int, float, bool)) and not (
                key in {'served-model-name', 'api-key'} and isinstance(item, list)
                and item and all(isinstance(x, str) and x for x in item)):
            raise ValueError('server.fixed_options values must be scalar connection options')
        if item is None:
            raise ValueError('server.fixed_options must not contain null')
    if 'port' in options:
        _number(options['port'], 'server.fixed_options.port', minimum=1, maximum=65535, integer=True)
    _env_mapping(server.setdefault('env', {}), 'server.env')
    _number(server.setdefault('startup_timeout_s', 900), 'server.startup_timeout_s', minimum=.01)
    _number(server.setdefault('shutdown_timeout_s', 20), 'server.shutdown_timeout_s', minimum=.01)

    discovery = _object(config.get('discovery'), 'discovery')
    _keys(discovery, {'command', 'help_file', 'timeout_s', 'env_names'}, 'discovery')
    if ('command' in discovery) == ('help_file' in discovery):
        raise ValueError('discovery requires exactly one of command or help_file')
    if 'command' in discovery:
        _templates(discovery['command'], 'discovery.command')
    else:
        _string(discovery['help_file'], 'discovery.help_file')
        discovery['help_file'] = str((Path(config['_source_dir']) / discovery['help_file']).resolve())
    _number(discovery.setdefault('timeout_s', 60), 'discovery.timeout_s', minimum=.01)
    names = discovery.setdefault('env_names', [])
    if not isinstance(names, list) or any(not isinstance(n, str) or not _ENV.fullmatch(n) for n in names) or len(names) != len(set(names)):
        raise ValueError('discovery.env_names must contain unique environment names')

    specs = _object(config.setdefault('workload_parameters', {}), 'workload_parameters')
    for key, spec in specs.items():
        if not _NAME.fullmatch(key):
            raise ValueError('workload_parameters contains an invalid name')
        _object(spec, f'workload_parameters.{key}')
        _keys(spec, {'type', 'description', 'minimum', 'maximum', 'choices'}, f'workload_parameters.{key}')
        if spec.get('type') not in {'integer', 'number', 'string', 'boolean'}:
            raise ValueError('workload parameter type must be integer, number, string or boolean')
        _string(spec.get('description'), f'workload_parameters.{key}.description')
        for bound in ('minimum', 'maximum'):
            if bound in spec:
                if spec['type'] not in {'integer', 'number'}:
                    raise ValueError('workload numeric limits require a numeric type')
                _number(spec[bound], f'workload_parameters.{key}.{bound}', integer=spec['type'] == 'integer')
        if 'minimum' in spec and 'maximum' in spec and spec['minimum'] > spec['maximum']:
            raise ValueError('workload minimum cannot exceed maximum')
        if 'choices' in spec and (not isinstance(spec['choices'], list) or not spec['choices']):
            raise ValueError('workload choices must be a nonempty list')
        if 'choices' in spec:
            # Check each adapter choice against its type and range, without itself.
            without_choices = {k: v for k, v in spec.items() if k != 'choices'}
            for item in spec['choices']:
                _check_workload({key: item}, {key: without_choices})

    scenarios = config.get('scenarios')
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError('scenarios must be a nonempty measurement adapter list')
    scene_names, bound_parameters = set(), set()
    for scene in scenarios:
        _object(scene, 'scenario')
        _keys(scene, {'name', 'command', 'result_file', 'metrics_format', 'timeout_s',
                      'quality_command', 'quality_result_file', 'warmup_command',
                      'before_each_command', 'telemetry_file'}, 'scenario')
        name = scene.get('name')
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name) or name in scene_names:
            raise ValueError('scenario names must be unique simple identifiers')
        scene_names.add(name)
        for key in _SCENARIO_COMMANDS:
            if key == 'command' or key in scene:
                _templates(scene.get(key), f'scenario.{key}', workload=True)
                for arg in scene[key]:
                    bound_parameters.update(_WORKLOAD.findall(arg))
        for key in _SCENARIO_OUTPUTS:
            if key == 'result_file' or key in scene:
                _string(scene.get(key), f'scenario.{key}')
                if '{workload.' in scene[key] or '{param.' in scene[key]:
                    raise ValueError('measurement result paths cannot be tuned')
        output_paths = [scene[k] for k in _SCENARIO_OUTPUTS if k in scene]
        if len(output_paths) != len(set(output_paths)):
            raise ValueError('measurement outputs must have distinct paths')
        if (('quality_command' in scene) != ('quality_result_file' in scene)):
            raise ValueError('quality command and result file must be provided together')
        if scene.setdefault('metrics_format', 'canonical') not in {'canonical', 'vllm'}:
            raise ValueError('scenario.metrics_format must be canonical or vllm')
        _number(scene.setdefault('timeout_s', 900), 'scenario.timeout_s', minimum=.01)
    if bound_parameters != set(specs):
        raise ValueError('workload parameters must exactly match placeholders in measurement commands')
    _env_mapping(config.setdefault('benchmark_env', {}), 'benchmark_env')
    if any('{workload.' in v or '{param.' in v for v in config['benchmark_env'].values()):
        raise ValueError('benchmark_env cannot contain tunable placeholders')

    metric_names = config.get('metric_names')
    if not isinstance(metric_names, list) or not metric_names or any(not isinstance(m, str) or not _METRIC.fullmatch(m) for m in metric_names) or len(metric_names) != len(set(metric_names)):
        raise ValueError('metric_names must list unique adapter metric names')
    baseline = config.get('baseline')
    _candidate_shape(baseline)
    if baseline['model'] not in seen:
        raise ValueError('baseline.model must name a supplied local asset')
    _check_workload(baseline['workload'], specs)

    llm = _object(config.get('llm'), 'llm')
    _keys(llm, {'base_url', 'model', 'api_key_env', 'timeout_s', 'max_tokens', 'temperature', 'json_mode', 'max_retries'}, 'llm')
    _endpoint(llm.get('base_url'))
    _string(llm.get('model'), 'llm.model')
    key_env = llm.setdefault('api_key_env', 'POC_LLM_API_KEY')
    if not isinstance(key_env, str) or not _ENV.fullmatch(key_env):
        raise ValueError('llm.api_key_env must be an environment variable name')
    _number(llm.setdefault('timeout_s', 60), 'llm.timeout_s', minimum=.01)
    _number(llm.setdefault('max_tokens', 3000), 'llm.max_tokens', minimum=1, integer=True)
    if llm.setdefault('temperature', .2) is not None:
        _number(llm['temperature'], 'llm.temperature', minimum=0, maximum=2)
    if type(llm.setdefault('json_mode', True)) is not bool:
        raise ValueError('llm.json_mode must be boolean')
    if 'max_retries' in llm:
        _number(llm['max_retries'], 'llm.max_retries', minimum=0, maximum=10, integer=True)
    if key_env in names or key_env in server['env'] or key_env in config['benchmark_env']:
        raise ValueError('planner API key cannot enter measurement or tuning environments')
    # Runtime paths/commands are trusted, but copying the planner key is still a wiring error.
    def contains_secret_ref(item):
        if isinstance(item, str):
            return '{env:' + key_env + '}' in item
        if isinstance(item, dict):
            return any(contains_secret_ref(x) for x in item.values())
        if isinstance(item, list):
            return any(contains_secret_ref(x) for x in item)
        return False
    if any(contains_secret_ref(config[k]) for k in ('models', 'server', 'discovery', 'scenarios', 'benchmark_env')):
        raise ValueError('measurement wiring cannot reference the planner API key')

    research = _object(config.setdefault('research', {}), 'research')
    _keys(research, {'transport', 'endpoint', 'repository', 'timeout_s', 'max_calls', 'on_error'}, 'research')
    transport = research.setdefault('transport', 'mcp')
    if not isinstance(transport, str) or transport not in {'mcp', 'a2a'}:
        raise ValueError('research.transport must be mcp or a2a')
    default_endpoint = 'http://127.0.0.1:8766/a2a' if transport == 'a2a' else 'https://mcp.deepwiki.com/mcp'
    validate_endpoint(research.setdefault('endpoint', default_endpoint))
    if research.setdefault('repository', 'vllm-project/vllm-ascend') != 'vllm-project/vllm-ascend':
        raise ValueError('research.repository must be vllm-project/vllm-ascend')
    _number(research.setdefault('timeout_s', 60), 'research.timeout_s', minimum=.01)
    _number(research.setdefault('max_calls', 8), 'research.max_calls', minimum=0, maximum=10000, integer=True)
    if research.setdefault('on_error', 'stop') not in {'stop', 'continue'}:
        raise ValueError('research.on_error must be stop or continue')
    budget = _object(config.setdefault('budget', {}), 'budget')
    defaults = {'max_experiments': 12, 'max_wall_time_s': 7200, 'max_llm_calls': 64,
                'repetitions': 1, 'final_repetitions': 3}
    _keys(budget, set(defaults), 'budget')
    for key, default in defaults.items():
        _number(budget.setdefault(key, default), 'budget.' + key,
                minimum=.01 if key == 'max_wall_time_s' else 1,
                maximum=None if key == 'max_wall_time_s' else 10000,
                integer=key != 'max_wall_time_s')
    verification = _object(config.setdefault('verification', {}), 'verification')
    _keys(verification, {'mode', 'max_baseline_drift', 'min_relative_improvement', 'drift_multiplier', 'frozen_files'}, 'verification')
    if verification.setdefault('mode', 'aba') != 'aba':
        raise ValueError('open tuning currently uses independent ABA verification')
    for key, default in {'max_baseline_drift': .05, 'min_relative_improvement': .02, 'drift_multiplier': 2.0}.items():
        _number(verification.setdefault(key, default), 'verification.' + key, minimum=0)
    if verification['max_baseline_drift'] > 1:
        raise ValueError('verification.max_baseline_drift must not exceed 1')
    files = verification.setdefault('frozen_files', [])
    if not isinstance(files, list):
        raise ValueError('verification.frozen_files must be a list')
    paths = set()
    for item in files:
        if not isinstance(item, dict) or set(item) != {'path', 'sha256'}:
            raise ValueError('frozen_files entries require path and sha256')
        _string(item['path'], 'frozen_files.path')
        if item['path'] in paths or not isinstance(item['sha256'], str) or not re.fullmatch(r'[0-9a-f]{64}', item['sha256']):
            raise ValueError('frozen_files requires unique paths and lowercase SHA256 digests')
        paths.add(item['path'])
    return config


def validate_candidate(runtime, candidate, intent, caps):
    """Validate an unrestricted-value proposal against resources and explicit intent."""
    from .open_capabilities import validate_engine, validate_env
    from .open_intent import check_candidate
    _candidate_shape(candidate)
    clean = copy.deepcopy(candidate)
    if clean['model'] not in {m['id'] for m in runtime['models']}:
        raise ValueError('candidate.model is not an available local model asset')
    clean['engine'] = validate_engine(clean['engine'], caps)
    _engine_assets(clean['engine'], runtime, resolve=False)
    clean['env'] = validate_env(clean['env'], runtime['discovery']['env_names'])
    clean['workload'] = _check_workload(clean['workload'], runtime['workload_parameters'])
    count = 1
    for key in ('tensor-parallel-size', 'pipeline-parallel-size', 'data-parallel-size'):
        value = clean['engine'].get(key)
        value = 1 if value is None else value
        if type(value) is not int or value < 1:
            raise ValueError(f'{key} must be a positive integer')
        count *= value
    if count > runtime['hardware']['device_count']:
        raise ValueError('TP × PP × DP exceeds the available device count')
    # Fixed transport configuration never has a second conflicting tunable value.
    if set(clean['engine']) & set(runtime['server']['fixed_options']):
        raise ValueError('candidate conflicts with fixed controller connection options')
    failures = check_candidate(intent, clean, runtime)
    if failures:
        raise ValueError('Explicit request constraints: ' + '; '.join(failures))
    return clean


def _scalar_text(value):
    if type(value) is bool:
        return 'true' if value else 'false'
    return str(value)


def _local_model_path(runtime, model):
    path = model['path']
    key_env = runtime['llm']['api_key_env']
    def replace(match):
        key = match.group(1)[4:]
        if key == key_env or not os.environ.get(key):
            raise ValueError(f'Local model asset requires environment variable {key}')
        return os.environ[key]
    path = _PLACEHOLDER.sub(replace, path)
    _untrusted_string(path, 'resolved model path')
    resolved = (Path(runtime['_source_dir']) / path).resolve()
    if not resolved.exists():
        raise ValueError('Selected local model asset is missing; prepare it before running an experiment')
    return str(resolved)


def _engine_assets(value, runtime, *, resolve):
    """Validate or resolve ``asset:MODEL_ID`` values without changing a proposal.

    Nested engine JSON may refer to a local draft model or another registered
    model asset. Persisted candidates retain these symbolic references; only the
    executable argv receives existing absolute paths. JSON supplied as a string
    receives the same treatment so it cannot bypass unknown-asset validation.
    """
    if isinstance(value, dict):
        return {key: _engine_assets(child, runtime, resolve=resolve)
                for key, child in value.items()}
    if isinstance(value, list):
        return [_engine_assets(child, runtime, resolve=resolve) for child in value]
    if not isinstance(value, str):
        return value
    if value.startswith('asset:'):
        identifier = value[6:]
        model = next((item for item in runtime['models'] if item['id'] == identifier), None)
        if model is None:
            raise ValueError('Engine asset reference names an unknown local model asset')
        return _local_model_path(runtime, model) if resolve else value
    if value.lstrip().startswith(('{', '[')):
        try:
            decoded = json.loads(value, object_pairs_hook=_json_object,
                                 parse_constant=_reject_constant)
        except (TypeError, ValueError, RecursionError):
            return value  # Engine startup validates other opaque string syntax.
        if isinstance(decoded, (dict, list)):
            transformed = _engine_assets(decoded, runtime, resolve=resolve)
            if resolve and transformed != decoded:
                return json.dumps(transformed, ensure_ascii=False, separators=(',', ':'))
    return value


def materialize(runtime, candidate, intent, caps, workdir):
    """Produce a runner contract; performance gates come only from the request.

    The result deliberately does not pass through legacy finite-search validation.
    The trusted executable prefix is preserved and all proposed values remain
    literal argv/environment values. Benchmark environment is separate from the
    candidate server environment even when no benchmark overrides are supplied.
    """
    from .open_capabilities import build_engine_argv
    candidate = validate_candidate(runtime, candidate, intent, caps)
    model = next(m for m in runtime['models'] if m['id'] == candidate['model'])
    model_path = _local_model_path(runtime, model)
    server = copy.deepcopy(runtime['server'])
    argv = [model_path if token == '{model}' else token for token in server['command']]
    for key, value in server.pop('fixed_options').items():
        if type(value) is bool:
            if value:
                argv.append('--' + key)
        elif isinstance(value, list):
            argv.append('--' + key)
            argv.extend(value)
        else:
            argv.append('--' + key + '=' + _scalar_text(value))
    resolved_engine = _engine_assets(candidate['engine'], runtime, resolve=True)
    argv.extend(build_engine_argv(resolved_engine, caps))
    server['command'] = argv
    server['env'].update(candidate['env'])
    scenes = copy.deepcopy(runtime['scenarios'])
    for scene in scenes:
        for key in _SCENARIO_COMMANDS:
            if key in scene:
                scene[key] = [_WORKLOAD.sub(lambda m: _scalar_text(candidate['workload'][m[1]]), arg)
                              for arg in scene[key]]
    requirements = copy.deepcopy(intent.get('metric_requirements', {}))
    _requirements(requirements, 'intent.metric_requirements')
    objective = copy.deepcopy(intent['objective'])
    if objective['metric'] not in runtime['metric_names'] or set(requirements) - set(runtime['metric_names']):
        raise ValueError('Requested metrics are not exposed by the measurement adapter')
    return {'schema_version': 1, 'name': runtime['name'], 'resource_id': runtime['_resource_id'],
            'context': {'hardware': copy.deepcopy(runtime['hardware']),
                        'model': {k: v for k, v in model.items() if k != 'path'},
                        'workload': copy.deepcopy(candidate['workload'])},
            'server': server, 'benchmark_env': copy.deepcopy(runtime['benchmark_env']),
            'scenarios': scenes, 'requirements': requirements, 'objective': objective,
            'budget': {'repetitions': runtime['budget']['repetitions'],
                       'final_repetitions': runtime['budget']['final_repetitions']},
            'llm': copy.deepcopy(runtime['llm']), '_repo_dir': str(Path(workdir).resolve())}
