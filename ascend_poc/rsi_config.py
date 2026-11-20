"""Strict configuration for the bounded inference improvement agent."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from .config import _json_object, _reject_constant, _keys, _number, _string
from .strategy import load_workflow, workflow_hash

DEFAULT_SETTINGS = {
    'max_steps': 40, 'max_experiments': 8, 'max_wall_time_s': 7200,
    'experiment_timeout_s': 900, 'max_llm_calls': 80, 'max_research_calls': 12,
    'max_consecutive_tool_errors': 3,
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_json(path):
    path = Path(path)
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError('Agent JSON input exceeds 8 MiB')
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_json_object,
                      parse_constant=_reject_constant)


def load_agent(path):
    source = Path(path).resolve()
    value = read_json(source)
    if not isinstance(value, dict):
        raise ValueError('Agent config must be an object')
    _keys(value, {'schema_version', 'name', 'goal', 'workflow', 'memory_file', 'settings'}, 'agent')
    if type(value.get('schema_version')) is not int or value['schema_version'] != 1:
        raise ValueError('agent.schema_version must be 1')
    for name in ('name', 'goal', 'workflow'):
        _string(value.get(name), 'agent.' + name)
    if len(value['goal']) > 3000:
        raise ValueError('agent.goal must be at most 3000 characters')
    if Path(value['workflow']).is_absolute():
        raise ValueError('agent.workflow must be relative to the agent config')
    _string(value.setdefault('memory_file', 'runs/rsi-memory.json'), 'agent.memory_file')
    settings = value.setdefault('settings', {})
    if not isinstance(settings, dict):
        raise ValueError('agent.settings must be an object')
    _keys(settings, set(DEFAULT_SETTINGS), 'agent.settings')
    for key, default in DEFAULT_SETTINGS.items():
        _number(settings.setdefault(key, default), 'agent.settings.' + key,
                minimum=.01 if key.endswith('_s') else 1,
                integer=not key.endswith('_s'))
    for key in ('max_steps', 'max_experiments', 'max_llm_calls', 'max_research_calls'):
        if settings[key] > 10000:
            raise ValueError('agent.settings.' + key + ' exceeds 10000')
    value['_workflow'] = load_workflow(source.parent / value['workflow'])
    value['_memory_file'] = str((source.parent / value['memory_file']).resolve())
    return value


def agent_hash(config):
    public = {k: copy.deepcopy(v) for k, v in config.items() if not k.startswith('_')}
    public['workflow_hash'] = workflow_hash(config['_workflow'])
    # Distinct output roots should not silently redirect persistent experience.
    public['resolved_memory_file'] = config['_memory_file']
    return digest(public)
