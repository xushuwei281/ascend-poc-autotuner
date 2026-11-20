"""One-request JSON transport for the open-configuration tuning controller.

Each ``ask`` performs at most one HTTP attempt. Retry policy and aggregate call
budgets belong to the controller; malformed responses are never retried here.
"""
from __future__ import annotations

import copy
import json
import math
import os
import re

from .config import _json_object, _reject_constant
from .llm import LLMError, LLMPlanner, _ProviderFailure


def _finite_json(value, depth=0):
    if depth > 30:
        raise ValueError('JSON nesting exceeds 30 levels')
    if value is None or isinstance(value, (str, bool)):
        if isinstance(value, str) and '\x00' in value:
            raise ValueError('JSON strings cannot contain NUL')
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError('JSON numbers must be finite')
        return
    if isinstance(value, list):
        for item in value:
            _finite_json(item, depth + 1)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for key, item in value.items():
            if '\x00' in key:
                raise ValueError('JSON keys cannot contain NUL')
            _finite_json(item, depth + 1)
        return
    raise ValueError('Only JSON values are supported')


def _positive_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + ' must be a positive finite number')
    return value


class JSONModel(LLMPlanner):
    """OpenAI-compatible JSON calls, with inherited bounded/no-redirect HTTP.

    ``max_retries`` is accepted for configuration compatibility, but deliberately
    has no effect: one ask is one attempt, including HTTP 429/5xx failures.
    """

    def __init__(self, settings):
        if not isinstance(settings, dict):
            raise ValueError('LLM settings must be an object')
        settings = copy.deepcopy(settings)
        allowed = {'base_url', 'model', 'api_key_env', 'timeout_s', 'max_tokens',
                   'temperature', 'json_mode', 'max_retries'}
        if set(settings) - allowed:
            raise ValueError('Unknown LLM settings')
        for name in ('base_url', 'model'):
            if not isinstance(settings.get(name), str) or not settings[name].strip() or '\x00' in settings[name]:
                raise ValueError('LLM ' + name + ' must be a nonempty string')
        env = settings.setdefault('api_key_env', 'POC_LLM_API_KEY')
        if not isinstance(env, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', env):
            raise ValueError('LLM api_key_env must name an environment variable')
        _positive_number(settings.setdefault('timeout_s', 60), 'LLM timeout_s')
        tokens = settings.setdefault('max_tokens', 4000)
        if type(tokens) is not int or tokens < 1:
            raise ValueError('LLM max_tokens must be a positive integer')
        temperature = settings.setdefault('temperature', 0.2)
        if temperature is not None and (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
                or not math.isfinite(temperature) or not 0 <= temperature <= 2):
            raise ValueError('LLM temperature must be null or between 0 and 2')
        if type(settings.setdefault('json_mode', True)) is not bool:
            raise ValueError('LLM json_mode must be a boolean')
        retries = settings.setdefault('max_retries', 0)
        if type(retries) is not int or not 0 <= retries <= 10:
            raise ValueError('LLM max_retries must be an integer from 0 to 10')
        super().__init__({'llm': settings})

    def ask(self, system, payload, remaining_s):
        self.last_usage = {}
        try:
            _positive_number(remaining_s, 'Remaining LLM time')
            if not isinstance(system, str) or not system.strip() or len(system) > 30000:
                raise ValueError('System instructions must be a bounded nonempty string')
            if not isinstance(payload, dict):
                raise ValueError('LLM payload must be an object')
            _finite_json(payload)
            content = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
            if len(content) > 200000:
                raise ValueError('LLM payload exceeds the context limit')
        except (ValueError, TypeError, RecursionError, OverflowError) as error:
            raise LLMError('Invalid JSON model request: ' + str(error)) from None
        key = os.environ.get(self.settings['api_key_env'], '')
        if not key.strip():
            raise LLMError('LLM API key is missing from the configured environment variable.')
        body = {'model': self.settings['model'], 'stream': False, 'max_tokens': self.settings['max_tokens'],
                'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}]}
        if self.settings['temperature'] is not None:
            body['temperature'] = self.settings['temperature']
        if self.settings['json_mode']:
            body['response_format'] = {'type': 'json_object'}
        encoded = json.dumps(body, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()
        self.calls += 1
        try:
            raw = self._request(encoded, key, min(float(remaining_s), float(self.settings['timeout_s'])))
        except _ProviderFailure as error:
            raise LLMError(str(error)) from None
        try:
            response = json.loads(raw, object_pairs_hook=_json_object, parse_constant=_reject_constant)
            _finite_json(response)
            content = response['choices'][0]['message']['content']
            if not isinstance(content, str):
                raise ValueError('Missing content')
            usage = response.get('usage', {})
            if isinstance(usage, dict):
                self.last_usage = {key: usage[key] for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')
                                   if type(usage.get(key)) is int and usage[key] >= 0}
        except (ValueError, TypeError, KeyError, IndexError, RecursionError, OverflowError):
            raise LLMError('LLM API response has an invalid Chat Completions structure.') from None
        text = content.strip()
        fence = re.fullmatch(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL | re.IGNORECASE)
        if fence:
            text = fence.group(1).strip()
        try:
            result = json.loads(text, object_pairs_hook=_json_object, parse_constant=_reject_constant)
            _finite_json(result)
            if not isinstance(result, dict):
                raise ValueError('Expected an object')
        except (ValueError, TypeError, RecursionError, OverflowError):
            raise LLMError('LLM response must be one finite JSON object without duplicate keys.') from None
        return result
