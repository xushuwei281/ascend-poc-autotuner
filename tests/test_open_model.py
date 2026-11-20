"""Open mode model transport makes every HTTP attempt visible to its budget."""
import json
import os
import unittest
from unittest.mock import patch

from ascend_poc.llm import LLMError
from ascend_poc.open_model import JSONModel
from tests.test_llm import provider


def settings(url='http://localhost/v1'):
    return {'base_url': url, 'model': 'test-model', 'api_key_env': 'POC_TEST_KEY',
            'timeout_s': 2, 'max_tokens': 1000, 'temperature': 0.2,
            'json_mode': True, 'max_retries': 2}


def response(content='{"next":"measure"}', usage=None):
    return {'choices': [{'message': {'content': content}}], 'usage': usage or {'total_tokens': 25}}


class JSONModelTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {'POC_TEST_KEY': 'test-only-key'})
        self.environment.start(); self.addCleanup(self.environment.stop)

    def test_actual_http_protocol_and_usage(self):
        with provider([{'payload': response()}]) as (url, captured):
            client = JSONModel(settings(url))
            self.assertEqual(client.ask('Use JSON.', {'stage': 'compile_intent'}, 3), {'next': 'measure'})
        self.assertEqual(client.calls, 1)
        self.assertEqual(client.last_usage, {'total_tokens': 25})
        self.assertEqual(captured[0]['path'], '/v1/chat/completions')
        body = captured[0]['body']
        self.assertEqual(body['messages'][0], {'role': 'system', 'content': 'Use JSON.'})
        self.assertEqual(json.loads(body['messages'][1]['content']), {'stage': 'compile_intent'})
        self.assertFalse(body['stream'])

    def test_no_hidden_retry_even_for_503_and_retry_compatible_settings(self):
        with provider([{'status': 503}]) as (url, captured):
            client = JSONModel(settings(url))
            with self.assertRaisesRegex(LLMError, 'HTTP 503'):
                client.ask('Use JSON.', {}, 3)
        self.assertEqual(client.calls, 1)
        self.assertEqual(len(captured), 1)

    def test_bad_json_duplicate_keys_and_infinite_numbers_are_rejected(self):
        bad = ['[]', 'null', 'not JSON PRIVATE', '{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}', '{"x":{"a":1,"a":2}}']
        for content in bad:
            client = JSONModel(settings())
            with self.subTest(content=content), patch.object(client, '_request', return_value=json.dumps(response(content)).encode()):
                with self.assertRaises(LLMError) as caught:
                    client.ask('Use JSON.', {}, 3)
                self.assertNotIn('PRIVATE', str(caught.exception))
                self.assertEqual(client.calls, 1)

    def test_duplicate_provider_envelope_keys_are_rejected(self):
        client = JSONModel(settings())
        raw = b'{"choices":[],"choices":[{"message":{"content":"{}"}}]}'
        with patch.object(client, '_request', return_value=raw), self.assertRaises(LLMError):
            client.ask('Use JSON.', {}, 3)

    def test_fenced_json_nested_configuration_and_optional_fields(self):
        conf = settings(); conf.update(temperature=None, json_mode=False)
        client = JSONModel(conf)
        content = '```json\n{"engine":{"speculative-config":{"method":"ngram"}}}\n```'
        with patch.object(client, '_request', return_value=json.dumps(response(content)).encode()) as request:
            result = client.ask('Use JSON.', {}, 3)
        self.assertEqual(result['engine']['speculative-config']['method'], 'ngram')
        body = json.loads(request.call_args.args[0])
        self.assertNotIn('temperature', body)
        self.assertNotIn('response_format', body)

    def test_missing_key_and_invalid_request_do_not_count_as_attempts(self):
        client = JSONModel(settings())
        with patch.dict(os.environ, {'POC_TEST_KEY': ''}), self.assertRaises(LLMError):
            client.ask('Use JSON.', {}, 3)
        for remaining in (0, -1, True, float('nan'), float('inf')):
            with self.subTest(remaining=remaining), self.assertRaises(LLMError):
                client.ask('Use JSON.', {}, remaining)
        for payload in ({'x': float('nan')}, [], {1: 'value'}, {'x': set()}):
            with self.subTest(payload=payload), self.assertRaises(LLMError):
                client.ask('Use JSON.', payload, 3)
        self.assertEqual(client.calls, 0)

    def test_timeout_is_bounded_by_remaining_budget_and_usage_resets(self):
        client = JSONModel(settings())
        raw = json.dumps(response(usage={'total_tokens': 10, 'prompt_tokens': True, 'completion_tokens': -1})).encode()
        with patch.object(client, '_request', return_value=raw) as request:
            client.ask('Use JSON.', {}, 0.75)
        self.assertEqual(request.call_args.args[2], 0.75)
        self.assertEqual(client.last_usage, {'total_tokens': 10})
        with patch.object(client, '_request', side_effect=LLMError('Timed out.')):
            with self.assertRaises(LLMError): client.ask('Use JSON.', {}, 1)
        self.assertEqual(client.last_usage, {})
        self.assertEqual(client.calls, 2)

    def test_settings_reject_plaintext_nonlocal_http_and_invalid_values(self):
        for change in ({'base_url': 'http://remote.example/v1'}, {'api_key_env': 'secret value'},
                       {'temperature': float('nan')}, {'max_tokens': True}, {'timeout_s': 0},
                       {'max_retries': -1}, {'json_mode': 1}, {'secret': 'no'}):
            config = settings(); config.update(change)
            with self.subTest(change=change), self.assertRaises((ValueError, LLMError)):
                JSONModel(config)


if __name__ == '__main__':
    unittest.main()
