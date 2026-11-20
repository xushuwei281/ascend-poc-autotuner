import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from ascend_poc.workload import load_telemetry, public_summary, summarize_requests


def request(identifier='r1', **updates):
    row = {'request_id': identifier, 'started_at_s': 0, 'completed_at_s': 1,
           'input_tokens': 100, 'output_tokens': 20}
    row.update(updates)
    return row


class WorkloadTests(unittest.TestCase):
    def test_missing_cache_is_unknown_and_partial_coverage_is_explicit(self):
        summary = summarize_requests([request(), request('r2', cached_tokens=80)])
        self.assertEqual(summary['cache_observation_coverage'], .5)
        self.assertEqual(summary['cache_unknown_requests'], 1)
        self.assertEqual(summary['overall']['mean_cached_tokens'], 80)
        self.assertEqual(summary['overall']['mean_uncached_tokens'], 20)
        self.assertEqual(summary['overall']['cache_hit_ratio'], .8)
        unknown = summarize_requests([request()])
        self.assertIsNone(unknown['overall']['mean_uncached_tokens'])
        self.assertIsNone(unknown['overall']['cache_hit_ratio'])

    def test_causal_proxy_uses_uncached_query_tokens_and_cached_prefix(self):
        summary = summarize_requests([request(input_tokens=12, cached_tokens=10,
                                              restored_tokens=4, accepted_tokens_per_step=1.5)])
        self.assertEqual(summary['overall']['mean_causal_attention_pairs'], 23)
        self.assertEqual(summary['overall']['mean_restored_tokens'], 4)
        self.assertEqual(summary['overall']['mean_accepted_tokens_per_step'], 1.5)
        self.assertIn('proxy_not_measured', summary['attention_work_semantics'])

    def test_errors_remain_in_counts_but_not_successful_token_means(self):
        summary = summarize_requests([request(), request('failed', input_tokens=900, output_tokens=800,
                                                        cached_tokens=900, status='error')])
        self.assertEqual((summary['request_count'], summary['successful'], summary['failed']), (2, 1, 1))
        self.assertEqual(summary['input_tokens_total'], 100)
        self.assertEqual(summary['output_tokens_total'], 20)
        self.assertEqual(summary['cache_observation_coverage'], 0)
        self.assertIsNone(summary['overall']['cache_hit_ratio'])

    def test_failed_request_can_omit_unknown_token_usage(self):
        row = {'request_id': 'failed', 'started_at_s': 0, 'completed_at_s': 2, 'status': 'error'}
        summary = summarize_requests([row])
        self.assertEqual(summary['failed'], 1)
        self.assertIsNone(summary['overall']['mean_input_tokens'])
        with self.assertRaises(ValueError):
            summarize_requests([dict(row, status='ok')])

    def test_public_summary_removes_untrusted_strings_and_nested_content(self):
        summary = summarize_requests([request(cached_tokens=20)])
        summary.update(prompt='PRIVATE', agent_id='PRIVATE', source='PRIVATE',
                       attention_work_semantics='PRIVATE', branch_count='PRIVATE')
        summary['overall'].update(secret='PRIVATE', mean_output_tokens='PRIVATE')
        summary['windows'][0].update(metadata={'secret': 'PRIVATE'}, start_s='PRIVATE')
        result = public_summary(summary)
        self.assertNotIn('PRIVATE', json.dumps(result))
        self.assertEqual(result['overall']['mean_input_tokens'], 100)
        self.assertNotIn('mean_output_tokens', result['overall'])
        self.assertNotIn('branch_count', result)

    def test_public_summary_samples_first_and_last_windows_with_full_count(self):
        summary = summarize_requests([request(completed_at_s=601)])
        safe = public_summary(summary, max_windows=4)
        self.assertEqual(safe['window_count'], 11)
        self.assertTrue(safe['windows_sampled'])
        self.assertEqual([w['start_s'] for w in safe['windows']], [0, 60, 540, 600])
        self.assertEqual(safe['request_count'], 1)
        self.assertEqual(public_summary(summary, max_windows=0)['windows'], [])
        self.assertEqual(public_summary(safe)['window_count'], 11)
        self.assertTrue(public_summary(safe)['windows_sampled'])

    def test_public_summary_rejects_bad_numbers_without_replacing_unknown_with_zero(self):
        safe = public_summary({'request_count': True, 'failed': -1, 'successful': 1,
                               'cache_observation_coverage': 2,
                               'overall': {'mean_cached_tokens': None, 'mean_input_tokens': math.inf,
                                           'cache_hit_ratio': -1}, 'windows': ['PRIVATE']})
        self.assertNotIn('request_count', safe)
        self.assertNotIn('failed', safe)
        self.assertNotIn('cache_observation_coverage', safe)
        self.assertIsNone(safe['overall']['mean_cached_tokens'])
        self.assertNotIn('mean_input_tokens', safe['overall'])
        self.assertEqual(safe['windows'], [])
        json.dumps(safe, allow_nan=False)

    def test_empty_and_error_only_are_not_successful_observations(self):
        empty = summarize_requests([])
        self.assertEqual(empty['windows'], [])
        self.assertIsNone(empty['observation_start_s'])
        failed = summarize_requests([request(status='error')])
        self.assertEqual(failed['successful'], 0)
        self.assertIsNone(failed['overall']['mean_input_tokens'])
        self.assertIsNone(failed['cache_observation_coverage'])
        json.dumps(failed, allow_nan=False)

    def test_windows_preserve_idle_periods_and_workload_shift(self):
        rows = [request(input_tokens=20, output_tokens=1000, cached_tokens=10),
                request('r2', started_at_s=120, completed_at_s=122, input_tokens=4000,
                        output_tokens=10, cached_tokens=100)]
        result = summarize_requests(rows)
        self.assertEqual(len(result['windows']), 3)
        self.assertEqual(result['windows'][1]['request_count'], 0)
        self.assertEqual(result['windows'][0]['overall']['mean_output_tokens'], 1000)
        self.assertEqual(result['windows'][2]['overall']['mean_uncached_tokens'], 3900)
        self.assertTrue(result['windows'][0]['complete_window'])
        self.assertFalse(result['windows'][2]['complete_window'])
        self.assertEqual(result['observation_end_s'], 122)

    def test_boundaries_are_anchored_at_first_start_and_use_completion(self):
        result = summarize_requests([request(started_at_s=10, completed_at_s=11),
                                     request('r2', started_at_s=12, completed_at_s=70)])
        self.assertEqual(result['windows'][0]['start_s'], 10)
        self.assertEqual(result['windows'][0]['request_count'], 1)
        self.assertEqual(result['windows'][1]['request_count'], 1)

    def test_identifiers_are_never_returned_and_branches_are_namespaced(self):
        rows = [request('SECRET_REQUEST_A', agent_id='SECRET_AGENT_A', branch_id='SECRET_BRANCH', context_version=0),
                request('SECRET_REQUEST_B', started_at_s=2, completed_at_s=4,
                        agent_id='SECRET_AGENT_A', branch_id='SECRET_BRANCH', context_version=1),
                request('SECRET_REQUEST_C', started_at_s=1, completed_at_s=5,
                        agent_id='SECRET_AGENT_B', branch_id='SECRET_BRANCH', context_version=0)]
        result = summarize_requests(rows)
        self.assertNotIn('SECRET', json.dumps(result))
        self.assertEqual(result['active_agent_count'], 2)
        self.assertEqual(result['branch_count'], 2)
        self.assertEqual(result['context_version_changes'], 1)
        self.assertIn('not_live_population', result['agent_count_semantics'])

    def test_context_version_changes_follow_submission_not_completion_order(self):
        rows = [request('later', started_at_s=2, completed_at_s=3,
                        agent_id='agent', branch_id='branch', context_version=1),
                request('first', started_at_s=0, completed_at_s=5,
                        agent_id='agent', branch_id='branch', context_version=0),
                request('last', started_at_s=6, completed_at_s=7,
                        agent_id='agent', branch_id='branch', context_version=1)]
        self.assertEqual(summarize_requests(rows)['context_version_changes'], 1)

    def test_invalid_counts_times_statuses_and_raw_content_fail_closed(self):
        invalid = [{'cached_tokens': 101}, {'restored_tokens': 1}, {'cached_tokens': 10, 'restored_tokens': 11},
                   {'input_tokens': -1}, {'output_tokens': True}, {'input_tokens': 1.2},
                   {'completed_at_s': math.nan}, {'started_at_s': math.inf}, {'started_at_s': 3},
                   {'accepted_tokens_per_step': 0}, {'accepted_tokens_per_step': math.inf},
                   {'context_version': -1}, {'event': 'compaction'}, {'status': 'maybe'},
                   {'status': []}, {'started_at_s': 10**1000},
                   {'messages': ['PRIVATE CUSTOMER TEXT']}, {'request_id': ''}, {'agent_id': None}]
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                summarize_requests([request(**changes)])
        with self.assertRaises(ValueError):
            summarize_requests([request(), request()])
        with self.assertRaises(ValueError):
            summarize_requests([{}])

    def test_limits_and_unrepresentable_windows_are_rejected(self):
        for interval in (0, -1, True, math.nan, math.inf):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                summarize_requests([request()], interval)
        with self.assertRaises(ValueError):
            summarize_requests([request(completed_at_s=600000)])
        with self.assertRaises(ValueError):
            summarize_requests([request(started_at_s=1e300, completed_at_s=1e300)], .1)
        with self.assertRaises(ValueError):
            summarize_requests([request()] * 100001)

    def test_input_is_not_mutated_and_output_is_finite_json(self):
        rows = [request(input_tokens=0, output_tokens=0, cached_tokens=0)]
        old = copy.deepcopy(rows)
        result = summarize_requests(rows)
        self.assertEqual(rows, old)
        self.assertIsNone(result['overall']['cache_hit_ratio'])
        json.dumps(result, allow_nan=False)

    def test_jsonl_loader_rejects_duplicate_keys_nonfinite_oversize_and_raw_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'telemetry.jsonl'
            path.write_text(json.dumps(request()) + '\n', encoding='utf-8')
            self.assertEqual(load_telemetry(path)['successful'], 1)
            with self.assertRaises(ValueError):
                load_telemetry(path, max_bytes=2)
            for text in ('{"request_id":"a","request_id":"b"}',
                         json.dumps(request(completed_at_s=math.nan)),
                         json.dumps(request(prompt='PRIVATE')), 'not-json'):
                path.write_text(text, encoding='utf-8')
                with self.subTest(text=text), self.assertRaises(ValueError):
                    load_telemetry(path)
            path.write_bytes(b'\xff')
            with self.assertRaises(ValueError):
                load_telemetry(path)
            path.write_text('', encoding='utf-8')
            self.assertEqual(load_telemetry(path)['request_count'], 0)


if __name__ == '__main__':
    unittest.main()
