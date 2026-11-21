"""Offline tests of exact-candidate research and model review boundaries."""
import copy
import json
from unittest import TestCase
from unittest.mock import patch

from ascend_poc.open_research import (REPOSITORY, build_review_question,
                                     candidate_changes, make_researcher, validate_review)
from ascend_poc.rsi_config import digest


DOC = 'https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/graph_mode.html'
CODE = 'https://github.com/vllm-project/vllm-ascend/blob/v0.11.0/vllm_ascend/worker/model_runner_v1.py#L100'
REPO_DOC = 'https://github.com/vllm-project/vllm-ascend/blob/v0.11.0/docs/source/user_guide/feature_guide/graph_mode.md'


class OpenResearchTests(TestCase):
    def setUp(self):
        self.reference = {'model': 'model-bf16', 'engine': {
            'tensor-parallel-size': 2,
            'compilation-config': {'cudagraph_mode': 'NONE', 'old_option': None}},
            'env': {}, 'workload': {'concurrency': 32}}
        self.candidate = copy.deepcopy(self.reference)
        self.candidate['engine']['compilation-config'] = {'cudagraph_mode': 'PIECEWISE'}
        self.evidence = {'id': 'research-0001', 'repository': REPOSITORY,
                         'answer': 'Fixture review of documented graph settings and their implementation.',
                         'sources': [DOC, CODE], 'transport': 'mcp', 'truncated': False}
        self.review = {'decision': 'run', 'reason': 'The cited sources support testing this candidate.',
                       'checked_changes': [row['path'] for row in candidate_changes(self.reference, self.candidate)],
                       'sources': [DOC, CODE], 'uncertainties': [], 'evidence_id': 'research-0001'}
        self.runtime = {'hardware': {'device_type': 'Ascend A2', 'device_count': 2, 'devices': ['0', '1']},
                        'models': [{'id': 'model-bf16', 'family': 'Qwen', 'precision': 'bf16',
                                    'path': '/private/customer/model'}],
                        'server': {'command': ['private-command'], 'api-key': 'secret-key'}}
        self.intent = {'objective': {'metric': 'output_tokens_per_second', 'direction': 'maximize',
                                     'scenario_aggregation': 'mean'},
                       'metric_requirements': {'p99_ttft_ms': {'max': 2000}},
                       'constraints': [{'scope': 'engine', 'key': 'tensor-parallel-size',
                                        'operator': 'eq', 'value': 2, 'source_quote': 'PRIVATE SOURCE QUOTE'}],
                       'summary': 'PRIVATE SUMMARY', 'objective_evidence': 'PRIVATE ORIGINAL REQUEST'}

    def test_transport_factory_defaults_to_existing_mcp_client(self):
        with patch('ascend_poc.open_research.DeepWikiClient') as client:
            result = make_researcher({})
        client.assert_called_once_with('https://mcp.deepwiki.com/mcp', timeout_s=60)
        self.assertIs(result, client.return_value)

    def test_transport_factory_reuses_a2a_client_and_endpoint(self):
        with patch('ascend_poc.open_research.A2AResearchClient') as client:
            result = make_researcher({'transport': 'a2a', 'endpoint': 'http://127.0.0.1:8766/a2a',
                                      'timeout_s': 12})
        client.assert_called_once_with('http://127.0.0.1:8766/a2a', timeout_s=12)
        self.assertIs(result, client.return_value)

    def test_invalid_transport_or_missing_a2a_endpoint_fails(self):
        for config in ({'transport': 'http'}, {'transport': []}, {'transport': 'a2a'}, []):
            with self.subTest(config=config), self.assertRaises(ValueError):
                make_researcher(config)

    def test_diff_includes_nested_removal_and_preserves_explicit_null(self):
        self.assertEqual(candidate_changes(self.reference, self.candidate), [
            {'path': '/engine/compilation-config/cudagraph_mode', 'operation': 'replace',
             'before': 'NONE', 'after': 'PIECEWISE'},
            {'path': '/engine/compilation-config/old_option', 'operation': 'remove', 'before': None}])
        self.assertEqual(candidate_changes({}, {'engine': {'x': None}}), [
            {'path': '/engine/x', 'operation': 'add', 'after': None}])

    def test_diff_handles_arrays_empty_objects_types_and_json_pointer_escaping(self):
        before = {'a/b~c': [1, 2], 'bool': True, 'removed': {}}
        after = {'a/b~c': [1, 3], 'bool': 1, 'added': {}}
        self.assertEqual(candidate_changes(before, after), [
            {'path': '/a~1b~0c', 'operation': 'replace', 'before': [1, 2], 'after': [1, 3]},
            {'path': '/added', 'operation': 'add', 'after': {}},
            {'path': '/bool', 'operation': 'replace', 'before': True, 'after': 1},
            {'path': '/removed', 'operation': 'remove', 'before': {}}])

    def test_diff_is_stable_and_does_not_mutate_candidates(self):
        original = copy.deepcopy(self.candidate)
        first = candidate_changes(self.reference, self.candidate)
        reordered = dict(reversed(list(self.candidate.items())))
        self.assertEqual(first, candidate_changes(self.reference, reordered))
        first[0]['after'] = 'changed externally'
        self.assertEqual(self.candidate, original)
        self.assertEqual(candidate_changes(original, self.candidate), [])

    def test_diff_rejects_nonfinite_or_non_json_values(self):
        for value in (float('nan'), float('inf'), {1: 'not a JSON key'}, {'x'}, 'nul\x00'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                candidate_changes({}, {'bad': value})

    def test_question_preserves_goal_constraints_changes_and_installed_versions(self):
        packages = {'vllm': '0.11.0', 'vllm-ascend': '0.11.0', 'torch': '2.7.1', 'torch-npu': None}
        question = build_review_question(self.runtime, self.intent, self.reference, self.candidate, packages)
        payload = json.loads(question.split('\n', 1)[1])
        self.assertEqual(payload['customer_objective'], self.intent['objective'])
        self.assertEqual(payload['customer_metric_requirements'], self.intent['metric_requirements'])
        self.assertEqual(payload['customer_constraints'][0], {
            'scope': 'engine', 'key': 'tensor-parallel-size', 'operator': 'eq', 'value': 2})
        self.assertEqual(payload['candidate'], self.candidate)
        self.assertEqual(payload['reference'], self.reference)
        self.assertEqual(payload['changes'], candidate_changes(self.reference, self.candidate))
        self.assertEqual(payload['installed_packages'], packages)
        self.assertIn('documentation AND implementation code', question)
        self.assertIn('unknown', question)
        self.assertLessEqual(len(question), 24000)

    def test_question_excludes_quotes_paths_commands_and_unrelated_metadata(self):
        self.candidate['engine']['draft-model'] = '/private/customer/model'
        question = build_review_question(self.runtime, self.intent, self.reference, self.candidate,
                                         {'vllm': None, 'unrelated': 'PRIVATE PACKAGE METADATA'})
        for private in ('PRIVATE', '/private/customer/model', 'private-command', 'secret-key', 'source_quote'):
            self.assertNotIn(private, question)
        self.assertIn('<local-model-asset>', question)
        payload = json.loads(question.split('\n', 1)[1])
        self.assertIsNone(payload['installed_packages']['vllm'])

    def test_oversized_question_is_rejected_without_silently_truncating_changes(self):
        self.candidate['engine']['large-config'] = 'x' * 24000
        with self.assertRaisesRegex(ValueError, '24000-character'):
            build_review_question(self.runtime, self.intent, self.reference, self.candidate, {})

    def test_review_is_bound_to_current_evidence_and_exact_candidate_hashes(self):
        result = validate_review(self.review, self.evidence, self.reference, self.candidate)
        self.assertEqual(result['evidence_id'], self.evidence['id'])
        self.assertEqual(result['reference_hash'], digest(self.reference))
        self.assertEqual(result['candidate_hash'], digest(self.candidate))
        result['sources'].append('external mutation')
        self.assertEqual(self.review['sources'], [DOC, CODE])

    def test_review_cannot_omit_changes_including_removals(self):
        self.review['checked_changes'].pop()
        with self.assertRaisesRegex(ValueError, 'every changed parameter'):
            validate_review(self.review, self.evidence, self.reference, self.candidate)

    def test_review_rejects_unknown_changes_and_duplicate_coverage(self):
        self.review['checked_changes'].append('/engine/unrelated')
        with self.assertRaisesRegex(ValueError, 'outside this proposal'):
            validate_review(self.review, self.evidence, self.reference, self.candidate)
        self.review['checked_changes'][-1] = self.review['checked_changes'][0]
        with self.assertRaisesRegex(ValueError, 'duplicates'):
            validate_review(self.review, self.evidence, self.reference, self.candidate)

    def test_old_evidence_or_invented_source_cannot_authorize_execution(self):
        self.review['evidence_id'] = 'research-old'
        with self.assertRaisesRegex(ValueError, 'different research evidence'):
            validate_review(self.review, self.evidence, self.reference, self.candidate)
        self.review['evidence_id'] = self.evidence['id']
        self.review['sources'].append(REPO_DOC)
        with self.assertRaisesRegex(ValueError, 'absent from current research'):
            validate_review(self.review, self.evidence, self.reference, self.candidate)

    def test_run_requires_both_documentation_and_implementation(self):
        for sources in ([DOC], [CODE], [], ['https://deepwiki.com/vllm-project/vllm-ascend']):
            with self.subTest(sources=sources):
                self.evidence['sources'] = self.review['sources'] = sources
                with self.assertRaisesRegex(ValueError, 'documentation and implementation'):
                    validate_review(self.review, self.evidence, self.reference, self.candidate)

    def test_repository_documentation_is_valid_alongside_code(self):
        self.evidence['sources'] = self.review['sources'] = [REPO_DOC, CODE]
        self.assertEqual(validate_review(self.review, self.evidence, self.reference, self.candidate)['decision'], 'run')

    def test_homepages_wrong_repo_untrusted_hosts_and_non_source_files_are_not_code(self):
        invalid_codes = [
            'https://github.com/vllm-project/vllm-ascend',
            'https://github.com/vllm-project/vllm-ascend/tree/main/vllm_ascend',
            'https://github.com/vllm-project/vllm/blob/main/vllm/config.py',
            'https://github.com.evil.example/vllm-project/vllm-ascend/blob/main/engine.py',
            'https://github.com/vllm-project/vllm-ascend/blob/main/README.md',
            'https://github.com/vllm-project/vllm-ascend/blob/main/model.json',
            'https://github.com/vllm-project/vllm-ascend/blob/main/../engine.py',
            'https://user:secret@github.com/vllm-project/vllm-ascend/blob/main/engine.py',
            'http://github.com/vllm-project/vllm-ascend/blob/main/engine.py',
        ]
        for source in invalid_codes:
            with self.subTest(source=source):
                self.evidence['sources'] = self.review['sources'] = [DOC, source]
                with self.assertRaisesRegex(ValueError, 'documentation and implementation'):
                    validate_review(self.review, self.evidence, self.reference, self.candidate)

    def test_run_with_uncertainty_or_truncated_evidence_is_rejected(self):
        self.review['uncertainties'] = ['Installed version compatibility is not established.']
        with self.assertRaisesRegex(ValueError, 'unresolved or truncated'):
            validate_review(self.review, self.evidence, self.reference, self.candidate)
        self.review['uncertainties'] = []
        self.evidence['truncated'] = True
        with self.assertRaisesRegex(ValueError, 'unresolved or truncated'):
            validate_review(self.review, self.evidence, self.reference, self.candidate)

    def test_non_run_decisions_preserve_model_reason_and_unknowns_without_full_coverage(self):
        for decision in ('reject', 'needs_evidence'):
            with self.subTest(decision=decision):
                self.review.update(decision=decision, checked_changes=[], sources=[],
                                   uncertainties=['No version-matched implementation source was found.'])
                checked = validate_review(self.review, self.evidence, self.reference, self.candidate)
                self.assertEqual(checked['decision'], decision)
                self.assertEqual(checked['uncertainties'], self.review['uncertainties'])

    def test_review_schema_does_not_allow_replacement_candidate_or_model_authored_hash(self):
        for key in ('candidate', 'candidate_hash', 'reference_hash'):
            with self.subTest(key=key):
                altered = dict(self.review, **{key: self.candidate})
                with self.assertRaisesRegex(ValueError, 'required schema'):
                    validate_review(altered, self.evidence, self.reference, self.candidate)

    def test_wrong_repository_empty_evidence_and_invalid_decision_fail_closed(self):
        for key, value in (('repository', 'another/repository'), ('id', ''), ('answer', '')):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_review(self.review, dict(self.evidence, **{key: value}), self.reference, self.candidate)
        self.review['decision'] = 'probably-run'
        with self.assertRaisesRegex(ValueError, 'run, reject or needs_evidence'):
            validate_review(self.review, self.evidence, self.reference, self.candidate)
