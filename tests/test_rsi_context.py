"""Long agent runs keep bounded prompts while preserving local evidence."""
import copy
import json
from unittest import TestCase
from ascend_poc.rsi import _model_state


class ContextTests(TestCase):
    def test_large_history_is_compacted_without_mutating_evidence(self):
        record = {'id': 'experiment-0000', 'candidate': {'profile_id': 'baseline', 'parameters': {'batch': 2}},
                  'source_generation': 0, 'reference': {'profile_id': 'baseline', 'parameters': {'batch': 1}},
                  'status': 'completed', 'hypothesis': 'test', 'expected_effect': 'test', 'lesson': 'x' * 1500,
                  'result': {'evaluation': {'feasible': False, 'objective': 4,
                             'checks': ([{'metric': 'throughput', 'pass': True, 'value': 1000}] * 150
                                        + [{'metric': 'latency', 'pass': False, 'value': 1000}] * 50)},
                             'verification': {}, 'analysis': {'violations': [{'metric': 'latency'}] * 100}}}
        state = {'experiments': [dict(copy.deepcopy(record), id=f'experiment-{i:04d}') for i in range(100)],
                 'generation': 4, 'champion': {'profile_id': 'baseline', 'parameters': {'batch': 2},
                                              'experiment_id': 'experiment-0000'}, 'steps': 1,
                 'memory': {'records': [{'id': i, 'lesson': 'y' * 1500} for i in range(12)], 'priors': []},
                 'research': [{'answer': 'z' * 16000}] * 12,
                 'last_result': {'tool': 'inspect', 'ok': True, 'result': {'duplicate': 'q' * 120000}}}
        before = copy.deepcopy(state)
        value = _model_state(state, {'max_steps': 100, 'max_experiments': 100}, 40)
        self.assertLess(len(json.dumps(value)), 80000)
        self.assertEqual(value['experiments'][0]['id'], 'experiment-0000')
        self.assertEqual(len(value['experiments']), 7)
        self.assertEqual(value['experiments'][0]['evaluation']['check_count'], 200)
        self.assertEqual(len(value['experiments'][0]['evaluation']['checks']), 12)
        self.assertEqual(value['experiments'][0]['evaluation']['checks'][0]['metric'], 'latency')
        self.assertNotIn('result', value['last_result'])
        self.assertEqual([r['id'] for r in value['memory']['records']], list(range(6)))
        self.assertEqual(state, before)

        # A large legal parameter catalog may still force entire old views out;
        # the current champion's experiment must survive that second-stage trim.
        for experiment in state['experiments']:
            experiment['candidate']['parameters'] = {f'knob_{i}': 'x' * 600 for i in range(30)}
        value = _model_state(state, {'max_steps': 100, 'max_experiments': 100}, 40)
        self.assertLess(len(json.dumps(value)), 80000)
        self.assertIn('experiment-0000', [item['id'] for item in value['experiments']])
