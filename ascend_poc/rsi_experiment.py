"""Independent ABA experiments for the bounded inference RSI agent.

The agent supplies only catalog identities and finite parameters. This adapter
owns the child workflow, freezes the customer evaluator, and never invokes a
second LLM or pretends its local execution facts came from DeepWiki.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import time

from .config import config_hash, validate_candidate
from .diagnosis import analyze_trial
from .metrics import evaluate_trial
from .reporting import _read_object, _saved_workflow
from .storage import atomic_json
from .strategy import eligible_profiles, load_workflow, materialize_profile, workflow_hash
from .verification import evaluate_aba
from .workflow import _checkpoint_hash, run_workflow
from .workflow_planner import _analysis_summary


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _public(value):
    return {key: copy.deepcopy(item) for key, item in value.items() if not key.startswith('_')}


def _selection(spec, value):
    if not isinstance(value, dict) or set(value) != {'profile_id', 'parameters'}:
        raise ValueError('Experiment selection requires only profile_id and parameters')
    if not isinstance(value['profile_id'], str):
        raise ValueError('Experiment profile_id must be a string')
    config = materialize_profile(spec, value['profile_id'])
    return {'profile_id': value['profile_id'],
            'parameters': validate_candidate(value['parameters'], config)}


def _positive_seconds(value, name):
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or value < .01):
        raise ValueError(f'{name} must be finite and at least 0.01 seconds')
    return float(value)


def _protocol(spec, candidate, reference, budget):
    configs = {}
    for label, selection in (('candidate', candidate), ('reference', reference)):
        config = materialize_profile(spec, selection['profile_id'], selection['parameters'])
        config['budget'] = copy.deepcopy(budget)
        configs[label] = config
    verification = dict(copy.deepcopy(spec['verification']), mode='aba')
    protocol = {'schema_version': 1, 'source_workflow_hash': workflow_hash(spec),
                'candidate': candidate, 'reference': reference,
                'candidate_config_hash': config_hash(configs['candidate']),
                'reference_config_hash': config_hash(configs['reference']),
                'budget': copy.deepcopy(budget), 'verification': verification,
                'workdir': str(Path(spec['_contract'].get('_repo_dir', Path.cwd())).resolve())}
    protocol['input_hash'] = _hash(protocol)
    return protocol, configs


def _write_protocol(spec, candidate, reference, output, budget):
    protocol, configs = _protocol(spec, candidate, reference, budget)
    metadata = {profile['id']: profile for profile in eligible_profiles(spec)}
    profiles = []
    for label, choice in (('reference', reference), ('candidate', candidate)):
        original = metadata[choice['profile_id']]
        profile = {'id': label, 'description': f'Operator-verified {label} recipe',
                   'enabled': True, 'compatibility': copy.deepcopy(original['compatibility']),
                   'requires_quality': original.get('requires_quality', False),
                   'quality_metric': original.get('quality_metric', 'accuracy')}
        if label == 'candidate':
            profile.update({key: copy.deepcopy(configs[label][key])
                            for key in ('server', 'search_space', 'baseline', 'constraints')})
        profiles.append(profile)
    child = {'schema_version': 1, 'name': 'RSI independent experiment ' + protocol['input_hash'],
             'contract': 'experiment-contract.json', 'budget': budget,
             'verification': protocol['verification'], 'profiles': profiles,
             # No network request is made: _LocalFacts supplies honest local facts.
             'research': {'transport': 'mcp', 'repository': 'vllm-project/vllm-ascend',
                          'max_calls': 1, 'on_error': 'stop'}}
    atomic_json(output / 'experiment-input.json', protocol)
    atomic_json(output / 'experiment-contract.json', _public(configs['reference']))
    atomic_json(output / 'experiment-workflow.json', child)
    normalized = load_workflow(output / 'experiment-workflow.json')
    # Keep exact normalized source snapshots for independent readback validation.
    atomic_json(output / 'experiment-workflow.json', _public(normalized))
    return protocol, normalized


class _FixedPlanner:
    calls = 0
    last_usage = {}

    def __init__(self, parameters):
        self.parameters = copy.deepcopy(parameters)

    def propose(self, history, analysis, evidence, remaining_s):
        return {'profile_id': 'candidate', 'parameters': copy.deepcopy(self.parameters),
                'hypothesis': 'Measure the parent agent selection against the fixed incumbent with independent ABA verification',
                'evidence_ids': [item['id'] for item in evidence]}


class _LocalFacts:
    def ask(self, repository, question, remaining_s=None):
        return {'repository': repository, 'sources': [], 'transport': 'local',
                'tool': 'fixed_experiment_protocol',
                'answer': 'Local execution fact: the parent agent selected an operator-verified catalog recipe. '
                          'This child only measures it against the incumbent. No DeepWiki query or external '
                          'compatibility claim is made by this local protocol.'}


def run_experiment(spec, candidate, reference, output, remaining_s, experiment_timeout_s, *, resume=False):
    """Run one fixed candidate against the incumbent, never change the evaluator.

    A child tests the reference and candidate, then independently verifies the
    best feasible exploration result. If that is the reference, the candidate
    has no accepted confirmation and cannot be promoted. Resume retains the
    original child deadline and refuses a shortened global budget it cannot fit.
    """
    candidate, reference = _selection(spec, candidate), _selection(spec, reference)
    if candidate == reference:
        raise ValueError('Candidate is identical to the reference')
    remaining_s = _positive_seconds(remaining_s, 'remaining_s')
    experiment_timeout_s = _positive_seconds(experiment_timeout_s, 'experiment_timeout_s')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / 'experiment-input.json'
    if resume:
        protocol = _read_object(protocol_path)
        expected, _ = _protocol(spec, candidate, reference, protocol.get('budget'))
        if protocol != expected:
            raise ValueError('Cannot resume after experiment inputs or working directory changed')
        if not (output / 'workflow_state.json').exists():
            raise ValueError('Experiment initialization did not complete; cannot restart its deadline')
        # Validate saved evidence before allowing run_workflow to resume it.
        read_experiment(output)
        child_state = _read_object(output / 'workflow_state.json')
        if child_state['status'] != 'completed' and child_state['deadline_epoch'] - time.time() > remaining_s + .01:
            raise ValueError('Remaining global budget cannot contain the original experiment deadline')
        normalized = load_workflow(output / 'experiment-workflow.json')
    else:
        if any(output.iterdir()):
            raise ValueError('Experiment output is not empty; use a new directory or resume')
        budget = dict(spec['budget'], max_trials=2, finalists=1,
                      max_wall_time_s=min(remaining_s, experiment_timeout_s))
        protocol, normalized = _write_protocol(spec, candidate, reference, output, budget)
    normalized['_contract']['_repo_dir'] = protocol['workdir']
    run_workflow(normalized, output, resume=resume,
                 planner=_FixedPlanner(candidate['parameters']), researcher=_LocalFacts())
    return read_experiment(output)


def read_experiment(output):
    """Read historical evidence and independently derive the candidate verdict.

    No hardware, API, or report export is touched. The original selection is
    bound into the workflow fingerprint by input_hash; the parent can bind the
    returned state_hash in its own journal before considering promotion.
    """
    output = Path(output).resolve()
    protocol = _read_object(output / 'experiment-input.json')
    fields = {'schema_version', 'source_workflow_hash', 'candidate', 'reference',
              'candidate_config_hash', 'reference_config_hash', 'budget', 'verification',
              'workdir', 'input_hash'}
    if set(protocol) != fields or protocol['schema_version'] != 1:
        raise ValueError('Malformed experiment protocol')
    if protocol['input_hash'] != _hash({key: value for key, value in protocol.items() if key != 'input_hash'}):
        raise ValueError('Experiment protocol hash does not match')
    state = _read_object(output / 'workflow_state.json')
    child = _saved_workflow(output, state)
    if child['name'] != 'RSI independent experiment ' + protocol['input_hash']:
        raise ValueError('Experiment protocol is not bound to the saved workflow')
    if (child['verification'] != protocol['verification'] or child['verification']['mode'] != 'aba'
            or child['budget'] != protocol['budget']
            or child['budget']['max_trials'] != 2 or child['budget']['finalists'] != 1
            or [profile['id'] for profile in child['profiles']] != ['reference', 'candidate']):
        raise ValueError('Saved experiment verification policy changed')
    if (_read_object(output / 'experiment-workflow.json') != _read_object(output / 'workflow.json')
            or _read_object(output / 'experiment-contract.json') != _read_object(output / 'contract.json')):
        raise ValueError('Experiment source snapshot changed')
    for label in ('candidate', 'reference'):
        config = materialize_profile(child, label)
        if (config_hash(config) != protocol[label + '_config_hash']
                or config['baseline'] != protocol[label]['parameters']):
            raise ValueError('Experiment configuration does not match the selected recipe')
    candidates = []
    for trial in state['trials']:
        label = trial['profile_id']
        if label not in ('candidate', 'reference') or trial['parameters'] != protocol[label]['parameters']:
            raise ValueError('Experiment measured an unplanned configuration')
        if label == 'candidate':
            if trial['phase'] not in ('exploration', 'confirmation'):
                raise ValueError('Candidate cannot be an ABA reference trial')
            candidates.append(trial)
        elif trial['phase'] not in ('exploration', 'confirmation', 'reference_before', 'reference_after'):
            raise ValueError('Invalid reference trial phase')
    confirmation = next((trial for trial in reversed(candidates) if trial['phase'] == 'confirmation'
                         and trial.get('verification', {}).get('verdict') != 'PENDING'), None)
    latest = confirmation or (candidates[-1] if candidates else None)
    evaluation = None
    if latest and latest['status'] == 'completed':
        repeats = child['budget']['final_repetitions'] if latest['phase'] == 'confirmation' else child['budget']['repetitions']
        evaluation = evaluate_trial(child['_contract'], latest.get('scenario_results', []), expected_repetitions=repeats)
        if evaluation != latest.get('evaluation'):
            raise ValueError('Stored experiment evaluation does not match its measurements')
    verification = {'mode': 'aba', 'verdict': 'INVALID', 'valid': False,
                    'accepted': False, 'improved': False,
                    'reasons': ['Candidate has no completed independent ABA confirmation']}
    if confirmation:
        recorded = confirmation['verification']
        references = recorded.get('reference_trials')
        if not isinstance(references, list) or len(references) != 2:
            raise ValueError('Candidate confirmation has invalid reference identities')
        before, after = [next((trial for trial in state['trials'] if trial['id'] == identifier), None)
                         for identifier in references]
        for trial, phase in ((before, 'reference_before'), (after, 'reference_after')):
            if trial and (trial['profile_id'] != 'reference' or trial['phase'] != phase):
                raise ValueError('Candidate confirmation used an invalid reference')
        verification = evaluate_aba(dict(child['_contract'], budget=child['budget']),
                                    before, confirmation, after, child['verification'])
        verification['reference_trials'] = references
        if verification != recorded:
            raise ValueError('Stored experiment verdict does not match independent ABA validation')
    if state['status'] != 'completed':
        verification = dict(verification, verdict='INVALID', valid=False, accepted=False, improved=False,
                            reasons=verification['reasons'] + ['Experiment campaign did not complete'])
    # Re-derive only allowlisted observations; persisted logs, prompts and error
    # text are never copied to the parent model's decision context.
    analysis = (_analysis_summary(analyze_trial(candidates[-1], child['_contract']), child['_contract'])
                if candidates else {})
    return {'candidate': copy.deepcopy(protocol['candidate']), 'reference': copy.deepcopy(protocol['reference']),
            'input_hash': protocol['input_hash'], 'source_workflow_hash': protocol['source_workflow_hash'],
            'workflow_hash': state['workflow_hash'], 'state_hash': _checkpoint_hash(state),
            'status': state['status'], 'candidate_status': latest['status'] if latest else 'not_measured',
            'evaluation': evaluation, 'verification': verification, 'analysis': analysis,
            'confirmed_objective': evaluation['objective'] if confirmation and verification['accepted'] and evaluation else None}
