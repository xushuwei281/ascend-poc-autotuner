"""Finite, resumable exploration followed by independent full-matrix confirmation."""
from __future__ import annotations

import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import time

from .config import config_hash, validate_candidate
from .budget import confirmation_cost, preflight, search_cutoff, trial_cost
from .llm import LLMError, LLMPlanner
from .metrics import evaluate_trial
from .provenance import runtime_signature
from .report import rank_key, write_report
from .runner import RunnerError, run_trial
from .storage import atomic_json, experiment_lock


def parameter_hash(parameters: dict) -> str:
    return hashlib.sha256(json.dumps(parameters, sort_keys=True, allow_nan=False).encode()).hexdigest()


def unseen_candidate(config: dict, trials: list[dict]) -> dict | None:
    """Sample legal configurations; bounded enumeration handles small exhausted spaces."""
    seen = {parameter_hash(t['parameters']) for t in trials if t.get('parameters')}
    rng = random.Random(41 + len(trials))
    space = config['search_space']
    for _ in range(1000):
        candidate = {key: rng.choice(values) for key, values in space.items()}
        try:
            validate_candidate(candidate, config)
        except ValueError:
            continue
        if parameter_hash(candidate) not in seen:
            return candidate
    combinations = 1
    for values in space.values():
        combinations *= len(values)
    if combinations <= 100_000:
        for values in itertools.product(*space.values()):
            candidate = dict(zip(space, values))
            try:
                validate_candidate(candidate, config)
            except ValueError:
                continue
            if parameter_hash(candidate) not in seen:
                return candidate
    return None


def tune(config: dict, output: Path, *, planner_name: str = 'llm', resume: bool = False) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with experiment_lock(output, config.get('resource_id', 'ascend-default')):
        return _tune(config, output, planner_name, resume)


def _tune(config: dict, output: Path, planner_name: str, resume: bool) -> dict:
    fingerprint = config_hash(config)
    runtime = runtime_signature(config)
    path = output / 'state.json'
    now = time.time()
    if path.exists():
        if not resume:
            raise ValueError('Output already contains an experiment; use --resume or a new --output')
        state = json.loads(path.read_text(encoding='utf-8'))
        if state['config_hash'] != fingerprint or state['planner'] != planner_name:
            raise ValueError('Cannot resume with a different contract or planner')
        if state.get('runtime_signature') != runtime:
            raise ValueError('Cannot resume after workdir, referenced environment, commands, packages or controller code changed')
        for trial in state['trials']:
            if trial['status'] == 'running':
                trial['status'] = 'interrupted'
        if state.get('status') == 'completed':
            write_report(config, state, output)
            return state
    else:
        if resume:
            raise ValueError('--resume requires an existing state.json')
        state = {'schema_version': 1, 'config_hash': fingerprint, 'planner': planner_name, 'runtime_signature': runtime,
                 'started_epoch': now, 'deadline_epoch': now + config['budget']['max_wall_time_s'],
                 'elapsed_s': 0, 'status': 'running', 'trials': [], 'events': [], 'llm_calls': 0}
        public_config = {k: v for k, v in config.items() if not k.startswith('_')}
        atomic_json(output / 'contract.json', public_config)
    if planner_name == 'llm' and not os.getenv(config['llm']['api_key_env']):
        raise ValueError(f"Set the API key environment variable {config['llm']['api_key_env']} before tuning")
    planner = LLMPlanner(config) if planner_name == 'llm' else None
    remaining = max(0.0, state['deadline_epoch'] - time.time())
    deadline = time.monotonic() + remaining
    # Reserve the last quarter of the original wall-time budget for confirmation.
    search_remaining = max(0.0, state['started_epoch'] + config['budget']['max_wall_time_s'] * .75 - time.time())
    legacy_search_deadline = min(deadline, time.monotonic() + search_remaining)
    search_deadline = legacy_search_deadline

    def persist():
        state['elapsed_s'] = max(0, time.time() - state['started_epoch'])
        atomic_json(path, state)
        write_report(config, state, output)

    def measure(parameters: dict, hypothesis: str, phase: str, repeats: int, limit: float):
        trial_id = f"{len(state['trials']):04d}-{phase}"
        record = {'id': trial_id, 'phase': phase, 'status': 'running',
                  'parameters': parameters, 'hypothesis': hypothesis, 'started_epoch': time.time()}
        state['trials'].append(record)
        persist()
        try:
            options = {'phase': phase} if phase == 'exploration' and config['budget'].get('early_stop', False) else {}
            data = run_trial(config, parameters, output / 'trials' / trial_id, limit, repeats, **options)
            record['scenario_results'] = data
            record['evaluation'] = evaluate_trial(config, data, expected_repetitions=repeats)
            record['status'] = getattr(data, 'status', 'completed')
            if getattr(data, 'timings', None):
                record['timings'] = data.timings
            if getattr(data, 'aborted_after', None):
                record['aborted_after'] = data.aborted_after
        except (RunnerError, ValueError, OSError) as error:
            record['status'] = 'failed'
            # Fixed diagnostic category, not raw process/API output that could contain secrets.
            record['error_type'] = type(error).__name__
            record['error'] = str(error) if isinstance(error, RunnerError) else 'Invalid or unavailable experiment result; inspect local trial logs'
        except BaseException:
            record['status'] = 'interrupted'
            raise
        finally:
            record['ended_epoch'] = time.time()
            persist()
        ev = record.get('evaluation') or {}
        print(f"{trial_id}: {record['status']} feasible={ev.get('feasible', False)} objective={ev.get('objective')}", flush=True)

    state['status'] = 'running'
    persist()
    try:
        while sum(t['phase'] == 'exploration' for t in state['trials']) < config['budget']['max_trials'] and time.monotonic() < deadline:
            explored = [t for t in state['trials'] if t['phase'] == 'exploration']
            search_deadline, reserve = search_cutoff(deadline, legacy_search_deadline, state['trials'],
                config['budget'], {'mode': 'single'}, None, [None] * config['budget']['finalists'])
            if time.monotonic() >= search_deadline:
                if reserve:
                    state['events'].append({'type': 'confirmation_budget_reserved', 'estimate': reserve})
                break
            if not explored:
                parameters = dict(config['baseline'])
                hypothesis = 'Frozen baseline'
            else:
                parameters = None
                hypothesis = 'Bounded random exploration of legal, unseen configurations'
                if planner:
                    try:
                        suggestion = planner.propose(state['trials'], search_deadline - time.monotonic())
                        proposed = validate_candidate(suggestion['parameters'], config)
                        if parameter_hash(proposed) not in {parameter_hash(t['parameters']) for t in explored}:
                            parameters = proposed
                            hypothesis = suggestion['hypothesis']
                        else:
                            state['events'].append({'type': 'duplicate_proposal', 'after_trial': explored[-1]['id']})
                    except (LLMError, ValueError) as error:
                        state['events'].append({'type': 'llm_fallback', 'error_type': type(error).__name__, 'after_trial': explored[-1]['id']})
                        print('LLM proposal unavailable/invalid; using a legal unseen fallback candidate.', flush=True)
                    finally:
                        state['llm_calls'] += planner.calls
                        planner.calls = 0
                        if planner.last_usage:
                            state.setdefault('llm_usage', []).append(planner.last_usage)
                if parameters is None:
                    parameters = unseen_candidate(config, explored)
                if parameters is None:
                    state['events'].append({'type': 'no_unseen_candidate'})
                    break
            if time.monotonic() >= search_deadline:
                break
            if explored and config['budget'].get('confirmation_reserve', False):
                cost = trial_cost(state['trials'], None, config['budget']['repetitions'], config['budget'])
                if cost['seconds'] is not None and cost['seconds'] * config['budget'].get('cost_safety_factor', 1.25) > search_deadline - time.monotonic():
                    state['events'].append({'type': 'exploration_skipped_insufficient_budget', 'estimate': cost})
                    break
            measure(parameters, hypothesis, 'exploration', config['budget']['repetitions'], search_deadline)
        qualified = [t for t in state['trials'] if t['phase'] == 'exploration' and t['status'] == 'completed' and t['evaluation']['feasible']]
        qualified.sort(key=lambda t: rank_key(t, config['objective']['direction']), reverse=True)
        confirmed_hashes = {parameter_hash(t['parameters']) for t in state['trials'] if t['phase'] == 'confirmation' and t['status'] != 'interrupted'}
        for candidate in qualified[:config['budget']['finalists']]:
            if time.monotonic() >= deadline:
                break
            if parameter_hash(candidate['parameters']) in confirmed_hashes:
                continue
            if config['budget'].get('confirmation_reserve', False):
                estimate = confirmation_cost(state['trials'], config['budget'], {'mode': 'single'}, None, [None])
                estimate = preflight(deadline - time.monotonic(), estimate)
                state['events'].append({'type': 'verification_budget_preflight', 'candidate_id': candidate['id'], 'estimate': estimate})
                if not estimate['can_start']:
                    state['events'].append({'type': 'skipped_insufficient_budget', 'candidate_id': candidate['id'], 'estimate': estimate})
                    persist()
                    continue
            measure(candidate['parameters'], f"Independent repeat confirmation of {candidate['id']}", 'confirmation', config['budget']['final_repetitions'], deadline)
        state['status'] = 'completed'
    except KeyboardInterrupt:
        state['status'] = 'interrupted'
        raise
    finally:
        persist()
    return state
