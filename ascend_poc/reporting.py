"""Regenerate historical reports from saved, validated experiment snapshots.

Reporting never connects to a model, research service or accelerator. Workflow
snapshots are checked against their stored fingerprint and checkpoint journal;
the current machine need not match the machine that produced the measurements.
"""
from __future__ import annotations

import copy
import fcntl
import json
from pathlib import Path
import tempfile

from .audit import read_trace
from .config import _json_object, _reject_constant, config_hash, load_config
from .report import write_report
from .strategy import load_workflow, workflow_hash
from .workflow import _checkpoint_hash, write_workflow_report


def _read_object(path: Path) -> dict:
    with path.open(encoding='utf-8') as stream:
        value = json.load(stream, object_pairs_hook=_json_object, parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError('Saved report input must be a JSON object')
    return value


def _saved_workflow(output: Path, state: dict) -> dict:
    snapshot = _read_object(output / 'workflow.json')
    contract = _read_object(output / 'contract.json')
    original_reference = snapshot.get('contract')
    # Validate using the saved contract, never the original customer path. The
    # snapshot preserves that original reference as part of its fingerprint.
    local_snapshot = copy.deepcopy(snapshot)
    local_snapshot['contract'] = 'contract.json'
    with tempfile.TemporaryDirectory(prefix='ascend-poc-report-') as directory:
        root = Path(directory)
        (root / 'workflow.json').write_text(json.dumps(local_snapshot, allow_nan=False), encoding='utf-8')
        (root / 'contract.json').write_text(json.dumps(contract, allow_nan=False), encoding='utf-8')
        spec = load_workflow(root / 'workflow.json')
    spec['contract'] = original_reference
    if not isinstance(original_reference, str) or not original_reference:
        raise ValueError('Saved workflow has an invalid contract reference')
    if workflow_hash(spec) != state.get('workflow_hash'):
        raise ValueError('Saved workflow or contract does not match the experiment fingerprint')

    trace = read_trace(output / 'events.jsonl')
    anchor = state.get('trace_anchor')
    checkpoint = next((item for item in trace if isinstance(anchor, dict)
                       and item['sequence'] == anchor.get('sequence')
                       and item['hash'] == anchor.get('hash')), None)
    if (checkpoint is None or checkpoint['event'].get('type') != 'checkpoint'
            or checkpoint['event'].get('state_hash') != _checkpoint_hash(state)):
        raise ValueError('Workflow trace does not match its saved checkpoint')
    if state.get('integrity_valid') is not True:
        raise ValueError('Workflow inputs were invalidated; cannot regenerate an accepted result')
    if spec['verification']['mode'] in {'aba', 'aba_ci'}:
        from .confirmation import revalidate_confirmation
        contract = dict(spec['_contract'], budget=spec['budget'])
        for trial in state['trials']:
            stored = trial.get('verification', {})
            if trial['phase'] != 'confirmation' or stored.get('verdict') in {None, 'PENDING'}:
                continue
            actual = revalidate_confirmation(contract, state['trials'], trial,
                                             spec['verification'], spec['profiles'][0]['id'])
            if actual != stored:
                raise ValueError('Saved independent ABA/CI verdict does not match raw measurements')
    return spec


def regenerate_report(output: Path) -> Path:
    """Detect workflow or legacy tune output, validate it, and return its report.

    The output lock prevents reporting from overwriting a live controller's
    exports. Historical runtime signatures are not rechecked: this command
    summarizes existing evidence and does not re-certify today's environment.
    """
    output = Path(output).resolve()
    if not output.is_dir():
        raise ValueError('Report output directory does not exist')
    with (output / '.run.lock').open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another process owns this experiment output') from None
        workflow_state = output / 'workflow_state.json'
        tune_state = output / 'state.json'
        if workflow_state.exists() and tune_state.exists():
            raise ValueError('Ambiguous output contains both workflow and tune state')
        if workflow_state.exists():
            state = _read_object(workflow_state)
            spec = _saved_workflow(output, state)
            write_workflow_report(spec, state, output)
            return output / 'workflow_report.md'
        if tune_state.exists():
            config = load_config(output / 'contract.json')
            state = _read_object(tune_state)
            if config_hash(config) != state.get('config_hash'):
                raise ValueError('Saved contract does not match the experiment fingerprint')
            write_report(config, state, output)
            return output / 'report.md'
        raise ValueError('No workflow_state.json or state.json found in output directory')
