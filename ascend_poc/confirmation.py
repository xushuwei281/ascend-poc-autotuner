"""Reconstruct independent verification from ordered, immutable trial records."""
from __future__ import annotations

from .verification import evaluate_aba, evaluate_aba_ci


def revalidate_confirmation(contract, trials, candidate, settings, reference_profile_id):
    """Never trust a stored verdict or reuse a measurement as another CI block."""
    identifiers = [trial['id'] for trial in trials]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError('Duplicate measurement identity')
    by_id = {trial['id']: trial for trial in trials}
    if by_id.get(candidate.get('id')) is not candidate:
        raise ValueError('Confirmation must be the recorded trial object')
    positions = {identifier: index for index, identifier in enumerate(identifiers)}
    recorded = candidate.get('verification', {})
    mode = settings['mode']
    if recorded.get('mode') != mode or candidate['phase'] != 'confirmation':
        raise ValueError('Confirmation protocol or phase changed')

    def resolve(identifier, role):
        if identifier is None:
            return None
        if not isinstance(identifier, str) or identifier not in by_id:
            raise ValueError('Missing referenced measurement')
        trial = by_id[identifier]
        if role in {'before', 'after'}:
            if (trial['profile_id'] != reference_profile_id
                    or trial['phase'] != 'reference_' + role
                    or trial['parameters'] != contract['baseline']):
                raise ValueError('Invalid reference measurement')
        elif (trial['profile_id'] != candidate['profile_id']
              or trial['parameters'] != candidate['parameters']
              or trial['phase'] not in {'confirmation', 'confirmation_block'}):
            raise ValueError('Invalid candidate block measurement')
        return trial

    def check_order(block):
        available = [positions[block[key]['id']] for key in ('before', 'candidate', 'after') if block[key] is not None]
        if available != sorted(set(available)):
            raise ValueError('Verification measurements are not in A/C/B order')

    if mode == 'aba':
        references = recorded.get('reference_trials')
        if not isinstance(references, list) or len(references) != 2:
            raise ValueError('Invalid ABA reference identities')
        block = {'before': resolve(references[0], 'before'), 'candidate': candidate,
                 'after': resolve(references[1], 'after')}
        check_order(block)
        result = evaluate_aba(contract, block['before'], candidate, block['after'], settings)
        result['reference_trials'] = references
        return result
    if mode != 'aba_ci':
        raise ValueError('Unsupported independent confirmation protocol')
    references = recorded.get('reference_blocks')
    if not isinstance(references, list) or len(references) > settings['ci_blocks']:
        raise ValueError('Invalid CI block identities')
    blocks, seen, previous = [], set(), -1
    for index, item in enumerate(references):
        if not isinstance(item, dict) or set(item) != {'before', 'candidate', 'after'}:
            raise ValueError('Malformed CI block reference')
        block = {role: resolve(item[role], role) for role in ('before', 'candidate', 'after')}
        measured = block['candidate']
        expected_phase = 'confirmation' if index == len(references) - 1 else 'confirmation_block'
        if measured is not None and measured['phase'] != expected_phase:
            raise ValueError('CI candidate has an invalid block/final phase')
        check_order(block)
        for role in ('before', 'candidate', 'after'):
            trial = block[role]
            if trial is None:
                continue
            identifier = trial['id']
            if identifier in seen or positions[identifier] <= previous:
                raise ValueError('CI blocks reuse or reorder measurements')
            seen.add(identifier)
            previous = positions[identifier]
        blocks.append(block)
    if not blocks or blocks[-1]['candidate'] is not candidate:
        raise ValueError('CI verdict is not attached to its final candidate block')
    result = evaluate_aba_ci(contract, blocks, settings)
    result['reference_blocks'] = references
    return result
