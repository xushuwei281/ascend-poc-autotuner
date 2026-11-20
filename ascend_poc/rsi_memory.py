"""Bounded, fingerprint-scoped experience hints for configuration-level RSI.

The controller must verify an experiment's journal before calling ``remember``.
This store does not independently authenticate measurements. Its checksum finds
accidental edits; a same-user process can rewrite the file and its checksum.
Even independently confirmed historical records are *hints*, never a substitute
for the current campaign's measurements, acceptance checks or promotion gate.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

from .config import _json_object, _reject_constant
from .storage import atomic_json
from .strategy import workflow_hash


_MAX_BYTES = 8 * 1024 * 1024
_MAX_RECORDS = 1000
_SCOPE = 'same_fingerprint_hints_only'
_HEX = re.compile(r'^[0-9a-f]{64}$')
_ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$')
_PROFILE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$')
_PARAM = re.compile(r'^[A-Za-z][A-Za-z0-9_]{0,63}$')


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False)


def _hash(value) -> str:
    return hashlib.sha256(_canonical(value).encode('utf-8')).hexdigest()


def _namespace(value: str) -> str:
    if not isinstance(value, str) or not _HEX.fullmatch(value):
        raise ValueError('Memory namespace must be a lowercase SHA-256 fingerprint')
    return value


def memory_namespace(spec: dict, runtime: dict, frozen: list) -> str:
    """Use the full workflow, runtime and frozen-file fingerprints conservatively.

    The workflow hash includes budgets and LLM settings. Changing these also
    isolates memory, rather than guessing which changes preserve comparability.
    Runtime/frozen contents are hashed locally; their paths and environment
    hashes are not exported by ``recall_memory``.
    """
    if not isinstance(runtime, dict) or not isinstance(frozen, list):
        raise ValueError('Memory namespace requires runtime object and frozen-file list')
    return _hash({'workflow_hash': workflow_hash(spec), 'runtime': runtime, 'frozen': frozen})


def _object(value, keys, label, optional=()):
    if not isinstance(value, dict) or not set(keys) <= set(value) or set(value) - set(keys) - set(optional):
        raise ValueError(f'{label} has missing or unsupported fields')


def _identifier(value, pattern, label):
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(f'{label} is invalid')


def _number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _parameters(value):
    if not isinstance(value, dict) or not value or len(value) > 64:
        raise ValueError('Memory parameters must be a nonempty bounded object')
    for key, scalar in value.items():
        _identifier(key, _PARAM, 'Memory parameter name')
        if type(scalar) is bool or _number(scalar):
            continue
        if isinstance(scalar, str) and scalar and len(scalar) <= 1000 and '\x00' not in scalar:
            continue
        raise ValueError('Memory parameters must be finite bounded scalars')


def _validate_record(record):
    _object(record, {'id', 'campaign_id', 'profile_id', 'parameters', 'reference',
                     'verification', 'objective'}, 'Memory record',
            {'lesson', 'observed_feasible', 'violated_metrics'})
    _identifier(record['id'], _ID, 'Memory record id')
    _identifier(record['campaign_id'], _ID, 'Memory campaign id')
    _identifier(record['profile_id'], _PROFILE, 'Memory profile id')
    _parameters(record['parameters'])
    reference = record['reference']
    _object(reference, {'profile_id', 'parameters'}, 'Memory reference')
    _identifier(reference['profile_id'], _PROFILE, 'Memory reference profile id')
    _parameters(reference['parameters'])
    verification = record['verification']
    _object(verification, {'mode', 'verdict', 'valid', 'accepted', 'improved'}, 'Memory verification')
    if verification['mode'] not in ('aba', 'single'):
        raise ValueError('Memory verification mode is invalid')
    if verification['verdict'] not in ('ACCEPTED', 'REJECTED', 'INVALID', 'PENDING'):
        raise ValueError('Memory verification verdict is invalid')
    if any(type(verification[field]) is not bool for field in ('valid', 'accepted', 'improved')):
        raise ValueError('Memory verification flags must be booleans')
    if verification['accepted'] != (verification['verdict'] == 'ACCEPTED'):
        raise ValueError('Memory accepted flag contradicts its verdict')
    if verification['accepted'] and not verification['valid']:
        raise ValueError('Memory cannot accept invalid measurements')
    if verification['improved'] and not (verification['mode'] == 'aba'
                                       and verification['accepted'] and verification['valid']):
        raise ValueError('Memory improvement requires valid accepted ABA verification')
    if verification['verdict'] in ('INVALID', 'PENDING') and verification['valid']:
        raise ValueError('Memory verification validity contradicts its verdict')
    objective = record['objective']
    if objective is not None and not _number(objective):
        raise ValueError('Memory objective must be finite or unknown')
    if verification['accepted'] and objective is None:
        raise ValueError('Memory accepted measurements require an objective')
    if verification['mode'] == 'aba' and verification['accepted'] and objective <= 0:
        raise ValueError('Memory accepted ABA objective must be positive')
    if 'observed_feasible' in record:
        feasible = record['observed_feasible']
        if feasible is not None and type(feasible) is not bool:
            raise ValueError('Memory observed feasibility must be a boolean or unknown')
        if feasible is False and verification['accepted']:
            raise ValueError('Memory observed failure contradicts accepted verification')
    if 'violated_metrics' in record:
        metrics = record['violated_metrics']
        if not isinstance(metrics, list) or len(metrics) > 32:
            raise ValueError('Memory violated metrics must be a list of at most 32 identifiers')
        for metric in metrics:
            _identifier(metric, _ID, 'Memory violated metric')
        if len(set(metrics)) != len(metrics):
            raise ValueError('Memory violated metric identifiers must be unique')
        if metrics and record.get('observed_feasible') is not False:
            raise ValueError('Memory violated metrics require observed infeasibility')
    if 'lesson' in record:
        lesson = record['lesson']
        if not isinstance(lesson, str) or not lesson.strip() or len(lesson) > 1500 or '\x00' in lesson:
            raise ValueError('Memory lesson must be nonempty bounded text')
    # No caller-owned objects are retained or mutated.
    return copy.deepcopy(record)


@contextmanager
def _lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + '.lock')
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NONBLOCK | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError('Memory lock must be a regular file')
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another process owns the experience memory') from None
        yield
    finally:
        os.close(fd)


def _load(path: Path):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    except FileNotFoundError:
        return {'schema_version': 1, 'records': []}
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_BYTES:
            raise ValueError('Memory must be a regular file no larger than 8 MiB')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            raw = stream.read(_MAX_BYTES + 1)
        if len(raw) > _MAX_BYTES:
            raise ValueError('Memory exceeds 8 MiB')
    finally:
        os.close(fd)
    value = json.loads(raw, object_pairs_hook=_json_object, parse_constant=_reject_constant)
    _object(value, {'schema_version', 'records', 'checksum'}, 'Memory file')
    if type(value['schema_version']) is not int or value['schema_version'] != 1:
        raise ValueError('Memory schema version is invalid')
    if not isinstance(value['records'], list) or len(value['records']) > _MAX_RECORDS:
        raise ValueError('Memory record count exceeds its limit')
    checksum = value.pop('checksum')
    if not isinstance(checksum, str) or checksum != _hash(value):
        raise ValueError('Experience memory checksum mismatch')
    seen = set()
    for item in value['records']:
        _object(item, {'namespace', 'record'}, 'Memory entry')
        _namespace(item['namespace'])
        _validate_record(item['record'])
        identity = (item['namespace'], item['record']['id'])
        if identity in seen:
            raise ValueError('Memory contains duplicate experiment identities')
        seen.add(identity)
    return value


def _evidence_level(record):
    verification = record['verification']
    if verification['mode'] == 'aba' and verification['valid'] and verification['accepted']:
        return 'independently_confirmed'
    # Invalid infrastructure/reference measurements are not evidence against a
    # strategy. Keep their details, but do not count them as candidate rejection.
    if verification['verdict'] == 'REJECTED':
        return 'rejected'
    return 'observed'


def remember_experiment(path: Path, namespace: str, record: dict) -> dict:
    """Store one controller-verified result; same-ID conflicting writes fail closed.

    At most 1,000 records are retained globally, oldest first. Evicted records
    are no longer available for deduplication. Keep the campaign journal as the
    authoritative complete history. Lessons are untrusted model hypotheses.
    """
    namespace = _namespace(namespace)
    clean = _validate_record(record)
    path = Path(path).absolute()
    with _lock(path):
        value = _load(path)
        previous = next((item['record'] for item in value['records']
                         if item['namespace'] == namespace and item['record']['id'] == clean['id']), None)
        if previous is not None:
            if _canonical(previous) != _canonical(clean):
                raise ValueError('Conflicting memory record for the same experiment identity')
            return {'stored': False, 'record': clean, 'scope': _SCOPE}
        value['records'].append({'namespace': namespace, 'record': clean})
        value['records'] = value['records'][-_MAX_RECORDS:]
        value['checksum'] = _hash(value)
        # Match atomic_json's readable encoding when enforcing the actual cap.
        if len((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')) > _MAX_BYTES:
            raise ValueError('Experience memory exceeds 8 MiB; use a separate memory file')
        atomic_json(path, value)
    return {'stored': True, 'record': clean, 'scope': _SCOPE}


def recall_memory(path: Path, namespace: str, limit: int = 12) -> dict:
    """Return only matching, bounded hints and descriptive profile counts.

    Counts describe retained experiments, not a calibrated probability model.
    Priors cover profiles present in the recalled records and count all retained
    matching experiments for those profiles.
    Confirmation labels describe historical evidence, not today's acceptance.
    """
    namespace = _namespace(namespace)
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError('Memory recall limit must be an integer from 1 to 100')
    path = Path(path).absolute()
    with _lock(path):
        value = _load(path)
    matching = [item['record'] for item in value['records'] if item['namespace'] == namespace]
    priors = {}
    for record in matching:
        profile = priors.setdefault(record['profile_id'], {'profile_id': record['profile_id'],
            'observed_count': 0, 'observed_rejected_count': 0, 'rejected_count': 0,
            'confirmed_count': 0, 'improved_count': 0})
        profile['observed_count'] += 1
        # A measured SLO failure is useful even when the candidate never reaches
        # ABA confirmation. Keep it distinct from an independently judged result.
        if record.get('observed_feasible') is False:
            profile['observed_rejected_count'] += 1
        level = _evidence_level(record)
        if level == 'rejected':
            profile['rejected_count'] += 1
        if level == 'independently_confirmed':
            profile['confirmed_count'] += 1
            if record['verification']['improved']:
                profile['improved_count'] += 1
    records = [dict(copy.deepcopy(record), evidence_level=_evidence_level(record))
               for record in reversed(matching[-limit:])]
    recalled_profiles = {record['profile_id'] for record in records}
    return {'records': records, 'priors': [priors[key] for key in sorted(recalled_profiles)], 'scope': _SCOPE}
