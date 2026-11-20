"""Tool-using inference RSI: measured configurations evolve; the judge stays fixed.

The agent selects experiments and remembers outcomes. A separate workflow owns
measurement and ABA adjudication. No tool can edit code, acceptance conditions,
model weights, or a live serving backend.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import time
import uuid

from .audit import append_trace, read_trace, verify_frozen_files
from .llm import LLMError
from .provenance import runtime_signatures
from .report import _cell
from .rsi_config import agent_hash, digest, read_json
from .rsi_experiment import run_experiment, read_experiment
from .rsi_memory import MEASUREMENT_PROTOCOL, memory_namespace, recall_memory, remember_experiment
from .rsi_research import research_question
from .rsi_planner import RSIPlanner, validate_action
from .storage import atomic_json
from .strategy import eligible_profiles, materialize_profile, workflow_hash
from .workflow import _checkpoint_hash, _public
from .workflow_planner import _public_sources


class AgentIntegrityError(RuntimeError):
    """Evidence or frozen inputs changed; all agent exports must be revoked."""


class ToolRejected(ValueError):
    """Controller-authored explanation safe to return to the planning model."""


@contextmanager
def _output_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    # The independent child workflow owns the accelerator resource lock. Keeping
    # that same lock here would deadlock; this lock only protects agent state.
    with (output / '.agent.lock').open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another agent owns this output directory') from None
        yield


def _runtime(spec):
    return runtime_signatures({p['id']: materialize_profile(spec, p['id'])
                               for p in eligible_profiles(spec)})


def _frozen(spec):
    return verify_frozen_files(spec['verification']['frozen_files'],
                               Path(spec['_contract'].get('_repo_dir', Path.cwd())))


def _revoke(output):
    for name in ('champion.json', 'champion-contract.json'):
        (output / name).unlink(missing_ok=True)


def _load_state(output):
    try:
        return _load_state_checked(output)
    except (ValueError, OSError, KeyError, TypeError, AttributeError) as error:
        raise AgentIntegrityError('Agent snapshots or audit journal are unreadable or invalid') from error


def _load_state_checked(output):
    state = read_json(output / 'agent_state.json')
    manifest = read_json(output / 'agent_manifest.json')
    if digest(manifest) != state.get('manifest_hash'):
        raise AgentIntegrityError('Agent manifest does not match its checkpoint')
    trace = read_trace(output / 'events.jsonl')
    anchor = state.get('trace_anchor', {})
    checkpoint = next((item for item in trace if item['sequence'] == anchor.get('sequence')
                       and item['hash'] == anchor.get('hash')), None)
    if (checkpoint is None or checkpoint['event'].get('type') != 'checkpoint'
            or checkpoint['event'].get('state_hash') != _checkpoint_hash(state)):
        raise AgentIntegrityError('Agent state does not match its audit checkpoint')
    if state.get('integrity_valid') is not True:
        raise AgentIntegrityError('Agent campaign has invalidated evidence')
    return state, manifest


def _verification(value):
    source = value.get('verification', {})
    return {key: source.get(key, False if key in {'valid', 'accepted', 'improved'} else 'INVALID')
            for key in ('mode', 'verdict', 'valid', 'accepted', 'improved')}


def _experiment_view(record):
    evaluation = record.get('result', {}).get('evaluation') or {}
    checks = [{k: c[k] for k in ('scenario', 'repetition', 'metric', 'value', 'passed', 'pass') if k in c}
              for c in evaluation.get('checks', [])[:100]]
    return {'id': record['id'], 'candidate': record['candidate'],
            'source_generation': record['source_generation'], 'reference': record['reference'],
            'status': record['status'], 'hypothesis': record['hypothesis'],
            'expected_effect': record['expected_effect'],
            'evaluation': {'feasible': evaluation.get('feasible', False),
                           'objective': evaluation.get('objective'), 'checks': checks},
            'verification': _verification(record.get('result', {})),
            'analysis': record.get('result', {}).get('analysis', {}),
            'lesson': record.get('lesson'), 'lesson_is_hypothesis': True}


def _validate_record(output, record):
    try:
        actual = read_experiment(output / 'experiments' / record['id'])
    except (ValueError, OSError, RuntimeError) as error:
        raise AgentIntegrityError('Independent experiment evidence is unavailable or changed') from error
    if digest(actual) != record.get('result_hash'):
        raise AgentIntegrityError('Independent experiment no longer matches the agent record')
    return actual


def _model_state(state, settings, remaining_s):
    """Bound long-horizon context; complete observations remain in local records."""
    from .workload import public_summary
    records = state['experiments'][-6:]
    champion_id = state['champion'].get('experiment_id')
    champion_record = next((r for r in state['experiments'] if r['id'] == champion_id), None)
    if champion_record is not None and champion_record not in records:
        records = [champion_record] + records
    views = []
    for record in records:
        view = copy.deepcopy(_experiment_view(record))
        checks = (record.get('result', {}).get('evaluation') or {}).get('checks', [])
        view['evaluation']['check_count'] = len(checks)
        view['evaluation']['checks'] = [
            {k: c[k] for k in ('scenario', 'repetition', 'metric', 'value', 'passed', 'pass') if k in c}
            for c in checks if c.get('passed', c.get('pass')) is False][:12]
        analysis = view['analysis']
        analysis['violations'] = analysis.get('violations', [])[:12]
        analysis['hypotheses'] = analysis.get('hypotheses', [])[:5]
        analysis['workloads'] = [dict(item, summary=public_summary(item.get('summary'), max_windows=2))
                                 for item in analysis.get('workloads', [])[:2]]
        if view.get('lesson'):
            view['lesson'] = view['lesson'][:500]
        views.append(view)
    memory = copy.deepcopy(state['memory'])
    memory['records'] = memory.get('records', [])[:6]  # recall is newest-first
    memory['priors'] = memory.get('priors', [])[:40]
    if len(memory.get('marginals', [])) > 32:
        memory['marginals_truncated'] = True
    memory['marginals'] = memory.get('marginals', [])[:32]
    for record in memory['records']:
        if record.get('lesson'):
            record['lesson'] = record['lesson'][:500]
    evidence = [dict(item, answer=item['answer'][:6000]) for item in state['research'][-2:]]
    # inspect/recall/research results already appear above. Do not send a second
    # full copy of the experiment history or case bank in last_result.
    last = {k: state['last_result'][k] for k in ('tool', 'ok', 'reason', 'error_type')
            if k in state['last_result']}
    result = {'generation': state['generation'], 'champion': state['champion'],
              'steps_remaining': settings['max_steps'] - state['steps'],
              'experiments_remaining': settings['max_experiments'] - len(state['experiments']),
              'remaining_s': remaining_s, 'experiments': views, 'memory': memory,
              'research': evidence, 'last_result': last, 'history_is_summarized': True}
    # Large optional histories cannot make the planner repeatedly exceed its
    # context limit. Never trim the fixed customer acceptance contract itself.
    while len(json.dumps(result, ensure_ascii=False, allow_nan=False)) > 80000:
        if memory['records']:
            memory['records'].pop()
        elif memory['marginals']:
            memory['marginals'].pop()
            memory['marginals_truncated'] = True
        elif evidence:
            evidence.pop(0)
        elif len(views) > 1:
            index = next((i for i, view in enumerate(views) if view['id'] != champion_id), 0)
            views.pop(index)
        else:
            raise ValueError('Single experiment summary is too large for the agent context')
    return result


def _validate_experiments(output, state):
    for record in state['experiments']:
        if record.get('result'):
            _validate_record(output, record)


def _write_report(output, state):
    lines = ['# Inference RSI Agent — 实验与代际报告', '',
             '配置与策略层的受控改进；不修改评测器、源码、模型权重或生产服务。', '',
             f"- 状态：{state['status']}；停止原因：{_cell(state.get('stop_reason', ''))}",
             f"- 当前代际：{state['generation']}；步骤：{state['steps']}；LLM HTTP 尝试：{state['llm_calls']}",
             f"- 研究调用：{state['research_calls']}；实验：{len(state['experiments'])}",
             f"- 历史经验模式：{state.get('memory_mode', 'enabled')}；当前任务实验仍保留",
             f"- 输入完整性：{state['integrity_valid']}；独立确认的当前配置：{bool(state['champion'].get('experiment_id'))}",
             '', '| 实验 | 父代 | 策略 | 完成 | 目标 | ABA | 已验收 | 有效提升 |',
             '|---|---:|---|---|---:|---|---|---|']
    ci_lines = []
    for record in state['experiments']:
        view = _experiment_view(record)
        verification = view['verification']
        lines.append('| ' + ' | '.join(_cell(x) for x in [record['id'], record['source_generation'],
                     record['candidate']['profile_id'], record['status'], view['evaluation']['objective'],
                     verification['verdict'], verification['accepted'], verification['improved']]) + ' |')
        result = record.get('result', {}).get('verification', {})
        if result.get('mode') == 'aba_ci':
            ci_lines.append(f"{_cell(record['id'])}：跨块效果点估计 {_cell(result.get('relative_improvement'))}；"
                         f"同时单侧置信下界 {_cell(result.get('relative_improvement_lower_bound'))}；"
                         f"统计假设与独立块证据见子实验 verification.json。")
    if ci_lines:
        lines += ['', *ci_lines]
    lines += ['', '## 到达验收与晋升的成本', '',
              '| 里程碑 | 已启动实验数 | 已用墙钟秒 |', '|---|---:|---:|']
    for name in ('first_observed_feasible', 'first_independent_acceptance', 'first_promotion'):
        milestone = state.get('milestones', {}).get(name, {})
        lines.append('| ' + ' | '.join(_cell(v) for v in (name, milestone.get('experiments', '未达到'),
                                                      milestone.get('elapsed_s', '未达到'))) + ' |')
    lines += ['', '观察达标不等于独立验收；跨任务比较须使用固定预算、负载、环境和独立输出目录。']
    lines += ['', '## 代际记录', '']
    for event in state['lineage']:
        lines.append('- ' + _cell(event['kind']) + f"：generation {event['generation']}；"
                     + _cell(event['champion']['profile_id']) + '；' + _cell(event.get('experiment_id', '')))
    lines += ['', '## 实验反思', '']
    for record in state['experiments']:
        lines += ['- ' + _cell(record['id']) + '：' + _cell(record.get('lesson', '尚无反思'))]
    lines += ['', '反思与历史先验是待验证建议，不是测量证据。只有独立 ABA 有效、全部客户门槛通过、'
              '改善超过噪声门槛且父代仍匹配，才允许晋升。回退只改变后续实验参考和导出配置。',
              '', '初始基线不因被称为 champion 而成为已验收交付。没有晋升时不会生成 champion.json。',
              '事件链检查一致性，不能防止同权限用户重写控制器和完整日志。']
    (output / 'agent_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _export_champion(output, state):
    champion = state['champion']
    if not state['integrity_valid'] or not champion.get('experiment_id'):
        _revoke(output)
        return
    record = next(r for r in state['experiments'] if r['id'] == champion['experiment_id'])
    _validate_record(output, record)
    child = output / 'experiments' / record['id'] / 'best-contract.json'
    # best-contract is a derived artifact, not authoritative evidence. Construct
    # it from the checked child snapshots and the confirmed candidate instead.
    from .reporting import _saved_workflow
    from .rsi_config import read_json as read
    folder = child.parent
    child_state = read(folder / 'workflow_state.json')
    child_spec = _saved_workflow(folder, child_state)
    effective = materialize_profile(child_spec, 'candidate')
    effective['baseline'] = champion['parameters']
    effective['budget'] = dict(effective['budget'], max_trials=1, finalists=1)
    atomic_json(output / 'champion-contract.json', _public(effective))
    atomic_json(output / 'champion.json', {'generation': state['generation'], **champion,
                'verification': record['result']['verification'], 'manifest_hash': state['manifest_hash'],
                'deployment': 'export_only', 'replay_contract': 'champion-contract.json'})


def report_agent(output):
    output = Path(output).resolve()
    if not output.is_dir():
        raise ValueError('Agent output does not exist')
    with _output_lock(output):
        try:
            state, _ = _load_state(output)
            _validate_experiments(output, state)
            _export_champion(output, state)
            _write_report(output, state)
        except (ValueError, OSError, RuntimeError):
            _revoke(output)
            raise
    return output / 'agent_report.md'


def run_agent(config, output, *, resume=False, planner=None, researcher=None, ignore_memory=False):
    output = Path(output).resolve()
    with _output_lock(output):
        try:
            return _run_agent(config, output, resume=resume, planner=planner, researcher=researcher,
                              ignore_memory=ignore_memory)
        except AgentIntegrityError:
            _revoke(output)
            raise


def _run_agent(config, output, *, resume, planner, researcher, ignore_memory):
    spec, settings = config['_workflow'], config['settings']
    try:
        runtime, frozen = _runtime(spec), _frozen(spec)
    except (ValueError, OSError, RuntimeError) as error:
        raise AgentIntegrityError('Agent environment or frozen inputs are unavailable') from error
    fingerprint = agent_hash(config)
    namespace = memory_namespace(spec, runtime, frozen)
    manifest = {'agent_config': _public(config), 'agent_hash': fingerprint,
                'workflow': _public(spec), 'contract': _public(spec['_contract']),
                'runtime': runtime, 'frozen': frozen, 'memory_namespace': namespace,
                'memory_mode': 'disabled' if ignore_memory else 'enabled'}
    state_path = output / 'agent_state.json'
    if state_path.exists():
        if not resume:
            raise ValueError('Agent output exists; use --resume or a new directory')
        state, previous = _load_state(output)
        if previous != manifest:
            raise AgentIntegrityError('Agent configuration, code, environment or frozen files changed')
        _validate_experiments(output, state)
        if state['status'] == 'completed':
            _export_champion(output, state)
            _write_report(output, state)
            return state
    else:
        if resume:
            raise ValueError('--resume requires agent_state.json')
        if read_trace(output / 'events.jsonl'):
            raise AgentIntegrityError('Agent journal exists without its state')
        first = eligible_profiles(spec)[0]['id']
        champion = {'profile_id': first, 'parameters': materialize_profile(spec, first)['baseline'],
                    'experiment_id': None}
        now = time.time()
        state = {'schema_version': 1, 'campaign_id': uuid.uuid4().hex, 'manifest_hash': digest(manifest),
                 'status': 'running', 'integrity_valid': True, 'started_epoch': now,
                 'deadline_epoch': now + settings['max_wall_time_s'], 'steps': 0,
                 'llm_calls': 0, 'research_calls': 0, 'generation': 0, 'champion': champion,
                 'stack': [copy.deepcopy(champion)], 'lineage': [], 'experiments': [],
                 'research': [], 'memory': {'records': [], 'priors': [], 'scope': 'same_fingerprint_hints_only'},
                 'pending': None, 'consecutive_errors': 0, 'last_result': {}}
        state.update(memory_mode=manifest['memory_mode'], milestones={})
        atomic_json(output / 'agent_manifest.json', manifest)
    planner = planner or RSIPlanner(spec, settings)
    if researcher is None:
        research = spec['research']
        if research['transport'] == 'a2a':
            from .a2a import A2AResearchClient
            researcher = A2AResearchClient(research['endpoint'], timeout_s=research['timeout_s'])
        else:
            from .deepwiki import DeepWikiClient
            researcher = DeepWikiClient(research['endpoint'], timeout_s=research['timeout_s'])
    end = time.monotonic() + max(0, state['deadline_epoch'] - time.time())
    memory_path = Path(config['_memory_file'])

    def emit(kind, **data):
        append_trace(output, {'type': kind, 'epoch': time.time(), **data})

    def persist(*, report=False):
        state['elapsed_s'] = max(0, time.time() - state['started_epoch'])
        try:
            anchor = append_trace(output, {'type': 'checkpoint', 'state_hash': _checkpoint_hash(state)})
            state['trace_anchor'] = {'sequence': anchor['sequence'], 'hash': anchor['hash']}
            atomic_json(state_path, state)
            _export_champion(output, state)
            if report:
                _write_report(output, state)
        except (OSError, ValueError, RuntimeError) as error:
            state['integrity_valid'] = False
            _revoke(output)
            raise AgentIntegrityError('Agent checkpoint or evidence could not be validated') from error

    def check_integrity():
        try:
            if (agent_hash(config) != fingerprint or _runtime(spec) != runtime or _frozen(spec) != frozen):
                raise ValueError('Changed inputs')
        except (ValueError, OSError, RuntimeError) as error:
            state['integrity_valid'] = False
            _revoke(output)
            raise AgentIntegrityError('Frozen agent inputs changed') from error

    def remember(record):
        if ignore_memory or record.get('remembered') or not record.get('result'):
            return
        result = _validate_record(output, record)
        evaluation = result.get('evaluation') or {}
        value = {'id': state['campaign_id'] + ':' + record['id'], 'campaign_id': state['campaign_id'],
                 'profile_id': record['candidate']['profile_id'], 'parameters': record['candidate']['parameters'],
                 'reference': record['reference'], 'verification': _verification(result),
                 'objective': evaluation.get('objective'), 'observed_feasible': evaluation.get('feasible'),
                 'source_workflow_hash': workflow_hash(spec), 'measurement_protocol': MEASUREMENT_PROTOCOL,
                 'violated_metrics': sorted({check['metric'] for check in evaluation.get('checks', [])
                     if check.get('passed', check.get('pass')) is False})[:32]}
        if record.get('lesson'):
            value['lesson'] = record['lesson']
        remember_experiment(memory_path, namespace, value)
        record['remembered'] = True

    def milestone(name, epoch=None):
        state.setdefault('milestones', {}).setdefault(name, {
            'experiments': len(state['experiments']),
            'elapsed_s': max(0, (time.time() if epoch is None else epoch) - state['started_epoch'])})

    def find_record(identifier):
        record = next((r for r in state['experiments'] if r['id'] == identifier), None)
        if record is None:
            raise ToolRejected('Unknown experiment ID')
        return record

    def execute(action):
        tool, args = action['tool'], action['arguments']
        if tool == 'inspect':
            return {'generation': state['generation'], 'champion': state['champion'],
                    'experiments': [_experiment_view(r) for r in state['experiments'][-12:]]}
        if tool == 'recall':
            state['memory'] = ({'records': [], 'priors': [], 'marginals': [], 'scope': 'disabled_control'}
                               if ignore_memory else recall_memory(memory_path, namespace))
            return state['memory']
        if tool == 'research':
            if state['research_calls'] >= settings['max_research_calls']:
                raise ToolRejected('Research budget exhausted')
            state['research_calls'] += 1
            persist()  # A killed external request still consumes its reservation.
            question = research_question(spec, state, args['topic'])
            response = researcher.ask(spec['research']['repository'], question,
                                      remaining_s=max(0, end - time.monotonic()))
            if (not isinstance(response, dict) or response.get('repository') != spec['research']['repository']
                    or not isinstance(response.get('answer'), str) or not response['answer'].strip()):
                raise ValueError('Invalid repository research response')
            evidence = {'id': f"research-{state['research_calls']:04d}", 'topic': args['topic'],
                        'answer': response['answer'][:16000], 'sources': _public_sources(response.get('sources', []))}
            state['research'].append(evidence)
            return evidence
        if tool == 'experiment':
            pending = state['pending']
            if 'experiment_id' not in pending:
                if len(state['experiments']) >= settings['max_experiments']:
                    raise ToolRejected('Experiment budget exhausted')
                candidate = {'profile_id': args['profile_id'], 'parameters': args['parameters']}
                reference = {k: state['champion'][k] for k in ('profile_id', 'parameters')}
                if candidate == reference:
                    raise ToolRejected('Candidate equals the current reference')
                if any(r['candidate'] == candidate and r['source_generation'] == state['generation']
                       for r in state['experiments']):
                    raise ToolRejected('Candidate already attempted against this generation')
                identifier = f"experiment-{len(state['experiments']):04d}"
                record = {'id': identifier, 'candidate': candidate, 'reference': reference,
                          'source_generation': state['generation'], 'hypothesis': args['hypothesis'],
                          'expected_effect': args['expected_effect'], 'status': 'running'}
                state['experiments'].append(record)
                pending['experiment_id'] = identifier
                persist()
            record = find_record(pending['experiment_id'])
            folder = output / 'experiments' / record['id']
            result = run_experiment(spec, record['candidate'], record['reference'], folder,
                                    max(0, end - time.monotonic()), settings['experiment_timeout_s'],
                                    resume=(folder / 'workflow_state.json').exists())
            record.update(status=result['status'], result=result, result_hash=digest(result))
            for name in ('first_observed_feasible', 'first_independent_acceptance'):
                epoch = result.get('milestones', {}).get(name + '_epoch')
                if epoch is not None:
                    milestone(name, epoch)
            return _experiment_view(record)
        if tool == 'reflect':
            record = find_record(args['experiment_id'])
            if record['status'] == 'running' or not record.get('result'):
                raise ToolRejected('Reflection requires a completed experiment record')
            if record.get('lesson') and record['lesson'] != args['lesson']:
                raise ToolRejected('Reflection is already recorded; start another experiment to revise evidence')
            record['lesson'] = args['lesson']
            remember(record)
            return {'lesson_status': 'untrusted_hypothesis', 'experiment': _experiment_view(record)}
        if tool == 'promote':
            record = find_record(args['experiment_id'])
            result = _validate_record(output, record)
            verification = result['verification']
            reference = {k: state['champion'][k] for k in ('profile_id', 'parameters')}
            if record['source_generation'] != state['generation'] or record['reference'] != reference:
                raise ToolRejected('Stale candidate must be measured against the current generation')
            if not (verification.get('mode') in {'aba', 'aba_ci'} and verification.get('valid') is True
                    and verification.get('accepted') is True and verification.get('improved') is True):
                raise ToolRejected('Promotion requires independent ABA acceptance and improvement')
            state['generation'] += 1
            state['champion'] = dict(record['candidate'], experiment_id=record['id'])
            state['stack'].append(copy.deepcopy(state['champion']))
            state['lineage'].append({'kind': 'promotion', 'generation': state['generation'],
                                     'champion': copy.deepcopy(state['champion']), 'experiment_id': record['id']})
            milestone('first_promotion')
            return {'promoted': True, 'generation': state['generation'], 'champion': state['champion']}
        if tool == 'rollback':
            if len(state['stack']) <= 1:
                raise ToolRejected('No previous generation is available')
            state['stack'].pop()
            state['generation'] += 1
            state['champion'] = copy.deepcopy(state['stack'][-1])
            state['lineage'].append({'kind': 'rollback', 'generation': state['generation'],
                                     'champion': copy.deepcopy(state['champion']), 'reason': args['reason']})
            return {'rolled_back': True, 'generation': state['generation'], 'deployment': 'export_only'}
        if tool == 'finish':
            state['stop_reason'] = args['reason']
            state['status'] = 'completed'
            return {'finished': True}
        raise ValueError('Unknown tool')

    state['status'] = 'running'
    persist(report=True)
    try:
        while time.monotonic() < end:
            check_integrity()
            if state['pending'] is None:
                if state['steps'] >= settings['max_steps']:
                    state['stop_reason'] = 'max_steps'
                    break
                remaining_calls = settings['max_llm_calls'] - state['llm_calls']
                if remaining_calls <= 0:
                    state['stop_reason'] = 'max_llm_calls'
                    break
                reserved = min(remaining_calls, spec['_contract']['llm']['max_retries'] + 1)
                state['steps'] += 1
                state['llm_calls'] += reserved
                persist()
                planner.call_budget = reserved
                planner.calls = 0
                observation = {'goal': config['goal'],
                               'state': _model_state(state, settings, max(0, end - time.monotonic()))}
                try:
                    action = validate_action(planner.decide(observation, max(0, end - time.monotonic())), spec)
                    state['pending'] = action
                except (LLMError, ValueError) as error:
                    state['consecutive_errors'] += 1
                    state['last_result'] = {'tool': 'planner', 'ok': False, 'error_type': type(error).__name__}
                finally:
                    state['llm_calls'] += planner.calls - reserved
                    if planner.last_usage:
                        state.setdefault('llm_usage', []).append(planner.last_usage)
                    persist()
            if state['pending'] is not None and time.monotonic() < end:
                action = {k: state['pending'][k] for k in ('tool', 'arguments')}
                emit('tool_started', action=action, generation=state['generation'])
                try:
                    result = execute(action)
                    state['last_result'] = {'tool': action['tool'], 'ok': True, 'result': result}
                    state['consecutive_errors'] = 0
                except AgentIntegrityError:
                    raise
                except (ValueError, OSError, RuntimeError) as error:
                    state['consecutive_errors'] += 1
                    state['last_result'] = {'tool': action['tool'], 'ok': False, 'error_type': type(error).__name__,
                        'reason': str(error) if isinstance(error, ToolRejected) else 'Tool failed; local evidence remains untrusted until verified'}
                    if 'experiment_id' in state['pending']:
                        record = find_record(state['pending']['experiment_id'])
                        record['status'] = 'failed'
                        record['error_type'] = type(error).__name__
                state['pending'] = None
                check_integrity()
                emit('tool_finished', tool=action['tool'], ok=state['last_result']['ok'])
                persist(report=True)
            if state['status'] == 'completed':
                break
            if state['consecutive_errors'] >= settings['max_consecutive_tool_errors']:
                state['stop_reason'] = 'consecutive_tool_errors'
                break
        check_integrity()
        # Acceptance evidence is independent of whether persistent experience is
        # enabled, already remembered, or intentionally disabled for a control.
        _validate_experiments(output, state)
        for record in state['experiments']:
            remember(record)
        state.setdefault('stop_reason', 'deadline')
        state['status'] = 'completed'
    except KeyboardInterrupt:
        state['status'] = 'interrupted'
        raise
    except Exception:
        state['status'] = 'failed'
        raise
    finally:
        persist(report=True)
    return state
