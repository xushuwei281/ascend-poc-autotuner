# Working on Ascend PoC Autotuner

## Purpose

Find reproducible inference configurations that optimize the customer's stated
objective while preserving conditions the customer explicitly specifies.
The recommended `autotune` mode compiles natural-language/text requirements,
discovers the installed engine's parameters, and lets the Agent search unspecified
inference settings. Do not silently inherit quality, precision, workload or model
locks from a baseline template. The older `agent/workflow/tune` modes retain their
fixed-contract, finite-domain behavior for compatibility. EvolvingInfra is
the primary reference for workload awareness and runtime adaptation;
ResearchBoost main supplies the goal-driven experiment process; its bench branch
supplies independent evaluation and evidence records. The Agent improves bounded
inference configurations and experiment experience, not its model weights.
This project does not currently port either project's accelerator engine.

## Start here

- Read `README.md` for commands and supported scope.
- Read `docs/open-tuning.md` for the new natural-language `autotune` mode,
  installed-capability discovery, runtime wiring and explicit-only constraints.
- Read `docs/tuning-explained.md` and `docs/quickstart.md` for the beginner-facing
  explanation and reproduction path. Keep their capability claims aligned with
  the actual example profiles. Regenerate `docs/assets/*.svg` with
  `python3 scripts/render_docs_diagrams.py` when changing illustrated behavior.
- Read `docs/rsi.md` for Agent actions, promotion, memory, budgets and recovery.
- Read `docs/guide.md` for the underlying contracts, execution and test adapters.
- Read `docs/architecture.md` for module boundaries and audit invariants.
- Read `docs/references.md` before proposing online engine changes.
- `examples/` contains illustrative configurations, not customer acceptance evidence.

## Code map

| Area | Files |
|---|---|
| Open request compilation and API | `open_intent.py`, `open_model.py` |
| Open capabilities and runtime wiring | `open_capabilities.py`, `open_runtime.py` |
| Open experiment loop and independent score verification | `open_tuner.py`, `open_verification.py` |
| Local synthetic open-mode demonstration | `open_demo.py` |
| Persistent Agent and budgets | `rsi.py`, `rsi_config.py` |
| Agent decisions, experiments and memory | `rsi_planner.py`, `rsi_experiment.py`, `rsi_memory.py` |
| Frozen contracts and strategy catalog | `config.py`, `strategy.py` |
| Research and slow-loop strategy selection | `deepwiki.py`, `a2a.py`, `rsi_research.py`, `workflow_planner.py` |
| Measurement and acceptance | `runner.py`, `metrics.py`, `verification.py`, `confirmation.py` |
| Confirmation costs and optional block statistics | `budget.py`, `statistics_ci.py` |
| Orchestration and audit | `workflow.py`, `audit.py`, `provenance.py` |
| Historical reporting | `reporting.py`, `report.py` |
| Agent request replay and observation | `replay.py`, `workload.py`, `diagnosis.py` |
| P/D capacity predictions only | `adaptive.py` |
| Local synthetic Agent demonstration | `rsi_demo.py` |

Paths in the table are under `ascend_poc/`.

## Implementation boundaries

- In open mode, compile only explicitly stated customer conditions as locks or
  metric gates, preserve source quotes, and surface unresolved requirements.
  Baseline values are starting points, not implicit customer constraints.
  Actual resource inventory and trusted execution wiring remain real limits.
  Model asset precision metadata does not prove runtime compute or KV precision;
  compile explicit requirements into the appropriate engine locks, or mark
  ambiguous requests unresolved.
- The LLM owns open-mode diagnosis, parameter/strategy choices and technical synthesis.
  Do not substitute a numerical optimizer or fixed strategy sequence for this role.
  Every changed candidate needs a fresh MCP or A2A-to-MCP query of vLLM-Ascend
  documentation and implementation code, followed by LLM review of the exact diff.
  Bind source evidence and review to candidate/reference hashes; a revised candidate
  needs a new review. Missing evidence, unresolved compatibility, or exhausted
  review budgets must not execute that candidate, including on_error=continue.
  Source review permits an experiment, never substitutes for real measurement.
- In legacy modes, preserve the original fixed workload, precision/quality rules
  and hard gates across candidates. Do not change historical evidence semantics.
- Keep the evaluator and metric definitions fixed in all modes. Open-mode changes
  to workload, model or precision must be visible alongside scores; distinguish
  different operating points from same-condition engine acceleration. Without a
  measured quality gate, do not claim quality acceptance.
- Improvement promotion requires fresh independent acceptance and improvement over
  the current generation. In open mode, an initial recipe that cannot be measured
  or fails explicit customer gates may be replaced by a bootstrap qualified
  baseline. All repetitions in each of its three independent starts must pass.
  A higher score from the unqualified reference must not block this bootstrap;
  do not claim acceleration against unqualified or failed measurements. Never
  reuse stale reference evidence. Zero/signed open scores use a labelled normalized absolute difference,
  not a percentage or speedup ratio.
- Keep generation monotonic, including rollback. Rollback changes the next
  experiment reference and exports; it never operates a production deployment.
- Memory is untrusted candidate guidance, not reusable acceptance. Keep measurement
  protocol, runtime and frozen-input namespaces separate; lessons cannot rewrite
  metrics. Search settings can be excluded only field by field; repetition counts
  and verification rules belong to the measurement scope. Resume remains strict.
- Bound tool steps, model/research calls, experiments and the absolute deadline.
  Resume pending work without resetting budgets or manufacturing success. Open
  mode supports complete checkpoints after graceful interruption; do not claim
  arbitrary kill/power-loss recovery when the evidence checkpoint is incomplete.
- Cost estimates distinguish startup from repeat-dependent measurements. A skipped
  confirmation is not a rejected or accepted independent result. Exploration early
  stop preserves partial evidence and never applies to references or confirmation.
- In `aba_ci`, preserve the planned block count and comparison cap, use independent
  service-start blocks rather than within-start repetitions as the statistical
  samples, and retain both within-block and between-block drift checks. Report
  log-ratio estimates separately from the customer's original objective values.
- Deduplicate integrity work only within one check; reread on later checks. Keep
  authoritative state bound to its trace checkpoint and revoke exports on errors.
- An ignore-memory run must neither read nor write persistent experience. Its own
  current-campaign measurements remain visible, and the mode is frozen on resume.
- Keep commands operator-authored and executed as argv, without a shell.
- Legacy LLMs select allowed, verified profiles. Open-mode candidates may use
  all discovered inference options and JSON configuration values, subject to
  explicit user constraints, actual resources and runtime wiring; compatibility
  comes from measured startup/test feedback rather than a manual profile label.
  Research text cannot add executable programs or alter the evaluator.
- Do not treat synthetic acceptance, force routing or fabricated measurements as
  normal-model inference performance. Missing user quality thresholds are not
  permission to mislabel a synthetic benchmark as real model behavior.
- Never turn absent cache or token telemetry into a measured zero.
- Use actual reported token usage; character counts are not token counts.
- Dynamic P/D roles, in-flight migration and agent-aware KV eviction require
  an implemented backend protocol and hardware tests. A launch flag or metadata
  header does not establish those capabilities.
- `adaptive-plan` remains prediction-only. Its supplied calibration is not
  automatically verified and cannot establish customer SLO compliance.
- Do not put secrets, customer prompts, logs, weights or `runs/` in commits.
  API credentials belong in environment variables; the planner key is stripped
  from measurement subprocesses.

## Verify changes

Run targeted tests while developing, then the relevant integration checks:

```bash
python -m unittest discover -s tests -v
python -m ascend_poc autotune-demo --output runs/fresh-open-demo
python -m ascend_poc autotune-report --output runs/fresh-open-demo
python -m ascend_poc agent-validate --config examples/inference-rsi.json
python -m ascend_poc agent-demo --output runs/fresh-rsi-demo
python -m ascend_poc agent-report --output runs/fresh-rsi-demo
python -m ascend_poc workflow-validate --config examples/ascend-agent-workflow.json
python -m ascend_poc adaptive-plan --config examples/adaptive-shadow.json --windows examples/adaptive-windows.json --output runs/shadow.json
python -m ascend_poc workflow-demo --output runs/fresh-demo
```

Use a fresh output directory for demos. Report exactly which checks ran and
distinguish local synthetic / HTTP-fixture tests from real Ascend measurements.
Before hardware execution, inspect the installed stack and the operator's
model, device, dataset and required evaluation interfaces. For open mode, do not
require a preapproved profile for every new candidate; record actual probe failures
and successes. Keep the
original deadline when resuming a campaign.
