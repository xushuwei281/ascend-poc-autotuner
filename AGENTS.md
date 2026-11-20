# Working on Ascend PoC Autotuner

## Purpose

Find reproducible inference configurations that satisfy the customer's fixed
workload, quality, latency, reliability and hardware budget. EvolvingInfra is
the primary reference for workload awareness and runtime adaptation;
ResearchBoost main supplies the goal-driven experiment process; its bench branch
supplies independent evaluation and evidence records. The Agent improves bounded
inference configurations and experiment experience, not its model weights.
This project does not currently port either project's accelerator engine.

## Start here

- Read `README.md` for commands and supported scope.
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

- Preserve workload, precision/quality rules and hard gates across candidates.
- Promotion requires fresh independent acceptance and improvement over the current
  generation. Never promote a result measured against an obsolete reference.
- Keep generation monotonic, including rollback. Rollback changes the next
  experiment reference and exports; it never operates a production deployment.
- Memory is untrusted candidate guidance, not reusable acceptance. Keep measurement
  protocol, runtime and frozen-input namespaces separate; lessons cannot rewrite
  metrics. Search settings can be excluded only field by field; repetition counts
  and verification rules belong to the measurement scope. Resume remains strict.
- Bound tool steps, model/research calls, experiments and the absolute deadline.
  Resume pending work without resetting budgets or manufacturing success.
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
- LLMs select allowed, verified profiles; research text cannot add executable code.
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
model, device, dataset, quality-evaluator and compatibility evidence. Keep the
original deadline when resuming a campaign.
