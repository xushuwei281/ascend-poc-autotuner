# Working on Ascend PoC Autotuner

## Purpose

Find reproducible inference configurations that satisfy the customer's fixed
workload, quality, latency, reliability and hardware budget. EvolvingInfra is
the primary reference for workload awareness and runtime adaptation;
ResearchBoost supplies ideas for independent evaluation and evidence records.
This project does not currently port either project's accelerator engine.

## Start here

- Read `README.md` for commands and supported scope.
- Read `docs/guide.md` for configuration, execution, reporting and troubleshooting.
- Read `docs/architecture.md` for module boundaries and audit invariants.
- Read `docs/references.md` before proposing online engine changes.
- `examples/` contains illustrative configurations, not customer acceptance evidence.

## Code map

| Area | Files |
|---|---|
| Frozen contracts and strategy catalog | `config.py`, `strategy.py` |
| Research and slow-loop strategy selection | `deepwiki.py`, `a2a.py`, `workflow_planner.py` |
| Measurement and acceptance | `runner.py`, `metrics.py`, `verification.py` |
| Orchestration and audit | `workflow.py`, `audit.py`, `provenance.py` |
| Historical reporting | `reporting.py`, `report.py` |
| Agent request replay and observation | `replay.py`, `workload.py`, `diagnosis.py` |
| P/D capacity predictions only | `adaptive.py` |

Paths in the table are under `ascend_poc/`.

## Implementation boundaries

- Preserve workload, precision/quality rules and hard gates across candidates.
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
python -m ascend_poc workflow-validate --config examples/ascend-agent-workflow.json
python -m ascend_poc adaptive-plan --config examples/adaptive-shadow.json --windows examples/adaptive-windows.json --output runs/shadow.json
python -m ascend_poc workflow-demo --output runs/fresh-demo
```

Use a fresh output directory for demos. Report exactly which checks ran and
distinguish local synthetic / HTTP-fixture tests from real Ascend measurements.
Before hardware execution, inspect the installed stack and the operator's
model, device, dataset, quality-evaluator and compatibility evidence. Keep the
original deadline when resuming a campaign.
