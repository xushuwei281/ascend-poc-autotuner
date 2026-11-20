"""Exercise the complete PoC controller with deterministic external measurements.

The expensive hardware runner and API provider are replaced; validation, ranking,
persistence, resume handling and final report export remain real.
"""

import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from ascend_poc.llm import LLMError
from ascend_poc.tuner import tune


def contract(resource_id):
    return {
        "schema_version": 1,
        "name": "Customer acceptance integration test",
        "resource_id": resource_id,
        "context": {"hardware": "CPU test double"},
        "search_space": {"batch": [1, 2, 3]},
        "baseline": {"batch": 1},
        "constraints": [],
        "server": {"command": ["fake-server", "{param.batch}"], "env": {},
                   "health_url": "http://127.0.0.1:8000/health",
                   "startup_timeout_s": 1, "shutdown_timeout_s": 1},
        "scenarios": [
            {"name": name, "command": ["customer-benchmark"],
             "result_file": "{trial_dir}/metrics.json", "metrics_format": "canonical",
             "timeout_s": 1, "requirements": {}}
            for name in ("chat", "long_context")
        ],
        "requirements": {"p99_ttft_ms": {"max": 2000}, "quality_score": {"min": 0.9}},
        "objective": {"metric": "output_tokens_per_second", "direction": "maximize",
                      "scenario_aggregation": "min"},
        "budget": {"max_trials": 3, "max_wall_time_s": 60, "repetitions": 2,
                   "final_repetitions": 3, "finalists": 1},
        "llm": {"base_url": "https://example.invalid/v1", "model": "fake-planner",
                "api_key_env": "POC_INTEGRATION_KEY", "timeout_s": 1,
                "max_tokens": 100, "temperature": 0.2, "json_mode": True, "max_retries": 0},
    }


def measured(config, candidate, repeats, *, latency=500, quality=1.0):
    """Distinct scenario results demonstrate a full acceptance matrix."""
    return [
        {"name": scenario["name"], "metrics": [
            {"output_tokens_per_second": candidate["batch"] * 100 - index * 10,
             "p99_ttft_ms": latency, "quality_score": quality}
            for _ in range(repeats)
        ]}
        for index, scenario in enumerate(config["scenarios"])
    ]


class FakePlanner:
    def __init__(self, parameters=None, events=None, error=None):
        self.parameters = parameters or {"batch": 2}
        self.events = events if events is not None else []
        self.error = error
        self.calls = 0
        self.last_usage = {}

    def propose(self, history, remaining_seconds):
        self.events.append("planner")
        self.calls += 1
        self.last_usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        if self.error:
            raise self.error
        return {"parameters": dict(self.parameters), "hypothesis": "Measure a larger allowed batch."}


class TunerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="poc-tuner-tests-")
        self.root = Path(self.temp.name)
        self.output = self.root / "run"
        self.config = contract(str(self.root))
        self.key = patch.dict(os.environ, {"POC_INTEGRATION_KEY": "test-only-key"})
        self.key.start()

    def tearDown(self):
        self.key.stop()
        self.temp.cleanup()

    def run_tune(self, runner, *, planner=None, config=None, resume=False):
        with patch("ascend_poc.tuner.run_trial", side_effect=runner) as mock_runner, \
             patch("ascend_poc.tuner.LLMPlanner", return_value=planner), redirect_stdout(io.StringIO()):
            result = tune(config or self.config, self.output,
                          planner_name="llm" if planner else "random", resume=resume)
        return result, mock_runner

    def test_baseline_runs_before_llm_and_each_candidate_covers_same_matrix(self):
        self.config["budget"]["max_trials"] = 2
        events = []
        planner = FakePlanner(events=events)

        def runner(config, candidate, trial_dir, deadline, repeats):
            events.append((dict(candidate), [item["name"] for item in config["scenarios"]], repeats))
            return measured(config, candidate, repeats)

        state, mock_runner = self.run_tune(runner, planner=planner)
        self.assertEqual(events[0], ({"batch": 1}, ["chat", "long_context"], 2))
        self.assertEqual(events[1], "planner")
        self.assertEqual(events[-1], ({"batch": 2}, ["chat", "long_context"], 3))
        self.assertEqual(mock_runner.call_count, 3)
        self.assertEqual(state["llm_calls"], 1)
        self.assertEqual(state["llm_usage"][0]["total_tokens"], 15)
        best = json.loads((self.output / "best.json").read_text())
        self.assertEqual(best["parameters"], {"batch": 2})
        self.assertEqual(best["status"], "confirmed")
        self.assertEqual(set(best["evaluation"]["scenario_objectives"]), {"chat", "long_context"})

    def test_a_fast_exploration_winner_that_fails_confirmation_is_not_exported(self):
        self.config["budget"]["max_trials"] = 2

        def runner(config, candidate, trial_dir, deadline, repeats):
            latency = 2500 if trial_dir.name.endswith("confirmation") else 500
            return measured(config, candidate, repeats, latency=latency)

        state, _ = self.run_tune(runner, planner=FakePlanner())
        self.assertTrue(all(trial["evaluation"]["feasible"] for trial in state["trials"] if trial["phase"] == "exploration"))
        self.assertFalse(state["trials"][-1]["evaluation"]["feasible"])
        self.assertFalse((self.output / "best.json").exists())
        self.assertIn("不能将探索候选视为最终验收通过", (self.output / "report.md").read_text())

    def test_failed_first_finalist_can_fall_back_to_a_verified_slower_configuration(self):
        self.config["budget"].update(max_trials=2, finalists=2)

        def runner(config, candidate, trial_dir, deadline, repeats):
            quality = 0.5 if trial_dir.name.endswith("confirmation") and candidate["batch"] == 2 else 1.0
            return measured(config, candidate, repeats, quality=quality)

        state, _ = self.run_tune(runner, planner=FakePlanner())
        self.assertEqual(len([t for t in state["trials"] if t["phase"] == "confirmation"]), 2)
        best = json.loads((self.output / "best.json").read_text())
        self.assertEqual(best["parameters"], {"batch": 1})

    def test_minimization_selects_lowest_latency_with_worst_scenario_aggregation(self):
        self.config["objective"] = {"metric": "p99_ttft_ms", "direction": "minimize", "scenario_aggregation": "max"}

        def runner(config, candidate, trial_dir, deadline, repeats):
            latencies = {1: 800, 2: 400, 3: 600}
            rows = measured(config, candidate, repeats, latency=latencies[candidate["batch"]])
            for metrics in rows[1]["metrics"]:
                metrics["p99_ttft_ms"] += 50
            return rows

        state, _ = self.run_tune(runner)
        best = json.loads((self.output / "best.json").read_text())
        self.assertEqual(best["parameters"], {"batch": 2})
        self.assertEqual(best["evaluation"]["objective"], 450)
        self.assertEqual(state["trials"][-1]["parameters"], {"batch": 2})

    def test_duplicate_llm_proposals_use_legal_unseen_fallback_and_stop(self):
        planner = FakePlanner(parameters={"batch": 1})

        def runner(config, candidate, trial_dir, deadline, repeats):
            return measured(config, candidate, repeats)

        state, _ = self.run_tune(runner, planner=planner)
        exploration = [trial for trial in state["trials"] if trial["phase"] == "exploration"]
        self.assertEqual(len(exploration), 3)
        self.assertEqual({trial["parameters"]["batch"] for trial in exploration}, {1, 2, 3})
        self.assertEqual(sum(event["type"] == "duplicate_proposal" for event in state["events"]), 2)
        self.assertEqual(state["status"], "completed")

    def test_provider_failure_uses_fallback_without_exposing_error_details(self):
        self.config["budget"]["max_trials"] = 2
        planner = FakePlanner(error=LLMError("PRIVATE PROVIDER DETAIL"))

        def runner(config, candidate, trial_dir, deadline, repeats):
            return measured(config, candidate, repeats)

        state, _ = self.run_tune(runner, planner=planner)
        self.assertEqual(sum(t["phase"] == "exploration" for t in state["trials"]), 2)
        self.assertTrue(any(event["type"] == "llm_fallback" for event in state["events"]))
        self.assertNotIn("PRIVATE PROVIDER DETAIL", (self.output / "state.json").read_text())

    def test_missing_api_key_fails_before_starting_any_service(self):
        with patch.dict(os.environ, {"POC_INTEGRATION_KEY": ""}), \
             patch("ascend_poc.tuner.run_trial") as runner, \
             patch("ascend_poc.tuner.LLMPlanner") as planner:
            with self.assertRaisesRegex(ValueError, "API key environment variable"):
                tune(self.config, self.output)
        runner.assert_not_called()
        planner.assert_not_called()
        self.assertFalse((self.output / "best.json").exists())

    def test_resume_rejects_changed_acceptance_contract_before_hardware_access(self):
        self.config["budget"]["max_trials"] = 1

        def runner(config, candidate, trial_dir, deadline, repeats):
            return measured(config, candidate, repeats)

        self.run_tune(runner)
        modified = copy.deepcopy(self.config)
        modified["requirements"]["p99_ttft_ms"]["max"] = 1000
        with patch("ascend_poc.tuner.run_trial") as hardware:
            with self.assertRaisesRegex(ValueError, "different contract"):
                tune(modified, self.output, planner_name="random", resume=True)
        hardware.assert_not_called()

    def test_completed_resume_rejects_changed_working_directory(self):
        self.config["budget"]["max_trials"] = 1
        self.config["_repo_dir"] = str(self.root)

        def runner(config, candidate, trial_dir, deadline, repeats):
            return measured(config, candidate, repeats)

        state, _ = self.run_tune(runner)
        self.assertEqual(state["status"], "completed")
        changed_root = self.root / "another-worktree"
        changed_root.mkdir()
        changed = copy.deepcopy(self.config)
        changed["_repo_dir"] = str(changed_root)
        with patch("ascend_poc.tuner.run_trial") as hardware:
            with self.assertRaisesRegex(ValueError, "Cannot resume after workdir"):
                tune(changed, self.output, planner_name="random", resume=True)
        hardware.assert_not_called()

    def test_completed_resume_rejects_changed_referenced_model_environment(self):
        self.config["budget"]["max_trials"] = 1
        self.config["server"]["command"] = ["fake-server", "{env:MODEL_PATH}", "{param.batch}"]

        def runner(config, candidate, trial_dir, deadline, repeats):
            return measured(config, candidate, repeats)

        first_model = "/private-customer-model-location/model-a"
        with patch.dict(os.environ, {"MODEL_PATH": first_model}):
            state, _ = self.run_tune(runner)
            self.assertEqual(state["status"], "completed")
            _, hardware = self.run_tune(runner, resume=True)
            hardware.assert_not_called()
        persisted = (self.output / "state.json").read_text()
        self.assertIn("MODEL_PATH", persisted)
        self.assertNotIn(first_model, persisted)
        with patch.dict(os.environ, {"MODEL_PATH": "/private-customer-model-location/model-b"}), \
             patch("ascend_poc.tuner.run_trial") as hardware:
            with self.assertRaisesRegex(ValueError, "referenced environment"):
                tune(self.config, self.output, planner_name="random", resume=True)
        hardware.assert_not_called()

    def test_completed_resume_rejects_changed_direct_benchmark_script_content(self):
        self.config["budget"]["max_trials"] = 1
        self.config["_repo_dir"] = str(self.root)
        benchmark = self.root / "customer_benchmark.py"
        benchmark.write_text("print('first')\n", encoding="utf-8")
        self.config["scenarios"][0]["command"] = ["python3", "{repo_dir}/customer_benchmark.py"]

        def runner(config, candidate, trial_dir, deadline, repeats):
            return measured(config, candidate, repeats)

        state, _ = self.run_tune(runner)
        self.assertEqual(state["status"], "completed")
        # Keep size and timestamps fixed: small scripts must be checked by content.
        before = benchmark.stat()
        benchmark.write_text("print('later')\n", encoding="utf-8")
        os.utime(benchmark, ns=(before.st_atime_ns, before.st_mtime_ns))
        with patch("ascend_poc.tuner.run_trial") as hardware:
            with self.assertRaisesRegex(ValueError, "commands, packages or controller code changed"):
                tune(self.config, self.output, planner_name="random", resume=True)
        hardware.assert_not_called()

    def test_no_feasible_configuration_never_creates_best(self):
        def runner(config, candidate, trial_dir, deadline, repeats):
            return measured(config, candidate, repeats, quality=0.7)

        state, hardware = self.run_tune(runner)
        self.assertEqual(hardware.call_count, 3)
        self.assertTrue(all(not trial["evaluation"]["feasible"] for trial in state["trials"]))
        self.assertFalse(any(trial["phase"] == "confirmation" for trial in state["trials"]))
        self.assertFalse((self.output / "best.json").exists())

    def test_search_limit_excludes_but_bounds_final_confirmation(self):
        self.config["search_space"]["batch"] = list(range(1, 21))
        self.config["budget"].update(max_trials=2, finalists=1)

        def runner(config, candidate, trial_dir, deadline, repeats):
            return measured(config, candidate, repeats)

        state, hardware = self.run_tune(runner)
        self.assertEqual(len(state["trials"]), 3)
        self.assertEqual(hardware.call_count, 3)
        self.assertEqual(sum(t["phase"] == "exploration" for t in state["trials"]), 2)
        self.assertEqual(sum(t["phase"] == "confirmation" for t in state["trials"]), 1)

    def test_one_bad_final_repetition_cannot_be_hidden_by_good_median(self):
        self.config["budget"]["max_trials"] = 1

        def runner(config, candidate, trial_dir, deadline, repeats):
            rows = measured(config, candidate, repeats)
            if trial_dir.name.endswith("confirmation"):
                rows[-1]["metrics"][-1]["p99_ttft_ms"] = 2500
            return rows

        state, _ = self.run_tune(runner)
        self.assertFalse(state["trials"][-1]["evaluation"]["feasible"])
        self.assertFalse((self.output / "best.json").exists())

    def test_interruption_is_persisted_and_resume_does_not_claim_interrupted_result(self):
        self.config["budget"]["max_trials"] = 2
        calls = []

        def interrupted_runner(config, candidate, trial_dir, deadline, repeats):
            calls.append(candidate)
            if len(calls) == 2:
                raise KeyboardInterrupt()
            return measured(config, candidate, repeats)

        with self.assertRaises(KeyboardInterrupt):
            self.run_tune(interrupted_runner)
        saved = json.loads((self.output / "state.json").read_text())
        self.assertEqual(saved["status"], "interrupted")
        self.assertEqual(saved["trials"][-1]["status"], "interrupted")
        self.assertNotIn("evaluation", saved["trials"][-1])
        self.assertFalse((self.output / "best.json").exists())

        def good_runner(config, candidate, trial_dir, deadline, repeats):
            return measured(config, candidate, repeats)

        state, resumed_hardware = self.run_tune(good_runner, resume=True)
        self.assertEqual(resumed_hardware.call_count, 1)
        self.assertEqual(state["status"], "completed")
        self.assertEqual(json.loads((self.output / "best.json").read_text())["parameters"], {"batch": 1})
        state, completed_hardware = self.run_tune(good_runner, resume=True)
        completed_hardware.assert_not_called()

    def test_missing_scenario_prevents_acceptance_even_with_good_metrics(self):
        self.config["budget"]["max_trials"] = 1

        def runner(config, candidate, trial_dir, deadline, repeats):
            return measured(config, candidate, repeats)[:1]

        state, _ = self.run_tune(runner)
        self.assertFalse(state["trials"][0]["evaluation"]["feasible"])
        self.assertFalse((self.output / "best.json").exists())


if __name__ == "__main__":
    unittest.main()
