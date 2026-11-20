import json
import math
import unittest

from ascend_poc.config import validate_config
from ascend_poc.diagnosis import analyze_trial
from ascend_poc.metrics import evaluate_trial
from tests.test_config import contract


class DiagnosisTests(unittest.TestCase):
    def setUp(self):
        base = contract()
        base["requirements"].update(p99_tpot_ms={"max": 50}, quality_score={"min": .98}, output_tokens_per_second={"min": 500}, error_rate={"max": 0})
        self.config = validate_config(base)

    def record(self, **updates):
        metrics = {"p99_ttft_ms": 1000, "p99_tpot_ms": 40, "quality_score": .99, "output_tokens_per_second": 600, "error_rate": 0}
        metrics.update(updates)
        evaluation = evaluate_trial(self.config, [{"name": "short", "metrics": [metrics]}], expected_repetitions=1)
        return {"id": "0001-exploration", "status": "completed", "evaluation": evaluation}

    def test_latency_failure_is_hypothesis_with_exact_gate_evidence(self):
        result = analyze_trial(self.record(p99_ttft_ms=2400), self.config)
        self.assertFalse(result["feasible"])
        self.assertEqual(result["violations"][0]["value"], 2400)
        self.assertEqual(result["violations"][0]["bounds"], {"max": 2000})
        self.assertEqual(result["hypotheses"][0]["kind"], "prefill_or_queueing")
        self.assertEqual(result["hypotheses"][0]["confidence"], "hypothesis")
        self.assertIn("prefix_caching", result["strategy_families"])

    def test_quality_precedes_speed_and_questions_are_bounded(self):
        result = analyze_trial(self.record(quality_score=.7, p99_ttft_ms=2500, p99_tpot_ms=80, output_tokens_per_second=100, error_rate=.1), self.config)
        self.assertEqual(result["hypotheses"][0]["kind"], "quality_regression")
        self.assertEqual(len(result["research_questions"]), 3)
        self.assertEqual(len(result["violations"]), 5)

    def test_raw_logs_and_arbitrary_text_cannot_reach_research(self):
        record = self.record()
        record.update(error="SECRET customer prompt", logs="SECRET api key", hypothesis="SECRET command")
        record["evaluation"]["failures"] = ["SECRET raw error"]
        record["evaluation"]["checks"].append({"metric": "SECRET unknown", "scenario": "short", "value": 1, "pass": False})
        result = analyze_trial(record, self.config)
        self.assertNotIn("SECRET", json.dumps(result))

    def test_failed_run_cannot_be_called_oom_or_speedup(self):
        result = analyze_trial({"id": "bad/path", "status": "failed", "error": "OutOfMemory fake evidence"}, self.config)
        self.assertIsNone(result["objective"])
        self.assertEqual(result["trial_id"], "unknown")
        self.assertEqual(result["hypotheses"][0]["kind"], "measurement_failure")
        self.assertNotIn("memory_pressure", [h["kind"] for h in result["hypotheses"]])

    def test_missing_metrics_are_diagnostic_failures(self):
        result = analyze_trial(self.record(p99_tpot_ms=math.nan), self.config)
        self.assertEqual(result["violations"][0]["reason"], "missing_metric")
        self.assertIsNone(result["violations"][0]["value"])
        self.assertEqual(result["hypotheses"][0]["kind"], "measurement_failure")
        json.dumps(result, allow_nan=False)

    def test_contract_bounds_override_untrusted_checks_and_pass_flags(self):
        record = self.record(p99_ttft_ms=2400)
        for check in record["evaluation"]["checks"]:
            check["pass"] = True
            check["requirement"] = {"max": 9999999}
        record["evaluation"]["feasible"] = True
        result = analyze_trial(record, self.config)
        self.assertFalse(result["feasible"])
        self.assertEqual(result["violations"][0]["bounds"], {"max": 2000})

    def test_passing_baseline_remains_evidence_not_claim_of_bottleneck(self):
        result = analyze_trial(self.record(), self.config)
        self.assertTrue(result["feasible"])
        self.assertEqual(result["violations"], [])
        self.assertEqual(result["hypotheses"][0]["kind"], "feasible_optimization")


if __name__ == "__main__":
    unittest.main()
