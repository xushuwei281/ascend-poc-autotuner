import copy
import unittest

from ascend_poc.metrics import evaluate_trial, normalize_metrics


def contract():
    return {
        "budget": {"repetitions": 2},
        "requirements": {"p99_ttft_ms": {"max": 2000}, "error_rate": {"max": 0.01}},
        "objective": {"metric": "output_tokens_per_second", "direction": "maximize", "scenario_aggregation": "min"},
        "scenarios": [{"name": "short"}, {"name": "long", "requirements": {"p99_ttft_ms": {"max": 3000}}}],
    }


def results():
    return [
        {"name": "short", "metrics": [{"p99_ttft_ms": 1000, "error_rate": 0, "output_tokens_per_second": 100}, {"p99_ttft_ms": 1100, "error_rate": 0, "output_tokens_per_second": 120}]},
        {"name": "long", "metrics": [{"p99_ttft_ms": 2500, "error_rate": 0, "output_tokens_per_second": 60}, {"p99_ttft_ms": 2600, "error_rate": 0, "output_tokens_per_second": 80}]},
    ]


class MetricTests(unittest.TestCase):
    def test_vllm_aliases_counts_and_percentiles(self):
        normalized = normalize_metrics({"completed": 99, "failed": 1, "output_throughput": 1234, "request_throughput": 12, "percentiles_ttft_ms": [[95, 40], [99, 50]], "quality_score": 0.94, "model": "metadata"}, "vllm")
        self.assertEqual(normalized["output_tokens_per_second"], 1234)
        self.assertEqual(normalized["request_per_second"], 12)
        self.assertEqual(normalized["p99_ttft_ms"], 50)
        self.assertEqual(normalized["error_rate"], 0.01)
        self.assertEqual(normalized["quality_score"], 0.94)
        self.assertNotIn("model", normalized)

    def test_missing_failed_count_is_not_assumed_zero(self):
        self.assertNotIn("error_rate", normalize_metrics({"completed": 10}, "vllm"))

    def test_invalid_counts_and_nonfinite_rejected(self):
        for raw in ({"completed": 0, "failed": 0}, {"completed": -1}, {"completed": True}, {"failed": 1.1}, {"output_throughput": float("nan")}, {"p99_ttft_ms": float("inf")}, {"error_rate": 3}, {"completed": 99, "failed": 1, "error_rate": 0}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                normalize_metrics(raw, "vllm")

    def test_all_scenarios_must_pass_same_candidate(self):
        evaluation = evaluate_trial(contract(), results())
        self.assertTrue(evaluation["feasible"])
        self.assertEqual(evaluation["objective"], 70)
        self.assertEqual(evaluation["scenario_objectives"], {"short": 110, "long": 70})
        self.assertEqual(evaluation["violation"], 0)
        self.assertEqual(len(evaluation["checks"]), 8)

    def test_p99_not_averaged_to_hide_one_failed_repetition(self):
        data = results()
        data[0]["metrics"][0]["p99_ttft_ms"] = 100
        data[0]["metrics"][1]["p99_ttft_ms"] = 2100
        evaluation = evaluate_trial(contract(), data)
        self.assertFalse(evaluation["feasible"])
        self.assertGreater(evaluation["violation"], 0)
        self.assertEqual(sum(not check["pass"] for check in evaluation["checks"]), 1)

    def test_missing_metric_or_scenario_or_repetition_fails(self):
        missing_metric = results()
        del missing_metric[0]["metrics"][0]["error_rate"]
        missing_objective = results()
        del missing_objective[0]["metrics"][0]["output_tokens_per_second"]
        partial = results()
        partial[0]["metrics"].pop()
        for data in (missing_metric, missing_objective, results()[:1], partial, results() + [copy.deepcopy(results()[0])]):
            with self.subTest(data=data):
                self.assertFalse(evaluate_trial(contract(), data)["feasible"])

    def test_final_repetition_count_can_differ(self):
        data = results()
        for scenario in data:
            scenario["metrics"] = scenario["metrics"][:1]
        self.assertTrue(evaluate_trial(contract(), data, expected_repetitions=1)["feasible"])

    def test_minimize_direction_uses_explicit_aggregation(self):
        config = contract()
        config["objective"] = {"metric": "p99_ttft_ms", "direction": "minimize", "scenario_aggregation": "max"}
        evaluation = evaluate_trial(config, results())
        self.assertTrue(evaluation["feasible"])
        self.assertEqual(evaluation["objective"], 2550)
        config["objective"]["scenario_aggregation"] = "mean"
        self.assertEqual(evaluate_trial(config, results())["objective"], 1800)

    def test_quality_gate_is_independent_of_performance(self):
        config = contract()
        config["requirements"]["quality_score"] = {"min": 0.9}
        data = results()
        for scenario in data:
            for metrics in scenario["metrics"]:
                metrics["quality_score"] = 0.95
        self.assertTrue(evaluate_trial(config, data)["feasible"])
        data[1]["metrics"][1]["quality_score"] = 0.89
        self.assertFalse(evaluate_trial(config, data)["feasible"])


if __name__ == "__main__":
    unittest.main()
