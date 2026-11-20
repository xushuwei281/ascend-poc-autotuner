import copy
import json
from pathlib import Path
import tempfile
import unittest

from ascend_poc.config import validate_config
from ascend_poc.strategy import eligible_profiles, load_workflow, materialize_profile, workflow_hash
from tests.test_config import contract


def workflow():
    return {
        "schema_version": 1, "name": "test workflow", "contract": "contract.json",
        "profiles": [
            {"id": "baseline", "description": "Known baseline", "compatibility": {"status": "verified", "evidence": ["Operator tested this fixed stack"]}},
            {"id": "eager", "description": "Alternative graph mode", "baseline": {"batch": 16, "graph": False}, "compatibility": {"status": "verified", "evidence": ["Operator tested eager mode"]}},
            {"id": "future", "description": "Unverified recipe", "compatibility": {"status": "unverified", "reason": "Model support not checked"}},
        ],
    }


class StrategyTests(unittest.TestCase):
    def test_verification_contract_rejects_malformed_or_unbounded_fields(self):
        for invalid in ({'mode': 'automatic'}, {'unknown': 1}, {'max_baseline_drift': 1.1},
                        {'max_baseline_drift': -1}, {'min_relative_improvement': True},
                        {'drift_multiplier': -1}, {'frozen_files': {}},
                        {'frozen_files': [{'path': 'data.json', 'sha256': 'not-a-hash'}]},
                        {'frozen_files': [{'path': 'data.json', 'sha256': 'a' * 64, 'command': 'bad'}]},
                        {'frozen_files': [{'path': 'data.json', 'sha256': 'a' * 64}] * 2}):
            raw = workflow()
            raw['verification'] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.load(raw)

    def test_verification_policy_is_frozen_but_files_need_not_exist_at_validation(self):
        baseline = self.load()
        self.assertEqual(baseline['verification']['mode'], 'single')
        raw = workflow()
        raw['verification'] = {'mode': 'aba', 'frozen_files': [{'path': '/customer/not-present-yet', 'sha256': 'a' * 64}]}
        spec = self.load(raw)
        self.assertNotEqual(workflow_hash(baseline), workflow_hash(spec))
        self.assertEqual(spec['verification']['max_baseline_drift'], .05)

    def load(self, spec=None, base=None):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "contract.json").write_text(json.dumps(base or contract()))
            path = root / "workflow.json"
            path.write_text(json.dumps(spec or workflow()))
            return load_workflow(path)

    def test_only_operator_enabled_verified_profiles_are_eligible(self):
        spec = self.load()
        self.assertEqual([p["id"] for p in eligible_profiles(spec)], ["baseline", "eager"])
        with self.assertRaisesRegex(ValueError, "unverified"):
            materialize_profile(spec, "future")
        raw = workflow()
        raw["profiles"][1]["enabled"] = False
        self.assertEqual(len(eligible_profiles(self.load(raw))), 1)

    def test_profile_keeps_all_workload_and_acceptance_fields_frozen(self):
        spec = self.load()
        chosen = materialize_profile(spec, "eager", {"batch": 32, "graph": False})
        for field in ("scenarios", "objective", "requirements", "llm", "context", "resource_id", "benchmark_env"):
            self.assertEqual(chosen[field], spec["_contract"][field])
        self.assertEqual(chosen["baseline"], {"batch": 32, "graph": False})
        chosen["scenarios"][0]["name"] = "modified"
        self.assertEqual(spec["_contract"]["scenarios"][0]["name"], "short")

    def test_unknown_profile_fields_cannot_change_workloads(self):
        for field, value in (("requirements", {}), ("scenarios", []), ("llm", {}), ("context", {}), ("benchmark_env", {}), ("command", ["echo", "oops"])):
            raw = workflow()
            raw["profiles"][1][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "unknown fields"):
                self.load(raw)

    def test_profile_ids_cannot_escape_output_directory(self):
        for name in ("../outside", "a/b", "..", "a.b", "", "x" * 65):
            raw = workflow()
            raw["profiles"][1]["id"] = name
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.load(raw)

    def test_baseline_must_be_original_and_verified(self):
        for change in ({"enabled": False}, {"baseline": {"batch": 32, "graph": True}}, {"compatibility": {"status": "unverified", "reason": "unknown"}}):
            raw = workflow()
            raw["profiles"][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load(raw)

    def test_verified_needs_evidence_and_unverified_needs_reason(self):
        raw = workflow()
        raw["profiles"][1]["compatibility"]["evidence"] = []
        with self.assertRaisesRegex(ValueError, "evidence"):
            self.load(raw)
        raw = workflow()
        raw["profiles"][2]["compatibility"].pop("reason")
        with self.assertRaisesRegex(ValueError, "reason"):
            self.load(raw)

    def test_unsafe_or_malformed_research_destinations_rejected(self):
        for url in ("http://external.example/mcp", "https://user:secret@example.org/mcp", "https://example.org/mcp?token=x", "https://example.org/mcp#x", "https://example.org:bad/mcp"):
            raw = workflow()
            raw["research"] = {"endpoint": url}
            with self.subTest(url=url), self.assertRaises(ValueError):
                self.load(raw)
        raw = workflow()
        raw["research"] = {"transport": "a2a"}
        self.assertEqual(self.load(raw)["research"]["endpoint"], "http://127.0.0.1:8766/a2a")

    def test_quality_profile_cannot_execute_without_all_quality_hooks_and_gates(self):
        raw = workflow()
        raw["profiles"][1]["requires_quality"] = True
        with self.assertRaisesRegex(ValueError, "quality hook"):
            self.load(raw)
        base = contract()
        base["scenarios"][0].update(quality_command=["python", "quality.py"], quality_result_file="{trial_dir}/quality.json")
        with self.assertRaisesRegex(ValueError, "threshold"):
            self.load(raw, base)
        base["requirements"]["accuracy"] = {"min": .95}
        self.assertIn("eager", self.load(raw, base)["_profiles"])
        # A disabled recipe remains reviewable before the operator supplies its evaluator.
        raw["profiles"][1]["enabled"] = False
        self.assertEqual(len(eligible_profiles(self.load(raw))), 1)

    def test_profile_parameter_references_and_constraints_still_validated(self):
        raw = workflow()
        raw["profiles"][1]["search_space"] = {"batch": [16, 32], "graph": [False], "unbound": [1]}
        raw["profiles"][1]["baseline"]["unbound"] = 1
        with self.assertRaisesRegex(ValueError, "must be bound"):
            self.load(raw)
        spec = self.load()
        with self.assertRaises(ValueError):
            materialize_profile(spec, "eager", {"batch": 9999, "graph": False})

    def test_search_cannot_change_benchmark_arguments(self):
        base = contract()
        base["scenarios"][0]["command"].extend(["--load", "{param.batch}"])
        with self.assertRaisesRegex(ValueError, "freeze"):
            self.load(base=base)

    def test_profile_service_endpoint_must_remain_fixed(self):
        raw = workflow()
        raw["profiles"][1]["server"] = copy.deepcopy(contract()["server"])
        raw["profiles"][1]["server"]["health_url"] = "http://127.0.0.1:9000/health"
        with self.assertRaisesRegex(ValueError, "health_url"):
            self.load(raw)

    def test_profile_changes_cannot_change_frozen_benchmark_environment(self):
        for field in ("command", "warmup_command", "before_each_command"):
            base = contract()
            base["scenarios"][0][field] = ["python", "bench.py", "{env:WORKLOAD}"]
            base["server"]["env"] = {"WORKLOAD": "fixed.json"}
            raw = workflow()
            raw["profiles"][1]["server"] = copy.deepcopy(base["server"])
            raw["profiles"][1]["server"]["env"]["WORKLOAD"] = "different.json"
            with self.subTest(field=field):
                chosen = materialize_profile(self.load(raw, base), "eager")
                self.assertEqual(chosen["server"]["env"]["WORKLOAD"], "different.json")
                self.assertEqual(chosen["benchmark_env"]["WORKLOAD"], "fixed.json")
        base = contract()
        base["server"]["env"] = {"RESULT": "{param.batch}", "GRAPH": "{param.graph}", "ROOT": "{repo_dir}", "SOURCE": "{env:SOURCE}"}
        chosen = materialize_profile(self.load(base=base), "eager", {"batch": 32, "graph": False})
        self.assertEqual(chosen["benchmark_env"], {"RESULT": "16", "GRAPH": "true", "ROOT": "{repo_dir}", "SOURCE": "{env:SOURCE}"})

    def test_explicit_benchmark_env_validated_and_preserved(self):
        base = contract()
        base["benchmark_env"] = {"OUTPUT_LEN": "512"}
        self.assertEqual(materialize_profile(self.load(base=base), "eager")["benchmark_env"], {"OUTPUT_LEN": "512"})
        for value in ({"BAD-NAME": "x"}, {"OUTPUT_LEN": 512}, {"OUTPUT_LEN": "{param.batch}"}, {"POC_LLM_API_KEY": "bad"}, {"LEAK": "{env:POC_LLM_API_KEY}"}):
            base["benchmark_env"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.load(base=base)

    def test_unrelated_operator_runtime_environment_can_change(self):
        raw = workflow()
        raw["profiles"][1]["server"] = copy.deepcopy(contract()["server"])
        raw["profiles"][1]["server"]["env"] = {"EXAMPLE_NPU_MODE": "checked-value"}
        self.assertEqual(materialize_profile(self.load(raw), "eager")["server"]["env"]["EXAMPLE_NPU_MODE"], "checked-value")

    def test_unchanged_environment_reference_preserves_profile_identity(self):
        base = contract()
        base["scenarios"][0]["command"].append("{env:WORKLOAD}")
        base["server"]["env"] = {"WORKLOAD": "fixed.json"}
        spec = self.load(base=base)
        self.assertEqual(set(spec["_profiles"]), {"baseline", "eager", "future"})
        self.assertEqual(materialize_profile(spec, "eager")["server"]["env"]["WORKLOAD"], "fixed.json")

    def test_hash_tracks_contract_and_ignores_runtime_paths(self):
        spec = self.load()
        before = workflow_hash(spec)
        spec["_source"] = "/new/path/workflow.json"
        spec["_contract"]["_repo_dir"] = "/runtime/path"
        self.assertEqual(workflow_hash(spec), before)
        self.assertEqual(materialize_profile(spec, "baseline")["_repo_dir"], "/runtime/path")
        spec["_contract"]["requirements"]["p99_ttft_ms"]["max"] = 999
        self.assertNotEqual(workflow_hash(spec), before)

    def test_in_memory_validation_does_not_mutate_input(self):
        base = contract()
        original = copy.deepcopy(base)
        normalized = validate_config(base)
        self.assertEqual(base, original)
        self.assertIn("budget", normalized)

    def test_duplicate_json_keys_nonfinite_and_invalid_budgets_rejected(self):
        for budget in ({"max_trials": True}, {"max_calls": 2}, {"max_wall_time_s": float("inf")}):
            raw = workflow()
            raw["budget"] = budget
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                self.load(raw)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.json"
            path.write_text('{"schema_version":1,"schema_version":1}')
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_workflow(path)


if __name__ == "__main__":
    unittest.main()
