import copy
import json
import tempfile
import unittest
from pathlib import Path

from ascend_poc.config import config_hash, load_config, validate_candidate


def contract():
    return {
        "schema_version": 1,
        "name": "unit-test",
        "search_space": {"batch": [16, 32], "graph": [True, False]},
        "baseline": {"batch": 16, "graph": True},
        "server": {"command": ["python", "server.py", "{param.batch}", "{param.graph}"], "health_url": "http://127.0.0.1:8123/health"},
        "scenarios": [{"name": "short", "command": ["python", "bench.py"], "result_file": "{trial_dir}/metrics.json"}],
        "objective": {"metric": "output_tokens_per_second"},
        "requirements": {"p99_ttft_ms": {"max": 2000}},
        "llm": {"base_url": "https://example.org/v1", "model": "test"},
    }


class ContractTests(unittest.TestCase):
    def load(self, data):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            return load_config(path)

    def test_defaults_and_hash_ignore_private_runtime_path(self):
        result = self.load(contract())
        self.assertEqual(result["budget"]["repetitions"], 3)
        self.assertEqual(result["resource_id"], "ascend-default")
        before = config_hash(result)
        result["_repo_dir"] = "/some/local/path"
        self.assertEqual(before, config_hash(result))
        result["baseline"]["batch"] = 32
        self.assertNotEqual(before, config_hash(result))

    def test_candidates_cannot_change_commands_or_thresholds(self):
        config = self.load(contract())
        for candidate in ({"batch": 16}, {"batch": 16, "graph": True, "command": "id"}, {"batch": 33, "graph": True}, {"batch": "16; touch /tmp/oops", "graph": True}):
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                validate_candidate(candidate, config)

    def test_boolean_is_not_integer_choice(self):
        config = self.load(contract())
        with self.assertRaises(ValueError):
            validate_candidate({"batch": 16, "graph": 1}, config)
        normalized = validate_candidate({"batch": 32.0, "graph": False}, config)
        self.assertEqual(normalized["batch"], 32)
        self.assertIs(type(normalized["batch"]), int)

    def test_conditional_constraint(self):
        config = contract()
        config["constraints"] = [{"when": {"graph": True}, "require": {"batch": [16]}}]
        config = self.load(config)
        with self.assertRaisesRegex(ValueError, "compatibility"):
            validate_candidate({"batch": 32, "graph": True}, config)
        self.assertEqual(validate_candidate({"batch": 32, "graph": False}, config)["batch"], 32)

    def test_invalid_baseline_rejected_before_running(self):
        config = contract()
        config["baseline"]["batch"] = 1024
        with self.assertRaises(ValueError):
            self.load(config)

    def test_invalid_shapes_and_typos_fail_loudly(self):
        variants = []
        config = contract()
        config["server"]["command"] = "vllm serve some-model"
        variants.append(config)
        config = contract()
        config["budget"] = {"max_trails": 4}
        variants.append(config)
        config = contract()
        config["scenarios"].append(copy.deepcopy(config["scenarios"][0]))
        variants.append(config)
        config = contract()
        config["requirements"]["quality_score"] = {"min": 0.9, "max": 0.8}
        variants.append(config)
        config = contract()
        config["search_space"]["bad.name"] = [1]
        variants.append(config)
        config = contract()
        config["scenarios"][0]["quality_command"] = ["python", "quality.py"]
        variants.append(config)
        config = contract()
        config["server"]["health_url"] = "http://user:password@localhost:8000/health"
        variants.append(config)
        config = contract()
        config["budget"] = {"repetitions": True}
        variants.append(config)
        for index, config in enumerate(variants):
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.load(config)

    def test_secrets_must_be_environment_references(self):
        config = contract()
        config["llm"]["api_key"] = "dummy-secret"
        with self.assertRaises(ValueError):
            self.load(config)
        config = contract()
        config["server"]["env"] = {"POC_LLM_API_KEY": "dummy-secret"}
        with self.assertRaises(ValueError):
            self.load(config)

    def test_nonfinite_and_duplicate_json_rejected(self):
        config = contract()
        config["search_space"]["batch"] = [float("nan")]
        with self.assertRaises(ValueError):
            self.load(config)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_config(path)
        config = contract()
        config["context"] = {"bad": float("inf")}
        with self.assertRaises(ValueError):
            self.load(config)

    def test_numeric_choice_duplicates(self):
        config = contract()
        config["search_space"]["batch"] = [16, 16.0]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.load(config)

    def test_temperature_can_be_omitted_for_provider_compatibility(self):
        config = contract()
        config["llm"]["temperature"] = None
        self.assertIsNone(self.load(config)["llm"]["temperature"])

    def test_every_search_parameter_must_change_server_configuration(self):
        config = contract()
        config["server"]["command"].remove("{param.graph}")
        with self.assertRaisesRegex(ValueError, "must be bound"):
            self.load(config)
        # Referencing it only in the benchmark does not make it a server knob.
        config["scenarios"][0]["command"].append("{param.graph}")
        with self.assertRaisesRegex(ValueError, "must be bound"):
            self.load(config)

    def test_search_parameter_can_bind_in_server_environment(self):
        config = contract()
        config["server"]["command"].remove("{param.graph}")
        config["server"]["env"] = {"EXAMPLE_GRAPH_MODE": "{param.graph}"}
        self.assertEqual(self.load(config)["server"]["env"]["EXAMPLE_GRAPH_MODE"], "{param.graph}")

    def test_unknown_server_parameter_references_are_rejected(self):
        for in_env in (False, True):
            config = contract()
            if in_env:
                config["server"]["env"] = {"EXAMPLE_MODE": "mode={param.unknown}"}
            else:
                config["server"]["command"].append("--mode={param.unknown}")
            with self.subTest(in_env=in_env), self.assertRaisesRegex(ValueError, "unknown parameter"):
                self.load(config)

    def test_malformed_server_parameter_placeholder_rejected(self):
        config = contract()
        config["server"]["command"].append("{param.typo")
        with self.assertRaisesRegex(ValueError, "malformed parameter"):
            self.load(config)


if __name__ == "__main__":
    unittest.main()
