import copy
import json
import os
import time
import unittest
from unittest.mock import patch

from ascend_poc.llm import LLMError
from ascend_poc.workflow_planner import WorkflowPlanner
from tests.test_llm import config, provider


def specification(url):
    contract = config(url)
    optimized = copy.deepcopy(contract)
    optimized["search_space"] = {"prefill_tokens": [2048, 4096]}
    optimized["baseline"] = {"prefill_tokens": 2048}
    return {"_contract": contract, "profiles": [
        {"id": "baseline", "description": "Conservative baseline", "enabled": True,
         "compatibility": {"status": "verified", "evidence": ["local test"]}},
        {"id": "chunked", "description": "Chunked prefill strategy", "enabled": True,
         "compatibility": {"status": "verified"}, "server": {"command": ["PROFILE-COMMAND-SECRET"]}},
        {"id": "disabled", "description": "DISABLED-SECRET", "enabled": False,
         "compatibility": {"status": "verified"}},
        {"id": "unverified", "description": "UNVERIFIED-SECRET", "enabled": True,
         "compatibility": {"status": "unverified"}},
    ], "_profiles": {"baseline": contract, "chunked": optimized, "disabled": contract, "unverified": contract}}


def answer(**changes):
    value = {"profile_id": "chunked", "parameters": {"prefill_tokens": 2048},
             "hypothesis": "Try chunked prefill to reduce scheduling interference; measure the tradeoff.",
             "evidence_ids": ["e0"]}
    value.update(changes)
    return {"choices": [{"message": {"content": json.dumps(value)}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70}}


EVIDENCE = [{"id": "e0", "answer": "Chunked prefill scheduling should be evaluated on this version.",
             "sources": ["https://github.com/vllm-project/vllm-ascend/blob/main/README.md#L1"],
             "private_logs": "EVIDENCE-SECRET"}]


class WorkflowPlannerTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"POC_TEST_KEY": "test-only-key"})
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_http_selection_contains_only_shareable_measurements_and_evidence(self):
        with provider([{"payload": answer()}]) as (url, captured):
            spec = specification(url)
            planner = WorkflowPlanner(spec)
            history = [{"profile_id": "baseline", "parameters": spec["_contract"]["baseline"],
                        "status": "completed", "logs": "HISTORY-SECRET", "error": "ERROR-SECRET",
                        "evaluation": {"feasible": False, "objective": 100, "violation": .4,
                                       "checks": [{"scenario": "customer-case", "metric": "p99_ttft_ms",
                                                   "value": 2800, "passed": False}]}}]
            analysis = {"trial_id": "0000-exploration", "status": "completed", "feasible": False,
                        "objective": 100, "error": "ANALYSIS-SECRET", "logs": "LOG-SECRET",
                        "violations": [{"metric": "p99_ttft_ms", "scenario": "customer-case", "value": 2800,
                                        "bounds": {"max": 2000}, "reason": "outside_requirement",
                                        "error": "VIOLATION-SECRET"}],
                        "hypotheses": [{"kind": "prefill_latency", "summary": "Prefill pressure is a hypothesis.",
                                        "evidence_metrics": ["p99_ttft_ms"], "logs": "HYPOTHESIS-SECRET"}],
                        "strategy_families": ["chunked_prefill"]}
            result = planner.propose(history, analysis, EVIDENCE, 3)
        self.assertEqual(result["profile_id"], "chunked")
        self.assertEqual(result["parameters"], {"prefill_tokens": 2048})
        self.assertEqual(result["evidence_ids"], ["e0"])
        self.assertEqual(planner.calls, 1)
        self.assertEqual(planner.last_usage["total_tokens"], 70)
        text = json.dumps(captured[0]["body"])
        for secret in ("-SECRET", "private-script", "private-serve", "/customer/private", "hidden-context-secret", "test-only-key"):
            self.assertNotIn(secret, text)
        data = json.loads(captured[0]["body"]["messages"][1]["content"])
        self.assertEqual([item["id"] for item in data["profiles"]], ["baseline", "chunked"])
        self.assertEqual(data["history"][0]["profile_id"], "baseline")
        self.assertEqual(data["analysis"]["violations"][0]["value"], 2800)
        self.assertIn("UNTRUSTED DATA", captured[0]["body"]["messages"][0]["content"])
        self.assertEqual(captured[0]["path"], "/v1/chat/completions")

    def test_unknown_disabled_unverified_profiles_and_wrong_parameter_domains_rejected(self):
        changes = [{"profile_id": "missing"}, {"profile_id": "disabled"}, {"profile_id": "unverified"},
                   {"profile_id": []}, {"parameters": {"prefill_tokens": 8192}},
                   {"parameters": {"max_num_seqs": 32}}, {"parameters": {"prefill_tokens": True}},
                   {"parameters": {"prefill_tokens": 2048, "command": "run"}}]
        for change in changes:
            with self.subTest(change=change), provider([{"payload": answer(**change)}]) as (url, captured):
                with self.assertRaises(LLMError):
                    WorkflowPlanner(specification(url)).propose([], {}, EVIDENCE, 3)
                self.assertEqual(len(captured), 1)

    def test_evidence_citations_must_be_present_real_unique_and_list(self):
        for citations in ([], ["invented"], ["e0", "e0"], "e0", [3], [{}]):
            with self.subTest(citations=citations), provider([{"payload": answer(evidence_ids=citations)}]) as (url, _):
                with self.assertRaisesRegex(LLMError, "evidence IDs"):
                    WorkflowPlanner(specification(url)).propose([], {}, EVIDENCE, 3)

    def test_absent_research_can_continue_without_invented_citation(self):
        with provider([{"payload": answer(evidence_ids=[])}]) as (url, _):
            result = WorkflowPlanner(specification(url)).propose([], {}, [], 3)
        self.assertEqual(result["evidence_ids"], [])

    def test_strict_json_duplicate_keys_and_extra_fields(self):
        contents = ["not JSON SECRET", "[]", '{"profile_id":"chunked","profile_id":"baseline"}',
                    json.dumps({"profile_id": "chunked", "parameters": {"prefill_tokens": 2048},
                                "hypothesis": "Try", "evidence_ids": ["e0"], "command": "run"})]
        for content in contents:
            payload = answer()
            payload["choices"][0]["message"]["content"] = content
            with self.subTest(content=content), provider([{"payload": payload}]) as (url, _):
                with self.assertRaises(LLMError) as error:
                    WorkflowPlanner(specification(url)).propose([], {}, EVIDENCE, 3)
                self.assertNotIn("SECRET", str(error.exception))

    def test_conditions_are_applied_to_selected_profile(self):
        with provider([{"payload": answer()}]) as (url, _):
            spec = specification(url)
            spec["_profiles"]["chunked"]["constraints"] = [
                {"when": {"prefill_tokens": 2048}, "require": {"prefill_tokens": [4096]}}]
            with self.assertRaisesRegex(LLMError, "constraints"):
                WorkflowPlanner(spec).propose([], {}, EVIDENCE, 3)

    def test_timeout_no_overlap_and_retry_accounting(self):
        with provider([{"delay": .5, "payload": answer()}]) as (url, _):
            planner = WorkflowPlanner(specification(url))
            start = time.monotonic()
            with self.assertRaises(LLMError):
                planner.propose([], {}, EVIDENCE, .08)
            self.assertLess(time.monotonic() - start, .4)
            self.assertEqual(planner.calls, 1)
        with provider([{"status": 429}, {"payload": answer()}]) as (url, _):
            planner = WorkflowPlanner(specification(url))
            self.assertEqual(planner.propose([], {}, EVIDENCE, 3)["profile_id"], "chunked")
            self.assertEqual(planner.calls, 2)

    def test_missing_key_deadline_and_invalid_url_do_not_send_request(self):
        with provider([]) as (url, captured):
            planner = WorkflowPlanner(specification(url))
            with patch.dict(os.environ, {"POC_TEST_KEY": ""}):
                with self.assertRaises(LLMError):
                    planner.propose([], {}, EVIDENCE, 3)
            for remaining in (0, -1, float("nan"), float("inf")):
                with self.assertRaises(LLMError):
                    planner.propose([], {}, EVIDENCE, remaining)
            self.assertEqual(captured, [])
        with self.assertRaises(LLMError):
            WorkflowPlanner(specification("http://remote.example/v1"))

    def test_references_are_bounded_and_credential_urls_are_omitted(self):
        evidence = [{"id": f"e{i}", "answer": "a" * 18000, "sources": [
            "https://user:secret@example.com/read", "javascript:alert(1)",
            "https://example.com/reference?token=hidden#L12"]} for i in range(8)]
        with provider([{"payload": answer(evidence_ids=["e7"])}]) as (url, captured):
            WorkflowPlanner(specification(url)).propose([], {}, evidence, 3)
        sent = json.loads(captured[0]["body"]["messages"][1]["content"])["evidence"]
        self.assertEqual(len(sent), 4)
        self.assertEqual(len(sent[0]["answer"]), 16000)
        self.assertEqual(sent[0]["sources"], ["https://example.com/reference#L12"])


if __name__ == "__main__":
    unittest.main()
