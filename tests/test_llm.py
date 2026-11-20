import json
import os
import threading
import time
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from ascend_poc.llm import LLMError, LLMPlanner


def config(url):
    return {
        "llm": {"base_url": url, "model": "test-model", "api_key_env": "POC_TEST_KEY",
                "timeout_s": 2, "max_tokens": 700, "temperature": 0.2,
                "json_mode": True, "max_retries": 1},
        "context": {"hardware": "Ascend test", "password": "hidden-context-secret"},
        "search_space": {"max_num_seqs": [32, 64], "max_num_batched_tokens": [2048, 4096]},
        "baseline": {"max_num_seqs": 32, "max_num_batched_tokens": 2048},
        "constraints": [],
        "requirements": {"p99_ttft_ms": {"max": 2000}},
        "objective": {"metric": "output_tokens_per_second", "direction": "maximize", "scenario_aggregation": "min"},
        "scenarios": [{"name": "customer-case", "command": ["private-script"],
                       "result_file": "/customer/private/path"}],
        "server": {"command": ["private-serve"]},
    }


def response(parameters=None, hypothesis="Try a larger batch."):
    return {
        "choices": [{"message": {"content": json.dumps({
            "parameters": parameters or {"max_num_seqs": 64, "max_num_batched_tokens": 4096},
            "hypothesis": hypothesis,
        })}}],
        "usage": {"prompt_tokens": 30, "completion_tokens": 20, "total_tokens": 50},
    }


@contextmanager
def provider(replies):
    captured = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            captured.append({"path": self.path, "headers": dict(self.headers), "body": json.loads(body)})
            reply = replies.pop(0)
            time.sleep(reply.get("delay", 0))
            self.send_response(reply.get("status", 200))
            for key, value in reply.get("headers", {}).items():
                self.send_header(key, value)
            self.end_headers()
            try:
                self.wfile.write(json.dumps(reply.get("payload", response())).encode())
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", captured
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


class LLMTests(unittest.TestCase):
    def setUp(self):
        self.environ = patch.dict(os.environ, {"POC_TEST_KEY": "test-only-key"})
        self.environ.start()

    def tearDown(self):
        self.environ.stop()

    def test_real_http_request_uses_api_and_strips_private_history(self):
        with provider([{}]) as (url, captured):
            planner = LLMPlanner(config(url))
            history = [{"parameters": config(url)["baseline"], "status": "completed",
                        "command": "SECRET-COMMAND", "logs": "SECRET-LOG",
                        "evaluation": {"feasible": False, "objective": 100, "violation": 0.2,
                                       "failures": ["SECRET-FAILURE"],
                                       "checks": [{"scenario": "customer-case", "metric": "p99_ttft_ms",
                                                   "value": 2400, "passed": False, "error": "SECRET-ERROR"}]}}]
            result = planner.propose(history, 3)
        self.assertEqual(result["parameters"]["max_num_seqs"], 64)
        self.assertEqual(planner.last_usage["total_tokens"], 50)
        self.assertEqual(planner.calls, 1)
        request = captured[0]
        self.assertEqual(request["path"], "/v1/chat/completions")
        self.assertEqual(request["headers"]["Authorization"], "Bearer test-only-key")
        self.assertEqual(request["body"]["model"], "test-model")
        self.assertEqual(request["body"]["max_tokens"], 700)
        self.assertEqual(request["body"]["response_format"], {"type": "json_object"})
        self.assertFalse(request["body"]["stream"])
        text = json.dumps(request["body"])
        for secret in ("SECRET-", "private-script", "/customer/private", "private-serve", "hidden-context-secret", "test-only-key"):
            self.assertNotIn(secret, text)
        self.assertIn("2400", text)

    def test_exact_endpoint_fenced_json_and_optional_provider_fields(self):
        payload = response()
        payload["choices"][0]["message"]["content"] = "```json\n" + payload["choices"][0]["message"]["content"] + "\n```"
        with provider([{"payload": payload}]) as (url, captured):
            settings = config(url + "/chat/completions/")
            settings["llm"]["json_mode"] = False
            settings["llm"]["temperature"] = None
            LLMPlanner(settings).propose([], 3)
        self.assertEqual(captured[0]["path"], "/v1/chat/completions")
        self.assertNotIn("response_format", captured[0]["body"])
        self.assertNotIn("temperature", captured[0]["body"])

    def test_missing_key_and_exhausted_budget_do_not_call_provider(self):
        with provider([]) as (url, captured):
            planner = LLMPlanner(config(url))
            with patch.dict(os.environ, {"POC_TEST_KEY": ""}):
                with self.assertRaisesRegex(LLMError, "key is missing"):
                    planner.propose([], 3)
            for remaining in (0, -1, float("nan"), float("inf")):
                with self.assertRaises(LLMError):
                    planner.propose([], remaining)
            self.assertEqual(captured, [])

    def test_reject_invalid_proposals_without_retrying(self):
        bad_contents = [
            "not JSON SECRET", "[]", '{"parameters": {}, "hypothesis": "x", "command": "rm"}',
            '{"parameters": {"max_num_seqs": 999, "max_num_batched_tokens": 4096}, "hypothesis": "x"}',
            '{"parameters": {"max_num_seqs": 64}, "hypothesis": "x"}',
            '{"parameters": {"max_num_seqs": 64, "max_num_batched_tokens": 4096}, "hypothesis": ""}',
            '{"parameters": {"max_num_seqs": NaN, "max_num_batched_tokens": 4096}, "hypothesis": "x"}',
        ]
        for content in bad_contents:
            payload = response()
            payload["choices"][0]["message"]["content"] = content
            with self.subTest(content=content), provider([{"payload": payload}]) as (url, captured):
                planner = LLMPlanner(config(url))
                with self.assertRaises(LLMError) as caught:
                    planner.propose([], 3)
                self.assertNotIn("SECRET", str(caught.exception))
                self.assertEqual(planner.calls, 1)

    def test_bad_response_structure_is_sanitized(self):
        for payload in ({"error": "test-only-key"}, {"choices": []}, {"choices": [{"message": {"content": None}}]}):
            with self.subTest(payload=payload), provider([{"payload": payload}]) as (url, _):
                with self.assertRaisesRegex(LLMError, "invalid Chat Completions structure"):
                    LLMPlanner(config(url)).propose([], 3)

    def test_retries_rate_limit_and_server_failure(self):
        for status in (429, 503):
            with self.subTest(status=status), provider([{"status": status}, {}]) as (url, _):
                planner = LLMPlanner(config(url))
                self.assertEqual(planner.propose([], 3)["parameters"]["max_num_seqs"], 64)
                self.assertEqual(planner.calls, 2)

    def test_no_retry_for_unauthorized_or_redirect_and_no_error_body(self):
        for status in (401, 400, 302):
            with self.subTest(status=status), provider([
                {"status": status, "payload": {"error": "test-only-key"},
                 "headers": {"Location": "http://127.0.0.1:1/stolen"}},
            ]) as (url, captured):
                planner = LLMPlanner(config(url))
                with self.assertRaises(LLMError) as caught:
                    planner.propose([], 3)
                self.assertIn(str(status), str(caught.exception))
                self.assertNotIn("test-only-key", str(caught.exception))
                self.assertEqual(planner.calls, 1)
                self.assertEqual(len(captured), 1)

    def test_request_deadline_is_bounded(self):
        with provider([{"delay": 0.5}]) as (url, _):
            planner = LLMPlanner(config(url))
            start = time.monotonic()
            with self.assertRaises(LLMError):
                planner.propose([], 0.08)
            self.assertLess(time.monotonic() - start, 0.4)
            self.assertEqual(planner.calls, 1)

    def test_endpoint_rejects_remote_plaintext_and_embedded_secrets(self):
        for url in ("http://example.com/v1", "https://user:secret@example.com/v1", "https://example.com/v1?key=secret", "ftp://localhost/v1", "https://example.com:bad/v1"):
            with self.subTest(url=url), self.assertRaises(LLMError):
                LLMPlanner(config(url))

    def test_conditional_constraints_cannot_be_bypassed_by_model(self):
        with provider([{}]) as (url, _):
            settings = config(url)
            settings["constraints"] = [{"when": {"max_num_seqs": 64}, "require": {"max_num_batched_tokens": [2048]}}]
            with self.assertRaises(LLMError):
                LLMPlanner(settings).propose([], 3)

    def test_history_is_bounded_and_retains_old_best(self):
        with provider([{}]) as (url, captured):
            settings = config(url)
            history = [{"parameters": settings["baseline"], "status": "completed",
                        "evaluation": {"feasible": True, "objective": 1000, "violation": 0}}]
            for i in range(20):
                history.append({"parameters": settings["baseline"], "status": "completed",
                                "evaluation": {"feasible": True, "objective": i, "violation": 0}})
            LLMPlanner(settings).propose(history, 3)
        compact = json.loads(captured[0]["body"]["messages"][1]["content"])["history"]
        self.assertEqual(len(compact), 11)
        self.assertEqual(compact[0]["evaluation"]["objective"], 1000)


if __name__ == "__main__":
    unittest.main()
