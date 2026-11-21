"""Local protocol tests for the optional A2A 0.3 research bridge."""
from contextlib import contextmanager
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
import unittest
import urllib.error
import urllib.request

from ascend_poc.a2a import A2AError, A2AResearchClient, create_research_server


REPO = "vllm-project/vllm-ascend"


class FakeResearcher:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def ask(self, repo, question, remaining_s=None):
        self.calls.append((repo, question, remaining_s))
        if self.fail:
            raise RuntimeError("secret upstream error")
        return {"repository": repo, "answer": "Test answer about chunked prefill.",
                "sources": ["https://deepwiki.com/vllm-project/vllm-ascend"],
                "transport": "mcp", "tool": "ask_wiki_question"}


@contextmanager
def bridge(researcher=None, **kwargs):
    server = create_research_server(port=0, researcher=researcher or FakeResearcher(), **kwargs)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def rpc(server, method, params, **headers):
    data = {"jsonrpc": "2.0", "id": "r1", "method": method, "params": params}
    request = urllib.request.Request(server.endpoint, data=json.dumps(data).encode(),
                                     headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(request, timeout=2) as response:
        return json.load(response)


def query_params(repo=REPO, question="Why is decode slow?"):
    return {"message": {"kind": "message", "messageId": "m1", "role": "user",
                        "parts": [{"kind": "data", "data": {"repository": repo, "question": question}}]}}


@contextmanager
def alternate_agent(change, delay=0):
    """A fake remote agent able to return malformed card/RPC/task data."""
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send(self, data):
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if delay:
                time.sleep(delay)
            card = {"protocolVersion": "0.3.0", "preferredTransport": "JSONRPC",
                    "url": f"http://127.0.0.1:{self.server.server_port}/a2a"}
            self.send(change("card", card))

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            result = {"jsonrpc": "2.0", "id": request["id"], "result": {
                "kind": "task", "id": "task1", "contextId": "ctx1", "status": {"state": "completed"},
                "artifacts": [{"artifactId": "art1", "parts": [{"kind": "data", "data": FakeResearcher().ask(REPO, "test")}]}]}}
            self.send(change("response", result))

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


class A2ATests(unittest.TestCase):
    def test_a2a_bridge_negotiates_real_mcp_client_transport(self):
        from ascend_poc.deepwiki import DeepWikiClient
        from tests.test_deepwiki import mcp_server
        with mcp_server(sse=True) as (endpoint, captured):
            with bridge(DeepWikiClient(endpoint)) as server:
                evidence = A2AResearchClient(server.endpoint).ask(REPO, "What affects TTFT?")
        self.assertEqual(evidence["transport"], "a2a")
        self.assertEqual(evidence["tool"], "ask_wiki_question")
        self.assertTrue(evidence["sources"])
        self.assertEqual([request["body"]["method"] for request in captured],
                         ["initialize", "notifications/initialized", "tools/list", "tools/call"])
        self.assertEqual(captured[-1]["body"]["params"]["arguments"],
                         {"repoName": REPO, "question": "What affects TTFT?"})

    def test_client_bridge_calls_researcher_and_returns_artifact(self):
        researcher = FakeResearcher()
        with bridge(researcher) as server:
            evidence = A2AResearchClient(server.endpoint).ask(REPO, "Analyze long TTFT", remaining_s=5)
            self.assertEqual(evidence["transport"], "a2a")
            self.assertEqual(evidence["repository"], REPO)
            self.assertEqual(researcher.calls[0][:2], (REPO, "Analyze long TTFT"))
            task = rpc(server, "tasks/get", {"id": evidence["task_id"]})["result"]
            self.assertEqual(task["status"]["state"], "completed")
            self.assertEqual(task["artifacts"][0]["parts"][0]["data"]["answer"], evidence["answer"])
            self.assertNotIn("history", task)

    def test_truncated_evidence_survives_bridge_and_cannot_authorize_candidate(self):
        from ascend_poc.open_research import candidate_changes, validate_review
        doc = "https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/graph_mode.html"
        code = "https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/envs.py"
        class TruncatedResearcher(FakeResearcher):
            def ask(self, repo, question, remaining_s=None):
                evidence = super().ask(repo, question, remaining_s)
                evidence.update(truncated=True, sources=[doc, code])
                return evidence
        with bridge(TruncatedResearcher()) as server:
            evidence = A2AResearchClient(server.endpoint).ask(REPO, "Review exact candidate")
        self.assertIs(evidence["truncated"], True)
        evidence["id"] = "research-1"
        reference, candidate = {"engine": {"x": 1}}, {"engine": {"x": 2}}
        review = {"decision": "run", "reason": "The fixture claims source support.",
                  "checked_changes": [row["path"] for row in candidate_changes(reference, candidate)],
                  "sources": [doc, code], "uncertainties": [], "evidence_id": evidence["id"]}
        with self.assertRaisesRegex(ValueError, "truncated"):
            validate_review(review, evidence, reference, candidate)

    def test_invalid_truncation_flags_are_rejected(self):
        for flag in ("true", 1, None):
            def change(stage, data):
                if stage == "response":
                    data["result"]["artifacts"][0]["parts"][0]["data"]["truncated"] = flag
                return data
            with self.subTest(flag=flag), alternate_agent(change) as endpoint:
                with self.assertRaisesRegex(A2AError, "truncation"):
                    A2AResearchClient(endpoint).ask(REPO, "q")

    def test_cancel_terminal_and_missing_task_errors(self):
        with bridge() as server:
            task = rpc(server, "message/send", query_params())["result"]
            self.assertEqual(rpc(server, "tasks/cancel", {"id": task["id"]})["error"]["code"], -32002)
            self.assertEqual(rpc(server, "tasks/get", {"id": "missing"})["error"]["code"], -32001)
            self.assertEqual(rpc(server, "message/stream", query_params())["error"]["code"], -32601)

    def test_localhost_card_keeps_same_origin(self):
        with bridge() as server:
            result = A2AResearchClient(server.endpoint.replace("127.0.0.1", "localhost")).ask(REPO, "q")
            self.assertEqual(result["transport"], "a2a")

    def test_unicode_question_budget_is_characters_not_ascii_json_bytes(self):
        question = "测试" * 7500
        fake = FakeResearcher()
        with bridge(fake) as server:
            result = A2AResearchClient(server.endpoint).ask(REPO, question)
            self.assertEqual(result["transport"], "a2a")
            # External JSON clients may use ensure_ascii=True (the rpc helper).
            self.assertEqual(rpc(server, "message/send", query_params(question=question))["result"]["status"]["state"], "completed")
            self.assertEqual(fake.calls[0][1], question)

    def test_repository_and_schema_cannot_expand_access(self):
        fake = FakeResearcher()
        with bridge(fake) as server:
            self.assertEqual(rpc(server, "message/send", query_params("other/repo"))["error"]["code"], -32602)
            bad = query_params()
            bad["message"]["parts"][0]["data"]["command"] = "exec"
            self.assertEqual(rpc(server, "message/send", bad)["error"]["code"], -32602)
            self.assertEqual(fake.calls, [])

    def test_failed_research_is_terminal_without_error_leak(self):
        with bridge(FakeResearcher(fail=True)) as server:
            task = rpc(server, "message/send", query_params())["result"]
            self.assertEqual(task["status"]["state"], "failed")
            self.assertNotIn("secret", json.dumps(task))
            with self.assertRaisesRegex(A2AError, "did not complete"):
                A2AResearchClient(server.endpoint).ask(REPO, "why?")

    def test_memory_bounded_task_store(self):
        with bridge(max_tasks=1) as server:
            first = rpc(server, "message/send", query_params())["result"]["id"]
            second = rpc(server, "message/send", query_params())["result"]["id"]
            self.assertEqual(len(server.tasks), 1)
            self.assertEqual(rpc(server, "tasks/get", {"id": first})["error"]["code"], -32001)
            self.assertEqual(rpc(server, "tasks/get", {"id": second})["result"]["id"], second)

    def test_nonlocal_binding_host_and_origin_blocked(self):
        with self.assertRaises(A2AError):
            create_research_server(host="0.0.0.0", researcher=FakeResearcher())
        with bridge() as server:
            for headers in ({"Host": "evil.example"}, {"Origin": "https://evil.example"}):
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    rpc(server, "message/send", query_params(), **headers)
                self.assertEqual(caught.exception.code, 403)

    def test_unsafe_endpoint_and_deadline_rejected(self):
        for endpoint in ("http://example.com", "https://x:secret@example.com/a2a", "https://example.com?a=1"):
            with self.assertRaises(A2AError):
                A2AResearchClient(endpoint)
        with self.assertRaises(A2AError):
            A2AResearchClient("http://127.0.0.1", timeout_s=float("nan"))
        with self.assertRaises(A2AError):
            A2AResearchClient("http://127.0.0.1").ask(REPO, "q", remaining_s=0)

    def test_card_version_and_origin_are_checked(self):
        for field, value in (("protocolVersion", "1.0"), ("url", "https://other.example/a2a"), ("preferredTransport", "HTTP+JSON")):
            def change(stage, data):
                if stage == "card":
                    data[field] = value
                return data
            with alternate_agent(change) as endpoint, self.assertRaises(A2AError):
                A2AResearchClient(endpoint).ask(REPO, "q")

    def test_rpc_id_and_incomplete_task_are_rejected(self):
        for issue in ("id", "working", "failed", "canceled", "input-required", "repository", "missing_artifact"):
            def change(stage, data):
                if stage == "response":
                    if issue == "id":
                        data["id"] = "wrong"
                    elif issue == "repository":
                        data["result"]["artifacts"][0]["parts"][0]["data"]["repository"] = "other/repo"
                    elif issue == "missing_artifact":
                        data["result"]["artifacts"] = []
                    else:
                        data["result"]["status"]["state"] = issue
                return data
            with alternate_agent(change) as endpoint, self.assertRaises(A2AError):
                A2AResearchClient(endpoint).ask(REPO, "q")

    def test_malformed_source_types_have_sanitized_errors(self):
        for source in (123, {}, [], "https://example.com:bad/source"):
            def change(stage, data):
                if stage == "response":
                    data["result"]["artifacts"][0]["parts"][0]["data"]["sources"] = [source]
                return data
            with alternate_agent(change) as endpoint, self.assertRaises(A2AError):
                A2AResearchClient(endpoint).ask(REPO, "q")

    def test_slow_response_does_not_exceed_caller_budget(self):
        with alternate_agent(lambda _stage, data: data, delay=.3) as endpoint:
            start = time.monotonic()
            with self.assertRaises(A2AError):
                A2AResearchClient(endpoint, timeout_s=2).ask(REPO, "q", remaining_s=.03)
            self.assertLess(time.monotonic() - start, .2)


if __name__ == "__main__":
    unittest.main()
