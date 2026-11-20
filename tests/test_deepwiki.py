import json
import threading
import time
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from ascend_poc.deepwiki import DeepWikiClient, DeepWikiError, extract_sources


def tool(name="ask_wiki_question", **extra):
    result = {"name": name, "inputSchema": {"type": "object", "properties": {
        "repoName": {"type": "string"}, "question": {"type": "string"}},
        "required": ["repoName", "question"]}}
    result.update(extra)
    return result


@contextmanager
def mcp_server(*, sse=False, advertised=None, customize=None, delay=0, hold_open=0):
    captured = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            captured.append({"headers": dict(self.headers), "body": data})
            method = data["method"]
            if delay:
                time.sleep(delay)
            if method == "notifications/initialized":
                self.send_response(202)
                self.end_headers()
                return
            result = {"initialize": {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}},
                      "tools/list": {"tools": advertised if advertised is not None else [tool()]},
                      "tools/call": {"content": [{"type": "text", "text": "Evidence [source](https://github.com/vllm-project/vllm-ascend/blob/main/README.md#L1)."}]}}[method]
            payload = {"jsonrpc": "2.0", "id": data["id"], "result": result}
            status, more_headers, override = customize(data, payload) if customize else (200, {}, None)
            encoded = json.dumps(payload).encode() if override is None else override
            self.send_response(status)
            self.send_header("Content-Type", "text/event-stream" if sse else "application/json")
            if method == "initialize":
                self.send_header("Mcp-Session-Id", "test-session")
            for key, value in more_headers.items():
                self.send_header(key, value)
            self.end_headers()
            try:
                if sse:
                    self.wfile.write(b': heartbeat\r\n\r\nevent: message\r\ndata: {"jsonrpc":"2.0","method":"notifications/progress","params":{}}\r\n\r\n')
                    self.wfile.write(b"event: message\r\ndata: " + encoded + b"\r\n\r\n")
                    self.wfile.flush()
                    if hold_open:
                        time.sleep(hold_open)
                else:
                    self.wfile.write(encoded)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=.005), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/mcp", captured
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


class DeepWikiTests(unittest.TestCase):
    def test_real_http_handshake_discovery_question_and_session(self):
        with mcp_server() as (endpoint, captured):
            answer = DeepWikiClient(endpoint).ask("vllm-project/vllm-ascend", "What affects TTFT?")
        self.assertEqual([request["body"]["method"] for request in captured],
                         ["initialize", "notifications/initialized", "tools/list", "tools/call"])
        self.assertNotIn("Authorization", captured[0]["headers"])
        for request in captured[1:]:
            normalized = {key.lower(): value for key, value in request["headers"].items()}
            self.assertEqual(normalized["mcp-session-id"], "test-session")
            self.assertEqual(normalized["mcp-protocol-version"], "2025-06-18")
        self.assertEqual(captured[-1]["body"]["params"]["arguments"],
                         {"repoName": "vllm-project/vllm-ascend", "question": "What affects TTFT?"})
        self.assertEqual(answer["tool"], "ask_wiki_question")
        self.assertEqual(answer["transport"], "mcp")
        self.assertEqual(answer["repository"], "vllm-project/vllm-ascend")
        self.assertEqual(len(answer["sources"]), 1)

    def test_sse_ignores_notifications_and_returns_before_stream_close(self):
        with mcp_server(sse=True, hold_open=.5) as (endpoint, captured):
            start = time.monotonic()
            result = DeepWikiClient(endpoint, timeout_s=1).ask("owner/repo", "question")
            self.assertLess(time.monotonic() - start, .4)
        self.assertIn("Evidence", result["answer"])

    def test_legacy_tool_and_array_repo_schema(self):
        legacy = tool("ask_question")
        legacy["inputSchema"]["properties"]["repoName"] = {"type": "array", "items": {"type": "string"}}
        with mcp_server(advertised=[legacy]) as (endpoint, captured):
            result = DeepWikiClient(endpoint).ask("owner/repo", "question")
        self.assertEqual(result["tool"], "ask_question")
        self.assertEqual(captured[-1]["body"]["params"]["arguments"]["repoName"], ["owner/repo"])

    def test_new_name_is_preferred_even_if_legacy_listed_first(self):
        with mcp_server(advertised=[tool("ask_question"), tool()]) as (endpoint, _):
            self.assertEqual(DeepWikiClient(endpoint).ask("owner/repo", "question")["tool"], "ask_wiki_question")

    def test_paginated_discovery_and_schema_names(self):
        renamed = tool()
        renamed["inputSchema"] = {"properties": {"repository": {"type": "string"}, "query": {"type": "string"}},
                                  "required": ["repository", "query"]}

        def customize(data, payload):
            if data["method"] == "tools/list" and not data.get("params", {}).get("cursor"):
                payload["result"] = {"tools": [], "nextCursor": "next-page"}
            return 200, {}, None

        with mcp_server(advertised=[renamed], customize=customize) as (endpoint, captured):
            DeepWikiClient(endpoint).ask("owner/repo", "question")
        self.assertEqual(captured[-1]["body"]["params"]["arguments"], {"repository": "owner/repo", "query": "question"})

    def test_tools_cannot_request_other_inputs(self):
        invalid = tool()
        invalid["inputSchema"]["required"].append("secret")
        for tools in ([], [tool("execute_shell")], [invalid]):
            with self.subTest(tools=tools), mcp_server(advertised=tools) as (endpoint, captured):
                with self.assertRaises(DeepWikiError):
                    DeepWikiClient(endpoint).ask("owner/repo", "question")
                self.assertNotIn("tools/call", [request["body"]["method"] for request in captured])

    def test_response_truncation_and_source_url_validation(self):
        with mcp_server() as (endpoint, _):
            result = DeepWikiClient(endpoint, max_answer_chars=8).ask("owner/repo", "question")
        self.assertEqual(result["answer"], "Evidence")
        self.assertTrue(result["truncated"])
        sources = extract_sources("bad https://user:password@example.com/ good https://example.com/a#L2). https://example.com/a#L2")
        self.assertEqual(sources, ["https://example.com/a#L2"])

    def test_remote_errors_are_sanitized(self):
        def customize(data, payload):
            if data["method"] == "tools/call":
                payload["result"] = {"isError": True, "content": [{"type": "text", "text": "PRIVATE-DIAGNOSTIC"}]}
            return 200, {}, None

        with mcp_server(customize=customize) as (endpoint, _):
            with self.assertRaises(DeepWikiError) as error:
                DeepWikiClient(endpoint).ask("owner/repo", "question")
        self.assertNotIn("PRIVATE", str(error.exception))

    def test_redirect_not_followed(self):
        def customize(data, payload):
            return 302, {"Location": "http://127.0.0.1:1/leak"}, b"PRIVATE"

        with mcp_server(customize=customize) as (endpoint, captured):
            with self.assertRaisesRegex(DeepWikiError, "HTTP 302"):
                DeepWikiClient(endpoint).ask("owner/repo", "question")
        self.assertEqual(len(captured), 1)

    def test_invalid_rpc_and_unknown_versions(self):
        def invalid_id(data, payload):
            payload["id"] = 999
            return 200, {}, None

        def unsupported(data, payload):
            payload["result"]["protocolVersion"] = "future-version"
            return 200, {}, None

        def invalid_json(data, payload):
            return 200, {}, b"not-json PRIVATE"

        def rpc_error(data, payload):
            payload.pop("result")
            payload["error"] = {"code": -1, "message": "PRIVATE"}
            return 200, {}, None

        for customize in (invalid_id, unsupported, invalid_json, rpc_error):
            with self.subTest(customize=customize), mcp_server(customize=customize) as (endpoint, _):
                with self.assertRaises(DeepWikiError) as error:
                    DeepWikiClient(endpoint).ask("owner/repo", "question")
                self.assertNotIn("PRIVATE", str(error.exception))

    def test_wall_clock_deadline_and_no_overlapping_query(self):
        with mcp_server(delay=.5) as (endpoint, captured):
            client = DeepWikiClient(endpoint, timeout_s=1)
            start = time.monotonic()
            with self.assertRaises(DeepWikiError):
                client.ask("owner/repo", "question", remaining_s=.05)
            self.assertLess(time.monotonic() - start, .3)
            # A request timeout may already have stopped the worker; either way
            # this exhausted-budget call cannot create another request.
            with self.assertRaises(DeepWikiError):
                client.ask("owner/repo", "question", remaining_s=0)
            self.assertEqual(len(captured), 1)

    def test_timed_out_worker_cannot_overlap_the_next_call(self):
        release = threading.Event()
        client = DeepWikiClient("http://127.0.0.1:1/mcp", timeout_s=.03)

        def blocked(*args):
            release.wait(1)
            return {"answer": "late"}

        with patch.object(client, "_ask", side_effect=blocked) as query:
            try:
                with self.assertRaisesRegex(DeepWikiError, "time budget"):
                    client.ask("owner/repo", "question")
                with self.assertRaisesRegex(DeepWikiError, "overlapping"):
                    client.ask("owner/repo", "question")
                self.assertEqual(query.call_count, 1)
            finally:
                release.set()

    def test_size_limit_for_json_and_sse(self):
        def customize(data, payload):
            return 200, {}, b'x' * 2048
        for sse in (True, False):
            with self.subTest(sse=sse), mcp_server(sse=sse, customize=customize) as (endpoint, _):
                client = DeepWikiClient(endpoint)
                client.MAX_RESPONSE_BYTES = 1024
                with self.assertRaisesRegex(DeepWikiError, "size limit"):
                    client.ask("owner/repo", "question")

    def test_invalid_input_does_not_touch_network(self):
        for endpoint in ("http://example.com/mcp", "https://a:b@example.com/", "https://example.com/?key=x", "https://example.com/#fragment", "file:///tmp/data"):
            with self.subTest(endpoint=endpoint), self.assertRaises(DeepWikiError):
                DeepWikiClient(endpoint)
        client = DeepWikiClient("http://127.0.0.1:1/mcp")
        for repo, question, remaining in (("https://github.com/a/b", "q", 1), ("a/b", "", 1), ("a/b", "q", float("nan")), ("a/b", "q", -1)):
            with self.subTest(repo=repo, question=question), self.assertRaises(DeepWikiError):
                client.ask(repo, question, remaining)


if __name__ == "__main__":
    unittest.main()
