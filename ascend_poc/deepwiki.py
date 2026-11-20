"""Bounded, read-only DeepWiki access through MCP Streamable HTTP.

This is an MCP client, not an A2A endpoint. Only the advertised repository
question tool is invoked; server instructions and answer text are untrusted data.
No local files, environment variables, or authentication headers are attached.
"""
from __future__ import annotations

import ipaddress
import json
import math
import queue
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


class DeepWikiError(RuntimeError):
    """Sanitized protocol, input, or network error."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def validate_endpoint(endpoint: str) -> str:
    """Allow configured HTTPS services or an HTTP loopback test/bridge."""
    try:
        url = urllib.parse.urlsplit(endpoint)
        host = url.hostname
        url.port
        local = host == "localhost"
        try:
            local = local or ipaddress.ip_address(host or "").is_loopback
        except ValueError:
            pass
        if (not host or url.username or url.password or url.query or url.fragment
                or (url.scheme != "https" and not (url.scheme == "http" and local))
                or any(ord(c) < 33 for c in endpoint)):
            raise ValueError()
    except (TypeError, ValueError):
        raise DeepWikiError("DeepWiki endpoint requires HTTPS or loopback HTTP without credentials, query or fragment.") from None
    return endpoint


def extract_sources(text: str) -> list[str]:
    """Keep source URLs as evidence, never as destinations to execute/fetch."""
    sources = []
    for match in re.findall(r"https?://[^\s<>\[\]{}\"']+", text):
        candidate = match.rstrip(".,;:!?)")
        try:
            parts = urllib.parse.urlsplit(candidate)
            if not parts.hostname or parts.username or parts.password:
                continue
            parts.port
        except ValueError:
            continue
        if candidate not in sources and len(candidate) <= 2000:
            sources.append(candidate)
        if len(sources) >= 30:
            break
    return sources


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class DeepWikiClient:
    """A fresh negotiated MCP session per question, without hidden retries.

    timeout_s bounds the entire initialize/discover/query exchange, including
    wall-clock DNS waits. A timed-out worker cannot overlap a second call on
    this client. Stream processing returns as soon as its RPC response arrives.
    """

    MAX_RESPONSE_BYTES = 1_048_576
    PROTOCOL_VERSIONS = {"2025-06-18", "2025-03-26"}

    def __init__(self, endpoint="https://mcp.deepwiki.com/mcp", timeout_s=60, max_answer_chars=16000):
        self.endpoint = validate_endpoint(endpoint)
        if not _number(timeout_s) or timeout_s <= 0:
            raise DeepWikiError("DeepWiki timeout_s must be a positive finite number.")
        if type(max_answer_chars) is not int or not 1 <= max_answer_chars <= 100_000:
            raise DeepWikiError("DeepWiki max_answer_chars must be an integer from 1 to 100000.")
        self.timeout_s = float(timeout_s)
        self.max_answer_chars = max_answer_chars
        self._busy = threading.Lock()

    def ask(self, repo: str, question: str, remaining_s: float | None = None) -> dict:
        if not isinstance(repo, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
            raise DeepWikiError("DeepWiki repository must be an owner/repo identifier.")
        if not isinstance(question, str) or not question.strip() or len(question) > 24000 or "\x00" in question:
            raise DeepWikiError("DeepWiki question must contain 1 to 24000 characters without NUL.")
        if remaining_s is not None and (not _number(remaining_s) or remaining_s <= 0):
            raise DeepWikiError("No time remains for DeepWiki research.")
        timeout = min(self.timeout_s, remaining_s) if remaining_s is not None else self.timeout_s
        if not self._busy.acquire(blocking=False):
            raise DeepWikiError("A previous DeepWiki request is still finishing; no overlapping query was started.")
        result = queue.Queue(maxsize=1)
        deadline = time.monotonic() + timeout

        def worker():
            try:
                result.put(self._ask(repo, question.strip(), deadline))
            except DeepWikiError as error:
                result.put(error)
            except Exception:
                result.put(DeepWikiError("DeepWiki request failed; no remote diagnostic text was retained."))
            finally:
                self._busy.release()

        threading.Thread(target=worker, daemon=True, name="poc-deepwiki-request").start()
        try:
            outcome = result.get(timeout=max(.001, deadline - time.monotonic()))
        except queue.Empty:
            raise DeepWikiError("DeepWiki request exceeded its time budget.") from None
        if isinstance(outcome, DeepWikiError):
            raise outcome
        return outcome

    def _ask(self, repo: str, question: str, deadline: float) -> dict:
        opener = urllib.request.build_opener(_NoRedirect())
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        sequence = 0

        def rpc(method, params=None, *, notification=False):
            nonlocal sequence
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DeepWikiError("DeepWiki request exceeded its time budget.")
            payload = {"jsonrpc": "2.0", "method": method}
            if params is not None:
                payload["params"] = params
            if not notification:
                sequence += 1
                payload["id"] = sequence
            request = urllib.request.Request(self.endpoint, data=json.dumps(payload, allow_nan=False).encode(), headers=headers)
            try:
                with opener.open(request, timeout=remaining) as response:
                    if method == "initialize":
                        session = response.headers.get("Mcp-Session-Id")
                        if session:
                            if len(session) > 1024 or not all(33 <= ord(c) <= 126 for c in session):
                                raise DeepWikiError("DeepWiki returned an invalid session identifier.")
                            headers["Mcp-Session-Id"] = session
                    if notification:
                        if response.status not in {200, 202, 204}:
                            raise DeepWikiError("DeepWiki rejected initialization notification.")
                        return None
                    content_type = response.headers.get_content_type()
                    if content_type == "text/event-stream":
                        return self._read_events(response, sequence, deadline)
                    if content_type != "application/json":
                        raise DeepWikiError("DeepWiki returned an unsupported response content type.")
                    data = response.read(self.MAX_RESPONSE_BYTES + 1)
                    if len(data) > self.MAX_RESPONSE_BYTES:
                        raise DeepWikiError("DeepWiki response exceeded the size limit.")
                    return self._result(self._json(data), sequence)
            except urllib.error.HTTPError as error:
                status = error.code
                error.close()
                raise DeepWikiError(f"DeepWiki returned HTTP {status}.") from None
            except (urllib.error.URLError, OSError, TimeoutError):
                raise DeepWikiError("DeepWiki network request failed or timed out.") from None

        initialized = rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                        "clientInfo": {"name": "ascend-poc-autotuner", "version": "0.2.0"}})
        if not isinstance(initialized, dict) or initialized.get("protocolVersion") not in self.PROTOCOL_VERSIONS:
            raise DeepWikiError("DeepWiki negotiated an unsupported MCP version.")
        if not isinstance(initialized.get("capabilities"), dict) or "tools" not in initialized["capabilities"]:
            raise DeepWikiError("DeepWiki does not advertise MCP tools.")
        headers["MCP-Protocol-Version"] = initialized["protocolVersion"]
        rpc("notifications/initialized", notification=True)
        discovered = []
        cursor = None
        seen_cursors = set()
        for _ in range(10):
            listing = rpc("tools/list", {"cursor": cursor} if cursor else {})
            if not isinstance(listing, dict) or not isinstance(listing.get("tools"), list):
                raise DeepWikiError("DeepWiki returned an invalid tools list.")
            discovered.extend(listing["tools"])
            cursor = listing.get("nextCursor")
            if not cursor:
                break
            if not isinstance(cursor, str) or len(cursor) > 2000 or cursor in seen_cursors:
                raise DeepWikiError("DeepWiki returned invalid tool pagination.")
            seen_cursors.add(cursor)
        else:
            raise DeepWikiError("DeepWiki tool pagination exceeded its limit.")
        tools = {item.get("name"): item for item in discovered if isinstance(item, dict) and isinstance(item.get("name"), str)}
        tool = next((tools[name] for name in ("ask_wiki_question", "ask_question") if name in tools), None)
        if tool is None:
            raise DeepWikiError("DeepWiki does not advertise a supported repository question tool.")
        schema = tool.get("inputSchema", {})
        if not isinstance(schema, dict) or schema.get("type", "object") != "object":
            raise DeepWikiError("DeepWiki question tool has an unsupported input schema.")
        props = schema.get("properties", {})
        if not isinstance(props, dict):
            raise DeepWikiError("DeepWiki question tool has an unsupported input schema.")
        repo_key = next((key for key in ("repoName", "repo_name", "repository", "repo") if key in props), None)
        question_key = next((key for key in ("question", "query") if key in props), None)
        if repo_key is None or question_key is None:
            raise DeepWikiError("DeepWiki question tool has an unsupported input schema.")
        repo_schema, question_schema = props[repo_key], props[question_key]
        if not isinstance(repo_schema, dict) or not isinstance(question_schema, dict):
            raise DeepWikiError("DeepWiki question tool has an unsupported input schema.")
        repo_type = repo_schema.get("type", "string")
        repo_types = repo_type if isinstance(repo_type, list) else [repo_type]
        if "string" in repo_types:
            repo_value = repo
        elif "array" in repo_types and repo_schema.get("items", {}).get("type") == "string":
            repo_value = [repo]
        else:
            raise DeepWikiError("DeepWiki repository field has an unsupported type.")
        if question_schema.get("type", "string") != "string":
            raise DeepWikiError("DeepWiki question field has an unsupported type.")
        required = schema.get("required", [])
        if not isinstance(required, list) or any(key not in {repo_key, question_key} for key in required):
            raise DeepWikiError("DeepWiki question tool requires unsupported arguments.")
        result = rpc("tools/call", {"name": tool["name"], "arguments": {repo_key: repo_value, question_key: question}})
        if not isinstance(result, dict) or result.get("isError"):
            raise DeepWikiError("DeepWiki question tool failed.")
        content = result.get("content", [])
        if not isinstance(content, list):
            raise DeepWikiError("DeepWiki question tool returned invalid content.")
        text = "\n".join(item["text"] for item in content if isinstance(item, dict)
                         and item.get("type") == "text" and isinstance(item.get("text"), str)).strip()
        if not text:
            raise DeepWikiError("DeepWiki question tool returned no textual evidence.")
        return {"answer": text[:self.max_answer_chars], "sources": extract_sources(text),
                "repository": repo, "transport": "mcp", "tool": tool["name"],
                "truncated": len(text) > self.max_answer_chars}

    @staticmethod
    def _json(data):
        try:
            return json.loads(data, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        except (UnicodeError, ValueError, TypeError):
            raise DeepWikiError("DeepWiki returned invalid JSON.") from None

    @staticmethod
    def _result(payload, request_id):
        if (not isinstance(payload, dict) or payload.get("jsonrpc") != "2.0"
                or type(payload.get("id")) is not int or payload["id"] != request_id):
            raise DeepWikiError("DeepWiki returned a mismatched JSON-RPC response.")
        if "error" in payload:
            raise DeepWikiError("DeepWiki returned a JSON-RPC error.")
        if "result" not in payload:
            raise DeepWikiError("DeepWiki returned no JSON-RPC result.")
        return payload["result"]

    def _read_events(self, response, request_id, deadline):
        total = 0
        event = []
        while True:
            if time.monotonic() >= deadline:
                raise DeepWikiError("DeepWiki event stream exceeded its time budget.")
            line = response.readline(self.MAX_RESPONSE_BYTES - total + 1)
            total += len(line)
            if total > self.MAX_RESPONSE_BYTES:
                raise DeepWikiError("DeepWiki event stream exceeded the size limit.")
            if line in {b"\n", b"\r\n", b""}:
                if event:
                    payload = self._json(b"\n".join(event))
                    event = []
                    if isinstance(payload, dict) and payload.get("id") == request_id and "method" not in payload:
                        return self._result(payload, request_id)
                if not line:
                    raise DeepWikiError("DeepWiki event stream ended without the requested result.")
            elif line.startswith(b"data:"):
                event.append(line[5:].lstrip(b" ").rstrip(b"\r\n"))
