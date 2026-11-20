"""Version-pinned A2A 0.3 JSON-RPC research bridge for DeepWiki MCP.

The public DeepWiki service speaks MCP, not A2A. This optional loopback agent
implements discovery, synchronous message/send, tasks/get and tasks/cancel.
Streaming, push notifications, remote deployment and A2A 1.x are not supported.
Only a structured repository/question request is accepted; no execution tools
or arbitrary instructions are exposed by this agent.
"""
from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import math
import queue
import re
import threading
import time
from urllib.parse import urlsplit, urlunsplit
import urllib.error
import urllib.request
import uuid


PROTOCOL_VERSION = "0.3.0"
# Permit a 16k-character question even when another JSON client emits Unicode
# escapes (six ASCII bytes for a BMP character). The semantic limit is separate.
MAX_BODY = 131_072
MAX_RESPONSE = 262_144
MAX_QUESTION = 16_000


class A2AError(RuntimeError):
    """Sanitized research transport, schema or deadline failure."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _url(value: str) -> tuple[str, str]:
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        port = parsed.port
        local = host == "localhost"
        if host:
            try:
                local = local or ipaddress.ip_address(host).is_loopback
            except ValueError:
                pass
        if (not host or parsed.username or parsed.password or parsed.query or parsed.fragment
                or (parsed.scheme != "https" and not (parsed.scheme == "http" and local))):
            raise ValueError
        # Normalize default ports for exact same-origin checks.
        origin = f"{parsed.scheme}://{host.lower()}:{port or (443 if parsed.scheme == 'https' else 80)}"
        return value.rstrip("/"), origin
    except (ValueError, TypeError, AttributeError):
        raise A2AError("A2A endpoint requires HTTPS or loopback HTTP without credentials or query.") from None


def _positive(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise A2AError(f"{name} must be a positive finite number.")
    return float(value)


def _json(raw: bytes):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))


def _request(url: str, payload: dict | None, deadline: float) -> dict:
    """Enforce a wall-clock caller deadline as well as socket/body limits."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise A2AError("A2A research deadline exhausted.")
    result = queue.Queue(maxsize=1)

    def perform():
        try:
            request = urllib.request.Request(url, data=None if payload is None else json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(),
                                             headers={"Accept": "application/json", "Content-Type": "application/json"})
            with urllib.request.build_opener(_NoRedirect()).open(request, timeout=remaining) as response:
                if response.headers.get_content_type() != "application/json":
                    raise A2AError("A2A endpoint returned a non-JSON response.")
                raw = response.read(MAX_RESPONSE + 1)
            if len(raw) > MAX_RESPONSE:
                raise A2AError("A2A response exceeds the size limit.")
            decoded = _json(raw)
            if not isinstance(decoded, dict):
                raise A2AError("A2A response must be an object.")
            result.put((decoded, None))
        except Exception as error:
            result.put((None, error if isinstance(error, A2AError) else A2AError("A2A request failed or returned invalid JSON.")))

    threading.Thread(target=perform, daemon=True).start()
    try:
        data, error = result.get(timeout=max(.001, deadline - time.monotonic()))
    except queue.Empty:
        raise A2AError("A2A research deadline exhausted.") from None
    if error:
        raise error from None
    return data


def _question(repo: str, question: str):
    if not isinstance(repo, str) or len(repo) > 200 or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise A2AError("Research requires a public owner/repository identifier.")
    if not isinstance(question, str) or not question.strip() or len(question) > MAX_QUESTION:
        raise A2AError("Research question is empty or exceeds the size limit.")


def _evidence(value: dict, repo: str) -> dict:
    if not isinstance(value, dict) or value.get("repository") != repo:
        raise A2AError("A2A research evidence has a different repository.")
    answer = value.get("answer")
    if not isinstance(answer, str) or not answer.strip() or len(answer) > 64_000:
        raise A2AError("A2A research evidence is missing a bounded answer.")
    links = value.get("sources", [])
    if not isinstance(links, list) or len(links) > 100:
        raise A2AError("A2A research evidence has invalid sources.")
    for link in links:
        try:
            if not isinstance(link, str) or len(link) > 4096:
                raise ValueError
            p = urlsplit(link)
            _ = p.port
            valid = p.scheme in {"http", "https"} and p.hostname and not p.username and not p.password
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise A2AError("A2A research evidence contains an invalid source URL.")
    tool = value.get("tool", "ask_wiki_question")
    if not isinstance(tool, str) or len(tool) > 100:
        raise A2AError("A2A research tool name is invalid.")
    return {"answer": answer, "sources": links, "repository": repo, "transport": "a2a", "tool": tool}


class A2AResearchClient:
    def __init__(self, endpoint: str, timeout_s: float = 60):
        self.endpoint, self.origin = _url(endpoint)
        self.timeout_s = _positive(timeout_s, "timeout_s")

    def ask(self, repo: str, question: str, remaining_s: float | None = None) -> dict:
        _question(repo, question)
        timeout = self.timeout_s if remaining_s is None else min(self.timeout_s, _positive(remaining_s, "remaining_s"))
        deadline = time.monotonic() + timeout
        parsed = urlsplit(self.endpoint)
        card_url = urlunsplit((parsed.scheme, parsed.netloc, "/.well-known/agent-card.json", "", ""))
        card = _request(card_url, None, deadline)
        if card.get("protocolVersion") != PROTOCOL_VERSION or card.get("preferredTransport") != "JSONRPC":
            raise A2AError("Research agent must advertise A2A 0.3.0 JSONRPC.")
        endpoint, origin = _url(card.get("url"))
        if origin != self.origin:
            raise A2AError("Research agent advertised a different origin.")
        if card.get("security"):
            raise A2AError("Authenticated remote A2A agents are not supported by this client.")
        request_id = uuid.uuid4().hex
        response = _request(endpoint, {
            "jsonrpc": "2.0", "id": request_id, "method": "message/send",
            "params": {"message": {"kind": "message", "role": "user", "messageId": uuid.uuid4().hex,
                                   "parts": [{"kind": "data", "data": {"repository": repo, "question": question}}]},
                       "configuration": {"blocking": True, "acceptedOutputModes": ["application/json"]}},
        }, deadline)
        if response.get("jsonrpc") != "2.0" or response.get("id") != request_id:
            raise A2AError("A2A response does not match the request.")
        if "error" in response or "result" not in response:
            raise A2AError("Research agent rejected the A2A request.")
        task = response["result"]
        if not isinstance(task, dict) or task.get("kind") != "task" or not isinstance(task.get("id"), str):
            raise A2AError("Research agent must return a task.")
        status = task.get("status")
        if not isinstance(status, dict) or status.get("state") != "completed":
            raise A2AError("Research agent did not complete the research task.")
        artifacts = task.get("artifacts")
        if not isinstance(artifacts, list):
            raise A2AError("Completed research task has no evidence artifact.")
        evidence = []
        for artifact in artifacts:
            if isinstance(artifact, dict) and isinstance(artifact.get("parts"), list):
                evidence.extend(part.get("data") for part in artifact["parts"]
                                if isinstance(part, dict) and part.get("kind") == "data")
        if len(evidence) != 1:
            raise A2AError("Research task must contain exactly one evidence artifact.")
        answer = _evidence(evidence[0], repo)
        answer["task_id"] = task["id"]
        return answer


class ResearchServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False
    allow_reuse_address = True

    def __init__(self, address, researcher, repository: str, timeout_s: float, max_tasks: int):
        self.researcher = researcher
        self.repository = repository
        self.timeout_s = timeout_s
        self.max_tasks = max_tasks
        self.tasks = OrderedDict()
        self.task_lock = threading.Lock()
        self.requests = threading.BoundedSemaphore(8)
        super().__init__(address, _ResearchHandler)
        self.endpoint = f"http://127.0.0.1:{self.server_port}/a2a"
        self.allowed_hosts = {f"127.0.0.1:{self.server_port}", f"localhost:{self.server_port}"}

    def get_request(self):
        sock, address = super().get_request()
        sock.settimeout(min(10, self.timeout_s))
        return sock, address

    def store_task(self, task):
        with self.task_lock:
            self.tasks[task["id"]] = task
            while len(self.tasks) > self.max_tasks:
                self.tasks.popitem(last=False)

    def card(self):
        return {"protocolVersion": PROTOCOL_VERSION, "name": "Ascend DeepWiki Research Agent",
                "description": "Synchronous repository research through the official DeepWiki MCP server.",
                "version": "0.1.0", "url": self.endpoint, "preferredTransport": "JSONRPC",
                "capabilities": {"streaming": False, "pushNotifications": False},
                "defaultInputModes": ["application/json"], "defaultOutputModes": ["application/json"],
                "skills": [{"id": "deepwiki_research", "name": "Repository research", "tags": ["deepwiki", "ascend"],
                            "description": "Accepts one DataPart with repository and question. Returns bounded research evidence."}]}


class _ResearchHandler(BaseHTTPRequestHandler):
    server: ResearchServer

    def log_message(self, *_args):
        # No questions, source contents or request credentials enter access logs.
        pass

    def _send(self, value, status=200):
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _safe_host(self):
        if self.headers.get("Host", "").lower() not in self.server.allowed_hosts:
            self._send({"error": "Invalid Host"}, 403)
            return False
        # The CLI does not send an Origin header. Browser calls must be local.
        origin = self.headers.get("Origin")
        if origin and origin not in {f"http://{host}" for host in self.server.allowed_hosts}:
            self._send({"error": "Invalid Origin"}, 403)
            return False
        return True

    def do_GET(self):
        if not self._safe_host():
            return
        if self.path == "/.well-known/agent-card.json":
            card = self.server.card()
            # Preserve the client's explicit localhost/127.0.0.1 origin after
            # validating Host. The advertised address never comes from an
            # unvalidated forwarding header.
            card["url"] = f"http://{self.headers['Host']}/a2a"
            self._send(card)
        else:
            self._send({"error": "Not found"}, 404)

    def _error(self, request_id, code, message):
        self._send({"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}})

    def do_POST(self):
        if not self._safe_host():
            return
        if self.path != "/a2a":
            self._send({"error": "Not found"}, 404)
            return
        if self.headers.get_content_type() != "application/json" or self.headers.get("Transfer-Encoding"):
            self._send({"error": "JSON Content-Length requests only"}, 415)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if not 0 < length <= MAX_BODY:
            self._send({"error": "Request size invalid"}, 413)
            return
        if not self.server.requests.acquire(blocking=False):
            self._send({"error": "Research agent busy"}, 503)
            return
        try:
            try:
                request = _json(self.rfile.read(length))
            except (ValueError, UnicodeDecodeError, OSError):
                self._error(None, -32700, "Parse error")
                return
            if (not isinstance(request, dict) or request.get("jsonrpc") != "2.0"
                    or isinstance(request.get("id"), bool) or not isinstance(request.get("id"), (str, int))):
                self._error(None, -32600, "Invalid Request")
                return
            request_id, method = request["id"], request.get("method")
            params = request.get("params", {})
            if not isinstance(params, dict):
                self._error(request_id, -32602, "Invalid params")
                return
            if method in {"tasks/get", "tasks/cancel"}:
                task_id = params.get("id")
                if not isinstance(task_id, str):
                    self._error(request_id, -32602, "Invalid params")
                    return
                with self.server.task_lock:
                    task = self.server.tasks.get(task_id)
                if task is None:
                    self._error(request_id, -32001, "Task not found")
                elif method == "tasks/cancel":
                    self._error(request_id, -32002, "Task cannot be canceled")
                else:
                    self._send({"jsonrpc": "2.0", "id": request_id, "result": task})
                return
            if method != "message/send":
                self._error(request_id, -32601, "Method not found")
                return
            message = params.get("message")
            try:
                if not isinstance(message, dict) or message.get("role") != "user" or message.get("kind") != "message":
                    raise A2AError("Invalid message")
                if not isinstance(message.get("messageId"), str) or not 0 < len(message["messageId"]) <= 200 or message.get("taskId"):
                    raise A2AError("Invalid message ID or unsupported task continuation")
                parts = message.get("parts")
                if not isinstance(parts, list) or len(parts) != 1 or not isinstance(parts[0], dict) or parts[0].get("kind") != "data":
                    raise A2AError("Use one data part")
                data = parts[0].get("data")
                if not isinstance(data, dict) or set(data) != {"repository", "question"}:
                    raise A2AError("Use repository and question")
                _question(data["repository"], data["question"])
                if data["repository"] != self.server.repository:
                    raise A2AError("Repository not allowed")
                context = message.get("contextId", uuid.uuid4().hex)
                if not isinstance(context, str) or not 0 < len(context) <= 200:
                    raise A2AError("Invalid context")
            except A2AError:
                self._error(request_id, -32602, "Invalid research request or repository")
                return
            task = {"kind": "task", "id": uuid.uuid4().hex, "contextId": context,
                    "status": {"state": "working"}}
            try:
                evidence = self.server.researcher.ask(data["repository"], data["question"], remaining_s=self.server.timeout_s)
                evidence = _evidence(evidence, data["repository"])
                task["status"]["state"] = "completed"
                task["artifacts"] = [{"artifactId": uuid.uuid4().hex, "name": "research-evidence",
                                      "parts": [{"kind": "data", "data": evidence}]}]
            except Exception:
                # Never reflect upstream error text, endpoint credentials or logs.
                task["status"]["state"] = "failed"
            task["status"]["timestamp"] = datetime.now(timezone.utc).isoformat()
            self.server.store_task(task)
            self._send({"jsonrpc": "2.0", "id": request_id, "result": task})
        finally:
            self.server.requests.release()


def create_research_server(host: str = "127.0.0.1", port: int = 8766, *,
                           mcp_endpoint: str = "https://mcp.deepwiki.com/mcp", timeout_s: float = 60,
                           repository: str = "vllm-project/vllm-ascend", researcher=None,
                           max_tasks: int = 256) -> ResearchServer:
    if host not in {"127.0.0.1", "localhost"}:
        raise A2AError("Built-in A2A research server binds only to IPv4 loopback.")
    if type(port) is not int or not 0 <= port <= 65535 or type(max_tasks) is not int or not 1 <= max_tasks <= 4096:
        raise A2AError("Invalid research server port or task limit.")
    timeout_s = _positive(timeout_s, "timeout_s")
    _question(repository, "validate")
    if researcher is None:
        from .deepwiki import DeepWikiClient
        researcher = DeepWikiClient(endpoint=mcp_endpoint, timeout_s=timeout_s)
    return ResearchServer(("127.0.0.1", port), researcher, repository, timeout_s, max_tasks)


def serve_research_agent(host: str = "127.0.0.1", port: int = 8766, *,
                         mcp_endpoint: str = "https://mcp.deepwiki.com/mcp", timeout_s: float = 60,
                         repository: str = "vllm-project/vllm-ascend") -> None:
    with create_research_server(host, port, mcp_endpoint=mcp_endpoint, timeout_s=timeout_s, repository=repository) as server:
        print(f"A2A 0.3.0 research agent: {server.endpoint}", flush=True)
        server.serve_forever(poll_interval=.2)
