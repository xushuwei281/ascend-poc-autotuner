import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from ascend_poc.runner import (
    RunnerError, _read_result, render_command, run_command, run_trial,
    stop_process_group,
)


SERVER = """
import http.server, sys
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'ok')
    def log_message(self, *args): pass
http.server.HTTPServer(('127.0.0.1', int(sys.argv[1])), Handler).serve_forever()
"""


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def alive(pid):
    try:
        # A zombie no longer executes and cannot retain accelerator resources.
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except (FileNotFoundError, ProcessLookupError):
        return False


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_process_probe_handles_exit_during_proc_read(self):
        with patch.object(Path, "read_text", side_effect=ProcessLookupError(3, "No such process")):
            self.assertFalse(alive(999999))
        with patch.object(Path, "read_text", side_effect=PermissionError(13, "Permission denied")):
            with self.assertRaises(PermissionError):
                alive(999999)

    def config(self, code=None):
        port = free_port()
        if code is None:
            code = "import json,sys; json.dump({'output_tokens_per_second':123,'p99_ttft_ms':40}, open(sys.argv[1], 'w'))"
        return {
            "_repo_dir": str(self.root),
            "llm": {"api_key_env": "TEST_POC_API_SECRET"},
            "server": {
                "command": [sys.executable, "-c", SERVER, str(port)],
                "health_url": f"http://127.0.0.1:{port}/health",
                "startup_timeout_s": 2,
                "shutdown_timeout_s": 0.2,
            },
            "scenarios": [{
                "name": "load", "command": [sys.executable, "-c", code, "{trial_dir}/metrics.json"],
                "result_file": "{trial_dir}/metrics.json", "metrics_format": "canonical",
                "timeout_s": 2,
            }],
        }

    def test_templates_preserve_json_and_shell_metacharacters(self):
        candidate = {"value": "$(touch /tmp/should-not-run); a b", "enabled": True}
        argv = render_command(["echo", "{param.value}", "{param.enabled}", '{"method":"mtp"}', "{env:MODEL}"],
                              candidate, self.root, self.root, {"MODEL": "/a model"})
        self.assertEqual(argv, ["echo", candidate["value"], "true", '{"method":"mtp"}', "/a model"])
        with self.assertRaises(RunnerError):
            render_command(["{env:MISSING}"], candidate, self.root, self.root, {})
        with self.assertRaises(RunnerError):
            render_command(["{unknown}"], candidate, self.root, self.root, {})

    def test_run_command_does_not_invoke_shell(self):
        marker = self.root / "injected"
        output = self.root / "argument"
        arg = f"$(touch {marker})"
        run_command([sys.executable, "-c", "import sys,pathlib; pathlib.Path(sys.argv[1]).write_text(sys.argv[2])", str(output), arg],
                    cwd=self.root, env=dict(os.environ), log_path=self.root / "log", timeout_s=2,
                    deadline=time.monotonic() + 3)
        self.assertEqual(output.read_text(), arg)
        self.assertFalse(marker.exists())

    def test_measure_repetitions_and_stop_server(self):
        config = self.config()
        trial = self.root / "trial"
        result = run_trial(config, {}, trial, time.monotonic() + 10, 2)
        self.assertEqual(len(result[0]["metrics"]), 2)
        self.assertEqual(result[0]["metrics"][0]["output_tokens_per_second"], 123)
        pid = json.loads((trial / "server-process.json").read_text())["pid"]
        self.assertFalse(alive(pid))
        self.assertTrue((trial / "scenario-00/repeat-001/normalized.json").exists())
        with self.assertRaisesRegex(RunnerError, "already exists"):
            run_trial(config, {}, trial, time.monotonic() + 5, 1)

    def test_removes_llm_api_key_from_children(self):
        code = "import os,json,sys; json.dump({'key_absent': int('TEST_POC_API_SECRET' not in os.environ)},open(sys.argv[1],'w'))"
        config = self.config(code)
        # An override cannot accidentally put the tuning API credential back.
        config["server"]["env"] = {"TEST_POC_API_SECRET": "secret-override"}
        with patch.dict(os.environ, {"TEST_POC_API_SECRET": "secret-original"}):
            result = run_trial(config, {}, self.root / "trial", time.monotonic() + 5, 1)
        self.assertEqual(result[0]["metrics"][0]["key_absent"], 1)
        self.assertNotIn("secret-", (self.root / "trial/commands.json").read_text())

    def test_server_environment_placeholders(self):
        code = "import os,json,sys; json.dump({'hccl':int(os.environ['HCCL_BUFFSIZE']),'original_env':int(os.environ['INHERITED_COPY'] == 'original'),'overridden_env':int(os.environ['TEST_SOURCE'] == 'changed'),'repo_matches':int(os.environ['REPO_DIR'] == os.getcwd()),'trial_exists':int(os.path.isdir(os.environ['TRIAL_DIR']))},open(sys.argv[1],'w'))"
        config = self.config(code)
        config["server"]["env"] = {
            "HCCL_BUFFSIZE": "{param.hccl}",
            "TEST_SOURCE": "changed",
            "INHERITED_COPY": "{env:TEST_SOURCE}",
            "REPO_DIR": "{repo_dir}",
            "TRIAL_DIR": "{trial_dir}",
        }
        with patch.dict(os.environ, {"TEST_SOURCE": "original"}):
            result = run_trial(config, {"hccl": 256}, self.root / "trial", time.monotonic() + 5, 1)
        self.assertEqual(result[0]["metrics"][0], {
            "hccl": 256, "original_env": 1, "overridden_env": 1, "repo_matches": 1, "trial_exists": 1,
        })

    def test_environment_cannot_expand_llm_credential(self):
        config = self.config()
        config["server"]["env"] = {"LEAK": "{env:TEST_POC_API_SECRET}"}
        with patch.dict(os.environ, {"TEST_POC_API_SECRET": "secret-value"}):
            with self.assertRaisesRegex(RunnerError, "Required environment variable is missing"):
                run_trial(config, {}, self.root / "trial", time.monotonic() + 5, 1)
        self.assertFalse((self.root / "trial/server-process.json").exists())

    def test_explicit_benchmark_env_isolated_from_server_knobs_and_secrets(self):
        code = "import os,json,sys; json.dump({'output_len':int(os.environ['OUTPUT_LEN']),'key_absent':int('TEST_POC_API_SECRET' not in os.environ),'inherited':int(os.environ['INHERITED'] == 'source')},open(sys.argv[1],'w'))"
        config = self.config(code)
        server_marker = self.root / "server-output-len.txt"
        config["server"]["command"][2] = "import os,pathlib;pathlib.Path(" + repr(str(server_marker)) + ").write_text(os.environ['OUTPUT_LEN'])\n" + SERVER
        config["server"]["env"] = {"OUTPUT_LEN": "{param.length}", "INHERITED": "server-only"}
        config["benchmark_env"] = {"OUTPUT_LEN": "512", "TEST_POC_API_SECRET": "blocked", "INHERITED": "{env:INHERITED}"}
        scenario = config["scenarios"][0]
        hook = [sys.executable, "-c", "import os; assert os.environ['OUTPUT_LEN'] == '512'; assert 'TEST_POC_API_SECRET' not in os.environ"]
        scenario["before_each_command"] = hook
        scenario["warmup_command"] = hook
        scenario["quality_command"] = [sys.executable, "-c", "import os,json,sys;json.dump({'quality_len':int(os.environ['OUTPUT_LEN'])},open(sys.argv[1],'w'))", "{trial_dir}/quality.json"]
        scenario["quality_result_file"] = "{trial_dir}/quality.json"
        with patch.dict(os.environ, {"INHERITED": "source", "TEST_POC_API_SECRET": "secret-original"}):
            result = run_trial(config, {"length": 64}, self.root / "isolated", time.monotonic() + 8, 1)
        self.assertEqual(server_marker.read_text(), "64")
        self.assertEqual(result[0]["metrics"][0], {"output_len": 512, "quality_len": 512, "key_absent": 1, "inherited": 1})

    def test_explicit_benchmark_env_used_for_result_paths_and_redaction(self):
        config = self.config()
        config["server"]["env"] = {"RESULT_NAME": "server.json", "SAFE_MARKER": "server-marker"}
        config["benchmark_env"] = {"RESULT_NAME": "measured.json", "SAFE_MARKER": "benchmark-marker"}
        scenario = config["scenarios"][0]
        scenario["command"][-1] = "{trial_dir}/{env:RESULT_NAME}"
        scenario["command"].append("{env:SAFE_MARKER}")
        scenario["result_file"] = "{trial_dir}/{env:RESULT_NAME}"
        run_trial(config, {}, self.root / "env-paths", time.monotonic() + 5, 1)
        self.assertTrue((self.root / "env-paths/scenario-00/repeat-000/measured.json").is_file())
        self.assertFalse((self.root / "env-paths/scenario-00/repeat-000/server.json").exists())
        saved = (self.root / "env-paths/commands.json").read_text()
        self.assertNotIn("benchmark-marker", saved)
        self.assertIn("<env:SAFE_MARKER>", saved)

    def test_does_not_kill_preexisting_listener(self):
        config = self.config()
        port = int(config["server"]["command"][-1])
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port)); sock.listen()
            with self.assertRaisesRegex(RunnerError, "occupied"):
                run_trial(config, {}, self.root / "trial", time.monotonic() + 5, 1)
            self.assertGreaterEqual(sock.fileno(), 0)

    def test_failed_benchmark_stops_server(self):
        config = self.config("import sys; sys.exit(7)")
        trial = self.root / "trial"
        with self.assertRaisesRegex(RunnerError, "status 7"):
            run_trial(config, {}, trial, time.monotonic() + 5, 1)
        pid = json.loads((trial / "server-process.json").read_text())["pid"]
        self.assertFalse(alive(pid))

    def test_missing_output_and_bad_json_fail(self):
        with self.assertRaisesRegex(RunnerError, "missing"):
            run_trial(self.config("pass"), {}, self.root / "trial", time.monotonic() + 5, 1)
        path = self.root / "invalid.json"
        path.write_text('{"x":NaN}')
        with self.assertRaises(RunnerError):
            _read_result(path)

    def test_hooks_cannot_supply_measured_result(self):
        config = self.config()
        config["scenarios"][0]["warmup_command"] = config["scenarios"][0]["command"]
        with self.assertRaisesRegex(RunnerError, "hook created"):
            run_trial(config, {}, self.root / "trial", time.monotonic() + 5, 1)

    def test_quality_merge_and_overwrite_rejection(self):
        config = self.config()
        scenario = config["scenarios"][0]
        scenario["quality_result_file"] = "{trial_dir}/quality.json"
        scenario["quality_command"] = [sys.executable, "-c", "import json,sys;json.dump({'quality_score':0.99},open(sys.argv[1],'w'))", "{trial_dir}/quality.json"]
        result = run_trial(config, {}, self.root / "good", time.monotonic() + 5, 1)
        self.assertEqual(result[0]["metrics"][0]["quality_score"], 0.99)
        scenario["quality_command"][2] = "import json,sys;json.dump({'output_tokens_per_second':999},open(sys.argv[1],'w'))"
        with self.assertRaisesRegex(RunnerError, "overwrite"):
            run_trial(config, {}, self.root / "bad", time.monotonic() + 5, 1)

    def test_global_deadline_cleans_child_workers(self):
        child_pid = self.root / "child.pid"
        child_code = "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)"
        code = "import subprocess,sys,pathlib,time;p=subprocess.Popen([sys.executable,'-c',sys.argv[2]]);pathlib.Path(sys.argv[1]).write_text(str(p.pid));time.sleep(30)"
        started = time.monotonic()
        with self.assertRaisesRegex(RunnerError, "timed out"):
            run_command([sys.executable, "-c", code, str(child_pid), child_code],
                        cwd=self.root, env=dict(os.environ), log_path=self.root / "log",
                        timeout_s=30, deadline=started + 0.5)
        self.assertLess(time.monotonic() - started, 2)
        pid = int(child_pid.read_text())
        for _ in range(20):
            if not alive(pid): break
            time.sleep(0.02)
        self.assertFalse(alive(pid))

    def test_group_cleanup_after_leader_exit(self):
        child_pid = self.root / "child.pid"
        code = "import subprocess,sys,pathlib;p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);pathlib.Path(sys.argv[1]).write_text(str(p.pid))"
        process = subprocess.Popen([sys.executable, "-c", code, str(child_pid)], start_new_session=True)
        process.wait(timeout=2)
        stop_process_group(process, 0.1)
        pid = int(child_pid.read_text())
        for _ in range(20):
            if not alive(pid): break
            time.sleep(0.02)
        self.assertFalse(alive(pid))


if __name__ == "__main__":
    unittest.main()
