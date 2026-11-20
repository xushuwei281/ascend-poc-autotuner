"""Content and integrity failures must not silently become trusted evidence."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ascend_poc.audit import append_trace, read_trace, verify_frozen_files


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def pin(self, name="dataset.jsonl", data=b"fixed customer workload\n"):
        path = self.root / name
        path.write_bytes(data)
        return {"path": name, "sha256": hashlib.sha256(data).hexdigest()}

    def test_streamed_large_file_and_absolute_path(self):
        data = b"x" * (3 * 1024 * 1024 + 37)
        pin = self.pin(data=data)
        manifest = verify_frozen_files([pin], self.root)
        self.assertEqual(manifest, [{**pin, "resolved_path": str(self.root / pin["path"]), "size": len(data)}])
        absolute = dict(pin, path=str(self.root / pin["path"]), sha256=pin["sha256"].upper())
        self.assertEqual(verify_frozen_files([absolute], self.root)[0]["sha256"], pin["sha256"])

    def test_changed_content_is_rejected_without_path_in_error(self):
        pin = self.pin(name="private-customer-name.jsonl")
        (self.root / pin["path"]).write_bytes(b"changed workload")
        with self.assertRaisesRegex(ValueError, "Frozen artifact 0: sha256 mismatch") as caught:
            verify_frozen_files([pin], self.root)
        self.assertNotIn("private-customer", str(caught.exception))

    def test_missing_file_directory_and_fifo_are_rejected(self):
        directory = self.root / "directory"
        directory.mkdir()
        fifo = self.root / "fifo"
        os.mkfifo(fifo)
        for name in ["missing", "directory", "fifo"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                verify_frozen_files([{"path": name, "sha256": "0" * 64}], self.root)

    def test_invalid_pins_fail_closed(self):
        for entries in [None, {}, [{"path": "x"}], [{"path": "x", "sha256": "z" * 64}],
                        [{"path": "x", "sha256": "0" * 63}], [{"path": "", "sha256": "0" * 64}],
                        [{"path": "x", "sha256": "0" * 64, "extra": True}],
                        [{"path": "x\x00y", "sha256": "0" * 64}]]:
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                verify_frozen_files(entries, self.root)

    def test_mutation_during_streaming_is_rejected(self):
        pin = self.pin(data=b"a" * (2 * 1024 * 1024))
        original = os.read
        changed = False

        def changing_read(fd, size):
            nonlocal changed
            result = original(fd, size)
            if not changed:
                changed = True
                with (self.root / pin["path"]).open("ab") as stream:
                    stream.write(b"new data")
            return result

        with patch("ascend_poc.audit.os.read", side_effect=changing_read), self.assertRaisesRegex(ValueError, "changed while hashing"):
            verify_frozen_files([pin], self.root)

    def test_symlink_target_resolution_is_recorded(self):
        pin = self.pin()
        (self.root / "link").symlink_to(self.root / pin["path"])
        result = verify_frozen_files([dict(pin, path="link")], self.root)
        self.assertEqual(result[0]["path"], "link")
        self.assertEqual(result[0]["resolved_path"], str(self.root / pin["path"]))

    def test_trace_can_continue_and_is_a_snapshot(self):
        path = self.root / "events.jsonl"
        self.assertEqual(read_trace(path), [])
        event = {"type": "measured", "metrics": {"ttft": 20.0}, "中文": "验收\u2028下一行"}
        first = append_trace(self.root, event)
        event["metrics"]["ttft"] = 999
        second = append_trace(self.root, {"type": "confirmed", "trial": "0001"})
        self.assertEqual(first["sequence"], 1)
        self.assertEqual(second["sequence"], 2)
        self.assertEqual(second["previous_hash"], first["hash"])
        self.assertEqual(first["event"]["metrics"]["ttft"], 20)
        self.assertEqual(read_trace(path), [first, second])

    def test_tampering_reordering_and_truncation_are_rejected(self):
        for event in [{"type": "start"}, {"type": "measured"}, {"type": "confirmed"}]:
            append_trace(self.root, event)
        path = self.root / "events.jsonl"
        content = path.read_text()
        rows = content.splitlines(keepends=True)
        cases = [content.replace("measured", "tampered"), rows[1] + rows[0] + rows[2],
                 content[:-1], content[:-10], rows[0] + rows[2], content + "\n"]
        for modified in cases:
            with self.subTest(modified=modified[-25:]):
                path.write_text(modified)
                with self.assertRaises(ValueError):
                    read_trace(path)
                with self.assertRaises(ValueError):
                    append_trace(self.root, {"type": "must-not-append"})
                self.assertEqual(path.read_text(), modified)

    def test_nonfinite_and_nonobject_event_rejected_before_creating_log(self):
        for event in [[], {"value": float("nan")}, {"value": float("inf")}, {"value": object()}]:
            with self.subTest(event=event), self.assertRaises(ValueError):
                append_trace(self.root, event)
        self.assertFalse((self.root / "events.jsonl").exists())

    def test_duplicate_json_keys_rejected(self):
        append_trace(self.root, {"type": "start"})
        path = self.root / "events.jsonl"
        content = path.read_text().replace('"sequence":1', '"sequence":1,"sequence":1')
        path.write_text(content)
        with self.assertRaises(ValueError):
            read_trace(path)

    def test_valid_hash_cannot_bypass_sequence_or_event_schema(self):
        for sequence, event in [(True, {}), (2, {}), (1, [])]:
            record = {"sequence": sequence, "event": event, "previous_hash": "0" * 64}
            digest = hashlib.sha256(json.dumps(record, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":")).encode()).hexdigest()
            (self.root / "events.jsonl").write_text(json.dumps(dict(record, hash=digest)) + "\n")
            with self.subTest(sequence=sequence, event=event), self.assertRaises(ValueError):
                read_trace(self.root / "events.jsonl")


if __name__ == "__main__":
    unittest.main()
