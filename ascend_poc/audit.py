"""Content-pinned inputs and a local, hash-linked experiment event journal.

These checks detect accidental input changes and damaged or edited event records.
They are not a sandbox or a signature: a process that can replace the controller
and rewrite the entire journal can also calculate a new hash chain.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat


_HEX = re.compile(r"^[0-9a-fA-F]{64}$")
_ZERO_HASH = "0" * 64
_CHUNK_BYTES = 1024 * 1024


def _identity(info):
    # Access time changes while reading and is deliberately excluded.
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def verify_frozen_files(entries: list, root: Path) -> list[dict]:
    """Hash every pinned regular file, failing closed on changes or bad pins.

    Relative paths are resolved against ``root``. Absolute paths are supported
    for customer datasets outside the project. Files are streamed regardless of
    size; directories must be represented by an operator-generated manifest
    rather than being treated as recursively frozen by this function.
    """
    if not isinstance(entries, list):
        raise ValueError("frozen_files must be a list")
    root = Path(root).resolve()
    manifest = []
    for index, entry in enumerate(entries):
        label = f"Frozen artifact {index}"
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise ValueError(f"{label}: expected path and sha256")
        configured = entry["path"]
        digest = entry["sha256"]
        if (not isinstance(configured, str) or not configured.strip()
                or "\x00" in configured):
            raise ValueError(f"{label}: invalid path")
        if not isinstance(digest, str) or not _HEX.fullmatch(digest):
            raise ValueError(f"{label}: sha256 must contain 64 hexadecimal characters")
        path = Path(configured)
        if not path.is_absolute():
            path = root / path
        descriptor = None
        try:
            resolved = path.resolve(strict=True)
            before = resolved.stat()
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"{label}: expected a regular file")
            # Nonblocking open also avoids hanging if a file is replaced by a
            # FIFO between the stat and open. The descriptor is checked again.
            descriptor = os.open(resolved, os.O_RDONLY | os.O_NONBLOCK)
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or _identity(opened) != _identity(before):
                raise ValueError(f"{label}: changed while opening")
            hasher = hashlib.sha256()
            while True:
                chunk = os.read(descriptor, _CHUNK_BYTES)
                if not chunk:
                    break
                hasher.update(chunk)
            after = os.fstat(descriptor)
            if (_identity(before) != _identity(after)
                    or path.resolve(strict=True) != resolved
                    or _identity(resolved.stat()) != _identity(before)):
                raise ValueError(f"{label}: changed while hashing")
            actual = hasher.hexdigest()
            if actual != digest.lower():
                raise ValueError(f"{label}: sha256 mismatch")
            manifest.append({"path": configured, "resolved_path": str(resolved),
                             "sha256": actual, "size": after.st_size})
        except (OSError, RuntimeError) as error:
            # Do not include arbitrary filesystem details or secret paths in
            # exceptions that a caller may later pass into an LLM diagnosis.
            raise ValueError(f"{label}: unavailable or changed while hashing") from error
        finally:
            if descriptor is not None:
                os.close(descriptor)
    return manifest


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate trace JSON key")
        result[key] = value
    return result


def _constant(_value):
    raise ValueError("Non-finite trace JSON number")


def _parse_trace(content: str) -> list[dict]:
    if content and not content.endswith("\n"):
        raise ValueError("Trace ends with an incomplete record")
    records = []
    previous = _ZERO_HASH
    try:
        # JSON strings may contain literal Unicode line separators (U+2028,
        # U+2029). Only the LF written by append_trace separates records.
        for sequence, line in enumerate(content.split("\n")[:-1], start=1):
            item = json.loads(line, object_pairs_hook=_pairs, parse_constant=_constant)
            if not isinstance(item, dict) or set(item) != {"sequence", "event", "previous_hash", "hash"}:
                raise ValueError("Invalid trace record fields")
            if type(item["sequence"]) is not int or item["sequence"] != sequence:
                raise ValueError("Invalid trace sequence")
            if not isinstance(item["event"], dict) or item["previous_hash"] != previous:
                raise ValueError("Invalid trace event or previous hash")
            payload = {key: value for key, value in item.items() if key != "hash"}
            digest = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
            if item["hash"] != digest:
                raise ValueError("Trace hash mismatch")
            previous = digest
            records.append(item)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Trace journal is invalid or has been modified") from error
    return records


def read_trace(path: Path) -> list[dict]:
    """Read and validate a journal. A missing file represents an empty journal."""
    try:
        with Path(path).open(encoding="utf-8", newline="") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_SH)
            return _parse_trace(stream.read())
    except FileNotFoundError:
        return []
    except (OSError, UnicodeError) as error:
        raise ValueError("Cannot read trace journal") from error


def append_trace(output: Path, event: dict) -> dict:
    """Append one caller-supplied event to ``output/events.jsonl`` and fsync it.

    No environment, request body, or logs are collected automatically. Callers
    must supply metadata appropriate for the local audit trail. Existing records
    are verified before appending, including when continuing after a restart.
    """
    if not isinstance(event, dict):
        raise ValueError("Trace event must be an object")
    try:
        # Roundtrip makes the returned event a snapshot independent of caller
        # mutation, and rejects non-JSON values and non-finite numbers up front.
        event = json.loads(_canonical(event), object_pairs_hook=_pairs, parse_constant=_constant)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Trace event must contain finite JSON data") from error
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        with (output / "events.jsonl").open("a+", encoding="utf-8", newline="") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            stream.seek(0)
            records = _parse_trace(stream.read())
            payload = {"sequence": len(records) + 1, "event": event,
                       "previous_hash": records[-1]["hash"] if records else _ZERO_HASH}
            record = dict(payload, hash=hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest())
            stream.seek(0, os.SEEK_END)
            stream.write(_canonical(record) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            return record
    except (OSError, UnicodeError) as error:
        raise ValueError("Cannot append trace journal") from error
