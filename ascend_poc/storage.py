"""Atomic state and exclusive local experiment ownership."""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


@contextlib.contextmanager
def experiment_lock(output: Path, resource_id: str):
    """Protect both output state and the local hardware pool, including other runs."""
    digest = hashlib.sha256(resource_id.encode()).hexdigest()[:24]
    paths = [output / ".run.lock", Path(tempfile.gettempdir()) / f"ascend-poc-{os.getuid()}-{digest}.lock"]
    handles = []
    try:
        for path in paths:
            handle = path.open("a+")
            handles.append(handle)
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("Another tuning process owns this output or resource_id") from None
        yield
    finally:
        for handle in reversed(handles):
            handle.close()
