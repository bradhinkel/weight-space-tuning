"""Session logging: every tool call (args + full result) as JSONL, weight diffs, session summary.

Everything is written by the (root) harness server under the session dir, which the driver's kernel
cannot read (/workspace/artifacts is mode 700; the kernel runs as an unprivileged user under Landlock).
"""
import json
import threading
import time
from pathlib import Path


class SessionLog:
    def __init__(self, session_dir: Path):
        self.dir = Path(session_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._f = open(self.dir / "calls.jsonl", "a", buffering=1)
        self._lock = threading.Lock()
        self.n = 0

    def event(self, kind: str, **fields):
        rec = {"t": round(time.time(), 3), "kind": kind, **fields}
        with self._lock:
            self._f.write(json.dumps(rec, default=str) + "\n")
            self.n += 1

    def call(self, tool: str, args: dict, result, seconds: float, budget: dict, error: str | None = None):
        self.event("tool_call", tool=tool, args=args, result=result, error=error,
                   seconds=round(seconds, 3), budget=budget)

    def write_json(self, name: str, obj):
        (self.dir / name).write_text(json.dumps(obj, indent=1, default=str))
