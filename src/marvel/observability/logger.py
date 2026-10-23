"""Strict JSONL records; opening a log never overwrites an existing run."""

import json
from pathlib import Path


class JsonLogger:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("x", encoding="utf-8")

    def write(self, record):
        line = json.dumps(record, allow_nan=False, sort_keys=True)
        self.stream.write(line + "\n")
        self.stream.flush()

    def close(self):
        self.stream.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
