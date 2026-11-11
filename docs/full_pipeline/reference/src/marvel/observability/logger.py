import json
from datetime import datetime, timezone


class JsonLogger:
    def __init__(self, path):
        self.file = open(path, "x", encoding="utf-8")

    def write(self, record):
        value = dict(record, wall_utc=datetime.now(timezone.utc).isoformat())
        self.file.write(json.dumps(value, allow_nan=False, separators=(",", ":"))+"\n")
        self.file.flush()

    def close(self):
        self.file.close()
