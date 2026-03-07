from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..utils import ensure_directory, now_utc_iso


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = path
        ensure_directory(path.parent)

    def log(self, level: str, event: str, **fields: Any) -> None:
        payload = {"ts": now_utc_iso(), "level": level.upper(), "event": event}
        payload.update(fields)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

