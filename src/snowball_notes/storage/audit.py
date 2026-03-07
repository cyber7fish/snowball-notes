from __future__ import annotations

import json
from typing import Any

from ..utils import new_id


def write_audit_log(
    db,
    event_type: str,
    detail: dict[str, Any],
    level: str = "info",
    trace_id: str | None = None,
    turn_id: str | None = None,
    task_id: str | None = None,
) -> None:
    if db is None:
        return
    db.execute(
        """
        INSERT INTO audit_logs (audit_id, event_type, level, trace_id, turn_id, task_id, detail_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (new_id("audit"), event_type, level, trace_id, turn_id, task_id, json.dumps(detail, ensure_ascii=False)),
    )

