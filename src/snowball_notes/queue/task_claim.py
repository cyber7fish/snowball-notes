from __future__ import annotations

from ..models import RunState, StandardEvent, TaskRecord
from ..utils import now_utc_iso


def claim_next_task(db, worker_id: str, claim_timeout_seconds: int) -> tuple[TaskRecord, StandardEvent] | None:
    with db.transaction():
        stale_cutoff = f"-{int(claim_timeout_seconds)} seconds"
        db.execute(
            """
            UPDATE tasks
            SET status = ?, claimed_by = NULL, claimed_at = NULL, updated_at = CURRENT_TIMESTAMP
            WHERE status = ? AND claimed_at IS NOT NULL
              AND datetime(claimed_at) < datetime('now', ?)
            """,
            (RunState.RECEIVED.value, RunState.PREPARED.value, stale_cutoff),
        )
        row = db.fetchone(
            """
            SELECT task_id, event_id, status, retry_count, max_retries
            FROM tasks
            WHERE status IN (?, ?)
              AND (next_retry_at IS NULL OR datetime(next_retry_at) <= datetime('now'))
            ORDER BY created_at ASC
            LIMIT 1
            """,
            (RunState.RECEIVED.value, RunState.FAILED_RETRYABLE.value),
        )
        if row is None:
            return None
        updated = db.execute(
            """
            UPDATE tasks
            SET status = ?, claimed_by = ?, claimed_at = ?, updated_at = ?
            WHERE task_id = ? AND status = ?
            """,
            (
                RunState.PREPARED.value,
                worker_id,
                now_utc_iso(),
                now_utc_iso(),
                row["task_id"],
                row["status"],
            ),
        )
        if updated.rowcount != 1:
            return None
        event_row = db.fetchone(
            "SELECT payload_json FROM conversation_events WHERE event_id = ?",
            (row["event_id"],),
        )
        if event_row is None:
            return None
        task = TaskRecord(
            task_id=row["task_id"],
            event_id=row["event_id"],
            status=RunState.PREPARED,
            retry_count=int(row["retry_count"]),
            max_retries=int(row["max_retries"]),
            claimed_by=worker_id,
            claimed_at=now_utc_iso(),
        )
        event = StandardEvent.from_dict(__import__("json").loads(event_row["payload_json"]))
        return task, event
