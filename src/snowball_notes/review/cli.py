from __future__ import annotations

from ..utils import now_utc_iso


def list_pending_reviews(db) -> str:
    rows = db.fetchall(
        """
        SELECT review_id, turn_id, trace_id, reason, created_at
        FROM review_actions
        WHERE final_action = 'pending_review'
        ORDER BY created_at DESC
        """
    )
    if not rows:
        return "No pending reviews."
    lines = []
    for row in rows:
        lines.append(
            f"{row['review_id']} turn={row['turn_id']} trace={row['trace_id']} reason={row['reason']} created_at={row['created_at']}"
        )
    return "\n".join(lines)


def update_review(db, review_id: str, final_action: str, reviewer: str = "local") -> bool:
    updated = db.execute(
        """
        UPDATE review_actions
        SET final_action = ?, reviewer = ?, created_at = ?
        WHERE review_id = ?
        """,
        (final_action, reviewer, now_utc_iso(), review_id),
    )
    return updated.rowcount == 1

