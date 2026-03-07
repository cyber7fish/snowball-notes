from __future__ import annotations


def render_status(db) -> str:
    task_rows = db.fetchall(
        "SELECT status, COUNT(*) AS count FROM tasks GROUP BY status ORDER BY status"
    )
    trace_stats = db.fetchone(
        """
        SELECT COUNT(*) AS total_runs,
               AVG(total_steps) AS avg_steps,
               AVG(total_duration_ms) AS avg_duration_ms
        FROM agent_traces
        """
    ) or {"total_runs": 0, "avg_steps": 0, "avg_duration_ms": 0}
    review_stats = db.fetchone(
        "SELECT COUNT(*) AS pending FROM review_actions WHERE final_action = 'pending_review'"
    ) or {"pending": 0}
    lines = ["Snowball Status", "----------------"]
    if not task_rows:
        lines.append("No tasks recorded yet.")
    else:
        for row in task_rows:
            lines.append(f"{row['status']}: {row['count']}")
    lines.append("")
    lines.append(f"agent_runs: {trace_stats['total_runs'] or 0}")
    lines.append(f"avg_steps: {round(trace_stats['avg_steps'] or 0, 2)}")
    lines.append(f"avg_duration_ms: {round(trace_stats['avg_duration_ms'] or 0, 2)}")
    lines.append(f"pending_reviews: {review_stats['pending'] or 0}")
    return "\n".join(lines)

