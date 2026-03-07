from __future__ import annotations

import json
from collections import Counter
from datetime import timedelta

from ..utils import now_utc, parse_datetime


APPROVED_REVIEW_ACTIONS = {"approved", "approve_create", "approve_append", "approve_archive"}


def collect_parser_health(db, sample_size: int = 50) -> dict[str, float | int | None]:
    rows = db.fetchall(
        """
        SELECT source_confidence
        FROM conversation_events
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (sample_size,),
    )
    scores = [float(row["source_confidence"]) for row in rows]
    if not scores:
        return {
            "sample_size": 0,
            "avg_confidence": None,
            "low_confidence_count": 0,
            "low_confidence_rate": None,
        }
    low_confidence_count = sum(score < 0.6 for score in scores)
    return {
        "sample_size": len(scores),
        "avg_confidence": sum(scores) / len(scores),
        "low_confidence_count": low_confidence_count,
        "low_confidence_rate": low_confidence_count / len(scores),
    }


def collect_agent_health(db, window_days: int = 7) -> dict[str, object]:
    cutoff = now_utc() - timedelta(days=window_days)
    trace_rows = _filter_recent_rows(
        db.fetchall(
            """
            SELECT created_at, final_decision, total_steps, total_duration_ms,
                   total_input_tokens, total_output_tokens, exceeded_max_steps, trace_json
            FROM agent_traces
            ORDER BY created_at DESC
            """
        ),
        cutoff,
        "created_at",
    )
    task_rows = _filter_recent_rows(
        db.fetchall("SELECT status, updated_at FROM tasks"),
        cutoff,
        "updated_at",
    )
    review_rows = _filter_recent_rows(
        db.fetchall("SELECT final_action, created_at FROM review_actions"),
        cutoff,
        "created_at",
    )
    commit_blocked_rows = _filter_recent_rows(
        db.fetchall(
            """
            SELECT created_at
            FROM audit_logs
            WHERE event_type = 'commit_blocked'
            """
        ),
        cutoff,
        "created_at",
    )
    agent_runs = len(trace_rows)
    task_state_counts = Counter(row["status"] for row in task_rows)
    decision_counts = Counter(row["final_decision"] for row in trace_rows if row["final_decision"])
    total_steps = sum(int(row["total_steps"] or 0) for row in trace_rows)
    total_duration_ms = sum(int(row["total_duration_ms"] or 0) for row in trace_rows)
    total_tokens = sum(
        int(row["total_input_tokens"] or 0) + int(row["total_output_tokens"] or 0)
        for row in trace_rows
    )
    exceeded_max_steps = sum(int(row["exceeded_max_steps"] or 0) for row in trace_rows)

    tool_calls = 0
    tool_errors = 0
    guardrail_blocks = 0
    for row in trace_rows:
        trace = json.loads(row["trace_json"])
        for step in trace.get("steps", []):
            if not step.get("tool_name"):
                continue
            tool_calls += 1
            if step.get("guardrail_blocked"):
                guardrail_blocks += 1
            elif step.get("tool_success") is False:
                tool_errors += 1

    pending_reviews = sum(1 for row in review_rows if row["final_action"] == "pending_review")
    resolved_reviews = [row for row in review_rows if row["final_action"] != "pending_review"]
    approved_reviews = sum(1 for row in resolved_reviews if row["final_action"] in APPROVED_REVIEW_ACTIONS)

    last_reconcile = _load_last_reconcile(db)

    return {
        "window_days": window_days,
        "agent_runs": agent_runs,
        "task_state_counts": dict(task_state_counts),
        "decision_counts": dict(decision_counts),
        "avg_steps": (total_steps / agent_runs) if agent_runs else None,
        "avg_duration_ms": (total_duration_ms / agent_runs) if agent_runs else None,
        "avg_tokens_per_run": (total_tokens / agent_runs) if agent_runs else None,
        "max_steps_exceeded_count": exceeded_max_steps,
        "max_steps_exceeded_rate": (exceeded_max_steps / agent_runs) if agent_runs else None,
        "tool_call_count": tool_calls,
        "tool_error_count": tool_errors,
        "tool_error_rate": (tool_errors / tool_calls) if tool_calls else None,
        "guardrail_block_count": guardrail_blocks,
        "guardrail_block_rate": (guardrail_blocks / tool_calls) if tool_calls else None,
        "commit_rejection_count": len(commit_blocked_rows),
        "commit_rejection_rate": (len(commit_blocked_rows) / agent_runs) if agent_runs else None,
        "review_count": len(review_rows),
        "review_rate": (len(review_rows) / agent_runs) if agent_runs else None,
        "pending_reviews": pending_reviews,
        "resolved_review_count": len(resolved_reviews),
        "review_acceptance_rate": (
            approved_reviews / len(resolved_reviews) if resolved_reviews else None
        ),
        "last_reconcile": last_reconcile,
    }


def _filter_recent_rows(rows: list[dict], cutoff, field: str) -> list[dict]:
    recent = []
    for row in rows:
        timestamp = parse_datetime(row.get(field))
        if timestamp is None or timestamp >= cutoff:
            recent.append(row)
    return recent


def _load_last_reconcile(db) -> dict[str, object] | None:
    row = db.fetchone(
        """
        SELECT event_type, detail_json, created_at
        FROM audit_logs
        WHERE event_type IN ('reconcile_completed', 'reconcile_mismatch')
        ORDER BY created_at DESC, rowid DESC
        LIMIT 1
        """
    )
    if row is None:
        return None
    detail = json.loads(row["detail_json"])
    return {
        "created_at": row["created_at"],
        "status": "mismatch" if row["event_type"] == "reconcile_mismatch" else "ok",
        "orphan_count": int(detail.get("orphan_count", len(detail.get("orphan_files", [])))),
        "missing_count": int(detail.get("missing_count", len(detail.get("missing_files", [])))),
    }
