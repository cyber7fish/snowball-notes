from __future__ import annotations

import json
from types import SimpleNamespace

from ..agent.commit import Committer
from ..agent.memory import SQLiteKnowledgeIndex, load_session_memory, update_session_memory
from ..agent.state import AgentState
from ..agent.tools import (
    ExtractKnowledgePointsTool,
    compose_append_content,
    compose_archive_payload,
    compose_atomic_note_content,
)
from ..config import SnowballConfig
from ..models import ActionProposal, StandardEvent
from ..storage.audit import write_audit_log
from ..utils import new_id, now_utc_iso, sha256_text


REVIEW_ACTION_ALIASES = {
    "create": "create_note",
    "create_note": "create_note",
    "append": "append_note",
    "append_note": "append_note",
    "archive": "archive_turn",
    "archive_turn": "archive_turn",
}
APPROVAL_FINAL_ACTIONS = {
    "create_note": "approve_create",
    "append_note": "approve_append",
    "archive_turn": "approve_archive",
}


def list_pending_reviews(db) -> str:
    rows = db.fetchall(
        """
        SELECT review_id, turn_id, trace_id, final_target_note_id, suggested_action,
               suggested_target_note_id, suggested_payload_json, reason, created_at
        FROM review_actions
        WHERE final_action = 'pending_review'
        ORDER BY created_at DESC
        """
    )
    if not rows:
        return "No pending reviews."
    lines = []
    for row in rows:
        suggested_action = _suggested_action(row)
        target = _suggested_target_note_id(row) or row.get("final_target_note_id") or "-"
        lines.append(
            f"{row['review_id']} turn={row['turn_id']} trace={row['trace_id']} "
            f"suggested={suggested_action} target={target} reason={row['reason']} created_at={row['created_at']}"
        )
    return "\n".join(lines)


def approve_review(
    db,
    vault,
    config: SnowballConfig,
    review_id: str,
    *,
    reviewer: str = "local",
    action: str | None = None,
    note_id: str | None = None,
    title: str | None = None,
) -> tuple[bool, str]:
    review_row = db.fetchone(
        """
        SELECT review_id, turn_id, trace_id, final_action, final_target_note_id,
               suggested_action, suggested_target_note_id, suggested_payload_json, reason
        FROM review_actions
        WHERE review_id = ?
        """,
        (review_id,),
    )
    if review_row is None:
        return False, f"review {review_id} not found"
    if review_row["final_action"] != "pending_review":
        return False, f"review {review_id} is already resolved as {review_row['final_action']}"
    event = _load_review_event(db, review_row["turn_id"])
    if event is None:
        return False, f"event for turn {review_row['turn_id']} not found"

    action_type = _resolve_action(action, note_id, review_row)
    target_note_id = note_id or _suggested_target_note_id(review_row) or review_row.get("final_target_note_id")
    if action_type == "append_note" and not target_note_id:
        return False, "append approval requires --note-id or a conflict note captured by the review row"

    proposal = _build_review_proposal(
        db=db,
        config=config,
        review_row=review_row,
        event=event,
        action_type=action_type,
        target_note_id=target_note_id,
        title=title,
    )
    state = AgentState(
        event=event,
        task_id=f"review_{review_id}",
        trace_id=review_row["trace_id"],
        session_memory=load_session_memory(db, event.conversation_id),
    )
    state.proposals.append(proposal)
    _save_proposal(db, proposal)

    committer = Committer(db, vault, state, config)
    validation_errors = committer.validate()
    if validation_errors:
        _discard_proposal(db, proposal.proposal_id)
        write_audit_log(
            db,
            "review_commit_blocked",
            {"review_id": review_id, "errors": validation_errors, "action_type": action_type},
            level="warn",
            trace_id=review_row["trace_id"],
            turn_id=event.turn_id,
        )
        db.commit()
        return False, "; ".join(validation_errors)

    commit_result = committer.commit()
    if not commit_result.success:
        _discard_proposal(db, proposal.proposal_id)
        write_audit_log(
            db,
            "review_commit_failed",
            {
                "review_id": review_id,
                "action_type": action_type,
                "disposition": commit_result.disposition,
                "reason": commit_result.reason,
            },
            level="error" if commit_result.disposition == "fatal" else "warn",
            trace_id=review_row["trace_id"],
            turn_id=event.turn_id,
        )
        db.commit()
        return False, commit_result.reason or "review commit failed"

    committed_note_id = proposal.target_note_id or (commit_result.committed_note_ids[-1] if commit_result.committed_note_ids else "")
    _finalize_approved_review(
        db=db,
        vault=vault,
        review_row=review_row,
        event=event,
        reviewer=reviewer,
        action_type=action_type,
        committed_note_id=committed_note_id,
    )
    db.commit()
    return True, committed_note_id or action_type


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


def _suggested_action(review_row: dict) -> str:
    suggested = review_row.get("suggested_action")
    if isinstance(suggested, str) and suggested:
        return suggested.replace("_note", "").replace("_turn", "")
    if review_row.get("final_target_note_id"):
        return "append"
    reason = (review_row.get("reason") or "").strip().lower()
    if "archive" in reason:
        return "archive"
    return "create"


def _load_review_event(db, turn_id: str) -> StandardEvent | None:
    row = db.fetchone(
        """
        SELECT payload_json
        FROM conversation_events
        WHERE turn_id = ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (turn_id,),
    )
    if row is None:
        return None
    return StandardEvent.from_dict(json.loads(row["payload_json"]))


def _resolve_action(action: str | None, note_id: str | None, review_row: dict) -> str:
    if action:
        normalized = REVIEW_ACTION_ALIASES.get(action.strip().lower())
        if normalized is None:
            raise ValueError(f"unsupported review action: {action}")
        return normalized
    suggested = review_row.get("suggested_action")
    if isinstance(suggested, str) and suggested:
        return suggested
    if note_id or _suggested_target_note_id(review_row) or review_row.get("final_target_note_id"):
        return "append_note"
    if "archive" in (review_row.get("reason") or "").lower():
        return "archive_turn"
    return "create_note"


def _build_review_proposal(
    db,
    config: SnowballConfig,
    review_row: dict,
    event: StandardEvent,
    action_type: str,
    target_note_id: str | None,
    title: str | None,
) -> ActionProposal:
    snapshot_proposal = _proposal_from_snapshot(review_row, action_type, target_note_id=target_note_id, title=title)
    if snapshot_proposal is not None:
        snapshot_proposal.trace_id = review_row["trace_id"]
        snapshot_proposal.turn_id = event.turn_id
        if action_type == "create_note":
            snapshot_proposal.payload.setdefault("source_event_id", event.event_id)
        if action_type == "append_note":
            snapshot_proposal.payload["source_turn_id"] = event.turn_id
            snapshot_proposal.payload["source_event_id"] = event.event_id
        if action_type == "archive_turn":
            snapshot_proposal.payload.setdefault("event_id", event.event_id)
            snapshot_proposal.payload.setdefault("user_message", event.user_message)
            snapshot_proposal.payload.setdefault("assistant_final_answer", event.assistant_final_answer)
        return snapshot_proposal
    extracted = ExtractKnowledgePointsTool().execute({}, SimpleNamespace(event=event)).data or {}
    if action_type == "create_note":
        knowledge_index = SQLiteKnowledgeIndex(db)
        related = [
            item.to_dict()
            for item in knowledge_index.search(
                extracted.get("candidate_title", event.user_message),
                config.retrieval.top_k,
            )[:2]
        ]
        resolved_title = (title or extracted.get("candidate_title") or event.user_message[:80]).strip() or "Untitled Note"
        return ActionProposal(
            proposal_id=new_id("proposal"),
            trace_id=review_row["trace_id"],
            turn_id=event.turn_id,
            action_type="create_note",
            target_note_id=None,
            payload={
                "title": resolved_title,
                "content": compose_atomic_note_content(extracted, event, related=related),
                "tags": extracted.get("tags", []),
                "topics": extracted.get("topics", []),
                "source_event_id": event.event_id,
            },
            idempotency_key=f"review-create:{event.turn_id}:{sha256_text(resolved_title)[:8]}",
        )
    if action_type == "append_note":
        if not target_note_id:
            raise ValueError("append_note requires target_note_id")
        return ActionProposal(
            proposal_id=new_id("proposal"),
            trace_id=review_row["trace_id"],
            turn_id=event.turn_id,
            action_type="append_note",
            target_note_id=target_note_id,
            payload={
                "content": compose_append_content(extracted, event),
                "source_turn_id": event.turn_id,
                "source_event_id": event.event_id,
            },
            idempotency_key=f"review-append:{event.turn_id}:{target_note_id}",
        )
    if action_type == "archive_turn":
        payload = compose_archive_payload(event)
        payload["content"] = event.assistant_final_answer
        return ActionProposal(
            proposal_id=new_id("proposal"),
            trace_id=review_row["trace_id"],
            turn_id=event.turn_id,
            action_type="archive_turn",
            target_note_id=None,
            payload=payload,
            idempotency_key=f"review-archive:{event.turn_id}",
        )
    raise ValueError(f"unsupported action_type {action_type}")


def _proposal_from_snapshot(
    review_row: dict,
    action_type: str,
    *,
    target_note_id: str | None,
    title: str | None,
) -> ActionProposal | None:
    snapshot = _snapshot_payload(review_row)
    if snapshot is None:
        return None
    payload = dict(snapshot)
    if action_type == "create_note":
        resolved_title = (title or payload.get("title") or "").strip()
        if resolved_title:
            payload["title"] = resolved_title
        if not payload.get("title") or not payload.get("content"):
            return None
        payload.setdefault("tags", [])
        payload.setdefault("topics", [])
        payload.setdefault("source_event_id", payload.get("source_event_id"))
        return ActionProposal(
            proposal_id=new_id("proposal"),
            trace_id="",
            turn_id="",
            action_type="create_note",
            target_note_id=None,
            payload=payload,
            idempotency_key=f"review-create:{sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))[:12]}",
        )
    if action_type == "append_note":
        resolved_note_id = target_note_id or payload.get("note_id")
        if not resolved_note_id or not payload.get("content"):
            return None
        return ActionProposal(
            proposal_id=new_id("proposal"),
            trace_id="",
            turn_id="",
            action_type="append_note",
            target_note_id=resolved_note_id,
            payload={
                "content": payload["content"],
                "source_turn_id": payload.get("source_turn_id"),
                "source_event_id": payload.get("source_event_id"),
            },
            idempotency_key=f"review-append:{resolved_note_id}:{sha256_text(str(payload['content']))[:8]}",
        )
    if action_type == "archive_turn":
        if not payload.get("title") or not payload.get("content"):
            return None
        return ActionProposal(
            proposal_id=new_id("proposal"),
            trace_id="",
            turn_id="",
            action_type="archive_turn",
            target_note_id=None,
            payload=payload,
            idempotency_key=f"review-archive:{sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))[:12]}",
        )
    return None


def _snapshot_payload(review_row: dict) -> dict | None:
    raw = review_row.get("suggested_payload_json")
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _suggested_target_note_id(review_row: dict) -> str | None:
    value = review_row.get("suggested_target_note_id")
    return value if isinstance(value, str) and value else None


def _save_proposal(db, proposal: ActionProposal) -> None:
    db.execute(
        """
        INSERT INTO action_proposals (
          proposal_id, trace_id, turn_id, action_type, target_note_id, payload_json,
          idempotency_key, status, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            proposal.proposal_id,
            proposal.trace_id,
            proposal.turn_id,
            proposal.action_type,
            proposal.target_note_id,
            json.dumps(proposal.payload, ensure_ascii=False),
            proposal.idempotency_key,
            proposal.status,
            proposal.created_at,
        ),
    )


def _discard_proposal(db, proposal_id: str) -> None:
    db.execute(
        "UPDATE action_proposals SET status = 'discarded' WHERE proposal_id = ? AND status = 'proposed'",
        (proposal_id,),
    )


def _finalize_approved_review(
    db,
    vault,
    review_row: dict,
    event: StandardEvent,
    reviewer: str,
    action_type: str,
    committed_note_id: str,
) -> None:
    note_title = None
    if committed_note_id:
        note_row = db.fetchone(
            "SELECT note_id, title, vault_path FROM notes WHERE note_id = ?",
            (committed_note_id,),
        )
        if note_row is not None:
            note_title = note_row["title"]
            if action_type in {"create_note", "append_note"}:
                content_hash = vault.update_note_status(note_row["vault_path"], "approved")
                db.execute(
                    """
                    UPDATE notes
                    SET status = 'approved', content_hash = ?, updated_at = ?
                    WHERE note_id = ?
                    """,
                    (content_hash, now_utc_iso(), committed_note_id),
                )
    db.execute(
        """
        UPDATE review_actions
        SET final_action = ?, final_target_note_id = ?, reviewer = ?, created_at = ?
        WHERE review_id = ?
        """,
        (
            APPROVAL_FINAL_ACTIONS[action_type],
            committed_note_id or None,
            reviewer,
            now_utc_iso(),
            review_row["review_id"],
        ),
    )
    actions = []
    if committed_note_id:
        actions.append(
            {
                "note_id": committed_note_id,
                "action_type": action_type,
                "note_title": note_title,
            }
        )
    update_session_memory(
        db,
        conversation_id=event.conversation_id,
        turn_id=event.turn_id,
        final_decision=action_type,
        actions=actions,
    )
    write_audit_log(
        db,
        "review_approved",
        {
            "review_id": review_row["review_id"],
            "action_type": action_type,
            "committed_note_id": committed_note_id,
        },
        trace_id=review_row["trace_id"],
        turn_id=event.turn_id,
    )
