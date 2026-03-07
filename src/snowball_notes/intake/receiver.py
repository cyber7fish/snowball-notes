from __future__ import annotations

import json

from ..config import SnowballConfig
from ..models import RunState, StandardEvent
from ..storage.audit import write_audit_log
from ..utils import new_id, now_utc_iso, sha256_text


def register_events(db, config: SnowballConfig, events: list[StandardEvent]) -> int:
    inserted = 0
    for event in events:
        if enqueue_event(db, config, event):
            inserted += 1
    db.commit()
    return inserted


def enqueue_event(db, config: SnowballConfig, event: StandardEvent) -> bool:
    existing = db.fetchone("SELECT event_id FROM conversation_events WHERE turn_id = ?", (event.turn_id,))
    if existing:
        return False
    db.execute(
        """
        INSERT INTO conversation_events (
          event_id, turn_id, conversation_id, session_file, user_message,
          assistant_final_answer, displayed_at, source_completeness,
          source_confidence, parser_version, context_meta_json, payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event.event_id,
            event.turn_id,
            event.conversation_id,
            event.session_file,
            event.user_message,
            event.assistant_final_answer,
            event.displayed_at,
            event.source_completeness,
            event.source_confidence,
            event.parser_version,
            json.dumps(event.context_meta, ensure_ascii=False),
            json.dumps(event.to_dict(), ensure_ascii=False),
        ),
    )
    if event.source_confidence < config.intake.min_confidence_to_run:
        write_audit_log(
            db,
            "intake_filtered",
            {"reason": "low_confidence", "source_confidence": event.source_confidence},
            level="info",
            turn_id=event.turn_id,
        )
        return False
    if len(event.assistant_final_answer.strip()) < config.intake.min_response_length:
        write_audit_log(
            db,
            "intake_filtered",
            {"reason": "response_too_short", "length": len(event.assistant_final_answer.strip())},
            level="info",
            turn_id=event.turn_id,
        )
        return False
    dedupe_key = sha256_text(f"{event.conversation_id}:{event.turn_id}")
    db.execute(
        """
        INSERT INTO tasks (task_id, event_id, dedupe_key, status, max_retries, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            new_id("task"),
            event.event_id,
            dedupe_key,
            RunState.RECEIVED.value,
            config.worker.max_retries,
            now_utc_iso(),
        ),
    )
    return True

