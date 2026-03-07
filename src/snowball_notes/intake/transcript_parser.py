from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..models import StandardEvent
from ..utils import sha256_text
from .confidence import compute_source_confidence


@dataclass
class TurnBuffer:
    turn_id: str
    user_message: str = ""
    assistant_final_answer: str = ""
    displayed_at: str = ""
    raw_events: list[dict[str, Any]] = field(default_factory=list)


def parse_session_file(path: Path, parser_version: str = "v1") -> list[StandardEvent]:
    session_meta: dict[str, Any] = {}
    current: TurnBuffer | None = None
    events: list[StandardEvent] = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        record_type = record.get("type")
        payload = record.get("payload", {})
        if record_type == "session_meta":
            session_meta = payload
            continue
        if record_type != "event_msg":
            continue
        subtype = payload.get("type")
        if subtype == "task_started":
            current = TurnBuffer(turn_id=payload["turn_id"], raw_events=[record])
            continue
        if current is None:
            continue
        current.raw_events.append(record)
        if subtype == "user_message":
            current.user_message = payload.get("message", "")
            continue
        if subtype == "agent_message" and record.get("phase") == "final_answer":
            current.assistant_final_answer = payload.get("message", "")
            current.displayed_at = record.get("timestamp", "")
            continue
        if subtype == "turn_aborted":
            current = None
            continue
        if subtype != "task_complete":
            continue
        if payload.get("turn_id") != current.turn_id:
            current = None
            continue
        displayed_at = current.displayed_at or record.get("timestamp", "")
        source_completeness = (
            "full"
            if current.user_message and current.assistant_final_answer
            else "partial"
        )
        source_confidence = compute_source_confidence(
            current.raw_events,
            current.assistant_final_answer,
            current.user_message,
            source_completeness,
            parser_version,
        )
        turn_seed = f"{session_meta.get('id', '')}:{current.turn_id}:{displayed_at}"
        event_id = f"evt_{sha256_text(turn_seed)[:16]}"
        events.append(
            StandardEvent(
                event_id=event_id,
                session_file=str(path),
                conversation_id=session_meta.get("id", ""),
                turn_id=current.turn_id,
                user_message=current.user_message,
                assistant_final_answer=current.assistant_final_answer,
                displayed_at=displayed_at,
                source_completeness=source_completeness,
                source_confidence=source_confidence,
                parser_version=parser_version,
                context_meta={
                    "client": session_meta.get("originator", "codex"),
                    "cwd": session_meta.get("cwd", ""),
                    "cli_version": session_meta.get("cli_version", ""),
                    "model_provider": session_meta.get("model_provider", ""),
                },
            )
        )
        current = None
    return events

