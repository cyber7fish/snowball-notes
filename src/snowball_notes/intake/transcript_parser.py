from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..models import StandardEvent
from ..utils import sha256_text
from .confidence import compute_source_confidence_breakdown


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
        if record_type == "event_msg" and payload.get("type") == "task_started":
            current = TurnBuffer(turn_id=payload["turn_id"], raw_events=[record])
            continue
        if current is None:
            continue
        current.raw_events.append(record)
        if record_type == "response_item":
            if payload.get("type") != "message" or payload.get("role") != "assistant":
                continue
            if _record_phase(record, payload) != "final_answer":
                continue
            final_answer = _extract_response_text(payload.get("content", []))
            if final_answer:
                current.assistant_final_answer = final_answer
                current.displayed_at = record.get("timestamp", "")
            continue
        if record_type != "event_msg":
            continue
        subtype = payload.get("type")
        if subtype == "user_message":
            current.user_message = payload.get("message", "")
            continue
        if subtype == "agent_message" and _record_phase(record, payload) == "final_answer":
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
        if not current.assistant_final_answer and payload.get("last_agent_message"):
            current.assistant_final_answer = payload["last_agent_message"]
            current.displayed_at = record.get("timestamp", "")
        displayed_at = current.displayed_at or record.get("timestamp", "")
        source_completeness = (
            "full"
            if current.user_message and current.assistant_final_answer
            else "partial"
        )
        confidence_breakdown = compute_source_confidence_breakdown(
            current.raw_events,
            current.assistant_final_answer,
            current.user_message,
            source_completeness,
            parser_version,
        )
        source_confidence = float(confidence_breakdown["score"])
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
                    "source_confidence_breakdown": confidence_breakdown,
                },
            )
        )
        current = None
    return events


def _record_phase(record: dict[str, Any], payload: dict[str, Any]) -> str:
    return str(payload.get("phase") or record.get("phase") or "")


def _extract_response_text(content: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for item in content:
        if item.get("type") != "output_text":
            continue
        text = item.get("text")
        if isinstance(text, str) and text:
            chunks.append(text)
    return "".join(chunks)
