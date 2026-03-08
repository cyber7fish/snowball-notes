from __future__ import annotations

import json
import re
from typing import Any

from ..models import ActionProposal, ToolResult
from ..utils import new_id, normalize_text, sha256_text, tokenize


TOOL_SCHEMAS = {
    "assess_turn_value": {"required": []},
    "extract_knowledge_points": {"required": []},
    "search_similar_notes": {"required": ["query"], "types": {"query": str, "top_k": int}},
    "read_note": {"required": ["note_id"], "types": {"note_id": str}},
    "propose_create_note": {
        "required": ["title", "content"],
        "types": {"title": str, "content": str, "tags": list, "topics": list},
    },
    "propose_append_to_note": {
        "required": ["note_id", "content"],
        "types": {"note_id": str, "content": str},
    },
    "propose_archive_turn": {
        "required": ["title", "content"],
        "types": {"title": str, "content": str},
    },
    "flag_for_review": {
        "required": ["reason"],
        "types": {"reason": str, "conflict_note_id": str, "suggested_action": str, "suggested_payload": dict},
    },
}


def compose_atomic_note_content(extracted: dict[str, Any], event, related: list[dict[str, Any]] | None = None) -> str:
    points = extracted.get("key_points") or []
    related = related or []
    lines = [
        "## Summary",
        extracted.get("summary", "").strip(),
        "",
        "## Key Points",
    ]
    if points:
        for point in points:
            lines.append(f"- {point}")
    else:
        lines.append("- No structured key points were extracted.")
    lines.extend(["", "## Source", f"- event_id: {event.event_id}", f"- turn_id: {event.turn_id}"])
    if related:
        lines.extend(["", "## Related"])
        for item in related:
            lines.append(f"- [[{item['title']}]] ({item['note_id']})")
    return "\n".join(lines).strip()


def compose_append_content(extracted: dict[str, Any], event) -> str:
    summary = extracted.get("summary", "").strip()
    points = extracted.get("key_points") or []
    line = summary or "; ".join(points[:2]) or "Additional supporting detail."
    return f"{line} (source turn: {event.turn_id})"


def compose_archive_payload(event) -> dict[str, Any]:
    return {
        "title": f"Conversation {event.turn_id[:8]}",
        "event_id": event.event_id,
        "user_message": event.user_message,
        "assistant_final_answer": event.assistant_final_answer,
    }


class Tool:
    name = ""

    def execute(self, payload: dict[str, Any], state) -> ToolResult:  # pragma: no cover - interface
        raise NotImplementedError


class AssessTurnValueTool(Tool):
    name = "assess_turn_value"

    def execute(self, payload: dict[str, Any], state) -> ToolResult:
        event = state.event
        answer = event.assistant_final_answer.strip()
        user = event.user_message.strip().lower()
        technical_signals = [
            "error",
            "debug",
            "design",
            "architecture",
            "implement",
            "python",
            "sql",
            "agent",
            "方案",
            "实现",
            "代码",
            "架构",
        ]
        is_short = len(answer) < 140
        is_small_talk = any(token in user for token in ["thanks", "thank you", "hello", "你好", "谢谢"])
        has_signal = any(token in answer.lower() or token in user for token in technical_signals)
        decision = "note"
        reasons = ["long_term_value"]
        if is_small_talk or (is_short and not has_signal):
            decision = "skip"
            reasons = ["low_information_density"]
        elif not has_signal and len(answer) < 260:
            decision = "archive"
            reasons = ["not_reusable_enough"]
        elif event.source_confidence < 0.7:
            decision = "archive"
            reasons = ["insufficient_confidence_for_note"]
        return ToolResult.ok({"decision": decision, "reason": reasons, "confidence": event.source_confidence})


class ExtractKnowledgePointsTool(Tool):
    name = "extract_knowledge_points"

    def execute(self, payload: dict[str, Any], state) -> ToolResult:
        event = state.event
        answer = event.assistant_final_answer.strip()
        title = _guess_title(event.user_message, answer)
        summary = _first_sentence(answer)
        key_points = _extract_points(answer)
        topics = _guess_topics(event.user_message, answer)
        tags = sorted(set(topics + ["codex", "snowball-notes"]))
        return ToolResult.ok(
            {
                "candidate_title": title,
                "summary": summary,
                "key_points": key_points[:5],
                "topics": topics[:4],
                "tags": tags[:6],
            }
        )


class SearchSimilarNotesTool(Tool):
    name = "search_similar_notes"

    def __init__(self, knowledge_index):
        self.knowledge_index = knowledge_index

    def execute(self, payload: dict[str, Any], state) -> ToolResult:
        results = self.knowledge_index.search(payload["query"], payload.get("top_k", 5))
        for item in results:
            state.knowledge_snapshot_refs.append(
                {
                    "note_id": item.note_id,
                    "content_hash": item.content_hash,
                    "title": item.title,
                    "similarity": item.similarity,
                }
            )
        return ToolResult.ok([item.to_dict() for item in results])


class ReadNoteTool(Tool):
    name = "read_note"

    def __init__(self, knowledge_index):
        self.knowledge_index = knowledge_index

    def execute(self, payload: dict[str, Any], state) -> ToolResult:
        return ToolResult.ok(self.knowledge_index.load_note(payload["note_id"]))


class ProposeCreateNoteTool(Tool):
    name = "propose_create_note"

    def execute(self, payload: dict[str, Any], state) -> ToolResult:
        proposal = ActionProposal(
            proposal_id=new_id("proposal"),
            trace_id=state.trace_id,
            turn_id=state.event.turn_id,
            action_type="create_note",
            target_note_id=None,
            payload={
                "title": payload["title"],
                "content": payload["content"],
                "tags": payload.get("tags", []),
                "topics": payload.get("topics", []),
                "source_event_id": state.event.event_id,
            },
            idempotency_key=f"create:{state.event.turn_id}:{sha256_text(payload['title'])[:8]}",
        )
        state.proposals.append(proposal)
        state.write_count += 1
        return ToolResult.ok({"proposal_id": proposal.proposal_id, "action_type": proposal.action_type})


class ProposeAppendToNoteTool(Tool):
    name = "propose_append_to_note"

    def execute(self, payload: dict[str, Any], state) -> ToolResult:
        proposal = ActionProposal(
            proposal_id=new_id("proposal"),
            trace_id=state.trace_id,
            turn_id=state.event.turn_id,
            action_type="append_note",
            target_note_id=payload["note_id"],
            payload={
                "content": payload["content"],
                "source_turn_id": state.event.turn_id,
                "source_event_id": state.event.event_id,
            },
            idempotency_key=f"append:{state.event.turn_id}:{payload['note_id']}",
        )
        state.proposals.append(proposal)
        state.write_count += 1
        state.append_count += 1
        return ToolResult.ok({"proposal_id": proposal.proposal_id, "action_type": proposal.action_type})


class ProposeArchiveTurnTool(Tool):
    name = "propose_archive_turn"

    def execute(self, payload: dict[str, Any], state) -> ToolResult:
        proposal = ActionProposal(
            proposal_id=new_id("proposal"),
            trace_id=state.trace_id,
            turn_id=state.event.turn_id,
            action_type="archive_turn",
            target_note_id=None,
            payload={
                "title": payload["title"],
                "content": payload["content"],
                "event_id": state.event.event_id,
                "user_message": state.event.user_message,
                "assistant_final_answer": state.event.assistant_final_answer,
            },
            idempotency_key=f"archive:{state.event.turn_id}",
        )
        state.proposals.append(proposal)
        state.write_count += 1
        return ToolResult.ok({"proposal_id": proposal.proposal_id, "action_type": proposal.action_type})


class FlagForReviewTool(Tool):
    name = "flag_for_review"

    def __init__(self, db):
        self.db = db

    def execute(self, payload: dict[str, Any], state) -> ToolResult:
        review_id = new_id("review")
        suggested_payload = payload.get("suggested_payload")
        if suggested_payload is not None and not isinstance(suggested_payload, dict):
            suggested_payload = None
        self.db.execute(
            """
            INSERT INTO review_actions (
              review_id, turn_id, trace_id, final_action, final_target_note_id,
              suggested_action, suggested_target_note_id, suggested_payload_json, reason
            )
            VALUES (?, ?, ?, 'pending_review', ?, ?, ?, ?, ?)
            """,
            (
                review_id,
                state.event.turn_id,
                state.trace_id,
                payload.get("conflict_note_id"),
                payload.get("suggested_action"),
                _snapshot_target_note_id(payload.get("suggested_action"), suggested_payload, payload.get("conflict_note_id")),
                json.dumps(suggested_payload, ensure_ascii=False) if suggested_payload is not None else None,
                payload["reason"],
            ),
        )
        state.is_terminated = True
        state.terminal_reason = payload["reason"]
        state.proposals.clear()
        return ToolResult.ok(
            {
                "flagged": True,
                "review_id": review_id,
                "reason": payload["reason"],
                "suggested_action": payload.get("suggested_action"),
            }
        )


def build_tool_registry(db, knowledge_index) -> dict[str, Tool]:
    return {
        "assess_turn_value": AssessTurnValueTool(),
        "extract_knowledge_points": ExtractKnowledgePointsTool(),
        "search_similar_notes": SearchSimilarNotesTool(knowledge_index),
        "read_note": ReadNoteTool(knowledge_index),
        "propose_create_note": ProposeCreateNoteTool(),
        "propose_append_to_note": ProposeAppendToNoteTool(),
        "propose_archive_turn": ProposeArchiveTurnTool(),
        "flag_for_review": FlagForReviewTool(db),
    }


def validated_tool_execute(tool_name: str, payload: dict[str, Any], registry: dict[str, Tool], state) -> ToolResult:
    tool = registry.get(tool_name)
    if tool is None:
        return ToolResult.error("unknown_tool", tool_name)
    errors = _validate_payload(tool_name, payload)
    if errors:
        return ToolResult.validation_error("; ".join(errors))
    return tool.execute(payload, state)


def _validate_payload(tool_name: str, payload: dict[str, Any]) -> list[str]:
    schema = TOOL_SCHEMAS.get(tool_name, {})
    errors = []
    for field_name in schema.get("required", []):
        if field_name not in payload:
            errors.append(f"missing required field: {field_name}")
    for field_name, expected_type in schema.get("types", {}).items():
        if field_name in payload and not isinstance(payload[field_name], expected_type):
            errors.append(f"{field_name} must be {expected_type.__name__}")
    return errors


def _snapshot_target_note_id(
    suggested_action: str | None,
    suggested_payload: dict[str, Any] | None,
    conflict_note_id: str | None,
) -> str | None:
    if suggested_action == "append_note" and isinstance(suggested_payload, dict):
        value = suggested_payload.get("note_id")
        if isinstance(value, str) and value:
            return value
    return conflict_note_id


def _guess_title(user_message: str, answer: str) -> str:
    prompt = user_message.strip().replace("\n", " ")
    prompt = re.sub(r"\s+", " ", prompt)
    if len(prompt) > 72:
        prompt = prompt[:72].rstrip() + "..."
    prompt = prompt.strip(" ??.。")
    if prompt:
        return prompt[:80]
    return _first_sentence(answer)[:80] or "Untitled Note"


def _first_sentence(answer: str) -> str:
    for separator in [". ", "\n", "。", "！", "!"]:
        if separator in answer:
            return answer.split(separator)[0].strip()
    return answer.strip()[:200]


def _extract_points(answer: str) -> list[str]:
    points = []
    for line in answer.splitlines():
        stripped = line.strip()
        if stripped.startswith(("- ", "* ")):
            points.append(stripped[2:].strip())
        elif re.match(r"^\d+\.\s+", stripped):
            points.append(re.sub(r"^\d+\.\s+", "", stripped))
    if points:
        return points
    sentences = [part.strip() for part in re.split(r"[。\n]", answer) if len(part.strip()) > 12]
    return sentences[:4]


def _guess_topics(user_message: str, answer: str) -> list[str]:
    tokens = tokenize(f"{user_message} {answer}")
    candidates = []
    for token in tokens:
        if token in {"the", "and", "with", "this", "that", "一个", "可以", "什么", "然后"}:
            continue
        if len(token) <= 2:
            continue
        candidates.append(token)
    seen = []
    for token in candidates:
        if token not in seen:
            seen.append(token)
    return seen[:6]
