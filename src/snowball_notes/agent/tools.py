from __future__ import annotations

import json
import re
from typing import Any

from ..models import ActionProposal, ToolResult
from ..note_cleanup import format_obsidian_link
from ..utils import new_id, normalize_text, sha256_text, tokenize

# ── Tunable constants ────────────────────────────────────────────
MIN_ANSWER_LENGTH_FOR_NOTE = 140
MIN_ANSWER_LENGTH_FOR_ARCHIVE = 260
CONFIDENCE_THRESHOLD_FOR_NOTE = 0.7
MAX_KEY_POINTS = 5
MAX_TOPICS = 4
MAX_TAGS = 6
MAX_TOPIC_CANDIDATES = 6
TITLE_PROMPT_TRUNCATION = 72
MAX_TITLE_FROM_PROMPT = 80
FIRST_SENTENCE_FALLBACK_LEN = 200
MIN_SENTENCE_LENGTH_FOR_POINT = 12
MAX_FALLBACK_KEY_POINTS = 4
SHORT_SUMMARY_PREFIX_THRESHOLD = 32
DEFAULT_TITLE_TRUNCATION = 64
DEFAULT_FALLBACK_TITLE = "Snowball Notes 知识笔记"
# ─────────────────────────────────────────────────────────────────

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
    "propose_link_notes": {
        "required": ["source_note_id", "target_note_id"],
        "types": {"source_note_id": str, "target_note_id": str},
    },
    "flag_for_review": {
        "required": ["reason"],
        "types": {"reason": str, "conflict_note_id": str, "suggested_action": str, "suggested_payload": dict},
    },
}

ENV_KEY_RE = re.compile(r"\b[A-Z][A-Z0-9_]{4,}\b")
ERROR_RE = re.compile(r"\b(?:[A-Za-z]+Error(?:: [^\n。！？!?]+)?)\b")
QUESTION_MARKERS = ("?", "？", "吗", "呢")
CONVERSATIONAL_PREFIXES = (
    "好的",
    "那",
    "现在",
    "你说的",
    "我",
    "怎么",
    "如何",
    "为什么",
    "什么叫",
    "在哪",
    "能不能",
    "是不是",
)
SUMMARY_LEADS = (
    "现在的用法是",
    "当前的用法是",
    "当前做法是",
    "主归属是",
    "原因很明确",
    "结论是",
    "根因是",
    "报错发生在",
    "这说明",
    "不是这个意思。",
    "这是一个关于",
    "这是关于",
    "可以这样记",
    "本质上是",
    "当前默认",
)
PROJECT_META_MARKERS = (
    "这一步属于哪个phase",
    "这一步属于哪一个phase",
    "phase归属",
    "进行到哪步",
    "做到哪步",
    "做到哪了",
    "做到哪",
    "已完成哪些",
    "完成了哪些",
    "还有哪部分没做完",
    "还有哪些没做完",
    "还没做完",
    "当前做到哪",
    "当前项目状态",
    "项目当前状态",
    "下一步",
    "都commit了吗",
    "commit了吗",
)


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
            lines.append(f"- {format_obsidian_link(item['title'], item.get('vault_path'))} ({item['note_id']})")
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
        combined_text = f"{event.user_message}\n{event.assistant_final_answer}"
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
        is_short = len(answer) < MIN_ANSWER_LENGTH_FOR_NOTE
        is_small_talk = any(token in user for token in ["thanks", "thank you", "hello", "你好", "谢谢"])
        has_signal = any(token in answer.lower() or token in user for token in technical_signals)
        decision = "note"
        reasons = ["long_term_value"]
        if _contains_secret_like_text(combined_text):
            decision = "skip"
            reasons = ["contains_secret_like_text"]
        elif is_project_meta_turn(event.user_message, answer):
            decision = "archive"
            reasons = ["project_meta_progress_tracking"]
        elif is_small_talk or (is_short and not has_signal):
            decision = "skip"
            reasons = ["low_information_density"]
        elif not has_signal and len(answer) < MIN_ANSWER_LENGTH_FOR_ARCHIVE:
            decision = "archive"
            reasons = ["not_reusable_enough"]
        elif event.source_confidence < CONFIDENCE_THRESHOLD_FOR_NOTE:
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
                "key_points": key_points[:MAX_KEY_POINTS],
                "topics": topics[:MAX_TOPICS],
                "tags": tags[:MAX_TAGS],
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


class ProposeLinkNotesTool(Tool):
    name = "propose_link_notes"

    def execute(self, payload: dict[str, Any], state) -> ToolResult:
        proposal = ActionProposal(
            proposal_id=new_id("proposal"),
            trace_id=state.trace_id,
            turn_id=state.event.turn_id,
            action_type="link_notes",
            target_note_id=payload["source_note_id"],
            payload={
                "source_note_id": payload["source_note_id"],
                "target_note_id": payload["target_note_id"],
                "source_event_id": state.event.event_id,
            },
            idempotency_key=f"link:{payload['source_note_id']}:{payload['target_note_id']}",
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
        "propose_link_notes": ProposeLinkNotesTool(),
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
    if len(prompt) > TITLE_PROMPT_TRUNCATION:
        prompt = prompt[:TITLE_PROMPT_TRUNCATION].rstrip() + "..."
    prompt = prompt.strip(" ??.。")
    if prompt:
        if re.search(r"[A-Za-z]", prompt) and not re.search(r"[\u4e00-\u9fff]", prompt):
            return prompt[:MAX_TITLE_FROM_PROMPT]
        return _canonicalize_candidate_title(prompt, f"## Summary\n{_first_sentence(answer)}")
    return _canonicalize_candidate_title(_first_sentence(answer)[:MAX_TITLE_FROM_PROMPT] or "Untitled Note")


def _contains_secret_like_text(value: str) -> bool:
    patterns = [
        r"github_pat_[A-Za-z0-9_]{20,}",
        r"\bghp_[A-Za-z0-9]{20,}\b",
        r"\bsk-[A-Za-z0-9]{16,}\b",
        r"\bAKIA[0-9A-Z]{16}\b",
        r"\bAIza[0-9A-Za-z\-_]{20,}\b",
    ]
    return any(re.search(pattern, value) for pattern in patterns)


def is_project_meta_turn(user_message: str, answer: str = "") -> bool:
    combined = normalize_text(f"{user_message}\n{answer}").replace(" ", "")
    if "phase" in combined and any(term in combined for term in ("哪个", "哪一个", "归属", "属于", "步骤")):
        return True
    return any(marker in combined for marker in PROJECT_META_MARKERS)


def _first_sentence(answer: str) -> str:
    for separator in [". ", "\n", "。", "！", "!"]:
        if separator in answer:
            return answer.split(separator)[0].strip()
    return answer.strip()[:FIRST_SENTENCE_FALLBACK_LEN]


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
    sentences = [part.strip() for part in re.split(r"[。\n]", answer) if len(part.strip()) > MIN_SENTENCE_LENGTH_FOR_POINT]
    return sentences[:MAX_FALLBACK_KEY_POINTS]


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
    return seen[:MAX_TOPIC_CANDIDATES]


def _canonicalize_candidate_title(raw_title: str, body: str = "") -> str:
    title = _clean_title_text(raw_title)
    if not title:
        return DEFAULT_FALLBACK_TITLE
    combined = f"{title}\n{body}"
    combined_norm = normalize_text(combined)
    mentions_snowball = "snowball-notes" in combined_norm or "snowball notes" in combined_norm
    if not _looks_conversational_title(title):
        return _truncate_title(title)
    if mentions_snowball and any(term in combined for term in ("怎么用", "如何用", "如何使用")):
        return "snowball-notes 使用方式"
    if all(term in combined for term in ("Archive", "Inbox", "Knowledge")):
        return "Archive、Inbox、Knowledge 目录设计与实现差异"
    if ("模型key" in combined or "model key" in combined_norm or "真实模型" in combined or "需要 key" in combined) and mentions_snowball:
        return "snowball-notes 命令分类与模型 key 依赖"
    if "status" in combined_norm and any(term in combined for term in ("报错", "API key", "api key", "key")):
        return "status 命令误触发 API key 检查"
    error_name = _extract_error_signature(combined)
    if error_name and any(term in combined for term in ("报错", "error", "Error", "missing", "失败")):
        return f"{error_name} 诊断与处理"
    env_key = _extract_env_key(combined)
    if env_key and any(term in combined for term in ("配置", "环境变量", "env", "api key", "API key", "读取")):
        return f"{env_key} 配置位置与读取方式"
    summary = _extract_summary_line(body)
    summary = _clean_summary_text(summary)
    if summary:
        if mentions_snowball and "snowball-notes" not in normalize_text(summary) and len(summary) <= SHORT_SUMMARY_PREFIX_THRESHOLD:
            summary = f"snowball-notes {summary}"
        return _truncate_title(summary)
    return _truncate_title(title.strip("？?"))


def _looks_conversational_title(title: str) -> bool:
    normalized = normalize_text(title)
    if any(marker in title for marker in QUESTION_MARKERS):
        return True
    if any(token in title for token in ("怎么", "如何", "为什么", "哪个", "哪一个", "什么叫", "在哪")):
        return True
    if title.startswith("是set") or "pyth" in normalized and "status" in normalized:
        return True
    return normalized.startswith(CONVERSATIONAL_PREFIXES)


def _clean_title_text(value: str) -> str:
    text = re.sub(r"\s+", " ", value).strip(" -:：")
    return text


def _clean_summary_text(value: str) -> str:
    text = re.sub(r"\s+", " ", value).strip()
    for prefix in SUMMARY_LEADS:
        if text.startswith(prefix):
            text = text[len(prefix) :].strip(" ：:-")
    text = re.sub(r"^这是一篇关于", "", text).strip(" ：:-")
    text = re.sub(r"的澄清$", "", text).strip(" ：:-")
    return text.strip("。！？!? ")


def _extract_summary_line(body: str) -> str:
    match = re.search(r"^## Summary\s+(.+?)(?:\n## |\Z)", body.strip(), re.MULTILINE | re.DOTALL)
    if match:
        summary = match.group(1).strip()
        first_line = next((line.strip() for line in summary.splitlines() if line.strip()), "")
        if first_line:
            return first_line
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "-", "*", "`")):
            continue
        return stripped
    return ""


def _extract_env_key(value: str) -> str | None:
    for match in ENV_KEY_RE.findall(value):
        if any(token in match for token in ("KEY", "TOKEN", "PAT", "SECRET")):
            return match
    return None


def _extract_error_signature(value: str) -> str | None:
    match = ERROR_RE.search(value)
    if not match:
        return None
    return match.group(0).strip("。！？!? ")


def _truncate_title(value: str, limit: int = DEFAULT_TITLE_TRUNCATION) -> str:
    compact = re.sub(r"\s+", " ", value).strip(" ：:-")
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip(" ，,:：-")
