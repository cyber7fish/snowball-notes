from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from .utils import now_utc, now_utc_iso


class RunState(str, Enum):
    RECEIVED = "received"
    PREPARED = "prepared"
    RUNNING = "running"
    PROPOSED_ACTIONS = "proposed_actions"
    COMMITTING = "committing"
    COMPLETED = "completed"
    FLAGGED = "flagged"
    FAILED_RETRYABLE = "failed_retryable"
    FAILED_FATAL = "failed_fatal"


@dataclass
class StandardEvent:
    event_id: str
    session_file: str
    conversation_id: str
    turn_id: str
    user_message: str
    assistant_final_answer: str
    displayed_at: str
    source_completeness: str
    source_confidence: float
    parser_version: str
    context_meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StandardEvent":
        return cls(
            event_id=payload["event_id"],
            session_file=payload["session_file"],
            conversation_id=payload["conversation_id"],
            turn_id=payload["turn_id"],
            user_message=payload.get("user_message", ""),
            assistant_final_answer=payload.get("assistant_final_answer", ""),
            displayed_at=payload["displayed_at"],
            source_completeness=payload["source_completeness"],
            source_confidence=float(payload["source_confidence"]),
            parser_version=payload["parser_version"],
            context_meta=dict(payload.get("context_meta", {})),
        )


@dataclass
class TaskRecord:
    task_id: str
    event_id: str
    status: RunState
    retry_count: int
    max_retries: int
    claimed_by: str | None = None
    claimed_at: str | None = None


@dataclass
class NoteMatch:
    note_id: str
    title: str
    vault_path: str
    similarity: float
    content_hash: str
    excerpt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ActionProposal:
    proposal_id: str
    trace_id: str
    turn_id: str
    action_type: str
    target_note_id: str | None
    payload: dict[str, Any]
    idempotency_key: str
    status: str = "proposed"
    created_at: str = field(default_factory=now_utc_iso)
    committed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolResult:
    success: bool
    data: Any = None
    error_code: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, data: Any = None, metadata: dict[str, Any] | None = None) -> "ToolResult":
        return cls(success=True, data=data, metadata=metadata or {})

    @classmethod
    def blocked(cls, reason: str) -> "ToolResult":
        return cls(success=False, error_code="guardrail_blocked", error_message=reason)

    @classmethod
    def error(cls, code: str, message: str) -> "ToolResult":
        return cls(success=False, error_code=code, error_message=message)

    @classmethod
    def validation_error(cls, message: str) -> "ToolResult":
        return cls(success=False, error_code="invalid_tool_input", error_message=message)


@dataclass
class GuardrailResult:
    allowed: bool
    reason: str = ""

    @classmethod
    def allow(cls) -> "GuardrailResult":
        return cls(allowed=True)

    @classmethod
    def block(cls, reason: str) -> "GuardrailResult":
        return cls(allowed=False, reason=reason)


@dataclass
class CommitResult:
    success: bool
    disposition: str
    reason: str = ""
    committed_note_ids: list[str] = field(default_factory=list)

    @classmethod
    def completed(cls, note_ids: list[str]) -> "CommitResult":
        return cls(success=True, disposition="completed", committed_note_ids=note_ids)

    @classmethod
    def rejected(cls, reason: str) -> "CommitResult":
        return cls(success=False, disposition="rejected", reason=reason)

    @classmethod
    def retryable(cls, reason: str) -> "CommitResult":
        return cls(success=False, disposition="retryable", reason=reason)

    @classmethod
    def fatal(cls, reason: str) -> "CommitResult":
        return cls(success=False, disposition="fatal", reason=reason)


@dataclass
class AgentResult:
    state: RunState
    reason: str = ""
    committed_note_ids: list[str] = field(default_factory=list)


@dataclass
class ToolCall:
    call_id: str
    name: str
    input: dict[str, Any]


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ModelResponse:
    stop_reason: str
    tool_use_blocks: list[ToolCall] = field(default_factory=list)
    decision_summary: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass
class TraceStep:
    step_index: int
    runtime_state: str
    decision_summary: str
    tool_name: str | None
    tool_input_json: str | None
    tool_result_json: str | None
    tool_success: bool | None
    proposal_ids: list[str]
    guardrail_blocked: bool
    duration_ms: int
    input_tokens: int
    output_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentTrace:
    trace_id: str
    event_id: str
    turn_id: str
    prompt_version: str
    model_name: str
    started_at: str = field(default_factory=now_utc_iso)
    finished_at: str | None = None
    total_steps: int = 0
    exceeded_max_steps: bool = False
    terminal_reason: str = ""
    final_decision: str = ""
    final_confidence: float | None = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_duration_ms: int = 0
    steps: list[TraceStep] = field(default_factory=list)

    def record_step(
        self,
        step_index: int,
        runtime_state: str,
        decision_summary: str,
        tool_name: str | None,
        tool_input_json: str | None,
        tool_result_json: str | None,
        tool_success: bool | None,
        proposal_ids: list[str],
        guardrail_blocked: bool,
        duration_ms: int,
        usage: TokenUsage,
    ) -> None:
        self.steps.append(
            TraceStep(
                step_index=step_index,
                runtime_state=runtime_state,
                decision_summary=decision_summary,
                tool_name=tool_name,
                tool_input_json=tool_input_json,
                tool_result_json=tool_result_json,
                tool_success=tool_success,
                proposal_ids=proposal_ids,
                guardrail_blocked=guardrail_blocked,
                duration_ms=duration_ms,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
            )
        )
        self.total_steps = max(self.total_steps, step_index + 1)
        self.total_input_tokens += usage.input_tokens
        self.total_output_tokens += usage.output_tokens

    def finish(self, final_decision: str, terminal_reason: str, confidence: float | None = None) -> None:
        self.finished_at = now_utc_iso()
        self.final_decision = final_decision
        self.terminal_reason = terminal_reason
        self.final_confidence = confidence
        started = datetime.fromisoformat(self.started_at)
        finished = datetime.fromisoformat(self.finished_at)
        self.total_duration_ms = int((finished - started).total_seconds() * 1000)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "event_id": self.event_id,
            "turn_id": self.turn_id,
            "prompt_version": self.prompt_version,
            "model_name": self.model_name,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_steps": self.total_steps,
            "exceeded_max_steps": int(self.exceeded_max_steps),
            "terminal_reason": self.terminal_reason,
            "final_decision": self.final_decision,
            "final_confidence": self.final_confidence,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_duration_ms": self.total_duration_ms,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass
class ReplayBundle:
    trace_id: str
    event_json: str
    prompt_snapshot: str
    config_snapshot_json: str
    tool_results_json: str
    knowledge_snapshot_refs_json: str
    model_name: str
    model_adapter_version: str
    created_at: str = field(default_factory=now_utc_iso)


@dataclass
class SessionTurn:
    turn_id: str
    processed_at: str
    final_decision: str
    note_id: str | None = None
    action_type: str | None = None
    note_title: str | None = None


@dataclass
class SessionMemory:
    conversation_id: str
    processed_turns: list[SessionTurn] = field(default_factory=list)


@dataclass
class ReconcileReport:
    orphan_files: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)

    def ok(self) -> bool:
        return not self.orphan_files and not self.missing_files

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
