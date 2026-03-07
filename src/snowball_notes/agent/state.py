from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..models import ActionProposal, SessionMemory, StandardEvent


@dataclass
class AgentState:
    event: StandardEvent
    task_id: str
    trace_id: str
    session_memory: SessionMemory
    write_count: int = 0
    append_count: int = 0
    is_terminated: bool = False
    terminal_reason: str = ""
    proposals: list[ActionProposal] = field(default_factory=list)
    tool_results_for_replay: list[dict[str, Any]] = field(default_factory=list)
    knowledge_snapshot_refs: list[dict[str, Any]] = field(default_factory=list)
    tool_context: dict[str, list[Any]] = field(default_factory=dict)

