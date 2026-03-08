from __future__ import annotations

import json
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import default_config
from ..models import (
    ActionProposal,
    ModelResponse,
    ReplayBundle,
    RunState,
    StandardEvent,
    TaskRecord,
    TokenUsage,
    ToolCall,
    ToolResult,
)
from ..embedding import build_embedding_provider, build_vector_store
from ..storage.sqlite import Database
from ..storage.vault import Vault
from ..utils import new_id, now_utc_iso, safe_read_text, sha256_text, write_atomic_text
from .adapter import build_model_adapter
from .memory import SQLiteKnowledgeIndex
from .runtime import SnowballAgent
from .tools import Tool, build_tool_registry


ACTION_TOOL_TO_TYPE = {
    "propose_create_note": "create_note",
    "propose_append_to_note": "append_note",
    "propose_archive_turn": "archive_turn",
    "propose_link_notes": "link_notes",
}


@dataclass
class ReplayOutcome:
    trace_id: str
    mode: str
    result_state: str
    final_decision: str
    matched_original: bool
    terminal_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "mode": self.mode,
            "result_state": self.result_state,
            "final_decision": self.final_decision,
            "matched_original": self.matched_original,
            "terminal_reason": self.terminal_reason,
        }


@dataclass
class _ReplayContext:
    bundle: ReplayBundle
    event: StandardEvent
    trace_json: dict[str, Any]
    tool_results: list[dict[str, Any]]
    action_proposals: list[dict[str, Any]]
    note_rows: list[dict[str, Any]]


class ReplayRunner:
    def __init__(self, config, db, vault):
        self.config = config
        self.db = db
        self.vault = vault

    def logical_replay(self, trace_id: str) -> ReplayOutcome:
        context = self._load_context(trace_id)
        with tempfile.TemporaryDirectory(prefix="snowball-logical-replay-") as temp_dir:
            sandbox_db, sandbox_vault, sandbox_config = self._build_sandbox(temp_dir, context)
            try:
                event, task = self._seed_task(sandbox_db, context.event)
                adapter = _TracePlaybackAdapter(context.trace_json.get("steps", []), context.trace_json.get("model_name", "trace-playback"))
                tools = _build_frozen_tools(sandbox_db, context)
                agent = SnowballAgent(sandbox_config, adapter, tools, sandbox_vault, sandbox_db)
                result = agent.run(task, event)
                replay_trace = sandbox_db.fetchone(
                    """
                    SELECT final_decision, terminal_reason
                    FROM agent_traces
                    ORDER BY created_at DESC, rowid DESC
                    LIMIT 1
                    """
                ) or {}
                final_decision = replay_trace.get("final_decision", result.state.value)
                return ReplayOutcome(
                    trace_id=trace_id,
                    mode="logical",
                    result_state=result.state.value,
                    final_decision=final_decision,
                    matched_original=final_decision == context.trace_json.get("final_decision"),
                    terminal_reason=replay_trace.get("terminal_reason", result.reason),
                )
            finally:
                sandbox_db.close()

    def live_replay(self, trace_id: str) -> ReplayOutcome:
        context = self._load_context(trace_id)
        with tempfile.TemporaryDirectory(prefix="snowball-live-replay-") as temp_dir:
            sandbox_db, sandbox_vault, sandbox_config = self._build_sandbox(temp_dir, context)
            try:
                event, task = self._seed_task(sandbox_db, context.event)
                embedding_provider = build_embedding_provider(sandbox_config)
                vector_store = build_vector_store(sandbox_config, sandbox_db)
                knowledge_index = SQLiteKnowledgeIndex(
                    sandbox_db,
                    config=sandbox_config,
                    embedding_provider=embedding_provider,
                    vector_store=vector_store,
                )
                tools = build_tool_registry(sandbox_db, knowledge_index)
                adapter = build_model_adapter(sandbox_config)
                agent = SnowballAgent(sandbox_config, adapter, tools, sandbox_vault, sandbox_db)
                result = agent.run(task, event)
                replay_trace = sandbox_db.fetchone(
                    """
                    SELECT final_decision, terminal_reason
                    FROM agent_traces
                    ORDER BY created_at DESC, rowid DESC
                    LIMIT 1
                    """
                ) or {}
                final_decision = replay_trace.get("final_decision", result.state.value)
                return ReplayOutcome(
                    trace_id=trace_id,
                    mode="live",
                    result_state=result.state.value,
                    final_decision=final_decision,
                    matched_original=final_decision == context.trace_json.get("final_decision"),
                    terminal_reason=replay_trace.get("terminal_reason", result.reason),
                )
            finally:
                sandbox_db.close()

    def _load_context(self, trace_id: str) -> _ReplayContext:
        bundle_row = self.db.fetchone("SELECT * FROM replay_bundles WHERE trace_id = ?", (trace_id,))
        if bundle_row is None:
            raise ValueError(f"trace {trace_id} not found")
        trace_row = self.db.fetchone("SELECT trace_json FROM agent_traces WHERE trace_id = ?", (trace_id,))
        if trace_row is None:
            raise ValueError(f"trace payload {trace_id} not found")
        proposal_rows = self.db.fetchall(
            """
            SELECT proposal_id, trace_id, turn_id, action_type, target_note_id, payload_json,
                   idempotency_key, status, created_at, committed_at
            FROM action_proposals
            WHERE trace_id = ?
            ORDER BY created_at ASC, proposal_id ASC
            """,
            (trace_id,),
        )
        bundle = ReplayBundle(
            trace_id=bundle_row["trace_id"],
            event_json=bundle_row["event_json"],
            prompt_snapshot=bundle_row["prompt_snapshot"],
            config_snapshot_json=bundle_row["config_snapshot_json"],
            tool_results_json=bundle_row["tool_results_json"],
            knowledge_snapshot_refs_json=bundle_row["knowledge_snapshot_refs_json"],
            model_name=bundle_row["model_name"],
            model_adapter_version=bundle_row["model_adapter_version"],
            created_at=bundle_row["created_at"],
        )
        trace_json = json.loads(trace_row["trace_json"])
        note_ids = {
            row["target_note_id"]
            for row in proposal_rows
            if row.get("target_note_id")
        }
        for item in json.loads(bundle.knowledge_snapshot_refs_json or "[]"):
            note_id = item.get("note_id")
            if note_id:
                note_ids.add(note_id)
        note_rows = []
        for note_id in sorted(note_ids):
            row = self.db.fetchone(
                """
                SELECT note_id, note_type, title, vault_path, content_hash, status, metadata_json
                FROM notes
                WHERE note_id = ?
                """,
                (note_id,),
            )
            if row:
                note_rows.append(row)
                continue
            note_rows.append(
                {
                    "note_id": note_id,
                    "note_type": "atomic",
                    "title": note_id,
                    "vault_path": str(self.vault.inbox_dir / f"{note_id}.md"),
                    "content_hash": "",
                    "status": "approved",
                    "metadata_json": "{}",
                }
            )
        return _ReplayContext(
            bundle=bundle,
            event=StandardEvent.from_dict(json.loads(bundle.event_json)),
            trace_json=trace_json,
            tool_results=json.loads(bundle.tool_results_json or "[]"),
            action_proposals=proposal_rows,
            note_rows=note_rows,
        )

    def _build_sandbox(self, temp_dir: str, context: _ReplayContext):
        sandbox_root = Path(temp_dir).resolve()
        config = default_config(sandbox_root)
        snapshot = json.loads(context.bundle.config_snapshot_json or "{}")
        _apply_config_snapshot(config, snapshot)
        config.project_root = sandbox_root
        db = Database(config.db_path)
        db.migrate()
        vault = Vault(config)
        self._seed_notes(db, vault, context.note_rows)
        return db, vault, config

    def _seed_notes(self, db, vault, note_rows: list[dict[str, Any]]) -> None:
        for row in note_rows:
            source_path = Path(row["vault_path"])
            if source_path.exists():
                content = safe_read_text(source_path)
            else:
                content = f"# {row['title']}\n\nReplayed note stub.\n"
            target_dir = vault.archive_dir if row.get("note_type") == "archive" else vault.inbox_dir
            target_path = target_dir / source_path.name
            write_atomic_text(target_path, content)
            db.execute(
                """
                INSERT OR REPLACE INTO notes (
                  note_id, note_type, title, vault_path, content_hash,
                  status, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["note_id"],
                    row.get("note_type", "atomic"),
                    row["title"],
                    str(target_path.resolve()),
                    sha256_text(content),
                    row.get("status", "approved"),
                    row.get("metadata_json") or "{}",
                    now_utc_iso(),
                    now_utc_iso(),
                ),
            )
        db.commit()

    def _seed_task(self, db, event: StandardEvent) -> tuple[StandardEvent, TaskRecord]:
        task_id = new_id("replay_task")
        payload_json = json.dumps(event.to_dict(), ensure_ascii=False)
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
                payload_json,
            ),
        )
        db.execute(
            """
            INSERT INTO tasks (
              task_id, event_id, dedupe_key, status, retry_count, max_retries, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 0, 1, ?, ?)
            """,
            (
                task_id,
                event.event_id,
                event.turn_id,
                RunState.PREPARED.value,
                now_utc_iso(),
                now_utc_iso(),
            ),
        )
        db.commit()
        return event, TaskRecord(
            task_id=task_id,
            event_id=event.event_id,
            status=RunState.PREPARED,
            retry_count=0,
            max_retries=1,
        )


class _TracePlaybackAdapter:
    version = "trace-playback-v1"

    def __init__(self, trace_steps: list[dict[str, Any]], model_name: str):
        self.steps = list(trace_steps)
        self.model_name = model_name
        self._cursor = 0

    def respond(self, event, state, messages, tools, step_index: int) -> ModelResponse:
        if self._cursor >= len(self.steps):
            return ModelResponse(
                stop_reason="end_turn",
                decision_summary="Trace replay reached end_turn.",
                usage=TokenUsage(),
            )
        step = self.steps[self._cursor]
        self._cursor += 1
        usage = TokenUsage(
            input_tokens=int(step.get("input_tokens") or 0),
            output_tokens=int(step.get("output_tokens") or 0),
        )
        tool_name = step.get("tool_name")
        if not tool_name:
            return ModelResponse(
                stop_reason="end_turn",
                decision_summary=str(step.get("decision_summary") or "Trace replay ended."),
                usage=usage,
            )
        tool_input = json.loads(step.get("tool_input_json") or "{}")
        return ModelResponse(
            stop_reason="tool_use",
            tool_use_blocks=[ToolCall(call_id=new_id("replay_tool"), name=tool_name, input=tool_input)],
            decision_summary=str(step.get("decision_summary") or tool_name),
            usage=usage,
            provider_response_id=f"trace-replay-{self._cursor}",
        )


class _FrozenReplayTool(Tool):
    def __init__(self, name: str, stateful_context: "_FrozenReplayContext"):
        self.name = name
        self.context = stateful_context

    def execute(self, payload: dict[str, Any], state) -> ToolResult:
        recorded = self.context.pop_tool_result(self.name)
        if recorded.get("success"):
            data = recorded.get("output")
            self.context.apply_side_effects(self.name, state, data)
            return ToolResult.ok(data)
        return ToolResult(
            success=False,
            data=recorded.get("output"),
            error_code=recorded.get("error_code") or "replay_error",
            error_message=recorded.get("error_message") or f"recorded failure for {self.name}",
        )


class _FrozenReplayContext:
    def __init__(self, replay_context: _ReplayContext):
        self.tool_results = list(replay_context.tool_results)
        self.action_proposals = list(replay_context.action_proposals)
        self._tool_index = 0
        self._proposal_index = 0

    def pop_tool_result(self, tool_name: str) -> dict[str, Any]:
        if self._tool_index >= len(self.tool_results):
            raise RuntimeError(f"no recorded tool result remaining for {tool_name}")
        item = self.tool_results[self._tool_index]
        self._tool_index += 1
        if item.get("tool") != tool_name:
            raise RuntimeError(
                f"replay tool order mismatch: expected {item.get('tool')} but got {tool_name}"
            )
        return item

    def apply_side_effects(self, tool_name: str, state, data: Any) -> None:
        action_type = ACTION_TOOL_TO_TYPE.get(tool_name)
        if action_type:
            proposal = self._pop_proposal(action_type)
            state.proposals.append(proposal)
            state.write_count += 1
            if action_type == "append_note":
                state.append_count += 1
            return
        if tool_name == "search_similar_notes" and isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                state.knowledge_snapshot_refs.append(
                    {
                        "note_id": item.get("note_id"),
                        "content_hash": item.get("content_hash", ""),
                        "title": item.get("title", ""),
                        "similarity": item.get("similarity"),
                    }
                )
            return
        if tool_name == "flag_for_review":
            reason = ""
            if isinstance(data, dict):
                reason = str(data.get("reason") or "")
            state.is_terminated = True
            state.terminal_reason = reason or "flagged"
            state.proposals.clear()

    def _pop_proposal(self, expected_action_type: str) -> ActionProposal:
        if self._proposal_index >= len(self.action_proposals):
            raise RuntimeError(f"no recorded proposal remaining for {expected_action_type}")
        row = self.action_proposals[self._proposal_index]
        self._proposal_index += 1
        if row["action_type"] != expected_action_type:
            raise RuntimeError(
                f"replay proposal order mismatch: expected {row['action_type']} but got {expected_action_type}"
            )
        return ActionProposal(
            proposal_id=row["proposal_id"],
            trace_id=row["trace_id"],
            turn_id=row["turn_id"],
            action_type=row["action_type"],
            target_note_id=row.get("target_note_id"),
            payload=json.loads(row["payload_json"]),
            idempotency_key=row["idempotency_key"],
            status="proposed",
            created_at=row["created_at"],
            committed_at=row.get("committed_at"),
        )


def _build_frozen_tools(db, replay_context: _ReplayContext) -> dict[str, Tool]:
    frozen_context = _FrozenReplayContext(replay_context)
    tools = {}
    for tool_name in [
        "assess_turn_value",
        "extract_knowledge_points",
        "search_similar_notes",
        "read_note",
        "propose_create_note",
        "propose_append_to_note",
        "propose_archive_turn",
        "propose_link_notes",
        "flag_for_review",
    ]:
        tools[tool_name] = _FrozenReplayTool(tool_name, frozen_context)
    return tools


def _apply_config_snapshot(config, payload: dict[str, Any]) -> None:
    for section_name, values in payload.items():
        target = getattr(config, section_name, None)
        if target is None or not isinstance(values, dict):
            continue
        for key, value in values.items():
            if hasattr(target, key):
                setattr(target, key, deepcopy(value))
