from __future__ import annotations

import json
from pathlib import Path

from ..config import SnowballConfig
from ..models import AgentResult, ReplayBundle, RunState, ToolResult
from ..storage.audit import write_audit_log
from ..utils import now_utc_iso
from .commit import Committer
from .guardrails import check_guardrail
from .memory import load_session_memory, update_session_memory
from .state import AgentState
from .state_machine import transition_state
from .tools import validated_tool_execute
from .trace import create_trace, save_agent_trace


class SnowballAgent:
    def __init__(self, config: SnowballConfig, model_adapter, tool_registry: dict, vault, db):
        self.config = config
        self.model_adapter = model_adapter
        self.tool_registry = tool_registry
        self.vault = vault
        self.db = db

    def run(self, task, event) -> AgentResult:
        trace = create_trace(event, self.config.agent.prompt_version, self.model_adapter.model_name)
        session_memory = load_session_memory(self.db, event.conversation_id)
        state = AgentState(
            event=event,
            task_id=task.task_id,
            trace_id=trace.trace_id,
            session_memory=session_memory,
        )
        messages = self._build_initial_messages(event, session_memory)
        try:
            transition_state(self.db, task.task_id, RunState.PREPARED, RunState.RUNNING)
            for step_index in range(self.config.agent.max_steps):
                response = self.model_adapter.respond(
                    event=event,
                    state=state,
                    messages=messages,
                    tools=self.tool_registry,
                    step_index=step_index,
                )
                if response.stop_reason == "end_turn":
                    break
                if response.stop_reason != "tool_use":
                    state.is_terminated = True
                    state.terminal_reason = "unexpected_model_response"
                    break
                for tool_call in response.tool_use_blocks:
                    guardrail = check_guardrail(self.config, state, tool_call.name)
                    if not guardrail.allowed:
                        observation = ToolResult.blocked(guardrail.reason)
                        trace.record_step(
                            step_index=step_index,
                            runtime_state=RunState.RUNNING.value,
                            decision_summary=response.decision_summary,
                            tool_name=tool_call.name,
                            tool_input_json=json.dumps(tool_call.input, ensure_ascii=False),
                            tool_result_json=json.dumps({"blocked": True, "reason": guardrail.reason}, ensure_ascii=False),
                            tool_success=False,
                            proposal_ids=[],
                            guardrail_blocked=True,
                            duration_ms=0,
                            usage=response.usage,
                        )
                        state.tool_context.setdefault(tool_call.name, []).append({"blocked": True, "reason": guardrail.reason})
                        continue
                    proposal_count = len(state.proposals)
                    observation = validated_tool_execute(
                        tool_call.name,
                        tool_call.input,
                        self.tool_registry,
                        state,
                    )
                    new_proposals = state.proposals[proposal_count:]
                    for proposal in new_proposals:
                        self._save_proposal(proposal)
                    payload = observation.data if observation.success else {
                        "error_code": observation.error_code,
                        "error_message": observation.error_message,
                    }
                    state.tool_results_for_replay.append(
                        {
                            "step": step_index,
                            "tool": tool_call.name,
                            "input": tool_call.input,
                            "output": payload,
                            "success": observation.success,
                        }
                    )
                    state.tool_context.setdefault(tool_call.name, []).append(payload)
                    trace.record_step(
                        step_index=step_index,
                        runtime_state=RunState.RUNNING.value,
                        decision_summary=response.decision_summary,
                        tool_name=tool_call.name,
                        tool_input_json=json.dumps(tool_call.input, ensure_ascii=False),
                        tool_result_json=json.dumps(payload, ensure_ascii=False),
                        tool_success=observation.success,
                        proposal_ids=[proposal.proposal_id for proposal in new_proposals],
                        guardrail_blocked=False,
                        duration_ms=0,
                        usage=response.usage,
                    )
                    if state.is_terminated:
                        break
                if state.is_terminated:
                    trace.finish("flagged", state.terminal_reason or "flagged", event.source_confidence)
                    transition_state(self.db, task.task_id, RunState.RUNNING, RunState.FLAGGED, state.terminal_reason)
                    self._persist_trace_and_replay(trace, state)
                    self._flush_session_memory(state, "flagged")
                    self.db.commit()
                    return AgentResult(state=RunState.FLAGGED, reason=state.terminal_reason)
            else:
                trace.exceeded_max_steps = True
                trace.finish("flagged", "exceeded_max_steps", event.source_confidence)
                transition_state(self.db, task.task_id, RunState.RUNNING, RunState.FLAGGED, "exceeded_max_steps")
                self._persist_trace_and_replay(trace, state)
                self._flush_session_memory(state, "flagged")
                self.db.commit()
                return AgentResult(state=RunState.FLAGGED, reason="exceeded_max_steps")

            transition_state(self.db, task.task_id, RunState.RUNNING, RunState.PROPOSED_ACTIONS)
            committer = Committer(self.db, self.vault, state, self.config)
            validation_errors = committer.validate()
            if validation_errors:
                reason = "; ".join(validation_errors)
                write_audit_log(
                    self.db,
                    "commit_blocked",
                    {"errors": validation_errors},
                    level="warn",
                    trace_id=trace.trace_id,
                    turn_id=event.turn_id,
                    task_id=task.task_id,
                )
                trace.finish("flagged", reason, event.source_confidence)
                transition_state(self.db, task.task_id, RunState.PROPOSED_ACTIONS, RunState.FLAGGED, reason)
                self._persist_trace_and_replay(trace, state)
                self._flush_session_memory(state, "flagged")
                self.db.commit()
                return AgentResult(state=RunState.FLAGGED, reason=reason)

            transition_state(self.db, task.task_id, RunState.PROPOSED_ACTIONS, RunState.COMMITTING)
            commit_result = committer.commit()
            if commit_result.success:
                final_decision = self._final_decision(state)
                trace.finish(final_decision, "completed", event.source_confidence)
                transition_state(self.db, task.task_id, RunState.COMMITTING, RunState.COMPLETED)
                self._persist_trace_and_replay(trace, state)
                self._flush_session_memory(state, final_decision)
                self.db.commit()
                return AgentResult(
                    state=RunState.COMPLETED,
                    committed_note_ids=commit_result.committed_note_ids,
                )

            if commit_result.disposition == "retryable":
                trace.finish("failed_retryable", commit_result.reason, event.source_confidence)
                transition_state(
                    self.db, task.task_id, RunState.COMMITTING, RunState.FAILED_RETRYABLE, commit_result.reason
                )
                self._persist_trace_and_replay(trace, state)
                self.db.commit()
                return AgentResult(state=RunState.FAILED_RETRYABLE, reason=commit_result.reason)

            trace.finish("failed_fatal", commit_result.reason, event.source_confidence)
            transition_state(self.db, task.task_id, RunState.COMMITTING, RunState.FAILED_FATAL, commit_result.reason)
            self._persist_trace_and_replay(trace, state)
            self.db.commit()
            return AgentResult(state=RunState.FAILED_FATAL, reason=commit_result.reason)
        except Exception as exc:
            write_audit_log(
                self.db,
                "runtime_error",
                {"error": str(exc)},
                level="error",
                trace_id=trace.trace_id,
                turn_id=event.turn_id,
                task_id=task.task_id,
            )
            trace.finish("failed_fatal", str(exc), event.source_confidence)
            current = self.db.fetchone("SELECT status FROM tasks WHERE task_id = ?", (task.task_id,))
            if current and current["status"] == RunState.RUNNING.value:
                transition_state(self.db, task.task_id, RunState.RUNNING, RunState.FAILED_FATAL, str(exc))
            self._persist_trace_and_replay(trace, state)
            self.db.commit()
            return AgentResult(state=RunState.FAILED_FATAL, reason=str(exc))

    def _save_proposal(self, proposal) -> None:
        self.db.execute(
            """
            INSERT OR IGNORE INTO action_proposals (
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

    def _build_initial_messages(self, event, session_memory) -> list[dict]:
        return [
            {
                "role": "user",
                "content": {
                    "turn_id": event.turn_id,
                    "user_message": event.user_message,
                    "assistant_final_answer": event.assistant_final_answer,
                    "source_confidence": event.source_confidence,
                    "previous_turns": len(session_memory.processed_turns),
                },
            }
        ]

    def _persist_trace_and_replay(self, trace, state) -> None:
        save_agent_trace(self.db, trace)
        prompt_path = Path(__file__).resolve().parents[1] / "prompts" / self.config.agent.prompt_version
        prompt_text = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""
        bundle = ReplayBundle(
            trace_id=trace.trace_id,
            event_json=json.dumps(state.event.to_dict(), ensure_ascii=False),
            prompt_snapshot=prompt_text,
            config_snapshot_json=json.dumps(self.config.to_dict(), ensure_ascii=False),
            tool_results_json=json.dumps(state.tool_results_for_replay, ensure_ascii=False),
            knowledge_snapshot_refs_json=json.dumps(state.knowledge_snapshot_refs, ensure_ascii=False),
            model_name=self.model_adapter.model_name,
            model_adapter_version=self.model_adapter.version,
        )
        self.db.execute(
            """
            INSERT OR REPLACE INTO replay_bundles (
              trace_id, event_json, prompt_snapshot, config_snapshot_json,
              tool_results_json, knowledge_snapshot_refs_json,
              model_name, model_adapter_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                bundle.trace_id,
                bundle.event_json,
                bundle.prompt_snapshot,
                bundle.config_snapshot_json,
                bundle.tool_results_json,
                bundle.knowledge_snapshot_refs_json,
                bundle.model_name,
                bundle.model_adapter_version,
                bundle.created_at,
            ),
        )

    def _flush_session_memory(self, state, final_decision: str) -> None:
        actions = []
        for proposal in state.proposals:
            if proposal.target_note_id:
                note_row = self.db.fetchone(
                    "SELECT title FROM notes WHERE note_id = ?",
                    (proposal.target_note_id,),
                )
                actions.append(
                    {
                        "note_id": proposal.target_note_id,
                        "action_type": proposal.action_type,
                        "note_title": note_row["title"] if note_row else None,
                    }
                )
        update_session_memory(
            self.db,
            conversation_id=state.event.conversation_id,
            turn_id=state.event.turn_id,
            final_decision=final_decision,
            actions=actions,
        )

    def _final_decision(self, state) -> str:
        if not state.proposals:
            return "skip"
        if any(proposal.action_type == "create_note" for proposal in state.proposals):
            return "create_note"
        if any(proposal.action_type == "append_note" for proposal in state.proposals):
            return "append_note"
        if any(proposal.action_type == "archive_turn" for proposal in state.proposals):
            return "archive_turn"
        return "completed"

