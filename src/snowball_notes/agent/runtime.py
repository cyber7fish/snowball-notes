from __future__ import annotations

import json
import time
from pathlib import Path

from ..config import SnowballConfig
from ..models import AgentResult, ReplayBundle, RunState, ToolResult
from ..storage.audit import write_audit_log
from ..utils import now_utc_iso
from .adapter import ModelRetryableError
from .commit import Committer
from .guardrails import check_guardrail
from .memory import load_session_memory, update_session_memory
from .state import AgentState
from .state_machine import transition_state
from .tools import validated_tool_execute
from .trace import create_trace, save_agent_trace


class RetryExhaustedError(RuntimeError):
    pass


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
        self._log_run_started(trace.trace_id, task.task_id, event)
        try:
            transition_state(self.db, task.task_id, RunState.PREPARED, RunState.RUNNING)
            for step_index in range(self.config.agent.max_steps):
                response = self._respond_with_retry(event, state, messages, step_index)
                state.model_context.pop("next_input_items", None)
                if response.provider_response_id:
                    state.model_context["previous_response_id"] = response.provider_response_id
                if response.stop_reason == "end_turn":
                    messages = self._advance_messages(messages, response, [])
                    break
                if response.stop_reason != "tool_use":
                    state.is_terminated = True
                    state.terminal_reason = "unexpected_model_response"
                    break
                tool_messages = []
                followup_items = []
                for tool_call in response.tool_use_blocks:
                    guardrail = check_guardrail(self.config, state, tool_call.name)
                    if not guardrail.allowed:
                        payload = {"blocked": True, "reason": guardrail.reason}
                        trace.record_step(
                            step_index=step_index,
                            runtime_state=RunState.RUNNING.value,
                            decision_summary=response.decision_summary,
                            tool_name=tool_call.name,
                            tool_input_json=json.dumps(tool_call.input, ensure_ascii=False),
                            tool_result_json=json.dumps(payload, ensure_ascii=False),
                            tool_success=False,
                            proposal_ids=[],
                            guardrail_blocked=True,
                            duration_ms=0,
                            usage=response.usage,
                        )
                        state.tool_context.setdefault(tool_call.name, []).append(payload)
                        tool_messages.append(self._tool_message(tool_call, payload))
                        followup_items.append(self._tool_followup_item(tool_call.call_id, payload))
                        continue
                    proposal_count = len(state.proposals)
                    t0 = time.monotonic()
                    observation = validated_tool_execute(
                        tool_call.name,
                        tool_call.input,
                        self.tool_registry,
                        state,
                    )
                    tool_duration_ms = int((time.monotonic() - t0) * 1000)
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
                    tool_messages.append(self._tool_message(tool_call, payload))
                    followup_items.append(self._tool_followup_item(tool_call.call_id, payload))
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
                        duration_ms=tool_duration_ms,
                        usage=response.usage,
                    )
                    if state.is_terminated:
                        break
                messages = self._advance_messages(messages, response, tool_messages)
                if response.provider_response_id and followup_items and not state.is_terminated:
                    state.model_context["next_input_items"] = followup_items
                else:
                    state.model_context.pop("next_input_items", None)
                if state.is_terminated:
                    return self._finalize_flagged(
                        trace, state, task.task_id, event,
                        RunState.RUNNING, state.terminal_reason or "flagged",
                    )
            else:
                trace.exceeded_max_steps = True
                return self._finalize_flagged(
                    trace, state, task.task_id, event,
                    RunState.RUNNING, "exceeded_max_steps",
                )

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
                return self._finalize_flagged(
                    trace, state, task.task_id, event,
                    RunState.PROPOSED_ACTIONS, reason,
                )

            transition_state(self.db, task.task_id, RunState.PROPOSED_ACTIONS, RunState.COMMITTING)
            commit_result = committer.commit()
            if commit_result.success:
                final_decision = self._final_decision(state)
                trace.finish(final_decision, "completed", event.source_confidence)
                transition_state(self.db, task.task_id, RunState.COMMITTING, RunState.COMPLETED)
                self._persist_trace_and_replay(trace, state)
                self._flush_session_memory(state, final_decision)
                self._log_run_finished(
                    trace, task.task_id, event, RunState.COMPLETED.value,
                    committed_note_ids=commit_result.committed_note_ids,
                )
                self.db.commit()
                return AgentResult(
                    state=RunState.COMPLETED,
                    committed_note_ids=commit_result.committed_note_ids,
                )

            target = RunState.FAILED_RETRYABLE if commit_result.disposition == "retryable" else RunState.FAILED_FATAL
            return self._finalize_failed(
                trace, state, task.task_id, event,
                RunState.COMMITTING, target, commit_result.reason,
            )
        except RetryExhaustedError as exc:
            return self._finalize_exception(
                trace, state, task.task_id, event,
                RunState.FAILED_RETRYABLE, str(exc),
            )
        except Exception as exc:
            write_audit_log(
                self.db, "runtime_error", {"error": str(exc)},
                level="error", trace_id=trace.trace_id,
                turn_id=event.turn_id, task_id=task.task_id,
            )
            return self._finalize_exception(
                trace, state, task.task_id, event,
                RunState.FAILED_FATAL, str(exc),
            )

    def _finalize_flagged(self, trace, state, task_id, event, from_state, reason) -> AgentResult:
        trace.finish("flagged", reason, event.source_confidence)
        transition_state(self.db, task_id, from_state, RunState.FLAGGED, reason)
        self._persist_trace_and_replay(trace, state)
        self._flush_session_memory(state, "flagged")
        self._log_run_finished(trace, task_id, event, RunState.FLAGGED.value, reason=reason)
        self.db.commit()
        return AgentResult(state=RunState.FLAGGED, reason=reason)

    def _finalize_failed(self, trace, state, task_id, event, from_state, target, reason) -> AgentResult:
        trace.finish(target.value, reason, event.source_confidence)
        transition_state(self.db, task_id, from_state, target, reason)
        self._persist_trace_and_replay(trace, state)
        self._log_run_finished(trace, task_id, event, target.value, reason=reason)
        self.db.commit()
        return AgentResult(state=target, reason=reason)

    def _finalize_exception(self, trace, state, task_id, event, target, reason) -> AgentResult:
        trace.finish(target.value, reason, event.source_confidence)
        current = self.db.fetchone("SELECT status FROM tasks WHERE task_id = ?", (task_id,))
        if current and current["status"] == RunState.RUNNING.value:
            transition_state(self.db, task_id, RunState.RUNNING, target, reason)
        self._persist_trace_and_replay(trace, state)
        self._log_run_finished(trace, task_id, event, target.value, reason=reason)
        self.db.commit()
        return AgentResult(state=target, reason=reason)

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

    def _respond_with_retry(self, event, state, messages, step_index: int):
        for attempt in range(self.config.agent.max_model_retries):
            try:
                return self.model_adapter.respond(
                    event=event,
                    state=state,
                    messages=messages,
                    tools=self.tool_registry,
                    step_index=step_index,
                )
            except ModelRetryableError as exc:
                if attempt + 1 >= self.config.agent.max_model_retries:
                    raise RetryExhaustedError(str(exc)) from exc
                time.sleep(2 ** attempt)

    def _advance_messages(self, messages: list[dict], response, tool_messages: list[dict]) -> list[dict]:
        assistant_message = {
            "role": "assistant",
            "content": {
                "decision_summary": response.decision_summary,
                "stop_reason": response.stop_reason,
                "tool_calls": [
                    {
                        "call_id": tool_call.call_id,
                        "name": tool_call.name,
                        "input": tool_call.input,
                    }
                    for tool_call in response.tool_use_blocks
                ],
            },
        }
        return messages + [assistant_message] + tool_messages

    def _tool_message(self, tool_call, payload: dict) -> dict:
        return {
            "role": "tool",
            "call_id": tool_call.call_id,
            "name": tool_call.name,
            "content": payload,
        }

    def _tool_followup_item(self, call_id: str, payload: dict) -> dict:
        return {
            "type": "function_call_output",
            "call_id": call_id,
            "output": json.dumps(payload, ensure_ascii=False),
        }

    def _build_initial_messages(self, event, session_memory) -> list[dict]:
        recent_actions = self._recent_session_actions(session_memory)
        return [
            {
                "role": "user",
                "content": {
                    "turn_id": event.turn_id,
                    "user_message": event.user_message,
                    "assistant_final_answer": event.assistant_final_answer,
                    "source_confidence": event.source_confidence,
                    "previous_turns": len(session_memory.processed_turns),
                    "session_context": self._format_session_context(session_memory),
                    "recent_actions": recent_actions,
                },
            }
        ]

    def _recent_session_actions(self, session_memory) -> list[dict]:
        recent = []
        seen = set()
        for turn in session_memory.processed_turns:
            key = (turn.turn_id, turn.action_type, turn.note_id)
            if key in seen:
                continue
            seen.add(key)
            recent.append(
                {
                    "turn_id": turn.turn_id,
                    "final_decision": turn.final_decision,
                    "note_id": turn.note_id,
                    "action_type": turn.action_type,
                    "note_title": turn.note_title,
                }
            )
        return recent[:5]

    def _format_session_context(self, session_memory) -> str:
        recent_actions = self._recent_session_actions(session_memory)
        if not recent_actions:
            return "No prior turns from this conversation have been processed yet."
        lines = [
            f"{len(session_memory.processed_turns)} prior turn records found in this conversation.",
            "Recent note actions:",
        ]
        for item in recent_actions:
            summary = item["final_decision"]
            if item["action_type"]:
                summary = f"{summary} via {item['action_type']}"
            if item["note_title"]:
                summary = f"{summary} on {item['note_title']}"
            elif item["note_id"]:
                summary = f"{summary} on {item['note_id']}"
            lines.append(f"- {item['turn_id']}: {summary}")
        lines.append("Do not repeat create or append actions against the same note within this conversation.")
        return "\n".join(lines)

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
            note_ids = []
            if proposal.action_type == "link_notes":
                source_note_id = proposal.payload.get("source_note_id")
                target_note_id = proposal.payload.get("target_note_id")
                if isinstance(source_note_id, str) and source_note_id:
                    note_ids.append(source_note_id)
                if isinstance(target_note_id, str) and target_note_id:
                    note_ids.append(target_note_id)
            elif proposal.target_note_id:
                note_ids.append(proposal.target_note_id)
            for note_id in note_ids:
                note_row = self.db.fetchone(
                    "SELECT title FROM notes WHERE note_id = ?",
                    (note_id,),
                )
                actions.append(
                    {
                        "note_id": note_id,
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
        if any(proposal.action_type == "link_notes" for proposal in state.proposals):
            return "link_notes"
        if any(proposal.action_type == "archive_turn" for proposal in state.proposals):
            return "archive_turn"
        return "completed"

    def _log_run_started(self, trace_id: str, task_id: str, event) -> None:
        write_audit_log(
            self.db,
            "agent_run_started",
            {
                "conversation_id": event.conversation_id,
                "source_confidence": event.source_confidence,
                "prompt_version": self.config.agent.prompt_version,
                "model_name": self.model_adapter.model_name,
            },
            trace_id=trace_id,
            turn_id=event.turn_id,
            task_id=task_id,
        )

    def _log_run_finished(
        self,
        trace,
        task_id: str,
        event,
        result_state: str,
        *,
        reason: str = "",
        committed_note_ids: list[str] | None = None,
    ) -> None:
        write_audit_log(
            self.db,
            "agent_run_finished",
            {
                "result_state": result_state,
                "final_decision": trace.final_decision,
                "terminal_reason": trace.terminal_reason,
                "reason": reason,
                "committed_note_ids": committed_note_ids or [],
            },
            trace_id=trace.trace_id,
            turn_id=event.turn_id,
            task_id=task_id,
        )
