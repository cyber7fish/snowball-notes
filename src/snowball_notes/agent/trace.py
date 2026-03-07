from __future__ import annotations

import json

from ..models import AgentTrace
from ..utils import new_id


def create_trace(event, prompt_version: str, model_name: str) -> AgentTrace:
    return AgentTrace(
        trace_id=new_id("trace"),
        event_id=event.event_id,
        turn_id=event.turn_id,
        prompt_version=prompt_version,
        model_name=model_name,
    )


def save_agent_trace(db, trace: AgentTrace) -> None:
    payload = trace.to_dict()
    db.execute(
        """
        INSERT OR REPLACE INTO agent_traces (
          trace_id, turn_id, event_id, prompt_version, model_name,
          started_at, finished_at, total_steps, exceeded_max_steps,
          terminal_reason, final_decision, final_confidence,
          total_input_tokens, total_output_tokens, total_duration_ms, trace_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            payload["trace_id"],
            payload["turn_id"],
            payload["event_id"],
            payload["prompt_version"],
            payload["model_name"],
            payload["started_at"],
            payload["finished_at"],
            payload["total_steps"],
            int(payload["exceeded_max_steps"]),
            payload["terminal_reason"],
            payload["final_decision"],
            payload["final_confidence"],
            payload["total_input_tokens"],
            payload["total_output_tokens"],
            payload["total_duration_ms"],
            json.dumps(payload, ensure_ascii=False),
        ),
    )

