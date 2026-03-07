from __future__ import annotations

from ..config import SnowballConfig
from ..models import ModelResponse, ToolCall, TokenUsage
from ..utils import new_id
from .tools import compose_append_content, compose_archive_payload, compose_atomic_note_content


class HeuristicModelAdapter:
    model_name = "heuristic-v1"
    version = "2026-03-07"

    def __init__(self, config: SnowballConfig):
        self.config = config

    def respond(self, event, state, messages, tools, step_index: int) -> ModelResponse:
        if self._last_result(state, "assess_turn_value") is None:
            return self._call("Assess the turn value", "assess_turn_value", {})

        assess = self._last_result(state, "assess_turn_value") or {}
        decision = assess.get("decision", "archive")
        if decision == "skip":
            return self._end("Skip the turn because it lacks durable knowledge value.")
        if decision == "archive":
            if not self._has_action(state, "archive_turn"):
                payload = compose_archive_payload(event)
                payload["content"] = event.assistant_final_answer
                return self._call("Archive the turn for future replay only.", "propose_archive_turn", payload)
            return self._end("Archive proposal prepared.")

        if self._last_result(state, "extract_knowledge_points") is None:
            return self._call("Extract reusable knowledge from the turn.", "extract_knowledge_points", {})

        extracted = self._last_result(state, "extract_knowledge_points") or {}
        if self._last_result(state, "search_similar_notes") is None:
            return self._call(
                "Search for similar notes before deciding whether to create or append.",
                "search_similar_notes",
                {"query": extracted.get("candidate_title", event.user_message), "top_k": self.config.retrieval.top_k},
            )

        search_results = self._last_result(state, "search_similar_notes") or []
        top_match = search_results[0] if search_results else None

        if top_match and top_match["similarity"] >= self.config.retrieval.append_threshold:
            if self._last_result(state, "read_note") is None:
                return self._call(
                    "Read the strongest candidate note before appending.",
                    "read_note",
                    {"note_id": top_match["note_id"]},
                )

        if not state.proposals:
            if top_match and top_match["similarity"] >= self.config.retrieval.append_threshold:
                if event.source_confidence < self.config.guardrails.min_confidence_for_append:
                    return self._call(
                        "Escalate the ambiguous high-similarity turn for review.",
                        "flag_for_review",
                        {
                            "reason": "high_similarity_low_confidence",
                            "conflict_note_id": top_match["note_id"],
                        },
                    )
                return self._call(
                    "Append to the matching note instead of creating a duplicate.",
                    "propose_append_to_note",
                    {
                        "note_id": top_match["note_id"],
                        "content": compose_append_content(extracted, event),
                    },
                )
            if top_match and top_match["similarity"] >= self.config.retrieval.review_threshold:
                return self._call(
                    "Ask for review because the candidate note is similar but not certain enough for append.",
                    "flag_for_review",
                    {
                        "reason": "ambiguous_existing_note",
                        "conflict_note_id": top_match["note_id"],
                    },
                )
            content = compose_atomic_note_content(extracted, event, related=search_results[:2])
            return self._call(
                "Create a new atomic note.",
                "propose_create_note",
                {
                    "title": extracted.get("candidate_title", event.user_message[:80]),
                    "content": content,
                    "tags": extracted.get("tags", []),
                    "topics": extracted.get("topics", []),
                },
            )
        return self._end("A proposal was already produced; the turn can end.")

    def _call(self, summary: str, tool_name: str, payload: dict) -> ModelResponse:
        token_usage = TokenUsage(input_tokens=60, output_tokens=24)
        return ModelResponse(
            stop_reason="tool_use",
            tool_use_blocks=[ToolCall(call_id=new_id("tool"), name=tool_name, input=payload)],
            decision_summary=summary,
            usage=token_usage,
        )

    def _end(self, summary: str) -> ModelResponse:
        return ModelResponse(
            stop_reason="end_turn",
            decision_summary=summary,
            usage=TokenUsage(input_tokens=32, output_tokens=16),
        )

    def _last_result(self, state, tool_name: str):
        values = state.tool_context.get(tool_name) or []
        return values[-1] if values else None

    def _has_action(self, state, action_type: str) -> bool:
        return any(proposal.action_type == action_type for proposal in state.proposals)

