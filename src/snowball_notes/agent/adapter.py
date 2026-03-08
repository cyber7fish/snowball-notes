from __future__ import annotations

import json
import os
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib import error, request

from ..config import SnowballConfig
from ..models import ModelResponse, ToolCall, TokenUsage
from ..utils import new_id, normalize_text
from .tools import TOOL_SCHEMAS, compose_append_content, compose_archive_payload, compose_atomic_note_content


class ModelRetryableError(RuntimeError):
    pass


class ModelFatalError(RuntimeError):
    pass


TOOL_DESCRIPTIONS = {
    "assess_turn_value": "Assess whether the turn should be skipped, archived, or turned into durable notes.",
    "extract_knowledge_points": "Extract reusable knowledge points, a candidate title, and tags from the turn.",
    "search_similar_notes": "Search existing notes before deciding whether to create or append.",
    "read_note": "Read an existing note before appending or flagging.",
    "propose_create_note": "Propose creating a new note without writing any side effects yet.",
    "propose_append_to_note": "Propose appending new knowledge into an existing note without writing yet.",
    "propose_archive_turn": "Propose archiving the turn as a conversation record.",
    "propose_link_notes": "Propose adding a bidirectional relationship between two existing notes.",
    "flag_for_review": "Escalate the turn for human review and terminate the current run.",
}

DEFAULT_OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"
DEFAULT_DEEPSEEK_CHAT_COMPLETIONS_URL = "https://api.deepseek.com/chat/completions"


class HeuristicModelAdapter:
    model_name = "heuristic-v1"
    version = "2026-03-07"

    def __init__(self, config: SnowballConfig):
        self.config = config

    def respond(self, event, state, messages, tools, step_index: int) -> ModelResponse:
        recent_actions = self._recent_actions(messages, state)
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
        if self._duplicate_created_title(recent_actions, extracted.get("candidate_title", "")):
            return self._end("Skip because this conversation already created a highly similar note.")
        if self._last_result(state, "search_similar_notes") is None:
            return self._call(
                "Search for similar notes before deciding whether to create or append.",
                "search_similar_notes",
                {"query": extracted.get("candidate_title", event.user_message), "top_k": self.config.retrieval.top_k},
            )

        search_results = self._last_result(state, "search_similar_notes") or []
        top_match = search_results[0] if search_results else None

        if (
            not state.proposals
            and len(search_results) >= 2
            and event.source_confidence >= self.config.guardrails.min_confidence_for_append
            and self._should_link_notes(event, extracted, search_results[:2])
        ):
            primary = search_results[0]
            secondary = search_results[1]
            return self._call(
                "Link two existing notes because the turn is explicitly describing their relationship.",
                "propose_link_notes",
                {
                    "source_note_id": primary["note_id"],
                    "target_note_id": secondary["note_id"],
                },
            )

        if top_match and top_match["similarity"] >= self.config.retrieval.append_threshold:
            if self._note_already_touched(recent_actions, top_match["note_id"]):
                return self._end("Skip because this conversation already wrote to the matching note.")
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
                            "suggested_action": "append_note",
                            "suggested_payload": {
                                "note_id": top_match["note_id"],
                                "content": compose_append_content(extracted, event),
                                "source_turn_id": event.turn_id,
                                "source_event_id": event.event_id,
                            },
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
                        "suggested_action": "append_note",
                        "suggested_payload": {
                            "note_id": top_match["note_id"],
                            "content": compose_append_content(extracted, event),
                            "source_turn_id": event.turn_id,
                            "source_event_id": event.event_id,
                        },
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

    def _recent_actions(self, messages, state) -> list[dict]:
        if messages:
            content = messages[0].get("content") or {}
            recent_actions = content.get("recent_actions")
            if isinstance(recent_actions, list):
                return [item for item in recent_actions if isinstance(item, dict)]
        actions = []
        for turn in state.session_memory.processed_turns:
            actions.append(
                {
                    "turn_id": turn.turn_id,
                    "final_decision": turn.final_decision,
                    "note_id": turn.note_id,
                    "action_type": turn.action_type,
                    "note_title": turn.note_title,
                }
            )
        return actions

    def _note_already_touched(self, recent_actions: list[dict], note_id: str) -> bool:
        for action in recent_actions:
            if action.get("note_id") != note_id:
                continue
            if action.get("action_type") in {"create_note", "append_note"}:
                return True
        return False

    def _duplicate_created_title(self, recent_actions: list[dict], title: str) -> bool:
        title_norm = normalize_text(title)
        if not title_norm:
            return False
        for action in recent_actions:
            if action.get("action_type") != "create_note":
                continue
            note_title = normalize_text(action.get("note_title") or "")
            if not note_title:
                continue
            if title_norm == note_title:
                return True
            if SequenceMatcher(None, title_norm, note_title).ratio() >= 0.94:
                return True
        return False

    def _should_link_notes(
        self,
        event,
        extracted: dict[str, Any],
        search_results: list[dict[str, Any]],
    ) -> bool:
        if len(search_results) < 2:
            return False
        first, second = search_results[0], search_results[1]
        if first.get("note_id") == second.get("note_id"):
            return False
        combined = " ".join(
            [
                str(event.user_message or ""),
                str(event.assistant_final_answer or ""),
                str(extracted.get("summary") or ""),
            ]
        )
        combined_norm = normalize_text(combined)
        first_title_norm = normalize_text(str(first.get("title") or ""))
        second_title_norm = normalize_text(str(second.get("title") or ""))
        explicit_title_match = (
            bool(first_title_norm)
            and bool(second_title_norm)
            and first_title_norm in combined_norm
            and second_title_norm in combined_norm
        )
        first_similarity = float(first.get("similarity") or 0.0)
        second_similarity = float(second.get("similarity") or 0.0)
        if (
            not explicit_title_match
            and min(first_similarity, second_similarity) < max(self.config.retrieval.review_threshold - 0.1, 0.45)
        ):
            return False
        combined = combined.lower()
        link_terms = [
            "link",
            "linked",
            "connect",
            "connection",
            "related",
            "relationship",
            "cross-reference",
            "associate",
            "关联",
            "链接",
            "连接",
            "关系",
        ]
        return any(term in combined for term in link_terms)


class _ToolAwareAdapter:
    def _load_prompt(self) -> str:
        prompt_path = Path(__file__).resolve().parents[1] / "prompts" / self.config.agent.prompt_version
        if not prompt_path.exists():
            return ""
        return prompt_path.read_text(encoding="utf-8")

    def _responses_tool_definitions(self) -> list[dict[str, Any]]:
        definitions = []
        for name, schema in TOOL_SCHEMAS.items():
            properties = {}
            for field_name, field_type in schema.get("types", {}).items():
                properties[field_name] = {"type": _json_schema_type(field_type)}
            definitions.append(
                {
                    "type": "function",
                    "name": name,
                    "description": TOOL_DESCRIPTIONS.get(name, name.replace("_", " ")),
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": schema.get("required", []),
                        "additionalProperties": False,
                    },
                    "strict": False,
                }
            )
        return definitions

    def _chat_tool_definitions(self) -> list[dict[str, Any]]:
        definitions = []
        for item in self._responses_tool_definitions():
            definitions.append(
                {
                    "type": "function",
                    "function": {
                        "name": item["name"],
                        "description": item["description"],
                        "parameters": item["parameters"],
                        "strict": item["strict"],
                    },
                }
            )
        return definitions

    def _render_initial_turn(self, messages: list[dict[str, Any]]) -> str:
        if not messages:
            return ""
        content = messages[0].get("content")
        if isinstance(content, str):
            return content
        if not isinstance(content, dict):
            return json.dumps(content, ensure_ascii=False)
        lines = [
            "## Current Turn",
            f"turn_id: {content.get('turn_id', '')}",
            "",
            "### User Message",
            str(content.get("user_message", "")).strip(),
            "",
            "### Assistant Final Answer",
            str(content.get("assistant_final_answer", "")).strip(),
            "",
            f"source_confidence: {content.get('source_confidence', '')}",
            "",
            "## Session Context",
            str(content.get("session_context", "")).strip(),
        ]
        recent_actions = content.get("recent_actions") or []
        if recent_actions:
            lines.extend(["", "## Recent Actions", json.dumps(recent_actions, ensure_ascii=False, indent=2)])
        return "\n".join(lines).strip()


class OpenAIResponsesAdapter(_ToolAwareAdapter):
    version = "responses-v1"

    def __init__(self, config: SnowballConfig):
        self.config = config
        self.model_name = config.agent.model
        api_key = os.environ.get(config.agent.api_key_env)
        if not api_key:
            raise RuntimeError(f"missing API key env: {config.agent.api_key_env}")
        self.api_key = api_key

    def respond(self, event, state, messages, tools, step_index: int) -> ModelResponse:
        previous_response_id = state.model_context.get("previous_response_id")
        next_input_items = state.model_context.get("next_input_items")
        payload = self._request_payload(messages, previous_response_id, next_input_items)
        function_calls = []
        for item in payload.get("output", []):
            if item.get("type") != "function_call":
                continue
            arguments = item.get("arguments") or "{}"
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ModelFatalError(f"invalid function arguments from model: {exc}") from exc
            function_calls.append(
                ToolCall(
                    call_id=item["call_id"],
                    name=item["name"],
                    input=parsed,
                )
            )
        usage_payload = payload.get("usage") or {}
        usage = TokenUsage(
            input_tokens=int(usage_payload.get("input_tokens") or 0),
            output_tokens=int(usage_payload.get("output_tokens") or 0),
        )
        decision_summary = self._decision_summary(payload, function_calls)
        stop_reason = "tool_use" if function_calls else "end_turn"
        return ModelResponse(
            stop_reason=stop_reason,
            tool_use_blocks=function_calls,
            decision_summary=decision_summary,
            usage=usage,
            provider_response_id=payload.get("id"),
        )

    def _request_payload(
        self,
        messages: list[dict[str, Any]],
        previous_response_id: str | None,
        next_input_items: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model_name,
            "instructions": self._load_prompt(),
            "tools": self._responses_tool_definitions(),
            "parallel_tool_calls": True,
            "store": True,
        }
        if self.config.agent.reasoning_effort:
            body["reasoning"] = {"effort": self.config.agent.reasoning_effort}
        if previous_response_id and next_input_items:
            body["previous_response_id"] = previous_response_id
            body["input"] = next_input_items
        else:
            body["input"] = [
                {
                    "role": "user",
                    "content": self._render_initial_turn(messages),
                }
            ]
        raw_body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            self.config.agent.api_base_url,
            data=raw_body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.config.agent.request_timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            message = f"{exc.code} {detail}".strip()
            if exc.code in {408, 409, 429, 500, 502, 503, 504}:
                raise ModelRetryableError(message) from exc
            raise ModelFatalError(message) from exc
        except error.URLError as exc:
            raise ModelRetryableError(str(exc)) from exc

    def _decision_summary(self, payload: dict[str, Any], function_calls: list[ToolCall]) -> str:
        output_text = payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()
        for item in payload.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                text = content.get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()
        if function_calls:
            return ", ".join(f"{tool.name}()" for tool in function_calls)
        return "Model ended the turn."


class DeepSeekChatCompletionsAdapter(_ToolAwareAdapter):
    version = "deepseek-chat-v1"

    def __init__(self, config: SnowballConfig):
        self.config = config
        self.model_name = config.agent.model if config.agent.model != "heuristic-v1" else "deepseek-chat"
        api_key_env = config.agent.api_key_env
        if api_key_env == DEFAULT_OPENAI_API_KEY_ENV:
            api_key_env = DEFAULT_DEEPSEEK_API_KEY_ENV
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"missing API key env: {api_key_env}")
        self.api_key = api_key
        self.api_base_url = config.agent.api_base_url
        if self.api_base_url == DEFAULT_OPENAI_RESPONSES_URL:
            self.api_base_url = DEFAULT_DEEPSEEK_CHAT_COMPLETIONS_URL

    def respond(self, event, state, messages, tools, step_index: int) -> ModelResponse:
        payload = self._request_payload(messages)
        choices = payload.get("choices") or []
        message = choices[0].get("message") if choices else {}
        if not isinstance(message, dict):
            message = {}
        function_calls = []
        for item in message.get("tool_calls") or []:
            function_payload = item.get("function") or {}
            arguments = function_payload.get("arguments") or "{}"
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ModelFatalError(f"invalid function arguments from model: {exc}") from exc
            function_calls.append(
                ToolCall(
                    call_id=item["id"],
                    name=function_payload["name"],
                    input=parsed,
                )
            )
        usage_payload = payload.get("usage") or {}
        usage = TokenUsage(
            input_tokens=int(usage_payload.get("prompt_tokens") or 0),
            output_tokens=int(usage_payload.get("completion_tokens") or 0),
        )
        stop_reason = "tool_use" if function_calls else "end_turn"
        return ModelResponse(
            stop_reason=stop_reason,
            tool_use_blocks=function_calls,
            decision_summary=self._decision_summary(message, function_calls),
            usage=usage,
            provider_response_id=payload.get("id"),
        )

    def _request_payload(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        body = {
            "model": self.model_name,
            "messages": self._chat_messages(messages),
            "tools": self._chat_tool_definitions(),
            "tool_choice": "auto",
            "stream": False,
        }
        raw_body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            self.api_base_url,
            data=raw_body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.config.agent.request_timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            message = f"{exc.code} {detail}".strip()
            if exc.code in {408, 409, 429, 500, 502, 503, 504}:
                raise ModelRetryableError(message) from exc
            raise ModelFatalError(message) from exc
        except error.URLError as exc:
            raise ModelRetryableError(str(exc)) from exc

    def _chat_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rendered: list[dict[str, Any]] = []
        prompt = self._load_prompt()
        if prompt:
            rendered.append({"role": "system", "content": prompt})
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role == "user":
                rendered.append({"role": "user", "content": self._render_initial_turn([message])})
                continue
            if role == "assistant":
                rendered.append(self._assistant_chat_message(content))
                continue
            if role == "tool":
                call_id = message.get("call_id")
                if not call_id:
                    continue
                rendered.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": self._stringify_chat_content(content),
                    }
                )
        return rendered

    def _assistant_chat_message(self, content: Any) -> dict[str, Any]:
        if isinstance(content, str):
            return {"role": "assistant", "content": content}
        if not isinstance(content, dict):
            return {"role": "assistant", "content": self._stringify_chat_content(content)}
        tool_calls = []
        for tool_call in content.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            tool_calls.append(
                {
                    "id": tool_call.get("call_id") or new_id("toolcall"),
                    "type": "function",
                    "function": {
                        "name": tool_call.get("name", ""),
                        "arguments": json.dumps(tool_call.get("input") or {}, ensure_ascii=False),
                    },
                }
            )
        message = {
            "role": "assistant",
            "content": str(content.get("decision_summary") or "") or None,
        }
        if tool_calls:
            message["tool_calls"] = tool_calls
        return message

    def _stringify_chat_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False)

    def _decision_summary(self, message: dict[str, Any], function_calls: list[ToolCall]) -> str:
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if function_calls:
            return ", ".join(f"{tool.name}()" for tool in function_calls)
        return "Model ended the turn."


def build_model_adapter(config: SnowballConfig):
    if config.agent.provider == "heuristic":
        return HeuristicModelAdapter(config)
    if config.agent.provider == "openai_responses":
        return OpenAIResponsesAdapter(config)
    if config.agent.provider in {"deepseek_v3", "deepseek_chat"}:
        return DeepSeekChatCompletionsAdapter(config)
    if config.agent.model == "heuristic-v1":
        return HeuristicModelAdapter(config)
    raise RuntimeError(f"unsupported agent.provider: {config.agent.provider}")


def _json_schema_type(field_type) -> str:
    if field_type is str:
        return "string"
    if field_type is int:
        return "integer"
    if field_type is float:
        return "number"
    if field_type is bool:
        return "boolean"
    if field_type is list:
        return "array"
    if field_type is dict:
        return "object"
    return "string"
