import tempfile
import unittest
from pathlib import Path
from unittest import mock

from snowball_notes.agent.adapter import (
    AnthropicMessagesAdapter,
    DeepSeekChatCompletionsAdapter,
    HeuristicModelAdapter,
    OpenAIResponsesAdapter,
    build_model_adapter,
)
from snowball_notes.agent.state import AgentState
from snowball_notes.config import default_config
from snowball_notes.models import SessionMemory, StandardEvent


def _sample_event() -> StandardEvent:
    return StandardEvent(
        event_id="evt_test",
        session_file="/tmp/session.jsonl",
        conversation_id="conv_test",
        turn_id="turn_test",
        user_message="How should I build a guarded agent runtime?",
        assistant_final_answer="Use proposals, guardrails, and replay bundles.",
        displayed_at="2026-03-08T00:00:00+00:00",
        source_completeness="full",
        source_confidence=0.92,
        parser_version="v1",
        context_meta={},
    )


class StubOpenAIResponsesAdapter(OpenAIResponsesAdapter):
    def __init__(self, config, payloads):
        self._payloads = list(payloads)
        self.calls = []
        super().__init__(config)

    def _request_payload(self, messages, previous_response_id, next_input_items):
        self.calls.append(
            {
                "messages": messages,
                "previous_response_id": previous_response_id,
                "next_input_items": next_input_items,
            }
        )
        return self._payloads.pop(0)


class StubDeepSeekChatCompletionsAdapter(DeepSeekChatCompletionsAdapter):
    def __init__(self, config, payloads):
        self._payloads = list(payloads)
        self.calls = []
        super().__init__(config)

    def _request_payload(self, messages):
        self.calls.append(messages)
        return self._payloads.pop(0)


class StubAnthropicMessagesAdapter(AnthropicMessagesAdapter):
    def __init__(self, config, payloads):
        self._payloads = list(payloads)
        self.bodies = []
        super().__init__(config)

    def _request_payload(self, messages):
        # Capture the exact wire body the real adapter would POST, so tests can
        # assert tool/system shape and prompt-cache placement without a network call.
        rendered_messages = self._anthropic_messages(messages)
        if self.config.agent.enable_prompt_cache:
            self._mark_conversation_cache_breakpoint(rendered_messages)
        body = {
            "model": self.model_name,
            "max_tokens": self.config.agent.max_output_tokens,
            "tools": self._anthropic_tool_definitions(),
            "messages": rendered_messages,
        }
        system_blocks = self._system_blocks()
        if system_blocks:
            body["system"] = system_blocks
        if self.config.agent.thinking == "adaptive":
            body["thinking"] = {"type": "adaptive"}
        self.bodies.append(body)
        return self._payloads.pop(0)


class OpenAIAdapterTests(unittest.TestCase):
    def test_responses_adapter_parses_function_call_and_followup(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            config = default_config(Path(temp_dir))
            config.agent.provider = "openai_responses"
            config.agent.model = "gpt-5.2-codex"
            adapter = StubOpenAIResponsesAdapter(
                config,
                [
                    {
                        "id": "resp_1",
                        "usage": {"input_tokens": 10, "output_tokens": 5},
                        "output": [
                            {
                                "type": "function_call",
                                "call_id": "call_1",
                                "name": "search_similar_notes",
                                "arguments": "{\"query\": \"agent runtime\", \"top_k\": 3}",
                            }
                        ],
                    },
                    {
                        "id": "resp_2",
                        "usage": {"input_tokens": 6, "output_tokens": 4},
                        "output_text": "The turn can end safely.",
                        "output": [],
                    },
                ],
            )
            state = AgentState(
                event=_sample_event(),
                task_id="task_1",
                trace_id="trace_1",
                session_memory=SessionMemory(conversation_id="conv_test"),
            )
            messages = [
                {
                    "role": "user",
                    "content": {
                        "turn_id": "turn_test",
                        "user_message": "How should I build a guarded agent runtime?",
                        "assistant_final_answer": "Use proposals, guardrails, and replay bundles.",
                        "source_confidence": 0.92,
                        "previous_turns": 0,
                        "session_context": "No prior turns from this conversation have been processed yet.",
                        "recent_actions": [],
                    },
                }
            ]

            first = adapter.respond(state.event, state, messages, {}, 0)
            self.assertEqual(first.stop_reason, "tool_use")
            self.assertEqual(first.provider_response_id, "resp_1")
            self.assertEqual(first.tool_use_blocks[0].name, "search_similar_notes")
            self.assertEqual(first.tool_use_blocks[0].input["top_k"], 3)

            state.model_context["previous_response_id"] = first.provider_response_id
            state.model_context["next_input_items"] = [
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "{\"results\": []}",
                }
            ]
            second = adapter.respond(state.event, state, messages, {}, 1)
            self.assertEqual(second.stop_reason, "end_turn")
            self.assertEqual(second.provider_response_id, "resp_2")
            self.assertEqual(adapter.calls[1]["previous_response_id"], "resp_1")
            self.assertEqual(adapter.calls[1]["next_input_items"][0]["call_id"], "call_1")

    def test_provider_selection_prefers_explicit_provider_over_default_model(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            config = default_config(Path(temp_dir))
            config.agent.provider = "deepseek_v3"
            adapter = build_model_adapter(config)
            self.assertIsInstance(adapter, DeepSeekChatCompletionsAdapter)
            self.assertNotIsInstance(adapter, HeuristicModelAdapter)
            self.assertEqual(adapter.model_name, "deepseek-chat")

    def test_deepseek_adapter_parses_tool_calls_and_replays_message_history(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            config = default_config(Path(temp_dir))
            config.agent.provider = "deepseek_v3"
            adapter = StubDeepSeekChatCompletionsAdapter(
                config,
                [
                    {
                        "id": "chatcmpl_1",
                        "usage": {"prompt_tokens": 12, "completion_tokens": 7},
                        "choices": [
                            {
                                "finish_reason": "tool_calls",
                                "message": {
                                    "content": "Read the candidate note first.",
                                    "tool_calls": [
                                        {
                                            "id": "call_1",
                                            "type": "function",
                                            "function": {
                                                "name": "read_note",
                                                "arguments": "{\"note_id\": \"note_123\"}",
                                            },
                                        }
                                    ],
                                },
                            }
                        ],
                    },
                    {
                        "id": "chatcmpl_2",
                        "usage": {"prompt_tokens": 18, "completion_tokens": 5},
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {
                                    "content": "The turn can end safely.",
                                },
                            }
                        ],
                    },
                ],
            )
            state = AgentState(
                event=_sample_event(),
                task_id="task_1",
                trace_id="trace_1",
                session_memory=SessionMemory(conversation_id="conv_test"),
            )
            messages = [
                {
                    "role": "user",
                    "content": {
                        "turn_id": "turn_test",
                        "user_message": "How should I build a guarded agent runtime?",
                        "assistant_final_answer": "Use proposals, guardrails, and replay bundles.",
                        "source_confidence": 0.92,
                        "previous_turns": 0,
                        "session_context": "No prior turns from this conversation have been processed yet.",
                        "recent_actions": [],
                    },
                }
            ]

            first = adapter.respond(state.event, state, messages, {}, 0)
            self.assertEqual(first.stop_reason, "tool_use")
            self.assertEqual(first.provider_response_id, "chatcmpl_1")
            self.assertEqual(first.tool_use_blocks[0].name, "read_note")
            self.assertEqual(first.tool_use_blocks[0].input["note_id"], "note_123")

            followup_messages = messages + [
                {
                    "role": "assistant",
                    "content": {
                        "decision_summary": first.decision_summary,
                        "stop_reason": first.stop_reason,
                        "tool_calls": [
                            {
                                "call_id": first.tool_use_blocks[0].call_id,
                                "name": first.tool_use_blocks[0].name,
                                "input": first.tool_use_blocks[0].input,
                            }
                        ],
                    },
                },
                {
                    "role": "tool",
                    "call_id": first.tool_use_blocks[0].call_id,
                    "name": first.tool_use_blocks[0].name,
                    "content": {"note_id": "note_123", "content": "## Summary\nExisting note."},
                },
            ]
            second = adapter.respond(state.event, state, followup_messages, {}, 1)
            self.assertEqual(second.stop_reason, "end_turn")
            self.assertEqual(second.provider_response_id, "chatcmpl_2")

            rendered_messages = adapter._chat_messages(followup_messages)
            self.assertEqual(rendered_messages[1]["role"], "user")
            self.assertEqual(rendered_messages[2]["role"], "assistant")
            self.assertEqual(rendered_messages[2]["tool_calls"][0]["id"], "call_1")
            self.assertEqual(rendered_messages[3]["role"], "tool")
            self.assertEqual(rendered_messages[3]["tool_call_id"], "call_1")


class AnthropicAdapterTests(unittest.TestCase):
    def test_messages_adapter_parses_tool_use_and_caches_system(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            config = default_config(Path(temp_dir))
            config.agent.provider = "anthropic"
            adapter = StubAnthropicMessagesAdapter(
                config,
                [
                    {
                        "id": "msg_1",
                        "stop_reason": "tool_use",
                        "usage": {"input_tokens": 20, "output_tokens": 8, "cache_read_input_tokens": 0},
                        "content": [
                            {"type": "text", "text": "Search before deciding."},
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "search_similar_notes",
                                "input": {"query": "agent runtime", "top_k": 3},
                            },
                        ],
                    },
                ],
            )
            state = AgentState(
                event=_sample_event(),
                task_id="task_1",
                trace_id="trace_1",
                session_memory=SessionMemory(conversation_id="conv_test"),
            )
            messages = [
                {
                    "role": "user",
                    "content": {
                        "turn_id": "turn_test",
                        "user_message": "How should I build a guarded agent runtime?",
                        "assistant_final_answer": "Use proposals, guardrails, and replay bundles.",
                        "source_confidence": 0.92,
                        "previous_turns": 0,
                        "session_context": "No prior turns from this conversation have been processed yet.",
                        "recent_actions": [],
                    },
                }
            ]

            response = adapter.respond(state.event, state, messages, {}, 0)
            self.assertEqual(response.stop_reason, "tool_use")
            self.assertEqual(response.provider_response_id, "msg_1")
            self.assertEqual(response.tool_use_blocks[0].name, "search_similar_notes")
            self.assertEqual(response.tool_use_blocks[0].input["top_k"], 3)
            self.assertEqual(response.decision_summary, "Search before deciding.")
            self.assertEqual(response.usage.input_tokens, 20)

            body = adapter.bodies[0]
            self.assertEqual(body["model"], "claude-opus-4-8")
            # System prompt carries a cache breakpoint (tools render before it, so
            # tools + system cache together).
            self.assertEqual(body["system"][0]["cache_control"], {"type": "ephemeral"})
            # Tools use Anthropic's input_schema shape, not OpenAI's parameters.
            self.assertIn("input_schema", body["tools"][0])

    def test_messages_adapter_rebuilds_tool_result_turns(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            config = default_config(Path(temp_dir))
            config.agent.provider = "claude"
            adapter = StubAnthropicMessagesAdapter(
                config,
                [
                    {
                        "id": "msg_2",
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 30, "output_tokens": 5},
                        "content": [{"type": "text", "text": "The turn can end safely."}],
                    }
                ],
            )
            state = AgentState(
                event=_sample_event(),
                task_id="task_1",
                trace_id="trace_1",
                session_memory=SessionMemory(conversation_id="conv_test"),
            )
            messages = [
                {"role": "user", "content": {"turn_id": "turn_test", "user_message": "u", "assistant_final_answer": "a"}},
                {
                    "role": "assistant",
                    "content": {
                        "decision_summary": "Read the candidate note.",
                        "stop_reason": "tool_use",
                        "tool_calls": [{"call_id": "toolu_1", "name": "read_note", "input": {"note_id": "note_123"}}],
                    },
                },
                {"role": "tool", "call_id": "toolu_1", "name": "read_note", "content": {"note_id": "note_123"}},
            ]

            response = adapter.respond(state.event, state, messages, {}, 1)
            self.assertEqual(response.stop_reason, "end_turn")

            rendered = adapter.bodies[0]["messages"]
            self.assertEqual(rendered[0]["role"], "user")
            self.assertEqual(rendered[1]["role"], "assistant")
            tool_use_block = rendered[1]["content"][-1]
            self.assertEqual(tool_use_block["type"], "tool_use")
            self.assertEqual(tool_use_block["id"], "toolu_1")
            # Tool output becomes a tool_result block inside a following user turn.
            self.assertEqual(rendered[2]["role"], "user")
            self.assertEqual(rendered[2]["content"][0]["type"], "tool_result")
            self.assertEqual(rendered[2]["content"][0]["tool_use_id"], "toolu_1")

    def _multiturn_messages(self):
        return [
            {"role": "user", "content": {"turn_id": "t", "user_message": "u", "assistant_final_answer": "a"}},
            {
                "role": "assistant",
                "content": {
                    "decision_summary": "Read the note.",
                    "stop_reason": "tool_use",
                    "tool_calls": [{"call_id": "toolu_1", "name": "read_note", "input": {"note_id": "n1"}}],
                },
            },
            {"role": "tool", "call_id": "toolu_1", "name": "read_note", "content": {"note_id": "n1"}},
        ]

    def test_conversation_cache_breakpoint_on_last_turn(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            config = default_config(Path(temp_dir))
            config.agent.provider = "anthropic"
            adapter = StubAnthropicMessagesAdapter(
                config,
                [{"id": "m", "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}, "content": []}],
            )
            state = AgentState(
                event=_sample_event(),
                task_id="task_1",
                trace_id="trace_1",
                session_memory=SessionMemory(conversation_id="conv_test"),
            )
            adapter.respond(state.event, state, self._multiturn_messages(), {}, 1)
            sent = adapter.bodies[0]["messages"]
            # Last turn's last block carries the conversation breakpoint...
            self.assertEqual(sent[-1]["content"][-1]["cache_control"], {"type": "ephemeral"})
            # ...and earlier turns do not (one breakpoint walks the prefix).
            self.assertNotIn("cache_control", sent[0]["content"][-1])

    def test_no_conversation_breakpoint_when_cache_disabled(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            config = default_config(Path(temp_dir))
            config.agent.provider = "anthropic"
            config.agent.enable_prompt_cache = False
            adapter = StubAnthropicMessagesAdapter(
                config,
                [{"id": "m", "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}, "content": []}],
            )
            state = AgentState(
                event=_sample_event(),
                task_id="task_1",
                trace_id="trace_1",
                session_memory=SessionMemory(conversation_id="conv_test"),
            )
            adapter.respond(state.event, state, self._multiturn_messages(), {}, 1)
            body = adapter.bodies[0]
            # System prompt is still sent, but without a cache breakpoint.
            self.assertNotIn("cache_control", body["system"][0])
            for message in body["messages"]:
                for block in message["content"]:
                    self.assertNotIn("cache_control", block)

    def test_build_model_adapter_selects_anthropic(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            config = default_config(Path(temp_dir))
            config.agent.provider = "anthropic"
            adapter = build_model_adapter(config)
            self.assertIsInstance(adapter, AnthropicMessagesAdapter)
            self.assertEqual(adapter.model_name, "claude-opus-4-8")
            self.assertEqual(adapter.api_base_url, "https://api.anthropic.com/v1/messages")


if __name__ == "__main__":
    unittest.main()
