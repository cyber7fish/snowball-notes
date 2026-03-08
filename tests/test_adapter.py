import tempfile
import unittest
from pathlib import Path
from unittest import mock

from snowball_notes.agent.adapter import (
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


if __name__ == "__main__":
    unittest.main()
