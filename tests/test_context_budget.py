import tempfile
import unittest
from pathlib import Path

from snowball_notes.agent.context_budget import (
    ContentReplacementState,
    apply_tool_result_budget,
)
from snowball_notes.agent.runtime import SnowballAgent
from snowball_notes.agent.state import AgentState
from snowball_notes.config import default_config
from snowball_notes.models import SessionMemory, StandardEvent


def _tool(call_id: str, name: str, size: int) -> dict:
    return {"role": "tool", "call_id": call_id, "name": name, "content": {"body": "x" * size}}


def _placeholder(name: str) -> str:
    return f"[{name} result cleared to fit context budget]"


class ContextBudgetTests(unittest.TestCase):
    def test_under_budget_clears_nothing(self):
        state = ContentReplacementState()
        messages = [
            {"role": "user", "content": {"turn_id": "t"}},
            _tool("c1", "search_similar_notes", 100),
            _tool("c2", "read_note", 100),
        ]
        out = apply_tool_result_budget(messages, state, max_chars=10000, keep_recent=0)
        self.assertEqual(out, messages)
        self.assertFalse(state.was_applied())
        self.assertEqual(state.chars_cleared, 0)

    def test_clears_largest_first_until_under_budget(self):
        state = ContentReplacementState()
        messages = [
            _tool("small", "search_similar_notes", 100),
            _tool("big", "read_note", 5000),
            _tool("medium", "search_similar_notes", 2000),
        ]
        out = apply_tool_result_budget(messages, state, max_chars=3000, keep_recent=0)
        # Only the largest (big) needed clearing to fit.
        self.assertIn("big", state.cleared_call_ids)
        self.assertNotIn("small", state.cleared_call_ids)
        self.assertNotIn("medium", state.cleared_call_ids)
        cleared = next(m for m in out if m["call_id"] == "big")
        self.assertEqual(cleared["content"], _placeholder("read_note"))
        # Untouched ones keep their original dict content.
        self.assertEqual(next(m for m in out if m["call_id"] == "small")["content"], {"body": "x" * 100})
        self.assertGreater(state.chars_cleared, 0)

    def test_keep_recent_protects_latest_eligible(self):
        state = ContentReplacementState()
        messages = [
            _tool("old", "read_note", 5000),
            _tool("recent", "read_note", 5000),
        ]
        apply_tool_result_budget(messages, state, max_chars=3000, keep_recent=1)
        # The most recent eligible result is protected; clearing falls on "old".
        self.assertIn("old", state.cleared_call_ids)
        self.assertNotIn("recent", state.cleared_call_ids)

    def test_only_compactable_tools_are_cleared(self):
        state = ContentReplacementState()
        messages = [
            _tool("assess", "assess_turn_value", 9000),
            _tool("create", "propose_create_note", 9000),
        ]
        out = apply_tool_result_budget(messages, state, max_chars=10, keep_recent=0)
        # Decision/action tool results are load-bearing and never cleared.
        self.assertEqual(state.cleared_call_ids, set())
        self.assertEqual(out, messages)

    def test_decisions_are_frozen_and_idempotent(self):
        state = ContentReplacementState()
        messages = [
            _tool("a", "search_similar_notes", 5000),
            _tool("b", "search_similar_notes", 100),
        ]
        first = apply_tool_result_budget(messages, state, max_chars=3000, keep_recent=0)
        chars_after_first = state.chars_cleared
        cleared_after_first = set(state.cleared_call_ids)
        # Re-applying with the same state + prefix must be byte-identical (the
        # invariant that keeps the prompt cache valid across steps) and must not
        # double-count cleared characters.
        second = apply_tool_result_budget(messages, state, max_chars=3000, keep_recent=0)
        self.assertEqual(first, second)
        self.assertEqual(state.cleared_call_ids, cleared_after_first)
        self.assertEqual(state.chars_cleared, chars_after_first)

    def test_cleared_result_stays_cleared_when_new_results_arrive(self):
        state = ContentReplacementState()
        messages = [_tool("a", "read_note", 5000)]
        apply_tool_result_budget(messages, state, max_chars=3000, keep_recent=0)
        self.assertIn("a", state.cleared_call_ids)
        # A later step appends a fresh large result; "a" must remain cleared.
        messages = messages + [_tool("b", "read_note", 5000)]
        out = apply_tool_result_budget(messages, state, max_chars=3000, keep_recent=0)
        self.assertIn("a", state.cleared_call_ids)
        self.assertEqual(next(m for m in out if m["call_id"] == "a")["content"], _placeholder("read_note"))


class RuntimeBudgetWiringTests(unittest.TestCase):
    """The runtime's _budget_messages must honor config and the shared state."""

    def _agent(self, **agent_overrides) -> SnowballAgent:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = default_config(Path(temp_dir))
        for key, value in agent_overrides.items():
            setattr(config.agent, key, value)
        # The budget path reads only config + state; db/vault/adapter are unused.
        return SnowballAgent(config, model_adapter=None, tool_registry={}, vault=None, db=None)

    def _state(self) -> AgentState:
        event = StandardEvent(
            event_id="e",
            session_file="/tmp/s.jsonl",
            conversation_id="c",
            turn_id="t",
            user_message="u",
            assistant_final_answer="a",
            displayed_at="2026-03-08T00:00:00+00:00",
            source_completeness="full",
            source_confidence=0.9,
            parser_version="v1",
            context_meta={},
        )
        return AgentState(event=event, task_id="task", trace_id="trace", session_memory=SessionMemory(conversation_id="c"))

    def _messages(self):
        return [
            {"role": "user", "content": {"turn_id": "t"}},
            {"role": "tool", "call_id": "c1", "name": "read_note", "content": {"body": "x" * 5000}},
        ]

    def test_enabled_budget_clears_through_runtime(self):
        agent = self._agent(enable_context_budget=True, max_tool_result_chars=50, keep_recent_tool_results=0)
        state = self._state()
        out = agent._budget_messages(self._messages(), state)
        self.assertTrue(state.replacement_state.was_applied())
        self.assertEqual(out[1]["content"], "[read_note result cleared to fit context budget]")
        self.assertGreater(state.replacement_state.chars_cleared, 0)

    def test_disabled_budget_is_passthrough(self):
        agent = self._agent(enable_context_budget=False, max_tool_result_chars=50)
        state = self._state()
        messages = self._messages()
        out = agent._budget_messages(messages, state)
        self.assertEqual(out, messages)
        self.assertFalse(state.replacement_state.was_applied())


if __name__ == "__main__":
    unittest.main()
