import tempfile
import unittest
from pathlib import Path

from snowball_notes.agent.context_recovery import (
    estimate_tokens,
    recover_context,
)
from snowball_notes.agent.runtime import SnowballAgent
from snowball_notes.agent.state import AgentState
from snowball_notes.config import default_config
from snowball_notes.models import ActionProposal, SessionMemory, StandardEvent


def _event() -> StandardEvent:
    return StandardEvent(
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


def _state() -> AgentState:
    return AgentState(
        event=_event(),
        task_id="task",
        trace_id="trace",
        session_memory=SessionMemory(conversation_id="c"),
    )


def _assistant(decision: str) -> dict:
    return {
        "role": "assistant",
        "content": {"decision_summary": decision, "stop_reason": "tool_use", "tool_calls": []},
    }


def _tool(call_id: str, name: str, content) -> dict:
    return {"role": "tool", "call_id": call_id, "name": name, "content": content}


def _conversation() -> list[dict]:
    # head + three [assistant, tool] groups with large-ish tool bodies.
    return [
        {"role": "user", "content": {"turn_id": "t", "user_message": "how do I cache prompts?"}},
        _assistant("search the index"),
        _tool("c1", "search_similar_notes", {"matches": [{"note_id": f"n{i}"} for i in range(8)], "body": "x" * 500}),
        _assistant("read the top match"),
        _tool("c2", "read_note", {"note_id": "n1", "content": "y" * 800}),
        _assistant("propose a new note"),
        _tool("c3", "propose_create_note", {"created": True, "note_id": "n_new"}),
    ]


def _config(**overrides):
    with tempfile.TemporaryDirectory() as temp_dir:
        config = default_config(Path(temp_dir))
    config.agent.enable_context_budget = False  # isolate recovery from the budget
    for key, value in overrides.items():
        setattr(config.agent, key, value)
    return config


class EstimateTokensTests(unittest.TestCase):
    def test_estimate_grows_with_content(self):
        small = [{"role": "user", "content": {"a": "x"}}]
        big = [{"role": "user", "content": {"a": "x" * 4000}}]
        self.assertLess(estimate_tokens(small), estimate_tokens(big))
        self.assertGreater(estimate_tokens(big), 900)


class RecoverContextTests(unittest.TestCase):
    def test_under_soft_limit_is_passthrough(self):
        config = _config(compact_token_soft_limit=100000, compact_token_hard_limit=200000)
        state = _state()
        messages = _conversation()
        out = recover_context(messages, state, config=config, step_index=3)
        self.assertEqual(out, messages)
        self.assertEqual(state.recovery_events, [])

    def test_history_compaction_folds_old_keeps_recent(self):
        config = _config(
            compact_token_soft_limit=1,
            compact_token_hard_limit=100000,
            keep_recent_turns=1,
        )
        state = _state()
        state.proposals.append(
            ActionProposal(
                proposal_id="p1",
                trace_id="trace",
                turn_id="t",
                action_type="create_note",
                target_note_id=None,
                payload={},
                idempotency_key="k1",
            )
        )
        out = recover_context(_conversation(), state, config=config, step_index=3)

        # One history_compaction event recorded, and it actually shrank the context.
        self.assertEqual(len(state.recovery_events), 1)
        event = state.recovery_events[0]
        self.assertEqual(event["level"], "history_compaction")
        self.assertLess(event["tokens_after"], event["tokens_before"])

        # Layout: original task, the digest, then the most recent group verbatim.
        self.assertEqual(out[0]["content"]["turn_id"], "t")
        digest = out[1]["content"]["compacted_history"]
        self.assertEqual(digest["summarized_steps"], 2)
        self.assertEqual(digest["proposals_so_far"], [{"action_type": "create_note", "target_note_id": None}])
        # The recent group (assistant + propose_create_note tool) is preserved untouched.
        self.assertEqual(out[-2], _assistant("propose a new note"))
        self.assertEqual(out[-1]["name"], "propose_create_note")
        # Folded search result is summarized to counts, not full body.
        first_step = digest["steps"][0]
        self.assertEqual(first_step["decision"], "search the index")
        self.assertEqual(first_step["tools"][0]["result"]["match_count"], 8)

    def test_full_summarize_collapses_to_task_plus_summary(self):
        config = _config(
            compact_token_soft_limit=1,
            compact_token_hard_limit=1,
            keep_recent_turns=1,
        )
        state = _state()
        out = recover_context(_conversation(), state, config=config, step_index=4)

        levels = [event["level"] for event in state.recovery_events]
        self.assertIn("full_summarize", levels)
        # Collapsed to exactly the original task plus a single summary message.
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["content"]["turn_id"], "t")
        summary = out[1]["content"]["full_conversation_summary"]
        self.assertEqual(summary["summarized_steps"], 3)

    def test_disabled_recovery_is_passthrough(self):
        config = _config(
            enable_context_recovery=False,
            compact_token_soft_limit=1,
            compact_token_hard_limit=1,
        )
        state = _state()
        messages = _conversation()
        out = recover_context(messages, state, config=config, step_index=1)
        self.assertEqual(out, messages)
        self.assertEqual(state.recovery_events, [])


class RuntimeRecoveryWiringTests(unittest.TestCase):
    def _agent(self, **agent_overrides) -> SnowballAgent:
        config = _config(**agent_overrides)
        return SnowballAgent(config, model_adapter=None, tool_registry={}, vault=None, db=None)

    def test_recover_context_records_events_through_runtime(self):
        agent = self._agent(compact_token_soft_limit=1, compact_token_hard_limit=100000, keep_recent_turns=1)
        state = _state()
        out = agent._recover_context(_conversation(), state, step_index=2)
        self.assertTrue(state.recovery_events)
        self.assertLess(len(out), len(_conversation()))


if __name__ == "__main__":
    unittest.main()
