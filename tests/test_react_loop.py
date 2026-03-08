import tempfile
import unittest
from pathlib import Path

from snowball_notes.agent.memory import SQLiteKnowledgeIndex
from snowball_notes.agent.state import AgentState
from snowball_notes.agent.runtime import SnowballAgent
from snowball_notes.agent.tools import AssessTurnValueTool, build_tool_registry
from snowball_notes.config import default_config
from snowball_notes.models import ModelResponse, RunState, SessionMemory, StandardEvent, TaskRecord, TokenUsage, ToolCall
from snowball_notes.storage.sqlite import Database
from snowball_notes.storage.vault import Vault


class ScriptedAdapter:
    model_name = "scripted-test"
    version = "test"

    def __init__(self):
        self.message_counts = []

    def respond(self, event, state, messages, tools, step_index: int) -> ModelResponse:
        self.message_counts.append(len(messages))
        if step_index == 0:
            return ModelResponse(
                stop_reason="tool_use",
                tool_use_blocks=[ToolCall(call_id="call_1", name="assess_turn_value", input={})],
                decision_summary="Assess the turn.",
                usage=TokenUsage(input_tokens=1, output_tokens=1),
            )
        if messages[-1]["role"] != "tool":
            raise AssertionError("expected the previous tool result to be appended into messages")
        return ModelResponse(
            stop_reason="end_turn",
            decision_summary="End the turn after observing the tool result.",
            usage=TokenUsage(input_tokens=1, output_tokens=1),
        )


def _event() -> StandardEvent:
    return StandardEvent(
        event_id="evt_runtime",
        session_file="/tmp/session.jsonl",
        conversation_id="conv_runtime",
        turn_id="turn_runtime",
        user_message="Should I keep this turn?",
        assistant_final_answer="This looks like a lightweight status update.",
        displayed_at="2026-03-08T00:00:00+00:00",
        source_completeness="full",
        source_confidence=0.92,
        parser_version="v1",
        context_meta={},
    )


class ReactLoopTests(unittest.TestCase):
    def test_runtime_advances_messages_with_tool_observations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = default_config(root)
            db = Database(root / "snowball.db")
            db.migrate()
            vault = Vault(config)
            adapter = ScriptedAdapter()
            tools = build_tool_registry(db, SQLiteKnowledgeIndex(db))
            agent = SnowballAgent(config, adapter, tools, vault, db)
            task = TaskRecord(
                task_id="task_runtime",
                event_id="evt_runtime",
                status=RunState.PREPARED,
                retry_count=0,
                max_retries=3,
            )
            db.execute(
                """
                INSERT INTO tasks (task_id, event_id, dedupe_key, status)
                VALUES ('task_runtime', 'evt_runtime', 'dedupe_runtime', 'prepared')
                """
            )
            db.commit()

            result = agent.run(task, _event())
            self.assertEqual(result.state, RunState.COMPLETED)
            self.assertEqual(adapter.message_counts, [1, 3])
            trace = db.fetchone("SELECT final_decision FROM agent_traces WHERE trace_id IS NOT NULL LIMIT 1")
            self.assertEqual(trace["final_decision"], "skip")
            db.close()

    def test_assess_turn_skips_secret_like_content(self):
        event = StandardEvent(
            event_id="evt_secret",
            session_file="/tmp/session.jsonl",
            conversation_id="conv_secret",
            turn_id="turn_secret",
            user_message="github_pat_11B6J7PKY0abcdefghijklmnop 这是 token，帮我配一下",
            assistant_final_answer="我会把这个 token 写到本地配置里。",
            displayed_at="2026-03-08T00:00:00+00:00",
            source_completeness="full",
            source_confidence=0.95,
            parser_version="v1",
            context_meta={},
        )
        state = AgentState(
            event=event,
            task_id="task_secret",
            trace_id="trace_secret",
            session_memory=SessionMemory(conversation_id=event.conversation_id),
        )
        result = AssessTurnValueTool().execute({}, state)
        self.assertTrue(result.success)
        self.assertEqual(result.data["decision"], "skip")
        self.assertEqual(result.data["reason"], ["contains_secret_like_text"])


if __name__ == "__main__":
    unittest.main()
