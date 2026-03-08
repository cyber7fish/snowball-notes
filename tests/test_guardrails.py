import tempfile
import unittest
from pathlib import Path

from snowball_notes.agent.guardrails import check_guardrail
from snowball_notes.agent.state import AgentState
from snowball_notes.config import default_config
from snowball_notes.models import SessionMemory, StandardEvent


def _sample_event(*, user_message: str = "How does X work?",
                  answer: str = "X works by doing Y and Z. " * 10,
                  source_confidence: float = 0.95) -> StandardEvent:
    return StandardEvent(
        event_id="evt_guardrail",
        session_file="/tmp/session.jsonl",
        conversation_id="conv_guardrail",
        turn_id="turn_guardrail",
        user_message=user_message,
        assistant_final_answer=answer,
        displayed_at="2026-03-08T00:00:00+00:00",
        source_completeness="full",
        source_confidence=source_confidence,
        parser_version="v1",
        context_meta={},
    )


def _make_state(event=None, **overrides):
    event = event or _sample_event()
    return AgentState(
        event=event,
        task_id="task_guardrail",
        trace_id="trace_guardrail",
        session_memory=SessionMemory(conversation_id="conv_guardrail"),
        **overrides,
    )


class GuardrailTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = default_config(Path(self.temp_dir))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_allows_action_within_limits(self):
        state = _make_state()
        result = check_guardrail(self.config, state, "propose_create_note")
        self.assertTrue(result.allowed)

    def test_allows_decision_tools_always(self):
        state = _make_state(write_count=99)
        for tool in ("assess_turn_value", "extract_knowledge_points",
                      "search_similar_notes", "read_note"):
            result = check_guardrail(self.config, state, tool)
            self.assertTrue(result.allowed, f"{tool} should always be allowed")

    def test_blocks_when_write_limit_exceeded(self):
        self.config.agent.max_writes_per_run = 1
        state = _make_state(write_count=1)
        for tool in ("propose_create_note", "propose_append_to_note",
                      "propose_archive_turn", "propose_link_notes"):
            result = check_guardrail(self.config, state, tool)
            self.assertFalse(result.allowed, f"{tool} should be blocked when write limit exceeded")
            self.assertIn("write limit", result.reason.lower())

    def test_blocks_create_when_low_confidence(self):
        state = _make_state(event=_sample_event(source_confidence=0.5))
        result = check_guardrail(self.config, state, "propose_create_note")
        self.assertFalse(result.allowed)
        self.assertIn("source_confidence", result.reason)

    def test_allows_create_at_confidence_boundary(self):
        state = _make_state(event=_sample_event(source_confidence=0.70))
        result = check_guardrail(self.config, state, "propose_create_note")
        self.assertTrue(result.allowed)

    def test_blocks_append_when_low_confidence(self):
        state = _make_state(event=_sample_event(source_confidence=0.80))
        result = check_guardrail(self.config, state, "propose_append_to_note")
        self.assertFalse(result.allowed)
        self.assertIn("confidence", result.reason.lower())

    def test_allows_append_at_confidence_boundary(self):
        state = _make_state(event=_sample_event(source_confidence=0.85))
        result = check_guardrail(self.config, state, "propose_append_to_note")
        self.assertTrue(result.allowed)

    def test_blocks_append_when_append_limit_exceeded(self):
        self.config.agent.max_appends_per_run = 1
        state = _make_state(append_count=1)
        result = check_guardrail(self.config, state, "propose_append_to_note")
        self.assertFalse(result.allowed)
        self.assertIn("append limit", result.reason.lower())

    def test_blocks_project_meta_note_creation(self):
        state = _make_state(event=_sample_event(
            user_message="你说的这一步属于技术方案里的哪个Phase？",
            answer="主归属是 Phase 4，但这一步本质上是在补 review action 的执行闭环。",
        ))
        result = check_guardrail(self.config, state, "propose_create_note")
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "project_meta_turn_not_durable_note")

    def test_allows_archive_despite_project_meta(self):
        state = _make_state(event=_sample_event(
            user_message="你说的这一步属于技术方案里的哪个Phase？",
            answer="主归属是 Phase 4。",
        ))
        result = check_guardrail(self.config, state, "propose_archive_turn")
        self.assertTrue(result.allowed)

    def test_allows_concept_explanation_even_when_answer_mentions_project_meta_examples(self):
        state = _make_state(event=_sample_event(
            user_message="什么是 Guardrails？",
            answer=(
                "Guardrails 是运行时的确定性安全检查。比如项目进度 / Phase 归属这类 meta turn "
                "不能变成知识 note；当问题只是“当前做到哪了”时，系统会直接 block。"
            ),
        ))
        result = check_guardrail(self.config, state, "propose_create_note")
        self.assertTrue(result.allowed)

    def test_allows_flag_for_review_always(self):
        state = _make_state(write_count=99, append_count=99,
                            event=_sample_event(source_confidence=0.1))
        result = check_guardrail(self.config, state, "flag_for_review")
        self.assertTrue(result.allowed)


if __name__ == "__main__":
    unittest.main()
