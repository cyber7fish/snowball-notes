import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from snowball_notes.models import StandardEvent
from snowball_notes.storage.sqlite import Database
from snowball_notes.utils import now_utc_iso


FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None

if FASTAPI_AVAILABLE:
    from fastapi.testclient import TestClient
    from snowball_notes.config import load_config
    from snowball_notes.review.server import build_review_app


def _write_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "paths:",
                "  db: \"./data/snowball.db\"",
                "  log: \"./logs/snowball.jsonl\"",
                "vault:",
                "  path: \"./vault\"",
                "  inbox_dir: \"Inbox\"",
                "  archive_dir: \"Archive/Conversations\"",
                "  atomic_dir: \"Knowledge/Atomic\"",
                "intake:",
                "  transcript_dir: \"./sessions\"",
                "  parser_version: \"v1\"",
                "  min_response_length: 120",
                "  min_confidence_to_run: 0.50",
                "agent:",
                "  provider: \"heuristic\"",
                "  model: \"heuristic-v1\"",
                "  max_steps: 8",
                "  prompt_version: \"agent_system/v1.md\"",
                "  max_writes_per_run: 1",
                "  max_appends_per_run: 1",
                "retrieval:",
                "  top_k: 5",
                "  append_threshold: 0.82",
                "  review_threshold: 0.62",
                "guardrails:",
                "  min_confidence_for_note: 0.70",
                "  min_confidence_for_append: 0.85",
                "worker:",
                "  poll_interval_seconds: 10",
                "  claim_timeout_seconds: 300",
                "  max_retries: 3",
            ]
        ),
        encoding="utf-8",
    )


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi test dependencies are not installed")
class ReviewServerTests(unittest.TestCase):
    def test_review_server_renders_and_handles_actions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.yaml"
            _write_config(config_path)

            config = load_config(config_path)
            db = Database(config.db_path)
            db.migrate()
            try:
                event = StandardEvent(
                    event_id="evt_review_1",
                    session_file="/tmp/review/session.jsonl",
                    conversation_id="conv_review_1",
                    turn_id="turn_review_1",
                    user_message="How should I review ambiguous agent writes?",
                    assistant_final_answer=(
                        "When the match is ambiguous, keep the run flagged, inspect the trace, "
                        "and only approve a write after checking the replay snapshot."
                    ),
                    displayed_at="2026-03-08T00:00:00+00:00",
                    source_completeness="full",
                    source_confidence=0.88,
                    parser_version="v1",
                    context_meta={"cwd": "/tmp/project", "client": "codex"},
                )
                db.execute(
                    """
                    INSERT INTO conversation_events (
                      event_id, turn_id, conversation_id, session_file, user_message,
                      assistant_final_answer, displayed_at, source_completeness,
                      source_confidence, parser_version, context_meta_json, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.event_id,
                        event.turn_id,
                        event.conversation_id,
                        event.session_file,
                        event.user_message,
                        event.assistant_final_answer,
                        event.displayed_at,
                        event.source_completeness,
                        event.source_confidence,
                        event.parser_version,
                        json.dumps(event.context_meta, ensure_ascii=False),
                        json.dumps(event.to_dict(), ensure_ascii=False),
                    ),
                )
                trace_json = {
                    "trace_id": "trace_review_1",
                    "event_id": event.event_id,
                    "turn_id": event.turn_id,
                    "prompt_version": "agent_system/v1.md",
                    "model_name": "heuristic-v1",
                    "started_at": now_utc_iso(),
                    "finished_at": now_utc_iso(),
                    "total_steps": 2,
                    "exceeded_max_steps": 0,
                    "terminal_reason": "ambiguous_existing_note",
                    "final_decision": "flagged",
                    "final_confidence": 0.88,
                    "total_input_tokens": 120,
                    "total_output_tokens": 40,
                    "total_duration_ms": 320,
                    "steps": [
                        {
                            "step_index": 0,
                            "runtime_state": "running",
                            "decision_summary": "Search for similar notes.",
                            "tool_name": "search_similar_notes",
                            "tool_input_json": json.dumps({"query": event.user_message}, ensure_ascii=False),
                            "tool_result_json": json.dumps([{"note_id": "note_1", "title": "Agent review"}], ensure_ascii=False),
                            "tool_success": True,
                            "proposal_ids": [],
                            "guardrail_blocked": False,
                            "duration_ms": 12,
                            "input_tokens": 60,
                            "output_tokens": 20,
                        },
                        {
                            "step_index": 1,
                            "runtime_state": "running",
                            "decision_summary": "Escalate for review.",
                            "tool_name": "flag_for_review",
                            "tool_input_json": json.dumps({"reason": "ambiguous_existing_note"}, ensure_ascii=False),
                            "tool_result_json": json.dumps({"flagged": True, "reason": "ambiguous_existing_note"}, ensure_ascii=False),
                            "tool_success": True,
                            "proposal_ids": [],
                            "guardrail_blocked": False,
                            "duration_ms": 8,
                            "input_tokens": 60,
                            "output_tokens": 20,
                        },
                    ],
                }
                db.execute(
                    """
                    INSERT INTO agent_traces (
                      trace_id, turn_id, event_id, prompt_version, model_name, started_at, finished_at,
                      total_steps, exceeded_max_steps, terminal_reason, final_decision, final_confidence,
                      total_input_tokens, total_output_tokens, total_duration_ms, trace_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "trace_review_1",
                        event.turn_id,
                        event.event_id,
                        "agent_system/v1.md",
                        "heuristic-v1",
                        trace_json["started_at"],
                        trace_json["finished_at"],
                        trace_json["total_steps"],
                        trace_json["exceeded_max_steps"],
                        trace_json["terminal_reason"],
                        trace_json["final_decision"],
                        trace_json["final_confidence"],
                        trace_json["total_input_tokens"],
                        trace_json["total_output_tokens"],
                        trace_json["total_duration_ms"],
                        json.dumps(trace_json, ensure_ascii=False),
                    ),
                )
                db.execute(
                    """
                    INSERT INTO replay_bundles (
                      trace_id, event_json, prompt_snapshot, config_snapshot_json, tool_results_json,
                      knowledge_snapshot_refs_json, model_name, model_adapter_version, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "trace_review_1",
                        json.dumps(event.to_dict(), ensure_ascii=False),
                        "prompt",
                        json.dumps(config.to_dict(), ensure_ascii=False),
                        json.dumps(
                            [
                                {"step": 0, "tool": "search_similar_notes", "input": {"query": event.user_message}, "output": [{"note_id": "note_1"}], "success": True},
                                {"step": 1, "tool": "flag_for_review", "input": {"reason": "ambiguous_existing_note"}, "output": {"flagged": True}, "success": True},
                            ],
                            ensure_ascii=False,
                        ),
                        json.dumps([{"note_id": "note_1", "content_hash": "abc", "title": "Agent review", "similarity": 0.73}], ensure_ascii=False),
                        "heuristic-v1",
                        "2026-03-08",
                        now_utc_iso(),
                    ),
                )
                db.execute(
                    """
                    INSERT INTO review_actions (
                      review_id, turn_id, trace_id, final_action, suggested_action,
                      suggested_target_note_id, suggested_payload_json, reason, created_at
                    ) VALUES (?, ?, ?, 'pending_review', ?, ?, ?, ?, ?)
                    """,
                    (
                        "review_1",
                        event.turn_id,
                        "trace_review_1",
                        "create_note",
                        None,
                        json.dumps(
                            {
                                "title": "How should I review ambiguous agent writes",
                                "content": "## Summary\nInspect trace and replay before approving a write.\n",
                                "tags": ["review"],
                                "topics": ["agent"],
                            },
                            ensure_ascii=False,
                        ),
                        "ambiguous_existing_note",
                        now_utc_iso(),
                    ),
                )
                db.execute(
                    """
                    INSERT INTO review_actions (
                      review_id, turn_id, trace_id, final_action, suggested_action,
                      suggested_target_note_id, suggested_payload_json, reason, created_at
                    ) VALUES (?, ?, ?, 'pending_review', ?, ?, ?, ?, ?)
                    """,
                    (
                        "review_2",
                        event.turn_id,
                        "trace_review_1",
                        "append_note",
                        "note_conflict",
                        json.dumps(
                            {
                                "note_id": "note_conflict",
                                "content": "## Summary\nCreate a sibling note instead of merging this turn.\n",
                                "title": "Create a Sibling Note for Ambiguous Merge",
                                "tags": ["review"],
                                "topics": ["agent"],
                            },
                            ensure_ascii=False,
                        ),
                        "needs_separate_note",
                        now_utc_iso(),
                    ),
                )
                db.execute(
                    """
                    INSERT INTO review_actions (
                      review_id, turn_id, trace_id, final_action, final_target_note_id,
                      suggested_action, suggested_target_note_id, suggested_payload_json, reason, created_at
                    ) VALUES (?, ?, ?, 'pending_review', ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "review_3",
                        event.turn_id,
                        "trace_review_1",
                        "note_conflict",
                        "append_note",
                        "note_conflict",
                        json.dumps(
                            {
                                "note_id": "note_conflict",
                                "content": "## Summary\nThis still conflicts with the existing note.\n",
                            },
                            ensure_ascii=False,
                        ),
                        "conflicting_evidence",
                        now_utc_iso(),
                    ),
                )
                db.execute(
                    """
                    INSERT INTO review_actions (
                      review_id, turn_id, trace_id, final_action, suggested_action,
                      suggested_target_note_id, suggested_payload_json, reason, created_at
                    ) VALUES (?, ?, ?, 'pending_review', ?, ?, ?, ?, ?)
                    """,
                    (
                        "review_4",
                        event.turn_id,
                        "trace_review_1",
                        "archive_turn",
                        None,
                        json.dumps(
                            {
                                "title": "Discarded Review",
                                "content": "No durable knowledge here.",
                            },
                            ensure_ascii=False,
                        ),
                        "not_useful_after_inspection",
                        now_utc_iso(),
                    ),
                )
                db.commit()
            finally:
                db.close()

            app = build_review_app(str(config_path))
            client = TestClient(app)
            try:
                home = client.get("/")
                self.assertEqual(home.status_code, 200)
                self.assertIn("Snowball Review Console", home.text)
                self.assertIn("Record Confidence Feedback", home.text)
                self.assertIn("Create Separate", home.text)
                self.assertIn("Mark Conflict", home.text)

                reviews = client.get("/api/reviews")
                self.assertEqual(reviews.status_code, 200)
                self.assertEqual(len(reviews.json()), 4)

                detail = client.get("/api/reviews/review_1")
                self.assertEqual(detail.status_code, 200)
                payload = detail.json()
                self.assertEqual(payload["review_id"], "review_1")
                self.assertEqual(payload["trace"]["final_decision"], "flagged")
                self.assertEqual(payload["event"]["turn_id"], "turn_review_1")

                feedback = client.post(
                    "/api/reviews/review_1/confidence-feedback",
                    json={"label": "partial", "annotator": "tester"},
                )
                self.assertEqual(feedback.status_code, 200)
                self.assertTrue(feedback.json()["recorded"])

                approve = client.post(
                    "/api/reviews/review_1/approve",
                    json={
                        "action": "create",
                        "reviewer": "tester",
                        "title": "Approved Review Note",
                    },
                )
                self.assertEqual(approve.status_code, 200)
                self.assertTrue(approve.json()["approved"])

                separate = client.post(
                    "/api/reviews/review_2/create-separate",
                    json={
                        "reviewer": "tester",
                        "title": "Separate Review Note",
                    },
                )
                self.assertEqual(separate.status_code, 200)
                self.assertEqual(separate.json()["final_action"], "create_separate")

                conflict = client.post(
                    "/api/reviews/review_3/mark-conflict",
                    json={
                        "reviewer": "tester",
                        "note_id": "note_conflict",
                    },
                )
                self.assertEqual(conflict.status_code, 200)
                self.assertEqual(conflict.json()["final_action"], "mark_conflict")

                discard = client.post(
                    "/api/reviews/review_4/discard",
                    json={"reviewer": "tester"},
                )
                self.assertEqual(discard.status_code, 200)
                self.assertEqual(discard.json()["final_action"], "discarded")

                reviews_after = client.get("/api/reviews")
                self.assertEqual(reviews_after.status_code, 200)
                self.assertEqual(reviews_after.json(), [])

                reopened_db = Database(config.db_path)
                try:
                    review_row = reopened_db.fetchone(
                        "SELECT final_action, reviewer, final_target_note_id FROM review_actions WHERE review_id = ?",
                        ("review_1",),
                    )
                    self.assertEqual(review_row["final_action"], "approve_create")
                    self.assertEqual(review_row["reviewer"], "tester")
                    self.assertTrue(review_row["final_target_note_id"])
                    separate_row = reopened_db.fetchone(
                        "SELECT final_action, reviewer, final_target_note_id FROM review_actions WHERE review_id = ?",
                        ("review_2",),
                    )
                    self.assertEqual(separate_row["final_action"], "create_separate")
                    self.assertEqual(separate_row["reviewer"], "tester")
                    self.assertTrue(separate_row["final_target_note_id"])
                    conflict_row = reopened_db.fetchone(
                        "SELECT final_action, reviewer, final_target_note_id FROM review_actions WHERE review_id = ?",
                        ("review_3",),
                    )
                    self.assertEqual(conflict_row["final_action"], "mark_conflict")
                    self.assertEqual(conflict_row["reviewer"], "tester")
                    self.assertEqual(conflict_row["final_target_note_id"], "note_conflict")
                    discard_row = reopened_db.fetchone(
                        "SELECT final_action, reviewer FROM review_actions WHERE review_id = ?",
                        ("review_4",),
                    )
                    self.assertEqual(discard_row["final_action"], "discarded")
                    self.assertEqual(discard_row["reviewer"], "tester")
                    feedback_count = reopened_db.fetchone(
                        "SELECT COUNT(*) AS count FROM confidence_feedback WHERE turn_id = ?",
                        ("turn_review_1",),
                    )
                    self.assertEqual(feedback_count["count"], 1)
                finally:
                    reopened_db.close()
            finally:
                client.close()
