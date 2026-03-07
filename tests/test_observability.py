import json
import tempfile
import unittest
from pathlib import Path

from snowball_notes.observability.metrics import render_status
from snowball_notes.storage.audit import write_audit_log
from snowball_notes.storage.sqlite import Database
from snowball_notes.utils import now_utc_iso


class ObservabilityStatusTests(unittest.TestCase):
    def test_render_status_includes_health_breakdown(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(Path(temp_dir) / "snowball.db")
            db.migrate()
            now = now_utc_iso()
            for index, confidence in enumerate((0.9, 0.4), start=1):
                db.execute(
                    """
                    INSERT INTO conversation_events (
                      event_id, turn_id, conversation_id, session_file, user_message,
                      assistant_final_answer, displayed_at, source_completeness,
                      source_confidence, parser_version, context_meta_json, payload_json
                    ) VALUES (?, ?, 'conv_1', '/tmp/session.jsonl', 'question', 'answer',
                              ?, 'full', ?, 'v1', '{}', ?)
                    """,
                    (
                        f"evt_{index}",
                        f"turn_{index}",
                        now,
                        confidence,
                        json.dumps({"turn_id": f"turn_{index}"}),
                    ),
                )
            db.execute(
                "INSERT INTO tasks (task_id, event_id, dedupe_key, status, updated_at) VALUES ('task_1', 'evt_1', 'd1', 'completed', ?)",
                (now,),
            )
            db.execute(
                "INSERT INTO tasks (task_id, event_id, dedupe_key, status, updated_at) VALUES ('task_2', 'evt_2', 'd2', 'flagged', ?)",
                (now,),
            )
            trace_one = {
                "steps": [
                    {
                        "tool_name": "search_similar_notes",
                        "tool_success": True,
                        "guardrail_blocked": False,
                    },
                    {
                        "tool_name": "propose_create_note",
                        "tool_success": False,
                        "guardrail_blocked": False,
                    },
                ]
            }
            trace_two = {
                "steps": [
                    {
                        "tool_name": "propose_append_to_note",
                        "tool_success": False,
                        "guardrail_blocked": True,
                    }
                ]
            }
            db.execute(
                """
                INSERT INTO agent_traces (
                  trace_id, turn_id, event_id, prompt_version, model_name, started_at,
                  finished_at, total_steps, exceeded_max_steps, terminal_reason,
                  final_decision, final_confidence, total_input_tokens, total_output_tokens,
                  total_duration_ms, trace_json
                ) VALUES (?, ?, ?, 'agent_system/v1.md', 'heuristic-v1', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "trace_1",
                    "turn_1",
                    "evt_1",
                    now,
                    now,
                    2,
                    0,
                    "completed",
                    "create_note",
                    0.9,
                    120,
                    40,
                    1500,
                    json.dumps(trace_one),
                ),
            )
            db.execute(
                """
                INSERT INTO agent_traces (
                  trace_id, turn_id, event_id, prompt_version, model_name, started_at,
                  finished_at, total_steps, exceeded_max_steps, terminal_reason,
                  final_decision, final_confidence, total_input_tokens, total_output_tokens,
                  total_duration_ms, trace_json
                ) VALUES (?, ?, ?, 'agent_system/v1.md', 'heuristic-v1', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "trace_2",
                    "turn_2",
                    "evt_2",
                    now,
                    now,
                    4,
                    1,
                    "flagged",
                    "append_note",
                    0.4,
                    80,
                    20,
                    2500,
                    json.dumps(trace_two),
                ),
            )
            db.execute(
                """
                INSERT INTO review_actions (review_id, turn_id, trace_id, final_action, reason, created_at)
                VALUES ('review_1', 'turn_1', 'trace_1', 'approved', 'safe', ?)
                """,
                (now,),
            )
            db.execute(
                """
                INSERT INTO review_actions (review_id, turn_id, trace_id, final_action, reason, created_at)
                VALUES ('review_2', 'turn_2', 'trace_2', 'pending_review', 'needs_review', ?)
                """,
                (now,),
            )
            write_audit_log(db, "commit_blocked", {"errors": ["source_confidence"]})
            write_audit_log(db, "reconcile_completed", {"orphan_count": 0, "missing_count": 0})

            rendered = render_status(db, window_days=7)
            self.assertIn("processed_runs: 2", rendered)
            self.assertIn("completed: 1", rendered)
            self.assertIn("flagged: 1", rendered)
            self.assertIn("create_note: 1", rendered)
            self.assertIn("append_note: 1", rendered)
            self.assertIn("tool_error_rate: 33.3% (1/3)", rendered)
            self.assertIn("guardrail_block_rate: 33.3% (1/3)", rendered)
            self.assertIn("commit_rejection_rate: 50.0% (1/2)", rendered)
            self.assertIn("avg_confidence_last_50: 0.65", rendered)
            self.assertIn("low_confidence_rate_last_50: 50.0% (1/2)", rendered)
            self.assertIn("last_result: ok", rendered)
            db.close()


if __name__ == "__main__":
    unittest.main()
