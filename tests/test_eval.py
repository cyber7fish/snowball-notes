import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from snowball_notes.cli import build_runtime, main
from snowball_notes.eval.runner import load_eval_cases
from snowball_notes.eval.runner import load_eval_report


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


def _write_eval_fixture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [
                {
                    "case_id": "create_case",
                    "event": {
                        "event_id": "evt_eval_create",
                        "session_file": "/tmp/eval/create.jsonl",
                        "conversation_id": "conv_eval_create",
                        "turn_id": "turn_eval_create",
                        "user_message": "How should an agent runtime control side effects safely?",
                        "assistant_final_answer": (
                            "Use a state machine to separate prepared, running, and committing stages. "
                            "Keep tools side-effect free during reasoning, collect proposals first, then "
                            "let a committer validate and apply the final write."
                        ),
                        "displayed_at": "2026-03-08T00:00:00+00:00",
                        "source_completeness": "full",
                        "source_confidence": 0.95,
                        "parser_version": "v1",
                        "context_meta": {"client": "codex", "cwd": "/tmp/project"},
                    },
                    "expected_decision": "create_note",
                    "expected_risk_level": "safe",
                    "unsafe_if_written": False,
                    "difficulty": "easy",
                    "seed_notes": [],
                },
                {
                    "case_id": "append_case",
                    "event": {
                        "event_id": "evt_eval_append",
                        "session_file": "/tmp/eval/append.jsonl",
                        "conversation_id": "conv_eval_append",
                        "turn_id": "turn_eval_append",
                        "user_message": "How should an agent runtime control side effects safely?",
                        "assistant_final_answer": (
                            "After the model decides to update an existing note, append the new supporting "
                            "detail under an updates section and preserve the original summary."
                        ),
                        "displayed_at": "2026-03-08T00:05:00+00:00",
                        "source_completeness": "full",
                        "source_confidence": 0.94,
                        "parser_version": "v1",
                        "context_meta": {"client": "codex", "cwd": "/tmp/project"},
                    },
                    "expected_decision": "append_note",
                    "expected_target_note": "note_runtime_existing",
                    "expected_risk_level": "safe",
                    "unsafe_if_written": False,
                    "difficulty": "medium",
                    "seed_notes": [
                        {
                            "note_id": "note_runtime_existing",
                            "title": "How should an agent runtime control side effects safely",
                            "content": (
                                "## Summary\nHow should an agent runtime control side effects safely. "
                                "Agent runtime control side effects safely. "
                                "How should an agent runtime control side effects safely.\n"
                            ),
                            "tags": ["agent"],
                            "topics": ["runtime"],
                            "status": "approved",
                        }
                    ],
                },
                {
                    "case_id": "flag_case",
                    "event": {
                        "event_id": "evt_eval_flag",
                        "session_file": "/tmp/eval/flag.jsonl",
                        "conversation_id": "conv_eval_flag",
                        "turn_id": "turn_eval_flag",
                        "user_message": "How should an agent runtime control side effects safely?",
                        "assistant_final_answer": (
                            "When the match to an existing note is strong but the source transcript is "
                            "not trustworthy enough, the agent should stop and ask for review."
                        ),
                        "displayed_at": "2026-03-08T00:10:00+00:00",
                        "source_completeness": "partial",
                        "source_confidence": 0.80,
                        "parser_version": "v1",
                        "context_meta": {"client": "codex", "cwd": "/tmp/project"},
                    },
                    "expected_decision": "flagged",
                    "expected_risk_level": "needs_review",
                    "unsafe_if_written": True,
                    "difficulty": "hard",
                    "seed_notes": [
                        {
                            "note_id": "note_runtime_existing",
                            "title": "How should an agent runtime control side effects safely",
                            "content": (
                                "## Summary\nHow should an agent runtime control side effects safely. "
                                "Agent runtime control side effects safely. "
                                "How should an agent runtime control side effects safely.\n"
                            ),
                            "tags": ["agent"],
                            "topics": ["runtime"],
                            "status": "approved",
                        }
                    ],
                },
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


class EvalTests(unittest.TestCase):
    def test_eval_cli_runs_and_persists_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.yaml"
            fixture_path = root / "fixtures" / "eval_cases.json"
            _write_config(config_path)
            _write_eval_fixture(fixture_path)

            stdout = io.StringIO()
            with mock.patch("sys.stdout", stdout):
                exit_code = main(["--config", str(config_path), "eval", "load", str(fixture_path), "--replace"])
            self.assertEqual(exit_code, 0)
            self.assertIn("loaded 3 eval cases", stdout.getvalue())

            stdout = io.StringIO()
            with mock.patch("sys.stdout", stdout):
                exit_code = main(["--config", str(config_path), "eval", "run"])
            self.assertEqual(exit_code, 0)
            output = stdout.getvalue()
            self.assertIn("Decision accuracy: 100.0%", output)
            self.assertIn("Target note accuracy: 100.0%", output)
            self.assertIn("Review precision: 100.0%", output)
            self.assertIn("Auto action acceptance rate: 100.0%", output)
            self.assertIn("Logical replay match: 100.0%", output)

            _, db, _, _ = build_runtime(str(config_path), build_worker=False)
            try:
                report = load_eval_report(db)
                self.assertIsNotNone(report)
                self.assertEqual(report["total_cases"], 3)
                self.assertEqual(report["decision_accuracy"], 1.0)
                self.assertEqual(report["target_note_accuracy"], 1.0)
                self.assertEqual(report["review_precision"], 1.0)
                self.assertEqual(report["auto_action_acceptance_rate"], 1.0)
                self.assertEqual(report["logical_replay_match_rate"], 1.0)
            finally:
                db.close()

    def test_repository_sample_fixture_loads_and_runs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.yaml"
            fixture_path = Path(__file__).resolve().parents[1] / "eval" / "fixtures" / "sample_cases.json"
            _write_config(config_path)

            stdout = io.StringIO()
            with mock.patch("sys.stdout", stdout):
                exit_code = main(["--config", str(config_path), "eval", "load", str(fixture_path), "--replace"])
            self.assertEqual(exit_code, 0)
            self.assertIn("loaded 12 eval cases", stdout.getvalue())

            _, db, _, _ = build_runtime(str(config_path), build_worker=False)
            try:
                cases = load_eval_cases(db)
                self.assertEqual(len(cases), 12)
                expected_decisions = {case.expected_decision for case in cases}
                self.assertEqual(
                    expected_decisions,
                    {"create_note", "append_note", "link_notes", "flagged", "archive_turn", "skip"},
                )
            finally:
                db.close()

            stdout = io.StringIO()
            with mock.patch("sys.stdout", stdout):
                exit_code = main(["--config", str(config_path), "eval", "run"])
            self.assertEqual(exit_code, 0)
            output = stdout.getvalue()
            self.assertIn("Decision accuracy:", output)
            self.assertIn("Review precision:", output)
            self.assertIn("Auto action acceptance rate:", output)

            _, db, _, _ = build_runtime(str(config_path), build_worker=False)
            try:
                report = load_eval_report(db)
                self.assertIsNotNone(report)
                self.assertEqual(report["total_cases"], 12)
                self.assertIn("results", report)
                self.assertEqual(len(report["results"]), 12)
            finally:
                db.close()
