import json
import tempfile
import unittest
from pathlib import Path

from snowball_notes.calibrate.confidence_feedback import (
    analyze_confidence_calibration,
    record_confidence_feedback,
    render_calibration_report,
)
from snowball_notes.storage.sqlite import Database


class ConfidenceCalibrationTests(unittest.TestCase):
    def test_record_feedback_requires_existing_turn(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(Path(temp_dir) / "snowball.db")
            db.migrate()
            self.assertFalse(record_confidence_feedback(db, "missing_turn", "trustworthy", "tester"))
            db.close()

    def test_analyze_feedback_reports_high_confidence_parser_issues(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(Path(temp_dir) / "snowball.db")
            db.migrate()
            for index, score in enumerate((0.91, 0.88, 0.95), start=1):
                turn_id = f"turn_{index}"
                db.execute(
                    """
                    INSERT INTO conversation_events (
                      event_id, turn_id, conversation_id, session_file, user_message,
                      assistant_final_answer, displayed_at, source_completeness,
                      source_confidence, parser_version, context_meta_json, payload_json
                    ) VALUES (?, ?, 'conv_1', '/tmp/session.jsonl', 'question', 'answer',
                              '2026-03-08T00:00:00+00:00', 'full', ?, 'v1', '{}', ?)
                    """,
                    (
                        f"evt_{index}",
                        turn_id,
                        score,
                        json.dumps({"turn_id": turn_id}),
                    ),
                )
                self.assertTrue(record_confidence_feedback(db, turn_id, "bad_parse", "tester"))
            report = analyze_confidence_calibration(db)
            rendered = render_calibration_report(report)
            self.assertEqual(report.total_feedback, 3)
            self.assertEqual(report.buckets[-1].total, 3)
            self.assertIn("High-confidence samples", report.recommendation)
            self.assertIn("[0.8, 1.0]", rendered)
            self.assertIn("bad_parse=3", rendered)
            db.close()


if __name__ == "__main__":
    unittest.main()

