import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from snowball_notes.cli import main
from snowball_notes.config import load_config
from snowball_notes.eval.runner import load_eval_report
from snowball_notes.storage.sqlite import Database


class DemoSetupTests(unittest.TestCase):
    def test_demo_setup_creates_workspace_with_review_and_eval_assets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dest = Path(temp_dir) / "demo-workspace"
            stdout = io.StringIO()
            with mock.patch("sys.stdout", stdout):
                exit_code = main(["demo", "setup", "--dest", str(dest)])
            self.assertEqual(exit_code, 0)
            self.assertIn("\"transcript_count\": 2", stdout.getvalue())
            self.assertTrue((dest / "config.yaml").exists())
            self.assertTrue((dest / "README.md").exists())
            self.assertEqual(len(list((dest / "sessions").glob("*.jsonl"))), 2)
            self.assertTrue((dest / "eval" / "fixtures" / "sample_cases.json").exists())
            report_path = dest / "reports" / "sample_eval_report.txt"
            self.assertTrue(report_path.exists())
            self.assertIn("Decision accuracy", report_path.read_text(encoding="utf-8"))
            self.assertIn("Review precision", report_path.read_text(encoding="utf-8"))

            config = load_config(dest / "config.yaml")
            db = Database(config.db_path)
            db.migrate()
            try:
                review_row = db.fetchone(
                    "SELECT review_id, final_action, suggested_action FROM review_actions WHERE review_id = ?",
                    ("review_demo_1",),
                )
                self.assertIsNotNone(review_row)
                assert review_row is not None
                self.assertEqual(review_row["final_action"], "pending_review")
                self.assertEqual(review_row["suggested_action"], "create_note")
                trace_row = db.fetchone(
                    "SELECT trace_id, final_decision FROM agent_traces WHERE trace_id = ?",
                    ("trace_demo_review_1",),
                )
                self.assertIsNotNone(trace_row)
                assert trace_row is not None
                self.assertEqual(trace_row["final_decision"], "flagged")
                report = load_eval_report(db)
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["total_cases"], 25)
            finally:
                db.close()
