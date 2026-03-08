import json
import tempfile
import unittest
from pathlib import Path

from snowball_notes.cli import build_runtime

from tests.test_runtime import _write_config, _write_transcript


class JsonlLoggingTests(unittest.TestCase):
    def test_worker_emits_structured_jsonl_lifecycle_events(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            transcripts = root / "sessions"
            transcripts.mkdir(parents=True)
            config_path = root / "config.yaml"
            _write_config(config_path, transcripts)
            _write_transcript(
                transcripts / "session.jsonl",
                "How should structured logs mirror the worker lifecycle?",
                (
                    "Emit scan, claim, run start, state transition, and completion events so the runtime "
                    "can be debugged from a JSONL stream without querying SQLite first."
                ),
            )
            config, db, vault, worker = build_runtime(str(config_path))
            try:
                result = worker.run_once()
                self.assertIsNotNone(result)
                self.assertEqual(result.state.value, "completed")

                log_path = config.log_path
                self.assertTrue(log_path.exists())
                lines = [
                    json.loads(line)
                    for line in log_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                events = [line["event"] for line in lines]
                for expected in [
                    "worker_scan_completed",
                    "task_claimed",
                    "agent_run_started",
                    "state_transition",
                    "agent_run_finished",
                    "worker_run_completed",
                ]:
                    self.assertIn(expected, events)

                finished = next(line for line in lines if line["event"] == "agent_run_finished")
                self.assertEqual(finished["detail"]["result_state"], "completed")
                self.assertEqual(finished["detail"]["final_decision"], "create_note")
                self.assertIn("ts", finished)
                self.assertIn("level", finished)
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main()
