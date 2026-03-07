import json
import tempfile
import unittest
from pathlib import Path

from snowball_notes.intake.transcript_parser import parse_session_file


class TranscriptParserTests(unittest.TestCase):
    def test_parse_completed_turn(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            session_file = Path(temp_dir) / "session.jsonl"
            lines = [
                {
                    "timestamp": "2026-03-07T04:24:18.100Z",
                    "type": "session_meta",
                    "payload": {"id": "conv_123", "cwd": "/tmp/project", "originator": "codex_cli_rs", "cli_version": "0.111.0"},
                },
                {
                    "timestamp": "2026-03-07T04:24:19.100Z",
                    "type": "event_msg",
                    "payload": {"type": "task_started", "turn_id": "turn_001"},
                },
                {
                    "timestamp": "2026-03-07T04:24:20.100Z",
                    "type": "event_msg",
                    "payload": {"type": "user_message", "message": "How do I build a queue?"},
                },
                {
                    "timestamp": "2026-03-07T04:24:21.100Z",
                    "type": "event_msg",
                    "phase": "commentary",
                    "payload": {"type": "agent_message", "message": "I am inspecting the queue."},
                },
                {
                    "timestamp": "2026-03-07T04:24:22.100Z",
                    "type": "event_msg",
                    "phase": "final_answer",
                    "payload": {"type": "agent_message", "message": "Build a claim-based queue with retries and durable state transitions."},
                },
                {
                    "timestamp": "2026-03-07T04:24:23.100Z",
                    "type": "event_msg",
                    "payload": {"type": "task_complete", "turn_id": "turn_001"},
                },
            ]
            session_file.write_text("\n".join(json.dumps(line, ensure_ascii=False) for line in lines), encoding="utf-8")
            events = parse_session_file(session_file)
            self.assertEqual(len(events), 1)
            event = events[0]
            self.assertEqual(event.turn_id, "turn_001")
            self.assertEqual(event.conversation_id, "conv_123")
            self.assertEqual(event.user_message, "How do I build a queue?")
            self.assertIn("claim-based queue", event.assistant_final_answer)
            self.assertGreaterEqual(event.source_confidence, 0.8)


if __name__ == "__main__":
    unittest.main()

