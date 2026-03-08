import json
import tempfile
import unittest
from pathlib import Path

from snowball_notes.cli import build_runtime
from snowball_notes.review.cli import approve_review


def _write_config(
    path: Path,
    transcript_dir: Path,
    parser_version: str = "v1",
    min_confidence_for_append: float = 0.85,
) -> None:
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
                f"  transcript_dir: \"{transcript_dir}\"",
                f"  parser_version: \"{parser_version}\"",
                "  min_response_length: 120",
                "  min_confidence_to_run: 0.50",
                "agent:",
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
                f"  min_confidence_for_append: {min_confidence_for_append:.2f}",
                "worker:",
                "  poll_interval_seconds: 10",
                "  claim_timeout_seconds: 300",
                "  max_retries: 3",
            ]
        ),
        encoding="utf-8",
    )


def _write_transcript(
    path: Path,
    user_message: str,
    answer: str,
    *,
    conversation_id: str = "conv_123",
    turn_id: str = "turn_001",
    duplicate_task_complete: bool = False,
) -> None:
    lines = [
        {
            "timestamp": "2026-03-07T04:24:18.100Z",
            "type": "session_meta",
            "payload": {"id": conversation_id, "cwd": "/tmp/project", "originator": "codex_cli_rs", "cli_version": "0.111.0"},
        },
        {
            "timestamp": "2026-03-07T04:24:19.100Z",
            "type": "event_msg",
            "payload": {"type": "task_started", "turn_id": turn_id},
        },
        {
            "timestamp": "2026-03-07T04:24:20.100Z",
            "type": "event_msg",
            "payload": {"type": "user_message", "message": user_message},
        },
        {
            "timestamp": "2026-03-07T04:24:21.100Z",
            "type": "event_msg",
            "phase": "final_answer",
            "payload": {"type": "agent_message", "message": answer},
        },
        {
            "timestamp": "2026-03-07T04:24:23.100Z",
            "type": "event_msg",
            "payload": {"type": "task_complete", "turn_id": turn_id},
        },
    ]
    if duplicate_task_complete:
        lines.append(
            {
                "timestamp": "2026-03-07T04:24:23.200Z",
                "type": "event_msg",
                "payload": {"type": "task_complete", "turn_id": turn_id},
            }
        )
    path.write_text("\n".join(json.dumps(line, ensure_ascii=False) for line in lines), encoding="utf-8")


def _write_current_transcript(
    path: Path,
    user_message: str,
    answer: str,
    *,
    conversation_id: str = "conv_123",
    turn_id: str = "turn_001",
) -> None:
    lines = [
        {
            "timestamp": "2026-03-07T04:24:18.100Z",
            "type": "session_meta",
            "payload": {"id": conversation_id, "cwd": "/tmp/project", "originator": "codex_cli_rs", "cli_version": "0.111.0"},
        },
        {
            "timestamp": "2026-03-07T04:24:19.100Z",
            "type": "event_msg",
            "payload": {"type": "task_started", "turn_id": turn_id},
        },
        {
            "timestamp": "2026-03-07T04:24:20.100Z",
            "type": "event_msg",
            "payload": {"type": "user_message", "message": user_message},
        },
        {
            "timestamp": "2026-03-07T04:24:21.100Z",
            "type": "event_msg",
            "payload": {"type": "agent_message", "message": "Inspecting the transcript shape.", "phase": "commentary"},
        },
        {
            "timestamp": "2026-03-07T04:24:22.100Z",
            "type": "event_msg",
            "payload": {"type": "agent_message", "message": answer, "phase": "final_answer"},
        },
        {
            "timestamp": "2026-03-07T04:24:22.101Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "phase": "final_answer",
                "content": [{"type": "output_text", "text": answer}],
            },
        },
        {
            "timestamp": "2026-03-07T04:24:23.100Z",
            "type": "event_msg",
            "payload": {"type": "task_complete", "turn_id": turn_id, "last_agent_message": answer},
        },
    ]
    path.write_text("\n".join(json.dumps(line, ensure_ascii=False) for line in lines), encoding="utf-8")


class RuntimeTests(unittest.TestCase):
    def test_worker_runs_startup_reconcile_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            transcripts = root / "sessions"
            transcripts.mkdir(parents=True)
            config_path = root / "config.yaml"
            _write_config(config_path, transcripts)
            config, db, vault, worker = build_runtime(str(config_path))
            try:
                self.assertIsNone(worker.run_once())
                self.assertIsNone(worker.run_once())
                reconcile_count = db.fetchone(
                    """
                    SELECT COUNT(*) AS count
                    FROM audit_logs
                    WHERE event_type = 'reconcile_completed'
                    """
                )
                self.assertEqual(reconcile_count["count"], 1)
            finally:
                db.close()

    def test_worker_creates_note_and_trace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            transcripts = root / "sessions"
            transcripts.mkdir(parents=True)
            config_path = root / "config.yaml"
            _write_config(config_path, transcripts)
            _write_transcript(
                transcripts / "session.jsonl",
                "How should I design an agent runtime with guarded side effects?",
                (
                    "Use a state machine, keep side effects behind proposals, add guardrails before tool "
                    "execution, and persist replay bundles so the run is debuggable over time."
                ),
            )
            config, db, vault, worker = build_runtime(str(config_path))
            try:
                result = worker.run_once()
                self.assertIsNotNone(result)
                self.assertEqual(result.state.value, "completed")
                note_rows = db.fetchall("SELECT note_id, vault_path FROM notes WHERE note_type = 'atomic'")
                self.assertEqual(len(note_rows), 1)
                self.assertTrue(Path(note_rows[0]["vault_path"]).exists())
                trace_rows = db.fetchall("SELECT trace_id FROM agent_traces")
                replay_rows = db.fetchall("SELECT trace_id FROM replay_bundles")
                self.assertEqual(len(trace_rows), 1)
                self.assertEqual(len(replay_rows), 1)
            finally:
                db.close()

    def test_worker_appends_to_existing_note(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            transcripts = root / "sessions"
            transcripts.mkdir(parents=True)
            config_path = root / "config.yaml"
            _write_config(config_path, transcripts)
            _write_transcript(
                transcripts / "session.jsonl",
                "How should I design an agent runtime with guarded side effects?",
                (
                    "Use a state machine, keep side effects behind proposals, add guardrails before tool "
                    "execution, and persist replay bundles so the run is debuggable over time."
                ),
            )
            config, db, vault, worker = build_runtime(str(config_path))
            try:
                existing_path, content_hash = vault.write_new_note(
                    note_id="note_existing",
                    title="How should I design an agent runtime with guarded side effects",
                    content="## Summary\nExisting note body.",
                    tags=["agent"],
                    topics=["runtime"],
                    source_event_ids=["evt_seed"],
                )
                db.execute(
                    """
                    INSERT INTO notes (note_id, note_type, title, vault_path, content_hash, status, metadata_json, created_at, updated_at)
                    VALUES (?, 'atomic', ?, ?, ?, 'approved', '{}', '2026-03-07T00:00:00+00:00', '2026-03-07T00:00:00+00:00')
                    """,
                    (
                        "note_existing",
                        "How should I design an agent runtime with guarded side effects",
                        str(existing_path.resolve()),
                        content_hash,
                    ),
                )
                db.commit()
                result = worker.run_once()
                self.assertIsNotNone(result)
                self.assertEqual(result.state.value, "completed")
                note_rows = db.fetchall("SELECT note_id FROM notes WHERE note_type = 'atomic'")
                self.assertEqual(len(note_rows), 1)
                updated = existing_path.read_text(encoding="utf-8")
                self.assertIn("## Updates", updated)
            finally:
                db.close()

    def test_worker_flags_ambiguous_existing_note(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            transcripts = root / "sessions"
            transcripts.mkdir(parents=True)
            config_path = root / "config.yaml"
            _write_config(config_path, transcripts, parser_version="legacy", min_confidence_for_append=0.95)
            _write_transcript(
                transcripts / "session.jsonl",
                "How should I design an agent runtime with guarded side effects?",
                (
                    "Use a state machine, keep side effects behind proposals, add guardrails before tool "
                    "execution, and persist replay bundles so the run stays debuggable while the vault remains safe."
                ),
                duplicate_task_complete=True,
            )
            config, db, vault, worker = build_runtime(str(config_path))
            try:
                existing_path, content_hash = vault.write_new_note(
                    note_id="note_existing",
                    title="How should I design an agent runtime with guarded side effects",
                    content="## Summary\nExisting note body.",
                    tags=["agent"],
                    topics=["runtime"],
                    source_event_ids=["evt_seed"],
                )
                db.execute(
                    """
                    INSERT INTO notes (note_id, note_type, title, vault_path, content_hash, status, metadata_json, created_at, updated_at)
                    VALUES (?, 'atomic', ?, ?, ?, 'approved', '{}', '2026-03-07T00:00:00+00:00', '2026-03-07T00:00:00+00:00')
                    """,
                    (
                        "note_existing",
                        "How should I design an agent runtime with guarded side effects",
                        str(existing_path.resolve()),
                        content_hash,
                    ),
                )
                db.commit()
                result = worker.run_once()
                self.assertIsNotNone(result)
                self.assertEqual(result.state.value, "flagged")
                review = db.fetchone(
                    """
                    SELECT final_action, final_target_note_id, suggested_action,
                           suggested_target_note_id, suggested_payload_json, reason
                    FROM review_actions
                    ORDER BY created_at DESC
                    LIMIT 1
                    """
                )
                self.assertEqual(review["final_action"], "pending_review")
                self.assertEqual(review["final_target_note_id"], "note_existing")
                self.assertEqual(review["suggested_action"], "append_note")
                self.assertEqual(review["suggested_target_note_id"], "note_existing")
                self.assertIn("\"note_id\": \"note_existing\"", review["suggested_payload_json"])
                self.assertEqual(review["reason"], "high_similarity_low_confidence")
                task_row = db.fetchone("SELECT status FROM tasks LIMIT 1")
                self.assertEqual(task_row["status"], "flagged")
                trace_rows = db.fetchall("SELECT trace_id FROM agent_traces")
                replay_rows = db.fetchall("SELECT trace_id FROM replay_bundles")
                self.assertEqual(len(trace_rows), 1)
                self.assertEqual(len(replay_rows), 1)
                note_rows = db.fetchall("SELECT note_id FROM notes WHERE note_type = 'atomic'")
                self.assertEqual(len(note_rows), 1)
            finally:
                db.close()

    def test_review_approve_commits_append_action(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            transcripts = root / "sessions"
            transcripts.mkdir(parents=True)
            config_path = root / "config.yaml"
            _write_config(config_path, transcripts, parser_version="legacy", min_confidence_for_append=0.95)
            _write_transcript(
                transcripts / "session.jsonl",
                "How should I design an agent runtime with guarded side effects?",
                (
                    "Use a state machine, keep side effects behind proposals, add guardrails before tool "
                    "execution, and persist replay bundles so the run stays debuggable while the vault remains safe."
                ),
                duplicate_task_complete=True,
            )
            config, db, vault, worker = build_runtime(str(config_path))
            try:
                existing_path, content_hash = vault.write_new_note(
                    note_id="note_existing",
                    title="How should I design an agent runtime with guarded side effects",
                    content="## Summary\nExisting note body.",
                    tags=["agent"],
                    topics=["runtime"],
                    source_event_ids=["evt_seed"],
                    status="approved",
                )
                db.execute(
                    """
                    INSERT INTO notes (note_id, note_type, title, vault_path, content_hash, status, metadata_json, created_at, updated_at)
                    VALUES (?, 'atomic', ?, ?, ?, 'approved', '{}', '2026-03-07T00:00:00+00:00', '2026-03-07T00:00:00+00:00')
                    """,
                    (
                        "note_existing",
                        "How should I design an agent runtime with guarded side effects",
                        str(existing_path.resolve()),
                        content_hash,
                    ),
                )
                db.commit()
                result = worker.run_once()
                self.assertIsNotNone(result)
                self.assertEqual(result.state.value, "flagged")

                review = db.fetchone(
                    """
                    SELECT review_id, final_action, final_target_note_id, suggested_action,
                           suggested_target_note_id, suggested_payload_json
                    FROM review_actions
                    ORDER BY created_at DESC
                    LIMIT 1
                    """
                )
                self.assertEqual(review["final_action"], "pending_review")
                self.assertEqual(review["suggested_action"], "append_note")
                self.assertEqual(review["suggested_target_note_id"], "note_existing")
                self.assertIn("\"note_id\": \"note_existing\"", review["suggested_payload_json"])
                ok, detail = approve_review(db, vault, config, review["review_id"], reviewer="tester")
                self.assertTrue(ok, detail)
                self.assertEqual(detail, "note_existing")

                resolved_review = db.fetchone(
                    """
                    SELECT final_action, final_target_note_id, reviewer
                    FROM review_actions
                    WHERE review_id = ?
                    """,
                    (review["review_id"],),
                )
                self.assertEqual(resolved_review["final_action"], "approve_append")
                self.assertEqual(resolved_review["final_target_note_id"], "note_existing")
                self.assertEqual(resolved_review["reviewer"], "tester")

                proposal = db.fetchone(
                    """
                    SELECT action_type, status, target_note_id
                    FROM action_proposals
                    WHERE turn_id = 'turn_001'
                    ORDER BY created_at DESC
                    LIMIT 1
                    """
                )
                self.assertEqual(proposal["action_type"], "append_note")
                self.assertEqual(proposal["status"], "committed")
                self.assertEqual(proposal["target_note_id"], "note_existing")

                note_row = db.fetchone(
                    "SELECT status FROM notes WHERE note_id = 'note_existing'"
                )
                self.assertEqual(note_row["status"], "approved")
                updated = existing_path.read_text(encoding="utf-8")
                self.assertIn("## Updates", updated)
                self.assertIn("status: approved", updated)

                memory_row = db.fetchone(
                    """
                    SELECT final_decision
                    FROM session_turns
                    WHERE conversation_id = 'conv_123' AND turn_id = 'turn_001'
                    """
                )
                self.assertEqual(memory_row["final_decision"], "append_note")
            finally:
                db.close()

    def test_worker_uses_session_memory_to_skip_duplicate_write(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            transcripts = root / "sessions"
            transcripts.mkdir(parents=True)
            config_path = root / "config.yaml"
            _write_config(config_path, transcripts)
            _write_transcript(
                transcripts / "session.jsonl",
                "How should I design an agent runtime with guarded side effects?",
                (
                    "Use a state machine, keep side effects behind proposals, add guardrails before tool "
                    "execution, and persist replay bundles so the run is debuggable over time."
                ),
                conversation_id="conv_repeat",
                turn_id="turn_002",
            )
            config, db, vault, worker = build_runtime(str(config_path))
            try:
                existing_path, content_hash = vault.write_new_note(
                    note_id="note_existing",
                    title="How should I design an agent runtime with guarded side effects",
                    content="## Summary\nExisting note body.",
                    tags=["agent"],
                    topics=["runtime"],
                    source_event_ids=["evt_seed"],
                )
                db.execute(
                    """
                    INSERT INTO notes (note_id, note_type, title, vault_path, content_hash, status, metadata_json, created_at, updated_at)
                    VALUES (?, 'atomic', ?, ?, ?, 'approved', '{}', '2026-03-07T00:00:00+00:00', '2026-03-07T00:00:00+00:00')
                    """,
                    (
                        "note_existing",
                        "How should I design an agent runtime with guarded side effects",
                        str(existing_path.resolve()),
                        content_hash,
                    ),
                )
                db.execute(
                    """
                    INSERT INTO session_turns (conversation_id, turn_id, processed_at, final_decision)
                    VALUES ('conv_repeat', 'turn_001', '2026-03-07T00:00:00+00:00', 'create_note')
                    """
                )
                db.execute(
                    """
                    INSERT INTO session_note_actions (
                      conversation_id, turn_id, note_id, action_type, note_title, created_at
                    ) VALUES ('conv_repeat', 'turn_001', 'note_existing', 'create_note', ?, '2026-03-07T00:00:00+00:00')
                    """,
                    ("How should I design an agent runtime with guarded side effects",),
                )
                db.commit()
                result = worker.run_once()
                self.assertIsNotNone(result)
                self.assertEqual(result.state.value, "completed")
                trace = db.fetchone(
                    """
                    SELECT final_decision
                    FROM agent_traces
                    ORDER BY created_at DESC
                    LIMIT 1
                    """
                )
                self.assertEqual(trace["final_decision"], "skip")
                updated = existing_path.read_text(encoding="utf-8")
                self.assertNotIn("## Updates", updated)
                review_count = db.fetchone("SELECT COUNT(*) AS count FROM review_actions")
                self.assertEqual(review_count["count"], 0)
                note_rows = db.fetchall("SELECT note_id FROM notes WHERE note_type = 'atomic'")
                self.assertEqual(len(note_rows), 1)
            finally:
                db.close()

    def test_worker_processes_current_transcript_format(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            transcripts = root / "sessions"
            transcripts.mkdir(parents=True)
            config_path = root / "config.yaml"
            _write_config(config_path, transcripts)
            _write_current_transcript(
                transcripts / "session.jsonl",
                "How should transcript intake recover from parser schema drift?",
                (
                    "Read final answers from the current payload shape, fall back to task_complete when needed, "
                    "and reparse stale rows so already-seen turns can still enter the queue."
                ),
            )
            config, db, vault, worker = build_runtime(str(config_path))
            try:
                result = worker.run_once()
                self.assertIsNotNone(result)
                self.assertEqual(result.state.value, "completed")
                task_count = db.fetchone("SELECT COUNT(*) AS count FROM tasks")
                self.assertEqual(task_count["count"], 1)
            finally:
                db.close()

    def test_worker_repairs_stale_partial_event_and_requeues(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            transcripts = root / "sessions"
            transcripts.mkdir(parents=True)
            config_path = root / "config.yaml"
            transcript_path = transcripts / "session.jsonl"
            answer = (
                "Read final answers from payload.phase, keep a task_complete fallback, and refresh stale intake "
                "rows so repaired turns can be queued without wiping the database."
            )
            _write_config(config_path, transcripts)
            _write_current_transcript(
                transcript_path,
                "How do I repair stale intake rows after a parser bug?",
                answer,
            )
            config, db, vault, worker = build_runtime(str(config_path))
            try:
                db.execute(
                    """
                    INSERT INTO conversation_events (
                      event_id, turn_id, conversation_id, session_file, user_message,
                      assistant_final_answer, displayed_at, source_completeness,
                      source_confidence, parser_version, context_meta_json, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "evt_stale",
                        "turn_001",
                        "conv_123",
                        str(transcript_path),
                        "How do I repair stale intake rows after a parser bug?",
                        "",
                        "2026-03-07T04:24:23.100Z",
                        "partial",
                        0.3,
                        "v1",
                        "{}",
                        "{}",
                    ),
                )
                stat = transcript_path.stat()
                db.upsert_cursor(str(transcript_path), stat.st_mtime, "2026-03-07T04:24:30+00:00")
                db.commit()

                result = worker.run_once()
                self.assertIsNotNone(result)
                self.assertEqual(result.state.value, "completed")
                repaired = db.fetchone(
                    """
                    SELECT assistant_final_answer, source_confidence
                    FROM conversation_events
                    WHERE event_id = 'evt_stale'
                    """
                )
                self.assertIn("payload.phase", repaired["assistant_final_answer"])
                self.assertGreater(repaired["source_confidence"], 0.8)
                task_count = db.fetchone("SELECT COUNT(*) AS count FROM tasks")
                self.assertEqual(task_count["count"], 1)
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main()
