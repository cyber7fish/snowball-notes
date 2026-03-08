import io
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from snowball_notes.cli import build_runtime, main
from snowball_notes.review.cli import approve_review


def _write_config(
    path: Path,
    transcript_dir: Path,
    parser_version: str = "v1",
    min_confidence_for_append: float = 0.85,
    intake_mode: str = "transcript_poll",
    cli_wrap_file: str | None = None,
    reconcile_run_on_startup: bool = True,
    reconcile_schedule_cron: str = "0 3 * * *",
) -> None:
    cli_wrap_value = "null" if cli_wrap_file is None else f"\"{cli_wrap_file}\""
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
                f"  mode: \"{intake_mode}\"",
                f"  transcript_dir: \"{transcript_dir}\"",
                f"  cli_wrap_file: {cli_wrap_value}",
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
                "reconcile:",
                "  enabled: true",
                f"  run_on_startup: {'true' if reconcile_run_on_startup else 'false'}",
                f"  schedule_cron: \"{reconcile_schedule_cron}\"",
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
    def test_status_command_does_not_require_model_api_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            transcripts = root / "sessions"
            transcripts.mkdir(parents=True)
            config_path = root / "config.yaml"
            _write_config(config_path, transcripts)
            config_path.write_text(
                config_path.read_text(encoding="utf-8").replace('  model: "heuristic-v1"', '  provider: "deepseek_v3"\n  model: "deepseek-chat"'),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with mock.patch("sys.stdout", stdout):
                exit_code = main(["--config", str(config_path), "status"])
            self.assertEqual(exit_code, 0)
            self.assertIn("Snowball Status", stdout.getvalue())

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

    def test_worker_runs_scheduled_reconcile_once_after_daily_slot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            transcripts = root / "sessions"
            transcripts.mkdir(parents=True)
            config_path = root / "config.yaml"
            _write_config(
                config_path,
                transcripts,
                reconcile_run_on_startup=False,
                reconcile_schedule_cron="0 3 * * *",
            )
            config, db, vault, worker = build_runtime(str(config_path))
            try:
                with mock.patch(
                    "snowball_notes.agent.orchestrator.now_utc",
                    return_value=datetime(2026, 3, 8, 2, 0, tzinfo=timezone.utc),
                ):
                    self.assertIsNone(worker.run_once())
                initial_count = db.fetchone(
                    """
                    SELECT COUNT(*) AS count
                    FROM audit_logs
                    WHERE event_type = 'reconcile_completed'
                    """
                )
                self.assertEqual(initial_count["count"], 0)

                with mock.patch(
                    "snowball_notes.agent.orchestrator.now_utc",
                    return_value=datetime(2026, 3, 8, 4, 0, tzinfo=timezone.utc),
                ):
                    self.assertIsNone(worker.run_once())
                after_first_slot = db.fetchone(
                    """
                    SELECT COUNT(*) AS count
                    FROM audit_logs
                    WHERE event_type = 'reconcile_completed'
                    """
                )
                self.assertEqual(after_first_slot["count"], 1)

                with mock.patch(
                    "snowball_notes.agent.orchestrator.now_utc",
                    return_value=datetime(2026, 3, 8, 5, 0, tzinfo=timezone.utc),
                ):
                    self.assertIsNone(worker.run_once())
                same_day_count = db.fetchone(
                    """
                    SELECT COUNT(*) AS count
                    FROM audit_logs
                    WHERE event_type = 'reconcile_completed'
                    """
                )
                self.assertEqual(same_day_count["count"], 1)

                with mock.patch(
                    "snowball_notes.agent.orchestrator.now_utc",
                    return_value=datetime(2026, 3, 9, 4, 0, tzinfo=timezone.utc),
                ):
                    self.assertIsNone(worker.run_once())
                next_day_count = db.fetchone(
                    """
                    SELECT COUNT(*) AS count
                    FROM audit_logs
                    WHERE event_type = 'reconcile_completed'
                    """
                )
                self.assertEqual(next_day_count["count"], 2)
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
                self.assertIn("Knowledge/Atomic", note_rows[0]["vault_path"])
                note_status = db.fetchone("SELECT status FROM notes WHERE note_id = ?", (note_rows[0]["note_id"],))
                self.assertEqual(note_status["status"], "approved")
                trace_rows = db.fetchall("SELECT trace_id FROM agent_traces")
                replay_rows = db.fetchall("SELECT trace_id FROM replay_bundles")
                self.assertEqual(len(trace_rows), 1)
                self.assertEqual(len(replay_rows), 1)
            finally:
                db.close()

    def test_worker_archives_project_meta_turn_without_creating_note(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            transcripts = root / "sessions"
            transcripts.mkdir(parents=True)
            config_path = root / "config.yaml"
            _write_config(config_path, transcripts)
            _write_transcript(
                transcripts / "session.jsonl",
                "可以告诉我现在这个项目进行到哪步了吗？和 snowball-notes-final.md 设计的还有哪部分没做完",
                (
                    "Phase 1 和 Phase 2 的主干已经完成，Phase 3 的 eval 和 Phase 4 的 review UI 刚补齐。"
                    "剩下的主要差距是 embedding provider 联调、更完整的 eval dataset，以及一些 demo 和文档打磨。"
                    "这个回答属于项目进度同步，不应该沉淀成长期知识 note。"
                ),
            )
            config, db, vault, worker = build_runtime(str(config_path))
            try:
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
                self.assertEqual(trace["final_decision"], "archive_turn")
                note_count = db.fetchone(
                    "SELECT COUNT(*) AS count FROM notes WHERE note_type = 'atomic' AND status != 'deleted'"
                )
                self.assertEqual(note_count["count"], 0)
                archive_count = db.fetchone(
                    "SELECT COUNT(*) AS count FROM notes WHERE note_type = 'archive' AND status != 'deleted'"
                )
                self.assertEqual(archive_count["count"], 1)
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

    def test_worker_links_two_existing_notes_when_turn_is_explicitly_relational(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            transcripts = root / "sessions"
            transcripts.mkdir(parents=True)
            config_path = root / "config.yaml"
            _write_config(config_path, transcripts)
            _write_transcript(
                transcripts / "session.jsonl",
                "How should I link the Agent Runtime State Machine note with the Guarded Side Effects note?",
                (
                    "Link these notes because the runtime state machine depends on guarded side effects. "
                    "Readers should be able to navigate between them in both directions."
                ),
            )
            config, db, vault, worker = build_runtime(str(config_path))
            try:
                source_path, source_hash = vault.write_new_note(
                    note_id="note_runtime",
                    title="Agent Runtime State Machine",
                    content="## Summary\nSeparate plan and commit phases.",
                    tags=["agent"],
                    topics=["runtime"],
                    source_event_ids=["evt_seed_runtime"],
                    status="approved",
                )
                target_path, target_hash = vault.write_new_note(
                    note_id="note_guardrails",
                    title="Guarded Side Effects",
                    content="## Summary\nProposals should stay side-effect free until commit time.",
                    tags=["agent"],
                    topics=["safety"],
                    source_event_ids=["evt_seed_guardrails"],
                    status="approved",
                )
                for note_id, title, note_path, content_hash in [
                    ("note_runtime", "Agent Runtime State Machine", source_path, source_hash),
                    ("note_guardrails", "Guarded Side Effects", target_path, target_hash),
                ]:
                    db.execute(
                        """
                        INSERT INTO notes (note_id, note_type, title, vault_path, content_hash, status, metadata_json, created_at, updated_at)
                        VALUES (?, 'atomic', ?, ?, ?, 'approved', '{}', '2026-03-07T00:00:00+00:00', '2026-03-07T00:00:00+00:00')
                        """,
                        (note_id, title, str(note_path.resolve()), content_hash),
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
                self.assertEqual(trace["final_decision"], "link_notes")

                proposal = db.fetchone(
                    """
                    SELECT action_type, status, payload_json
                    FROM action_proposals
                    ORDER BY created_at DESC
                    LIMIT 1
                    """
                )
                self.assertEqual(proposal["action_type"], "link_notes")
                self.assertEqual(proposal["status"], "committed")
                self.assertIn("\"source_note_id\": \"note_runtime\"", proposal["payload_json"])
                self.assertIn("\"target_note_id\": \"note_guardrails\"", proposal["payload_json"])

                runtime_note = source_path.read_text(encoding="utf-8")
                guardrails_note = target_path.read_text(encoding="utf-8")
                self.assertIn("[[Guarded Side Effects]]", runtime_note)
                self.assertIn("[[Agent Runtime State Machine]]", guardrails_note)
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
                self.assertIn('status: "approved"', updated)

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

    def test_worker_transcript_watch_detects_new_file_after_boot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            transcripts = root / "sessions"
            transcripts.mkdir(parents=True)
            config_path = root / "config.yaml"
            _write_config(config_path, transcripts, intake_mode="transcript_watch")
            config, db, vault, worker = build_runtime(str(config_path))
            try:
                self.assertIsNone(worker.run_once())

                _write_transcript(
                    transcripts / "session.jsonl",
                    "How should transcript watch mode detect new session files?",
                    (
                        "Keep an in-memory snapshot of seen files, reparse new or changed transcripts, "
                        "and still rely on queue dedupe for safety."
                    ),
                )
                result = worker.run_once()
                self.assertIsNotNone(result)
                self.assertEqual(result.state.value, "completed")
                task_count = db.fetchone("SELECT COUNT(*) AS count FROM tasks")
                self.assertEqual(task_count["count"], 1)
            finally:
                db.close()

    def test_worker_cli_wrap_mode_reads_configured_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            transcripts = root / "sessions"
            transcripts.mkdir(parents=True)
            wrapped_dir = root / "wrapped"
            wrapped_dir.mkdir(parents=True)
            wrapped_path = wrapped_dir / "current.jsonl"
            config_path = root / "config.yaml"
            _write_config(
                config_path,
                transcripts,
                intake_mode="cli_wrap",
                cli_wrap_file="./wrapped/current.jsonl",
            )
            _write_current_transcript(
                wrapped_path,
                "How should cli wrap mode ingest the current transcript file?",
                (
                    "Point intake at the wrapped transcript path and apply the same parser and queue flow "
                    "without scanning the whole sessions tree."
                ),
            )
            config, db, vault, worker = build_runtime(str(config_path))
            try:
                result = worker.run_once()
                self.assertIsNotNone(result)
                self.assertEqual(result.state.value, "completed")
                event = db.fetchone("SELECT session_file FROM conversation_events LIMIT 1")
                self.assertEqual(Path(event["session_file"]).resolve(), wrapped_path.resolve())
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

    def test_reconcile_promotes_auto_approved_notes_from_inbox_to_knowledge(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            transcripts = root / "sessions"
            transcripts.mkdir(parents=True)
            config_path = root / "config.yaml"
            _write_config(config_path, transcripts, reconcile_run_on_startup=False)
            config, db, vault, _ = build_runtime(str(config_path), build_worker=False)
            try:
                legacy_path, content_hash = vault.write_new_note(
                    note_id="note_legacy",
                    title="Legacy Auto Approved Note",
                    content="## Summary\nThis note should be promoted out of Inbox.",
                    tags=["agent"],
                    topics=["runtime"],
                    source_event_ids=["evt_legacy"],
                    status="pending_review",
                )
                db.execute(
                    """
                    INSERT INTO notes (note_id, note_type, title, vault_path, content_hash, status, metadata_json, created_at, updated_at)
                    VALUES (?, 'atomic', ?, ?, ?, 'pending_review', '{}', '2026-03-07T00:00:00+00:00', '2026-03-07T00:00:00+00:00')
                    """,
                    (
                        "note_legacy",
                        "Legacy Auto Approved Note",
                        str(legacy_path.resolve()),
                        content_hash,
                    ),
                )
                db.execute(
                    """
                    INSERT INTO agent_traces (
                      trace_id, turn_id, event_id, prompt_version, model_name, started_at, finished_at,
                      total_steps, exceeded_max_steps, terminal_reason, final_decision, final_confidence,
                      total_input_tokens, total_output_tokens, total_duration_ms, trace_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "trace_legacy",
                        "turn_legacy",
                        "evt_legacy",
                        "agent_system/v1.md",
                        "heuristic-v1",
                        "2026-03-07T00:00:00+00:00",
                        "2026-03-07T00:00:01+00:00",
                        3,
                        0,
                        "completed",
                        "create_note",
                        0.92,
                        120,
                        40,
                        240,
                        "{}",
                    ),
                )
                db.execute(
                    """
                    INSERT INTO action_proposals (
                      proposal_id, trace_id, turn_id, action_type, target_note_id, payload_json,
                      idempotency_key, status, created_at, committed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "proposal_legacy",
                        "trace_legacy",
                        "turn_legacy",
                        "create_note",
                        "note_legacy",
                        "{}",
                        "legacy:create:note_legacy",
                        "committed",
                        "2026-03-07T00:00:00+00:00",
                        "2026-03-07T00:00:01+00:00",
                    ),
                )
                db.commit()

                stdout = io.StringIO()
                with mock.patch("sys.stdout", stdout):
                    exit_code = main(["--config", str(config_path), "reconcile"])
                self.assertEqual(exit_code, 0)
                self.assertIn("\"promoted_auto_approved\": 1", stdout.getvalue())
                self.assertIn("\"normalized_note_files\": 1", stdout.getvalue())

                note_row = db.fetchone(
                    "SELECT status, vault_path FROM notes WHERE note_id = 'note_legacy'"
                )
                self.assertEqual(note_row["status"], "approved")
                self.assertIn("Knowledge/Atomic", note_row["vault_path"])
                self.assertFalse(legacy_path.exists())
                promoted_path = Path(note_row["vault_path"])
                self.assertTrue(promoted_path.exists())
                promoted_text = promoted_path.read_text(encoding="utf-8")
                self.assertIn('status: "approved"', promoted_text)
            finally:
                db.close()

    def test_reconcile_reports_missing_file_without_crashing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            transcripts = root / "sessions"
            transcripts.mkdir(parents=True)
            config_path = root / "config.yaml"
            _write_config(config_path, transcripts, reconcile_run_on_startup=False)
            config, db, vault, _ = build_runtime(str(config_path), build_worker=False)
            try:
                missing_path = root / "vault" / "Archive" / "Conversations" / "Conversation 019cb98c.md"
                db.execute(
                    """
                    INSERT INTO notes (note_id, note_type, title, vault_path, content_hash, status, metadata_json, created_at, updated_at)
                    VALUES (?, 'archive', ?, ?, ?, 'archived', '{}', '2026-03-07T00:00:00+00:00', '2026-03-07T00:00:00+00:00')
                    """,
                    (
                        "note_missing_archive",
                        "Conversation 019cb98c",
                        str(missing_path.resolve()),
                        "missing",
                    ),
                )
                db.commit()

                stdout = io.StringIO()
                with mock.patch("sys.stdout", stdout):
                    exit_code = main(["--config", str(config_path), "reconcile"])
                self.assertEqual(exit_code, 0)
                self.assertIn(str(missing_path.resolve()), stdout.getvalue())
            finally:
                db.close()

    def test_vault_quotes_frontmatter_and_removes_duplicate_title_heading(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            transcripts = root / "sessions"
            transcripts.mkdir(parents=True)
            config_path = root / "config.yaml"
            _write_config(config_path, transcripts, reconcile_run_on_startup=False)
            config, db, vault, _ = build_runtime(str(config_path), build_worker=False)
            try:
                title = "RuntimeError: missing API key env: DEEPSEEK_API_KEY 为什么报错了"
                path, _ = vault.write_new_note(
                    note_id="note_frontmatter",
                    title=title,
                    content=f"# {title}\n\n## Summary\nKeep keys out of notes.\n",
                    tags=["debug"],
                    topics=["secrets"],
                    source_event_ids=["evt_frontmatter"],
                    status="approved",
                )
                rendered = path.read_text(encoding="utf-8")
                self.assertIn(f'title: "{title}"', rendered)
                self.assertEqual(rendered.count(f"# {title}"), 1)
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main()
