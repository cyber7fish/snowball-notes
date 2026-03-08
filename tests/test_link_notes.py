import tempfile
import unittest
from pathlib import Path

from snowball_notes.agent.commit import Committer
from snowball_notes.agent.state import AgentState
from snowball_notes.cli import build_runtime
from snowball_notes.models import ActionProposal, SessionMemory, StandardEvent


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
                "  max_writes_per_run: 2",
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


class LinkNotesTests(unittest.TestCase):
    def test_committer_links_two_existing_notes_bidirectionally(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.yaml"
            _write_config(config_path)
            config, db, vault, _ = build_runtime(str(config_path), build_worker=False)
            try:
                source_path, source_hash = vault.write_new_note(
                    note_id="note_source",
                    title="Agent Runtime State Machine",
                    content="## Summary\nSeparate planning from commits.",
                    tags=["agent"],
                    topics=["runtime"],
                    source_event_ids=["evt_seed_source"],
                    status="approved",
                )
                target_path, target_hash = vault.write_new_note(
                    note_id="note_target",
                    title="Guarded Side Effects",
                    content="## Summary\nSide effects should be deferred behind proposals.",
                    tags=["agent"],
                    topics=["safety"],
                    source_event_ids=["evt_seed_target"],
                    status="approved",
                )
                for note_id, title, note_path, content_hash in [
                    ("note_source", "Agent Runtime State Machine", source_path, source_hash),
                    ("note_target", "Guarded Side Effects", target_path, target_hash),
                ]:
                    db.execute(
                        """
                        INSERT INTO notes (
                          note_id, note_type, title, vault_path, content_hash, status,
                          metadata_json, created_at, updated_at
                        ) VALUES (?, 'atomic', ?, ?, ?, 'approved', '{}', '2026-03-08T00:00:00+00:00', '2026-03-08T00:00:00+00:00')
                        """,
                        (note_id, title, str(note_path.resolve()), content_hash),
                    )
                db.commit()

                event = StandardEvent(
                    event_id="evt_link_1",
                    session_file="/tmp/link/session.jsonl",
                    conversation_id="conv_link_1",
                    turn_id="turn_link_1",
                    user_message="Link the runtime and guardrail notes.",
                    assistant_final_answer="These notes should reference each other because they form one design pattern.",
                    displayed_at="2026-03-08T00:00:00+00:00",
                    source_completeness="full",
                    source_confidence=0.96,
                    parser_version="v1",
                    context_meta={"cwd": "/tmp/project", "client": "codex"},
                )
                state = AgentState(
                    event=event,
                    task_id="task_link_1",
                    trace_id="trace_link_1",
                    session_memory=SessionMemory(conversation_id=event.conversation_id),
                )
                proposal = ActionProposal(
                    proposal_id="proposal_link_1",
                    trace_id=state.trace_id,
                    turn_id=event.turn_id,
                    action_type="link_notes",
                    target_note_id="note_source",
                    payload={
                        "source_note_id": "note_source",
                        "target_note_id": "note_target",
                        "source_event_id": event.event_id,
                    },
                    idempotency_key="link:note_source:note_target",
                )
                state.proposals.append(proposal)

                committer = Committer(db, vault, state, config)
                self.assertEqual(committer.validate(), [])
                result = committer.commit()
                self.assertTrue(result.success)
                self.assertEqual(result.committed_note_ids, ["note_source"])

                source_text = source_path.read_text(encoding="utf-8")
                target_text = target_path.read_text(encoding="utf-8")
                self.assertIn("## Links", source_text)
                self.assertIn("[[Guarded Side Effects]]", source_text)
                self.assertIn("## Links", target_text)
                self.assertIn("[[Agent Runtime State Machine]]", target_text)

                status_rows = db.fetchall(
                    "SELECT note_id, status FROM notes WHERE note_id IN ('note_source', 'note_target') ORDER BY note_id ASC"
                )
                self.assertEqual([row["status"] for row in status_rows], ["approved", "approved"])
                source_count = db.fetchone(
                    """
                    SELECT COUNT(*) AS count
                    FROM note_sources
                    WHERE event_id = 'evt_link_1' AND relation_type = 'linked_from'
                    """
                )
                self.assertEqual(source_count["count"], 2)
            finally:
                db.close()
