import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from snowball_notes.agent.commit import Committer
from snowball_notes.agent.state import AgentState
from snowball_notes.config import default_config
from snowball_notes.models import ActionProposal, SessionMemory, StandardEvent
from snowball_notes.storage.sqlite import Database
from snowball_notes.storage.vault import Vault
from snowball_notes.utils import new_id, now_utc_iso, sha256_text


def _sample_event(**overrides) -> StandardEvent:
    defaults = dict(
        event_id="evt_commit_test",
        session_file="/tmp/session.jsonl",
        conversation_id="conv_commit_test",
        turn_id="turn_commit_test",
        user_message="How does X work?",
        assistant_final_answer="X works by doing Y and Z. " * 10,
        displayed_at="2026-03-08T00:00:00+00:00",
        source_completeness="full",
        source_confidence=0.95,
        parser_version="v1",
        context_meta={},
    )
    defaults.update(overrides)
    return StandardEvent(**defaults)


def _make_state(event=None, **overrides):
    event = event or _sample_event()
    defaults = dict(
        event=event,
        task_id="task_commit_test",
        trace_id="trace_commit_test",
        session_memory=SessionMemory(conversation_id=event.conversation_id),
    )
    defaults.update(overrides)
    return AgentState(**defaults)


def _create_proposal(state, action_type, **payload_overrides):
    if action_type == "create_note":
        payload = {
            "title": "Test Note",
            "content": "Some content.",
            "tags": ["test"],
            "topics": ["testing"],
            "source_event_id": state.event.event_id,
        }
        payload.update(payload_overrides)
        return ActionProposal(
            proposal_id=new_id("proposal"),
            trace_id=state.trace_id,
            turn_id=state.event.turn_id,
            action_type="create_note",
            target_note_id=None,
            payload=payload,
            idempotency_key=f"create:{state.event.turn_id}:{sha256_text(payload['title'])[:8]}",
        )
    if action_type == "append_note":
        payload = {
            "content": "Additional detail.",
            "source_turn_id": state.event.turn_id,
            "source_event_id": state.event.event_id,
        }
        payload.update(payload_overrides)
        target = payload_overrides.get("_target_note_id", "note_existing")
        return ActionProposal(
            proposal_id=new_id("proposal"),
            trace_id=state.trace_id,
            turn_id=state.event.turn_id,
            action_type="append_note",
            target_note_id=target,
            payload=payload,
            idempotency_key=f"append:{state.event.turn_id}:{target}",
        )
    if action_type == "archive_turn":
        payload = {
            "title": f"Conversation {state.event.turn_id[:8]}",
            "content": "Archive content.",
            "event_id": state.event.event_id,
            "user_message": state.event.user_message,
            "assistant_final_answer": state.event.assistant_final_answer,
        }
        payload.update(payload_overrides)
        return ActionProposal(
            proposal_id=new_id("proposal"),
            trace_id=state.trace_id,
            turn_id=state.event.turn_id,
            action_type="archive_turn",
            target_note_id=None,
            payload=payload,
            idempotency_key=f"archive:{state.event.turn_id}",
        )
    if action_type == "link_notes":
        payload = {
            "source_note_id": payload_overrides.get("source_note_id", "note_a"),
            "target_note_id": payload_overrides.get("target_note_id", "note_b"),
            "source_event_id": state.event.event_id,
        }
        return ActionProposal(
            proposal_id=new_id("proposal"),
            trace_id=state.trace_id,
            turn_id=state.event.turn_id,
            action_type="link_notes",
            target_note_id=payload["source_note_id"],
            payload=payload,
            idempotency_key=f"link:{payload['source_note_id']}:{payload['target_note_id']}",
        )
    raise ValueError(f"unsupported action_type: {action_type}")


def _seed_note(db, vault, note_id, title, status="approved"):
    path, content_hash = vault.write_new_note(
        note_id=note_id,
        title=title,
        content=f"## Summary\n{title}\n",
        tags=["test"],
        status=status,
    )
    db.execute(
        """
        INSERT INTO notes (note_id, note_type, title, vault_path, content_hash,
                           status, metadata_json, created_at, updated_at)
        VALUES (?, 'atomic', ?, ?, ?, ?, '{}', ?, ?)
        """,
        (note_id, title, str(path.resolve()), content_hash, status,
         now_utc_iso(), now_utc_iso()),
    )
    db.commit()
    return path


class CommitterValidateTests(unittest.TestCase):
    def _setup(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = default_config(Path(self.temp_dir))
        self.db = Database(self.config.db_path)
        self.db.migrate()
        self.vault = Vault(self.config)
        self.state = _make_state()
        return Committer(self.db, self.vault, self.state, self.config)

    def tearDown(self):
        import shutil
        if hasattr(self, "temp_dir"):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_validate_passes_for_single_create(self):
        committer = self._setup()
        self.state.proposals.append(_create_proposal(self.state, "create_note"))
        errors = committer.validate()
        self.assertEqual(errors, [])

    def test_validate_blocks_too_many_writes(self):
        committer = self._setup()
        self.config.agent.max_writes_per_run = 1
        self.state.proposals.append(_create_proposal(self.state, "create_note", title="A"))
        self.state.proposals.append(_create_proposal(self.state, "create_note", title="B"))
        errors = committer.validate()
        self.assertTrue(any("too many" in e for e in errors))

    def test_validate_blocks_low_confidence_create(self):
        committer = self._setup()
        self.state.event = _sample_event(source_confidence=0.5)
        self.state.proposals.append(_create_proposal(self.state, "create_note"))
        errors = committer.validate()
        self.assertTrue(any("confidence" in e.lower() or "blocked" in e.lower() for e in errors))

    def test_validate_allows_archive_despite_low_confidence(self):
        committer = self._setup()
        self.state.event = _sample_event(source_confidence=0.5)
        self.state.proposals.append(_create_proposal(self.state, "archive_turn"))
        errors = committer.validate()
        self.assertEqual(errors, [])

    def test_validate_blocks_append_to_missing_note(self):
        committer = self._setup()
        self.state.proposals.append(
            _create_proposal(self.state, "append_note", _target_note_id="note_nonexistent")
        )
        errors = committer.validate()
        self.assertTrue(any("missing" in e.lower() for e in errors))

    def test_validate_blocks_duplicate_target_note(self):
        committer = self._setup()
        _seed_note(self.db, self.vault, "note_dup", "Dup Note")
        p1 = _create_proposal(self.state, "append_note", _target_note_id="note_dup")
        p2 = _create_proposal(self.state, "append_note", _target_note_id="note_dup")
        p2.idempotency_key = "append:other:note_dup"
        self.config.agent.max_writes_per_run = 10
        self.state.proposals.extend([p1, p2])
        errors = committer.validate()
        self.assertTrue(any("duplicate" in e.lower() for e in errors))

    def test_validate_blocks_link_self_referencing(self):
        committer = self._setup()
        _seed_note(self.db, self.vault, "note_self", "Self Note")
        self.state.proposals.append(
            _create_proposal(self.state, "link_notes",
                             source_note_id="note_self", target_note_id="note_self")
        )
        errors = committer.validate()
        self.assertTrue(any("distinct" in e.lower() for e in errors))

    def test_validate_blocks_link_missing_target(self):
        committer = self._setup()
        _seed_note(self.db, self.vault, "note_exists", "Exists")
        self.state.proposals.append(
            _create_proposal(self.state, "link_notes",
                             source_note_id="note_exists", target_note_id="note_ghost")
        )
        errors = committer.validate()
        self.assertTrue(any("missing" in e.lower() for e in errors))


class CommitterCommitTests(unittest.TestCase):
    def _setup(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = default_config(Path(self.temp_dir))
        self.db = Database(self.config.db_path)
        self.db.migrate()
        self.vault = Vault(self.config)
        self.state = _make_state()
        return Committer(self.db, self.vault, self.state, self.config)

    def tearDown(self):
        import shutil
        if hasattr(self, "temp_dir"):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _persist_proposals(self):
        for p in self.state.proposals:
            self.db.execute(
                """
                INSERT OR IGNORE INTO action_proposals (
                  proposal_id, trace_id, turn_id, action_type, target_note_id,
                  payload_json, idempotency_key, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (p.proposal_id, p.trace_id, p.turn_id, p.action_type,
                 p.target_note_id, json.dumps(p.payload, ensure_ascii=False),
                 p.idempotency_key, p.status, p.created_at),
            )
        self.db.commit()

    def test_commit_create_note(self):
        committer = self._setup()
        self.state.proposals.append(_create_proposal(self.state, "create_note"))
        self._persist_proposals()
        result = committer.commit()
        self.assertTrue(result.success)
        self.assertEqual(len(result.committed_note_ids), 1)
        note = self.db.fetchone(
            "SELECT * FROM notes WHERE note_id = ?",
            (result.committed_note_ids[0],),
        )
        self.assertIsNotNone(note)
        self.assertEqual(note["note_type"], "atomic")

    def test_commit_append_note(self):
        committer = self._setup()
        note_path = _seed_note(self.db, self.vault, "note_append", "Append Target")
        self.state.proposals.append(
            _create_proposal(self.state, "append_note", _target_note_id="note_append")
        )
        self._persist_proposals()
        result = committer.commit()
        self.assertTrue(result.success)
        self.assertIn("note_append", result.committed_note_ids)

    def test_commit_archive_turn(self):
        committer = self._setup()
        self.state.proposals.append(_create_proposal(self.state, "archive_turn"))
        self._persist_proposals()
        result = committer.commit()
        self.assertTrue(result.success)
        note = self.db.fetchone(
            "SELECT * FROM notes WHERE note_id = ?",
            (result.committed_note_ids[0],),
        )
        self.assertEqual(note["note_type"], "archive")

    def test_commit_link_notes(self):
        committer = self._setup()
        _seed_note(self.db, self.vault, "note_link_a", "Link A")
        _seed_note(self.db, self.vault, "note_link_b", "Link B")
        self.state.proposals.append(
            _create_proposal(self.state, "link_notes",
                             source_note_id="note_link_a", target_note_id="note_link_b")
        )
        self._persist_proposals()
        result = committer.commit()
        self.assertTrue(result.success)

    def test_commit_vault_write_error_returns_retryable(self):
        committer = self._setup()
        self.state.proposals.append(_create_proposal(self.state, "create_note"))
        self._persist_proposals()
        with patch.object(self.vault, "write_new_note", side_effect=OSError("disk full")):
            result = committer.commit()
        self.assertFalse(result.success)
        self.assertEqual(result.disposition, "retryable")

    def test_commit_unknown_action_returns_fatal(self):
        committer = self._setup()
        bad_proposal = _create_proposal(self.state, "create_note")
        bad_proposal.action_type = "unknown_action"
        self.state.proposals.append(bad_proposal)
        self._persist_proposals()
        result = committer.commit()
        self.assertFalse(result.success)
        self.assertEqual(result.disposition, "fatal")


if __name__ == "__main__":
    unittest.main()
