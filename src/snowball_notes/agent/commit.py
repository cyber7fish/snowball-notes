from __future__ import annotations

import json
from pathlib import Path

from ..config import SnowballConfig
from ..models import CommitResult
from ..storage.audit import write_audit_log
from ..utils import new_id, now_utc_iso


class Committer:
    def __init__(self, db, vault, state, config: SnowballConfig):
        self.db = db
        self.vault = vault
        self.state = state
        self.config = config

    def validate(self) -> list[str]:
        errors = []
        proposals = self.state.proposals
        write_proposals = [proposal for proposal in proposals if proposal.action_type != "archive_turn"]
        if len(write_proposals) > self.config.agent.max_writes_per_run:
            errors.append("too many write proposals in a single run")
        if self.state.event.source_confidence < self.config.guardrails.min_confidence_for_note:
            if any(proposal.action_type == "create_note" for proposal in proposals):
                errors.append("create_note blocked by source_confidence")
        touched_notes = set()
        for proposal in proposals:
            if proposal.action_type == "link_notes":
                source_note_id = proposal.payload.get("source_note_id")
                target_note_id = proposal.payload.get("target_note_id")
                if not isinstance(source_note_id, str) or not source_note_id:
                    errors.append("link_notes missing source_note_id")
                    continue
                if not isinstance(target_note_id, str) or not target_note_id:
                    errors.append("link_notes missing target_note_id")
                    continue
                if source_note_id == target_note_id:
                    errors.append("link_notes requires two distinct notes")
                    continue
                for note_id in (source_note_id, target_note_id):
                    if note_id in touched_notes:
                        errors.append(f"duplicate proposal target {note_id}")
                    touched_notes.add(note_id)
                    note_row = self.db.fetchone(
                        "SELECT vault_path FROM notes WHERE note_id = ?",
                        (note_id,),
                    )
                    if note_row is None:
                        errors.append(f"link target missing: {note_id}")
                continue
            if proposal.target_note_id:
                if proposal.target_note_id in touched_notes:
                    errors.append(f"duplicate proposal target {proposal.target_note_id}")
                touched_notes.add(proposal.target_note_id)
                note_row = self.db.fetchone(
                    "SELECT vault_path FROM notes WHERE note_id = ?",
                    (proposal.target_note_id,),
                )
                if proposal.action_type == "append_note" and note_row is None:
                    errors.append(f"append target missing: {proposal.target_note_id}")
        return errors

    def commit(self) -> CommitResult:
        committed_note_ids = []
        try:
            with self.db.transaction():
                for proposal in self.state.proposals:
                    note_id = self._commit_proposal(proposal)
                    committed_note_ids.append(note_id)
                    proposal.status = "committed"
                    proposal.committed_at = now_utc_iso()
                    self.db.execute(
                        """
                        UPDATE action_proposals
                        SET status = 'committed', committed_at = ?, target_note_id = ?
                        WHERE proposal_id = ?
                        """,
                        (proposal.committed_at, proposal.target_note_id, proposal.proposal_id),
                    )
            return CommitResult.completed(committed_note_ids)
        except OSError as exc:
            write_audit_log(self.db, "commit_vault_error", {"error": str(exc)}, level="error")
            return CommitResult.retryable(str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            write_audit_log(self.db, "commit_fatal_error", {"error": str(exc)}, level="error")
            return CommitResult.fatal(str(exc))

    def _commit_proposal(self, proposal) -> str:
        payload = proposal.payload
        if proposal.action_type == "create_note":
            note_id = new_id("note")
            path, content_hash = self.vault.write_new_note(
                note_id=note_id,
                title=payload["title"],
                content=payload["content"],
                tags=payload.get("tags", []),
                topics=payload.get("topics", []),
                source_event_ids=[payload["source_event_id"]],
            )
            metadata = {"tags": payload.get("tags", []), "topics": payload.get("topics", [])}
            self.db.execute(
                """
                INSERT INTO notes (note_id, note_type, title, vault_path, content_hash, status, metadata_json, created_at, updated_at)
                VALUES (?, 'atomic', ?, ?, ?, 'pending_review', ?, ?, ?)
                """,
                (
                    note_id,
                    payload["title"],
                    str(path.resolve()),
                    content_hash,
                    json.dumps(metadata, ensure_ascii=False),
                    now_utc_iso(),
                    now_utc_iso(),
                ),
            )
            self.db.execute(
                "INSERT INTO note_sources (note_id, event_id, relation_type) VALUES (?, ?, 'derived_from')",
                (note_id, self.state.event.event_id),
            )
            proposal.target_note_id = note_id
            return note_id
        if proposal.action_type == "append_note":
            row = self.db.fetchone(
                "SELECT vault_path, title, metadata_json FROM notes WHERE note_id = ?",
                (proposal.target_note_id,),
            )
            if row is None:
                raise RuntimeError(f"missing note {proposal.target_note_id}")
            note_path = Path(row["vault_path"])
            content_hash = self.vault.append_to_updates_section(
                note_path, payload["content"], payload["source_turn_id"]
            )
            self.db.execute(
                """
                UPDATE notes
                SET content_hash = ?, updated_at = ?, status = 'pending_review'
                WHERE note_id = ?
                """,
                (content_hash, now_utc_iso(), proposal.target_note_id),
            )
            self.db.execute(
                "INSERT OR IGNORE INTO note_sources (note_id, event_id, relation_type) VALUES (?, ?, 'appended_from')",
                (proposal.target_note_id, self.state.event.event_id),
            )
            return proposal.target_note_id
        if proposal.action_type == "link_notes":
            source_note_id = payload["source_note_id"]
            target_note_id = payload["target_note_id"]
            source_row = self.db.fetchone(
                "SELECT vault_path, title FROM notes WHERE note_id = ?",
                (source_note_id,),
            )
            target_row = self.db.fetchone(
                "SELECT vault_path, title FROM notes WHERE note_id = ?",
                (target_note_id,),
            )
            if source_row is None:
                raise RuntimeError(f"missing note {source_note_id}")
            if target_row is None:
                raise RuntimeError(f"missing note {target_note_id}")
            source_hash, target_hash = self.vault.add_bidirectional_link(
                source_row["vault_path"],
                source_row["title"],
                target_row["vault_path"],
                target_row["title"],
            )
            timestamp = now_utc_iso()
            self.db.execute(
                """
                UPDATE notes
                SET content_hash = ?, updated_at = ?, status = 'pending_review'
                WHERE note_id = ?
                """,
                (source_hash, timestamp, source_note_id),
            )
            self.db.execute(
                """
                UPDATE notes
                SET content_hash = ?, updated_at = ?, status = 'pending_review'
                WHERE note_id = ?
                """,
                (target_hash, timestamp, target_note_id),
            )
            self.db.execute(
                "INSERT OR IGNORE INTO note_sources (note_id, event_id, relation_type) VALUES (?, ?, 'linked_from')",
                (source_note_id, self.state.event.event_id),
            )
            self.db.execute(
                "INSERT OR IGNORE INTO note_sources (note_id, event_id, relation_type) VALUES (?, ?, 'linked_from')",
                (target_note_id, self.state.event.event_id),
            )
            proposal.target_note_id = source_note_id
            return source_note_id
        if proposal.action_type == "archive_turn":
            note_id = new_id("archive")
            path, content_hash = self.vault.write_archive_note(note_id, payload)
            self.db.execute(
                """
                INSERT INTO notes (note_id, note_type, title, vault_path, content_hash, status, metadata_json, created_at, updated_at)
                VALUES (?, 'archive', ?, ?, ?, 'archived', '{}', ?, ?)
                """,
                (
                    note_id,
                    payload["title"],
                    str(path.resolve()),
                    content_hash,
                    now_utc_iso(),
                    now_utc_iso(),
                ),
            )
            self.db.execute(
                "INSERT INTO note_sources (note_id, event_id, relation_type) VALUES (?, ?, 'archived_from')",
                (note_id, self.state.event.event_id),
            )
            proposal.target_note_id = note_id
            return note_id
        raise ValueError(f"unknown action_type {proposal.action_type}")
