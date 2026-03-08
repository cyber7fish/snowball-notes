from __future__ import annotations

import json

from ..models import ReconcileReport
from .audit import write_audit_log


def reconcile_vault_and_db(vault, db) -> ReconcileReport:
    promoted_count = promote_auto_approved_notes(vault, db)
    normalized_count = normalize_note_files(vault, db)
    vault_files = {
        str(path.resolve())
        for path in vault.root.rglob("*.md")
        if not path.name.startswith(".")
    }
    db_files = {
        row["vault_path"]
        for row in db.fetchall("SELECT vault_path FROM notes WHERE status != 'deleted'")
    }
    report = ReconcileReport(
        orphan_files=sorted(vault_files - db_files),
        missing_files=sorted(db_files - vault_files),
        promoted_auto_approved=promoted_count,
        normalized_note_files=normalized_count,
    )
    write_audit_log(
        db,
        "reconcile_completed",
        {
            "vault_root": str(vault.root.resolve()),
            "orphan_count": len(report.orphan_files),
            "missing_count": len(report.missing_files),
            "promoted_auto_approved": promoted_count,
            "normalized_note_files": normalized_count,
        },
    )
    if not report.ok():
        write_audit_log(db, "reconcile_mismatch", report.to_dict(), level="warn")
    return report


def promote_auto_approved_notes(vault, db) -> int:
    rows = db.fetchall(
        """
        SELECT DISTINCT n.note_id, n.title, n.vault_path
        FROM notes n
        JOIN action_proposals p
          ON p.target_note_id = n.note_id
         AND p.status = 'committed'
        JOIN agent_traces t
          ON t.trace_id = p.trace_id
        LEFT JOIN review_actions r
          ON r.trace_id = t.trace_id
         AND r.final_action = 'pending_review'
        WHERE n.status = 'pending_review'
          AND t.final_decision IN ('create_note', 'append_note', 'link_notes')
          AND r.review_id IS NULL
        """
    )
    promoted = 0
    for row in rows:
        promoted_path, content_hash = vault.promote_note_to_atomic(
            row["note_id"],
            row["title"],
            row["vault_path"],
        )
        db.execute(
            """
            UPDATE notes
            SET status = 'approved', vault_path = ?, content_hash = ?, updated_at = CURRENT_TIMESTAMP
            WHERE note_id = ?
            """,
            (str(promoted_path.resolve()), content_hash, row["note_id"]),
        )
        promoted += 1
    if promoted:
        write_audit_log(
            db,
            "auto_approved_notes_promoted",
            {"count": promoted},
        )
    return promoted


def normalize_note_files(vault, db) -> int:
    rows = db.fetchall(
        """
        SELECT note_id, note_type, title, vault_path, status, metadata_json, created_at, updated_at
        FROM notes
        WHERE status != 'deleted'
        """
    )
    normalized = 0
    for row in rows:
        source_rows = db.fetchall(
            """
            SELECT event_id
            FROM note_sources
            WHERE note_id = ?
            ORDER BY event_id ASC
            """,
            (row["note_id"],),
        )
        changed, content_hash = vault.normalize_note_file(
            note_type=row["note_type"],
            note_id=row["note_id"],
            title=row["title"],
            note_path=row["vault_path"],
            status=row["status"],
            metadata=json.loads(row.get("metadata_json") or "{}"),
            source_event_ids=[item["event_id"] for item in source_rows],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
        if not changed:
            continue
        db.execute(
            "UPDATE notes SET content_hash = ? WHERE note_id = ?",
            (content_hash, row["note_id"]),
        )
        normalized += 1
    if normalized:
        write_audit_log(
            db,
            "note_files_normalized",
            {"count": normalized},
        )
    return normalized
