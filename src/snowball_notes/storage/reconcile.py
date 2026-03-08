from __future__ import annotations

from ..models import ReconcileReport
from .audit import write_audit_log


def reconcile_vault_and_db(vault, db) -> ReconcileReport:
    promoted_count = promote_auto_approved_notes(vault, db)
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
    )
    write_audit_log(
        db,
        "reconcile_completed",
        {
            "vault_root": str(vault.root.resolve()),
            "orphan_count": len(report.orphan_files),
            "missing_count": len(report.missing_files),
            "promoted_auto_approved": promoted_count,
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
