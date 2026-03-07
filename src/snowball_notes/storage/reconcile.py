from __future__ import annotations

from pathlib import Path

from ..models import ReconcileReport
from .audit import write_audit_log


def reconcile_vault_and_db(vault_path: Path, db) -> ReconcileReport:
    vault_files = {
        str(path.resolve())
        for path in vault_path.rglob("*.md")
        if not path.name.startswith(".")
    }
    db_files = {
        row["vault_path"]
        for row in db.fetchall("SELECT vault_path FROM notes WHERE status != 'deleted'")
    }
    report = ReconcileReport(
        orphan_files=sorted(vault_files - db_files),
        missing_files=sorted(db_files - vault_files),
    )
    write_audit_log(
        db,
        "reconcile_completed",
        {
            "vault_root": str(vault_path.resolve()),
            "orphan_count": len(report.orphan_files),
            "missing_count": len(report.missing_files),
        },
    )
    if not report.ok():
        write_audit_log(db, "reconcile_mismatch", report.to_dict(), level="warn")
    return report
