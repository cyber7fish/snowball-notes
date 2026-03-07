from __future__ import annotations

from pathlib import Path

from ..models import ReconcileReport


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
    return ReconcileReport(
        orphan_files=sorted(vault_files - db_files),
        missing_files=sorted(db_files - vault_files),
    )

