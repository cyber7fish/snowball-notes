from __future__ import annotations

import json
from pathlib import Path

from snowball_notes.config import load_config
from snowball_notes.storage.sqlite import Database
from snowball_notes.storage.vault import Vault
from snowball_notes.utils import now_utc_iso


def main() -> int:
    root = Path(__file__).resolve().parent
    config = load_config(root / "config.yaml")
    db = Database(config.db_path)
    db.migrate()
    vault = Vault(config)
    try:
        title = "KV Cache Decode Latency"
        path, content_hash = vault.write_new_note(
            note_id="note_seed_kv_cache_review",
            title=title,
            content=(
                "## Summary\n"
                "KV cache reduces decode latency by reusing historical attention state instead of recomputing the full prefix.\n\n"
                "## Key Points\n"
                "- Prefill writes prompt-side key/value tensors into cache.\n"
                "- Decode reuses cache and only computes projections for the newest token.\n"
                "- System performance depends on memory layout and batching strategy as much as on saved compute.\n"
            ),
            tags=["llm", "inference", "kv-cache"],
            topics=["decode", "latency", "serving"],
            source_event_ids=["evt_seed_kv_cache_review"],
            status="approved",
        )
        timestamp = now_utc_iso()
        db.execute(
            """
            INSERT OR REPLACE INTO notes (
              note_id, note_type, title, vault_path, content_hash,
              status, metadata_json, created_at, updated_at
            ) VALUES (?, 'atomic', ?, ?, ?, 'approved', ?, ?, ?)
            """,
            (
                "note_seed_kv_cache_review",
                title,
                str(path.resolve()),
                content_hash,
                json.dumps(
                    {"tags": ["llm", "inference", "kv-cache"], "topics": ["decode", "latency", "serving"]},
                    ensure_ascii=False,
                ),
                timestamp,
                timestamp,
            ),
        )
        db.commit()
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
