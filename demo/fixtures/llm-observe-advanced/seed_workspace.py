from __future__ import annotations

import json
from pathlib import Path

from snowball_notes.config import load_config
from snowball_notes.storage.sqlite import Database
from snowball_notes.storage.vault import Vault
from snowball_notes.utils import now_utc_iso


SEED_NOTES = [
    {
        "note_id": "note_seed_kv_cache",
        "title": "KV Cache Decode Latency",
        "content": (
            "## Summary\n"
            "KV cache reduces decode latency by reusing historical attention state instead of recomputing the full prefix.\n\n"
            "## Key Points\n"
            "- Prefill computes key/value tensors for the existing context once.\n"
            "- Decode reuses cached tensors and only computes projections for the newest token.\n"
            "- Real systems must balance latency gains against memory pressure and batching constraints.\n"
        ),
        "tags": ["llm", "inference", "kv-cache"],
        "topics": ["decode", "latency", "serving"],
        "source_event_ids": ["evt_seed_kv_cache"],
    },
    {
        "note_id": "note_seed_rag",
        "title": "Long-Context RAG Tradeoffs",
        "content": (
            "## Summary\n"
            "Long-context RAG systems trade retrieval precision against evidence coverage and prompt budget.\n\n"
            "## Key Points\n"
            "- Smaller chunks improve retrieval precision but fragment evidence.\n"
            "- Larger chunks preserve context but waste context window on irrelevant text.\n"
            "- Practical systems combine chunking, reranking, and synthesis-time evidence merging.\n"
        ),
        "tags": ["llm", "rag", "retrieval"],
        "topics": ["chunking", "recall", "context"],
        "source_event_ids": ["evt_seed_rag"],
    },
]


def main() -> int:
    root = Path(__file__).resolve().parent
    config = load_config(root / "config.yaml")
    db = Database(config.db_path)
    db.migrate()
    vault = Vault(config)
    try:
        for note in SEED_NOTES:
            path, content_hash = vault.write_new_note(
                note_id=note["note_id"],
                title=note["title"],
                content=note["content"],
                tags=note["tags"],
                topics=note["topics"],
                source_event_ids=note["source_event_ids"],
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
                    note["note_id"],
                    note["title"],
                    str(path.resolve()),
                    content_hash,
                    json.dumps(
                        {"tags": note["tags"], "topics": note["topics"]},
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
