from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from ..utils import ensure_directory


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS conversation_events (
  event_id TEXT PRIMARY KEY,
  turn_id TEXT NOT NULL,
  conversation_id TEXT NOT NULL,
  session_file TEXT NOT NULL,
  user_message TEXT NOT NULL,
  assistant_final_answer TEXT NOT NULL,
  displayed_at TEXT NOT NULL,
  source_completeness TEXT NOT NULL,
  source_confidence REAL NOT NULL,
  parser_version TEXT NOT NULL,
  context_meta_json TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_conversation_events_turn_id
  ON conversation_events(turn_id);

CREATE TABLE IF NOT EXISTS tasks (
  task_id TEXT PRIMARY KEY,
  event_id TEXT NOT NULL,
  dedupe_key TEXT NOT NULL UNIQUE,
  status TEXT NOT NULL,
  retry_count INTEGER NOT NULL DEFAULT 0,
  max_retries INTEGER NOT NULL DEFAULT 3,
  claimed_by TEXT,
  claimed_at TEXT,
  next_retry_at TEXT,
  last_error TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS notes (
  note_id TEXT PRIMARY KEY,
  note_type TEXT NOT NULL,
  title TEXT NOT NULL,
  vault_path TEXT NOT NULL UNIQUE,
  content_hash TEXT NOT NULL,
  status TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS note_sources (
  note_id TEXT NOT NULL,
  event_id TEXT NOT NULL,
  relation_type TEXT NOT NULL,
  PRIMARY KEY (note_id, event_id, relation_type)
);

CREATE TABLE IF NOT EXISTS merge_logs (
  merge_id TEXT PRIMARY KEY,
  candidate_event_id TEXT NOT NULL,
  target_note_id TEXT,
  decision TEXT NOT NULL,
  reason TEXT NOT NULL,
  detail_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS note_embeddings (
  note_id TEXT PRIMARY KEY,
  embedding_model TEXT NOT NULL,
  embedding_vector BLOB,
  content_hash TEXT,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transcript_cursors (
  session_file TEXT PRIMARY KEY,
  last_mtime REAL NOT NULL,
  last_scanned_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_traces (
  trace_id TEXT PRIMARY KEY,
  turn_id TEXT NOT NULL,
  event_id TEXT NOT NULL,
  prompt_version TEXT NOT NULL,
  model_name TEXT NOT NULL,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  total_steps INTEGER NOT NULL,
  exceeded_max_steps INTEGER NOT NULL DEFAULT 0,
  terminal_reason TEXT NOT NULL,
  final_decision TEXT NOT NULL,
  final_confidence REAL,
  total_input_tokens INTEGER,
  total_output_tokens INTEGER,
  total_duration_ms INTEGER,
  trace_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS session_turns (
  conversation_id TEXT NOT NULL,
  turn_id TEXT NOT NULL,
  processed_at TEXT NOT NULL,
  final_decision TEXT NOT NULL,
  PRIMARY KEY (conversation_id, turn_id)
);

CREATE TABLE IF NOT EXISTS session_note_actions (
  conversation_id TEXT NOT NULL,
  turn_id TEXT NOT NULL,
  note_id TEXT NOT NULL,
  action_type TEXT NOT NULL,
  note_title TEXT,
  created_at TEXT NOT NULL,
  PRIMARY KEY (conversation_id, turn_id, note_id, action_type)
);

CREATE TABLE IF NOT EXISTS action_proposals (
  proposal_id TEXT PRIMARY KEY,
  trace_id TEXT NOT NULL,
  turn_id TEXT NOT NULL,
  action_type TEXT NOT NULL,
  target_note_id TEXT,
  payload_json TEXT NOT NULL,
  idempotency_key TEXT NOT NULL UNIQUE,
  status TEXT NOT NULL DEFAULT 'proposed',
  created_at TEXT NOT NULL,
  committed_at TEXT
);

CREATE TABLE IF NOT EXISTS replay_bundles (
  trace_id TEXT PRIMARY KEY,
  event_json TEXT NOT NULL,
  prompt_snapshot TEXT NOT NULL,
  config_snapshot_json TEXT NOT NULL,
  tool_results_json TEXT NOT NULL,
  knowledge_snapshot_refs_json TEXT NOT NULL,
  model_name TEXT NOT NULL,
  model_adapter_version TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS review_actions (
  review_id TEXT PRIMARY KEY,
  turn_id TEXT NOT NULL,
  trace_id TEXT NOT NULL,
  final_action TEXT NOT NULL,
  final_target_note_id TEXT,
  suggested_action TEXT,
  suggested_target_note_id TEXT,
  suggested_payload_json TEXT,
  reviewer TEXT,
  reason TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS eval_cases (
  case_id TEXT PRIMARY KEY,
  turn_id TEXT,
  input_json TEXT NOT NULL,
  expected_decision TEXT NOT NULL,
  expected_target_note TEXT,
  expected_risk_level TEXT NOT NULL,
  unsafe_if_written INTEGER NOT NULL DEFAULT 0,
  difficulty TEXT NOT NULL,
  annotator TEXT,
  notes TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS eval_runs (
  run_id TEXT PRIMARY KEY,
  prompt_version TEXT NOT NULL,
  model_name TEXT NOT NULL,
  total_cases INTEGER NOT NULL,
  decision_accuracy REAL NOT NULL,
  target_note_accuracy REAL,
  false_write_rate REAL NOT NULL,
  unsafe_merge_rate REAL,
  proposal_rejection_rate REAL,
  logical_replay_match_rate REAL,
  live_replay_drift_rate REAL,
  review_precision REAL,
  auto_action_acceptance_rate REAL,
  avg_steps REAL,
  avg_tokens REAL,
  avg_duration_ms REAL,
  result_json TEXT NOT NULL,
  ran_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS confidence_feedback (
  feedback_id TEXT PRIMARY KEY,
  turn_id TEXT NOT NULL,
  source_confidence REAL NOT NULL,
  human_label TEXT NOT NULL,
  annotator TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS audit_logs (
  audit_id TEXT PRIMARY KEY,
  event_type TEXT NOT NULL,
  level TEXT NOT NULL,
  trace_id TEXT,
  turn_id TEXT,
  task_id TEXT,
  detail_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


class Database:
    def __init__(self, path: Path):
        self.path = path
        self.event_logger = None
        ensure_directory(path.parent)
        self._connection = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA synchronous = NORMAL")

    def migrate(self) -> None:
        self._connection.executescript(SCHEMA_SQL)
        self._ensure_column("note_embeddings", "content_hash", "TEXT")
        self._ensure_column("review_actions", "suggested_action", "TEXT")
        self._ensure_column("review_actions", "suggested_target_note_id", "TEXT")
        self._ensure_column("review_actions", "suggested_payload_json", "TEXT")
        self._connection.commit()

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        cursor = self._connection.execute(sql, params)
        return cursor

    def executemany(self, sql: str, params: list[tuple[Any, ...]]) -> sqlite3.Cursor:
        cursor = self._connection.executemany(sql, params)
        return cursor

    def fetchone(self, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        row = self._connection.execute(sql, params).fetchone()
        if row is None:
            return None
        return dict(row)

    def fetchall(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        return [dict(row) for row in self._connection.execute(sql, params).fetchall()]

    def commit(self) -> None:
        self._connection.commit()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            yield
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

    def close(self) -> None:
        self._connection.close()

    def _ensure_column(self, table: str, column: str, ddl: str) -> None:
        columns = {
            row["name"]
            for row in self._connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column in columns:
            return
        self._connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    def upsert_cursor(self, session_file: str, last_mtime: float, scanned_at: str) -> None:
        self.execute(
            """
            INSERT INTO transcript_cursors (session_file, last_mtime, last_scanned_at)
            VALUES (?, ?, ?)
            ON CONFLICT(session_file)
            DO UPDATE SET last_mtime = excluded.last_mtime,
                          last_scanned_at = excluded.last_scanned_at
            """,
            (session_file, last_mtime, scanned_at),
        )

    def get_cursor(self, session_file: str) -> dict[str, Any] | None:
        return self.fetchone(
            "SELECT session_file, last_mtime, last_scanned_at FROM transcript_cursors WHERE session_file = ?",
            (session_file,),
        )

    _IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    def save_json_row(self, table: str, payload: dict[str, Any]) -> None:
        for name in [table, *payload.keys()]:
            if not self._IDENTIFIER_RE.match(name):
                raise ValueError(f"unsafe SQL identifier: {name!r}")
        keys = list(payload.keys())
        placeholders = ", ".join("?" for _ in keys)
        columns = ", ".join(keys)
        values = tuple(json.dumps(value) if isinstance(value, (dict, list)) else value for value in payload.values())
        self.execute(f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", values)
