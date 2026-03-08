import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from snowball_notes.agent.memory import SQLiteKnowledgeIndex
from snowball_notes.cli import main
from snowball_notes.config import default_config
from snowball_notes.embedding import build_embedding_provider, build_vector_store, run_embedding_check
from snowball_notes.embedding.local import LocalHashEmbeddingProvider
from snowball_notes.storage.sqlite import Database
from snowball_notes.storage.vault import Vault
from snowball_notes.utils import now_utc_iso


class EmbeddingTests(unittest.TestCase):
    def test_local_hash_embedding_provider_is_deterministic(self):
        config = default_config(Path.cwd())
        provider = LocalHashEmbeddingProvider(config)
        left = provider.embed("agent runtime guardrails")
        right = provider.embed("agent runtime guardrails")
        self.assertEqual(len(left), config.embedding.local_dimensions)
        self.assertEqual(left, right)
        self.assertAlmostEqual(sum(value * value for value in left), 1.0, places=6)

    def test_knowledge_index_populates_embeddings_and_ranks_relevant_note_first(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = default_config(root)
            db = Database(config.db_path)
            db.migrate()
            vault = Vault(config)
            try:
                semantic_path, semantic_hash = vault.write_new_note(
                    note_id="note_semantic",
                    title="Guarded Side Effect Playbook",
                    content=(
                        "## Summary\n"
                        "Agent runtime control side effects safely with proposals, guardrails, commit validation, "
                        "and replay bundles for debugging.\n"
                    ),
                    tags=["agent", "runtime", "guardrails"],
                    topics=["side-effects", "replay"],
                    source_event_ids=["evt_seed_1"],
                    status="approved",
                )
                other_path, other_hash = vault.write_new_note(
                    note_id="note_other",
                    title="Gardening Journal",
                    content="## Summary\nTomatoes need consistent watering and sunlight.\n",
                    tags=["garden"],
                    topics=["plants"],
                    source_event_ids=["evt_seed_2"],
                    status="approved",
                )
                for note_id, title, path, content_hash, metadata in [
                    (
                        "note_semantic",
                        "Guarded Side Effect Playbook",
                        semantic_path,
                        semantic_hash,
                        {"tags": ["agent", "runtime", "guardrails"], "topics": ["side-effects", "replay"]},
                    ),
                    (
                        "note_other",
                        "Gardening Journal",
                        other_path,
                        other_hash,
                        {"tags": ["garden"], "topics": ["plants"]},
                    ),
                ]:
                    db.execute(
                        """
                        INSERT INTO notes (
                          note_id, note_type, title, vault_path, content_hash,
                          status, metadata_json, created_at, updated_at
                        ) VALUES (?, 'atomic', ?, ?, ?, 'approved', ?, ?, ?)
                        """,
                        (
                            note_id,
                            title,
                            str(path.resolve()),
                            content_hash,
                            json.dumps(metadata, ensure_ascii=False),
                            now_utc_iso(),
                            now_utc_iso(),
                        ),
                    )
                db.commit()

                provider = build_embedding_provider(config)
                vector_store = build_vector_store(config, db)
                index = SQLiteKnowledgeIndex(
                    db,
                    config=config,
                    embedding_provider=provider,
                    vector_store=vector_store,
                )

                results = index.search("How should an agent runtime control side effects safely?", top_k=2)

                self.assertEqual(results[0].note_id, "note_semantic")
                embeddings = db.fetchall(
                    "SELECT note_id, embedding_model, content_hash FROM note_embeddings ORDER BY note_id ASC"
                )
                self.assertEqual(len(embeddings), 2)
                self.assertEqual(embeddings[0]["embedding_model"], config.embedding.local_model)
                self.assertEqual({row["note_id"] for row in embeddings}, {"note_other", "note_semantic"})
            finally:
                db.close()

    def test_embedding_check_succeeds_for_local_provider(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = default_config(root)
            db = Database(config.db_path)
            db.migrate()
            try:
                result = run_embedding_check(config, db, provider_override="local")
                self.assertTrue(result["ok"])
                self.assertEqual(result["provider"], "local")
                self.assertEqual(result["dimensions"], config.embedding.local_dimensions)
                self.assertTrue(result["roundtrip_ok"])
            finally:
                db.close()

    def test_embedding_check_reports_missing_voyage_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.yaml"
            config_path.write_text(
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
                        "embedding:",
                        "  provider: \"voyage\"",
                        "  api_key_env: \"VOYAGE_API_KEY\"",
                        "worker:",
                        "  poll_interval_seconds: 10",
                        "  claim_timeout_seconds: 300",
                        "  max_retries: 3",
                    ]
                ),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            stderr = io.StringIO()
            with mock.patch.dict(os.environ, {"SNOWBALL_ENV_FILE": ""}, clear=True):
                with mock.patch("sys.stdout", stdout), mock.patch("sys.stderr", stderr):
                    exit_code = main(["--config", str(config_path), "embedding", "check", "--provider", "voyage"])
            self.assertEqual(exit_code, 1)
            self.assertIn("missing embedding API key env", stderr.getvalue())

    def test_embedding_check_reports_missing_dashscope_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.yaml"
            config_path.write_text(
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
                        "embedding:",
                        "  provider: \"dashscope\"",
                        "  dashscope_api_key_env: \"DASHSCOPE_API_KEY\"",
                        "worker:",
                        "  poll_interval_seconds: 10",
                        "  claim_timeout_seconds: 300",
                        "  max_retries: 3",
                    ]
                ),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            stderr = io.StringIO()
            with mock.patch.dict(os.environ, {"SNOWBALL_ENV_FILE": ""}, clear=True):
                with mock.patch("sys.stdout", stdout), mock.patch("sys.stderr", stderr):
                    exit_code = main(["--config", str(config_path), "embedding", "check", "--provider", "dashscope"])
            self.assertEqual(exit_code, 1)
            self.assertIn("missing embedding API key env", stderr.getvalue())

    def test_embedding_check_uses_mocked_voyage_provider(self):
        class _FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(
                    {
                        "data": [
                            {"embedding": [1.0, 0.0, 0.0]},
                        ]
                    }
                ).encode("utf-8")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = default_config(root)
            config.embedding.provider = "voyage"
            config.embedding.api_key_env = "VOYAGE_API_KEY"
            db = Database(config.db_path)
            db.migrate()
            try:
                with mock.patch.dict(os.environ, {"VOYAGE_API_KEY": "test-key"}):
                    with mock.patch("snowball_notes.embedding.voyage.request.urlopen", return_value=_FakeResponse()):
                        result = run_embedding_check(config, db)
                self.assertTrue(result["ok"])
                self.assertEqual(result["provider"], "voyage")
                self.assertEqual(result["dimensions"], 3)
                self.assertTrue(result["roundtrip_ok"])
            finally:
                db.close()

    def test_embedding_check_uses_mocked_dashscope_provider(self):
        class _FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(
                    {
                        "data": [
                            {"embedding": [0.1, 0.2, 0.3, 0.4]},
                        ]
                    }
                ).encode("utf-8")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = default_config(root)
            config.embedding.provider = "dashscope"
            config.embedding.dashscope_api_key_env = "DASHSCOPE_API_KEY"
            config.embedding.dashscope_dimensions = 4
            db = Database(config.db_path)
            db.migrate()
            try:
                with mock.patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}):
                    with mock.patch("snowball_notes.embedding.dashscope.request.urlopen", return_value=_FakeResponse()):
                        result = run_embedding_check(config, db)
                self.assertTrue(result["ok"])
                self.assertEqual(result["provider"], "dashscope")
                self.assertEqual(result["dimensions"], 4)
                self.assertTrue(result["roundtrip_ok"])
            finally:
                db.close()
