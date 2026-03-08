import json
import tempfile
import unittest
from pathlib import Path

from snowball_notes.agent.memory import SQLiteKnowledgeIndex
from snowball_notes.config import default_config
from snowball_notes.embedding import build_embedding_provider, build_vector_store
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
