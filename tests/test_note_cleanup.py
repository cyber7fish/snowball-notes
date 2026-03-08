import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from snowball_notes.cli import build_runtime, main
from snowball_notes.maintenance.repair_note_titles import repair_note_titles, repaired_title_for_metadata
from snowball_notes.note_cleanup import format_obsidian_link, rewrite_wikilink_target, sanitize_note_markdown


def _write_config(path: Path, transcript_dir: Path) -> None:
    path.write_text(
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
                "  mode: \"transcript_poll\"",
                f"  transcript_dir: \"{transcript_dir}\"",
                "  parser_version: \"v1\"",
                "  min_response_length: 120",
                "  min_confidence_to_run: 0.50",
                "agent:",
                "  model: \"heuristic-v1\"",
                "  max_steps: 8",
                "  prompt_version: \"agent_system/v1.md\"",
                "  max_writes_per_run: 1",
                "  max_appends_per_run: 1",
                "retrieval:",
                "  top_k: 5",
                "  append_threshold: 0.82",
                "  review_threshold: 0.62",
                "guardrails:",
                "  min_confidence_for_note: 0.70",
                "  min_confidence_for_append: 0.85",
                "worker:",
                "  poll_interval_seconds: 10",
                "  claim_timeout_seconds: 300",
                "  max_retries: 3",
                "reconcile:",
                "  enabled: true",
                "  run_on_startup: false",
                "  schedule_cron: \"0 3 * * *\"",
            ]
        ),
        encoding="utf-8",
    )


class NoteCleanupTests(unittest.TestCase):
    def test_sanitize_local_file_links(self):
        sanitized = sanitize_note_markdown(
            "See [vault.py](/Users/7fish/project/snowball-notes/src/snowball_notes/storage/vault.py#L51)."
        )
        self.assertEqual(sanitized, "See `vault.py`.")

    def test_format_obsidian_link_uses_alias_when_filename_differs(self):
        self.assertEqual(
            format_obsidian_link("技术方案步骤的 Phase 归属", "/tmp/技术方案步骤的 Phase 归属 (01ceff).md"),
            "[[技术方案步骤的 Phase 归属 (01ceff)|技术方案步骤的 Phase 归属]]",
        )

    def test_rewrite_wikilink_target_upgrades_plain_link(self):
        self.assertEqual(
            rewrite_wikilink_target(
                "See [[RuntimeError: missing API key env: DEEPSEEK_API_KEY 诊断与处理]].",
                "RuntimeError: missing API key env: DEEPSEEK_API_KEY 诊断与处理",
                "/tmp/RuntimeError missing API key env DEEPSEEK_API_KEY 诊断与处理.md",
            ),
            "See [[RuntimeError missing API key env DEEPSEEK_API_KEY 诊断与处理|RuntimeError: missing API key env: DEEPSEEK_API_KEY 诊断与处理]].",
        )

    def test_reconcile_sanitizes_body_but_preserves_agent_title(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            transcripts = root / "sessions"
            transcripts.mkdir(parents=True)
            config_path = root / "config.yaml"
            _write_config(config_path, transcripts)
            config, db, vault, _ = build_runtime(str(config_path), build_worker=False)
            try:
                note_path, content_hash = vault.write_new_note(
                    note_id="note_agent_title",
                    title="现在怎么用这个snowball-notes呢",
                    content=(
                        "## Summary\n"
                        "现在的用法是“后台 worker 扫 Codex transcript，然后把结果写进你的 Obsidian Vault”。\n\n"
                        "See [config.yaml](/Users/7fish/project/snowball-notes/config.yaml#L5).\n"
                    ),
                    tags=["snowball-notes"],
                    topics=["usage"],
                    source_event_ids=["evt_usage"],
                    status="approved",
                )
                db.execute(
                    """
                    INSERT INTO notes (note_id, note_type, title, vault_path, content_hash, status, metadata_json, created_at, updated_at)
                    VALUES (?, 'atomic', ?, ?, ?, 'approved', ?, '2026-03-08T00:00:00+00:00', '2026-03-08T00:00:00+00:00')
                    """,
                    (
                        "note_agent_title",
                        "现在怎么用这个snowball-notes呢",
                        str(note_path.resolve()),
                        content_hash,
                        '{"tags":["snowball-notes"],"topics":["usage"]}',
                    ),
                )
                db.commit()

                stdout = io.StringIO()
                with mock.patch("sys.stdout", stdout):
                    exit_code = main(["--config", str(config_path), "reconcile"])

                self.assertEqual(exit_code, 0)
                note_row = db.fetchone("SELECT title, vault_path FROM notes WHERE note_id = 'note_agent_title'")
                self.assertEqual(note_row["title"], "现在怎么用这个snowball-notes呢")
                self.assertEqual(note_row["vault_path"], str(note_path.resolve()))
                normalized_text = note_path.read_text(encoding="utf-8")
                self.assertIn("# 现在怎么用这个snowball-notes呢", normalized_text)
                self.assertIn("`config.yaml`", normalized_text)
                self.assertNotIn("/Users/7fish/project/snowball-notes/config.yaml", normalized_text)
            finally:
                db.close()

    def test_reconcile_renames_slug_style_note_to_title_filename(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            transcripts = root / "sessions"
            transcripts.mkdir(parents=True)
            config_path = root / "config.yaml"
            _write_config(config_path, transcripts)
            config, db, vault, _ = build_runtime(str(config_path), build_worker=False)
            try:
                canonical_path, content_hash = vault.write_new_note(
                    note_id="note_phase",
                    title="技术方案步骤的 Phase 归属",
                    content="## Summary\nThis should become a readable filename.\n",
                    tags=["phase"],
                    topics=["design"],
                    source_event_ids=["evt_phase"],
                    status="approved",
                )
                legacy_path = canonical_path.with_name("技术方案步骤的-phase-归属-01ceff.md")
                canonical_path.replace(legacy_path)

                ref_path, ref_hash = vault.write_new_note(
                    note_id="note_ref",
                    title="Reference Note",
                    content="## Summary\nReference.\n\n## Related\n- [[技术方案步骤的 Phase 归属]]\n",
                    tags=["reference"],
                    topics=["links"],
                    source_event_ids=["evt_ref"],
                    status="approved",
                )
                db.execute(
                    """
                    INSERT INTO notes (note_id, note_type, title, vault_path, content_hash, status, metadata_json, created_at, updated_at)
                    VALUES (?, 'atomic', ?, ?, ?, 'approved', ?, '2026-03-08T00:00:00+00:00', '2026-03-08T00:00:00+00:00')
                    """,
                    (
                        "note_phase",
                        "技术方案步骤的 Phase 归属",
                        str(legacy_path.resolve()),
                        content_hash,
                        '{"tags":["phase"],"topics":["design"]}',
                    ),
                )
                db.execute(
                    """
                    INSERT INTO notes (note_id, note_type, title, vault_path, content_hash, status, metadata_json, created_at, updated_at)
                    VALUES (?, 'atomic', ?, ?, ?, 'approved', ?, '2026-03-08T00:00:00+00:00', '2026-03-08T00:00:00+00:00')
                    """,
                    (
                        "note_ref",
                        "Reference Note",
                        str(ref_path.resolve()),
                        ref_hash,
                        '{"tags":["reference"],"topics":["links"]}',
                    ),
                )
                db.commit()

                stdout = io.StringIO()
                with mock.patch("sys.stdout", stdout):
                    exit_code = main(["--config", str(config_path), "reconcile"])

                self.assertEqual(exit_code, 0)
                row = db.fetchone("SELECT vault_path FROM notes WHERE note_id = 'note_phase'")
                self.assertEqual(Path(row["vault_path"]).name, "技术方案步骤的 Phase 归属.md")
                self.assertFalse(legacy_path.exists())
                self.assertTrue(Path(row["vault_path"]).exists())

                ref_text = ref_path.read_text(encoding="utf-8")
                self.assertIn("[[技术方案步骤的 Phase 归属]]", ref_text)
            finally:
                db.close()

    def test_reconcile_rewrites_links_when_filename_differs_from_title(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            transcripts = root / "sessions"
            transcripts.mkdir(parents=True)
            config_path = root / "config.yaml"
            _write_config(config_path, transcripts)
            config, db, vault, _ = build_runtime(str(config_path), build_worker=False)
            try:
                primary_path, primary_hash = vault.write_new_note(
                    note_id="note_error",
                    title="RuntimeError: missing API key env: DEEPSEEK_API_KEY 诊断与处理",
                    content="## Summary\nPreserve the title, sanitize only the filename.\n",
                    tags=["runtime_error"],
                    topics=["secrets"],
                    source_event_ids=["evt_error"],
                    status="approved",
                )
                ref_path, ref_hash = vault.write_new_note(
                    note_id="note_ref",
                    title="Reference Note",
                    content=(
                        "## Summary\nReference.\n\n## Related\n"
                        "- [[RuntimeError: missing API key env: DEEPSEEK_API_KEY 诊断与处理]]\n"
                    ),
                    tags=["reference"],
                    topics=["links"],
                    source_event_ids=["evt_ref"],
                    status="approved",
                )
                for note_id, title, note_path, content_hash, metadata in [
                    (
                        "note_error",
                        "RuntimeError: missing API key env: DEEPSEEK_API_KEY 诊断与处理",
                        primary_path,
                        primary_hash,
                        '{"tags":["runtime_error"],"topics":["secrets"]}',
                    ),
                    (
                        "note_ref",
                        "Reference Note",
                        ref_path,
                        ref_hash,
                        '{"tags":["reference"],"topics":["links"]}',
                    ),
                ]:
                    db.execute(
                        """
                        INSERT INTO notes (note_id, note_type, title, vault_path, content_hash, status, metadata_json, created_at, updated_at)
                        VALUES (?, 'atomic', ?, ?, ?, 'approved', ?, '2026-03-08T00:00:00+00:00', '2026-03-08T00:00:00+00:00')
                        """,
                        (note_id, title, str(note_path.resolve()), content_hash, metadata),
                    )
                db.commit()

                stdout = io.StringIO()
                with mock.patch("sys.stdout", stdout):
                    exit_code = main(["--config", str(config_path), "reconcile"])

                self.assertEqual(exit_code, 0)
                ref_text = ref_path.read_text(encoding="utf-8")
                self.assertIn(
                    "[[RuntimeError missing API key env DEEPSEEK_API_KEY 诊断与处理|RuntimeError: missing API key env: DEEPSEEK_API_KEY 诊断与处理]]",
                    ref_text,
                )
            finally:
                db.close()

    def test_repaired_title_for_metadata_is_kept_in_one_off_script(self):
        self.assertEqual(
            repaired_title_for_metadata(
                "status 命令误触发 API key 检查",
                {"tags": ["archive", "inbox", "knowledge"], "topics": ["directory-structure"]},
            ),
            "项目里 Archive、Inbox、Knowledge 目录的设计意图与当前实现差异",
        )

    def test_one_off_title_repair_script_updates_existing_notes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            transcripts = root / "sessions"
            transcripts.mkdir(parents=True)
            config_path = root / "config.yaml"
            _write_config(config_path, transcripts)
            config, db, vault, _ = build_runtime(str(config_path), build_worker=False)
            try:
                note_path, content_hash = vault.write_new_note(
                    note_id="note_bad",
                    title="现在怎么用这个snowball-notes呢",
                    content=(
                        "## Summary\n"
                        "现在的用法是“后台 worker 扫 Codex transcript，然后把结果写进你的 Obsidian Vault”。\n\n"
                        "See [config.yaml](/Users/7fish/project/snowball-notes/config.yaml#L5).\n"
                    ),
                    tags=["snowball-notes", "transcript", "worker"],
                    topics=["现在怎么用这个snowball"],
                    source_event_ids=["evt_bad"],
                    status="approved",
                )
                ref_path, ref_hash = vault.write_new_note(
                    note_id="note_ref",
                    title="Reference Note",
                    content="## Summary\nReference.\n\n## Related\n- [[现在怎么用这个snowball-notes呢]]\n",
                    tags=["reference"],
                    topics=["links"],
                    source_event_ids=["evt_ref"],
                    status="approved",
                )
                db.execute(
                    """
                    INSERT INTO notes (note_id, note_type, title, vault_path, content_hash, status, metadata_json, created_at, updated_at)
                    VALUES (?, 'atomic', ?, ?, ?, 'approved', ?, '2026-03-08T00:00:00+00:00', '2026-03-08T00:00:00+00:00')
                    """,
                    (
                        "note_bad",
                        "现在怎么用这个snowball-notes呢",
                        str(note_path.resolve()),
                        content_hash,
                        '{"tags":["snowball-notes","transcript","worker"],"topics":["现在怎么用这个snowball"]}',
                    ),
                )
                db.execute(
                    """
                    INSERT INTO notes (note_id, note_type, title, vault_path, content_hash, status, metadata_json, created_at, updated_at)
                    VALUES (?, 'atomic', ?, ?, ?, 'approved', ?, '2026-03-08T00:00:00+00:00', '2026-03-08T00:00:00+00:00')
                    """,
                    (
                        "note_ref",
                        "Reference Note",
                        str(ref_path.resolve()),
                        ref_hash,
                        '{"tags":["reference"],"topics":["links"]}',
                    ),
                )
                db.commit()

                report = repair_note_titles(str(config_path), apply=True)
                self.assertEqual(report["changed"], 1)

                note_row = db.fetchone("SELECT title, vault_path FROM notes WHERE note_id = 'note_bad'")
                self.assertEqual(note_row["title"], "snowball-notes 使用方式")
                self.assertNotEqual(note_row["vault_path"], str(note_path.resolve()))
                self.assertFalse(note_path.exists())

                repaired_path = Path(note_row["vault_path"])
                self.assertTrue(repaired_path.exists())
                repaired_text = repaired_path.read_text(encoding="utf-8")
                self.assertIn("# snowball-notes 使用方式", repaired_text)
                self.assertIn("`config.yaml`", repaired_text)

                ref_text = ref_path.read_text(encoding="utf-8")
                self.assertIn("[[snowball-notes 使用方式]]", ref_text)
                self.assertNotIn("[[现在怎么用这个snowball-notes呢]]", ref_text)
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main()
