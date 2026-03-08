from __future__ import annotations

import argparse
import json
from typing import Any

from ..config import load_config
from ..storage.sqlite import Database
from ..storage.vault import Vault


def repaired_title_for_metadata(title: str, metadata: dict[str, Any] | None) -> str:
    metadata = metadata or {}
    tags = {_signal_key(item) for item in metadata.get("tags", [])}
    topics = {_signal_key(item) for item in metadata.get("topics", [])}
    signals = {item for item in tags | topics if item}
    if {"transcript", "worker"} <= signals or "现在怎么用这个snowball" in signals:
        return "snowball-notes 使用方式"
    if {"deepseek", "runtime-error"} <= signals or {"deepseek", "environment-variable", "runtime-error"} <= signals:
        return "RuntimeError: missing API key env: DEEPSEEK_API_KEY 诊断与处理"
    if {"cli", "status", "bug-fix"} <= signals:
        return "status 命令误触发 API key 检查"
    if {"archive", "inbox", "knowledge"} <= signals:
        return "项目里 Archive、Inbox、Knowledge 目录的设计意图与当前实现差异"
    if {"project-status", "implementation-gap", "phase-assessment"} <= signals:
        return "snowball-notes项目当前状态评估：已完成部分与最终设计差距"
    if {"snowball-config", "环境配置"} <= signals or {"snowball-config", "api密钥管理"} <= signals:
        return "snowball-notes环境配置架构：SNOWBALL_CONFIG与API密钥管理"
    if {"phase-3", "phase-4"} <= signals:
        return "完成snowball-notes-final.md的Phase 3和Phase 4实现"
    if {"shell", "zsh", "line-continuation"} <= signals:
        return "Shell命令多行续行符缺失导致zsh报错：command not found: --provider"
    if {"embedding", "阿里云百炼", "智谱"} <= signals:
        return "中国embedding替代方案推荐：阿里云百炼、智谱、BAAI/bge-m3对比"
    if {"obsidian", "frontmatter", "yaml"} <= signals:
        return "Obsidian frontmatter格式问题诊断与修复：YAML标题冒号与重复标题处理"
    if {"phase", "final"} <= signals:
        return "技术方案步骤的 Phase 归属"
    if {"snowball-notes", "model-key", "command-classification"} <= signals:
        return "snowball-notes 命令分类与模型 key 依赖"
    return title


def repair_note_titles(config_path: str | None = None, *, apply: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    db = Database(config.db_path)
    db.migrate()
    vault = Vault(config)
    changes: list[dict[str, str]] = []
    try:
        rows = db.fetchall(
            """
            SELECT note_id, note_type, title, vault_path, status, metadata_json, created_at, updated_at
            FROM notes
            WHERE note_type = 'atomic' AND status = 'approved'
            ORDER BY created_at ASC
            """
        )
        for row in rows:
            metadata = json.loads(row.get("metadata_json") or "{}")
            repaired_title = repaired_title_for_metadata(row["title"], metadata)
            if repaired_title == row["title"]:
                continue
            source_rows = db.fetchall(
                """
                SELECT event_id
                FROM note_sources
                WHERE note_id = ?
                ORDER BY event_id ASC
                """,
                (row["note_id"],),
            )
            change = {
                "note_id": row["note_id"],
                "old_title": row["title"],
                "new_title": repaired_title,
                "old_path": row["vault_path"],
            }
            if apply:
                vault.replace_note_title_references(row["title"], repaired_title)
                renamed_path = vault.rename_note_path(
                    row["note_id"],
                    repaired_title,
                    row["vault_path"],
                    status=row["status"],
                )
                _, content_hash, _, normalized_path = vault.normalize_note_file(
                    note_type=row["note_type"],
                    note_id=row["note_id"],
                    title=repaired_title,
                    note_path=renamed_path,
                    status=row["status"],
                    metadata=metadata,
                    source_event_ids=[item["event_id"] for item in source_rows],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
                db.execute(
                    """
                    UPDATE notes
                    SET title = ?, vault_path = ?, content_hash = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE note_id = ?
                    """,
                    (repaired_title, str(normalized_path.resolve()), content_hash, row["note_id"]),
                )
                db.execute(
                    "UPDATE session_note_actions SET note_title = ? WHERE note_id = ?",
                    (repaired_title, row["note_id"]),
                )
                change["new_path"] = str(normalized_path.resolve())
            changes.append(change)
        if apply:
            db.commit()
        return {
            "apply": apply,
            "changed": len(changes),
            "changes": changes,
        }
    finally:
        db.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One-off title repair for existing approved knowledge notes.")
    parser.add_argument("--config", help="Path to config.yaml")
    parser.add_argument("--apply", action="store_true", help="Write repaired titles back to the vault and database.")
    args = parser.parse_args(argv)
    report = repair_note_titles(args.config, apply=args.apply)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _signal_key(value: str) -> str:
    return str(value or "").strip().lower().replace("_", "-")


if __name__ == "__main__":
    raise SystemExit(main())
