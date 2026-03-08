from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import SnowballConfig
from ..utils import ensure_directory, now_utc_iso, safe_read_text, sha256_text, slugify, write_atomic_text


class Vault:
    def __init__(self, config: SnowballConfig):
        self.root = config.vault_path
        self.inbox_dir = self.root / config.vault.inbox_dir
        self.archive_dir = self.root / config.vault.archive_dir
        self.atomic_dir = self.root / config.vault.atomic_dir
        self.ensure_layout()

    def ensure_layout(self) -> None:
        ensure_directory(self.inbox_dir)
        ensure_directory(self.archive_dir)
        ensure_directory(self.atomic_dir)

    def write_new_note(
        self,
        note_id: str,
        title: str,
        content: str,
        tags: list[str] | None = None,
        topics: list[str] | None = None,
        source_event_ids: list[str] | None = None,
        status: str = "pending_review",
    ) -> tuple[Path, str]:
        path = self.inbox_dir / f"{slugify(title)}-{note_id[-6:]}.md"
        frontmatter = self._frontmatter(
            {
                "id": note_id,
                "type": "atomic",
                "title": title,
                "tags": tags or [],
                "topics": topics or [],
                "source_event_ids": source_event_ids or [],
                "created_at": now_utc_iso(),
                "updated_at": now_utc_iso(),
                "status": status,
            }
        )
        full_content = f"{frontmatter}\n\n# {title}\n\n{content.strip()}\n"
        write_atomic_text(path, full_content)
        return path, sha256_text(full_content)

    def write_archive_note(self, note_id: str, payload: dict[str, Any]) -> tuple[Path, str]:
        title = payload["title"]
        path = self.archive_dir / f"{slugify(title)}-{note_id[-6:]}.md"
        frontmatter = self._frontmatter(
            {
                "id": note_id,
                "type": "archive",
                "title": title,
                "source_event_ids": [payload["event_id"]],
                "created_at": now_utc_iso(),
                "updated_at": now_utc_iso(),
                "status": "archived",
            }
        )
        full_content = (
            f"{frontmatter}\n\n# {title}\n\n"
            f"## User\n{payload['user_message'].strip()}\n\n"
            f"## Assistant\n{payload['assistant_final_answer'].strip()}\n"
        )
        write_atomic_text(path, full_content)
        return path, sha256_text(full_content)

    def append_to_updates_section(self, note_path: Path, content: str, turn_id: str) -> str:
        existing = safe_read_text(note_path)
        block = f"- {turn_id}: {content.strip()}"
        marker = "\n## Updates\n"
        if marker in existing:
            updated = existing.rstrip() + "\n" + block + "\n"
        else:
            updated = existing.rstrip() + f"{marker}\n{block}\n"
        write_atomic_text(note_path, updated)
        return sha256_text(updated)

    def read_note(self, note_path: str | Path) -> str:
        return safe_read_text(Path(note_path))

    def content_hash(self, note_path: str | Path) -> str:
        return sha256_text(self.read_note(note_path))

    def update_note_status(self, note_path: str | Path, status: str) -> str:
        path = Path(note_path)
        existing = safe_read_text(path)
        if not existing.startswith("---\n"):
            return sha256_text(existing)
        lines = existing.splitlines()
        try:
            end_index = lines.index("---", 1)
        except ValueError:
            return sha256_text(existing)
        frontmatter = lines[1:end_index]
        updated = []
        status_seen = False
        timestamp_seen = False
        for line in frontmatter:
            if line.startswith("status:"):
                updated.append(f"status: {status}")
                status_seen = True
                continue
            if line.startswith("updated_at:"):
                updated.append(f"updated_at: {now_utc_iso()}")
                timestamp_seen = True
                continue
            updated.append(line)
        if not status_seen:
            updated.append(f"status: {status}")
        if not timestamp_seen:
            updated.append(f"updated_at: {now_utc_iso()}")
        rendered = "\n".join(["---", *updated, "---", *lines[end_index + 1 :]])
        if existing.endswith("\n"):
            rendered += "\n"
        write_atomic_text(path, rendered)
        return sha256_text(rendered)

    def _frontmatter(self, payload: dict[str, Any]) -> str:
        lines = ["---"]
        for key, value in payload.items():
            if isinstance(value, list):
                lines.append(f"{key}:")
                for item in value:
                    lines.append(f"  - {item}")
            else:
                lines.append(f"{key}: {value}")
        lines.append("---")
        return "\n".join(lines)
