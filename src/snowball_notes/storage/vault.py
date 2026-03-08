from __future__ import annotations

import json
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
        path = self.note_path(note_id, title, status=status)
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
        body = self._normalize_atomic_body(title, content)
        full_content = f"{frontmatter}\n\n# {title}\n\n{body}\n"
        write_atomic_text(path, full_content)
        return path, sha256_text(full_content)

    def note_path(self, note_id: str, title: str, *, status: str = "pending_review") -> Path:
        base_dir = self.atomic_dir if status == "approved" else self.inbox_dir
        return base_dir / f"{slugify(title)}-{note_id[-6:]}.md"

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

    def add_bidirectional_link(
        self,
        source_path: str | Path,
        source_title: str,
        target_path: str | Path,
        target_title: str,
    ) -> tuple[str, str]:
        source_hash = self._ensure_links_section_entry(Path(source_path), target_title)
        target_hash = self._ensure_links_section_entry(Path(target_path), source_title)
        return source_hash, target_hash

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
                updated.append(f"status: {self._yaml_scalar(status)}")
                status_seen = True
                continue
            if line.startswith("updated_at:"):
                updated.append(f"updated_at: {self._yaml_scalar(now_utc_iso())}")
                timestamp_seen = True
                continue
            updated.append(line)
        if not status_seen:
            updated.append(f"status: {self._yaml_scalar(status)}")
        if not timestamp_seen:
            updated.append(f"updated_at: {self._yaml_scalar(now_utc_iso())}")
        rendered = "\n".join(["---", *updated, "---", *lines[end_index + 1 :]])
        if existing.endswith("\n"):
            rendered += "\n"
        write_atomic_text(path, rendered)
        return sha256_text(rendered)

    def promote_note_to_atomic(self, note_id: str, title: str, note_path: str | Path) -> tuple[Path, str]:
        source_path = Path(note_path)
        target_path = self.note_path(note_id, title, status="approved")
        if source_path.resolve() != target_path.resolve():
            ensure_directory(target_path.parent)
            source_path.replace(target_path)
        content_hash = self.update_note_status(target_path, "approved")
        return target_path, content_hash

    def _ensure_links_section_entry(self, note_path: Path, linked_title: str) -> str:
        existing = safe_read_text(note_path)
        link_line = f"- [[{linked_title}]]"
        if link_line in existing:
            return sha256_text(existing)
        marker = "\n## Links\n"
        if marker in existing:
            updated = existing.rstrip() + "\n" + link_line + "\n"
        else:
            updated = existing.rstrip() + f"{marker}\n{link_line}\n"
        write_atomic_text(note_path, updated)
        return sha256_text(updated)

    def _frontmatter(self, payload: dict[str, Any]) -> str:
        lines = ["---"]
        for key, value in payload.items():
            if isinstance(value, list):
                lines.append(f"{key}:")
                for item in value:
                    lines.append(f"  - {self._yaml_scalar(item)}")
            else:
                lines.append(f"{key}: {self._yaml_scalar(value)}")
        lines.append("---")
        return "\n".join(lines)

    def normalize_note_file(
        self,
        *,
        note_type: str,
        note_id: str,
        title: str,
        note_path: str | Path,
        status: str,
        metadata: dict[str, Any],
        source_event_ids: list[str],
        created_at: str,
        updated_at: str,
    ) -> tuple[bool, str]:
        path = Path(note_path)
        existing = safe_read_text(path)
        if note_type == "atomic":
            frontmatter = self._frontmatter(
                {
                    "id": note_id,
                    "type": note_type,
                    "title": title,
                    "tags": metadata.get("tags", []),
                    "topics": metadata.get("topics", []),
                    "source_event_ids": source_event_ids,
                    "created_at": created_at,
                    "updated_at": updated_at,
                    "status": status,
                }
            )
            body = self._normalize_atomic_body(title, self._body_without_frontmatter(existing))
            rendered = f"{frontmatter}\n\n# {title}\n\n{body}\n"
        else:
            frontmatter = self._frontmatter(
                {
                    "id": note_id,
                    "type": note_type,
                    "title": title,
                    "source_event_ids": source_event_ids,
                    "created_at": created_at,
                    "updated_at": updated_at,
                    "status": status,
                }
            )
            body = self._body_without_frontmatter(existing).strip()
            rendered = f"{frontmatter}\n\n{body}\n"
        if rendered == existing:
            return False, sha256_text(existing)
        write_atomic_text(path, rendered)
        return True, sha256_text(rendered)

    def _normalize_atomic_body(self, title: str, content: str) -> str:
        body = content.strip()
        title_line = f"# {title}".strip()
        while body.startswith(title_line):
            remainder = body[len(title_line) :].lstrip()
            if remainder == body:
                break
            body = remainder
        return body or "## Summary\nNo durable summary captured yet."

    def _body_without_frontmatter(self, content: str) -> str:
        if not content.startswith("---\n"):
            return content.strip()
        lines = content.splitlines()
        try:
            end_index = lines.index("---", 1)
        except ValueError:
            return content.strip()
        return "\n".join(lines[end_index + 1 :]).strip()

    def _yaml_scalar(self, value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        return json.dumps(str(value), ensure_ascii=False)
