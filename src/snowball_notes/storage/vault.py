from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..config import SnowballConfig
from ..note_cleanup import (
    format_obsidian_link,
    sanitize_note_markdown,
    strip_leading_page_heading,
    rewrite_wikilink_target,
    update_wikilink_titles,
)
from ..utils import ensure_directory, now_utc_iso, safe_read_text, sha256_text, write_atomic_text


INVALID_FILENAME_CHARS_RE = re.compile(r'[<>:"/\\\\|?*\x00-\x1f]')
SPACE_RE = re.compile(r"\s+")


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

    def note_path(
        self,
        note_id: str,
        title: str,
        *,
        status: str = "pending_review",
        current_path: str | Path | None = None,
    ) -> Path:
        base_dir = self.atomic_dir if status == "approved" else self.inbox_dir
        return self._preferred_note_path(base_dir, note_id, title, current_path=current_path)

    def write_archive_note(self, note_id: str, payload: dict[str, Any]) -> tuple[Path, str]:
        title = payload["title"]
        path = self._preferred_note_path(self.archive_dir, note_id, title)
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
            f"## User\n{sanitize_note_markdown(payload['user_message'])}\n\n"
            f"## Assistant\n{sanitize_note_markdown(payload['assistant_final_answer'])}\n"
        )
        write_atomic_text(path, full_content)
        return path, sha256_text(full_content)

    def append_to_updates_section(self, note_path: Path, content: str, turn_id: str) -> str:
        existing = safe_read_text(note_path)
        block = f"- {turn_id}: {sanitize_note_markdown(content)}"
        marker = "\n## Updates\n"
        if marker in existing:
            updated = existing.rstrip() + "\n" + block + "\n"
        else:
            updated = existing.rstrip() + f"{marker}\n{block}\n"
        write_atomic_text(note_path, updated)
        return sha256_text(updated)

    def rename_note_path(self, note_id: str, title: str, note_path: str | Path, *, status: str) -> Path:
        source_path = Path(note_path)
        target_path = self.note_path(note_id, title, status=status, current_path=source_path)
        if source_path.resolve(strict=False) == target_path.resolve(strict=False):
            return source_path
        ensure_directory(target_path.parent)
        source_path.replace(target_path)
        return target_path

    def replace_note_title_references(self, old_title: str, new_title: str) -> int:
        if old_title == new_title:
            return 0
        updated_count = 0
        for path in self.root.rglob("*.md"):
            if path.name.startswith("."):
                continue
            existing = safe_read_text(path)
            updated = update_wikilink_titles(existing, old_title, new_title)
            if updated == existing:
                continue
            write_atomic_text(path, updated)
            updated_count += 1
        return updated_count

    def normalize_link_targets(self, note_rows: list[dict[str, str]]) -> int:
        updated_count = 0
        for path in self.root.rglob("*.md"):
            if path.name.startswith("."):
                continue
            existing = safe_read_text(path)
            updated = existing
            for row in note_rows:
                updated = rewrite_wikilink_target(updated, row["title"], row["vault_path"])
            if updated == existing:
                continue
            write_atomic_text(path, updated)
            updated_count += 1
        return updated_count

    def add_bidirectional_link(
        self,
        source_path: str | Path,
        source_title: str,
        target_path: str | Path,
        target_title: str,
    ) -> tuple[str, str]:
        source_hash = self._ensure_links_section_entry(Path(source_path), target_title, Path(target_path))
        target_hash = self._ensure_links_section_entry(Path(target_path), source_title, Path(source_path))
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
        target_path = self.note_path(note_id, title, status="approved", current_path=source_path)
        if source_path.resolve(strict=False) != target_path.resolve(strict=False):
            ensure_directory(target_path.parent)
            source_path.replace(target_path)
        content_hash = self.update_note_status(target_path, "approved")
        return target_path, content_hash

    def _ensure_links_section_entry(self, note_path: Path, linked_title: str, linked_path: Path) -> str:
        existing = safe_read_text(note_path)
        link_line = f"- {format_obsidian_link(linked_title, linked_path)}"
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
    ) -> tuple[bool, str, str, Path]:
        path = Path(note_path)
        existing = safe_read_text(path)
        normalized_path = self._preferred_note_path(
            self._base_dir_for_status(status, note_type),
            note_id,
            title,
            current_path=path,
        )
        path_changed = path.resolve(strict=False) != normalized_path.resolve(strict=False)
        if path_changed:
            ensure_directory(normalized_path.parent)
            path.replace(normalized_path)
            path = normalized_path
            existing = safe_read_text(path)
        if note_type == "atomic":
            body_without_frontmatter = self._body_without_frontmatter(existing)
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
            body = self._normalize_atomic_body(title, body_without_frontmatter)
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
            body = sanitize_note_markdown(self._body_without_frontmatter(existing))
            rendered = f"{frontmatter}\n\n{body}\n"
        if rendered == existing and not path_changed:
            return False, sha256_text(existing), title, path
        if rendered != existing:
            write_atomic_text(path, rendered)
            return True, sha256_text(rendered), title, path
        return True, sha256_text(existing), title, path

    def _normalize_atomic_body(self, title: str, content: str) -> str:
        body = sanitize_note_markdown(content)
        body = strip_leading_page_heading(body)
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

    def _preferred_note_path(
        self,
        base_dir: Path,
        note_id: str,
        title: str,
        *,
        current_path: str | Path | None = None,
    ) -> Path:
        stem = self._filename_stem(title, fallback=f"note {note_id[-6:]}")
        candidate = base_dir / f"{stem}.md"
        if self._path_is_available(candidate, current_path):
            return candidate
        return base_dir / f"{stem} ({note_id[-6:]}).md"

    def _filename_stem(self, title: str, *, fallback: str) -> str:
        text = INVALID_FILENAME_CHARS_RE.sub("", str(title or ""))
        text = SPACE_RE.sub(" ", text).strip().rstrip(".")
        return text or fallback

    def _path_is_available(self, candidate: Path, current_path: str | Path | None) -> bool:
        if current_path is not None and candidate.resolve(strict=False) == Path(current_path).resolve(strict=False):
            return True
        return not candidate.exists()

    def _base_dir_for_status(self, status: str, note_type: str) -> Path:
        if note_type == "archive":
            return self.archive_dir
        if status == "approved":
            return self.atomic_dir
        return self.inbox_dir
