from __future__ import annotations

import re
from pathlib import Path


LOCAL_LINK_RE = re.compile(r"\[([^\]]+)\]\((/Users/[^)\s]+(?:#[^)]+)?)\)")
WIKILINK_ALIAS_RE = re.compile(r"\[\[([^\]|]+)\|([^\]]+)\]\]")


def sanitize_note_markdown(content: str) -> str:
    sanitized = LOCAL_LINK_RE.sub(_replace_local_link, content)
    return sanitized.replace("\r\n", "\n").strip()


def strip_leading_page_heading(content: str) -> str:
    body = content.lstrip()
    if not body.startswith("# "):
        return content.strip()
    first_line, _, remainder = body.partition("\n")
    if not first_line.startswith("# "):
        return content.strip()
    return remainder.lstrip()


def update_wikilink_titles(content: str, old_title: str, new_title: str) -> str:
    if old_title == new_title:
        return content
    updated = content.replace(f"[[{old_title}]]", f"[[{new_title}]]")
    updated = updated.replace(f"[[{old_title}|", f"[[{new_title}|")
    updated = WIKILINK_ALIAS_RE.sub(
        lambda match: f"[[{match.group(1)}|{new_title}]]" if match.group(2) == old_title else match.group(0),
        updated,
    )
    return updated


def format_obsidian_link(title: str, note_path: str | Path | None = None) -> str:
    if note_path is None:
        return f"[[{title}]]"
    basename = Path(note_path).stem
    if not basename or basename == title:
        return f"[[{title}]]"
    return f"[[{basename}|{title}]]"


def rewrite_wikilink_target(content: str, title: str, note_path: str | Path | None) -> str:
    desired = format_obsidian_link(title, note_path)
    plain = f"[[{title}]]"
    updated = content.replace(plain, desired)
    updated = updated.replace(f"[[{title}|{title}]]", desired)
    updated = WIKILINK_ALIAS_RE.sub(
        lambda match: desired if match.group(2) == title else match.group(0),
        updated,
    )
    return updated


def _replace_local_link(match: re.Match[str]) -> str:
    label = match.group(1).strip()
    if not label:
        return ""
    if any(token in label for token in (".py", ".md", ".yaml", ".yml", ".toml", ".json", "/", "#L", "#")):
        return f"`{label}`"
    return label
