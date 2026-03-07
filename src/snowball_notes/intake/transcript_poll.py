from __future__ import annotations

from pathlib import Path

from ..config import SnowballConfig
from ..utils import now_utc_iso
from .transcript_parser import parse_session_file


def scan_transcripts(config: SnowballConfig, db) -> list:
    events = []
    transcript_dir = config.transcript_dir
    if not transcript_dir.exists():
        return events
    for path in sorted(transcript_dir.rglob("*.jsonl")):
        stat = path.stat()
        cursor = db.get_cursor(str(path))
        if cursor and float(cursor["last_mtime"]) >= stat.st_mtime:
            continue
        events.extend(parse_session_file(path, parser_version=config.intake.parser_version))
        db.upsert_cursor(str(path), stat.st_mtime, now_utc_iso())
    db.commit()
    return events

