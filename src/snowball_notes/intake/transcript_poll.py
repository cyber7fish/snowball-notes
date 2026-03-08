from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..config import SnowballConfig
from ..utils import now_utc_iso
from .transcript_parser import parse_session_file


@dataclass
class IntakeWatchState:
    known_files: dict[str, float] = field(default_factory=dict)


def collect_transcript_events(
    config: SnowballConfig,
    db,
    watch_state: IntakeWatchState | None = None,
) -> list:
    mode = str(getattr(config.intake, "mode", "transcript_poll") or "transcript_poll").strip().lower()
    if mode == "transcript_watch":
        return watch_transcripts(config, db, watch_state or IntakeWatchState())
    if mode == "cli_wrap":
        return scan_cli_wrap_file(config, db)
    return scan_transcripts(config, db)


def scan_transcripts(config: SnowballConfig, db) -> list:
    transcript_dir = config.transcript_dir
    if not transcript_dir.exists():
        return []
    return _scan_paths(
        config,
        db,
        sorted(transcript_dir.rglob("*.jsonl")),
        respect_cursor=True,
    )


def watch_transcripts(config: SnowballConfig, db, watch_state: IntakeWatchState) -> list:
    events = []
    transcript_dir = config.transcript_dir
    if not transcript_dir.exists():
        return events
    current_files: dict[str, float] = {}
    candidate_paths: list[Path] = []
    for path in sorted(transcript_dir.rglob("*.jsonl")):
        stat = path.stat()
        resolved = str(path.resolve())
        current_files[resolved] = stat.st_mtime
        previous_mtime = watch_state.known_files.get(resolved)
        if previous_mtime is None or previous_mtime < stat.st_mtime or _has_stale_events(db, path):
            candidate_paths.append(path)
    watch_state.known_files = current_files
    return _scan_paths(config, db, candidate_paths, respect_cursor=True)


def scan_cli_wrap_file(config: SnowballConfig, db) -> list:
    cli_wrap_path = config.cli_wrap_path
    if cli_wrap_path is None or not cli_wrap_path.exists():
        return []
    return _scan_paths(config, db, [cli_wrap_path], respect_cursor=True)


def _scan_paths(config: SnowballConfig, db, paths: list[Path], *, respect_cursor: bool) -> list:
    events = []
    for path in paths:
        stat = path.stat()
        cursor = db.get_cursor(str(path))
        if respect_cursor and cursor and float(cursor["last_mtime"]) >= stat.st_mtime and not _has_stale_events(db, path):
            continue
        events.extend(parse_session_file(path, parser_version=config.intake.parser_version))
        db.upsert_cursor(str(path), stat.st_mtime, now_utc_iso())
    db.commit()
    return events


def _has_stale_events(db, path: Path) -> bool:
    row = db.fetchone(
        """
        SELECT 1 AS has_stale
        FROM conversation_events
        WHERE session_file = ?
          AND assistant_final_answer = ''
        LIMIT 1
        """,
        (str(path),),
    )
    return row is not None
