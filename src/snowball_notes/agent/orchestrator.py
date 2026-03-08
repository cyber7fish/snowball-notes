from __future__ import annotations

import socket
import time
from datetime import datetime, timezone

from ..config import SnowballConfig
from ..intake.receiver import register_events
from ..intake.transcript_poll import IntakeWatchState, collect_transcript_events
from ..models import AgentResult
from ..queue.task_claim import claim_next_task
from ..storage.reconcile import reconcile_vault_and_db
from ..storage.audit import write_audit_log
from ..utils import now_utc, parse_datetime


class SnowballWorker:
    def __init__(self, config: SnowballConfig, db, agent, vault):
        self.config = config
        self.db = db
        self.agent = agent
        self.vault = vault
        self.worker_id = f"{socket.gethostname()}:{id(self)}"
        self._startup_reconcile_done = False
        self._watch_state = IntakeWatchState()
        self._last_reconcile_at = self._load_last_reconcile_at()

    def run_once(self) -> AgentResult | None:
        self._maybe_run_startup_reconcile()
        self._maybe_run_scheduled_reconcile()
        events = collect_transcript_events(self.config, self.db, self._watch_state)
        enqueued = register_events(self.db, self.config, events)
        write_audit_log(
            self.db,
            "worker_scan_completed",
            {
                "intake_mode": self.config.intake.mode,
                "discovered_events": len(events),
                "enqueued_tasks": enqueued,
            },
        )
        claimed = claim_next_task(self.db, self.worker_id, self.config.worker.claim_timeout_seconds)
        self.db.commit()
        if claimed is None:
            return None
        task, event = claimed
        result = self.agent.run(task, event)
        write_audit_log(
            self.db,
            "worker_run_completed",
            {"result_state": result.state.value, "reason": result.reason},
            task_id=task.task_id,
            turn_id=event.turn_id,
        )
        self.db.commit()
        return result

    def run_forever(self) -> None:
        while True:
            self.run_once()
            time.sleep(self.config.worker.poll_interval_seconds)

    def _maybe_run_startup_reconcile(self) -> None:
        if self._startup_reconcile_done:
            return
        self._startup_reconcile_done = True
        if not (self.config.reconcile.enabled and self.config.reconcile.run_on_startup):
            return
        self._run_reconcile("startup")

    def _maybe_run_scheduled_reconcile(self) -> None:
        if not self.config.reconcile.enabled:
            return
        now = now_utc()
        if not _scheduled_reconcile_due(now, self._last_reconcile_at, self.config.reconcile.schedule_cron):
            return
        self._run_reconcile("scheduled")

    def _run_reconcile(self, trigger: str) -> None:
        try:
            reconcile_vault_and_db(self.vault, self.db)
            self._last_reconcile_at = now_utc()
            self.db.commit()
        except Exception as exc:  # pragma: no cover - defensive
            write_audit_log(
                self.db,
                "reconcile_error",
                {"error": str(exc), "trigger": trigger},
                level="error",
            )
            self.db.commit()

    def _load_last_reconcile_at(self):
        row = self.db.fetchone(
            """
            SELECT created_at
            FROM audit_logs
            WHERE event_type IN ('reconcile_completed', 'reconcile_mismatch', 'reconcile_error')
            ORDER BY created_at DESC, rowid DESC
            LIMIT 1
            """
        )
        return parse_datetime(row["created_at"]) if row else None


def _scheduled_reconcile_due(
    now: datetime,
    last_reconcile_at: datetime | None,
    schedule_cron: str,
) -> bool:
    scheduled_for = _scheduled_slot_utc(now, schedule_cron)
    if scheduled_for is None or now < scheduled_for:
        return False
    if last_reconcile_at is None:
        return True
    return last_reconcile_at < scheduled_for


def _scheduled_slot_utc(now: datetime, schedule_cron: str) -> datetime | None:
    parts = schedule_cron.strip().split()
    if len(parts) != 5:
        return None
    minute, hour, day, month, weekday = parts
    if day != "*" or month != "*" or weekday != "*":
        return None
    try:
        minute_value = int(minute)
        hour_value = int(hour)
    except ValueError:
        return None
    if not (0 <= minute_value <= 59 and 0 <= hour_value <= 23):
        return None
    current = now.astimezone(timezone.utc)
    return current.replace(hour=hour_value, minute=minute_value, second=0, microsecond=0)
