from __future__ import annotations

import socket
import time

from ..config import SnowballConfig
from ..intake.receiver import register_events
from ..intake.transcript_poll import scan_transcripts
from ..models import AgentResult
from ..queue.task_claim import claim_next_task
from ..storage.reconcile import reconcile_vault_and_db
from ..storage.audit import write_audit_log


class SnowballWorker:
    def __init__(self, config: SnowballConfig, db, agent, vault):
        self.config = config
        self.db = db
        self.agent = agent
        self.vault = vault
        self.worker_id = f"{socket.gethostname()}:{id(self)}"
        self._startup_reconcile_done = False

    def run_once(self) -> AgentResult | None:
        self._maybe_run_startup_reconcile()
        events = scan_transcripts(self.config, self.db)
        register_events(self.db, self.config, events)
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
        try:
            reconcile_vault_and_db(self.vault.root, self.db)
            self.db.commit()
        except Exception as exc:  # pragma: no cover - defensive
            write_audit_log(
                self.db,
                "reconcile_error",
                {"error": str(exc)},
                level="error",
            )
            self.db.commit()
