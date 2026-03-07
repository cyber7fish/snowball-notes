from __future__ import annotations

import socket
import time

from ..config import SnowballConfig
from ..intake.receiver import register_events
from ..intake.transcript_poll import scan_transcripts
from ..models import AgentResult
from ..queue.task_claim import claim_next_task
from ..storage.audit import write_audit_log


class SnowballWorker:
    def __init__(self, config: SnowballConfig, db, agent):
        self.config = config
        self.db = db
        self.agent = agent
        self.worker_id = f"{socket.gethostname()}:{id(self)}"

    def run_once(self) -> AgentResult | None:
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

