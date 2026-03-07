from __future__ import annotations

from ..models import RunState
from ..storage.audit import write_audit_log


class InvalidStateTransition(RuntimeError):
    pass


class StateTransitionConflict(RuntimeError):
    pass


VALID_TRANSITIONS = {
    RunState.RECEIVED: {RunState.PREPARED},
    RunState.PREPARED: {RunState.RUNNING},
    RunState.RUNNING: {
        RunState.PROPOSED_ACTIONS,
        RunState.FLAGGED,
        RunState.FAILED_RETRYABLE,
        RunState.FAILED_FATAL,
    },
    RunState.PROPOSED_ACTIONS: {RunState.COMMITTING, RunState.FLAGGED},
    RunState.COMMITTING: {RunState.COMPLETED, RunState.FAILED_RETRYABLE, RunState.FAILED_FATAL},
    RunState.COMPLETED: set(),
    RunState.FLAGGED: set(),
    RunState.FAILED_RETRYABLE: {RunState.RECEIVED, RunState.PREPARED},
    RunState.FAILED_FATAL: set(),
}


def transition_state(db, task_id: str, current: RunState, target: RunState, reason: str = "") -> None:
    if target not in VALID_TRANSITIONS[current]:
        raise InvalidStateTransition(f"{current.value} -> {target.value} is invalid")
    updated = db.execute(
        """
        UPDATE tasks
        SET status = ?, updated_at = CURRENT_TIMESTAMP
        WHERE task_id = ? AND status = ?
        """,
        (target.value, task_id, current.value),
    )
    if updated.rowcount != 1:
        raise StateTransitionConflict(f"task {task_id} state changed concurrently")
    write_audit_log(
        db,
        "state_transition",
        {"from": current.value, "to": target.value, "reason": reason},
        task_id=task_id,
    )

