import tempfile
import unittest
from pathlib import Path

from snowball_notes.agent.state_machine import (
    InvalidStateTransition,
    StateTransitionConflict,
    VALID_TRANSITIONS,
    transition_state,
)
from snowball_notes.models import RunState
from snowball_notes.storage.sqlite import Database


def _insert_task(db, task_id, status):
    db.execute(
        """
        INSERT INTO tasks (task_id, event_id, dedupe_key, status)
        VALUES (?, ?, ?, ?)
        """,
        (task_id, f"evt_{task_id}", f"dedupe_{task_id}", status.value),
    )
    db.commit()


class ValidTransitionTests(unittest.TestCase):
    """Verify every valid transition in VALID_TRANSITIONS succeeds."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db = Database(Path(self.temp_dir) / "snowball.db")
        self.db.migrate()
        self._counter = 0

    def tearDown(self):
        self.db.close()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _run_transition(self, from_state, to_state):
        self._counter += 1
        task_id = f"task_{self._counter}"
        _insert_task(self.db, task_id, from_state)
        transition_state(self.db, task_id, from_state, to_state)
        row = self.db.fetchone("SELECT status FROM tasks WHERE task_id = ?", (task_id,))
        self.assertEqual(row["status"], to_state.value)

    def test_received_to_prepared(self):
        self._run_transition(RunState.RECEIVED, RunState.PREPARED)

    def test_prepared_to_running(self):
        self._run_transition(RunState.PREPARED, RunState.RUNNING)

    def test_running_to_proposed_actions(self):
        self._run_transition(RunState.RUNNING, RunState.PROPOSED_ACTIONS)

    def test_running_to_flagged(self):
        self._run_transition(RunState.RUNNING, RunState.FLAGGED)

    def test_running_to_failed_retryable(self):
        self._run_transition(RunState.RUNNING, RunState.FAILED_RETRYABLE)

    def test_running_to_failed_fatal(self):
        self._run_transition(RunState.RUNNING, RunState.FAILED_FATAL)

    def test_proposed_actions_to_committing(self):
        self._run_transition(RunState.PROPOSED_ACTIONS, RunState.COMMITTING)

    def test_proposed_actions_to_flagged(self):
        self._run_transition(RunState.PROPOSED_ACTIONS, RunState.FLAGGED)

    def test_committing_to_completed(self):
        self._run_transition(RunState.COMMITTING, RunState.COMPLETED)

    def test_committing_to_failed_retryable(self):
        self._run_transition(RunState.COMMITTING, RunState.FAILED_RETRYABLE)

    def test_committing_to_failed_fatal(self):
        self._run_transition(RunState.COMMITTING, RunState.FAILED_FATAL)

    def test_failed_retryable_to_received(self):
        self._run_transition(RunState.FAILED_RETRYABLE, RunState.RECEIVED)

    def test_failed_retryable_to_prepared(self):
        self._run_transition(RunState.FAILED_RETRYABLE, RunState.PREPARED)


class InvalidTransitionTests(unittest.TestCase):
    """Verify that invalid transitions are rejected."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db = Database(Path(self.temp_dir) / "snowball.db")
        self.db.migrate()

    def tearDown(self):
        self.db.close()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_received_to_completed_is_invalid(self):
        _insert_task(self.db, "task_inv_1", RunState.RECEIVED)
        with self.assertRaises(InvalidStateTransition):
            transition_state(self.db, "task_inv_1", RunState.RECEIVED, RunState.COMPLETED)

    def test_completed_has_no_outgoing_transitions(self):
        _insert_task(self.db, "task_inv_2", RunState.COMPLETED)
        for target in RunState:
            if target == RunState.COMPLETED:
                continue
            with self.assertRaises(InvalidStateTransition):
                transition_state(self.db, "task_inv_2", RunState.COMPLETED, target)

    def test_flagged_has_no_outgoing_transitions(self):
        _insert_task(self.db, "task_inv_3", RunState.FLAGGED)
        with self.assertRaises(InvalidStateTransition):
            transition_state(self.db, "task_inv_3", RunState.FLAGGED, RunState.RUNNING)

    def test_failed_fatal_has_no_outgoing_transitions(self):
        _insert_task(self.db, "task_inv_4", RunState.FAILED_FATAL)
        with self.assertRaises(InvalidStateTransition):
            transition_state(self.db, "task_inv_4", RunState.FAILED_FATAL, RunState.RECEIVED)


class ConcurrencyTests(unittest.TestCase):
    """Verify rowcount-based concurrent modification detection."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db = Database(Path(self.temp_dir) / "snowball.db")
        self.db.migrate()

    def tearDown(self):
        self.db.close()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_conflict_when_status_already_changed(self):
        _insert_task(self.db, "task_conc", RunState.RUNNING)
        self.db.execute(
            "UPDATE tasks SET status = ? WHERE task_id = ?",
            (RunState.FLAGGED.value, "task_conc"),
        )
        self.db.commit()
        with self.assertRaises(StateTransitionConflict):
            transition_state(self.db, "task_conc", RunState.RUNNING, RunState.PROPOSED_ACTIONS)

    def test_conflict_when_task_does_not_exist(self):
        with self.assertRaises(StateTransitionConflict):
            transition_state(self.db, "task_ghost", RunState.RUNNING, RunState.FLAGGED)


class TransitionCompletenessTests(unittest.TestCase):
    """Verify VALID_TRANSITIONS covers all RunState values."""

    def test_all_states_have_entries(self):
        for state in RunState:
            self.assertIn(state, VALID_TRANSITIONS,
                          f"{state} missing from VALID_TRANSITIONS")


if __name__ == "__main__":
    unittest.main()
