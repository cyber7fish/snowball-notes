import tempfile
import unittest
from pathlib import Path

from snowball_notes.agent.state_machine import InvalidStateTransition, transition_state
from snowball_notes.models import RunState
from snowball_notes.storage.sqlite import Database


class StateMachineTests(unittest.TestCase):
    def test_valid_transition_updates_task(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(Path(temp_dir) / "snowball.db")
            db.migrate()
            db.execute(
                """
                INSERT INTO tasks (task_id, event_id, dedupe_key, status)
                VALUES ('task_1', 'evt_1', 'dedupe_1', ?)
                """,
                (RunState.RECEIVED.value,),
            )
            db.commit()
            transition_state(db, "task_1", RunState.RECEIVED, RunState.PREPARED)
            row = db.fetchone("SELECT status FROM tasks WHERE task_id = 'task_1'")
            self.assertEqual(row["status"], RunState.PREPARED.value)
            db.close()

    def test_invalid_transition_raises(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Database(Path(temp_dir) / "snowball.db")
            db.migrate()
            db.execute(
                """
                INSERT INTO tasks (task_id, event_id, dedupe_key, status)
                VALUES ('task_1', 'evt_1', 'dedupe_1', ?)
                """,
                (RunState.RECEIVED.value,),
            )
            db.commit()
            with self.assertRaises(InvalidStateTransition):
                transition_state(db, "task_1", RunState.RECEIVED, RunState.COMPLETED)
            db.close()


if __name__ == "__main__":
    unittest.main()

