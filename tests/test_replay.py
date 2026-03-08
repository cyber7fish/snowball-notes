import json
import tempfile
import unittest
from pathlib import Path

from snowball_notes.agent.replay import ReplayRunner, _apply_config_snapshot
from snowball_notes.config import default_config
from snowball_notes.models import RunState, StandardEvent
from snowball_notes.storage.sqlite import Database
from snowball_notes.storage.vault import Vault
from snowball_notes.utils import new_id, now_utc_iso, sha256_text


def _sample_event(**overrides) -> StandardEvent:
    defaults = dict(
        event_id="evt_replay_test",
        session_file="/tmp/session.jsonl",
        conversation_id="conv_replay_test",
        turn_id="turn_replay_test",
        user_message="How does replay work?",
        assistant_final_answer="Replay works by freezing tool inputs and outputs into a bundle, then replaying the agent with those frozen results. " * 3,
        displayed_at="2026-03-08T00:00:00+00:00",
        source_completeness="full",
        source_confidence=0.95,
        parser_version="v1",
        context_meta={"client": "codex", "cwd": "/tmp"},
    )
    defaults.update(overrides)
    return StandardEvent(**defaults)


def _seed_full_trace(db, vault, event, config, decision="create_note"):
    """Seed a complete trace + replay bundle + proposals to enable replay."""
    trace_id = new_id("trace")

    tool_results = [
        {
            "step": 0,
            "tool": "assess_turn_value",
            "input": {},
            "output": {"decision": "note", "reason": ["long_term_value"], "confidence": event.source_confidence},
            "success": True,
        },
        {
            "step": 1,
            "tool": "extract_knowledge_points",
            "input": {},
            "output": {
                "candidate_title": "How does replay work",
                "summary": "Replay works by freezing tool inputs.",
                "key_points": ["Freeze tool results", "Compare decisions"],
                "topics": ["replay"],
                "tags": ["replay", "codex", "snowball-notes"],
            },
            "success": True,
        },
        {
            "step": 2,
            "tool": "search_similar_notes",
            "input": {"query": "How does replay work"},
            "output": [],
            "success": True,
        },
        {
            "step": 3,
            "tool": "propose_create_note",
            "input": {
                "title": "How does replay work",
                "content": "## Summary\nReplay works by freezing tool inputs.\n\n## Key Points\n- Freeze tool results\n- Compare decisions",
                "tags": ["replay", "codex", "snowball-notes"],
                "topics": ["replay"],
            },
            "output": {"proposal_id": "proposal_replay", "action_type": "create_note"},
            "success": True,
        },
    ]

    trace_steps = [
        {
            "step_index": i,
            "runtime_state": "running",
            "decision_summary": tr["tool"],
            "tool_name": tr["tool"],
            "tool_input_json": json.dumps(tr["input"]),
            "tool_result_json": json.dumps(tr["output"]),
            "tool_success": tr["success"],
            "proposal_ids": ["proposal_replay"] if tr["tool"].startswith("propose_") else [],
            "guardrail_blocked": False,
            "duration_ms": 1,
            "input_tokens": 100,
            "output_tokens": 50,
        }
        for i, tr in enumerate(tool_results)
    ]

    trace_json = {
        "trace_id": trace_id,
        "event_id": event.event_id,
        "turn_id": event.turn_id,
        "prompt_version": config.agent.prompt_version,
        "model_name": "heuristic-v1",
        "started_at": now_utc_iso(),
        "finished_at": now_utc_iso(),
        "total_steps": len(trace_steps),
        "exceeded_max_steps": 0,
        "terminal_reason": "completed",
        "final_decision": decision,
        "final_confidence": event.source_confidence,
        "total_input_tokens": 400,
        "total_output_tokens": 200,
        "total_duration_ms": 50,
        "steps": trace_steps,
    }

    db.execute(
        """
        INSERT INTO agent_traces (
          trace_id, turn_id, event_id, prompt_version, model_name,
          started_at, finished_at, total_steps, exceeded_max_steps,
          terminal_reason, final_decision, final_confidence,
          total_input_tokens, total_output_tokens, total_duration_ms, trace_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            trace_id, event.turn_id, event.event_id,
            config.agent.prompt_version, "heuristic-v1",
            now_utc_iso(), now_utc_iso(),
            len(trace_steps), 0,
            "completed", decision, event.source_confidence,
            400, 200, 50, json.dumps(trace_json),
        ),
    )

    prompt_path = Path(__file__).resolve().parents[1] / "src" / "snowball_notes" / "prompts" / config.agent.prompt_version
    prompt_text = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""

    db.execute(
        """
        INSERT INTO replay_bundles (
          trace_id, event_json, prompt_snapshot, config_snapshot_json,
          tool_results_json, knowledge_snapshot_refs_json,
          model_name, model_adapter_version, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            trace_id,
            json.dumps(event.to_dict(), ensure_ascii=False),
            prompt_text,
            json.dumps(config.to_dict(), ensure_ascii=False),
            json.dumps(tool_results, ensure_ascii=False),
            json.dumps([]),
            "heuristic-v1",
            "heuristic-v1",
            now_utc_iso(),
        ),
    )

    db.execute(
        """
        INSERT INTO action_proposals (
          proposal_id, trace_id, turn_id, action_type, target_note_id,
          payload_json, idempotency_key, status, created_at, committed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "proposal_replay", trace_id, event.turn_id, "create_note", None,
            json.dumps({
                "title": "How does replay work",
                "content": "## Summary\nReplay works by freezing tool inputs.",
                "tags": ["replay"],
                "topics": ["replay"],
                "source_event_id": event.event_id,
            }),
            f"create:{event.turn_id}:replay",
            "committed", now_utc_iso(), now_utc_iso(),
        ),
    )
    db.commit()
    return trace_id


class LogicalReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = default_config(Path(self.temp_dir))
        self.db = Database(self.config.db_path)
        self.db.migrate()
        self.vault = Vault(self.config)

    def tearDown(self):
        self.db.close()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_logical_replay_produces_same_decision(self):
        event = _sample_event()
        trace_id = _seed_full_trace(self.db, self.vault, event, self.config)
        runner = ReplayRunner(self.config, self.db, self.vault)
        outcome = runner.logical_replay(trace_id)
        self.assertEqual(outcome.mode, "logical")
        self.assertEqual(outcome.final_decision, "create_note")
        self.assertTrue(outcome.matched_original)

    def test_logical_replay_missing_trace_raises(self):
        runner = ReplayRunner(self.config, self.db, self.vault)
        with self.assertRaises(ValueError):
            runner.logical_replay("trace_nonexistent")


class LiveReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = default_config(Path(self.temp_dir))
        self.db = Database(self.config.db_path)
        self.db.migrate()
        self.vault = Vault(self.config)

    def tearDown(self):
        self.db.close()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_live_replay_runs_to_completion(self):
        event = _sample_event()
        trace_id = _seed_full_trace(self.db, self.vault, event, self.config)
        runner = ReplayRunner(self.config, self.db, self.vault)
        outcome = runner.live_replay(trace_id)
        self.assertEqual(outcome.mode, "live")
        self.assertIn(outcome.result_state, [s.value for s in RunState])


class ConfigSnapshotTests(unittest.TestCase):
    def test_apply_config_snapshot_overrides_values(self):
        config = default_config(Path("/tmp"))
        _apply_config_snapshot(config, {"agent": {"max_steps": 4}})
        self.assertEqual(config.agent.max_steps, 4)

    def test_apply_config_snapshot_ignores_unknown_sections(self):
        config = default_config(Path("/tmp"))
        _apply_config_snapshot(config, {"nonexistent_section": {"key": "value"}})
        self.assertFalse(hasattr(config, "nonexistent_section"))

    def test_apply_config_snapshot_ignores_non_dict_values(self):
        config = default_config(Path("/tmp"))
        original_model = config.agent.model
        _apply_config_snapshot(config, {"agent": "not_a_dict"})
        self.assertEqual(config.agent.model, original_model)


if __name__ == "__main__":
    unittest.main()
