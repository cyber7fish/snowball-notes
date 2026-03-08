from __future__ import annotations

import json
import shutil
from pathlib import Path

from .config import load_config
from .eval import EvalRunner, import_eval_cases, load_eval_cases, render_eval_report
from .models import StandardEvent
from .storage.sqlite import Database
from .utils import ensure_directory, now_utc_iso, write_atomic_text


REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_FIXTURE_ROOT = REPO_ROOT / "demo" / "fixtures"
DEMO_TRANSCRIPT_ROOT = DEMO_FIXTURE_ROOT / "transcripts"
DEMO_EVAL_FIXTURE_PATH = REPO_ROOT / "eval" / "fixtures" / "sample_cases.json"


def setup_demo_workspace(destination: str | Path) -> dict[str, str | int]:
    root = Path(destination).resolve()
    ensure_directory(root)
    ensure_directory(root / "sessions")
    ensure_directory(root / "reports")
    ensure_directory(root / "eval" / "fixtures")

    write_atomic_text(root / "config.yaml", _demo_config_text())
    write_atomic_text(root / "README.md", _demo_workspace_readme(root))
    transcript_count = _copy_demo_transcripts(root / "sessions")
    shutil.copy2(DEMO_EVAL_FIXTURE_PATH, root / "eval" / "fixtures" / "sample_cases.json")

    config = load_config(root / "config.yaml")
    db = Database(config.db_path)
    db.migrate()
    try:
        review_id = _seed_review_fixture(db, config)
        imported = import_eval_cases(db, root / "eval" / "fixtures" / "sample_cases.json", replace=True)
        report = EvalRunner(config, db).run(load_eval_cases(db))
        report_payload = report.to_dict()
        report_text = render_eval_report(report_payload)
        write_atomic_text(root / "reports" / "sample_eval_report.txt", report_text + "\n")
        write_atomic_text(
            root / "reports" / "sample_eval_report.json",
            json.dumps(report_payload, ensure_ascii=False, indent=2) + "\n",
        )
        return {
            "workspace_root": str(root),
            "config_path": str(root / "config.yaml"),
            "sessions_dir": str(root / "sessions"),
            "report_path": str(root / "reports" / "sample_eval_report.txt"),
            "review_id": review_id,
            "transcript_count": transcript_count,
            "eval_case_count": imported,
            "eval_run_id": report.run_id,
        }
    finally:
        db.commit()
        db.close()


def _copy_demo_transcripts(destination: Path) -> int:
    ensure_directory(destination)
    copied = 0
    for source in sorted(DEMO_TRANSCRIPT_ROOT.glob("*.jsonl")):
        shutil.copy2(source, destination / source.name)
        copied += 1
    return copied


def _seed_review_fixture(db, config) -> str:
    event = StandardEvent(
        event_id="evt_demo_review_1",
        session_file=str((config.project_root / "sessions" / "03-ambiguous-review.jsonl").resolve()),
        conversation_id="conv_demo_review",
        turn_id="turn_demo_review_1",
        user_message="How should a guarded agent handle an ambiguous merge into the knowledge base?",
        assistant_final_answer=(
            "When a new turn partially overlaps an existing note, the agent should stop before writing, "
            "capture the reasoning trace, and let a human reviewer inspect the replay bundle before approving "
            "any append or create action."
        ),
        displayed_at="2026-03-08T09:30:00+00:00",
        source_completeness="full",
        source_confidence=0.91,
        parser_version="v1",
        context_meta={"cwd": str(config.project_root), "client": "demo_fixture"},
    )
    started_at = now_utc_iso()
    finished_at = now_utc_iso()
    trace_json = {
        "trace_id": "trace_demo_review_1",
        "event_id": event.event_id,
        "turn_id": event.turn_id,
        "prompt_version": config.agent.prompt_version,
        "model_name": config.agent.model,
        "started_at": started_at,
        "finished_at": finished_at,
        "total_steps": 4,
        "exceeded_max_steps": 0,
        "terminal_reason": "ambiguous_existing_note",
        "final_decision": "flagged",
        "final_confidence": event.source_confidence,
        "total_input_tokens": 180,
        "total_output_tokens": 64,
        "total_duration_ms": 420,
        "steps": [
            {
                "step_index": 0,
                "runtime_state": "running",
                "decision_summary": "Assess whether the turn should write, archive, or stop.",
                "tool_name": "assess_turn_value",
                "tool_input_json": json.dumps({}, ensure_ascii=False),
                "tool_result_json": json.dumps(
                    {"decision": "note", "reason": ["long_term_value"], "confidence": event.source_confidence},
                    ensure_ascii=False,
                ),
                "tool_success": True,
                "proposal_ids": [],
                "guardrail_blocked": False,
                "duration_ms": 14,
                "input_tokens": 40,
                "output_tokens": 10,
            },
            {
                "step_index": 1,
                "runtime_state": "running",
                "decision_summary": "Extract a durable summary before searching.",
                "tool_name": "extract_knowledge_points",
                "tool_input_json": json.dumps({}, ensure_ascii=False),
                "tool_result_json": json.dumps(
                    {
                        "candidate_title": "How should a guarded agent handle an ambiguous merge into the knowledge base",
                        "summary": "Stop before writing and inspect the replay bundle when the merge is ambiguous.",
                        "tags": ["agent", "review", "snowball-notes"],
                        "topics": ["guardrails", "review"],
                    },
                    ensure_ascii=False,
                ),
                "tool_success": True,
                "proposal_ids": [],
                "guardrail_blocked": False,
                "duration_ms": 21,
                "input_tokens": 42,
                "output_tokens": 16,
            },
            {
                "step_index": 2,
                "runtime_state": "running",
                "decision_summary": "Search for a potentially overlapping note.",
                "tool_name": "search_similar_notes",
                "tool_input_json": json.dumps(
                    {"query": "How should a guarded agent handle an ambiguous merge into the knowledge base", "top_k": 5},
                    ensure_ascii=False,
                ),
                "tool_result_json": json.dumps(
                    [
                        {
                            "note_id": "demo_note_guardrails",
                            "title": "Agent Guardrails for Ambiguous Note Merges",
                            "similarity": 0.74,
                        }
                    ],
                    ensure_ascii=False,
                ),
                "tool_success": True,
                "proposal_ids": [],
                "guardrail_blocked": False,
                "duration_ms": 29,
                "input_tokens": 54,
                "output_tokens": 20,
            },
            {
                "step_index": 3,
                "runtime_state": "running",
                "decision_summary": "Escalate the ambiguous merge for review.",
                "tool_name": "flag_for_review",
                "tool_input_json": json.dumps(
                    {
                        "reason": "ambiguous_existing_note",
                        "suggested_action": "create_note",
                        "suggested_payload": {
                            "title": "Guarded Review Flow for Ambiguous Merges",
                            "content": "## Summary\nFlag the turn, inspect trace and replay, then approve a clean write.\n",
                            "tags": ["agent", "review"],
                            "topics": ["guardrails", "replay"],
                        },
                    },
                    ensure_ascii=False,
                ),
                "tool_result_json": json.dumps(
                    {"flagged": True, "review_id": "review_demo_1", "reason": "ambiguous_existing_note"},
                    ensure_ascii=False,
                ),
                "tool_success": True,
                "proposal_ids": [],
                "guardrail_blocked": False,
                "duration_ms": 18,
                "input_tokens": 44,
                "output_tokens": 18,
            },
        ],
    }

    db.execute(
        """
        INSERT OR REPLACE INTO conversation_events (
          event_id, turn_id, conversation_id, session_file, user_message,
          assistant_final_answer, displayed_at, source_completeness,
          source_confidence, parser_version, context_meta_json, payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event.event_id,
            event.turn_id,
            event.conversation_id,
            event.session_file,
            event.user_message,
            event.assistant_final_answer,
            event.displayed_at,
            event.source_completeness,
            event.source_confidence,
            event.parser_version,
            json.dumps(event.context_meta, ensure_ascii=False),
            json.dumps(event.to_dict(), ensure_ascii=False),
        ),
    )
    db.execute(
        """
        INSERT OR REPLACE INTO agent_traces (
          trace_id, turn_id, event_id, prompt_version, model_name, started_at, finished_at,
          total_steps, exceeded_max_steps, terminal_reason, final_decision, final_confidence,
          total_input_tokens, total_output_tokens, total_duration_ms, trace_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "trace_demo_review_1",
            event.turn_id,
            event.event_id,
            config.agent.prompt_version,
            config.agent.model,
            started_at,
            finished_at,
            trace_json["total_steps"],
            trace_json["exceeded_max_steps"],
            trace_json["terminal_reason"],
            trace_json["final_decision"],
            trace_json["final_confidence"],
            trace_json["total_input_tokens"],
            trace_json["total_output_tokens"],
            trace_json["total_duration_ms"],
            json.dumps(trace_json, ensure_ascii=False),
        ),
    )
    db.execute(
        """
        INSERT OR REPLACE INTO replay_bundles (
          trace_id, event_json, prompt_snapshot, config_snapshot_json, tool_results_json,
          knowledge_snapshot_refs_json, model_name, model_adapter_version, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "trace_demo_review_1",
            json.dumps(event.to_dict(), ensure_ascii=False),
            "Demo prompt snapshot for the review workspace.",
            json.dumps(config.to_dict(), ensure_ascii=False),
            json.dumps(
                [
                    {"step": 0, "tool": "assess_turn_value", "output": {"decision": "note"}, "success": True},
                    {"step": 1, "tool": "extract_knowledge_points", "output": {"candidate_title": "Guarded Review Flow"}, "success": True},
                    {
                        "step": 2,
                        "tool": "search_similar_notes",
                        "output": [{"note_id": "demo_note_guardrails", "similarity": 0.74}],
                        "success": True,
                    },
                    {
                        "step": 3,
                        "tool": "flag_for_review",
                        "output": {"flagged": True, "review_id": "review_demo_1"},
                        "success": True,
                    },
                ],
                ensure_ascii=False,
            ),
            json.dumps(
                [
                    {
                        "note_id": "demo_note_guardrails",
                        "content_hash": "demo-content-hash",
                        "title": "Agent Guardrails for Ambiguous Note Merges",
                        "similarity": 0.74,
                    }
                ],
                ensure_ascii=False,
            ),
            config.agent.model,
            "demo-fixture-v1",
            now_utc_iso(),
        ),
    )
    db.execute(
        """
        INSERT OR REPLACE INTO review_actions (
          review_id, turn_id, trace_id, final_action, final_target_note_id,
          suggested_action, suggested_target_note_id, suggested_payload_json,
          reviewer, reason, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "review_demo_1",
            event.turn_id,
            "trace_demo_review_1",
            "pending_review",
            None,
            "create_note",
            None,
            json.dumps(
                {
                    "title": "Guarded Review Flow for Ambiguous Merges",
                    "content": (
                        "## Summary\n"
                        "Flag ambiguous merges, inspect trace and replay, then approve a clean write.\n\n"
                        "## Key Points\n"
                        "- stop before mutating the vault when overlap is uncertain\n"
                        "- use replay to inspect the exact tool path\n"
                        "- let a reviewer choose create, discard, or mark conflict\n"
                    ),
                    "tags": ["agent", "review", "snowball-notes"],
                    "topics": ["guardrails", "replay", "review"],
                },
                ensure_ascii=False,
            ),
            "demo-seed",
            "ambiguous_existing_note",
            now_utc_iso(),
        ),
    )
    return "review_demo_1"


def _demo_config_text() -> str:
    return "\n".join(
        [
            "paths:",
            "  db: \"./data/snowball.db\"",
            "  log: \"./logs/snowball.jsonl\"",
            "vault:",
            "  path: \"./vault\"",
            "  inbox_dir: \"Inbox\"",
            "  archive_dir: \"Archive/Conversations\"",
            "  atomic_dir: \"Knowledge/Atomic\"",
            "intake:",
            "  mode: \"transcript_poll\"",
            "  transcript_dir: \"./sessions\"",
            "  parser_version: \"v1\"",
            "  min_response_length: 120",
            "  min_confidence_to_run: 0.50",
            "agent:",
            "  provider: \"heuristic\"",
            "  model: \"heuristic-v1\"",
            "  max_steps: 8",
            "  prompt_version: \"agent_system/v1.md\"",
            "  max_writes_per_run: 1",
            "  max_appends_per_run: 1",
            "retrieval:",
            "  top_k: 5",
            "  append_threshold: 0.82",
            "  review_threshold: 0.62",
            "guardrails:",
            "  min_confidence_for_note: 0.70",
            "  min_confidence_for_append: 0.85",
            "embedding:",
            "  provider: \"local\"",
            "  local_model: \"hash-384\"",
            "  vector_store: \"sqlite_blob\"",
            "worker:",
            "  poll_interval_seconds: 10",
            "  claim_timeout_seconds: 300",
            "  max_retries: 3",
            "reconcile:",
            "  enabled: true",
            "  run_on_startup: true",
            "  schedule_cron: \"0 3 * * *\"",
            "",
        ]
    )


def _demo_workspace_readme(root: Path) -> str:
    return "\n".join(
        [
            "# Demo Workspace",
            "",
            "This workspace is generated by `snowball demo setup`.",
            "",
            "What is included:",
            "",
            "- `config.yaml`: heuristic + local embedding config for offline runs",
            "- `sessions/`: sample transcript fixtures for intake and worker demos",
            "- `reports/sample_eval_report.txt`: a generated eval report from the bundled fixture set",
            "- `review_demo_1`: a seeded pending review with trace and replay data",
            "",
            "Recommended next steps from the repository root:",
            "",
            f"- `PYTHONPATH=src .venv/bin/python -m snowball_notes.cli --config {root / 'config.yaml'} worker --once`",
            f"- `PYTHONPATH=src .venv/bin/python -m snowball_notes.cli --config {root / 'config.yaml'} review list`",
            f"- `PYTHONPATH=src .venv/bin/python -m snowball_notes.cli --config {root / 'config.yaml'} review serve --host 127.0.0.1 --port 8000`",
            f"- `PYTHONPATH=src .venv/bin/python -m snowball_notes.cli --config {root / 'config.yaml'} eval report`",
            "",
        ]
    )
