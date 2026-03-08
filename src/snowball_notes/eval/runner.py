from __future__ import annotations

import json
import tempfile
from copy import deepcopy
from pathlib import Path

from ..agent.adapter import build_model_adapter
from ..agent.memory import SQLiteKnowledgeIndex
from ..agent.replay import ReplayRunner
from ..agent.runtime import SnowballAgent
from ..agent.tools import build_tool_registry
from ..embedding import build_embedding_provider, build_vector_store
from ..models import EvalCase, EvalCaseResult, EvalReport, RunState, TaskRecord
from ..storage.sqlite import Database
from ..storage.vault import Vault
from ..utils import new_id, now_utc_iso


WRITE_DECISIONS = {"create_note", "append_note", "archive_turn", "link_notes"}
DECISION_ALIASES = {
    "create": "create_note",
    "create_note": "create_note",
    "append": "append_note",
    "append_note": "append_note",
    "archive": "archive_turn",
    "archive_turn": "archive_turn",
    "link": "link_notes",
    "link_notes": "link_notes",
    "flag": "flagged",
    "flagged": "flagged",
    "skip": "skip",
}


def load_eval_cases(db) -> list[EvalCase]:
    rows = db.fetchall(
        """
        SELECT case_id, turn_id, input_json, expected_decision, expected_target_note,
               expected_risk_level, unsafe_if_written, difficulty, annotator, notes
        FROM eval_cases
        ORDER BY created_at ASC, case_id ASC
        """
    )
    return [EvalCase.from_row(row) for row in rows]


def import_eval_cases(db, fixture_path: str | Path, *, replace: bool = False) -> int:
    path = Path(fixture_path).resolve()
    files = [path] if path.is_file() else sorted(path.rglob("*.json"))
    if replace:
        db.execute("DELETE FROM eval_cases")
    total = 0
    for file_path in files:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        raw_cases = payload.get("cases", payload) if isinstance(payload, dict) else payload
        if not isinstance(raw_cases, list):
            raise ValueError(f"unsupported eval fixture shape: {file_path}")
        for raw_case in raw_cases:
            case = EvalCase.from_dict(raw_case)
            row = case.to_row()
            db.execute(
                """
                INSERT OR REPLACE INTO eval_cases (
                  case_id, turn_id, input_json, expected_decision, expected_target_note,
                  expected_risk_level, unsafe_if_written, difficulty, annotator, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["case_id"],
                    row["turn_id"],
                    json.dumps(row["input_json"], ensure_ascii=False),
                    row["expected_decision"],
                    row["expected_target_note"],
                    row["expected_risk_level"],
                    row["unsafe_if_written"],
                    row["difficulty"],
                    row["annotator"],
                    row["notes"],
                ),
            )
            total += 1
    db.commit()
    return total


def load_eval_report(db, run_id: str | None = None) -> dict | None:
    if run_id is None:
        row = db.fetchone(
            """
            SELECT run_id, result_json, ran_at
            FROM eval_runs
            ORDER BY ran_at DESC, rowid DESC
            LIMIT 1
            """
        )
    else:
        row = db.fetchone(
            """
            SELECT run_id, result_json, ran_at
            FROM eval_runs
            WHERE run_id = ?
            """,
            (run_id,),
        )
    if row is None:
        return None
    report = json.loads(row["result_json"])
    report["run_id"] = row["run_id"]
    report["ran_at"] = row["ran_at"]
    return report


class EvalRunner:
    def __init__(self, config, db):
        self.config = config
        self.db = db

    def run(self, dataset: list[EvalCase], *, prompt_version: str | None = None) -> EvalReport:
        if not dataset:
            raise ValueError("no eval cases loaded")
        effective_prompt_version = prompt_version or self.config.agent.prompt_version
        results = [self._run_case(case, effective_prompt_version) for case in dataset]
        report = self._build_report(results, effective_prompt_version)
        self._save_report(report)
        self.db.commit()
        return report

    def _run_case(self, case: EvalCase, prompt_version: str) -> EvalCaseResult:
        with tempfile.TemporaryDirectory(prefix=f"snowball-eval-{case.case_id}-") as temp_dir:
            sandbox_root = Path(temp_dir).resolve()
            sandbox_config = deepcopy(self.config)
            sandbox_config.project_root = sandbox_root
            sandbox_config.agent.prompt_version = prompt_version
            sandbox_db = Database(sandbox_config.db_path)
            sandbox_db.migrate()
            sandbox_vault = Vault(sandbox_config)
            try:
                task = self._seed_case(sandbox_db, sandbox_vault, case)
                embedding_provider = build_embedding_provider(sandbox_config)
                vector_store = build_vector_store(sandbox_config, sandbox_db)
                knowledge_index = SQLiteKnowledgeIndex(
                    sandbox_db,
                    config=sandbox_config,
                    embedding_provider=embedding_provider,
                    vector_store=vector_store,
                )
                tools = build_tool_registry(sandbox_db, knowledge_index)
                adapter = build_model_adapter(sandbox_config)
                agent = SnowballAgent(sandbox_config, adapter, tools, sandbox_vault, sandbox_db)
                agent.run(task, case.event)
                sandbox_db.commit()

                trace_row = sandbox_db.fetchone(
                    """
                    SELECT trace_id, final_decision, total_steps, total_duration_ms,
                           total_input_tokens, total_output_tokens
                    FROM agent_traces
                    ORDER BY created_at DESC, rowid DESC
                    LIMIT 1
                    """
                )
                if trace_row is None:
                    raise RuntimeError(f"missing trace for eval case {case.case_id}")

                trace_id = trace_row["trace_id"]
                actual_decision = _normalize_decision(trace_row["final_decision"])
                actual_target_note = self._load_actual_target_note(sandbox_db, trace_id)
                has_write = actual_decision in WRITE_DECISIONS
                proposal_count = self._count_rows(
                    sandbox_db,
                    "SELECT COUNT(*) AS count FROM action_proposals WHERE trace_id = ?",
                    (trace_id,),
                )
                proposal_rejected = bool(
                    self._count_rows(
                        sandbox_db,
                        """
                        SELECT COUNT(*) AS count
                        FROM audit_logs
                        WHERE event_type = 'commit_blocked' AND trace_id = ?
                        """,
                        (trace_id,),
                    )
                )
                replay_runner = ReplayRunner(sandbox_config, sandbox_db, sandbox_vault)
                logical = replay_runner.logical_replay(trace_id)
                live = replay_runner.live_replay(trace_id)

                return EvalCaseResult(
                    case_id=case.case_id,
                    trace_id=trace_id,
                    actual_decision=actual_decision,
                    actual_target_note=actual_target_note,
                    decision_correct=actual_decision == _normalize_decision(case.expected_decision),
                    target_note_correct=(
                        None
                        if case.expected_target_note is None
                        else actual_target_note == case.expected_target_note
                    ),
                    has_write=has_write,
                    unsafe_write=has_write and case.unsafe_if_written,
                    unsafe_merge=(
                        case.expected_risk_level == "needs_review"
                        and has_write
                        and actual_decision != "flagged"
                    ),
                    unsafe_merge_eligible=case.expected_risk_level == "needs_review",
                    proposal_rejected=proposal_rejected and proposal_count > 0,
                    proposals_present=proposal_count > 0,
                    logical_replay_matched=logical.matched_original,
                    live_replay_drifted=not live.matched_original,
                    steps=int(trace_row["total_steps"] or 0),
                    total_tokens=int(trace_row["total_input_tokens"] or 0)
                    + int(trace_row["total_output_tokens"] or 0),
                    duration_ms=int(trace_row["total_duration_ms"] or 0),
                )
            finally:
                sandbox_db.close()

    def _seed_case(self, db, vault, case: EvalCase) -> TaskRecord:
        event = case.event
        db.execute(
            """
            INSERT INTO conversation_events (
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
        for seed_note in case.seed_notes:
            if seed_note.note_type == "archive":
                path, content_hash = vault.write_archive_note(
                    seed_note.note_id,
                    {
                        "title": seed_note.title,
                        "event_id": seed_note.source_event_ids[0] if seed_note.source_event_ids else event.event_id,
                        "user_message": event.user_message,
                        "assistant_final_answer": seed_note.content,
                    },
                )
            else:
                path, content_hash = vault.write_new_note(
                    note_id=seed_note.note_id,
                    title=seed_note.title,
                    content=seed_note.content,
                    tags=seed_note.tags,
                    topics=seed_note.topics,
                    source_event_ids=seed_note.source_event_ids,
                    status=seed_note.status,
                )
            db.execute(
                """
                INSERT INTO notes (
                  note_id, note_type, title, vault_path, content_hash,
                  status, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    seed_note.note_id,
                    seed_note.note_type,
                    seed_note.title,
                    str(path.resolve()),
                    content_hash,
                    seed_note.status,
                    json.dumps({"tags": seed_note.tags, "topics": seed_note.topics}, ensure_ascii=False),
                    now_utc_iso(),
                    now_utc_iso(),
                ),
            )
            for source_event_id in seed_note.source_event_ids:
                db.execute(
                    """
                    INSERT OR IGNORE INTO note_sources (note_id, event_id, relation_type)
                    VALUES (?, ?, 'seeded_from')
                    """,
                    (seed_note.note_id, source_event_id),
                )
        task_id = f"eval_task_{case.case_id}"
        db.execute(
            """
            INSERT INTO tasks (
              task_id, event_id, dedupe_key, status, retry_count, max_retries, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 0, 1, ?, ?)
            """,
            (
                task_id,
                event.event_id,
                event.turn_id,
                RunState.PREPARED.value,
                now_utc_iso(),
                now_utc_iso(),
            ),
        )
        db.commit()
        return TaskRecord(
            task_id=task_id,
            event_id=event.event_id,
            status=RunState.PREPARED,
            retry_count=0,
            max_retries=1,
        )

    def _load_actual_target_note(self, db, trace_id: str) -> str | None:
        row = db.fetchone(
            """
            SELECT target_note_id
            FROM action_proposals
            WHERE trace_id = ? AND status = 'committed'
            ORDER BY created_at DESC, proposal_id DESC
            LIMIT 1
            """,
            (trace_id,),
        )
        if row is None:
            return None
        return row.get("target_note_id")

    def _build_report(self, results: list[EvalCaseResult], prompt_version: str) -> EvalReport:
        total_cases = len(results)
        target_results = [result for result in results if result.target_note_correct is not None]
        unsafe_merge_population = [result for result in results if result.unsafe_merge_eligible]
        replay_results = [result for result in results if result.logical_replay_matched is not None]
        proposal_population = [result for result in results if result.proposals_present]
        review_population = [result for result in results if result.actual_decision == "flagged"]
        auto_action_population = [result for result in results if result.actual_decision in WRITE_DECISIONS]
        report = EvalReport(
            run_id=new_id("eval"),
            prompt_version=prompt_version,
            model_name=self.config.agent.model,
            total_cases=total_cases,
            decision_accuracy=sum(result.decision_correct for result in results) / total_cases,
            target_note_accuracy=(
                sum(bool(result.target_note_correct) for result in target_results) / len(target_results)
                if target_results
                else None
            ),
            false_write_rate=sum(result.unsafe_write for result in results) / total_cases,
            unsafe_merge_rate=(
                sum(result.unsafe_merge for result in results) / len(unsafe_merge_population)
                if unsafe_merge_population
                else None
            ),
            proposal_rejection_rate=(
                sum(result.proposal_rejected for result in proposal_population) / len(proposal_population)
                if proposal_population
                else None
            ),
            logical_replay_match_rate=(
                sum(bool(result.logical_replay_matched) for result in replay_results) / len(replay_results)
                if replay_results
                else None
            ),
            live_replay_drift_rate=(
                sum(bool(result.live_replay_drifted) for result in replay_results) / len(replay_results)
                if replay_results
                else None
            ),
            review_precision=(
                sum(result.unsafe_merge_eligible for result in review_population) / len(review_population)
                if review_population
                else None
            ),
            auto_action_acceptance_rate=(
                sum(
                    result.decision_correct
                    and result.target_note_correct is not False
                    and not result.unsafe_write
                    for result in auto_action_population
                )
                / len(auto_action_population)
                if auto_action_population
                else None
            ),
            avg_steps=(sum(result.steps for result in results) / total_cases) if results else None,
            avg_tokens=(sum(result.total_tokens for result in results) / total_cases) if results else None,
            avg_duration_ms=(sum(result.duration_ms for result in results) / total_cases) if results else None,
            results=results,
        )
        return report

    def _save_report(self, report: EvalReport) -> None:
        self.db.execute(
            """
            INSERT INTO eval_runs (
              run_id, prompt_version, model_name, total_cases, decision_accuracy,
              target_note_accuracy, false_write_rate, unsafe_merge_rate,
              proposal_rejection_rate, logical_replay_match_rate, live_replay_drift_rate,
              review_precision, auto_action_acceptance_rate, avg_steps, avg_tokens,
              avg_duration_ms, result_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report.run_id,
                report.prompt_version,
                report.model_name,
                report.total_cases,
                report.decision_accuracy,
                report.target_note_accuracy,
                report.false_write_rate,
                report.unsafe_merge_rate,
                report.proposal_rejection_rate,
                report.logical_replay_match_rate,
                report.live_replay_drift_rate,
                report.review_precision,
                report.auto_action_acceptance_rate,
                report.avg_steps,
                report.avg_tokens,
                report.avg_duration_ms,
                json.dumps(report.to_dict(), ensure_ascii=False),
            ),
        )

    def _count_rows(self, db, sql: str, params: tuple) -> int:
        row = db.fetchone(sql, params)
        return int(row["count"]) if row else 0


def _normalize_decision(value: str) -> str:
    return DECISION_ALIASES.get(value, value)
