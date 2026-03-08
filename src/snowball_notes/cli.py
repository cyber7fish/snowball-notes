from __future__ import annotations

import argparse
import json
import sys

from .agent.adapter import build_model_adapter
from .agent.memory import SQLiteKnowledgeIndex
from .agent.orchestrator import SnowballWorker
from .agent.replay import ReplayRunner
from .agent.runtime import SnowballAgent
from .agent.tools import build_tool_registry
from .calibrate.confidence_feedback import (
    HUMAN_LABELS,
    analyze_confidence_calibration,
    record_confidence_feedback,
    render_calibration_report,
)
from .config import load_config
from .demo import setup_demo_workspace
from .embedding import build_embedding_provider, build_vector_store, run_embedding_check
from .eval import EvalRunner, import_eval_cases, load_eval_cases, load_eval_report, render_eval_report
from .observability.logger import JsonlLogger
from .observability.metrics import render_status
from .review.cli import approve_review, list_pending_reviews, update_review
from .review.server import serve_review_app
from .storage.reconcile import reconcile_vault_and_db
from .storage.sqlite import Database
from .storage.vault import Vault


def build_runtime(config_path: str | None = None, *, build_worker: bool = True):
    config = load_config(config_path)
    db = Database(config.db_path)
    db.migrate()
    db.event_logger = JsonlLogger(config.log_path)
    vault = Vault(config)
    worker = None
    if build_worker:
        embedding_provider = build_embedding_provider(config)
        vector_store = build_vector_store(config, db)
        knowledge_index = SQLiteKnowledgeIndex(
            db,
            config=config,
            embedding_provider=embedding_provider,
            vector_store=vector_store,
        )
        tools = build_tool_registry(db, knowledge_index)
        adapter = build_model_adapter(config)
        agent = SnowballAgent(config, adapter, tools, vault, db)
        worker = SnowballWorker(config, db, agent, vault)
    return config, db, vault, worker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="snowball")
    parser.add_argument("--config", dest="config_path", default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--once", action="store_true")
    worker_parser.add_argument("--forever", action="store_true")

    review_parser = subparsers.add_parser("review")
    review_subparsers = review_parser.add_subparsers(dest="review_command", required=True)
    review_subparsers.add_parser("list")
    serve_parser = review_subparsers.add_parser("serve")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    approve_parser = review_subparsers.add_parser("approve")
    approve_parser.add_argument("review_id")
    approve_parser.add_argument("--action", choices=["create", "append", "archive", "link"], default=None)
    approve_parser.add_argument("--note-id", default=None)
    approve_parser.add_argument("--title", default=None)
    approve_parser.add_argument("--reviewer", default="local")
    reject_parser = review_subparsers.add_parser("reject")
    reject_parser.add_argument("review_id")
    reject_parser.add_argument("--reviewer", default="local")
    mark_conflict_parser = review_subparsers.add_parser("mark-conflict")
    mark_conflict_parser.add_argument("review_id")
    mark_conflict_parser.add_argument("--reviewer", default="local")
    mark_conflict_parser.add_argument("--note-id", default=None)
    mark_conflict_parser.add_argument("--reason", default="marked_conflict")
    discard_parser = review_subparsers.add_parser("discard")
    discard_parser.add_argument("review_id")
    discard_parser.add_argument("--reviewer", default="local")
    discard_parser.add_argument("--reason", default="discarded_by_reviewer")

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--days", type=int, default=7)
    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("trace_id")
    replay_parser.add_argument("--mode", choices=["dump", "logical", "live"], default="dump")
    subparsers.add_parser("reconcile")
    embedding_parser = subparsers.add_parser("embedding")
    embedding_subparsers = embedding_parser.add_subparsers(dest="embedding_command", required=True)
    embedding_check_parser = embedding_subparsers.add_parser("check")
    embedding_check_parser.add_argument("--provider", choices=["local", "dashscope", "voyage"], default=None)
    embedding_check_parser.add_argument("--vector-store", choices=["sqlite_blob", "sqlite_vec"], default=None)
    embedding_check_parser.add_argument("--text", default=None)
    eval_parser = subparsers.add_parser("eval")
    eval_subparsers = eval_parser.add_subparsers(dest="eval_command", required=True)
    eval_load_parser = eval_subparsers.add_parser("load")
    eval_load_parser.add_argument("fixture_path")
    eval_load_parser.add_argument("--replace", action="store_true")
    eval_run_parser = eval_subparsers.add_parser("run")
    eval_run_parser.add_argument("--fixtures", dest="fixture_path", default=None)
    eval_run_parser.add_argument("--replace", action="store_true")
    eval_run_parser.add_argument("--prompt-version", default=None)
    eval_run_parser.add_argument("--baseline-run", default=None)
    eval_report_parser = eval_subparsers.add_parser("report")
    eval_report_parser.add_argument("run_id", nargs="?")
    eval_report_parser.add_argument("--baseline-run", default=None)
    demo_parser = subparsers.add_parser("demo")
    demo_subparsers = demo_parser.add_subparsers(dest="demo_command", required=True)
    demo_setup_parser = demo_subparsers.add_parser("setup")
    demo_setup_parser.add_argument("--dest", default="./demo-workspace")
    calibrate_parser = subparsers.add_parser("calibrate")
    calibrate_subparsers = calibrate_parser.add_subparsers(dest="calibrate_command", required=True)
    feedback_parser = calibrate_subparsers.add_parser("add-feedback")
    feedback_parser.add_argument("turn_id")
    feedback_parser.add_argument("label", choices=sorted(HUMAN_LABELS))
    feedback_parser.add_argument("--annotator", default="local")
    calibrate_subparsers.add_parser("report")

    args = parser.parse_args(argv)
    if args.command == "demo":
        if args.demo_command == "setup":
            payload = setup_demo_workspace(args.dest)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
    config, db, vault, worker = build_runtime(args.config_path, build_worker=args.command == "worker")
    try:
        if args.command == "worker":
            if worker is None:
                print("worker runtime unavailable", file=sys.stderr)
                return 1
            if args.forever:
                worker.run_forever()
                return 0
            worker.run_once()
            return 0
        if args.command == "review":
            if args.review_command == "list":
                print(list_pending_reviews(db))
                return 0
            if args.review_command == "serve":
                serve_review_app(args.config_path, host=args.host, port=args.port)
                return 0
            if args.review_command == "approve":
                try:
                    approved, detail = approve_review(
                        db,
                        vault,
                        config,
                        args.review_id,
                        reviewer=args.reviewer,
                        action=args.action,
                        note_id=args.note_id,
                        title=args.title,
                    )
                except ValueError as exc:
                    print(str(exc), file=sys.stderr)
                    return 1
                if not approved:
                    print(detail, file=sys.stderr)
                    return 1
                print(f"approved {args.review_id}: {detail}")
                return 0
            if args.review_command == "reject":
                return 0 if update_review(db, args.review_id, "rejected", reviewer=args.reviewer) else 1
            if args.review_command == "mark-conflict":
                return (
                    0
                    if update_review(
                        db,
                        args.review_id,
                        "mark_conflict",
                        reviewer=args.reviewer,
                        final_target_note_id=args.note_id,
                        reason=args.reason,
                    )
                    else 1
                )
            if args.review_command == "discard":
                return (
                    0
                    if update_review(
                        db,
                        args.review_id,
                        "discarded",
                        reviewer=args.reviewer,
                        reason=args.reason,
                    )
                    else 1
                )
        if args.command == "status":
            print(render_status(db, window_days=args.days))
            return 0
        if args.command == "replay":
            row = db.fetchone("SELECT * FROM replay_bundles WHERE trace_id = ?", (args.trace_id,))
            if row is None:
                print(f"trace {args.trace_id} not found", file=sys.stderr)
                return 1
            if args.mode == "dump":
                print(json.dumps(row, ensure_ascii=False, indent=2))
                return 0
            runner = ReplayRunner(config, db, vault)
            outcome = (
                runner.logical_replay(args.trace_id)
                if args.mode == "logical"
                else runner.live_replay(args.trace_id)
            )
            print(json.dumps(outcome.to_dict(), ensure_ascii=False, indent=2))
            return 0
        if args.command == "reconcile":
            report = reconcile_vault_and_db(vault, db)
            print(
                json.dumps(
                    {
                        "orphan_files": report.orphan_files,
                        "missing_files": report.missing_files,
                        "promoted_auto_approved": report.promoted_auto_approved,
                        "normalized_note_files": report.normalized_note_files,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.command == "embedding":
            if args.embedding_command == "check":
                try:
                    result = run_embedding_check(
                        config,
                        db,
                        provider_override=args.provider,
                        vector_store_override=args.vector_store,
                        text=args.text,
                    )
                except Exception as exc:
                    print(str(exc), file=sys.stderr)
                    return 1
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return 0 if result.get("ok") else 1
        if args.command == "eval":
            if args.eval_command == "load":
                imported = import_eval_cases(db, args.fixture_path, replace=args.replace)
                print(f"loaded {imported} eval cases")
                return 0
            if args.eval_command == "run":
                if args.fixture_path:
                    import_eval_cases(db, args.fixture_path, replace=args.replace)
                dataset = load_eval_cases(db)
                if not dataset:
                    print("no eval cases loaded", file=sys.stderr)
                    return 1
                report = EvalRunner(config, db).run(dataset, prompt_version=args.prompt_version)
                baseline = load_eval_report(db, args.baseline_run) if args.baseline_run else None
                print(render_eval_report(report.to_dict(), baseline))
                return 0
            if args.eval_command == "report":
                report = load_eval_report(db, args.run_id)
                if report is None:
                    print("eval report not found", file=sys.stderr)
                    return 1
                baseline = load_eval_report(db, args.baseline_run) if args.baseline_run else None
                print(render_eval_report(report, baseline))
                return 0
        if args.command == "calibrate":
            if args.calibrate_command == "add-feedback":
                created = record_confidence_feedback(db, args.turn_id, args.label, annotator=args.annotator)
                if not created:
                    print(f"turn {args.turn_id} not found", file=sys.stderr)
                    return 1
                print(f"recorded confidence feedback for {args.turn_id}")
                return 0
            if args.calibrate_command == "report":
                print(render_calibration_report(analyze_confidence_calibration(db)))
                return 0
        return 1
    finally:
        db.commit()
        db.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
