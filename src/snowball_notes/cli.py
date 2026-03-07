from __future__ import annotations

import argparse
import json
import sys

from .agent.adapter import HeuristicModelAdapter
from .agent.memory import SQLiteKnowledgeIndex
from .agent.orchestrator import SnowballWorker
from .agent.runtime import SnowballAgent
from .agent.tools import build_tool_registry
from .calibrate.confidence_feedback import (
    HUMAN_LABELS,
    analyze_confidence_calibration,
    record_confidence_feedback,
    render_calibration_report,
)
from .config import load_config
from .observability.metrics import render_status
from .review.cli import list_pending_reviews, update_review
from .storage.reconcile import reconcile_vault_and_db
from .storage.sqlite import Database
from .storage.vault import Vault


def build_runtime(config_path: str | None = None):
    config = load_config(config_path)
    db = Database(config.db_path)
    db.migrate()
    vault = Vault(config)
    knowledge_index = SQLiteKnowledgeIndex(db)
    tools = build_tool_registry(db, knowledge_index)
    adapter = HeuristicModelAdapter(config)
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
    approve_parser = review_subparsers.add_parser("approve")
    approve_parser.add_argument("review_id")
    reject_parser = review_subparsers.add_parser("reject")
    reject_parser.add_argument("review_id")

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--days", type=int, default=7)
    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("trace_id")
    subparsers.add_parser("reconcile")
    calibrate_parser = subparsers.add_parser("calibrate")
    calibrate_subparsers = calibrate_parser.add_subparsers(dest="calibrate_command", required=True)
    feedback_parser = calibrate_subparsers.add_parser("add-feedback")
    feedback_parser.add_argument("turn_id")
    feedback_parser.add_argument("label", choices=sorted(HUMAN_LABELS))
    feedback_parser.add_argument("--annotator", default="local")
    calibrate_subparsers.add_parser("report")

    args = parser.parse_args(argv)
    config, db, vault, worker = build_runtime(args.config_path)
    try:
        if args.command == "worker":
            if args.forever:
                worker.run_forever()
                return 0
            worker.run_once()
            return 0
        if args.command == "review":
            if args.review_command == "list":
                print(list_pending_reviews(db))
                return 0
            if args.review_command == "approve":
                return 0 if update_review(db, args.review_id, "approved") else 1
            if args.review_command == "reject":
                return 0 if update_review(db, args.review_id, "rejected") else 1
        if args.command == "status":
            print(render_status(db, window_days=args.days))
            return 0
        if args.command == "replay":
            row = db.fetchone("SELECT * FROM replay_bundles WHERE trace_id = ?", (args.trace_id,))
            if row is None:
                print(f"trace {args.trace_id} not found", file=sys.stderr)
                return 1
            print(json.dumps(row, ensure_ascii=False, indent=2))
            return 0
        if args.command == "reconcile":
            report = reconcile_vault_and_db(vault.root, db)
            print(json.dumps({"orphan_files": report.orphan_files, "missing_files": report.missing_files}, ensure_ascii=False, indent=2))
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
