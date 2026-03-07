from __future__ import annotations

import argparse
import json
import sys

from .agent.adapter import HeuristicModelAdapter
from .agent.memory import SQLiteKnowledgeIndex
from .agent.orchestrator import SnowballWorker
from .agent.runtime import SnowballAgent
from .agent.tools import build_tool_registry
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
    worker = SnowballWorker(config, db, agent)
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

    subparsers.add_parser("status")
    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("trace_id")
    subparsers.add_parser("reconcile")

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
            print(render_status(db))
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
        return 1
    finally:
        db.commit()
        db.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

