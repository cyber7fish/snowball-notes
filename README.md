# Snowball Notes

Snowball Notes turns completed Codex turns into reviewable Obsidian notes with a guarded agent runtime.

## What is implemented

- Transcript intake from `~/.codex/sessions/**/*.jsonl`
- `source_confidence` scoring and pre-queue filtering
- SQLite-backed task queue with claim semantics and state transitions
- Heuristic agent runtime with tool calls, guardrails, proposals, trace, and replay bundle
- Vault writer for archive, create, and append flows
- Review CLI for flagged runs
- Status CLI and reconciliation helper
- `unittest` coverage for confidence, parser, state machine, and end-to-end runtime

## Project layout

```text
snowball-notes/
  bin/
  src/snowball_notes/
  tests/
  config.yaml
```

## Quick start

```bash
cd snowball-notes
PYTHONPATH=src python3 -m unittest discover -s tests
PYTHONPATH=src python3 -m snowball_notes.cli worker --once
PYTHONPATH=src python3 -m snowball_notes.cli status
```

The default configuration writes runtime data under `./data`, logs under `./logs`, and notes under `./vault`. Update `config.yaml` to point at your real Obsidian vault when you are ready.

## Commands

- `worker --once`: scan transcripts, enqueue events, claim one task, and run the agent once
- `worker --forever`: continuous polling worker
- `review list`: show pending review actions
- `review approve <review_id>`: mark a flagged case approved
- `review reject <review_id>`: mark a flagged case rejected
- `status`: print queue and runtime metrics
- `replay <trace_id>`: dump a saved replay bundle

## Design notes

This implementation follows the runtime shape from `snowball-notes-final.md`, but fixes the state handoff around commit validation. Validation happens before the `proposed_actions -> committing` transition so rejected proposal batches can move cleanly to `flagged`.
