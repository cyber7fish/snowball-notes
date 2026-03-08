# Snowball Notes

Snowball Notes turns completed Codex turns into reviewable Obsidian notes with a guarded agent runtime.

## What is implemented

- Transcript intake via `transcript_poll`, `transcript_watch`, or single-file `cli_wrap`
- `source_confidence` scoring and pre-queue filtering
- SQLite-backed task queue with claim semantics and state transitions
- Heuristic agent runtime with tool calls, guardrails, proposals, trace, and replay bundle
- Hybrid retrieval with local embeddings stored in SQLite
- Vault writer for archive, create, append, and note-link flows
- Review CLI for flagged runs
- Optional FastAPI review server for pending approvals, trace detail, and confidence feedback
- Status CLI with agent/parser/reconcile health metrics
- Structured JSONL lifecycle logs mirrored alongside SQLite audit logs
- Sandbox eval runner with fixture import, score aggregation, review precision, and replay consistency metrics
- Startup plus scheduled reconciliation audit and confidence calibration feedback loop
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
PYTHONPATH=src python3 -m snowball_notes.cli status --days 7
PYTHONPATH=src python3 -m snowball_notes.cli embedding check
PYTHONPATH=src python3 -m snowball_notes.cli eval load eval/fixtures/sample_cases.json --replace
PYTHONPATH=src python3 -m snowball_notes.cli eval run
PYTHONPATH=src python3 -m snowball_notes.cli calibrate report
```

The default configuration writes runtime data under `./data`, logs under `./logs`, and notes under `./vault`. Update `config.yaml` to point at your real Obsidian vault when you are ready.

Intake modes:

```yaml
intake:
  mode: "transcript_poll"   # recursive directory scan with SQLite cursors
  transcript_dir: "~/.codex/sessions"
```

```yaml
intake:
  mode: "transcript_watch"  # incremental in-process watch over the transcript tree
  transcript_dir: "~/.codex/sessions"
```

```yaml
intake:
  mode: "cli_wrap"          # parse one rolling transcript file
  cli_wrap_file: "./wrapped/current.jsonl"
```

Reconcile scheduling is configured in UTC. The default runs once on startup and once per day after `03:00 UTC`:

```yaml
reconcile:
  enabled: true
  run_on_startup: true
  schedule_cron: "0 3 * * *"
```

If no config file is present, the runtime falls back to the local `heuristic-v1` adapter. The checked-in `config.yaml` is preconfigured for DeepSeek. To run a real tool-calling model with OpenAI Responses instead, set:

```yaml
agent:
  provider: "openai_responses"
  model: "gpt-5.2-codex"
```

and export `OPENAI_API_KEY` before starting the worker.

To run against DeepSeek's tool-calling chat API, set:

```yaml
agent:
  provider: "deepseek_v3"
  model: "deepseek-chat"
  api_key_env: "DEEPSEEK_API_KEY"
  api_base_url: "https://api.deepseek.com/chat/completions"
```

and export `DEEPSEEK_API_KEY` before starting the worker.

Retrieval defaults to an offline local embedding provider backed by SQLite. To switch to Alibaba Cloud DashScope `text-embedding-v4`, set:

```yaml
embedding:
  provider: "dashscope"
  dashscope_model: "text-embedding-v4"
  dashscope_dimensions: 1024
  dashscope_api_key_env: "DASHSCOPE_API_KEY"
```

and export `DASHSCOPE_API_KEY`.

Voyage is still supported if you prefer it:

```yaml
embedding:
  provider: "voyage"
  voyage_model: "voyage-3-lite"
```

and export `VOYAGE_API_KEY`.

To verify the configured provider and vector store end to end, run:

```bash
PYTHONPATH=src python3 -m snowball_notes.cli embedding check
PYTHONPATH=src python3 -m snowball_notes.cli embedding check --provider dashscope
PYTHONPATH=src python3 -m snowball_notes.cli embedding check --provider voyage
```

The check performs a real embedding call for the selected provider and then does a vector-store round trip using a temporary probe vector.

To run the review server, install the optional review dependencies first:

```bash
pip install -e ".[review]"
```

## Commands

- `worker --once`: scan transcripts, enqueue events, claim one task, and run the agent once
- `worker --forever`: continuous polling worker
- `review list`: show pending review actions
- `review serve [--host HOST] [--port PORT]`: start the FastAPI review server
- `review approve <review_id> [--action create|append|archive|link] [--note-id NOTE_ID] [--title TITLE]`: generate a proposal from a pending review and commit it
- `review mark-conflict <review_id> [--note-id NOTE_ID]`: resolve a review as a conflict without writing
- `review discard <review_id>`: resolve a review as intentionally discarded
- `review reject <review_id>`: mark a flagged case rejected
- `status [--days N]`: print queue, runtime, parser, and reconcile health metrics
- `embedding check [--provider local|dashscope|voyage] [--vector-store sqlite_blob|sqlite_vec] [--text TEXT]`: verify provider and vector-store round-trip behavior
- `replay <trace_id> [--mode dump|logical|live]`: dump or rerun a saved replay bundle
- `eval load <fixture_path> [--replace]`: import eval fixtures into `eval_cases`
- `eval run [--fixtures PATH] [--prompt-version VERSION] [--baseline-run RUN_ID]`: run sandbox eval and print a comparable report
- `eval report [run_id] [--baseline-run RUN_ID]`: render a stored eval report
- `calibrate add-feedback <turn_id> <trustworthy|partial|bad_parse>`: record parser confidence feedback
- `calibrate report`: summarize confidence calibration buckets and recommendations

## Design notes

This implementation follows the runtime shape from `snowball-notes-final.md`, but fixes the state handoff around commit validation. Validation happens before the `proposed_actions -> committing` transition so rejected proposal batches can move cleanly to `flagged`.
