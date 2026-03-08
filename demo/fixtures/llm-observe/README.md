# LLM Observe Fixtures

This fixture pack is for observing note capture behavior in an isolated workspace.

It uses:

- `heuristic` agent routing
- `local` embeddings
- a separate demo database and vault under `workspace/`

Run it from the project root:

```bash
PYTHONPATH=src python3 -m snowball_notes.cli --config demo/fixtures/llm-observe/config.yaml worker --once
```

Run that command six times to process all six transcript files.

Then inspect:

- `demo/fixtures/llm-observe/workspace/vault/Knowledge/Atomic`
- `demo/fixtures/llm-observe/workspace/vault/Archive/Conversations`

Current observed results in this repo:

- `01-rag-chunking-create.jsonl`: create a new atomic note
- `02-rag-chunking-append.jsonl`: append to the first RAG chunking note
- `03-kv-cache-create.jsonl`: archive the turn under the current heuristic
- `04-post-training-create.jsonl`: create a new atomic note
- `05-llm-demo-archive.jsonl`: archive the turn instead of creating knowledge
- `06-kv-cache-create-with-implementation-signal.jsonl`: create a new atomic note

Why `03` is useful:

- it is durable LLM knowledge from a human point of view
- the current heuristic still archives it as `not_reusable_enough`
- that makes it a good probe for whether your routing rules are too narrow on LLM topics

Useful follow-up checks:

- `PYTHONPATH=src python3 -m snowball_notes.cli --config demo/fixtures/llm-observe/config.yaml status`
- inspect `demo/fixtures/llm-observe/workspace/data/snowball.db`
