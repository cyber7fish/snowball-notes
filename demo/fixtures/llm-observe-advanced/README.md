# LLM Observe Advanced Fixtures

This pack is for observing more interesting branching behavior:

- `create_note`
- `append_note`
- `skip` from duplicate session memory
- `link_notes`

It uses a separate workspace plus a small seeded note set.

For `flag_for_review`, use the separate `demo/fixtures/llm-observe-review` pack.

Seed the workspace:

```bash
PYTHONPATH=src python3 demo/fixtures/llm-observe-advanced/seed_workspace.py
```

Then run the worker five times:

```bash
PYTHONPATH=src python3 -m snowball_notes.cli --config demo/fixtures/llm-observe-advanced/config.yaml worker --once
```

Inspect:

- `demo/fixtures/llm-observe-advanced/workspace-main/vault/Knowledge/Atomic`
- `demo/fixtures/llm-observe-advanced/workspace-main/vault/Archive/Conversations`
- `demo/fixtures/llm-observe-advanced/workspace-main/data/snowball.db`

Expected behavior:

- `01-moe-create.jsonl`: create a new atomic note
- `02-moe-duplicate-skip.jsonl`: skip because the same conversation already created that note
- `03-link-kv-cache-and-rag.jsonl`: link two seeded notes
- `04-kv-cache-append.jsonl`: append to the seeded KV cache note
- `05-llm-small-talk-skip.jsonl`: skip because the user message is just a thank-you
