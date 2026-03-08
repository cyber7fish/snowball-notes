# LLM Observe Review Fixture

This pack is for one specific branch: `flag_for_review`.

It seeds one existing note, then submits a highly similar turn under a lower-confidence parser configuration.
The result should be a review record instead of an automatic append.

Commands:

```bash
PYTHONPATH=src python3 demo/fixtures/llm-observe-review/seed_workspace.py
PYTHONPATH=src python3 -m snowball_notes.cli --config demo/fixtures/llm-observe-review/config.yaml worker --once
```

Inspect:

- `demo/fixtures/llm-observe-review/workspace-review/data/snowball.db`
- `review_actions` in that database

Expected behavior:

- `01-kv-cache-review-flag.jsonl`: `flagged` with `reason=high_similarity_low_confidence`
