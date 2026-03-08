# Eval Fixtures

`eval/fixtures/sample_cases.json` is a small but broader seed dataset for local regression checks.

Current coverage:

- `create_note`: 4 cases
- `append_note`: 3 cases
- `link_notes`: 1 case
- `flagged`: 1 case
- `archive_turn`: 2 cases
- `skip`: 1 case

The sample set is intentionally mixed:

- some cases are straightforward smoke tests for stable local development
- some cases exercise guardrails and review routing
- some cases cover newer flows like note linking and reconcile scheduling

Recommended usage:

```bash
PYTHONPATH=src python3 -m snowball_notes.cli eval load eval/fixtures/sample_cases.json --replace
PYTHONPATH=src python3 -m snowball_notes.cli eval run
```

The expected labels are the ground-truth targets for evaluation. A given model or prompt version is not expected to score 100% on every case.
