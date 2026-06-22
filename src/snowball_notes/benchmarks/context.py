"""Offline benchmark for the context-engineering machinery.

Heuristic / stub adapters return hard-coded token counts, so just running the
agent end-to-end can't show whether the tool-result budget or prefix caching
actually save anything. This module simulates instead: it builds the exact
message list ``SnowballAgent`` would send to a real model on step N, then
measures bytes — once with the optimizations on, once with them off — and
reconstructs the cache-hit accounting the same way the Anthropic API does
(longest byte-identical message prefix between consecutive requests).

Four configurations are compared:

- ``baseline``        : no budget, no prompt cache.
- ``budget_only``     : the frozen tool-result budget, no cache.
- ``cache_only``      : prefix caching, no budget.
- ``budget_and_cache``: both — what the project ships.

The result is a deterministic, reproducible table of fresh vs cached input
chars (and the implied input-token cost at ``chars/4``) showing what each piece
of the context-engineering stack is actually worth on its own and combined.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..agent.context_budget import ContentReplacementState, apply_tool_result_budget

# Same byte→token estimate the recovery module uses; good enough to drive a
# comparative benchmark without pulling in a tokenizer dependency.
ESTIMATED_CHARS_PER_TOKEN = 4


def _content_chars(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    return len(json.dumps(content, ensure_ascii=False))


def _msg_chars(message: dict[str, Any]) -> int:
    return (
        len(str(message.get("role", "")))
        + len(str(message.get("name", "")))
        + _content_chars(message.get("content"))
    )


def _msg_key(message: dict[str, Any]) -> str:
    return json.dumps(message, ensure_ascii=False, sort_keys=True)


def _common_prefix_chars(prev: list[dict[str, Any]], curr: list[dict[str, Any]]) -> int:
    chars = 0
    for prev_msg, curr_msg in zip(prev, curr):
        if _msg_key(prev_msg) != _msg_key(curr_msg):
            break
        chars += _msg_chars(curr_msg)
    return chars


@dataclass
class BenchmarkResult:
    config: str
    fresh_chars: int
    cache_read_chars: int
    max_step_chars: int

    @property
    def total_chars(self) -> int:
        return self.fresh_chars + self.cache_read_chars

    @property
    def cache_read_rate(self) -> float:
        return (self.cache_read_chars / self.total_chars) if self.total_chars else 0.0

    @property
    def fresh_tokens(self) -> int:
        return self.fresh_chars // ESTIMATED_CHARS_PER_TOKEN

    @property
    def cache_read_tokens(self) -> int:
        return self.cache_read_chars // ESTIMATED_CHARS_PER_TOKEN

    @property
    def max_step_tokens(self) -> int:
        return self.max_step_chars // ESTIMATED_CHARS_PER_TOKEN

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "fresh_chars": self.fresh_chars,
            "cache_read_chars": self.cache_read_chars,
            "total_chars": self.total_chars,
            "cache_read_rate": self.cache_read_rate,
            "fresh_tokens": self.fresh_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "max_step_chars": self.max_step_chars,
            "max_step_tokens": self.max_step_tokens,
        }


def synthetic_messages(step: int, *, tool_result_size: int) -> list[dict[str, Any]]:
    """Reconstruct the message list after `step` ReAct steps in the synthetic loop.

    Each step appends an assistant message and a single ``search_similar_notes``
    tool result whose match excerpts add up to roughly ``tool_result_size * 3``
    characters — large enough to make the tool-result budget kick in over a
    handful of steps, mirroring a real heavy retrieval turn.
    """
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": {
                "turn_id": "bench",
                "user_message": "Synthesize notes from this turn.",
                "previous_turns": 0,
            },
        }
    ]
    for index in range(1, step + 1):
        messages.append(
            {
                "role": "assistant",
                "content": {
                    "decision_summary": f"step {index} reasoning",
                    "stop_reason": "tool_use",
                    "tool_calls": [
                        {
                            "call_id": f"c{index}",
                            "name": "search_similar_notes",
                            "input": {"query": f"step {index}"},
                        }
                    ],
                },
            }
        )
        messages.append(
            {
                "role": "tool",
                "call_id": f"c{index}",
                "name": "search_similar_notes",
                "content": {
                    "matches": [
                        {
                            "note_id": f"n{index}_{match}",
                            "title": f"note {index}-{match}",
                            "excerpt": "x" * tool_result_size,
                        }
                        for match in range(3)
                    ],
                },
            }
        )
    return messages


def run_benchmark(
    *,
    steps: int = 6,
    tool_result_size: int = 1500,
    max_tool_result_chars: int = 16000,
    keep_recent_tool_results: int = 2,
) -> list[BenchmarkResult]:
    configs = (
        ("baseline", False, False),
        ("budget_only", True, False),
        ("cache_only", False, True),
        ("budget_and_cache", True, True),
    )
    results: list[BenchmarkResult] = []
    for name, use_budget, use_cache in configs:
        budget_state = ContentReplacementState()
        previous_payload: list[dict[str, Any]] = []
        fresh = 0
        cache_read = 0
        max_step = 0
        for step in range(1, steps + 1):
            raw = synthetic_messages(step, tool_result_size=tool_result_size)
            if use_budget:
                payload = apply_tool_result_budget(
                    raw,
                    budget_state,
                    max_chars=max_tool_result_chars,
                    keep_recent=keep_recent_tool_results,
                )
            else:
                payload = raw
            payload_chars = sum(_msg_chars(message) for message in payload)
            max_step = max(max_step, payload_chars)
            if use_cache:
                hit = _common_prefix_chars(previous_payload, payload)
                fresh += payload_chars - hit
                cache_read += hit
            else:
                fresh += payload_chars
            previous_payload = payload
        results.append(
            BenchmarkResult(
                config=name,
                fresh_chars=fresh,
                cache_read_chars=cache_read,
                max_step_chars=max_step,
            )
        )
    return results


def format_table(
    results: list[BenchmarkResult],
    *,
    steps: int,
    tool_result_size: int,
) -> str:
    if not results:
        return "(no results)"
    baseline_fresh = results[0].fresh_chars
    lines = [
        f"Context-engineering benchmark — {steps} steps, ~{tool_result_size}-char tool excerpts.",
        "",
        "| config | fresh tokens | cache reads | cache rate | savings vs baseline | peak step tokens |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        savings = (
            f"{(1 - result.fresh_chars / baseline_fresh) * 100:.1f}%"
            if baseline_fresh
            else "n/a"
        )
        rate = f"{result.cache_read_rate * 100:.1f}%"
        lines.append(
            f"| {result.config} | {result.fresh_tokens:,} | {result.cache_read_tokens:,} "
            f"| {rate} | {savings} | {result.max_step_tokens:,} |"
        )
    return "\n".join(lines)
