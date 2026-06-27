"""Tool-result context budget with frozen replacement decisions.

Modeled on Claude Code's ``applyToolResultBudget`` + ``ContentReplacementState``.
The agent re-sends the full message history to the model on every step, so the
large read-only tool outputs (note searches, note bodies) accumulate and inflate
every subsequent request. This module caps the aggregate size of *compactable*
tool results and replaces the overflow with a compact placeholder.

The decisive property is that replacement decisions are **frozen**: once a tool
result is cleared it stays cleared, and re-applying the budget to the same prefix
yields byte-identical output. That stability is what keeps the Anthropic prompt
cache (and any prefix cache) valid across steps — any change to an earlier byte
would invalidate every cached token after it. See ``AnthropicMessagesAdapter``,
which caches the system prefix, and the README "Context Budget" section.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# Read-only tools whose outputs are large but not decision-state. assess/extract
# and the action/flag tools are small and load-bearing, so they are never cleared.
COMPACTABLE_TOOLS = frozenset({"search_similar_notes", "read_note"})


@dataclass
class ContentReplacementState:
    """Frozen record of which tool results have been cleared this run."""

    cleared_call_ids: set[str] = field(default_factory=set)
    chars_cleared: int = 0

    def was_applied(self) -> bool:
        return bool(self.cleared_call_ids)


def _content_chars(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    return len(json.dumps(content, ensure_ascii=False))


def _placeholder(tool_name: str) -> str:
    # Deterministic and content-free so the cleared block is byte-stable across
    # steps (a size or timestamp here would re-break the cache).
    return f"[{tool_name} result cleared to fit context budget]"


def apply_tool_result_budget(
    messages: list[dict[str, Any]],
    state: ContentReplacementState,
    *,
    max_chars: int,
    keep_recent: int,
    compactable_tools: frozenset[str] = COMPACTABLE_TOOLS,
) -> list[dict[str, Any]]:
    """Return a copy of ``messages`` with overflowing tool results cleared.

    - Only ``compactable_tools`` results are eligible.
    - The most recent ``keep_recent`` eligible results are never cleared (the
      agent is most likely still reasoning over them).
    - Already-cleared results (tracked in ``state``) stay cleared.
    - When over budget, the largest eligible results are cleared first until the
      aggregate fits, mutating ``state`` so the decision is frozen for next time.
    """
    eligible = [
        (index, message)
        for index, message in enumerate(messages)
        if message.get("role") == "tool"
        and message.get("name") in compactable_tools
        and isinstance(message.get("call_id"), str)
    ]
    if not eligible:
        return list(messages)

    protected_indices = {index for index, _ in eligible[len(eligible) - keep_recent :]} if keep_recent > 0 else set()

    def effective_chars(message: dict[str, Any]) -> int:
        call_id = message["call_id"]
        if call_id in state.cleared_call_ids:
            return len(_placeholder(message.get("name", "")))
        return _content_chars(message.get("content"))

    total = sum(effective_chars(message) for _, message in eligible)

    if total > max_chars:
        # Clear largest-first among uncleared, unprotected results. Ties broken by
        # index (oldest first) for determinism.
        candidates = sorted(
            (
                (index, message)
                for index, message in eligible
                if index not in protected_indices
                and message["call_id"] not in state.cleared_call_ids
                # Only clear results bigger than the placeholder — clearing a tiny
                # result would add bytes, not save them.
                and _content_chars(message.get("content")) > len(_placeholder(message.get("name", "")))
            ),
            key=lambda pair: (-_content_chars(pair[1].get("content")), pair[0]),
        )
        for index, message in candidates:
            if total <= max_chars:
                break
            saved = _content_chars(message.get("content")) - len(_placeholder(message.get("name", "")))
            state.cleared_call_ids.add(message["call_id"])
            state.chars_cleared += saved
            total -= saved

    if not state.cleared_call_ids:
        return list(messages)

    rendered: list[dict[str, Any]] = []
    for message in messages:
        if (
            message.get("role") == "tool"
            and message.get("call_id") in state.cleared_call_ids
            and message.get("content") != _placeholder(message.get("name", ""))
        ):
            replaced = dict(message)
            replaced["content"] = _placeholder(message.get("name", ""))
            rendered.append(replaced)
        else:
            rendered.append(message)
    return rendered
