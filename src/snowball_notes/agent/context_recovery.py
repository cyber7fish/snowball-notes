"""Multi-level context recovery — an escalating gradient for context pressure.

The agent re-sends its whole message history to the model on every ReAct step.
The tool-result budget (:mod:`context_budget`) caps the *largest* tool outputs
while keeping the prefix byte-stable, so it is cheap and cache-preserving. But a
long, tool-heavy turn can still grow the conversation past the context window
even after the budget runs. This module adds the next levels of Claude Code's
recovery gradient on top of it:

1. **microcompact** — clear overflowing tool results (the frozen tool-result
   budget). Cache-preserving; measured by ``context_chars_cleared``.
2. **history_compaction** — when the estimate is still over the soft limit, fold
   all but the most recent few ReAct exchanges into one compact digest of the
   decisions, tool outcomes, and proposals so far, keeping recent turns verbatim.
3. **full_summarize** — when even that is over the hard limit, collapse the whole
   turn down to the original task plus a single summary and continue from there.

Levels 2 and 3 deliberately rewrite the prefix, so they invalidate the prompt
cache — that is the trade-off of reclaiming context room, and the gradient only
pays it when the cheaper, cache-preserving level is not enough. The full history
is retained in ``messages`` and the ``ReplayBundle``; only the model-facing copy
is compacted. The digests are built mechanically from the structured message
history (each assistant message already carries ``decision_summary`` and
``tool_calls``), so recovery stays deterministic and offline — no extra model
call, and replay sees the original, uncompacted tool outputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .context_budget import apply_tool_result_budget

# Rough byte→token ratio. Good enough to drive escalation thresholds without a
# tokenizer dependency; the estimate only needs to be monotone, not exact.
ESTIMATED_CHARS_PER_TOKEN = 4

HISTORY_COMPACTION = "history_compaction"
FULL_SUMMARIZE = "full_summarize"


@dataclass
class RecoveryEvent:
    step_index: int
    level: str
    tokens_before: int
    tokens_after: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "level": self.level,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
        }


def _content_chars(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    return len(json.dumps(content, ensure_ascii=False))


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    chars = 0
    for message in messages:
        chars += len(str(message.get("role", "")))
        chars += len(str(message.get("name", "")))
        chars += _content_chars(message.get("content"))
    return (chars + ESTIMATED_CHARS_PER_TOKEN - 1) // ESTIMATED_CHARS_PER_TOKEN


def recover_context(
    messages: list[dict[str, Any]],
    state,
    *,
    config,
    step_index: int,
) -> list[dict[str, Any]]:
    """Apply the recovery gradient to the model-facing message list.

    Always runs the cache-preserving tool-result budget first (when enabled),
    then escalates to history compaction / full summarization only when the token
    estimate exceeds the configured soft / hard limits. Compaction events are
    recorded on ``state.recovery_events`` for telemetry.
    """
    agent = config.agent
    working = list(messages)
    if agent.enable_context_budget:
        working = apply_tool_result_budget(
            working,
            state.replacement_state,
            max_chars=agent.max_tool_result_chars,
            keep_recent=agent.keep_recent_tool_results,
        )
    if not agent.enable_context_recovery:
        return working

    tokens = estimate_tokens(working)
    if tokens <= agent.compact_token_soft_limit:
        return working

    # `base` is the post-budget history with every step intact. History
    # compaction folds the older steps but keeps recent ones verbatim; full
    # summarization (the harder reset) re-derives from `base` so it captures
    # every step rather than the partial digest L2 produced.
    base = working
    compacted = _compact_history(base, state, keep_recent_turns=agent.keep_recent_turns)
    after = estimate_tokens(compacted)
    if after < tokens:
        _record(state, step_index, HISTORY_COMPACTION, tokens, after)
        working, tokens = compacted, after

    if tokens <= agent.compact_token_hard_limit:
        return working

    summarized = _full_summarize(base, state)
    final_tokens = estimate_tokens(summarized)
    if final_tokens < tokens:
        _record(state, step_index, FULL_SUMMARIZE, tokens, final_tokens)
        working = summarized
    return working


def _record(state, step_index: int, level: str, before: int, after: int) -> None:
    state.recovery_events.append(
        RecoveryEvent(
            step_index=step_index,
            level=level,
            tokens_before=before,
            tokens_after=after,
        ).to_dict()
    )


def _split_turns(messages: list[dict[str, Any]]) -> tuple[list[dict], list[list[dict]]]:
    """Split into the leading non-assistant head and per-step [assistant, tool*] groups."""
    head: list[dict] = []
    index = 0
    while index < len(messages) and messages[index].get("role") != "assistant":
        head.append(messages[index])
        index += 1
    groups: list[list[dict]] = []
    current: list[dict] = []
    for message in messages[index:]:
        if message.get("role") == "assistant" and current:
            groups.append(current)
            current = []
        current.append(message)
    if current:
        groups.append(current)
    return head, groups


def _compact_history(messages: list[dict[str, Any]], state, *, keep_recent_turns: int) -> list[dict[str, Any]]:
    head, groups = _split_turns(messages)
    if len(groups) <= max(keep_recent_turns, 0):
        return messages
    if keep_recent_turns > 0:
        old, recent = groups[:-keep_recent_turns], groups[-keep_recent_turns:]
    else:
        old, recent = groups, []
    digest = _digest_message(old, state, full=False)
    tail = [message for group in recent for message in group]
    return head + [digest] + tail


def _full_summarize(messages: list[dict[str, Any]], state) -> list[dict[str, Any]]:
    head, groups = _split_turns(messages)
    if not groups:
        return messages
    digest = _digest_message(groups, state, full=True)
    # Keep only the original task message (head[0]); drop any earlier summary.
    task = head[:1]
    return task + [digest]


def _digest_message(groups: list[list[dict]], state, *, full: bool) -> dict[str, Any]:
    steps = []
    for group in groups:
        assistant = group[0]
        content = assistant.get("content")
        decision = content.get("decision_summary", "") if isinstance(content, dict) else ""
        tools = [
            {"tool": message.get("name"), "result": _brief_result(message.get("content"))}
            for message in group[1:]
        ]
        steps.append({"decision": decision, "tools": tools})
    proposals = [
        {"action_type": proposal.action_type, "target_note_id": proposal.target_note_id}
        for proposal in state.proposals
    ]
    label = "full_conversation_summary" if full else "compacted_history"
    return {
        "role": "user",
        "content": {
            label: {
                "summarized_steps": len(groups),
                "steps": steps,
                "proposals_so_far": proposals,
                "note": (
                    "Earlier reasoning was compacted to fit the context window. "
                    "These are the decisions, tool outcomes, and proposals so far; "
                    "continue from this state."
                ),
            }
        },
    }


def _brief_result(content: Any) -> Any:
    if isinstance(content, str):
        return content[:120]
    if not isinstance(content, dict):
        return content
    brief: dict[str, Any] = {}
    for key in ("note_id", "decision", "error_code", "error_message", "created"):
        if key in content:
            brief[key] = content[key]
    matches = content.get("matches")
    if isinstance(matches, list):
        brief["match_count"] = len(matches)
        brief["top_note_ids"] = [
            match.get("note_id") for match in matches[:3] if isinstance(match, dict)
        ]
    if brief:
        return brief
    return {"keys": sorted(content.keys())[:5]}
