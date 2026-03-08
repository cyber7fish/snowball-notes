from __future__ import annotations

from typing import Any


CURRENT_STABLE_PARSER = "v1"
MAX_CONFIDENCE_SCORE = 1.0
MIN_CONFIDENCE_SCORE = 0.0


def compute_source_confidence_breakdown(
    turn_events: list[dict],
    final_answer: str | None,
    user_message: str | None,
    source_completeness: str,
    parser_version: str,
) -> dict[str, Any]:
    penalties: list[dict[str, Any]] = []
    score = MAX_CONFIDENCE_SCORE
    final_answer_length = len((final_answer or "").strip())
    task_complete_count = sum(
        1
        for event in turn_events
        if event.get("payload", {}).get("type") == "task_complete"
    )

    if not final_answer:
        score = _apply_penalty(
            penalties,
            score,
            "missing_final_answer",
            -0.50,
            "final answer missing from parsed turn",
        )
    if not user_message:
        score = _apply_penalty(
            penalties,
            score,
            "missing_user_message",
            -0.20,
            "user message missing from parsed turn",
        )
    if source_completeness == "partial":
        score = _apply_penalty(
            penalties,
            score,
            "partial_source",
            -0.20,
            "turn is only partially reconstructed",
        )
    if parser_version != CURRENT_STABLE_PARSER:
        score = _apply_penalty(
            penalties,
            score,
            "parser_version_drift",
            -0.10,
            f"parser_version={parser_version} differs from stable={CURRENT_STABLE_PARSER}",
        )
    if final_answer and final_answer_length < 50:
        score = _apply_penalty(
            penalties,
            score,
            "short_final_answer",
            -0.15,
            f"final answer too short ({final_answer_length} chars)",
        )
    if task_complete_count > 1:
        score = _apply_penalty(
            penalties,
            score,
            "duplicate_task_complete",
            -0.20,
            f"turn emitted {task_complete_count} task_complete events",
        )

    return {
        "score": round(max(MIN_CONFIDENCE_SCORE, min(MAX_CONFIDENCE_SCORE, score)), 2),
        "base_score": MAX_CONFIDENCE_SCORE,
        "penalties": penalties,
        "signals": {
            "has_final_answer": bool(final_answer),
            "has_user_message": bool(user_message),
            "source_completeness": source_completeness,
            "parser_version": parser_version,
            "stable_parser_version": CURRENT_STABLE_PARSER,
            "final_answer_length": final_answer_length,
            "task_complete_count": task_complete_count,
        },
    }


def compute_source_confidence(
    turn_events: list[dict],
    final_answer: str | None,
    user_message: str | None,
    source_completeness: str,
    parser_version: str,
) -> float:
    return float(
        compute_source_confidence_breakdown(
            turn_events,
            final_answer,
            user_message,
            source_completeness,
            parser_version,
        )["score"]
    )


def _apply_penalty(
    penalties: list[dict[str, Any]],
    score: float,
    code: str,
    delta: float,
    detail: str,
) -> float:
    next_score = score + delta
    penalties.append(
        {
            "code": code,
            "delta": delta,
            "detail": detail,
            "score_after": round(max(MIN_CONFIDENCE_SCORE, min(MAX_CONFIDENCE_SCORE, next_score)), 2),
        }
    )
    return next_score
