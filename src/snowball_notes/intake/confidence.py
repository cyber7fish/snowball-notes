from __future__ import annotations


CURRENT_STABLE_PARSER = "v1"


def compute_source_confidence(
    turn_events: list[dict],
    final_answer: str | None,
    user_message: str | None,
    source_completeness: str,
    parser_version: str,
) -> float:
    score = 1.0
    if not final_answer:
        score -= 0.50
    if not user_message:
        score -= 0.20
    if source_completeness == "partial":
        score -= 0.20
    if parser_version != CURRENT_STABLE_PARSER:
        score -= 0.10
    if final_answer and len(final_answer.strip()) < 50:
        score -= 0.15
    task_complete_count = sum(
        1
        for event in turn_events
        if event.get("payload", {}).get("type") == "task_complete"
    )
    if task_complete_count > 1:
        score -= 0.20
    return round(max(0.0, min(1.0, score)), 2)

