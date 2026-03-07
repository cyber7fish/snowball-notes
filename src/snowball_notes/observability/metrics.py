from __future__ import annotations

from .health import collect_agent_health, collect_parser_health


def render_status(db, window_days: int = 7) -> str:
    agent_health = collect_agent_health(db, window_days=window_days)
    parser_health = collect_parser_health(db)

    lines = [f"Snowball Status ({window_days}d)", "----------------------"]
    lines.append(f"processed_runs: {agent_health['agent_runs']}")

    task_state_counts = agent_health["task_state_counts"]
    if not task_state_counts:
        lines.append("task_states: none")
    else:
        lines.append("task_states:")
        for status, count in sorted(task_state_counts.items()):
            lines.append(f"  {status}: {count}")

    decision_counts = agent_health["decision_counts"]
    if decision_counts:
        lines.append("decisions:")
        for decision, count in sorted(decision_counts.items()):
            lines.append(f"  {decision}: {count}")

    lines.append("agent_health:")
    lines.append(f"  avg_steps: {_fmt_number(agent_health['avg_steps'])}")
    lines.append(
        f"  max_steps_exceeded: {agent_health['max_steps_exceeded_count']} "
        f"({_fmt_rate(agent_health['max_steps_exceeded_rate'])})"
    )
    lines.append(
        f"  tool_error_rate: {_fmt_rate(agent_health['tool_error_rate'])} "
        f"({agent_health['tool_error_count']}/{agent_health['tool_call_count']})"
    )
    lines.append(
        f"  guardrail_block_rate: {_fmt_rate(agent_health['guardrail_block_rate'])} "
        f"({agent_health['guardrail_block_count']}/{agent_health['tool_call_count']})"
    )
    lines.append(
        f"  commit_rejection_rate: {_fmt_rate(agent_health['commit_rejection_rate'])} "
        f"({agent_health['commit_rejection_count']}/{agent_health['agent_runs']})"
    )
    lines.append(f"  avg_duration_ms: {_fmt_number(agent_health['avg_duration_ms'])}")
    lines.append(f"  avg_tokens_per_run: {_fmt_number(agent_health['avg_tokens_per_run'])}")

    lines.append("review:")
    lines.append(
        f"  review_rate: {_fmt_rate(agent_health['review_rate'])} "
        f"({agent_health['review_count']}/{agent_health['agent_runs']})"
    )
    lines.append(f"  pending_reviews: {agent_health['pending_reviews']}")
    lines.append(
        f"  acceptance_rate: {_fmt_rate(agent_health['review_acceptance_rate'])} "
        f"({agent_health['resolved_review_count']} resolved)"
    )

    lines.append("parser_health:")
    lines.append(f"  avg_confidence_last_50: {_fmt_number(parser_health['avg_confidence'])}")
    lines.append(
        f"  low_confidence_rate_last_50: {_fmt_rate(parser_health['low_confidence_rate'])} "
        f"({parser_health['low_confidence_count']}/{parser_health['sample_size']})"
    )

    reconcile = agent_health["last_reconcile"]
    lines.append("reconcile:")
    if reconcile is None:
        lines.append("  last_run: never")
        lines.append("  last_result: n/a")
    else:
        lines.append(f"  last_run: {reconcile['created_at']}")
        lines.append(f"  last_result: {reconcile['status']}")
        lines.append(f"  orphan_files: {reconcile['orphan_count']}")
        lines.append(f"  missing_files: {reconcile['missing_count']}")
    return "\n".join(lines)


def _fmt_rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _fmt_number(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"
