from __future__ import annotations


METRICS = [
    ("decision_accuracy", "Decision accuracy"),
    ("target_note_accuracy", "Target note accuracy"),
    ("false_write_rate", "False write rate"),
    ("unsafe_merge_rate", "Unsafe merge rate"),
    ("proposal_rejection_rate", "Proposal rejection rate"),
    ("review_precision", "Review precision"),
    ("auto_action_acceptance_rate", "Auto action acceptance rate"),
    ("logical_replay_match_rate", "Logical replay match"),
    ("live_replay_drift_rate", "Live replay drift"),
    ("avg_steps", "Avg steps"),
    ("avg_tokens", "Avg tokens"),
    ("avg_duration_ms", "Avg duration ms"),
]


def render_eval_report(report: dict, baseline: dict | None = None) -> str:
    lines = [
        f"Eval Results ({report['prompt_version']})",
        "------------------------------",
        f"run_id: {report['run_id']}",
        f"model: {report['model_name']}",
        f"total_cases: {report['total_cases']}",
    ]
    for key, label in METRICS:
        value = report.get(key)
        if value is None:
            lines.append(f"{label}: n/a")
            continue
        rendered = _format_metric(key, value)
        if baseline is None or baseline.get(key) is None:
            lines.append(f"{label}: {rendered}")
            continue
        delta = value - baseline[key]
        lines.append(f"{label}: {rendered} ({_format_delta(key, delta)})")
    case_results = report.get("results") or []
    if case_results:
        incorrect = [item for item in case_results if not item.get("decision_correct")]
        lines.append(f"failed_cases: {len(incorrect)}")
        if incorrect:
            lines.append("case_errors:")
            for item in incorrect[:5]:
                lines.append(
                    f"  {item['case_id']}: actual={item['actual_decision']} target={item.get('actual_target_note') or '-'}"
                )
    return "\n".join(lines)


def _format_metric(key: str, value: float) -> str:
    if key.endswith("_rate") or key.endswith("_accuracy") or key.endswith("_match") or key == "review_precision":
        return f"{value * 100:.1f}%"
    return f"{value:.2f}"


def _format_delta(key: str, value: float) -> str:
    if key.endswith("_rate") or key.endswith("_accuracy") or key.endswith("_match") or key == "review_precision":
        return f"{value * 100:+.1f}pp"
    return f"{value:+.2f}"
