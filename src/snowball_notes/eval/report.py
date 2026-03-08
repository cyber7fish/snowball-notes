from __future__ import annotations


QUALITY_METRICS = [
    ("decision_accuracy", "Decision accuracy"),
    ("target_note_accuracy", "Target note accuracy"),
]
SAFETY_METRICS = [
    ("false_write_rate", "False write rate"),
    ("unsafe_merge_rate", "Unsafe merge rate"),
    ("proposal_rejection_rate", "Proposal rejection rate"),
]
REVIEW_METRICS = [
    ("review_precision", "Review precision"),
    ("auto_action_acceptance_rate", "Auto action acceptance rate"),
]
COST_METRICS = [
    ("avg_steps", "Avg steps"),
    ("avg_tokens", "Avg tokens"),
    ("avg_duration_ms", "Avg duration ms"),
]
REPLAY_METRICS = [
    ("logical_replay_match_rate", "Logical replay match"),
    ("live_replay_drift_rate", "Live replay drift"),
]

METRICS = QUALITY_METRICS + SAFETY_METRICS + REVIEW_METRICS + COST_METRICS + REPLAY_METRICS

SECTIONS = [
    ("Decision quality", QUALITY_METRICS),
    ("Safety", SAFETY_METRICS),
    ("Review burden", REVIEW_METRICS),
    ("Cost", COST_METRICS),
    ("Replay consistency", REPLAY_METRICS),
]

REGRESSION_KEYS = {"false_write_rate", "unsafe_merge_rate", "live_replay_drift_rate"}
IMPROVEMENT_KEYS = {"decision_accuracy", "target_note_accuracy", "review_precision",
                    "auto_action_acceptance_rate", "logical_replay_match_rate"}


def render_eval_report(report: dict, baseline: dict | None = None) -> str:
    has_baseline = baseline is not None
    prompt_ver = report["prompt_version"]
    header = f"Eval Results — {prompt_ver}"
    if has_baseline:
        header += f" vs {baseline['prompt_version']}"
    sep = "─" * max(50, len(header) + 4)
    lines = [header, sep, f"run_id: {report['run_id']}", f"model: {report['model_name']}",
             f"total_cases: {report['total_cases']}", ""]

    for section_label, section_metrics in SECTIONS:
        has_data = any(report.get(k) is not None for k, _ in section_metrics)
        if not has_data:
            continue
        lines.append(f"{section_label}:")
        for key, label in section_metrics:
            value = report.get(key)
            if value is None:
                continue
            rendered = _format_metric(key, value)
            if not has_baseline or baseline.get(key) is None:
                lines.append(f"  {label:.<36s} {rendered}")
            else:
                base_val = baseline[key]
                delta = value - base_val
                indicator = _change_indicator(key, delta)
                lines.append(
                    f"  {label:.<36s} {_format_metric(key, base_val)} → {rendered}  "
                    f"({_format_delta(key, delta)}) {indicator}"
                )
        lines.append("")

    regressions = _count_regressions(report, baseline) if has_baseline else []
    if has_baseline:
        if regressions:
            lines.append(f"Regressions: {len(regressions)} metric(s) degraded")
            for key, delta in regressions:
                label = dict(METRICS).get(key, key)
                lines.append(f"  - {label}: {_format_delta(key, delta)}")
        verdict = "FAIL" if regressions else "PASS"
        reason = f"regressions in {', '.join(k for k, _ in regressions)}" if regressions else f"{prompt_ver} is equal or better on all metrics"
        lines.append(f"\nVerdict: {verdict} — {reason}")

    case_results = report.get("results") or []
    if case_results:
        incorrect = [item for item in case_results if not item.get("decision_correct")]
        if incorrect:
            lines.append(f"\nFailed cases ({len(incorrect)}):")
            for item in incorrect[:10]:
                lines.append(
                    f"  {item['case_id']}: actual={item['actual_decision']} "
                    f"target={item.get('actual_target_note') or '-'}"
                )
    lines.append(sep)
    return "\n".join(lines)


def _format_metric(key: str, value: float) -> str:
    if _is_rate_key(key):
        return f"{value * 100:.1f}%"
    return f"{value:.2f}"


def _format_delta(key: str, value: float) -> str:
    if _is_rate_key(key):
        return f"{value * 100:+.1f}pp"
    return f"{value:+.2f}"


def _is_rate_key(key: str) -> bool:
    return (key.endswith("_rate") or key.endswith("_accuracy")
            or key.endswith("_match_rate") or key == "review_precision"
            or key == "auto_action_acceptance_rate")


def _change_indicator(key: str, delta: float) -> str:
    if abs(delta) < 1e-9:
        return "="
    if key in REGRESSION_KEYS:
        return "✓" if delta < 0 else "✗"
    if key in IMPROVEMENT_KEYS:
        return "✓" if delta > 0 else "✗"
    if key in {"avg_steps", "avg_tokens", "avg_duration_ms"}:
        return "✓" if delta < 0 else "✗"
    return ""


def _count_regressions(report: dict, baseline: dict | None) -> list[tuple[str, float]]:
    if baseline is None:
        return []
    regressions = []
    for key, _ in METRICS:
        value = report.get(key)
        base = baseline.get(key)
        if value is None or base is None:
            continue
        delta = value - base
        if abs(delta) < 1e-9:
            continue
        if key in REGRESSION_KEYS and delta > 0:
            regressions.append((key, delta))
        elif key in IMPROVEMENT_KEYS and delta < 0:
            regressions.append((key, delta))
        elif key in {"avg_steps", "avg_tokens", "avg_duration_ms"} and delta > 0:
            regressions.append((key, delta))
    return regressions
