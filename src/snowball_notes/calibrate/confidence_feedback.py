from __future__ import annotations

from dataclasses import dataclass, field

from ..utils import new_id, now_utc_iso


HUMAN_LABELS = {"trustworthy", "partial", "bad_parse"}
BUCKET_SPECS = (
    (0.0, 0.3),
    (0.3, 0.6),
    (0.6, 0.8),
    (0.8, 1.01),
)


@dataclass
class CalibrationBucket:
    lower_bound: float
    upper_bound: float
    total: int = 0
    label_counts: dict[str, int] = field(
        default_factory=lambda: {label: 0 for label in sorted(HUMAN_LABELS)}
    )

    @property
    def name(self) -> str:
        upper = min(self.upper_bound, 1.0)
        return f"[{self.lower_bound:.1f}, {upper:.1f}{']' if upper == 1.0 else ')'}"

    @property
    def trustworthy_rate(self) -> float:
        return 0.0 if self.total == 0 else self.label_counts["trustworthy"] / self.total

    @property
    def bad_parse_rate(self) -> float:
        return 0.0 if self.total == 0 else self.label_counts["bad_parse"] / self.total


@dataclass
class CalibrationReport:
    buckets: list[CalibrationBucket]
    total_feedback: int
    recommendation: str


def record_confidence_feedback(
    db,
    turn_id: str,
    human_label: str,
    annotator: str | None = None,
) -> bool:
    if human_label not in HUMAN_LABELS:
        raise ValueError(f"unsupported human_label: {human_label}")
    event_row = db.fetchone(
        """
        SELECT turn_id, source_confidence
        FROM conversation_events
        WHERE turn_id = ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (turn_id,),
    )
    if event_row is None:
        return False
    db.execute(
        """
        INSERT INTO confidence_feedback (
          feedback_id, turn_id, source_confidence, human_label, annotator, created_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            new_id("feedback"),
            event_row["turn_id"],
            float(event_row["source_confidence"]),
            human_label,
            annotator,
            now_utc_iso(),
        ),
    )
    return True


def analyze_confidence_calibration(db) -> CalibrationReport:
    buckets = [CalibrationBucket(lower, upper) for lower, upper in BUCKET_SPECS]
    rows = db.fetchall(
        """
        SELECT source_confidence, human_label
        FROM confidence_feedback
        ORDER BY created_at ASC
        """
    )
    for row in rows:
        score = float(row["source_confidence"])
        label = row["human_label"]
        bucket = _bucket_for_score(buckets, score)
        bucket.total += 1
        bucket.label_counts[label] = bucket.label_counts.get(label, 0) + 1
    return CalibrationReport(
        buckets=buckets,
        total_feedback=len(rows),
        recommendation=_generate_weight_recommendation(buckets, len(rows)),
    )


def render_calibration_report(report: CalibrationReport) -> str:
    lines = ["Confidence Calibration", "----------------------"]
    lines.append(f"feedback_count: {report.total_feedback}")
    for bucket in report.buckets:
        counts = ", ".join(
            f"{label}={bucket.label_counts.get(label, 0)}"
            for label in ("trustworthy", "partial", "bad_parse")
        )
        lines.append(
            f"{bucket.name}: total={bucket.total} {counts}"
        )
    lines.append("")
    lines.append(f"recommendation: {report.recommendation}")
    return "\n".join(lines)


def _bucket_for_score(buckets: list[CalibrationBucket], score: float) -> CalibrationBucket:
    for bucket in buckets:
        if bucket.lower_bound <= score < bucket.upper_bound:
            return bucket
    return buckets[-1]


def _generate_weight_recommendation(buckets: list[CalibrationBucket], total_feedback: int) -> str:
    if total_feedback == 0:
        return "No confidence feedback recorded yet."
    high_conf_bucket = buckets[-1]
    if high_conf_bucket.total >= 3 and high_conf_bucket.bad_parse_rate >= 0.25:
        return (
            "High-confidence samples still contain bad_parse labels; tighten parser drift penalties "
            "or discount suspicious task_complete patterns."
        )
    low_conf_buckets = [bucket for bucket in buckets if bucket.upper_bound <= 0.6 and bucket.total >= 2]
    if any(bucket.trustworthy_rate >= 0.5 for bucket in low_conf_buckets):
        return (
            "Low-confidence buckets contain many trustworthy turns; consider relaxing penalties for "
            "partial sources or short final answers."
        )
    mid_conf_buckets = [bucket for bucket in buckets if 0.6 <= bucket.lower_bound < 0.8 and bucket.total >= 2]
    if any(bucket.bad_parse_rate >= 0.34 for bucket in mid_conf_buckets):
        return (
            "Mid-confidence samples show frequent bad_parse labels; refine parser_version and "
            "source_completeness weighting."
        )
    return "Current confidence heuristics look directionally calibrated. Keep collecting feedback."
