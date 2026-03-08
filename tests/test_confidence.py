import unittest

from snowball_notes.intake.confidence import (
    compute_source_confidence,
    compute_source_confidence_breakdown,
)


class ConfidenceTests(unittest.TestCase):
    def test_full_source_scores_high(self):
        score = compute_source_confidence(
            turn_events=[{"payload": {"type": "task_complete"}}],
            final_answer="A sufficiently long final answer that should count as complete and stable.",
            user_message="How should I design the system?",
            source_completeness="full",
            parser_version="v1",
        )
        self.assertEqual(score, 1.0)

    def test_missing_fields_reduce_score(self):
        score = compute_source_confidence(
            turn_events=[{"payload": {"type": "task_complete"}}, {"payload": {"type": "task_complete"}}],
            final_answer="short",
            user_message=None,
            source_completeness="partial",
            parser_version="legacy",
        )
        self.assertLess(score, 0.3)

    def test_breakdown_lists_penalties_and_score(self):
        breakdown = compute_source_confidence_breakdown(
            turn_events=[{"payload": {"type": "task_complete"}}, {"payload": {"type": "task_complete"}}],
            final_answer="short",
            user_message=None,
            source_completeness="partial",
            parser_version="legacy",
        )
        self.assertEqual(breakdown["base_score"], 1.0)
        self.assertEqual(breakdown["score"], 0.15)
        self.assertEqual(
            [item["code"] for item in breakdown["penalties"]],
            [
                "missing_user_message",
                "partial_source",
                "parser_version_drift",
                "short_final_answer",
                "duplicate_task_complete",
            ],
        )
        self.assertEqual(breakdown["signals"]["task_complete_count"], 2)
        self.assertEqual(breakdown["signals"]["final_answer_length"], 5)


if __name__ == "__main__":
    unittest.main()
