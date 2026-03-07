import unittest

from snowball_notes.intake.confidence import compute_source_confidence


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


if __name__ == "__main__":
    unittest.main()

