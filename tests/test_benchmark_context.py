import contextlib
import io
import json
import unittest

from snowball_notes.benchmarks.context import (
    format_table,
    run_benchmark,
    synthetic_messages,
)
from snowball_notes.cli import main as cli_main


class SyntheticMessagesTests(unittest.TestCase):
    def test_message_count_and_shape(self):
        messages = synthetic_messages(3, tool_result_size=200)
        # 1 user turn + 3 (assistant + tool) pairs.
        self.assertEqual(len(messages), 7)
        self.assertEqual(messages[0]["role"], "user")
        for index in range(1, 4):
            self.assertEqual(messages[2 * index - 1]["role"], "assistant")
            tool_message = messages[2 * index]
            self.assertEqual(tool_message["role"], "tool")
            self.assertEqual(tool_message["name"], "search_similar_notes")
            self.assertEqual(len(tool_message["content"]["matches"]), 3)


class RunBenchmarkTests(unittest.TestCase):
    def test_each_layer_helps_against_baseline(self):
        results = run_benchmark(steps=8, tool_result_size=4000)
        by_config = {result.config: result for result in results}

        baseline = by_config["baseline"]
        budget = by_config["budget_only"]
        cache = by_config["cache_only"]
        both = by_config["budget_and_cache"]

        # Sanity: ordered the four configs as documented.
        self.assertEqual([r.config for r in results], ["baseline", "budget_only", "cache_only", "budget_and_cache"])

        # Cache must reduce fresh chars by reading the cumulative prefix.
        self.assertGreater(cache.cache_read_chars, 0)
        self.assertLess(cache.fresh_chars, baseline.fresh_chars)
        self.assertEqual(baseline.cache_read_chars, 0)
        self.assertEqual(budget.cache_read_chars, 0)

        # Budget must cap peak step pressure; cache_only does not.
        self.assertLess(budget.max_step_chars, baseline.max_step_chars)
        self.assertEqual(cache.max_step_chars, baseline.max_step_chars)
        self.assertEqual(both.max_step_chars, budget.max_step_chars)

        # Budget reduces total sent chars (each step's payload is trimmed).
        self.assertLess(budget.fresh_chars, baseline.fresh_chars)

        # Both layers together beat baseline on every axis we care about.
        self.assertLess(both.fresh_chars, baseline.fresh_chars)
        self.assertLess(both.max_step_chars, baseline.max_step_chars)

    def test_deterministic(self):
        first = [r.to_dict() for r in run_benchmark(steps=4, tool_result_size=500)]
        second = [r.to_dict() for r in run_benchmark(steps=4, tool_result_size=500)]
        self.assertEqual(first, second)


class FormatTableTests(unittest.TestCase):
    def test_table_includes_all_configs_and_peak_column(self):
        table = format_table(
            run_benchmark(steps=3, tool_result_size=500),
            steps=3,
            tool_result_size=500,
        )
        self.assertIn("baseline", table)
        self.assertIn("budget_only", table)
        self.assertIn("cache_only", table)
        self.assertIn("budget_and_cache", table)
        self.assertIn("peak step tokens", table)


class CliBenchTests(unittest.TestCase):
    def test_cli_bench_json_emits_results(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            exit_code = cli_main(["bench", "context", "--steps", "3", "--tool-result-size", "300", "--format", "json"])
        self.assertEqual(exit_code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertEqual([row["config"] for row in payload], ["baseline", "budget_only", "cache_only", "budget_and_cache"])


if __name__ == "__main__":
    unittest.main()
