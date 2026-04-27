from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

import context  # noqa: F401
from balatro_ai.eval.scenario_score import main


class ScenarioScoreTests(unittest.TestCase):
    def test_scenario_score_prints_evaluation(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            result = main(["--cards", "KS", "--jokers", "Hanging Chad,Photograph"])

        self.assertEqual(result, 0)
        self.assertIn("Hand type: High Card", output.getvalue())
        self.assertIn("Score: 280", output.getvalue())


if __name__ == "__main__":
    unittest.main()
