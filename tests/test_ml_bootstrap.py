"""Tests for the bootstrap dataset generator (Stage 1.1).

Uses `greedy_bot` (fast, deterministic) so the parallel generate → persist →
reload → expand path is exercised end-to-end quickly. The real teacher
(`basic_strategy_bot`) runs through the identical machinery.
"""

from __future__ import annotations

import os
import tempfile
import unittest

from balatro_ai.ml import bootstrap as bs
from balatro_ai.ml.dataset import RunCapture
from balatro_ai.ml.encoding import ENCODING_VERSION


class TestBootstrap(unittest.TestCase):
    def test_generate_read_load(self) -> None:
        seeds = ["0000001", "0000002", "0000003"]
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "caps.jsonl")
            stats = bs.generate_captures(
                seeds, out, bot_name="greedy_bot", max_steps=120, workers=2)
            self.assertEqual(stats.n_runs, 3)
            self.assertEqual(stats.bot, "greedy_bot")

            caps = list(bs.read_captures(out))
            self.assertEqual(len(caps), 3)
            self.assertTrue(all(isinstance(c, RunCapture) for c in caps))
            self.assertTrue(all(c.n_steps > 0 for c in caps))

            examples = bs.load_examples(out)
            self.assertGreater(len(examples), 0)
            self.assertEqual(examples[0].encoded_state.version, ENCODING_VERSION)
            # One example per recorded step across all runs.
            self.assertEqual(len(examples), sum(c.n_steps for c in caps))

    def test_resume_skips_done(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "caps.jsonl")
            bs.generate_captures(
                ["0000001", "0000002"], out, bot_name="greedy_bot", max_steps=80, workers=2)
            self.assertEqual(len(list(bs.read_captures(out))), 2)
            # Re-run with an extra seed; the two done seeds are skipped.
            stats = bs.generate_captures(
                ["0000001", "0000002", "0000003"], out,
                bot_name="greedy_bot", max_steps=80, workers=2)
            self.assertEqual(stats.n_runs, 3)


if __name__ == "__main__":
    unittest.main()
