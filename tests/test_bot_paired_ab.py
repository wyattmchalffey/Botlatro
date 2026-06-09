"""Tests for the paired bot A/B harness."""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


paired = importlib.import_module("scripts.bot_paired_ab")


class TestBotPairedAB(unittest.TestCase):
    def test_summarize_tracks_win_flips_and_aborts(self) -> None:
        rows = [
            {
                "seed": "0000001",
                "a": {"won": False, "run_over": True, "termination": "run_over", "ante": 5, "score": 100, "loss_frac": 0.5, "cpu_s": 1.0},
                "b": {"won": True, "run_over": True, "termination": "run_over", "ante": 8, "score": 0, "loss_frac": None, "cpu_s": 2.0},
                "d_ante": 3,
                "d_score": -100,
                "d_win": 1,
            },
            {
                "seed": "0000002",
                "a": {"won": True, "run_over": True, "termination": "run_over", "ante": 8, "score": 0, "loss_frac": None, "cpu_s": 1.5},
                "b": {"won": False, "run_over": False, "termination": "no_action", "ante": 4, "score": 10, "loss_frac": 0.1, "cpu_s": 2.5},
                "d_ante": -4,
                "d_score": 10,
                "d_win": -1,
            },
        ]
        out = paired.summarize(rows, bot_a="a", bot_b="b", faithful=False, wall_s=3.0)
        self.assertEqual(out["a"]["wins"], 1)
        self.assertEqual(out["b"]["wins"], 1)
        self.assertEqual(out["paired"]["win_flips_for_b"], 1)
        self.assertEqual(out["paired"]["win_flips_for_a"], 1)
        self.assertEqual(out["paired"]["d_score_same_ante_loss_mean"], None)
        self.assertEqual(out["paired"]["d_loss_frac_mean"], None)
        self.assertEqual(out["b"]["aborts"], 1)
        self.assertEqual(out["b"]["abort_reasons"], {"no_action": 1})

    def test_summarize_limits_score_delta_to_same_ante_losses(self) -> None:
        rows = [
            {
                "seed": "0000001",
                "a": {"won": False, "run_over": True, "termination": "run_over", "ante": 5, "score": 100, "loss_frac": 0.5, "cpu_s": 1.0},
                "b": {"won": False, "run_over": True, "termination": "run_over", "ante": 5, "score": 180, "loss_frac": 0.9, "cpu_s": 2.0},
                "d_ante": 0,
                "d_score": 80,
                "d_win": 0,
            },
            {
                "seed": "0000002",
                "a": {"won": False, "run_over": True, "termination": "run_over", "ante": 4, "score": 500, "loss_frac": 0.8, "cpu_s": 1.0},
                "b": {"won": False, "run_over": True, "termination": "run_over", "ante": 6, "score": 50, "loss_frac": 0.2, "cpu_s": 2.0},
                "d_ante": 2,
                "d_score": -450,
                "d_win": 0,
            },
        ]
        out = paired.summarize(rows, bot_a="a", bot_b="b", faithful=False, wall_s=3.0)
        self.assertEqual(out["paired"]["d_score_same_ante_loss_mean"], 80)
        self.assertAlmostEqual(out["paired"]["d_loss_frac_mean"], -0.1)

    def test_write_metrics_records_partial_completion(self) -> None:
        rows = [
            {
                "seed": "0000002",
                "a": {"won": False, "run_over": True, "termination": "run_over", "ante": 4, "score": 100, "loss_frac": 0.5, "cpu_s": 1.0},
                "b": {"won": False, "run_over": True, "termination": "run_over", "ante": 5, "score": 200, "loss_frac": 0.6, "cpu_s": 2.0},
                "d_ante": 1,
                "d_score": 100,
                "d_win": 0,
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.json"
            out = paired.write_metrics(
                path,
                rows,
                bot_a="a",
                bot_b="b",
                faithful=False,
                started=0.0,
                expected_n=2,
            )
            saved = json.loads(path.read_text(encoding="utf-8"))

        self.assertFalse(out["complete"])
        self.assertEqual(saved["expected_n"], 2)
        self.assertFalse(saved["complete"])
        self.assertEqual(saved["rows"][0]["seed"], "0000002")

    def test_auto_backend_falls_back_to_subprocess_when_process_pool_is_denied(self) -> None:
        rows = [
            {
                "seed": "0000001",
                "a": {"won": False, "run_over": True, "termination": "run_over", "ante": 5, "score": 100, "loss_frac": 0.5, "cpu_s": 1.0},
                "b": {"won": True, "run_over": True, "termination": "run_over", "ante": 8, "score": 0, "loss_frac": None, "cpu_s": 2.0},
                "d_ante": 3,
                "d_score": -100,
                "d_win": 1,
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = Path(tmpdir) / "metrics.json"
            argv = [
                "bot_paired_ab.py",
                "--bot-a",
                "a",
                "--bot-b",
                "b",
                "--seeds",
                "1",
                "--jobs",
                "2",
                "--metrics",
                str(metrics),
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(paired, "_run_pairs_process", side_effect=PermissionError),
                patch.object(paired, "_run_pairs_subprocess", return_value=rows) as subprocess_runner,
                patch("builtins.print"),
            ):
                code = paired.main()
            saved = json.loads(metrics.read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        subprocess_runner.assert_called_once()
        self.assertEqual(saved["bot_b"], "b")
        self.assertTrue(saved["complete"])


if __name__ == "__main__":
    unittest.main()
