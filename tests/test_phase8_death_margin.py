"""Regression tests for the Phase 8 death-margin diagnostic."""

from __future__ import annotations

import importlib
import unittest


death_margin = importlib.import_module("scripts.phase8_death_margin")


class TestPhase8DeathMargin(unittest.TestCase):
    def test_nonterminal_aborts_are_not_loss_margins(self) -> None:
        summary = death_margin._summarize_results([
            {
                "seed": "0000001",
                "won": False,
                "run_over": True,
                "phase": "run_over",
                "termination": "run_over",
                "ante": 5,
                "cur": 800,
                "req": 1000,
                "ratio": 0.8,
            },
            {
                "seed": "0000002",
                "won": False,
                "run_over": False,
                "phase": "shop",
                "termination": "no_action",
                "ante": 3,
                "cur": 0,
                "req": 800,
                "ratio": None,
            },
            {
                "seed": "0000003",
                "won": True,
                "run_over": True,
                "phase": "run_over",
                "termination": "run_over",
                "ante": 8,
                "cur": 400000,
                "req": 400000,
                "ratio": 1.0,
            },
        ])
        self.assertEqual(summary["n_losses"], 1)
        self.assertEqual(summary["n_nonterminal_aborts"], 1)
        self.assertEqual(summary["nonterminal_abort_reasons"], {"no_action": 1})
        self.assertEqual(summary["overall_median_loss_ratio"], 0.8)
        self.assertEqual(summary["late_deaths_ante5plus"]["n"], 1)


if __name__ == "__main__":
    unittest.main()
