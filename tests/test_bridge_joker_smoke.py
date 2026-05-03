from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import context  # noqa: F401
from balatro_ai.api.actions import ActionType
from balatro_ai.sim.bridge_joker_smoke import (
    SHOP_POOLS_PATH,
    iter_base_joker_smoke_scenarios,
    iter_joker_smoke_scenarios,
    write_manifest,
)


class BridgeJokerSmokeTests(unittest.TestCase):
    def test_generated_smoke_scenarios_cover_every_shop_pool_joker(self) -> None:
        pool = json.loads(SHOP_POOLS_PATH.read_text(encoding="utf-8"))["jokers"]
        expected = {(item["key"], item["name"]) for item in pool}
        scenarios = tuple(iter_base_joker_smoke_scenarios())
        actual = {(scenario.joker_key, scenario.joker_name) for scenario in scenarios}

        self.assertEqual(actual, expected)
        self.assertEqual(len(scenarios), 150)

    def test_generated_smoke_scenarios_include_rare_and_shop_event_cases(self) -> None:
        scenarios = tuple(iter_joker_smoke_scenarios())
        categories = {scenario.category for scenario in scenarios}
        ids = [scenario.scenario_id for scenario in scenarios]

        self.assertIn("single_joker", categories)
        self.assertIn("rare_combo", categories)
        self.assertIn("shop_event", categories)
        self.assertIn("edge_case", categories)
        self.assertIn("boss_blind", categories)
        self.assertIn("stochastic", categories)
        self.assertGreater(len(scenarios), 150)
        self.assertEqual(len(ids), len(set(ids)))

    def test_generated_scenarios_are_bridge_safe_shapes(self) -> None:
        scenarios = tuple(iter_joker_smoke_scenarios())

        self.assertTrue(scenarios)
        for scenario in scenarios:
            self.assertLessEqual(scenario.selected_count, 5)
            self.assertLessEqual(scenario.selected_count, len(scenario.hand))
            self.assertLessEqual(len(scenario.hand), 8)
            self.assertGreaterEqual(scenario.money, 0)
            self.assertGreaterEqual(scenario.hands, 0)
            self.assertGreaterEqual(scenario.discards, 0)
            self.assertIn(
                scenario.action_type,
                {
                    ActionType.PLAY_HAND,
                    ActionType.DISCARD,
                    ActionType.BUY,
                    ActionType.CASH_OUT,
                    ActionType.CHOOSE_PACK_CARD,
                    ActionType.SELECT_BLIND,
                    ActionType.SELL,
                    ActionType.REROLL,
                    ActionType.END_SHOP,
                },
            )
            setup = scenario.setup_params()
            self.assertTrue(setup["clear_hand"])
            self.assertTrue(setup["clear_jokers"])
            self.assertEqual(len(setup["hand"]), len(scenario.hand))
            self.assertGreaterEqual(len(setup["jokers"]), 1)

    def test_manifest_writer_outputs_every_scenario(self) -> None:
        scenarios = tuple(iter_joker_smoke_scenarios())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_manifest(scenarios, path)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["count"], len(scenarios))
        self.assertEqual(len(payload["scenarios"]), len(scenarios))
        self.assertEqual(payload["scenarios"][0]["joker_key"], scenarios[0].joker_key)


if __name__ == "__main__":
    unittest.main()
