from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import context  # noqa: F401
from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import Card, GamePhase, GameState, Joker
from balatro_ai.data.replay_logger import ReplayLogger


class ReplayLoggerTests(unittest.TestCase):
    def test_log_step_includes_compact_shop_and_chosen_item_details(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            state = GameState(
                phase=GamePhase.SHOP,
                ante=1,
                money=8,
                deck_size=44,
                hand=(Card("A", "S"), Card("A", "H", enhancement="Bonus")),
                known_deck=(Card("K", "S"), Card("Q", "D", seal="Red")),
                jokers=(Joker("Ice Cream", metadata={"ability": {"extra": {"chips": 85}}}),),
                hand_levels={"Pair": 2},
                modifiers={
                    "hands": {"Pair": {"level": 2, "played_this_round": 1}},
                    "shop_cards": (
                        {"label": "Joker", "set": "Joker", "cost": {"buy": 3}, "rarity": 1},
                        {"label": "Cavendish", "set": "Joker", "cost": {"buy": 4}, "rarity": 2},
                    ),
                    "booster_packs": ({"label": "Arcana Pack", "set": "Booster", "cost": {"buy": 4}},),
                },
            )
            action = Action(ActionType.BUY, target_id="card", amount=1, metadata={"kind": "card", "index": 1})

            ReplayLogger(path).log_step(state=state, legal_actions=(action,), chosen_action=action, reward=0.0)

            row = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(row["state_detail"]["shop"][1]["name"], "Cavendish")
        self.assertEqual(row["state_detail"]["booster_packs"][0]["name"], "Arcana Pack")
        self.assertEqual(row["state_detail"]["hand"][0]["name"], "AS")
        self.assertEqual(row["state_detail"]["hand"][1]["enhancement"], "Bonus")
        self.assertEqual(row["state_detail"]["known_deck"][0]["name"], "KS")
        self.assertEqual(row["state_detail"]["known_deck"][1]["seal"], "Red")
        self.assertEqual(row["state_detail"]["hand_levels"], {"Pair": 2})
        self.assertEqual(row["state_detail"]["hands"]["Pair"]["played_this_round"], 1)
        self.assertEqual(row["state_detail"]["jokers"][0]["metadata"]["ability"]["extra"]["chips"], 85)
        self.assertEqual(row["state_detail"]["deck_size"], 44)
        self.assertEqual(row["chosen_item"]["name"], "Cavendish")


if __name__ == "__main__":
    unittest.main()
