"""Regression tests for the Phase 8 shop action-label prototype."""

from __future__ import annotations

import importlib
import unittest

from balatro_ai.api.actions import ActionType
from balatro_ai.api.state import GamePhase, GameState


labeler = importlib.import_module("scripts.phase8_shop_action_label")


def _shop_state() -> GameState:
    return GameState(
        phase=GamePhase.SHOP,
        money=20,
        modifiers={
            "shop_cards": (
                {"key": "j_joker", "name": "Joker", "set": "Joker", "cost": {"buy": 4}},
            ),
            "booster_packs": (
                {"key": "p_buffoon_normal_2", "name": "Buffoon Pack", "set": "BOOSTER", "cost": {"buy": 4}},
            ),
        },
    )


class TestPhase8ShopActionLabel(unittest.TestCase):
    def test_shop_actions_include_booster_pack_open(self) -> None:
        actions = labeler._shop_actions(_shop_state())
        self.assertIn(ActionType.BUY, {a.action_type for a in actions})
        self.assertIn(ActionType.OPEN_PACK, {a.action_type for a in actions})
        self.assertTrue(any(a.action_type == ActionType.OPEN_PACK and a.amount == 0 for a in actions))

    def test_action_key_uses_stable_action_value(self) -> None:
        pack = next(a for a in labeler._shop_actions(_shop_state()) if a.action_type == ActionType.OPEN_PACK)
        self.assertEqual(labeler._action_key(pack), ("open_pack", "pack", 0))


if __name__ == "__main__":
    unittest.main()
