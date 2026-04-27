from __future__ import annotations

import unittest

import context  # noqa: F401
from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import Card, GameState
from balatro_ai.bots.greedy_bot import GreedyBot


class GreedyBotTests(unittest.TestCase):
    def test_choose_action_prefers_highest_immediate_score(self) -> None:
        state = GameState(
            hand=(
                Card("A", "S"),
                Card("A", "H"),
                Card("A", "D"),
                Card("K", "C"),
                Card("2", "D"),
            ),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0,)),
                Action(ActionType.PLAY_HAND, card_indices=(0, 1, 2)),
            ),
        )
        bot = GreedyBot(seed=1)

        self.assertEqual(bot.choose_action(state).card_indices, (0, 1, 2))

    def test_choose_action_selects_blind_before_random_fallback(self) -> None:
        state = GameState(
            legal_actions=(
                Action(ActionType.SKIP_BLIND),
                Action(ActionType.SELECT_BLIND),
            )
        )
        bot = GreedyBot(seed=1)

        self.assertEqual(bot.choose_action(state).action_type, ActionType.SELECT_BLIND)

    def test_choose_action_leaves_shop_before_random_buying(self) -> None:
        state = GameState(
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0),
                Action(ActionType.END_SHOP),
            )
        )
        bot = GreedyBot(seed=1)

        self.assertEqual(bot.choose_action(state).action_type, ActionType.END_SHOP)


if __name__ == "__main__":
    unittest.main()
