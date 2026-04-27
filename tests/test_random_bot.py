from __future__ import annotations

import unittest

import context  # noqa: F401
from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GameState
from balatro_ai.bots.random_bot import RandomBot


class RandomBotTests(unittest.TestCase):
    def test_choose_action_returns_legal_action(self) -> None:
        actions = (
            Action(ActionType.PLAY_HAND, card_indices=(0, 1)),
            Action(ActionType.DISCARD, card_indices=(2,)),
        )
        state = GameState(legal_actions=actions)
        bot = RandomBot(seed=1)

        self.assertIn(bot.choose_action(state), actions)

    def test_choose_action_requires_legal_actions(self) -> None:
        bot = RandomBot(seed=1)

        with self.assertRaises(ValueError):
            bot.choose_action(GameState())


if __name__ == "__main__":
    unittest.main()

