from __future__ import annotations

import unittest

import context  # noqa: F401
from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import Card, GameState
from balatro_ai.bots.basic_strategy_bot import BasicStrategyBot


class BasicStrategyBotTests(unittest.TestCase):
    def test_plays_when_best_hand_beats_remaining_score(self) -> None:
        state = GameState(
            required_score=100,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            hand=(
                Card("A", "S"),
                Card("A", "H"),
                Card("A", "D"),
                Card("K", "C"),
                Card("2", "D"),
            ),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1, 2)),
                Action(ActionType.DISCARD, card_indices=(3, 4)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.PLAY_HAND)
        self.assertEqual(action.card_indices, (0, 1, 2))

    def test_discards_when_best_hand_is_behind_score_pace(self) -> None:
        state = GameState(
            required_score=600,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            hand=(
                Card("A", "S"),
                Card("A", "H"),
                Card("K", "D"),
                Card("9", "C"),
                Card("5", "D"),
                Card("4", "S"),
                Card("2", "C"),
            ),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1)),
                Action(ActionType.DISCARD, card_indices=(0, 1, 2, 3, 4)),
                Action(ActionType.DISCARD, card_indices=(3, 4, 5, 6)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.DISCARD)
        self.assertNotIn(0, action.card_indices)
        self.assertNotIn(1, action.card_indices)

    def test_plays_on_last_hand_even_when_score_is_low(self) -> None:
        state = GameState(
            required_score=600,
            current_score=0,
            hands_remaining=1,
            discards_remaining=4,
            hand=(Card("A", "S"), Card("A", "H")),
            legal_actions=(
                Action(ActionType.PLAY_HAND, card_indices=(0, 1)),
                Action(ActionType.DISCARD, card_indices=(0,)),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.PLAY_HAND)

    def test_buys_first_affordable_joker_before_leaving_shop(self) -> None:
        state = GameState(
            money=6,
            modifiers={
                "shop_cards": (
                    {"label": "Joker", "set": "JOKER", "cost": {"buy": 4}},
                    {"label": "Mars", "set": "PLANET", "cost": {"buy": 3}},
                )
            },
            legal_actions=(
                Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0}),
                Action(ActionType.BUY, target_id="card", amount=1, metadata={"kind": "card", "index": 1}),
                Action(ActionType.END_SHOP),
            ),
        )

        action = BasicStrategyBot(seed=1).choose_action(state)

        self.assertEqual(action.action_type, ActionType.BUY)
        self.assertEqual(action.amount, 0)


if __name__ == "__main__":
    unittest.main()
