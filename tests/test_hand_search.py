from __future__ import annotations

import unittest

import context  # noqa: F401
from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import Card, GamePhase, GameState, Joker, with_derived_legal_actions
from balatro_ai.search.hand_search import HandSearchConfig, best_hand_action


class HandSearchTests(unittest.TestCase):
    def test_prefers_immediate_clear_to_first_blind_discard_value(self) -> None:
        play_pair = Action(ActionType.PLAY_HAND, card_indices=(0, 1))
        discard = Action(ActionType.DISCARD, card_indices=(2,))
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            ante=1,
            blind="Small Blind",
            required_score=60,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            deck_size=10,
            hand=(Card("A", "S"), Card("A", "H"), Card("2", "D")),
            known_deck=(Card("K", "C"),),
            legal_actions=(play_pair, discard),
        )

        action = best_hand_action(state, config=HandSearchConfig(draw_samples=2, leaf_samples=1, seed=1))

        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, ActionType.PLAY_HAND)
        self.assertEqual(action.card_indices, (0, 1))
        self.assertEqual(action.metadata["search"], "hand_expectimax")

    def test_hunts_for_one_hand_clear_when_first_blind_hand_is_weak(self) -> None:
        state = with_derived_legal_actions(
            GameState(
                phase=GamePhase.SELECTING_HAND,
                ante=1,
                blind="Small Blind",
                required_score=300,
                current_score=0,
                hands_remaining=4,
                discards_remaining=3,
                deck_size=45,
                hand=(
                    Card("9", "H"),
                    Card("7", "D"),
                    Card("6", "H"),
                    Card("6", "D"),
                    Card("5", "C"),
                    Card("4", "D"),
                    Card("3", "H"),
                    Card("3", "C"),
                ),
            )
        )

        action = best_hand_action(state, config=HandSearchConfig(draw_samples=2, leaf_samples=1, seed=1))

        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, ActionType.DISCARD)
        self.assertEqual(action.metadata["search"], "hand_expectimax")

    def test_preserves_sixth_sense_setup_play(self) -> None:
        play_six = Action(ActionType.PLAY_HAND, card_indices=(0,))
        play_pair = Action(ActionType.PLAY_HAND, card_indices=(1, 2))
        discard = Action(ActionType.DISCARD, card_indices=(3,))
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            ante=1,
            blind="Small Blind",
            required_score=100,
            current_score=0,
            hands_remaining=4,
            discards_remaining=4,
            deck_size=10,
            hand=(Card("6", "S"), Card("A", "S"), Card("A", "H"), Card("2", "D")),
            jokers=(Joker("Sixth Sense"),),
            legal_actions=(play_six, play_pair, discard),
        )

        action = best_hand_action(state, config=HandSearchConfig(draw_samples=2, leaf_samples=1, seed=1))

        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, ActionType.PLAY_HAND)
        self.assertEqual(action.card_indices, (0,))
        self.assertIn("joker_setup", action.metadata["reason"])


if __name__ == "__main__":
    unittest.main()
