from __future__ import annotations

from collections import Counter
import unittest

import context  # noqa: F401
from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import Card, GamePhase, GameState
from balatro_ai.search.discard_search import DiscardSearchConfig, best_discard_action, discard_action_value


def pair_value(state: GameState) -> float:
    ranks = Counter(card.rank for card in state.hand)
    return 1.0 if any(count >= 2 for count in ranks.values()) else 0.0


class DiscardSearchTests(unittest.TestCase):
    def test_best_discard_action_uses_expected_draw_value(self) -> None:
        discard_ace = Action(ActionType.DISCARD, card_indices=(0,))
        discard_king = Action(ActionType.DISCARD, card_indices=(1,))
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            discards_remaining=1,
            deck_size=2,
            hand=(Card("A", "S"), Card("K", "H"), Card("2", "D")),
            known_deck=(Card("A", "H"), Card("3", "C")),
            legal_actions=(discard_ace, discard_king),
        )

        action = best_discard_action(
            state,
            config=DiscardSearchConfig(draw_samples=4, seed=1),
            value_fn=pair_value,
        )

        self.assertIsNotNone(action)
        self.assertEqual(action.card_indices, (1,))
        self.assertEqual(action.metadata["search"], "discard_expectimax")
        self.assertEqual(action.metadata["search_value"], 0.5)

    def test_discard_action_value_enumerates_small_draws(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            blind="Small Blind",
            discards_remaining=1,
            deck_size=2,
            hand=(Card("A", "S"), Card("K", "H"), Card("2", "D")),
            known_deck=(Card("A", "H"), Card("3", "C")),
        )

        value = discard_action_value(
            state,
            Action(ActionType.DISCARD, card_indices=(1,)),
            config=DiscardSearchConfig(draw_samples=10, seed=1),
            value_fn=pair_value,
        )

        self.assertEqual(value, 0.5)

    def test_best_discard_action_returns_none_without_legal_discards(self) -> None:
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            legal_actions=(Action(ActionType.PLAY_HAND, card_indices=(0,)),),
        )

        self.assertIsNone(best_discard_action(state, value_fn=pair_value))

    def test_best_discard_action_prunes_large_action_sets(self) -> None:
        actions = tuple(Action(ActionType.DISCARD, card_indices=(index,)) for index in range(6))
        state = GameState(
            phase=GamePhase.SELECTING_HAND,
            hand=(
                Card("A", "S"),
                Card("K", "H"),
                Card("Q", "D"),
                Card("3", "C"),
                Card("2", "S"),
                Card("2", "H"),
            ),
            known_deck=(Card("A", "H"),),
            legal_actions=actions,
        )
        evaluated: list[tuple[int, ...]] = []

        def value_fn(next_state: GameState) -> float:
            discarded = tuple(
                action.card_indices
                for action in actions
                if len(next_state.hand) == len(state.hand)
            )
            del discarded
            return 1.0

        def tracking_value(next_state: GameState) -> float:
            missing = set(card.short_name for card in state.hand) - set(card.short_name for card in next_state.hand)
            evaluated.append(tuple(index for index, card in enumerate(state.hand) if card.short_name in missing))
            return 1.0

        action = best_discard_action(
            state,
            config=DiscardSearchConfig(draw_samples=1, seed=1, max_actions=2),
            value_fn=tracking_value,
        )

        self.assertIsNotNone(action)
        self.assertLessEqual(len(set(evaluated)), 2)

    def test_discard_action_value_rejects_non_discard_actions(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires discard"):
            discard_action_value(
                GameState(hand=(Card("A", "S"),)),
                Action(ActionType.PLAY_HAND, card_indices=(0,)),
                value_fn=pair_value,
            )


if __name__ == "__main__":
    unittest.main()
