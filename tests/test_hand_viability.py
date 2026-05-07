from __future__ import annotations

import math
import unittest

import context  # noqa: F401
from balatro_ai.api.state import Card, GameState, Joker
from balatro_ai.bots.basic_strategy_bot import _planet_capacity_gain, _preferred_hand_type
from balatro_ai.rules.hand_evaluator import HandType
from balatro_ai.search.forward_sim import PLANET_TO_HAND
from balatro_ai.search.hand_viability import hand_draw_odds


class HandViabilityTests(unittest.TestCase):
    def test_standard_five_card_pair_probability_is_exact(self) -> None:
        state = GameState(deck_size=52)

        odds = hand_draw_odds(state, draw_size=5)

        no_pair = (math.comb(13, 5) * (4**5)) / math.comb(52, 5)
        self.assertAlmostEqual(odds.probability(HandType.PAIR), 1.0 - no_pair, places=9)

    def test_known_straight_flush_draw_is_certain(self) -> None:
        state = GameState(
            deck_size=5,
            known_deck=(
                Card("9", "S"),
                Card("8", "S"),
                Card("7", "S"),
                Card("6", "S"),
                Card("5", "S"),
            ),
        )

        odds = hand_draw_odds(state, draw_size=5)

        self.assertEqual(odds.probability(HandType.STRAIGHT_FLUSH), 1.0)

    def test_duplicate_exact_card_deck_can_make_flush_five(self) -> None:
        state = GameState(deck_size=5, known_deck=tuple(Card("A", "S") for _ in range(5)))

        odds = hand_draw_odds(state, draw_size=5)

        self.assertEqual(odds.probability(HandType.FLUSH_FIVE), 1.0)
        self.assertEqual(odds.probability(HandType.FIVE_OF_A_KIND), 1.0)

    def test_single_neptune_level_does_not_make_straight_flush_preferred(self) -> None:
        state = GameState(ante=5, deck_size=52, hand_levels={HandType.STRAIGHT_FLUSH.value: 2})

        self.assertIsNone(_preferred_hand_type(state))

    def test_viable_straight_flush_level_can_define_preference(self) -> None:
        state = GameState(
            ante=5,
            deck_size=5,
            known_deck=(
                Card("9", "S"),
                Card("8", "S"),
                Card("7", "S"),
                Card("6", "S"),
                Card("5", "S"),
            ),
            hand_levels={HandType.STRAIGHT_FLUSH.value: 2},
        )

        self.assertEqual(_preferred_hand_type(state), HandType.STRAIGHT_FLUSH)

    def test_plain_flush_plan_does_not_credit_flush_house_planet_capacity(self) -> None:
        state = GameState(
            ante=5,
            deck_size=52,
            jokers=(Joker("Droll Joker"),),
            hand_levels={HandType.FLUSH.value: 1, HandType.FLUSH_HOUSE.value: 1},
        )

        self.assertEqual(_preferred_hand_type(state), HandType.FLUSH)
        self.assertEqual(_planet_capacity_gain(state, PLANET_TO_HAND["Ceres"]), 0.0)


if __name__ == "__main__":
    unittest.main()
