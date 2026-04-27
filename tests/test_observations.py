from __future__ import annotations

import unittest

import context  # noqa: F401
from balatro_ai.api.state import Card, GameState, Joker
from balatro_ai.env.observations import BASIC_FEATURES, vector_from_state


class ObservationTests(unittest.TestCase):
    def test_basic_vector_matches_feature_count(self) -> None:
        state = GameState(
            ante=2,
            money=7,
            hand=(Card("A", "S"), Card("K", "H")),
            jokers=(Joker("Greedy Joker"),),
            hand_levels={"Pair": 2, "Flush": 1},
        )

        vector = vector_from_state(state)

        self.assertEqual(len(vector), len(BASIC_FEATURES))
        self.assertEqual(vector[BASIC_FEATURES.index("ante")], 2.0)
        self.assertEqual(vector[BASIC_FEATURES.index("hand_size")], 2.0)
        self.assertEqual(vector[BASIC_FEATURES.index("joker_count")], 1.0)
        self.assertEqual(vector[BASIC_FEATURES.index("hand_level_sum")], 3.0)


if __name__ == "__main__":
    unittest.main()

