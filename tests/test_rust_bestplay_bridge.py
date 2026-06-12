"""Safety gates for the opt-in Rust best-play batch bridge."""

from __future__ import annotations

import unittest

import context  # noqa: F401
from balatro_ai.api.state import Card, Joker
from balatro_ai.search.rust_bridge import rust_best_play_scores, rust_joker_data

try:
    import balatro_core  # noqa: F401

    BALATRO_CORE_AVAILABLE = True
except ImportError:
    BALATRO_CORE_AVAILABLE = False


def _cards() -> tuple[Card, ...]:
    return (
        Card("A", "S"),
        Card("A", "H"),
        Card("K", "S"),
        Card("Q", "S"),
        Card("2", "C"),
    )


@unittest.skipUnless(BALATRO_CORE_AVAILABLE, "balatro_core (Rust) not installed")
class RustBestPlayBridgeTests(unittest.TestCase):
    def test_fast_path_handles_simple_joker_context(self) -> None:
        result = rust_best_play_scores(
            _cards(),
            {"High Card": 1},
            "Small Blind",
            (Joker("Joker"),),
            deck_size=40,
        )
        self.assertIsNotNone(result)

    def test_fast_path_resolves_blueprint_copy_before_safety_gate(self) -> None:
        result = rust_best_play_scores(
            _cards(),
            {"High Card": 1},
            "Small Blind",
            (Joker("Blueprint"), Joker("Joker")),
            deck_size=40,
        )
        self.assertIsNotNone(result)

    def test_copied_swashbuckler_uses_copied_metadata_not_sell_sum(self) -> None:
        data = rust_joker_data(
            (
                Joker("Blueprint", sell_value=5),
                Joker("Swashbuckler", sell_value=2),
                Joker("Joker", sell_value=4),
            )
        )
        self.assertIsNotNone(data)
        names = data[0]
        plus_mult = data[2]
        self.assertEqual(names, ["Swashbuckler", "Swashbuckler", "Joker"])
        self.assertEqual(plus_mult[0], 0)
        self.assertEqual(plus_mult[1], 9)

    def test_the_eye_zeroes_repeated_hand_types_after_rust_scoring(self) -> None:
        actions, scores = rust_best_play_scores(
            _cards(),
            {"High Card": 1, "Pair": 1},
            "The Eye",
            (),
            played_hand_types=("High Card",),
            deck_size=40,
        )
        score_by_action = dict(zip(actions, scores))
        self.assertEqual(score_by_action[(2,)], 0)
        self.assertGreater(score_by_action[(0, 1)], 0)

    def test_fast_path_handles_formerly_unsafe_blue_joker_context(self) -> None:
        result = rust_best_play_scores(
            _cards(),
            {"High Card": 1},
            "Small Blind",
            (Joker("Blue Joker"),),
            deck_size=40,
        )
        self.assertIsNotNone(result)

    def test_fast_path_zeroes_non_five_card_plays_under_psychic(self) -> None:
        """The Psychic no longer bails (Psychic lift, commit 371a1c2): the
        Rust batch runs and !=5-card subsets are zeroed post-batch, matching
        Python's evaluator (which zeroes any !=5-card play under The
        Psychic). The argmax must therefore be the full 5-card play with a
        Python-equal score."""

        from balatro_ai.rules.hand_evaluator import evaluate_played_cards

        result = rust_best_play_scores(
            _cards(),
            {"High Card": 1},
            "The Psychic",
            (Joker("Joker"),),
            deck_size=40,
        )
        self.assertIsNotNone(result)
        actions, scores = result
        score_by_action = dict(zip(actions, scores))
        for idxs, score in score_by_action.items():
            if len(idxs) != 5:
                self.assertEqual(score, 0, f"non-5-card play {idxs} must score 0 under The Psychic")
        five_card = (0, 1, 2, 3, 4)
        self.assertGreater(score_by_action[five_card], 0)
        python_eval = evaluate_played_cards(
            _cards(),
            hand_levels={"High Card": 1},
            blind_name="The Psychic",
            jokers=(Joker("Joker"),),
            deck_size=40,
        )
        self.assertEqual(score_by_action[five_card], python_eval.score)
        best_action = max(score_by_action, key=lambda key: score_by_action[key])
        self.assertEqual(best_action, five_card)


if __name__ == "__main__":
    unittest.main()
