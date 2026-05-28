from __future__ import annotations

import unittest

import context  # noqa: F401
from balatro_ai.rng.deck import (
    STANDARD_RANKS_LEXICOGRAPHIC,
    STANDARD_SUITS_CDHS_ORDER,
    build_standard_red_deck,
    build_standard_red_deck_short_names,
)
from balatro_ai.rng.validate import (
    PredictionConfig,
    check_initial_hand_matches_prediction,
    extract_initial_hand_short_names,
    predict_starting_hand,
    search_matching_configs,
)


def _fixture_with_hand(short_names: tuple[str, ...]) -> dict:
    return {
        "hand": {
            "cards": [
                {"value": {"rank": rank, "suit": suit}}
                for rank, suit in (name[:-1], name[-1])  # type: ignore[misc]  # narrow tuple
                for name in (short_names if False else [])  # placeholder; real builder below
            ]
        }
    }


def _hand_fixture(short_names: tuple[str, ...]) -> dict:
    cards = [
        {"value": {"rank": name[:-1], "suit": name[-1]}}
        for name in short_names
    ]
    return {"hand": {"cards": cards}}


class StandardDeckTests(unittest.TestCase):
    def test_deck_has_52_unique_cards(self) -> None:
        deck = build_standard_red_deck()
        self.assertEqual(len(deck), 52)
        self.assertEqual(len({(c.rank, c.suit) for c in deck}), 52)

    def test_first_card_is_two_of_clubs_in_verified_order(self) -> None:
        # Suit-outer C,D,H,S / rank-inner 2..9, A, J, K, Q, T (lex string
        # order, not poker-ascending). Recovered empirically from fixtures.
        deck = build_standard_red_deck_short_names()
        self.assertEqual(deck[0], "2C")
        self.assertEqual(deck[12], "TC")
        self.assertEqual(deck[13], "2D")
        self.assertEqual(deck[51], "TS")

    def test_suit_and_rank_orders_match_constants(self) -> None:
        # Guard against drift between the constants and the construction.
        self.assertEqual(STANDARD_SUITS_CDHS_ORDER, ("C", "D", "H", "S"))
        self.assertEqual(len(STANDARD_RANKS_LEXICOGRAPHIC), 13)
        self.assertEqual(STANDARD_RANKS_LEXICOGRAPHIC[0], "2")
        # Lex order: A < J < K < Q < T (because '9' < 'A' < 'J' < 'K' < 'Q' < 'T').
        self.assertEqual(STANDARD_RANKS_LEXICOGRAPHIC[-5:], ("A", "J", "K", "Q", "T"))


class ExtractInitialHandTests(unittest.TestCase):
    def test_extracts_rank_suit_pairs_in_payload_order(self) -> None:
        fixture = _hand_fixture(("QC", "JH", "9C"))
        self.assertEqual(extract_initial_hand_short_names(fixture), ("QC", "JH", "9C"))

    def test_returns_empty_tuple_when_no_hand_present(self) -> None:
        self.assertEqual(extract_initial_hand_short_names({}), ())
        self.assertEqual(extract_initial_hand_short_names({"hand": {"cards": []}}), ())


class PredictStartingHandTests(unittest.TestCase):
    def test_is_deterministic_for_same_seed_and_config(self) -> None:
        config = PredictionConfig(shuffle_key="shuffle", mix_hashed_seed=False)
        first = predict_starting_hand("AAAAAAA", config)
        second = predict_starting_hand("AAAAAAA", config)
        self.assertEqual(first, second)

    def test_different_seed_yields_different_hand(self) -> None:
        config = PredictionConfig(shuffle_key="shuffle", mix_hashed_seed=False)
        self.assertNotEqual(
            predict_starting_hand("AAAAAAA", config),
            predict_starting_hand("BBBBBBB", config),
        )

    def test_different_config_yields_different_hand(self) -> None:
        a = predict_starting_hand("AAAAAAA", PredictionConfig(shuffle_key="shuffle", mix_hashed_seed=False))
        b = predict_starting_hand("AAAAAAA", PredictionConfig(shuffle_key="front", mix_hashed_seed=False))
        # Different keys → different bootstrapped streams → different shuffles.
        self.assertNotEqual(a, b)

    def test_returns_requested_hand_size(self) -> None:
        config = PredictionConfig(shuffle_key="shuffle", mix_hashed_seed=False)
        self.assertEqual(len(predict_starting_hand("AAAAAAA", config, hand_size=5)), 5)


class CheckInitialHandTests(unittest.TestCase):
    def test_marks_ok_on_match(self) -> None:
        fixture = _hand_fixture(("QC", "JH"))
        result = check_initial_hand_matches_prediction(fixture, ("QC", "JH"))
        self.assertEqual(result.status, "ok")

    def test_marks_mismatch_on_difference(self) -> None:
        fixture = _hand_fixture(("QC", "JH"))
        result = check_initial_hand_matches_prediction(fixture, ("AS", "KD"))
        self.assertEqual(result.status, "mismatch")
        self.assertIn("predicted", result.detail)
        self.assertIn("actual", result.detail)

    def test_marks_unsupported_when_fixture_has_no_hand(self) -> None:
        result = check_initial_hand_matches_prediction({"hand": {"cards": []}}, ("QC",))
        self.assertEqual(result.status, "unsupported")


class SearchMatchingConfigsTests(unittest.TestCase):
    def test_finds_config_that_produces_the_fixture(self) -> None:
        # Build a fake fixture by running our own predictor with a known
        # config. The search should find at least that config.
        target = PredictionConfig(shuffle_key="shuffle", mix_hashed_seed=False)
        predicted = predict_starting_hand("AAAAAAA", target)
        fixture = _hand_fixture(predicted)
        matches = search_matching_configs(fixture, "AAAAAAA")
        self.assertIn(target, matches)

    def test_returns_empty_when_no_config_matches(self) -> None:
        # Hand that does not exist in any standard-deck shuffle result.
        fixture = _hand_fixture(("ZZ", "ZZ"))
        self.assertEqual(search_matching_configs(fixture, "AAAAAAA"), ())


if __name__ == "__main__":
    unittest.main()
