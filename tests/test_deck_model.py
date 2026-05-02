from __future__ import annotations

import json
from pathlib import Path
from random import Random
import tempfile
import unittest

import context  # noqa: F401
from balatro_ai.api.state import Card, GameState
from balatro_ai.search.deck_model import DeckModel, validate_replay_draws


def card_names(cards: tuple[Card, ...]) -> tuple[str, ...]:
    return tuple(card.short_name for card in cards)


def state_detail(
    *,
    current_score: int = 0,
    hands_remaining: int = 4,
    discards_remaining: int = 3,
    deck_size: int = 4,
    hand: list[str] | None = None,
    known_deck: list[str] | None = None,
) -> dict[str, object]:
    return {
        "phase": "selecting_hand",
        "ante": 1,
        "blind": "Small Blind",
        "required_score": 500,
        "current_score": current_score,
        "hands_remaining": hands_remaining,
        "discards_remaining": discards_remaining,
        "money": 4,
        "deck_size": deck_size,
        "hand": [_card_detail(card) for card in (hand or [])],
        "known_deck": [_card_detail(card) for card in (known_deck or [])],
        "hand_levels": {},
        "hands": {},
        "jokers": [],
    }


def _card_detail(name: str) -> dict[str, str]:
    return {"rank": name[:-1], "suit": name[-1], "name": name}


class DeckModelTests(unittest.TestCase):
    def test_from_state_uses_known_deck_as_exact_multiset(self) -> None:
        state = GameState(
            deck_size=3,
            hand=(Card("A", "S"),),
            known_deck=(Card("K", "S"), Card("K", "S"), Card("2", "H")),
        )

        model = DeckModel.from_state(state)

        self.assertTrue(model.exact)
        self.assertEqual(model.total_cards, 3)
        self.assertEqual(model.card_count(Card("K", "S")), 2)
        self.assertEqual(model.card_count(Card("A", "S")), 0)

    def test_from_state_without_known_deck_uses_standard_deck_candidates(self) -> None:
        state = GameState(deck_size=50, hand=(Card("A", "S"), Card("K", "H")))

        model = DeckModel.from_state(state)

        self.assertFalse(model.exact)
        self.assertEqual(model.total_cards, 50)
        self.assertFalse(model.contains(Card("A", "S")))
        self.assertTrue(model.contains(Card("Q", "H")))

    def test_from_state_without_known_deck_removes_visible_card_zones(self) -> None:
        state = GameState(
            deck_size=48,
            hand=(Card("A", "S"),),
            modifiers={
                "played_pile": (Card("K", "H"),),
                "discard_pile": [{"rank": "Q", "suit": "D"}],
                "destroyed_cards": ("6S",),
            },
        )

        model = DeckModel.from_state(state)

        self.assertFalse(model.exact)
        self.assertEqual(model.total_cards, 48)
        self.assertFalse(model.contains(Card("A", "S")))
        self.assertFalse(model.contains(Card("K", "H")))
        self.assertFalse(model.contains(Card("Q", "D")))
        self.assertFalse(model.contains(Card("6", "S")))

    def test_from_state_exact_known_deck_ignores_out_of_draw_zones(self) -> None:
        state = GameState(
            deck_size=1,
            hand=(Card("K", "H"),),
            known_deck=(Card("Q", "S"),),
            modifiers={"discard_pile": (Card("A", "S"),)},
        )

        model = DeckModel.from_state(state)

        self.assertTrue(model.exact)
        self.assertEqual(model.total_cards, 1)
        self.assertTrue(model.contains(Card("Q", "S")))
        self.assertFalse(model.contains(Card("A", "S")))

    def test_remove_seen_tracks_within_blind_draws(self) -> None:
        model = DeckModel.from_cards((Card("A", "S"), Card("K", "H"), Card("K", "H")))

        updated = model.remove_seen((Card("K", "H"), Card("A", "S")))

        self.assertEqual(updated.total_cards, 1)
        self.assertEqual(updated.card_count(Card("K", "H")), 1)
        self.assertEqual(updated.card_count(Card("A", "S")), 0)
        self.assertEqual(model.total_cards, 3)

    def test_sample_draws_samples_without_replacement(self) -> None:
        model = DeckModel.from_cards((Card("A", "S"), Card("K", "H"), Card("Q", "D")))

        draw = model.sample_draws(2, Random(7))

        self.assertEqual(len(draw), 2)
        self.assertEqual(len(set(card_names(draw))), 2)
        self.assertTrue(all(model.contains(card) for card in draw))

    def test_all_possible_draws_enumerates_multiset_combinations(self) -> None:
        model = DeckModel.from_cards((Card("A", "S"), Card("A", "S"), Card("K", "H")))

        outcomes = {card_names(draw) for draw in model.all_possible_draws(2)}

        self.assertEqual(outcomes, {("AS", "AS"), ("AS", "KH")})

    def test_validate_replay_draws_accepts_draws_from_known_deck(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory) / "run.jsonl"
            rows = [
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        deck_size=2,
                        hand=["AS", "AH"],
                        known_deck=["QS", "QC"],
                    ),
                    "chosen_action": {"type": "play_hand", "card_indices": [0]},
                },
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        current_score=16,
                        hands_remaining=3,
                        deck_size=1,
                        hand=["AH", "QS"],
                        known_deck=["QC"],
                    ),
                    "chosen_action": {"type": "play_hand", "card_indices": [0]},
                },
            ]
            replay_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            summary = validate_replay_draws((replay_path,))

        self.assertTrue(summary.passed)
        self.assertEqual(summary.draws_checked, 1)

    def test_validate_replay_draws_reports_impossible_known_deck_draws(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory) / "run.jsonl"
            rows = [
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        deck_size=1,
                        hand=["AS", "AH"],
                        known_deck=["QC"],
                    ),
                    "chosen_action": {"type": "play_hand", "card_indices": [0]},
                },
                {
                    "seed": 1,
                    "state_detail": state_detail(
                        current_score=16,
                        hands_remaining=3,
                        deck_size=0,
                        hand=["AH", "QS"],
                        known_deck=[],
                    ),
                    "chosen_action": {"type": "play_hand", "card_indices": [0]},
                },
            ]
            replay_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            summary = validate_replay_draws((replay_path,))

        self.assertFalse(summary.passed)
        self.assertEqual(len(summary.impossible_draws), 1)
        self.assertIn("impossible_draw=QS", summary.to_text())


if __name__ == "__main__":
    unittest.main()
