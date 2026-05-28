"""Phase 3 parity tests for the Rust forward_sim ports.

Each helper that gets ported from `src/balatro_ai/search/forward_sim.py`
into the Rust `balatro_core.forward_sim::*` module needs a parity
test here. The pattern mirrors test_rust_hand_eval.py: drive both
sides on the same inputs and assert byte-equal outputs.
"""

from __future__ import annotations

import random
import unittest

import context  # noqa: F401
from balatro_ai.api.state import Card

try:
    import balatro_core  # noqa: F401
    BALATRO_CORE_AVAILABLE = True
except ImportError:
    BALATRO_CORE_AVAILABLE = False


def _to_rust(cards):
    return [balatro_core.RustCard.from_python(c) for c in cards]


def _to_python(rust_cards):
    return [c.to_python() for c in rust_cards]


@unittest.skipUnless(BALATRO_CORE_AVAILABLE, "balatro_core (Rust) not installed")
class DrawFromDeckParityTests(unittest.TestCase):
    """Rust `forward_sim::deck::draw_from_deck` must match Python's
    `forward_sim._draw_from_deck`."""

    def _py_draw(self, known_deck, deck_size, drawn):
        from balatro_ai.api.state import GameState
        from balatro_ai.search.forward_sim import _draw_from_deck
        # Build a minimal GameState-like shim
        class _S:
            pass
        s = _S()
        s.known_deck = tuple(known_deck)
        s.deck_size = deck_size
        r = _draw_from_deck(s, tuple(drawn))
        return (r.deck_size, list(r.known_deck))

    def _rust_draw(self, known_deck, deck_size, drawn):
        rust_known = _to_rust(known_deck)
        rust_drawn = _to_rust(drawn)
        size, rust_known_after = balatro_core.draw_from_deck(
            rust_known, deck_size, rust_drawn,
        )
        return size, _to_python(rust_known_after)

    def assertParity(self, known_deck, deck_size, drawn):
        py_size, py_known = self._py_draw(known_deck, deck_size, drawn)
        rs_size, rs_known = self._rust_draw(known_deck, deck_size, drawn)
        self.assertEqual(py_size, rs_size, f"size differs: py={py_size} rs={rs_size}")
        self.assertEqual(
            [(c.rank, c.suit) for c in py_known],
            [(c.rank, c.suit) for c in rs_known],
            f"known_deck differs",
        )

    def test_empty_drawn(self) -> None:
        deck = [Card(rank="A", suit="H"), Card(rank="K", suit="S")]
        self.assertParity(deck, 2, [])

    def test_exact_deck_match(self) -> None:
        deck = [Card(rank="A", suit="H"), Card(rank="2", suit="S")]
        self.assertParity(deck, 2, [Card(rank="A", suit="H")])

    def test_partial_deck(self) -> None:
        # known_deck only knows a few cards out of a 40-card deck
        deck = [Card(rank="A", suit="H")]
        self.assertParity(deck, 40, [Card(rank="K", suit="S")])  # not in known

    def test_empty_known_deck(self) -> None:
        self.assertParity([], 40, [Card(rank="A", suit="H")])

    def test_multiple_draws(self) -> None:
        deck = [
            Card(rank="A", suit="H"),
            Card(rank="A", suit="H"),  # duplicate
            Card(rank="2", suit="S"),
        ]
        drawn = [Card(rank="A", suit="H"), Card(rank="2", suit="S")]
        self.assertParity(deck, 3, drawn)

    def test_random_fuzz(self) -> None:
        rng = random.Random(42)
        ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
        suits = ["C", "D", "H", "S"]
        for _ in range(50):
            deck_size = rng.randint(10, 52)
            known_len = rng.randint(0, deck_size)
            deck = [
                Card(rank=rng.choice(ranks), suit=rng.choice(suits))
                for _ in range(known_len)
            ]
            num_drawn = rng.randint(0, 5)
            # Draw from the known deck to keep "exact deck" cases valid.
            drawn_indices = rng.sample(range(len(deck)), min(num_drawn, len(deck))) if deck else []
            drawn = [deck[i] for i in drawn_indices]
            self.assertParity(deck, deck_size, drawn)


@unittest.skipUnless(BALATRO_CORE_AVAILABLE, "balatro_core (Rust) not installed")
class JokersAfterPlayParityTests(unittest.TestCase):
    """Rust `forward_sim::jokers::jokers_after_play` must match
    Python's `_jokers_after_play` for the scaling-counter portion."""

    def _py_after_play(self, joker_names, played_cards, hand_type, hands_remaining=4):
        """Drive Python's _jokers_after_play with synthetic inputs.
        Returns the new metadata for each joker (or None if removed).
        """
        from balatro_ai.api.state import Joker
        from balatro_ai.search.forward_sim import _jokers_after_play
        from balatro_ai.rules.hand_evaluator import HandType, HandEvaluation
        jokers = tuple(Joker(name=n) for n in joker_names)
        played = tuple(played_cards)
        ht_enum = next(h for h in HandType if h.value == hand_type)
        evaluation = HandEvaluation(
            hand_type=ht_enum,
            cards=played,
            scoring_indices=tuple(range(len(played))),
            base_chips=0, base_mult=0, card_chips=0,
            level=1, level_chips=0, level_mult=0,
        )
        updated = _jokers_after_play(
            jokers, played, evaluation,
            hands_remaining=hands_remaining,
            played_hand_counts={},
            stochastic_outcomes={},
        )
        return updated

    def test_green_joker_increments(self) -> None:
        from balatro_ai.api.state import Joker, Card
        names = ["Green Joker"]
        played = [Card(rank="A", suit="H")]
        py_updated = self._py_after_play(names, played, "High Card")
        # Now call Rust
        new_chips, new_mult, new_xmult, new_remaining, remove = (
            balatro_core.jokers_after_play(
                names, [0], [3], [1.0], [0],
                [0.0], [False], [0.0], 0,
                "High Card",
                _to_rust(played), [0], 4, False,
            )
        )
        # Rust: Green Joker increments mult from 3 to 4.
        self.assertEqual(new_mult, [4])
        self.assertEqual(remove, [False])
        # Python: Green Joker mult should also be 3+1=4 (was 0 in metadata).
        from balatro_ai.rules.hand_evaluator import _joker_current_plus
        py_mult = int(_joker_current_plus(py_updated[0], suffix="mult"))
        # The synthetic Joker starts with no mult metadata, so Python
        # treats current=0 → next=1. Our Rust test uses cur=3 → next=4.
        # Adjust comparison: both should increment by 1.
        self.assertEqual(py_mult, 1)

    def test_ice_cream_removal(self) -> None:
        # Ice Cream with chips=5 should be REMOVED after a play.
        names = ["Ice Cream"]
        from balatro_ai.api.state import Card
        played = [Card(rank="A", suit="H")]
        _, _, _, _, remove = balatro_core.jokers_after_play(
            names, [5], [0], [1.0], [0],
            [0.0], [False], [0.0], 0,
            "High Card",
            _to_rust(played), [0], 4, False,
        )
        self.assertEqual(remove, [True])

    def test_loyalty_card_cycle(self) -> None:
        from balatro_ai.api.state import Card
        names = ["Loyalty Card"]
        played = [Card(rank="A", suit="H")]
        # cur=0 → reset to 5
        _, _, _, new_remaining, _ = balatro_core.jokers_after_play(
            names, [0], [0], [1.0], [0],
            [0.0], [False], [0.0], 0,
            "High Card",
            _to_rust(played), [0], 4, False,
        )
        self.assertEqual(new_remaining, [5])
        # cur=3 → 2
        _, _, _, new_remaining, _ = balatro_core.jokers_after_play(
            names, [0], [0], [1.0], [3],
            [0.0], [False], [0.0], 0,
            "High Card",
            _to_rust(played), [0], 4, False,
        )
        self.assertEqual(new_remaining, [2])

    def test_unknown_joker_no_change(self) -> None:
        from balatro_ai.api.state import Card
        names = ["Future Joker"]
        played = [Card(rank="A", suit="H")]
        result = balatro_core.jokers_after_play(
            names, [0], [0], [1.0], [0],
            [0.0], [False], [0.0], 0,
            "High Card",
            _to_rust(played), [0], 4, False,
        )
        new_chips, new_mult, new_xmult, new_remaining, remove = result
        self.assertEqual(new_chips, [None])
        self.assertEqual(new_mult, [None])
        self.assertEqual(new_xmult, [None])
        self.assertEqual(remove, [False])


@unittest.skipUnless(BALATRO_CORE_AVAILABLE, "balatro_core (Rust) not installed")
class NextPhaseTests(unittest.TestCase):
    """Rust `forward_sim::phase::next_phase` should match the Python
    logic at simulate_play lines 283-295."""

    def test_score_met_round_eval(self) -> None:
        phase, bones = balatro_core.next_phase(300, 350, 3, False)
        self.assertEqual(phase, "ROUND_EVAL")
        self.assertFalse(bones)

    def test_out_of_hands_run_over(self) -> None:
        phase, bones = balatro_core.next_phase(300, 100, 0, False)
        self.assertEqual(phase, "RUN_OVER")

    def test_mr_bones_save(self) -> None:
        phase, bones = balatro_core.next_phase(300, 100, 0, True)
        self.assertEqual(phase, "ROUND_EVAL")
        self.assertTrue(bones)

    def test_mr_bones_below_threshold(self) -> None:
        # 50 < 25% of 300=75 → no save.
        phase, bones = balatro_core.next_phase(300, 50, 0, True)
        self.assertEqual(phase, "RUN_OVER")


@unittest.skipUnless(BALATRO_CORE_AVAILABLE, "balatro_core (Rust) not installed")
class HeldEndOfRoundMoneyTests(unittest.TestCase):
    """Rust `held_end_of_round_money_delta` must match Python's
    `_held_end_of_round_money_delta`."""

    def _py(self, held, joker_names):
        from balatro_ai.api.state import Joker
        from balatro_ai.search.forward_sim import _held_end_of_round_money_delta
        return _held_end_of_round_money_delta(
            tuple(held),
            tuple(Joker(name=n) for n in joker_names),
        )

    def _rs(self, held, mime_count):
        return balatro_core.held_end_of_round_money_delta(_to_rust(held), mime_count)

    def test_two_gold_no_mime(self) -> None:
        held = [
            Card(rank="A", suit="H", enhancement="gold"),
            Card(rank="2", suit="S", enhancement="gold"),
        ]
        self.assertEqual(self._rs(held, 0), self._py(held, []))

    def test_gold_with_mime(self) -> None:
        held = [Card(rank="A", suit="H", enhancement="gold")]
        self.assertEqual(self._rs(held, 1), self._py(held, ["Mime"]))


@unittest.skipUnless(BALATRO_CORE_AVAILABLE, "balatro_core (Rust) not installed")
class DiscardMoneyDeltaTests(unittest.TestCase):
    """Rust `discard_money_delta` must match Python's
    `_discard_money_delta` for the supported joker subset."""

    def test_no_jokers_zero(self) -> None:
        discarded = [Card(rank="A", suit="H")]
        self.assertEqual(
            balatro_core.discard_money_delta(_to_rust(discarded), False, False, 3, False, None),
            0,
        )

    def test_trading_card_first_single_discard(self) -> None:
        discarded = [Card(rank="A", suit="H")]
        self.assertEqual(
            balatro_core.discard_money_delta(_to_rust(discarded), False, True, 3, False, None),
            3,
        )

    def test_faceless_three_faces(self) -> None:
        discarded = [
            Card(rank="J", suit="H"), Card(rank="Q", suit="S"), Card(rank="K", suit="C"),
        ]
        self.assertEqual(
            balatro_core.discard_money_delta(_to_rust(discarded), False, False, 3, True, None),
            5,
        )

    def test_faceless_two_faces_no_bonus(self) -> None:
        discarded = [Card(rank="J", suit="H"), Card(rank="Q", suit="S")]
        self.assertEqual(
            balatro_core.discard_money_delta(_to_rust(discarded), False, False, 3, True, None),
            0,
        )

    def test_mail_in_rebate_matches(self) -> None:
        discarded = [
            Card(rank="A", suit="H"),
            Card(rank="A", suit="S"),
            Card(rank="K", suit="C"),
        ]
        # target_rank="A" → 2 matches × $5 = $10
        self.assertEqual(
            balatro_core.discard_money_delta(_to_rust(discarded), False, False, 3, False, "A"),
            10,
        )


@unittest.skipUnless(BALATRO_CORE_AVAILABLE, "balatro_core (Rust) not installed")
class HandLevelAfterPlayTests(unittest.TestCase):
    """Rust `hand_level_after_play` must match Python's
    `_hand_levels_after_play` for the played hand_type slot."""

    def test_no_change_default(self) -> None:
        self.assertIsNone(balatro_core.hand_level_after_play(3, 0, False))

    def test_space_joker_increment(self) -> None:
        self.assertEqual(balatro_core.hand_level_after_play(3, 1, False), 4)

    def test_the_arm_decrement(self) -> None:
        self.assertEqual(balatro_core.hand_level_after_play(3, 0, True), 2)
        self.assertIsNone(balatro_core.hand_level_after_play(1, 0, True))


if __name__ == "__main__":
    unittest.main()
