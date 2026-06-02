"""Tests for the learnable state encoder (Step 0.1).

Covers: vocabulary sanity, identity capture (the whole point — jokers/cards/
shop carry identity, editions, counters), index-bounds safety (every emitted
index fits the embedding table sized by `encoding_spec`), UNK-safety on unknown
keys, padding/empty states, hand-level mapping, and encode determinism.

Built against the real `GameState` API: jokers carry display `name`, blind type
lives in `modifiers["current_blind"]["type"]`, shop items are dicts in
`modifiers["shop_cards"]`, and `hand_levels` is a direct field.
"""

from __future__ import annotations

import unittest

from balatro_ai.api.state import Card, GamePhase, GameState, Joker
from balatro_ai.ml import encoding as enc


def _rich_state() -> GameState:
    return GameState(
        phase=GamePhase.SELECTING_HAND,
        ante=3,
        blind="The Hook",
        required_score=2000,
        current_score=450,
        hands_remaining=3,
        discards_remaining=2,
        money=27,
        deck_size=52,
        hand=(
            Card("A", "Spades", enhancement="GLASS", edition="e_foil", seal="Red"),
            Card("K", "Hearts"),
            Card("10", "Clubs", debuffed=True),
        ),
        jokers=(
            Joker(name="Blueprint", edition="e_polychrome", metadata={"eternal": True}),
            Joker(
                name="Ride the Bus",
                sell_value=3,
                metadata={"effect": "gains +1 Mult per hand (currently +12 Mult)"},
            ),
        ),
        consumables=("The Fool",),
        vouchers=("Overstock",),
        known_deck=(
            Card("A", "Spades"), Card("2", "Hearts"), Card("K", "Clubs"),
            Card("7", "Diamonds", enhancement="STEEL"),
        ),
        hand_levels={"Full House": 6, "Pair": 2},
        modifiers={
            "current_blind": {"type": "BOSS", "name": "The Hook"},
            "shop_cards": (
                {"key": "j_joker", "name": "Joker", "set": "Joker", "cost": {"buy": 4}},
                {"key": "c_pluto", "name": "Pluto", "set": "Planet", "cost": {"buy": 3}},
            ),
        },
    )


class TestEncodingSpec(unittest.TestCase):
    def test_version_and_dims(self) -> None:
        spec = enc.encoding_spec()
        self.assertEqual(spec["version"], enc.ENCODING_VERSION)
        self.assertEqual(spec["scalar_dim"], len(enc.SCALAR_NAMES))
        self.assertEqual(spec["hand_levels_dim"], len(enc.POKER_HANDS))

    def test_joker_vocab_is_complete(self) -> None:
        spec = enc.encoding_spec()
        # 150 jokers (incl. legendaries) + PAD + UNK.
        self.assertEqual(spec["joker_vocab_size"], 150 + enc._RESERVED)
        self.assertGreaterEqual(spec["consumable_vocab_size"], 45)
        self.assertGreaterEqual(spec["item_vocab_size"], 200)


class TestEncodeState(unittest.TestCase):
    def test_scalar_length(self) -> None:
        e = enc.encode_state(_rich_state())
        self.assertEqual(len(e.scalars), len(enc.SCALAR_NAMES))
        self.assertEqual(e.version, enc.ENCODING_VERSION)

    def test_joker_identity_captured(self) -> None:
        e = enc.encode_state(_rich_state())
        self.assertEqual(len(e.jokers), 2)
        self.assertGreater(e.jokers[0].key_index, enc.UNK)   # Blueprint known
        self.assertGreater(e.jokers[1].key_index, enc.UNK)   # Ride the Bus known
        self.assertNotEqual(e.jokers[0].key_index, e.jokers[1].key_index)
        self.assertNotEqual(e.jokers[0].edition_index, 0)    # polychrome
        self.assertEqual(e.jokers[0].eternal, 1.0)           # metadata flag
        self.assertGreater(e.jokers[1].counter, 0.0)         # currently +12 Mult

    def test_card_attributes_captured(self) -> None:
        e = enc.encode_state(_rich_state())
        ace = e.hand[0]
        self.assertEqual(ace.rank_index, enc._RANK_INDEX["A"])
        self.assertEqual(ace.suit_index, enc._SUIT_INDEX["Spades"])
        self.assertNotEqual(ace.enhancement_index, 0)  # GLASS
        self.assertNotEqual(ace.edition_index, 0)      # e_foil -> foil
        self.assertNotEqual(ace.seal_index, 0)         # Red
        self.assertEqual(e.hand[2].debuffed, 1.0)

    def test_shop_consumables_vouchers(self) -> None:
        e = enc.encode_state(_rich_state())
        self.assertEqual(len(e.shop), 2)
        self.assertGreater(e.shop[0].key_index, enc.UNK)  # j_joker known
        self.assertEqual(e.shop[0].item_type_index, enc._ITEM_TYPE_INDEX["joker"])
        self.assertEqual(e.shop[1].item_type_index, enc._ITEM_TYPE_INDEX["planet"])
        self.assertEqual(e.consumables, (enc._consumable_name_vocab()["The Fool"],))
        self.assertEqual(e.vouchers, (enc._voucher_name_vocab()["Overstock"],))

    def test_hand_levels_mapping(self) -> None:
        e = enc.encode_state(_rich_state())
        fh = enc.POKER_HANDS.index("Full House")
        pair = enc.POKER_HANDS.index("Pair")
        high = enc.POKER_HANDS.index("High Card")
        self.assertGreater(e.hand_levels[fh], e.hand_levels[pair])  # 6 > 2
        self.assertEqual(e.hand_levels[high], 0.0)                  # default level 1

    def test_boss_index(self) -> None:
        e = enc.encode_state(_rich_state())
        self.assertEqual(e.boss_index, enc._BOSS_INDEX["The Hook"])
        self.assertEqual(e.scalars[enc.SCALAR_NAMES.index("is_boss")], 1.0)

    def test_deck_counts_normalized(self) -> None:
        e = enc.encode_state(_rich_state())
        rank_sum = sum(e.deck_counts[: len(enc.RANKS)])
        self.assertAlmostEqual(rank_sum, 1.0, places=6)
        self.assertGreater(e.deck_counts[-1], 0.0)  # one STEEL-enhanced card


class TestRobustness(unittest.TestCase):
    def test_unknown_joker_is_unk(self) -> None:
        s = GameState(phase=GamePhase.SHOP, jokers=(Joker(name="Not A Real Joker"),))
        e = enc.encode_state(s)
        self.assertEqual(e.jokers[0].key_index, enc.UNK)

    def test_empty_state_encodes(self) -> None:
        e = enc.encode_state(GameState(phase=GamePhase.BLIND_SELECT))
        self.assertEqual(len(e.jokers), 0)
        self.assertEqual(len(e.hand), 0)
        self.assertEqual(len(e.scalars), len(enc.SCALAR_NAMES))
        self.assertEqual(e.boss_index, enc.PAD)

    def test_determinism(self) -> None:
        self.assertEqual(enc.encode_state(_rich_state()), enc.encode_state(_rich_state()))

    def test_all_indices_within_vocab_bounds(self) -> None:
        spec = enc.encoding_spec()
        e = enc.encode_state(_rich_state())
        for j in e.jokers:
            self.assertLess(j.key_index, spec["joker_vocab_size"])
            self.assertLess(j.edition_index, spec["edition_vocab_size"])
        for c in e.hand:
            self.assertLess(c.rank_index, spec["rank_vocab_size"])
            self.assertLess(c.suit_index, spec["suit_vocab_size"])
            self.assertLess(c.enhancement_index, spec["enhancement_vocab_size"])
            self.assertLess(c.seal_index, spec["seal_vocab_size"])
        for sh in e.shop:
            self.assertLess(sh.key_index, spec["item_vocab_size"])
            self.assertLessEqual(sh.item_type_index, spec["item_type_vocab_size"])
        for ci in e.consumables:
            self.assertLess(ci, spec["consumable_vocab_size"])
        for vi in e.vouchers:
            self.assertLess(vi, spec["voucher_vocab_size"])
        self.assertLess(e.boss_index, spec["boss_vocab_size"])


class TestRealCardFormat(unittest.TestCase):
    """Real seed/local-sim decks use single-letter suits (H/S/D/C) and "T" for ten,
    not the full "Hearts"/"10" forms the other tests use. The encoder MUST normalize
    both to the same index — the prior bug sent every real card to SUIT_NONE and every
    ten to rank 0, silently corrupting value/policy inputs on live data."""

    def test_short_suit_and_ten_map_to_real_indices(self) -> None:
        s = GameState(
            phase=GamePhase.SELECTING_HAND,
            hand=(Card("T", "H"), Card("A", "S"), Card("Q", "C"), Card("9", "D")),
        )
        e = enc.encode_state(s)
        self.assertEqual(e.hand[0].rank_index, enc._RANK_INDEX["10"])   # T -> 10, not 0
        self.assertEqual(e.hand[0].suit_index, enc._SUIT_INDEX["Hearts"])
        self.assertEqual(e.hand[1].suit_index, enc._SUIT_INDEX["Spades"])
        self.assertEqual(e.hand[2].suit_index, enc._SUIT_INDEX["Clubs"])
        self.assertEqual(e.hand[3].suit_index, enc._SUIT_INDEX["Diamonds"])
        # None fell through to the stone/unknown bucket.
        self.assertFalse(any(c.suit_index == enc.SUIT_NONE for c in e.hand))

    def test_short_form_equals_full_form(self) -> None:
        short = enc.encode_state(GameState(phase=GamePhase.SELECTING_HAND, hand=(Card("T", "H"),)))
        full = enc.encode_state(GameState(phase=GamePhase.SELECTING_HAND, hand=(Card("10", "Hearts"),)))
        self.assertEqual(short.hand[0], full.hand[0])

    def test_deck_counts_capture_short_suits(self) -> None:
        s = GameState(
            phase=GamePhase.SELECTING_HAND,
            known_deck=(Card("T", "H"), Card("9", "S"), Card("2", "D"), Card("K", "C")),
        )
        e = enc.encode_state(s)
        suit_counts = e.deck_counts[len(enc.RANKS): len(enc.RANKS) + len(enc.SUITS)]
        self.assertGreater(sum(suit_counts), 0.0)        # not all zero anymore
        # The ten is counted in the "10" rank slot, not the "2" slot.
        self.assertGreater(e.deck_counts[enc._RANK_INDEX["10"]], 0.0)


if __name__ == "__main__":
    unittest.main()
