"""Parity tests for `balatro_core.RustGameState` (Phase 1 of RUST_PORT_PLAN.md).

GameStateNative round-trip preserves all solver-relevant fields:
- Scalar run state (phase, ante, blind, scores, hands/discards/money, deck_size)
- Hand + known_deck (Cards)
- Jokers
- Consumables, vouchers
- hand_levels dict
- modifiers dict
- run_over, won

NOT preserved (acceptable):
- stake, seed (not used by solver hot path)
- shop, pack (handled separately)
- legal_actions (re-derived after every state change anyway)

The round-trip uses canonical-seed initial states + mid-blind states
driven by basic_strategy_bot to a SELECTING_HAND. If a new GameState
field gets added without porting to GameStateNative, this test
won't catch it (it only checks fields it knows about) — but the
solver-integration test (Phase 4 acceptance gate) will surface
any divergence in behavior.
"""

from __future__ import annotations

import unittest

import context  # noqa: F401
from balatro_ai.api.state import GamePhase
from balatro_ai.bots.basic_strategy_bot import BasicStrategyBot
from balatro_ai.sim.local_runner import LocalBalatroSimulator
from balatro_ai.solver.seed_game import SeedGame
from balatro_ai.solver.trajectory import _stable_seed_int

try:
    import balatro_core
    BALATRO_CORE_AVAILABLE = True
except ImportError:
    balatro_core = None
    BALATRO_CORE_AVAILABLE = False


def _drive_to_selecting_hand(seed: str):
    """Run basic_strategy_bot until the sim hits SELECTING_HAND."""

    game = SeedGame(seed)
    sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
    sim.state = game.initial_state()
    bot = BasicStrategyBot(seed=0)
    while sim.state.phase != GamePhase.SELECTING_HAND:
        if sim.state.run_over or sim.state.phase == GamePhase.RUN_OVER:
            break
        sim.step(bot.choose_action(sim.state))
    return sim.state


def _normalize_rank(rank: str) -> str:
    """Round-trip normalization: "T" and "10" both mean ten.

    Rust stores Rank::Ten and renders it back as "10" canonically.
    The Python source occasionally uses "T" (in deck representations
    parsed from the bridge). Both are valid for `_card_chip_value`
    (RANK_VALUES dict has both keys mapping to 10), so the
    normalization is semantically neutral.
    """

    return "10" if rank == "T" else rank


def _cards_equal_modulo_rank_alias(a, b) -> bool:
    """Card-tuple equality that treats "T" == "10"."""

    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if (
            _normalize_rank(x.rank) != _normalize_rank(y.rank)
            or x.suit != y.suit
            or x.enhancement != y.enhancement
            or x.edition != y.edition
            or x.seal != y.seal
            or x.debuffed != y.debuffed
        ):
            return False
    return True


def _assert_state_round_trip(testcase: unittest.TestCase, gs) -> None:
    """Round-trip gs through RustGameState and assert all visible fields equal.

    Documented divergence: card ranks "T" and "10" both normalize
    to "10" on the Rust side. The `_cards_equal_modulo_rank_alias`
    helper treats this as equivalent. Other Card fields are
    compared exactly.
    """

    rs = balatro_core.RustGameState.from_python(gs)
    back = rs.to_python()

    testcase.assertEqual(back.phase, gs.phase, "phase changed")
    testcase.assertEqual(back.ante, gs.ante, "ante changed")
    testcase.assertEqual(back.blind, gs.blind, "blind changed")
    testcase.assertEqual(back.required_score, gs.required_score, "required_score changed")
    testcase.assertEqual(back.current_score, gs.current_score, "current_score changed")
    testcase.assertEqual(back.hands_remaining, gs.hands_remaining, "hands_remaining changed")
    testcase.assertEqual(back.discards_remaining, gs.discards_remaining, "discards_remaining changed")
    testcase.assertEqual(back.money, gs.money, "money changed")
    testcase.assertEqual(back.deck_size, gs.deck_size, "deck_size changed")
    testcase.assertEqual(back.run_over, gs.run_over, "run_over changed")
    testcase.assertEqual(back.won, gs.won, "won changed")
    testcase.assertTrue(
        _cards_equal_modulo_rank_alias(back.hand, gs.hand),
        "hand changed (after T/10 normalization)",
    )
    testcase.assertTrue(
        _cards_equal_modulo_rank_alias(back.known_deck, gs.known_deck),
        "known_deck changed (after T/10 normalization)",
    )
    testcase.assertEqual(back.jokers, gs.jokers, "jokers changed")
    testcase.assertEqual(back.consumables, gs.consumables, "consumables changed")
    testcase.assertEqual(back.vouchers, gs.vouchers, "vouchers changed")
    testcase.assertEqual(back.hand_levels, gs.hand_levels, "hand_levels changed")
    testcase.assertEqual(back.modifiers, gs.modifiers, "modifiers changed")


@unittest.skipUnless(BALATRO_CORE_AVAILABLE, "balatro_core (Rust) not installed")
class RustGameStateInitialStateRoundTripTests(unittest.TestCase):
    """SeedGame.initial_state() goes through the round-trip cleanly."""

    def test_aaaaaaa_initial(self) -> None:
        gs = SeedGame("AAAAAAA").initial_state()
        _assert_state_round_trip(self, gs)

    def test_bbbbbbb_initial(self) -> None:
        gs = SeedGame("BBBBBBB").initial_state()
        _assert_state_round_trip(self, gs)

    def test_ccccccc_initial(self) -> None:
        gs = SeedGame("CCCCCCC").initial_state()
        _assert_state_round_trip(self, gs)

    def test_1234567_initial(self) -> None:
        gs = SeedGame("1234567").initial_state()
        _assert_state_round_trip(self, gs)


@unittest.skipUnless(BALATRO_CORE_AVAILABLE, "balatro_core (Rust) not installed")
class RustGameStateMidBlindRoundTripTests(unittest.TestCase):
    """A real SELECTING_HAND state (8 cards in hand) round-trips."""

    def test_aaaaaaa_first_selecting_hand(self) -> None:
        gs = _drive_to_selecting_hand("AAAAAAA")
        # We expect to be in SELECTING_HAND with 8 cards drawn.
        self.assertEqual(gs.phase, GamePhase.SELECTING_HAND)
        self.assertEqual(len(gs.hand), 8)
        _assert_state_round_trip(self, gs)


@unittest.skipUnless(BALATRO_CORE_AVAILABLE, "balatro_core (Rust) not installed")
class RustGameStateGettersTests(unittest.TestCase):
    """Quick-access Python getters return correct values."""

    def test_initial_state_getters(self) -> None:
        gs = SeedGame("AAAAAAA").initial_state()
        rs = balatro_core.RustGameState.from_python(gs)
        self.assertEqual(rs.ante, gs.ante)
        self.assertEqual(rs.blind, gs.blind)
        self.assertEqual(rs.required_score, gs.required_score)
        self.assertEqual(rs.current_score, gs.current_score)
        self.assertEqual(rs.hands_remaining, gs.hands_remaining)
        self.assertEqual(rs.discards_remaining, gs.discards_remaining)
        self.assertEqual(rs.money, gs.money)
        self.assertEqual(rs.deck_size, gs.deck_size)
        self.assertEqual(rs.phase, gs.phase.value)
        self.assertEqual(rs.won, gs.won)
        self.assertEqual(rs.run_over, gs.run_over)
        self.assertEqual(rs.hand_size, len(gs.hand))
        self.assertEqual(rs.n_jokers, len(gs.jokers))


if __name__ == "__main__":
    unittest.main()
