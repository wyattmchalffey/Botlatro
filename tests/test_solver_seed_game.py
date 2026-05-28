"""Tests for `balatro_ai.solver.seed_game` (Milestone M1).

These tests verify the seed materializer constructs a `GameState` that
matches the captured bridge fixtures for the deck and ante-1 boss, and
that the result is structurally valid for downstream `forward_sim`
consumption.
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import context  # noqa: F401
from balatro_ai.api.state import GamePhase, Stake
from balatro_ai.solver.seed_game import SeedGame, deck_for_seed


FIXTURE_DIR = Path(__file__).resolve().parent.parent / ".data" / "rng-validation"
CANONICAL_SEEDS = ("AAAAAAA", "BBBBBBB", "CCCCCCC", "1234567")


def _deck_fixture(seed: str) -> dict | None:
    path = FIXTURE_DIR / f"seed_{seed}_red_white.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _shop_fixture(seed: str) -> dict | None:
    path = FIXTURE_DIR / f"shop_seed_{seed}_red_white.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _bridge_deck_short_names(fixture: dict) -> tuple[str, ...]:
    """Pull the post-shuffle deck order out of a bridge fixture."""

    cards = fixture.get("cards", {}).get("cards", [])
    out: list[str] = []
    for card in cards:
        value = card.get("value", {}) if isinstance(card, dict) else {}
        out.append(f"{value.get('rank')}{value.get('suit')}")
    return tuple(out)


class DeckPredictionTests(unittest.TestCase):
    def test_deck_for_each_canonical_seed_matches_bridge_fixture(self) -> None:
        any_checked = False
        for seed in CANONICAL_SEEDS:
            fixture = _deck_fixture(seed)
            if fixture is None:
                continue
            any_checked = True
            expected = _bridge_deck_short_names(fixture)
            predicted = tuple(card.short_name for card in deck_for_seed(seed))
            self.assertEqual(
                predicted,
                expected,
                f"Deck mismatch for seed {seed}: pred[:5]={predicted[:5]} actual[:5]={expected[:5]}",
            )
        if not any_checked:
            self.skipTest(
                "No captured deck fixtures available; run `python -m balatro_ai.rng.capture --all`"
            )


class SeedGameInitialStateTests(unittest.TestCase):
    """Structural validity of the initial GameState produced by SeedGame."""

    def test_initial_state_is_blind_select_ante_1(self) -> None:
        state = SeedGame("AAAAAAA").initial_state()
        self.assertEqual(state.phase, GamePhase.BLIND_SELECT)
        self.assertEqual(state.ante, 1)

    def test_initial_state_has_starting_resources(self) -> None:
        state = SeedGame("AAAAAAA").initial_state()
        self.assertEqual(state.money, 4)
        self.assertEqual(state.hands_remaining, 4)
        self.assertEqual(state.discards_remaining, 4)
        self.assertEqual(state.deck_size, 52)
        self.assertEqual(len(state.known_deck), 52)
        self.assertEqual(state.stake, Stake.WHITE)

    def test_initial_state_seed_faithful_deck(self) -> None:
        # Both pathways through SeedGame should return the same deck order
        # as deck_for_seed (memoized identity check).
        game = SeedGame("AAAAAAA")
        from_game = tuple(c.short_name for c in game.initial_state().known_deck)
        from_helper = tuple(c.short_name for c in deck_for_seed("AAAAAAA"))
        self.assertEqual(from_game, from_helper)

    def test_initial_surface_exposes_boss_voucher_tags(self) -> None:
        game = SeedGame("AAAAAAA")
        surface = game.initial_surface()
        self.assertTrue(surface.boss_key.startswith("bl_"))
        self.assertTrue(surface.voucher_key.startswith("v_"))
        self.assertTrue(surface.small_tag_key.startswith("tag_"))
        self.assertTrue(surface.big_tag_key.startswith("tag_"))

    def test_initial_state_modifiers_carry_surface(self) -> None:
        game = SeedGame("AAAAAAA")
        state = game.initial_state()
        surface = state.modifiers.get("seed_game_initial_surface")
        self.assertIsInstance(surface, dict)
        self.assertIn("boss_key", surface)
        self.assertIn("voucher_key", surface)

    def test_hand_levels_initialized_to_one(self) -> None:
        state = SeedGame("AAAAAAA").initial_state()
        # Every hand type should be present at level 1.
        self.assertTrue(all(level == 1 for level in state.hand_levels.values()))
        # And we should have all 12 hand types.
        self.assertEqual(len(state.hand_levels), 12)


class SeedGameForwardSimSmokeTests(unittest.TestCase):
    """End-to-end check: initial state can drive forward_sim without bridge.

    This is the M1 acceptance test — confirms a SeedGame can be the entry
    point for the solver's search loop, not just a static prediction.
    """

    def test_initial_state_has_blind_surface(self) -> None:
        state = SeedGame("AAAAAAA").initial_state()
        self.assertEqual(state.blind, "Small Blind")
        self.assertEqual(state.required_score, 300)
        self.assertIn("current_blind", state.modifiers)

    def test_initial_state_flows_through_simulate_select_blind(self) -> None:
        from balatro_ai.search.forward_sim import simulate_select_blind

        state = SeedGame("AAAAAAA").initial_state()
        hand_size = state.modifiers.get("hand_size", 8)
        drawn = state.known_deck[:hand_size]
        next_state = simulate_select_blind(state, drawn_cards=drawn)
        self.assertEqual(next_state.phase, GamePhase.SELECTING_HAND)
        self.assertEqual(len(next_state.hand), hand_size)
        self.assertEqual(next_state.required_score, 300)

    def test_initial_state_flows_through_simulate_play(self) -> None:
        from balatro_ai.api.actions import Action, ActionType
        from balatro_ai.search.forward_sim import simulate_play, simulate_select_blind

        state = SeedGame("AAAAAAA").initial_state()
        hand_size = state.modifiers.get("hand_size", 8)
        drawn = state.known_deck[:hand_size]
        state = simulate_select_blind(state, drawn_cards=drawn)

        # Play the first 5 cards regardless of hand quality; this just
        # exercises the pipeline.
        play = Action(action_type=ActionType.PLAY_HAND, card_indices=(0, 1, 2, 3, 4))
        next_draws = state.known_deck[:5]
        next_state = simulate_play(state, play, drawn_cards=next_draws)
        self.assertEqual(next_state.hands_remaining, 3)
        self.assertGreater(next_state.current_score, 0)


class SeedGameAnteOneBossTests(unittest.TestCase):
    """Cross-validate predicted ante-1 boss against captured shop fixtures."""

    def test_predicted_ante1_boss_matches_shop_fixture(self) -> None:
        any_checked = False
        for seed in CANONICAL_SEEDS:
            fixture = _shop_fixture(seed)
            if fixture is None:
                continue
            any_checked = True
            game = SeedGame(seed)
            predicted_boss_key = game.boss_for_ante(1)
            # The fixture's blinds.boss.name (display string) maps from key
            # via rng.surfaces.BOSS_NAMES; check via the property.
            predicted_boss_name = game.initial_surface().boss_name
            actual_boss_name = fixture.get("blinds", {}).get("boss", {}).get("name", "?")
            self.assertEqual(
                predicted_boss_name,
                actual_boss_name,
                f"Boss mismatch for seed {seed}: pred={predicted_boss_name!r} "
                f"(key {predicted_boss_key}) actual={actual_boss_name!r}",
            )
        if not any_checked:
            self.skipTest(
                "No captured shop fixtures available; run `python -m balatro_ai.rng.capture_shop --all`"
            )


if __name__ == "__main__":
    unittest.main()
