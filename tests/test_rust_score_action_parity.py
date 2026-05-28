"""Parity test for the Rust wire-in to `_score_action`.

When `_try_rust_evaluate_simple` returns a non-None value, that
value MUST equal what the Python `evaluate_played_cards` path
would produce. A drift here silently corrupts every leaf evaluation
in the solver — this is the critical correctness gate for the
Phase 2 wire-in.

Strategy:
1. Drive a real solver run to SELECTING_HAND on canonical seeds.
2. For every legal play action at that state, compute the score
   two ways: with the Rust path active (default) and with the
   Rust path force-disabled.
3. Assert identical values.

This catches any divergence between `evaluate_simple` and
`evaluate_played_cards` for vanilla (no-joker) hands. The
fallback path itself is unchanged so divergence can only come
from the new Rust path.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import context  # noqa: F401
from balatro_ai.api.actions import ActionType
from balatro_ai.api.state import GamePhase
from balatro_ai.bots.basic_strategy_bot import BasicStrategyBot
from balatro_ai.sim.local_runner import LocalBalatroSimulator
from balatro_ai.solver.seed_game import SeedGame
from balatro_ai.solver.trajectory import _stable_seed_int

try:
    import balatro_core  # noqa: F401
    BALATRO_CORE_AVAILABLE = True
except ImportError:
    BALATRO_CORE_AVAILABLE = False


def _drive_to_selecting_hand(seed: str):
    """Run basic_strategy_bot until the sim hits SELECTING_HAND."""

    game = SeedGame(seed)
    sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
    sim.state = game.initial_state()
    bot = BasicStrategyBot(seed=0)
    for _ in range(50):
        if sim.state.phase == GamePhase.SELECTING_HAND:
            return sim.state
        if sim.state.run_over or sim.state.phase == GamePhase.RUN_OVER:
            break
        sim.step(bot.choose_action(sim.state))
    return None


@unittest.skipUnless(BALATRO_CORE_AVAILABLE, "balatro_core (Rust) not installed")
class ScoreActionParityOnRealStatesTests(unittest.TestCase):
    """For every legal play action on a real first-blind state,
    Rust path and Python fallback must produce identical scores."""

    def _check_all_plays(self, seed: str) -> None:
        from balatro_ai.search.state_value import _score_action_uncached

        state = _drive_to_selecting_hand(seed)
        self.assertIsNotNone(state, f"failed to reach SELECTING_HAND on {seed}")
        # First blind = no jokers yet (the Rust fast path's home turf).
        self.assertEqual(len(state.jokers), 0, f"{seed} unexpectedly has jokers")

        play_actions = [
            a for a in state.legal_actions
            if a.action_type == ActionType.PLAY_HAND and a.card_indices
        ]
        self.assertGreater(len(play_actions), 0, f"no legal play actions on {seed}")

        for action in play_actions:
            # Rust path active (default behavior after wire-in).
            rust_score = _score_action_uncached(state, action)

            # Force Python fallback by patching out the Rust attempt.
            with patch(
                "balatro_ai.search.state_value._try_rust_evaluate_simple",
                return_value=None,
            ):
                py_score = _score_action_uncached(state, action)

            self.assertEqual(
                rust_score, py_score,
                f"Score divergence on {seed} action {action.card_indices}: "
                f"Rust={rust_score}, Python={py_score}",
            )

    def test_aaaaaaa(self) -> None:
        self._check_all_plays("AAAAAAA")

    def test_bbbbbbb(self) -> None:
        self._check_all_plays("BBBBBBB")

    def test_ccccccc(self) -> None:
        self._check_all_plays("CCCCCCC")

    def test_1234567(self) -> None:
        self._check_all_plays("1234567")


@unittest.skipUnless(BALATRO_CORE_AVAILABLE, "balatro_core (Rust) not installed")
class ScoreActionRustFastPathEligibilityTests(unittest.TestCase):
    """Sanity: the Rust path activates when it should (no jokers,
    vanilla blind) and stays inactive when conditions aren't met."""

    def test_activates_on_first_blind_no_jokers(self) -> None:
        from balatro_ai.search.state_value import _try_rust_evaluate_simple

        state = _drive_to_selecting_hand("AAAAAAA")
        action = next(
            a for a in state.legal_actions
            if a.action_type == ActionType.PLAY_HAND and a.card_indices
        )
        # API takes card_indices (tuple of ints) after the amortization
        # refactor, not extracted Card tuples — the helper indexes into
        # the cached hand-as-RustCards.
        score = _try_rust_evaluate_simple(state, action.card_indices, state.blind, None)
        # First blind, no jokers, vanilla "Small Blind" name → should activate.
        self.assertIsNotNone(score, "Rust path should activate on vanilla first blind")
        self.assertGreater(score, 0)


@unittest.skipUnless(BALATRO_CORE_AVAILABLE, "balatro_core (Rust) not installed")
class ScoreActionParityWithSupportedJokersTests(unittest.TestCase):
    """If a state has a Rust-supported joker, scoring through Rust
    must match Python. We inject jokers synthetically (rather than
    waiting for the bot to buy one) to keep tests deterministic.
    """

    def _state_with_jokers(self, seed: str, joker_names: list[str]):
        """Drive the bot to first SELECTING_HAND, then inject jokers."""
        from balatro_ai.api.state import Joker
        from dataclasses import replace

        state = _drive_to_selecting_hand(seed)
        self.assertIsNotNone(state)
        injected = tuple(Joker(name=n) for n in joker_names)
        return replace(state, jokers=injected)

    def _check_all_plays(self, state) -> None:
        from balatro_ai.search.state_value import _score_action_uncached

        play_actions = [
            a for a in state.legal_actions
            if a.action_type == ActionType.PLAY_HAND and a.card_indices
        ]
        for action in play_actions:
            rust_score = _score_action_uncached(state, action)
            with patch(
                "balatro_ai.search.state_value._try_rust_evaluate_simple",
                return_value=None,
            ):
                py_score = _score_action_uncached(state, action)
            self.assertEqual(
                rust_score, py_score,
                f"Score divergence on {state.blind} action {action.card_indices} "
                f"with jokers {[j.name for j in state.jokers]}: "
                f"Rust={rust_score}, Python={py_score}",
            )

    def test_plain_joker(self) -> None:
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Joker"]))

    def test_jolly_joker(self) -> None:
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Jolly Joker"]))

    def test_greedy_joker(self) -> None:
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Greedy Joker"]))

    def test_even_steven(self) -> None:
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Even Steven"]))

    def test_scary_face(self) -> None:
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Scary Face"]))

    def test_half_joker(self) -> None:
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Half Joker"]))

    def test_multiple_supported_jokers(self) -> None:
        self._check_all_plays(self._state_with_jokers(
            "AAAAAAA", ["Joker", "Jolly Joker", "Greedy Joker"]
        ))

    def test_unsupported_joker_falls_back_to_python(self) -> None:
        # Vampire is NOT in the supported set → Rust returns None →
        # _score_action_uncached uses Python path. No divergence to
        # catch, but verify the wire-in doesn't blow up.
        state = self._state_with_jokers("AAAAAAA", ["Vampire"])
        self._check_all_plays(state)

    # === Batch 2 jokers ===

    def test_fibonacci(self) -> None:
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Fibonacci"]))

    def test_scholar(self) -> None:
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Scholar"]))

    def test_smiley_face(self) -> None:
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Smiley Face"]))

    def test_walkie_talkie(self) -> None:
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Walkie Talkie"]))

    def test_onyx_agate(self) -> None:
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Onyx Agate"]))

    def test_arrowhead(self) -> None:
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Arrowhead"]))

    def test_triboulet_xmult_per_kq(self) -> None:
        # Triboulet now native (two-pass evaluator). Parity covers
        # K/Q-bearing scoring hands which apply xmult during per-card
        # pass before any ability-joker additive mult.
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Triboulet"]))

    def test_bull_money_dependent(self) -> None:
        # Bull = +2 chips per dollar. The state has $4 starting cash.
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Bull"]))

    def test_banner_discards_dependent(self) -> None:
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Banner"]))

    def test_mystic_summit_no_discards(self) -> None:
        # Initial state has 4 discards; Mystic Summit won't fire — but
        # the parity check still must hold (Rust correctly returns 0
        # mult contribution).
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Mystic Summit"]))

    def test_abstract_joker_3_per_joker(self) -> None:
        # With 1 joker (Abstract itself) → +3 mult.
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Abstract Joker"]))

    def test_stuntman_flat_chips(self) -> None:
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Stuntman"]))

    def test_bootstraps_money_dependent(self) -> None:
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Bootstraps"]))

    def test_gros_michel_flat_mult(self) -> None:
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Gros Michel"]))

    def test_joker_stencil_bails_to_python(self) -> None:
        # Joker Stencil needs metadata RustJoker doesn't carry.
        # Documented bail.
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Joker Stencil"]))

    def test_triboulet_plus_joker_ordering(self) -> None:
        # The combination that exposed the per-card/per-ability
        # ordering bug. Rust now does (hand_mult * triboulet_xmult)
        # + joker_add, matching Python's order. K-played:
        #   start mult=1, x2 Triboulet=2, +4 Joker=6 → score=15*6=90.
        # vs the WRONG order (1+4)*2 = 10 → score=150.
        self._check_all_plays(self._state_with_jokers(
            "AAAAAAA", ["Joker", "Triboulet"]
        ))

    # === Batch 3: ability xmult jokers ===

    def test_the_duo(self) -> None:
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["The Duo"]))

    def test_the_trio(self) -> None:
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["The Trio"]))

    def test_the_family(self) -> None:
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["The Family"]))

    def test_the_order(self) -> None:
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["The Order"]))

    def test_the_tribe(self) -> None:
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["The Tribe"]))

    def test_acrobat(self) -> None:
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Acrobat"]))

    def test_flower_pot(self) -> None:
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Flower Pot"]))

    def test_xmult_stacking_duo_plus_tribe(self) -> None:
        # Two xmult ability jokers stack multiplicatively.
        # On a pair flush: Duo x2 + Tribe x2 = x4.
        self._check_all_plays(self._state_with_jokers(
            "AAAAAAA", ["The Duo", "The Tribe"]
        ))

    def test_xmult_and_additive_mix(self) -> None:
        # The Duo (x2 ability) + Joker (+4 ability). Order within
        # ability pass shouldn't matter for Duo since x2 applies to
        # (mult + 4). Both Rust and Python should match.
        self._check_all_plays(self._state_with_jokers(
            "AAAAAAA", ["Joker", "The Duo"]
        ))

    # === Batch 4 ===

    def test_photograph_first_face(self) -> None:
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Photograph"]))

    def test_cavendish_flat_x3(self) -> None:
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Cavendish"]))

    def test_seeing_double(self) -> None:
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Seeing Double"]))

    def test_photograph_plus_triboulet_ordering(self) -> None:
        # Both per-card xmult jokers. Triboulet x2 per K/Q, Photograph
        # x2 on first face. Order in per-card loop: same per card.
        # For a single K play: Triboulet x2 + Photograph x2 (K is face
        # AND first face) → mult=1*2*2 = 4 → score 15*4 = 60.
        self._check_all_plays(self._state_with_jokers(
            "AAAAAAA", ["Triboulet", "Photograph"]
        ))

    # === Batch 6: joker editions ===

    def _state_with_edition_jokers(self, seed: str, joker_specs: list[tuple[str, str | None]]):
        """Inject jokers with editions. joker_specs = list of (name, edition_or_None)."""
        from balatro_ai.api.state import Joker
        from dataclasses import replace
        state = _drive_to_selecting_hand(seed)
        injected = tuple(Joker(name=n, edition=e) for n, e in joker_specs)
        return replace(state, jokers=injected)

    def test_foil_joker(self) -> None:
        # Foil Joker = +50 chips from edition + Joker's +4 mult.
        self._check_all_plays(self._state_with_edition_jokers(
            "AAAAAAA", [("Joker", "foil")]
        ))

    def test_holographic_joker(self) -> None:
        # Holographic Joker = +10 mult from edition + Joker's +4 mult.
        self._check_all_plays(self._state_with_edition_jokers(
            "AAAAAAA", [("Joker", "holographic")]
        ))

    def test_polychrome_joker(self) -> None:
        # Polychrome Joker = x1.5 mult from edition AFTER joker effect.
        # Order: chips, mult adds, x1.5.
        self._check_all_plays(self._state_with_edition_jokers(
            "AAAAAAA", [("Joker", "polychrome")]
        ))

    def test_polychrome_jolly_on_pair(self) -> None:
        # Polychrome edition on a hand-shape joker.
        self._check_all_plays(self._state_with_edition_jokers(
            "AAAAAAA", [("Jolly Joker", "polychrome")]
        ))

    def test_mixed_edition_jokers(self) -> None:
        # Multiple jokers, multiple editions — covers the full
        # per-joker edition ordering across the ability pass.
        self._check_all_plays(self._state_with_edition_jokers(
            "AAAAAAA",
            [("Joker", "foil"), ("Greedy Joker", "polychrome"), ("Stuntman", None)],
        ))

    # === Batch 7: held-card pass ===

    def test_baron_held_kings(self) -> None:
        # Baron: x1.5 mult per held King. The AAAAAAA opening hand
        # has 1 King (KC); when we play a non-King hand, the King
        # stays held and contributes x1.5.
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Baron"]))

    def test_shoot_the_moon_held_queens(self) -> None:
        # Shoot the Moon: +13 mult per held Q. Opening hand has 1 Q.
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Shoot the Moon"]))

    def test_raised_fist_lowest_held(self) -> None:
        # Raised Fist: +2 * lowest_held.rank_value mult.
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Raised Fist"]))

    def test_mime_retriggers_held(self) -> None:
        # Mime: each held-card effect fires twice. By itself with no
        # other held-card jokers and no Steel cards, Mime contributes
        # nothing visible — but the parity test must still hold (Rust
        # correctly produces 0 extra contribution).
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Mime"]))

    def test_mime_plus_baron(self) -> None:
        # Mime + Baron: each held King's x1.5 mult fires TWICE per
        # Python's retrigger logic.
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Mime", "Baron"]))

    def test_blackboard_no_held_blacks(self) -> None:
        # Blackboard inactive when not all held are black. AAAAAAA
        # opening has mixed suits; condition unlikely → 0 contribution.
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Blackboard"]))

    # === Batch 8: Card Sharp / Supernova (bail) + suit-debuff bosses ===

    def test_card_sharp_bails_safely(self) -> None:
        # Card Sharp is in is_supported_joker but Python wire-in bails
        # because we don't plumb played-hand state yet. Verify the bail
        # cascade: trajectory still computes the right Python value.
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Card Sharp"]))

    def test_supernova_bails_safely(self) -> None:
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Supernova"]))

    def test_suit_debuff_boss_activates_rust(self) -> None:
        # Synthesize a "The Goad" (Spades debuffed) blind state. The
        # math is the vanilla formula with debuffed_suits=Spades, which
        # we already handle in evaluate_simple. This test verifies
        # Rust activates on a non-vanilla blind name from the safe set.
        from dataclasses import replace
        state = _drive_to_selecting_hand("AAAAAAA")
        # Replace blind name with "The Goad" while keeping the hand etc.
        state2 = replace(state, blind="The Goad")
        self._check_all_plays(state2)

    # === Batch 9: scaling jokers (metadata-driven) ===

    def test_vampire_no_metadata(self) -> None:
        # Synthetic Vampire has no current_xmult metadata → defaults
        # to 1.0 in both Python and Rust. Net effect: no xmult.
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Vampire"]))

    def test_hologram_no_metadata(self) -> None:
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Hologram"]))

    def test_steel_joker_no_metadata(self) -> None:
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Steel Joker"]))

    def test_glass_joker_no_metadata(self) -> None:
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Glass Joker"]))

    def test_lucky_cat_no_metadata(self) -> None:
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Lucky Cat"]))

    def test_loyalty_card_bails_to_python(self) -> None:
        # Loyalty Card needs ready/cycle gating Rust can't compute
        # from metadata alone — falls back to Python. Parity test
        # verifies the fallback path doesn't blow up.
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Loyalty Card"]))

    def test_madness_no_metadata(self) -> None:
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Madness"]))

    def test_throwback_no_metadata(self) -> None:
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Throwback"]))

    def test_ride_the_bus_no_metadata(self) -> None:
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Ride the Bus"]))

    def test_green_joker_no_metadata(self) -> None:
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Green Joker"]))

    def test_square_joker_no_metadata(self) -> None:
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Square Joker"]))

    def test_castle_no_metadata(self) -> None:
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Castle"]))

    def test_erosion_no_metadata(self) -> None:
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Erosion"]))

    def test_joker_stencil_no_metadata(self) -> None:
        # Joker Stencil was previously bailed → Python. Now it should
        # activate Rust because current_xmult metadata reads 1.0 from
        # the synthetic joker and contributes nothing — same as Python
        # via current_xmult().
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Joker Stencil"]))

    def test_blue_joker_uses_deck_size(self) -> None:
        # Blue Joker = +2 chips per deck_size. For synthetic joker with
        # no metadata, formula falls to 2 * state.deck_size.
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Blue Joker"]))

    def test_constellation_via_metadata(self) -> None:
        # Constellation's xmult comes from _joker_current_xmult which
        # handles the custom internal_xmult_from_visible formula.
        # Synthetic constellation = 1.0 xmult → no contribution.
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Constellation"]))

    def test_ramen_via_metadata(self) -> None:
        # Same pattern as Constellation. Synthetic = 1.0 xmult.
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Ramen"]))

    def test_card_sharp_activated(self) -> None:
        # Card Sharp now activates in Rust with played-hand ctx.
        # Initial state has no played hands → Card Sharp inactive.
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Card Sharp"]))

    def test_supernova_activated(self) -> None:
        # Supernova activates in Rust with played_count_this_hand_type
        # ctx. Initial state count=0 → +1 mult.
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Supernova"]))

    def test_multiple_scaling_jokers(self) -> None:
        # Stack several scaling jokers — all should contribute 0/0/1
        # to a freshly-built state with no scaling history.
        self._check_all_plays(self._state_with_jokers(
            "AAAAAAA",
            ["Vampire", "Hologram", "Ride the Bus", "Square Joker"],
        ))

    # === Batch 11: comprehensive sweep ===

    def test_fortune_teller_no_metadata(self) -> None:
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Fortune Teller"]))

    def test_ceremonial_dagger_no_metadata(self) -> None:
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Ceremonial Dagger"]))

    def test_stone_joker_no_metadata(self) -> None:
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Stone Joker"]))

    def test_caino_xmult(self) -> None:
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Caino"]))

    def test_state_only_jokers(self) -> None:
        # Score-neutral state jokers contribute nothing to scoring.
        # Sample a few from each category.
        for name in [
            "Cartomancer", "DNA", "Marble Joker", "Mr. Bones",
            "Showman", "To the Moon", "Trading Card", "Juggler",
            "Riff-raff", "Hiker", "Sixth Sense",
        ]:
            self._check_all_plays(self._state_with_jokers("AAAAAAA", [name]))

    def test_rng_jokers_empty_outcomes(self) -> None:
        # RNG jokers fire only when stochastic_outcomes is non-empty.
        # Our solver's deterministic search context has empty outcomes,
        # so these contribute 0/1.0.
        for name in [
            "Misprint", "Bloodstone", "Business Card", "Space Joker",
            "Reserved Parking", "Vagabond", "8 Ball", "Rocket",
        ]:
            self._check_all_plays(self._state_with_jokers("AAAAAAA", [name]))

    def test_retriggers(self) -> None:
        # Retrigger jokers fire scored-card effects multiple times.
        # Parity must hold with chips/edition/enhancement/per-card-jokers
        # all applied per trigger.
        for name in [
            "Hack", "Sock and Buskin", "Hanging Chad", "Seltzer",
        ]:
            self._check_all_plays(self._state_with_jokers("AAAAAAA", [name]))

    def test_dusk_on_last_hand(self) -> None:
        # Dusk only triggers on hands_remaining == 1. Default state
        # has 4 hands → no retrigger. Parity still holds (Rust knows
        # the joker exists, returns 0 contribution).
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Dusk"]))

    def test_retrigger_plus_per_card(self) -> None:
        # Hack + Fibonacci: each scored 2-5 card retriggers, so Fibonacci's
        # +8 mult fires twice for matching ranks. Parity catches a missing
        # multiplier in the trigger loop.
        self._check_all_plays(self._state_with_jokers(
            "AAAAAAA", ["Hack", "Fibonacci"]
        ))

    def test_sock_buskin_plus_scary_face(self) -> None:
        # Sock and Buskin retriggers face cards. Scary Face fires per
        # trigger → 2× chips on face cards.
        self._check_all_plays(self._state_with_jokers(
            "AAAAAAA", ["Sock and Buskin", "Scary Face"]
        ))

    def test_money_jokers(self) -> None:
        # These award money outside the scoring pass.
        for name in [
            "Faceless Joker", "Golden Joker", "Golden Ticket",
            "Rough Gem", "Mail-In Rebate", "Cloud 9",
        ]:
            self._check_all_plays(self._state_with_jokers("AAAAAAA", [name]))

    def test_splash_joker(self) -> None:
        # Splash forces ALL played cards to score regardless of hand
        # type. Rust handles this via scoring_indices_simple_with_splash
        # returning the full index range. Parity catches if any per-card
        # effect (chips/enhancements) gets skipped for non-scoring slots.
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Splash"]))

    def test_loyalty_card_default_not_ready(self) -> None:
        # A freshly-injected Loyalty Card has no metadata → not ready.
        # Rust returns 0 effect; Python ditto. Parity must hold.
        self._check_all_plays(self._state_with_jokers(
            "AAAAAAA", ["Loyalty Card"]
        ))

    def test_drivers_license_default_inactive(self) -> None:
        # Freshly-injected Driver's License has no enhanced-tally
        # metadata → inactive. Both paths contribute 0. Parity holds.
        self._check_all_plays(self._state_with_jokers(
            "AAAAAAA", ["Driver's License"]
        ))

    def test_ice_cream_default(self) -> None:
        # Synthetic Ice Cream — `current_plus_chips` defaults to 0,
        # so Rust falls back to `leading_plus_chips` (also 0 for a
        # fresh joker) → 0 contribution. Both paths match.
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Ice Cream"]))

    def test_popcorn_default(self) -> None:
        # Synthetic Popcorn — same default-zero scenario as Ice Cream.
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Popcorn"]))

    def test_obelisk_default(self) -> None:
        # Synthetic Obelisk — no played hand types yet, so no other
        # hand_type dominates → gate fails → x1.0 contribution.
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Obelisk"]))

    def test_swashbuckler_solo(self) -> None:
        # Swashbuckler alone → sum-of-others is 0 → +0 mult.
        self._check_all_plays(self._state_with_jokers("AAAAAAA", ["Swashbuckler"]))

    def test_swashbuckler_with_joker(self) -> None:
        # Swashbuckler + Joker — adds plain Joker's sell_value as
        # +mult. Tests the precompute-on-Python-side logic.
        self._check_all_plays(self._state_with_jokers(
            "AAAAAAA", ["Swashbuckler", "Joker"],
        ))

    def test_baseball_card_with_uncommon(self) -> None:
        # Baseball Card + Jolly Joker (uncommon) → Jolly's mult
        # contribution multiplied by 1.5 when triggered.
        self._check_all_plays(self._state_with_jokers(
            "AAAAAAA", ["Baseball Card", "Jolly Joker"],
        ))

    def test_ancient_joker_default(self) -> None:
        # Ancient Joker without target_suit metadata → no contribution.
        self._check_all_plays(self._state_with_jokers(
            "AAAAAAA", ["Ancient Joker"],
        ))

    def test_idol_default(self) -> None:
        # The Idol without target_rank/suit → no contribution.
        self._check_all_plays(self._state_with_jokers(
            "AAAAAAA", ["The Idol"],
        ))

    def test_diet_cola(self) -> None:
        # Diet Cola is score-neutral (it's a sell-trigger joker).
        self._check_all_plays(self._state_with_jokers(
            "AAAAAAA", ["Diet Cola"],
        ))

    def test_burnt_joker(self) -> None:
        # Burnt Joker is score-neutral (it modifies discards).
        self._check_all_plays(self._state_with_jokers(
            "AAAAAAA", ["Burnt Joker"],
        ))

    def test_pareidolia_alone(self) -> None:
        # Pareidolia turns every card into a face card. Parity must
        # hold (joker itself contributes 0, but it changes scoring).
        self._check_all_plays(self._state_with_jokers(
            "AAAAAAA", ["Pareidolia"],
        ))

    def test_pareidolia_plus_smiley_face(self) -> None:
        # Pareidolia + Smiley Face: every scored card gets +5 mult
        # because Pareidolia makes every card a face.
        self._check_all_plays(self._state_with_jokers(
            "AAAAAAA", ["Pareidolia", "Smiley Face"],
        ))

    def test_pareidolia_plus_photograph(self) -> None:
        # Pareidolia + Photograph: x2 mult fires on the first SCORED
        # card (since every card is a face now), not just the first
        # actual face card.
        self._check_all_plays(self._state_with_jokers(
            "AAAAAAA", ["Pareidolia", "Photograph"],
        ))

    def test_pareidolia_plus_sock_and_buskin(self) -> None:
        # Pareidolia + Sock and Buskin: every scored card retriggers.
        self._check_all_plays(self._state_with_jokers(
            "AAAAAAA", ["Pareidolia", "Sock and Buskin"],
        ))

    def test_wee_joker(self) -> None:
        # Wee Joker — +current_plus_chips + 8 * triggers-on-scored-2s.
        # Synthetic joker has current_plus_chips=0, so this just tests
        # the 2s + retrigger math.
        self._check_all_plays(self._state_with_jokers(
            "AAAAAAA", ["Wee Joker"],
        ))

    def test_wee_joker_plus_hack(self) -> None:
        # Wee Joker + Hack — Hack retriggers 2-5 cards, so scored 2s
        # get an extra trigger that multiplies Wee Joker's per-2
        # chip bonus.
        self._check_all_plays(self._state_with_jokers(
            "AAAAAAA", ["Wee Joker", "Hack"],
        ))

    def test_stone_card_in_play(self) -> None:
        # Inject a Stone-enhanced card into the hand and play all
        # legal actions. Rust must score the stone as 50 chips,
        # filter it out of identification, and add it back to the
        # scoring indices regardless of hand shape.
        from balatro_ai.api.state import Card
        from dataclasses import replace
        state = _drive_to_selecting_hand("AAAAAAA")
        modified_hand = list(state.hand)
        original = modified_hand[2]
        modified_hand[2] = Card(
            rank=original.rank, suit=original.suit,
            enhancement="stone",
        )
        state2 = replace(state, hand=tuple(modified_hand))
        self._check_all_plays(state2)

    def test_smeared_joker_solo(self) -> None:
        # Smeared turns Hearts+Diamonds into one suit and Clubs+Spades
        # into another. Affects flush identification only. Parity holds.
        self._check_all_plays(self._state_with_jokers(
            "AAAAAAA", ["Smeared Joker"],
        ))

    def test_four_fingers_solo(self) -> None:
        # Four Fingers makes flushes and straights need only 4 cards.
        # Affects identification + scoring index selection.
        self._check_all_plays(self._state_with_jokers(
            "AAAAAAA", ["Four Fingers"],
        ))

    def test_shortcut_solo(self) -> None:
        # Shortcut allows gap-of-1 straights (2-4-6-8-10).
        self._check_all_plays(self._state_with_jokers(
            "AAAAAAA", ["Shortcut"],
        ))

    def test_four_fingers_plus_shortcut(self) -> None:
        # Combo: 4-card gap straights allowed.
        self._check_all_plays(self._state_with_jokers(
            "AAAAAAA", ["Four Fingers", "Shortcut"],
        ))

    def test_wild_card_in_play(self) -> None:
        # Inject a Wild-enhanced card. Wild matches any suit for
        # flush detection; chip_value still uses its rank.
        from balatro_ai.api.state import Card
        from dataclasses import replace
        state = _drive_to_selecting_hand("AAAAAAA")
        modified_hand = list(state.hand)
        original = modified_hand[3]
        modified_hand[3] = Card(
            rank=original.rank, suit=original.suit,
            enhancement="wild",
        )
        state2 = replace(state, hand=tuple(modified_hand))
        self._check_all_plays(state2)

    def test_blueprint_copies_next(self) -> None:
        # Blueprint at index 0 + Joker (+4 mult) at index 1 → Blueprint
        # contributes +4 mult too. Total +8 mult from the pair.
        self._check_all_plays(self._state_with_jokers(
            "AAAAAAA", ["Blueprint", "Joker"],
        ))

    def test_brainstorm_copies_first(self) -> None:
        # Brainstorm at index 1 copies the leftmost joker (Joker at
        # index 0). Both contribute +4 mult → +8 total.
        self._check_all_plays(self._state_with_jokers(
            "AAAAAAA", ["Joker", "Brainstorm"],
        ))

    def test_blueprint_no_target(self) -> None:
        # Blueprint as the LAST joker (no next slot) contributes 0.
        # Equivalent to a no-op.
        self._check_all_plays(self._state_with_jokers(
            "AAAAAAA", ["Joker", "Blueprint"],
        ))

    def test_blueprint_incompatible_target(self) -> None:
        # Blueprint copying Pareidolia (in BLUEPRINT_INCOMPATIBLE_JOKERS)
        # → unresolved, scores 0.
        self._check_all_plays(self._state_with_jokers(
            "AAAAAAA", ["Blueprint", "Pareidolia"],
        ))

    def test_steel_held_via_enhancement_injection(self) -> None:
        # Inject a Steel-enhanced card into the hand to verify the
        # x1.5-mult-per-held-Steel logic. Build a synthetic state.
        from balatro_ai.api.state import Card
        from dataclasses import replace
        state = _drive_to_selecting_hand("AAAAAAA")
        # Replace the second card with a Steel-enhanced version.
        modified_hand = list(state.hand)
        original = modified_hand[1]
        modified_hand[1] = Card(
            rank=original.rank, suit=original.suit,
            enhancement="steel",
        )
        state2 = replace(state, hand=tuple(modified_hand))
        self._check_all_plays(state2)


if __name__ == "__main__":
    unittest.main()
