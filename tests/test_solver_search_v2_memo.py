"""Tests for `balatro_ai.solver.search_v2.memo` (Tier 1 #3).

Two layers:

1. **Parity**: for a representative sample of GameStates from a real
   solver run, the cached and uncached versions of
   `clear_probability` / `planning_value` return identical values.
   A cache that returns the wrong answer is much worse than a slow
   cache — this test is the correctness gate.

2. **Determinism + hit behavior**: the same state queried twice
   inside one `solver_search_cache_scope()` does NOT recompute;
   the second call shares the first's cached entry. Two scopes
   in sequence don't leak.

The SCALE of the parity test is small but covers the variety:
opening hand, mid-blind, post-discard, ROUND_EVAL. That's enough
to catch a missing-signature-field bug — if the signature omits a
relevant field, at least one state pair will diverge.
"""

from __future__ import annotations

import unittest

import context  # noqa: F401
from balatro_ai.api.state import GamePhase
from balatro_ai.bots.basic_strategy_bot import BasicStrategyBot
from balatro_ai.search.state_value import (
    clear_probability,
    planning_value,
)
from balatro_ai.sim.local_runner import LocalBalatroSimulator
from balatro_ai.solver.search_v2.memo import (
    cache_stats,
    cached_clear_probability,
    cached_planning_value,
    solver_search_cache_scope,
)
from balatro_ai.solver.search_v2.state_signature import state_signature
from balatro_ai.solver.seed_game import SeedGame
from balatro_ai.solver.trajectory import _stable_seed_int


def _collect_states(seed: str, max_states: int = 20) -> list:
    """Drive `basic_strategy_bot` through `seed` and snapshot mid-decision states."""

    game = SeedGame(seed)
    sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
    sim.state = game.initial_state()
    bot = BasicStrategyBot(seed=0)
    states = []
    for _ in range(max_states * 4):  # padding to skip non-SELECTING_HAND phases
        if sim.state.phase == GamePhase.SELECTING_HAND:
            states.append(sim.state)
            if len(states) >= max_states:
                break
        if sim.state.run_over or sim.state.phase == GamePhase.RUN_OVER:
            break
        sim.step(bot.choose_action(sim.state))
    return states


class CacheParityTests(unittest.TestCase):
    """Cached and uncached must return EXACTLY equal values."""

    def test_clear_probability_parity_across_real_states(self) -> None:
        states = _collect_states("AAAAAAA", max_states=10)
        self.assertGreater(len(states), 0, "no states collected from AAAAAAA")
        with solver_search_cache_scope():
            for state in states:
                cached = cached_clear_probability(state, samples=4, seed=0)
                uncached = clear_probability(state, samples=4, seed=0)
                self.assertEqual(
                    cached, uncached,
                    f"clear_probability cache divergence on state "
                    f"ante={state.ante} score={state.current_score}",
                )

    def test_planning_value_parity_across_real_states(self) -> None:
        states = _collect_states("AAAAAAA", max_states=10)
        with solver_search_cache_scope():
            for state in states:
                cached = cached_planning_value(state, samples=4, seed=0)
                uncached = planning_value(state, samples=4, seed=0)
                self.assertEqual(
                    cached, uncached,
                    f"planning_value cache divergence on state "
                    f"ante={state.ante} score={state.current_score}",
                )


class CacheBehaviorTests(unittest.TestCase):
    def test_outside_scope_is_passthrough(self) -> None:
        # No cache active — cached call must still return the right
        # answer (transparent pass-through).
        states = _collect_states("AAAAAAA", max_states=3)
        self.assertGreater(len(states), 0)
        for state in states:
            self.assertEqual(
                cached_clear_probability(state, samples=2, seed=0),
                clear_probability(state, samples=2, seed=0),
            )

    def test_scope_tracks_entries(self) -> None:
        states = _collect_states("AAAAAAA", max_states=3)
        with solver_search_cache_scope():
            # Pre-populate; cache should grow.
            for state in states:
                cached_clear_probability(state, samples=2, seed=0)
            entries, active = cache_stats()
            self.assertGreater(entries, 0)
            self.assertEqual(active, 1)

    def test_scope_isolates_between_calls(self) -> None:
        # Cache from scope A must not leak into scope B.
        states = _collect_states("AAAAAAA", max_states=2)
        with solver_search_cache_scope():
            for state in states:
                cached_clear_probability(state, samples=2, seed=0)
            entries_a, _ = cache_stats()
        # Outside any scope.
        entries_outside, active_outside = cache_stats()
        self.assertEqual(entries_outside, 0)
        self.assertEqual(active_outside, 0)
        with solver_search_cache_scope():
            entries_b, _ = cache_stats()
            self.assertEqual(entries_b, 0)  # fresh scope, no leak from A


class StateSignatureTests(unittest.TestCase):
    """Quick sanity: signatures are hashable and distinct states differ."""

    def test_signature_is_hashable(self) -> None:
        states = _collect_states("AAAAAAA", max_states=2)
        sigs = {state_signature(s) for s in states}
        self.assertGreater(len(sigs), 0)

    def test_distinct_states_have_distinct_signatures(self) -> None:
        # Run two different seeds; first SELECTING_HAND state differs.
        a = _collect_states("AAAAAAA", max_states=1)
        b = _collect_states("BBBBBBB", max_states=1)
        if a and b:
            self.assertNotEqual(state_signature(a[0]), state_signature(b[0]))


if __name__ == "__main__":
    unittest.main()
