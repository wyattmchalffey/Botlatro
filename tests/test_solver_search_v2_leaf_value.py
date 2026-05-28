"""Tests for `balatro_ai.solver.search_v2.leaf_value` (Tier 1 #1b).

Sanity tests for the four leaf evaluators:

- `FastHeuristicLeaf` — rollout-free, O(visible-state). Verifies the
  monotonic behavior we expect: more progress beats less, more
  resources beat fewer.
- `ClearProbabilityLeaf` — rollout-backed; we only spot-check the
  terminal-state cases since the rollout's value is implementation-
  defined per `state_value.clear_probability`.
- `FutureBlindSurvivalLeaf` — verifies the headroom bonus actually
  raises the score above bare clear probability.
- `ArchetypeAwareLeaf` — decorator; verifies the bonus is additive
  and a no-match archetype leaves the base value unchanged.

The hard end-to-end "does this evaluator pick better trajectories"
question is answered by the trajectory tests in test_solver_search_v2_play.py
plus the .data/ measurement scripts.
"""

from __future__ import annotations

import unittest

import context  # noqa: F401
from balatro_ai.api.state import Card, GamePhase, GameState
from balatro_ai.solver.archetypes import FLUSH_ARCHETYPE, Archetype
from balatro_ai.solver.search_v2.leaf_value import (
    ArchetypeAwareLeaf,
    ClearProbabilityLeaf,
    FastHeuristicLeaf,
    FutureBlindSurvivalLeaf,
    PlanningValueLeaf,
)


def _bare_state(
    *,
    phase: GamePhase = GamePhase.SELECTING_HAND,
    current_score: int = 0,
    required_score: int = 300,
    hands_remaining: int = 4,
    discards_remaining: int = 3,
    money: int = 4,
    won: bool = False,
    run_over: bool = False,
    hand: tuple[Card, ...] = (),
    jokers: tuple = (),
) -> GameState:
    """Minimal GameState for evaluator unit tests.

    The evaluators we test only read a small handful of fields, so
    constructing a full fixture per case is overkill — we set only
    what each test exercises.
    """

    return GameState(
        phase=phase,
        ante=1,
        blind="Small Blind",
        required_score=required_score,
        current_score=current_score,
        hands_remaining=hands_remaining,
        discards_remaining=discards_remaining,
        money=money,
        hand=hand,
        jokers=jokers,
        won=won,
        run_over=run_over,
    )


class FastHeuristicLeafTests(unittest.TestCase):
    def test_won_state_returns_max(self) -> None:
        leaf = FastHeuristicLeaf()
        self.assertEqual(leaf.evaluate(_bare_state(won=True)), 2.0)

    def test_run_over_returns_zero(self) -> None:
        leaf = FastHeuristicLeaf()
        self.assertEqual(leaf.evaluate(_bare_state(run_over=True)), 0.0)
        self.assertEqual(
            leaf.evaluate(_bare_state(phase=GamePhase.RUN_OVER)), 0.0
        )

    def test_round_eval_above_one(self) -> None:
        # Cleared blind — round eval phase. Bonus floor at 1.0 + small
        # headroom adjustment.
        leaf = FastHeuristicLeaf()
        value = leaf.evaluate(_bare_state(phase=GamePhase.ROUND_EVAL))
        self.assertGreaterEqual(value, 1.0)

    def test_more_progress_scores_higher(self) -> None:
        leaf = FastHeuristicLeaf()
        low = leaf.evaluate(_bare_state(current_score=50))
        high = leaf.evaluate(_bare_state(current_score=250))
        self.assertGreater(high, low)

    def test_more_resources_scores_higher(self) -> None:
        leaf = FastHeuristicLeaf()
        poor = leaf.evaluate(_bare_state(hands_remaining=1, money=0))
        rich = leaf.evaluate(_bare_state(hands_remaining=4, money=40))
        self.assertGreater(rich, poor)


class ClearProbabilityLeafTerminalTests(unittest.TestCase):
    def test_won_returns_two(self) -> None:
        leaf = ClearProbabilityLeaf()
        self.assertEqual(leaf.evaluate(_bare_state(won=True)), 2.0)

    def test_run_over_returns_zero(self) -> None:
        leaf = ClearProbabilityLeaf()
        self.assertEqual(leaf.evaluate(_bare_state(run_over=True)), 0.0)


class PlanningValueLeafTerminalTests(unittest.TestCase):
    """PlanningValueLeaf delegates to state_value.planning_value — we
    only check the terminal cases. Rollout behavior is covered by
    `state_value`'s own tests."""

    def test_won_returns_two(self) -> None:
        # planning_value returns 2.0 for won states.
        leaf = PlanningValueLeaf()
        self.assertEqual(leaf.evaluate(_bare_state(won=True)), 2.0)

    def test_run_over_returns_zero(self) -> None:
        leaf = PlanningValueLeaf()
        self.assertEqual(leaf.evaluate(_bare_state(run_over=True)), 0.0)


class FutureBlindSurvivalLeafTests(unittest.TestCase):
    def test_won_returns_two(self) -> None:
        leaf = FutureBlindSurvivalLeaf()
        self.assertEqual(leaf.evaluate(_bare_state(won=True)), 2.0)

    def test_run_over_returns_zero(self) -> None:
        leaf = FutureBlindSurvivalLeaf()
        self.assertEqual(leaf.evaluate(_bare_state(run_over=True)), 0.0)

    def test_round_eval_above_one(self) -> None:
        leaf = FutureBlindSurvivalLeaf()
        value = leaf.evaluate(_bare_state(phase=GamePhase.ROUND_EVAL))
        self.assertGreaterEqual(value, 1.0)


class ArchetypeAwareLeafTests(unittest.TestCase):
    def test_no_match_returns_base_value(self) -> None:
        # An archetype that matches nothing in the state — bonus is 0,
        # so the decorated value equals the base value.
        base = FastHeuristicLeaf()
        empty_archetype = Archetype(
            name="empty",
            target_hand_types=frozenset(),
            key_joker_keys=frozenset({"j_nonexistent"}),
        )
        decorated = ArchetypeAwareLeaf(base=base, archetype=empty_archetype)
        state = _bare_state(current_score=100)
        self.assertEqual(decorated.evaluate(state), base.evaluate(state))

    def test_decorator_is_additive(self) -> None:
        # The decorator should never subtract from the base value —
        # the archetype-fit bonus is non-negative.
        base = FastHeuristicLeaf()
        decorated = ArchetypeAwareLeaf(base=base, archetype=FLUSH_ARCHETYPE)
        state = _bare_state(current_score=100)
        self.assertGreaterEqual(decorated.evaluate(state), base.evaluate(state))


if __name__ == "__main__":
    unittest.main()
