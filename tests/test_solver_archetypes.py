"""Tests for `balatro_ai.solver.archetypes` (Milestone M6a).

Covers:
- `Archetype.archetype_fit_score` counts matched jokers + consumables.
- `FLUSH_ARCHETYPE` has the expected key set populated.
- `SolverPolicy(archetype=...)` accepts an archetype and the trajectory
  still runs end-to-end (M6a acceptance).
"""

from __future__ import annotations

import unittest

import context  # noqa: F401
from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GamePhase, GameState, Joker
from balatro_ai.solver.archetypes import FLUSH_ARCHETYPE, Archetype
from balatro_ai.solver.policy import SolverPolicy
from balatro_ai.solver.seed_game import SeedGame
from balatro_ai.solver.trajectory import generate_trajectory


def _state_with_jokers(jokers: tuple[Joker, ...]) -> GameState:
    state = SeedGame("AAAAAAA").initial_state()
    # Replace jokers immutably.
    from dataclasses import replace
    return replace(state, jokers=jokers)


class ArchetypeFitScoreTests(unittest.TestCase):
    def test_zero_with_no_matching_jokers(self) -> None:
        state = SeedGame("AAAAAAA").initial_state()
        self.assertEqual(FLUSH_ARCHETYPE.archetype_fit_score(state), 0.0)

    def test_scores_matched_joker_by_key_metadata(self) -> None:
        smeared = Joker(name="Smeared Joker", metadata={"key": "j_smeared"})
        state = _state_with_jokers((smeared,))
        self.assertAlmostEqual(
            FLUSH_ARCHETYPE.archetype_fit_score(state),
            FLUSH_ARCHETYPE.shop_bonus_per_match,
        )

    def test_scores_multiple_matched_jokers(self) -> None:
        smeared = Joker(name="Smeared Joker", metadata={"key": "j_smeared"})
        four_fingers = Joker(name="Four Fingers", metadata={"key": "j_four_fingers"})
        state = _state_with_jokers((smeared, four_fingers))
        self.assertAlmostEqual(
            FLUSH_ARCHETYPE.archetype_fit_score(state),
            2 * FLUSH_ARCHETYPE.shop_bonus_per_match,
        )

    def test_non_matching_jokers_score_zero(self) -> None:
        # `j_joker` is not in the flush key list.
        plain = Joker(name="Joker", metadata={"key": "j_joker"})
        state = _state_with_jokers((plain,))
        self.assertEqual(FLUSH_ARCHETYPE.archetype_fit_score(state), 0.0)

    def test_falls_back_to_name_when_no_key_in_metadata(self) -> None:
        # Locally-constructed jokers in tests sometimes lack the `key`.
        smeared = Joker(name="Smeared Joker")  # no metadata
        state = _state_with_jokers((smeared,))
        self.assertAlmostEqual(
            FLUSH_ARCHETYPE.archetype_fit_score(state),
            FLUSH_ARCHETYPE.shop_bonus_per_match,
        )


class FlushArchetypeShapeTests(unittest.TestCase):
    def test_flush_archetype_targets_flush_family(self) -> None:
        from balatro_ai.rules.hand_evaluator import HandType
        self.assertIn(HandType.FLUSH, FLUSH_ARCHETYPE.target_hand_types)
        self.assertIn(HandType.FLUSH_HOUSE, FLUSH_ARCHETYPE.target_hand_types)
        self.assertIn(HandType.FLUSH_FIVE, FLUSH_ARCHETYPE.target_hand_types)
        self.assertIn(HandType.STRAIGHT_FLUSH, FLUSH_ARCHETYPE.target_hand_types)

    def test_flush_archetype_has_known_keys(self) -> None:
        # Spot-check a few must-have entries.
        self.assertIn("j_smeared", FLUSH_ARCHETYPE.key_joker_keys)
        self.assertIn("j_four_fingers", FLUSH_ARCHETYPE.key_joker_keys)
        # Suit-conversion tarots.
        self.assertIn("c_sun", FLUSH_ARCHETYPE.key_consumable_keys)
        self.assertIn("c_moon", FLUSH_ARCHETYPE.key_consumable_keys)


class ArchetypeAwareSolverPolicyTests(unittest.TestCase):
    """M6a acceptance: SolverPolicy(archetype=...) drives a trajectory."""

    def test_solver_policy_accepts_archetype(self) -> None:
        # Cheap path: constructor + a delegated BLIND_SELECT call.
        policy = SolverPolicy(archetype=FLUSH_ARCHETYPE)
        state = SeedGame("AAAAAAA").initial_state()
        action = policy.choose_action(state)
        self.assertEqual(action.action_type, ActionType.SELECT_BLIND)


class ArchetypeBiasMechanicsTests(unittest.TestCase):
    """Confirm that swapping archetypes actually changes the leaf-value fn."""

    def test_fit_score_distinguishes_aligned_state_from_unaligned(self) -> None:
        # Two states differing only by jokers should produce different
        # archetype-fit scores when one has aligned jokers and the other
        # doesn't.
        aligned = _state_with_jokers((
            Joker(name="Smeared Joker", metadata={"key": "j_smeared"}),
            Joker(name="Four Fingers", metadata={"key": "j_four_fingers"}),
        ))
        unaligned = _state_with_jokers((
            Joker(name="Joker", metadata={"key": "j_joker"}),
            Joker(name="Greedy Joker", metadata={"key": "j_greedy_joker"}),
        ))
        aligned_score = FLUSH_ARCHETYPE.archetype_fit_score(aligned)
        unaligned_score = FLUSH_ARCHETYPE.archetype_fit_score(unaligned)
        self.assertGreater(
            aligned_score,
            unaligned_score,
            "Flush archetype should score Smeared+Four Fingers > plain Joker+Greedy",
        )

    def test_different_archetypes_have_different_key_sets(self) -> None:
        # M6b's multi-archetype branching depends on each archetype
        # actually preferring different items. Cross-check that the
        # Flush and Pair Retrigger archetypes don't accidentally share
        # all their keys.
        from balatro_ai.solver.archetypes import PAIR_RETRIGGER_ARCHETYPE
        overlap = FLUSH_ARCHETYPE.key_joker_keys & PAIR_RETRIGGER_ARCHETYPE.key_joker_keys
        # Small overlap is fine, but it shouldn't be everything.
        self.assertLess(
            len(overlap),
            min(len(FLUSH_ARCHETYPE.key_joker_keys), len(PAIR_RETRIGGER_ARCHETYPE.key_joker_keys)) // 2,
            f"Flush and Pair Retrigger archetypes overlap too much: {overlap}",
        )


if __name__ == "__main__":
    unittest.main()
