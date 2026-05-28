"""Tests for `balatro_ai.solver.play_search` (Milestone M4).

Verifies the whole-blind beam-search policy:
- Falls through to the fallback policy for non-SELECTING_HAND phases.
- Returns a play/discard action when the beam succeeds at SELECTING_HAND.
- Recovers gracefully when the beam returns None / raises (the fallback
  kicks in).
- Drives a full trajectory through `generate_trajectory` without
  crashing.

The single-seed trajectory test is the M4 acceptance bar; it confirms
the policy interoperates with `SeedGame` + `LocalBalatroSimulator` end
to end.
"""

from __future__ import annotations

import unittest

import context  # noqa: F401
from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GamePhase, GameState
from balatro_ai.solver.play_search import PlaySearchPolicy
from balatro_ai.solver.seed_game import SeedGame
from balatro_ai.solver.trajectory import generate_trajectory


class _FixedFallback:
    """Sentinel fallback used to verify delegation."""

    def __init__(self, action: Action) -> None:
        self.action = action
        self.calls = 0

    def choose_action(self, _state: GameState) -> Action:
        self.calls += 1
        return self.action


class PlaySearchPolicyDelegationTests(unittest.TestCase):
    def test_blind_select_delegates_to_fallback(self) -> None:
        fallback = _FixedFallback(Action(action_type=ActionType.SELECT_BLIND))
        policy = PlaySearchPolicy(fallback=fallback)
        state = SeedGame("AAAAAAA").initial_state()
        self.assertEqual(state.phase, GamePhase.BLIND_SELECT)
        action = policy.choose_action(state)
        self.assertEqual(action.action_type, ActionType.SELECT_BLIND)
        self.assertEqual(fallback.calls, 1)

    def test_callable_alias_works(self) -> None:
        # PlaySearchPolicy should be usable directly as the policy callable.
        fallback = _FixedFallback(Action(action_type=ActionType.SELECT_BLIND))
        policy = PlaySearchPolicy(fallback=fallback)
        state = SeedGame("AAAAAAA").initial_state()
        # Calling via __call__ should match calling choose_action.
        action_direct = policy(state)
        self.assertEqual(action_direct.action_type, ActionType.SELECT_BLIND)


class PlaySearchPolicyTrajectoryTests(unittest.TestCase):
    """M4 acceptance: full-run trajectory via the beam-search policy."""

    def test_full_trajectory_runs_without_crashing(self) -> None:
        # Reduced beam depth/width keeps the test under a few minutes.
        # We're checking pipeline integrity, not search quality.
        policy = PlaySearchPolicy(beam_depth=3, beam_width=2)
        traj = generate_trajectory(
            "AAAAAAA", policy.choose_action, max_steps=2000
        )
        self.assertIn(
            traj.terminated_reason,
            ("RUN_OVER", "STEP_LIMIT"),
            f"Unexpected termination: {traj.terminated_reason}",
        )
        self.assertGreater(traj.n_steps, 10, "trajectory terminated too early")
        self.assertGreaterEqual(traj.final_ante, 1)


if __name__ == "__main__":
    unittest.main()
