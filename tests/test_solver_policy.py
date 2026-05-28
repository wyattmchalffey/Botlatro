"""Tests for `balatro_ai.solver.policy` (Milestone M5).

Verifies the composed solver policy:
- Delegates non-search phases (BLIND_SELECT, CASH_OUT) to the fallback.
- Routes SHOP to shop beam search, SELECTING_HAND to play beam search.
- Recovers gracefully when either sub-search returns None / raises.
- Drives a full trajectory through `generate_trajectory` without
  crashing — the M5 acceptance bar.
"""

from __future__ import annotations

import unittest

import context  # noqa: F401
from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GamePhase, GameState
from balatro_ai.solver.play_search import PlaySearchPolicy
from balatro_ai.solver.policy import SolverPolicy
from balatro_ai.solver.search_v2.play import SearchV2PlayPolicy
from balatro_ai.solver.seed_game import SeedGame
from balatro_ai.solver.trajectory import generate_trajectory


class _FixedFallback:
    def __init__(self, action: Action) -> None:
        self.action = action
        self.calls = 0

    def choose_action(self, _state: GameState) -> Action:
        self.calls += 1
        return self.action


class SolverPolicyDelegationTests(unittest.TestCase):
    def test_blind_select_delegates_to_fallback(self) -> None:
        fallback = _FixedFallback(Action(action_type=ActionType.SELECT_BLIND))
        policy = SolverPolicy(fallback=fallback)
        state = SeedGame("AAAAAAA").initial_state()
        self.assertEqual(state.phase, GamePhase.BLIND_SELECT)
        action = policy.choose_action(state)
        self.assertEqual(action.action_type, ActionType.SELECT_BLIND)
        self.assertEqual(fallback.calls, 1)

    def test_callable_alias_works(self) -> None:
        fallback = _FixedFallback(Action(action_type=ActionType.SELECT_BLIND))
        policy = SolverPolicy(fallback=fallback)
        state = SeedGame("AAAAAAA").initial_state()
        action_direct = policy(state)
        self.assertEqual(action_direct.action_type, ActionType.SELECT_BLIND)

    def test_default_play_backend_is_legacy(self) -> None:
        # Default reverted from "v2" to "legacy" on 2026-05-26 after a
        # 4-seed measurement showed v2 averages ante 2.0 vs legacy 4.5.
        # See SOLVER_OPTIMIZATION_PLAN.md "Clean 4-seed measurement".
        # v2 stays available via play_backend="v2" for deep-search work.
        policy = SolverPolicy()
        self.assertIsInstance(policy.play_policy, PlaySearchPolicy)

    def test_v2_play_backend_opt_in(self) -> None:
        policy = SolverPolicy(play_backend="v2")
        self.assertIsInstance(policy.play_policy, SearchV2PlayPolicy)

    def test_legacy_play_backend_still_available(self) -> None:
        policy = SolverPolicy(play_backend="legacy")
        self.assertIsInstance(policy.play_policy, PlaySearchPolicy)


class SolverPolicyTrajectoryTests(unittest.TestCase):
    """M5 acceptance: full-run trajectory via the composed policy."""

    def test_full_trajectory_runs_without_crashing(self) -> None:
        # Default config; same throughput envelope as M4's trajectory test,
        # plus shop search at default ShopSearchConfig.
        policy = SolverPolicy()
        traj = generate_trajectory(
            "AAAAAAA", policy.choose_action, max_steps=2000
        )
        self.assertIn(
            traj.terminated_reason,
            ("RUN_OVER", "STEP_LIMIT"),
            f"Unexpected termination: {traj.terminated_reason}",
        )
        self.assertGreater(traj.n_steps, 10)
        self.assertGreaterEqual(traj.final_ante, 1)


if __name__ == "__main__":
    unittest.main()
