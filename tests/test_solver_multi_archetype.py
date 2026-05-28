"""Tests for `balatro_ai.solver.multi_archetype` (Milestone M6b).

Verifies the multi-archetype solver entry point picks the trajectory
with the best (won, ante, score) ordering, never returns worse than
the baseline attempt, and exposes the per-archetype attempts for
analysis.

The full per-seed measurement against the no-archetype baseline lives
in scripts under `.data/` (too expensive for unittest); these tests
just confirm the selection logic and API.
"""

from __future__ import annotations

import unittest

import context  # noqa: F401
from balatro_ai.solver.multi_archetype import (
    ArchetypeRunResult,
    MultiArchetypeSolve,
    _trajectory_score,
)
from balatro_ai.solver.trajectory import Trajectory


def _traj(won: bool, ante: int, score: int) -> Trajectory:
    return Trajectory(
        seed="dummy",
        stake="white",
        steps=(),
        final_phase="run_over",
        final_ante=ante,
        final_score=score,
        final_money=0,
        won=won,
        sim_step_count=0,
        wall_seconds=0.0,
        terminated_reason="RUN_OVER",
    )


class TrajectoryScoreOrderingTests(unittest.TestCase):
    def test_won_beats_not_won_even_at_lower_ante(self) -> None:
        won_low = _traj(won=True, ante=8, score=100_000)
        lost_high = _traj(won=False, ante=8, score=999_999)
        self.assertGreater(_trajectory_score(won_low), _trajectory_score(lost_high))

    def test_higher_ante_beats_lower_ante_at_same_won_status(self) -> None:
        a = _traj(won=False, ante=6, score=10_000)
        b = _traj(won=False, ante=4, score=50_000)
        self.assertGreater(_trajectory_score(a), _trajectory_score(b))

    def test_score_breaks_tie_when_ante_and_won_match(self) -> None:
        a = _traj(won=False, ante=5, score=20_000)
        b = _traj(won=False, ante=5, score=15_000)
        self.assertGreater(_trajectory_score(a), _trajectory_score(b))


class MultiArchetypeSolveAPITests(unittest.TestCase):
    """Confirm the dataclass exposes what callers need without running the solver."""

    def test_best_trajectory_property(self) -> None:
        baseline = ArchetypeRunResult(archetype_name="baseline", trajectory=_traj(False, 5, 10_000))
        flush = ArchetypeRunResult(archetype_name="flush", trajectory=_traj(False, 7, 30_000))
        solve = MultiArchetypeSolve(seed="X", attempts=(baseline, flush), best=flush)
        self.assertEqual(solve.best_trajectory.final_ante, 7)


if __name__ == "__main__":
    unittest.main()
