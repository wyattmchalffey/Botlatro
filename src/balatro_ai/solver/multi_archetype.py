"""Multi-archetype solver entry point (Milestone M6b).

`solve_seed_multi_archetype(seed)` runs the solver multiple times per
seed — once with no archetype (baseline) plus once per candidate
archetype — and returns the best trajectory. This guarantees the
solver quality floor is at least the no-archetype baseline, with
upside when one of the archetypes happens to fit the seed's available
items.

Why this matters: M6a confirmed that committing to a single archetype
soft-biases shop decisions in a way that often hurts when the archetype
doesn't fit the seed. Multi-archetype solves run all candidates and
keep only the best, side-stepping the "wrong commit" failure mode.

Compute cost: ~5× per seed (1 baseline + 4 archetypes). Without
adaptive budget this projects to >2 weeks for a 10k-seed dataset.
M6b ships the correctness baseline; M7 will add adaptive budget /
early termination / parallel-within-seed if throughput is a problem.

Status: M6b of `SOLVER_PLAN.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from balatro_ai.solver.archetypes import BUILT_IN_ARCHETYPES, Archetype
from balatro_ai.solver.policy import SolverPolicy
from balatro_ai.solver.trajectory import Trajectory, generate_trajectory


@dataclass(frozen=True, slots=True)
class ArchetypeRunResult:
    """One archetype-attempt result for a given seed."""

    archetype_name: str  # "baseline" for the no-archetype run.
    trajectory: Trajectory


@dataclass(frozen=True, slots=True)
class MultiArchetypeSolve:
    """All attempts and the best one selected."""

    seed: str
    attempts: tuple[ArchetypeRunResult, ...]
    best: ArchetypeRunResult

    @property
    def best_trajectory(self) -> Trajectory:
        return self.best.trajectory


def _trajectory_score(trajectory: Trajectory) -> tuple[int, int, int]:
    """Comparable key for selecting the best trajectory.

    Sort key (descending preference):
    1. Won runs first.
    2. Higher final ante next.
    3. Higher final score breaks ties.
    """

    return (
        1 if trajectory.won else 0,
        trajectory.final_ante,
        trajectory.final_score,
    )


def solve_seed_multi_archetype(
    seed: str,
    *,
    archetypes: Iterable[Archetype] | None = None,
    stake: str = "white",
    max_steps: int = 5000,
    record_steps: bool = True,
    policy_seed: int = 0,
    play_backend: str | None = None,
    play_depth: int | None = None,
    play_width: int | None = None,
) -> MultiArchetypeSolve:
    """Run the solver under each archetype + baseline, return the best.

    The "baseline" attempt uses ``SolverPolicy(archetype=None)``; each
    other attempt uses ``SolverPolicy(archetype=arch)``. The best is
    chosen by `_trajectory_score` (won > final_ante > final_score).

    Optional `play_backend` / `play_depth` / `play_width` override the
    SolverPolicy defaults for ALL attempts (baseline + each archetype).
    They're optional so existing callers continue to use SolverPolicy
    defaults.
    """

    candidate_archetypes = tuple(archetypes) if archetypes is not None else BUILT_IN_ARCHETYPES
    attempts: list[ArchetypeRunResult] = []

    def _make_policy(archetype: Archetype | None) -> SolverPolicy:
        kwargs: dict[str, object] = {"seed": policy_seed}
        if archetype is not None:
            kwargs["archetype"] = archetype
        if play_backend is not None:
            kwargs["play_backend"] = play_backend
        if play_depth is not None:
            kwargs["play_depth"] = play_depth
        if play_width is not None:
            kwargs["play_width"] = play_width
        return SolverPolicy(**kwargs)

    # Always include the baseline (no archetype) as a candidate so we
    # never produce a result strictly worse than the M5.5 baseline.
    baseline_policy = _make_policy(None)
    baseline_traj = generate_trajectory(
        seed, baseline_policy.choose_action,
        stake=stake, max_steps=max_steps, record_steps=record_steps,
    )
    attempts.append(ArchetypeRunResult(archetype_name="baseline", trajectory=baseline_traj))

    for arch in candidate_archetypes:
        policy = _make_policy(arch)
        traj = generate_trajectory(
            seed, policy.choose_action,
            stake=stake, max_steps=max_steps, record_steps=record_steps,
        )
        attempts.append(ArchetypeRunResult(archetype_name=arch.name, trajectory=traj))

    best = max(attempts, key=lambda attempt: _trajectory_score(attempt.trajectory))
    return MultiArchetypeSolve(seed=seed, attempts=tuple(attempts), best=best)
