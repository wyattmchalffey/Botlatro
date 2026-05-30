"""Per-seed worker entry point for the dataset multiprocessing pool.

The worker function `solve_seed` is the boundary between the CLI's
worker pool and the solver. It must be:

- **Pickleable.** `multiprocessing.Pool.imap_unordered` pickles the
  callable + args to ship them to each worker process. So `solve_seed`
  is a module-level function and its args are plain data
  (dataclasses, strings, ints) — no lambdas or closures.
- **Self-contained.** Each call creates its own policy instance
  inside the worker process so any per-policy caches (memoization,
  RNG) stay worker-local. This is also what lets us add Tier 1 #3's
  content-keyed memoization later without redesigning the worker
  boundary.
- **Failure-tolerant.** The worker catches BaseException and returns
  a `SeedResult` with `error_type`/`error_message` populated rather
  than re-raising. One bad seed should never kill the pool.

The CLI calls `solve_seed(seed, config)` per seed and writes the
returned `SeedResult` to the output JSONL.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass

from balatro_ai.dataset.schema import (
    ArchetypeAttemptRow,
    SeedResult,
    StepRow,
)


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    """Plain-data config for one worker run.

    Stays minimal on purpose — every field has to survive pickling.
    Policy construction inside the worker reads from these fields.

    Fields:
        policy_kind:    one of "v2", "legacy", "multi-archetype". Selects
                        which solver entry point to call.
        stake:          stake name ("white" default).
        max_steps:      hard cap on action count per trajectory.
        record_steps:   if True, the per-step record is included in
                        the output (~50KB per 100-step trajectory).
        play_depth:     SearchV2PlayPolicy depth (used for v2 + ma).
        play_width:     SearchV2PlayPolicy width (used for v2 + ma).
        leaf_kind:      "planning" (default) or "fast" — picks the
                        leaf evaluator for v2 / ma backends.
        timeout_seconds: per-seed wall clock budget. If exceeded the
                        worker returns a partial `SeedResult` with
                        `terminated_reason="TIMEOUT"`. (Note: enforced
                        by the POOL not the worker — the worker just
                        runs to completion.)
    """

    policy_kind: str = "v2"
    stake: str = "white"
    max_steps: int = 2000
    record_steps: bool = False
    play_depth: int = 3
    play_width: int = 1  # ~1.8x faster data-gen, keeps full depth-3 lookahead
    leaf_kind: str = "planning"
    timeout_seconds: float = 300.0


def solve_seed(seed: str, config: WorkerConfig) -> SeedResult:
    """Top-level worker. Returns a `SeedResult`; never raises."""

    try:
        return _solve_seed_impl(seed, config)
    except BaseException as exc:  # noqa: BLE001 — workers MUST not raise
        return SeedResult(
            seed=seed,
            stake=config.stake,
            policy=_policy_name(config),
            won=False,
            final_ante=0,
            final_score=0,
            final_money=0,
            n_steps=0,
            wall_seconds=0.0,
            terminated_reason="WORKER_ERROR",
            error_type=type(exc).__name__,
            error_message=f"{exc}\n{traceback.format_exc()[-2000:]}",
        )


def _solve_seed_impl(seed: str, config: WorkerConfig) -> SeedResult:
    """The actual work. Wrapped by `solve_seed` for error containment."""

    # Local imports so the parent process doesn't pay the heavy
    # solver import cost just to ship the worker function.
    from balatro_ai.solver.trajectory import generate_trajectory

    start = time.perf_counter()
    policy_name = _policy_name(config)

    if config.policy_kind == "multi-archetype":
        from balatro_ai.solver.multi_archetype import solve_seed_multi_archetype

        # `solve_seed_multi_archetype` builds one policy per archetype
        # internally; we pass our play knobs through as keyword
        # overrides so every attempt uses the same v2 depth/width.
        result = solve_seed_multi_archetype(
            seed,
            stake=config.stake,
            max_steps=config.max_steps,
            record_steps=config.record_steps,
            play_backend="v2",
            play_depth=config.play_depth,
            play_width=config.play_width,
        )
        best = result.best.trajectory
        attempts = tuple(
            ArchetypeAttemptRow(
                archetype_name=a.archetype_name,
                won=a.trajectory.won,
                final_ante=a.trajectory.final_ante,
                final_score=a.trajectory.final_score,
                final_money=a.trajectory.final_money,
                n_steps=a.trajectory.n_steps,
                wall_seconds=a.trajectory.wall_seconds,
                terminated_reason=a.trajectory.terminated_reason,
            )
            for a in result.attempts
        )
        steps = (
            tuple(_step_to_row(s) for s in best.steps)
            if config.record_steps
            else ()
        )
        return SeedResult(
            seed=seed,
            stake=config.stake,
            policy=policy_name,
            won=best.won,
            final_ante=best.final_ante,
            final_score=best.final_score,
            final_money=best.final_money,
            n_steps=best.n_steps,
            wall_seconds=time.perf_counter() - start,
            terminated_reason=best.terminated_reason,
            best_archetype=result.best.archetype_name,
            attempts=attempts,
            steps=steps,
        )

    # Single-policy paths (v2 / legacy).
    from balatro_ai.solver.policy import SolverPolicy

    policy = SolverPolicy(
        play_backend=config.policy_kind,
        play_depth=config.play_depth,
        play_width=config.play_width,
    )
    traj = generate_trajectory(
        seed,
        policy.choose_action,
        stake=config.stake,
        max_steps=config.max_steps,
        record_steps=config.record_steps,
    )
    steps = (
        tuple(_step_to_row(s) for s in traj.steps)
        if config.record_steps
        else ()
    )
    return SeedResult(
        seed=seed,
        stake=config.stake,
        policy=policy_name,
        won=traj.won,
        final_ante=traj.final_ante,
        final_score=traj.final_score,
        final_money=traj.final_money,
        n_steps=traj.n_steps,
        wall_seconds=time.perf_counter() - start,
        terminated_reason=traj.terminated_reason,
        steps=steps,
    )


def _policy_name(config: WorkerConfig) -> str:
    return f"{config.policy_kind}-d{config.play_depth}w{config.play_width}-{config.leaf_kind}"


def _step_to_row(step) -> StepRow:
    return StepRow(
        step=step.step,
        phase_before=step.phase_before,
        action_type=step.action_type,
        card_indices=tuple(step.card_indices),
        money_before=step.money_before,
        money_after=step.money_after,
        score_after=step.score_after,
        ante_after=step.ante_after,
        hands_after=step.hands_after,
        discards_after=step.discards_after,
    )
