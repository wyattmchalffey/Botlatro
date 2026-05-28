"""Trajectory generation (Milestone M2, hardened in M6).

Drives a `SeedGame`'s initial state forward via a `policy` callable and
records every (state-summary, action) pair until the run ends. The first
policy supported is `basic_strategy_bot.choose_action`; the result is a
`Trajectory` record that downstream Phase 8 imitation training can
consume.

For M2 (throughput baseline) we delegate action dispatch to
`LocalBalatroSimulator`, which already implements the full
play/discard/shop/pack/blind-select state machine. The simulator uses its
own Python-RNG for shop and pack sampling — that's NOT seed-faithful, but
M2's goal is to measure compute cost per decision, not to validate
trajectory contents against the live game. M4+ swaps in seed-faithful
shop predictions for the actual dataset run.

Status: M2 of `SOLVER_PLAN.md`.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GamePhase, GameState
from balatro_ai.sim.local_runner import LocalBalatroSimulator
from balatro_ai.solver.seed_game import SeedGame


Policy = Callable[[GameState], Action]


def _stable_seed_int(seed: str) -> int:
    """Cross-process stable integer derived from a seed string.

    Used to seed `LocalBalatroSimulator`'s internal Python `Random` so
    the same seed string produces the same shop samples every time the
    solver runs, regardless of Python process. Built-in `hash()` is
    randomized per process and can't be used here.

    Uses the first 4 bytes of MD5 — collisions are fine (we just need
    diverse but reproducible seeds), MD5 is fast and stdlib, and we
    don't care about cryptographic strength.
    """

    digest = hashlib.md5(seed.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


@dataclass(frozen=True, slots=True)
class StepRecord:
    """One step of a trajectory: the action taken plus the post-state summary."""

    step: int
    phase_before: str
    action_type: str
    card_indices: tuple[int, ...]
    money_before: int
    money_after: int
    score_after: int
    ante_after: int
    hands_after: int
    discards_after: int


@dataclass(frozen=True, slots=True)
class Trajectory:
    """A full run trajectory: ordered actions plus the final outcome."""

    seed: str
    stake: str
    steps: tuple[StepRecord, ...]
    final_phase: str
    final_ante: int
    final_score: int
    final_money: int
    won: bool
    sim_step_count: int
    wall_seconds: float
    terminated_reason: str  # "RUN_OVER", "STEP_LIMIT", "STUCK", "POLICY_ERROR"

    @property
    def n_steps(self) -> int:
        return len(self.steps)


def _record_step(
    step: int,
    state_before: GameState,
    action: Action,
    state_after: GameState,
) -> StepRecord:
    return StepRecord(
        step=step,
        phase_before=state_before.phase.value,
        action_type=action.action_type.value,
        card_indices=tuple(action.card_indices),
        money_before=state_before.money,
        money_after=state_after.money,
        score_after=state_after.current_score,
        ante_after=state_after.ante,
        hands_after=state_after.hands_remaining,
        discards_after=state_after.discards_remaining,
    )


def generate_trajectory(
    seed: str,
    policy: Policy,
    *,
    stake: str = "white",
    max_steps: int = 5000,
    record_steps: bool = True,
) -> Trajectory:
    """Drive `policy` through a single run starting from `SeedGame(seed)`.

    Returns a `Trajectory` even when the run ends abnormally. The
    `terminated_reason` field tells you why the loop stopped.

    Parameters
    ----------
    seed:        the run seed string (e.g. "AAAAAAA").
    policy:      a callable `GameState -> Action`. Typical: `bot.choose_action`.
    stake:       stake name, defaults to "white".
    max_steps:   hard cap on action count; protects against catastrophic
                 loops while we're still validating the pipeline.
    record_steps: when False, skip per-step record building (faster, for
                 throughput-only runs). The returned trajectory then has
                 an empty `steps` tuple.
    """

    # We use LocalBalatroSimulator's action dispatch but override its
    # initial state with the seed-faithful one from SeedGame. The
    # simulator's internal _rng still drives shop/pack sampling for M2;
    # that's documented as a known not-yet-seed-faithful gap.
    game = SeedGame(seed, stake=stake)
    initial_state = game.initial_state()

    # LocalBalatroSimulator wants an integer seed for its Random; we
    # derive it from a STABLE hash of the seed string so the same string
    # produces the same int across Python processes. Python's built-in
    # hash() is randomized per process (PYTHONHASHSEED), which causes
    # cross-process trajectory divergence — caught in the M6 variance
    # check after spurious archetype regressions turned out to be
    # cross-process noise from this exact bug.
    int_seed = _stable_seed_int(seed)
    # Opt-in seed-faithful shop sourcing (BALATRO_SEED_FAITHFUL=1):
    # pass the original seed STRING so the simulator can source the
    # first shop from the validated rng/ predictors. Default off keeps
    # the legacy generic-Random shop path.
    balatro_seed = seed if os.environ.get("BALATRO_SEED_FAITHFUL") == "1" else None
    sim = LocalBalatroSimulator(seed=int_seed, stake=stake, balatro_seed=balatro_seed)
    sim.state = initial_state

    steps: list[StepRecord] = []
    start = time.perf_counter()
    reason = "STEP_LIMIT"
    last_state_signature: tuple | None = None
    stuck_repeats = 0

    for step_index in range(max_steps):
        state = sim.state
        if state.phase == GamePhase.RUN_OVER or state.run_over:
            reason = "RUN_OVER"
            break

        # Stuck detection: same state signature N times in a row → bail.
        # The signature has to be specific enough that any legitimate action
        # changes at least one field, otherwise we'll bail mid-blind on a
        # legitimate discard sequence. Discards keep money/score/hands_remaining
        # constant but change discards_remaining + hand contents; shop browsing
        # changes money/jokers; etc. Tuple of cheap-to-hash fields covers all
        # the action types this policy actually emits.
        signature = (
            state.phase,
            state.ante,
            state.current_score,
            state.money,
            state.hands_remaining,
            state.discards_remaining,
            state.deck_size,
            len(state.hand),
            len(state.jokers),
            len(state.consumables),
        )
        if signature == last_state_signature:
            stuck_repeats += 1
            # 10 in a row is genuinely stuck — most legitimate same-signature
            # sequences clear within a few steps.
            if stuck_repeats >= 10:
                reason = "STUCK"
                break
        else:
            stuck_repeats = 0
        last_state_signature = signature

        try:
            action = policy(state)
        except Exception as exc:  # noqa: BLE001 — policy errors are recorded, not raised
            reason = f"POLICY_ERROR: {type(exc).__name__}: {exc}"
            break

        if action.action_type == ActionType.NO_OP:
            reason = "POLICY_NOOP"
            break

        next_state = sim.step(action)
        if record_steps:
            steps.append(_record_step(step_index, state, action, next_state))

    elapsed = time.perf_counter() - start
    final = sim.state
    return Trajectory(
        seed=seed,
        stake=stake,
        steps=tuple(steps),
        final_phase=final.phase.value,
        final_ante=final.ante,
        final_score=final.current_score,
        final_money=final.money,
        won=bool(final.won),
        sim_step_count=step_index + 1 if reason != "STEP_LIMIT" else max_steps,
        wall_seconds=elapsed,
        terminated_reason=reason,
    )
