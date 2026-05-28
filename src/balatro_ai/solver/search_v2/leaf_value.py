"""Leaf evaluators for the solver beam (Tier 1 #1b).

Three evaluators ship in this module:

- `ClearProbabilityLeaf` — wraps the existing `state_value.clear_probability`
  with a configurable sample count. The default sample count (4) is
  deliberately lower than `planning_value`'s 16 because the solver
  beam visits many more leaves per decision and the marginal precision
  from extra samples isn't worth the cost.

- `FutureBlindSurvivalLeaf` — projects survival across the current
  blind AND the next one using `headroom_value` as a build-strength
  proxy. Rewards leaves that not only clear the current blind but
  also leave the run in a state that's likely to clear future blinds.

- `ArchetypeAwareLeaf` — decorator that adds the archetype-fit bonus
  on top of any base evaluator. Used by the multi-archetype solver
  when running an archetype attempt.

All three implement the `LeafEvaluator` protocol: a single
`evaluate(state) -> float` method. Higher is better; range is
roughly [0, 4] for the base evaluators but no hard upper bound.

See `SOLVER_OPTIMIZATION_PLAN.md` §3 #1b for the design rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from balatro_ai.api.state import GamePhase, GameState
from balatro_ai.search.state_value import (
    clear_probability,
    headroom_value,
    planning_value,
)


class LeafEvaluator(Protocol):
    """Protocol for solver-beam leaf valuation.

    The beam calls `evaluate(state)` on every leaf state (terminal or
    depth-exhausted) and propagates the value up via max selection.
    """

    def evaluate(self, state: GameState) -> float:
        ...


@dataclass(frozen=True, slots=True)
class FastHeuristicLeaf:
    """Cheap deterministic leaf evaluator for solver beam expansion.

    Calling ``clear_probability`` or ``headroom_value`` at every child state
    makes depth-3+ beams spend most of their time re-enumerating playable
    hands. This evaluator is intentionally O(visible state): it combines
    current-blind progress, remaining resources, economy, and joker count.
    """

    progress_weight: float = 0.45
    resource_weight: float = 0.25
    build_weight: float = 0.30

    def evaluate(self, state: GameState) -> float:
        if state.won:
            return 2.0
        if state.run_over or state.phase == GamePhase.RUN_OVER:
            return 0.0

        if state.phase == GamePhase.ROUND_EVAL or state.current_score >= state.required_score:
            return 1.0 + min(0.25, max(0, state.hands_remaining) * 0.04 + max(0, state.money) * 0.002)

        target = max(1, state.required_score)
        progress = min(1.0, max(0.0, state.current_score / target))
        hand_resource = min(1.0, max(0, state.hands_remaining) / 4)
        discard_resource = min(1.0, max(0, state.discards_remaining) / 4)
        money_resource = min(1.0, max(0, state.money) / 40)
        joker_build = min(1.0, len(state.jokers) / max(1, int(state.modifiers.get("joker_slot_limit", 5) or 5)))
        return (
            progress * self.progress_weight
            + ((hand_resource * 0.5 + discard_resource * 0.2 + money_resource * 0.3) * self.resource_weight)
            + (joker_build * self.build_weight)
        )


@dataclass(frozen=True, slots=True)
class PlanningValueLeaf:
    """Leaf evaluator that delegates to `state_value.planning_value`.

    This is the closest analog to what `best_blind_beam_action` uses
    today (via `_beam_leaf_value`). It runs a small number of greedy
    rollouts to estimate clear probability, then layers headroom +
    in-progress signal on top so the beam can distinguish:
        - leaves that clear with margin (preferred over bare clear)
        - leaves that don't clear but made progress (preferred over
          leaves that didn't, when no path clears)

    Default `samples=4` (lower than M4's planning_value default of 16
    because the solver beam visits many more leaves per decision and
    the marginal precision from extra samples isn't worth the cost).

    Use `FastHeuristicLeaf` if you can afford the quality hit for
    ~10x more leaves per decision (e.g. depth-5+ experiments where
    the deeper search compensates for the cheaper leaf).
    """

    samples: int = 4
    seed: int = 0

    def evaluate(self, state: GameState) -> float:
        return planning_value(state, samples=self.samples, seed=self.seed)


@dataclass(frozen=True, slots=True)
class ClearProbabilityLeaf:
    """Leaf evaluator that mirrors `state_value.clear_probability`.

    Default `samples=4` is the solver tuning — the beam visits
    significantly more leaves than the live bot, and the variance
    of clear-probability across 4 vs 16 samples is small enough to
    not matter at depth=3-5. Tune up if leaf evaluations are noisy
    enough to cause beam thrash.

    `seed` is mixed into the rollout RNG so independent leaf calls
    on the same state are deterministic but different invocations
    with different seeds give independent estimates (useful when
    debugging variance).
    """

    samples: int = 4
    seed: int = 0

    def evaluate(self, state: GameState) -> float:
        if state.won:
            return 2.0
        if state.run_over or state.phase == GamePhase.RUN_OVER:
            return 0.0
        if state.phase == GamePhase.ROUND_EVAL:
            # Already cleared this blind; reward bonus matches what
            # `planning_value` gives.
            return 1.0 + min(0.75, headroom_value(state) * 0.25)
        return clear_probability(state, samples=self.samples, seed=self.seed)


@dataclass(frozen=True, slots=True)
class FutureBlindSurvivalLeaf:
    """Project survival probability across the current + next blind.

    Approach:
      base_value = clear_probability(state, samples=samples)
      if base_value clears:
          add a "next blind" bonus proportional to headroom_value —
          builds with surplus chips/jokers/money survive better.

    This is intentionally cheap: it does NOT actually simulate the
    next blind's required-score or run another rollout. The headroom
    value already captures the relevant signal (build strength
    relative to current blind requirement, which correlates with
    next-blind difficulty).

    Tuning: `future_weight` scales how much the next-blind bonus
    can shift the leaf value. The default 0.4 means a fully-built
    leaf can score up to ~40% above a bare-clear leaf at the same
    clear probability.
    """

    samples: int = 4
    seed: int = 0
    future_weight: float = 0.4

    def evaluate(self, state: GameState) -> float:
        if state.won:
            return 2.0
        if state.run_over or state.phase == GamePhase.RUN_OVER:
            return 0.0
        if state.phase == GamePhase.ROUND_EVAL:
            # Cleared. Reward proportional to headroom so cleared-with-
            # margin beats cleared-by-a-hair.
            head = headroom_value(state)
            return 1.0 + min(1.0, head * (0.25 + self.future_weight * 0.25))

        clear = clear_probability(state, samples=self.samples, seed=self.seed)
        head = headroom_value(state)
        # If we likely clear, headroom is forward signal for next blind.
        # If we likely DON'T clear, headroom is mostly noise (we won't
        # get there) — multiply by `clear` to dampen.
        return clear + (clear * head * self.future_weight)


@dataclass(frozen=True, slots=True)
class ArchetypeAwareLeaf:
    """Decorator: base evaluator + archetype-fit bonus.

    Wraps any other `LeafEvaluator` and adds `archetype.archetype_fit_score(state)`
    scaled by `bonus_weight`. The bonus is additive — the archetype
    is a soft bias, not a hard override.

    `bonus_weight` keeps the bonus small relative to the base value
    so a strong build for the wrong archetype still beats a weak
    build for the right archetype. Default 0.05 means a 5-match
    archetype fit can shift the value by ~0.5 (vs base of 0-2).
    """

    base: LeafEvaluator
    archetype: object  # Avoid circular import: archetypes.Archetype
    bonus_weight: float = 0.05

    def evaluate(self, state: GameState) -> float:
        base_value = self.base.evaluate(state)
        fit_bonus = self.archetype.archetype_fit_score(state) * self.bonus_weight
        return base_value + fit_bonus
