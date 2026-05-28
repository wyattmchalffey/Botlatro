"""Branch-and-bound play search (Tier 2 #5 of SOLVER_OPTIMIZATION_PLAN.md).

Different angle from the d3w2 beam: instead of fixed-depth minimax,
this uses depth-first search with branch pruning based on an upper-
bound on the score achievable from each state. Branches whose upper
bound can't beat the current best leaf value are pruned without
expansion — vastly fewer nodes visited than the beam's full
enumeration when the bound is tight enough.

Key insight: Balatro blind-clearing is a single-player deterministic
planning problem (given a seeded RNG for draws). No expectimax
needed — every branch has a deterministic outcome. Branch-and-bound
with `current_score + hands_remaining * best_immediate_score` as
the upper bound is the natural fit.

Heuristic choices made here:
- **Upper bound** is `current_score + hands_remaining * max_per_hand`
  where `max_per_hand = _best_immediate_score(state)` (cheap and
  already content-cached in state_value.py). This is loose but
  cheap; later we could replace with a tighter estimate.
- **Search depth cap** = `hands_remaining + discards_remaining` so
  the search can't expand past resources actually available.
- **Per-ply candidate cap** matches the existing beam (width+1 = 3)
  so we don't blow up the branching factor on rich hands.
- **Leaf evaluator** unchanged: PlanningValueLeaf at terminal /
  bound-cutoff states. Same leaf logic as the beam, just a
  different exploration order + pruning criterion.

This is INTENTIONALLY a smaller change than the original Tier 2 #5
plan called for (full IDA* with iterative deepening). The
branch-and-bound version is simpler to validate and gives most of
the same value — pruning unpromising branches early. If the
measurement is encouraging we can promote to true IDA*.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Protocol

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import Card, GamePhase, GameState, with_derived_legal_actions
from balatro_ai.search.deck_model import DeckModel
from balatro_ai.search.forward_sim import simulate_discard, simulate_play
from balatro_ai.search.state_value import _best_immediate_score, state_value_cache_scope
from balatro_ai.solver.search_v2.leaf_value import LeafEvaluator, PlanningValueLeaf
from balatro_ai.solver.search_v2.play import TopKByImmediateScore, CandidateProvider


DEFAULT_MAX_DEPTH = 7  # bounded by hands+discards typically; safe upper limit
DEFAULT_WIDTH = 3


def bnb_play_action(
    state: GameState,
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    width: int = DEFAULT_WIDTH,
    leaf_evaluator: LeafEvaluator | None = None,
    candidate_provider: CandidateProvider | None = None,
    rng_seed: int = 0,
) -> Action | None:
    """Branch-and-bound search returning the first action of the best line.

    Parameters
    ----------
    state: SELECTING_HAND state with non-empty hand.
    max_depth: hard cap on plies expanded below the root. Real depth
        is also bounded by `hands_remaining + discards_remaining` so
        a value of 7 is enough for any normal blind state.
    width: per-ply branching cap. Smaller than beam's width because
        we explore fewer nodes via pruning.
    leaf_evaluator: scores leaf states (terminal or bound-cutoff).
        Default `PlanningValueLeaf`.
    candidate_provider: action enumeration. Default `TopKByImmediateScore`.
    rng_seed: deterministic seed for sampled draws.
    """

    if not state.hand:
        return None
    if state.phase != GamePhase.SELECTING_HAND:
        return None

    if leaf_evaluator is None:
        leaf_evaluator = PlanningValueLeaf()
    if candidate_provider is None:
        candidate_provider = TopKByImmediateScore()

    root_candidates = candidate_provider(state, width=width)
    if not root_candidates:
        return None

    with state_value_cache_scope():
        # First pass: get an initial best estimate by running the natural
        # greedy line. This gives us a non-trivial pruning threshold.
        initial_best_value = _baseline_value(state, leaf_evaluator)

        ctx = _SearchContext(
            leaf_evaluator=leaf_evaluator,
            candidate_provider=candidate_provider,
            width=width,
            max_depth=max_depth,
            best_value=initial_best_value,
            rng_seed=rng_seed,
        )

        best_action: Action | None = None
        for index, action in enumerate(root_candidates):
            child = _simulate(state, action, rng_seed + index)
            if child is None:
                continue
            # Try expanding this root branch with the current best as
            # pruning threshold. If we beat it, update.
            value = _search(child, depth=1, ctx=ctx, rng_seed=rng_seed + index + 1)
            if value > ctx.best_value:
                ctx.best_value = value
                best_action = action

        # Fallback: if no branch beat the baseline, return the highest-
        # ranked root candidate (matches the beam's "always return
        # something legal" contract).
        if best_action is None:
            best_action = root_candidates[0]

    return best_action


@dataclass
class _SearchContext:
    """Mutable context threaded through recursive DFS calls."""

    leaf_evaluator: LeafEvaluator
    candidate_provider: CandidateProvider
    width: int
    max_depth: int
    best_value: float
    rng_seed: int


def _baseline_value(state: GameState, leaf_evaluator: LeafEvaluator) -> float:
    """Cheap initial estimate to seed the pruning threshold.

    Just evaluates the current state's leaf value — usually below
    what any reasonable action sequence achieves, so almost any
    branch will improve on it. The point isn't a tight bound; it's
    to AVOID having `best_value = -inf` which would never prune.
    """

    return leaf_evaluator.evaluate(state)


def _search(
    state: GameState,
    *,
    depth: int,
    ctx: _SearchContext,
    rng_seed: int,
) -> float:
    """DFS with bound pruning. Returns best leaf value reachable from state."""

    # Terminal conditions.
    if state.run_over or state.phase == GamePhase.RUN_OVER:
        return 0.0
    if state.won:
        return 2.0
    if (state.required_score > 0 and state.current_score >= state.required_score) \
            or state.phase == GamePhase.ROUND_EVAL:
        return ctx.leaf_evaluator.evaluate(state)
    if depth >= ctx.max_depth:
        return ctx.leaf_evaluator.evaluate(state)
    if state.hands_remaining <= 0 or not state.hand:
        return ctx.leaf_evaluator.evaluate(state)

    # Pruning DISABLED. The naive upper bound (`current_score + best *
    # hands_remaining`) is too pessimistic for discard branches —
    # discarding to draw better cards can pay off in ways the bound
    # doesn't see, so pruning on it cuts winning lines. A tighter
    # admissible upper bound for Balatro is a real research project;
    # for now, plain DFS at this depth. With pruning off this is
    # essentially equivalent to the existing beam, so the test below
    # mostly validates the per-ply candidate cap behavior.

    candidates = ctx.candidate_provider(state, width=ctx.width)
    if not candidates:
        return ctx.leaf_evaluator.evaluate(state)

    per_ply_cap = max(1, ctx.width + 1)
    branches = candidates[:per_ply_cap]

    best_in_subtree = float("-inf")
    for index, action in enumerate(branches):
        child = _simulate(state, action, rng_seed + index)
        if child is None:
            continue
        value = _search(
            child,
            depth=depth + 1,
            ctx=ctx,
            rng_seed=rng_seed + index + 1,
        )
        if value > best_in_subtree:
            best_in_subtree = value
            # Update global best as soon as we find a better one —
            # tightens the pruning threshold for sibling branches.
            if value > ctx.best_value:
                ctx.best_value = value

    if best_in_subtree == float("-inf"):
        return ctx.leaf_evaluator.evaluate(state)
    return best_in_subtree


def _upper_bound(state: GameState) -> float:
    """Loose upper bound on the leaf value achievable from `state`.

    Calculation: assume we can play the current `_best_immediate_score`
    (the score of the best play action from the current hand) once
    per remaining hand. If `current_score + best * hands_remaining`
    clears the blind, we project a clear-leaf value (1.0); otherwise
    we project a partial-progress leaf value (~0.3, matching how
    the `planning_value` formula scores in-progress states).

    This is loose — actual achievable score is often LARGER (we
    might draw better cards mid-blind) — but it's admissible-ish
    in the sense that if THIS bound predicts a clear, the search
    is likely to find one. The bound is mainly used to prune
    obviously-doomed branches (where even max greedy can't clear).
    """

    if state.run_over or state.phase == GamePhase.RUN_OVER:
        return 0.0
    if state.won:
        return 2.0
    remaining = max(0, state.required_score - state.current_score)
    if remaining == 0:
        return 1.0
    best = _best_immediate_score(state)
    if best <= 0:
        return 0.0
    projected_max = best * max(1, state.hands_remaining)
    if projected_max >= remaining:
        # Could clear → cap at the planning_value "we cleared" zone.
        return 1.5
    # Can't clear; projected leaf is partial-progress weighted by
    # ratio of progress made.
    progress = min(1.0, (state.current_score + projected_max) / state.required_score)
    return progress * 0.4


def _simulate(state: GameState, action: Action, rng_seed: int) -> GameState | None:
    """Run forward_sim, return resulting state with legal_actions populated."""

    try:
        if action.action_type == ActionType.PLAY_HAND:
            drawn = _sample_draw(state, action, rng_seed)
            simulated = simulate_play(state, action, drawn_cards=drawn)
        elif action.action_type == ActionType.DISCARD:
            drawn = _sample_draw(state, action, rng_seed)
            simulated = simulate_discard(state, action, drawn_cards=drawn)
        else:
            return None
    except (ValueError, IndexError, KeyError, TypeError, AttributeError):
        return None
    try:
        return with_derived_legal_actions(simulated)
    except (ValueError, IndexError, KeyError, TypeError, AttributeError):
        return None


def _sample_draw(state: GameState, action: Action, rng_seed: int) -> tuple[Card, ...]:
    draw_count = len(action.card_indices)
    if draw_count <= 0:
        return ()
    try:
        model = DeckModel.from_state(state)
    except (ValueError, AttributeError):
        return ()
    draw_count = min(draw_count, model.total_cards)
    if draw_count <= 0:
        return ()
    return model.sample_draws(draw_count, Random(rng_seed))
