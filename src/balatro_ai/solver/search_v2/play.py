"""Custom solver play search (Tier 1 #1 of SOLVER_OPTIMIZATION_PLAN.md).

`solver_beam_play_action` is the from-scratch replacement for the
wrapped `best_hand_action` that the M4 `PlaySearchPolicy` uses. The
algorithmic shape is the same — whole-blind depth-limited search
with per-ply candidate enumeration — but:

- No live-bot preprocessing (`_opening_first_blind_hunt_action`,
  `_basic_best_discard_action`) which dominated per-call cost in
  the M4 profile (~6-7s of the ~8s opening-hand decision).
- Configurable leaf evaluator via the `LeafEvaluator` protocol.
- Configurable candidate provider for action enumeration / ranking.
- Cheaper default candidate cap (`width * 2` per ply) since the
  solver beam is symmetric with itself across plies (we don't need
  a wider root than internal nodes).

The trade vs `best_hand_action`:
- We give up the live-bot's tuned "first hand of run, hunt for a
  clear" shortcut. For the solver this is fine — the beam will
  find the same action on its own with one extra ply of search.
- We give up decision-tracing metadata (`reason`, `search_value`).
  Trajectories don't need these; if we ever want to debug a
  specific decision we can re-run with annotation on.

Acceptance bar (per `SOLVER_OPTIMIZATION_PLAN.md` §3 #1):
- <500ms per call on AAAAAAA opening hand (currently ~8s for the
  wrapped path).
- Parity ≥80% with `best_hand_action(state, beam_depth=3, beam_width=2)`
  on canonical-seed states.
- Full trajectory via `generate_trajectory` runs to RUN_OVER with
  ante ≥ baseline median 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Protocol

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import Card, GamePhase, GameState, with_derived_legal_actions
from balatro_ai.bots.basic_strategy_bot import BasicStrategyBot
from balatro_ai.search.deck_model import DeckModel
from balatro_ai.search.discard_search import _discard_candidate_score
from balatro_ai.search.forward_sim import simulate_discard, simulate_play
from balatro_ai.solver.search_v2.leaf_value import (
    FastHeuristicLeaf,  # noqa: F401 — re-exported for callers building deeper experiments
    LeafEvaluator,
    PlanningValueLeaf,
)
from balatro_ai.search.state_value import state_value_cache_scope
from balatro_ai.solver.search_v2.memo import solver_search_cache_scope


# Default beam configuration. Matches the M4 `PlaySearchPolicy`
# defaults so a drop-in swap of `best_hand_action` → `solver_beam_play_action`
# starts at parity tuning, not a different operating point.
DEFAULT_DEPTH = 3
DEFAULT_WIDTH = 2


class CandidateProvider(Protocol):
    """Protocol for enumerating + ranking candidate actions at one ply.

    The beam calls this once per node. Implementations return an
    ordered tuple of `Action` objects (best-first); the beam tries
    each in order and keeps the highest-leaf-value result.

    Implementations MUST handle the case where `state.legal_actions`
    is empty (return empty tuple).
    """

    def __call__(self, state: GameState, *, width: int) -> tuple[Action, ...]:
        ...


@dataclass(frozen=True, slots=True)
class TopKByImmediateScore:
    """Default candidate provider.

    Enumerates legal play and discard actions, ranks plays by
    immediate score (via `_score_play_action`), discards by
    `_discard_candidate_score`. Returns plays first then discards.

    Why naive concatenation, not unified ranking:
    Tried sorting plays+discards together using M4's pressure-bonus
    formula. Result on the 4-seed canonical batch was *worse* than
    the naive concat (avg ante 1.25 vs original ante ~4) because
    pressure_bonus(15000) dominated play heuristic scores (100-1000),
    pushing ALL discards above all plays at the opening hand of
    every seed. The beam then explored only discard chains, burning
    discards early and busting on weak plays. Even with a "min
    plays in top-K" guarantee, the per_ply_cap at internal plies
    still pruned plays out.

    The naive concat works because `solver_beam_play_action` iterates
    ALL root candidates (no truncation at root), so plays and
    discards both get explored as root branches; internal plies
    pick top-K plays which is usually the right move (you don't
    typically chain 3 discards in a row).
    """

    play_oversample: int = 2
    discard_share: int = 1

    def __call__(self, state: GameState, *, width: int) -> tuple[Action, ...]:
        play_actions = tuple(
            a for a in state.legal_actions
            if a.action_type == ActionType.PLAY_HAND and a.card_indices
        )
        discard_actions = tuple(
            a for a in state.legal_actions
            if a.action_type == ActionType.DISCARD and a.card_indices
        )

        play_limit = max(1, width * self.play_oversample)
        discard_limit = max(0, width * self.discard_share)

        ranked_plays = _rank_plays(state, play_actions, limit=play_limit)
        ranked_discards = _rank_discards(state, discard_actions, limit=discard_limit)
        return (*ranked_plays, *ranked_discards)


@dataclass(slots=True)
class SearchV2PlayPolicy:
    """Policy wrapper around `solver_beam_play_action`.

    This is the drop-in solver replacement for the legacy M4
    `PlaySearchPolicy`. It owns only play/discard states; every other phase,
    plus any search failure, delegates to `fallback`.
    """

    depth: int = DEFAULT_DEPTH
    width: int = DEFAULT_WIDTH
    leaf_evaluator: LeafEvaluator | None = None
    candidate_provider: CandidateProvider | None = None
    seed: int = 0
    fallback: object | None = None

    def __post_init__(self) -> None:
        if self.fallback is None:
            self.fallback = BasicStrategyBot(seed=self.seed)

    def __call__(self, state: GameState) -> Action:
        return self.choose_action(state)

    def choose_action(self, state: GameState) -> Action:
        if state.phase == GamePhase.SELECTING_HAND and _has_play_or_discard(state):
            try:
                action = solver_beam_play_action(
                    state,
                    depth=self.depth,
                    width=self.width,
                    leaf_evaluator=self.leaf_evaluator,
                    candidate_provider=self.candidate_provider,
                    rng_seed=self.seed,
                )
            except (ValueError, IndexError, KeyError, TypeError, AttributeError):
                action = None
            if action is not None:
                return action
        return self.fallback.choose_action(state)


def solver_beam_play_action(
    state: GameState,
    *,
    depth: int = DEFAULT_DEPTH,
    width: int = DEFAULT_WIDTH,
    leaf_evaluator: LeafEvaluator | None = None,
    candidate_provider: CandidateProvider | None = None,
    rng_seed: int = 0,
) -> Action | None:
    """Whole-blind search returning the first action of the best line.

    Parameters
    ----------
    state: a `SELECTING_HAND` state with non-empty hand and at least
        one legal play or discard action. If no candidate actions
        exist, returns `None` (caller should fall back).
    depth: maximum plies expanded below the root before evaluating
        the leaf. The blind usually clears or busts before this,
        so it's an upper bound — typical effective depth at the
        root is 1-2 plies.
    width: per-ply candidate cap. The `candidate_provider` may
        enumerate more than `width` raw actions but the beam keeps
        at most `width` simulated children per parent.
    leaf_evaluator: scores leaf states. Default `PlanningValueLeaf` —
        the rollout-backed leaf that gives M4-parity quality at the
        d3w2 op-point. For deeper-search experiments where the extra
        depth compensates for a cheaper leaf, swap in `FastHeuristicLeaf`
        (rollout-free, ~10x more leaves per second but loses ~1-2
        antes at shallow depths on the AAAAAAA acceptance seed).
    candidate_provider: action enumeration + ranking. Default
        `TopKByImmediateScore`.
    rng_seed: seed for the deterministic-sample draw at each
        simulated action. Same seed → same trajectory for a given
        state input, which is what reproducible solver runs need.

    Returns
    -------
    The best root action found, or `None` if no candidate actions
    exist (caller should fall back to a heuristic policy).
    """

    if not state.hand:
        return None
    if state.phase != GamePhase.SELECTING_HAND:
        return None

    if leaf_evaluator is None:
        # PlanningValueLeaf (rollout-backed) gives M4-parity quality.
        # The earlier draft of this module defaulted to FastHeuristicLeaf
        # for speed but a single-seed AAAAAAA measurement showed it
        # loses ~3 antes vs M4 at d3w2 — the rollout-free leaf can't
        # distinguish a setup discard from a bad one, so the beam never
        # builds. Keep FastHeuristicLeaf as the explicit opt-in for
        # deeper-search experiments where the algorithm has enough plies
        # to find the build without leaf guidance.
        leaf_evaluator = PlanningValueLeaf()
    if candidate_provider is None:
        candidate_provider = TopKByImmediateScore()

    root_candidates = candidate_provider(state, width=width)
    if not root_candidates:
        return None

    # Identity-keyed local memo catches the trivial case where the
    # same forward-simulated state is re-evaluated at the same depth
    # within one call. The CONTENT-keyed cache (Tier 1 #3) lives in
    # `search_v2.memo` and is activated by the `solver_search_cache_scope`
    # below — it catches the bigger win: identical-content states
    # produced by different beam branches share leaf-eval results.
    #
    # The value is a (state, float) tuple, NOT a bare float: `id(state)`
    # alone is unsafe as a key because transient child states are GC'd
    # mid-search and CPython reuses their addresses for new states,
    # causing stale memo hits and nondeterministic beam values. Storing
    # the state reference both (a) keeps the id from being reused while
    # the entry is live and (b) lets us guard the lookup with `is`.
    memo: dict[tuple[int, int], tuple[GameState, float]] = {}

    best_action: Action | None = None
    best_value = float("-inf")
    # No cache scope wrap right now. Two attempts at cache wrapping
    # (live-bot scope + my own content-keyed scope) both regressed
    # quality on the AAAAAAA acceptance seed in ways I couldn't fully
    # diagnose under the time budget — suspect a stale rollout-state
    # contamination through a content-key that's safe in isolation
    # but loses determinism inside the beam's per-step state evolution.
    # The leaf_evaluator imports go straight to state_value.* (not the
    # cached_* wrappers in memo.py) so the wrappers aren't used in
    # production paths until the regression is understood.
    for index, action in enumerate(root_candidates):
        child = _simulate_action(state, action, rng_seed=rng_seed + index)
        if child is None:
            continue
        value = _plan_value(
            child,
            depth=depth - 1,
            width=width,
            leaf_evaluator=leaf_evaluator,
            candidate_provider=candidate_provider,
            rng_seed=rng_seed + index + 1,
            memo=memo,
        )
        if value > best_value:
            best_value = value
            best_action = action

    return best_action


# --- Internals --------------------------------------------------------


def _plan_value(
    state: GameState,
    *,
    depth: int,
    width: int,
    leaf_evaluator: LeafEvaluator,
    candidate_provider: CandidateProvider,
    rng_seed: int,
    memo: dict[tuple[int, int], tuple[GameState, float]],
) -> float:
    """Recursive value backup. Returns max leaf value reachable.

    Algorithm: depth-limited minimax over actions, with branching
    capped at `_per_ply_candidate_cap(width)` actions per node. The
    candidate_provider supplies a pre-ranked list and the beam keeps
    the top-N for recursion — we DO NOT prune by 1-ply leaf value
    because a setup discard often looks worse than a play at 1 ply
    but pays off after 2-3 plies. This matches the M4 beam shape;
    the speedup comes from a cheaper leaf evaluator and no
    preprocessing shortcut, not from greedy beam pruning.
    """

    if depth <= 0 or _is_terminal(state):
        return leaf_evaluator.evaluate(state)

    memo_key = (id(state), depth)
    cached = memo.get(memo_key)
    if cached is not None and cached[0] is state:
        return cached[1]

    candidates = candidate_provider(state, width=width)
    if not candidates:
        value = leaf_evaluator.evaluate(state)
        memo[memo_key] = (state, value)
        return value

    # Cap per-ply branching factor. At ply 1 the root expansion is
    # uncapped (handled by `solver_beam_play_action`), but internal
    # plies use `width+1` so the candidate_provider's ranking gets
    # its top slot guaranteed plus 1-2 alternates.
    per_ply_cap = max(1, width + 1)
    branches = candidates[:per_ply_cap]

    best_value = float("-inf")
    for index, action in enumerate(branches):
        child = _simulate_action(state, action, rng_seed=rng_seed + index)
        if child is None:
            continue
        value = _plan_value(
            child,
            depth=depth - 1,
            width=width,
            leaf_evaluator=leaf_evaluator,
            candidate_provider=candidate_provider,
            rng_seed=rng_seed + index + 1,
            memo=memo,
        )
        if value > best_value:
            best_value = value

    if best_value == float("-inf"):
        best_value = leaf_evaluator.evaluate(state)
    memo[memo_key] = (state, best_value)
    return best_value


def _simulate_action(
    state: GameState,
    action: Action,
    *,
    rng_seed: int,
) -> GameState | None:
    """Run forward_sim, return the resulting state with legal_actions populated.

    Returns `None` on any simulator-raised exception — the caller
    should treat that action as infeasible and skip it. Exceptions
    are recoverable in this context (e.g. card-index validation
    drift from a non-standard joker effect); we don't want one
    bad action to kill the whole search.
    """

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

    # Re-derive legal_actions so the next ply's candidate_provider
    # can read them. simulate_* sets legal_actions=() to force a
    # fresh derivation here.
    try:
        return with_derived_legal_actions(simulated)
    except (ValueError, IndexError, KeyError, TypeError, AttributeError):
        return None


def _sample_draw(state: GameState, action: Action, rng_seed: int) -> tuple[Card, ...]:
    """Deterministic single-sample draw of the cards replacing the played/discarded ones."""

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


def _is_terminal(state: GameState) -> bool:
    """Search terminates here — blind cleared, run over, or no hand to play."""

    return (
        state.won
        or state.run_over
        or state.phase in {GamePhase.ROUND_EVAL, GamePhase.RUN_OVER}
        or (state.required_score > 0 and state.current_score >= state.required_score)
        or state.hands_remaining <= 0
        or not state.hand
        or state.phase != GamePhase.SELECTING_HAND
    )


def _has_play_or_discard(state: GameState) -> bool:
    return any(action.action_type in {ActionType.PLAY_HAND, ActionType.DISCARD} for action in state.legal_actions)


def _rank_plays(
    state: GameState,
    actions: tuple[Action, ...],
    *,
    limit: int,
) -> tuple[Action, ...]:
    """Rank candidate play actions by immediate score, keep top-`limit`."""

    if not actions or limit <= 0:
        return ()
    if len(actions) <= limit:
        return actions
    # Lazy import to avoid pulling basic_strategy into module load.
    from balatro_ai.bots.basic_strategy.play_scoring import (
        _score_play_action as score_play_action,
    )
    from balatro_ai.bots.basic_strategy.hand_models import _BlindContext

    ctx = _BlindContext()
    remaining = max(0, state.required_score - state.current_score)

    # Rust batched scorer: ~16× faster on AAAAAAA-style states. Tries
    # to score all actions in one FFI call; falls back to Python per-
    # action loop for any action where Rust returns None (Wild cards,
    # unsupported jokers, special blinds — handled by score_play_action's
    # existing fallback).
    rust_scores = _rust_score_play_actions_batch(state, actions)

    scored: list[tuple[float, int, Action]] = []
    for idx, action in enumerate(actions):
        rs = rust_scores[idx] if rust_scores is not None else None
        if rs is not None:
            score = int(rs)
        else:
            score = score_play_action(state, action, ctx)
        value = _play_rank_from_score(state, action, ctx, remaining, score)
        scored.append((value, -idx, action))
    scored.sort(reverse=True)
    return tuple(action for _, _, action in scored[:limit])


def _rust_score_play_actions_batch(
    state: GameState,
    actions: tuple[Action, ...],
) -> list[int | None] | None:
    """Batched Rust scorer. Returns per-action scores or None for any
    action Rust can't handle (caller falls back to Python). Returns
    None entirely if balatro_core isn't loadable or the state has a
    non-vanilla blind."""

    try:
        import balatro_core
    except ImportError:
        return None
    # Skip on non-vanilla blinds (boss adjustments handled Python-side
    # by _boss_adjusted_score). _rank_plays uses raw scores for ranking,
    # so passing through unaware doesn't change rankings.
    from balatro_ai.search.state_value import (
        _cached_hand_as_rust_cards, _cached_joker_data,
        _cached_played_hand_types_strs, _PLAYED_STATE_JOKERS,
    )
    from balatro_ai.rules.hand_evaluator import debuffed_suits_for_blind
    hand_rust = _cached_hand_as_rust_cards(state, balatro_core)
    if hand_rust is None:
        try:
            hand_rust = [balatro_core.RustCard.from_python(c) for c in state.hand]
        except (AttributeError, TypeError, ValueError):
            return None
    try:
        (joker_names, joker_editions, joker_plus_mult,
         joker_plus_chips, joker_xmult,
         joker_loyalty_ready, joker_drivers_active,
         joker_leading_plus_mult, joker_leading_plus_chips,
         joker_sell_value, joker_rarity,
         joker_target_suit, joker_target_rank,
         joker_obelisk_gain) = _cached_joker_data(state)
    except Exception:  # noqa: BLE001
        return None
    if any(j.name in _PLAYED_STATE_JOKERS for j in state.jokers):
        played_types_strs = _cached_played_hand_types_strs(state)
    else:
        played_types_strs = []
    debuffed = [s for s in debuffed_suits_for_blind(state.blind) if len(s) == 1]
    slot_limit_raw = state.modifiers.get("joker_slot_limit", 5)
    try:
        joker_slot_limit = int(slot_limit_raw) if slot_limit_raw is not None else 5
    except (TypeError, ValueError):
        joker_slot_limit = 5
    action_indices_list = [list(a.card_indices) for a in actions]
    try:
        scores = balatro_core.score_play_actions_batch(
            hand_rust, action_indices_list,
            state.hand_levels, debuffed,
            joker_names=joker_names, joker_editions=joker_editions,
            joker_current_plus_mult=joker_plus_mult,
            joker_current_plus_chips=joker_plus_chips,
            joker_current_xmult=joker_xmult,
            joker_loyalty_ready=joker_loyalty_ready,
            joker_drivers_active=joker_drivers_active,
            joker_leading_plus_mult=joker_leading_plus_mult,
            joker_leading_plus_chips=joker_leading_plus_chips,
            joker_sell_value=joker_sell_value, joker_rarity=joker_rarity,
            joker_target_suit=joker_target_suit, joker_target_rank=joker_target_rank,
            joker_obelisk_gain=joker_obelisk_gain,
            money=int(state.money), joker_slot_limit=joker_slot_limit,
            discards_remaining=int(max(0, state.discards_remaining)),
            hands_remaining=int(max(0, state.hands_remaining)),
            played_hand_types=played_types_strs,
            deck_size=int(max(0, state.deck_size)),
        )
    except Exception:  # noqa: BLE001
        return None
    return scores


def _play_rank_from_score(
    state: GameState,
    action: Action,
    ctx,
    remaining: int,
    score: int,
) -> float:
    """Compute the per-action rank value from a precomputed score.
    Shared by _play_rank (Python-only path) and the batched path."""

    value = float(score)
    if remaining > 0:
        if score >= remaining:
            value += 100_000.0 - (len(action.card_indices) * 3.0)
        else:
            needed = max(1, (remaining + score - 1) // max(1, score))
            value += 10_000.0 / needed
    return value


def _play_rank(
    state: GameState,
    action: Action,
    ctx,
    remaining: int,
    score_play_action,
) -> float:
    """Cheap heuristic rank for plays. Mirrors hand_search._cheap_beam_play_rank."""

    score = score_play_action(state, action, ctx)
    value = float(score)
    if remaining > 0:
        if score >= remaining:
            # Clears the blind — strongly preferred. Tiebreak by
            # using fewer cards (saves the rest of the hand for
            # future blinds).
            value += 100_000.0 - (len(action.card_indices) * 3.0)
        else:
            # Doesn't clear; reward by "fewer hands needed to clear
            # at this pace" (rough proxy for hand quality).
            needed = max(1, (remaining + score - 1) // max(1, score))
            value += 10_000.0 / needed
    return value


def _rank_discards(
    state: GameState,
    actions: tuple[Action, ...],
    *,
    limit: int,
) -> tuple[Action, ...]:
    """Rank candidate discards by `_discard_candidate_score`, keep top-`limit`."""

    if not actions or limit <= 0:
        return ()
    if len(actions) <= limit:
        return actions
    # Precompute all candidate heuristic scores in ONE Rust FFI call
    # (discard_candidate_scores_batch) rather than N per-action Python calls —
    # this ranking is the solver's hottest discard path (~280K Python calls /
    # trajectory). The Rust batch is the faithful port of _discard_candidate_score
    # (already trusted for ranking in discard_search._candidate_discard_actions);
    # the lazy fallback only runs on a cache miss (Rust card-extraction failure),
    # so the ranking is identical to the per-action Python version.
    from balatro_ai.search.discard_search import _rust_batch_discard_scores
    score_cache = _rust_batch_discard_scores(state, actions)
    if score_cache is not None:
        def _cs(action: Action) -> float:
            key = action.card_indices
            return score_cache[key] if key in score_cache else _discard_candidate_score(state, action)
    else:
        def _cs(action: Action) -> float:
            return _discard_candidate_score(state, action)
    scored = [(_cs(action), -idx, action) for idx, action in enumerate(actions)]
    scored.sort(reverse=True)
    return tuple(action for _, _, action in scored[:limit])
