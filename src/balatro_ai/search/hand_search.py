"""Integrated play/discard search for Phase 7 hand decisions."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from random import Random

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import Card, GamePhase, GameState, with_derived_legal_actions
from balatro_ai.search.deck_model import DeckModel
from balatro_ai.search.discard_search import (
    _candidate_discard_actions,
    _discard_candidate_score,
    _rust_batch_discard_scores,
)
from balatro_ai.search.forward_sim import simulate_discard, simulate_play
from balatro_ai.search.state_value import _card_cache_key, _freeze_for_cache, _jokers_cache_key, planning_value, state_value

ValueFn = Callable[[GameState], float]

# Opt-in A/B switch for the native (Rust) beam subtree evaluator.
# Default OFF so production / default runs use the pure-Python legacy
# beam. Set BALATRO_NATIVE_BEAM=1 to route `_beam_plan_value`'s subtree
# through `beam_play_value_native` for quality/speed comparison against
# the deterministic legacy beam (Phase 4d.2 re-validation).
_NATIVE_BEAM_ENABLED = os.environ.get("BALATRO_NATIVE_BEAM") == "1"

# ROOT-level native beam: one FFI call per play decision picks the best root
# action (the whole depth-3 tree expands in Rust). This is the architecture
# that can actually amortize FFI (vs the per-subtree _NATIVE_BEAM_ENABLED, which
# re-translates per node and was speed-neutral). Set BALATRO_NATIVE_BEAM_ROOT=1.
_NATIVE_BEAM_ROOT_ENABLED = os.environ.get("BALATRO_NATIVE_BEAM_ROOT") == "1"


@dataclass(frozen=True, slots=True)
class HandSearchConfig:
    draw_samples: int = 1
    leaf_samples: int = 1
    seed: int = 0
    enumerate_draws_up_to: int = 0
    max_play_actions: int = 16
    max_discard_actions: int = 8
    beam_depth: int = 0
    beam_width: int = 0
    beam_draw_samples: int = 1
    beam_leaf_samples: int = 1


def best_hand_action(
    state: GameState,
    *,
    config: HandSearchConfig | None = None,
    value_fn: ValueFn | None = None,
    context=None,
) -> Action | None:
    """Return the best modeled hand action across legal plays and discards."""

    search_config = config or HandSearchConfig()
    blind_context = context or _blind_context()
    play_actions = tuple(action for action in state.legal_actions if action.action_type == ActionType.PLAY_HAND and action.card_indices)
    discard_actions = tuple(action for action in state.legal_actions if action.action_type == ActionType.DISCARD and action.card_indices)
    if not play_actions and not discard_actions:
        return None

    opening_hunt = _opening_first_blind_hunt_action(state, discard_actions, blind_context)
    if opening_hunt is not None:
        return _annotated_action(opening_hunt, search_value=128.0, reason="first_blind_hunt hand_search_discard")

    if search_config.beam_depth > 1 and search_config.beam_width > 0:
        beam_action = best_blind_beam_action(state, config=search_config, context=blind_context)
        if beam_action is not None:
            return beam_action

    ranked_plays = _candidate_play_actions(state, play_actions, blind_context, limit=search_config.max_play_actions)
    ranked_discards = _candidate_discard_actions(state, discard_actions, limit=search_config.max_discard_actions)
    evaluator = value_fn or _default_value_fn(search_config)
    best_action: Action | None = None
    best_value = float("-inf")
    action_index = 0
    for action in (*ranked_plays, *ranked_discards):
        try:
            value = hand_action_value(
                state,
                action,
                config=search_config,
                value_fn=evaluator,
                action_index=action_index,
                context=blind_context,
            )
        except (ValueError, IndexError, TypeError, AttributeError):
            action_index += 1
            continue
        if value > best_value:
            best_action = action
            best_value = value
        action_index += 1

    if best_action is None:
        return None
    return _annotated_action(best_action, search_value=best_value, reason=_action_reason(state, best_action, blind_context))


def hand_action_value(
    state: GameState,
    action: Action,
    *,
    config: HandSearchConfig | None = None,
    value_fn: ValueFn | None = None,
    action_index: int = 0,
    context=None,
) -> float:
    """Evaluate a play or discard action through the forward simulator."""

    if action.action_type not in {ActionType.PLAY_HAND, ActionType.DISCARD}:
        raise ValueError(f"hand_action_value requires play/discard, got {action.action_type.value}")
    search_config = config or HandSearchConfig()
    blind_context = context or _blind_context()
    evaluator = value_fn or _default_value_fn(search_config)
    draw_count = _draw_count_for_action(state, action, context=blind_context)
    draws = _draw_outcomes(state, draw_count, config=search_config, action_index=action_index)
    total = 0.0
    for drawn_cards in draws:
        simulated = _simulate_hand_action(state, action, drawn_cards)
        total += evaluator(simulated)
    expected = total / max(1, len(draws))
    if action.action_type == ActionType.PLAY_HAND:
        expected += _play_action_bonus(state, action, context=blind_context)
    else:
        expected += _discard_action_bonus(state, action, context=blind_context)
    return expected


def best_blind_beam_action(
    state: GameState,
    *,
    config: HandSearchConfig | None = None,
    context=None,
) -> Action | None:
    """Return the best root action from a small whole-blind expectimax beam."""

    search_config = config or HandSearchConfig()
    blind_context = context or _context_from_state(state)
    actions = _beam_root_actions(state, search_config, blind_context)
    if not actions:
        return None

    # ROOT-level native beam (opt-in): one FFI call expands the whole tree for
    # every root candidate and returns the argmax index. Falls through to the
    # Python beam on bail (unsafe blind / complex jokers / Rust unavailable).
    if _NATIVE_BEAM_ROOT_ENABLED:
        native_idx = _try_rust_best_beam_action(state, search_config, actions)
        if native_idx is not None:
            chosen = actions[native_idx]
            return _annotated_action(
                chosen, search_value=0.0,
                reason=_beam_action_reason(chosen), search="hand_beam_native",
            )

    memo: dict[tuple[object, ...], float] | None = None
    best_action: Action | None = None
    best_value = float("-inf")
    for action_index, action in enumerate(actions):
        value = _beam_action_value(
            state,
            action,
            config=search_config,
            depth=max(0, search_config.beam_depth - 1),
            action_index=action_index,
            seed_offset=action_index * 1_000_003,
            memo=memo,
        )
        if value > best_value:
            best_action = action
            best_value = value

    if best_action is None:
        return None
    return _annotated_action(best_action, search_value=best_value, reason=_beam_action_reason(best_action), search="hand_beam")


def _beam_action_value(
    state: GameState,
    action: Action,
    *,
    config: HandSearchConfig,
    depth: int,
    action_index: int,
    seed_offset: int,
    memo: dict[tuple[object, ...], float] | None = None,
) -> float:
    if action.action_type not in {ActionType.PLAY_HAND, ActionType.DISCARD}:
        return float("-inf")

    context = _context_from_state(state)
    draw_count = _draw_count_for_action(state, action, context=context)
    draws = _beam_draw_outcomes(state, draw_count, config=config, action_index=action_index, seed_offset=seed_offset)
    if not draws:
        draws = ((),)

    total = 0.0
    valid = 0
    for draw_index, drawn_cards in enumerate(draws):
        try:
            child = _state_after_beam_action(state, action, drawn_cards)
        except (ValueError, IndexError, TypeError, AttributeError):
            continue
        total += _beam_plan_value(
            child,
            config=config,
            depth=depth,
            seed_offset=seed_offset + ((draw_index + 1) * 65_537),
            memo=memo,
        )
        valid += 1

    if valid <= 0:
        return float("-inf")
    return total / valid


def _beam_plan_value(
    state: GameState,
    *,
    config: HandSearchConfig,
    depth: int,
    seed_offset: int,
    memo: dict[tuple[object, ...], float] | None = None,
) -> float:
    if depth <= 0 or _beam_terminal(state):
        return _beam_leaf_value(state, config=config, seed_offset=seed_offset, memo=memo)

    memo_key = ("plan", depth, _beam_state_key(state))
    if memo is not None and memo_key in memo:
        return memo[memo_key]

    # A/B opt-in (BALATRO_NATIVE_BEAM=1): route the whole subtree
    # value through the native Rust beam. Compared against the
    # deterministic legacy Python beam on a seed batch (quality gate,
    # not step-identity). Default OFF.
    if _NATIVE_BEAM_ENABLED:
        rust_value = _try_rust_beam_plan_value(state, depth, config, seed_offset)
        if rust_value is not None:
            if memo is not None:
                memo[memo_key] = rust_value
            return rust_value

    # NOTE: `beam.rs` has a complete native beam with discards +
    # play ranking + xoshiro rollouts (Phase 4d.2 proper). When
    # wired into _beam_plan_value, both the minimal version (plays
    # only) AND the full version (plays + discards) broke parity
    # severely (130 steps → 33-34 steps on AAAAAAA). The bot's
    # value estimates diverge enough from Python's that it makes
    # quality-eroding decisions. Likely causes that need more
    # careful work to fix:
    #   - apply_play's joker metadata + modifier updates may
    #     differ subtly from Python's simulate_play
    #   - xoshiro RNG draws differ from Python's random.Random,
    #     leading to different rollout outcomes the beam treats
    #     as ground truth at each branch
    #   - The mixed play/discard ranking and limit selection may
    #     not exactly match Python's _beam_future_actions
    # Leaving `beam.rs` in the tree as scaffolding. To productize,
    # would need a careful side-by-side instrumentation pass to
    # find exactly which sub-decision diverges first.

    actions = _beam_future_actions(state, config)
    if not actions:
        return _beam_leaf_value(state, config=config, seed_offset=seed_offset, memo=memo)

    best_value = float("-inf")
    for action_index, action in enumerate(actions):
        value = _beam_action_value(
            state,
            action,
            config=config,
            depth=depth - 1,
            action_index=action_index,
            seed_offset=seed_offset + ((action_index + 1) * 131_071),
            memo=memo,
        )
        if value > best_value:
            best_value = value
    if best_value == float("-inf"):
        best_value = _beam_leaf_value(state, config=config, seed_offset=seed_offset, memo=memo)
    if memo is not None:
        memo[memo_key] = best_value
    return best_value


def _try_rust_beam_plan_value(
    state: GameState,
    depth: int,
    config: HandSearchConfig,
    seed_offset: int,
) -> float | None:
    """Bridge to Rust `beam_play_value_native`. Returns the plan
    value or None to bail."""

    try:
        import balatro_core
    except ImportError:
        return None
    from balatro_ai.search.state_value import (
        _RUST_BLIND_SAFE, _cached_hand_as_rust_cards, _cached_joker_data,
        _cached_played_hand_types_strs, _PLAYED_STATE_JOKERS,
    )
    # Use the wider rollout blind set since the beam's leaf evaluator
    # is the rollout — same constraints apply.
    try:
        from balatro_ai.search.state_value import _RUST_ROLLOUT_BLIND_SAFE
    except ImportError:
        _RUST_ROLLOUT_BLIND_SAFE = _RUST_BLIND_SAFE
    if state.blind not in _RUST_ROLLOUT_BLIND_SAFE:
        return None
    hand_rust = _cached_hand_as_rust_cards(state, balatro_core)
    if hand_rust is None:
        try:
            hand_rust = [balatro_core.RustCard.from_python(c) for c in state.hand]
        except (AttributeError, TypeError, ValueError):
            return None
    try:
        known_deck_rust = [balatro_core.RustCard.from_python(c) for c in state.known_deck]
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
    try:
        from balatro_ai.rules.hand_evaluator import _metadata_loyalty_remaining
    except ImportError:
        return None
    joker_current_remaining: list[int] = []
    for j in state.jokers:
        if j.name in {"Loyalty Card", "Seltzer", "Yorick"}:
            v = _metadata_loyalty_remaining(j)
            joker_current_remaining.append(int(v) if v is not None else 0)
        else:
            joker_current_remaining.append(0)
    if any(j.name in _PLAYED_STATE_JOKERS for j in state.jokers) or any(
        j.name == "Obelisk" for j in state.jokers
    ):
        played_types_strs = _cached_played_hand_types_strs(state)
    else:
        played_types_strs = []
    from balatro_ai.rules.hand_evaluator import debuffed_suits_for_blind
    debuffed = [s for s in debuffed_suits_for_blind(state.blind) if len(s) == 1]
    pareidolia_active = any(j.name == "Pareidolia" for j in state.jokers)
    joker_vampire_gain = [0.0] * len(joker_names)
    joker_obelisk_should_scale = [False] * len(joker_names)
    try:
        return balatro_core.beam_play_value_native(
            hand_rust, known_deck_rust,
            state.hand_levels, debuffed,
            int(max(0, depth)), int(max(1, config.beam_width)),
            joker_names=joker_names, joker_editions=joker_editions,
            joker_current_plus_mult=joker_plus_mult,
            joker_current_plus_chips=joker_plus_chips,
            joker_current_xmult=joker_xmult,
            joker_current_remaining=joker_current_remaining,
            joker_loyalty_ready=joker_loyalty_ready,
            joker_drivers_active=joker_drivers_active,
            joker_leading_plus_mult=joker_leading_plus_mult,
            joker_leading_plus_chips=joker_leading_plus_chips,
            joker_sell_value=joker_sell_value, joker_rarity=joker_rarity,
            joker_target_suit=joker_target_suit, joker_target_rank=joker_target_rank,
            joker_obelisk_gain=joker_obelisk_gain,
            joker_vampire_gain=joker_vampire_gain,
            joker_obelisk_should_scale=joker_obelisk_should_scale,
            money=int(state.money),
            discards_remaining=int(max(0, state.discards_remaining)),
            hands_remaining=int(max(0, state.hands_remaining)),
            played_hand_types=played_types_strs,
            deck_size=int(max(0, state.deck_size)),
            pareidolia_active=pareidolia_active,
            current_score=int(state.current_score),
            required_score=int(state.required_score),
            # base seed + per-action offset are added INSIDE Rust so the
            # leaf's rollout seeds XoshiroRng::new(config.seed + seed_offset),
            # matching Python `planning_value(.., seed=config.seed + seed_offset)`.
            seed=int(config.seed) & 0xFFFFFFFFFFFFFFFF,
            seed_offset=int(seed_offset) & 0xFFFFFFFFFFFFFFFF,
        )
    except Exception:  # noqa: BLE001
        return None


def _beam_rust_state_kwargs(state: GameState) -> dict | None:
    """Translate a GameState into the shared kwargs both native-beam
    pyfunctions take (hand/deck RustCards + joker arrays + scalars).
    Returns None to bail (no Rust, unsafe blind, conversion failure)."""

    try:
        import balatro_core
    except ImportError:
        return None
    from balatro_ai.search.state_value import (
        _RUST_BLIND_SAFE, _cached_hand_as_rust_cards, _cached_joker_data,
        _cached_played_hand_types_strs, _PLAYED_STATE_JOKERS,
    )
    try:
        from balatro_ai.search.state_value import _RUST_ROLLOUT_BLIND_SAFE
    except ImportError:
        _RUST_ROLLOUT_BLIND_SAFE = _RUST_BLIND_SAFE
    if state.blind not in _RUST_ROLLOUT_BLIND_SAFE:
        return None
    hand_rust = _cached_hand_as_rust_cards(state, balatro_core)
    if hand_rust is None:
        try:
            hand_rust = [balatro_core.RustCard.from_python(c) for c in state.hand]
        except (AttributeError, TypeError, ValueError):
            return None
    try:
        known_deck_rust = [balatro_core.RustCard.from_python(c) for c in state.known_deck]
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
    try:
        from balatro_ai.rules.hand_evaluator import (
            _metadata_loyalty_remaining, debuffed_suits_for_blind,
        )
    except ImportError:
        return None
    joker_current_remaining: list[int] = []
    for j in state.jokers:
        if j.name in {"Loyalty Card", "Seltzer", "Yorick"}:
            v = _metadata_loyalty_remaining(j)
            joker_current_remaining.append(int(v) if v is not None else 0)
        else:
            joker_current_remaining.append(0)
    if any(j.name in _PLAYED_STATE_JOKERS for j in state.jokers) or any(
        j.name == "Obelisk" for j in state.jokers
    ):
        played_types_strs = _cached_played_hand_types_strs(state)
    else:
        played_types_strs = []
    debuffed = [s for s in debuffed_suits_for_blind(state.blind) if len(s) == 1]
    n = len(joker_names)
    return {
        "hand_rust": hand_rust,
        "known_deck_rust": known_deck_rust,
        "hand_levels": state.hand_levels,
        "debuffed": debuffed,
        "joker_names": joker_names,
        "joker_editions": joker_editions,
        "joker_current_plus_mult": joker_plus_mult,
        "joker_current_plus_chips": joker_plus_chips,
        "joker_current_xmult": joker_xmult,
        "joker_current_remaining": joker_current_remaining,
        "joker_loyalty_ready": joker_loyalty_ready,
        "joker_drivers_active": joker_drivers_active,
        "joker_leading_plus_mult": joker_leading_plus_mult,
        "joker_leading_plus_chips": joker_leading_plus_chips,
        "joker_sell_value": joker_sell_value,
        "joker_rarity": joker_rarity,
        "joker_target_suit": joker_target_suit,
        "joker_target_rank": joker_target_rank,
        "joker_obelisk_gain": joker_obelisk_gain,
        "joker_vampire_gain": [0.0] * n,
        "joker_obelisk_should_scale": [False] * n,
        "money": int(state.money),
        "discards_remaining": int(max(0, state.discards_remaining)),
        "hands_remaining": int(max(0, state.hands_remaining)),
        "played_hand_types": played_types_strs,
        "deck_size": int(max(0, state.deck_size)),
        "pareidolia_active": any(j.name == "Pareidolia" for j in state.jokers),
        "current_score": int(state.current_score),
        "required_score": int(state.required_score),
    }


def _try_rust_best_beam_action(
    state: GameState,
    config: HandSearchConfig,
    actions: tuple[Action, ...],
) -> int | None:
    """ROOT-level native beam: one FFI call returns the index (into
    `actions`) of the best root action. Returns None to bail."""

    try:
        import balatro_core
    except ImportError:
        return None
    kw = _beam_rust_state_kwargs(state)
    if kw is None:
        return None
    cand_is_play = [a.action_type == ActionType.PLAY_HAND for a in actions]
    cand_indices = [[int(i) for i in (a.card_indices or ())] for a in actions]
    hand_rust = kw.pop("hand_rust")
    known_deck_rust = kw.pop("known_deck_rust")
    hand_levels = kw.pop("hand_levels")
    debuffed = kw.pop("debuffed")
    try:
        idx = balatro_core.best_beam_action_native(
            hand_rust, known_deck_rust, hand_levels, debuffed,
            cand_is_play, cand_indices,
            int(max(0, config.beam_depth - 1)), int(max(1, config.beam_width)),
            seed=int(config.seed) & 0xFFFFFFFFFFFFFFFF,
            **kw,
        )
    except Exception:  # noqa: BLE001
        return None
    if idx is None or idx < 0 or idx >= len(actions):
        return None
    return int(idx)


def _beam_leaf_value(
    state: GameState,
    *,
    config: HandSearchConfig,
    seed_offset: int,
    memo: dict[tuple[object, ...], float] | None = None,
) -> float:
    samples = max(1, config.beam_leaf_samples)
    memo_key = ("leaf", samples, _beam_state_key(state))
    if memo is not None and memo_key in memo:
        return memo[memo_key]
    value = planning_value(state, samples=samples, seed=config.seed + seed_offset)
    if memo is not None:
        memo[memo_key] = value
    return value


def _beam_terminal(state: GameState) -> bool:
    return (
        state.won
        or state.run_over
        or state.phase in {GamePhase.ROUND_EVAL, GamePhase.RUN_OVER}
        or state.current_score >= state.required_score > 0
        or state.hands_remaining <= 0
        or not state.hand
    )


def _beam_root_actions(state: GameState, config: HandSearchConfig, context) -> tuple[Action, ...]:
    play_actions = tuple(action for action in state.legal_actions if action.action_type == ActionType.PLAY_HAND and action.card_indices)
    discard_actions = tuple(action for action in state.legal_actions if action.action_type == ActionType.DISCARD and action.card_indices)
    ranked_plays = _candidate_play_actions(state, play_actions, context, limit=config.max_play_actions)
    ranked_discards = _candidate_discard_actions(state, discard_actions, limit=config.max_discard_actions)
    return (*ranked_plays, *ranked_discards)


def _beam_future_actions(state: GameState, config: HandSearchConfig) -> tuple[Action, ...]:
    context = _context_from_state(state)
    play_actions = tuple(action for action in state.legal_actions if action.action_type == ActionType.PLAY_HAND and action.card_indices)
    discard_actions = tuple(action for action in state.legal_actions if action.action_type == ActionType.DISCARD and action.card_indices)
    limit = _beam_future_action_limit(config)
    play_scores = _cheap_beam_play_scores(state, play_actions, context)
    best_play_score = max(play_scores.values(), default=0)
    ranked_plays = _cheap_beam_play_actions(state, play_actions, context, limit=limit, play_scores=play_scores)
    ranked_discards = _cheap_beam_discard_actions(state, discard_actions, context, limit=limit, best_play_score=best_play_score)
    ranked = sorted(
        (*ranked_plays, *ranked_discards),
        key=lambda action: _cheap_beam_action_rank(
            state,
            action,
            context,
            play_scores=play_scores,
            best_play_score=best_play_score,
        ),
        reverse=True,
    )
    return tuple(ranked[:limit])


def _beam_future_action_limit(config: HandSearchConfig) -> int:
    return max(1, min(4, config.beam_width + 1))


def _cheap_beam_play_actions(
    state: GameState,
    actions: tuple[Action, ...],
    context,
    *,
    limit: int,
    play_scores: dict[tuple[int, ...], int],
) -> tuple[Action, ...]:
    if not actions:
        return ()
    indexed = tuple(enumerate(actions))
    ranked = sorted(
        indexed,
        key=lambda item: (_cheap_beam_play_rank(state, item[1], context, play_scores=play_scores), -item[0]),
        reverse=True,
    )
    return tuple(action for _, action in ranked[:limit])


def _cheap_beam_discard_actions(
    state: GameState,
    actions: tuple[Action, ...],
    context,
    *,
    limit: int,
    best_play_score: int,
) -> tuple[Action, ...]:
    if not actions:
        return ()
    indexed = tuple(enumerate(actions))
    # Precompute all candidate discard scores in ONE Rust FFI call, then rank.
    discard_scores = _rust_batch_discard_scores(state, actions)
    ranked = sorted(
        indexed,
        key=lambda item: (
            _cheap_beam_discard_rank(
                state, item[1], context,
                best_play_score=best_play_score, discard_scores=discard_scores,
            ),
            -item[0],
        ),
        reverse=True,
    )
    return tuple(action for _, action in ranked[:limit])


def _cheap_beam_action_rank(
    state: GameState,
    action: Action,
    context,
    *,
    play_scores: dict[tuple[int, ...], int],
    best_play_score: int,
) -> float:
    if action.action_type == ActionType.PLAY_HAND:
        return _cheap_beam_play_rank(state, action, context, play_scores=play_scores)
    if action.action_type == ActionType.DISCARD:
        return _cheap_beam_discard_rank(state, action, context, best_play_score=best_play_score)
    return float("-inf")


def _cheap_beam_play_scores(state: GameState, actions: tuple[Action, ...], context) -> dict[tuple[int, ...], int]:
    if not actions:
        return {}
    # Rust batched scorer: ~16× faster than the per-action Python loop
    # on AAAAAAA-style states. Returns RAW scores; we still need
    # boss-adjustment (The Eye / The Mouth) on top. Falls back to
    # Python for any action Rust returns None for.
    rust_scores = _rust_batch_play_scores(state, actions)
    if rust_scores is None:
        return {action.card_indices: _score_play_action(state, action, context) for action in actions}
    out: dict[tuple[int, ...], int] = {}
    needs_boss_adj = state.blind in {"The Eye", "The Mouth"}
    for idx, action in enumerate(actions):
        rs = rust_scores[idx]
        if rs is None:
            out[action.card_indices] = _score_play_action(state, action, context)
            continue
        score = int(rs)
        # Apply The Eye / The Mouth adjustment Python-side. We need
        # the played hand_type for the action; cheapest path is to
        # ask the Python evaluator for it (only when boss adjustment
        # is relevant). For other blinds, raw score is correct.
        if needs_boss_adj and score > 0 and context.played_hand_types:
            # Re-score Python-side to get the hand_type + adjustment
            # in one go. Boss-blind plays are rare so the slow path
            # here is fine.
            out[action.card_indices] = _score_play_action(state, action, context)
        else:
            out[action.card_indices] = score
    return out


def _rust_batch_play_scores(
    state: GameState,
    actions: tuple[Action, ...],
) -> list[int | None] | None:
    """Try the Rust batched action scorer. Returns per-action scores
    or None if the Rust extension isn't loadable / extraction fails.
    Per-action entries may be None to signal Python fallback.

    Bails entirely when the blind is one Rust doesn't handle (The
    Flint halves chips/mult, The Arm decrements level, The Tooth
    money penalty, The Psychic requires 5 cards, The Eye / The
    Mouth / The Plant per-hand-type rules). Same set as the
    Phase 2 wire-in's `_RUST_BLIND_SAFE`.
    """

    try:
        import balatro_core
    except ImportError:
        return None
    from balatro_ai.search.state_value import (
        _RUST_BLIND_SAFE,
        _cached_hand_as_rust_cards, _cached_joker_data,
        _cached_played_hand_types_strs, _PLAYED_STATE_JOKERS,
    )
    if state.blind not in _RUST_BLIND_SAFE:
        return None
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
        return balatro_core.score_play_actions_batch(
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


def _cheap_beam_play_rank(
    state: GameState,
    action: Action,
    context,
    *,
    play_scores: dict[tuple[int, ...], int],
) -> float:
    remaining_score = max(0, state.required_score - state.current_score)
    score = play_scores.get(action.card_indices)
    if score is None:
        score = _score_play_action(state, action, context)
    value = float(score)
    if remaining_score > 0:
        if score >= remaining_score:
            value += 100_000.0 - (len(action.card_indices) * 3.0)
        else:
            value += max(0.0, 10_000.0 / max(1, _estimated_hands_needed(remaining_score, score)))
    return value


def _cheap_beam_discard_rank(
    state: GameState, action: Action, context, *, best_play_score: int,
    discard_scores: dict[tuple[int, ...], float] | None = None,
) -> float:
    remaining_score = max(0, state.required_score - state.current_score)
    pace_score = remaining_score / max(1, state.hands_remaining)
    pressure_bonus = 0.0
    if state.discards_remaining > 0 and best_play_score < pace_score:
        pressure_bonus = 15_000.0
    # discard_scores: optional batch-precomputed _discard_candidate_score values
    # (one Rust FFI call for all candidates) — this rank is the solver's hottest
    # discard path (~310K per-action Python calls/trajectory). Lazy fallback on a
    # cache miss keeps the value byte-identical to the per-action version.
    if discard_scores is not None:
        key = action.card_indices
        cand = discard_scores[key] if key in discard_scores else _discard_candidate_score(state, action)
    else:
        cand = _discard_candidate_score(state, action)
    return pressure_bonus + (cand * 500.0)


def _beam_action_rank(state: GameState, action: Action, context) -> float:
    remaining_score = max(0, state.required_score - state.current_score)
    if action.action_type == ActionType.PLAY_HAND:
        return _play_candidate_rank(state, action, context, remaining_score)
    if action.action_type == ActionType.DISCARD:
        best_score = _best_play_score(state, context)
        pace_score = remaining_score / max(1, state.hands_remaining)
        pressure_bonus = 15_000.0 if state.discards_remaining > 0 and best_score < pace_score else 0.0
        return pressure_bonus + (_discard_candidate_score(state, action) * 500.0)
    return float("-inf")


def _beam_state_key(state: GameState) -> tuple[object, ...]:
    return (
        state.phase,
        state.blind,
        state.required_score,
        state.current_score,
        state.hands_remaining,
        state.discards_remaining,
        state.money,
        state.deck_size,
        state.run_over,
        state.won,
        _beam_hand_multiset_key(state.hand),
        _beam_hand_multiset_key(state.known_deck),
        _jokers_cache_key(state.jokers),
        _freeze_for_cache(state.hand_levels),
        _freeze_for_cache(state.consumables),
        _freeze_for_cache(state.vouchers),
        _freeze_for_cache(state.modifiers),
    )


def _beam_hand_multiset_key(cards: tuple[Card, ...]) -> tuple[object, ...]:
    return tuple(sorted((_card_cache_key(card) for card in cards), key=repr))


def _beam_draw_outcomes(
    state: GameState,
    draw_count: int,
    *,
    config: HandSearchConfig,
    action_index: int,
    seed_offset: int,
) -> tuple[tuple[Card, ...], ...]:
    if draw_count <= 0:
        return ((),)
    model = DeckModel.from_state(state)
    draw_count = min(draw_count, model.total_cards)
    if draw_count <= 0:
        return ((),)
    if draw_count <= config.enumerate_draws_up_to:
        outcomes = model.all_possible_draws(draw_count)
        if outcomes and len(outcomes) <= max(1, config.beam_draw_samples):
            return tuple(tuple(card for card in outcome if isinstance(card, Card)) for outcome in outcomes)

    rng = Random(config.seed + seed_offset + (action_index * 1_000_003))
    return tuple(model.sample_draws(draw_count, rng) for _ in range(max(1, config.beam_draw_samples)))


def _state_after_beam_action(state: GameState, action: Action, drawn_cards: tuple[Card, ...]) -> GameState:
    simulated = _simulate_hand_action(state, action, drawn_cards)
    return with_derived_legal_actions(simulated)


def _simulate_hand_action(state: GameState, action: Action, drawn_cards: tuple[Card, ...]) -> GameState:
    if action.action_type == ActionType.PLAY_HAND:
        return simulate_play(state, action, drawn_cards=drawn_cards)
    if action.action_type == ActionType.DISCARD:
        return simulate_discard(state, action, drawn_cards=drawn_cards)
    raise ValueError(f"Unsupported hand action: {action.action_type.value}")


def _default_value_fn(config: HandSearchConfig) -> ValueFn:
    return lambda state: state_value(state, samples=config.leaf_samples, seed=config.seed) * 100.0


def _candidate_play_actions(
    state: GameState,
    actions: tuple[Action, ...],
    context,
    *,
    limit: int,
) -> tuple[Action, ...]:
    if limit <= 0 or len(actions) <= limit:
        return actions
    best_play = _best_play_action(state, context)
    indexed = tuple(enumerate(actions))
    remaining_score = max(0, state.required_score - state.current_score)
    ranked = sorted(indexed, key=lambda item: (_play_candidate_rank(state, item[1], context, remaining_score), -item[0]), reverse=True)
    selected: list[Action] = []
    if best_play is not None:
        selected.append(best_play)
    for _index, action in ranked:
        if action not in selected:
            selected.append(action)
        if len(selected) >= limit:
            break
    return tuple(selected)


def _play_candidate_rank(state: GameState, action: Action, context, remaining_score: int) -> float:
    score = _score_play_action(state, action, context)
    return _play_candidate_rank_from_score(state, action, context, remaining_score, score)


def _play_candidate_rank_cached(
    state: GameState,
    action: Action,
    context,
    remaining_score: int,
    score_cache: dict[tuple[int, ...], int],
) -> float:
    score = score_cache.get(action.card_indices)
    if score is None:
        score = _score_play_action(state, action, context)
    return _play_candidate_rank_from_score(state, action, context, remaining_score, score)


def _play_candidate_rank_from_score(
    state: GameState,
    action: Action,
    context,
    remaining_score: int,
    score: int,
) -> float:
    value = float(score)
    if remaining_score > 0:
        if score >= remaining_score:
            value += 100_000.0 - (len(action.card_indices) * 3.0)
        else:
            value += max(0.0, 10_000.0 / max(1, _estimated_hands_needed(remaining_score, score)))
    value += _setup_play_bonus(state, action, score=score, remaining_score=remaining_score, context=context) * 25.0
    return value


def _draw_count_for_action(state: GameState, action: Action, *, context) -> int:
    if action.action_type == ActionType.PLAY_HAND and _score_play_action(state, action, context) >= max(0, state.required_score - state.current_score):
        return 0
    return min(len(action.card_indices), DeckModel.from_state(state).total_cards)


def _draw_outcomes(
    state: GameState,
    draw_count: int,
    *,
    config: HandSearchConfig,
    action_index: int,
) -> tuple[tuple[Card, ...], ...]:
    if draw_count <= 0:
        return ((),)
    model = DeckModel.from_state(state)
    draw_count = min(draw_count, model.total_cards)
    if draw_count <= 0:
        return ((),)
    if draw_count <= config.enumerate_draws_up_to:
        outcomes = model.all_possible_draws(draw_count)
        if outcomes and len(outcomes) <= max(1, config.draw_samples):
            return tuple(tuple(card for card in outcome if isinstance(card, Card)) for outcome in outcomes)

    rng = Random(config.seed + (action_index * 1_000_003))
    return tuple(model.sample_draws(draw_count, rng) for _ in range(max(1, config.draw_samples)))


def _play_action_bonus(state: GameState, action: Action, *, context) -> float:
    remaining_score = max(0, state.required_score - state.current_score)
    score = _score_play_action(state, action, context)
    bonus = 0.0
    if remaining_score <= 0 or score >= remaining_score:
        bonus += 26.0
        bonus += max(0, state.hands_remaining - 1) * 1.2
        bonus += max(0, 5 - len(action.card_indices)) * 0.15
    elif score > 0:
        bonus += min(6.0, score / max(1.0, remaining_score) * 6.0)
    bonus += _setup_play_bonus(state, action, score=score, remaining_score=remaining_score, context=context)
    return bonus


def _setup_play_bonus(
    state: GameState,
    action: Action,
    *,
    score: int,
    remaining_score: int,
    context,
) -> float:
    if (
        state.current_score != 0
        or context.played_hand_types
        or state.hands_remaining <= 2
        or remaining_score <= 0
        or len(action.card_indices) != 1
    ):
        return 0.0
    card = state.hand[action.card_indices[0]]
    names = {joker.name for joker in state.jokers}
    bonus = 0.0
    if "DNA" in names:
        bonus += 18.0
    if "Sixth Sense" in names and card.rank == "6" and _has_consumable_room(state):
        bonus += 24.0
    if bonus <= 0.0:
        return 0.0
    after_remaining = max(0, remaining_score - score)
    if after_remaining > 0 and _estimated_hands_needed(after_remaining, max(1, _best_play_score(state, context))) > max(1, state.hands_remaining - 1):
        return 0.0
    return bonus


def _discard_action_bonus(state: GameState, action: Action, *, context) -> float:
    remaining_score = max(0, state.required_score - state.current_score)
    if (
        state.ante == 1
        and state.blind == "Small Blind"
        and not state.jokers
        and getattr(context, "discards_taken", 0) == 0
        and remaining_score > 0
        and _best_play_score(state, context) < remaining_score
    ):
        return 28.0
    if context.played_hand_types:
        return 0.0
    selected = tuple(state.hand[index] for index in action.card_indices if 0 <= index < len(state.hand))
    names = {joker.name for joker in state.jokers}
    bonus = 0.0
    if "Trading Card" in names and len(selected) == 1:
        bonus += 10.0
    if "Burnt Joker" in names:
        bonus += 8.0
    if "Hit the Road" in names:
        bonus += 5.0 * sum(1 for card in selected if card.rank == "J")
    return bonus


def _opening_first_blind_hunt_action(
    state: GameState,
    discard_actions: tuple[Action, ...],
    context,
) -> Action | None:
    if (
        state.ante != 1
        or state.blind != "Small Blind"
        or state.current_score != 0
        or state.hands_remaining < 3
        or state.discards_remaining <= 0
        or state.jokers
        or getattr(context, "discards_taken", 0) != 0
    ):
        return None
    remaining_score = max(0, state.required_score - state.current_score)
    if remaining_score <= 0 or _best_play_score(state, context) >= remaining_score:
        return None
    action = _basic_best_discard_action(state, current_best_score=_best_play_score(state, context), context=context)
    if action is None:
        return None
    for legal_action in discard_actions:
        if legal_action.card_indices == action.card_indices:
            return legal_action
    return action


def _best_play_score(state: GameState, context) -> int:
    best_play = _best_play_action(state, context)
    return _score_play_action(state, best_play, context) if best_play is not None else 0


def _has_consumable_room(state: GameState) -> bool:
    limit = 2
    for key in ("consumable_slots", "consumeable_slots", "tarot_slots"):
        raw = state.modifiers.get(key)
        if isinstance(raw, int | float):
            limit = max(limit, int(raw))
    if "Crystal Ball" in state.vouchers:
        limit += 1
    return len(state.consumables) < limit


def _action_reason(state: GameState, action: Action, context) -> str:
    if action.action_type == ActionType.PLAY_HAND:
        remaining_score = max(0, state.required_score - state.current_score)
        score = _score_play_action(state, action, context)
        if _setup_play_bonus(state, action, score=score, remaining_score=remaining_score, context=context) > 0:
            return "joker_setup hand_search_play"
        return "hand_search_play"
    if action.action_type == ActionType.DISCARD:
        return "hand_search_discard"
    return "hand_search"


def _blind_context():
    from balatro_ai.bots.basic_strategy.hand_models import _BlindContext

    return _BlindContext()


def _context_from_state(state: GameState):
    from balatro_ai.bots.basic_strategy.hand_models import _BlindContext
    from balatro_ai.bots.basic_strategy.play_scoring import _played_hand_types_this_round

    discards_taken = 0
    for key in ("round_discards_used", "discards_used", "discards_used_this_round"):
        if key in state.modifiers:
            try:
                discards_taken = int(state.modifiers.get(key) or 0)
            except (TypeError, ValueError):
                discards_taken = 0
            break
    return _BlindContext(played_hand_types=_played_hand_types_this_round(state), discards_taken=discards_taken)


def _best_play_action(state: GameState, context):
    from balatro_ai.bots.basic_strategy.play_scoring import _best_play_action as basic_best_play_action

    return basic_best_play_action(state, context)


def _score_play_action(state: GameState, action: Action, context=None) -> int:
    from balatro_ai.bots.basic_strategy.play_scoring import _score_play_action as basic_score_play_action

    return basic_score_play_action(state, action, context)


def _estimated_hands_needed(remaining_score: int, score: int | float) -> int:
    from balatro_ai.bots.basic_strategy.discard_policy import _estimated_hands_needed as basic_estimated_hands_needed

    return basic_estimated_hands_needed(remaining_score, score)


def _basic_best_discard_action(state: GameState, *, current_best_score: int, context):
    from balatro_ai.bots.basic_strategy.discard_policy import _best_discard_action as basic_best_discard_action

    return basic_best_discard_action(state, current_best_score=current_best_score, context=context)


def _beam_action_reason(action: Action) -> str:
    if action.action_type == ActionType.PLAY_HAND:
        return "whole_blind_beam_play"
    if action.action_type == ActionType.DISCARD:
        return "whole_blind_beam_discard"
    return "whole_blind_beam"


def _annotated_action(action: Action, *, search_value: float, reason: str, search: str = "hand_expectimax") -> Action:
    return Action(
        action.action_type,
        card_indices=action.card_indices,
        target_id=action.target_id,
        amount=action.amount,
        metadata={
            **action.metadata,
            "search": search,
            "search_value": round(search_value, 6),
            "reason": action.metadata.get("reason", reason),
        },
    )
