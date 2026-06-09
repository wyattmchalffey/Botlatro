"""Leaf-state value estimates for Phase 7 search."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from random import Random
from threading import local

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import Card, GamePhase, GameState, Joker
from balatro_ai.rules.hand_evaluator import _prepare_joker_evaluation_context, debuffed_suits_for_blind, evaluate_played_cards
from balatro_ai.search.deck_model import DeckModel
from balatro_ai.search.forward_sim import simulate_cash_out, simulate_discard, simulate_play


@dataclass(frozen=True, slots=True)
class RolloutConfig:
    samples: int = 64
    seed: int = 0


_STATE_VALUE_CACHE_LOCAL = local()


@contextmanager
def state_value_cache_scope():
    """Cache expensive state-value helpers for one bot decision."""

    previous_cache = _current_state_value_cache()
    _STATE_VALUE_CACHE_LOCAL.cache = {}
    try:
        yield
    finally:
        if previous_cache is None:
            try:
                del _STATE_VALUE_CACHE_LOCAL.cache
            except AttributeError:
                pass
        else:
            _STATE_VALUE_CACHE_LOCAL.cache = previous_cache


def _current_state_value_cache() -> dict[tuple[object, ...], object] | None:
    return getattr(_STATE_VALUE_CACHE_LOCAL, "cache", None)


def _state_identity_cached_value(kind: str, state: GameState, extra: tuple[object, ...], factory):
    cache = _current_state_value_cache()
    if cache is None:
        return factory()
    key = (kind, id(state), *extra)
    cached = cache.get(key)
    if cached is not None and cached[0] is state:
        return cached[1]
    value = factory()
    cache[key] = (state, value)
    return value


def _identity_cached(kind: str, obj, factory):
    cache = _current_state_value_cache()
    if cache is None:
        return factory()
    key = (kind, id(obj))
    cached = cache.get(key)
    if cached is not None and cached[0] is obj:
        return cached[1]
    value = factory()
    cache[key] = (obj, value)
    return value


def _state_content_cached_value(kind: str, state: GameState, extra: tuple[object, ...], factory):
    cache = _current_state_value_cache()
    if cache is None:
        return factory()
    key = ("content", kind, _search_state_cache_key(state), *extra)
    if key not in cache:
        cache[key] = factory()
    return cache[key]


def clear_probability(state: GameState, *, samples: int = 64, seed: int = 0) -> float:
    """Estimate the chance that greedy play clears the current blind."""

    return _state_identity_cached_value(
        "clear_probability",
        state,
        (samples, seed),
        lambda: _clear_probability_uncached(state, samples=samples, seed=seed),
    )


def _clear_probability_uncached(state: GameState, *, samples: int, seed: int) -> float:
    """Estimate the chance that greedy play clears the current blind."""

    if state.run_over or state.phase == GamePhase.RUN_OVER:
        return 0.0
    if state.phase == GamePhase.ROUND_EVAL:
        return 1.0
    if state.required_score <= 0 or state.current_score >= state.required_score:
        return 1.0
    if state.hands_remaining <= 0:
        return 0.0
    if not state.hand:
        return 0.0

    # Rust fast path: runs all N rollouts internally without crossing
    # the Python boundary per simulate_play. Bails on complex jokers,
    # boss blinds, Glass cards, blue seals. May give a slightly
    # different clear-probability estimate than Python (internal RNG
    # differs from random.Random) — that's acceptable because this is
    # an estimator and decisions remain quality-equivalent.
    #
    # The Rust rollout NOW models discards (botlatro-core search/rollout.rs:
    # _should_rollout_discard / _rollout_discard_action ports), so the earlier
    # discard-blindness (returning ~0 on discard-clearable blinds) is fixed and
    # the discard-aware Python fallback that used to live here is no longer
    # needed. Verified: noise-averaged Rust vs Python discard-aware rollout has
    # 0 systematic underestimates; remaining gap is xoshiro-vs-random.Random noise.
    rust_result = _try_rust_clear_probability(state, samples, seed)
    if rust_result is not None:
        return rust_result

    total_samples = max(1, samples)
    rng = Random(seed)
    clears = 0
    for _ in range(total_samples):
        if _greedy_rollout_clears(state, rng):
            clears += 1
    return clears / total_samples


# Wider blind set for the rollout estimator. The rollout is a noisy
# clear-probability estimate, so we can include boss blinds whose
# scoring effects are either absent (Manacle: just hand-size, score
# math unchanged) or stochastic in a way the deterministic rollout
# already approximates by not modeling them (Wheel: 1-in-7 face-down
# per card — rollout overestimates by ignoring; Psychic: must play
# 5 cards — rollout's argmax usually picks 5-card hands anyway).
# Hook stays bailed because mid-play discards meaningfully shift
# which cards get scored.
_RUST_ROLLOUT_BLIND_SAFE: frozenset[str] = frozenset({
    "", "Small Blind", "Big Blind",
    # Suit-debuff bosses
    "The Club", "The Goad", "The Head", "The Window",
    # Non-scoring bosses whose impact is small in rollout context.
    "The Manacle", "The Wheel", "The Psychic",
    # NOTE: The Wall is deliberately NOT here. It is rollout-SAFE (parity
    # clean, no bias), but a 24-seed process_time A/B showed un-bailing it
    # gives ~0% speedup while perturbing trajectories (noisy-estimator
    # decision flips: score 11650->10465). The earlier 3-seed profile
    # over-sampled Wall states; on a representative seed population Wall
    # rollouts are a tiny fraction of compute. Not worth the data drift.
})


def _try_rust_clear_probability(state: GameState, samples: int, seed: int) -> float | None:
    """Bridge to Rust `clear_probability_native`. Returns the clear
    fraction (0.0-1.0) or None to bail to Python."""

    try:
        import balatro_core
    except ImportError:
        return None
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
    # joker_current_remaining for Loyalty Card / Seltzer (drives
    # in-rollout joker scaling updates).
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
    played_count_this = sum(1 for s in played_types_strs if s == "")  # placeholder
    # Initial state: no played hand_types this rollout yet — we use
    # the round's existing played_hand_types as the starting list.
    debuffed = [s for s in debuffed_suits_for_blind(state.blind) if len(s) == 1]
    pareidolia_active = any(j.name == "Pareidolia" for j in state.jokers)
    # Vampire gain + Obelisk should_scale: we bail when Vampire is
    # present (SIMPLE_BAIL_JOKERS), so all-zeros is safe. Obelisk
    # works inside Rust based on played_count_max_other_hand_type
    # which we recompute internally per rollout step.
    joker_vampire_gain = [0.0] * len(joker_names)
    joker_obelisk_should_scale = [False] * len(joker_names)

    try:
        result = balatro_core.clear_probability_native(
            hand_rust, known_deck_rust,
            state.hand_levels, debuffed,
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
            samples=int(max(1, samples)),
            seed=int(seed) & 0xFFFFFFFFFFFFFFFF,
        )
    except Exception:  # noqa: BLE001
        return None
    return result


def future_value(state: GameState) -> float:
    """Small monotonic build-strength estimate for search leaves."""

    return _state_identity_cached_value("future_value", state, (), lambda: _future_value_uncached(state))


def headroom_value(state: GameState) -> float:
    """Unclamped build-strength estimate for planners that need surplus signal."""

    return _state_identity_cached_value("headroom_value", state, (), lambda: _headroom_value_uncached(state))


def _future_value_uncached(state: GameState) -> float:
    """Small monotonic build-strength estimate for search leaves."""

    state = _cash_out_leaf_state(state)
    if state.run_over or state.phase == GamePhase.RUN_OVER:
        return 0.0
    if state.won:
        return 1.0
    best_score = _best_immediate_score(state)
    score_target = max(1, state.required_score - state.current_score)
    score_component = min(1.0, best_score / score_target)
    money_component = min(1.0, max(0, state.money) / 50)
    joker_component = min(1.0, len(state.jokers) / 5)
    return _clamp01((score_component * 0.7) + (money_component * 0.2) + (joker_component * 0.1))


def _headroom_value_uncached(state: GameState) -> float:
    """Build-strength score that preserves useful surplus above a bare clear."""

    state = _cash_out_leaf_state(state)
    if state.run_over or state.phase == GamePhase.RUN_OVER:
        return 0.0
    if state.won:
        return 2.0
    best_score = _best_immediate_score(state)
    score_target = max(1, state.required_score - state.current_score)
    score_component = min(3.0, best_score / score_target)
    money_component = min(1.5, max(0, state.money) / 40)
    joker_component = min(1.25, len(state.jokers) / 5)
    hand_component = min(1.0, max(0, state.hands_remaining) / 4)
    return (score_component * 0.60) + (money_component * 0.20) + (joker_component * 0.12) + (hand_component * 0.08)


def state_value(state: GameState, *, samples: int = 64, seed: int = 0) -> float:
    """Combine current-blind survival with a conservative future-strength score."""

    return _state_identity_cached_value(
        "state_value",
        state,
        (samples, seed),
        lambda: _state_value_uncached(state, samples=samples, seed=seed),
    )


def _state_value_uncached(state: GameState, *, samples: int, seed: int) -> float:
    """Combine current-blind survival with a conservative future-strength score."""

    if state.won:
        return 1.0
    if state.run_over or state.phase == GamePhase.RUN_OVER:
        return 0.0
    clear = clear_probability(state, samples=samples, seed=seed)
    future = future_value(state)
    return _clamp01((clear * 0.8) + (clear * future * 0.2))


def planning_value(state: GameState, *, samples: int = 16, seed: int = 0) -> float:
    """Value for bounded search planners, preserving clear headroom and economy."""

    return _state_identity_cached_value(
        "planning_value",
        state,
        (samples, seed),
        lambda: _planning_value_uncached(state, samples=samples, seed=seed),
    )


def _planning_value_uncached(state: GameState, *, samples: int, seed: int) -> float:
    """Value for bounded search planners, preserving clear headroom and economy."""

    if state.won:
        return 2.0
    if state.run_over or state.phase == GamePhase.RUN_OVER:
        return 0.0

    clear = clear_probability(state, samples=samples, seed=seed)
    headroom = headroom_value(state)
    if state.phase == GamePhase.ROUND_EVAL or state.current_score >= state.required_score:
        # CLEARED the blind — must value strictly ABOVE every non-cleared state.
        # The non-cleared branch below maxes at ~1.58 (headroom <= 2.33), and a
        # near-cleared state has an INFLATED headroom score-component
        # (best_play / tiny_remaining -> capped 3.0). With the old base of 1.0
        # here, "almost cleared with a strong hand in hand" (~1.39) outscored
        # "actually cleared" (~1.03), so the solver deferred its good hands,
        # played junk to keep the strong hand "available", and ran out of hands
        # — the ~1% data-gen winrate / ante-1 collapse (fixed 2026-05-30).
        # Base 1.75 (> 1.58 non-cleared max) makes clearing dominate; headroom
        # still tie-breaks bigger/better clears. (won-run stays 2.0.)
        return 1.75 + min(0.25, headroom * 0.25)

    progress = min(1.0, max(0.0, state.current_score / max(1, state.required_score)))
    return (clear * 1.0) + (clear * headroom * 0.25) + ((1.0 - clear) * progress * 0.15)


def _greedy_rollout_clears(state: GameState, rng: Random) -> bool:
    current = state
    while current.current_score < current.required_score and current.hands_remaining > 0 and current.hand:
        action = _best_greedy_play_action(current)
        if action is None:
            return False
        if _should_rollout_discard(current, action):
            discard_action = _rollout_discard_action(current, action)
            if discard_action is not None:
                deck_model = _deck_model_for_state(current)
                draw_count = min(len(discard_action.card_indices), deck_model.total_cards)
                drawn_cards = deck_model.sample_draws(draw_count, rng) if draw_count > 0 else ()
                current = simulate_discard(current, discard_action, drawn_cards=drawn_cards)
                continue
        deck_model = _deck_model_for_state(current)
        draw_count = min(len(action.card_indices), deck_model.total_cards)
        drawn_cards = deck_model.sample_draws(draw_count, rng) if draw_count > 0 else ()
        current = simulate_play(current, action, drawn_cards=drawn_cards)
        if current.phase == GamePhase.ROUND_EVAL or current.current_score >= current.required_score:
            return True
        if current.phase == GamePhase.RUN_OVER or current.run_over:
            return False
    return current.current_score >= current.required_score


def _should_rollout_discard(state: GameState, best_play: Action) -> bool:
    if state.discards_remaining <= 0 or _deck_model_for_state(state).total_cards <= 0:
        return False
    best_score = _score_action(state, best_play)
    remaining_score = max(0, state.required_score - state.current_score)
    if best_score >= remaining_score:
        return False
    pace_score = remaining_score / max(1, state.hands_remaining)
    return state.hands_remaining <= 1 or best_score < pace_score


def _rollout_discard_action(state: GameState, best_play: Action) -> Action | None:
    protected = set(best_play.card_indices)
    candidates = tuple(index for index in range(len(state.hand)) if index not in protected)
    if not candidates:
        return None
    discard_limit = min(5, len(candidates), _deck_model_for_state(state).total_cards)
    if discard_limit <= 0:
        return None
    ordered = sorted(candidates, key=lambda index: (_rollout_discard_rank_value(state.hand[index]), index))
    return Action(ActionType.DISCARD, card_indices=tuple(ordered[:discard_limit]))


def _deck_model_for_state(state: GameState) -> DeckModel:
    return _state_identity_cached_value("deck_model", state, (), lambda: DeckModel.from_state(state))


def _cash_out_leaf_state(state: GameState) -> GameState:
    if state.phase != GamePhase.ROUND_EVAL:
        return state
    try:
        return simulate_cash_out(state, next_to_do_targets=_to_do_list_cash_out_targets(state))
    except ValueError:
        return state


def _to_do_list_cash_out_targets(state: GameState) -> tuple[str, ...]:
    targets: list[str] = []
    for joker in state.jokers:
        if joker.name != "To Do List":
            continue
        targets.append(_to_do_list_current_target(joker))
    return tuple(targets)


def _to_do_list_current_target(joker: object) -> str:
    metadata = getattr(joker, "metadata", {})
    if isinstance(metadata, dict):
        for source in _metadata_sources(metadata):
            for key in ("target_hand", "to_do_poker_hand", "poker_hand", "hand_type"):
                value = source.get(key)
                if value:
                    return str(value)
            value = source.get("value")
            if isinstance(value, dict):
                effect = value.get("effect")
                if isinstance(effect, str):
                    target = _to_do_target_from_effect(effect)
                    if target:
                        return target
    return "High Card"


def _metadata_sources(metadata: dict[str, object]) -> tuple[dict[str, object], ...]:
    sources: list[dict[str, object]] = [metadata]
    for key in ("ability", "config", "extra"):
        value = metadata.get(key)
        if isinstance(value, dict):
            sources.append(value)
            nested_extra = value.get("extra")
            if isinstance(nested_extra, dict):
                sources.append(nested_extra)
    return tuple(sources)


def _to_do_target_from_effect(effect: str) -> str | None:
    lowered = effect.lower()
    hand_names = (
        "Flush Five",
        "Flush House",
        "Five of a Kind",
        "Straight Flush",
        "Four of a Kind",
        "Full House",
        "Three of a Kind",
        "Two Pair",
        "Straight",
        "Flush",
        "Pair",
        "High Card",
    )
    for name in hand_names:
        if name.lower() in lowered:
            return name
    return None


def _best_greedy_play_action(state: GameState) -> Action | None:
    cache = _current_state_value_cache()
    if cache is None:
        return _best_greedy_play_action_uncached(state)
    key = ("content", "best_greedy_play_action", _search_state_cache_key(state))
    if key in cache:
        selected_key = cache[key]
        return _play_action_for_card_multiset(state.hand, selected_key) if isinstance(selected_key, tuple) else None

    action = _best_greedy_play_action_uncached(state)
    cache[key] = _selected_action_multiset_key(state, action)
    return action


def _best_greedy_play_action_uncached(state: GameState) -> Action | None:
    # Rust fast path: combined enumerate+score+argmax in one FFI call
    # (~16x faster than the Python per-action loop; this variant also
    # eliminates the Python action-enumeration tuple allocations).
    # Called inside greedy rollouts — the dominant rollout cost.
    rust_best = _try_rust_best_play_action(state)
    if rust_best is not None:
        indices, _score = rust_best
        return Action(ActionType.PLAY_HAND, card_indices=tuple(indices))

    # Falls back to the batched scorer + Python argmax when the
    # combined call bails entirely (Wild cards, unsupported jokers,
    # boss-adjusted blinds).
    actions = _play_actions_for_hand(state.hand)
    if not actions:
        return None
    rust_scores = _try_rust_batch_scores(state, actions)
    if rust_scores is not None:
        best_action: Action | None = None
        best_score = -1
        blind_name = _effective_blind_name(state)
        debuffed_suits = debuffed_suits_for_blind(blind_name)
        for action, rs in zip(actions, rust_scores, strict=False):
            score = int(rs) if rs is not None else _score_action(
                state, action, blind_name=blind_name, debuffed_suits=debuffed_suits,
            )
            if score > best_score:
                best_score = score
                best_action = action
        return best_action
    # Python fallback (boss blinds + unsupported jokers).
    best_action = None
    best_score = -1
    joker_context = _prepare_joker_evaluation_context(state.jokers)
    blind_name = _effective_blind_name(state)
    debuffed_suits = debuffed_suits_for_blind(blind_name)
    for action in actions:
        score = _score_action(state, action, joker_context=joker_context, blind_name=blind_name, debuffed_suits=debuffed_suits)
        if score > best_score:
            best_score = score
            best_action = action
    return best_action


def _try_rust_best_play_action(state: GameState) -> tuple[list[int], int] | None:
    """Bridge to Rust `best_play_action`. Returns (indices, score) or
    None to bail. Same restrictions as `_try_rust_batch_scores`."""

    try:
        import balatro_core
    except ImportError:
        return None
    if state.blind not in _RUST_BLIND_SAFE:
        return None
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
    try:
        return balatro_core.best_play_action(
            hand_rust, state.hand_levels, debuffed,
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


def _best_immediate_score(state: GameState) -> int:
    return _state_content_cached_value("best_immediate_score", state, (), lambda: _best_immediate_score_uncached(state))


def _best_immediate_score_uncached(state: GameState) -> int:
    actions = _play_actions_for_hand(state.hand)
    if not actions:
        return 0
    rust_scores = _try_rust_batch_scores(state, actions)
    if rust_scores is not None:
        best = 0
        blind_name = _effective_blind_name(state)
        debuffed_suits = debuffed_suits_for_blind(blind_name)
        for action, rs in zip(actions, rust_scores, strict=False):
            score = int(rs) if rs is not None else _score_action(
                state, action, blind_name=blind_name, debuffed_suits=debuffed_suits,
            )
            if score > best:
                best = score
        return best
    # Python fallback.
    best = 0
    joker_context = _prepare_joker_evaluation_context(state.jokers)
    blind_name = _effective_blind_name(state)
    debuffed_suits = debuffed_suits_for_blind(blind_name)
    for action in actions:
        best = max(best, _score_action(state, action, joker_context=joker_context, blind_name=blind_name, debuffed_suits=debuffed_suits))
    return best


def _try_rust_batch_scores(
    state: GameState,
    actions: tuple[Action, ...],
) -> list[int | None] | None:
    """Bridge to Rust `score_play_actions_batch` from state_value.
    Returns per-action raw scores or None to bail. Restricted to
    `_RUST_BLIND_SAFE` blinds since the Rust path doesn't apply
    boss-blind scoring effects (Flint, Arm, Eye, Mouth, etc.)."""

    try:
        import balatro_core
    except ImportError:
        return None
    if state.blind not in _RUST_BLIND_SAFE:
        return None
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


def _play_actions_for_hand(hand: tuple[Card, ...]) -> tuple[Action, ...]:
    return _play_actions_for_hand_size(len(hand))


@lru_cache(maxsize=16)
def _play_actions_for_hand_size(hand_size: int) -> tuple[Action, ...]:
    actions: list[Action] = []
    for size in range(1, min(5, hand_size) + 1):
        for indices in combinations(range(hand_size), size):
            actions.append(Action(ActionType.PLAY_HAND, card_indices=indices))
    return tuple(actions)


def _score_action(
    state: GameState,
    action: Action,
    *,
    joker_context=None,
    blind_name: str | None = None,
    debuffed_suits: set[str] | frozenset[str] | None = None,
) -> int:
    # Profiling note (2026-05-26): tried wrapping this in
    # `_state_content_cached_value` for the solver beam. Hit rate was
    # 0.2% — states are too content-unique within a single decision
    # (each rollout step changes hand+deck_size). Removed; the right
    # fix is to make `_score_action_uncached` faster, not to avoid
    # calls to it. Tier 2 #4 (Cython port of evaluate_played_cards)
    # addresses this directly.
    return _score_action_uncached(
        state,
        action,
        joker_context=joker_context,
        blind_name=blind_name,
        debuffed_suits=debuffed_suits,
    )


def _score_action_uncached(
    state: GameState,
    action: Action,
    *,
    joker_context=None,
    blind_name: str | None = None,
    debuffed_suits: set[str] | frozenset[str] | None = None,
) -> int:
    selected_indices = set(action.card_indices)
    selected = tuple(card for index, card in enumerate(state.hand) if index in selected_indices)
    held = tuple(card for index, card in enumerate(state.hand) if index not in selected_indices)
    effective_blind = _effective_blind_name(state) if blind_name is None else blind_name

    # Native (Rust) fast path. Activates when:
    #   - No jokers (joker effects need Python's full _effect_adjustments)
    #   - Blind has no score-multiplier override (most non-boss blinds)
    #   - Played cards have no Stone/Wild enhancement (Rust bails internally)
    # Returns int score identical to what the Python path computes,
    # or None to fall through. Parity verified by
    # `tests/test_rust_score_action_parity.py` on canonical-seed
    # state corpus before this hook went live. We pass card_indices
    # so the helper can use the cached whole-hand RustCard conversion.
    rust_score = _try_rust_evaluate_simple(
        state, action.card_indices, effective_blind, debuffed_suits,
    )
    if rust_score is not None:
        return rust_score

    try:
        evaluation = evaluate_played_cards(
            selected,
            state.hand_levels,
            debuffed_suits=debuffed_suits_for_blind(effective_blind) if debuffed_suits is None else debuffed_suits,
            blind_name=effective_blind,
            jokers=state.jokers,
            discards_remaining=state.discards_remaining,
            hands_remaining=state.hands_remaining,
            held_cards=held,
            deck_size=state.deck_size,
            money=state.money,
            _joker_context=joker_context,
        )
    except ValueError:
        return 0
    return evaluation.score


# Blinds whose score math is the same vanilla (chips * mult) formula
# Rust implements. We bail to Python for any blind not in this set.
#
# Includes:
# - "Small Blind", "Big Blind", "" — vanilla.
# - Suit-debuff bosses: The Club/Goad/Head/Window. Math is unchanged
#   except one suit is debuffed; we already pass debuffed_suits to
#   Rust, so these activate Rust on most mid-run plays.
#
# Bosses with effects beyond suit debuffs (Hook, Pillar, Plant, Wall,
# Mark, Manacle, Ox, etc.) bail to Python because Rust's
# `evaluate_simple` doesn't implement their score/play-rule overrides.
_RUST_BLIND_SAFE = frozenset({
    "", "Small Blind", "Big Blind",
    "The Club", "The Goad", "The Head", "The Window",
    # Score-neutral bosses (audited 2026-05-28 against
    # evaluate_played_cards: none touch chips/mult/score — their
    # effects are money / hands / forward-sim / card-debuff). Parity
    # verified Rust==Python on 495 samples incl. Bull/Bootstraps
    # money jokers + debuffed cards. The Ox alone was ~84% of this
    # path's Python fallbacks. NOT The Tooth (its $/card deduction
    # feeds money-scaling jokers within the same eval) and NOT
    # Flint/Psychic/Eye/Mouth (real score effects).
    "The Ox", "The Needle", "The Wheel", "The Manacle",
    "The Hook", "The Mark", "The Pillar",
    # Pure target-multiplier bosses (added 2026-05-29): The Wall is ×2
    # required score, Violet Vessel is ×3. Neither touches chips/mult/score
    # or which cards are drawn/scored — the multiplier is data-driven boss
    # metadata applied to required_score, which the Rust rollout already
    # reads from state (clear_probability_native(required_score=...)). The
    # name "The Wall" appears nowhere in evaluate_played_cards. The Wall was
    # ~51% of this trajectory's Rust-eval bails on AAAAAAA.
    "The Wall", "Violet Vessel",
    # The Water (added 2026-05-29): start with 0 discards. The ONLY effect is
    # discards_remaining=0, which the rollout already receives via
    # clear_probability_native(discards_remaining=...). No draw/face-down/score
    # change, so the Rust rollout models it exactly. (The House/Fish/Serpent are
    # deliberately kept OUT of this rollout set — the Rust rollout doesn't model
    # their face-down/draw-count mechanics — but ARE in the evaluator set.)
    "The Water",
})

# Jokers whose effect needs run-state Rust can't currently compute.
# (Card Sharp / Supernova used to be here — now they get the
# played-hand state via the ctx fields populated below.)
_RUST_JOKERS_NEEDING_PLAYED_STATE: frozenset[str] = frozenset()

# Jokers that consult played-hand-types this round. We compute the
# `played_types_strs` argument ONLY when one of these is present, so
# the common case (run without Card Sharp / Supernova) doesn't pay
# the per-call lookup cost.
_PLAYED_STATE_JOKERS = frozenset({"Card Sharp", "Supernova"})


def _try_rust_evaluate_simple(
    state: GameState,
    card_indices: tuple[int, ...],
    effective_blind: str,
    debuffed_suits: set[str] | frozenset[str] | None,
) -> int | None:
    """Try to score the selected indices via the Rust fast path.

    Returns the score (int) on success, None to fall through to
    the Python implementation. The decision is deliberately
    conservative — when in doubt, fall back to Python.

    Joker support: passes joker NAMES (no metadata) to Rust. Rust
    bails (returns None) if any joker isn't in its supported set —
    that's the safety net that prevents silent miscalculation when
    a complex joker (Photograph, Vampire, etc.) is in the run.
    """

    # Eligibility: vanilla blind, non-empty selection. Jokers are no
    # longer a blanket blocker — Rust will bail on unsupported names.
    # Joker editions are now handled natively (Phase 2d batch 6).
    if effective_blind == "The Psychic":
        # The Psychic: a play with != 5 cards scores 0; a 5-card play scores
        # normally (Psychic adds no scoring effect of its own). Mirror
        # rust_bridge's short-circuit so this per-action scorer stops bailing
        # to Python on Psychic — a very common ante-1/2 boss that was ~97% of
        # this path's fallbacks. Matches the Python evaluator exactly, so it's
        # behavior-preserving. (Psychic stays OUT of _RUST_BLIND_SAFE, so the
        # whole-rollout clear_probability_native still bails to Python.)
        if not card_indices:
            return None
        if len(card_indices) != 5:
            return 0
        # 5-card Psychic play: fall through to the normal Rust scoring path.
    elif effective_blind not in _RUST_BLIND_SAFE:
        return None
    if not card_indices:
        return None
    # Bail when Rust support exists but needs ctx fields we don't yet
    # plumb through. Solver runs that don't have these jokers (most
    # opening-blind decisions) are unaffected.
    if any(j.name in _RUST_JOKERS_NEEDING_PLAYED_STATE for j in state.jokers):
        return None

    try:
        import balatro_core
    except ImportError:
        return None

    hand_rust = _cached_hand_as_rust_cards(state, balatro_core)
    if hand_rust is None:
        try:
            rs_cards = [
                balatro_core.RustCard.from_python(state.hand[i])
                for i in card_indices
            ]
            # Held cards = the rest of the hand (not in selected).
            selected_set = set(card_indices)
            rs_held = [
                balatro_core.RustCard.from_python(state.hand[i])
                for i in range(len(state.hand))
                if i not in selected_set
            ]
        except (ValueError, AttributeError):
            return None
    else:
        try:
            rs_cards = [hand_rust[i] for i in card_indices]
            selected_set = set(card_indices)
            rs_held = [hand_rust[i] for i in range(len(hand_rust)) if i not in selected_set]
        except IndexError:
            return None

    if debuffed_suits is None:
        debuffed = debuffed_suits_for_blind(effective_blind)
    else:
        debuffed = debuffed_suits
    debuffed_list = [s for s in debuffed if len(s) == 1]
    # Joker metadata is extracted ONCE per state.jokers reference
    # via identity-cache. state.jokers is a frozen tuple — same
    # id means same content means same extracted values. Saves
    # ~25 Python helper calls per _score_action invocation when
    # multiple jokers are held.
    (joker_names, joker_editions, joker_current_plus_mult,
     joker_current_plus_chips, joker_current_xmult,
     joker_loyalty_ready, joker_drivers_active,
     joker_leading_plus_mult, joker_leading_plus_chips,
     joker_sell_value, joker_rarity_list,
     joker_target_suit, joker_target_rank,
     joker_obelisk_gain) = _cached_joker_data(state)

    # Played-hand state for Card Sharp + Supernova. Only compute when
    # one of those jokers is actually present — the per-call lookup
    # adds measurable overhead across ~70K evals/decision and is a
    # net loss when those jokers aren't held (the dominant case).
    if any(j.name in _PLAYED_STATE_JOKERS for j in state.jokers):
        played_types_strs = _cached_played_hand_types_strs(state)
    else:
        played_types_strs = []

    # joker_slot_limit: Python reads state.modifiers["joker_slot_limit"]
    # with default 5. Match the same default.
    slot_limit_raw = state.modifiers.get("joker_slot_limit", 5)
    try:
        joker_slot_limit = int(slot_limit_raw) if slot_limit_raw is not None else 5
    except (TypeError, ValueError):
        joker_slot_limit = 5

    try:
        result = balatro_core.evaluate_simple_with_levels(
            rs_cards,
            state.hand_levels,
            debuffed_suits=debuffed_list,
            joker_names=joker_names,
            joker_editions=joker_editions,
            held_cards=rs_held,
            joker_current_plus_mult=joker_current_plus_mult,
            joker_current_plus_chips=joker_current_plus_chips,
            joker_current_xmult=joker_current_xmult,
            joker_loyalty_ready=joker_loyalty_ready,
            joker_drivers_active=joker_drivers_active,
            joker_leading_plus_mult=joker_leading_plus_mult,
            joker_leading_plus_chips=joker_leading_plus_chips,
            joker_sell_value=joker_sell_value,
            joker_rarity=joker_rarity_list,
            joker_target_suit=joker_target_suit,
            joker_target_rank=joker_target_rank,
            joker_obelisk_gain=joker_obelisk_gain,
            money=int(state.money),
            joker_slot_limit=joker_slot_limit,
            discards_remaining=int(max(0, state.discards_remaining)),
            hands_remaining=int(max(0, state.hands_remaining)),
            played_hand_types=played_types_strs,
            deck_size=int(max(0, state.deck_size)),
        )
    except Exception:  # noqa: BLE001 — never crash the bot path
        return None
    if result is None:
        return None
    _chips, _mult, score, _ht = result
    return int(score)


_RARITY_TO_INT = {"common": 0, "uncommon": 1, "rare": 2, "legendary": 3}


def _cached_joker_data(state: GameState):
    """Identity-cached extraction of joker names/editions/metadata.

    state.jokers is a frozen tuple — same id() means same content.
    Returns a 14-tuple of parallel lists used by the Rust wire-in:
      (names, editions, plus_mult, plus_chips, xmult,
       loyalty_ready, drivers_active,
       leading_plus_mult, leading_plus_chips,
       sell_value, rarity,
       target_suit, target_rank, obelisk_gain)
    Computed at most once per (state.jokers, scope) pair.
    """

    cache = _current_state_value_cache()
    jokers = state.jokers
    if cache is not None:
        key = ("rust_joker_data", id(jokers))
        entry = cache.get(key)
        if entry is not None and entry[0] is jokers:
            return entry[1]
    from balatro_ai.rules.hand_evaluator import (
        _joker_current_plus, _joker_current_xmult,
        _joker_leading_plus,
        _loyalty_card_ready, _drivers_license_active,
        _joker_rarity, _joker_target_suit, _joker_target_rank_suit,
        _obelisk_xmult_gain, _effective_ability_joker_indices,
    )
    # Blueprint/Brainstorm copy-effect resolution: returns the index
    # of the joker whose EFFECT logic and METADATA each slot should
    # use. The PHYSICAL edition + rarity + sell_value stay at the
    # original slot — only the name + scaling metadata get copied.
    ability_indices = _effective_ability_joker_indices(jokers)
    ability = [jokers[ability_indices[i]] for i in range(len(jokers))]
    # `names` and the metadata fields read from the COPIED joker
    # (ability_jokers); editions/rarity/sell read from the PHYSICAL
    # joker. This mirrors Python's `_effect_adjustments` ability pass
    # which uses `ability_joker.name` but `physical_joker.edition`.
    names = [a.name for a in ability]
    editions = [getattr(j, "edition", None) for j in jokers]
    plus_mult = [int(_joker_current_plus(a, suffix="mult")) for a in ability]
    plus_chips = [int(_joker_current_plus(a, suffix="chips")) for a in ability]
    xmult = [float(_joker_current_xmult(a)) for a in ability]
    # Gating bools — cheap evaluation, only used by the named jokers
    # in Rust's ability pass. Compute always for simplicity (the per-
    # joker helper is a tight loop over metadata dicts).
    # Gating bools, scaling fallbacks, and target metadata all read
    # from the COPIED joker (ability) — Blueprint copying Loyalty
    # Card should use Loyalty Card's ready flag, etc.
    loyalty_ready = [
        bool(_loyalty_card_ready(a)) if a.name == "Loyalty Card" else False
        for a in ability
    ]
    drivers_active = [
        bool(_drivers_license_active(a)) if a.name == "Driver's License" else False
        for a in ability
    ]
    leading_plus_mult = [
        int(_joker_leading_plus(a, suffix="mult")) if a.name == "Popcorn" else 0
        for a in ability
    ]
    leading_plus_chips = [
        int(_joker_leading_plus(a, suffix="chips")) if a.name == "Ice Cream" else 0
        for a in ability
    ]
    # Sell value: PHYSICAL — used by Swashbuckler to sum over other
    # physical jokers (a Blueprint's sell value still counts toward
    # an actual Swashbuckler's tally).
    sell_value = [int(getattr(j, "sell_value", 0) or 0) for j in jokers]
    # Rarity: PHYSICAL — Baseball Card gates on physical_joker's
    # rarity per Python (`joker_rarity(physical_joker)`).
    rarity = [_RARITY_TO_INT.get(_joker_rarity(j), 0) for j in jokers]
    target_suit: list[str | None] = []
    target_rank: list[str | None] = []
    for a in ability:
        if a.name == "Ancient Joker":
            target_suit.append(_joker_target_suit(a))
            target_rank.append(None)
        elif a.name == "The Idol":
            ts = _joker_target_rank_suit(a)
            if ts is None:
                target_rank.append(None)
                target_suit.append(None)
            else:
                rk, st = ts
                target_rank.append(rk)
                target_suit.append(st)
        else:
            target_suit.append(None)
            target_rank.append(None)
    obelisk_gain = [
        float(_obelisk_xmult_gain(a)) if a.name == "Obelisk" else 0.0
        for a in ability
    ]
    # Swashbuckler precompute mirrors Python's ability arm: a physical
    # Swashbuckler uses the sum of other physical jokers' sell values;
    # a Blueprint/Brainstorm copying Swashbuckler uses the copied
    # joker's current_plus_mult metadata.
    if "Swashbuckler" in names:
        total_sell = sum(sell_value)
        for i, n in enumerate(names):
            if n == "Swashbuckler":
                ability_index = ability_indices[i]
                if ability_index == i:
                    plus_mult[i] = total_sell - sell_value[i]
                else:
                    plus_mult[i] = int(_joker_current_plus(jokers[ability_index], suffix="mult"))
    result = (names, editions, plus_mult, plus_chips, xmult,
              loyalty_ready, drivers_active,
              leading_plus_mult, leading_plus_chips,
              sell_value, rarity,
              target_suit, target_rank, obelisk_gain)
    if cache is not None:
        cache[("rust_joker_data", id(jokers))] = (jokers, result)
    return result


def _cached_played_hand_types_strs(state: GameState) -> list[str]:
    """Identity-cached `[ht.value for ht in played_hand_types(state)]`.

    Same key (id(state.modifiers)) as Python's identity-cached
    `_played_hand_types_this_round`. Computed once per unique
    state.modifiers reference; on hot beam paths this means once
    per branch, not once per `_score_action`.
    """

    cache = _current_state_value_cache()
    mods = state.modifiers
    if cache is not None:
        key = ("rust_played_types_strs", id(mods))
        entry = cache.get(key)
        if entry is not None and entry[0] is mods:
            return entry[1]
    try:
        from balatro_ai.bots.basic_strategy.play_scoring import (
            _played_hand_types_this_round,
        )
        result = [ht.value for ht in _played_hand_types_this_round(state)]
    except (ImportError, AttributeError, ValueError, TypeError):
        result = []
    if cache is not None:
        cache[("rust_played_types_strs", id(mods))] = (mods, result)
    return result


def _cached_hand_as_rust_cards(state: GameState, balatro_core_mod) -> list | None:
    """Return a list of RustCard mirroring `state.hand`, cached for
    the duration of the active `state_value_cache_scope` (if any).

    Returns None when no scope is active (caller falls back to
    per-call conversion). Cache key is `id(state.hand)` since the
    hand tuple is frozen — if it ever changes identity, we get a
    fresh conversion automatically.
    """

    cache = _current_state_value_cache()
    if cache is None:
        return None
    key = ("rust_hand", id(state.hand))
    entry = cache.get(key)
    if entry is not None and entry[0] is state.hand:
        return entry[1]
    try:
        rust_hand = [balatro_core_mod.RustCard.from_python(c) for c in state.hand]
    except (ValueError, AttributeError):
        cache[key] = (state.hand, None)
        return None
    cache[key] = (state.hand, rust_hand)
    return rust_hand


def _rollout_discard_rank_value(card: Card) -> int:
    return {
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "10": 10,
        "T": 10,
        "J": 11,
        "Q": 12,
        "K": 13,
        "A": 14,
    }.get(card.rank, 0)


def _effective_blind_name(state: GameState) -> str:
    return "" if _truthy(state.modifiers.get("boss_disabled")) else state.blind


def _truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _search_state_cache_key(state: GameState) -> tuple[object, ...]:
    return (
        _hand_multiset_cache_key(state.hand),
        _jokers_cache_key(state.jokers),
        _freeze_for_cache(state.hand_levels),
        _scoring_state_cache_key(state),
    )


def _scoring_state_cache_key(state: GameState) -> tuple[object, ...]:
    return _state_identity_cached_value(
        "scoring_state_cache_key",
        state,
        (),
        lambda: (
            _effective_blind_name(state),
            state.discards_remaining,
            state.hands_remaining,
            state.deck_size,
            state.money,
            _freeze_for_cache(state.modifiers.get("hands", state.modifiers.get("hand_stats", {}))),
        ),
    )


def _selected_action_multiset_key(state: GameState, action: Action | None) -> tuple[object, ...] | None:
    if action is None:
        return None
    return _hand_multiset_cache_key(tuple(state.hand[index] for index in action.card_indices if 0 <= index < len(state.hand)))


def _play_action_for_card_multiset(hand: tuple[Card, ...], selected_key: tuple[object, ...]) -> Action | None:
    remaining: dict[object, int] = {}
    for key in selected_key:
        remaining[key] = remaining.get(key, 0) + 1

    indices: list[int] = []
    for index, card in enumerate(hand):
        key = _card_cache_key(card)
        count = remaining.get(key, 0)
        if count <= 0:
            continue
        indices.append(index)
        if count == 1:
            del remaining[key]
        else:
            remaining[key] = count - 1

    if remaining:
        return None
    return Action(ActionType.PLAY_HAND, card_indices=tuple(indices))


def _hand_multiset_cache_key(cards: tuple[Card, ...]) -> tuple[object, ...]:
    return tuple(sorted((_card_cache_key(card) for card in cards), key=repr))


def _jokers_cache_key(jokers: tuple[Joker, ...]) -> tuple[object, ...]:
    return tuple(_joker_cache_key(joker) for joker in jokers)


def _card_cache_key(card: Card) -> object:
    return _identity_cached(
        "card_cache_key",
        card,
        lambda: (
            "Card",
            card.rank,
            card.suit,
            card.enhancement,
            card.seal,
            card.edition,
            card.debuffed,
            _freeze_for_cache(card.metadata),
        ),
    )


def _joker_cache_key(joker: Joker) -> object:
    return _identity_cached(
        "joker_cache_key",
        joker,
        lambda: (
            "Joker",
            joker.name,
            joker.edition,
            joker.sell_value,
            _freeze_for_cache(joker.metadata),
        ),
    )


def _freeze_for_cache(value: object) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Card):
        return _card_cache_key(value)
    if isinstance(value, Joker):
        return _joker_cache_key(value)
    if isinstance(value, dict):
        return _identity_cached(
            "dict_freeze",
            value,
            lambda: tuple(sorted((str(key), _freeze_for_cache(item)) for key, item in value.items())),
        )
    if isinstance(value, list | tuple):
        return _identity_cached(
            "seq_freeze",
            value,
            lambda: tuple(_freeze_for_cache(item) for item in value),
        )
    if isinstance(value, set | frozenset):
        return _identity_cached(
            "set_freeze",
            value,
            lambda: tuple(sorted((_freeze_for_cache(item) for item in value), key=repr)),
        )
    return repr(value)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
