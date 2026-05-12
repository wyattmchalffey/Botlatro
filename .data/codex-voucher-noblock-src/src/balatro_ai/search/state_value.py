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

    total_samples = max(1, samples)
    rng = Random(seed)
    clears = 0
    for _ in range(total_samples):
        if _greedy_rollout_clears(state, rng):
            clears += 1
    return clears / total_samples


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
        return 1.0 + min(0.75, headroom * 0.25)

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
    best_action: Action | None = None
    best_score = -1
    joker_context = _prepare_joker_evaluation_context(state.jokers)
    blind_name = _effective_blind_name(state)
    debuffed_suits = debuffed_suits_for_blind(blind_name)
    for action in _play_actions_for_hand(state.hand):
        score = _score_action(state, action, joker_context=joker_context, blind_name=blind_name, debuffed_suits=debuffed_suits)
        if score > best_score:
            best_score = score
            best_action = action
    return best_action


def _best_immediate_score(state: GameState) -> int:
    return _state_content_cached_value("best_immediate_score", state, (), lambda: _best_immediate_score_uncached(state))


def _best_immediate_score_uncached(state: GameState) -> int:
    best = 0
    joker_context = _prepare_joker_evaluation_context(state.jokers)
    blind_name = _effective_blind_name(state)
    debuffed_suits = debuffed_suits_for_blind(blind_name)
    for action in _play_actions_for_hand(state.hand):
        best = max(best, _score_action(state, action, joker_context=joker_context, blind_name=blind_name, debuffed_suits=debuffed_suits))
    return best


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
