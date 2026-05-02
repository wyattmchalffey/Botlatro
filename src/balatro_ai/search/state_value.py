"""Leaf-state value estimates for Phase 7 search."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from random import Random

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import Card, GamePhase, GameState
from balatro_ai.rules.hand_evaluator import debuffed_suits_for_blind, evaluate_played_cards
from balatro_ai.search.deck_model import DeckModel
from balatro_ai.search.forward_sim import simulate_cash_out, simulate_discard, simulate_play


@dataclass(frozen=True, slots=True)
class RolloutConfig:
    samples: int = 64
    seed: int = 0


def clear_probability(state: GameState, *, samples: int = 64, seed: int = 0) -> float:
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


def state_value(state: GameState, *, samples: int = 64, seed: int = 0) -> float:
    """Combine current-blind survival with a conservative future-strength score."""

    if state.won:
        return 1.0
    if state.run_over or state.phase == GamePhase.RUN_OVER:
        return 0.0
    clear = clear_probability(state, samples=samples, seed=seed)
    future = future_value(state)
    return _clamp01((clear * 0.8) + (clear * future * 0.2))


def _greedy_rollout_clears(state: GameState, rng: Random) -> bool:
    current = state
    while current.current_score < current.required_score and current.hands_remaining > 0 and current.hand:
        action = _best_greedy_play_action(current)
        if action is None:
            return False
        if _should_rollout_discard(current, action):
            discard_action = _rollout_discard_action(current, action)
            if discard_action is not None:
                draw_count = min(len(discard_action.card_indices), DeckModel.from_state(current).total_cards)
                drawn_cards = DeckModel.from_state(current).sample_draws(draw_count, rng) if draw_count > 0 else ()
                current = simulate_discard(current, discard_action, drawn_cards=drawn_cards)
                continue
        draw_count = min(len(action.card_indices), DeckModel.from_state(current).total_cards)
        drawn_cards = DeckModel.from_state(current).sample_draws(draw_count, rng) if draw_count > 0 else ()
        current = simulate_play(current, action, drawn_cards=drawn_cards)
        if current.phase == GamePhase.ROUND_EVAL or current.current_score >= current.required_score:
            return True
        if current.phase == GamePhase.RUN_OVER or current.run_over:
            return False
    return current.current_score >= current.required_score


def _should_rollout_discard(state: GameState, best_play: Action) -> bool:
    if state.discards_remaining <= 0 or DeckModel.from_state(state).total_cards <= 0:
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
    discard_limit = min(5, len(candidates), DeckModel.from_state(state).total_cards)
    if discard_limit <= 0:
        return None
    ordered = sorted(candidates, key=lambda index: (_rollout_discard_rank_value(state.hand[index]), index))
    return Action(ActionType.DISCARD, card_indices=tuple(ordered[:discard_limit]))


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
    best_action: Action | None = None
    best_score = -1
    for action in _play_actions_for_hand(state.hand):
        score = _score_action(state, action)
        if score > best_score:
            best_score = score
            best_action = action
    return best_action


def _best_immediate_score(state: GameState) -> int:
    best = 0
    for action in _play_actions_for_hand(state.hand):
        best = max(best, _score_action(state, action))
    return best


def _play_actions_for_hand(hand: tuple[Card, ...]) -> tuple[Action, ...]:
    actions: list[Action] = []
    for size in range(1, min(5, len(hand)) + 1):
        for indices in combinations(range(len(hand)), size):
            actions.append(Action(ActionType.PLAY_HAND, card_indices=indices))
    return tuple(actions)


def _score_action(state: GameState, action: Action) -> int:
    selected = tuple(card for index, card in enumerate(state.hand) if index in set(action.card_indices))
    held = tuple(card for index, card in enumerate(state.hand) if index not in set(action.card_indices))
    try:
        evaluation = evaluate_played_cards(
            selected,
            state.hand_levels,
            debuffed_suits=debuffed_suits_for_blind(_effective_blind_name(state)),
            blind_name=_effective_blind_name(state),
            jokers=state.jokers,
            discards_remaining=state.discards_remaining,
            hands_remaining=state.hands_remaining,
            held_cards=held,
            deck_size=state.deck_size,
            money=state.money,
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


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
