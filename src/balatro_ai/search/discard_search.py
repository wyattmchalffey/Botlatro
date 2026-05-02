"""Discard expectimax for Phase 7 search."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from random import Random

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import Card, GameState
from balatro_ai.search.deck_model import DeckModel
from balatro_ai.search.forward_sim import simulate_discard
from balatro_ai.search.state_value import state_value

ValueFn = Callable[[GameState], float]


@dataclass(frozen=True, slots=True)
class DiscardSearchConfig:
    draw_samples: int = 32
    leaf_samples: int = 16
    seed: int = 0
    enumerate_draws_up_to: int = 2
    max_actions: int = 48


def best_discard_action(
    state: GameState,
    *,
    config: DiscardSearchConfig | None = None,
    value_fn: ValueFn | None = None,
) -> Action | None:
    """Return the legal discard action with the highest expected leaf value."""

    discard_actions = tuple(action for action in state.legal_actions if action.action_type == ActionType.DISCARD)
    if not discard_actions:
        return None

    search_config = config or DiscardSearchConfig()
    discard_actions = _candidate_discard_actions(state, discard_actions, limit=search_config.max_actions)
    best_action = discard_actions[0]
    best_value = float("-inf")
    for action_index, action in enumerate(discard_actions):
        value = discard_action_value(
            state,
            action,
            config=search_config,
            value_fn=value_fn,
            action_index=action_index,
        )
        if value > best_value:
            best_action = action
            best_value = value
    return _annotated_action(best_action, search_value=best_value)


def discard_action_value(
    state: GameState,
    action: Action,
    *,
    config: DiscardSearchConfig | None = None,
    value_fn: ValueFn | None = None,
    action_index: int = 0,
) -> float:
    """Average leaf value over caller-visible draw possibilities."""

    if action.action_type != ActionType.DISCARD:
        raise ValueError(f"discard_action_value requires discard, got {action.action_type.value}")
    search_config = config or DiscardSearchConfig()
    evaluator = value_fn or _default_value_fn(search_config, action_index=action_index)
    draw_count = len(action.card_indices)
    draws = _draw_outcomes(
        state,
        draw_count,
        config=search_config,
        action_index=action_index,
    )
    if not draws:
        simulated = simulate_discard(state, action, drawn_cards=())
        return evaluator(simulated)

    total = 0.0
    for drawn_cards in draws:
        simulated = simulate_discard(state, action, drawn_cards=drawn_cards)
        total += evaluator(simulated)
    return total / len(draws)


def _draw_outcomes(
    state: GameState,
    draw_count: int,
    *,
    config: DiscardSearchConfig,
    action_index: int,
) -> tuple[tuple[object, ...], ...]:
    if draw_count <= 0:
        return ((),)
    model = DeckModel.from_state(state)
    draw_count = min(draw_count, model.total_cards)
    if draw_count <= 0:
        return ((),)
    if draw_count <= config.enumerate_draws_up_to:
        outcomes = model.all_possible_draws(draw_count)
        if outcomes and len(outcomes) <= max(1, config.draw_samples):
            return outcomes

    rng = Random(config.seed + (action_index * 1_000_003))
    samples = max(1, config.draw_samples)
    return tuple(model.sample_draws(draw_count, rng) for _ in range(samples))


def _candidate_discard_actions(
    state: GameState,
    actions: tuple[Action, ...],
    *,
    limit: int,
) -> tuple[Action, ...]:
    if limit <= 0 or len(actions) <= limit:
        return actions
    indexed = tuple(enumerate(actions))
    ranked = sorted(indexed, key=lambda item: (_discard_candidate_score(state, item[1]), -item[0]), reverse=True)
    return tuple(action for _, action in ranked[:limit])


def _discard_candidate_score(state: GameState, action: Action) -> float:
    selected = tuple(state.hand[index] for index in action.card_indices if 0 <= index < len(state.hand))
    if not selected:
        return float("-inf")
    held = tuple(card for index, card in enumerate(state.hand) if index not in set(action.card_indices))
    score = len(selected) * 0.5
    score += sum(15 - _rank_value(card) for card in selected) * 0.25
    score += _duplicate_support_score(held) * 2.0
    score += _suit_support_score(held) * 1.2
    score += _straight_support_score(held) * 0.9
    score -= _duplicate_support_score(selected) * 1.5
    score -= _suit_support_score(selected) * 0.6
    return score


def _duplicate_support_score(cards: tuple[Card, ...]) -> float:
    counts: dict[str, int] = {}
    for card in cards:
        counts[card.rank] = counts.get(card.rank, 0) + 1
    return float(sum(count * count for count in counts.values() if count >= 2))


def _suit_support_score(cards: tuple[Card, ...]) -> float:
    counts: dict[str, int] = {}
    for card in cards:
        counts[card.suit] = counts.get(card.suit, 0) + 1
    return float(max(counts.values(), default=0))


def _straight_support_score(cards: tuple[Card, ...]) -> float:
    values = sorted({_rank_value(card) for card in cards if _rank_value(card) > 0})
    if 14 in values:
        values = sorted(set(values + [1]))
    best = 0
    for start in range(len(values)):
        run = 1
        for index in range(start + 1, len(values)):
            if values[index] == values[index - 1] + 1:
                run += 1
                best = max(best, run)
            elif values[index] > values[index - 1] + 1:
                break
    return float(best)


def _rank_value(card: Card) -> int:
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


def _default_value_fn(config: DiscardSearchConfig, *, action_index: int) -> ValueFn:
    def evaluate(state: GameState) -> float:
        return state_value(
            state,
            samples=config.leaf_samples,
            seed=config.seed + (action_index * 10_007),
        )

    return evaluate


def _annotated_action(action: Action, *, search_value: float) -> Action:
    return Action(
        action.action_type,
        card_indices=action.card_indices,
        target_id=action.target_id,
        amount=action.amount,
        metadata={**action.metadata, "search_value": round(search_value, 6), "search": "discard_expectimax"},
    )
