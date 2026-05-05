"""Integrated play/discard search for Phase 7 hand decisions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from random import Random

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import Card, GameState
from balatro_ai.search.deck_model import DeckModel
from balatro_ai.search.discard_search import _candidate_discard_actions
from balatro_ai.search.forward_sim import simulate_discard, simulate_play
from balatro_ai.search.state_value import state_value

ValueFn = Callable[[GameState], float]


@dataclass(frozen=True, slots=True)
class HandSearchConfig:
    draw_samples: int = 1
    leaf_samples: int = 1
    seed: int = 0
    enumerate_draws_up_to: int = 0
    max_play_actions: int = 16
    max_discard_actions: int = 8


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
    from balatro_ai.bots.basic_strategy_bot import _BlindContext

    return _BlindContext()


def _best_play_action(state: GameState, context):
    from balatro_ai.bots.basic_strategy_bot import _best_play_action as basic_best_play_action

    return basic_best_play_action(state, context)


def _score_play_action(state: GameState, action: Action, context=None) -> int:
    from balatro_ai.bots.basic_strategy_bot import _score_play_action as basic_score_play_action

    return basic_score_play_action(state, action, context)


def _estimated_hands_needed(remaining_score: int, score: int | float) -> int:
    from balatro_ai.bots.basic_strategy_bot import _estimated_hands_needed as basic_estimated_hands_needed

    return basic_estimated_hands_needed(remaining_score, score)


def _annotated_action(action: Action, *, search_value: float, reason: str) -> Action:
    return Action(
        action.action_type,
        card_indices=action.card_indices,
        target_id=action.target_id,
        amount=action.amount,
        metadata={
            **action.metadata,
            "search": "hand_expectimax",
            "search_value": round(search_value, 6),
            "reason": action.metadata.get("reason", reason),
        },
    )
