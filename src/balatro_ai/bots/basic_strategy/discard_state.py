"""State projection helpers for discard decisions."""

from __future__ import annotations

from dataclasses import replace

from balatro_ai.api.actions import Action
from balatro_ai.api.state import Card, GameState, Joker
from balatro_ai.bots.basic_strategy.blind_state import _is_boss_blind
from balatro_ai.bots.basic_strategy.cards import _normalize_suit
from balatro_ai.bots.basic_strategy.hand_models import _BlindContext
from balatro_ai.bots.basic_strategy.jokers import (
    _castle_target_suit,
    _format_xmult,
    _joker_current_xmult_value,
    _joker_metadata_int_value,
    _joker_metadata_numeric_value,
    _joker_with_added_current_plus,
    _joker_with_added_current_xmult,
    _truthy_modifier,
)
from balatro_ai.bots.basic_strategy.play_scoring import _played_hand_counts
from balatro_ai.bots.basic_strategy.utils import _int_or_default
from balatro_ai.rules.hand_evaluator import evaluate_played_cards, debuffed_suits_for_blind


def _known_draw_for_discard(state: GameState, action: Action) -> tuple[Card, ...]:
    kept_count = max(0, len(state.hand) - len(action.card_indices))
    draw_count = min(_discard_draw_count(state, action, kept_count), len(state.known_deck))
    if draw_count <= 0:
        return ()
    return tuple(state.known_deck[:draw_count])


def _discard_draw_count(state: GameState, action: Action, kept_count: int) -> int:
    if _serpent_draws_three_for_strategy(state):
        draw_count = 3
    else:
        draw_count = max(0, _effective_hand_size(state) - kept_count)
    if state.deck_size > 0:
        return min(draw_count, state.deck_size)
    if state.known_deck:
        return min(draw_count, len(state.known_deck))
    if not _has_explicit_hand_size_modifier(state):
        return min(draw_count, len(action.card_indices))
    return draw_count


def _has_explicit_hand_size_modifier(state: GameState) -> bool:
    return any(key in state.modifiers for key in ("hand_size", "hand_size_limit", "hand_size_max", "hand_size_delta"))


def _effective_hand_size(state: GameState) -> int:
    for key in ("hand_size", "hand_size_limit", "hand_size_max"):
        raw = state.modifiers.get(key)
        try:
            if raw is not None:
                return max(1, int(raw))
        except (TypeError, ValueError):
            continue
    return max(1, 8 + _int_or_default(state.modifiers.get("hand_size_delta"), 0))


def _serpent_draws_three_for_strategy(state: GameState) -> bool:
    return (
        state.blind == "The Serpent"
        and _is_boss_blind(state)
        and not _truthy_modifier(state.modifiers.get("boss_disabled"))
        and not any(joker.name == "Chicot" and not joker.effect.disabled for joker in state.jokers)
    )


def _state_after_known_discard_for_economy_hunt(
    state: GameState,
    action: Action,
    drawn_cards: tuple[Card, ...],
    context: _BlindContext,
) -> GameState:
    return _state_after_discard_for_projection(
        state,
        action,
        drawn_cards=drawn_cards,
        context=context,
        decrement_discard=True,
    )


def _state_after_discard_for_projection(
    state: GameState,
    action: Action,
    *,
    drawn_cards: tuple[Card, ...],
    context: _BlindContext,
    decrement_discard: bool,
) -> GameState:
    discarded_cards = tuple(state.hand[index] for index in action.card_indices)
    kept_cards = tuple(card for index, card in enumerate(state.hand) if index not in action.card_indices)
    scoring_state = _discard_scoring_state(state, discarded_cards, context)
    discards_remaining = max(0, state.discards_remaining - 1) if decrement_discard else state.discards_remaining
    modifiers = (
        _modifiers_after_discard_for_economy_hunt(state.modifiers, context)
        if decrement_discard
        else state.modifiers
    )
    return replace(
        state,
        hand=(*kept_cards, *drawn_cards),
        known_deck=tuple(state.known_deck[len(drawn_cards):]),
        deck_size=max(0, state.deck_size - len(drawn_cards)),
        discards_remaining=discards_remaining,
        money=scoring_state.money,
        jokers=scoring_state.jokers,
        hand_levels=scoring_state.hand_levels,
        modifiers=modifiers,
    )


def _discard_scoring_state(
    state: GameState,
    discarded_cards: tuple[Card, ...],
    context: _BlindContext,
) -> GameState:
    discard_money = _conditional_discard_money_delta_for_economy_hunt(state, discarded_cards, context)
    return replace(
        state,
        money=state.money + discard_money,
        jokers=_jokers_after_discard_for_scoring(state, discarded_cards),
        hand_levels=_hand_levels_after_discard_for_economy_hunt(state, discarded_cards, context),
    )


def _conditional_discard_money_delta_for_economy_hunt(
    state: GameState,
    discarded_cards: tuple[Card, ...],
    context: _BlindContext,
) -> int:
    if _round_discard_used_count(state, context) > 0 or len(discarded_cards) != 1:
        return 0
    total = 0
    for joker in state.jokers:
        if joker.name == "Trading Card" and not joker.effect.disabled:
            total += _trading_card_dollars(joker)
    return total


def _trading_card_dollars(joker: Joker) -> int:
    if joker.effect.earn_dollars is not None:
        return joker.effect.earn_dollars
    value = _joker_metadata_numeric_value(joker, ("dollars", "money", "extra"))
    try:
        return int(value) if value is not None else 3
    except (TypeError, ValueError):
        return 3


def _hand_levels_after_discard_for_economy_hunt(
    state: GameState,
    discarded_cards: tuple[Card, ...],
    context: _BlindContext,
) -> dict[str, int]:
    if (
        not discarded_cards
        or state.blind == "The Hook"
        or _round_discard_used_count(state, context) > 0
        or not any(joker.name == "Burnt Joker" and not joker.effect.disabled for joker in state.jokers)
    ):
        return state.hand_levels
    evaluation = evaluate_played_cards(
        discarded_cards,
        state.hand_levels,
        debuffed_suits=debuffed_suits_for_blind(state.blind),
        blind_name=state.blind,
        jokers=state.jokers,
        discards_remaining=state.discards_remaining,
        hands_remaining=state.hands_remaining,
        held_cards=(),
        deck_size=state.deck_size,
        money=state.money,
        played_hand_types_this_round=context.played_hand_types,
        played_hand_counts=_played_hand_counts(state),
    )
    updated = dict(state.hand_levels)
    updated[evaluation.hand_type.value] = max(1, _int_or_default(updated.get(evaluation.hand_type.value), 1)) + 1
    return updated


def _modifiers_after_discard_for_economy_hunt(
    modifiers: dict[str, object],
    context: _BlindContext,
) -> dict[str, object]:
    updated = dict(modifiers)
    next_count = max(0, _round_discard_used_count_from_modifiers(modifiers), context.discards_taken) + 1
    for key in ("round_discards_used", "discards_used", "discards_used_this_round"):
        updated[key] = next_count
    return updated


def _round_discard_used_count(state: GameState, context: _BlindContext) -> int:
    return max(context.discards_taken, _round_discard_used_count_from_modifiers(state.modifiers))


def _round_discard_used_count_from_modifiers(modifiers: dict[str, object]) -> int:
    for key in ("round_discards_used", "discards_used", "discards_used_this_round"):
        if key not in modifiers:
            continue
        value = _int_or_default(modifiers.get(key), 0)
        return max(0, value)
    return 0


def _jokers_after_discard_for_scoring(state: GameState, discarded_cards: tuple[Card, ...]) -> tuple[Joker, ...]:
    if not discarded_cards:
        return state.jokers

    castle_suit = _castle_target_suit(state)
    discarded_castle_suit = (
        sum(1 for card in discarded_cards if not card.debuffed and _normalize_suit(card.suit) == castle_suit)
        if castle_suit is not None
        else 0
    )
    discarded_jacks = sum(1 for card in discarded_cards if not card.debuffed and card.rank == "J")
    discard_count = len(discarded_cards)

    adjusted: list[Joker] = []
    for joker in state.jokers:
        if joker.effect.disabled:
            adjusted.append(joker)
        elif joker.name == "Green Joker":
            adjusted.append(_joker_with_added_current_plus(joker, -1, suffix="mult"))
        elif joker.name == "Castle" and discarded_castle_suit:
            adjusted.append(_joker_with_added_current_plus(joker, discarded_castle_suit * 3, suffix="chips"))
        elif joker.name == "Hit the Road" and discarded_jacks:
            adjusted.append(_joker_with_added_current_xmult(joker, discarded_jacks * 0.5))
        elif joker.name == "Yorick" and discard_count:
            adjusted.append(_yorick_after_discard(joker, discard_count))
        elif joker.name == "Ramen" and discard_count:
            adjusted.append(_joker_with_added_current_xmult(joker, discard_count * -0.01, minimum=1.0))
        else:
            adjusted.append(joker)
    return tuple(adjusted)


def _yorick_after_discard(joker: Joker, discard_count: int) -> Joker:
    remaining = _joker_metadata_int_value(
        joker,
        ("current_remaining", "remaining", "discards_remaining", "yorick_discards"),
        default=23,
    )
    current = _joker_current_xmult_value(joker)
    for _ in range(discard_count):
        if remaining <= 1:
            remaining = 23
            current += 1.0
        else:
            remaining -= 1
    metadata = dict(joker.metadata)
    metadata["current_remaining"] = remaining
    metadata["current_xmult"] = current
    metadata["effect"] = f"Currently X{_format_xmult(current)} ({remaining} discards remaining)"
    return Joker(joker.name, edition=joker.edition, sell_value=joker.sell_value, metadata=metadata)
