"""Economy-value helpers used by blind discard and play choices."""

from __future__ import annotations

from balatro_ai.api.state import Card, GameState, Joker
from balatro_ai.bots.basic_strategy.cards import (
    _basic_consumable_open_slots,
    _is_blue_seal,
    _is_gold_enhancement,
    _is_gold_seal,
)
from balatro_ai.bots.basic_strategy.discard_state import _round_discard_used_count
from balatro_ai.bots.basic_strategy.hand_models import _BlindContext
from balatro_ai.bots.basic_strategy.jokers import _joker_metadata_numeric_value


BLUE_SEAL_ROUND_END_VALUE = 12.0


def _discard_sensitive_cash_out_value(state: GameState, context: _BlindContext) -> float:
    if state.discards_remaining <= 0 or _round_discard_used_count(state, context) > 0:
        return 0.0

    value = 0.0
    for joker in state.jokers:
        if joker.name == "Delayed Gratification" and not joker.effect.disabled:
            value += float(state.discards_remaining * _delayed_gratification_dollars(joker))
    return value


def _delayed_gratification_dollars(joker: Joker) -> int:
    if joker.effect.earn_dollars is not None:
        return joker.effect.earn_dollars
    value = _joker_metadata_numeric_value(joker, ("dollars", "money", "extra"))
    try:
        return int(value) if value is not None else 2
    except (TypeError, ValueError):
        return 2


def _held_round_end_economy_value(state: GameState, held_cards: tuple[Card, ...]) -> float:
    value = 0.0
    trigger_count = 1 + sum(1 for joker in state.jokers if joker.name == "Mime" and not joker.effect.disabled)
    value += 3.0 * trigger_count * sum(1 for card in held_cards if _is_gold_enhancement(card))
    blue_seals = sum(1 for card in held_cards if _is_blue_seal(card))
    if blue_seals > 0:
        value += BLUE_SEAL_ROUND_END_VALUE * min(_basic_consumable_open_slots(state), blue_seals)
    return value


def _drawn_economy_hunt_value(state: GameState, drawn_cards: tuple[Card, ...]) -> float:
    return sum(_economy_hunt_card_value(state, card) for card in drawn_cards)


def _economy_hunt_card_value(state: GameState, card: Card) -> float:
    value = 0.0
    if _is_gold_enhancement(card):
        value += 3.0
    if _is_blue_seal(card) and _basic_consumable_open_slots(state) > 0:
        value += BLUE_SEAL_ROUND_END_VALUE
    if _is_gold_seal(card):
        value += 3.0
    return value


def _card_is_economy_hunt_target(state: GameState, card: Card) -> bool:
    return _economy_hunt_card_value(state, card) > 0.0
