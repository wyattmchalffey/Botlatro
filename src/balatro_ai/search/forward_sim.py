"""Small forward simulators used by Phase 7 search.

The simulator never invents random outcomes. Callers must inject drawn cards,
shop rolls, and pack contents so search code can use enumeration or sampling.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import re
from typing import Iterable

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import Card, GamePhase, GameState, Joker
from balatro_ai.rules.hand_evaluator import HandType, debuffed_suits_for_blind, evaluate_played_cards


@dataclass(frozen=True, slots=True)
class _DrawResult:
    deck_size: int
    known_deck: tuple[Card, ...]


@dataclass(frozen=True, slots=True)
class StochasticPlayOutcomes:
    """Known random outcomes for one played hand.

    Forward sim stays deterministic: callers inject these values from replay
    data or from a search sampler instead of the simulator rolling randomness.
    """

    bloodstone_triggers: int = 0
    business_card_triggers: int = 0
    reserved_parking_triggers: int = 0
    misprint_mult: int = 0
    space_joker_triggers: int = 0
    lucky_card_triggers: int = 0
    matador_triggered: bool | None = None


PLANET_TO_HAND = {
    "Pluto": HandType.HIGH_CARD,
    "Mercury": HandType.PAIR,
    "Uranus": HandType.TWO_PAIR,
    "Venus": HandType.THREE_OF_A_KIND,
    "Earth": HandType.FULL_HOUSE,
    "Mars": HandType.FOUR_OF_A_KIND,
    "Jupiter": HandType.FLUSH,
    "Saturn": HandType.STRAIGHT,
    "Neptune": HandType.STRAIGHT_FLUSH,
    "Planet X": HandType.FIVE_OF_A_KIND,
    "Ceres": HandType.FLUSH_HOUSE,
    "Eris": HandType.FLUSH_FIVE,
}


def simulate_play(
    state: GameState,
    action: Action,
    drawn_cards: Iterable[Card] = (),
    *,
    created_consumables: Iterable[object] = (),
    stochastic_outcomes: StochasticPlayOutcomes | dict[str, object] | None = None,
) -> GameState:
    """Apply a deterministic play-hand transition."""

    if action.action_type != ActionType.PLAY_HAND:
        raise ValueError(f"simulate_play requires play_hand, got {action.action_type.value}")
    selected_cards, held_cards = _split_action_cards(state, action)
    if not selected_cards:
        raise ValueError("Cannot simulate playing zero cards")

    outcome_map = _play_outcome_mapping(stochastic_outcomes)
    blind_name = _effective_blind_name(state)
    evaluation = evaluate_played_cards(
        selected_cards,
        state.hand_levels,
        debuffed_suits=debuffed_suits_for_blind(blind_name),
        blind_name=blind_name,
        jokers=state.jokers,
        discards_remaining=state.discards_remaining,
        hands_remaining=state.hands_remaining,
        held_cards=held_cards,
        deck_size=state.deck_size,
        money=state.money,
        played_hand_types_this_round=_played_hand_types_this_round(state),
        played_hand_counts=_played_hand_counts(state),
        stochastic_outcomes=outcome_map,
    )
    draw = tuple(drawn_cards)
    draw_result = _draw_from_deck(state, draw)
    next_score = state.current_score + evaluation.score
    next_hands = max(0, state.hands_remaining - 1)
    next_money = state.money + evaluation.money_delta
    next_jokers = _jokers_after_play(
        state.jokers,
        selected_cards,
        evaluation,
        hands_remaining=state.hands_remaining,
        stochastic_outcomes=outcome_map,
    )
    next_hand_levels = _hand_levels_after_play(state.hand_levels, evaluation, stochastic_outcomes=outcome_map)
    mutated_played_cards = _played_cards_after_play(state, selected_cards, evaluation)
    dna_cards = _dna_cards_after_play(state, selected_cards, evaluation)
    sixth_sense_triggered = _sixth_sense_triggers(state, selected_cards)
    consumable_creations = _play_consumable_creation_requests(
        state,
        selected_cards,
        evaluation,
        sixth_sense_triggered=sixth_sense_triggered,
    )
    created_consumable_labels = _created_labels(created_consumables)
    _validate_injected_count_at_most(
        label="play-created consumables",
        expected=_creation_request_total(consumable_creations),
        injected=len(created_consumable_labels),
    )
    kept_played_cards = () if sixth_sense_triggered else mutated_played_cards
    next_modifiers = _with_played_hand_type(state.modifiers, evaluation.hand_type)
    next_modifiers = _with_played_hand_level(next_modifiers, evaluation.hand_type, _outcome_int(outcome_map, "space_joker_triggers"))
    next_modifiers = _with_played_pile(next_modifiers, kept_played_cards)
    if sixth_sense_triggered:
        next_modifiers = _with_destroyed_cards(next_modifiers, mutated_played_cards)
    next_modifiers = _with_creation_counts(next_modifiers, consumable_creations, created=len(created_consumable_labels))
    next_phase = state.phase
    run_over = state.run_over
    if state.required_score > 0 and next_score >= state.required_score:
        next_phase = GamePhase.ROUND_EVAL
        next_money += _held_end_of_round_money_delta(held_cards, state.jokers)
    elif next_hands <= 0:
        if _mr_bones_saves(state, next_score):
            next_phase = GamePhase.ROUND_EVAL
            next_money += _held_end_of_round_money_delta(held_cards, state.jokers)
            next_jokers = _jokers_after_mr_bones_save(next_jokers)
        else:
            next_phase = GamePhase.RUN_OVER
            run_over = True
    return replace(
        state,
        phase=next_phase,
        current_score=next_score,
        money=next_money,
        hands_remaining=next_hands,
        deck_size=draw_result.deck_size,
        hand_levels=next_hand_levels,
        hand=held_cards + dna_cards + draw,
        known_deck=draw_result.known_deck,
        jokers=next_jokers,
        consumables=state.consumables + created_consumable_labels,
        modifiers=next_modifiers,
        legal_actions=(),
        run_over=run_over,
    )


def simulate_discard(state: GameState, action: Action, drawn_cards: Iterable[Card] = ()) -> GameState:
    """Apply a deterministic discard transition."""

    if action.action_type != ActionType.DISCARD:
        raise ValueError(f"simulate_discard requires discard, got {action.action_type.value}")
    selected_cards, held_cards = _split_action_cards(state, action)
    if not selected_cards:
        raise ValueError("Cannot simulate discarding zero cards")

    draw = tuple(drawn_cards)
    draw_result = _draw_from_deck(state, draw)
    next_jokers = _jokers_after_discard(state.jokers, selected_cards)
    next_hand_levels = _hand_levels_after_discard(state, selected_cards)
    next_money = state.money + _discard_money_delta(state, selected_cards)
    discarded_pile_cards = _discarded_cards_after_discard(state, selected_cards)
    destroyed_cards = _destroyed_cards_after_discard(state, selected_cards)
    next_modifiers = _with_discard_used(state.modifiers)
    next_modifiers = _with_discard_pile(next_modifiers, discarded_pile_cards)
    if destroyed_cards:
        next_modifiers = _with_destroyed_cards(next_modifiers, destroyed_cards)
    return replace(
        state,
        discards_remaining=max(0, state.discards_remaining - 1),
        money=next_money,
        deck_size=draw_result.deck_size,
        hand=held_cards + draw,
        known_deck=draw_result.known_deck,
        jokers=next_jokers,
        hand_levels=next_hand_levels,
        modifiers=next_modifiers,
        legal_actions=(),
    )


def simulate_buy(state: GameState, action: Action) -> GameState:
    """Apply a deterministic shop buy transition."""

    if action.action_type != ActionType.BUY:
        raise ValueError(f"simulate_buy requires buy, got {action.action_type.value}")
    kind = _action_kind(action, default="card")
    if kind == "voucher":
        item_key = "voucher_cards"
    elif kind == "card":
        item_key = "shop_cards"
    else:
        raise ValueError(f"Unsupported buy kind: {kind}")

    items = _modifier_items(state.modifiers, item_key)
    index = _action_index(action)
    if index is None or not 0 <= index < len(items):
        raise ValueError(f"Buy action index is outside {item_key}: {action.amount}")
    item = items[index]
    cost = _effective_item_buy_cost(state, item)
    if cost > state.money:
        raise ValueError(f"Cannot buy {_item_label(item)} for ${cost} with ${state.money}")
    if _item_is_consumable(item) and not _has_consumable_room(state):
        raise ValueError(
            f"Cannot buy {_item_label(item)} with full consumable slots: "
            f"{len(state.consumables)}/{_consumable_slot_limit(state)}"
        )

    next_modifiers = _with_modifier_items(state.modifiers, item_key, _without_index(items, index))
    acquired = _state_after_acquiring_item(replace(state, modifiers=next_modifiers), item, force_voucher=kind == "voucher")
    return replace(
        acquired,
        money=state.money - cost,
        shop=tuple(_item_label(item) for item in _modifier_items(next_modifiers, "shop_cards")),
        legal_actions=(),
    )


def simulate_sell(state: GameState, action: Action) -> GameState:
    """Apply a deterministic joker sell transition."""

    if action.action_type != ActionType.SELL:
        raise ValueError(f"simulate_sell requires sell, got {action.action_type.value}")
    kind = _action_kind(action, default="joker")
    if kind != "joker":
        raise ValueError(f"Unsupported sell kind: {kind}")
    index = _action_index(action)
    if index is None or not 0 <= index < len(state.jokers):
        raise ValueError(f"Sell action index is outside jokers: {action.amount}")
    sold = state.jokers[index]
    remaining = tuple(joker for joker_index, joker in enumerate(state.jokers) if joker_index != index)
    next_modifiers = _modifiers_after_joker_removed(state.modifiers, sold)
    next_modifiers = _modifiers_after_sell_trigger(state, next_modifiers, sold)
    return replace(
        state,
        money=state.money + (sold.sell_value or 0),
        jokers=_jokers_after_sell(remaining),
        modifiers=next_modifiers,
        legal_actions=(),
    )


def simulate_reroll(state: GameState, action: Action, new_shop_cards: Iterable[object]) -> GameState:
    """Apply a deterministic reroll transition using caller-injected shop cards."""

    if action.action_type != ActionType.REROLL:
        raise ValueError(f"simulate_reroll requires reroll, got {action.action_type.value}")
    cost, next_modifiers = _spend_reroll(state)
    if cost > state.money:
        raise ValueError(f"Cannot reroll for ${cost} with ${state.money}")
    shop_cards = tuple(new_shop_cards)
    next_modifiers = _with_modifier_items(next_modifiers, "shop_cards", shop_cards)
    return replace(
        state,
        money=state.money - cost,
        shop=tuple(_item_label(item) for item in shop_cards),
        jokers=_jokers_after_reroll(state.jokers),
        modifiers=next_modifiers,
        legal_actions=(),
    )


def simulate_open_pack(
    state: GameState,
    action: Action,
    pack_contents: Iterable[object],
    *,
    created_consumables: Iterable[object] = (),
) -> GameState:
    """Open a booster, using caller-injected pack contents."""

    if action.action_type != ActionType.OPEN_PACK:
        raise ValueError(f"simulate_open_pack requires open_pack, got {action.action_type.value}")
    packs = _modifier_items(state.modifiers, "booster_packs")
    index = _action_index(action)
    if index is None or not 0 <= index < len(packs):
        raise ValueError(f"Open-pack action index is outside booster_packs: {action.amount}")
    pack = packs[index]
    cost = _effective_item_buy_cost(state, pack)
    if cost > state.money:
        raise ValueError(f"Cannot open {_item_label(pack)} for ${cost} with ${state.money}")
    contents = tuple(pack_contents)
    hallucination_cards = _created_labels(created_consumables)
    _validate_injected_count_at_most(
        label="Hallucination",
        expected=_hallucination_creation_count(state),
        injected=len(hallucination_cards),
    )
    next_modifiers = _with_modifier_items(state.modifiers, "booster_packs", _without_index(packs, index))
    next_modifiers = _with_modifier_items(next_modifiers, "pack_cards", contents)
    next_modifiers = _with_created_tarot_count(next_modifiers, len(hallucination_cards))
    next_modifiers = _increment_modifier_by(next_modifiers, "created_consumable_count", len(hallucination_cards))
    return replace(
        state,
        phase=GamePhase.BOOSTER_OPENED,
        money=state.money - cost,
        pack=tuple(_item_label(item) for item in contents),
        consumables=state.consumables + hallucination_cards,
        modifiers=next_modifiers,
        legal_actions=(),
    )


def simulate_choose_pack_card(
    state: GameState,
    action: Action,
    *,
    remaining_pack_cards: Iterable[object] | None = None,
) -> GameState:
    """Choose or skip a card from an opened booster."""

    if action.action_type != ActionType.CHOOSE_PACK_CARD:
        raise ValueError(f"simulate_choose_pack_card requires choose_pack_card, got {action.action_type.value}")
    if action.target_id == "skip" or _action_kind(action, default="card") == "skip":
        return replace(
            state,
            phase=GamePhase.SHOP,
            pack=(),
            modifiers=_with_modifier_items(state.modifiers, "pack_cards", ()),
            legal_actions=(),
        )

    pack_cards = _modifier_items(state.modifiers, "pack_cards")
    index = _action_index(action)
    if index is None or not 0 <= index < len(pack_cards):
        raise ValueError(f"Pack choice index is outside pack_cards: {action.amount}")
    item = pack_cards[index]
    if _item_is_joker(item) and _joker_item_uses_normal_slot(item) and _normal_joker_open_slots(state) <= 0:
        raise ValueError(
            f"Cannot choose {_item_label(item)} with full joker slots: "
            f"{_normal_joker_slots_used(state)}/{_normal_joker_slot_limit(state)}"
        )
    next_pack_cards = tuple(remaining_pack_cards) if remaining_pack_cards is not None else ()
    next_phase = GamePhase.BOOSTER_OPENED if remaining_pack_cards is not None else GamePhase.SHOP
    next_modifiers = _with_modifier_items(state.modifiers, "pack_cards", next_pack_cards)
    acquired = _state_after_pack_choice(
        replace(
            state,
            phase=next_phase,
            pack=tuple(_item_label(item) for item in next_pack_cards),
            modifiers=next_modifiers,
        ),
        item,
    )
    return replace(acquired, legal_actions=())


def simulate_end_shop(state: GameState, *, created_consumables: Iterable[object] = ()) -> GameState:
    """Leave the shop and move to blind selection."""

    created_consumable_labels = _created_labels(created_consumables)
    _validate_injected_count_at_most(
        label="Perkeo",
        expected=_perkeo_creation_count(state),
        injected=len(created_consumable_labels),
    )
    next_modifiers = _without_shop_surface(state.modifiers)
    if created_consumable_labels:
        next_modifiers = _increment_modifier_by(
            next_modifiers,
            "created_negative_consumable_count",
            len(created_consumable_labels),
        )
    return replace(
        state,
        phase=GamePhase.BLIND_SELECT,
        shop=(),
        pack=(),
        consumables=state.consumables + created_consumable_labels,
        modifiers=next_modifiers,
        legal_actions=(),
    )


def simulate_cash_out(
    state: GameState,
    *,
    next_shop_cards: Iterable[object] = (),
    next_voucher_cards: Iterable[object] = (),
    next_booster_packs: Iterable[object] = (),
    next_to_do_targets: Iterable[str] = (),
    removed_jokers: Iterable[str] = (),
) -> GameState:
    returning_cards = state.hand + _zone_cards(state.modifiers, "played_pile") + _zone_cards(state.modifiers, "discard_pile")
    shop_cards = tuple(next_shop_cards)
    voucher_cards = tuple(next_voucher_cards)
    booster_packs = tuple(next_booster_packs)
    known_deck = state.known_deck + returning_cards if state.known_deck else ()
    remaining_deck_count = max(state.deck_size, len(state.known_deck)) if state.known_deck else state.deck_size
    gift_card_gain = _gift_card_sell_value_gain(state.jokers)
    next_jokers = _jokers_after_cash_out(state, next_to_do_targets=tuple(next_to_do_targets))
    next_jokers = _jokers_after_stochastic_removals(next_jokers, tuple(removed_jokers))
    next_jokers = _jokers_after_gift_card_cash_out(next_jokers, gift_card_gain=gift_card_gain)
    next_modifiers = _without_round_card_zones(state.modifiers)
    next_modifiers = _with_modifier_items(next_modifiers, "shop_cards", shop_cards)
    next_modifiers = _with_modifier_items(next_modifiers, "voucher_cards", voucher_cards)
    next_modifiers = _with_modifier_items(next_modifiers, "booster_packs", booster_packs)
    next_modifiers = _with_modifier_items(next_modifiers, "pack_cards", ())
    next_modifiers = _modifiers_after_cash_out(state.modifiers, next_modifiers, state.jokers, next_jokers)
    next_modifiers = _modifiers_after_gift_card_cash_out(next_modifiers, state, gift_card_gain=gift_card_gain)
    return replace(
        state,
        phase=GamePhase.SHOP,
        current_score=0,
        money=state.money + _cash_out_money_delta(state),
        hands_remaining=_cash_out_base_hands(state),
        discards_remaining=_cash_out_base_discards(state),
        hand=(),
        shop=tuple(_item_label(item) for item in shop_cards),
        pack=(),
        known_deck=known_deck,
        deck_size=remaining_deck_count + len(returning_cards),
        jokers=next_jokers,
        modifiers=next_modifiers,
        legal_actions=(),
    )


def simulate_select_blind(
    state: GameState,
    drawn_cards: Iterable[Card] = (),
    *,
    created_hand_cards: Iterable[Card] = (),
    created_deck_cards: Iterable[Card] = (),
    created_consumables: Iterable[object] = (),
    created_jokers: Iterable[object] = (),
) -> GameState:
    """Start a blind, drawing the opening hand and applying injected first-draw card creation."""

    draw = tuple(drawn_cards)
    certificate_cards = tuple(created_hand_cards)
    marble_cards = tuple(_stone_card(card) for card in created_deck_cards)
    cartomancer_cards = _created_labels(created_consumables)
    riff_raff_jokers = tuple(_joker_from_item(item) for item in created_jokers)
    _validate_injected_count(
        label="Certificate",
        expected=_first_hand_drawn_effective_count(state, "Certificate"),
        injected=len(certificate_cards),
    )
    _validate_injected_count(
        label="Marble Joker",
        expected=_first_hand_drawn_effective_count(state, "Marble Joker"),
        injected=len(marble_cards),
    )
    _validate_injected_count_at_most(
        label="Cartomancer",
        expected=_cartomancer_creation_count(state),
        injected=len(cartomancer_cards),
    )
    _validate_injected_count_at_most(
        label="Riff-raff",
        expected=_riff_raff_creation_count(state),
        injected=len(riff_raff_jokers),
    )

    draw_result = _draw_from_deck(state, draw)
    next_modifiers = _with_new_round_counters(_without_round_card_zones(state.modifiers))
    next_modifiers = _with_created_tarot_count(next_modifiers, len(cartomancer_cards))
    next_modifiers = _increment_modifier_by(next_modifiers, "created_consumable_count", len(cartomancer_cards))
    next_modifiers = _increment_modifier_by(next_modifiers, "created_common_joker_count", len(riff_raff_jokers))
    known_deck = marble_cards + draw_result.known_deck if draw_result.known_deck or marble_cards else ()
    next_hands = max(0, state.hands_remaining + _int_value(next_modifiers.get("round_hands_delta")))
    next_discards = max(0, state.discards_remaining + _int_value(next_modifiers.get("round_discards_delta")))
    blind_name = _effective_blind_name(state)
    if blind_name == "The Needle":
        next_hands = 1
    if blind_name == "The Water":
        next_discards = 0
    burglar_count = _effective_joker_count(state.jokers, "Burglar")
    if burglar_count:
        next_hands += 3 * burglar_count
        next_discards = 0
    if _blind_is_boss(state) and _effective_joker_count(state.jokers, "Chicot"):
        next_modifiers["boss_disabled"] = True
    return replace(
        state,
        phase=GamePhase.SELECTING_HAND,
        current_score=0,
        hands_remaining=next_hands,
        discards_remaining=next_discards,
        hand=draw + certificate_cards,
        known_deck=known_deck,
        deck_size=draw_result.deck_size + len(marble_cards),
        jokers=state.jokers + riff_raff_jokers,
        consumables=state.consumables + cartomancer_cards,
        modifiers=next_modifiers,
        legal_actions=(),
        run_over=False,
    )


def _draw_from_deck(state: GameState, drawn_cards: tuple[Card, ...]) -> _DrawResult:
    if not drawn_cards:
        return _DrawResult(deck_size=state.deck_size, known_deck=state.known_deck)
    if not state.known_deck:
        return _DrawResult(deck_size=max(0, state.deck_size - len(drawn_cards)), known_deck=())

    exact_known_deck = len(state.known_deck) == state.deck_size
    remaining = list(state.known_deck)
    for drawn in drawn_cards:
        index = _find_matching_card_index(remaining, drawn)
        if index is None:
            if exact_known_deck:
                raise ValueError(f"Drawn card {drawn.short_name} was not present in exact known_deck")
            continue
        del remaining[index]

    known_deck = tuple(remaining)
    if exact_known_deck:
        return _DrawResult(deck_size=len(known_deck), known_deck=known_deck)
    return _DrawResult(deck_size=max(len(known_deck), state.deck_size - len(drawn_cards)), known_deck=known_deck)


def _play_outcome_mapping(outcomes: StochasticPlayOutcomes | dict[str, object] | None) -> dict[str, object]:
    if outcomes is None:
        return {}
    if isinstance(outcomes, StochasticPlayOutcomes):
        return {
            "bloodstone_triggers": outcomes.bloodstone_triggers,
            "business_card_triggers": outcomes.business_card_triggers,
            "reserved_parking_triggers": outcomes.reserved_parking_triggers,
            "misprint_mult": outcomes.misprint_mult,
            "space_joker_triggers": outcomes.space_joker_triggers,
            "lucky_card_triggers": outcomes.lucky_card_triggers,
            "matador_triggered": outcomes.matador_triggered,
        }
    return dict(outcomes)


def _find_matching_card_index(cards: list[Card], target: Card) -> int | None:
    target_key = _card_identity_key(target)
    for index, card in enumerate(cards):
        if _card_identity_key(card) == target_key:
            return index
    return None


def _card_identity_key(card: Card) -> tuple[str, str, str | None, str | None, str | None]:
    return (_normalize_rank(card.rank), _normalize_suit(card.suit), card.enhancement, card.seal, card.edition)


def _dna_cards_after_play(state: GameState, played_cards: tuple[Card, ...], evaluation) -> tuple[Card, ...]:
    del evaluation
    if len(played_cards) != 1 or not _first_hand_of_round(state):
        return ()
    dna_count = sum(1 for joker in state.jokers if joker.name == "DNA" and not _joker_is_disabled(joker))
    if dna_count <= 0:
        return ()
    return tuple(_copy_card(played_cards[0]) for _ in range(dna_count))


def _sixth_sense_triggers(state: GameState, played_cards: tuple[Card, ...]) -> bool:
    if len(played_cards) != 1 or not _first_hand_of_round(state) or not _has_consumable_room(state):
        return False
    card = played_cards[0]
    return _normalize_rank(card.rank) == "6" and any(
        joker.name == "Sixth Sense" and not _joker_is_disabled(joker) for joker in state.jokers
    )


def _copy_card(card: Card) -> Card:
    return replace(card, metadata=_copy_metadata(card.metadata))


def _copy_metadata(value):
    if isinstance(value, dict):
        return {key: _copy_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_metadata(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_metadata(item) for item in value)
    return value


def _stone_card(card: Card) -> Card:
    metadata = dict(card.metadata)
    modifier = metadata.get("modifier")
    if isinstance(modifier, dict):
        modifier = dict(modifier)
    else:
        modifier = {}
    modifier["enhancement"] = "STONE"
    metadata["modifier"] = modifier
    return replace(card, enhancement="STONE", metadata=metadata)


def _first_hand_drawn_effective_count(state: GameState, joker_name: str) -> int:
    return sum(1 for joker in _active_effective_jokers(state.jokers) if joker.name == joker_name)


def _validate_injected_count(*, label: str, expected: int, injected: int) -> None:
    if injected != expected:
        raise ValueError(f"{label} needs {expected} injected created card(s), got {injected}")


def _validate_injected_count_at_most(*, label: str, expected: int, injected: int) -> None:
    if injected > expected:
        raise ValueError(f"{label} can create at most {expected} injected card(s), got {injected}")


def _action_kind(action: Action, *, default: str) -> str:
    return str(action.metadata.get("kind") or action.target_id or default)


def _action_index(action: Action) -> int | None:
    raw = action.metadata.get("index", action.amount)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _modifier_items(modifiers: dict[str, object], key: str) -> tuple[object, ...]:
    raw = modifiers.get(key, ())
    if isinstance(raw, dict):
        raw = raw.get("cards", ())
    if isinstance(raw, list | tuple):
        return tuple(raw)
    return ()


def _with_modifier_items(modifiers: dict[str, object], key: str, items: tuple[object, ...]) -> dict[str, object]:
    updated = dict(modifiers)
    updated[key] = items
    return updated


def _without_index(items: tuple[object, ...], index: int) -> tuple[object, ...]:
    return tuple(item for item_index, item in enumerate(items) if item_index != index)


def _state_after_acquiring_item(state: GameState, item: object, *, force_voucher: bool = False) -> GameState:
    if force_voucher or _item_is_voucher(item):
        return replace(state, vouchers=state.vouchers + (_item_label(item),))
    if _item_is_joker(item):
        joker = _joker_from_item(item)
        return replace(
            state,
            jokers=state.jokers + (joker,),
            modifiers=_modifiers_after_joker_added(state.modifiers, joker),
        )
    if _item_is_consumable(item):
        return replace(state, consumables=state.consumables + (_item_label(item),))

    playing_card = _playing_card_from_item(item)
    if playing_card is not None:
        known_deck = state.known_deck + (playing_card,) if state.known_deck else ()
        return replace(state, deck_size=state.deck_size + 1, known_deck=known_deck)
    return replace(state, consumables=state.consumables + (_item_label(item),))


def _state_after_pack_choice(state: GameState, item: object) -> GameState:
    planet_hand = _planet_hand_type(item)
    if planet_hand is not None:
        return _state_after_planet_used(state, planet_hand)

    playing_card = _playing_card_from_item(item)
    if playing_card is not None:
        known_deck = state.known_deck + (playing_card,) if state.known_deck else ()
        return replace(state, deck_size=state.deck_size + 1, known_deck=known_deck)

    if _item_is_consumable(item):
        return state

    return _state_after_acquiring_item(state, item)


def _state_after_planet_used(state: GameState, hand_type: HandType) -> GameState:
    hand_name = hand_type.value
    updated_levels = dict(state.hand_levels)
    updated_levels[hand_name] = max(1, _int_value(updated_levels.get(hand_name))) + 1
    updated_modifiers = dict(state.modifiers)
    hands = updated_modifiers.get("hands")
    if isinstance(hands, dict):
        updated_hands = dict(hands)
        entry = updated_hands.get(hand_name)
        if isinstance(entry, dict):
            updated_entry = dict(entry)
            updated_entry["level"] = max(1, _int_value(updated_entry.get("level"))) + 1
            updated_hands[hand_name] = updated_entry
        else:
            updated_hands[hand_name] = {"level": updated_levels[hand_name], "played": 0, "played_this_round": 0}
        updated_modifiers["hands"] = updated_hands
    return replace(state, hand_levels=updated_levels, modifiers=updated_modifiers)


def _hand_levels_after_play(
    hand_levels: dict[str, int],
    evaluation,
    *,
    stochastic_outcomes: dict[str, object],
) -> dict[str, int]:
    level_gain = _outcome_int(stochastic_outcomes, "space_joker_triggers")
    if level_gain <= 0:
        return hand_levels
    updated = dict(hand_levels)
    hand_name = evaluation.hand_type.value
    updated[hand_name] = max(1, _int_value(updated.get(hand_name))) + level_gain
    return updated


def _planet_hand_type(item: object) -> HandType | None:
    return PLANET_TO_HAND.get(_item_label(item))


def _joker_from_item(item: object) -> Joker:
    joker = Joker.from_mapping(item) if isinstance(item, dict | str) else Joker(str(item))
    if joker.sell_value is not None:
        return joker
    buy_cost = _item_buy_cost(item)
    sell_value = max(1, buy_cost // 2) if buy_cost > 0 else None
    return Joker(joker.name, edition=joker.edition, sell_value=sell_value, metadata=joker.metadata)


def _playing_card_from_item(item: object) -> Card | None:
    if not isinstance(item, dict):
        return None
    value = item.get("value")
    value = value if isinstance(value, dict) else {}
    key_card = _playing_card_from_key(item)
    if key_card is not None:
        return key_card
    if "rank" not in item and "rank" not in value:
        return None
    if "suit" not in item and "suit" not in value:
        return None
    return Card.from_mapping(item)


def _playing_card_from_key(item: dict[str, object]) -> Card | None:
    key = str(item.get("key", ""))
    match = re.fullmatch(r"([SHCD])_([2-9TJQKA])", key)
    if not match:
        return None
    suit, rank = match.groups()
    return Card(
        rank=rank,
        suit=suit,
        enhancement=_pack_card_enhancement(item),
        seal=_optional_str(item.get("seal")),
        edition=_optional_str(item.get("edition")),
        metadata=dict(item),
    )


def _pack_card_enhancement(item: dict[str, object]) -> str | None:
    label = _item_label(item).lower()
    if "base" in label:
        return None
    for name, enhancement in (
        ("bonus", "BONUS"),
        ("mult", "MULT"),
        ("wild", "WILD"),
        ("glass", "GLASS"),
        ("steel", "STEEL"),
        ("stone", "STONE"),
        ("gold", "GOLD"),
        ("lucky", "LUCKY"),
    ):
        if name in label:
            return enhancement
    return None


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _item_is_voucher(item: object) -> bool:
    return _item_set(item) == "VOUCHER"


def _item_is_joker(item: object) -> bool:
    label = _item_label(item).lower()
    key = _item_key(item)
    return _item_set(item) == "JOKER" or "joker" in label or key.startswith("j_")


def _item_is_consumable(item: object) -> bool:
    return _item_set(item) in {"TAROT", "PLANET", "SPECTRAL"} or _item_key(item).startswith("c_")


def _item_label(item: object) -> str:
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return str(item)
    return str(item.get("label", item.get("name", item.get("key", "unknown"))))


def _item_key(item: object) -> str:
    if not isinstance(item, dict):
        return ""
    return str(item.get("key", ""))


def _item_set(item: object) -> str:
    if not isinstance(item, dict):
        return ""
    return str(item.get("set", "")).upper()


def _item_buy_cost(item: object) -> int:
    if not isinstance(item, dict):
        return 0
    cost = item.get("cost", {})
    if isinstance(cost, dict):
        return _int_value(cost.get("buy", cost.get("cost", 0)))
    return _int_value(cost)


def _effective_item_buy_cost(state: GameState, item: object) -> int:
    if _astronomer_makes_item_free(state, item):
        return 0
    return _item_buy_cost(item)


def _astronomer_makes_item_free(state: GameState, item: object) -> bool:
    if not _effective_joker_count(state.jokers, "Astronomer"):
        return False
    item_set = _item_set(item)
    if item_set == "PLANET":
        return True
    label = _item_label(item).lower()
    key = _item_key(item).lower()
    return item_set == "BOOSTER" and ("celestial" in label or key.startswith("p_celestial"))


def _created_labels(items: Iterable[object]) -> tuple[str, ...]:
    return tuple(_item_label(item) for item in items)


def _play_consumable_creation_requests(
    state: GameState,
    played_cards: tuple[Card, ...],
    evaluation,
    *,
    sixth_sense_triggered: bool,
) -> tuple[tuple[str, int], ...]:
    requests: list[tuple[str, int]] = []
    room = _consumable_open_slots(state)

    if sixth_sense_triggered and room > 0:
        count = min(room, _effective_joker_count(state.jokers, "Sixth Sense"))
        if count > 0:
            requests.append(("spectral", count))
            room -= count

    tarot_count = 0
    if room > 0 and _hand_contains_straight(evaluation.hand_type) and _played_hand_has_ace(played_cards):
        tarot_count += _effective_joker_count(state.jokers, "Superposition")
    if room > 0 and _hand_is_straight_flush_family(evaluation.hand_type):
        tarot_count += _effective_joker_count(state.jokers, "Seance")
    if room > 0 and state.money <= 4:
        tarot_count += _effective_joker_count(state.jokers, "Vagabond")
    if room > 0:
        tarot_count += _effective_joker_count(state.jokers, "8 Ball") * _scored_rank_trigger_count(
            played_cards,
            evaluation,
            state.jokers,
            rank="8",
            hands_remaining=state.hands_remaining,
        )
    tarot_count = min(room, tarot_count)
    if tarot_count > 0:
        requests.append(("tarot", tarot_count))

    return tuple(requests)


def _creation_request_total(requests: tuple[tuple[str, int], ...]) -> int:
    return sum(count for _, count in requests)


def _with_creation_counts(
    modifiers: dict[str, object],
    requests: tuple[tuple[str, int], ...],
    *,
    created: int,
) -> dict[str, object]:
    updated = dict(modifiers)
    remaining = created
    for kind, requested_count in requests:
        if remaining <= 0:
            break
        count = min(remaining, requested_count)
        remaining -= count
        if kind == "spectral":
            updated = _increment_modifier_by(updated, "created_spectral_count", count)
            updated = _increment_modifier_by(updated, "created_consumable_count", count)
        elif kind == "tarot":
            updated = _with_created_tarot_count(updated, count)
            updated = _increment_modifier_by(updated, "created_consumable_count", count)
        else:
            updated = _increment_modifier_by(updated, f"created_{kind}_count", count)
            updated = _increment_modifier_by(updated, "created_consumable_count", count)
    return updated


def _with_created_tarot_count(modifiers: dict[str, object], count: int) -> dict[str, object]:
    if count <= 0:
        return modifiers
    return _increment_modifier_by(modifiers, "created_tarot_count", count)


def _cartomancer_creation_count(state: GameState) -> int:
    if not _has_consumable_room(state):
        return 0
    return min(_consumable_open_slots(state), _effective_joker_count(state.jokers, "Cartomancer"))


def _riff_raff_creation_count(state: GameState) -> int:
    count = 2 * _effective_joker_count(state.jokers, "Riff-raff")
    return min(count, _normal_joker_open_slots(state))


def _perkeo_creation_count(state: GameState) -> int:
    if not state.consumables:
        return 0
    return _effective_joker_count(state.jokers, "Perkeo")


def _hallucination_creation_count(state: GameState) -> int:
    if not _has_consumable_room(state):
        return 0
    return min(_consumable_open_slots(state), _effective_joker_count(state.jokers, "Hallucination"))


def _modifiers_after_joker_added(modifiers: dict[str, object], joker: Joker) -> dict[str, object]:
    return _modifiers_after_passive_joker(modifiers, joker, sign=1)


def _modifiers_after_joker_removed(modifiers: dict[str, object], joker: Joker) -> dict[str, object]:
    return _modifiers_after_passive_joker(modifiers, joker, sign=-1)


def _modifiers_after_passive_joker(modifiers: dict[str, object], joker: Joker, *, sign: int) -> dict[str, object]:
    if _joker_is_disabled(joker):
        return dict(modifiers)

    updated = dict(modifiers)
    if joker.name == "Credit Card":
        _add_modifier_int(updated, "bankrupt_at", -20 * sign)
    elif joker.name == "Drunkard":
        _add_modifier_int(updated, "round_discards_delta", _joker_int(joker, keys=("d_size", "discards"), default=1) * sign)
    elif joker.name == "Juggler":
        _add_modifier_int(updated, "hand_size_delta", _joker_int(joker, keys=("h_size", "hand_size"), default=1) * sign)
    elif joker.name == "Merry Andy":
        _add_modifier_int(updated, "round_discards_delta", _joker_int(joker, keys=("d_size", "discards"), default=3) * sign)
        _add_modifier_int(updated, "hand_size_delta", _joker_int(joker, keys=("h_size", "hand_size"), default=-1) * sign)
    elif joker.name == "Troubadour":
        _add_modifier_int(updated, "hand_size_delta", _joker_int(joker, keys=("h_size", "hand_size"), default=2) * sign)
        _add_modifier_int(updated, "round_hands_delta", _joker_int(joker, keys=("h_plays", "hands"), default=-1) * sign)
    elif joker.name == "Stuntman":
        _add_modifier_int(updated, "hand_size_delta", -_joker_int(joker, keys=("h_size", "hand_size"), default=2) * sign)
    elif joker.name == "Turtle Bean":
        hand_size = _joker_int(joker, keys=("current_h_size", "current_hand_size", "h_size", "hand_size"), default=5)
        _add_modifier_int(updated, "hand_size_delta", hand_size * sign)
    elif joker.name == "Chaos the Clown":
        next_free = _int_value(updated.get("free_rerolls")) + (
            _joker_int(joker, keys=("extra", "free_rerolls"), default=1) * sign
        )
        updated["free_rerolls"] = max(0, next_free)
    elif joker.name == "To the Moon":
        default_interest = 2 if sign < 0 else 1
        current = _int_value(updated.get("interest_amount", default_interest))
        updated["interest_amount"] = max(0, current + (_joker_int(joker, keys=("extra", "interest"), default=1) * sign))
    elif joker.name == "Oops! All 6s":
        current = _float_value(updated.get("probability_multiplier", 1.0))
        updated["probability_multiplier"] = current * (2.0 if sign > 0 else 0.5)
    elif joker.name == "Showman":
        next_count = max(0, _int_value(updated.get("showman_count")) + sign)
        updated["showman_count"] = next_count
        updated["allow_duplicate_jokers"] = next_count > 0
    return updated


def _add_modifier_int(modifiers: dict[str, object], key: str, delta: int) -> None:
    modifiers[key] = _int_value(modifiers.get(key)) + delta


def _modifiers_after_cash_out(
    previous_modifiers: dict[str, object],
    next_modifiers: dict[str, object],
    previous_jokers: tuple[Joker, ...],
    next_jokers: tuple[Joker, ...],
) -> dict[str, object]:
    if "hand_size_delta" not in previous_modifiers:
        return next_modifiers
    previous_turtle_size = _active_turtle_bean_hand_size(previous_jokers)
    next_turtle_size = _active_turtle_bean_hand_size(next_jokers)
    if previous_turtle_size == next_turtle_size:
        return next_modifiers
    updated = dict(next_modifiers)
    _add_modifier_int(updated, "hand_size_delta", next_turtle_size - previous_turtle_size)
    return updated


def _active_turtle_bean_hand_size(jokers: tuple[Joker, ...]) -> int:
    total = 0
    for joker in jokers:
        if joker.name == "Turtle Bean" and not _joker_is_disabled(joker):
            total += _joker_int(joker, keys=("current_h_size", "current_hand_size", "h_size", "hand_size"), default=5)
    return total


def _modifiers_after_sell_trigger(
    state: GameState,
    modifiers: dict[str, object],
    sold: Joker,
) -> dict[str, object]:
    if _joker_is_disabled(sold):
        return modifiers
    if sold.name == "Diet Cola":
        return _with_added_tag(modifiers, "Double Tag")
    if sold.name == "Luchador" and _blind_is_boss(state):
        updated = dict(modifiers)
        updated["boss_disabled"] = True
        return updated
    return modifiers


def _with_added_tag(modifiers: dict[str, object], tag: str) -> dict[str, object]:
    updated = dict(modifiers)
    raw = updated.get("tags", ())
    tags = tuple(raw) if isinstance(raw, list | tuple) else ()
    updated["tags"] = tags + (tag,)
    return updated


def _modifiers_after_gift_card_cash_out(
    modifiers: dict[str, object],
    state: GameState,
    *,
    gift_card_gain: int,
) -> dict[str, object]:
    if gift_card_gain <= 0 or not state.consumables:
        return modifiers
    updated = dict(modifiers)
    _add_modifier_int(updated, "consumable_sell_value_bonus", gift_card_gain * len(state.consumables))
    return updated


def _spend_reroll(state: GameState) -> tuple[int, dict[str, object]]:
    updated = dict(state.modifiers)
    free_rerolls = _int_value(updated.get("free_rerolls"))
    if free_rerolls > 0:
        updated["free_rerolls"] = free_rerolls - 1
        return 0, updated

    cost = _int_value(updated.get("reroll_cost", updated.get("current_reroll_cost", 5))) or 5
    updated["reroll_cost"] = cost + 1
    updated["current_reroll_cost"] = cost + 1
    updated["reroll_cost_increase"] = _int_value(updated.get("reroll_cost_increase")) + 1
    return cost, updated


def _jokers_after_reroll(jokers: tuple[Joker, ...]) -> tuple[Joker, ...]:
    updated: list[Joker] = []
    for joker in jokers:
        if joker.name == "Flash Card" and not _joker_is_disabled(joker):
            current = _joker_int(joker, keys=("current_mult", "mult"), default=0)
            gain = _joker_int(joker, keys=("extra", "mult_gain"), default=2)
            updated.append(_with_joker_metadata(joker, {"current_mult": current + gain}))
        else:
            updated.append(joker)
    return tuple(updated)


def _jokers_after_gift_card_cash_out(jokers: tuple[Joker, ...], *, gift_card_gain: int) -> tuple[Joker, ...]:
    if gift_card_gain <= 0:
        return jokers
    return tuple(_with_joker_sell_value_delta(joker, gift_card_gain) for joker in jokers)


def _with_joker_sell_value_delta(joker: Joker, delta: int) -> Joker:
    return _with_joker_metadata(joker, {}, sell_value=(joker.sell_value or 0) + delta)


def _jokers_after_sell(jokers: tuple[Joker, ...]) -> tuple[Joker, ...]:
    updated: list[Joker] = []
    for joker in jokers:
        if joker.name == "Campfire" and not _joker_is_disabled(joker):
            current = _joker_float(joker, keys=("current_xmult", "xmult", "current_x_mult", "x_mult"), default=1.0)
            gain = _joker_float(joker, keys=("extra", "xmult_gain"), default=0.25)
            updated.append(_with_joker_metadata(joker, {"current_xmult": current + gain}))
        else:
            updated.append(joker)
    return tuple(updated)


def _jokers_after_play(
    jokers: tuple[Joker, ...],
    played_cards: tuple[Card, ...],
    evaluation,
    *,
    hands_remaining: int,
    stochastic_outcomes: dict[str, object],
) -> tuple[Joker, ...]:
    updated: list[Joker] = []
    vampire_gains = _vampire_xmult_gains_after_play(jokers, played_cards, evaluation)
    for joker_index, joker in enumerate(jokers):
        if joker.name == "Ice Cream":
            current = _joker_int(joker, keys=("current_chips", "chips"), default=100)
            if current - 5 > 0:
                updated.append(_with_joker_metadata(joker, {"current_chips": current - 5}))
        elif joker.name == "Seltzer":
            current = _joker_int(joker, keys=("current_remaining", "remaining", "uses", "extra"), default=10)
            if current - 1 > 0:
                updated.append(_with_joker_metadata(joker, {"current_remaining": current - 1}))
        elif joker.name == "Green Joker":
            current = _joker_int(joker, keys=("current_mult", "mult"), default=0)
            updated.append(_with_joker_metadata(joker, {"current_mult": current + 1}))
        elif joker.name == "Ride the Bus":
            current = _joker_int(joker, keys=("current_mult", "mult"), default=0)
            next_mult = 0 if _scored_hand_has_face(played_cards, evaluation, jokers) else current + 1
            updated.append(_with_joker_metadata(joker, {"current_mult": next_mult}))
        elif joker.name == "Square Joker":
            current = _joker_int(joker, keys=("current_chips", "chips"), default=0)
            gain = 4 if len(played_cards) == 4 else 0
            updated.append(_with_joker_metadata(joker, {"current_chips": current + gain}))
        elif joker.name == "Runner":
            current = _joker_int(joker, keys=("current_chips", "chips"), default=0)
            gain = 15 if _hand_contains_straight(evaluation.hand_type) else 0
            updated.append(_with_joker_metadata(joker, {"current_chips": current + gain}))
        elif joker.name == "Spare Trousers":
            current = _joker_int(joker, keys=("current_mult", "mult"), default=0)
            gain = 2 if _hand_contains_two_pair_for_trousers(evaluation.hand_type) else 0
            updated.append(_with_joker_metadata(joker, {"current_mult": current + gain}))
        elif joker.name == "Wee Joker":
            current = _joker_int(joker, keys=("current_chips", "chips"), default=0)
            gain = 8 * _scored_two_trigger_count(played_cards, evaluation, jokers, hands_remaining=hands_remaining)
            updated.append(_with_joker_metadata(joker, {"current_chips": current + gain}))
        elif joker.name == "Loyalty Card":
            remaining = _joker_int_or_none(joker, keys=("current_remaining", "remaining", "hands_remaining", "hands_left"))
            if remaining is None:
                updated.append(joker)
            else:
                updated.append(_with_joker_metadata(joker, {"current_remaining": 5 if remaining <= 0 else remaining - 1}))
        elif joker.name == "Vampire":
            current = _joker_float(joker, keys=("current_xmult", "xmult", "current_x_mult", "x_mult"), default=1.0)
            gain = vampire_gains[joker_index] if joker_index < len(vampire_gains) else 0.0
            updated.append(_with_joker_metadata(joker, {"current_xmult": current + gain}))
        elif joker.name == "Lucky Cat":
            current = _joker_float(joker, keys=("current_xmult", "xmult", "current_x_mult", "x_mult"), default=1.0)
            gain = 0.25 * max(0, _outcome_int(stochastic_outcomes, "lucky_card_triggers"))
            updated.append(_with_joker_metadata(joker, {"current_xmult": current + gain}) if gain else joker)
        else:
            updated.append(joker)
    return tuple(updated)


def _played_cards_after_play(
    state: GameState,
    played_cards: tuple[Card, ...],
    evaluation,
) -> tuple[Card, ...]:
    scoring_indices = set(evaluation.scoring_indices)
    updated = list(played_cards)

    for joker in state.jokers:
        if _joker_is_disabled(joker):
            continue
        if joker.name == "Midas Mask":
            for index in scoring_indices:
                card = updated[index]
                if _is_face_card(card, state.jokers):
                    updated[index] = _gold_card(card)
        elif joker.name == "Vampire":
            for index in scoring_indices:
                card = updated[index]
                if _vampire_can_strip(card):
                    updated[index] = _base_card(card)

    hiker_count = sum(1 for joker in _active_effective_jokers(state.jokers) if joker.name == "Hiker")
    if hiker_count:
        for index in scoring_indices:
            updated[index] = _with_card_perma_bonus(updated[index], 5 * hiker_count)
    return tuple(updated)


def _vampire_xmult_gains_after_play(
    jokers: tuple[Joker, ...],
    played_cards: tuple[Card, ...],
    evaluation,
) -> tuple[float, ...]:
    scoring_indices = set(evaluation.scoring_indices)
    updated = list(played_cards)
    gains = [0.0 for _ in jokers]
    for joker_index, joker in enumerate(jokers):
        if _joker_is_disabled(joker):
            continue
        if joker.name == "Midas Mask":
            for index in scoring_indices:
                card = updated[index]
                if _is_face_card(card, jokers):
                    updated[index] = _gold_card(card)
        elif joker.name == "Vampire":
            stripped = 0
            for index in scoring_indices:
                card = updated[index]
                if _vampire_can_strip(card):
                    updated[index] = _base_card(card)
                    stripped += 1
            gains[joker_index] = 0.1 * stripped
    return tuple(gains)


def _jokers_after_discard(jokers: tuple[Joker, ...], discarded_cards: tuple[Card, ...]) -> tuple[Joker, ...]:
    updated: list[Joker] = []
    for joker in jokers:
        if joker.name == "Ramen":
            current = _joker_float(joker, keys=("current_xmult", "xmult", "current_x_mult", "x_mult"), default=2.0)
            eaten = False
            for _ in discarded_cards:
                if current - 0.01 <= 1:
                    eaten = True
                    break
                current -= 0.01
            if not eaten:
                updated.append(_with_joker_metadata(joker, {"current_xmult": current}))
        elif joker.name == "Green Joker":
            current = _joker_int(joker, keys=("current_mult", "mult"), default=0)
            updated.append(_with_joker_metadata(joker, {"current_mult": max(0, current - 1)}))
        elif joker.name == "Hit the Road":
            current = _joker_float(joker, keys=("current_xmult", "xmult", "current_x_mult", "x_mult"), default=1.0)
            jack_count = sum(1 for card in discarded_cards if not card.debuffed and card.rank == "J")
            updated.append(_with_joker_metadata(joker, {"current_xmult": current + (0.5 * jack_count)}))
        elif joker.name == "Castle":
            target_suit = _joker_target_suit(joker)
            if target_suit is None:
                updated.append(joker)
                continue
            current = _joker_int(joker, keys=("current_chips", "chips"), default=0)
            gain = 3 * sum(
                1 for card in discarded_cards if not card.debuffed and _normalize_suit(card.suit) == target_suit
            )
            updated.append(_with_joker_metadata(joker, {"current_chips": current + gain}))
        elif joker.name == "Yorick":
            remaining = _joker_countdown_or_none(
                joker,
                keys=("current_remaining", "remaining", "discards_remaining", "yorick_discards"),
            )
            if remaining is None:
                updated.append(joker)
                continue
            current = _joker_float(joker, keys=("current_xmult", "xmult", "current_x_mult", "x_mult"), default=1.0)
            for _ in discarded_cards:
                if remaining <= 1:
                    remaining = 23
                    current += 1.0
                else:
                    remaining -= 1
            updated.append(_with_joker_metadata(joker, {"current_remaining": remaining, "current_xmult": current}))
        else:
            updated.append(joker)
    return tuple(updated)


def _jokers_after_cash_out(state: GameState, *, next_to_do_targets: tuple[str, ...]) -> tuple[Joker, ...]:
    to_do_count = sum(
        1 for joker in state.jokers if joker.name == "To Do List" and not _joker_is_disabled(joker)
    )
    if len(next_to_do_targets) != to_do_count:
        raise ValueError(f"To Do List needs {to_do_count} injected target(s), got {len(next_to_do_targets)}")

    to_do_index = 0
    updated: list[Joker] = []
    boss = _blind_is_boss(state)
    for joker in state.jokers:
        if _joker_is_disabled(joker):
            updated.append(joker)
        elif joker.name == "Campfire" and boss:
            current = _joker_float(joker, keys=("current_xmult", "xmult", "current_x_mult", "x_mult"), default=1.0)
            updated.append(_with_joker_metadata(joker, {"current_xmult": 1.0}) if current > 1 else joker)
        elif joker.name == "Rocket" and boss:
            current = _rocket_dollars(joker)
            increase = _joker_int(joker, keys=("increase", "dollar_increase"), default=2)
            updated.append(_with_joker_metadata(joker, {"current_dollars": current + increase}))
        elif joker.name == "Popcorn":
            current = _joker_int(joker, keys=("current_mult", "mult"), default=20)
            decay = _joker_int(joker, keys=("decay", "mult_decay", "extra"), default=4)
            next_mult = current - decay
            if next_mult > 0:
                updated.append(_with_joker_metadata(joker, {"current_mult": next_mult}))
        elif joker.name == "Turtle Bean":
            current = _joker_int(joker, keys=("current_h_size", "current_hand_size", "h_size", "hand_size"), default=5)
            decay = _joker_int(joker, keys=("h_mod", "hand_size_decay", "decay"), default=1)
            next_h_size = current - decay
            if next_h_size > 0:
                updated.append(_with_joker_metadata(joker, {"current_h_size": next_h_size}))
        elif joker.name == "Invisible Joker":
            current = _joker_int(joker, keys=("current_rounds", "invis_rounds", "rounds"), default=0)
            updated.append(_with_joker_metadata(joker, {"current_rounds": current + 1, "invis_rounds": current + 1}))
        elif joker.name == "Egg":
            gain = _joker_int(joker, keys=("extra", "value_gain"), default=3)
            extra_value = _joker_int(joker, keys=("extra_value",), default=0) + gain
            sell_value = (joker.sell_value + gain) if joker.sell_value is not None else None
            updated.append(_with_joker_metadata(joker, {"extra_value": extra_value}, sell_value=sell_value))
        elif joker.name == "Hit the Road":
            current = _joker_float(joker, keys=("current_xmult", "xmult", "current_x_mult", "x_mult"), default=1.0)
            updated.append(_with_joker_metadata(joker, {"current_xmult": 1.0}) if current > 1 else joker)
        elif joker.name == "To Do List":
            target = next_to_do_targets[to_do_index]
            to_do_index += 1
            updated.append(_with_joker_metadata(joker, {"target_hand": target, "to_do_poker_hand": target}))
        else:
            updated.append(joker)
    return tuple(updated)


def _jokers_after_stochastic_removals(jokers: tuple[Joker, ...], removed_names: tuple[str, ...]) -> tuple[Joker, ...]:
    if not removed_names:
        return jokers
    remaining_removals = Counter(removed_names)
    updated: list[Joker] = []
    for joker in jokers:
        if remaining_removals[joker.name] > 0:
            remaining_removals[joker.name] -= 1
            continue
        updated.append(joker)
    return tuple(updated)


def _gift_card_sell_value_gain(jokers: tuple[Joker, ...]) -> int:
    return _effective_joker_count(jokers, "Gift Card")


def _hand_levels_after_discard(state: GameState, discarded_cards: tuple[Card, ...]) -> dict[str, int]:
    if not _has_joker(state.jokers, "Burnt Joker"):
        return state.hand_levels
    blind_name = _effective_blind_name(state)
    if blind_name == "The Hook" or _discard_used_count(state) > 0:
        return state.hand_levels
    evaluation = evaluate_played_cards(
        discarded_cards,
        state.hand_levels,
        debuffed_suits=debuffed_suits_for_blind(blind_name),
        blind_name=blind_name,
        jokers=state.jokers,
        discards_remaining=state.discards_remaining,
        hands_remaining=state.hands_remaining,
        held_cards=(),
        deck_size=state.deck_size,
        money=state.money,
        played_hand_types_this_round=_played_hand_types_this_round(state),
        played_hand_counts=_played_hand_counts(state),
    )
    updated = dict(state.hand_levels)
    updated[evaluation.hand_type.value] = max(1, _int_value(updated.get(evaluation.hand_type.value))) + 1
    return updated


def _discard_money_delta(state: GameState, discarded_cards: tuple[Card, ...]) -> int:
    delta = 0
    if _discard_used_count(state) == 0 and len(discarded_cards) == 1:
        for joker in state.jokers:
            if joker.name == "Trading Card":
                delta += _joker_int(joker, keys=("dollars", "money", "extra"), default=3)
    if _has_joker(state.jokers, "Faceless Joker"):
        face_count = sum(1 for card in discarded_cards if _is_face_card(card, state.jokers))
        if face_count >= 3:
            delta += 5
    mail_rank = _mail_in_rebate_rank(state.jokers)
    if mail_rank is not None:
        delta += 5 * sum(1 for card in discarded_cards if not card.debuffed and _rank_matches(card.rank, mail_rank))
    return delta


def _held_end_of_round_money_delta(held_cards: tuple[Card, ...], jokers: tuple[Joker, ...]) -> int:
    mime_retriggers = sum(1 for joker in jokers if joker.name == "Mime" and not _joker_is_disabled(joker))
    trigger_count = 1 + mime_retriggers
    return 3 * trigger_count * sum(1 for card in held_cards if _is_gold_card(card))


def _cash_out_money_delta(state: GameState) -> int:
    delta = _blind_reward(state)
    delta += max(0, state.hands_remaining) * _modifier_int(state.modifiers, ("money_per_hand", "dollars_per_hand"), default=1)

    money_per_discard = _modifier_int_or_none(state.modifiers, ("money_per_discard", "dollars_per_discard"))
    if money_per_discard is not None and state.discards_remaining > 0:
        delta += state.discards_remaining * money_per_discard

    delta += _joker_cash_out_money_delta(state)
    delta += _interest_money_delta(state)
    return delta


def _cash_out_base_hands(state: GameState) -> int:
    configured = _modifier_int_or_none(
        state.modifiers,
        ("base_hands", "base_hands_remaining", "hands_per_round", "hands_per_blind", "default_hands"),
    )
    if configured is not None:
        return max(0, configured)
    passive_base = max(0, 4 + _int_value(state.modifiers.get("round_hands_delta")))
    round_start = state.hands_remaining + len(_played_hand_types_this_round(state))
    temporary_bonus = 3 * _effective_joker_count(state.jokers, "Burglar")
    return max(passive_base, round_start - temporary_bonus)


def _cash_out_base_discards(state: GameState) -> int:
    configured = _modifier_int_or_none(
        state.modifiers,
        (
            "base_discards",
            "base_discards_remaining",
            "discards_per_round",
            "discards_per_blind",
            "default_discards",
        ),
    )
    if configured is not None:
        return max(0, configured)
    used = _modifier_int_or_none(
        state.modifiers,
        ("round_discards_used", "discards_used", "discards_used_this_round"),
    )
    if used is not None:
        return max(4, state.discards_remaining + used)
    return max(4, state.discards_remaining)


def _blind_reward(state: GameState) -> int:
    for key in ("blind_reward", "blind_dollars", "current_blind_reward"):
        if key in state.modifiers:
            return _int_value(state.modifiers.get(key))

    current_blind = state.modifiers.get("current_blind")
    if isinstance(current_blind, dict):
        for key in ("dollars", "reward", "cash_reward"):
            if key in current_blind:
                return _int_value(current_blind.get(key))
        blind_type = str(current_blind.get("type", current_blind.get("kind", ""))).upper()
        if blind_type == "SMALL":
            return 3
        if blind_type == "BIG":
            return 4
        if blind_type == "BOSS":
            return 5

    if state.blind == "Small Blind":
        return 3
    if state.blind == "Big Blind":
        return 4
    return 5 if state.blind else 0


def _joker_cash_out_money_delta(state: GameState) -> int:
    delta = 0
    for joker in state.jokers:
        if _joker_is_disabled(joker):
            continue
        if joker.name == "Golden Joker":
            delta += _joker_int(joker, keys=("current_dollars", "dollars", "extra"), default=4)
        elif joker.name == "Rocket":
            delta += _rocket_dollars(joker)
        elif joker.name == "Delayed Gratification":
            if _discard_used_count(state) == 0 and state.discards_remaining > 0:
                dollars = _joker_int(joker, keys=("dollars", "extra"), default=2)
                delta += state.discards_remaining * dollars
        elif joker.name == "Cloud 9":
            dollars = _joker_int(joker, keys=("dollars", "extra"), default=1)
            delta += _cloud_9_nine_count(state, joker) * dollars
        elif joker.name == "Satellite":
            dollars = _joker_int(joker, keys=("dollars", "extra"), default=1)
            delta += _satellite_planet_count(state, joker) * dollars
    return delta


def _cloud_9_nine_count(state: GameState, joker: Joker) -> int:
    visible_count = _joker_int_or_none(joker, keys=("current_nines", "nine_tally", "nines"))
    if visible_count is not None:
        return visible_count
    return sum(1 for card in _known_full_deck_cards(state) if _normalize_rank(card.rank) == "9")


def _satellite_planet_count(state: GameState, joker: Joker) -> int:
    visible_count = _joker_int_or_none(joker, keys=("current_planets", "planet_count", "unique_planets", "planets"))
    if visible_count is not None:
        return visible_count
    for key in ("planet_cards_used", "used_planets", "consumable_usage", "consumeable_usage"):
        raw = state.modifiers.get(key)
        if isinstance(raw, dict):
            return sum(1 for value in raw.values() if _int_value(value) > 0)
        if isinstance(raw, list | tuple | set):
            return len({str(item) for item in raw})
    return 0


def _rocket_dollars(joker: Joker) -> int:
    visible_value = _joker_int_or_none(joker, keys=("current_dollars", "dollars"))
    if visible_value is not None:
        return visible_value
    match = re.search(r"earn\s+\$?\s*([0-9]+)", _joker_effect_text(joker), flags=re.IGNORECASE)
    return int(match.group(1)) if match else 1


def _known_full_deck_cards(state: GameState) -> tuple[Card, ...]:
    return (
        state.hand
        + state.known_deck
        + _zone_cards(state.modifiers, "played_pile")
        + _zone_cards(state.modifiers, "discard_pile")
    )


def _interest_money_delta(state: GameState) -> int:
    if _truthy_modifier(state.modifiers.get("no_interest")):
        return 0
    cap_money = _interest_cap_money(state)
    if cap_money <= 0 or state.money < 5:
        return 0
    configured_amount = _modifier_int_or_none(state.modifiers, ("interest_amount", "interest_per_5"))
    amount = configured_amount if configured_amount is not None else 1 + _effective_joker_count(state.jokers, "To the Moon")
    return amount * (min(state.money, cap_money) // 5)


def _interest_cap_money(state: GameState) -> int:
    cap = _modifier_int(state.modifiers, ("interest_cap_money", "interest_cap"), default=25)
    if "Seed Money" in state.vouchers:
        cap = max(cap, 50)
    if "Money Tree" in state.vouchers:
        cap = max(cap, 100)
    return cap


def _modifier_int(modifiers: dict[str, object], keys: tuple[str, ...], *, default: int) -> int:
    value = _modifier_int_or_none(modifiers, keys)
    return default if value is None else value


def _modifier_int_or_none(modifiers: dict[str, object], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        if key in modifiers:
            try:
                return int(modifiers[key])
            except (TypeError, ValueError):
                return None
    return None


def _truthy_modifier(value: object) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _scored_hand_has_face(cards: tuple[Card, ...], evaluation, jokers: tuple[Joker, ...]) -> bool:
    return any(_is_face_card(cards[index], jokers) for index in evaluation.scoring_indices)


def _scored_two_trigger_count(
    cards: tuple[Card, ...],
    evaluation,
    jokers: tuple[Joker, ...],
    *,
    hands_remaining: int,
) -> int:
    return _scored_rank_trigger_count(cards, evaluation, jokers, rank="2", hands_remaining=hands_remaining)


def _scored_rank_trigger_count(
    cards: tuple[Card, ...],
    evaluation,
    jokers: tuple[Joker, ...],
    *,
    rank: str,
    hands_remaining: int,
) -> int:
    total = 0
    first_scored_index = evaluation.scoring_indices[0] if evaluation.scoring_indices else None
    for index in evaluation.scoring_indices:
        card = cards[index]
        if not _rank_matches(card.rank, rank):
            continue
        triggers = 1
        if _normalized(card.seal) == "red":
            triggers += 1
        if _has_joker(jokers, "Hack") and _normalize_rank(rank) in {"2", "3", "4", "5"}:
            triggers += 1
        if _has_joker(jokers, "Dusk") and hands_remaining == 1:
            triggers += 1
        if _has_joker(jokers, "Seltzer"):
            triggers += 1
        if _has_joker(jokers, "Hanging Chad") and index == first_scored_index:
            triggers += 2
        total += triggers
    return total


def _hand_contains_straight(hand_type: HandType) -> bool:
    return hand_type in {HandType.STRAIGHT, HandType.STRAIGHT_FLUSH}


def _hand_is_straight_flush_family(hand_type: HandType) -> bool:
    return hand_type == HandType.STRAIGHT_FLUSH


def _hand_contains_two_pair_for_trousers(hand_type: HandType) -> bool:
    return hand_type in {HandType.TWO_PAIR, HandType.FULL_HOUSE, HandType.FLUSH_HOUSE}


def _played_hand_has_ace(cards: tuple[Card, ...]) -> bool:
    return any(_normalize_rank(card.rank) == "A" for card in cards)


def _is_face_card(card: Card, jokers: tuple[Joker, ...]) -> bool:
    if _normalized(card.enhancement) in {"stone", "stone card"}:
        return False
    return card.rank in {"J", "Q", "K"} or _has_joker(jokers, "Pareidolia")


def _vampire_can_strip(card: Card) -> bool:
    if card.debuffed:
        return False
    return _normalized(card.enhancement) not in {"", "base", "base card"}


def _is_gold_card(card: Card) -> bool:
    return not card.debuffed and _normalized(card.enhancement) in {"gold", "gold card"}


def _gold_card(card: Card) -> Card:
    metadata = dict(card.metadata)
    modifier = metadata.get("modifier")
    if isinstance(modifier, dict):
        modifier = dict(modifier)
    else:
        modifier = {}
    modifier["enhancement"] = "GOLD"
    metadata["modifier"] = modifier
    return replace(card, enhancement="GOLD", metadata=metadata)


def _base_card(card: Card) -> Card:
    metadata = dict(card.metadata)
    modifier = metadata.get("modifier")
    if isinstance(modifier, dict):
        modifier = dict(modifier)
        modifier["enhancement"] = ""
        metadata["modifier"] = modifier
    return replace(card, enhancement=None, metadata=metadata)


def _with_card_perma_bonus(card: Card, amount: int) -> Card:
    metadata = dict(card.metadata)
    modifier = metadata.get("modifier")
    if isinstance(modifier, dict):
        modifier = dict(modifier)
    else:
        modifier = {}
    modifier["perma_bonus"] = _card_perma_bonus(card) + amount
    metadata["modifier"] = modifier
    return replace(card, metadata=metadata)


def _card_perma_bonus(card: Card) -> int:
    for source in _card_metadata_sources(card):
        for key in ("perma_bonus", "permanent_chips", "bonus_chips", "extra_chips"):
            if key in source:
                return _int_value(source[key])
    return 0


def _card_metadata_sources(card: Card) -> tuple[dict[str, object], ...]:
    sources: list[dict[str, object]] = [card.metadata]
    for key in ("modifier", "value"):
        value = card.metadata.get(key)
        if isinstance(value, dict):
            sources.append(value)
    return tuple(sources)


def _has_joker(jokers: tuple[Joker, ...], name: str) -> bool:
    return any(joker.name == name for joker in jokers)


def _effective_joker_count(jokers: tuple[Joker, ...], name: str) -> int:
    return sum(1 for joker in _active_effective_jokers(jokers) if joker.name == name)


def _effective_blind_name(state: GameState) -> str:
    return "" if _truthy_modifier(state.modifiers.get("boss_disabled")) else state.blind


def _mr_bones_saves(state: GameState, score: int) -> bool:
    if state.required_score <= 0 or score < state.required_score * 0.25:
        return False
    return any(joker.name == "Mr. Bones" and not _joker_is_disabled(joker) for joker in state.jokers)


def _jokers_after_mr_bones_save(jokers: tuple[Joker, ...]) -> tuple[Joker, ...]:
    removed = False
    updated: list[Joker] = []
    for joker in jokers:
        if not removed and joker.name == "Mr. Bones" and not _joker_is_disabled(joker):
            removed = True
            continue
        updated.append(joker)
    return tuple(updated)


def _active_effective_jokers(jokers: tuple[Joker, ...]) -> tuple[Joker, ...]:
    sources = _effective_ability_joker_indices(jokers)
    effective = tuple(jokers[index] for index in sources)
    return tuple(
        ability
        for physical, ability in zip(jokers, effective, strict=False)
        if not _joker_is_disabled(physical) and not _joker_is_disabled(ability)
    )


def _effective_ability_joker_indices(jokers: tuple[Joker, ...]) -> tuple[int, ...]:
    def resolve(index: int, seen: frozenset[int] = frozenset()) -> int:
        joker = jokers[index]
        if index in seen:
            return index
        if joker.name == "Blueprint" and index + 1 < len(jokers):
            return resolve(index + 1, seen | {index})
        if joker.name == "Brainstorm" and index != 0:
            return resolve(0, seen | {index})
        return index

    return tuple(resolve(index) for index in range(len(jokers)))


def _joker_is_disabled(joker: Joker) -> bool:
    state = joker.metadata.get("state")
    if isinstance(state, dict) and state.get("debuff"):
        return True
    return "all abilities are disabled" in _joker_effect_text(joker).lower()


def _with_joker_metadata(joker: Joker, updates: dict[str, object], *, sell_value: int | None = None) -> Joker:
    return Joker(
        name=joker.name,
        edition=joker.edition,
        sell_value=joker.sell_value if sell_value is None else sell_value,
        metadata={**joker.metadata, **updates},
    )


def _joker_int(joker: Joker, *, keys: tuple[str, ...], default: int) -> int:
    value = _joker_int_or_none(joker, keys=keys)
    return default if value is None else value


def _joker_int_or_none(joker: Joker, *, keys: tuple[str, ...]) -> int | None:
    value = _joker_numeric_value(joker, keys=keys)
    if value is None:
        return _current_int_from_effect(joker)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _joker_float(joker: Joker, *, keys: tuple[str, ...], default: float) -> float:
    value = _joker_numeric_value(joker, keys=keys)
    if value is None:
        value = _current_xmult_from_effect(joker)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _joker_numeric_value(joker: Joker, *, keys: tuple[str, ...]) -> object | None:
    for source in _metadata_sources(joker.metadata):
        for key in keys:
            if key in source:
                return source[key]
    return None


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


def _current_int_from_effect(joker: Joker) -> int | None:
    match = re.search(r"currently\s+([+-]?\d+)", _joker_effect_text(joker), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _joker_countdown_or_none(joker: Joker, *, keys: tuple[str, ...]) -> int | None:
    value = _joker_numeric_value(joker, keys=keys)
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    effect = _joker_effect_text(joker)
    match = re.search(r"([0-9]+)\s+(?:cards?\s+)?remaining", effect, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _current_xmult_from_effect(joker: Joker) -> float | None:
    effect = _joker_effect_text(joker)
    match = re.search(r"currently\s+x\s*([0-9]+(?:\.[0-9]+)?)", effect, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"\bx\s*([0-9]+(?:\.[0-9]+)?)\s*mult", effect, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def _joker_effect_text(joker: Joker) -> str:
    value = joker.metadata.get("value")
    if isinstance(value, dict):
        return str(value.get("effect", ""))
    return str(joker.metadata.get("effect", ""))


def _normalized(value: object) -> str:
    return str(value or "").removeprefix("m_").replace("_", " ").lower()


def _normalize_suit(suit: str) -> str:
    return {
        "Spades": "S",
        "Spade": "S",
        "Hearts": "H",
        "Heart": "H",
        "Clubs": "C",
        "Club": "C",
        "Diamonds": "D",
        "Diamond": "D",
    }.get(suit, suit)


def _rank_matches(card_rank: str, target_rank: str) -> bool:
    if target_rank == "10":
        return card_rank in {"10", "T"}
    return card_rank == target_rank


def _joker_target_suit(joker: Joker) -> str | None:
    for source in _metadata_sources(joker.metadata):
        for key in ("current_suit", "target_suit", "suit"):
            value = source.get(key)
            if isinstance(value, str) and value:
                return _normalize_suit(value)
    effect = _joker_effect_text(joker)
    match = re.search(r"discarded\s+(Spade|Heart|Club|Diamond)", effect, flags=re.IGNORECASE)
    return _normalize_suit(match.group(1)) if match else None


def _blind_is_boss(state: GameState) -> bool:
    current_blind = state.modifiers.get("current_blind")
    if isinstance(current_blind, dict):
        blind_type = str(current_blind.get("type", current_blind.get("kind", ""))).upper()
        if blind_type == "BOSS":
            return True
        if blind_type in {"SMALL", "BIG"}:
            return False
    name = state.blind.strip()
    return bool(name and name not in {"Small Blind", "Big Blind"})


def _mail_in_rebate_rank(jokers: tuple[Joker, ...]) -> str | None:
    for joker in jokers:
        if joker.name != "Mail-In Rebate":
            continue
        for source in _metadata_sources(joker.metadata):
            for key in ("current_rank", "target_rank", "rank"):
                value = source.get(key)
                if isinstance(value, str) and value:
                    return _normalize_rank(value)
        return _rank_from_text(_joker_effect_text(joker))
    return None


def _rank_from_text(text: str) -> str | None:
    match = re.search(
        r"\b(Ace|King|Queen|Jack|Ten|Nine|Eight|Seven|Six|Five|Four|Three|Two|K|Q|J|10|T|[2-9])\b",
        text,
        flags=re.IGNORECASE,
    )
    return _normalize_rank(match.group(1)) if match else None


def _normalize_rank(rank: str) -> str:
    value = rank.strip().lower()
    return {
        "ace": "A",
        "king": "K",
        "queen": "Q",
        "jack": "J",
        "ten": "10",
        "nine": "9",
        "eight": "8",
        "seven": "7",
        "six": "6",
        "five": "5",
        "four": "4",
        "three": "3",
        "two": "2",
        "a": "A",
        "k": "K",
        "q": "Q",
        "j": "J",
        "t": "10",
    }.get(value, value.upper())


def _split_action_cards(state: GameState, action: Action) -> tuple[tuple[Card, ...], tuple[Card, ...]]:
    selected = set(action.card_indices)
    if any(index < 0 or index >= len(state.hand) for index in selected):
        raise ValueError(f"Action card indices are outside the hand: {action.card_indices}")
    selected_cards = tuple(card for index, card in enumerate(state.hand) if index in selected)
    held_cards = tuple(card for index, card in enumerate(state.hand) if index not in selected)
    return selected_cards, held_cards


def _played_hand_types_this_round(state: GameState) -> tuple[str, ...]:
    hands = state.modifiers.get("hands", {})
    if not isinstance(hands, dict):
        return ()

    played: list[tuple[int, str]] = []
    for name, value in hands.items():
        if not isinstance(value, dict):
            continue
        try:
            played_count = int(value.get("played_this_round", 0) or 0)
        except (TypeError, ValueError):
            played_count = 0
        if played_count <= 0:
            continue
        try:
            order = int(value.get("order", 0) or 0)
        except (TypeError, ValueError):
            order = 0
        played.extend((order, str(name)) for _ in range(played_count))
    return tuple(name for _, name in sorted(played, key=lambda item: item[0]))


def _played_hand_counts(state: GameState) -> dict[str, int]:
    hands = state.modifiers.get("hands", {})
    if not isinstance(hands, dict):
        return {}

    counts: dict[str, int] = {}
    for name, value in hands.items():
        if isinstance(value, dict):
            counts[str(name)] = _int_value(value.get("played"))
        else:
            counts[str(name)] = _int_value(value)
    return counts


def _with_played_hand_type(modifiers: dict[str, object], hand_type: HandType) -> dict[str, object]:
    updated = dict(modifiers)
    raw_hands = modifiers.get("hands", {})
    hands = dict(raw_hands) if isinstance(raw_hands, dict) else {}
    current = hands.get(hand_type.value, {})
    current_mapping = dict(current) if isinstance(current, dict) else {}
    played_this_round = _int_value(current_mapping.get("played_this_round")) + 1
    played_total = _int_value(current_mapping.get("played")) + 1
    current_mapping["played_this_round"] = played_this_round
    current_mapping["played"] = played_total
    if "order" not in current_mapping:
        current_mapping["order"] = len(_played_names_from_hands(hands)) + 1
    hands[hand_type.value] = current_mapping
    updated["hands"] = hands
    return updated


def _with_played_hand_level(modifiers: dict[str, object], hand_type: HandType, level_gain: int) -> dict[str, object]:
    if level_gain <= 0:
        return modifiers
    updated = dict(modifiers)
    raw_hands = modifiers.get("hands", {})
    hands = dict(raw_hands) if isinstance(raw_hands, dict) else {}
    current = hands.get(hand_type.value, {})
    current_mapping = dict(current) if isinstance(current, dict) else {}
    current_mapping["level"] = max(1, _int_value(current_mapping.get("level"))) + level_gain
    hands[hand_type.value] = current_mapping
    updated["hands"] = hands
    return updated


def _with_played_pile(modifiers: dict[str, object], played_cards: tuple[Card, ...]) -> dict[str, object]:
    updated = dict(modifiers)
    updated["played_pile"] = _zone_cards(modifiers, "played_pile") + played_cards
    return updated


def _with_discard_pile(modifiers: dict[str, object], discarded_cards: tuple[Card, ...]) -> dict[str, object]:
    updated = dict(modifiers)
    updated["discard_pile"] = _zone_cards(modifiers, "discard_pile") + discarded_cards
    return updated


def _with_destroyed_cards(modifiers: dict[str, object], destroyed_cards: tuple[Card, ...]) -> dict[str, object]:
    updated = dict(modifiers)
    updated["destroyed_cards"] = _zone_cards(modifiers, "destroyed_cards") + destroyed_cards
    return updated


def _discarded_cards_after_discard(state: GameState, discarded_cards: tuple[Card, ...]) -> tuple[Card, ...]:
    if _trading_card_removes_discard(state, discarded_cards):
        return ()
    return discarded_cards


def _destroyed_cards_after_discard(state: GameState, discarded_cards: tuple[Card, ...]) -> tuple[Card, ...]:
    if _trading_card_removes_discard(state, discarded_cards):
        return discarded_cards
    return ()


def _trading_card_removes_discard(state: GameState, discarded_cards: tuple[Card, ...]) -> bool:
    return (
        _discard_used_count(state) == 0
        and len(discarded_cards) == 1
        and any(joker.name == "Trading Card" and not _joker_is_disabled(joker) for joker in state.jokers)
    )


def _increment_modifier(modifiers: dict[str, object], key: str) -> dict[str, object]:
    return _increment_modifier_by(modifiers, key, 1)


def _increment_modifier_by(modifiers: dict[str, object], key: str, amount: int) -> dict[str, object]:
    updated = dict(modifiers)
    updated[key] = _int_value(modifiers.get(key)) + amount
    return updated


def _zone_cards(modifiers: dict[str, object], key: str) -> tuple[Card, ...]:
    raw = modifiers.get(key, ())
    if isinstance(raw, Card):
        return (raw,)
    if isinstance(raw, dict):
        raw = raw.get("cards", ())
    if not isinstance(raw, list | tuple):
        return ()

    cards: list[Card] = []
    for item in raw:
        if isinstance(item, Card):
            cards.append(item)
        elif isinstance(item, dict | str):
            cards.append(Card.from_mapping(item))
    return tuple(cards)


def _without_round_card_zones(modifiers: dict[str, object]) -> dict[str, object]:
    updated = dict(modifiers)
    updated.pop("played_pile", None)
    updated.pop("discard_pile", None)
    updated.pop("boss_disabled", None)
    return updated


def _without_shop_surface(modifiers: dict[str, object]) -> dict[str, object]:
    updated = dict(modifiers)
    updated["shop_cards"] = ()
    updated["voucher_cards"] = ()
    updated["booster_packs"] = ()
    updated["pack_cards"] = ()
    return updated


def _with_new_round_counters(modifiers: dict[str, object]) -> dict[str, object]:
    updated = dict(modifiers)
    updated["round_hands_played"] = 0
    updated["hands_played"] = 0
    updated["round_discards_used"] = 0
    updated["discards_used"] = 0
    updated["discards_used_this_round"] = 0
    raw_hands = modifiers.get("hands", {})
    if isinstance(raw_hands, dict):
        hands: dict[object, object] = {}
        for name, value in raw_hands.items():
            if isinstance(value, dict):
                current = dict(value)
                current["played_this_round"] = 0
                hands[name] = current
            else:
                hands[name] = value
        updated["hands"] = hands
    return updated


def _first_hand_of_round(state: GameState) -> bool:
    for key in ("round_hands_played", "hands_played"):
        if key in state.modifiers:
            return _int_value(state.modifiers.get(key)) == 0
    return not _played_hand_types_this_round(state) and state.current_score <= 0


def _has_consumable_room(state: GameState) -> bool:
    return _consumable_open_slots(state) > 0


def _consumable_open_slots(state: GameState) -> int:
    return max(0, _consumable_slot_limit(state) - len(state.consumables))


def _consumable_slot_limit(state: GameState) -> int:
    limit = 2
    for key in ("consumable_slot_limit", "consumeable_slot_limit", "consumable_slots", "consumeable_slots"):
        raw = state.modifiers.get(key)
        try:
            if raw is not None:
                limit = max(0, int(raw))
                break
        except (TypeError, ValueError):
            continue
    return limit


def _normal_joker_open_slots(state: GameState) -> int:
    return max(0, _normal_joker_slot_limit(state) - _normal_joker_slots_used(state))


def _normal_joker_slots_used(state: GameState) -> int:
    return sum(1 for joker in state.jokers if _joker_uses_normal_slot(joker))


def _normal_joker_slot_limit(state: GameState) -> int:
    for key in ("joker_slot_limit", "joker_slots"):
        raw = state.modifiers.get(key)
        try:
            if raw is not None:
                return max(0, int(raw))
        except (TypeError, ValueError):
            continue
    return 5


def _joker_uses_normal_slot(joker: Joker) -> bool:
    return "negative" not in (joker.edition or "").lower()


def _joker_item_uses_normal_slot(item: object) -> bool:
    return _joker_uses_normal_slot(_joker_from_item(item))


def _discard_used_count(state: GameState) -> int:
    for key in ("round_discards_used", "discards_used", "discards_used_this_round"):
        if key in state.modifiers:
            return _int_value(state.modifiers.get(key))
    if _played_hand_types_this_round(state) or state.current_score > 0:
        return 1
    if 0 < state.discards_remaining < 3:
        return 1
    return 0


def _with_discard_used(modifiers: dict[str, object]) -> dict[str, object]:
    updated = dict(modifiers)
    next_count = _int_value(
        modifiers.get(
            "round_discards_used",
            modifiers.get("discards_used", modifiers.get("discards_used_this_round", 0)),
        )
    ) + 1
    updated["round_discards_used"] = next_count
    updated["discards_used"] = next_count
    updated["discards_used_this_round"] = next_count
    return updated


def _played_names_from_hands(hands: dict[object, object]) -> tuple[str, ...]:
    names: list[str] = []
    for name, value in hands.items():
        if isinstance(value, dict) and _int_value(value.get("played_this_round")) > 0:
            names.append(str(name))
    return tuple(names)


def _int_value(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _outcome_int(outcomes: dict[str, object], key: str, default: int = 0) -> int:
    try:
        return int(outcomes.get(key, default))
    except (TypeError, ValueError):
        return default


def _float_value(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
