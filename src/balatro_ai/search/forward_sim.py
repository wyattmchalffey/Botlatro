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
from balatro_ai.rules.joker_compat import is_blueprint_compatible


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
    lucky_card_mult_triggers: int = 0
    lucky_card_money_triggers: int = 0
    lucky_card_triggers: int = 0
    matador_triggered: bool | None = None
    removed_jokers: tuple[str, ...] = ()
    hook_discarded_cards: tuple[Card, ...] = ()
    shattered_glass_cards: tuple[Card, ...] = ()
    crimson_heart_disabled_joker_index: int | None = None


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

ANTE_SMALL_BLIND_SCORES = {
    1: 300,
    2: 800,
    3: 2000,
    4: 5000,
    5: 11000,
    6: 20000,
    7: 35000,
    8: 50000,
}

TAROT_NAMES = frozenset(
    {
        "The Fool",
        "The Magician",
        "The High Priestess",
        "The Empress",
        "The Emperor",
        "The Hierophant",
        "The Lovers",
        "The Chariot",
        "Justice",
        "The Hermit",
        "The Wheel of Fortune",
        "Strength",
        "The Hanged Man",
        "Death",
        "Temperance",
        "The Devil",
        "The Tower",
        "The Star",
        "The Moon",
        "The Sun",
        "Judgement",
        "The World",
    }
)
SPECTRAL_NAMES = frozenset(
    {
        "Familiar",
        "Grim",
        "Incantation",
        "Talisman",
        "Aura",
        "Wraith",
        "Sigil",
        "Ouija",
        "Ectoplasm",
        "Immolate",
        "Ankh",
        "Deja Vu",
        "Hex",
        "Trance",
        "Medium",
        "Cryptid",
        "The Soul",
        "Black Hole",
    }
)
TAROT_ENHANCEMENTS = {
    "The Magician": "LUCKY",
    "The Empress": "MULT",
    "The Hierophant": "BONUS",
    "The Lovers": "WILD",
    "The Chariot": "STEEL",
    "Justice": "GLASS",
    "The Devil": "GOLD",
    "The Tower": "STONE",
}
TAROT_SUIT_CONVERSIONS = {
    "The Star": "D",
    "The Moon": "C",
    "The Sun": "H",
    "The World": "S",
}
SPECTRAL_SEALS = {
    "Talisman": "Gold",
    "Deja Vu": "Red",
    "Trance": "Blue",
    "Medium": "Purple",
}
STRENGTH_RANKS = {
    "2": "3",
    "3": "4",
    "4": "5",
    "5": "6",
    "6": "7",
    "7": "8",
    "8": "9",
    "9": "10",
    "10": "J",
    "T": "J",
    "J": "Q",
    "Q": "K",
    "K": "A",
    "A": "2",
}
RANK_CHIPS = {
    "A": 11,
    "K": 10,
    "Q": 10,
    "J": 10,
    "10": 10,
    "T": 10,
    "9": 9,
    "8": 8,
    "7": 7,
    "6": 6,
    "5": 5,
    "4": 4,
    "3": 3,
    "2": 2,
}
SHOP_EDITION_BY_TAG = {
    "Negative Tag": "NEGATIVE",
    "Foil Tag": "FOIL",
    "Holographic Tag": "HOLOGRAPHIC",
    "Polychrome Tag": "POLYCHROME",
}
EDITION_EXTRA_COST = {"FOIL": 2, "HOLOGRAPHIC": 3, "POLYCHROME": 5, "NEGATIVE": 5}


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
    hook_discarded_cards = _outcome_cards(outcome_map, "hook_discarded_cards")
    held_cards_after_hook = _without_card_instances(held_cards, hook_discarded_cards)
    state_jokers = _jokers_with_dynamic_state_values(state)
    hook_jokers = _jokers_after_discard(state_jokers, hook_discarded_cards) if hook_discarded_cards else state_jokers
    hook_money_delta = _discard_money_delta(state, hook_discarded_cards) if hook_discarded_cards else 0
    evaluation = evaluate_played_cards(
        selected_cards,
        state.hand_levels,
        debuffed_suits=debuffed_suits_for_blind(blind_name),
        blind_name=blind_name,
        jokers=hook_jokers,
        discards_remaining=state.discards_remaining,
        hands_remaining=state.hands_remaining,
        held_cards=held_cards_after_hook,
        deck_size=state.deck_size,
        money=state.money + hook_money_delta,
        played_hand_types_this_round=_played_hand_types_this_round(state),
        played_hand_counts=_played_hand_counts(state),
        stochastic_outcomes=outcome_map,
    )
    money_before_scoring = state.money + hook_money_delta
    if _ox_should_reset_money(state, evaluation.hand_type):
        money_before_scoring = 0
        evaluation = evaluate_played_cards(
            selected_cards,
            state.hand_levels,
            debuffed_suits=debuffed_suits_for_blind(blind_name),
            blind_name=blind_name,
            jokers=hook_jokers,
            discards_remaining=state.discards_remaining,
            hands_remaining=state.hands_remaining,
            held_cards=held_cards_after_hook,
            deck_size=state.deck_size,
            money=money_before_scoring,
            played_hand_types_this_round=_played_hand_types_this_round(state),
            played_hand_counts=_played_hand_counts(state),
            stochastic_outcomes=outcome_map,
        )
    scored_points = int(evaluation.score * _observatory_multiplier(state, evaluation.hand_type))
    next_score = state.current_score + scored_points
    next_hands = max(0, state.hands_remaining - 1)
    next_money = money_before_scoring + evaluation.money_delta
    next_jokers = _jokers_after_play(
        hook_jokers,
        selected_cards,
        evaluation,
        hands_remaining=state.hands_remaining,
        stochastic_outcomes=outcome_map,
    )
    next_hand_levels = _hand_levels_after_play(state.hand_levels, evaluation, stochastic_outcomes=outcome_map)
    mutated_played_cards = _played_cards_after_play(state, selected_cards, evaluation)
    dna_cards = _dna_cards_after_play(state, selected_cards, evaluation)
    shattered_glass_cards = _outcome_cards(outcome_map, "shattered_glass_cards")
    if shattered_glass_cards:
        _validate_shattered_glass_cards(mutated_played_cards, shattered_glass_cards)
    sixth_sense_triggered = _sixth_sense_triggers(state, selected_cards)
    consumable_creations = _play_consumable_creation_requests(
        state,
        selected_cards,
        evaluation,
        sixth_sense_triggered=sixth_sense_triggered,
    )
    created_consumable_labels = _created_labels(created_consumables)
    marked_played_cards = _cards_marked_played_this_ante(mutated_played_cards)
    kept_played_cards = () if sixth_sense_triggered else _without_card_instances(marked_played_cards, shattered_glass_cards)
    destroyed_played_cards = marked_played_cards if sixth_sense_triggered else shattered_glass_cards
    if destroyed_played_cards:
        next_jokers = _jokers_after_playing_cards_destroyed(next_jokers, destroyed_played_cards, glass_shattered=shattered_glass_cards)
    next_modifiers = _with_played_hand_type(state.modifiers, evaluation.hand_type)
    next_modifiers = _with_played_hand_level(next_modifiers, evaluation.hand_type, _outcome_int(outcome_map, "space_joker_triggers"))
    next_modifiers = _with_played_pile(next_modifiers, kept_played_cards)
    if hook_discarded_cards:
        next_modifiers = _with_discard_pile(next_modifiers, hook_discarded_cards)
    if destroyed_played_cards:
        next_modifiers = _with_destroyed_cards(next_modifiers, destroyed_played_cards)
    next_phase = state.phase
    run_over = state.run_over
    if state.required_score > 0 and next_score >= state.required_score:
        next_phase = GamePhase.ROUND_EVAL
        next_money += _held_end_of_round_money_delta(held_cards_after_hook, next_jokers)
    elif next_hands <= 0:
        if _mr_bones_saves(state, next_score):
            next_phase = GamePhase.ROUND_EVAL
            next_money += _held_end_of_round_money_delta(held_cards_after_hook, next_jokers)
            next_jokers = _jokers_after_mr_bones_save(next_jokers)
        else:
            next_phase = GamePhase.RUN_OVER
            run_over = True
    if next_phase == GamePhase.ROUND_EVAL:
        consumable_creations += _blue_seal_creation_requests(
            state,
            held_cards_after_hook,
            reserved_slots=_creation_request_total(consumable_creations),
        )
    _validate_injected_count_at_most(
        label="play-created consumables",
        expected=_creation_request_total(consumable_creations),
        injected=len(created_consumable_labels),
    )
    next_modifiers = _with_creation_counts(next_modifiers, consumable_creations, created=len(created_consumable_labels))
    draw = () if next_phase == GamePhase.ROUND_EVAL else tuple(drawn_cards)
    draw_result = _draw_from_deck(state, draw)
    next_hand = _sort_hand_cards(held_cards_after_hook + dna_cards + draw)
    next_known_deck = draw_result.known_deck
    next_deck_size = draw_result.deck_size
    if next_phase == GamePhase.ROUND_EVAL:
        next_deck_size, next_known_deck = _round_eval_deck_after_play(
            state,
            draw_result,
            returning_hand_cards=held_cards_after_hook + dna_cards + kept_played_cards + hook_discarded_cards + draw,
        )
        next_hand = ()
        next_modifiers = _without_round_card_zones(next_modifiers)
    if next_phase in {GamePhase.ROUND_EVAL, GamePhase.RUN_OVER}:
        next_jokers = (
            _jokers_after_played_round_ends(next_jokers)
            if next_phase == GamePhase.ROUND_EVAL
            else _jokers_after_egg_end_of_round(next_jokers)
        )
        next_jokers = _jokers_after_stochastic_removals(next_jokers, _outcome_names(outcome_map, "removed_jokers"))
        if next_phase == GamePhase.ROUND_EVAL:
            gift_card_gain = _gift_card_sell_value_gain(next_jokers)
            next_jokers = _jokers_after_gift_card_round_end(next_jokers, gift_card_gain=gift_card_gain)
            next_modifiers = _modifiers_after_gift_card_round_end(next_modifiers, state, gift_card_gain=gift_card_gain)
            if _blind_is_boss(state):
                next_known_deck = _cards_without_played_this_ante(next_known_deck)
    else:
        next_jokers = _jokers_after_crimson_heart_draw(state, next_jokers, outcome_map, next_phase=next_phase)
    next_state = replace(
        state,
        phase=next_phase,
        current_score=next_score,
        money=next_money,
        hands_remaining=next_hands,
        deck_size=next_deck_size,
        hand_levels=next_hand_levels,
        hand=next_hand,
        known_deck=next_known_deck,
        jokers=next_jokers,
        consumables=state.consumables + created_consumable_labels,
        modifiers=next_modifiers,
        legal_actions=(),
        run_over=run_over,
    )
    return _state_with_dynamic_joker_values(next_state)


def simulate_discard(state: GameState, action: Action, drawn_cards: Iterable[Card] = ()) -> GameState:
    """Apply a deterministic discard transition."""

    if action.action_type != ActionType.DISCARD:
        raise ValueError(f"simulate_discard requires discard, got {action.action_type.value}")
    selected_cards, held_cards = _split_action_cards(state, action)
    if not selected_cards:
        raise ValueError("Cannot simulate discarding zero cards")

    draw = tuple(drawn_cards)
    draw_result = _draw_from_deck(state, draw)
    next_jokers = _jokers_after_discard(_jokers_with_dynamic_state_values(state), selected_cards)
    next_hand_levels = _hand_levels_after_discard(state, selected_cards)
    next_money = state.money + _discard_money_delta(state, selected_cards)
    discarded_pile_cards = _discarded_cards_after_discard(state, selected_cards)
    destroyed_cards = _destroyed_cards_after_discard(state, selected_cards)
    if destroyed_cards:
        next_jokers = _jokers_after_playing_cards_destroyed(next_jokers, destroyed_cards, glass_shattered=())
    next_modifiers = _with_discard_used(state.modifiers)
    next_modifiers = _with_discard_pile(next_modifiers, discarded_pile_cards)
    if destroyed_cards:
        next_modifiers = _with_destroyed_cards(next_modifiers, destroyed_cards)
    next_state = replace(
        state,
        discards_remaining=max(0, state.discards_remaining - 1),
        money=next_money,
        deck_size=draw_result.deck_size,
        hand=_sort_hand_cards(held_cards + draw),
        known_deck=draw_result.known_deck,
        jokers=next_jokers,
        hand_levels=next_hand_levels,
        modifiers=next_modifiers,
        legal_actions=(),
    )
    return _state_with_dynamic_joker_values(next_state)


def simulate_buy(
    state: GameState,
    action: Action,
    *,
    next_shop_cards: Iterable[object] | None = None,
    next_voucher_cards: Iterable[object] | None = None,
    next_booster_packs: Iterable[object] | None = None,
) -> GameState:
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
    edition_tag_name, edition = _active_shop_edition_tag(state, item)
    edition_target_index = _first_shop_edition_tag_target_index(items) if edition is not None else None
    if edition is not None and index == edition_target_index:
        item = _shop_item_with_tag_edition(item, edition)
    cost = _effective_item_buy_cost(state, item)
    if cost > state.money:
        raise ValueError(f"Cannot buy {_item_label(item)} for ${cost} with ${state.money}")
    if _item_is_consumable(item) and not _has_consumable_room(state):
        raise ValueError(
            f"Cannot buy {_item_label(item)} with full consumable slots: "
            f"{len(state.consumables)}/{_consumable_slot_limit(state)}"
        )
    if _item_is_joker(item) and _joker_item_uses_normal_slot(item) and _normal_joker_open_slots(state) <= 0:
        raise ValueError(
            f"Cannot buy {_item_label(item)} with full joker slots: "
            f"{_normal_joker_slots_used(state)}/{_normal_joker_slot_limit(state)}"
        )

    remaining_items = _without_index(items, index)
    if edition is not None and edition_target_index is not None and index != edition_target_index:
        adjusted_index = edition_target_index - (1 if index < edition_target_index else 0)
        remaining_items = _with_shop_edition_at_index(remaining_items, adjusted_index, edition)
    next_modifiers = _with_modifier_items(state.modifiers, item_key, remaining_items)
    if edition_tag_name is not None and index == edition_target_index:
        next_modifiers = _modifiers_after_shop_edition_tag_used(next_modifiers, edition_tag_name)
    acquired = _state_after_acquiring_item(replace(state, modifiers=next_modifiers), item, force_voucher=kind == "voucher")
    if kind == "voucher":
        acquired = _state_after_voucher_purchase_effects(acquired, item)
    final_modifiers = acquired.modifiers
    if next_shop_cards is not None:
        final_modifiers = _with_modifier_items(final_modifiers, "shop_cards", tuple(next_shop_cards))
    if next_voucher_cards is not None:
        final_modifiers = _with_modifier_items(final_modifiers, "voucher_cards", tuple(next_voucher_cards))
    if next_booster_packs is not None:
        final_modifiers = _with_modifier_items(final_modifiers, "booster_packs", tuple(next_booster_packs))
    next_state = replace(
        acquired,
        money=state.money - cost,
        modifiers=final_modifiers,
        shop=tuple(_item_label(item) for item in _modifier_items(final_modifiers, "shop_cards")),
        legal_actions=(),
    )
    return _state_with_dynamic_joker_values(next_state)


def simulate_sell(state: GameState, action: Action, *, created_joker: object | None = None) -> GameState:
    """Apply a deterministic sell transition for jokers or consumables."""

    if action.action_type != ActionType.SELL:
        raise ValueError(f"simulate_sell requires sell, got {action.action_type.value}")
    kind = _action_kind(action, default="joker")
    if kind == "consumable":
        index = _action_index(action)
        if index is None or not 0 <= index < len(state.consumables):
            raise ValueError(f"Sell action index is outside consumables: {action.amount}")
        sold = state.consumables[index]
        remaining = tuple(name for consumable_index, name in enumerate(state.consumables) if consumable_index != index)
        next_state = replace(
            state,
            money=state.money + _consumable_sell_value(state, sold),
            consumables=remaining,
            modifiers=_modifiers_after_consumable_removed(state.modifiers, previous_count=len(state.consumables)),
            legal_actions=(),
        )
        return _state_with_dynamic_joker_values(next_state)
    if kind != "joker":
        raise ValueError(f"Unsupported sell kind: {kind}")
    index = _action_index(action)
    if index is None or not 0 <= index < len(state.jokers):
        raise ValueError(f"Sell action index is outside jokers: {action.amount}")
    sold = state.jokers[index]
    remaining = tuple(joker for joker_index, joker in enumerate(state.jokers) if joker_index != index)
    if created_joker is not None:
        if sold.name != "Invisible Joker":
            raise ValueError(f"Only Invisible Joker can inject a created joker on sell, got {sold.name}")
        if not _invisible_joker_ready(sold):
            raise ValueError("Invisible Joker is not ready to create a duplicate")
        remaining = remaining + (_joker_from_item(created_joker),)
    next_modifiers = _modifiers_after_joker_removed(state.modifiers, sold)
    next_modifiers = _modifiers_after_sell_trigger(state, next_modifiers, sold)
    next_state = replace(
        state,
        money=state.money + (sold.sell_value or 0),
        jokers=_jokers_after_sell(remaining),
        modifiers=next_modifiers,
        legal_actions=(),
    )
    next_state = _state_with_current_discard_delta_change(state, next_state)
    if _effective_blind_name(state) == "Verdant Leaf":
        next_state = _state_after_verdant_leaf_disabled(next_state)
    return _state_with_dynamic_joker_values(next_state)


def simulate_use_consumable(
    state: GameState,
    action: Action,
    *,
    created_consumables: Iterable[object] = (),
    created_jokers: Iterable[object] = (),
    created_hand_cards: Iterable[Card] = (),
    destroyed_hand_card_indices: Iterable[int] = (),
    aura_edition: str | None = None,
    wheel_of_fortune_joker_index: int | None = None,
    wheel_of_fortune_edition: str | None = None,
    sigil_suit: str | None = None,
    ouija_rank: str | None = None,
    spectral_joker_index: int | None = None,
) -> GameState:
    """Use a stored consumable without inventing stochastic outcomes."""

    if action.action_type != ActionType.USE_CONSUMABLE:
        raise ValueError(f"simulate_use_consumable requires use_consumable, got {action.action_type.value}")
    kind = _action_kind(action, default="consumable")
    if kind != "consumable":
        raise ValueError(f"Unsupported consumable-use kind: {kind}")
    index = _action_index(action)
    if index is None or not 0 <= index < len(state.consumables):
        raise ValueError(f"Consumable action index is outside consumables: {action.amount}")
    name = state.consumables[index]
    remaining = tuple(item for item_index, item in enumerate(state.consumables) if item_index != index)
    used_state = replace(state, consumables=remaining)
    next_state = _state_after_consumable_used(
        used_state,
        name,
        card_indices=action.card_indices,
        created_consumables=created_consumables,
        created_jokers=created_jokers,
        created_hand_cards=created_hand_cards,
        destroyed_hand_card_indices=destroyed_hand_card_indices,
        aura_edition=aura_edition,
        wheel_of_fortune_joker_index=wheel_of_fortune_joker_index,
        wheel_of_fortune_edition=wheel_of_fortune_edition,
        sigil_suit=sigil_suit,
        ouija_rank=ouija_rank,
        spectral_joker_index=spectral_joker_index,
    )
    return _state_with_dynamic_joker_values(replace(next_state, legal_actions=()))


def simulate_reroll(
    state: GameState,
    action: Action,
    new_shop_cards: Iterable[object],
    *,
    new_voucher_cards: Iterable[object] | None = None,
    new_booster_packs: Iterable[object] | None = None,
) -> GameState:
    """Apply a deterministic reroll transition using caller-injected shop cards."""

    if action.action_type != ActionType.REROLL:
        raise ValueError(f"simulate_reroll requires reroll, got {action.action_type.value}")
    cost, next_modifiers = _spend_reroll(state)
    if cost > state.money:
        raise ValueError(f"Cannot reroll for ${cost} with ${state.money}")
    shop_cards = tuple(new_shop_cards)
    next_modifiers = _with_modifier_items(next_modifiers, "shop_cards", shop_cards)
    if new_voucher_cards is not None:
        next_modifiers = _with_modifier_items(next_modifiers, "voucher_cards", tuple(new_voucher_cards))
    if new_booster_packs is not None:
        next_modifiers = _with_modifier_items(next_modifiers, "booster_packs", tuple(new_booster_packs))
    next_state = replace(
        state,
        money=state.money - cost,
        shop=tuple(_item_label(item) for item in shop_cards),
        jokers=_jokers_after_reroll(state.jokers),
        modifiers=next_modifiers,
        legal_actions=(),
    )
    return _state_with_dynamic_joker_values(next_state)


def simulate_open_pack(
    state: GameState,
    action: Action,
    pack_contents: Iterable[object],
    *,
    created_consumables: Iterable[object] = (),
    drawn_cards: Iterable[Card] = (),
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
    draw = tuple(drawn_cards)
    draw_result = _draw_from_deck(state, draw)
    _validate_injected_count_at_most(
        label="Hallucination",
        expected=_hallucination_creation_count(state),
        injected=len(hallucination_cards),
    )
    next_modifiers = _with_modifier_items(state.modifiers, "booster_packs", _without_index(packs, index))
    next_modifiers = _with_modifier_items(next_modifiers, "pack_cards", contents)
    next_modifiers = _with_created_tarot_count(next_modifiers, len(hallucination_cards))
    next_modifiers = _increment_modifier_by(next_modifiers, "created_consumable_count", len(hallucination_cards))
    next_state = replace(
        state,
        phase=GamePhase.BOOSTER_OPENED,
        money=state.money - cost,
        deck_size=draw_result.deck_size,
        hand=_sort_hand_cards(draw),
        known_deck=draw_result.known_deck,
        pack=tuple(_item_label(item) for item in contents),
        consumables=state.consumables + hallucination_cards,
        modifiers=next_modifiers,
        legal_actions=(),
    )
    return _state_with_dynamic_joker_values(next_state)


def simulate_choose_pack_card(
    state: GameState,
    action: Action,
    *,
    remaining_pack_cards: Iterable[object] | None = None,
    return_hand_to_deck: bool = True,
    created_consumables: Iterable[object] = (),
    created_jokers: Iterable[object] = (),
    created_hand_cards: Iterable[Card] = (),
    destroyed_hand_card_indices: Iterable[int] = (),
    aura_edition: str | None = None,
    wheel_of_fortune_joker_index: int | None = None,
    wheel_of_fortune_edition: str | None = None,
    sigil_suit: str | None = None,
    ouija_rank: str | None = None,
    spectral_joker_index: int | None = None,
) -> GameState:
    """Choose or skip a card from an opened booster."""

    if action.action_type != ActionType.CHOOSE_PACK_CARD:
        raise ValueError(f"simulate_choose_pack_card requires choose_pack_card, got {action.action_type.value}")
    if action.target_id == "skip" or _action_kind(action, default="card") == "skip":
        returned_deck_size, returned_known_deck = (
            _deck_after_returning_hand(state) if return_hand_to_deck else (state.deck_size, state.known_deck)
        )
        next_state = replace(
            state,
            phase=GamePhase.SHOP,
            hand=() if return_hand_to_deck else state.hand,
            deck_size=returned_deck_size,
            known_deck=returned_known_deck,
            jokers=_jokers_after_pack_skip(state.jokers),
            pack=(),
            modifiers=_with_modifier_items(state.modifiers, "pack_cards", ()),
            legal_actions=(),
        )
        return _state_with_dynamic_joker_values(next_state)

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
    base_state = replace(
        state,
        phase=next_phase,
        pack=tuple(_item_label(item) for item in next_pack_cards),
        modifiers=next_modifiers,
    )
    if not _item_is_consumable(item) and next_phase == GamePhase.SHOP and return_hand_to_deck:
        next_deck_size, next_known_deck = _deck_after_returning_hand(state)
        base_state = replace(base_state, hand=(), deck_size=next_deck_size, known_deck=next_known_deck)
    acquired = _state_after_pack_choice(
        base_state,
        item,
        card_indices=action.card_indices,
        created_consumables=created_consumables,
        created_jokers=created_jokers,
        created_hand_cards=created_hand_cards,
        destroyed_hand_card_indices=destroyed_hand_card_indices,
        aura_edition=aura_edition,
        wheel_of_fortune_joker_index=wheel_of_fortune_joker_index,
        wheel_of_fortune_edition=wheel_of_fortune_edition,
        sigil_suit=sigil_suit,
        ouija_rank=ouija_rank,
        spectral_joker_index=spectral_joker_index,
    )
    if _item_is_consumable(item) and next_phase == GamePhase.SHOP and return_hand_to_deck:
        returned_deck_size, returned_known_deck = _deck_after_returning_hand(acquired)
        acquired = replace(acquired, hand=(), deck_size=returned_deck_size, known_deck=returned_known_deck)
    return _state_with_dynamic_joker_values(replace(acquired, legal_actions=()))


def simulate_end_shop(state: GameState, *, created_consumables: Iterable[object] = ()) -> GameState:
    """Leave the shop and move to blind selection."""

    created_consumable_labels = _created_labels(created_consumables)
    _validate_injected_count_at_most(
        label="Perkeo",
        expected=_perkeo_creation_count(state),
        injected=len(created_consumable_labels),
    )
    next_modifiers = _without_shop_surface(state.modifiers)
    next_hands, next_discards, next_modifiers = _round_counts_after_shop_exit(state, next_modifiers)
    if created_consumable_labels:
        next_modifiers = _increment_modifier_by(
            next_modifiers,
            "created_negative_consumable_count",
            len(created_consumable_labels),
        )
    next_state = replace(
        state,
        phase=GamePhase.BLIND_SELECT,
        shop=(),
        pack=(),
        consumables=state.consumables + created_consumable_labels,
        hands_remaining=next_hands,
        discards_remaining=next_discards,
        modifiers=next_modifiers,
        legal_actions=(),
    )
    return _state_with_dynamic_joker_values(next_state)


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
    next_jokers = _jokers_after_cash_out(state, next_to_do_targets=tuple(next_to_do_targets))
    next_jokers = _jokers_after_stochastic_removals(next_jokers, tuple(removed_jokers))
    next_modifiers = _without_round_card_zones(state.modifiers)
    next_modifiers = _with_modifier_items(next_modifiers, "shop_cards", shop_cards)
    next_modifiers = _with_modifier_items(next_modifiers, "voucher_cards", voucher_cards)
    next_modifiers = _with_modifier_items(next_modifiers, "booster_packs", booster_packs)
    next_modifiers = _with_modifier_items(next_modifiers, "pack_cards", ())
    next_modifiers = _modifiers_after_cash_out(state.modifiers, next_modifiers, state.jokers, next_jokers)
    next_modifiers = _with_applied_round_count_deltas(next_modifiers)
    next_state = replace(
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
    return _state_with_dynamic_joker_values(next_state)


def simulate_select_blind(
    state: GameState,
    drawn_cards: Iterable[Card] = (),
    *,
    created_hand_cards: Iterable[Card] = (),
    created_deck_cards: Iterable[Card] = (),
    created_consumables: Iterable[object] = (),
    created_jokers: Iterable[object] = (),
    next_ancient_suit: str | None = None,
    next_idol_card: Card | None = None,
    next_castle_suit: str | None = None,
    next_mail_rank: str | None = None,
    ceremonial_destroyed_joker_index: int | None = None,
    madness_destroyed_joker_index: int | None = None,
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
    next_modifiers.pop("round_count_deltas_applied", None)
    next_hands, next_discards, next_modifiers = _round_counts_after_blind_start(state, next_modifiers)
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
    for removed_joker in _blind_select_removed_jokers(
        state.jokers,
        ceremonial_destroyed_joker_index=ceremonial_destroyed_joker_index,
        madness_destroyed_joker_index=madness_destroyed_joker_index,
        madness_can_destroy=not _blind_is_boss(state),
    ):
        before_discards_delta = _int_value(next_modifiers.get("round_discards_delta"))
        next_modifiers = _modifiers_after_joker_removed(next_modifiers, removed_joker)
        after_discards_delta = _int_value(next_modifiers.get("round_discards_delta"))
        discard_delta_change = after_discards_delta - before_discards_delta
        if discard_delta_change:
            next_discards = max(0, next_discards + discard_delta_change)
            next_modifiers["applied_round_discards_delta"] = after_discards_delta
    next_jokers = state.jokers
    next_jokers = _jokers_after_blind_select(
        state,
        next_jokers,
        next_ancient_suit=next_ancient_suit,
        next_idol_card=next_idol_card,
        next_castle_suit=next_castle_suit,
        next_mail_rank=next_mail_rank,
        ceremonial_destroyed_joker_index=ceremonial_destroyed_joker_index,
        madness_destroyed_joker_index=madness_destroyed_joker_index,
    )
    next_jokers = next_jokers + riff_raff_jokers
    next_state = replace(
        state,
        phase=GamePhase.SELECTING_HAND,
        current_score=0,
        hands_remaining=next_hands,
        discards_remaining=next_discards,
        hand=_sort_hand_cards(draw + certificate_cards),
        known_deck=known_deck,
        deck_size=draw_result.deck_size + len(marble_cards),
        jokers=next_jokers,
        consumables=state.consumables + cartomancer_cards,
        modifiers=next_modifiers,
        legal_actions=(),
        run_over=False,
    )
    return _state_with_dynamic_joker_values(next_state)


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


def _jokers_after_blind_select(
    state: GameState,
    jokers: tuple[Joker, ...],
    *,
    next_ancient_suit: str | None,
    next_idol_card: Card | None,
    next_castle_suit: str | None,
    next_mail_rank: str | None,
    ceremonial_destroyed_joker_index: int | None,
    madness_destroyed_joker_index: int | None,
) -> tuple[Joker, ...]:
    updated = _jokers_with_round_targets(
        jokers,
        next_ancient_suit=next_ancient_suit,
        next_idol_card=next_idol_card,
        next_castle_suit=next_castle_suit,
        next_mail_rank=next_mail_rank,
    )
    updated = _jokers_after_ceremonial_dagger_blind_select(updated, ceremonial_destroyed_joker_index)
    if not _blind_is_boss(state):
        updated = _jokers_after_madness_blind_select(updated, madness_destroyed_joker_index)
    elif madness_destroyed_joker_index is not None:
        raise ValueError("Madness does not destroy jokers on boss blinds")
    return updated


def _blind_select_removed_jokers(
    jokers: tuple[Joker, ...],
    *,
    ceremonial_destroyed_joker_index: int | None,
    madness_destroyed_joker_index: int | None,
    madness_can_destroy: bool,
) -> tuple[Joker, ...]:
    removed: list[Joker] = []
    remaining = jokers
    if ceremonial_destroyed_joker_index is not None and 0 <= ceremonial_destroyed_joker_index < len(jokers):
        removed.append(jokers[ceremonial_destroyed_joker_index])
        remaining = tuple(joker for index, joker in enumerate(jokers) if index != ceremonial_destroyed_joker_index)
    if madness_can_destroy and madness_destroyed_joker_index is not None and 0 <= madness_destroyed_joker_index < len(remaining):
        removed.append(remaining[madness_destroyed_joker_index])
    return tuple(removed)


def _jokers_with_round_targets(
    jokers: tuple[Joker, ...],
    *,
    next_ancient_suit: str | None,
    next_idol_card: Card | None,
    next_castle_suit: str | None,
    next_mail_rank: str | None,
) -> tuple[Joker, ...]:
    updated: list[Joker] = []
    for joker in jokers:
        if joker.name == "Ancient Joker" and next_ancient_suit:
            suit = _normalize_suit(next_ancient_suit)
            updated.append(_with_joker_metadata(joker, {"current_suit": suit, "target_suit": suit}))
        elif joker.name == "The Idol" and next_idol_card is not None:
            rank = _normalize_rank(next_idol_card.rank)
            suit = _normalize_suit(next_idol_card.suit)
            updated.append(
                _with_joker_metadata(
                    joker,
                    {
                        "current_rank": rank,
                        "target_rank": rank,
                        "current_suit": suit,
                        "target_suit": suit,
                    },
                )
            )
        elif joker.name == "Castle" and next_castle_suit:
            suit = _normalize_suit(next_castle_suit)
            updated.append(_with_joker_metadata(joker, {"current_suit": suit, "target_suit": suit}))
        elif joker.name == "Mail-In Rebate" and next_mail_rank:
            rank = _normalize_rank(next_mail_rank)
            updated.append(_with_joker_metadata(joker, {"current_rank": rank, "target_rank": rank}))
        else:
            updated.append(joker)
    return tuple(updated)


def _jokers_after_ceremonial_dagger_blind_select(
    jokers: tuple[Joker, ...],
    destroyed_index: int | None,
) -> tuple[Joker, ...]:
    if destroyed_index is None:
        return jokers
    dagger_index = destroyed_index - 1
    if dagger_index < 0 or dagger_index >= len(jokers):
        raise ValueError(f"Ceremonial Dagger destroyed index has no dagger to its left: {destroyed_index}")
    dagger = jokers[dagger_index]
    destroyed = jokers[destroyed_index] if 0 <= destroyed_index < len(jokers) else None
    if dagger.name != "Ceremonial Dagger":
        raise ValueError("Ceremonial Dagger can only destroy the joker immediately to its right")
    if destroyed is None:
        raise ValueError(f"Ceremonial Dagger destroyed index is outside jokers: {destroyed_index}")
    current = _joker_int(dagger, keys=("current_mult", "mult"), default=0)
    gained = 2 * (destroyed.sell_value or 0)
    updated_dagger = _with_joker_metadata(dagger, {"current_mult": current + gained, "mult": current + gained})
    return tuple(
        updated_dagger if index == dagger_index else joker
        for index, joker in enumerate(jokers)
        if index != destroyed_index
    )


def _jokers_after_madness_blind_select(
    jokers: tuple[Joker, ...],
    destroyed_index: int | None,
) -> tuple[Joker, ...]:
    madness_indexes = [index for index, joker in enumerate(jokers) if joker.name == "Madness" and not _joker_is_disabled(joker)]
    if not madness_indexes:
        if destroyed_index is not None:
            raise ValueError("Madness destroyed joker was injected but no active Madness exists")
        return jokers

    updated: list[Joker] = []
    for index, joker in enumerate(jokers):
        if index == destroyed_index:
            continue
        if joker.name == "Madness" and not _joker_is_disabled(joker):
            current = _joker_float(joker, keys=("current_xmult", "xmult", "current_x_mult", "x_mult"), default=1.0)
            gain = _joker_float(joker, keys=("extra", "xmult_gain"), default=0.5)
            updated.append(_with_joker_metadata(joker, {"current_xmult": current + gain, "xmult": current + gain}))
        else:
            updated.append(joker)
    return tuple(updated)


def _round_eval_deck_after_play(
    state: GameState,
    draw_result: _DrawResult,
    *,
    returning_hand_cards: tuple[Card, ...],
) -> tuple[int, tuple[Card, ...]]:
    returning_cards = (
        returning_hand_cards
        + _zone_cards(state.modifiers, "played_pile")
        + _zone_cards(state.modifiers, "discard_pile")
    )
    known_deck = draw_result.known_deck + returning_cards if draw_result.known_deck else ()
    remaining_deck_count = max(draw_result.deck_size, len(draw_result.known_deck)) if draw_result.known_deck else draw_result.deck_size
    return remaining_deck_count + len(returning_cards), known_deck


def _deck_after_returning_hand(state: GameState) -> tuple[int, tuple[Card, ...]]:
    if not state.hand:
        return state.deck_size, state.known_deck
    known_deck = state.known_deck + state.hand if state.known_deck else ()
    return state.deck_size + len(state.hand), known_deck


def _sort_hand_cards(cards: tuple[Card, ...]) -> tuple[Card, ...]:
    rank_order = {rank: index for index, rank in enumerate(("A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"))}
    suit_order = {"S": 0, "H": 1, "C": 2, "D": 3}
    return tuple(
        sorted(
            cards,
            key=lambda card: (
                rank_order.get(card.rank, len(rank_order)),
                suit_order.get(card.suit, len(suit_order)),
                card.enhancement or "",
                card.seal or "",
                card.edition or "",
                card.debuffed,
            ),
        )
    )


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
            "lucky_card_mult_triggers": outcomes.lucky_card_mult_triggers,
            "lucky_card_money_triggers": outcomes.lucky_card_money_triggers,
            "lucky_card_triggers": outcomes.lucky_card_triggers,
            "matador_triggered": outcomes.matador_triggered,
            "removed_jokers": outcomes.removed_jokers,
            "hook_discarded_cards": outcomes.hook_discarded_cards,
            "shattered_glass_cards": outcomes.shattered_glass_cards,
            "crimson_heart_disabled_joker_index": outcomes.crimson_heart_disabled_joker_index,
        }
    return dict(outcomes)


def _without_card_instances(cards: tuple[Card, ...], removed_cards: tuple[Card, ...]) -> tuple[Card, ...]:
    if not removed_cards:
        return cards
    remaining = list(cards)
    for removed in removed_cards:
        index = _find_matching_card_index(remaining, removed)
        if index is None:
            raise ValueError(f"Cannot remove {removed.short_name}; it is not in held cards")
        del remaining[index]
    return tuple(remaining)


def _validate_shattered_glass_cards(played_cards: tuple[Card, ...], shattered_cards: tuple[Card, ...]) -> None:
    for card in shattered_cards:
        if _normalized(card.enhancement) not in {"glass", "glass card"}:
            raise ValueError(f"Only Glass Cards can shatter, got {card.short_name} {card.enhancement!r}")
    _without_card_instances(played_cards, shattered_cards)


def _find_matching_card_index(cards: list[Card], target: Card) -> int | None:
    target_key = _card_identity_key(target)
    for index, card in enumerate(cards):
        if _card_identity_key(card) == target_key:
            return index
    return None


def _card_identity_key(card: Card) -> tuple[str, str, str | None, str | None, str | None]:
    return (_normalize_rank(card.rank), _normalize_suit(card.suit), card.enhancement, card.seal, card.edition)


def _cards_marked_played_this_ante(cards: tuple[Card, ...]) -> tuple[Card, ...]:
    return tuple(_card_with_metadata_flag(card, "played_this_ante", True) for card in cards)


def _cards_without_played_this_ante(cards: tuple[Card, ...]) -> tuple[Card, ...]:
    return tuple(_card_without_metadata_flag(card, "played_this_ante") for card in cards)


def _card_with_metadata_flag(card: Card, key: str, value: object) -> Card:
    metadata = dict(card.metadata)
    metadata[key] = value
    ability = dict(metadata.get("ability")) if isinstance(metadata.get("ability"), dict) else {}
    ability[key] = value
    metadata["ability"] = ability
    return replace(card, metadata=metadata)


def _card_without_metadata_flag(card: Card, key: str) -> Card:
    if key not in card.metadata and not (isinstance(card.metadata.get("ability"), dict) and key in card.metadata["ability"]):
        return card
    metadata = dict(card.metadata)
    metadata.pop(key, None)
    ability = dict(metadata.get("ability")) if isinstance(metadata.get("ability"), dict) else {}
    ability.pop(key, None)
    if ability:
        metadata["ability"] = ability
    else:
        metadata.pop("ability", None)
    return replace(card, metadata=metadata)


def _card_without_blind_debuff(card: Card, *, force: bool = False) -> Card:
    metadata = dict(card.metadata)
    if not force and not metadata.get("debuffed_by_blind"):
        return card
    metadata.pop("debuffed_by_blind", None)
    state = dict(metadata.get("state")) if isinstance(metadata.get("state"), dict) else {}
    state.pop("debuff", None)
    if state:
        metadata["state"] = state
    else:
        metadata.pop("state", None)
    return replace(card, debuffed=False, metadata=metadata)


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
        acquired = replace(
            state,
            jokers=state.jokers + (joker,),
            modifiers=_modifiers_after_joker_added(state.modifiers, joker),
        )
        return _state_with_current_discard_delta_change(state, acquired)
    if _item_is_consumable(item):
        return replace(state, consumables=state.consumables + (_item_label(item),))

    playing_card = _playing_card_from_item(item)
    if playing_card is not None:
        known_deck = state.known_deck + (playing_card,) if state.known_deck else ()
        return _state_with_dynamic_joker_values(
            replace(
                state,
                deck_size=state.deck_size + 1,
                known_deck=known_deck,
                jokers=_jokers_after_playing_cards_added(state.jokers, (playing_card,)),
            )
        )
    return replace(state, consumables=state.consumables + (_item_label(item),))


def _state_after_voucher_purchase_effects(state: GameState, voucher: object) -> GameState:
    label = _item_label(voucher)
    updated_state = state
    hand_delta = _voucher_hand_delta(label)
    discard_delta = _voucher_discard_delta(label)
    shop_slot_delta = _voucher_shop_slot_delta(label)
    if hand_delta or discard_delta or shop_slot_delta:
        modifiers = dict(updated_state.modifiers)
        if hand_delta:
            modifiers["round_hands_delta"] = _int_value(modifiers.get("round_hands_delta")) + hand_delta
            modifiers["applied_round_hands_delta"] = _int_value(modifiers.get("applied_round_hands_delta")) + hand_delta
        if discard_delta:
            modifiers["round_discards_delta"] = _int_value(modifiers.get("round_discards_delta")) + discard_delta
            modifiers["applied_round_discards_delta"] = _int_value(modifiers.get("applied_round_discards_delta")) + discard_delta
        if shop_slot_delta:
            for key in ("shop_joker_max", "shop_slots", "shop_card_slots"):
                if key in modifiers:
                    modifiers[key] = max(0, _int_value(modifiers.get(key)) + shop_slot_delta)
                    break
        updated_state = replace(
            updated_state,
            hands_remaining=max(0, updated_state.hands_remaining + hand_delta),
            discards_remaining=max(0, updated_state.discards_remaining + discard_delta),
            modifiers=modifiers,
        )

    updated_state = _state_after_voucher_static_effects(updated_state, label)

    discount_percent = _voucher_discount_percent(label)
    if discount_percent <= 0:
        return updated_state
    modifiers = dict(updated_state.modifiers)
    modifiers["discount_percent"] = discount_percent
    for key in ("shop_cards", "voucher_cards", "booster_packs", "pack_cards"):
        items = _modifier_items(modifiers, key)
        if items:
            modifiers = _with_modifier_items(modifiers, key, tuple(_with_discounted_item_cost(item, discount_percent) for item in items))
    return replace(
        updated_state,
        jokers=tuple(_with_discounted_joker_sell_value(joker, discount_percent) for joker in updated_state.jokers),
        modifiers=modifiers,
    )


def _state_after_voucher_static_effects(state: GameState, label: str) -> GameState:
    modifiers = dict(state.modifiers)
    updated_state = state

    if label in {"Tarot Merchant", "Tarot Tycoon"}:
        modifiers["tarot_rate"] = 32.0 if label == "Tarot Tycoon" else 9.6
    elif label in {"Planet Merchant", "Planet Tycoon"}:
        modifiers["planet_rate"] = 32.0 if label == "Planet Tycoon" else 9.6
    elif label in {"Hone", "Glow Up"}:
        modifiers["edition_rate"] = 4.0 if label == "Glow Up" else 2.0
    elif label in {"Magic Trick", "Illusion"}:
        modifiers["playing_card_rate"] = 4.0

    if label == "Crystal Ball":
        modifiers["consumable_slot_limit"] = _int_value(modifiers.get("consumable_slot_limit", 2)) + 1
    elif label == "Antimatter":
        modifiers["joker_slot_limit"] = _int_value(modifiers.get("joker_slot_limit", 5)) + 1
    elif label in {"Paint Brush", "Palette"}:
        if "hand_size" in modifiers:
            modifiers["hand_size"] = _int_value(modifiers.get("hand_size")) + 1
        else:
            modifiers["hand_size_delta"] = _int_value(modifiers.get("hand_size_delta")) + 1
    elif label in {"Reroll Surplus", "Reroll Glut"}:
        for key in ("round_reset_reroll_cost", "base_reroll_cost"):
            if key in modifiers:
                modifiers[key] = _int_value(modifiers.get(key)) - 2
        for key in ("reroll_cost", "current_reroll_cost"):
            if key in modifiers:
                modifiers[key] = max(0, _int_value(modifiers.get(key)) - 2)
        if "round_reset_reroll_cost" not in modifiers and "base_reroll_cost" not in modifiers:
            modifiers["round_reset_reroll_cost"] = 3
        if "reroll_cost" not in modifiers and "current_reroll_cost" not in modifiers:
            modifiers["reroll_cost"] = 3
            modifiers["current_reroll_cost"] = 3
    elif label == "Seed Money":
        modifiers["interest_cap"] = 50
        modifiers["interest_cap_money"] = 50
    elif label == "Money Tree":
        modifiers["interest_cap"] = 100
        modifiers["interest_cap_money"] = 100
    elif label in {"Director's Cut", "Retcon"}:
        modifiers["boss_reroll_cost"] = 10
        if label == "Retcon":
            modifiers["boss_rerolls_unlimited"] = True
        else:
            modifiers["boss_rerolled"] = False
    elif label in {"Hieroglyph", "Petroglyph"}:
        updated_state, modifiers = _state_and_modifiers_after_ante_shift(updated_state, modifiers, -1)

    if modifiers == state.modifiers and updated_state is state:
        return state
    return replace(updated_state, modifiers=modifiers)


def _voucher_discount_percent(label: str) -> int:
    if label == "Liquidation":
        return 50
    if label == "Clearance Sale":
        return 25
    return 0


def _voucher_hand_delta(label: str) -> int:
    if label in {"Grabber", "Nacho Tong"}:
        return 1
    if label == "Hieroglyph":
        return -1
    return 0


def _voucher_discard_delta(label: str) -> int:
    if label in {"Wasteful", "Recyclomancy"}:
        return 1
    if label == "Petroglyph":
        return -1
    return 0


def _voucher_shop_slot_delta(label: str) -> int:
    if label in {"Overstock", "Overstock Plus"}:
        return 1
    return 0


def _with_discounted_item_cost(item: object, discount_percent: int) -> object:
    if not isinstance(item, dict):
        return item
    buy_cost = _item_buy_cost(item)
    if buy_cost <= 0:
        return item
    discounted_buy = _discounted_cost(buy_cost, discount_percent)
    updated = dict(item)
    cost = dict(updated.get("cost")) if isinstance(updated.get("cost"), dict) else {}
    cost["buy"] = discounted_buy
    cost["sell"] = max(1, discounted_buy // 2)
    updated["cost"] = cost
    return updated


def _shop_item_with_tag_edition(item: object, edition: str) -> object:
    normalized = _normalized_edition(edition)
    if isinstance(item, Joker):
        return _joker_with_edition(item, normalized)
    if not isinstance(item, dict):
        return item

    updated = dict(item)
    modifier = dict(updated.get("modifier")) if isinstance(updated.get("modifier"), dict) else {}
    modifier["edition"] = normalized
    updated["modifier"] = modifier
    updated["edition"] = normalized

    base_buy_cost = _item_buy_cost(item)
    if base_buy_cost <= 0:
        raw_sell_value = updated.get("sell_value")
        if raw_sell_value is None and isinstance(updated.get("cost"), dict):
            raw_sell_value = updated["cost"].get("sell")  # type: ignore[index]
        base_buy_cost = max(0, _int_value(raw_sell_value) * 2)
    sell_value = max(1, (base_buy_cost + EDITION_EXTRA_COST.get(normalized, 0)) // 2)
    cost = dict(updated.get("cost")) if isinstance(updated.get("cost"), dict) else {}
    cost["buy"] = 0
    cost["sell"] = sell_value
    updated["cost"] = cost
    updated["sell_value"] = sell_value
    return updated


def _active_shop_edition_tag(state: GameState, item: object) -> tuple[str | None, str | None]:
    if not _item_is_joker(item) or _item_has_edition(item) or state.modifiers.get("shop_edition_tag_consumed"):
        return None, None

    for tag_name in _pending_tag_names(state.modifiers):
        edition = SHOP_EDITION_BY_TAG.get(tag_name)
        if edition is not None:
            return tag_name, edition

    current_blind = state.modifiers.get("current_blind")
    if isinstance(current_blind, dict):
        if str(current_blind.get("status", "")).upper() != "SKIPPED":
            return None, None
        for key in ("tag_name", "tag", "skip_tag", "tag_label", "tag_key"):
            tag_name = _normalize_tag_name(current_blind.get(key))
            edition = SHOP_EDITION_BY_TAG.get(tag_name)
            if edition is not None:
                return tag_name, edition
    return None, None


def _first_shop_edition_tag_target_index(items: tuple[object, ...]) -> int | None:
    for index, item in enumerate(items):
        if _item_is_joker(item) and not _item_has_edition(item):
            return index
    return None


def _with_shop_edition_at_index(items: tuple[object, ...], index: int, edition: str) -> tuple[object, ...]:
    if not 0 <= index < len(items):
        return items
    updated = list(items)
    updated[index] = _shop_item_with_tag_edition(updated[index], edition)
    return tuple(updated)


def _item_has_edition(item: object) -> bool:
    if isinstance(item, Joker):
        return bool(item.edition)
    if not isinstance(item, dict):
        return False
    modifier = item.get("modifier")
    return bool(item.get("edition") or (modifier.get("edition") if isinstance(modifier, dict) else None))


def _modifiers_after_shop_edition_tag_used(modifiers: dict[str, object], tag_name: str) -> dict[str, object]:
    updated = dict(modifiers)
    tags = list(_pending_tag_names(modifiers))
    if tag_name in tags:
        tags.remove(tag_name)
        updated["tags"] = tuple(tags)
    updated["shop_edition_tag_consumed"] = True
    return updated


def _pending_tag_names(modifiers: dict[str, object]) -> tuple[str, ...]:
    raw = modifiers.get("tags", ())
    if not isinstance(raw, list | tuple):
        return ()
    names: list[str] = []
    for tag in raw:
        name = _normalize_tag_name(tag)
        if name:
            names.append(name)
    return tuple(names)


def _normalize_tag_name(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    key_map = {
        "tag_negative": "Negative Tag",
        "negative": "Negative Tag",
        "negative tag": "Negative Tag",
        "tag_foil": "Foil Tag",
        "foil": "Foil Tag",
        "foil tag": "Foil Tag",
        "tag_holo": "Holographic Tag",
        "tag_holographic": "Holographic Tag",
        "holographic": "Holographic Tag",
        "holographic tag": "Holographic Tag",
        "tag_polychrome": "Polychrome Tag",
        "polychrome": "Polychrome Tag",
        "polychrome tag": "Polychrome Tag",
    }
    return key_map.get(text.lower(), text)


def _with_discounted_joker_sell_value(joker: Joker, discount_percent: int) -> Joker:
    extra_value = _joker_extra_value(joker)
    current_buy_cost = _joker_buy_cost_for_discount(joker, extra_value=extra_value)
    if current_buy_cost <= 0:
        return joker
    discounted_buy = _discounted_cost(current_buy_cost, discount_percent)
    sell_value = max(1, discounted_buy // 2) + extra_value
    metadata = dict(joker.metadata)
    cost = dict(metadata.get("cost")) if isinstance(metadata.get("cost"), dict) else {}
    cost["buy"] = discounted_buy
    cost["sell"] = sell_value
    metadata["cost"] = cost
    return Joker(joker.name, edition=joker.edition, sell_value=sell_value, metadata=metadata)


def _joker_buy_cost_for_discount(joker: Joker, *, extra_value: int) -> int:
    for source in _metadata_sources(joker.metadata):
        cost = source.get("cost")
        if isinstance(cost, dict):
            buy_cost = _int_value(cost.get("buy", cost.get("cost", 0)))
            if buy_cost > 0:
                return buy_cost
    if joker.sell_value is None:
        return 0
    return max(0, (joker.sell_value - extra_value) * 2)


def _joker_extra_value(joker: Joker) -> int:
    value = _joker_numeric_value(joker, keys=("extra_value",))
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0


def _discounted_cost(current_cost: int, discount_percent: int) -> int:
    return max(1, ((current_cost * 2 + 1) * (100 - discount_percent)) // 200)


def _state_after_pack_choice(
    state: GameState,
    item: object,
    *,
    card_indices: Iterable[int] = (),
    created_consumables: Iterable[object] = (),
    created_jokers: Iterable[object] = (),
    created_hand_cards: Iterable[Card] = (),
    destroyed_hand_card_indices: Iterable[int] = (),
    aura_edition: str | None = None,
    wheel_of_fortune_joker_index: int | None = None,
    wheel_of_fortune_edition: str | None = None,
    sigil_suit: str | None = None,
    ouija_rank: str | None = None,
    spectral_joker_index: int | None = None,
) -> GameState:
    planet_hand = _planet_hand_type(item)
    if planet_hand is not None:
        return _state_after_planet_used(state, planet_hand)

    tarot_name = _tarot_name(item)
    if tarot_name is not None:
        return _state_after_tarot_used(
            state,
            tarot_name,
            card_indices=card_indices,
            created_consumables=created_consumables,
            created_jokers=created_jokers,
            wheel_of_fortune_joker_index=wheel_of_fortune_joker_index,
            wheel_of_fortune_edition=wheel_of_fortune_edition,
        )

    spectral_name = _spectral_name(item)
    if spectral_name is not None:
        return _state_after_spectral_used(
            state,
            spectral_name,
            card_indices=card_indices,
            created_jokers=created_jokers,
            created_hand_cards=created_hand_cards,
            destroyed_hand_card_indices=destroyed_hand_card_indices,
            aura_edition=aura_edition,
            sigil_suit=sigil_suit,
            ouija_rank=ouija_rank,
            spectral_joker_index=spectral_joker_index,
        )

    playing_card = _playing_card_from_item(item)
    if playing_card is not None:
        known_deck = state.known_deck + (playing_card,) if state.known_deck else ()
        return _state_with_dynamic_joker_values(
            replace(
                state,
                deck_size=state.deck_size + 1,
                known_deck=known_deck,
                jokers=_jokers_after_playing_cards_added(state.jokers, (playing_card,)),
            )
        )

    if _item_is_consumable(item):
        return state

    return _state_after_acquiring_item(state, item)


def _state_after_planet_used(state: GameState, hand_type: HandType) -> GameState:
    hand_name = hand_type.value
    updated_levels = dict(state.hand_levels)
    updated_levels[hand_name] = max(1, _int_value(updated_levels.get(hand_name))) + 1
    updated_modifiers = dict(state.modifiers)
    updated_modifiers = _record_planet_used(updated_modifiers, hand_type)
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
    return replace(
        state,
        hand_levels=updated_levels,
        jokers=_jokers_after_planet_used(state.jokers),
        modifiers=updated_modifiers,
    )


def _state_after_consumable_used(
    state: GameState,
    name: str,
    *,
    card_indices: Iterable[int] = (),
    created_consumables: Iterable[object] = (),
    created_jokers: Iterable[object] = (),
    created_hand_cards: Iterable[Card] = (),
    destroyed_hand_card_indices: Iterable[int] = (),
    aura_edition: str | None = None,
    wheel_of_fortune_joker_index: int | None = None,
    wheel_of_fortune_edition: str | None = None,
    sigil_suit: str | None = None,
    ouija_rank: str | None = None,
    spectral_joker_index: int | None = None,
) -> GameState:
    planet_hand = PLANET_TO_HAND.get(name)
    if planet_hand is not None:
        return _state_after_planet_used(state, planet_hand)
    if name in TAROT_NAMES:
        return _state_after_tarot_used(
            state,
            name,
            card_indices=card_indices,
            created_consumables=created_consumables,
            created_jokers=created_jokers,
            wheel_of_fortune_joker_index=wheel_of_fortune_joker_index,
            wheel_of_fortune_edition=wheel_of_fortune_edition,
        )
    if name in SPECTRAL_NAMES:
        return _state_after_spectral_used(
            state,
            name,
            card_indices=card_indices,
            created_jokers=created_jokers,
            created_hand_cards=created_hand_cards,
            destroyed_hand_card_indices=destroyed_hand_card_indices,
            aura_edition=aura_edition,
            sigil_suit=sigil_suit,
            ouija_rank=ouija_rank,
            spectral_joker_index=spectral_joker_index,
        )
    raise ValueError(f"Unsupported consumable: {name}")


def _state_after_tarot_used(
    state: GameState,
    tarot_name: str,
    *,
    card_indices: Iterable[int] = (),
    created_consumables: Iterable[object] = (),
    created_jokers: Iterable[object] = (),
    wheel_of_fortune_joker_index: int | None = None,
    wheel_of_fortune_edition: str | None = None,
) -> GameState:
    updated_modifiers = _record_tarot_used(state.modifiers, tarot_name)
    money = state.money
    jokers = state.jokers
    hand = state.hand
    deck_size = state.deck_size
    known_deck = state.known_deck
    consumables = state.consumables
    indices = tuple(card_indices)

    if tarot_name in TAROT_ENHANCEMENTS:
        _require_selected_indices(state, indices, min_count=1, max_count=2 if tarot_name in {"The Magician", "The Empress", "The Hierophant"} else 1, label=tarot_name)
        hand = _hand_with_replaced_cards(hand, indices, lambda card: _card_with_enhancement(card, TAROT_ENHANCEMENTS[tarot_name]))
    elif tarot_name in TAROT_SUIT_CONVERSIONS:
        _require_selected_indices(state, indices, min_count=1, max_count=3, label=tarot_name)
        hand = _hand_with_replaced_cards(hand, indices, lambda card: _card_with_suit(card, TAROT_SUIT_CONVERSIONS[tarot_name]))
    elif tarot_name == "Strength":
        _require_selected_indices(state, indices, min_count=1, max_count=2, label=tarot_name)
        hand = _hand_with_replaced_cards(hand, indices, lambda card: _card_with_rank(card, STRENGTH_RANKS[_normalize_rank(card.rank)]))
    elif tarot_name == "Death":
        _require_selected_indices(state, indices, min_count=2, max_count=2, label=tarot_name)
        source_index = max(indices)
        source = hand[source_index]
        hand = _hand_with_replacement_map(hand, {index: _copy_card(source) for index in indices if index != source_index})
    elif tarot_name == "The Hanged Man":
        _require_selected_indices(state, indices, min_count=1, max_count=2, label=tarot_name)
        destroyed_cards = _hand_cards_at_indices(state, indices)
        hand = _hand_without_indices(hand, indices)
        updated_modifiers = _with_destroyed_cards(updated_modifiers, destroyed_cards)
        jokers = _jokers_after_playing_cards_destroyed(jokers, destroyed_cards, glass_shattered=())
    if tarot_name == "The Hermit":
        money += min(max(0, money), 20)
    elif tarot_name == "Temperance":
        money += min(50, sum(max(0, joker.sell_value or 0) for joker in jokers))
    elif tarot_name == "The Fool":
        created = _created_labels(created_consumables)
        _validate_consumable_creations(state, created, max_count=1, label=tarot_name)
        consumables = consumables + created
        updated_modifiers = _increment_modifier_by(updated_modifiers, "created_consumable_count", len(created))
        updated_modifiers = _with_created_tarot_count(updated_modifiers, sum(1 for name in created if name in TAROT_NAMES))
    elif tarot_name == "The Emperor":
        created = _created_labels(created_consumables)
        _validate_consumable_creations(state, created, max_count=2, label=tarot_name)
        consumables = consumables + created
        updated_modifiers = _increment_modifier_by(updated_modifiers, "created_consumable_count", len(created))
        updated_modifiers = _with_created_tarot_count(updated_modifiers, len(created))
    elif tarot_name == "The High Priestess":
        created = _created_labels(created_consumables)
        _validate_consumable_creations(state, created, max_count=2, label=tarot_name)
        consumables = consumables + created
        updated_modifiers = _increment_modifier_by(updated_modifiers, "created_consumable_count", len(created))
    elif tarot_name in {"Judgement"}:
        created = tuple(_joker_from_item(item) for item in created_jokers)
        if len(created) > 1:
            raise ValueError(f"{tarot_name} can create at most 1 joker, got {len(created)}")
        jokers, updated_modifiers = _jokers_and_modifiers_after_created_jokers(state, jokers, updated_modifiers, created)
    elif tarot_name == "The Wheel of Fortune":
        jokers = _jokers_after_wheel_of_fortune(
            jokers,
            joker_index=wheel_of_fortune_joker_index,
            edition=wheel_of_fortune_edition,
        )
    return replace(
        state,
        money=money,
        hand=hand,
        deck_size=deck_size,
        known_deck=known_deck,
        jokers=_jokers_after_tarot_used(jokers),
        consumables=consumables,
        modifiers=updated_modifiers,
    )


def _state_after_spectral_used(
    state: GameState,
    spectral_name: str,
    *,
    card_indices: Iterable[int] = (),
    created_jokers: Iterable[object] = (),
    created_hand_cards: Iterable[Card] = (),
    destroyed_hand_card_indices: Iterable[int] = (),
    aura_edition: str | None = None,
    sigil_suit: str | None = None,
    ouija_rank: str | None = None,
    spectral_joker_index: int | None = None,
) -> GameState:
    updated_modifiers = _record_spectral_used(state.modifiers, spectral_name)
    money = state.money
    hand = state.hand
    deck_size = state.deck_size
    known_deck = state.known_deck
    jokers = state.jokers
    indices = tuple(card_indices)

    if spectral_name in SPECTRAL_SEALS:
        _require_selected_indices(state, indices, min_count=1, max_count=1, label=spectral_name)
        hand = _hand_with_replaced_cards(hand, indices, lambda card: _card_with_seal(card, SPECTRAL_SEALS[spectral_name]))
    elif spectral_name == "Aura":
        _require_selected_indices(state, indices, min_count=1, max_count=1, label=spectral_name)
        if aura_edition is None:
            raise ValueError("Aura requires an injected edition")
        hand = _hand_with_replaced_cards(hand, indices, lambda card: _card_with_edition(card, _normalized_edition(aura_edition)))
    elif spectral_name == "Cryptid":
        _require_selected_indices(state, indices, min_count=1, max_count=1, label=spectral_name)
        source = state.hand[indices[0]]
        created = (_copy_card(source), _copy_card(source))
        hand = _sort_hand_cards(hand + created)
        deck_size += len(created)
        known_deck = known_deck + created if known_deck else ()
        jokers = _jokers_after_playing_cards_added(jokers, created)
    elif spectral_name in {"Familiar", "Grim", "Incantation", "Immolate"}:
        destroyed_cards = _destroyed_hand_cards_for_consumable(state, destroyed_hand_card_indices)
        if spectral_name == "Immolate":
            _validate_injected_count_at_most(label=spectral_name, expected=min(5, len(state.hand)), injected=len(destroyed_cards))
            money += 20
        else:
            _validate_injected_count_at_most(label=spectral_name, expected=min(1, len(state.hand)), injected=len(destroyed_cards))
        hand = _without_card_instances(hand, destroyed_cards)
        if destroyed_cards:
            updated_modifiers = _with_destroyed_cards(updated_modifiers, destroyed_cards)
            jokers = _jokers_after_playing_cards_destroyed(jokers, destroyed_cards, glass_shattered=())
        created = tuple(created_hand_cards)
        hand = _sort_hand_cards(hand + created)
        deck_size += len(created)
        known_deck = known_deck + created if known_deck else ()
        jokers = _jokers_after_playing_cards_added(jokers, created)
    elif spectral_name == "Sigil":
        if sigil_suit is None:
            raise ValueError("Sigil requires an injected suit")
        hand = tuple(_card_with_suit(card, _normalize_suit(sigil_suit)) for card in hand)
    elif spectral_name == "Ouija":
        if ouija_rank is None:
            raise ValueError("Ouija requires an injected rank")
        hand = tuple(_card_with_rank(card, _normalize_rank(ouija_rank)) for card in hand)
        updated_modifiers = _add_hand_size_delta(updated_modifiers, -1)
    elif spectral_name == "Black Hole":
        hand_levels = {hand_type.value: max(1, _int_value(state.hand_levels.get(hand_type.value))) + 1 for hand_type in HandType}
        updated_modifiers = _with_all_hand_levels(updated_modifiers, hand_levels)
        return replace(state, hand_levels=hand_levels, modifiers=updated_modifiers)
    elif spectral_name in {"Wraith", "The Soul"}:
        created = tuple(_joker_from_item(item) for item in created_jokers)
        if len(created) > 1:
            raise ValueError(f"{spectral_name} can create at most 1 joker, got {len(created)}")
        jokers, updated_modifiers = _jokers_and_modifiers_after_created_jokers(state, jokers, updated_modifiers, created)
        if spectral_name == "Wraith" and created:
            money = 0
    elif spectral_name == "Ectoplasm":
        if spectral_joker_index is None:
            raise ValueError("Ectoplasm requires an injected joker index")
        jokers = _jokers_with_single_edition(jokers, spectral_joker_index, "NEGATIVE")
        updated_modifiers = _add_hand_size_delta(updated_modifiers, -1)
    elif spectral_name == "Hex":
        if spectral_joker_index is None:
            raise ValueError("Hex requires an injected joker index")
        jokers, updated_modifiers = _jokers_after_hex(state, jokers, updated_modifiers, spectral_joker_index)
    elif spectral_name == "Ankh":
        if spectral_joker_index is None:
            raise ValueError("Ankh requires an injected joker index")
        jokers, updated_modifiers = _jokers_after_ankh(state, jokers, updated_modifiers, spectral_joker_index)

    return replace(
        state,
        money=money,
        hand=hand,
        deck_size=deck_size,
        known_deck=known_deck,
        jokers=jokers,
        modifiers=updated_modifiers,
    )


def _record_tarot_used(modifiers: dict[str, object], tarot_name: str) -> dict[str, object]:
    updated = dict(modifiers)
    usage = updated.get("tarot_cards_used")
    used = dict(usage) if isinstance(usage, dict) else {}
    used[tarot_name] = _int_value(used.get(tarot_name)) + 1
    updated["tarot_cards_used"] = used
    updated["tarot_cards_used_total"] = _int_value(updated.get("tarot_cards_used_total")) + 1
    if tarot_name != "The Fool":
        updated["last_tarot_planet"] = tarot_name
        updated["last_tarot_planet_name"] = tarot_name
    return updated


def _record_planet_used(modifiers: dict[str, object], hand_type: HandType) -> dict[str, object]:
    updated = dict(modifiers)
    usage = updated.get("planet_cards_used")
    used = dict(usage) if isinstance(usage, dict) else {}
    planet_name = _planet_for_hand_type(hand_type)
    used[planet_name] = _int_value(used.get(planet_name)) + 1
    updated["planet_cards_used"] = used
    updated["planet_cards_used_total"] = _int_value(updated.get("planet_cards_used_total")) + 1
    updated["last_tarot_planet"] = planet_name
    updated["last_tarot_planet_name"] = planet_name
    return updated


def _record_spectral_used(modifiers: dict[str, object], spectral_name: str) -> dict[str, object]:
    updated = dict(modifiers)
    usage = updated.get("spectral_cards_used")
    used = dict(usage) if isinstance(usage, dict) else {}
    used[spectral_name] = _int_value(used.get(spectral_name)) + 1
    updated["spectral_cards_used"] = used
    updated["spectral_cards_used_total"] = _int_value(updated.get("spectral_cards_used_total")) + 1
    return updated


def _planet_for_hand_type(hand_type: HandType) -> str:
    for planet, mapped_hand in PLANET_TO_HAND.items():
        if mapped_hand == hand_type:
            return planet
    return hand_type.value


def _require_selected_indices(
    state: GameState,
    indices: tuple[int, ...],
    *,
    min_count: int,
    max_count: int,
    label: str,
) -> None:
    if len(indices) < min_count or len(indices) > max_count:
        if min_count == max_count:
            raise ValueError(f"{label} requires exactly {min_count} selected card(s), got {len(indices)}")
        raise ValueError(f"{label} requires {min_count}-{max_count} selected card(s), got {len(indices)}")
    if any(index < 0 or index >= len(state.hand) for index in indices):
        raise ValueError(f"{label} selected card indices are outside the hand: {indices}")


def _hand_cards_at_indices(state: GameState, indices: tuple[int, ...]) -> tuple[Card, ...]:
    if any(index < 0 or index >= len(state.hand) for index in indices):
        raise ValueError(f"Selected card indices are outside the hand: {indices}")
    return tuple(state.hand[index] for index in indices)


def _hand_with_replaced_cards(hand: tuple[Card, ...], indices: tuple[int, ...], transform) -> tuple[Card, ...]:
    selected = set(indices)
    return tuple(transform(card) if index in selected else card for index, card in enumerate(hand))


def _hand_with_replacement_map(hand: tuple[Card, ...], replacements: dict[int, Card]) -> tuple[Card, ...]:
    return tuple(replacements.get(index, card) for index, card in enumerate(hand))


def _hand_without_indices(hand: tuple[Card, ...], indices: tuple[int, ...]) -> tuple[Card, ...]:
    selected = set(indices)
    return tuple(card for index, card in enumerate(hand) if index not in selected)


def _destroyed_hand_cards_for_consumable(state: GameState, destroyed_hand_card_indices: Iterable[int]) -> tuple[Card, ...]:
    indices = tuple(destroyed_hand_card_indices)
    if indices:
        return _hand_cards_at_indices(state, indices)
    return ()


def _card_with_enhancement(card: Card, enhancement: str) -> Card:
    metadata = dict(card.metadata)
    modifier = dict(metadata.get("modifier")) if isinstance(metadata.get("modifier"), dict) else {}
    modifier["enhancement"] = enhancement
    metadata["modifier"] = modifier
    return replace(card, enhancement=enhancement, metadata=metadata)


def _card_with_suit(card: Card, suit: str) -> Card:
    normalized = _normalize_suit(suit)
    metadata = dict(card.metadata)
    value = dict(metadata.get("value")) if isinstance(metadata.get("value"), dict) else {}
    value["suit"] = normalized
    metadata["value"] = value
    return replace(card, suit=normalized, metadata=metadata)


def _card_with_rank(card: Card, rank: str) -> Card:
    normalized = _normalize_rank(rank)
    metadata = dict(card.metadata)
    value = dict(metadata.get("value")) if isinstance(metadata.get("value"), dict) else {}
    value["rank"] = normalized
    metadata["value"] = value
    return replace(card, rank=normalized, metadata=metadata)


def _card_with_seal(card: Card, seal: str) -> Card:
    metadata = dict(card.metadata)
    modifier = dict(metadata.get("modifier")) if isinstance(metadata.get("modifier"), dict) else {}
    modifier["seal"] = seal
    metadata["modifier"] = modifier
    return replace(card, seal=seal, metadata=metadata)


def _card_with_edition(card: Card, edition: str) -> Card:
    metadata = dict(card.metadata)
    modifier = dict(metadata.get("modifier")) if isinstance(metadata.get("modifier"), dict) else {}
    modifier["edition"] = edition
    metadata["modifier"] = modifier
    return replace(card, edition=edition, metadata=metadata)


def _validate_consumable_creations(state: GameState, created: tuple[str, ...], *, max_count: int, label: str) -> None:
    if len(created) > min(max_count, _consumable_open_slots(state)):
        raise ValueError(f"{label} can create at most {min(max_count, _consumable_open_slots(state))} consumable(s), got {len(created)}")


def _jokers_and_modifiers_after_created_jokers(
    state: GameState,
    jokers: tuple[Joker, ...],
    modifiers: dict[str, object],
    created: tuple[Joker, ...],
) -> tuple[tuple[Joker, ...], dict[str, object]]:
    if not created:
        return jokers, modifiers
    normal_created = sum(1 for joker in created if _joker_uses_normal_slot(joker))
    if normal_created > max(0, _normal_joker_slot_limit(state) - sum(1 for joker in jokers if _joker_uses_normal_slot(joker))):
        raise ValueError(f"Created joker does not fit in joker slots: {normal_created}")
    updated_modifiers = modifiers
    for joker in created:
        updated_modifiers = _modifiers_after_joker_added(updated_modifiers, joker)
    return jokers + created, updated_modifiers


def _jokers_with_single_edition(jokers: tuple[Joker, ...], index: int, edition: str) -> tuple[Joker, ...]:
    if not 0 <= index < len(jokers):
        raise ValueError(f"Spectral joker index is outside jokers: {index}")
    updated = list(jokers)
    updated[index] = _joker_with_edition(updated[index], edition)
    return tuple(updated)


def _jokers_after_hex(
    state: GameState,
    jokers: tuple[Joker, ...],
    modifiers: dict[str, object],
    index: int,
) -> tuple[tuple[Joker, ...], dict[str, object]]:
    if not 0 <= index < len(jokers):
        raise ValueError(f"Hex joker index is outside jokers: {index}")
    updated_modifiers = modifiers
    kept: list[Joker] = []
    for joker_index, joker in enumerate(jokers):
        if joker_index == index:
            kept.append(_joker_with_edition(joker, "POLYCHROME"))
        elif _joker_is_eternal(joker):
            kept.append(joker)
        else:
            updated_modifiers = _modifiers_after_joker_removed(updated_modifiers, joker)
    return tuple(kept), updated_modifiers


def _jokers_after_ankh(
    state: GameState,
    jokers: tuple[Joker, ...],
    modifiers: dict[str, object],
    index: int,
) -> tuple[tuple[Joker, ...], dict[str, object]]:
    if not 0 <= index < len(jokers):
        raise ValueError(f"Ankh joker index is outside jokers: {index}")
    selected = jokers[index]
    copied = _copy_joker_for_ankh(selected)
    updated_modifiers = modifiers
    kept: list[Joker] = []
    for joker_index, joker in enumerate(jokers):
        if joker_index == index or _joker_is_eternal(joker):
            kept.append(joker)
        else:
            updated_modifiers = _modifiers_after_joker_removed(updated_modifiers, joker)
    kept.append(copied)
    updated_modifiers = _modifiers_after_joker_added(updated_modifiers, copied)
    return tuple(kept), updated_modifiers


def _copy_joker_for_ankh(joker: Joker) -> Joker:
    copied = _joker_with_initial_dynamic_values(Joker(joker.name, edition=joker.edition, sell_value=joker.sell_value, metadata=_copy_metadata(joker.metadata)))
    if copied.edition != "NEGATIVE":
        return copied
    metadata = dict(copied.metadata)
    modifier = dict(metadata.get("modifier")) if isinstance(metadata.get("modifier"), dict) else {}
    modifier.pop("edition", None)
    metadata["modifier"] = modifier
    return Joker(copied.name, edition=None, sell_value=copied.sell_value, metadata=metadata)


def _joker_is_eternal(joker: Joker) -> bool:
    for source in _metadata_sources(joker.metadata):
        if _truthy_modifier(source.get("eternal")):
            return True
        stickers = source.get("stickers")
        if isinstance(stickers, dict) and _truthy_modifier(stickers.get("eternal")):
            return True
        if isinstance(stickers, list | tuple | set) and any(str(sticker).lower() == "eternal" for sticker in stickers):
            return True
    return False


def _add_hand_size_delta(modifiers: dict[str, object], amount: int) -> dict[str, object]:
    updated = dict(modifiers)
    _add_modifier_int(updated, "hand_size_delta", amount)
    return updated


def _with_all_hand_levels(modifiers: dict[str, object], hand_levels: dict[str, int]) -> dict[str, object]:
    updated = dict(modifiers)
    raw_hands = updated.get("hands")
    hands = dict(raw_hands) if isinstance(raw_hands, dict) else {}
    for hand_name, level in hand_levels.items():
        current = hands.get(hand_name)
        entry = dict(current) if isinstance(current, dict) else {}
        entry["level"] = level
        hands[hand_name] = entry
    updated["hands"] = hands
    return updated


def _jokers_after_planet_used(jokers: tuple[Joker, ...]) -> tuple[Joker, ...]:
    updated: list[Joker] = []
    for joker in jokers:
        if joker.name == "Constellation" and not _joker_is_disabled(joker):
            current = _joker_float(joker, keys=("current_xmult", "xmult", "current_x_mult", "x_mult"), default=1.0)
            gain = _joker_float(joker, keys=("extra", "xmult_gain"), default=0.1)
            updated.append(_with_joker_metadata(joker, {"current_xmult": current + gain, "xmult": current + gain}))
        else:
            updated.append(joker)
    return tuple(updated)


def _jokers_after_tarot_used(jokers: tuple[Joker, ...]) -> tuple[Joker, ...]:
    updated: list[Joker] = []
    for joker in jokers:
        if joker.name == "Fortune Teller" and not _joker_is_disabled(joker):
            current = _joker_int(joker, keys=("current_mult", "mult"), default=0)
            gain = _joker_int(joker, keys=("extra", "mult_gain"), default=1)
            updated.append(_with_joker_metadata(joker, {"current_mult": current + gain, "mult": current + gain}))
        else:
            updated.append(joker)
    return tuple(updated)


def _jokers_after_wheel_of_fortune(
    jokers: tuple[Joker, ...],
    *,
    joker_index: int | None,
    edition: str | None,
) -> tuple[Joker, ...]:
    if joker_index is None:
        return jokers
    if edition is None:
        raise ValueError("Wheel of Fortune hit requires an injected edition")
    if not 0 <= joker_index < len(jokers):
        raise ValueError(f"Wheel of Fortune joker index is outside jokers: {joker_index}")
    joker = jokers[joker_index]
    if joker.edition:
        raise ValueError(f"Wheel of Fortune target already has an edition: {joker.name}")
    updated = list(jokers)
    updated[joker_index] = _joker_with_edition(joker, edition)
    return tuple(updated)


def _joker_with_edition(joker: Joker, edition: str) -> Joker:
    normalized = _normalized_edition(edition)
    metadata = dict(joker.metadata)
    modifier = dict(metadata.get("modifier")) if isinstance(metadata.get("modifier"), dict) else {}
    modifier["edition"] = normalized
    metadata["modifier"] = modifier
    sell_value = _sell_value_after_adding_edition(joker, normalized)
    return Joker(joker.name, edition=normalized, sell_value=sell_value, metadata=metadata)


def _normalized_edition(edition: str) -> str:
    value = edition.strip().lower()
    if value in {"holo", "holographic"}:
        return "HOLOGRAPHIC"
    if value in {"poly", "polychrome"}:
        return "POLYCHROME"
    if value == "foil":
        return "FOIL"
    if value == "negative":
        return "NEGATIVE"
    return edition


def _sell_value_after_adding_edition(joker: Joker, edition: str) -> int | None:
    if joker.sell_value is None:
        return None
    extra_cost = EDITION_EXTRA_COST.get(edition, 0)
    extra_value = _joker_extra_value(joker)
    current_cost = _joker_buy_cost_for_discount(joker, extra_value=extra_value)
    if current_cost <= 0:
        current_cost = max(1, (joker.sell_value - extra_value) * 2)
    return max(1, (current_cost + extra_cost) // 2) + extra_value


def _jokers_after_playing_cards_added(jokers: tuple[Joker, ...], cards: tuple[Card, ...]) -> tuple[Joker, ...]:
    if not cards:
        return jokers
    updated: list[Joker] = []
    for joker in jokers:
        if joker.name == "Hologram" and not _joker_is_disabled(joker):
            current = _joker_float(joker, keys=("current_xmult", "xmult", "current_x_mult", "x_mult"), default=1.0)
            gain = len(cards) * _joker_float(joker, keys=("extra", "xmult_gain"), default=0.25)
            updated.append(_with_joker_metadata(joker, {"current_xmult": current + gain, "xmult": current + gain}))
        else:
            updated.append(joker)
    return tuple(updated)


def _jokers_after_playing_cards_destroyed(
    jokers: tuple[Joker, ...],
    destroyed_cards: tuple[Card, ...],
    *,
    glass_shattered: tuple[Card, ...],
) -> tuple[Joker, ...]:
    if not destroyed_cards:
        return jokers
    face_count = sum(1 for card in destroyed_cards if _is_face_card(card, jokers))
    shattered_count = len(glass_shattered)
    updated: list[Joker] = []
    for joker in jokers:
        if _joker_is_disabled(joker):
            updated.append(joker)
        elif joker.name in {"Caino", "Canio"} and face_count:
            current = _joker_float(joker, keys=("current_xmult", "caino_xmult", "xmult", "current_x_mult"), default=1.0)
            gain = face_count * _joker_float(joker, keys=("extra", "xmult_gain"), default=1.0)
            updated.append(_with_joker_metadata(joker, {"current_xmult": current + gain, "caino_xmult": current + gain}))
        elif joker.name == "Glass Joker" and shattered_count:
            current = _joker_float(joker, keys=("current_xmult", "xmult", "current_x_mult", "x_mult"), default=1.0)
            gain = shattered_count * _joker_float(joker, keys=("extra", "xmult_gain"), default=0.75)
            updated.append(_with_joker_metadata(joker, {"current_xmult": current + gain, "xmult": current + gain}))
        else:
            updated.append(joker)
    return tuple(updated)


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


def _tarot_name(item: object) -> str | None:
    label = _item_label(item)
    if label in TAROT_NAMES:
        return label
    if _item_set(item) == "TAROT":
        return label
    key = _item_key(item)
    if key.startswith("c_") and label in TAROT_NAMES:
        return label
    return None


def _spectral_name(item: object) -> str | None:
    label = _item_label(item)
    if label in SPECTRAL_NAMES:
        return label
    if _item_set(item) == "SPECTRAL":
        return label
    key = _item_key(item)
    if key.startswith("c_") and label in SPECTRAL_NAMES:
        return label
    return None


def _joker_from_item(item: object) -> Joker:
    joker = Joker.from_mapping(item) if isinstance(item, dict | str) else Joker(str(item))
    if joker.sell_value is not None:
        return _joker_with_initial_dynamic_values(joker)
    buy_cost = _item_buy_cost(item)
    sell_value = max(1, buy_cost // 2) if buy_cost > 0 else None
    return _joker_with_initial_dynamic_values(
        Joker(joker.name, edition=joker.edition, sell_value=sell_value, metadata=joker.metadata)
    )


def _joker_with_initial_dynamic_values(joker: Joker) -> Joker:
    defaults: dict[str, object]
    if joker.name == "Ice Cream":
        defaults = {"current_chips": 100}
    elif joker.name == "Popcorn":
        defaults = {"current_mult": 20}
    elif joker.name == "Ramen":
        defaults = {"current_xmult": 2.0}
    elif joker.name == "Loyalty Card":
        defaults = {"current_remaining": 5}
    elif joker.name in {"Square Joker", "Runner", "Wee Joker", "Castle"}:
        defaults = {"current_chips": 0}
    elif joker.name in {"Green Joker", "Flash Card", "Red Card", "Spare Trousers", "Erosion", "Fortune Teller"}:
        defaults = {"current_mult": 0}
    elif joker.name in {"Constellation", "Hologram", "Lucky Cat", "Campfire", "Madness", "Glass Joker"}:
        defaults = {"current_xmult": 1.0}
    elif joker.name in {"Caino", "Canio"}:
        defaults = {"current_xmult": 1.0, "caino_xmult": 1.0}
    else:
        return joker

    metadata = dict(joker.metadata)
    for key, value in defaults.items():
        metadata.setdefault(key, value)
    return Joker(joker.name, edition=joker.edition, sell_value=joker.sell_value, metadata=metadata)


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
    label = _item_label(item)
    return (
        _item_set(item) in {"TAROT", "PLANET", "SPECTRAL"}
        or _item_key(item).startswith("c_")
        or label in TAROT_NAMES
        or label in SPECTRAL_NAMES
        or label in PLANET_TO_HAND
    )


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

    if room > 0 and _hand_is_straight_flush_family(evaluation.hand_type):
        count = min(room, _effective_joker_count(state.jokers, "Seance"))
        if count > 0:
            requests.append(("spectral", count))
            room -= count

    tarot_count = 0
    if room > 0 and _hand_contains_straight(evaluation.hand_type) and _played_hand_has_ace(played_cards):
        tarot_count += _effective_joker_count(state.jokers, "Superposition")
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


def _blue_seal_creation_requests(
    state: GameState,
    held_cards: tuple[Card, ...],
    *,
    reserved_slots: int,
) -> tuple[tuple[str, int], ...]:
    room = max(0, _consumable_open_slots(state) - reserved_slots)
    if room <= 0:
        return ()
    blue_seals = sum(1 for card in held_cards if not card.debuffed and str(card.seal or "").lower() == "blue")
    count = min(room, blue_seals)
    return (("planet", count),) if count > 0 else ()


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


def _state_with_current_discard_delta_change(previous_state: GameState, next_state: GameState) -> GameState:
    previous_delta = _int_value(previous_state.modifiers.get("round_discards_delta"))
    next_delta = _int_value(next_state.modifiers.get("round_discards_delta"))
    change = next_delta - previous_delta
    if change == 0:
        return next_state
    modifiers = dict(next_state.modifiers)
    modifiers["applied_round_discards_delta"] = _int_value(modifiers.get("applied_round_discards_delta")) + change
    return replace(
        next_state,
        discards_remaining=max(0, next_state.discards_remaining + change),
        modifiers=modifiers,
    )


def _state_and_modifiers_after_ante_shift(
    state: GameState,
    modifiers: dict[str, object],
    delta: int,
) -> tuple[GameState, dict[str, object]]:
    previous_ante = max(1, state.ante)
    next_ante = max(0, state.ante + delta)
    updated = dict(modifiers)
    updated["blind_ante"] = _int_value(updated.get("blind_ante", previous_ante)) + delta
    next_required = _score_shifted_between_antes(state.required_score, previous_ante, max(1, next_ante))
    updated = _modifiers_with_shifted_blind_scores(updated, previous_ante, max(1, next_ante), next_required)
    cleared_blind = updated.get("cleared_blind")
    if isinstance(cleared_blind, dict):
        cleared = dict(cleared_blind)
        raw_cleared_ante = _int_value(cleared.get("ante"))
        cleared_ante = max(1, raw_cleared_ante or previous_ante)
        shifted_cleared_ante = max(0, cleared_ante + delta)
        cleared["ante"] = shifted_cleared_ante
        if "required_score" in cleared:
            cleared["required_score"] = _score_shifted_between_antes(
                _int_value(cleared.get("required_score")),
                cleared_ante,
                max(1, shifted_cleared_ante),
            )
        updated["cleared_blind"] = cleared
    return replace(state, ante=next_ante, required_score=next_required), updated


def _modifiers_with_shifted_blind_scores(
    modifiers: dict[str, object],
    previous_ante: int,
    next_ante: int,
    next_required: int,
) -> dict[str, object]:
    updated = dict(modifiers)
    if "blind_score" in updated:
        updated["blind_score"] = next_required
    round_detail = updated.get("round")
    if isinstance(round_detail, dict):
        next_round = dict(round_detail)
        if "blind_score" in next_round:
            next_round["blind_score"] = next_required
        updated["round"] = next_round
    current_blind = updated.get("current_blind")
    if isinstance(current_blind, dict):
        next_blind = dict(current_blind)
        if "score" in next_blind:
            next_blind["score"] = next_required
        updated["current_blind"] = next_blind
    blinds = updated.get("blinds")
    if isinstance(blinds, dict):
        shifted_blinds: dict[object, object] = {}
        for key, payload in blinds.items():
            if isinstance(payload, dict):
                shifted = dict(payload)
                if "score" in shifted:
                    shifted["score"] = _score_shifted_between_antes(_int_value(shifted.get("score")), previous_ante, next_ante)
                shifted_blinds[key] = shifted
            else:
                shifted_blinds[key] = payload
        updated["blinds"] = shifted_blinds
    return updated


def _score_shifted_between_antes(score: int, previous_ante: int, next_ante: int) -> int:
    previous_small = _small_blind_score_for_ante(previous_ante)
    next_small = _small_blind_score_for_ante(next_ante)
    if previous_small <= 0:
        return score
    return int(score * next_small / previous_small)


def _small_blind_score_for_ante(ante: int) -> int:
    if ante in ANTE_SMALL_BLIND_SCORES:
        return ANTE_SMALL_BLIND_SCORES[ante]
    last_ante = max(ANTE_SMALL_BLIND_SCORES)
    last_score = ANTE_SMALL_BLIND_SCORES[last_ante]
    return int(last_score * (1.6 ** max(0, ante - last_ante)))


def _round_counts_after_shop_exit(
    state: GameState,
    modifiers: dict[str, object],
) -> tuple[int, int, dict[str, object]]:
    current_discards_delta = _int_value(modifiers.get("round_discards_delta"))
    applied_discards_delta = _int_value(modifiers.get("applied_round_discards_delta"))
    updated = dict(modifiers)
    updated["applied_round_discards_delta"] = current_discards_delta
    return (
        state.hands_remaining,
        max(0, state.discards_remaining + current_discards_delta - applied_discards_delta),
        updated,
    )


def _round_counts_after_blind_start(
    state: GameState,
    modifiers: dict[str, object],
) -> tuple[int, int, dict[str, object]]:
    current_hands_delta = _int_value(modifiers.get("round_hands_delta"))
    current_discards_delta = _int_value(modifiers.get("round_discards_delta"))
    applied_hands_delta = _int_value(modifiers.get("applied_round_hands_delta"))
    applied_discards_delta = _int_value(modifiers.get("applied_round_discards_delta"))
    updated = dict(modifiers)
    updated["applied_round_hands_delta"] = current_hands_delta
    updated["applied_round_discards_delta"] = current_discards_delta
    return (
        max(0, state.hands_remaining + current_hands_delta - applied_hands_delta),
        max(0, state.discards_remaining + current_discards_delta - applied_discards_delta),
        updated,
    )


def _with_applied_round_count_deltas(modifiers: dict[str, object]) -> dict[str, object]:
    updated = dict(modifiers)
    updated["applied_round_hands_delta"] = _int_value(updated.get("round_hands_delta"))
    updated["applied_round_discards_delta"] = _int_value(updated.get("round_discards_delta"))
    updated["round_count_deltas_applied"] = True
    return updated


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


def _consumable_sell_value(state: GameState, name: str) -> int:
    base = 2 if name in SPECTRAL_NAMES else 1
    bonus_total = _int_value(state.modifiers.get("consumable_sell_value_bonus"))
    bonus_each = bonus_total // max(1, len(state.consumables)) if bonus_total > 0 else 0
    return base + bonus_each


def _modifiers_after_consumable_removed(modifiers: dict[str, object], *, previous_count: int) -> dict[str, object]:
    bonus_total = _int_value(modifiers.get("consumable_sell_value_bonus"))
    if bonus_total <= 0:
        return modifiers
    bonus_each = bonus_total // max(1, previous_count)
    updated = dict(modifiers)
    updated["consumable_sell_value_bonus"] = max(0, bonus_total - bonus_each)
    if updated["consumable_sell_value_bonus"] <= 0:
        updated.pop("consumable_sell_value_bonus", None)
    return updated


def _with_added_tag(modifiers: dict[str, object], tag: str) -> dict[str, object]:
    updated = dict(modifiers)
    raw = updated.get("tags", ())
    tags = tuple(raw) if isinstance(raw, list | tuple) else ()
    updated["tags"] = tags + (tag,)
    return updated


def _modifiers_after_gift_card_round_end(
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


def _jokers_after_pack_skip(jokers: tuple[Joker, ...]) -> tuple[Joker, ...]:
    updated: list[Joker] = []
    for joker in jokers:
        if joker.name == "Red Card" and not _joker_is_disabled(joker):
            current = _joker_int(joker, keys=("current_mult", "mult"), default=0)
            gain = _joker_int(joker, keys=("extra", "mult_gain"), default=3)
            updated.append(_with_joker_metadata(joker, {"current_mult": current + gain, "mult": current + gain}))
        else:
            updated.append(joker)
    return tuple(updated)


def _jokers_after_gift_card_round_end(jokers: tuple[Joker, ...], *, gift_card_gain: int) -> tuple[Joker, ...]:
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


def _state_after_verdant_leaf_disabled(state: GameState) -> GameState:
    modifiers = dict(state.modifiers)
    modifiers["boss_disabled"] = True
    return replace(
        state,
        hand=tuple(_card_without_blind_debuff(card, force=True) for card in state.hand),
        known_deck=tuple(_card_without_blind_debuff(card, force=True) for card in state.known_deck),
        modifiers=modifiers,
    )


def _invisible_joker_ready(joker: Joker) -> bool:
    rounds = _joker_int(joker, keys=("current_rounds", "invis_rounds", "rounds"), default=0)
    needed = _joker_int(joker, keys=("extra", "rounds_needed"), default=2)
    return rounds >= needed


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
        if _joker_is_disabled(joker):
            updated.append(joker)
        elif joker.name == "Ice Cream":
            current = _joker_int(joker, keys=("current_chips", "chips"), default=100)
            if current - 5 > 0:
                updated.append(_with_joker_metadata(joker, {"current_chips": current - 5}))
        elif joker.name == "Seltzer":
            current = _seltzer_remaining(joker)
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
        if _joker_is_disabled(joker):
            updated.append(joker)
        elif joker.name == "Ramen":
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


def _jokers_after_egg_end_of_round(jokers: tuple[Joker, ...]) -> tuple[Joker, ...]:
    updated: list[Joker] = []
    for joker in jokers:
        if joker.name != "Egg":
            updated.append(joker)
            continue
        gain = _joker_int(joker, keys=("extra", "value_gain"), default=3)
        extra_value = _joker_int(joker, keys=("extra_value",), default=0) + gain
        sell_value = (joker.sell_value + gain) if joker.sell_value is not None else None
        updated.append(_with_joker_metadata(joker, {"extra_value": extra_value}, sell_value=sell_value))
    return tuple(updated)


def _jokers_after_played_round_ends(jokers: tuple[Joker, ...]) -> tuple[Joker, ...]:
    updated: list[Joker] = []
    for joker in _jokers_after_egg_end_of_round(jokers):
        if joker.name == "Popcorn":
            current = _joker_int(joker, keys=("current_mult", "mult"), default=20)
            decay = _joker_int(joker, keys=("decay", "mult_decay", "extra"), default=4)
            next_mult = current - decay
            if next_mult > 0:
                updated.append(_with_joker_metadata(joker, {"current_mult": next_mult}))
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
    if _effective_joker_count(state.jokers, "Burnt Joker") <= 0:
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
            if joker.name == "Trading Card" and not _joker_is_disabled(joker):
                delta += _joker_int(joker, keys=("dollars", "money", "extra"), default=3)
    if any(joker.name == "Faceless Joker" and not _joker_is_disabled(joker) for joker in state.jokers):
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
    oracle_delta = _modifier_int_or_none(
        state.modifiers,
        ("cash_out_money_delta", "current_round_dollars", "round_eval_dollars", "round_dollars"),
    )
    if oracle_delta is not None:
        return oracle_delta

    delta = _blind_reward(state)
    if not _truthy_modifier(state.modifiers.get("no_extra_hand_money")):
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
    no_reward = state.modifiers.get("no_blind_reward")
    blind_type = _blind_reward_type(state)
    if isinstance(no_reward, dict) and blind_type and _truthy_modifier(no_reward.get(blind_type)):
        return 0
    if isinstance(no_reward, (list, tuple, set)) and blind_type in {str(item) for item in no_reward}:
        return 0
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


def _blind_reward_type(state: GameState) -> str | None:
    current_blind = state.modifiers.get("current_blind")
    if isinstance(current_blind, dict):
        blind_type = str(current_blind.get("type", current_blind.get("kind", ""))).title()
        if blind_type in {"Small", "Big", "Boss"}:
            return blind_type
    if state.blind == "Small Blind":
        return "Small"
    if state.blind == "Big Blind":
        return "Big"
    return "Boss" if state.blind else None


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


def _state_with_dynamic_joker_values(state: GameState) -> GameState:
    return replace(state, jokers=_jokers_with_dynamic_state_values(state))


def _jokers_with_dynamic_state_values(state: GameState) -> tuple[Joker, ...]:
    if not state.jokers:
        return state.jokers

    full_deck = _known_full_deck_cards(state)
    visible_out_of_draw_deck = (
        len(state.hand)
        + len(_zone_cards(state.modifiers, "played_pile"))
        + len(_zone_cards(state.modifiers, "discard_pile"))
    )
    stone_count = sum(1 for card in full_deck if _normalized(card.enhancement) in {"stone", "stone card"})
    steel_count = sum(1 for card in full_deck if _normalized(card.enhancement) in {"steel", "steel card"})
    enhanced_count = sum(1 for card in full_deck if _is_enhanced_playing_card(card))
    current_deck_size = max(state.deck_size + visible_out_of_draw_deck, len(full_deck))
    starting_deck_size = max(
        current_deck_size,
        _modifier_int(state.modifiers, ("starting_deck_size", "starting_playing_card_count"), default=52),
    )
    missing_cards = max(0, starting_deck_size - current_deck_size)
    stencil_xmult = _normal_joker_slot_limit(state) - _normal_joker_slots_used(state) + sum(
        1 for joker in state.jokers if joker.name == "Joker Stencil"
    )

    updated: list[Joker] = []
    for joker in state.jokers:
        if joker.name == "Joker Stencil":
            updated.append(_with_joker_metadata(joker, {"current_xmult": float(stencil_xmult), "xmult": float(stencil_xmult)}))
        elif joker.name == "Stone Joker":
            chips = 25 * stone_count
            updated.append(_with_joker_metadata(joker, {"stone_tally": stone_count, "current_chips": chips}))
        elif joker.name == "Steel Joker":
            xmult = 1.0 + (0.2 * steel_count)
            updated.append(_with_joker_metadata(joker, {"steel_tally": steel_count, "current_xmult": xmult, "xmult": xmult}))
        elif joker.name == "Driver's License":
            updated.append(_with_joker_metadata(joker, {"driver_tally": enhanced_count}))
        elif joker.name == "Erosion":
            updated.append(_with_joker_metadata(joker, {"current_mult": 4 * missing_cards}))
        else:
            updated.append(joker)
    return tuple(updated)


def _is_enhanced_playing_card(card: Card) -> bool:
    return _normalized(card.enhancement) not in {"", "base", "base card"}


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


def _ox_should_reset_money(state: GameState, hand_type: HandType) -> bool:
    if _effective_blind_name(state) != "The Ox":
        return False
    return hand_type.value == _most_played_hand_name(state)


def _most_played_hand_name(state: GameState) -> str:
    explicit = state.modifiers.get("most_played_poker_hand")
    if explicit:
        return str(explicit)
    hands = state.modifiers.get("hands", {})
    if not isinstance(hands, dict):
        return HandType.HIGH_CARD.value
    best_name = HandType.HIGH_CARD.value
    best_played = -1
    best_order = 100
    for name, raw in hands.items():
        if not isinstance(raw, dict):
            continue
        played = _int_value(raw.get("played"))
        order = _int_value(raw.get("order")) or 100
        if played > best_played or (played == best_played and order < best_order):
            best_name = str(name)
            best_played = played
            best_order = order
    return best_name


def _observatory_multiplier(state: GameState, hand_type: HandType) -> float:
    if "Observatory" not in state.vouchers:
        return 1.0
    matching_planets = sum(1 for consumable in state.consumables if PLANET_TO_HAND.get(consumable) == hand_type)
    return 1.5**matching_planets


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
        if _normalize_rank(rank) in {"2", "3", "4", "5"}:
            triggers += _effective_joker_count(jokers, "Hack")
        if hands_remaining == 1:
            triggers += _effective_joker_count(jokers, "Dusk")
        triggers += _effective_joker_count(jokers, "Seltzer")
        if index == first_scored_index:
            triggers += 2 * _effective_joker_count(jokers, "Hanging Chad")
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
    if card.debuffed:
        return False
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
    value = metadata.get("value")
    if isinstance(value, dict):
        value = dict(value)
        value["effect"] = f"+{RANK_CHIPS.get(_normalize_rank(card.rank), 0)} chips"
        metadata["value"] = value
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
        if joker.name == "Blueprint" and index + 1 < len(jokers) and is_blueprint_compatible(jokers[index + 1]):
            return resolve(index + 1, seen | {index})
        if joker.name == "Brainstorm" and index != 0 and is_blueprint_compatible(jokers[0]):
            return resolve(0, seen | {index})
        return index

    return tuple(resolve(index) for index in range(len(jokers)))


def _joker_is_disabled(joker: Joker) -> bool:
    state = joker.metadata.get("state")
    if isinstance(state, dict) and state.get("debuff"):
        return True
    return "all abilities are disabled" in _joker_effect_text(joker).lower()


def _jokers_after_crimson_heart_draw(
    state: GameState,
    jokers: tuple[Joker, ...],
    outcome_map: dict[str, object],
    *,
    next_phase: GamePhase,
) -> tuple[Joker, ...]:
    if _effective_blind_name(state) != "Crimson Heart":
        return jokers
    if next_phase not in {GamePhase.SELECTING_HAND, GamePhase.PLAYING_BLIND}:
        return jokers
    disabled_index = _outcome_int_or_none(outcome_map, "crimson_heart_disabled_joker_index")
    if disabled_index is None:
        return jokers
    return _jokers_with_only_disabled_index(jokers, disabled_index)


def _jokers_with_only_disabled_index(jokers: tuple[Joker, ...], disabled_index: int) -> tuple[Joker, ...]:
    if disabled_index < 0 or disabled_index >= len(jokers):
        raise ValueError(f"Crimson Heart disabled joker index is outside jokers: {disabled_index}")
    return tuple(_joker_with_disabled_state(joker, index == disabled_index) for index, joker in enumerate(jokers))


def _joker_with_disabled_state(joker: Joker, disabled: bool) -> Joker:
    metadata = _copy_metadata(joker.metadata)
    state_value = metadata.get("state")
    state_data = dict(state_value) if isinstance(state_value, dict) else {}
    if disabled:
        state_data["debuff"] = True
        _set_disabled_effect_text(metadata)
    else:
        state_data.pop("debuff", None)
        _restore_disabled_effect_text(metadata)

    if state_data:
        metadata["state"] = state_data
    elif "state" in metadata:
        metadata["state"] = []
    return replace(joker, metadata=metadata)


def _set_disabled_effect_text(metadata: dict[str, object]) -> None:
    value = metadata.get("value")
    if not isinstance(value, dict):
        return
    effect = value.get("effect")
    if isinstance(effect, str) and effect and "all abilities are disabled" not in effect.lower():
        metadata.setdefault("_crimson_heart_original_effect", effect)
    value["effect"] = "All abilities are disabled"


def _restore_disabled_effect_text(metadata: dict[str, object]) -> None:
    value = metadata.get("value")
    if not isinstance(value, dict):
        metadata.pop("_crimson_heart_original_effect", None)
        return
    effect = str(value.get("effect", ""))
    if "all abilities are disabled" not in effect.lower():
        metadata.pop("_crimson_heart_original_effect", None)
        return
    original = metadata.pop("_crimson_heart_original_effect", None)
    if isinstance(original, str) and original:
        value["effect"] = original
    else:
        value.pop("effect", None)


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


def _seltzer_remaining(joker: Joker) -> int:
    value = _joker_numeric_value(joker, keys=("current_remaining", "remaining", "uses", "extra"))
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    effect = _joker_effect_text(joker)
    match = re.search(r"next\s+([0-9]+)\s+hands?", effect, flags=re.IGNORECASE)
    return int(match.group(1)) if match else 10


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
        if joker.name != "Mail-In Rebate" or _joker_is_disabled(joker):
            continue
        for source in _metadata_sources(joker.metadata):
            for key in ("current_rank", "target_rank", "rank"):
                value = source.get(key)
                if isinstance(value, str) and value:
                    return _normalize_rank(value)
        return _mail_in_rebate_rank_from_text(_joker_effect_text(joker))
    return None


def _mail_in_rebate_rank_from_text(text: str) -> str | None:
    match = re.search(
        r"\bdiscarded\s+(Ace|King|Queen|Jack|Ten|Nine|Eight|Seven|Six|Five|Four|Three|Two|K|Q|J|10|T|[2-9])\b",
        text,
        flags=re.IGNORECASE,
    )
    return _normalize_rank(match.group(1)) if match else None


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


def _outcome_int_or_none(outcomes: dict[str, object], key: str) -> int | None:
    if key not in outcomes or outcomes[key] is None:
        return None
    try:
        return int(outcomes[key])
    except (TypeError, ValueError):
        return None


def _outcome_names(outcomes: dict[str, object], key: str) -> tuple[str, ...]:
    raw = outcomes.get(key, ())
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, list | tuple | set):
        return tuple(str(item) for item in raw)
    return ()


def _outcome_cards(outcomes: dict[str, object], key: str) -> tuple[Card, ...]:
    raw = outcomes.get(key, ())
    if isinstance(raw, Card):
        return (raw,)
    if isinstance(raw, list | tuple | set):
        cards: list[Card] = []
        for item in raw:
            if isinstance(item, Card):
                cards.append(item)
            elif isinstance(item, dict):
                cards.append(Card.from_mapping(item))
        return tuple(cards)
    return ()


def _float_value(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
