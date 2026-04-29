"""Basic rule bot with simple play/discard and shop discipline."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import ceil
from typing import Any

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import Card, GameState, Joker
from balatro_ai.bots.random_bot import RandomBot
from balatro_ai.rules.hand_evaluator import (
    HandType,
    RANK_VALUES,
    STRAIGHT_VALUES,
    best_play_from_hand,
    debuffed_suits_for_blind,
    evaluate_played_cards,
)


@dataclass(frozen=True, slots=True)
class _PlayCandidate:
    action: Action
    score: int


@dataclass(frozen=True, slots=True)
class _ShopPressure:
    target_score: float
    build_capacity: float
    ratio: float

    @property
    def danger(self) -> float:
        return max(0.0, min(2.0, self.ratio - 1.0))

    @property
    def safe_margin(self) -> float:
        return max(0.0, min(1.0, 1.0 - self.ratio))


@dataclass(slots=True)
class BasicStrategyBot:
    """A small step above immediate-score greed.

    The bot avoids random shop spending, plays hands that are good enough for
    the current blind, and uses discards when the best immediate hand is behind
    the score pace.
    """

    seed: int | None = None
    name: str = "basic_strategy_bot"
    _fallback: RandomBot = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._fallback = RandomBot(seed=self.seed)

    def choose_action(self, state: GameState) -> Action:
        for preferred_type in (
            ActionType.SELECT_BLIND,
            ActionType.CASH_OUT,
        ):
            action = _first_action_of_type(state, preferred_type)
            if action is not None:
                return action

        shop_action = _shop_action(state)
        if shop_action is not None:
            return shop_action

        end_shop = _first_action_of_type(state, ActionType.END_SHOP)
        if end_shop is not None:
            return end_shop

        pack_choice = _pack_choice_action(state)
        if pack_choice is not None:
            return pack_choice

        best_play = _best_play_action(state)
        if best_play is None:
            return self._fallback.choose_action(state)

        if _should_play_now(state, best_play):
            return best_play

        discard = _best_discard_action(state)
        return discard or best_play


def _first_action_of_type(state: GameState, action_type: ActionType) -> Action | None:
    for action in state.legal_actions:
        if action.action_type == action_type:
            return action
    return None


def _annotated_action(action: Action, *, reason: str) -> Action:
    return Action(
        action.action_type,
        card_indices=action.card_indices,
        target_id=action.target_id,
        amount=action.amount,
        metadata={**action.metadata, "reason": reason},
    )


def _pack_choice_action(state: GameState) -> Action | None:
    pack_cards = state.modifiers.get("pack_cards", ())
    best_action: Action | None = None
    best_value = 0.0

    for action in state.legal_actions:
        if action.action_type != ActionType.CHOOSE_PACK_CARD or action.target_id == "skip":
            continue
        index = int(action.metadata.get("index", action.amount or 0))
        if index >= len(pack_cards):
            continue
        value = _pack_card_value(state, pack_cards[index])
        if value > best_value:
            best_action = action
            best_value = value

    if best_action is not None and best_value >= 20:
        return _annotated_action(best_action, reason=f"pack_pick value={best_value:.1f}")

    for action in state.legal_actions:
        if action.action_type == ActionType.CHOOSE_PACK_CARD and action.target_id == "skip":
            return action
    return None


def _shop_action(state: GameState) -> Action | None:
    pressure = _shop_pressure(state)
    replacement = _replacement_sell_action(state, pressure)
    if replacement is not None:
        return replacement

    best_action: Action | None = None
    best_value = 0.0

    for action in state.legal_actions:
        value = _shop_action_value(state, action, pressure)
        if value > best_value:
            best_action = action
            best_value = value

    if best_action is not None and best_value >= _shop_buy_threshold(state, pressure):
        return _annotated_action(
            best_action,
            reason=(
                f"shop_value value={best_value:.1f} pressure={pressure.ratio:.2f} "
                f"target={pressure.target_score:.0f} capacity={pressure.build_capacity:.0f}"
            ),
        )
    return None


def _replacement_sell_action(state: GameState, pressure: _ShopPressure) -> Action | None:
    if len(state.jokers) < 5:
        return None

    sell_actions = {
        int(action.metadata.get("index", action.amount or 0)): action
        for action in state.legal_actions
        if action.action_type == ActionType.SELL
    }
    if not sell_actions:
        return None

    weakest_index, weakest_joker = min(
        enumerate(state.jokers),
        key=lambda item: _owned_joker_value(state, item[1], remove_index=item[0]),
    )
    weakest_value = _owned_joker_value(state, weakest_joker, remove_index=weakest_index)
    sell_value = weakest_joker.sell_value or 0

    shop_cards = state.modifiers.get("shop_cards", ())
    best_upgrade = 0.0
    best_label = ""
    for card in shop_cards:
        if not _is_joker_card(card):
            continue
        cost = _card_cost(card)
        if state.money + sell_value < cost:
            continue
        candidate = _joker_from_shop_card(card)
        candidate_value = _candidate_joker_value_for_replacement(state, candidate)
        upgrade = candidate_value - weakest_value - max(0, cost - sell_value) * _replacement_cost_weight(pressure)
        if upgrade > best_upgrade:
            best_upgrade = upgrade
            best_label = candidate.name

    if best_upgrade >= _replacement_upgrade_threshold(pressure) and weakest_index in sell_actions:
        return _annotated_action(
            sell_actions[weakest_index],
            reason=(
                f"replace {weakest_joker.name} value={weakest_value:.1f} "
                f"with {best_label} upgrade={best_upgrade:.1f} pressure={pressure.ratio:.2f}"
            ),
        )
    return None


def _shop_action_value(state: GameState, action: Action, pressure: _ShopPressure) -> float:
    kind = str(action.metadata.get("kind", ""))
    if action.action_type == ActionType.BUY and kind == "card":
        shop_cards = state.modifiers.get("shop_cards", ())
        index = int(action.metadata.get("index", action.amount or 0))
        if index >= len(shop_cards):
            return 0.0
        value = _shop_card_value(state, shop_cards[index])
        if _is_joker_card(shop_cards[index]):
            value += pressure.danger * 14
        return value - _cost_penalty(state, shop_cards[index], pressure)
    if action.action_type == ActionType.BUY and kind == "voucher":
        vouchers = state.modifiers.get("voucher_cards", ())
        index = int(action.metadata.get("index", action.amount or 0))
        if index >= len(vouchers):
            return 0.0
        return _voucher_value(state, vouchers[index]) - _cost_penalty(state, vouchers[index], pressure)
    if action.action_type == ActionType.OPEN_PACK:
        packs = state.modifiers.get("booster_packs", ())
        index = int(action.metadata.get("index", action.amount or 0))
        if index >= len(packs):
            return 0.0
        if state.money - _card_cost(packs[index]) < 4 and pressure.ratio < 1.15:
            return 0.0
        return _pack_value(state, packs[index]) + pressure.danger * 16 - _cost_penalty(state, packs[index], pressure)
    if action.action_type == ActionType.REROLL:
        if state.money < 9:
            return 0.0
        if pressure.ratio < 0.95 and state.money < 14:
            return 0.0
        if len(state.jokers) >= 5 and pressure.ratio < 1.1:
            return 0.0
        pressure_bonus = max(0, 4 - len(state.jokers)) * 7
        pressure_bonus += pressure.danger * 22
        return 24 + pressure_bonus - _money_after_spend_penalty(state, 5, pressure)
    return 0.0


def _shop_card_value(state: GameState, card: object) -> float:
    if _is_joker_card(card):
        return _joker_card_value(state, card)
    if _is_planet_card(card):
        if not _has_consumable_room(state):
            return 0.0
        return _planet_card_value(state, card)
    if _is_tarot_card(card):
        if not _has_consumable_room(state):
            return 0.0
        return _tarot_card_value(state, card)
    if _is_playing_card(card):
        return _playing_card_shop_value(state, card)
    return 0.0


def _pack_card_value(state: GameState, card: object) -> float:
    if _is_joker_card(card):
        return _joker_card_value(state, card) + 15
    if _is_planet_card(card):
        if not _has_consumable_room(state):
            return 0.0
        return _planet_card_value(state, card) + 10
    if _is_tarot_card(card):
        if not _has_consumable_room(state):
            return 0.0
        return _tarot_card_value(state, card) + 10
    if _is_playing_card(card):
        return _playing_card_shop_value(state, card)
    return 0.0


def _shop_buy_threshold(state: GameState, pressure: _ShopPressure) -> float:
    if len(state.jokers) == 0:
        base = 18.0
    elif state.ante <= 2 and len(state.jokers) < 3:
        base = 24.0
    elif state.money >= 25:
        base = 22.0
    else:
        base = 30.0
    return max(12.0, base - pressure.danger * 14 + pressure.safe_margin * 14)


def _replacement_upgrade_threshold(pressure: _ShopPressure) -> float:
    return max(16.0, 28.0 - pressure.danger * 10 + pressure.safe_margin * 8)


def _replacement_cost_weight(pressure: _ShopPressure) -> float:
    return max(1.3, 2.5 - pressure.danger * 0.8 + pressure.safe_margin * 0.5)


def _shop_pressure(state: GameState) -> _ShopPressure:
    target = _estimated_next_required_score(state)
    current_score = _sample_build_score(state, state.jokers)
    hands = float(state.hands_remaining or 4)
    preferred_hands = max(2.0, min(4.0, hands) - 0.5)
    capacity = max(1.0, current_score * preferred_hands * 0.85)
    ratio = max(target / capacity, _early_build_pressure_floor(state))
    return _ShopPressure(target_score=target, build_capacity=capacity, ratio=ratio)


def _early_build_pressure_floor(state: GameState) -> float:
    joker_count = len(state.jokers)
    if state.ante <= 1 and joker_count == 0:
        return 1.25
    if state.ante <= 2 and joker_count < 2:
        return 1.15
    if state.ante <= 3 and joker_count < 3:
        return 1.05
    return 0.0


def _estimated_next_required_score(state: GameState) -> float:
    ante = max(1, state.ante)
    small = ANTE_SMALL_BLIND_SCORES.get(ante, _extrapolated_small_blind_score(ante))
    next_small = ANTE_SMALL_BLIND_SCORES.get(ante + 1, _extrapolated_small_blind_score(ante + 1))

    if state.blind == "Small Blind":
        return small * 1.5
    if state.blind == "Big Blind":
        return small * 2.0
    if state.required_score > 0 and state.blind:
        return max(next_small, state.required_score * 1.25)
    if state.required_score > 0:
        return state.required_score * 1.5
    return small


def _extrapolated_small_blind_score(ante: int) -> float:
    if ante <= 8:
        return ANTE_SMALL_BLIND_SCORES.get(ante, 300)
    return 50000 * (1.6 ** (ante - 8))


def _has_consumable_room(state: GameState) -> bool:
    return len(state.consumables) < 2


def _is_joker_card(card: object) -> bool:
    if not isinstance(card, dict):
        return False
    label = str(card.get("label", card.get("name", card.get("key", ""))))
    key = str(card.get("key", ""))
    card_set = str(card.get("set", "")).upper()
    return card_set == "JOKER" or "joker" in label.lower() or key.startswith("j_")


def _is_planet_card(card: object) -> bool:
    return _card_set(card) == "PLANET" or _card_label(card) in PLANET_TO_HAND


def _is_tarot_card(card: object) -> bool:
    return _card_set(card) == "TAROT"


def _is_playing_card(card: object) -> bool:
    if not isinstance(card, dict):
        return False
    card_set = _card_set(card)
    return card_set in {"DEFAULT", "ENHANCED", "PLAYING_CARD"} or (
        _card_rank(card) != "" and _card_suit(card) != ""
    )


def _joker_card_value(state: GameState, card: object) -> float:
    if len(state.jokers) >= 5:
        return 0.0

    joker = _joker_from_shop_card(card)
    name = joker.name
    sample_gain = _sample_score_gain_for_joker(state, joker)
    value = sample_gain * 0.08
    value += max(0, 5 - len(state.jokers)) * 6
    value += max(0, 4 - state.ante) * _early_power_bonus(name)
    value += _joker_heuristic_value(state, joker)
    value += _edition_bonus(joker.edition)

    if any(existing.name == name for existing in state.jokers):
        value -= 18
    if len(state.jokers) == 0 and value < 35:
        value += 10
    return value


def _candidate_joker_value_for_replacement(state: GameState, joker: Joker) -> float:
    sample_gain = _sample_score_gain_for_joker(state, joker)
    value = sample_gain * 0.08
    value += _joker_heuristic_value(state, joker)
    value += _edition_bonus(joker.edition)
    if any(existing.name == joker.name for existing in state.jokers):
        value -= 18
    return value


def _owned_joker_value(state: GameState, joker: Joker, *, remove_index: int) -> float:
    without = tuple(existing for index, existing in enumerate(state.jokers) if index != remove_index)
    score_loss = max(0.0, _sample_build_score(state, state.jokers) - _sample_build_score(state, without))
    value = score_loss * 0.08
    value += _joker_heuristic_value(state, joker) * 0.75
    value += _edition_bonus(joker.edition)
    value += (joker.sell_value or 0) * 1.5
    if joker.name in LOW_PRIORITY_JOKERS:
        value -= 20
    return value


def _sample_score_gain_for_joker(state: GameState, joker: Joker) -> float:
    current = _sample_build_score(state, state.jokers)
    with_candidate = _sample_build_score(state, (*state.jokers, joker))
    return max(0.0, with_candidate - current)


def _sample_build_score(state: GameState, jokers: tuple[Joker, ...]) -> float:
    scores = []
    for sample in SAMPLE_HANDS:
        scores.append(
            evaluate_played_cards(
                sample.cards,
                state.hand_levels,
                debuffed_suits=debuffed_suits_for_blind(state.blind),
                blind_name=state.blind,
                jokers=jokers,
                discards_remaining=state.discards_remaining,
                hands_remaining=max(1, state.hands_remaining),
                held_cards=sample.held_cards,
                deck_size=max(30, state.deck_size),
                money=state.money,
            ).score
        )
    best = max(scores, default=0)
    average_top = sum(sorted(scores, reverse=True)[:3]) / 3
    return (best * 0.65) + (average_top * 0.35)


def _joker_heuristic_value(state: GameState, joker: Joker) -> float:
    name = joker.name
    value = 0.0
    value += JOKER_BASE_VALUES.get(name, 0)
    value += JOKER_SCALING_VALUES.get(name, 0) * (1 + max(0, 4 - state.ante) * 0.15)
    value += JOKER_ECONOMY_VALUES.get(name, 0)
    value += _build_synergy_value(state, name)
    if name in GLASS_CANNON_JOKERS and state.ante <= 2:
        value += 8
    if name in LOW_PRIORITY_JOKERS and len(state.jokers) >= 2:
        value -= 16
    return value


def _build_synergy_value(state: GameState, joker_name: str) -> float:
    preferred = _preferred_hand_type(state)
    if preferred is None:
        return 0.0
    if preferred in JOKER_HAND_SYNERGY.get(joker_name, ()):
        return 18.0
    if preferred == HandType.PAIR and joker_name in {"Spare Trousers", "Mad Joker", "Clever Joker"}:
        return 10.0
    if preferred == HandType.FLUSH and joker_name in {"Four Fingers", "Smeared Joker"}:
        return 16.0
    return 0.0


def _planet_card_value(state: GameState, card: object) -> float:
    hand_type = PLANET_TO_HAND.get(_card_label(card))
    if hand_type is None:
        return 0.0

    preferred = _preferred_hand_type(state)
    current_level = state.hand_levels.get(hand_type.value, 1)
    value = 14 + min(current_level, 5) * 2
    if hand_type == preferred:
        value += 24
    elif hand_type in _flexible_hand_types(state):
        value += 8
    else:
        value -= 8
    return value


def _tarot_card_value(state: GameState, card: object) -> float:
    name = _card_label(card)
    value = TAROT_VALUES.get(name, 18)
    preferred = _preferred_hand_type(state)
    if preferred == HandType.FLUSH and name in {"The Sun", "The Moon", "The Star", "The World"}:
        value += 14
    if preferred in {HandType.PAIR, HandType.TWO_PAIR, HandType.THREE_OF_A_KIND, HandType.FULL_HOUSE} and name in {
        "Strength",
        "Death",
        "The Hanged Man",
    }:
        value += 12
    if state.money < 8 and name in {"Temperance", "The Hermit"}:
        value += 12
    return value


def _playing_card_shop_value(state: GameState, card: object) -> float:
    rank = _card_rank(card)
    suit = _card_suit(card)
    value = RANK_VALUES.get(rank, 5)
    if _preferred_hand_type(state) == HandType.FLUSH:
        dominant_suit = _dominant_suit(state)
        if dominant_suit and suit == dominant_suit:
            value += 18
    enhancement = str(_card_modifier(card).get("enhancement", card.get("enhancement", ""))) if isinstance(card, dict) else ""
    if enhancement:
        value += 10
    return float(value)


def _voucher_value(state: GameState, voucher: object) -> float:
    name = _card_label(voucher)
    value = VOUCHER_VALUES.get(name, 22)
    if state.ante <= 2:
        value += 8
    if state.money >= 20:
        value += 8
    if any(existing == name for existing in state.vouchers):
        value = 0
    return float(value)


def _pack_value(state: GameState, pack: object) -> float:
    name = _card_label(pack).lower()
    if "buffoon" in name:
        value = 42 if len(state.jokers) < 5 else 8
    elif "celestial" in name:
        value = 28
    elif "arcana" in name:
        value = 26
    elif "standard" in name:
        value = 18
    elif "spectral" in name:
        value = 20 if state.ante <= 2 else 14
    else:
        value = 16
    if "mega" in name or "jumbo" in name:
        value += 8
    return float(value)


def _cost_penalty(state: GameState, card: object, pressure: _ShopPressure) -> float:
    cost = _card_cost(card)
    cost_weight = max(1.6, 3.0 - pressure.danger * 0.9 + pressure.safe_margin * 1.0)
    return cost * cost_weight + _money_after_spend_penalty(state, cost, pressure)


def _money_after_spend_penalty(state: GameState, cost: int, pressure: _ShopPressure) -> float:
    after = state.money - cost
    if after < 0:
        return 1000.0
    penalty = 0.0
    before_interest = min(state.money, 25) // 5
    after_interest = min(after, 25) // 5
    interest_weight = max(1.0, (5 if state.ante >= 2 else 3) - pressure.danger * 3 + pressure.safe_margin * 4)
    penalty += max(0, before_interest - after_interest) * interest_weight
    if after < 4 and state.ante >= 2:
        penalty += max(2.0, 10 - pressure.danger * 6 + pressure.safe_margin * 4)
    return penalty


def _best_play_action(state: GameState) -> Action | None:
    candidates = _play_candidates(state)
    if not candidates:
        return None

    remaining_score = max(0, state.required_score - state.current_score)
    if remaining_score > 0:
        winning_candidates = [candidate for candidate in candidates if candidate.score >= remaining_score]
        if winning_candidates:
            return min(winning_candidates, key=_minimum_sufficient_play_key).action

        feasible_candidates = [candidate for candidate in candidates if candidate.score > 0]
        if feasible_candidates:
            return min(feasible_candidates, key=lambda candidate: _hands_to_clear_key(candidate, remaining_score)).action

    return max(candidates, key=_maximum_play_key).action


def _play_candidates(state: GameState) -> list[_PlayCandidate]:
    play_actions = [
        action
        for action in state.legal_actions
        if action.action_type == ActionType.PLAY_HAND and action.card_indices
    ]
    return [_PlayCandidate(action=action, score=_score_play_action(state, action)) for action in play_actions]


def _minimum_sufficient_play_key(candidate: _PlayCandidate) -> tuple[int, int, int]:
    return (candidate.score, len(candidate.action.card_indices), _action_index_sum(candidate.action))


def _maximum_play_key(candidate: _PlayCandidate) -> tuple[int, int]:
    return (candidate.score, -len(candidate.action.card_indices))


def _hands_to_clear_key(candidate: _PlayCandidate, remaining_score: int) -> tuple[int, int, int, int]:
    estimated_hands = ceil(remaining_score / max(1, candidate.score))
    return (estimated_hands, -candidate.score, len(candidate.action.card_indices), _action_index_sum(candidate.action))


def _action_index_sum(action: Action) -> int:
    return sum(action.card_indices)


def _score_play_action(state: GameState, action: Action) -> int:
    cards = tuple(state.hand[index] for index in action.card_indices)
    held_cards = tuple(card for index, card in enumerate(state.hand) if index not in action.card_indices)
    return evaluate_played_cards(
        cards,
        state.hand_levels,
        debuffed_suits=debuffed_suits_for_blind(state.blind),
        blind_name=state.blind,
        jokers=state.jokers,
        discards_remaining=state.discards_remaining,
        hands_remaining=state.hands_remaining,
        held_cards=held_cards,
        deck_size=state.deck_size,
        money=state.money,
    ).score


def _should_play_now(state: GameState, action: Action) -> bool:
    score = _score_play_action(state, action)
    remaining_score = max(0, state.required_score - state.current_score)
    if remaining_score == 0:
        return True
    if score >= remaining_score:
        return True
    if state.hands_remaining <= 1 or state.discards_remaining <= 0:
        return True

    estimated_hands_to_clear = ceil(remaining_score / max(1, score))
    if estimated_hands_to_clear < state.hands_remaining:
        return True

    best_discard = _best_discard_action(state)
    return best_discard is None


def _best_discard_action(state: GameState) -> Action | None:
    discard_actions = [
        action
        for action in state.legal_actions
        if action.action_type == ActionType.DISCARD and action.card_indices
    ]
    if not discard_actions:
        return None

    keep_scores = _card_keep_scores(state.hand)
    protected_count = max(2, min(4, len(state.hand) // 2))
    protected = {
        index
        for index, _ in sorted(
            enumerate(keep_scores),
            key=lambda item: (item[1], RANK_VALUES[state.hand[item[0]].rank]),
            reverse=True,
        )[:protected_count]
    }

    def discard_score(action: Action) -> tuple[float, float, int]:
        protected_penalty = sum(1 for index in action.card_indices if index in protected) * 1000
        desirability = sum(100 - keep_scores[index] for index in action.card_indices)
        kept_cards = tuple(card for index, card in enumerate(state.hand) if index not in action.card_indices)
        kept_potential = _kept_hand_potential(state, kept_cards)
        return (desirability + kept_potential - protected_penalty, kept_potential, len(action.card_indices))

    best = max(discard_actions, key=discard_score)
    return best if discard_score(best)[0] > -500 else None


def _kept_hand_potential(state: GameState, kept_cards: tuple[Card, ...]) -> float:
    if not kept_cards:
        return 0.0

    rank_counts = Counter(card.rank for card in kept_cards)
    suit_counts = Counter(card.suit for card in kept_cards)
    pair_bonus = sum(count * 20 for count in rank_counts.values() if count >= 2)
    flush_draw_bonus = max(suit_counts.values(), default=0) * 8
    straight_draw_bonus = _straight_draw_potential(kept_cards) * 6
    high_card_bonus = sum(RANK_VALUES[card.rank] for card in kept_cards) / max(1, len(kept_cards))

    immediate_score = best_play_from_hand(
        kept_cards,
        state.hand_levels,
        debuffed_suits=debuffed_suits_for_blind(state.blind),
        blind_name=state.blind,
        jokers=state.jokers,
        discards_remaining=state.discards_remaining,
        hands_remaining=state.hands_remaining,
        deck_size=state.deck_size,
        money=state.money,
    ).score

    return pair_bonus + flush_draw_bonus + straight_draw_bonus + high_card_bonus + (immediate_score * 0.03)


def _straight_draw_potential(cards: tuple[Card, ...]) -> int:
    values = {STRAIGHT_VALUES[card.rank] for card in cards}
    if "A" in {card.rank for card in cards}:
        values.add(1)
    best = 0
    for start in range(1, 11):
        best = max(best, sum(1 for value in range(start, start + 5) if value in values))
    return best


def _card_keep_scores(hand: tuple[Card, ...]) -> tuple[float, ...]:
    rank_counts = Counter(card.rank for card in hand)
    suit_counts = Counter(card.suit for card in hand)
    straight_values = [STRAIGHT_VALUES[card.rank] for card in hand]

    scores: list[float] = []
    for card in hand:
        straight_value = STRAIGHT_VALUES[card.rank]
        nearby = sum(
            1
            for value in straight_values
            if value != straight_value and abs(value - straight_value) <= 4
        )
        ace_low_nearby = 0
        if card.rank == "A":
            ace_low_nearby = sum(1 for value in straight_values if value in {2, 3, 4, 5})

        rank_count = rank_counts[card.rank]
        score = RANK_VALUES[card.rank]
        score += suit_counts[card.suit] * 8
        score += max(nearby, ace_low_nearby) * 6
        if rank_count >= 2:
            score += rank_count * 100
        scores.append(float(score))

    return tuple(scores)


def _card_label(card: object) -> str:
    if not isinstance(card, dict):
        return str(card)
    return str(card.get("label", card.get("name", card.get("key", ""))))


def _card_set(card: object) -> str:
    if not isinstance(card, dict):
        return ""
    return str(card.get("set", "")).upper()


def _card_cost(card: object) -> int:
    if not isinstance(card, dict):
        return 0
    cost = card.get("cost", {})
    if isinstance(cost, dict):
        return int(cost.get("buy", cost.get("cost", 0)) or 0)
    return int(cost or 0)


def _card_modifier(card: object) -> dict[str, Any]:
    if not isinstance(card, dict) or not isinstance(card.get("modifier"), dict):
        return {}
    return card["modifier"]


def _card_value(card: object) -> dict[str, Any]:
    if not isinstance(card, dict) or not isinstance(card.get("value"), dict):
        return {}
    return card["value"]


def _card_rank(card: object) -> str:
    if not isinstance(card, dict):
        return ""
    value = _card_value(card)
    rank = str(card.get("rank", value.get("rank", "")))
    return "T" if rank == "10" else rank


def _card_suit(card: object) -> str:
    if not isinstance(card, dict):
        return ""
    value = _card_value(card)
    suit = str(card.get("suit", value.get("suit", "")))
    return {"Spades": "S", "Spade": "S", "Hearts": "H", "Heart": "H", "Clubs": "C", "Club": "C", "Diamonds": "D", "Diamond": "D"}.get(suit, suit)


def _joker_from_shop_card(card: object) -> Joker:
    if isinstance(card, dict):
        return Joker.from_mapping(card)
    return Joker(str(card))


def _edition_bonus(edition: str | None) -> float:
    if edition is None:
        return 0.0
    text = edition.lower()
    if "negative" in text:
        return 60.0
    if "polychrome" in text:
        return 45.0
    if "holo" in text or "holographic" in text:
        return 24.0
    if "foil" in text:
        return 18.0
    return 0.0


def _early_power_bonus(name: str) -> float:
    if name in EARLY_POWER_JOKERS:
        return 5.0
    if name in JOKER_SCALING_VALUES:
        return 2.0
    return 0.0


def _preferred_hand_type(state: GameState) -> HandType | None:
    joker_votes: Counter[HandType] = Counter()
    for joker in state.jokers:
        for hand_type in JOKER_HAND_SYNERGY.get(joker.name, ()):
            joker_votes[hand_type] += 2

    level_votes = Counter(
        {
            hand_type: max(0, state.hand_levels.get(hand_type.value, 1) - 1)
            for hand_type in HandType
        }
    )
    combined = joker_votes + level_votes
    if combined:
        best, score = combined.most_common(1)[0]
        if score > 0:
            return best
    if any(joker.name in {"Smeared Joker", "Four Fingers", "The Tribe", "Droll Joker"} for joker in state.jokers):
        return HandType.FLUSH
    if any(joker.name in {"Spare Trousers", "Mad Joker", "Clever Joker"} for joker in state.jokers):
        return HandType.TWO_PAIR
    if state.ante <= 2:
        return HandType.PAIR
    return None


def _flexible_hand_types(state: GameState) -> set[HandType]:
    preferred = _preferred_hand_type(state)
    if preferred in {HandType.PAIR, HandType.TWO_PAIR, HandType.THREE_OF_A_KIND, HandType.FULL_HOUSE}:
        return {HandType.PAIR, HandType.TWO_PAIR, HandType.THREE_OF_A_KIND, HandType.FULL_HOUSE}
    if preferred in {HandType.STRAIGHT, HandType.STRAIGHT_FLUSH}:
        return {HandType.STRAIGHT, HandType.STRAIGHT_FLUSH}
    if preferred in {HandType.FLUSH, HandType.STRAIGHT_FLUSH, HandType.FLUSH_HOUSE, HandType.FLUSH_FIVE}:
        return {HandType.FLUSH, HandType.STRAIGHT_FLUSH, HandType.FLUSH_HOUSE, HandType.FLUSH_FIVE}
    return {HandType.PAIR, HandType.TWO_PAIR, HandType.FLUSH, HandType.STRAIGHT}


def _dominant_suit(state: GameState) -> str | None:
    cards = state.hand or state.known_deck
    if not cards:
        return None
    return Counter(card.suit for card in cards).most_common(1)[0][0]


@dataclass(frozen=True, slots=True)
class _SampleHand:
    cards: tuple[Card, ...]
    held_cards: tuple[Card, ...] = ()


SAMPLE_HANDS = (
    _SampleHand((Card("A", "S"),), (Card("K", "H"), Card("Q", "C"), Card("7", "D"))),
    _SampleHand((Card("A", "S"), Card("A", "H")), (Card("K", "D"), Card("4", "C"))),
    _SampleHand((Card("K", "S"), Card("K", "H"), Card("7", "D"), Card("7", "C"))),
    _SampleHand((Card("Q", "S"), Card("Q", "H"), Card("Q", "D"))),
    _SampleHand((Card("T", "S"), Card("9", "H"), Card("8", "D"), Card("7", "C"), Card("6", "S"))),
    _SampleHand((Card("A", "H"), Card("K", "H"), Card("Q", "H"), Card("7", "H"), Card("2", "H"))),
    _SampleHand((Card("J", "S"), Card("J", "H"), Card("J", "D"), Card("4", "S"), Card("4", "C"))),
)


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


JOKER_HAND_SYNERGY = {
    "Jolly Joker": (HandType.PAIR,),
    "Sly Joker": (HandType.PAIR,),
    "Zany Joker": (HandType.THREE_OF_A_KIND, HandType.FULL_HOUSE),
    "Wily Joker": (HandType.THREE_OF_A_KIND, HandType.FULL_HOUSE),
    "Mad Joker": (HandType.TWO_PAIR, HandType.FULL_HOUSE),
    "Clever Joker": (HandType.TWO_PAIR, HandType.FULL_HOUSE),
    "Crazy Joker": (HandType.STRAIGHT, HandType.STRAIGHT_FLUSH),
    "Devious Joker": (HandType.STRAIGHT, HandType.STRAIGHT_FLUSH),
    "Droll Joker": (HandType.FLUSH, HandType.STRAIGHT_FLUSH, HandType.FLUSH_HOUSE, HandType.FLUSH_FIVE),
    "Crafty Joker": (HandType.FLUSH, HandType.STRAIGHT_FLUSH, HandType.FLUSH_HOUSE, HandType.FLUSH_FIVE),
    "Spare Trousers": (HandType.TWO_PAIR, HandType.FULL_HOUSE),
    "Runner": (HandType.STRAIGHT, HandType.STRAIGHT_FLUSH),
    "The Duo": (HandType.PAIR,),
    "The Trio": (HandType.THREE_OF_A_KIND,),
    "The Family": (HandType.FOUR_OF_A_KIND,),
    "The Order": (HandType.STRAIGHT, HandType.STRAIGHT_FLUSH),
    "The Tribe": (HandType.FLUSH, HandType.STRAIGHT_FLUSH, HandType.FLUSH_HOUSE, HandType.FLUSH_FIVE),
}


EARLY_POWER_JOKERS = {
    "Joker",
    "Jolly Joker",
    "Zany Joker",
    "Mad Joker",
    "Crazy Joker",
    "Droll Joker",
    "Sly Joker",
    "Wily Joker",
    "Clever Joker",
    "Devious Joker",
    "Crafty Joker",
    "Greedy Joker",
    "Lusty Joker",
    "Wrathful Joker",
    "Gluttonous Joker",
    "Half Joker",
    "Mystic Summit",
    "Misprint",
    "Gros Michel",
    "Popcorn",
    "Ice Cream",
    "Even Steven",
    "Odd Todd",
    "Scary Face",
    "Abstract Joker",
}


JOKER_BASE_VALUES = {
    "Blueprint": 75,
    "Brainstorm": 72,
    "Cavendish": 58,
    "Gros Michel": 42,
    "Popcorn": 34,
    "Ice Cream": 38,
    "Misprint": 32,
    "Abstract Joker": 28,
    "Half Joker": 24,
    "Mystic Summit": 22,
    "Photograph": 32,
    "Hanging Chad": 36,
    "Sock and Buskin": 34,
    "Hack": 28,
    "Dusk": 30,
    "Seltzer": 26,
    "Seeing Double": 34,
    "Flower Pot": 26,
    "Ancient Joker": 24,
    "The Idol": 24,
    "Baseball Card": 30,
    "Steel Joker": 28,
    "Stone Joker": 22,
    "Driver's License": 32,
    "Joker Stencil": 35,
    "Blackboard": 30,
    "Baron": 36,
}


JOKER_SCALING_VALUES = {
    "Green Joker": 34,
    "Ride the Bus": 34,
    "Supernova": 30,
    "Square Joker": 28,
    "Runner": 28,
    "Spare Trousers": 38,
    "Hologram": 40,
    "Constellation": 38,
    "Flash Card": 30,
    "Red Card": 28,
    "Castle": 30,
    "Erosion": 24,
    "Wee Joker": 34,
    "Lucky Cat": 34,
    "Glass Joker": 34,
    "Campfire": 32,
    "Throwback": 24,
    "Obelisk": 18,
}


JOKER_ECONOMY_VALUES = {
    "Golden Joker": 28,
    "Rocket": 32,
    "Cloud 9": 18,
    "Business Card": 18,
    "Reserved Parking": 16,
    "Delayed Gratification": 14,
    "To the Moon": 22,
    "Mail-In Rebate": 18,
    "Golden Ticket": 20,
    "Faceless Joker": 12,
    "Egg": 14,
    "Gift Card": 18,
    "Trading Card": 20,
    "Satellite": 16,
}


LOW_PRIORITY_JOKERS = {
    "Credit Card",
    "Loyalty Card",
    "Seance",
    "Superposition",
    "To Do List",
    "Matador",
    "Hallucination",
}


GLASS_CANNON_JOKERS = {
    "Popcorn",
    "Ice Cream",
    "Seltzer",
    "Gros Michel",
    "Ramen",
}


TAROT_VALUES = {
    "The Fool": 18,
    "The Magician": 24,
    "The High Priestess": 22,
    "The Empress": 22,
    "The Emperor": 20,
    "The Hierophant": 18,
    "The Lovers": 20,
    "The Chariot": 24,
    "Justice": 22,
    "The Hermit": 30,
    "The Wheel of Fortune": 16,
    "Strength": 28,
    "The Hanged Man": 28,
    "Death": 34,
    "Temperance": 28,
    "The Devil": 20,
    "The Tower": 18,
    "The Star": 22,
    "The Moon": 22,
    "The Sun": 22,
    "Judgement": 32,
    "The World": 22,
}


VOUCHER_VALUES = {
    "Overstock": 34,
    "Overstock Plus": 42,
    "Clearance Sale": 38,
    "Liquidation": 46,
    "Hone": 30,
    "Glow Up": 40,
    "Reroll Surplus": 32,
    "Reroll Glut": 40,
    "Crystal Ball": 26,
    "Omen Globe": 34,
    "Telescope": 26,
    "Observatory": 36,
    "Grabber": 28,
    "Nacho Tong": 36,
    "Wasteful": 22,
    "Recyclomancy": 30,
    "Paint Brush": 34,
    "Palette": 44,
    "Seed Money": 38,
    "Money Tree": 46,
    "Hieroglyph": 22,
    "Petroglyph": 28,
    "Director's Cut": 28,
    "Retcon": 36,
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
