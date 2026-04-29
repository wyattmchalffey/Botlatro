"""Basic rule bot with simple play/discard and shop discipline."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
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
    hand_type: HandType
    scoring_card_indices: tuple[int, ...]
    cycle_value: float
    cycle_count: int


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


@dataclass(frozen=True, slots=True)
class _BuildProfile:
    preferred_hand: HandType | None
    archetype: str
    has_chips: bool
    has_mult: bool
    has_xmult: bool
    has_scaling: bool
    has_economy: bool
    open_joker_slots: int
    money: int
    ante: int

    @property
    def missing_roles(self) -> tuple[str, ...]:
        roles: list[str] = []
        if not self.has_chips:
            roles.append("chips")
        if not self.has_mult:
            roles.append("mult")
        if not self.has_xmult:
            roles.append("xmult")
        if not self.has_scaling:
            roles.append("scaling")
        if not self.has_economy:
            roles.append("economy")
        return tuple(roles)

    @property
    def rich(self) -> bool:
        return self.money >= 50

    @property
    def late(self) -> bool:
        return self.ante >= 5


@dataclass(frozen=True, slots=True)
class _ShopContext:
    rerolls_in_shop: int = 0
    packs_opened_in_shop: int = 0


DISCARD_DETAIL_LIMIT = 28


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
    _shop_key: tuple[int | None, int, str, int] | None = field(default=None, init=False, repr=False)
    _rerolls_in_shop: int = field(default=0, init=False, repr=False)
    _packs_opened_in_shop: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._fallback = RandomBot(seed=self.seed)

    def choose_action(self, state: GameState) -> Action:
        self._sync_shop_memory(state)

        for preferred_type in (
            ActionType.SELECT_BLIND,
            ActionType.CASH_OUT,
        ):
            action = _first_action_of_type(state, preferred_type)
            if action is not None:
                return action

        shop_action = _shop_action(state, self._shop_context())
        if shop_action is not None:
            self._record_shop_action(state, shop_action)
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

        return _tactical_blind_action(state, best_play)

    def _shop_context(self) -> _ShopContext:
        return _ShopContext(
            rerolls_in_shop=self._rerolls_in_shop,
            packs_opened_in_shop=self._packs_opened_in_shop,
        )

    def _sync_shop_memory(self, state: GameState) -> None:
        if not (_has_shop_decision_surface(state) or state.modifiers.get("pack_cards")):
            return

        key = _shop_memory_key(state)
        if key != self._shop_key:
            self._shop_key = key
            self._rerolls_in_shop = 0
            self._packs_opened_in_shop = 0

    def _record_shop_action(self, state: GameState, action: Action) -> None:
        if action.action_type not in {ActionType.REROLL, ActionType.OPEN_PACK}:
            return

        if action.action_type == ActionType.REROLL:
            self._rerolls_in_shop += 1
        elif action.action_type == ActionType.OPEN_PACK:
            self._packs_opened_in_shop += 1


def _first_action_of_type(state: GameState, action_type: ActionType) -> Action | None:
    for action in state.legal_actions:
        if action.action_type == action_type:
            return action
    return None


def _annotated_action(action: Action, *, reason: str, audit: dict[str, Any] | None = None) -> Action:
    metadata = {**action.metadata, "reason": reason}
    if audit is not None:
        metadata["shop_audit"] = audit
    return Action(
        action.action_type,
        card_indices=action.card_indices,
        target_id=action.target_id,
        amount=action.amount,
        metadata=metadata,
    )


def _shop_memory_key(state: GameState) -> tuple[int | None, int, str, int]:
    return (state.seed, state.ante, state.blind, state.required_score)


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
        if _pack_card_requires_targets(pack_cards[index]):
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


def _shop_action(state: GameState, context: _ShopContext | None = None) -> Action | None:
    context = context or _ShopContext()
    pressure = _shop_pressure(state)
    replacement = _replacement_sell_action(state, pressure)
    if replacement is not None:
        return replacement

    best_action: Action | None = None
    best_value = 0.0

    for action in state.legal_actions:
        value = _shop_action_value(state, action, pressure, context)
        if value > best_value:
            best_action = action
            best_value = value

    threshold = _shop_buy_threshold(state, pressure)
    if best_action is not None and best_value >= threshold:
        return _annotated_action(
            best_action,
            reason=(
                f"shop_value value={best_value:.1f} pressure={pressure.ratio:.2f} "
                f"target={pressure.target_score:.0f} capacity={pressure.build_capacity:.0f}"
            ),
            audit=_shop_decision_audit(
                state,
                pressure,
                chosen_action=best_action,
                chosen_value=best_value,
                threshold=threshold,
                decision="take",
                context=context,
            ),
        )
    end_shop = _first_action_of_type(state, ActionType.END_SHOP)
    if end_shop is not None and _has_shop_decision_surface(state):
        return _annotated_action(
            end_shop,
            reason=(
                f"shop_skip best_value={best_value:.1f} threshold={threshold:.1f} "
                f"pressure={pressure.ratio:.2f} target={pressure.target_score:.0f} "
                f"capacity={pressure.build_capacity:.0f}"
            ),
            audit=_shop_decision_audit(
                state,
                pressure,
                chosen_action=end_shop,
                chosen_value=0.0,
                threshold=threshold,
                decision="skip",
                context=context,
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
    candidate_options: list[dict[str, object]] = []
    for card in shop_cards:
        if not _is_joker_card(card):
            continue
        cost = _card_cost(card)
        if state.money + sell_value < cost:
            continue
        candidate = _joker_from_shop_card(card)
        candidate_value = _candidate_joker_value_for_replacement(state, candidate)
        upgrade = candidate_value - weakest_value - max(0, cost - sell_value) * _replacement_cost_weight(pressure)
        candidate_options.append(
            {
                "name": candidate.name,
                "cost": cost,
                "candidate_value": round(candidate_value, 2),
                "upgrade": round(upgrade, 2),
            }
        )
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
            audit={
                "decision": "replace",
                "pressure": _pressure_payload(pressure),
                "threshold": round(_replacement_upgrade_threshold(pressure), 2),
                "owned_jokers": _owned_joker_value_payloads(state),
                "sold_joker": {
                    "index": weakest_index,
                    "name": weakest_joker.name,
                    "value": round(weakest_value, 2),
                    "sell_value": sell_value,
                },
                "replacement_options": sorted(candidate_options, key=lambda item: item["upgrade"], reverse=True),
                "chosen_replacement": best_label,
                "chosen_upgrade": round(best_upgrade, 2),
            },
        )
    return None


def _has_shop_decision_surface(state: GameState) -> bool:
    return bool(
        state.modifiers.get("shop_cards")
        or state.modifiers.get("voucher_cards")
        or state.modifiers.get("booster_packs")
        or _first_action_of_type(state, ActionType.REROLL)
    )


def _shop_decision_audit(
    state: GameState,
    pressure: _ShopPressure,
    *,
    chosen_action: Action,
    chosen_value: float,
    threshold: float,
    decision: str,
    context: _ShopContext | None = None,
) -> dict[str, object]:
    context = context or _ShopContext()
    options = [
        _shop_option_payload(state, action, _shop_action_value(state, action, pressure, context))
        for action in state.legal_actions
        if action.action_type in {ActionType.BUY, ActionType.OPEN_PACK, ActionType.REROLL, ActionType.END_SHOP}
    ]
    return {
        "decision": decision,
        "build_profile": _build_profile_payload(_build_profile(state)),
        "pressure": _pressure_payload(pressure),
        "threshold": round(threshold, 2),
        "chosen_value": round(chosen_value, 2),
        "chosen_action": _action_payload(chosen_action),
        "chosen_item": _item_payload_for_action(state, chosen_action),
        "money": state.money,
        "jokers": [joker.name for joker in state.jokers],
        "shop_context": {
            "rerolls_in_shop": context.rerolls_in_shop,
            "packs_opened_in_shop": context.packs_opened_in_shop,
        },
        "options": sorted(options, key=lambda item: item["value"], reverse=True),
    }


def _shop_option_payload(state: GameState, action: Action, value: float) -> dict[str, object]:
    payload = _action_payload(action)
    payload["value"] = round(value, 2)
    item = _item_payload_for_action(state, action)
    if item is not None:
        payload["item"] = item
    return payload


def _action_payload(action: Action) -> dict[str, object]:
    return {
        "type": action.action_type.value,
        "kind": str(action.metadata.get("kind", action.target_id or "")),
        "index": action.metadata.get("index", action.amount),
        "target_id": action.target_id,
        "amount": action.amount,
        "stable_key": action.stable_key,
    }


def _item_payload_for_action(state: GameState, action: Action) -> dict[str, object] | None:
    kind = str(action.metadata.get("kind", action.target_id or ""))
    index = action.metadata.get("index", action.amount)
    try:
        item_index = int(index)
    except (TypeError, ValueError):
        item_index = -1
    if action.action_type == ActionType.BUY and kind == "voucher":
        return _indexed_shop_item_payload(state.modifiers.get("voucher_cards", ()), item_index)
    if action.action_type == ActionType.BUY and kind == "card":
        return _indexed_shop_item_payload(state.modifiers.get("shop_cards", ()), item_index)
    if action.action_type == ActionType.OPEN_PACK:
        return _indexed_shop_item_payload(state.modifiers.get("booster_packs", ()), item_index)
    if action.action_type == ActionType.CHOOSE_PACK_CARD:
        return _indexed_shop_item_payload(state.modifiers.get("pack_cards", ()), item_index)
    return None


def _indexed_shop_item_payload(items: tuple[object, ...], index: int) -> dict[str, object] | None:
    if 0 <= index < len(items):
        return _shop_item_payload(items[index])
    return None


def _shop_item_payload(item: object) -> dict[str, object]:
    if not isinstance(item, dict):
        return {"name": str(item)}
    return {
        "name": _card_label(item),
        "set": _card_set(item),
        "cost": _card_cost(item),
        "key": item.get("key"),
        "rarity": item.get("rarity"),
    }


def _pressure_payload(pressure: _ShopPressure) -> dict[str, float]:
    return {
        "target_score": round(pressure.target_score, 2),
        "build_capacity": round(pressure.build_capacity, 2),
        "ratio": round(pressure.ratio, 4),
        "danger": round(pressure.danger, 4),
        "safe_margin": round(pressure.safe_margin, 4),
    }


def _owned_joker_value_payloads(state: GameState) -> list[dict[str, object]]:
    return [
        {
            "index": index,
            "name": joker.name,
            "value": round(_owned_joker_value(state, joker, remove_index=index), 2),
            "sell_value": joker.sell_value or 0,
        }
        for index, joker in enumerate(state.jokers)
    ]


def _shop_action_value(
    state: GameState,
    action: Action,
    pressure: _ShopPressure,
    context: _ShopContext | None = None,
) -> float:
    context = context or _ShopContext()
    kind = str(action.metadata.get("kind", ""))
    profile = _build_profile(state)
    if action.action_type == ActionType.BUY and kind == "card":
        shop_cards = state.modifiers.get("shop_cards", ())
        index = int(action.metadata.get("index", action.amount or 0))
        if index >= len(shop_cards):
            return 0.0
        card = shop_cards[index]
        value = _shop_card_value(state, card)
        value += _early_shop_safety_adjustment(state, card)
        if _is_joker_card(card):
            value += pressure.danger * 14
        return value - _cost_penalty(state, card, pressure)
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
        pack = packs[index]
        if state.money - _card_cost(pack) < 4 and pressure.ratio < 1.15:
            return 0.0
        if not _late_pack_is_worth_opening(state, pack, pressure, context):
            return 0.0
        return _pack_value(state, pack) + pressure.danger * 16 - _cost_penalty(state, pack, pressure)
    if action.action_type == ActionType.REROLL:
        if state.money < _minimum_reroll_bank(state):
            return 0.0
        if pressure.ratio < 0.95 and state.money < 14:
            return 0.0
        if not _late_reroll_is_worth_it(state, pressure, profile, context):
            return 0.0
        if len(state.jokers) >= 5 and pressure.ratio < 1.1 and not _rich_late_role_hunt(profile):
            return 0.0
        pressure_bonus = max(0, 4 - len(state.jokers)) * 7
        pressure_bonus += pressure.danger * 22
        pressure_bonus += _reroll_role_hunt_bonus(profile, pressure)
        return 24 + pressure_bonus - _money_after_spend_penalty(state, 5, pressure)
    return 0.0


def _minimum_reroll_bank(state: GameState) -> int:
    if state.ante >= 5 and len(state.jokers) >= 5:
        return 65
    if state.ante >= 5:
        return 18
    return 9


def _late_reroll_is_worth_it(
    state: GameState,
    pressure: _ShopPressure,
    profile: _BuildProfile,
    context: _ShopContext,
) -> bool:
    if state.ante < 5:
        return True
    if context.rerolls_in_shop >= _late_reroll_limit(state, pressure, profile):
        return False
    if pressure.ratio >= 1.05:
        return True
    if not _rich_late_role_hunt(profile):
        return False
    if len(state.jokers) >= 5 and pressure.ratio < 0.58:
        return False
    return state.money >= 75


def _late_reroll_limit(state: GameState, pressure: _ShopPressure, profile: _BuildProfile) -> int:
    if state.ante < 5:
        return 99
    if pressure.ratio >= 1.15:
        return 2
    if _rich_late_role_hunt(profile):
        return 1
    return 0


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
        if _pack_card_requires_targets(card):
            return 0.0
        return _tarot_card_value(state, card)
    if _is_playing_card(card):
        return _playing_card_shop_value(state, card)
    return 0.0


def _pack_card_value(state: GameState, card: object) -> float:
    if _pack_card_requires_targets(card):
        return 0.0
    if _is_joker_card(card):
        return _joker_card_value(state, card) + 15 + _early_shop_safety_adjustment(state, card)
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


def _pack_card_requires_targets(card: object) -> bool:
    return _is_tarot_card(card) and _card_label(card) in TARGET_REQUIRED_TAROTS


def _shop_buy_threshold(state: GameState, pressure: _ShopPressure) -> float:
    if len(state.jokers) == 0:
        base = 18.0
    elif state.ante <= 2 and len(state.jokers) < 3:
        base = 24.0
    elif state.ante >= 5 and state.money >= 50:
        base = 16.0
    elif state.money >= 25:
        base = 22.0
    else:
        base = 30.0
    safe_margin_weight = 8 if state.ante >= 5 and state.money >= 50 else 14
    return max(10.0, base - pressure.danger * 14 + pressure.safe_margin * safe_margin_weight)


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
    value += _joker_role_bonus(state, joker)
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
    value += _joker_role_bonus(state, joker)
    value += _edition_bonus(joker.edition)
    if any(existing.name == joker.name for existing in state.jokers):
        value -= 18
    return value


def _owned_joker_value(state: GameState, joker: Joker, *, remove_index: int) -> float:
    without = tuple(existing for index, existing in enumerate(state.jokers) if index != remove_index)
    score_loss = max(0.0, _sample_build_score(state, state.jokers) - _sample_build_score(state, without))
    value = score_loss * 0.08
    value += _joker_heuristic_value(state, joker) * 0.75
    value += _owned_role_value(state, joker, remove_index=remove_index)
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


def _build_profile(state: GameState) -> _BuildProfile:
    joker_roles = [_joker_roles(joker) for joker in state.jokers]
    preferred = _preferred_hand_type(state)
    return _BuildProfile(
        preferred_hand=preferred,
        archetype=_build_archetype(state, preferred),
        has_chips=any("chips" in roles for roles in joker_roles),
        has_mult=any("mult" in roles for roles in joker_roles),
        has_xmult=any("xmult" in roles for roles in joker_roles),
        has_scaling=any("scaling" in roles for roles in joker_roles),
        has_economy=state.money >= 25 or any("economy" in roles for roles in joker_roles),
        open_joker_slots=max(0, 5 - len(state.jokers)),
        money=state.money,
        ante=state.ante,
    )


def _build_profile_payload(profile: _BuildProfile) -> dict[str, object]:
    return {
        "preferred_hand": profile.preferred_hand.value if profile.preferred_hand is not None else None,
        "archetype": profile.archetype,
        "has_chips": profile.has_chips,
        "has_mult": profile.has_mult,
        "has_xmult": profile.has_xmult,
        "has_scaling": profile.has_scaling,
        "has_economy": profile.has_economy,
        "missing_roles": list(profile.missing_roles),
        "open_joker_slots": profile.open_joker_slots,
        "money": profile.money,
        "ante": profile.ante,
    }


def _joker_roles(joker: Joker) -> frozenset[str]:
    roles: set[str] = set()
    name = joker.name
    if name in CHIP_JOKERS:
        roles.add("chips")
    if name in MULT_JOKERS:
        roles.add("mult")
    if name in XMULT_JOKERS:
        roles.add("xmult")
    if name in JOKER_SCALING_VALUES or name in SCALING_JOKERS:
        roles.add("scaling")
    if name in JOKER_ECONOMY_VALUES:
        roles.add("economy")

    edition = (joker.edition or "").lower()
    if "foil" in edition:
        roles.add("chips")
    if "holo" in edition or "holographic" in edition:
        roles.add("mult")
    if "polychrome" in edition:
        roles.add("xmult")
    if "negative" in edition:
        roles.add("economy")
    return frozenset(roles)


def _joker_role_bonus(state: GameState, joker: Joker) -> float:
    profile = _build_profile(state)
    roles = _joker_roles(joker)
    bonus = 0.0
    for role in profile.missing_roles:
        if role in roles:
            bonus += ROLE_MISSING_BONUSES[role]

    if profile.late and profile.rich:
        if "xmult" in roles and not profile.has_xmult:
            bonus += 22
        if "scaling" in roles and not profile.has_scaling:
            bonus += 16
        if "economy" in roles and profile.has_economy:
            bonus -= 10
    if profile.ante <= 2 and ("chips" in roles or "mult" in roles):
        bonus += 6
    if profile.open_joker_slots <= 1 and roles.isdisjoint({"xmult", "scaling"}) and profile.late:
        bonus -= 12
    return bonus


def _build_archetype(state: GameState, preferred: HandType | None) -> str:
    if preferred in {HandType.FLUSH, HandType.STRAIGHT_FLUSH, HandType.FLUSH_HOUSE, HandType.FLUSH_FIVE}:
        return "flush"
    if preferred in {HandType.STRAIGHT, HandType.STRAIGHT_FLUSH}:
        return "straight"
    if preferred in {
        HandType.PAIR,
        HandType.TWO_PAIR,
        HandType.THREE_OF_A_KIND,
        HandType.FULL_HOUSE,
        HandType.FOUR_OF_A_KIND,
        HandType.FIVE_OF_A_KIND,
    }:
        return "rank"
    if any(joker.name in FLEX_SCALING_JOKERS for joker in state.jokers):
        return "flex_scaling"
    if not state.jokers:
        return "open"
    return "flex"


def _owned_role_value(state: GameState, joker: Joker, *, remove_index: int) -> float:
    roles = _joker_roles(joker)
    if not roles:
        return 0.0

    remaining_roles = [
        _joker_roles(existing)
        for index, existing in enumerate(state.jokers)
        if index != remove_index
    ]
    value = 0.0
    for role in ("chips", "mult", "xmult", "scaling", "economy"):
        if role in roles and not any(role in existing_roles for existing_roles in remaining_roles):
            value += ROLE_UNIQUE_VALUES[role]
    return value


def _reroll_role_hunt_bonus(profile: _BuildProfile, pressure: _ShopPressure) -> float:
    if not profile.rich:
        return 0.0
    bonus = 0.0
    if not profile.has_xmult and (profile.late or pressure.ratio >= 1.0):
        bonus += 16
    if not profile.has_scaling and (profile.late or pressure.ratio >= 1.0):
        bonus += 10
    if profile.open_joker_slots >= 2 and ("chips" in profile.missing_roles or "mult" in profile.missing_roles):
        bonus += 8
    return bonus


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
        if state.ante >= 4 and not _build_profile(state).has_scaling:
            value += 8
    elif hand_type in _flexible_hand_types(state):
        value += 8
    else:
        value -= 8
    return value


def _tarot_card_value(state: GameState, card: object) -> float:
    name = _card_label(card)
    value = TAROT_VALUES.get(name, 18)
    preferred = _preferred_hand_type(state)
    profile = _build_profile(state)
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
    if profile.rich and profile.ante >= 4 and name in {"Death", "The Hanged Man", "Strength"}:
        value += 10
    if not profile.has_economy and name in {"The Hermit", "Temperance"}:
        value += 8
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
    profile = _build_profile(state)
    if "buffoon" in name:
        value = 42 if len(state.jokers) < 5 else 8
        if profile.rich and ("xmult" in profile.missing_roles or "scaling" in profile.missing_roles):
            value += 18
    elif "celestial" in name:
        value = 28
        if profile.ante >= 4 and not profile.has_scaling:
            value += 10
        if profile.rich and profile.late:
            value += 12
    elif "arcana" in name:
        value = 26
        if profile.rich and profile.ante >= 4:
            value += 14
        if profile.rich and profile.late:
            value += 6
    elif "standard" in name:
        value = 18
    elif "spectral" in name:
        value = 20 if state.ante <= 2 else 14
    else:
        value = 16
    if "mega" in name or "jumbo" in name:
        value += 8
    return float(value)


def _late_pack_is_worth_opening(
    state: GameState,
    pack: object,
    pressure: _ShopPressure,
    context: _ShopContext,
) -> bool:
    if state.ante < 5:
        return True
    if context.packs_opened_in_shop >= _late_pack_limit(state, pressure):
        return False
    if pressure.ratio >= 1.05:
        return True

    capacity_gain = _pack_capacity_gain(state, pack, pressure)
    if capacity_gain <= 0:
        return False

    floor = _minimum_late_pack_capacity_gain(state, pressure)
    if pressure.ratio < 0.65:
        return capacity_gain >= floor
    return capacity_gain >= floor * 0.65


def _late_pack_limit(state: GameState, pressure: _ShopPressure) -> int:
    if state.ante < 5:
        return 99
    if pressure.ratio >= 1.05:
        return 2
    return 1


def _pack_capacity_gain(state: GameState, pack: object, pressure: _ShopPressure) -> float:
    name = _card_label(pack).lower()
    current = _sample_build_score(state, state.jokers)
    spend_loss = _score_loss_after_spending(state, _card_cost(pack), current_score=current)
    profile = _build_profile(state)

    if "celestial" in name:
        return _celestial_pack_capacity_gain(state, pack, current_score=current)
    if "buffoon" in name:
        if len(state.jokers) < 5:
            role_bonus = 0.18 if ("xmult" in profile.missing_roles or "scaling" in profile.missing_roles) else 0.10
            return (current * role_bonus) - spend_loss
        if _rich_late_role_hunt(profile):
            return (current * 0.08) - spend_loss
        return -spend_loss
    if "arcana" in name:
        if pressure.ratio < 0.75:
            return -spend_loss
        return (current * 0.06) - spend_loss
    if "standard" in name:
        return (current * 0.04) - spend_loss
    if "spectral" in name:
        return (current * 0.05) - spend_loss
    return -spend_loss


def _celestial_pack_capacity_gain(state: GameState, pack: object, *, current_score: float) -> float:
    after_spend = replace(state, money=max(0, state.money - _card_cost(pack)))
    candidates = _celestial_candidate_hand_types(state)
    if not candidates:
        return -_score_loss_after_spending(state, _card_cost(pack), current_score=current_score)

    best_score = 0.0
    for hand_type in candidates:
        hand_levels = dict(after_spend.hand_levels)
        hand_levels[hand_type.value] = hand_levels.get(hand_type.value, 1) + 1
        leveled = replace(after_spend, hand_levels=hand_levels)
        best_score = max(best_score, _sample_build_score(leveled, leveled.jokers))
    return best_score - current_score


def _celestial_candidate_hand_types(state: GameState) -> tuple[HandType, ...]:
    preferred = _preferred_hand_type(state)
    flexible = _flexible_hand_types(state)
    ordered: list[HandType] = []
    for hand_type in (*((preferred,) if preferred is not None else ()), *tuple(flexible)):
        if hand_type not in ordered:
            ordered.append(hand_type)
    return tuple(ordered[:4])


def _score_loss_after_spending(state: GameState, cost: int, *, current_score: float) -> float:
    if cost <= 0:
        return 0.0
    after_spend = replace(state, money=max(0, state.money - cost))
    return max(0.0, current_score - _sample_build_score(after_spend, after_spend.jokers))


def _minimum_late_pack_capacity_gain(state: GameState, pressure: _ShopPressure) -> float:
    current = _sample_build_score(state, state.jokers)
    if pressure.ratio < 0.65:
        return max(350.0, current * 0.14)
    return max(250.0, current * 0.09)


def _cost_penalty(state: GameState, card: object, pressure: _ShopPressure) -> float:
    cost = _card_cost(card)
    cost_weight = max(1.6, 3.0 - pressure.danger * 0.9 + pressure.safe_margin * 1.0)
    if state.ante >= 5 and state.money >= 50:
        cost_weight = max(0.9, cost_weight - 0.8)
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


def _early_shop_safety_adjustment(state: GameState, card: object) -> float:
    if state.ante > 2:
        return 0.0

    adjustment = 0.0
    if _is_tarot_card(card) and _pack_card_requires_targets(card):
        return -100.0
    if _is_planet_card(card) and not _has_real_scoring_joker(state):
        adjustment -= 18.0
    if _is_joker_card(card):
        joker = _joker_from_shop_card(card)
        roles = _joker_roles(joker)
        if not _has_real_scoring_joker(state):
            if roles == {"economy"}:
                adjustment -= 55.0
            if joker.name == "Swashbuckler" and _joker_sell_total(state) < 8:
                adjustment -= 45.0
            if joker.name in NARROW_EARLY_JOKERS and not _early_hand_type_is_supported(state, joker.name):
                adjustment -= 95.0
        if state.money - _card_cost(card) <= 1 and roles.isdisjoint({"chips", "mult", "xmult", "scaling"}):
            adjustment -= 20.0
    return adjustment


def _has_real_scoring_joker(state: GameState) -> bool:
    return any(_joker_roles(joker) & {"chips", "mult", "xmult", "scaling"} for joker in state.jokers)


def _joker_sell_total(state: GameState) -> int:
    return sum(joker.sell_value or 0 for joker in state.jokers)


def _early_hand_type_is_supported(state: GameState, joker_name: str) -> bool:
    wanted = set(JOKER_HAND_SYNERGY.get(joker_name, ()))
    if not wanted:
        return True
    if state.hand_levels and any(state.hand_levels.get(hand_type.value, 1) > 1 for hand_type in wanted):
        return True
    return _hand_has_natural_support(state.hand, wanted)


def _hand_has_natural_support(hand: tuple[Card, ...], wanted: set[HandType]) -> bool:
    if not hand:
        return False
    rank_counts = Counter(card.rank for card in hand)
    suit_counts = Counter(card.suit for card in hand)
    if wanted & {HandType.STRAIGHT, HandType.STRAIGHT_FLUSH}:
        return _straight_draw_potential(hand) >= 4
    if wanted & {HandType.FLUSH, HandType.STRAIGHT_FLUSH, HandType.FLUSH_HOUSE, HandType.FLUSH_FIVE}:
        return max(suit_counts.values(), default=0) >= 4
    if wanted & {HandType.THREE_OF_A_KIND, HandType.FULL_HOUSE, HandType.FOUR_OF_A_KIND, HandType.FIVE_OF_A_KIND}:
        return max(rank_counts.values(), default=0) >= 3
    if wanted & {HandType.PAIR, HandType.TWO_PAIR}:
        return max(rank_counts.values(), default=0) >= 2
    return False


def _rich_late_role_hunt(profile: _BuildProfile) -> bool:
    return profile.rich and profile.late and ("xmult" in profile.missing_roles or "scaling" in profile.missing_roles)


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
    return [_play_candidate(state, action) for action in play_actions]


def _play_candidate(state: GameState, action: Action) -> _PlayCandidate:
    evaluation = _evaluate_play_action(state, action)
    scoring_card_indices = tuple(action.card_indices[index] for index in evaluation.scoring_indices)
    cycle_value = _cycle_value_for_play(state, action, scoring_card_indices)
    cycle_count = sum(1 for index in action.card_indices if index not in scoring_card_indices)
    return _PlayCandidate(
        action=action,
        score=evaluation.score,
        hand_type=evaluation.hand_type,
        scoring_card_indices=scoring_card_indices,
        cycle_value=cycle_value,
        cycle_count=cycle_count,
    )


def _minimum_sufficient_play_key(candidate: _PlayCandidate) -> tuple[int, float, int, int]:
    return (candidate.score, -candidate.cycle_value, -candidate.cycle_count, _action_index_sum(candidate.action))


def _maximum_play_key(candidate: _PlayCandidate) -> tuple[int, float, int]:
    return (candidate.score, candidate.cycle_value, -len(candidate.action.card_indices))


def _hands_to_clear_key(candidate: _PlayCandidate, remaining_score: int) -> tuple[int, int, float, int, int]:
    estimated_hands = ceil(remaining_score / max(1, candidate.score))
    return (
        estimated_hands,
        -candidate.score,
        -candidate.cycle_value,
        -candidate.cycle_count,
        _action_index_sum(candidate.action),
    )


def _action_index_sum(action: Action) -> int:
    return sum(action.card_indices)


def _score_play_action(state: GameState, action: Action) -> int:
    return _evaluate_play_action(state, action).score


def _evaluate_play_action(state: GameState, action: Action):
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
    )


def _cycle_value_for_play(
    state: GameState,
    action: Action,
    scoring_card_indices: tuple[int, ...],
) -> float:
    if len(action.card_indices) <= len(scoring_card_indices):
        return 0.0

    preferred = _preferred_hand_type(state)
    keep_scores = _card_keep_scores(state.hand, preferred)
    scoring_set = set(scoring_card_indices)
    value = 0.0
    for index in action.card_indices:
        if index in scoring_set:
            continue
        value += 72.0 - keep_scores[index]
    return value


def _tactical_blind_action(state: GameState, best_play: Action) -> Action:
    score = _score_play_action(state, best_play)
    remaining_score = max(0, state.required_score - state.current_score)
    if (
        remaining_score == 0
        or score >= remaining_score
        or state.hands_remaining <= 1
        or state.discards_remaining <= 0
        or _estimated_hands_needed(remaining_score, score) < state.hands_remaining
    ):
        return _annotated_action(best_play, reason=_play_reason(state, best_play))

    best_discard = _best_discard_action(state, current_best_score=score)
    if best_discard is not None and _discard_can_reduce_hands_needed(state, best_discard, score, remaining_score):
        return _annotated_action(best_discard, reason=_discard_reason(state, best_play, best_discard))

    if _score_is_on_pace(state, score, remaining_score):
        return _annotated_action(best_play, reason=_play_reason(state, best_play))

    if best_discard is not None:
        return _annotated_action(best_discard, reason=_discard_reason(state, best_play, best_discard))
    return _annotated_action(best_play, reason=_play_reason(state, best_play))


def _should_play_now(state: GameState, action: Action) -> bool:
    score = _score_play_action(state, action)
    remaining_score = max(0, state.required_score - state.current_score)
    if remaining_score == 0:
        return True
    if score >= remaining_score:
        return True
    if state.hands_remaining <= 1 or state.discards_remaining <= 0:
        return True

    estimated_hands_to_clear = _estimated_hands_needed(remaining_score, score)
    if estimated_hands_to_clear < state.hands_remaining:
        return True

    best_discard = _best_discard_action(state, current_best_score=score)
    if best_discard is None:
        return True

    if _discard_can_reduce_hands_needed(state, best_discard, score, remaining_score):
        return False

    return _score_is_on_pace(state, score, remaining_score)


def _best_discard_action(state: GameState, *, current_best_score: int | None = None) -> Action | None:
    discard_actions = [
        action
        for action in state.legal_actions
        if action.action_type == ActionType.DISCARD and action.card_indices
    ]
    if not discard_actions:
        return None

    preferred = _preferred_hand_type(state)
    keep_scores = _card_keep_scores(state.hand, preferred)
    protected_count = max(2, min(4, len(state.hand) // 2))
    protected = {
        index
        for index, _ in sorted(
            enumerate(keep_scores),
            key=lambda item: (item[1], RANK_VALUES[state.hand[item[0]].rank]),
            reverse=True,
        )[:protected_count]
    }

    remaining_score = max(0, state.required_score - state.current_score)
    if current_best_score is None:
        current_best_play = _best_play_action(state)
        current_best_score = _score_play_action(state, current_best_play) if current_best_play is not None else 0
    current_hands_needed = _estimated_hands_needed(remaining_score, current_best_score or 0)
    detailed_actions = _prefilter_discard_actions(state, discard_actions, keep_scores, protected, preferred)

    def discard_score(action: Action) -> tuple[float, float, int, int]:
        protected_penalty = sum(1 for index in action.card_indices if index in protected) * 1000
        desirability = sum(100 - keep_scores[index] for index in action.card_indices)
        kept_cards = tuple(card for index, card in enumerate(state.hand) if index not in action.card_indices)
        kept_potential = _kept_hand_potential(state, kept_cards)
        projected_score = _projected_score_after_discard(state, action)
        projected_hands_needed = _estimated_hands_needed(remaining_score, projected_score)
        speed_bonus = max(0, current_hands_needed - projected_hands_needed) * 900
        score_bonus = min(projected_score, remaining_score) * 0.08
        return (
            desirability + kept_potential + speed_bonus + score_bonus - protected_penalty,
            projected_score,
            kept_potential,
            len(action.card_indices),
        )

    best = max(detailed_actions, key=discard_score)
    return best if discard_score(best)[0] > -500 else None


def _prefilter_discard_actions(
    state: GameState,
    discard_actions: list[Action],
    keep_scores: tuple[float, ...],
    protected: set[int],
    preferred: HandType | None,
) -> list[Action]:
    if len(discard_actions) <= DISCARD_DETAIL_LIMIT:
        return discard_actions

    def cheap_score(action: Action) -> tuple[float, int]:
        protected_penalty = sum(1 for index in action.card_indices if index in protected) * 1000
        desirability = sum(100 - keep_scores[index] for index in action.card_indices)
        kept_cards = tuple(card for index, card in enumerate(state.hand) if index not in action.card_indices)
        return (
            desirability + _cheap_kept_hand_potential(kept_cards, preferred) - protected_penalty,
            len(action.card_indices),
        )

    return sorted(discard_actions, key=cheap_score, reverse=True)[:DISCARD_DETAIL_LIMIT]


def _estimated_hands_needed(remaining_score: int, score: int | float) -> int:
    if remaining_score <= 0:
        return 0
    return ceil(remaining_score / max(1, score))


def _score_is_on_pace(state: GameState, score: int, remaining_score: int) -> bool:
    if state.hands_remaining <= 0:
        return True
    pace_score = remaining_score / max(1, state.hands_remaining)
    return score >= pace_score * _pace_safety_multiplier(state)


def _pace_safety_multiplier(state: GameState) -> float:
    multiplier = 1.0
    if state.ante >= 5:
        multiplier += 0.08
    if _is_boss_blind(state):
        multiplier += 0.05
    if state.hands_remaining <= 2:
        multiplier += 0.04
    if state.discards_remaining >= 3 and state.hands_remaining >= 3:
        multiplier += 0.03
    return multiplier


def _is_boss_blind(state: GameState) -> bool:
    if not state.blind:
        return False
    return state.blind not in {"Small Blind", "Big Blind"}


def _discard_can_reduce_hands_needed(
    state: GameState,
    action: Action,
    current_score: int,
    remaining_score: int,
) -> bool:
    current_hands_needed = _estimated_hands_needed(remaining_score, current_score)
    if current_hands_needed <= 1:
        return False

    projected_score = _projected_score_after_discard(state, action)
    projected_hands_needed = _estimated_hands_needed(remaining_score, projected_score)
    if projected_hands_needed >= current_hands_needed:
        return False
    if state.known_deck:
        return True

    kept_cards = tuple(card for index, card in enumerate(state.hand) if index not in action.card_indices)
    return _strong_draw_size(kept_cards) >= 4 and projected_score >= current_score * 1.35


def _projected_score_after_discard(state: GameState, action: Action) -> int:
    kept_cards = tuple(card for index, card in enumerate(state.hand) if index not in action.card_indices)
    draw_count = min(len(action.card_indices), max(0, 8 - len(kept_cards)))

    if state.known_deck and draw_count > 0:
        known_draw = tuple(state.known_deck[:draw_count])
        return _best_score_from_cards(state, (*kept_cards, *known_draw))

    kept_score = _best_score_from_cards(state, kept_cards)
    optimistic_score = _optimistic_completion_score(state, kept_cards, draw_count)
    realism = _discard_realism_factor(kept_cards, draw_count)
    return int(max(kept_score, (optimistic_score * realism) + (kept_score * (1.0 - realism))))


def _best_score_from_cards(state: GameState, cards: tuple[Card, ...]) -> int:
    if not cards:
        return 0
    return best_play_from_hand(
        cards,
        state.hand_levels,
        debuffed_suits=debuffed_suits_for_blind(state.blind),
        blind_name=state.blind,
        jokers=state.jokers,
        discards_remaining=max(0, state.discards_remaining - 1),
        hands_remaining=state.hands_remaining,
        deck_size=state.deck_size,
        money=state.money,
    ).score


def _optimistic_completion_score(state: GameState, kept_cards: tuple[Card, ...], draw_count: int) -> int:
    if draw_count <= 0:
        return _best_score_from_cards(state, kept_cards)

    candidates = [_fill_with_high_cards(kept_cards, draw_count)]
    flush_cards = _flush_completion_cards(kept_cards, draw_count)
    if flush_cards:
        candidates.append(flush_cards)
    straight_cards = _straight_completion_cards(kept_cards, draw_count)
    if straight_cards:
        candidates.append(straight_cards)
    rank_cards = _rank_completion_cards(kept_cards, draw_count)
    if rank_cards:
        candidates.append(rank_cards)

    return max(_best_score_from_cards(state, candidate) for candidate in candidates)


def _fill_with_high_cards(kept_cards: tuple[Card, ...], draw_count: int) -> tuple[Card, ...]:
    existing_ranks = {card.rank for card in kept_cards}
    fill_pool = (
        Card("A", "S"),
        Card("K", "H"),
        Card("Q", "D"),
        Card("J", "C"),
        Card("10", "S"),
        Card("9", "H"),
        Card("8", "D"),
    )
    fill = tuple(card for card in fill_pool if card.rank not in existing_ranks)
    return (*kept_cards, *fill[:draw_count])[:8]


def _flush_completion_cards(kept_cards: tuple[Card, ...], draw_count: int) -> tuple[Card, ...]:
    suit_counts = Counter(card.suit for card in kept_cards)
    if not suit_counts:
        return ()
    suit, count = max(suit_counts.items(), key=lambda item: item[1])
    missing = max(0, 5 - count)
    if missing == 0 or missing > draw_count:
        return ()
    existing_ranks = {card.rank for card in kept_cards if card.suit == suit}
    fill = tuple(Card(rank, suit) for rank in ("A", "K", "Q", "J", "10") if rank not in existing_ranks)
    return (*kept_cards, *fill[:draw_count])[:8]


def _straight_completion_cards(kept_cards: tuple[Card, ...], draw_count: int) -> tuple[Card, ...]:
    if not kept_cards:
        return ()

    present_values = {STRAIGHT_VALUES[card.rank] for card in kept_cards}
    if 14 in present_values:
        present_values.add(1)

    best_missing: list[int] | None = None
    for start in range(1, 11):
        run = set(range(start, start + 5))
        missing = sorted(run - present_values)
        if len(missing) <= draw_count and (best_missing is None or len(missing) < len(best_missing)):
            best_missing = missing

    if not best_missing:
        return ()

    suit = _dominant_suit_from_cards(kept_cards) or "S"
    fill = tuple(Card(_straight_rank_for_value(value), suit) for value in best_missing)
    return (*kept_cards, *fill, *_fill_with_high_cards((), max(0, draw_count - len(fill))))[:8]


def _rank_completion_cards(kept_cards: tuple[Card, ...], draw_count: int) -> tuple[Card, ...]:
    rank_counts = Counter(card.rank for card in kept_cards)
    if not rank_counts:
        return ()

    rank, count = max(rank_counts.items(), key=lambda item: (item[1], RANK_VALUES[item[0]]))
    if count < 3:
        return ()

    target_count = 4 if count + draw_count >= 4 else 3
    missing = max(0, target_count - count)
    if missing == 0 or missing > draw_count:
        return ()

    used_suits = {card.suit for card in kept_cards if card.rank == rank}
    fill_suits = tuple(suit for suit in ("S", "H", "D", "C") if suit not in used_suits)
    fill = tuple(Card(rank, suit) for suit in fill_suits[:missing])
    remaining_draw = max(0, draw_count - len(fill))
    return (*kept_cards, *fill, *_fill_with_high_cards((), remaining_draw))[:8]


def _dominant_suit_from_cards(cards: tuple[Card, ...]) -> str | None:
    suit_counts = Counter(card.suit for card in cards)
    if not suit_counts:
        return None
    return max(suit_counts.items(), key=lambda item: item[1])[0]


def _straight_rank_for_value(value: int) -> str:
    if value == 1:
        return "A"
    for rank, rank_value in STRAIGHT_VALUES.items():
        if rank_value == value and rank != "T":
            return rank
    return "A"


def _discard_realism_factor(kept_cards: tuple[Card, ...], draw_count: int) -> float:
    if draw_count <= 0:
        return 0.0
    draw_size = _strong_draw_size(kept_cards)
    if draw_size >= 4:
        return min(0.72, 0.30 + draw_size * 0.08 + draw_count * 0.04)
    if draw_size == 3:
        return min(0.52, 0.24 + draw_count * 0.05)
    return min(0.45, 0.20 + draw_count * 0.05)


def _strong_draw_size(cards: tuple[Card, ...]) -> int:
    if not cards:
        return 0
    rank_counts = Counter(card.rank for card in cards)
    suit_counts = Counter(card.suit for card in cards)
    return max(
        max(rank_counts.values(), default=0),
        max(suit_counts.values(), default=0),
        _straight_draw_potential(cards),
    )


def _play_reason(state: GameState, action: Action) -> str:
    score = _score_play_action(state, action)
    remaining_score = max(0, state.required_score - state.current_score)
    hands_needed = _estimated_hands_needed(remaining_score, score)
    return (
        f"tactical_play score={score} remaining={remaining_score} "
        f"hands_needed={hands_needed} hands_left={state.hands_remaining}"
    )


def _discard_reason(state: GameState, best_play: Action, discard: Action) -> str:
    best_score = _score_play_action(state, best_play)
    remaining_score = max(0, state.required_score - state.current_score)
    projected_score = _projected_score_after_discard(state, discard)
    current_needed = _estimated_hands_needed(remaining_score, best_score)
    projected_needed = _estimated_hands_needed(remaining_score, projected_score)
    return (
        f"tactical_discard current_score={best_score} projected_score={projected_score} "
        f"hands_needed={current_needed}->{projected_needed} hands_left={state.hands_remaining}"
    )


def _kept_hand_potential(state: GameState, kept_cards: tuple[Card, ...]) -> float:
    if not kept_cards:
        return 0.0

    cheap_potential = _cheap_kept_hand_potential(kept_cards, _preferred_hand_type(state))
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

    return cheap_potential + (immediate_score * 0.03)


def _cheap_kept_hand_potential(kept_cards: tuple[Card, ...], preferred: HandType | None = None) -> float:
    if not kept_cards:
        return 0.0

    rank_counts = Counter(card.rank for card in kept_cards)
    suit_counts = Counter(card.suit for card in kept_cards)
    rank_weight = 20
    flush_weight = 8
    straight_weight = 6
    if preferred in {HandType.PAIR, HandType.TWO_PAIR, HandType.THREE_OF_A_KIND, HandType.FULL_HOUSE}:
        rank_weight = 28
        flush_weight = 5
        straight_weight = 4
    elif preferred in {HandType.FLUSH, HandType.STRAIGHT_FLUSH, HandType.FLUSH_HOUSE, HandType.FLUSH_FIVE}:
        rank_weight = 8
        flush_weight = 24
        straight_weight = 4
    elif preferred in {HandType.STRAIGHT, HandType.STRAIGHT_FLUSH}:
        rank_weight = 10
        flush_weight = 5
        straight_weight = 22

    pair_bonus = sum(count * rank_weight for count in rank_counts.values() if count >= 2)
    flush_draw_bonus = max(suit_counts.values(), default=0) * flush_weight
    straight_draw_bonus = _straight_draw_potential(kept_cards) * straight_weight
    high_card_bonus = sum(RANK_VALUES[card.rank] for card in kept_cards) / max(1, len(kept_cards))
    return pair_bonus + flush_draw_bonus + straight_draw_bonus + high_card_bonus


def _straight_draw_potential(cards: tuple[Card, ...]) -> int:
    values = {STRAIGHT_VALUES[card.rank] for card in cards}
    if "A" in {card.rank for card in cards}:
        values.add(1)
    best = 0
    for start in range(1, 11):
        best = max(best, sum(1 for value in range(start, start + 5) if value in values))
    return best


def _card_keep_scores(hand: tuple[Card, ...], preferred: HandType | None = None) -> tuple[float, ...]:
    rank_counts = Counter(card.rank for card in hand)
    suit_counts = Counter(card.suit for card in hand)
    straight_values = [STRAIGHT_VALUES[card.rank] for card in hand]

    rank_weight = 100
    suit_weight = 8
    straight_weight = 6
    if preferred in {HandType.PAIR, HandType.TWO_PAIR, HandType.THREE_OF_A_KIND, HandType.FULL_HOUSE}:
        rank_weight = 140
        suit_weight = 5
        straight_weight = 4
    elif preferred in {HandType.FLUSH, HandType.STRAIGHT_FLUSH, HandType.FLUSH_HOUSE, HandType.FLUSH_FIVE}:
        rank_weight = 35
        suit_weight = 28
        straight_weight = 4
    elif preferred in {HandType.STRAIGHT, HandType.STRAIGHT_FLUSH}:
        rank_weight = 45
        suit_weight = 5
        straight_weight = 24

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
        score += suit_counts[card.suit] * suit_weight
        score += max(nearby, ace_low_nearby) * straight_weight
        if rank_count >= 2:
            score += rank_count * rank_weight
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


NARROW_EARLY_JOKERS = {
    "Crazy Joker",
    "Devious Joker",
    "Droll Joker",
    "Crafty Joker",
    "Runner",
    "The Order",
    "The Tribe",
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


CHIP_JOKERS = {
    "Sly Joker",
    "Wily Joker",
    "Clever Joker",
    "Devious Joker",
    "Crafty Joker",
    "Scary Face",
    "Odd Todd",
    "Scholar",
    "Arrowhead",
    "Banner",
    "Stuntman",
    "Bull",
    "Blue Joker",
    "Square Joker",
    "Wee Joker",
    "Runner",
    "Ice Cream",
    "Stone Joker",
    "Castle",
}


MULT_JOKERS = {
    "Joker",
    "Jolly Joker",
    "Zany Joker",
    "Mad Joker",
    "Crazy Joker",
    "Droll Joker",
    "Greedy Joker",
    "Lusty Joker",
    "Wrathful Joker",
    "Gluttonous Joker",
    "Half Joker",
    "Mystic Summit",
    "Misprint",
    "Gros Michel",
    "Popcorn",
    "Abstract Joker",
    "Swashbuckler",
    "Supernova",
    "Bootstraps",
    "Fibonacci",
    "Scholar",
    "Even Steven",
    "Smiley Face",
    "Walkie Talkie",
    "Onyx Agate",
    "Green Joker",
    "Ride the Bus",
    "Spare Trousers",
    "Fortune Teller",
    "Flash Card",
    "Red Card",
    "Erosion",
    "Shoot the Moon",
    "Raised Fist",
}


XMULT_JOKERS = {
    "Blueprint",
    "Brainstorm",
    "Cavendish",
    "Ramen",
    "Acrobat",
    "Blackboard",
    "The Duo",
    "The Trio",
    "The Family",
    "The Order",
    "The Tribe",
    "Seeing Double",
    "Flower Pot",
    "Ancient Joker",
    "The Idol",
    "Baseball Card",
    "Steel Joker",
    "Glass Joker",
    "Driver's License",
    "Joker Stencil",
    "Baron",
    "Hologram",
    "Constellation",
    "Lucky Cat",
    "Campfire",
    "Throwback",
    "Obelisk",
    "Canio",
    "Caino",
    "Yorick",
    "Madness",
    "Vampire",
    "Card Sharp",
    "Loyalty Card",
}


SCALING_JOKERS = {
    "Green Joker",
    "Ride the Bus",
    "Supernova",
    "Square Joker",
    "Runner",
    "Spare Trousers",
    "Hologram",
    "Constellation",
    "Flash Card",
    "Red Card",
    "Castle",
    "Erosion",
    "Wee Joker",
    "Lucky Cat",
    "Glass Joker",
    "Campfire",
    "Throwback",
    "Obelisk",
    "Fortune Teller",
    "Steel Joker",
    "Vampire",
    "Madness",
}


FLEX_SCALING_JOKERS = {
    "Abstract Joker",
    "Banner",
    "Blue Joker",
    "Bull",
    "Green Joker",
    "Hologram",
    "Ride the Bus",
    "Square Joker",
    "Supernova",
}


ROLE_MISSING_BONUSES = {
    "chips": 18.0,
    "mult": 18.0,
    "xmult": 34.0,
    "scaling": 26.0,
    "economy": 14.0,
}


ROLE_UNIQUE_VALUES = {
    "chips": 18.0,
    "mult": 18.0,
    "xmult": 34.0,
    "scaling": 24.0,
    "economy": 10.0,
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


TARGET_REQUIRED_TAROTS = {
    "The Magician",
    "The Empress",
    "The Hierophant",
    "The Lovers",
    "The Chariot",
    "Justice",
    "Strength",
    "The Hanged Man",
    "Death",
    "The Devil",
    "The Tower",
    "The Star",
    "The Moon",
    "The Sun",
    "The World",
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
