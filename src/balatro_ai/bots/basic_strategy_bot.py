"""Basic rule bot with simple play/discard and shop discipline."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from itertools import permutations
from math import ceil
import re
from typing import Any

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import Card, GamePhase, GameState, Joker
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
    raw_ratio: float
    safety_multiplier: float
    capacity_safety_factor: float
    boss_name: str | None = None
    boss_target_multiplier: float = 1.0
    boss_capacity_factor: float = 1.0

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
    chip_score: float
    mult_score: float
    xmult_score: float
    scaling_score: float
    economy_score: float
    has_chips: bool
    has_mult: bool
    has_xmult: bool
    has_scaling: bool
    has_economy: bool
    open_joker_slots: int
    money: int
    spendable_money: int
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
        return self.spendable_money >= 20

    @property
    def late(self) -> bool:
        return self.ante >= 5

    def role_score(self, role: str) -> float:
        return {
            "chips": self.chip_score,
            "mult": self.mult_score,
            "xmult": self.xmult_score,
            "scaling": self.scaling_score,
            "economy": self.economy_score,
        }.get(role, 0.0)

    def role_requirement(self, role: str) -> float:
        return _role_requirement(role, self.ante)

    def role_deficit_ratio(self, role: str) -> float:
        requirement = self.role_requirement(role)
        if requirement <= 0:
            return 0.0
        return max(0.0, min(1.0, (requirement - self.role_score(role)) / requirement))


@dataclass(frozen=True, slots=True)
class _ShopContext:
    rerolls_in_shop: int = 0
    packs_opened_in_shop: int = 0
    filled_last_joker_slot: bool = False


@dataclass(frozen=True, slots=True)
class _BlindContext:
    played_hand_types: tuple[HandType, ...] = ()
    discards_taken: int = 0


DISCARD_DETAIL_LIMIT = 28
MAX_JOKER_REARRANGE_COUNT = 6
JOKER_REARRANGE_EXHAUSTIVE_COUNT = 5
JOKER_REARRANGE_MIN_GAIN = 1
SHOP_VALUE_TOLERANCE = 0.25
SHOP_TARGET_SAFETY_BASE = 1.15
HAND_PACE_SAFETY_BASE = 1.05
BASE_INTEREST_CAP_MONEY = 25
VOUCHER_INTEREST_CAP_MONEY = {
    "Seed Money": 50,
    "Money Tree": 100,
}
MONEY_SCALING_RESERVE_TARGETS = {
    "Bull": 75,
    "Bootstraps": 75,
}
RARE_HAND_TYPES = frozenset(
    {
        HandType.FOUR_OF_A_KIND,
        HandType.FIVE_OF_A_KIND,
        HandType.FLUSH_HOUSE,
        HandType.FLUSH_FIVE,
    }
)
IMPOSSIBLE_STARTER_DECK_HANDS = frozenset(
    {
        HandType.FIVE_OF_A_KIND,
        HandType.FLUSH_HOUSE,
        HandType.FLUSH_FIVE,
    }
)
IMPOSSIBLE_HAND_MANIPULATION_GAP = {
    HandType.FIVE_OF_A_KIND: 1.5,
    HandType.FLUSH_HOUSE: 1.5,
    HandType.FLUSH_FIVE: 2.0,
}
RARE_RANK_HAND_TYPES = frozenset(
    {
        HandType.FOUR_OF_A_KIND,
        HandType.FIVE_OF_A_KIND,
        HandType.FLUSH_HOUSE,
        HandType.FLUSH_FIVE,
    }
)
RARE_FLUSH_HAND_TYPES = frozenset({HandType.FLUSH_HOUSE, HandType.FLUSH_FIVE})
RARE_HAND_JOKER_TARGETS = {
    "The Family": HandType.FOUR_OF_A_KIND,
}
RANK_DECK_MANIPULATION_TAROTS = {
    "Strength",
    "Death",
    "The Hanged Man",
}
FLUSH_DECK_MANIPULATION_TAROTS = {
    "Death",
    "The Hanged Man",
    "The Star",
    "The Moon",
    "The Sun",
    "The World",
}
RARE_HAND_SUPPORT_TAROTS = RANK_DECK_MANIPULATION_TAROTS | FLUSH_DECK_MANIPULATION_TAROTS
DEDICATED_TWO_PAIR_BUILD_JOKERS = {
    "Spare Trousers",
}
DEDICATED_PAIR_BUILD_JOKERS = {
    "The Duo",
    "Jolly Joker",
    "Sly Joker",
}
TWO_PAIR_SUPPORT_JOKERS = {
    "Mad Joker",
    "Clever Joker",
}


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
    _filled_last_joker_slot_in_shop: bool = field(default=False, init=False, repr=False)
    _blind_key: tuple[int | None, int, str, int] | None = field(default=None, init=False, repr=False)
    _played_hand_types_in_blind: tuple[HandType, ...] = field(default=(), init=False, repr=False)
    _discards_in_blind: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._fallback = RandomBot(seed=self.seed)

    def choose_action(self, state: GameState) -> Action:
        self._sync_shop_memory(state)
        self._sync_blind_memory(state)

        blind_select = _blind_select_action(state)
        if blind_select is not None:
            return blind_select

        cash_out = _first_action_of_type(state, ActionType.CASH_OUT)
        if cash_out is not None:
            return cash_out

        shop_action = _shop_action(state, self._shop_context())
        if shop_action is not None:
            self._record_shop_action(state, shop_action)
            return shop_action

        end_shop = _first_action_of_type(state, ActionType.END_SHOP)
        if end_shop is not None:
            return end_shop

        stale_empty_pack = _stale_empty_pack_action(state)
        if stale_empty_pack is not None:
            return stale_empty_pack

        pack_choice = _pack_choice_action(state, self._shop_context())
        if pack_choice is not None:
            self._record_shop_action(state, pack_choice)
            return pack_choice

        blind_context = self._blind_context(state)
        joker_rearrange = _joker_rearrange_action(state, blind_context)
        if joker_rearrange is not None:
            return joker_rearrange

        best_play = _best_play_action(state, blind_context)
        if best_play is None:
            return self._fallback.choose_action(state)

        action = _tactical_blind_action(state, best_play, blind_context)
        self._record_blind_action(state, action, blind_context)
        return action

    def _shop_context(self) -> _ShopContext:
        return _ShopContext(
            rerolls_in_shop=self._rerolls_in_shop,
            packs_opened_in_shop=self._packs_opened_in_shop,
            filled_last_joker_slot=self._filled_last_joker_slot_in_shop,
        )

    def _blind_context(self, state: GameState) -> _BlindContext:
        played_from_state = _played_hand_types_this_round(state)
        if state.blind == "The Mouth" and self._played_hand_types_in_blind:
            return _BlindContext(played_hand_types=self._played_hand_types_in_blind, discards_taken=self._discards_in_blind)
        if played_from_state:
            return _BlindContext(played_hand_types=played_from_state, discards_taken=self._discards_in_blind)
        return _BlindContext(played_hand_types=self._played_hand_types_in_blind, discards_taken=self._discards_in_blind)

    def _sync_shop_memory(self, state: GameState) -> None:
        if not (_has_shop_decision_surface(state) or state.modifiers.get("pack_cards")):
            return

        key = _shop_memory_key(state)
        if key != self._shop_key:
            self._shop_key = key
            self._rerolls_in_shop = 0
            self._packs_opened_in_shop = 0
            self._filled_last_joker_slot_in_shop = False

    def _record_shop_action(self, state: GameState, action: Action) -> None:
        if action.action_type == ActionType.REROLL:
            self._rerolls_in_shop += 1
        elif action.action_type == ActionType.OPEN_PACK:
            self._packs_opened_in_shop += 1
        elif action.action_type == ActionType.SELL:
            self._filled_last_joker_slot_in_shop = False
        elif action.action_type == ActionType.BUY:
            item = _shop_item_for_action(state, action)
            if (
                _normal_slot_joker_card(item)
                and _normal_joker_open_slots(state) <= 1
            ):
                self._filled_last_joker_slot_in_shop = True
        elif action.action_type == ActionType.CHOOSE_PACK_CARD:
            item = _shop_item_for_action(state, action)
            if (
                _normal_slot_joker_card(item)
                and _normal_joker_open_slots(state) <= 1
            ):
                self._filled_last_joker_slot_in_shop = True

    def _sync_blind_memory(self, state: GameState) -> None:
        if not (state.hand or state.current_score or state.hands_remaining):
            return

        key = _blind_memory_key(state)
        if key != self._blind_key:
            self._blind_key = key
            self._played_hand_types_in_blind = ()
            self._discards_in_blind = 0

    def _record_blind_action(self, state: GameState, action: Action, context: _BlindContext) -> None:
        if action.action_type == ActionType.DISCARD:
            self._discards_in_blind = context.discards_taken + 1
            return
        if action.action_type != ActionType.PLAY_HAND or not action.card_indices:
            return
        hand_type = _evaluate_play_action(state, action, context).hand_type
        self._played_hand_types_in_blind = (*context.played_hand_types, hand_type)


def _first_action_of_type(state: GameState, action_type: ActionType) -> Action | None:
    for action in state.legal_actions:
        if action.action_type == action_type:
            return action
    return None


def _blind_select_action(state: GameState) -> Action | None:
    select = _first_action_of_type(state, ActionType.SELECT_BLIND)
    if select is None:
        return None

    skip = _first_action_of_type(state, ActionType.SKIP_BLIND)
    if skip is not None and _throwback_skip_is_worth_it(state):
        return _annotated_action(skip, reason="throwback_skip")
    return select


def _throwback_skip_is_worth_it(state: GameState) -> bool:
    if not any(joker.name == "Throwback" for joker in state.jokers):
        return False
    if state.blind != "Small Blind" or state.ante < 2:
        return False
    if state.money < 10 and state.ante <= 3:
        return False
    current_score = _sample_build_score(state, state.jokers)
    hands = max(3.0, float(state.hands_remaining or 4))
    capacity = current_score * hands * 0.85
    target = max(float(state.required_score), _estimated_next_required_score(state)) * 1.2
    return capacity >= target


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


def _blind_memory_key(state: GameState) -> tuple[int | None, int, str, int]:
    return (state.seed, state.ante, state.blind, state.required_score)


def _pack_choice_action(state: GameState, context: _ShopContext | None = None) -> Action | None:
    context = context or _ShopContext()
    pack_cards = state.modifiers.get("pack_cards", ())
    if not pack_cards:
        return None
    best_action: Action | None = None
    best_value = 0.0
    skip_action = _pack_skip_action(state, has_pack_cards=bool(pack_cards))
    skip_value = _pack_skip_value(state)

    for action in state.legal_actions:
        if action.action_type != ActionType.CHOOSE_PACK_CARD or action.target_id == "skip":
            continue
        index = int(action.metadata.get("index", action.amount or 0))
        if index >= len(pack_cards):
            continue
        pack_card = pack_cards[index]
        if context.filled_last_joker_slot and _normal_slot_joker_card(pack_card):
            continue
        if not _pack_card_is_pickable(state, pack_card):
            continue
        target_indices = _pack_card_target_indices(state, pack_card)
        if _pack_card_requires_targets(pack_card) and not target_indices:
            continue
        value = _pack_card_value(state, pack_card)
        if value > best_value:
            best_action = _with_target_indices(action, target_indices)
            best_value = value

    if skip_action is not None and skip_value >= max(20.0, best_value - 4.0):
        return _annotated_action(skip_action, reason=f"pack_skip value={skip_value:.1f}")

    if best_action is not None and best_value >= 20:
        return _annotated_action(best_action, reason=f"pack_pick value={best_value:.1f}")

    if skip_action is not None:
        return skip_action
    return None


def _pack_skip_action(state: GameState, *, has_pack_cards: bool) -> Action | None:
    for action in state.legal_actions:
        if action.action_type == ActionType.CHOOSE_PACK_CARD and action.target_id == "skip":
            return action
    if has_pack_cards:
        return Action(ActionType.CHOOSE_PACK_CARD, target_id="skip", metadata={"kind": "skip", "index": True})
    return None


def _stale_empty_pack_action(state: GameState) -> Action | None:
    if state.phase != GamePhase.BOOSTER_OPENED:
        return None
    if state.pack or state.modifiers.get("pack_cards"):
        return None
    return _first_action_of_type(state, ActionType.NO_OP) or Action(ActionType.NO_OP)


def _pack_skip_value(state: GameState) -> float:
    return _red_card_pack_skip_value(state)


def _red_card_pack_skip_value(state: GameState) -> float:
    if not any(joker.name == "Red Card" for joker in state.jokers):
        return 0.0

    boosted_jokers = tuple(
        _joker_with_added_current_plus(joker, 3, suffix="mult") if joker.name == "Red Card" else joker
        for joker in state.jokers
    )
    current_score = _sample_build_score(state, state.jokers)
    boosted_score = _sample_build_score(replace(state, jokers=boosted_jokers), boosted_jokers)
    score_delta = max(0.0, boosted_score - current_score)
    value = 18.0 + min(90.0, score_delta / 8.0)
    if state.ante <= 2:
        value += 8.0
    return value


def _with_target_indices(action: Action, target_indices: tuple[int, ...]) -> Action:
    if not target_indices:
        return action
    return Action(
        action.action_type,
        card_indices=target_indices,
        target_id=action.target_id,
        amount=action.amount,
        metadata=action.metadata,
    )


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
    if best_action is not None and best_value + SHOP_VALUE_TOLERANCE >= threshold:
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
    if _normal_joker_open_slots(state) > 0:
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
        role_upgrade = _replacement_role_upgrade_bonus(state, weakest_joker, candidate, pressure)
        upgrade = (
            candidate_value
            + role_upgrade
            - weakest_value
            - max(0, cost - sell_value) * _replacement_cost_weight(pressure)
        )
        candidate_options.append(
            {
                "name": candidate.name,
                "cost": cost,
                "candidate_value": round(candidate_value, 2),
                "role_upgrade": round(role_upgrade, 2),
                "upgrade": round(upgrade, 2),
            }
        )
        if upgrade > best_upgrade:
            best_upgrade = upgrade
            best_label = candidate.name

    if best_upgrade >= _replacement_upgrade_threshold(pressure, state) and weakest_index in sell_actions:
        return _annotated_action(
            sell_actions[weakest_index],
            reason=(
                f"replace {weakest_joker.name} value={weakest_value:.1f} "
                f"with {best_label} upgrade={best_upgrade:.1f} pressure={pressure.ratio:.2f}"
            ),
            audit={
                "decision": "replace",
                "pressure": _pressure_payload(pressure),
                "threshold": round(_replacement_upgrade_threshold(pressure, state), 2),
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
        "money_plan": _money_plan_payload(state, pressure),
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
    item = _shop_item_for_action(state, action)
    if item is None:
        return None
    return _shop_item_payload(item)


def _shop_item_for_action(state: GameState, action: Action) -> object | None:
    kind = str(action.metadata.get("kind", action.target_id or ""))
    index = action.metadata.get("index", action.amount)
    try:
        item_index = int(index)
    except (TypeError, ValueError):
        item_index = -1
    if action.action_type == ActionType.BUY and kind == "voucher":
        return _indexed_shop_item(state.modifiers.get("voucher_cards", ()), item_index)
    if action.action_type == ActionType.BUY and kind == "card":
        return _indexed_shop_item(state.modifiers.get("shop_cards", ()), item_index)
    if action.action_type == ActionType.OPEN_PACK:
        return _indexed_shop_item(state.modifiers.get("booster_packs", ()), item_index)
    if action.action_type == ActionType.CHOOSE_PACK_CARD:
        return _indexed_shop_item(state.modifiers.get("pack_cards", ()), item_index)
    return None


def _indexed_shop_item(items: tuple[object, ...], index: int) -> object | None:
    if 0 <= index < len(items):
        return items[index]
    return None


def _indexed_shop_item_payload(items: tuple[object, ...], index: int) -> dict[str, object] | None:
    item = _indexed_shop_item(items, index)
    if item is None:
        return None
    return _shop_item_payload(item)


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


def _pressure_payload(pressure: _ShopPressure) -> dict[str, object]:
    payload: dict[str, float | str | None] = {
        "target_score": round(pressure.target_score, 2),
        "build_capacity": round(pressure.build_capacity, 2),
        "ratio": round(pressure.ratio, 4),
        "raw_ratio": round(pressure.raw_ratio, 4),
        "danger": round(pressure.danger, 4),
        "safe_margin": round(pressure.safe_margin, 4),
        "safety_multiplier": round(pressure.safety_multiplier, 3),
        "capacity_safety_factor": round(pressure.capacity_safety_factor, 3),
        "boss_name": pressure.boss_name,
        "boss_target_multiplier": round(pressure.boss_target_multiplier, 3),
        "boss_capacity_factor": round(pressure.boss_capacity_factor, 3),
    }
    return payload


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
        if context.filled_last_joker_slot and _normal_slot_joker_card(card):
            return 0.0
        if _buy_would_overfill_joker_slots(state, card):
            return 0.0
        value = _shop_card_value(state, card)
        value += _early_shop_safety_adjustment(state, card)
        if _is_joker_card(card):
            value += pressure.danger * 14
            value += _pressure_joker_role_bonus(state, _joker_from_shop_card(card), pressure)
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
        if _shop_pack_can_trigger_hidden_target_error(state, pack):
            return 0.0
        if state.money - _card_cost(pack) < 4 and pressure.ratio < 1.15:
            return 0.0
        if not _late_pack_is_worth_opening(state, pack, pressure, context):
            return 0.0
        return (
            _pack_value(state, pack)
            + pressure.danger * 16
            + _pressure_pack_bonus(state, pack, pressure)
            - _cost_penalty(state, pack, pressure)
        )
    if action.action_type == ActionType.REROLL:
        if not _early_reroll_is_allowed(state, pressure):
            return 0.0
        if state.money < _minimum_reroll_bank(state, pressure):
            return 0.0
        if pressure.ratio < 0.95 and state.money < 14:
            return 0.0
        if not _late_reroll_is_worth_it(state, pressure, profile, context):
            return 0.0
        if _normal_joker_open_slots(state) <= 0 and pressure.ratio < 1.1 and not _rich_late_role_hunt(profile):
            return 0.0
        if _visible_safety_pack_before_reroll(state, pressure, profile, context):
            return 0.0
        pressure_bonus = max(0, 4 - _normal_joker_slots_used(state)) * 7
        pressure_bonus += pressure.danger * 22
        pressure_bonus += _reroll_role_hunt_bonus(profile, pressure)
        if any(joker.name == "Flash Card" for joker in state.jokers):
            pressure_bonus += 26
        return 24 + pressure_bonus - _money_after_spend_penalty(state, 5, pressure)
    return 0.0


def _minimum_reroll_bank(state: GameState, pressure: _ShopPressure | None = None) -> int:
    if state.ante >= 4:
        reserve = _desired_money_reserve(state, pressure)
        if _normal_joker_open_slots(state) <= 0:
            reserve = max(reserve, _interest_cap_money(state))
        return reserve + 5
    if state.ante <= 1 and not _has_real_scoring_joker(state) and not _visible_early_power_path(state):
        return 5
    if state.ante <= 2 and not _has_real_scoring_joker(state) and pressure is not None and pressure.ratio >= 1.1:
        return 6
    if _has_money_scaling_joker(state):
        return min(_desired_money_reserve(state, pressure) + 5, 30)
    return 9


def _early_reroll_is_allowed(state: GameState, pressure: _ShopPressure) -> bool:
    if state.ante > 2:
        return True
    if _visible_early_power_path(state):
        return False
    if not _has_real_scoring_joker(state):
        return state.money >= _minimum_reroll_bank(state, pressure)
    if state.money < 9:
        return False
    return pressure.ratio >= 1.1 or not _has_real_scoring_joker(state)


def _visible_early_power_path(state: GameState) -> bool:
    shop_cards = state.modifiers.get("shop_cards", ())
    for card in shop_cards:
        cost = _card_cost(card)
        if cost > state.money or not _is_joker_card(card):
            continue
        joker = _joker_from_shop_card(card)
        roles = _joker_roles(joker)
        if roles & {"chips", "mult", "xmult", "scaling"}:
            return True
        if joker.name not in LOW_PRIORITY_JOKERS and JOKER_ECONOMY_VALUES.get(joker.name, 0) >= 18:
            return True

    for pack in state.modifiers.get("booster_packs", ()):
        if _card_cost(pack) <= state.money and "buffoon" in _card_label(pack).lower():
            return True
    return False


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
    if pressure.raw_ratio >= 1.05:
        return True
    if pressure.ratio >= 1.05 and _spendable_money(state) >= 20:
        return True
    if not _rich_late_role_hunt(profile):
        return False
    if _urgent_late_role_hunt(state, pressure, profile):
        return _spendable_money(state, pressure) >= 15
    if _normal_joker_open_slots(state) <= 0 and pressure.ratio < 0.58:
        return False
    return _spendable_money(state, pressure) >= 20


def _late_reroll_limit(state: GameState, pressure: _ShopPressure, profile: _BuildProfile) -> int:
    if state.ante < 5:
        return 99
    if pressure.raw_ratio >= 1.35:
        return 3 if _urgent_late_role_hunt(state, pressure, profile) else 2
    if pressure.raw_ratio >= 1.15:
        return 2
    if _urgent_late_role_hunt(state, pressure, profile):
        return 2
    if _rich_late_role_hunt(profile):
        return 1
    return 0


def _shop_card_value(state: GameState, card: object) -> float:
    if _is_joker_card(card):
        return _joker_card_value(state, card)
    if _is_black_hole_card(card):
        if not _has_consumable_room(state):
            return 0.0
        return _black_hole_card_value(state)
    if _is_planet_card(card):
        if not _has_consumable_room(state):
            return 0.0
        return _planet_card_value(state, card)
    if _is_tarot_card(card):
        if not _has_consumable_room(state):
            return 0.0
        if _pack_card_requires_targets(card) and not _target_required_tarot_is_supported(state, card):
            return 0.0
        return _tarot_card_value(state, card)
    if _is_playing_card(card):
        return _playing_card_shop_value(state, card)
    return 0.0


def _pack_card_value(state: GameState, card: object) -> float:
    if _pack_card_requires_targets(card) and not _target_required_tarot_is_supported(state, card):
        return 0.0
    if _is_black_hole_card(card):
        return _black_hole_card_value(state) + 100
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


def _pack_card_is_pickable(state: GameState, card: object) -> bool:
    if _is_black_hole_card(card):
        return True
    if _is_joker_card(card) and _buy_would_overfill_joker_slots(state, card):
        return False
    if _is_consumable_card(card) and not _has_consumable_room(state):
        return False
    return True


def _target_required_tarot_is_supported(state: GameState, card: object) -> bool:
    name = _card_label(card)
    rare_plan = _rare_hand_plan(state)
    if rare_plan is not None and _tarot_supports_rare_hand(name, rare_plan):
        return True
    preferred = _preferred_hand_type(state)
    if preferred not in FLUSH_ARCHETYPE_HANDS and _hand_archetype_support_count(state, HandType.FLUSH) <= 0:
        return False
    target_suit = SUIT_TAROT_TARGET_SUITS.get(name)
    if target_suit is None:
        return name in FLUSH_DECK_MANIPULATION_TAROTS
    dominant_suit = _dominant_suit(state)
    return dominant_suit in {None, target_suit} or sum(1 for card in state.hand if card.suit == target_suit) >= 2


def _pack_card_target_indices(state: GameState, card: object) -> tuple[int, ...]:
    if not _pack_card_requires_targets(card):
        return ()
    name = _card_label(card)
    target_suit = SUIT_TAROT_TARGET_SUITS.get(name)
    if target_suit is None or not _target_required_tarot_is_supported(state, card):
        return ()
    return _suit_tarot_target_indices(state, target_suit)


def _suit_tarot_target_indices(state: GameState, target_suit: str) -> tuple[int, ...]:
    if not state.hand:
        return ()

    candidates = [
        (index, card)
        for index, card in enumerate(state.hand)
        if card.suit != target_suit and not card.debuffed
    ]
    if not candidates:
        return ()

    preferred = _preferred_hand_type(state)
    keep_scores = _card_keep_scores(state.hand, preferred, state=state)
    ranked = sorted(
        candidates,
        key=lambda item: (RANK_VALUES.get(item[1].rank, 0), -keep_scores[item[0]]),
        reverse=True,
    )
    return tuple(index for index, _ in ranked[:3])


def _shop_buy_threshold(state: GameState, pressure: _ShopPressure) -> float:
    spendable_money = _spendable_money(state, pressure)
    profile = _build_profile(state)
    normal_jokers = _normal_joker_slots_used(state)
    if normal_jokers == 0:
        base = 18.0
    elif state.ante <= 2 and normal_jokers < 3:
        base = 24.0
    elif state.ante >= 5 and spendable_money >= 20:
        base = 16.0
    elif state.money >= 25:
        base = 22.0
    else:
        base = 30.0
    if pressure.ratio >= 1.05 and ("xmult" in profile.missing_roles or "scaling" in profile.missing_roles):
        base -= 5.0
    if state.ante >= 4 and spendable_money >= 4 and not _has_money_scaling_joker(state):
        if pressure.ratio >= 0.85:
            base -= 4.0
        if _normal_joker_open_slots(state) <= 0 and not profile.has_xmult:
            base -= 2.0
        if state.money >= 75 and ("xmult" in profile.missing_roles or "scaling" in profile.missing_roles):
            base -= 6.0
    if state.ante >= 4 and pressure.ratio >= 1.15:
        base -= 4.0
    safe_margin_weight = 8 if state.ante >= 5 and spendable_money >= 20 else 14
    return max(10.0, base - pressure.danger * 14 + pressure.safe_margin * safe_margin_weight)


def _replacement_upgrade_threshold(pressure: _ShopPressure, state: GameState | None = None) -> float:
    threshold = max(16.0, 28.0 - pressure.danger * 10 + pressure.safe_margin * 8)
    if state is not None and _urgent_late_role_hunt(state, pressure, _build_profile(state)):
        threshold -= 6.0
    return max(12.0, threshold)


def _replacement_cost_weight(pressure: _ShopPressure) -> float:
    return max(1.3, 2.5 - pressure.danger * 0.8 + pressure.safe_margin * 0.5)


def _replacement_role_upgrade_bonus(
    state: GameState,
    sold_joker: Joker,
    candidate: Joker,
    pressure: _ShopPressure,
) -> float:
    profile = _build_profile(state)
    sold_roles = _joker_roles(sold_joker)
    candidate_roles = _joker_roles(candidate)
    missing = set(profile.missing_roles)
    bonus = 0.0

    if "xmult" in candidate_roles and "xmult" in missing and "xmult" not in sold_roles:
        bonus += 30.0 * max(0.35, profile.role_deficit_ratio("xmult"))
    if "scaling" in candidate_roles and "scaling" in missing and "scaling" not in sold_roles:
        bonus += 24.0 * max(0.35, profile.role_deficit_ratio("scaling"))
    if _urgent_late_role_hunt(state, pressure, profile) and candidate_roles & {"xmult", "scaling"}:
        bonus += 14.0
    if sold_roles & {"xmult", "scaling"} and not candidate_roles & {"xmult", "scaling"}:
        bonus -= 24.0
    if sold_roles & {"chips", "mult"} and candidate_roles.isdisjoint({"chips", "mult", "xmult", "scaling"}):
        bonus -= 12.0
    if candidate_roles.isdisjoint({"xmult", "scaling"}) and _urgent_late_role_hunt(state, pressure, profile):
        bonus -= 14.0
    return bonus


def _shop_pressure(state: GameState) -> _ShopPressure:
    boss_name = _upcoming_boss_blind_name(state)
    boss_target_multiplier = _effective_boss_target_multiplier(state, boss_name)
    boss_capacity_factor = _weighted_boss_capacity_factor(state, boss_name)
    raw_target = _estimated_next_required_score(state)
    safety_multiplier = _shop_target_safety_multiplier(state)
    target = raw_target * safety_multiplier
    current_score = _sample_build_score(state, state.jokers)
    hands = float(state.hands_remaining or 4)
    preferred_hands = max(2.0, min(4.0, hands) - 0.5)
    raw_capacity = max(1.0, current_score * preferred_hands * 0.85)
    capacity_safety_factor = _shop_capacity_safety_factor(state) * boss_capacity_factor
    capacity = max(1.0, raw_capacity * capacity_safety_factor)
    raw_ratio = raw_target / raw_capacity
    ratio = max(target / capacity, _early_build_pressure_floor(state))
    return _ShopPressure(
        target_score=target,
        build_capacity=capacity,
        ratio=ratio,
        raw_ratio=raw_ratio,
        safety_multiplier=safety_multiplier,
        capacity_safety_factor=capacity_safety_factor,
        boss_name=boss_name,
        boss_target_multiplier=boss_target_multiplier,
        boss_capacity_factor=boss_capacity_factor,
    )


def _shop_target_safety_multiplier(state: GameState) -> float:
    multiplier = SHOP_TARGET_SAFETY_BASE
    if state.ante >= 3:
        multiplier += 0.10
    if state.ante >= 4:
        multiplier += 0.10
    if state.ante >= 5:
        multiplier += 0.05
    if _normal_joker_open_slots(state) <= 0 and not _has_money_scaling_joker(state):
        multiplier += 0.05
    if state.blind in {"The Wall", "The Needle"}:
        multiplier += 0.12
    elif state.blind in {"The Eye", "The Mouth"}:
        multiplier += 0.08
    elif state.blind in {"The Water", "The Arm"}:
        multiplier += 0.06
    boss_name = _upcoming_boss_blind_name(state)
    if boss_name is not None:
        multiplier += _boss_target_safety_bonus(boss_name) * _boss_preview_weight(state)
    return min(1.45, multiplier)


def _shop_capacity_safety_factor(state: GameState) -> float:
    if state.ante < 4:
        return 1.0
    if _has_money_scaling_joker(state):
        return 1.0

    profile = _build_profile(state)
    factor = 1.0
    if not profile.has_xmult:
        factor -= 0.10
    if not _has_planet_investment(state):
        factor -= 0.08
    if _normal_joker_open_slots(state) <= 0 and not _has_money_scaling_joker(state):
        factor -= 0.06
    if profile.preferred_hand in {
        HandType.FLUSH,
        HandType.STRAIGHT,
        HandType.FULL_HOUSE,
        HandType.THREE_OF_A_KIND,
        HandType.FOUR_OF_A_KIND,
        HandType.FIVE_OF_A_KIND,
        HandType.STRAIGHT_FLUSH,
        HandType.FLUSH_HOUSE,
        HandType.FLUSH_FIVE,
    }:
        factor -= 0.05
    if state.ante >= 5:
        factor -= 0.04
    return max(0.72, factor)


def _has_planet_investment(state: GameState) -> bool:
    return any(level > 1 for level in state.hand_levels.values())


def _early_build_pressure_floor(state: GameState) -> float:
    joker_count = _normal_joker_slots_used(state)
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
        boss_base = small * 2.0
        boss_score = _upcoming_boss_score(state)
        if boss_score > 0:
            return max(boss_base, boss_score)
        boss_name = _upcoming_boss_blind_name(state)
        return boss_base * _boss_score_target_multiplier(boss_name)
    if state.required_score > 0 and state.blind:
        return max(next_small, state.required_score * 1.25)
    if state.required_score > 0:
        return state.required_score * 1.5
    return small


def _upcoming_boss_blind_name(state: GameState) -> str | None:
    direct_keys = ("upcoming_boss", "next_boss", "boss_blind", "boss")
    for key in direct_keys:
        name = _blind_name_from_mapping(state.modifiers.get(key))
        if name:
            return name

    blinds = state.modifiers.get("blinds")
    if isinstance(blinds, dict):
        for key in ("boss", "Boss", "boss_blind"):
            name = _blind_name_from_mapping(blinds.get(key))
            if name:
                return name
        for value in blinds.values():
            if not isinstance(value, dict):
                continue
            blind_type = str(value.get("type", value.get("kind", ""))).upper()
            if blind_type == "BOSS":
                name = _blind_name_from_mapping(value)
                if name:
                    return name
    return None


def _upcoming_boss_score(state: GameState) -> float:
    direct_keys = ("upcoming_boss", "next_boss", "boss_blind", "boss")
    for key in direct_keys:
        score = _blind_score_from_mapping(state.modifiers.get(key))
        if score > 0:
            return score

    blinds = state.modifiers.get("blinds")
    if isinstance(blinds, dict):
        for key in ("boss", "Boss", "boss_blind"):
            score = _blind_score_from_mapping(blinds.get(key))
            if score > 0:
                return score
        for value in blinds.values():
            if not isinstance(value, dict):
                continue
            blind_type = str(value.get("type", value.get("kind", ""))).upper()
            if blind_type == "BOSS":
                score = _blind_score_from_mapping(value)
                if score > 0:
                    return score
    return 0.0


def _blind_name_from_mapping(value: object) -> str | None:
    if isinstance(value, str):
        return value or None
    if not isinstance(value, dict):
        return None
    name = str(value.get("name", value.get("label", "")))
    return name or None


def _blind_score_from_mapping(value: object) -> float:
    if not isinstance(value, dict):
        return 0.0
    for key in ("score", "required_score", "score_required", "chips"):
        try:
            raw = value.get(key)
            if raw is not None:
                return float(raw)
        except (TypeError, ValueError):
            continue
    return 0.0


def _effective_boss_target_multiplier(state: GameState, boss_name: str | None) -> float:
    if state.blind != "Big Blind" or _upcoming_boss_score(state) > 0:
        return 1.0
    return _boss_score_target_multiplier(boss_name)


def _boss_score_target_multiplier(boss_name: str | None) -> float:
    if boss_name == "The Wall":
        return 2.0
    if boss_name == "Violet Vessel":
        return 3.0
    return 1.0


def _boss_target_safety_bonus(boss_name: str) -> float:
    if boss_name in {"The Wall", "The Needle", "Violet Vessel"}:
        return 0.14
    if boss_name in {"The Eye", "The Mouth", "The Pillar", "The Psychic", "The Flint"}:
        return 0.10
    if boss_name in {"The Tooth", "The Water", "The Arm", "The Manacle"}:
        return 0.07
    if boss_name in {"The Club", "The Goad", "The Head", "The Window"}:
        return 0.06
    return 0.04


def _weighted_boss_capacity_factor(state: GameState, boss_name: str | None) -> float:
    if boss_name is None:
        return 1.0
    weight = _boss_preview_weight(state)
    if weight <= 0:
        return 1.0
    factor = _boss_capacity_factor(state, boss_name)
    return 1.0 - ((1.0 - factor) * weight)


def _boss_preview_weight(state: GameState) -> float:
    if state.blind == "Big Blind":
        return 1.0
    if state.blind == "Small Blind":
        return 0.45
    return 0.0


def _boss_capacity_factor(state: GameState, boss_name: str) -> float:
    if boss_name == "The Flint":
        return 0.66
    if boss_name == "The Needle":
        return 0.74
    if boss_name in {"The Eye", "The Mouth"}:
        return 0.8
    if boss_name == "The Pillar":
        return 0.82
    if boss_name == "The Psychic":
        return 0.86
    if boss_name in {"The Water", "The Manacle"}:
        return 0.88
    if boss_name == "The Arm":
        return 0.9
    if boss_name == "The Tooth":
        return 0.95
    if boss_name in {"The Club", "The Goad", "The Head", "The Window"}:
        return _suit_boss_capacity_factor(state, boss_name)
    return 1.0


def _suit_boss_capacity_factor(state: GameState, boss_name: str) -> float:
    debuffed = debuffed_suits_for_blind(boss_name)
    if not debuffed:
        return 1.0
    dominant = _dominant_suit(state)
    if dominant is not None and _normalize_suit(dominant) in debuffed:
        return 0.76
    if _preferred_hand_type(state) in FLUSH_ARCHETYPE_HANDS:
        return 0.86
    return 0.92


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


def _is_black_hole_card(card: object) -> bool:
    return _card_label(card) == "Black Hole" or _card_key(card) == "c_black_hole"


def _is_tarot_card(card: object) -> bool:
    return _card_set(card) == "TAROT"


def _is_consumable_card(card: object) -> bool:
    return _is_tarot_card(card) or _is_planet_card(card) or _card_set(card) == "SPECTRAL"


def _is_playing_card(card: object) -> bool:
    if not isinstance(card, dict):
        return False
    card_set = _card_set(card)
    return card_set in {"DEFAULT", "ENHANCED", "PLAYING_CARD"} or (
        _card_rank(card) != "" and _card_suit(card) != ""
    )


def _joker_card_value(state: GameState, card: object) -> float:
    joker = _joker_from_shop_card(card)
    if _joker_would_overfill_slots(state, joker):
        return 0.0

    name = joker.name
    if _joker_stencil_would_fill_slots(state, joker):
        return 0.0
    sample_delta = _sample_score_delta_for_joker(state, joker) * _joker_sample_reliability(state, joker)
    value = sample_delta * 0.08
    value += _normal_joker_open_slots(state) * 6
    value += max(0, 4 - state.ante) * _early_power_bonus(name)
    value += _joker_heuristic_value(state, joker)
    value += _joker_role_bonus(state, joker)
    value -= _unsupported_two_pair_joker_penalty(state, name)
    value -= _unsupported_rare_joker_extra_penalty(state, name)
    value += _edition_bonus(joker.edition)

    if any(existing.name == name for existing in state.jokers):
        value -= 18
    if _normal_joker_slots_used(state) == 0 and value < 35:
        value += 10
    return value


def _candidate_joker_value_for_replacement(state: GameState, joker: Joker) -> float:
    if _joker_stencil_would_fill_slots(state, joker):
        return 0.0
    sample_gain = _sample_score_gain_for_joker(state, joker) * _joker_sample_reliability(state, joker)
    value = sample_gain * 0.08
    value += _joker_heuristic_value(state, joker)
    value += _joker_role_bonus(state, joker)
    value -= _unsupported_two_pair_joker_penalty(state, joker.name)
    value -= _unsupported_rare_joker_extra_penalty(state, joker.name)
    value += _edition_bonus(joker.edition)
    if any(existing.name == joker.name for existing in state.jokers):
        value -= 18
    return value


def _owned_joker_value(state: GameState, joker: Joker, *, remove_index: int) -> float:
    without = _jokers_after_sell_for_scoring(state, remove_index=remove_index)
    score_loss = max(0.0, _sample_build_score(state, state.jokers) - _sample_build_score(state, without))
    value = score_loss * 0.08
    value += _joker_heuristic_value(state, joker) * 0.75
    value += _owned_role_value(state, joker, remove_index=remove_index)
    value += _edition_bonus(joker.edition)
    value += (joker.sell_value or 0) * 1.5
    if joker.name in LOW_PRIORITY_JOKERS:
        value -= 20
    if joker.name in TWO_PAIR_SUPPORT_JOKERS and not _has_dedicated_two_pair_plan(state):
        value -= 45
    return value


def _sample_score_gain_for_joker(state: GameState, joker: Joker) -> float:
    return max(0.0, _sample_score_delta_for_joker(state, joker))


def _sample_score_delta_for_joker(state: GameState, joker: Joker) -> float:
    current = _sample_build_score(state, state.jokers)
    with_candidate = _sample_build_score(state, _jokers_after_buy_for_scoring(state, joker))
    return with_candidate - current


def _jokers_after_buy_for_scoring(state: GameState, joker: Joker) -> tuple[Joker, ...]:
    jokers = (*state.jokers, joker)
    if not any(existing.name == "Joker Stencil" for existing in jokers):
        return jokers

    normal_slots_used = sum(1 for existing in jokers if _uses_normal_joker_slot(existing))
    stencil_xmult = max(1.0, float(_normal_joker_slot_limit(state) + 1 - normal_slots_used))
    return tuple(
        _joker_with_current_xmult(existing, stencil_xmult)
        if existing.name == "Joker Stencil"
        else existing
        for existing in jokers
    )


def _jokers_after_sell_for_scoring(state: GameState, *, remove_index: int) -> tuple[Joker, ...]:
    jokers = tuple(existing for index, existing in enumerate(state.jokers) if index != remove_index)
    if not any(existing.name == "Joker Stencil" for existing in jokers):
        return jokers

    normal_slots_used = sum(1 for existing in jokers if _uses_normal_joker_slot(existing))
    stencil_xmult = max(1.0, float(_normal_joker_slot_limit(state) + 1 - normal_slots_used))
    return tuple(
        _joker_with_current_xmult(existing, stencil_xmult)
        if existing.name == "Joker Stencil"
        else existing
        for existing in jokers
    )


def _uses_normal_joker_slot(joker: Joker) -> bool:
    return "negative" not in (joker.edition or "").lower()


def _normal_joker_slots_used(state: GameState) -> int:
    return sum(1 for joker in state.jokers if _uses_normal_joker_slot(joker))


def _normal_joker_slot_limit(state: GameState) -> int:
    for key in ("joker_slot_limit", "joker_slots"):
        raw = state.modifiers.get(key)
        try:
            if raw is not None:
                return max(0, int(raw))
        except (TypeError, ValueError):
            continue
    return 5


def _normal_joker_open_slots(state: GameState) -> int:
    return max(0, _normal_joker_slot_limit(state) - _normal_joker_slots_used(state))


def _buy_would_overfill_joker_slots(state: GameState, card: object) -> bool:
    return _is_joker_card(card) and _joker_would_overfill_slots(state, _joker_from_shop_card(card))


def _joker_would_overfill_slots(state: GameState, joker: Joker) -> bool:
    return _uses_normal_joker_slot(joker) and _normal_joker_open_slots(state) <= 0


def _normal_slot_joker_card(card: object) -> bool:
    return _is_joker_card(card) and _uses_normal_joker_slot(_joker_from_shop_card(card))


def _joker_with_current_xmult(joker: Joker, xmult: float) -> Joker:
    metadata = dict(joker.metadata)
    effect = _joker_effect_text(joker)
    replacement = f"Currently X{_format_xmult(xmult)}"
    if effect:
        effect = re.sub(
            r"currently\s+x\s*[0-9]+(?:\.[0-9]+)?",
            replacement,
            effect,
            flags=re.IGNORECASE,
        )
        if replacement not in effect:
            effect = f"{effect} ({replacement})"
    else:
        effect = replacement

    value = metadata.get("value")
    if isinstance(value, dict):
        value = dict(value)
        value["effect"] = effect
        metadata["value"] = value
    else:
        metadata["effect"] = effect
    return Joker(joker.name, edition=joker.edition, sell_value=joker.sell_value, metadata=metadata)


def _joker_with_added_current_plus(joker: Joker, amount: int, *, suffix: str) -> Joker:
    metadata = dict(joker.metadata)
    effect = _joker_effect_text(joker)
    current = _joker_current_plus_value(joker, suffix=suffix)
    new_value = max(0, current + amount)
    sign = "+" if new_value >= 0 else ""
    replacement = f"Currently {sign}{new_value} {suffix.title()}"
    if effect:
        effect = re.sub(
            rf"currently\s+[+-]?\d+(?:\s+{re.escape(suffix)})?",
            replacement,
            effect,
            flags=re.IGNORECASE,
        )
        if replacement not in effect:
            effect = f"{effect} ({replacement})"
    else:
        effect = replacement

    value = metadata.get("value")
    if isinstance(value, dict):
        value = dict(value)
        value["effect"] = effect
        metadata["value"] = value
    else:
        metadata["effect"] = effect
    return Joker(joker.name, edition=joker.edition, sell_value=joker.sell_value, metadata=metadata)


def _joker_with_added_current_xmult(joker: Joker, amount: float, *, minimum: float = 1.0) -> Joker:
    current = _joker_current_xmult_value(joker)
    return _joker_with_current_xmult(joker, max(minimum, current + amount))


def _joker_current_plus_value(joker: Joker, *, suffix: str) -> int:
    effect = _joker_effect_text(joker).replace("$", " ").replace("(", " ").replace(")", " ")
    match = re.search(
        rf"currently\s+([+-]?\d+)(?:\s+{re.escape(suffix)})?",
        effect,
        flags=re.IGNORECASE,
    )
    return int(match.group(1)) if match else 0


def _joker_current_xmult_value(joker: Joker) -> float:
    effect = _joker_effect_text(joker)
    match = re.search(r"currently\s+x\s*([0-9]+(?:\.[0-9]+)?)", effect, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    match = re.search(r"\bx\s*([0-9]+(?:\.[0-9]+)?)\s*mult", effect, flags=re.IGNORECASE)
    return float(match.group(1)) if match else 1.0


def _joker_effect_text(joker: Joker) -> str:
    value = joker.metadata.get("value")
    if isinstance(value, dict):
        return str(value.get("effect", ""))
    return str(joker.metadata.get("effect", ""))


def _format_xmult(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:.2f}".rstrip("0").rstrip(".")


def _sample_build_score(state: GameState, jokers: tuple[Joker, ...]) -> float:
    scoring_state = replace(state, jokers=jokers)
    weighted_total = 0.0
    total_weight = 0.0
    raw_scores: list[float] = []

    for sample in _score_samples_for_state(scoring_state):
        score = float(
            evaluate_played_cards(
                sample.cards,
                scoring_state.hand_levels,
                debuffed_suits=debuffed_suits_for_blind(scoring_state.blind),
                blind_name=scoring_state.blind,
                jokers=jokers,
                discards_remaining=scoring_state.discards_remaining,
                hands_remaining=max(1, scoring_state.hands_remaining),
                held_cards=sample.held_cards,
                deck_size=max(30, scoring_state.deck_size),
                money=scoring_state.money,
                played_hand_counts=_played_hand_counts(scoring_state),
            ).score
        )
        weighted_total += score * sample.weight
        total_weight += sample.weight
        raw_scores.append(score)

    visible_score = _visible_hand_sample_score(scoring_state, jokers)
    if visible_score > 0:
        weighted_total += visible_score * 1.15
        total_weight += 1.15
        raw_scores.append(float(visible_score))

    if total_weight <= 0 or not raw_scores:
        return 0.0
    expected = weighted_total / total_weight
    average_top = sum(sorted(raw_scores, reverse=True)[:3]) / min(3, len(raw_scores))
    return (expected * 0.78) + (average_top * 0.22)


def _score_samples_for_state(state: GameState) -> tuple["_SampleHand", ...]:
    preferred = _preferred_hand_type(state)
    samples = list(WHITE_STAKE_SAMPLE_HANDS)
    samples.extend(_archetype_score_samples(state, preferred))
    return tuple(samples)


def _archetype_score_samples(state: GameState, preferred: HandType | None) -> tuple["_SampleHand", ...]:
    if preferred == HandType.PAIR:
        return (
            _SampleHand((Card("3", "S"), Card("3", "H")), (Card("9", "D"), Card("5", "C")), weight=1.2),
            _SampleHand((Card("8", "S"), Card("8", "D")), (Card("K", "C"), Card("4", "H")), weight=1.0),
            _SampleHand((Card("4", "S"), Card("4", "H"), Card("9", "D"), Card("9", "C")), weight=0.6),
        )
    if preferred == HandType.TWO_PAIR:
        return (
            _SampleHand((Card("4", "S"), Card("4", "H"), Card("9", "D"), Card("9", "C")), weight=1.4),
            _SampleHand((Card("6", "S"), Card("6", "D"), Card("J", "H"), Card("J", "C")), weight=1.0),
            _SampleHand((Card("7", "S"), Card("7", "H")), (Card("Q", "D"), Card("3", "C")), weight=0.6),
        )
    if preferred in {HandType.THREE_OF_A_KIND, HandType.FULL_HOUSE}:
        return (
            _SampleHand((Card("7", "S"), Card("7", "H"), Card("7", "D")), weight=1.0),
            _SampleHand((Card("6", "S"), Card("6", "H"), Card("6", "D"), Card("J", "S"), Card("J", "C")), weight=0.8),
            _SampleHand((Card("5", "S"), Card("5", "H")), (Card("Q", "D"), Card("4", "C")), weight=0.8),
        )
    if preferred in {HandType.FLUSH, HandType.STRAIGHT_FLUSH, HandType.FLUSH_HOUSE, HandType.FLUSH_FIVE}:
        dominant = _dominant_suit(state) or "H"
        samples = [
            _SampleHand(
                (
                    Card("A", dominant),
                    Card("Q", dominant),
                    Card("9", dominant),
                    Card("6", dominant),
                    Card("3", dominant),
                ),
                weight=1.4,
            ),
            _SampleHand(
                (
                    Card("K", dominant),
                    Card("J", dominant),
                    Card("8", dominant),
                    Card("5", dominant),
                    Card("2", dominant),
                ),
                weight=1.0,
            ),
        ]
        if _rare_hand_deck_manipulation_need(state, preferred) <= 0:
            samples.append(
                _SampleHand(
                    (
                        Card("7", dominant),
                        Card("7", dominant),
                        Card("7", dominant),
                        Card("4", dominant),
                        Card("4", dominant),
                    ),
                    weight=0.6,
                )
            )
        return tuple(samples)
    if preferred in {HandType.STRAIGHT, HandType.STRAIGHT_FLUSH}:
        return (
            _SampleHand((Card("9", "S"), Card("8", "H"), Card("7", "D"), Card("6", "C"), Card("5", "S")), weight=1.35),
            _SampleHand((Card("A", "S"), Card("K", "H"), Card("Q", "D"), Card("J", "C"), Card("10", "S")), weight=0.9),
            _SampleHand((Card("6", "S"), Card("6", "H")), (Card("Q", "D"), Card("4", "C")), weight=0.6),
        )
    if preferred in {HandType.FOUR_OF_A_KIND, HandType.FIVE_OF_A_KIND}:
        if _rare_hand_deck_manipulation_need(state, preferred) > 0:
            return (
                _SampleHand((Card("8", "S"), Card("8", "H"), Card("8", "D")), weight=0.8),
                _SampleHand((Card("5", "S"), Card("5", "H")), (Card("Q", "D"), Card("4", "C")), weight=0.8),
            )
        return (
            _SampleHand((Card("8", "S"), Card("8", "H"), Card("8", "D"), Card("8", "C")), weight=1.0),
            _SampleHand((Card("6", "S"), Card("6", "H"), Card("6", "D"), Card("J", "S"), Card("J", "C")), weight=0.6),
        )
    return ()


def _visible_hand_sample_score(state: GameState, jokers: tuple[Joker, ...]) -> int:
    if not state.hand:
        return 0
    return best_play_from_hand(
        state.hand,
        state.hand_levels,
        debuffed_suits=debuffed_suits_for_blind(state.blind),
        blind_name=state.blind,
        jokers=jokers,
        discards_remaining=state.discards_remaining,
        hands_remaining=max(1, state.hands_remaining),
        deck_size=max(30, state.deck_size),
        money=state.money,
    ).score


def _joker_heuristic_value(state: GameState, joker: Joker) -> float:
    name = joker.name
    if _joker_stencil_would_fill_slots(state, joker):
        return 0.0
    value = 0.0
    value += JOKER_BASE_VALUES.get(name, 0)
    value += JOKER_SCALING_VALUES.get(name, 0) * (1 + max(0, 4 - state.ante) * 0.15)
    value += JOKER_ECONOMY_VALUES.get(name, 0)
    if name == "Hallucination":
        value += _hallucination_utility_bonus(state)
    value += _build_synergy_value(state, name)
    value -= _build_conflict_penalty(state, name)
    value -= _rare_hand_inconsistency_penalty_for_joker(state, name)
    value -= _narrow_rank_inconsistency_penalty_for_joker(state, name)
    if name in GLASS_CANNON_JOKERS and state.ante <= 2:
        value += 8
    if name in LOW_PRIORITY_JOKERS and len(state.jokers) >= 2:
        value -= 16
    return value


def _hallucination_utility_bonus(state: GameState) -> float:
    if not _has_consumable_room(state):
        return -24.0
    if not _has_real_scoring_joker(state):
        return 0.0
    bonus = 18.0
    if state.ante <= 3:
        bonus += 26.0
    affordable_packs = sum(
        1
        for pack in state.modifiers.get("booster_packs", ())
        if _card_cost(pack) <= max(0, state.money)
    )
    bonus += min(18.0, affordable_packs * 9.0)
    return bonus


def _joker_stencil_would_fill_slots(state: GameState, joker: Joker) -> bool:
    return joker.name == "Joker Stencil" and _normal_joker_open_slots(state) <= 1


def _joker_sample_reliability(state: GameState, joker: Joker) -> float:
    primary = JOKER_PRIMARY_HAND.get(joker.name)
    if primary == HandType.TWO_PAIR and not _has_dedicated_two_pair_plan(state):
        return 0.65
    if primary in {HandType.THREE_OF_A_KIND, HandType.FOUR_OF_A_KIND, HandType.FIVE_OF_A_KIND}:
        wanted = set(JOKER_HAND_SYNERGY.get(joker.name, ()))
        if not _hand_has_natural_support((*state.hand, *state.known_deck), wanted):
            return 0.35 if state.ante >= 3 else 0.55
    return 1.0


def _build_profile(state: GameState) -> _BuildProfile:
    joker_roles = [_joker_roles(joker) for joker in state.jokers]
    preferred = _preferred_hand_type(state)
    role_scores = _build_role_scores(state)
    return _BuildProfile(
        preferred_hand=preferred,
        archetype=_build_archetype(state, preferred),
        chip_score=role_scores["chips"],
        mult_score=role_scores["mult"],
        xmult_score=role_scores["xmult"],
        scaling_score=role_scores["scaling"],
        economy_score=role_scores["economy"],
        has_chips=role_scores["chips"] >= _role_requirement("chips", state.ante),
        has_mult=role_scores["mult"] >= _role_requirement("mult", state.ante),
        has_xmult=role_scores["xmult"] >= _role_requirement("xmult", state.ante),
        has_scaling=role_scores["scaling"] >= _role_requirement("scaling", state.ante),
        has_economy=(
            state.money >= _interest_cap_money(state)
            or any("economy" in roles for roles in joker_roles)
            or role_scores["economy"] >= _role_requirement("economy", state.ante)
        ),
        open_joker_slots=_normal_joker_open_slots(state),
        money=state.money,
        spendable_money=_spendable_money(state),
        ante=state.ante,
    )


def _build_profile_payload(profile: _BuildProfile) -> dict[str, object]:
    return {
        "preferred_hand": profile.preferred_hand.value if profile.preferred_hand is not None else None,
        "archetype": profile.archetype,
        "role_scores": {
            "chips": round(profile.chip_score, 2),
            "mult": round(profile.mult_score, 2),
            "xmult": round(profile.xmult_score, 2),
            "scaling": round(profile.scaling_score, 2),
            "economy": round(profile.economy_score, 2),
        },
        "role_requirements": {
            "chips": round(profile.role_requirement("chips"), 2),
            "mult": round(profile.role_requirement("mult"), 2),
            "xmult": round(profile.role_requirement("xmult"), 2),
            "scaling": round(profile.role_requirement("scaling"), 2),
            "economy": round(profile.role_requirement("economy"), 2),
        },
        "has_chips": profile.has_chips,
        "has_mult": profile.has_mult,
        "has_xmult": profile.has_xmult,
        "has_scaling": profile.has_scaling,
        "has_economy": profile.has_economy,
        "missing_roles": list(profile.missing_roles),
        "open_joker_slots": profile.open_joker_slots,
        "money": profile.money,
        "spendable_money": profile.spendable_money,
        "ante": profile.ante,
    }


def _role_requirement(role: str, ante: int) -> float:
    if role == "chips":
        if ante <= 2:
            return 28.0
        if ante <= 4:
            return 55.0
        return 85.0
    if role == "mult":
        if ante <= 2:
            return 8.0
        if ante <= 4:
            return 16.0
        return 24.0
    if role == "xmult":
        return 26.0 if ante <= 4 else 34.0
    if role == "scaling":
        return 22.0 if ante <= 4 else 30.0
    if role == "economy":
        return 18.0
    return 0.0


def _build_role_scores(state: GameState) -> dict[str, float]:
    scores = {"chips": 0.0, "mult": 0.0, "xmult": 0.0, "scaling": 0.0, "economy": 0.0}
    for joker in state.jokers:
        joker_scores = _joker_role_scores(joker)
        for role, value in joker_scores.items():
            scores[role] += value
    scores["economy"] += min(18.0, max(0.0, state.money - 5) * 0.6)
    if state.money >= _interest_cap_money(state):
        scores["economy"] += 18.0
    return scores


def _joker_role_scores(joker: Joker) -> dict[str, float]:
    name = joker.name
    scores = {
        "chips": float(_edition_chips_value(joker.edition) + _joker_current_plus_value(joker, suffix="chips")),
        "mult": float(_edition_mult_value(joker.edition) + _joker_current_plus_value(joker, suffix="mult")),
        "xmult": max(0.0, (_edition_xmult_value(joker.edition) - 1.0) * 42.0),
        "scaling": 0.0,
        "economy": float(JOKER_ECONOMY_VALUES.get(name, 0)),
    }

    current_xmult = _joker_current_xmult_value(joker)
    if current_xmult > 1.0:
        scores["xmult"] += (current_xmult - 1.0) * 42.0

    scores["chips"] += _static_chip_role_score(name)
    scores["mult"] += _static_mult_role_score(name)
    scores["xmult"] += _static_xmult_role_score(name)
    if name in SCALING_JOKERS or name in JOKER_SCALING_VALUES:
        scores["scaling"] += float(JOKER_SCALING_VALUES.get(name, 24))
        if name in {"Green Joker", "Ride the Bus", "Square Joker", "Runner", "Spare Trousers"}:
            scores["scaling"] += max(0.0, _joker_current_plus_value(joker, suffix="mult") * 0.8)
            scores["scaling"] += max(0.0, _joker_current_plus_value(joker, suffix="chips") * 0.25)
        if current_xmult > 1.0 and name in {"Hologram", "Constellation", "Lucky Cat", "Glass Joker", "Campfire"}:
            scores["scaling"] += min(24.0, (current_xmult - 1.0) * 22.0)
    return scores


def _static_chip_role_score(name: str) -> float:
    if name == "Stuntman":
        return 240.0
    if name == "Bull":
        return 95.0
    if name == "Blue Joker":
        return 70.0
    if name in {"Banner", "Ice Cream", "Runner", "Square Joker", "Wee Joker", "Castle"}:
        return 55.0
    if name in CHIP_JOKERS:
        return 34.0
    return 0.0


def _static_mult_role_score(name: str) -> float:
    if name == "Joker":
        return 4.0
    if name in {"Gros Michel", "Popcorn", "Mystic Summit", "Half Joker"}:
        return 16.0
    if name in {"Abstract Joker", "Swashbuckler", "Bootstraps"}:
        return 14.0
    if name in {"Green Joker", "Ride the Bus", "Spare Trousers", "Red Card", "Flash Card"}:
        return 10.0
    if name in MULT_JOKERS:
        return 8.0
    return 0.0


def _static_xmult_role_score(name: str) -> float:
    if name == "Loyalty Card":
        return 0.0
    if name in {"Cavendish", "Joker Stencil"}:
        return 70.0
    if name in {"The Duo", "The Trio", "The Family", "The Order", "The Tribe", "Seeing Double", "Blackboard"}:
        return 42.0
    if name in {"Acrobat", "Card Sharp", "Driver's License", "Ancient Joker", "The Idol", "Flower Pot"}:
        return 36.0
    if name in XMULT_JOKERS:
        return 30.0
    return 0.0


def _joker_roles(joker: Joker) -> frozenset[str]:
    roles: set[str] = set()
    name = joker.name
    if name in CHIP_JOKERS:
        roles.add("chips")
    if name in MULT_JOKERS:
        roles.add("mult")
    if name in XMULT_JOKERS and name != "Loyalty Card":
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
    if _joker_stencil_would_fill_slots(state, joker):
        return -30.0
    profile = _build_profile(state)
    roles = _joker_roles(joker)
    bonus = 0.0
    for role in profile.missing_roles:
        if role in roles:
            bonus += ROLE_MISSING_BONUSES[role] * max(0.35, profile.role_deficit_ratio(role))

    if profile.late and profile.rich:
        if "xmult" in roles and not profile.has_xmult:
            bonus += 22 * max(0.4, profile.role_deficit_ratio("xmult"))
        if "scaling" in roles and not profile.has_scaling:
            bonus += 16 * max(0.4, profile.role_deficit_ratio("scaling"))
        if "economy" in roles and profile.has_economy:
            bonus -= 10
    if profile.ante <= 2 and ("chips" in roles or "mult" in roles):
        bonus += 6
    if profile.open_joker_slots <= 1 and roles.isdisjoint({"xmult", "scaling"}) and profile.late:
        bonus -= 12
    return bonus


def _pressure_joker_role_bonus(state: GameState, joker: Joker, pressure: _ShopPressure) -> float:
    if pressure.ratio < 0.95:
        return 0.0
    rare_target = RARE_HAND_JOKER_TARGETS.get(joker.name)
    if rare_target is not None and _rare_hand_deck_manipulation_need(state, rare_target) > 0:
        return 0.0

    profile = _build_profile(state)
    roles = _joker_roles(joker)
    bonus = 0.0
    if "scaling" in roles and not profile.has_scaling:
        deficit = max(0.35, profile.role_deficit_ratio("scaling"))
        bonus += (18.0 + pressure.danger * 28.0 + max(0, state.ante - 2) * 4.0) * deficit
        if _urgent_late_role_hunt(state, pressure, profile):
            bonus += 16.0 * deficit
    if "xmult" in roles and (not profile.has_xmult or pressure.ratio >= 1.15):
        deficit = max(0.35, profile.role_deficit_ratio("xmult"))
        bonus += (10.0 + pressure.danger * 18.0) * deficit
        if _urgent_late_role_hunt(state, pressure, profile):
            bonus += 18.0 * deficit
    if roles.isdisjoint({"xmult", "scaling"}) and profile.open_joker_slots <= 1 and pressure.ratio >= 1.1:
        bonus -= 10.0
    if roles.isdisjoint({"xmult", "scaling"}) and _urgent_late_role_hunt(state, pressure, profile):
        bonus -= 12.0
    if joker.name in TWO_PAIR_SUPPORT_JOKERS and not _has_dedicated_two_pair_plan(state):
        bonus -= 12.0
    return bonus


def _pressure_pack_bonus(state: GameState, pack: object, pressure: _ShopPressure) -> float:
    name = _card_label(pack).lower()
    profile = _build_profile(state)
    bonus = 0.0
    if pressure.ratio >= 1.0:
        if "buffoon" in name and ("xmult" in profile.missing_roles or "scaling" in profile.missing_roles):
            bonus += 8.0 + pressure.danger * 16.0
        if "celestial" in name and not profile.has_scaling:
            bonus += 8.0 + pressure.danger * 10.0
    if _urgent_late_role_hunt(state, pressure, profile):
        if "buffoon" in name:
            bonus += 14.0
        elif "celestial" in name:
            bonus += 8.0
    bonus += _shop_safety_pack_bonus(state, pack, pressure, profile)
    return bonus


def _visible_safety_pack_before_reroll(
    state: GameState,
    pressure: _ShopPressure,
    profile: _BuildProfile,
    context: _ShopContext,
) -> bool:
    if state.ante < 4 or _normal_joker_open_slots(state) > 0:
        return False

    packs = state.modifiers.get("booster_packs", ())
    for action in state.legal_actions:
        if action.action_type != ActionType.OPEN_PACK:
            continue
        index = int(action.metadata.get("index", action.amount or 0))
        if index >= len(packs):
            continue
        pack = packs[index]
        if _shop_pack_can_trigger_hidden_target_error(state, pack):
            continue
        if not _late_pack_is_worth_opening(state, pack, pressure, context):
            continue
        if _shop_safety_pack_bonus(state, pack, pressure, profile) > 0:
            return True
    return False


def _shop_safety_pack_bonus(
    state: GameState,
    pack: object,
    pressure: _ShopPressure,
    profile: _BuildProfile,
) -> float:
    if state.ante < 4 or _has_money_scaling_joker(state):
        return 0.0

    cost = _card_cost(pack)
    if cost <= 0 or state.money - cost < _interest_cap_money(state):
        return 0.0
    if pressure.ratio < 0.78 and profile.has_xmult and profile.has_scaling and _has_planet_investment(state):
        return 0.0

    name = _card_label(pack).lower()
    bonus = 0.0
    if "arcana" in name:
        bonus += 8.0
    elif "celestial" in name:
        bonus += 6.0
    elif "standard" in name:
        bonus += 4.0
    elif "spectral" in name:
        bonus += 3.0
    elif "buffoon" in name and ("xmult" in profile.missing_roles or "scaling" in profile.missing_roles):
        bonus += 5.0

    if bonus <= 0:
        return 0.0
    if not profile.has_xmult:
        bonus += 3.0
    if not _has_planet_investment(state):
        bonus += 2.0
    if pressure.ratio >= 0.95:
        bonus += 3.0
    return bonus


def _shop_pack_can_trigger_hidden_target_error(state: GameState, pack: object) -> bool:
    name = _card_label(pack).lower()
    return "arcana" in name and not state.hand


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
    if not profile.rich and pressure.ratio < 1.05:
        return 0.0
    bonus = 0.0
    if not profile.has_xmult and (profile.late or pressure.ratio >= 1.0):
        bonus += 16 * max(0.35, profile.role_deficit_ratio("xmult"))
    if not profile.has_scaling and (profile.late or pressure.ratio >= 1.0):
        bonus += 10 * max(0.35, profile.role_deficit_ratio("scaling"))
    if not profile.has_scaling and pressure.ratio >= 1.15:
        bonus += 12 * max(0.35, profile.role_deficit_ratio("scaling"))
    if profile.open_joker_slots >= 2 and ("chips" in profile.missing_roles or "mult" in profile.missing_roles):
        bonus += 8
    return bonus


def _urgent_late_role_hunt(
    state: GameState,
    pressure: _ShopPressure,
    profile: _BuildProfile,
) -> bool:
    if not profile.late or _has_money_scaling_joker(state):
        return False
    if not ({"xmult", "scaling"} & set(profile.missing_roles)):
        return False
    return state.money >= 75 or pressure.ratio >= 0.85 or pressure.raw_ratio >= 0.95


def _build_synergy_value(state: GameState, joker_name: str) -> float:
    preferred = _preferred_hand_type(state)
    if preferred is None:
        return 0.0
    if preferred in JOKER_HAND_SYNERGY.get(joker_name, ()):
        return 10.0 if preferred == HandType.TWO_PAIR and joker_name not in DEDICATED_TWO_PAIR_BUILD_JOKERS else 18.0
    if preferred == HandType.PAIR and joker_name in {"Spare Trousers", "Mad Joker", "Clever Joker"}:
        return 10.0
    if preferred == HandType.FLUSH and joker_name in {"Four Fingers", "Smeared Joker"}:
        return 16.0
    return 0.0


def _build_conflict_penalty(state: GameState, joker_name: str) -> float:
    candidate_hands = set(JOKER_HAND_SYNERGY.get(joker_name, ()))
    if not candidate_hands:
        return 0.0

    existing_hand_jokers = [
        set(JOKER_HAND_SYNERGY.get(joker.name, ()))
        for joker in state.jokers
        if JOKER_HAND_SYNERGY.get(joker.name)
    ]
    if not existing_hand_jokers:
        return 0.0
    if any(candidate_hands & existing_hands for existing_hands in existing_hand_jokers):
        return 0.0

    preferred = _preferred_hand_type(state)
    if preferred is not None and preferred in candidate_hands:
        return 0.0

    return min(30.0, 14.0 + len(existing_hand_jokers) * 6.0)


def _rare_hand_inconsistency_penalty_for_joker(state: GameState, joker_name: str) -> float:
    hand_type = RARE_HAND_JOKER_TARGETS.get(joker_name)
    if hand_type is None:
        return 0.0
    return _rare_hand_investment_penalty(state, hand_type)


def _narrow_rank_inconsistency_penalty_for_joker(state: GameState, joker_name: str) -> float:
    primary = JOKER_PRIMARY_HAND.get(joker_name)
    if primary not in {HandType.THREE_OF_A_KIND, HandType.FOUR_OF_A_KIND, HandType.FIVE_OF_A_KIND}:
        return 0.0

    wanted = set(JOKER_HAND_SYNERGY.get(joker_name, ()))
    if _hand_has_natural_support((*state.hand, *state.known_deck), wanted):
        return 0.0

    penalty = 22.0 if state.ante <= 2 else 58.0
    if _normal_joker_open_slots(state) <= 1:
        penalty += 18.0
    return penalty


def _unsupported_rare_joker_extra_penalty(state: GameState, joker_name: str) -> float:
    hand_type = RARE_HAND_JOKER_TARGETS.get(joker_name)
    if hand_type is None:
        return 0.0
    need = _rare_hand_deck_manipulation_need(state, hand_type)
    if need <= 0:
        return 0.0
    return min(32.0, 14.0 + need * 4.0)


def _unsupported_two_pair_joker_penalty(state: GameState, joker_name: str) -> float:
    if joker_name not in TWO_PAIR_SUPPORT_JOKERS:
        return 0.0
    if _has_dedicated_two_pair_plan(state):
        return 0.0
    return 54.0 if state.ante <= 3 else 72.0


def _rare_hand_plan(state: GameState) -> HandType | None:
    preferred = _preferred_hand_type(state)
    if preferred in RARE_HAND_TYPES:
        return preferred

    for joker in state.jokers:
        target = RARE_HAND_JOKER_TARGETS.get(joker.name)
        if target is not None:
            return target

    leveled_rare_hands = [
        hand_type
        for hand_type in RARE_HAND_TYPES
        if state.hand_levels.get(hand_type.value, 1) > 1
    ]
    if leveled_rare_hands:
        return max(leveled_rare_hands, key=lambda hand_type: state.hand_levels.get(hand_type.value, 1))
    return None


def _rare_hand_investment_penalty(state: GameState, hand_type: HandType) -> float:
    if hand_type not in RARE_HAND_TYPES:
        return 0.0

    support_gap = _rare_hand_support_gap(state, hand_type)
    if support_gap <= 0:
        return 0.0
    early_risk = 1.0 if state.ante <= 3 else 0.75
    return min(42.0, (18.0 + support_gap * 10.0) * early_risk)


def _rare_hand_deck_manipulation_need(state: GameState, hand_type: HandType) -> float:
    if hand_type not in RARE_HAND_TYPES:
        return 0.0
    return _rare_hand_support_gap(state, hand_type)


def _rare_hand_support_gap(state: GameState, hand_type: HandType) -> float:
    gap = max(0.0, _rare_hand_support_threshold(hand_type) - _rare_hand_support_score(state, hand_type))
    if hand_type in IMPOSSIBLE_STARTER_DECK_HANDS and not _has_impossible_hand_manipulation_path(state, hand_type):
        gap += IMPOSSIBLE_HAND_MANIPULATION_GAP[hand_type]
    return gap


def _rare_hand_support_threshold(hand_type: HandType) -> float:
    if hand_type == HandType.FOUR_OF_A_KIND:
        return 3.0
    if hand_type in {HandType.FIVE_OF_A_KIND, HandType.FLUSH_FIVE}:
        return 4.0
    if hand_type == HandType.FLUSH_HOUSE:
        return 3.5
    return 0.0


def _rare_hand_support_score(state: GameState, hand_type: HandType) -> float:
    cards = (*state.hand, *state.known_deck)
    rank_counts = Counter(card.rank for card in cards)
    suit_counts = Counter(card.suit for card in cards)
    suited_rank_counts = Counter((card.rank, card.suit) for card in cards)

    max_rank = max(rank_counts.values(), default=0)
    max_suit = max(suit_counts.values(), default=0)
    max_suited_rank = max(suited_rank_counts.values(), default=0)

    if hand_type == HandType.FOUR_OF_A_KIND:
        score = float(max_rank)
    elif hand_type == HandType.FIVE_OF_A_KIND:
        score = float(max_rank)
    elif hand_type == HandType.FLUSH_FIVE:
        score = float(max_suited_rank)
        if max_rank >= 3 and max_suit >= 4:
            score += 0.75
    elif hand_type == HandType.FLUSH_HOUSE:
        score = min(float(max_rank), 3.0) + min(float(max_suit), 5.0) * 0.25
    else:
        score = 0.0

    level = state.hand_levels.get(hand_type.value, 1)
    score += min(1.5, max(0, level - 1) * 0.5)
    score += min(1.5, sum(1 for name in state.consumables if name in RARE_HAND_SUPPORT_TAROTS) * 0.5)
    return score


def _has_impossible_hand_manipulation_path(state: GameState, hand_type: HandType) -> bool:
    if hand_type not in IMPOSSIBLE_STARTER_DECK_HANDS:
        return True
    if state.hand_levels.get(hand_type.value, 1) > 1:
        return True
    if any(_tarot_supports_rare_hand(name, hand_type) for name in state.consumables):
        return True

    cards = (*state.hand, *state.known_deck)
    if not cards:
        return False
    rank_counts = Counter(card.rank for card in cards)
    exact_counts = Counter((card.rank, card.suit) for card in cards)

    max_rank = max(rank_counts.values(), default=0)
    max_exact = max(exact_counts.values(), default=0)
    if hand_type == HandType.FIVE_OF_A_KIND:
        return max_rank >= 5 or max_exact >= 2
    if hand_type == HandType.FLUSH_FIVE:
        return max_exact >= 2
    if hand_type == HandType.FLUSH_HOUSE:
        return max_exact >= 2
    return False


def _tarot_supports_rare_hand(name: str, hand_type: HandType) -> bool:
    if hand_type in RARE_RANK_HAND_TYPES and name in RANK_DECK_MANIPULATION_TAROTS:
        return True
    if hand_type in RARE_FLUSH_HAND_TYPES and name in FLUSH_DECK_MANIPULATION_TAROTS:
        return True
    return False


def _planet_card_value(state: GameState, card: object) -> float:
    hand_type = PLANET_TO_HAND.get(_card_label(card))
    if hand_type is None:
        return 0.0

    preferred = _preferred_hand_type(state)
    current_level = state.hand_levels.get(hand_type.value, 1)
    value = 14 + min(current_level, 5) * 2
    capacity_gain = _planet_capacity_gain(state, hand_type)
    flexible = _flexible_hand_types(state)
    alignment_score = _hand_joker_alignment_score(state, hand_type)
    if hand_type == preferred:
        value += min(48.0, capacity_gain / 12.0)
        value += min(20.0, alignment_score * 5.0)
    elif hand_type in flexible:
        value += min(24.0, capacity_gain / 18.0)
        value += min(10.0, alignment_score * 3.0)
    else:
        value += min(6.0, capacity_gain / 36.0)
    if hand_type in RARE_HAND_TYPES:
        value -= _rare_hand_investment_penalty(state, hand_type) * 0.8
    value -= _weak_hand_plan_penalty(state, hand_type)
    if hand_type == preferred:
        value += 24
        if hand_type in RARE_HAND_TYPES and _rare_hand_deck_manipulation_need(state, hand_type) > 0:
            value -= 10
        elif state.ante >= 4 and not _build_profile(state).has_scaling:
            value += 8
    elif hand_type in flexible:
        value += 8
        if state.ante <= 2 and alignment_score <= 0 and current_level <= 1:
            value -= 20
    else:
        value -= 8
        if state.ante <= 2:
            value -= 24
    return value


def _planet_capacity_gain(state: GameState, hand_type: HandType) -> float:
    current = _sample_build_score(state, state.jokers)
    hand_levels = dict(state.hand_levels)
    hand_levels[hand_type.value] = hand_levels.get(hand_type.value, 1) + 1
    leveled = replace(state, hand_levels=hand_levels)
    return max(0.0, _sample_build_score(leveled, leveled.jokers) - current)


def _hand_joker_alignment_score(state: GameState, hand_type: HandType) -> float:
    score = 0.0
    for joker in state.jokers:
        if hand_type == JOKER_PRIMARY_HAND.get(joker.name):
            score += 1.25
        elif hand_type in JOKER_HAND_SYNERGY.get(joker.name, ()):
            score += 1.0
    return score


def _black_hole_card_value(state: GameState) -> float:
    preferred = _preferred_hand_type(state)
    matching_planet_value = 0.0
    for planet, hand_type in PLANET_TO_HAND.items():
        if hand_type == preferred:
            matching_planet_value = _planet_card_value(state, {"label": planet, "set": "PLANET"})
            break
    levels = sum(max(1, int(level)) for level in state.hand_levels.values()) if state.hand_levels else 12
    return max(160.0, matching_planet_value + 90.0 + min(levels, 36) * 1.5)


def _weak_hand_plan_penalty(state: GameState, hand_type: HandType) -> float:
    if hand_type == HandType.PAIR:
        if _has_dedicated_pair_plan(state):
            return 0.0
        penalty = 8.0
        if state.ante >= 4:
            penalty += 8.0
        if state.hand_levels.get(hand_type.value, 1) >= 3:
            penalty -= 8.0
        return max(0.0, penalty)
    if hand_type != HandType.TWO_PAIR or _has_dedicated_two_pair_plan(state):
        return 0.0
    current_level = state.hand_levels.get(hand_type.value, 1)
    penalty = 18.0
    if current_level >= 2:
        penalty -= 6.0
    if state.ante >= 4:
        penalty += 6.0
    return max(8.0, penalty)


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
    value += _rare_hand_deck_manipulation_bonus(state, name)
    return value


def _rare_hand_deck_manipulation_bonus(state: GameState, name: str) -> float:
    hand_type = _rare_hand_plan(state)
    if hand_type is None or not _tarot_supports_rare_hand(name, hand_type):
        return 0.0

    need = _rare_hand_deck_manipulation_need(state, hand_type)
    if need <= 0:
        return 10.0
    return min(34.0, 16.0 + need * 7.0)


def _playing_card_shop_value(state: GameState, card: object) -> float:
    rank = _card_rank(card)
    suit = _card_suit(card)
    value = RANK_VALUES.get(rank, 5)
    if _preferred_hand_type(state) == HandType.FLUSH:
        dominant_suit = _dominant_suit(state)
        if dominant_suit and suit == dominant_suit:
            value += 18
    rare_plan = _rare_hand_plan(state)
    if rare_plan in RARE_RANK_HAND_TYPES:
        target_rank = _rare_rank_target(state)
        if target_rank and rank == target_rank:
            value += 26
        elif rank and _visible_rank_count(state, rank) >= 2:
            value += 14
    if rare_plan in RARE_FLUSH_HAND_TYPES:
        dominant_suit = _dominant_suit(state)
        if dominant_suit and suit == dominant_suit:
            value += 14
    enhancement = str(_card_modifier(card).get("enhancement", card.get("enhancement", ""))) if isinstance(card, dict) else ""
    if enhancement:
        value += 10
        if any(joker.name == "Vampire" for joker in state.jokers):
            value += 22
    if any(joker.name == "Hologram" for joker in state.jokers):
        value += 20
    if any(joker.name == "Midas Mask" for joker in state.jokers) and rank in {"J", "Q", "K"}:
        value += 12
    return float(value)


def _rare_rank_target(state: GameState) -> str | None:
    cards = (*state.hand, *state.known_deck)
    if not cards:
        return None
    rank, count = Counter(card.rank for card in cards).most_common(1)[0]
    return rank if count >= 2 else None


def _visible_rank_count(state: GameState, rank: str) -> int:
    return sum(1 for card in (*state.hand, *state.known_deck) if card.rank == rank)


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
        value = 42 if _normal_joker_open_slots(state) > 0 else 8
        if profile.rich and ("xmult" in profile.missing_roles or "scaling" in profile.missing_roles):
            value += 18
        if profile.late and state.money >= 75 and ("xmult" in profile.missing_roles or "scaling" in profile.missing_roles):
            value += 12
    elif "celestial" in name:
        value = 28
        if profile.ante >= 4 and not profile.has_scaling:
            value += 10
        if profile.rich and profile.late:
            value += 12
        if profile.late and state.money >= 75 and not profile.has_scaling:
            value += 8
    elif "arcana" in name:
        value = 26
        if profile.rich and profile.ante >= 4:
            value += 14
        if profile.rich and profile.late:
            value += 6
        value += _rare_hand_pack_bonus(state, "arcana")
    elif "standard" in name:
        value = 18
        value += _rare_hand_pack_bonus(state, "standard")
        if any(joker.name in {"Hologram", "Vampire", "Midas Mask"} for joker in state.jokers):
            value += 16
    elif "spectral" in name:
        value = 20 if state.ante <= 2 else 14
        value += _rare_hand_pack_bonus(state, "spectral")
    else:
        value = 16
    if "mega" in name or "jumbo" in name:
        value += 8
    return float(value)


def _rare_hand_pack_bonus(state: GameState, pack_kind: str) -> float:
    hand_type = _rare_hand_plan(state)
    if hand_type is None:
        return 0.0

    need = _rare_hand_deck_manipulation_need(state, hand_type)
    if need <= 0:
        return 0.0
    if pack_kind == "arcana":
        return min(22.0, 8.0 + need * 5.0)
    if pack_kind == "standard":
        return min(16.0, 5.0 + need * 4.0)
    if pack_kind == "spectral":
        return min(14.0, 4.0 + need * 3.0)
    return 0.0


def _rare_hand_pack_capacity_bonus(state: GameState, current_score: float, pack_kind: str) -> float:
    bonus = _rare_hand_pack_bonus(state, pack_kind)
    if bonus <= 0:
        return 0.0
    return current_score * (bonus / 220.0)


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
        if _normal_joker_open_slots(state) > 0:
            role_bonus = 0.18 if ("xmult" in profile.missing_roles or "scaling" in profile.missing_roles) else 0.10
            return (current * role_bonus) - spend_loss
        if _rich_late_role_hunt(profile):
            return (current * 0.08) - spend_loss
        return -spend_loss
    if "arcana" in name:
        rare_bonus = _rare_hand_pack_capacity_bonus(state, current, "arcana")
        if pressure.ratio < 0.75:
            return rare_bonus - spend_loss
        return (current * 0.06) + rare_bonus - spend_loss
    if "standard" in name:
        return (current * 0.04) + _rare_hand_pack_capacity_bonus(state, current, "standard") - spend_loss
    if "spectral" in name:
        return (current * 0.05) + _rare_hand_pack_capacity_bonus(state, current, "spectral") - spend_loss
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
    if state.ante >= 5 and _spendable_money(state, pressure) >= 20:
        cost_weight = max(0.9, cost_weight - 0.8)
    profile = _build_profile(state)
    if _urgent_late_role_hunt(state, pressure, profile):
        cost_weight = max(0.75, cost_weight - 0.35)
    return cost * cost_weight + _money_after_spend_penalty(state, cost, pressure)


def _money_after_spend_penalty(state: GameState, cost: int, pressure: _ShopPressure) -> float:
    after = state.money - cost
    if after < 0:
        return 1000.0
    penalty = 0.0
    interest_cap = _interest_cap_money(state)
    before_interest = min(state.money, interest_cap) // 5
    after_interest = min(after, interest_cap) // 5
    interest_weight = max(1.0, (5 if state.ante >= 2 else 3) - pressure.danger * 3 + pressure.safe_margin * 4)
    penalty += max(0, before_interest - after_interest) * interest_weight
    reserve = _desired_money_reserve(state, pressure)
    before_shortfall = max(0, reserve - state.money)
    after_shortfall = max(0, reserve - after)
    new_shortfall = max(0, after_shortfall - before_shortfall)
    if new_shortfall:
        reserve_weight = max(0.8, 2.4 - pressure.danger * 1.3 + pressure.safe_margin * 0.8)
        if _has_money_scaling_joker(state):
            reserve_weight += 0.8
        penalty += new_shortfall * reserve_weight
    if after < 4 and state.ante >= 2:
        penalty += max(2.0, 10 - pressure.danger * 6 + pressure.safe_margin * 4)
    return penalty


def _interest_cap_money(state: GameState) -> int:
    cap = BASE_INTEREST_CAP_MONEY
    owned_vouchers = set(state.vouchers)
    for voucher, voucher_cap in VOUCHER_INTEREST_CAP_MONEY.items():
        if voucher in owned_vouchers:
            cap = max(cap, voucher_cap)
    return cap


def _desired_money_reserve(state: GameState, pressure: _ShopPressure | None = None) -> int:
    reserve = _interest_cap_money(state)
    owned_jokers = {joker.name for joker in state.jokers}
    for joker_name, target in MONEY_SCALING_RESERVE_TARGETS.items():
        if joker_name in owned_jokers:
            reserve = max(reserve, target)
    if {"Bull", "Bootstraps"}.issubset(owned_jokers):
        reserve = max(reserve, 100)
    if pressure is not None and pressure.raw_ratio >= 1.05 and not owned_jokers.intersection(MONEY_SCALING_RESERVE_TARGETS):
        reserve -= int(min(15.0, (pressure.raw_ratio - 1.0) * 36.0))
        if pressure.raw_ratio >= 1.2:
            reserve -= 4
        reserve = max(8, reserve)
    return reserve


def _spendable_money(state: GameState, pressure: _ShopPressure | None = None) -> int:
    return max(0, state.money - _desired_money_reserve(state, pressure))


def _has_money_scaling_joker(state: GameState) -> bool:
    return any(joker.name in MONEY_SCALING_RESERVE_TARGETS for joker in state.jokers)


def _money_plan_payload(state: GameState, pressure: _ShopPressure | None = None) -> dict[str, int | bool]:
    return {
        "interest_cap_money": _interest_cap_money(state),
        "reserve_money": _desired_money_reserve(state, pressure),
        "spendable_money": _spendable_money(state, pressure),
        "has_money_scaling_joker": _has_money_scaling_joker(state),
    }


def _early_shop_safety_adjustment(state: GameState, card: object) -> float:
    if state.ante > 2:
        return 0.0

    adjustment = 0.0
    if _is_tarot_card(card) and _pack_card_requires_targets(card) and not _target_required_tarot_is_supported(state, card):
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


def _joker_rearrange_action(state: GameState, context: _BlindContext | None = None) -> Action | None:
    context = context or _BlindContext()
    if state.phase != GamePhase.SELECTING_HAND:
        return None
    if len(state.jokers) < 2 or len(state.jokers) > MAX_JOKER_REARRANGE_COUNT:
        return None
    if not _joker_order_can_matter(state.jokers):
        return None

    current_order = tuple(range(len(state.jokers)))
    current_score = _best_play_score_for_joker_order(state, state.jokers, context)
    best_order = current_order
    best_score = current_score

    for order in _joker_rearrange_candidate_orders(state.jokers):
        if order == current_order:
            continue
        ordered_jokers = tuple(state.jokers[index] for index in order)
        score = _best_play_score_for_joker_order(state, ordered_jokers, context)
        if score > best_score:
            best_score = score
            best_order = order

    if best_order == current_order or best_score < current_score + JOKER_REARRANGE_MIN_GAIN:
        return None
    return Action(
        ActionType.REARRANGE,
        card_indices=best_order,
        target_id="jokers",
        metadata={
            "kind": "jokers",
            "reason": f"rearrange_jokers score={current_score}->{best_score}",
        },
    )


def _best_play_score_for_joker_order(state: GameState, jokers: tuple[Joker, ...], context: _BlindContext) -> int:
    ordered_state = replace(state, jokers=jokers)
    return max((candidate.score for candidate in _play_candidates(ordered_state, context)), default=0)


def _joker_rearrange_candidate_orders(jokers: tuple[Joker, ...]) -> tuple[tuple[int, ...], ...]:
    current_order = tuple(range(len(jokers)))
    if len(jokers) <= JOKER_REARRANGE_EXHAUSTIVE_COUNT:
        return tuple(permutations(current_order))

    orders: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()

    def add(order: tuple[int, ...]) -> None:
        if len(order) != len(jokers) or set(order) != set(current_order) or order in seen:
            return
        seen.add(order)
        orders.append(order)

    role_order = tuple(sorted(current_order, key=lambda index: _joker_order_sort_key(jokers[index], index)))
    add(current_order)
    add(role_order)

    copy_indices = tuple(
        index for index, joker in enumerate(jokers) if joker.name in {"Blueprint", "Brainstorm"}
    )
    if copy_indices:
        for target_index in current_order:
            if jokers[target_index].name in {"Blueprint", "Brainstorm"}:
                continue
            add(_order_with_target_first(role_order, target_index))
            for copy_index in copy_indices:
                add(_order_with_copy_before_target(role_order, copy_index, target_index))

    return tuple(orders)


def _joker_order_sort_key(joker: Joker, original_index: int) -> tuple[int, int]:
    role = _joker_order_role(joker)
    if role in {"chips", "mult"}:
        return (0, original_index)
    if joker.name in {"Blueprint", "Brainstorm"}:
        return (1, original_index)
    if role == "xmult":
        return (2, original_index)
    return (1, original_index)


def _order_with_target_first(order: tuple[int, ...], target_index: int) -> tuple[int, ...]:
    return (target_index, *(index for index in order if index != target_index))


def _order_with_copy_before_target(
    order: tuple[int, ...],
    copy_index: int,
    target_index: int,
) -> tuple[int, ...]:
    without_copy = [index for index in order if index != copy_index]
    try:
        target_position = without_copy.index(target_index)
    except ValueError:
        return tuple(order)
    without_copy.insert(target_position, copy_index)
    return tuple(without_copy)


def _joker_order_can_matter(jokers: tuple[Joker, ...]) -> bool:
    names = {joker.name for joker in jokers}
    if names & {"Blueprint", "Brainstorm", "Baseball Card"}:
        return True
    has_xmult = any(_joker_order_role(joker) == "xmult" for joker in jokers)
    has_pre_xmult_value = any(_joker_order_role(joker) in {"chips", "mult"} for joker in jokers)
    return has_xmult and has_pre_xmult_value


def _joker_order_role(joker: Joker) -> str:
    if _edition_xmult_value(joker.edition) > 1:
        return "xmult"
    if _edition_mult_value(joker.edition) > 0:
        return "mult"
    if _edition_chips_value(joker.edition) > 0:
        return "chips"
    if joker.name in JOKER_ORDER_XMULT:
        return "xmult"
    if joker.name in JOKER_ORDER_CHIPS:
        return "chips"
    if joker.name in JOKER_ORDER_MULT:
        return "mult"
    if _joker_current_xmult_value(joker) > 1:
        return "xmult"
    if _joker_current_plus_value(joker, suffix="mult") > 0:
        return "mult"
    if _joker_current_plus_value(joker, suffix="chips") > 0:
        return "chips"
    return ""


def _best_play_action(state: GameState, context: _BlindContext | None = None) -> Action | None:
    context = context or _BlindContext()
    candidates = _play_candidates(state, context)
    if not candidates:
        return None

    remaining_score = max(0, state.required_score - state.current_score)
    if remaining_score > 0:
        winning_candidates = [candidate for candidate in candidates if candidate.score >= remaining_score]
        if winning_candidates:
            return min(
                winning_candidates,
                key=lambda candidate: _minimum_sufficient_play_key(state, candidate, context),
            ).action

        feasible_candidates = [candidate for candidate in candidates if candidate.score > 0]
        if feasible_candidates:
            if _should_set_up_card_sharp(state, context):
                return min(
                    feasible_candidates,
                    key=lambda candidate: _card_sharp_setup_play_key(candidate, remaining_score),
                ).action
            return min(
                feasible_candidates,
                key=lambda candidate: _hands_to_clear_key(state, candidate, remaining_score, context),
            ).action

    return max(candidates, key=lambda candidate: _maximum_play_key(state, candidate, context)).action


def _play_candidates(state: GameState, context: _BlindContext | None = None) -> list[_PlayCandidate]:
    context = context or _BlindContext()
    play_actions = [
        action
        for action in state.legal_actions
        if action.action_type == ActionType.PLAY_HAND and action.card_indices
    ]
    return [_play_candidate(state, action, context) for action in play_actions]


def _play_candidate(state: GameState, action: Action, context: _BlindContext | None = None) -> _PlayCandidate:
    context = context or _BlindContext()
    evaluation = _evaluate_play_action(state, action, context)
    scoring_card_indices = tuple(action.card_indices[index] for index in evaluation.scoring_indices)
    cycle_value = _cycle_value_for_play(state, action, scoring_card_indices)
    cycle_count = sum(1 for index in action.card_indices if index not in scoring_card_indices)
    return _PlayCandidate(
        action=action,
        score=_boss_adjusted_score(state, evaluation.hand_type, evaluation.score, context),
        hand_type=evaluation.hand_type,
        scoring_card_indices=scoring_card_indices,
        cycle_value=cycle_value,
        cycle_count=cycle_count,
    )


def _minimum_sufficient_play_key(
    state: GameState,
    candidate: _PlayCandidate,
    context: _BlindContext,
) -> tuple[int, float, float, int, int]:
    bonus = _playstyle_play_bonus(state, candidate, context)
    return (candidate.score, -bonus, -candidate.cycle_value, -candidate.cycle_count, _action_index_sum(candidate.action))


def _maximum_play_key(
    state: GameState,
    candidate: _PlayCandidate,
    context: _BlindContext,
) -> tuple[float, int, float, int]:
    return (
        candidate.score + _playstyle_play_bonus(state, candidate, context),
        candidate.score,
        candidate.cycle_value,
        -len(candidate.action.card_indices),
    )


def _hands_to_clear_key(
    state: GameState,
    candidate: _PlayCandidate,
    remaining_score: int,
    context: _BlindContext,
) -> tuple[int, float, int, float, int, int]:
    bonus = _playstyle_play_bonus(state, candidate, context)
    adjusted_score = max(1.0, candidate.score + bonus)
    estimated_hands = ceil(remaining_score / adjusted_score)
    return (
        estimated_hands,
        -bonus,
        -candidate.score,
        -candidate.cycle_value,
        -candidate.cycle_count,
        _action_index_sum(candidate.action),
    )


def _should_set_up_card_sharp(state: GameState, context: _BlindContext) -> bool:
    return (
        any(joker.name == "Card Sharp" for joker in state.jokers)
        and not context.played_hand_types
        and state.blind != "The Eye"
        and state.hands_remaining > 1
    )


def _card_sharp_setup_play_key(candidate: _PlayCandidate, remaining_score: int) -> tuple[int, float, int, float, int, int]:
    adjusted_score = _card_sharp_setup_score(candidate)
    estimated_hands = ceil(remaining_score / max(1.0, adjusted_score))
    return (
        estimated_hands,
        -adjusted_score,
        -candidate.score,
        -candidate.cycle_value,
        -candidate.cycle_count,
        _action_index_sum(candidate.action),
    )


def _card_sharp_setup_score(candidate: _PlayCandidate) -> float:
    repeatability = CARD_SHARP_REPEATABILITY_WEIGHTS.get(candidate.hand_type, 0.0)
    return candidate.score * (1.0 + repeatability)


def _playstyle_play_bonus(state: GameState, candidate: _PlayCandidate, context: _BlindContext) -> float:
    names = _joker_names(state)
    bonus = 0.0
    scoring_cards = _candidate_scoring_cards(state, candidate)

    if "Square Joker" in names:
        bonus += 90.0 if len(candidate.action.card_indices) == 4 else -20.0
    if "Green Joker" in names:
        bonus += 35.0
    if "Supernova" in names:
        if candidate.hand_type == _most_played_hand_type(state):
            bonus += 95.0
        if context.played_hand_types and candidate.hand_type == context.played_hand_types[-1]:
            bonus += 35.0
    if "Ride the Bus" in names and any(_is_face_card_for_state(state, card) for card in scoring_cards):
        bonus -= 240.0 + (_current_plus_for_joker(state, "Ride the Bus", suffix="mult") * 35.0)
    if "Wee Joker" in names:
        bonus += 70.0 * sum(1 for card in scoring_cards if card.rank == "2")
    if "Hack" in names:
        bonus += 12.0 * sum(1 for card in scoring_cards if card.rank in {"2", "3", "4", "5"})
    if "Fibonacci" in names:
        bonus += 16.0 * sum(1 for card in scoring_cards if card.rank in {"A", "2", "3", "5", "8"})
    if "Vampire" in names:
        bonus += 75.0 * sum(1 for card in scoring_cards if card.enhancement)
    if "Midas Mask" in names:
        bonus += 30.0 * sum(1 for card in scoring_cards if _is_face_card_for_state(state, card))
    if "Hiker" in names:
        bonus += 18.0 * len(scoring_cards)
    return bonus


def _candidate_scoring_cards(state: GameState, candidate: _PlayCandidate) -> tuple[Card, ...]:
    return tuple(state.hand[index] for index in candidate.scoring_card_indices)


def _most_played_hand_type(state: GameState) -> HandType | None:
    hands = state.modifiers.get("hands", state.modifiers.get("hand_stats", {}))
    if not isinstance(hands, dict):
        return None

    best: tuple[int, HandType] | None = None
    for name, value in hands.items():
        if not isinstance(value, dict):
            continue
        try:
            hand_type = HandType(str(name))
        except ValueError:
            continue
        played = _hand_play_count(value)
        if played <= 0:
            continue
        if best is None or played > best[0]:
            best = (played, hand_type)
    return best[1] if best is not None else None


def _hand_play_count(value: dict[str, object]) -> int:
    for key in ("played", "played_this_run", "played_count", "times_played", "count"):
        try:
            raw = value.get(key)
            if raw is not None:
                return int(raw)
        except (TypeError, ValueError):
            continue
    try:
        return int(value.get("played_this_round", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _action_index_sum(action: Action) -> int:
    return sum(action.card_indices)


def _score_play_action(state: GameState, action: Action, context: _BlindContext | None = None) -> int:
    context = context or _BlindContext()
    evaluation = _evaluate_play_action(state, action, context)
    return _boss_adjusted_score(state, evaluation.hand_type, evaluation.score, context)


def _evaluate_play_action(
    state: GameState,
    action: Action,
    context: _BlindContext | None = None,
):
    context = context or _BlindContext()
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
        played_hand_types_this_round=context.played_hand_types,
        played_hand_counts=_played_hand_counts(state),
    )


def _boss_adjusted_score(
    state: GameState,
    hand_type: HandType,
    score: int,
    context: _BlindContext,
) -> int:
    if score <= 0 or not context.played_hand_types:
        return score
    if state.blind == "The Eye" and hand_type in context.played_hand_types:
        return 0
    if state.blind == "The Mouth" and hand_type != context.played_hand_types[0]:
        return 0
    return score


def _played_hand_types_this_round(state: GameState) -> tuple[HandType, ...]:
    hands = state.modifiers.get("hands", state.modifiers.get("hand_stats", {}))
    if not isinstance(hands, dict):
        return ()

    played: list[tuple[int, HandType]] = []
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
            hand_type = HandType(str(name))
        except ValueError:
            continue
        try:
            order = int(value.get("order", 0) or 0)
        except (TypeError, ValueError):
            order = 0
        played.extend((order, hand_type) for _ in range(played_count))

    return tuple(hand_type for _, hand_type in sorted(played, key=lambda item: item[0]))


def _played_hand_counts(state: GameState) -> dict[str, int]:
    hands = state.modifiers.get("hands", state.modifiers.get("hand_stats", {}))
    if not isinstance(hands, dict):
        return {}

    counts: dict[str, int] = {}
    for name, value in hands.items():
        if isinstance(value, dict):
            counts[str(name)] = _hand_play_count(value)
        else:
            try:
                counts[str(name)] = int(value)
            except (TypeError, ValueError):
                counts[str(name)] = 0
    return counts


def _cycle_value_for_play(
    state: GameState,
    action: Action,
    scoring_card_indices: tuple[int, ...],
) -> float:
    if len(action.card_indices) <= len(scoring_card_indices):
        return 0.0

    preferred = _preferred_hand_type(state)
    keep_scores = _card_keep_scores(state.hand, preferred, state=state)
    scoring_set = set(scoring_card_indices)
    value = 0.0
    for index in action.card_indices:
        if index in scoring_set:
            continue
        value += 72.0 - keep_scores[index]
    return value


def _tactical_blind_action(
    state: GameState,
    best_play: Action,
    context: _BlindContext | None = None,
) -> Action:
    context = context or _BlindContext()
    score = _score_play_action(state, best_play, context)
    remaining_score = max(0, state.required_score - state.current_score)
    first_blind_discard = _first_blind_one_hand_hunt_action(state, best_play, score, remaining_score, context)
    if first_blind_discard is not None:
        return first_blind_discard

    opening_setup = _opening_joker_setup_play_action(state, best_play, score, remaining_score, context)
    if opening_setup is not None:
        return opening_setup

    strategic_discard = _strategic_joker_discard_action(state, best_play, score, remaining_score, context)
    if strategic_discard is not None:
        return strategic_discard

    if (
        remaining_score == 0
        or score >= remaining_score
        or state.hands_remaining <= 1
        or state.discards_remaining <= 0
        or _estimated_hands_needed(remaining_score, score) < state.hands_remaining
    ):
        return _annotated_action(best_play, reason=_play_reason(state, best_play, context))

    best_discard = _best_discard_action(state, current_best_score=score, context=context)
    if best_discard is not None and _discard_can_reduce_hands_needed(state, best_discard, score, remaining_score, context):
        return _annotated_action(best_discard, reason=_discard_reason(state, best_play, best_discard, context))

    if best_discard is not None and _should_panic_discard(state, best_discard, score, remaining_score, context):
        return _annotated_action(best_discard, reason=_panic_discard_reason(state, best_play, best_discard, context))

    if best_discard is not None and _should_safety_discard(state, best_discard, score, remaining_score, context):
        return _annotated_action(best_discard, reason=_safety_discard_reason(state, best_play, best_discard, context))

    if _score_is_on_pace(state, score, remaining_score):
        return _annotated_action(best_play, reason=_play_reason(state, best_play, context))

    if best_discard is not None and _should_chase_discard(state, best_discard, score, remaining_score, context):
        return _annotated_action(best_discard, reason=_discard_reason(state, best_play, best_discard, context))
    return _annotated_action(best_play, reason=_play_reason(state, best_play, context))


def _opening_joker_setup_play_action(
    state: GameState,
    best_play: Action,
    best_score: int,
    remaining_score: int,
    context: _BlindContext | None = None,
) -> Action | None:
    context = context or _BlindContext()
    if (
        state.current_score != 0
        or context.played_hand_types
        or state.hands_remaining <= 2
        or remaining_score <= 0
    ):
        return None

    single_play_actions = [
        action
        for action in state.legal_actions
        if action.action_type == ActionType.PLAY_HAND and len(action.card_indices) == 1
    ]
    if not single_play_actions:
        return None

    names = _joker_names(state)
    candidates: list[tuple[float, Action]] = []
    for action in single_play_actions:
        card = state.hand[action.card_indices[0]]
        value = 0.0
        if "DNA" in names:
            value += 450.0 + _card_long_term_value(state, card)
        if "Sixth Sense" in names and card.rank == "6" and _has_consumable_room(state):
            value += 520.0
        if value <= 0:
            continue
        score = _score_play_action(state, action, context)
        if not _opening_setup_is_safe(state, remaining_score, setup_score=score, fallback_score=best_score):
            continue
        candidates.append((value + score, action))

    if not candidates:
        return None
    _, action = max(candidates, key=lambda item: item[0])
    return _annotated_action(action, reason=f"joker_setup {_play_reason(state, action, context)}")


def _opening_setup_is_safe(
    state: GameState,
    remaining_score: int,
    *,
    setup_score: int,
    fallback_score: int,
) -> bool:
    after_remaining = max(0, remaining_score - setup_score)
    if after_remaining <= 0:
        return True
    if state.ante <= 2 and state.hands_remaining >= 3:
        return _estimated_hands_needed(after_remaining, max(1, fallback_score)) <= state.hands_remaining
    return _estimated_hands_needed(after_remaining, max(1, fallback_score)) <= max(1, state.hands_remaining - 1)


def _strategic_joker_discard_action(
    state: GameState,
    best_play: Action,
    best_score: int,
    remaining_score: int,
    context: _BlindContext | None = None,
) -> Action | None:
    context = context or _BlindContext()
    if state.discards_remaining <= 0 or state.hands_remaining <= 1 or remaining_score <= 0:
        return None

    discard_actions = [
        action
        for action in state.legal_actions
        if action.action_type == ActionType.DISCARD and action.card_indices
    ]
    if not discard_actions:
        return None

    preferred = _preferred_hand_type(state)
    keep_scores = _card_keep_scores(state.hand, preferred, state=state)
    ranked: list[tuple[float, int, Action]] = []
    for action in discard_actions:
        discarded_cards = tuple(state.hand[index] for index in action.card_indices)
        bonus = _discard_action_playstyle_bonus(
            state,
            action,
            discarded_cards=discarded_cards,
            keep_scores=keep_scores,
            context=context,
        )
        if bonus < 400.0:
            continue
        projected_score = _projected_score_after_discard(state, action, context)
        if not _strategic_discard_is_safe(state, remaining_score, projected_score, best_score):
            continue
        ranked.append((bonus, projected_score, action))

    if not ranked:
        return None
    _, _, action = max(ranked, key=lambda item: (item[0], item[1], len(item[2].card_indices)))
    return _annotated_action(action, reason=_joker_discard_reason(state, best_play, action, context))


def _strategic_discard_is_safe(
    state: GameState,
    remaining_score: int,
    projected_score: int,
    best_score: int,
) -> bool:
    if state.ante <= 2 and state.hands_remaining >= 3:
        return _estimated_hands_needed(remaining_score, max(1, projected_score, best_score)) <= state.hands_remaining
    if projected_score <= 0:
        return False
    pace_score = remaining_score / max(1, state.hands_remaining)
    if projected_score < pace_score * 0.65:
        return False
    return _estimated_hands_needed(remaining_score, projected_score) <= state.hands_remaining


def _first_blind_one_hand_hunt_action(
    state: GameState,
    best_play: Action,
    score: int,
    remaining_score: int,
    context: _BlindContext | None = None,
) -> Action | None:
    if (
        state.ante != 1
        or state.blind != "Small Blind"
        or state.current_score != 0
        or state.hands_remaining < 3
        or state.discards_remaining <= 0
        or state.jokers
        or remaining_score <= 0
        or score >= remaining_score
    ):
        return None

    best_discard = _best_discard_action(state, current_best_score=score, context=context)
    if best_discard is None:
        return None
    return _annotated_action(best_discard, reason=_first_blind_discard_reason(state, best_play, best_discard, context))


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


def _should_chase_discard(
    state: GameState,
    action: Action,
    current_score: int,
    remaining_score: int,
    context: _BlindContext | None = None,
) -> bool:
    if state.known_deck:
        return True

    kept_cards = tuple(card for index, card in enumerate(state.hand) if index not in action.card_indices)
    projected_score = _projected_score_after_discard(state, action, context)
    return _unknown_discard_projection_is_trustworthy(
        state,
        kept_cards=kept_cards,
        current_score=current_score,
        projected_score=projected_score,
        remaining_score=remaining_score,
        weak_chase=True,
    )


def _should_panic_discard(
    state: GameState,
    action: Action,
    current_score: int,
    remaining_score: int,
    context: _BlindContext | None = None,
) -> bool:
    if remaining_score <= 0 or state.hands_remaining <= 1 or state.discards_remaining <= 0:
        return False

    pace_score = remaining_score / max(1, state.hands_remaining)
    if pace_score <= 0 or current_score >= pace_score * _panic_discard_ratio(state):
        return False

    kept_cards = tuple(card for index, card in enumerate(state.hand) if index not in action.card_indices)
    projected_score = _projected_score_after_discard(state, action, context)
    if projected_score > current_score:
        return True

    return _strong_draw_size(kept_cards) >= 3 and len(action.card_indices) >= 2


def _should_safety_discard(
    state: GameState,
    action: Action,
    current_score: int,
    remaining_score: int,
    context: _BlindContext | None = None,
) -> bool:
    if remaining_score <= 0 or state.hands_remaining <= 1 or state.discards_remaining <= 0:
        return False

    pace_score = remaining_score / max(1, state.hands_remaining)
    safety_score = pace_score * _pace_safety_multiplier(state)
    if current_score >= safety_score:
        return False

    projected_score = _projected_score_after_discard(state, action, context)
    if (
        state.ante >= 3
        and state.discards_remaining <= 1
        and not state.known_deck
        and projected_score < remaining_score
    ):
        projected_hands_needed = _estimated_hands_needed(remaining_score, projected_score)
        if current_score >= pace_score or projected_hands_needed >= state.hands_remaining:
            return False
    if projected_score >= safety_score:
        return True

    kept_cards = tuple(card for index, card in enumerate(state.hand) if index not in action.card_indices)
    if state.discards_remaining <= 1 and not state.known_deck:
        return projected_score >= max(current_score * 1.35, pace_score)

    return projected_score >= max(current_score * 1.2, pace_score * 0.9) or _strong_draw_size(kept_cards) >= 4


def _panic_discard_ratio(state: GameState) -> float:
    ratio = 0.45
    if state.ante >= 4:
        ratio += 0.10
    if _is_boss_blind(state):
        ratio += 0.05
    if state.hands_remaining <= 2:
        ratio += 0.10
    if state.discards_remaining <= 1 and not state.known_deck:
        ratio -= 0.25
    return max(0.20, min(0.70, ratio))


def _best_discard_action(
    state: GameState,
    *,
    current_best_score: int | None = None,
    context: _BlindContext | None = None,
) -> Action | None:
    context = context or _BlindContext()
    discard_actions = [
        action
        for action in state.legal_actions
        if action.action_type == ActionType.DISCARD and action.card_indices
    ]
    if not discard_actions:
        return None

    preferred = _preferred_hand_type(state)
    keep_scores = _card_keep_scores(state.hand, preferred, state=state)
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
        current_best_play = _best_play_action(state, context)
        current_best_score = _score_play_action(state, current_best_play, context) if current_best_play is not None else 0
    current_hands_needed = _estimated_hands_needed(remaining_score, current_best_score or 0)
    detailed_actions = _prefilter_discard_actions(state, discard_actions, keep_scores, protected, preferred)

    def discard_score(action: Action) -> tuple[float, float, int, int]:
        protected_penalty = sum(1 for index in action.card_indices if index in protected) * 1000
        desirability = sum(100 - keep_scores[index] for index in action.card_indices)
        kept_cards = tuple(card for index, card in enumerate(state.hand) if index not in action.card_indices)
        discarded_cards = tuple(state.hand[index] for index in action.card_indices)
        kept_potential = _kept_hand_potential(state, kept_cards)
        projected_score = _projected_score_after_discard(state, action, context)
        projected_hands_needed = _estimated_hands_needed(remaining_score, projected_score)
        speed_bonus = max(0, current_hands_needed - projected_hands_needed) * 900
        score_bonus = min(projected_score, remaining_score) * 0.08
        playstyle_bonus = _discard_action_playstyle_bonus(
            state,
            action,
            discarded_cards=discarded_cards,
            keep_scores=keep_scores,
            context=context,
        )
        return (
            desirability + kept_potential + speed_bonus + score_bonus + playstyle_bonus - protected_penalty,
            projected_score,
            kept_potential,
            len(action.card_indices),
        )

    best = max(detailed_actions, key=discard_score)
    return best if discard_score(best)[0] > -500 else None


def _discard_action_playstyle_bonus(
    state: GameState,
    action: Action,
    *,
    discarded_cards: tuple[Card, ...],
    keep_scores: tuple[float, ...],
    context: _BlindContext | None = None,
) -> float:
    context = context or _BlindContext()
    names = _joker_names(state)
    bonus = 0.0
    if not discarded_cards:
        return bonus

    if "Trading Card" in names and _is_first_discard_window(state, context) and len(discarded_cards) == 1:
        index = action.card_indices[0]
        bonus += 780.0 + max(0.0, 140.0 - keep_scores[index])
    if "Burnt Joker" in names and _is_first_discard_window(state, context):
        hand_type = evaluate_played_cards(discarded_cards, state.hand_levels).hand_type
        bonus += BURNT_JOKER_DISCARD_HAND_VALUES.get(hand_type, 0.0)
    if "Hit the Road" in names:
        bonus += 520.0 * sum(1 for card in discarded_cards if card.rank == "J")
    castle_suit = _castle_target_suit(state)
    if castle_suit is not None:
        bonus += 240.0 * sum(1 for card in discarded_cards if _normalize_suit(card.suit) == castle_suit)
    mail_rank = _mail_in_rebate_rank(state)
    if mail_rank is not None:
        bonus += 180.0 * sum(1 for card in discarded_cards if _rank_matches(card.rank, mail_rank))
    if "Faceless Joker" in names:
        face_count = sum(1 for card in discarded_cards if _is_face_card_for_state(state, card))
        if face_count >= 3:
            bonus += 520.0 + face_count * 120.0
    if "Ride the Bus" in names and "Pareidolia" not in names:
        bonus += 90.0 * sum(1 for card in discarded_cards if _is_face_card_for_state(state, card))
    if "Green Joker" in names:
        bonus -= 280.0 + _current_plus_for_joker(state, "Green Joker", suffix="mult") * 30.0
    if "Delayed Gratification" in names:
        bonus -= 420.0
    if "Ramen" in names:
        bonus -= 80.0 * len(discarded_cards)
    return bonus


def _is_first_discard_window(state: GameState, context: _BlindContext) -> bool:
    if context.discards_taken > 0:
        return False
    if context.played_hand_types or state.current_score > 0:
        return False
    raw_discards = state.modifiers.get("round_discards_used", state.modifiers.get("discards_used"))
    try:
        if raw_discards is not None:
            return int(raw_discards) == 0
    except (TypeError, ValueError):
        pass
    return state.discards_remaining >= 3


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
    multiplier = HAND_PACE_SAFETY_BASE
    if state.ante >= 4:
        multiplier += 0.05
    if state.ante >= 5:
        multiplier += 0.06
    if _is_boss_blind(state):
        multiplier += 0.05
    if state.blind in {"The Wall", "The Needle"}:
        multiplier += 0.12
    elif state.blind in {"The Eye", "The Mouth"}:
        multiplier += 0.08
    elif state.blind in {"The Water", "The Arm"}:
        multiplier += 0.06
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
    context: _BlindContext | None = None,
) -> bool:
    current_hands_needed = _estimated_hands_needed(remaining_score, current_score)
    if current_hands_needed <= 1:
        return False

    projected_score = _projected_score_after_discard(state, action, context)
    projected_hands_needed = _estimated_hands_needed(remaining_score, projected_score)
    if projected_hands_needed >= current_hands_needed:
        return False
    if state.known_deck:
        return True

    kept_cards = tuple(card for index, card in enumerate(state.hand) if index not in action.card_indices)
    return _unknown_discard_projection_is_trustworthy(
        state,
        kept_cards=kept_cards,
        current_score=current_score,
        projected_score=projected_score,
        remaining_score=remaining_score,
    )


def _unknown_discard_projection_is_trustworthy(
    state: GameState,
    *,
    kept_cards: tuple[Card, ...],
    current_score: int,
    projected_score: int,
    remaining_score: int,
    weak_chase: bool = False,
) -> bool:
    if state.ante >= 3 and state.discards_remaining <= 1 and projected_score < remaining_score:
        return False

    draw_size = _strong_draw_size(kept_cards)
    if state.ante >= 3:
        if state.discards_remaining >= 3 and state.hands_remaining >= 3:
            if weak_chase:
                return projected_score >= current_score * 1.1 or draw_size >= 3
            return projected_score >= current_score * 1.25 or draw_size >= 3
        if draw_size < 4:
            return False
        return projected_score >= max(
            current_score * 1.6,
            (remaining_score / max(1, state.hands_remaining)) * _pace_safety_multiplier(state),
        )

    if weak_chase:
        return projected_score >= current_score * 1.1 or draw_size >= 2
    return draw_size >= 4 and projected_score >= current_score * 1.35


def _projected_score_after_discard(
    state: GameState,
    action: Action,
    context: _BlindContext | None = None,
) -> int:
    context = context or _BlindContext()
    kept_cards = tuple(card for index, card in enumerate(state.hand) if index not in action.card_indices)
    discarded_cards = tuple(state.hand[index] for index in action.card_indices)
    projected_state = replace(state, jokers=_jokers_after_discard_for_scoring(state, discarded_cards))
    draw_count = min(len(action.card_indices), max(0, 8 - len(kept_cards)))

    if state.known_deck and draw_count > 0:
        known_draw = tuple(state.known_deck[:draw_count])
        return _best_score_from_cards(projected_state, (*kept_cards, *known_draw), context)

    kept_score = _best_score_from_cards(projected_state, kept_cards, context)
    optimistic_score = _optimistic_completion_score(projected_state, kept_cards, draw_count, context)
    realism = _discard_realism_factor(projected_state, kept_cards, draw_count)
    return int(max(kept_score, (optimistic_score * realism) + (kept_score * (1.0 - realism))))


def _jokers_after_discard_for_scoring(state: GameState, discarded_cards: tuple[Card, ...]) -> tuple[Joker, ...]:
    if not discarded_cards:
        return state.jokers

    castle_suit = _castle_target_suit(state)
    discarded_castle_suit = (
        sum(1 for card in discarded_cards if _normalize_suit(card.suit) == castle_suit)
        if castle_suit is not None
        else 0
    )
    discarded_jacks = sum(1 for card in discarded_cards if card.rank == "J")
    discard_count = len(discarded_cards)

    adjusted: list[Joker] = []
    for joker in state.jokers:
        if joker.name == "Green Joker":
            adjusted.append(_joker_with_added_current_plus(joker, -1, suffix="mult"))
        elif joker.name == "Castle" and discarded_castle_suit:
            adjusted.append(_joker_with_added_current_plus(joker, discarded_castle_suit * 3, suffix="chips"))
        elif joker.name == "Hit the Road" and discarded_jacks:
            adjusted.append(_joker_with_added_current_xmult(joker, discarded_jacks * 0.5))
        elif joker.name == "Ramen" and discard_count:
            adjusted.append(_joker_with_added_current_xmult(joker, discard_count * -0.01, minimum=1.0))
        else:
            adjusted.append(joker)
    return tuple(adjusted)


def _best_score_from_cards(
    state: GameState,
    cards: tuple[Card, ...],
    context: _BlindContext | None = None,
) -> int:
    context = context or _BlindContext()
    if not cards:
        return 0
    evaluation = best_play_from_hand(
        cards,
        state.hand_levels,
        debuffed_suits=debuffed_suits_for_blind(state.blind),
        blind_name=state.blind,
        jokers=state.jokers,
        discards_remaining=max(0, state.discards_remaining - 1),
        hands_remaining=state.hands_remaining,
        deck_size=state.deck_size,
        money=state.money,
        played_hand_types_this_round=context.played_hand_types,
        played_hand_counts=_played_hand_counts(state),
    )
    return _boss_adjusted_score(state, evaluation.hand_type, evaluation.score, context)


def _optimistic_completion_score(
    state: GameState,
    kept_cards: tuple[Card, ...],
    draw_count: int,
    context: _BlindContext | None = None,
) -> int:
    context = context or _BlindContext()
    if draw_count <= 0:
        return _best_score_from_cards(state, kept_cards, context)

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

    return max(_best_score_from_cards(state, candidate, context) for candidate in candidates)


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


def _discard_realism_factor(state: GameState, kept_cards: tuple[Card, ...], draw_count: int) -> float:
    if draw_count <= 0:
        return 0.0
    draw_size = _strong_draw_size(kept_cards)
    if draw_size >= 4:
        factor = min(0.72, 0.30 + draw_size * 0.08 + draw_count * 0.04)
    elif draw_size == 3:
        factor = min(0.52, 0.24 + draw_count * 0.05)
    else:
        factor = min(0.45, 0.20 + draw_count * 0.05)

    if not state.known_deck:
        if state.ante >= 3:
            factor *= 0.65
        if _is_boss_blind(state):
            factor *= 0.85
        if state.discards_remaining <= 1:
            factor *= 0.75
    return factor


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


def _play_reason(state: GameState, action: Action, context: _BlindContext | None = None) -> str:
    score = _score_play_action(state, action, context)
    remaining_score = max(0, state.required_score - state.current_score)
    hands_needed = _estimated_hands_needed(remaining_score, score)
    return (
        f"tactical_play score={score} remaining={remaining_score} "
        f"hands_needed={hands_needed} hands_left={state.hands_remaining}"
    )


def _discard_reason(
    state: GameState,
    best_play: Action,
    discard: Action,
    context: _BlindContext | None = None,
) -> str:
    best_score = _score_play_action(state, best_play, context)
    remaining_score = max(0, state.required_score - state.current_score)
    projected_score = _projected_score_after_discard(state, discard, context)
    current_needed = _estimated_hands_needed(remaining_score, best_score)
    projected_needed = _estimated_hands_needed(remaining_score, projected_score)
    return (
        f"tactical_discard current_score={best_score} projected_score={projected_score} "
        f"hands_needed={current_needed}->{projected_needed} hands_left={state.hands_remaining}"
    )


def _joker_discard_reason(
    state: GameState,
    best_play: Action,
    discard: Action,
    context: _BlindContext | None = None,
) -> str:
    context = context or _BlindContext()
    best_score = _score_play_action(state, best_play, context)
    projected_score = _projected_score_after_discard(state, discard, context)
    discarded_cards = tuple(state.hand[index] for index in discard.card_indices)
    names = sorted(_joker_discard_triggers(state, discarded_cards))
    label = ",".join(names) if names else "joker"
    return (
        f"joker_discard triggers={label} current_score={best_score} "
        f"projected_score={projected_score} discarding={len(discard.card_indices)}"
    )


def _joker_discard_triggers(state: GameState, discarded_cards: tuple[Card, ...]) -> set[str]:
    names = _joker_names(state)
    triggers: set[str] = set()
    if "Trading Card" in names and len(discarded_cards) == 1:
        triggers.add("Trading Card")
    if "Burnt Joker" in names:
        triggers.add("Burnt Joker")
    if "Hit the Road" in names and any(card.rank == "J" for card in discarded_cards):
        triggers.add("Hit the Road")
    castle_suit = _castle_target_suit(state)
    if castle_suit is not None and any(_normalize_suit(card.suit) == castle_suit for card in discarded_cards):
        triggers.add("Castle")
    mail_rank = _mail_in_rebate_rank(state)
    if mail_rank is not None and any(_rank_matches(card.rank, mail_rank) for card in discarded_cards):
        triggers.add("Mail-In Rebate")
    if "Faceless Joker" in names and sum(1 for card in discarded_cards if _is_face_card_for_state(state, card)) >= 3:
        triggers.add("Faceless Joker")
    return triggers


def _safety_discard_reason(
    state: GameState,
    best_play: Action,
    discard: Action,
    context: _BlindContext | None = None,
) -> str:
    best_score = _score_play_action(state, best_play, context)
    remaining_score = max(0, state.required_score - state.current_score)
    projected_score = _projected_score_after_discard(state, discard, context)
    pace_score = remaining_score / max(1, state.hands_remaining)
    safety_score = pace_score * _pace_safety_multiplier(state)
    return (
        f"safety_discard current_score={best_score} projected_score={projected_score} "
        f"safety_score={safety_score:.1f} hands_left={state.hands_remaining} "
        f"discards_left={state.discards_remaining}"
    )


def _panic_discard_reason(
    state: GameState,
    best_play: Action,
    discard: Action,
    context: _BlindContext | None = None,
) -> str:
    best_score = _score_play_action(state, best_play, context)
    remaining_score = max(0, state.required_score - state.current_score)
    projected_score = _projected_score_after_discard(state, discard, context)
    pace_score = remaining_score / max(1, state.hands_remaining)
    return (
        f"panic_discard current_score={best_score} projected_score={projected_score} "
        f"pace_score={pace_score:.1f} hands_left={state.hands_remaining} "
        f"discards_left={state.discards_remaining}"
    )


def _first_blind_discard_reason(
    state: GameState,
    best_play: Action,
    discard: Action,
    context: _BlindContext | None = None,
) -> str:
    best_score = _score_play_action(state, best_play, context)
    projected_score = _projected_score_after_discard(state, discard, context)
    return (
        f"first_blind_hunt current_score={best_score} projected_score={projected_score} "
        f"discarding={len(discard.card_indices)}"
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


def _joker_card_keep_bonus(state: GameState, card: Card) -> float:
    names = _joker_names(state)
    bonus = 0.0
    face = _is_face_card_for_state(state, card)
    black = _normalize_suit(card.suit) in {"S", "C"}

    if "Ride the Bus" in names and face and "Pareidolia" not in names:
        bonus -= 95.0
    if "Wee Joker" in names and card.rank == "2":
        bonus += 130.0
    if "Hack" in names and card.rank in {"2", "3", "4", "5"}:
        bonus += 55.0
    if "Fibonacci" in names and card.rank in {"A", "2", "3", "5", "8"}:
        bonus += 45.0
    if "Scholar" in names and card.rank == "A":
        bonus += 45.0
    if "Even Steven" in names and card.rank in {"2", "4", "6", "8", "10", "T"}:
        bonus += 30.0
    if "Odd Todd" in names and card.rank in {"A", "3", "5", "7", "9"}:
        bonus += 30.0
    if "Walkie Talkie" in names and card.rank in {"10", "T", "4"}:
        bonus += 35.0
    if "Shoot the Moon" in names and card.rank == "Q":
        bonus += 95.0 * _held_effect_multiplier(state)
    if "Baron" in names and card.rank == "K":
        bonus += 110.0 * _held_effect_multiplier(state)
    if "Reserved Parking" in names and face:
        bonus += 35.0 * _held_effect_multiplier(state)
    if "Raised Fist" in names:
        bonus += RANK_VALUES[card.rank] * 4.0 * _held_effect_multiplier(state)
    if "Blackboard" in names:
        bonus += 80.0 if black else -55.0
    castle_suit = _castle_target_suit(state)
    if castle_suit is not None and _normalize_suit(card.suit) == castle_suit:
        bonus -= 35.0
    mail_rank = _mail_in_rebate_rank(state)
    if mail_rank is not None and _rank_matches(card.rank, mail_rank):
        bonus -= 25.0
    return bonus


def _held_effect_multiplier(state: GameState) -> float:
    return 2.0 if any(joker.name == "Mime" for joker in state.jokers) else 1.0


def _card_keep_scores(
    hand: tuple[Card, ...],
    preferred: HandType | None = None,
    *,
    state: GameState | None = None,
) -> tuple[float, ...]:
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
        score += _joker_card_keep_bonus(state, card) if state is not None else 0.0
        scores.append(float(score))

    return tuple(scores)


def _card_label(card: object) -> str:
    if not isinstance(card, dict):
        return str(card)
    return str(card.get("label", card.get("name", card.get("key", ""))))


def _card_key(card: object) -> str:
    if not isinstance(card, dict):
        return ""
    return str(card.get("key", ""))


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


def _joker_names(state: GameState) -> set[str]:
    return {joker.name for joker in state.jokers}


def _is_face_card_for_state(state: GameState, card: Card) -> bool:
    return card.rank in {"J", "Q", "K"} or any(joker.name == "Pareidolia" for joker in state.jokers)


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


def _current_plus_for_joker(state: GameState, joker_name: str, *, suffix: str) -> int:
    for joker in state.jokers:
        if joker.name == joker_name:
            return _joker_current_plus_value(joker, suffix=suffix)
    return 0


def _castle_target_suit(state: GameState) -> str | None:
    for joker in state.jokers:
        if joker.name != "Castle":
            continue
        effect = _joker_effect_text(joker)
        match = re.search(r"discarded\s+(Spade|Heart|Club|Diamond)", effect, flags=re.IGNORECASE)
        if match:
            return _normalize_suit(match.group(1))
    return None


def _mail_in_rebate_rank(state: GameState) -> str | None:
    for joker in state.jokers:
        if joker.name != "Mail-In Rebate":
            continue
        return _rank_from_text(_joker_effect_text(joker))
    return None


def _rank_from_text(text: str) -> str | None:
    match = re.search(
        r"\b(Ace|King|Queen|Jack|Ten|Nine|Eight|Seven|Six|Five|Four|Three|Two|K|Q|J|10|[2-9])\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    value = match.group(1).lower()
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
        "k": "K",
        "q": "Q",
        "j": "J",
    }.get(value, value.upper())


def _card_long_term_value(state: GameState, card: Card) -> float:
    value = float(RANK_VALUES.get(card.rank, 0))
    if card.enhancement:
        value += 18.0
    if card.edition:
        value += 20.0
    if card.seal:
        value += 16.0
    if card.rank in {"A", "K", "Q", "J"}:
        value += 4.0
    if _preferred_hand_type(state) in FLUSH_ARCHETYPE_HANDS:
        dominant = _dominant_suit(state)
        if dominant and card.suit == dominant:
            value += 10.0
    return value


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


def _edition_chips_value(edition: str | None) -> int:
    if edition is None:
        return 0
    return 50 if "foil" in edition.lower() else 0


def _edition_mult_value(edition: str | None) -> int:
    if edition is None:
        return 0
    text = edition.lower()
    return 10 if "holo" in text or "holographic" in text else 0


def _edition_xmult_value(edition: str | None) -> float:
    if edition is None:
        return 1.0
    return 1.5 if "polychrome" in edition.lower() else 1.0


def _early_power_bonus(name: str) -> float:
    if name in EARLY_POWER_JOKERS:
        return 5.0
    if name in JOKER_SCALING_VALUES:
        return 2.0
    return 0.0


def _preferred_hand_type(state: GameState) -> HandType | None:
    joker_votes: Counter[HandType] = Counter()
    for joker in state.jokers:
        primary = JOKER_PRIMARY_HAND.get(joker.name)
        if primary is not None:
            joker_votes[primary] += _primary_hand_vote_weight(state, joker.name, primary)
        for hand_type in JOKER_HAND_SYNERGY.get(joker.name, ()):
            if hand_type == primary or hand_type in RARE_HAND_TYPES:
                continue
            if hand_type == HandType.TWO_PAIR and joker.name not in DEDICATED_TWO_PAIR_BUILD_JOKERS:
                continue
            joker_votes[hand_type] += 1

    level_votes = Counter(
        {
            hand_type: _hand_level_vote(state, hand_type)
            for hand_type in HandType
        }
    )
    combined = joker_votes + level_votes
    if combined:
        if combined.get(HandType.PAIR, 0) > 0 and not _has_dedicated_pair_plan(state) and state.ante >= 3:
            combined[HandType.PAIR] -= 1
        if combined.get(HandType.TWO_PAIR, 0) > 0 and not _has_dedicated_two_pair_plan(state):
            combined[HandType.TWO_PAIR] -= 2
        best, score = combined.most_common(1)[0]
        if score > 0:
            if _single_narrow_chip_signal_is_noise(state, best, score):
                return HandType.PAIR if state.ante <= 3 else None
            return best
    if any(joker.name in {"Smeared Joker", "Four Fingers", "The Tribe", "Droll Joker"} for joker in state.jokers):
        return HandType.FLUSH
    if _has_dedicated_two_pair_plan(state):
        return HandType.TWO_PAIR
    if state.ante <= 2:
        return HandType.PAIR
    return None


def _primary_hand_vote_weight(state: GameState, joker_name: str, hand_type: HandType) -> int:
    if hand_type == HandType.PAIR and not _has_dedicated_pair_plan(state):
        return 1 if state.ante <= 2 else 0
    if hand_type == HandType.TWO_PAIR and joker_name not in DEDICATED_TWO_PAIR_BUILD_JOKERS:
        return 1 if state.ante <= 2 else 0
    if joker_name in NARROW_CHIP_PRIMARY_JOKERS and state.ante <= 3:
        if _hand_archetype_support_count(state, hand_type) <= 1 and _hand_level_vote(state, hand_type) <= 0:
            return 1
        return 2
    return 3


def _hand_level_vote(state: GameState, hand_type: HandType) -> int:
    level = max(0, state.hand_levels.get(hand_type.value, 1) - 1)
    if hand_type == HandType.PAIR and not _has_dedicated_pair_plan(state):
        return max(0, level - 1)
    if hand_type == HandType.TWO_PAIR and not _has_dedicated_two_pair_plan(state):
        return max(0, level - 1)
    return level


def _single_narrow_chip_signal_is_noise(state: GameState, hand_type: HandType, score: int) -> bool:
    if state.ante > 3 or score > 1:
        return False
    if hand_type not in {HandType.STRAIGHT, HandType.FLUSH}:
        return False
    return any(JOKER_PRIMARY_HAND.get(joker.name) == hand_type for joker in state.jokers if joker.name in NARROW_CHIP_PRIMARY_JOKERS)


def _hand_archetype_support_count(state: GameState, hand_type: HandType) -> int:
    count = 0
    for joker in state.jokers:
        if JOKER_PRIMARY_HAND.get(joker.name) == hand_type:
            count += 1
        elif hand_type in JOKER_HAND_SYNERGY.get(joker.name, ()):
            count += 1
    return count


def _has_dedicated_pair_plan(state: GameState) -> bool:
    if any(joker.name in DEDICATED_PAIR_BUILD_JOKERS for joker in state.jokers):
        return True
    return state.hand_levels.get(HandType.PAIR.value, 1) >= 3


def _has_dedicated_two_pair_plan(state: GameState) -> bool:
    if any(joker.name in DEDICATED_TWO_PAIR_BUILD_JOKERS for joker in state.jokers):
        return True
    return state.hand_levels.get(HandType.TWO_PAIR.value, 1) >= 3


def _flexible_hand_types(state: GameState) -> set[HandType]:
    preferred = _preferred_hand_type(state)
    if preferred in RANK_ARCHETYPE_HANDS:
        hands = set(RANK_ARCHETYPE_HANDS)
        if not _has_dedicated_two_pair_plan(state):
            hands.discard(HandType.TWO_PAIR)
        return hands
    if preferred in {HandType.STRAIGHT, HandType.STRAIGHT_FLUSH}:
        return {HandType.STRAIGHT, HandType.STRAIGHT_FLUSH}
    if preferred in FLUSH_ARCHETYPE_HANDS:
        return set(FLUSH_ARCHETYPE_HANDS)
    hands = {HandType.PAIR, HandType.THREE_OF_A_KIND, HandType.FULL_HOUSE, HandType.FLUSH, HandType.STRAIGHT}
    if _has_dedicated_two_pair_plan(state):
        hands.add(HandType.TWO_PAIR)
    return hands


def _dominant_suit(state: GameState) -> str | None:
    cards = state.hand or state.known_deck
    if not cards:
        return None
    return Counter(card.suit for card in cards).most_common(1)[0][0]


@dataclass(frozen=True, slots=True)
class _SampleHand:
    cards: tuple[Card, ...]
    held_cards: tuple[Card, ...] = ()
    weight: float = 1.0


WHITE_STAKE_SAMPLE_HANDS = (
    _SampleHand((Card("A", "S"),), (Card("K", "H"), Card("Q", "C"), Card("7", "D")), weight=0.55),
    _SampleHand((Card("2", "S"), Card("2", "H")), (Card("9", "D"), Card("5", "C")), weight=1.35),
    _SampleHand((Card("7", "S"), Card("7", "H")), (Card("K", "D"), Card("4", "C")), weight=1.2),
    _SampleHand((Card("A", "S"), Card("A", "H")), (Card("8", "D"), Card("4", "C")), weight=0.75),
    _SampleHand((Card("4", "S"), Card("4", "H"), Card("9", "D"), Card("9", "C")), weight=0.95),
    _SampleHand((Card("Q", "S"), Card("Q", "H"), Card("Q", "D")), weight=0.45),
    _SampleHand((Card("9", "S"), Card("8", "H"), Card("7", "D"), Card("6", "C"), Card("5", "S")), weight=0.3),
    _SampleHand((Card("A", "H"), Card("K", "H"), Card("Q", "H"), Card("7", "H"), Card("2", "H")), weight=0.3),
    _SampleHand((Card("J", "S"), Card("J", "H"), Card("J", "D"), Card("4", "S"), Card("4", "C")), weight=0.25),
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

PAIR_CONTAINS_HANDS = (
    HandType.PAIR,
    HandType.TWO_PAIR,
    HandType.THREE_OF_A_KIND,
    HandType.FULL_HOUSE,
    HandType.FOUR_OF_A_KIND,
    HandType.FIVE_OF_A_KIND,
    HandType.FLUSH_HOUSE,
    HandType.FLUSH_FIVE,
)

TWO_PAIR_CONTAINS_HANDS = (
    HandType.TWO_PAIR,
    HandType.FULL_HOUSE,
    HandType.FLUSH_HOUSE,
)

THREE_KIND_CONTAINS_HANDS = (
    HandType.THREE_OF_A_KIND,
    HandType.FULL_HOUSE,
    HandType.FOUR_OF_A_KIND,
    HandType.FIVE_OF_A_KIND,
    HandType.FLUSH_HOUSE,
    HandType.FLUSH_FIVE,
)

FOUR_KIND_CONTAINS_HANDS = (
    HandType.FOUR_OF_A_KIND,
    HandType.FIVE_OF_A_KIND,
    HandType.FLUSH_FIVE,
)

FLUSH_ARCHETYPE_HANDS = (
    HandType.FLUSH,
    HandType.STRAIGHT_FLUSH,
    HandType.FLUSH_HOUSE,
    HandType.FLUSH_FIVE,
)

RANK_ARCHETYPE_HANDS = tuple(
    dict.fromkeys((*PAIR_CONTAINS_HANDS, HandType.FOUR_OF_A_KIND, HandType.FIVE_OF_A_KIND))
)

CARD_SHARP_REPEATABILITY_WEIGHTS = {
    HandType.HIGH_CARD: 1.9,
    HandType.PAIR: 2.2,
    HandType.TWO_PAIR: 1.4,
    HandType.THREE_OF_A_KIND: 1.0,
    HandType.STRAIGHT: 0.7,
    HandType.FLUSH: 0.7,
    HandType.FULL_HOUSE: 0.45,
    HandType.FOUR_OF_A_KIND: 0.15,
    HandType.STRAIGHT_FLUSH: 0.1,
    HandType.FIVE_OF_A_KIND: 0.0,
    HandType.FLUSH_HOUSE: 0.0,
    HandType.FLUSH_FIVE: 0.0,
}

BURNT_JOKER_DISCARD_HAND_VALUES = {
    HandType.HIGH_CARD: 180.0,
    HandType.PAIR: 420.0,
    HandType.TWO_PAIR: 500.0,
    HandType.THREE_OF_A_KIND: 540.0,
    HandType.STRAIGHT: 580.0,
    HandType.FLUSH: 580.0,
    HandType.FULL_HOUSE: 620.0,
    HandType.FOUR_OF_A_KIND: 650.0,
    HandType.STRAIGHT_FLUSH: 700.0,
    HandType.FIVE_OF_A_KIND: 720.0,
    HandType.FLUSH_HOUSE: 720.0,
    HandType.FLUSH_FIVE: 740.0,
}


JOKER_HAND_SYNERGY = {
    "Jolly Joker": PAIR_CONTAINS_HANDS,
    "Sly Joker": PAIR_CONTAINS_HANDS,
    "Zany Joker": THREE_KIND_CONTAINS_HANDS,
    "Wily Joker": THREE_KIND_CONTAINS_HANDS,
    "Mad Joker": TWO_PAIR_CONTAINS_HANDS,
    "Clever Joker": TWO_PAIR_CONTAINS_HANDS,
    "Crazy Joker": (HandType.STRAIGHT, HandType.STRAIGHT_FLUSH),
    "Devious Joker": (HandType.STRAIGHT, HandType.STRAIGHT_FLUSH),
    "Droll Joker": FLUSH_ARCHETYPE_HANDS,
    "Crafty Joker": FLUSH_ARCHETYPE_HANDS,
    "Spare Trousers": TWO_PAIR_CONTAINS_HANDS,
    "Runner": (HandType.STRAIGHT, HandType.STRAIGHT_FLUSH),
    "The Duo": PAIR_CONTAINS_HANDS,
    "The Trio": THREE_KIND_CONTAINS_HANDS,
    "The Family": FOUR_KIND_CONTAINS_HANDS,
    "The Order": (HandType.STRAIGHT, HandType.STRAIGHT_FLUSH),
    "The Tribe": FLUSH_ARCHETYPE_HANDS,
}


JOKER_PRIMARY_HAND = {
    "Jolly Joker": HandType.PAIR,
    "Sly Joker": HandType.PAIR,
    "Zany Joker": HandType.THREE_OF_A_KIND,
    "Wily Joker": HandType.THREE_OF_A_KIND,
    "Mad Joker": HandType.TWO_PAIR,
    "Clever Joker": HandType.TWO_PAIR,
    "Crazy Joker": HandType.STRAIGHT,
    "Devious Joker": HandType.STRAIGHT,
    "Droll Joker": HandType.FLUSH,
    "Crafty Joker": HandType.FLUSH,
    "Spare Trousers": HandType.TWO_PAIR,
    "Runner": HandType.STRAIGHT,
    "The Duo": HandType.PAIR,
    "The Trio": HandType.THREE_OF_A_KIND,
    "The Family": HandType.FOUR_OF_A_KIND,
    "The Order": HandType.STRAIGHT,
    "The Tribe": HandType.FLUSH,
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


NARROW_CHIP_PRIMARY_JOKERS = {
    "Crafty Joker",
    "Devious Joker",
    "Runner",
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
    "Hallucination": 24,
}


JOKER_ORDER_XMULT = {
    "The Duo",
    "The Trio",
    "The Family",
    "The Order",
    "The Tribe",
    "Acrobat",
    "Seeing Double",
    "Flower Pot",
    "Blackboard",
    "Baron",
    "Constellation",
    "Madness",
    "Vampire",
    "Hologram",
    "Obelisk",
    "Lucky Cat",
    "Canio",
    "Caino",
    "Yorick",
    "Ramen",
    "Campfire",
    "Throwback",
    "Steel Joker",
    "Glass Joker",
    "Joker Stencil",
    "Hit the Road",
    "Cavendish",
    "Loyalty Card",
    "Driver's License",
    "Card Sharp",
    "Ancient Joker",
    "The Idol",
    "Triboulet",
}


JOKER_ORDER_MULT = {
    "Joker",
    "Gros Michel",
    "Mystic Summit",
    "Abstract Joker",
    "Swashbuckler",
    "Supernova",
    "Bootstraps",
    "Fibonacci",
    "Scholar",
    "Even Steven",
    "Half Joker",
    "Odd Todd",
    "Smiley Face",
    "Walkie Talkie",
    "Onyx Agate",
    "Greedy Joker",
    "Lusty Joker",
    "Wrathful Joker",
    "Gluttonous Joker",
    "Jolly Joker",
    "Zany Joker",
    "Mad Joker",
    "Crazy Joker",
    "Droll Joker",
    "Shoot the Moon",
    "Raised Fist",
    "Green Joker",
    "Ride the Bus",
    "Spare Trousers",
    "Fortune Teller",
    "Red Card",
    "Flash Card",
    "Popcorn",
    "Ceremonial Dagger",
    "Erosion",
}


JOKER_ORDER_CHIPS = {
    "Stuntman",
    "Bull",
    "Banner",
    "Scary Face",
    "Arrowhead",
    "Walkie Talkie",
    "Sly Joker",
    "Wily Joker",
    "Clever Joker",
    "Devious Joker",
    "Crafty Joker",
    "Blue Joker",
    "Wee Joker",
    "Runner",
    "Ice Cream",
    "Square Joker",
    "Stone Joker",
    "Castle",
}


LOW_PRIORITY_JOKERS = {
    "Credit Card",
    "Loyalty Card",
    "Seance",
    "Superposition",
    "To Do List",
    "Matador",
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


SUIT_TAROT_TARGET_SUITS = {
    "The Star": "D",
    "The Moon": "C",
    "The Sun": "H",
    "The World": "S",
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
