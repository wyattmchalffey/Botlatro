"""Basic rule bot with simple play/discard and shop discipline."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from itertools import combinations, permutations
from math import ceil, comb
import re
from threading import local
from typing import Any

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import Card, GamePhase, GameState, Joker
from balatro_ai.bots.random_bot import RandomBot
from balatro_ai.rules.hand_evaluator import (
    HandType,
    RANK_VALUES,
    STRAIGHT_VALUES,
    _prepare_joker_evaluation_context,
    best_play_from_hand,
    debuffed_suits_for_blind,
    evaluate_played_cards,
)
from balatro_ai.search.hand_viability import (
    ADVANCED_HAND_TYPES,
    hand_type_is_viable,
    hand_type_viability_multiplier,
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


@dataclass(frozen=True, slots=True)
class _StraightDrawEvaluation:
    present_count: int
    missing_count: int
    missing_values: tuple[int, ...]
    out_count: int
    top_draw_out_count: int
    completion_probability: float
    completion_score: int
    quality: float
    window_high: int
    open_ended: bool
    gutshot: bool
    completes_from_known_draw: bool


@dataclass(frozen=True, slots=True)
class _TargetDrawEvaluation:
    hand_type: HandType
    label: str
    present_count: int
    missing_count: int
    out_count: int
    completion_probability: float
    completion_score: int
    quality: float


DISCARD_DETAIL_LIMIT = 28
LATE_DISCARD_DETAIL_LIMIT = 12
ANTE_ONE_UPGRADE_NEAR_CLEAR_RATIO = 0.80
ANTE_ONE_UPGRADE_TARGET_RATIO = 0.96
ANTE_ONE_UPGRADE_MIN_GAIN = 16
WINNING_ECONOMY_HUNT_MIN_GAIN = 2.5
BLUE_SEAL_ROUND_END_VALUE = 12.0
MAX_JOKER_REARRANGE_COUNT = 6
JOKER_REARRANGE_EXHAUSTIVE_COUNT = 4
JOKER_REARRANGE_MIN_GAIN = 1
SHOP_VALUE_TOLERANCE = 0.25
SHOP_TARGET_SAFETY_BASE = 1.15
HAND_PACE_SAFETY_BASE = 1.05
BANNER_DISCARD_FUTURE_TAX_WEIGHT = 0.65
DISCARD_PENALTY_JOKERS = {"Banner", "Delayed Gratification", "Green Joker", "Ramen"}
CRITICAL_BUILD_ROLES = {"chips", "mult", "xmult", "scaling"}
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
_DECISION_CACHE_LOCAL = local()


def _decision_cached(key: tuple[object, ...], factory):
    cache = _current_decision_cache()
    if cache is None:
        return factory()
    if key not in cache:
        cache[key] = factory()
    return cache[key]


def _current_decision_cache() -> dict[tuple[object, ...], object] | None:
    return getattr(_DECISION_CACHE_LOCAL, "cache", None)


@contextmanager
def decision_cache_scope():
    previous_cache = _current_decision_cache()
    _DECISION_CACHE_LOCAL.cache = {}
    try:
        yield
    finally:
        if previous_cache is None:
            try:
                del _DECISION_CACHE_LOCAL.cache
            except AttributeError:
                pass
        else:
            _DECISION_CACHE_LOCAL.cache = previous_cache


def _sample_build_score_cache_key(state: GameState, jokers: tuple[Joker, ...]) -> tuple[object, ...]:
    state_key = _identity_cached_value(
        "sample_build_score_state_key",
        state,
        lambda: (
            state.ante,
            state.blind,
            state.hands_remaining,
            state.discards_remaining,
            state.money,
            state.deck_size,
            _freeze_for_cache(state.hand_levels),
            _freeze_for_cache(state.modifiers.get("hands", state.modifiers.get("hand_stats", {}))),
            _freeze_for_cache(state.hand),
            _freeze_for_cache(state.known_deck),
            tuple(state.consumables),
        ),
    )
    jokers_key = tuple(_joker_cache_key(joker) for joker in jokers)
    return ("sample_build_score", state_key, jokers_key)


def _joker_cache_key(joker: Joker) -> object:
    return _identity_cached_value("joker_content_key", joker, lambda: _freeze_for_cache(joker))


def _card_cache_key(card: Card) -> object:
    return _identity_cached_value(
        "card_content_key",
        card,
        lambda: (
            "Card",
            card.rank,
            card.suit,
            card.enhancement,
            card.seal,
            card.edition,
            card.debuffed,
            _freeze_for_cache(card.metadata),
        ),
    )


def _identity_cached_value(kind: str, obj: object, factory):
    cache = _current_decision_cache()
    if cache is None:
        return factory()
    key = (kind, id(obj))
    cached = cache.get(key)
    if isinstance(cached, tuple) and len(cached) == 2 and cached[0] is obj:
        return cached[1]
    value = factory()
    cache[key] = (obj, value)
    return value


def _state_scoped_cache(kind: str, state: GameState) -> dict[tuple[object, ...], object] | None:
    cache = _current_decision_cache()
    if cache is None:
        return None
    return _identity_cached_value(kind, state, dict)


def _decision_scoped_cache(kind: str) -> dict[tuple[object, ...], object] | None:
    cache = _current_decision_cache()
    if cache is None:
        return None
    key = ("decision_scoped_cache", kind)
    bucket = cache.get(key)
    if isinstance(bucket, dict):
        return bucket
    bucket = {}
    cache[key] = bucket
    return bucket


def _freeze_for_cache(value: object) -> object:
    if isinstance(value, Card):
        return _card_cache_key(value)
    if isinstance(value, Joker):
        return (
            "Joker",
            value.name,
            value.edition,
            value.sell_value,
            _freeze_for_cache(value.metadata),
        )
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze_for_cache(item)) for key, item in value.items()))
    if isinstance(value, list | tuple):
        return tuple(_freeze_for_cache(item) for item in value)
    if isinstance(value, set | frozenset):
        return tuple(sorted(_freeze_for_cache(item) for item in value))
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return repr(value)


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
        with decision_cache_scope():
            return self._choose_action_uncached(state)

    def _choose_action_uncached(self, state: GameState) -> Action:
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
            if shop_action.action_type == ActionType.END_SHOP:
                held_consumable = _held_consumable_action(state)
                if held_consumable is not None:
                    return held_consumable
                return shop_action
            else:
                self._record_shop_action(state, shop_action)
                return shop_action

        end_shop = _first_action_of_type(state, ActionType.END_SHOP)
        if end_shop is not None:
            held_consumable = _held_consumable_action(state)
            if held_consumable is not None:
                return held_consumable
            return end_shop

        stale_empty_pack = _stale_empty_pack_action(state)
        if stale_empty_pack is not None:
            return stale_empty_pack

        pack_choice = _pack_choice_action(state, self._shop_context())
        if pack_choice is not None:
            self._record_shop_action(state, pack_choice)
            return pack_choice

        held_consumable = _held_consumable_action(state)
        if held_consumable is not None:
            return held_consumable

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
    return select


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
    best_value = float("-inf")
    best_target_count = -1
    best_should_take_without_edge = False
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
        target_indices = _pack_card_target_indices(state, pack_card) or tuple(action.card_indices)
        if _pack_card_requires_targets(pack_card) and not target_indices:
            continue
        value = _pack_card_value(state, pack_card)
        target_count = len(target_indices)
        if value > best_value or (value == best_value and target_count > best_target_count):
            best_action = _with_target_indices(action, target_indices)
            best_value = value
            best_target_count = target_count
            best_should_take_without_edge = not _is_joker_card(pack_card) and not _is_spectral_card(pack_card)

    if skip_action is not None and skip_value > 0.0 and skip_value >= best_value:
        return _annotated_action(skip_action, reason=f"pack_skip value={skip_value:.1f}")

    if best_action is not None and (best_value > skip_value or best_should_take_without_edge):
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
        sequenced_info_action = _shop_information_first_action(
            state,
            pressure,
            context,
            best_action=best_action,
            best_value=best_value,
            threshold=threshold,
        )
        if sequenced_info_action is not None:
            info_action, info_value, planned_item = sequenced_info_action
            return _annotated_action(
                info_action,
                reason=(
                    f"shop_sequence_info_first value={info_value:.1f} "
                    f"planned_buy={_card_label(planned_item)} buy_value={best_value:.1f} "
                    f"pressure={pressure.ratio:.2f}"
                ),
                audit=_shop_decision_audit(
                    state,
                    pressure,
                    chosen_action=info_action,
                    chosen_value=info_value,
                    threshold=threshold,
                    decision="sequence_info_first",
                    context=context,
                ),
            )
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
    forced_action = _pressure_forced_shop_action(
        state,
        pressure,
        context,
        best_action=best_action,
        best_value=best_value,
        threshold=threshold,
    )
    if forced_action is not None:
        forced_choice, forced_value = forced_action
        return _annotated_action(
            forced_choice,
            reason=(
                f"shop_pressure_forced_spend value={forced_value:.1f} threshold={threshold:.1f} "
                f"pressure={pressure.ratio:.2f} target={pressure.target_score:.0f} "
                f"capacity={pressure.build_capacity:.0f}"
            ),
            audit=_shop_decision_audit(
                state,
                pressure,
                chosen_action=forced_choice,
                chosen_value=forced_value,
                threshold=threshold,
                decision="forced_spend",
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


def _pressure_forced_shop_action(
    state: GameState,
    pressure: _ShopPressure,
    context: _ShopContext,
    *,
    best_action: Action | None,
    best_value: float,
    threshold: float,
) -> tuple[Action, float] | None:
    profile = _build_profile(state)
    if not _pressure_spend_mode(state, pressure, profile):
        return None
    if best_action is None or best_action.action_type == ActionType.END_SHOP:
        return None
    cost = _shop_action_cost(state, best_action)
    if cost > max(0, state.money - 4):
        return None
    if cost > _spendable_money(state, pressure) + _pressure_spend_reserve_slack(state, pressure):
        return None
    minimum_value = 0.0 if pressure.ratio >= 3.0 or pressure.raw_ratio >= 1.75 else 3.0
    if best_value + SHOP_VALUE_TOLERANCE < minimum_value:
        return None
    if best_value + SHOP_VALUE_TOLERANCE >= threshold:
        return None
    return best_action, best_value


def _shop_information_first_action(
    state: GameState,
    pressure: _ShopPressure,
    context: _ShopContext,
    *,
    best_action: Action,
    best_value: float,
    threshold: float,
) -> tuple[Action, float, object] | None:
    if context.packs_opened_in_shop > 0:
        return None
    if best_action.action_type != ActionType.BUY or str(best_action.metadata.get("kind", "")) != "card":
        return None
    planned_item = _shop_item_for_action(state, best_action)
    if planned_item is None or not _is_joker_card(planned_item):
        return None
    planned_cost = _card_cost(planned_item)
    if planned_cost <= 0 or state.money < planned_cost:
        return None

    best_info_action: Action | None = None
    best_info_value = float("-inf")
    for action in state.legal_actions:
        if not _shop_action_reveals_information_before_joker_buy(state, action):
            continue
        info_cost = _shop_action_cost(state, action)
        if planned_cost + info_cost > state.money:
            continue
        if (
            _normal_slot_joker_card(planned_item)
            and _shop_information_action_can_take_joker_slot(state, action)
            and _normal_joker_open_slots(state) <= 1
        ):
            continue
        value = _shop_action_value(state, action, pressure, context)
        if value + SHOP_VALUE_TOLERANCE < threshold:
            continue
        if value > best_info_value:
            best_info_action = action
            best_info_value = value

    if best_info_action is None:
        return None
    return best_info_action, best_info_value, planned_item


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
        if action.action_type in {
            ActionType.BUY,
            ActionType.OPEN_PACK,
            ActionType.REROLL,
            ActionType.END_SHOP,
        }
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
        value += _scaling_commitment_shop_bonus(state, card, pressure)
        if _is_joker_card(card):
            joker = _joker_from_shop_card(card)
            value += pressure.danger * 14
            value += _pressure_joker_role_bonus(state, joker, pressure)
            value += _future_score_headroom_joker_bonus(state, joker, pressure)
        return value - _cost_penalty(state, card, pressure)
    if action.action_type == ActionType.BUY and kind == "voucher":
        vouchers = state.modifiers.get("voucher_cards", ())
        index = int(action.metadata.get("index", action.amount or 0))
        if index >= len(vouchers):
            return 0.0
        return _voucher_value(state, vouchers[index], pressure) - _cost_penalty(state, vouchers[index], pressure)
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
            + _scaling_commitment_pack_bonus(state, pack, pressure)
            - _cost_penalty(state, pack, pressure)
        )
    if action.action_type == ActionType.REROLL:
        reroll_cost = _shop_reroll_cost(state)
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
        return (
            24
            + pressure_bonus
            - _money_after_spend_penalty(state, reroll_cost, pressure)
            - _reroll_cost_escalation_penalty(state, reroll_cost, pressure)
        )
    return 0.0


def _shop_action_reveals_information_before_joker_buy(state: GameState, action: Action) -> bool:
    if action.action_type == ActionType.OPEN_PACK:
        pack = _shop_item_for_action(state, action)
        return (
            pack is not None
            and _is_buffoon_pack(pack)
            and not _shop_pack_can_trigger_hidden_target_error(state, pack)
        )
    return False


def _shop_information_action_can_take_joker_slot(state: GameState, action: Action) -> bool:
    if action.action_type == ActionType.OPEN_PACK:
        pack = _shop_item_for_action(state, action)
        return pack is not None and _is_buffoon_pack(pack)
    return False


def _shop_action_cost(state: GameState, action: Action) -> int:
    if action.action_type == ActionType.REROLL:
        return _shop_reroll_cost(state)
    item = _shop_item_for_action(state, action)
    return _card_cost(item) if item is not None else 0


def _shop_reroll_cost(state: GameState) -> int:
    if _int_or_default(state.modifiers.get("free_rerolls"), 0) > 0:
        return 0
    for key in ("reroll_cost", "current_reroll_cost"):
        if key in state.modifiers:
            return max(0, _int_or_default(state.modifiers.get(key), 5))
    reset_cost = _int_or_default(
        state.modifiers.get("round_reset_reroll_cost", state.modifiers.get("base_reroll_cost")),
        5,
    )
    increase = _int_or_default(state.modifiers.get("reroll_cost_increase"), 0)
    return max(0, reset_cost + increase)


def _held_consumable_action(state: GameState) -> Action | None:
    best: tuple[float, int, Action, str] | None = None
    for action in state.legal_actions:
        if action.action_type != ActionType.USE_CONSUMABLE:
            continue
        index = _action_index_for_strategy(action)
        if index is None or index >= len(state.consumables):
            continue
        name = state.consumables[index]
        value = _held_consumable_value(state, name, action)
        if value <= 0.0:
            continue
        candidate = (value, len(action.card_indices), action, name)
        if best is None or candidate[0] > best[0] or (candidate[0] == best[0] and candidate[1] > best[1]):
            best = candidate
    if best is None:
        return None
    value, _, action, name = best
    return _annotated_action(action, reason=f"use_consumable name={name} value={value:.1f}")


def _action_index_for_strategy(action: Action) -> int | None:
    raw = action.metadata.get("index", action.amount)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _held_consumable_value(state: GameState, name: str, action: Action) -> float:
    card = _consumable_card_for_name(name)
    if _is_planet_card(card):
        return _planet_card_value(state, card) + 30.0
    if _is_black_hole_card(card):
        return _black_hole_card_value(state) + 100.0
    if _is_tarot_card(card):
        if _pack_card_requires_targets(card) and not action.card_indices:
            return 0.0
        if name == "Judgement" and _normal_joker_open_slots(state) <= 0:
            return 0.0
        if name in {"The Emperor", "The High Priestess", "The Fool"} and _consumable_open_slots_after_storage_use(state) <= 0:
            return 0.0
        return _tarot_card_value(state, card) + 8.0
    if _is_spectral_card(card):
        value = _spectral_card_value(state, card)
        return value + 8.0 if value > 0.0 else 0.0
    return 0.0


def _consumable_card_for_name(name: str) -> dict[str, object]:
    if name in PLANET_TO_HAND:
        card_set = "PLANET"
    elif name in TAROT_VALUES:
        card_set = "TAROT"
    elif name in SPECTRAL_CARD_NAMES or name == "Black Hole":
        card_set = "SPECTRAL"
    else:
        card_set = "CONSUMABLE"
    return {"label": name, "set": card_set}


def _consumable_open_slots_after_storage_use(state: GameState) -> int:
    return _basic_consumable_open_slots(state) + 1


def _minimum_reroll_bank(state: GameState, pressure: _ShopPressure | None = None) -> int:
    reroll_cost = _shop_reroll_cost(state)
    if state.ante >= 4:
        reserve = _desired_money_reserve(state, pressure)
        if _normal_joker_open_slots(state) <= 0:
            owned_jokers = {joker.name for joker in state.jokers}
            closing_cap = (
                _late_closing_money_reserve_cap(state, pressure, owned_jokers)
                if pressure is not None
                else None
            )
            if closing_cap is None:
                reserve = max(reserve, _interest_cap_money(state))
        return reserve + reroll_cost
    if state.ante <= 1 and not _has_real_scoring_joker(state) and not _visible_early_power_path(state):
        return 5
    if state.ante <= 2 and not _has_real_scoring_joker(state) and pressure is not None and pressure.ratio >= 1.1:
        return 6
    if _has_money_scaling_joker(state):
        return min(_desired_money_reserve(state, pressure) + reroll_cost, 30)
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


def _reroll_cost_escalation_penalty(state: GameState, cost: int, pressure: _ShopPressure) -> float:
    if cost <= 5:
        return 0.0
    extra_cost = cost - 5
    if pressure.ratio >= 1.75 or pressure.raw_ratio >= 1.2:
        weight = 1.4
    elif pressure.ratio >= 1.05 or pressure.raw_ratio >= 0.95:
        weight = 2.2
    else:
        weight = 3.0
    if state.ante >= 5:
        weight += 0.4
    return extra_cost * weight


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
    reroll_limit = _late_reroll_limit(state, pressure, profile)
    if context.rerolls_in_shop >= reroll_limit and not _late_extra_pressure_reroll_is_worth_it(
        state,
        pressure,
        profile,
        context,
        reroll_limit,
    ):
        return False
    pressure_spendable = _spendable_money(state, pressure)
    if pressure.raw_ratio >= 1.05:
        return True
    if pressure.ratio >= 1.05 and pressure_spendable >= 20:
        if (
            not profile.rich
            and pressure.ratio < 2.5
            and pressure.raw_ratio < 1.05
            and not _pressure_spend_mode(state, pressure, profile)
        ):
            return False
        return True
    if _urgent_late_role_hunt(state, pressure, profile):
        if (
            not profile.rich
            and pressure.ratio < 2.5
            and pressure.raw_ratio < 1.05
            and not _pressure_spend_mode(state, pressure, profile)
        ):
            return False
        return pressure_spendable >= 15
    if not _rich_late_role_hunt(profile):
        return False
    if _normal_joker_open_slots(state) <= 0 and pressure.ratio < 0.58:
        return False
    return pressure_spendable >= 20


def _late_extra_pressure_reroll_is_worth_it(
    state: GameState,
    pressure: _ShopPressure,
    profile: _BuildProfile,
    context: _ShopContext,
    reroll_limit: int,
) -> bool:
    allowance = _late_extra_pressure_reroll_allowance(state, pressure, profile, reroll_limit)
    if allowance <= 0 or context.rerolls_in_shop >= reroll_limit + allowance:
        return False
    if _spendable_money(state, pressure) < 15:
        return False

    visible_value = _best_visible_non_reroll_shop_value(state, pressure, context)
    threshold = _shop_buy_threshold(state, pressure)
    return visible_value + SHOP_VALUE_TOLERANCE < threshold


def _late_extra_pressure_reroll_allowance(
    state: GameState,
    pressure: _ShopPressure,
    profile: _BuildProfile,
    reroll_limit: int,
) -> int:
    if state.ante < 5:
        return 0
    missing_critical = _missing_critical_roles(profile)
    if not missing_critical:
        return 0
    if pressure.ratio < 1.75 and pressure.raw_ratio < 1.2:
        return 0
    spendable = _spendable_money(state, pressure)
    if spendable < max(15, _shop_reroll_cost(state) * 2):
        return 0
    if _shop_has_followup_big_blind_shop(state) and pressure.ratio < 3.0 and pressure.raw_ratio < 1.75:
        return 0
    if missing_critical.isdisjoint({"xmult", "scaling"}) and pressure.ratio < 2.5 and pressure.raw_ratio < 1.35:
        return 0
    if reroll_limit >= 6:
        return 0
    if pressure.ratio >= 3.0 or pressure.raw_ratio >= 1.75:
        return 4 if spendable >= 30 else 2
    return 2 if pressure.ratio >= 2.0 or pressure.raw_ratio >= 1.35 else 1


def _missing_critical_roles(profile: _BuildProfile) -> set[str]:
    return set(profile.missing_roles) & CRITICAL_BUILD_ROLES


def _pressure_spend_mode(state: GameState, pressure: _ShopPressure, profile: _BuildProfile) -> bool:
    if state.ante < 5:
        return False
    if not _missing_critical_roles(profile):
        return False
    if _late_bank_conversion_mode(state, pressure, profile):
        return True
    if pressure.ratio < 2.0 and pressure.raw_ratio < 1.2:
        return False
    return _spendable_money(state, pressure) >= 20


def _pressure_spend_reserve_slack(state: GameState, pressure: _ShopPressure) -> int:
    if pressure.ratio >= 3.0 or pressure.raw_ratio >= 1.75:
        return 8
    if pressure.ratio >= 2.0 or pressure.raw_ratio >= 1.2:
        return 4
    return 0


def _best_visible_non_reroll_shop_value(
    state: GameState,
    pressure: _ShopPressure,
    context: _ShopContext,
) -> float:
    best_value = 0.0
    for action in state.legal_actions:
        if action.action_type in {ActionType.END_SHOP, ActionType.REROLL}:
            continue
        best_value = max(best_value, _shop_action_value(state, action, pressure, context))
    return best_value


def _late_reroll_limit(state: GameState, pressure: _ShopPressure, profile: _BuildProfile) -> int:
    if state.ante < 5:
        return 99
    closer_limit = _late_pressure_closer_reroll_limit(state, pressure, profile)
    if closer_limit is not None:
        return closer_limit
    if state.ante >= 7 and pressure.ratio >= 1.2:
        if _urgent_late_role_hunt(state, pressure, profile):
            return 4 if pressure.ratio >= 1.6 else 3
        if _rich_late_role_hunt(profile) or {"xmult", "scaling"} & set(profile.missing_roles):
            return 3 if pressure.ratio >= 1.6 else 2
    if state.ante >= 8 and pressure.ratio >= 1.05 and _rich_late_role_hunt(profile):
        return 2
    if pressure.raw_ratio >= 1.35:
        return 3 if _urgent_late_role_hunt(state, pressure, profile) else 2
    if pressure.raw_ratio >= 1.15:
        return 2
    if _urgent_late_role_hunt(state, pressure, profile):
        return 2
    if _rich_late_role_hunt(profile):
        return 1
    return 0


def _late_pressure_closer_mode(
    state: GameState,
    pressure: _ShopPressure,
    profile: _BuildProfile | None = None,
) -> bool:
    if state.ante < 5 or _has_money_scaling_joker(state):
        return False
    if state.money < 45:
        return False

    profile = profile or _build_profile(state)
    owned_jokers = {joker.name for joker in state.jokers}
    bank_conversion_cap = _late_bank_conversion_reserve_cap(state, pressure, owned_jokers, profile=profile)
    spendable = max(0, state.money - (bank_conversion_cap if bank_conversion_cap is not None else _late_pressure_interest_floor(state, pressure)))
    if spendable < 15:
        return False

    if bank_conversion_cap is not None:
        return True
    if _shop_has_followup_big_blind_shop(state):
        return pressure.raw_ratio >= 1.75 or pressure.ratio >= 3.0
    if pressure.raw_ratio >= 1.2:
        return True
    if pressure.ratio >= 1.75:
        return True

    return state.money >= 75 and profile.rich and pressure.raw_ratio >= 1.05


def _late_bank_conversion_mode(state: GameState, pressure: _ShopPressure, profile: _BuildProfile) -> bool:
    owned_jokers = {joker.name for joker in state.jokers}
    return _late_bank_conversion_reserve_cap(state, pressure, owned_jokers, profile=profile) is not None


def _late_bank_conversion_reserve_cap(
    state: GameState,
    pressure: _ShopPressure,
    owned_jokers: set[str],
    *,
    profile: _BuildProfile | None = None,
) -> int | None:
    if state.ante < 7 or owned_jokers.intersection(MONEY_SCALING_RESERVE_TARGETS):
        return None

    profile = profile or _build_profile(state)
    missing = _missing_critical_roles(profile)
    if not (missing & {"xmult", "scaling"}):
        return None
    if state.money < 65:
        return None

    boss_push = pressure.boss_name in FINAL_BOSS_BLINDS and (pressure.raw_ratio >= 0.85 or pressure.ratio >= 1.0)
    pressure_push = pressure.raw_ratio >= 0.95 or pressure.ratio >= 1.15
    final_push = state.ante >= 8 and (pressure.raw_ratio >= 0.80 or pressure.ratio >= 0.95)
    if not (boss_push or pressure_push or final_push):
        return None

    interest_cap = _interest_cap_money(state)
    severe = pressure.raw_ratio >= 1.2 or pressure.ratio >= 1.75
    reserve = max(20 if severe else 25, int(interest_cap * (0.65 if severe else 0.75)))
    if state.money - reserve < 15:
        return None
    return reserve


def _late_pressure_closer_reroll_limit(
    state: GameState,
    pressure: _ShopPressure,
    profile: _BuildProfile,
) -> int | None:
    if not _late_pressure_closer_mode(state, pressure, profile):
        return None
    if _shop_has_followup_big_blind_shop(state):
        if _late_bank_conversion_mode(state, pressure, profile):
            return 3
        return 3 if pressure.ratio >= 3.0 or pressure.raw_ratio >= 1.75 else 2
    if pressure.ratio >= 3.0 or pressure.raw_ratio >= 1.75:
        return 8
    if pressure.ratio >= 1.25 or pressure.raw_ratio >= 1.05:
        return 6
    return 4


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
        return _planet_card_value(state, card) + 10
    if _is_tarot_card(card):
        return _tarot_card_value(state, card) + 10
    if _is_spectral_card(card):
        return _spectral_card_value(state, card)
    if _is_playing_card(card):
        return _playing_card_shop_value(state, card)
    return 0.0


def _spectral_card_value(state: GameState, card: object) -> float:
    name = _card_label(card)
    if name in SPECTRAL_SEAL_VALUES:
        return SPECTRAL_SEAL_VALUES[name]
    if name == "Aura":
        return 46.0
    if name == "Cryptid":
        return 42.0
    if name == "Immolate":
        destroyed_count = min(5, len(state.hand))
        deck_penalty = destroyed_count * 1.6
        if state.deck_size and state.deck_size <= 30:
            deck_penalty += (31 - state.deck_size) * 0.4
        return _money_gain_value(state, 20) - deck_penalty
    if name == "The Soul":
        return 70.0 if _normal_joker_open_slots(state) > 0 else 0.0
    if name == "Wraith":
        if _normal_joker_open_slots(state) <= 0:
            return 0.0
        return 44.0 - (state.money * 1.4)
    return 0.0


def _money_gain_value(state: GameState, amount: int) -> float:
    if amount <= 0:
        return 0.0
    value = float(amount)
    interest_gap = max(0, _interest_cap_money(state) - state.money)
    value += min(amount, interest_gap) * (0.45 if state.ante <= 2 else 0.25)
    if _has_money_scaling_joker(state):
        value += amount * 0.35
    return value


def _pack_card_requires_targets(card: object) -> bool:
    return _is_tarot_card(card) and _card_label(card) in TARGET_REQUIRED_TAROTS


def _pack_card_is_pickable(state: GameState, card: object) -> bool:
    if _is_black_hole_card(card):
        return True
    if _is_joker_card(card) and _buy_would_overfill_joker_slots(state, card):
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
    return tuple(sorted(index for index, _ in ranked[:3]))


def _shop_buy_threshold(state: GameState, pressure: _ShopPressure) -> float:
    spendable_money = _spendable_money(state, pressure)
    profile = _build_profile(state)
    bank_conversion = _late_bank_conversion_mode(state, pressure, profile)
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
    if bank_conversion:
        base -= 5.0
    if state.ante >= 4 and pressure.ratio >= 1.15:
        base -= 4.0
    safe_margin_weight = 4 if bank_conversion else 8 if state.ante >= 5 and spendable_money >= 20 else 14
    floor = 8.0 if bank_conversion else 10.0
    if state.ante >= 5 and spendable_money >= 4:
        if pressure.ratio >= 1.75 or pressure.raw_ratio >= 1.2:
            floor = 9.0
        elif pressure.ratio >= 1.25 and {"xmult", "scaling"} & set(profile.missing_roles):
            floor = 9.0
    return max(floor, base - pressure.danger * 14 + pressure.safe_margin * safe_margin_weight)


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
    return _identity_cached_value("shop_pressure", state, lambda: _shop_pressure_uncached(state))


def _shop_pressure_uncached(state: GameState) -> _ShopPressure:
    boss_name = _upcoming_boss_blind_name(state)
    boss_target_multiplier = _effective_boss_target_multiplier(state, boss_name)
    boss_capacity_factor = _shop_pressure_boss_capacity_factor(state, boss_name)
    raw_target = _estimated_shop_planning_required_score(state)
    safety_multiplier = _shop_target_safety_multiplier(state)
    target = raw_target * safety_multiplier
    score_state = _shop_pressure_score_state(state, boss_name, raw_target=raw_target)
    current_score = _sample_build_score(score_state, score_state.jokers) * _shop_hand_realism_factor(score_state)
    effective_hands = _shop_pressure_effective_hands(state, boss_name)
    raw_capacity = max(1.0, current_score * effective_hands * 0.85)
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


def _shop_hand_realism_factor(state: GameState) -> float:
    if state.ante < 4 or _has_money_scaling_joker(state):
        return 1.0

    preferred = _preferred_hand_type(state)
    if preferred not in PREFERRED_HAND_HUNT_TYPES:
        return 1.0

    factor = {
        HandType.THREE_OF_A_KIND: 0.72,
        HandType.FULL_HOUSE: 0.68,
        HandType.STRAIGHT: 0.66,
        HandType.FLUSH: 0.74,
        HandType.FOUR_OF_A_KIND: 0.55,
        HandType.FIVE_OF_A_KIND: 0.50,
        HandType.STRAIGHT_FLUSH: 0.52,
        HandType.FLUSH_HOUSE: 0.50,
        HandType.FLUSH_FIVE: 0.48,
    }.get(preferred, 1.0)

    if _has_planet_investment(state):
        factor += 0.05
    if state.discards_remaining >= 4:
        factor += 0.04
    elif state.discards_remaining <= 2:
        factor -= 0.05
    if _normal_joker_open_slots(state) > 0:
        factor += 0.03
    if preferred in RARE_HAND_TYPES and _rare_hand_deck_manipulation_need(state, preferred) > 0:
        factor -= 0.08
    return max(0.45, min(1.0, factor))


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


def _shop_pressure_boss_capacity_factor(state: GameState, boss_name: str | None) -> float:
    if _shop_pressure_uses_exact_needle_hand(state, boss_name):
        return 1.0
    return _weighted_boss_capacity_factor(state, boss_name) * _weighted_final_boss_fragility_factor(state, boss_name)


def _shop_pressure_effective_hands(state: GameState, boss_name: str | None) -> float:
    if _shop_pressure_uses_exact_needle_hand(state, boss_name):
        return 1.0
    hands = float(state.hands_remaining or 4)
    return max(2.0, min(4.0, hands) - 0.5)


def _shop_pressure_score_state(state: GameState, boss_name: str | None, *, raw_target: float) -> GameState:
    if not _shop_pressure_uses_boss_score_state(state, boss_name):
        return state
    return replace(
        state,
        blind=str(boss_name),
        required_score=int(raw_target),
        hands_remaining=_shop_pressure_boss_hands_remaining(state, boss_name),
        discards_remaining=_shop_pressure_boss_discards_remaining(state, boss_name),
    )


def _shop_pressure_uses_boss_score_state(state: GameState, boss_name: str | None) -> bool:
    return bool(boss_name) and _shop_cleared_blind_kind(state) == "BIG"


def _shop_pressure_boss_hands_remaining(state: GameState, boss_name: str | None) -> int:
    if boss_name == "The Needle":
        return 1
    return state.hands_remaining


def _shop_pressure_boss_discards_remaining(state: GameState, boss_name: str | None) -> int:
    if boss_name == "The Water":
        return 0
    return state.discards_remaining


def _shop_pressure_uses_exact_needle_hand(state: GameState, boss_name: str | None) -> bool:
    return boss_name == "The Needle" and _shop_cleared_blind_kind(state) == "BIG"


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
    cleared_kind = _shop_cleared_blind_kind(state)

    if state.blind == "Small Blind" and cleared_kind != "BIG":
        return small * 1.5
    if state.blind == "Big Blind" or cleared_kind == "BIG":
        boss_base = small * 2.0
        boss_score = _upcoming_boss_score(state)
        if cleared_kind == "BIG" and boss_score > 0:
            return boss_score
        if boss_score > 0:
            return max(boss_base, boss_score)
        boss_name = _upcoming_boss_blind_name(state)
        return boss_base * _boss_score_target_multiplier(boss_name)
    if state.required_score > 0 and state.blind:
        return max(next_small, state.required_score * 1.25)
    if state.required_score > 0:
        return state.required_score * 1.5
    return small


def _estimated_shop_planning_required_score(state: GameState) -> float:
    next_required = _estimated_next_required_score(state)
    final_weight = _final_win_target_weight(state)
    if final_weight <= 0.0:
        return next_required
    final_required = _estimated_final_win_required_score(state) * final_weight
    return max(next_required, final_required)


def _estimated_final_win_required_score(state: GameState) -> float:
    final_small = ANTE_SMALL_BLIND_SCORES.get(8, _extrapolated_small_blind_score(8))
    target = final_small * 2.0
    if state.ante >= 8:
        boss_score = _upcoming_boss_score(state)
        if boss_score > 0:
            target = max(target, boss_score)
        elif state.blind == "Big Blind" or _shop_cleared_blind_kind(state) == "BIG":
            target *= _boss_score_target_multiplier(_upcoming_boss_blind_name(state))
    return target


def _final_win_target_weight(state: GameState) -> float:
    if state.ante >= 8:
        return 1.0
    cleared_kind = _shop_cleared_blind_kind(state)
    if state.ante == 7:
        if cleared_kind == "BIG" or state.blind == "Big Blind":
            return 1.0
        if cleared_kind == "SMALL" or state.blind == "Small Blind":
            return 0.75
        return 0.65
    return 0.0


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
    if (state.blind != "Big Blind" and _shop_cleared_blind_kind(state) != "BIG") or _upcoming_boss_score(state) > 0:
        return 1.0
    return _boss_score_target_multiplier(boss_name)


def _boss_score_target_multiplier(boss_name: str | None) -> float:
    if boss_name == "The Wall":
        return 2.0
    if boss_name == "Violet Vessel":
        return 3.0
    return 1.0


def _boss_target_safety_bonus(boss_name: str) -> float:
    if boss_name in {"Violet Vessel", "Verdant Leaf", "Crimson Heart"}:
        return 0.18
    if boss_name in {"The Wall", "The Needle", "Amber Acorn", "Cerulean Bell"}:
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
    cleared_kind = _shop_cleared_blind_kind(state)
    if cleared_kind == "BIG":
        return 1.0
    if cleared_kind == "SMALL":
        return 0.45
    if state.blind == "Big Blind":
        return 1.0
    if state.blind == "Small Blind":
        return 0.45
    return 0.0


def _shop_has_followup_big_blind_shop(state: GameState) -> bool:
    cleared_kind = _shop_cleared_blind_kind(state)
    return cleared_kind == "SMALL" or (state.blind == "Small Blind" and cleared_kind != "BIG")


def _shop_cleared_blind_kind(state: GameState) -> str:
    raw = state.modifiers.get("cleared_blind")
    if not isinstance(raw, dict):
        return ""
    kind = str(raw.get("kind", "")).upper()
    return kind if kind in {"SMALL", "BIG", "BOSS"} else ""


def _boss_capacity_factor(state: GameState, boss_name: str) -> float:
    if boss_name == "Verdant Leaf":
        return 0.58
    if boss_name == "Crimson Heart":
        return 0.60
    if boss_name == "Amber Acorn":
        return 0.72
    if boss_name == "Cerulean Bell":
        return 0.72
    if boss_name == "The Flint":
        return 0.66
    if boss_name == "The Needle":
        return 0.34
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


def _weighted_final_boss_fragility_factor(state: GameState, boss_name: str | None) -> float:
    if boss_name not in FINAL_BOSS_BLINDS:
        return 1.0
    factor = _final_boss_fragility_factor(state, boss_name)
    weight = _boss_preview_weight(state)
    return 1.0 - ((1.0 - factor) * weight)


def _final_boss_fragility_factor(state: GameState, boss_name: str | None) -> float:
    names = _active_joker_names(state)
    if boss_name == "Crimson Heart":
        factor = 1.0
        if len(state.jokers) >= 4:
            factor *= 0.84
        if names & FINAL_BOSS_FRAGILE_JOKERS:
            factor *= 0.88
        return factor
    if boss_name == "Amber Acorn" and names & ORDER_SENSITIVE_JOKERS:
        return 0.90
    if boss_name == "Cerulean Bell" and _preferred_hand_type(state) in PREFERRED_HAND_HUNT_TYPES:
        return 0.92
    return 1.0


def _extrapolated_small_blind_score(ante: int) -> float:
    if ante <= 8:
        return ANTE_SMALL_BLIND_SCORES.get(ante, 300)
    return 50000 * (1.6 ** (ante - 8))


def _has_consumable_room(state: GameState) -> bool:
    return _basic_consumable_open_slots(state) > 0


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
    return _card_set(card) == "TAROT" or _card_label(card) in TAROT_VALUES


def _is_spectral_card(card: object) -> bool:
    return _card_set(card) == "SPECTRAL" or _card_label(card) in SPECTRAL_CARD_NAMES


def _is_consumable_card(card: object) -> bool:
    return _is_tarot_card(card) or _is_planet_card(card) or _is_spectral_card(card)


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
    durability = _joker_late_durability_factor(state, joker)
    sample_delta = _sample_score_delta_for_joker(state, joker) * _joker_sample_reliability(state, joker) * durability
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
    value = _capped_red_card_candidate_value(state, joker, value)
    return value


def _candidate_joker_value_for_replacement(state: GameState, joker: Joker) -> float:
    if _joker_stencil_would_fill_slots(state, joker):
        return 0.0
    durability = _joker_late_durability_factor(state, joker)
    sample_gain = _sample_score_gain_for_joker(state, joker) * _joker_sample_reliability(state, joker) * durability
    value = sample_gain * 0.08
    value += _joker_heuristic_value(state, joker)
    value += _joker_role_bonus(state, joker)
    value -= _unsupported_two_pair_joker_penalty(state, joker.name)
    value -= _unsupported_rare_joker_extra_penalty(state, joker.name)
    value += _edition_bonus(joker.edition)
    value = _capped_red_card_candidate_value(state, joker, value)
    if any(existing.name == joker.name for existing in state.jokers):
        value -= 18
    return value


def _owned_joker_value(state: GameState, joker: Joker, *, remove_index: int) -> float:
    without = _jokers_after_sell_for_scoring(state, remove_index=remove_index)
    score_loss = max(0.0, _sample_build_score(state, state.jokers) - _sample_build_score(state, without))
    durability = _joker_late_durability_factor(state, joker)
    value = score_loss * 0.08 * durability
    value += _joker_heuristic_value(state, joker) * 0.75
    value += _owned_role_value(state, joker, remove_index=remove_index)
    value += _edition_bonus(joker.edition)
    value += (joker.sell_value or 0) * 1.5
    if joker.name in LOW_PRIORITY_JOKERS:
        value -= 20
    if joker.name in TWO_PAIR_SUPPORT_JOKERS and not _has_dedicated_two_pair_plan(state):
        value -= 45
    return value


def _capped_red_card_candidate_value(state: GameState, joker: Joker, value: float) -> float:
    if joker.name != "Red Card" or _joker_current_plus_value(joker, suffix="mult") > 0:
        return value
    cap = 24.0
    if _red_card_has_visible_skip_plan(state):
        cap += 16.0
    if state.ante <= 1:
        cap += 4.0
    return min(value, cap)


def _red_card_has_visible_skip_plan(state: GameState) -> bool:
    return any(
        _card_cost(pack) <= max(0, state.money - 4)
        for pack in state.modifiers.get("booster_packs", ())
    )


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
    return joker.effect.current_plus_value(suffix)


def _joker_current_xmult_value(joker: Joker) -> float:
    return joker.effect.current_xmult_visible if joker.effect.current_xmult_visible is not None else 1.0


def _joker_effect_text(joker: Joker) -> str:
    return joker.effect.text


def _format_xmult(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:.2f}".rstrip("0").rstrip(".")


def _sample_build_score(state: GameState, jokers: tuple[Joker, ...]) -> float:
    return _decision_cached(
        _sample_build_score_cache_key(state, jokers),
        lambda: _sample_build_score_uncached(state, jokers),
    )


def _sample_build_score_uncached(state: GameState, jokers: tuple[Joker, ...]) -> float:
    scoring_state = replace(state, jokers=jokers)
    joker_context = _prepare_joker_evaluation_context(jokers)
    weighted_total = 0.0
    total_weight = 0.0
    raw_scores: list[float] = []

    for sample in _score_samples_for_state(scoring_state):
        score = _sample_hand_build_score(scoring_state, jokers, sample, joker_context=joker_context)
        weighted_total += score * sample.weight
        total_weight += sample.weight
        raw_scores.append(score)

    visible_score = _visible_hand_sample_score(scoring_state, jokers, joker_context=joker_context)
    if visible_score > 0:
        weighted_total += visible_score * 1.15
        total_weight += 1.15
        raw_scores.append(float(visible_score))

    if total_weight <= 0 or not raw_scores:
        return 0.0
    expected = weighted_total / total_weight
    average_top = sum(sorted(raw_scores, reverse=True)[:3]) / min(3, len(raw_scores))
    return (expected * 0.78) + (average_top * 0.22)


def _sample_hand_build_score(
    state: GameState,
    jokers: tuple[Joker, ...],
    sample: "_SampleHand",
    *,
    joker_context,
) -> float:
    played_types = _played_hand_types_this_round(state)
    evaluation = evaluate_played_cards(
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
        played_hand_types_this_round=played_types,
        played_hand_counts=_played_hand_counts(state),
        _joker_context=joker_context,
    )
    score = float(evaluation.score)
    if not _should_project_card_sharp_repeat_value(state, joker_context, played_types):
        return score

    repeated_evaluation = evaluate_played_cards(
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
        played_hand_types_this_round=(evaluation.hand_type,),
        played_hand_counts=_played_hand_counts(state),
        _joker_context=joker_context,
    )
    active_weight = _card_sharp_repeat_projection_weight(state)
    return (score * (1.0 - active_weight)) + (float(repeated_evaluation.score) * active_weight)


def _should_project_card_sharp_repeat_value(
    state: GameState,
    joker_context,
    played_types: tuple[HandType, ...],
) -> bool:
    if "Card Sharp" not in joker_context.active_ability_names:
        return False
    if played_types:
        return False
    if state.blind == "The Eye":
        return False
    return _card_sharp_repeat_projection_weight(state) > 0.0


def _card_sharp_repeat_projection_weight(state: GameState) -> float:
    hands = max(1, int(state.hands_remaining or 4))
    if hands <= 1:
        return 0.0
    setup_discount = 0.85 if state.blind == "The Mouth" else 0.78
    return min(0.82, ((hands - 1) / hands) * setup_discount)


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
    if preferred in {HandType.FLUSH, HandType.FLUSH_HOUSE, HandType.FLUSH_FIVE}:
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
        if preferred in {HandType.FLUSH_HOUSE, HandType.FLUSH_FIVE} and _rare_hand_deck_manipulation_need(state, preferred) <= 0:
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
        samples = [
            _SampleHand((Card("9", "S"), Card("8", "H"), Card("7", "D"), Card("6", "C"), Card("5", "S")), weight=1.35),
            _SampleHand((Card("A", "S"), Card("K", "H"), Card("Q", "D"), Card("J", "C"), Card("10", "S")), weight=0.9),
            _SampleHand((Card("6", "S"), Card("6", "H")), (Card("Q", "D"), Card("4", "C")), weight=0.6),
        ]
        if preferred == HandType.STRAIGHT_FLUSH and hand_type_is_viable(state, HandType.STRAIGHT_FLUSH):
            dominant = _dominant_suit(state) or "H"
            samples.append(
                _SampleHand(
                    (
                        Card("9", dominant),
                        Card("8", dominant),
                        Card("7", dominant),
                        Card("6", dominant),
                        Card("5", dominant),
                    ),
                    weight=0.55,
                )
            )
        return tuple(samples)
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


def _visible_hand_sample_score(state: GameState, jokers: tuple[Joker, ...], *, joker_context=None) -> int:
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
        _joker_context=joker_context,
    ).score


def _joker_heuristic_value(state: GameState, joker: Joker) -> float:
    name = joker.name
    if _joker_is_disabled_for_build(joker):
        return 0.0
    if _joker_stencil_would_fill_slots(state, joker):
        return 0.0
    value = 0.0
    value += JOKER_BASE_VALUES.get(name, 0)
    if name == "Red Card":
        value += _red_card_heuristic_value(state, joker)
    else:
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
    durability = _joker_late_durability_factor(state, joker)
    if durability < 1.0:
        value *= durability
        if state.ante >= 7:
            value -= _temporary_score_late_penalty(joker)
    return value


def _temporary_score_late_penalty(joker: Joker) -> float:
    if _joker_has_sticker(joker, "perishable"):
        return 18.0
    if joker.name in DECAYING_SCORE_JOKERS:
        return 12.0
    if joker.name in TEMPORARY_SCORE_JOKERS:
        return 8.0
    return 0.0


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


def _red_card_heuristic_value(state: GameState, joker: Joker) -> float:
    """Red Card is only real scaling once the bot is actually skipping packs."""

    current_mult = _joker_current_plus_value(joker, suffix="mult")
    if current_mult <= 0:
        return 4.0
    scaled_value = min(42.0, current_mult * 2.4)
    if state.ante <= 2:
        scaled_value += min(10.0, current_mult * 0.8)
    return 6.0 + scaled_value


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
    return _identity_cached_value("build_profile", state, lambda: _build_profile_uncached(state))


def _build_profile_uncached(state: GameState) -> _BuildProfile:
    joker_roles = [_joker_roles(joker) for joker in state.jokers if not _joker_is_disabled_for_build(joker)]
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
        joker_scores = _joker_role_scores(state, joker)
        for role, value in joker_scores.items():
            scores[role] += value
    scores["economy"] += min(18.0, max(0.0, state.money - 5) * 0.6)
    if state.money >= _interest_cap_money(state):
        scores["economy"] += 18.0
    return scores


def _joker_role_scores(state: GameState, joker: Joker) -> dict[str, float]:
    if _joker_is_disabled_for_build(joker):
        return {"chips": 0.0, "mult": 0.0, "xmult": 0.0, "scaling": 0.0, "economy": 0.0}

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

    if name == "Red Card":
        current_mult = _joker_current_plus_value(joker, suffix="mult")
        if current_mult > 0:
            scores["scaling"] += min(30.0, current_mult * 1.5)
        return _durable_joker_role_scores(state, joker, scores)

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

    return _durable_joker_role_scores(state, joker, scores)


def _durable_joker_role_scores(
    state: GameState,
    joker: Joker,
    scores: dict[str, float],
) -> dict[str, float]:
    durability = _joker_late_durability_factor(state, joker)
    if durability < 1.0:
        for role in ("chips", "mult", "xmult", "scaling"):
            scores[role] *= durability
        if _joker_has_sticker(joker, "perishable"):
            scores["economy"] *= durability
    return scores


def _joker_late_durability_factor(state: GameState, joker: Joker) -> float:
    if _joker_is_disabled_for_build(joker):
        return 0.0

    factor = 1.0
    if _joker_has_sticker(joker, "perishable"):
        if state.ante >= 7:
            factor *= 0.35
        elif state.ante >= 5:
            factor *= 0.55
        else:
            factor *= 0.75

    if joker.name in DECAYING_SCORE_JOKERS:
        if state.ante >= 7:
            factor *= 0.45
        elif state.ante >= 5:
            factor *= 0.68
        else:
            factor *= 0.90

    if joker.name in FINITE_SCORE_JOKERS:
        remaining = _joker_remaining_count(joker)
        if remaining is not None:
            factor *= min(1.0, max(0.35, remaining / 8.0))

    if joker.name in ROUND_RESET_SCORE_JOKERS:
        if state.ante >= 7:
            factor *= 0.62
        elif state.ante >= 5:
            factor *= 0.78

    if joker.name == "Gros Michel":
        if state.ante >= 7:
            factor *= 0.72
        elif state.ante >= 5:
            factor *= 0.88

    return max(0.0, min(1.0, factor))


def _joker_is_disabled_for_build(joker: Joker) -> bool:
    if joker.effect.disabled:
        return True
    for source in _joker_metadata_sources(joker.metadata):
        state = source.get("state")
        if isinstance(state, dict) and _truthy_modifier(state.get("debuff")):
            return True
        if _truthy_modifier(source.get("debuff")) or _truthy_modifier(source.get("disabled")):
            return True
    return False


def _joker_has_sticker(joker: Joker, sticker: str) -> bool:
    wanted = sticker.lower()
    for source in _joker_metadata_sources(joker.metadata):
        for key in (wanted, sticker, f"is_{wanted}"):
            if _truthy_modifier(source.get(key)):
                return True
        stickers = source.get("stickers")
        if isinstance(stickers, dict):
            if _truthy_modifier(stickers.get(wanted)) or _truthy_modifier(stickers.get(sticker)):
                return True
        elif isinstance(stickers, list | tuple | set):
            if any(str(item).strip().lower() == wanted for item in stickers):
                return True
        elif isinstance(stickers, str):
            normalized = stickers.lower().replace("_", " ")
            if wanted in normalized:
                return True
    return False


def _joker_remaining_count(joker: Joker) -> int | None:
    value = _joker_metadata_value(
        joker,
        (
            "current_remaining",
            "remaining",
            "uses",
            "hands_remaining",
            "hands_left",
            "cards_remaining",
        ),
    )
    if value is None:
        value = joker.effect.remaining or joker.effect.cards_remaining or joker.effect.next_hands
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _joker_metadata_value(joker: Joker, keys: tuple[str, ...]) -> object | None:
    for source in _joker_metadata_sources(joker.metadata):
        for key in keys:
            if key in source:
                return source[key]
    return None


def _joker_metadata_sources(metadata: dict[str, object]) -> tuple[dict[str, object], ...]:
    sources: list[dict[str, object]] = [metadata]
    for key in ("ability", "config", "extra", "modifier", "stickers"):
        value = metadata.get(key)
        if isinstance(value, dict):
            sources.append(value)
            nested_extra = value.get("extra")
            if isinstance(nested_extra, dict):
                sources.append(nested_extra)
    return tuple(sources)


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
    if _joker_is_disabled_for_build(joker):
        return frozenset()

    roles: set[str] = set()
    name = joker.name
    if name in CHIP_JOKERS:
        roles.add("chips")
    if name in MULT_JOKERS:
        roles.add("mult")
    if name in XMULT_JOKERS and name != "Loyalty Card":
        roles.add("xmult")
    if name == "Red Card":
        if _joker_current_plus_value(joker, suffix="mult") > 0:
            roles.add("mult")
            roles.add("scaling")
    elif name in JOKER_SCALING_VALUES or name in SCALING_JOKERS:
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
    durability = _joker_late_durability_factor(state, joker)
    bonus = 0.0
    for role in profile.missing_roles:
        if role in roles:
            bonus += ROLE_MISSING_BONUSES[role] * max(0.35, profile.role_deficit_ratio(role)) * durability

    if profile.late and profile.rich:
        if "xmult" in roles and not profile.has_xmult:
            bonus += 22 * max(0.4, profile.role_deficit_ratio("xmult")) * durability
        if "scaling" in roles and not profile.has_scaling:
            bonus += 16 * max(0.4, profile.role_deficit_ratio("scaling")) * durability
        if "economy" in roles and profile.has_economy:
            bonus -= 10
    if profile.ante <= 2 and ("chips" in roles or "mult" in roles):
        bonus += 6
    if profile.open_joker_slots <= 1 and roles.isdisjoint({"xmult", "scaling"}) and profile.late:
        bonus -= 12
    if profile.late and durability < 1.0:
        bonus -= (1.0 - durability) * 16.0
    return bonus


def _pressure_joker_role_bonus(state: GameState, joker: Joker, pressure: _ShopPressure) -> float:
    if pressure.ratio < 0.95:
        return 0.0
    rare_target = RARE_HAND_JOKER_TARGETS.get(joker.name)
    if rare_target is not None and _rare_hand_deck_manipulation_need(state, rare_target) > 0:
        return 0.0

    profile = _build_profile(state)
    roles = _joker_roles(joker)
    durability = _joker_late_durability_factor(state, joker)
    bonus = 0.0
    if "scaling" in roles and not profile.has_scaling:
        deficit = max(0.35, profile.role_deficit_ratio("scaling"))
        bonus += (18.0 + pressure.danger * 28.0 + max(0, state.ante - 2) * 4.0) * deficit * durability
        if _urgent_late_role_hunt(state, pressure, profile):
            bonus += 16.0 * deficit * durability
    if "xmult" in roles and (not profile.has_xmult or pressure.ratio >= 1.15):
        deficit = max(0.35, profile.role_deficit_ratio("xmult"))
        bonus += (10.0 + pressure.danger * 18.0) * deficit * durability
        if _urgent_late_role_hunt(state, pressure, profile):
            bonus += 18.0 * deficit * durability
    if roles.isdisjoint({"xmult", "scaling"}) and profile.open_joker_slots <= 1 and pressure.ratio >= 1.1:
        bonus -= 10.0
    if roles.isdisjoint({"xmult", "scaling"}) and _urgent_late_role_hunt(state, pressure, profile):
        bonus -= 12.0
    if joker.name in TWO_PAIR_SUPPORT_JOKERS and not _has_dedicated_two_pair_plan(state):
        bonus -= 12.0
    if profile.late and durability < 1.0:
        bonus -= (1.0 - durability) * 14.0
    return bonus


def _future_score_headroom_joker_bonus(state: GameState, joker: Joker, pressure: _ShopPressure) -> float:
    if state.ante >= 8 or _joker_is_disabled_for_build(joker):
        return 0.0

    current_capacity = _shop_build_capacity_for_jokers(state, state.jokers, pressure)
    candidate_jokers = _jokers_after_buy_for_scoring(state, joker)
    candidate_capacity = _shop_build_capacity_for_jokers(state, candidate_jokers, pressure)
    if candidate_capacity <= current_capacity:
        return 0.0

    future_target = _future_headroom_required_score(state)
    current_margin = current_capacity / max(1.0, future_target)
    candidate_margin = candidate_capacity / max(1.0, future_target)
    gain_ratio = candidate_capacity / max(1.0, current_capacity)

    if candidate_margin >= 1.05 and current_margin < 0.95:
        bonus = 24.0
        bonus += min(30.0, (candidate_margin - 1.0) * 24.0)
        bonus += min(18.0, (gain_ratio - 1.0) * 8.0)
    elif candidate_margin >= 0.85 and gain_ratio >= 1.75:
        bonus = 14.0 + min(24.0, (gain_ratio - 1.0) * 7.0)
    else:
        return 0.0

    if state.ante <= 3:
        bonus += 6.0
    durability = _joker_late_durability_factor(state, joker)
    if durability < 1.0:
        bonus *= durability
    return min(64.0, bonus)


def _shop_build_capacity_for_jokers(
    state: GameState,
    jokers: tuple[Joker, ...],
    pressure: _ShopPressure,
) -> float:
    raw_target = pressure.target_score / max(0.1, pressure.safety_multiplier)
    score_state = replace(state, jokers=jokers)
    score_state = _shop_pressure_score_state(score_state, pressure.boss_name, raw_target=raw_target)
    current_score = _sample_build_score(score_state, score_state.jokers) * _shop_hand_realism_factor(score_state)
    effective_hands = _shop_pressure_effective_hands(state, pressure.boss_name)
    raw_capacity = max(1.0, current_score * effective_hands * 0.85)
    return max(1.0, raw_capacity * pressure.capacity_safety_factor)


def _future_headroom_required_score(state: GameState) -> float:
    ante = min(8, max(1, state.ante) + 2)
    small = ANTE_SMALL_BLIND_SCORES.get(ante, _extrapolated_small_blind_score(ante))
    return small * 2.0


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


def _scaling_commitment_shop_bonus(state: GameState, card: object, pressure: _ShopPressure) -> float:
    names = _active_joker_names(state)
    if _is_planet_card(card):
        bonus = 0.0
        if "Constellation" in names:
            bonus += 18.0 + pressure.danger * 12.0
        if "Obelisk" in names:
            bonus += 8.0
        return bonus
    if _is_playing_card(card):
        bonus = 0.0
        if "Hologram" in names:
            bonus += 18.0 + pressure.danger * 8.0
        if "Vampire" in names and _card_modifier(card).get("enhancement"):
            bonus += 18.0 + pressure.danger * 8.0
        return bonus
    return 0.0


def _scaling_commitment_pack_bonus(state: GameState, pack: object, pressure: _ShopPressure) -> float:
    names = _active_joker_names(state)
    label = _card_label(pack).lower()
    bonus = 0.0
    if "celestial" in label and "Constellation" in names:
        bonus += 18.0 + pressure.danger * 12.0
    if "standard" in label and names & {"Hologram", "Vampire"}:
        bonus += 16.0 + pressure.danger * 10.0
    if "arcana" in label and "Vampire" in names:
        bonus += 8.0 + pressure.danger * 8.0
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
    if hand_type in ADVANCED_HAND_TYPES:
        viability = hand_type_viability_multiplier(state, hand_type)
        if viability < 1.0:
            value *= 0.20 + (0.80 * viability)
            value -= (1.0 - viability) * 18.0
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
    if "Constellation" in _active_joker_names(state):
        value += 18.0
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
    if name == "The Hermit":
        value += 8
        if state.money >= 10:
            value += 5
    elif name == "Temperance" and state.ante <= 2:
        value -= 3
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


def _voucher_value(state: GameState, voucher: object, pressure: _ShopPressure | None = None) -> float:
    name = _card_label(voucher)
    if any(existing == name for existing in state.vouchers):
        return 0.0
    if _voucher_buy_is_blocked(state, name):
        return 0.0

    pressure = pressure or _shop_pressure(state)
    if _voucher_does_not_solve_current_boss_shop(state, name, pressure):
        return 0.0
    if _late_pressure_blocks_voucher(state, name, pressure):
        return 0.0

    value = float(VOUCHER_VALUES.get(name, 22))
    value += _voucher_dynamic_adjustment(state, name, pressure)
    return max(0.0, value)


def _voucher_buy_is_blocked(state: GameState, name: str) -> bool:
    if name == "Blank" and state.ante >= 8:
        return True
    return _voucher_name_key(name) in VOUCHER_BUY_DENYLIST


def _voucher_name_key(name: str) -> str:
    return name.replace("'", "").strip().lower()


def _voucher_does_not_solve_current_boss_shop(
    state: GameState,
    name: str,
    pressure: _ShopPressure,
) -> bool:
    if name not in {"Grabber", "Nacho Tong"}:
        return False
    return _shop_pressure_uses_exact_needle_hand(state, pressure.boss_name)


def _late_pressure_blocks_voucher(state: GameState, name: str, pressure: _ShopPressure) -> bool:
    if state.ante < 5 or pressure.ratio < 1.15:
        return False
    if name == "Blank":
        return state.ante >= 8
    if name in VOUCHER_PRESSURE_ALLOWED_NAMES:
        return False
    if name == "Observatory":
        return not any(planet in PLANET_TO_HAND for planet in state.consumables)
    if name in {"Seed Money", "Money Tree"}:
        return not _has_money_scaling_joker(state)
    return True


def _voucher_dynamic_adjustment(state: GameState, name: str, pressure: _ShopPressure) -> float:
    profile = _build_profile(state)
    danger = pressure.danger
    spendable = _spendable_money(state, pressure)
    bonus = 0.0

    if state.ante <= 2:
        bonus += 8.0
    if state.money >= 20:
        bonus += 8.0

    if name in {"Grabber", "Nacho Tong"}:
        bonus += _hand_voucher_adjustment(state, pressure, profile)
    elif name in {"Wasteful", "Recyclomancy"}:
        bonus += _discard_voucher_adjustment(state, pressure, profile)
    elif name in {"Paint Brush", "Palette"}:
        bonus += _hand_size_voucher_adjustment(state, pressure, profile)
    elif name in {"Overstock", "Overstock Plus"}:
        bonus += _shop_slot_voucher_adjustment(state, pressure, profile)
    elif name in {"Clearance Sale", "Liquidation"}:
        bonus += _discount_voucher_adjustment(state, pressure, profile)
    elif name in {"Reroll Surplus", "Reroll Glut"}:
        bonus += _reroll_voucher_adjustment(state, pressure, profile)
    elif name in {"Hone", "Glow Up"}:
        bonus += _edition_voucher_adjustment(state, pressure, profile)
    elif name == "Omen Globe":
        bonus += _omen_globe_voucher_adjustment(state, pressure, profile)
    elif name == "Observatory":
        bonus += _observatory_voucher_adjustment(state, pressure, profile)
    elif name in {"Seed Money", "Money Tree"}:
        bonus += _interest_cap_voucher_adjustment(state, name, pressure, profile)
    elif name in {"Hieroglyph", "Petroglyph"}:
        bonus += _ante_step_voucher_adjustment(state, pressure, profile)
    elif name == "Retcon":
        bonus += _retcon_voucher_adjustment(state, pressure, profile)
    elif name == "Antimatter":
        bonus += _antimatter_voucher_adjustment(state, pressure, profile)
    elif name in {"Tarot Merchant", "Tarot Tycoon", "Planet Tycoon", "Illusion"}:
        bonus += _generator_voucher_adjustment(state, name, pressure, profile)

    if state.ante >= 7 and pressure.raw_ratio >= 1.0 and name not in VOUCHER_IMMEDIATE_SCORE_NAMES:
        bonus -= 10.0 + min(14.0, danger * 8.0)
    if state.ante >= 8 and spendable < 15 and name not in VOUCHER_IMMEDIATE_SCORE_NAMES:
        bonus -= 8.0
    return bonus


def _hand_voucher_adjustment(
    state: GameState,
    pressure: _ShopPressure,
    profile: _BuildProfile,
) -> float:
    bonus = 0.0
    if pressure.ratio >= 0.9:
        bonus += 8.0 + pressure.danger * 18.0
    if state.ante >= 5 and pressure.ratio < 0.75:
        bonus -= 8.0
    if profile.preferred_hand in {HandType.STRAIGHT, HandType.FULL_HOUSE, HandType.FOUR_OF_A_KIND, HandType.FIVE_OF_A_KIND}:
        bonus += 4.0
    return bonus


def _discard_voucher_adjustment(
    state: GameState,
    pressure: _ShopPressure,
    profile: _BuildProfile,
) -> float:
    if pressure.boss_name == "The Water" and _shop_cleared_blind_kind(state) == "BIG":
        return -18.0
    bonus = 0.0
    if state.ante <= 3:
        bonus += 4.0
    if profile.preferred_hand in {
        HandType.STRAIGHT,
        HandType.FLUSH,
        HandType.FULL_HOUSE,
        HandType.FOUR_OF_A_KIND,
        HandType.FIVE_OF_A_KIND,
        HandType.STRAIGHT_FLUSH,
        HandType.FLUSH_HOUSE,
        HandType.FLUSH_FIVE,
    }:
        bonus += 8.0
    if pressure.ratio >= 0.95 and profile.preferred_hand in RARE_HAND_TYPES:
        bonus += 6.0 + pressure.danger * 8.0
    if state.ante >= 6 and pressure.ratio < 0.75:
        bonus -= 6.0
    return bonus


def _hand_size_voucher_adjustment(
    state: GameState,
    pressure: _ShopPressure,
    profile: _BuildProfile,
) -> float:
    bonus = 6.0 if state.ante >= 4 else 2.0
    if pressure.boss_name == "The Needle" and _shop_cleared_blind_kind(state) == "BIG":
        bonus += 10.0 + pressure.danger * 16.0
    elif pressure.ratio >= 0.9:
        bonus += 6.0 + pressure.danger * 14.0
    if profile.preferred_hand in {
        HandType.STRAIGHT,
        HandType.FLUSH,
        HandType.FULL_HOUSE,
        HandType.FOUR_OF_A_KIND,
        HandType.FIVE_OF_A_KIND,
        HandType.STRAIGHT_FLUSH,
        HandType.FLUSH_HOUSE,
        HandType.FLUSH_FIVE,
    }:
        bonus += 8.0
    if not profile.has_xmult and state.ante >= 5:
        bonus += 4.0
    return bonus


def _shop_slot_voucher_adjustment(
    state: GameState,
    pressure: _ShopPressure,
    profile: _BuildProfile,
) -> float:
    bonus = max(0.0, (6 - state.ante) * 3.0)
    if profile.open_joker_slots > 0 and ("xmult" in profile.missing_roles or "scaling" in profile.missing_roles):
        bonus += 8.0
    if profile.rich and state.ante <= 6:
        bonus += 6.0
    if state.ante >= 7:
        bonus -= 8.0
    if pressure.ratio >= 1.15 and state.ante >= 6:
        bonus -= 6.0
    return bonus


def _discount_voucher_adjustment(
    state: GameState,
    pressure: _ShopPressure,
    profile: _BuildProfile,
) -> float:
    if pressure.ratio >= 0.95:
        penalty = 8.0 + min(18.0, pressure.danger * 12.0)
        if state.ante >= 4:
            penalty += 6.0
        if state.ante >= 5:
            penalty += 8.0
        if profile.rich and pressure.ratio < 1.05 and state.ante <= 4:
            penalty -= 4.0
        return -penalty

    bonus = max(0.0, (7 - state.ante) * 3.0)
    if profile.rich and state.ante <= 6:
        bonus += 8.0
    if pressure.ratio >= 1.0 and state.money >= 30:
        bonus += 6.0
    if state.ante >= 7 and pressure.ratio < 1.0:
        bonus -= 8.0
    return bonus


def _reroll_voucher_adjustment(
    state: GameState,
    pressure: _ShopPressure,
    profile: _BuildProfile,
) -> float:
    bonus = 0.0
    if "xmult" in profile.missing_roles or "scaling" in profile.missing_roles:
        bonus += 8.0
    if pressure.ratio >= 0.95:
        bonus += 6.0 + pressure.danger * 14.0
    if profile.rich:
        bonus += 6.0
    if state.ante >= 7 and not profile.rich:
        bonus -= 8.0
    return bonus


def _edition_voucher_adjustment(
    state: GameState,
    pressure: _ShopPressure,
    profile: _BuildProfile,
) -> float:
    bonus = 4.0 if state.ante <= 5 else 0.0
    if profile.open_joker_slots > 0:
        bonus += 4.0
    if pressure.ratio >= 0.95 and ("xmult" in profile.missing_roles or "scaling" in profile.missing_roles):
        bonus += 4.0 + pressure.danger * 8.0
    if state.ante >= 7:
        bonus -= 6.0
    return bonus


def _omen_globe_voucher_adjustment(
    state: GameState,
    pressure: _ShopPressure,
    profile: _BuildProfile,
) -> float:
    if not _has_consumable_room(state):
        return -10.0
    bonus = 0.0
    if profile.rich and state.ante <= 6:
        bonus += 6.0
    if _rare_hand_plan(state) is not None:
        bonus += 6.0
    if state.ante >= 7 and pressure.ratio >= 1.0:
        bonus -= 6.0
    return bonus


def _observatory_voucher_adjustment(
    state: GameState,
    pressure: _ShopPressure,
    profile: _BuildProfile,
) -> float:
    planet_count = sum(1 for name in state.consumables if name in PLANET_TO_HAND)
    if planet_count <= 0:
        return -18.0
    bonus = planet_count * 14.0
    if profile.preferred_hand is not None and _planet_for_hand(profile.preferred_hand) in state.consumables:
        bonus += 10.0
    if pressure.ratio >= 0.9:
        bonus += 6.0 + pressure.danger * 10.0
    return bonus


def _interest_cap_voucher_adjustment(
    state: GameState,
    name: str,
    pressure: _ShopPressure,
    profile: _BuildProfile,
) -> float:
    current_cap = _interest_cap_money(state)
    new_cap = VOUCHER_INTEREST_CAP_MONEY.get(name, current_cap)
    cap_gain = max(0, new_cap - current_cap)
    if cap_gain <= 0:
        return -10.0
    runway = max(0, 8 - state.ante)
    bonus = min(18.0, cap_gain / 5.0) + runway * 2.0
    if state.money >= current_cap:
        bonus += 6.0
    if _has_money_scaling_joker(state):
        bonus += 8.0
    if state.ante >= 6 and pressure.ratio >= 1.0 and not _has_money_scaling_joker(state):
        bonus -= 16.0
    if profile.late and not profile.rich:
        bonus -= 8.0
    return bonus


def _ante_step_voucher_adjustment(
    state: GameState,
    pressure: _ShopPressure,
    profile: _BuildProfile,
) -> float:
    if state.ante <= 1:
        return -30.0
    if pressure.ratio >= 0.75 or pressure.raw_ratio >= 0.75:
        return -28.0
    if not profile.has_xmult and state.ante >= 4:
        return -18.0
    return 6.0 if state.ante <= 5 else -10.0


def _retcon_voucher_adjustment(
    state: GameState,
    pressure: _ShopPressure,
    profile: _BuildProfile,
) -> float:
    if _shop_cleared_blind_kind(state) != "BIG" or not pressure.boss_name:
        return -12.0
    if pressure.boss_name not in DANGEROUS_BOSS_BLINDS:
        return -6.0
    bonus = 12.0 + pressure.danger * 18.0
    if profile.rich:
        bonus += 6.0
    if state.money < 20:
        bonus -= 8.0
    return bonus


def _antimatter_voucher_adjustment(
    state: GameState,
    pressure: _ShopPressure,
    profile: _BuildProfile,
) -> float:
    bonus = 26.0
    if profile.open_joker_slots <= 0 and ("xmult" in profile.missing_roles or "scaling" in profile.missing_roles):
        bonus += 14.0
    if pressure.ratio >= 0.9:
        bonus += 8.0 + pressure.danger * 12.0
    return bonus


def _generator_voucher_adjustment(
    state: GameState,
    name: str,
    pressure: _ShopPressure,
    profile: _BuildProfile,
) -> float:
    bonus = 0.0
    if name in {"Tarot Merchant", "Tarot Tycoon", "Illusion"} and _rare_hand_plan(state) is not None:
        bonus += 8.0
    if name == "Planet Tycoon" and profile.preferred_hand is not None and not _has_planet_investment(state):
        bonus += 6.0
    if state.ante >= 6 and pressure.ratio >= 1.0:
        bonus -= 10.0
    return bonus


def _planet_for_hand(hand_type: HandType) -> str | None:
    for planet, planet_hand in PLANET_TO_HAND.items():
        if planet_hand == hand_type:
            return planet
    return None


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


def _is_buffoon_pack(pack: object) -> bool:
    return "buffoon" in _card_label(pack).lower()


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
    if _late_pressure_closer_mode(state, pressure):
        if pressure.ratio >= 3.0 or pressure.raw_ratio >= 1.75:
            return 5
        return 4
    if pressure.ratio >= 1.05:
        return 2
    return 1


def _pack_capacity_gain(state: GameState, pack: object, pressure: _ShopPressure) -> float:
    name = _card_label(pack).lower()
    current = _sample_build_score(state, state.jokers)
    spend_loss = _score_loss_after_spending(state, _card_cost(pack), current_score=current)
    profile = _build_profile(state)

    if "celestial" in name:
        gain = _celestial_pack_capacity_gain(state, pack, current_score=current)
        if "Constellation" in _active_joker_names(state):
            gain += current * 0.10
        return gain
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
        rate = 0.04
        if _active_joker_names(state) & {"Hologram", "Vampire"}:
            rate += 0.08
        return (current * rate) + _rare_hand_pack_capacity_bonus(state, current, "standard") - spend_loss
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
    money_penalty = _money_after_spend_penalty(state, cost, pressure)
    if _is_joker_card(card):
        money_penalty *= _economy_joker_interest_penalty_scale(state, _joker_from_shop_card(card), pressure)
    return cost * cost_weight + money_penalty


def _economy_joker_interest_penalty_scale(state: GameState, joker: Joker, pressure: _ShopPressure) -> float:
    """Strong econ buys can dip below interest when the build is already safe."""

    economy_value = JOKER_ECONOMY_VALUES.get(joker.name, 0)
    if economy_value <= 0:
        return 1.0
    if state.ante <= 1 and not _has_real_scoring_joker(state):
        return 1.0
    if pressure.raw_ratio >= 1.0 or pressure.ratio >= 0.95:
        return 1.0
    if economy_value >= 24:
        return 0.25
    if economy_value >= 18:
        return 0.40
    return 0.60


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


def _late_pressure_interest_floor(state: GameState, pressure: _ShopPressure) -> int:
    cap = _interest_cap_money(state)
    if state.ante < 5:
        return cap
    if pressure.ratio >= 3.0 or pressure.raw_ratio >= 1.75:
        return max(20, int(cap * 0.65))
    if pressure.ratio >= 1.75 or pressure.raw_ratio >= 1.2:
        return max(20, int(cap * 0.8))
    return cap


def _desired_money_reserve(state: GameState, pressure: _ShopPressure | None = None) -> int:
    reserve = _interest_cap_money(state)
    owned_jokers = {joker.name for joker in state.jokers}
    for joker_name, target in MONEY_SCALING_RESERVE_TARGETS.items():
        if joker_name in owned_jokers:
            reserve = max(reserve, target)
    if {"Bull", "Bootstraps"}.issubset(owned_jokers):
        reserve = max(reserve, 100)
    if pressure is not None and not owned_jokers.intersection(MONEY_SCALING_RESERVE_TARGETS):
        reserve = min(reserve, _late_pressure_interest_floor(state, pressure))
    if pressure is not None:
        closing_cap = _late_closing_money_reserve_cap(state, pressure, owned_jokers)
        if closing_cap is not None:
            reserve = min(reserve, closing_cap)
    return reserve


def _late_closing_money_reserve_cap(
    state: GameState,
    pressure: _ShopPressure,
    owned_jokers: set[str],
) -> int | None:
    if state.ante < 5:
        return None
    if state.money < 45:
        return None

    bank_conversion_cap = _late_bank_conversion_reserve_cap(state, pressure, owned_jokers)
    if bank_conversion_cap is not None:
        return bank_conversion_cap

    if pressure.ratio < 1.05 and pressure.raw_ratio < 0.95:
        return None

    has_money_scaling = bool(owned_jokers.intersection(MONEY_SCALING_RESERVE_TARGETS))
    if has_money_scaling:
        missing_roles = set(_build_profile(state).missing_roles)
        if not (missing_roles & {"mult", "xmult", "scaling"}):
            return None
        if state.ante < 7 and pressure.ratio < 3.0 and pressure.raw_ratio < 1.75:
            return None
        if pressure.ratio >= 1.75 or pressure.raw_ratio >= 1.25:
            return 35
        if state.ante >= 8 or pressure.ratio >= 1.25:
            return 50
        return None

    floor = _late_pressure_interest_floor(state, pressure)
    return floor if floor < _interest_cap_money(state) else None


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
    if _can_use_direct_best_play_score(ordered_state, context):
        try:
            evaluation = best_play_from_hand(
                ordered_state.hand,
                ordered_state.hand_levels,
                debuffed_suits=debuffed_suits_for_blind(ordered_state.blind),
                blind_name=ordered_state.blind,
                jokers=jokers,
                discards_remaining=ordered_state.discards_remaining,
                hands_remaining=ordered_state.hands_remaining,
                deck_size=ordered_state.deck_size,
                money=ordered_state.money,
                played_hand_types_this_round=context.played_hand_types,
                played_hand_counts=_played_hand_counts(ordered_state),
            )
            return evaluation.score
        except ValueError:
            return 0
    return max((candidate.score for candidate in _play_candidates(ordered_state, context)), default=0)


def _can_use_direct_best_play_score(state: GameState, context: _BlindContext) -> bool:
    if context.played_hand_types and state.blind in {"The Eye", "The Mouth"}:
        return False
    legal_play_count = sum(1 for action in state.legal_actions if action.action_type == ActionType.PLAY_HAND and action.card_indices)
    return legal_play_count == _full_play_action_count(len(state.hand))


def _full_play_action_count(card_count: int) -> int:
    total = 0
    for size in range(1, min(5, card_count) + 1):
        total += comb(card_count, size)
    return total


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
    joker_context = _prepare_joker_evaluation_context(state.jokers)
    return [_play_candidate(state, action, context, joker_context=joker_context) for action in play_actions]


def _play_candidate(
    state: GameState,
    action: Action,
    context: _BlindContext | None = None,
    *,
    joker_context=None,
) -> _PlayCandidate:
    context = context or _BlindContext()
    evaluation = _evaluate_play_action(state, action, context, joker_context=joker_context)
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


def _score_play_action(
    state: GameState,
    action: Action,
    context: _BlindContext | None = None,
    *,
    joker_context=None,
) -> int:
    context = context or _BlindContext()
    evaluation = _evaluate_play_action(state, action, context, joker_context=joker_context)
    return _boss_adjusted_score(state, evaluation.hand_type, evaluation.score, context)


def _evaluate_play_action(
    state: GameState,
    action: Action,
    context: _BlindContext | None = None,
    *,
    joker_context=None,
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
        _joker_context=joker_context,
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
    return _identity_cached_value("played_hand_counts", state, lambda: _played_hand_counts_uncached(state))


def _played_hand_counts_uncached(state: GameState) -> dict[str, int]:
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
    if score >= remaining_score > 0:
        economy_hunt = _winning_economy_hunt_discard_action(state, best_play, score, remaining_score, context)
        if economy_hunt is not None:
            return economy_hunt

    first_blind_discard = _first_blind_one_hand_hunt_action(state, best_play, score, remaining_score, context)
    if first_blind_discard is not None:
        return first_blind_discard

    opening_setup = _opening_joker_setup_play_action(state, best_play, score, remaining_score, context)
    if opening_setup is not None:
        return opening_setup

    strategic_discard = _strategic_joker_discard_action(state, best_play, score, remaining_score, context)
    if strategic_discard is not None:
        return strategic_discard

    mystic_setup_discard = _mystic_summit_setup_discard_action(state, best_play, score, remaining_score, context)
    if mystic_setup_discard is not None:
        return mystic_setup_discard

    clear_line_hunt = _preferred_hand_hunt_discard_action(
        state,
        best_play,
        score,
        remaining_score,
        context,
        clear_line_only=True,
    )
    if clear_line_hunt is not None:
        banner_play = _banner_vetoed_play_action(state, best_play, clear_line_hunt, score, remaining_score, context)
        if banner_play is not None:
            return banner_play
        return clear_line_hunt

    clear_line_redraw = _preferred_hand_hunt_redraw_play_action(state, best_play, score, remaining_score, context)
    if clear_line_redraw is not None:
        return clear_line_redraw

    if (
        remaining_score == 0
        or score >= remaining_score
        or state.discards_remaining <= 0
        or _estimated_hands_needed(remaining_score, score) < state.hands_remaining
    ):
        return _annotated_action(best_play, reason=_play_reason(state, best_play, context))

    best_discard = _best_discard_action(state, current_best_score=score, context=context)
    if (
        best_discard is not None
        and state.hands_remaining <= 1
        and _should_last_hand_hunt_discard(state, best_discard, score, remaining_score, context)
    ):
        banner_play = _banner_vetoed_play_action(state, best_play, best_discard, score, remaining_score, context)
        if banner_play is not None:
            return banner_play
        return _annotated_action(best_discard, reason=_last_hand_hunt_discard_reason(state, best_play, best_discard, context))

    preferred_hunt = _preferred_hand_hunt_discard_action(state, best_play, score, remaining_score, context)
    if preferred_hunt is not None:
        banner_play = _banner_vetoed_play_action(state, best_play, preferred_hunt, score, remaining_score, context)
        if banner_play is not None:
            return banner_play
        return preferred_hunt

    if best_discard is not None and _should_panic_discard(state, best_discard, score, remaining_score, context):
        banner_play = _banner_vetoed_play_action(state, best_play, best_discard, score, remaining_score, context)
        if banner_play is not None:
            return banner_play
        return _annotated_action(best_discard, reason=_panic_discard_reason(state, best_play, best_discard, context))

    if best_discard is not None and _discard_can_reduce_hands_needed(state, best_discard, score, remaining_score, context):
        banner_play = _banner_vetoed_play_action(state, best_play, best_discard, score, remaining_score, context)
        if banner_play is not None:
            return banner_play
        return _annotated_action(best_discard, reason=_discard_reason(state, best_play, best_discard, context))

    if best_discard is not None and _should_safety_discard(state, best_discard, score, remaining_score, context):
        banner_play = _banner_vetoed_play_action(state, best_play, best_discard, score, remaining_score, context)
        if banner_play is not None:
            return banner_play
        return _annotated_action(best_discard, reason=_safety_discard_reason(state, best_play, best_discard, context))

    if _score_is_on_pace(state, score, remaining_score):
        return _annotated_action(best_play, reason=_play_reason(state, best_play, context))

    if best_discard is not None and _should_chase_discard(state, best_discard, score, remaining_score, context):
        banner_play = _banner_vetoed_play_action(state, best_play, best_discard, score, remaining_score, context)
        if banner_play is not None:
            return banner_play
        return _annotated_action(best_discard, reason=_discard_reason(state, best_play, best_discard, context))
    return _annotated_action(best_play, reason=_play_reason(state, best_play, context))


def _banner_vetoed_play_action(
    state: GameState,
    best_play: Action,
    discard: Action,
    current_score: int,
    remaining_score: int,
    context: _BlindContext,
) -> Action | None:
    reason = _banner_preserve_play_reason(state, best_play, discard, current_score, remaining_score, context)
    if reason is None:
        return None
    return _annotated_action(best_play, reason=reason)


def _banner_preserve_play_reason(
    state: GameState,
    best_play: Action,
    best_discard: Action,
    current_score: int,
    remaining_score: int,
    context: _BlindContext,
) -> str | None:
    if (
        "Banner" not in _active_joker_names(state)
        or state.hands_remaining <= 0
        or state.discards_remaining <= 0
        or current_score <= 0
        or remaining_score <= 0
        or not best_discard.card_indices
    ):
        return None

    reduced_state = replace(state, discards_remaining=max(0, state.discards_remaining - 1))
    reduced_current_score = _score_play_action(reduced_state, best_play, context)
    banner_tax = max(0, current_score - reduced_current_score)
    if banner_tax <= 0:
        return None

    projected_score = _projected_score_after_discard(state, best_discard, context)
    if projected_score >= remaining_score:
        return None

    current_hands_needed = _estimated_hands_needed(remaining_score, current_score)
    projected_hands_needed = _estimated_hands_needed(remaining_score, projected_score)

    if state.hands_remaining <= 1:
        return None

    raw_pace = remaining_score / max(1, state.hands_remaining)
    if current_score < raw_pace:
        if state.ante >= 8 and _is_boss_blind(state) and projected_score < current_score:
            immediate_gain = projected_score - current_score
            future_hands = max(0, min(state.hands_remaining - 1, projected_hands_needed - 1))
            future_banner_tax = banner_tax * future_hands
            required_gain = banner_tax * BANNER_DISCARD_FUTURE_TAX_WEIGHT
            return _banner_ev_reason(
                state,
                current_score=current_score,
                projected_score=projected_score,
                immediate_gain=immediate_gain,
                banner_tax=banner_tax,
                future_banner_tax=future_banner_tax,
                required_gain=required_gain,
                current_hands_needed=current_hands_needed,
                projected_hands_needed=projected_hands_needed,
            )
        return None

    if projected_hands_needed < current_hands_needed and projected_hands_needed <= state.hands_remaining:
        return None

    immediate_gain = projected_score - current_score
    future_hands = max(0, min(state.hands_remaining - 1, current_hands_needed - 1, projected_hands_needed - 1))
    future_banner_tax = banner_tax * future_hands
    required_gain = future_banner_tax * BANNER_DISCARD_FUTURE_TAX_WEIGHT
    if projected_score > current_score:
        if state.ante < 8:
            return None
        if immediate_gain >= required_gain:
            return None
        return _banner_ev_reason(
            state,
            current_score=current_score,
            projected_score=projected_score,
            immediate_gain=immediate_gain,
            banner_tax=banner_tax,
            future_banner_tax=future_banner_tax,
            required_gain=required_gain,
            current_hands_needed=current_hands_needed,
            projected_hands_needed=projected_hands_needed,
        )

    return _banner_ev_reason(
        state,
        current_score=current_score,
        projected_score=projected_score,
        immediate_gain=immediate_gain,
        banner_tax=banner_tax,
        future_banner_tax=future_banner_tax,
        required_gain=required_gain,
        current_hands_needed=current_hands_needed,
        projected_hands_needed=projected_hands_needed,
    )


def _banner_ev_reason(
    state: GameState,
    *,
    current_score: int,
    projected_score: int,
    immediate_gain: int,
    banner_tax: int,
    future_banner_tax: float,
    required_gain: float,
    current_hands_needed: int,
    projected_hands_needed: int,
) -> str:
    return (
        "preserve_banner_ev "
        f"current_score={current_score} projected_score={projected_score} "
        f"gain={immediate_gain} banner_tax={banner_tax} "
        f"future_tax={future_banner_tax:.1f} required_gain={required_gain:.1f} "
        f"hands_needed={current_hands_needed}->{projected_hands_needed} "
        f"hands_left={state.hands_remaining} discards_left={state.discards_remaining}"
    )


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


def _winning_economy_hunt_discard_action(
    state: GameState,
    best_play: Action,
    score: int,
    remaining_score: int,
    context: _BlindContext | None = None,
) -> Action | None:
    context = context or _BlindContext()
    if (
        remaining_score <= 0
        or score < remaining_score
        or state.discards_remaining <= 0
        or not state.known_deck
    ):
        return None

    baseline_value = _clear_economy_value_for_play(state, best_play, context)
    ranked: list[tuple[tuple[float, float, int, int, int], Action, float, int]] = []
    for action in state.legal_actions:
        if action.action_type != ActionType.DISCARD or not action.card_indices:
            continue
        drawn_cards = _known_draw_for_discard(state, action)
        if not drawn_cards or not any(_card_is_economy_hunt_target(state, card) for card in drawn_cards):
            continue
        discarded_cards = tuple(state.hand[index] for index in action.card_indices)
        discard_value = _conditional_discard_money_delta_for_economy_hunt(state, discarded_cards, context)
        projected_state = _state_after_known_discard_for_economy_hunt(state, action, drawn_cards, context)
        projected_context = replace(context, discards_taken=context.discards_taken + 1)
        projected_value, projected_score = _best_clear_economy_value(projected_state, remaining_score, projected_context)
        projected_value += discard_value
        if projected_score < remaining_score:
            continue
        gain = projected_value - baseline_value
        if gain + SHOP_VALUE_TOLERANCE < WINNING_ECONOMY_HUNT_MIN_GAIN:
            continue
        ranked.append(
            (
                (
                    gain,
                    projected_value,
                    _drawn_economy_hunt_value(state, drawn_cards),
                    -len(action.card_indices),
                    -_action_index_sum(action),
                ),
                action,
                gain,
                projected_score,
            )
        )

    if not ranked:
        return None

    _, action, gain, projected_score = max(ranked, key=lambda item: item[0])
    return _annotated_action(
        action,
        reason=_winning_economy_hunt_discard_reason(state, best_play, action, gain, projected_score, context),
    )


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


def _truthy_modifier(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "none", "nil"}
    return bool(value)


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


def _int_or_default(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _best_clear_economy_value(
    state: GameState,
    remaining_score: int,
    context: _BlindContext,
) -> tuple[float, int]:
    best_value = float("-inf")
    best_score = 0
    max_cards = min(5, len(state.hand))
    for count in range(1, max_cards + 1):
        for indices in combinations(range(len(state.hand)), count):
            action = Action(ActionType.PLAY_HAND, card_indices=tuple(indices))
            evaluation = _evaluate_play_action(state, action, context)
            score = _boss_adjusted_score(state, evaluation.hand_type, evaluation.score, context)
            if score < remaining_score:
                continue
            value = _clear_economy_value_for_evaluated_play(state, action, evaluation, context)
            if value > best_value or (value == best_value and score > best_score):
                best_value = value
                best_score = score
    return best_value, best_score


def _clear_economy_value_for_play(
    state: GameState,
    action: Action,
    context: _BlindContext,
) -> float:
    evaluation = _evaluate_play_action(state, action, context)
    return _clear_economy_value_for_evaluated_play(state, action, evaluation, context)


def _clear_economy_value_for_evaluated_play(
    state: GameState,
    action: Action,
    evaluation,
    context: _BlindContext,
) -> float:
    played = set(action.card_indices)
    held_cards = tuple(card for index, card in enumerate(state.hand) if index not in played)
    return (
        float(evaluation.money_delta)
        + _held_round_end_economy_value(state, held_cards)
        + _discard_sensitive_cash_out_value(state, context)
    )


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


def _joker_metadata_numeric_value(joker: Joker, keys: tuple[str, ...]) -> object | None:
    sources: list[dict[str, object]] = [joker.metadata]
    for key in ("ability", "config", "extra", "value"):
        value = joker.metadata.get(key)
        if isinstance(value, dict):
            sources.append(value)
            nested = value.get("extra")
            if isinstance(nested, dict):
                sources.append(nested)
    for source in sources:
        for key in keys:
            if key in source:
                return source[key]
    return None


def _joker_metadata_int_value(joker: Joker, keys: tuple[str, ...], *, default: int) -> int:
    value = _joker_metadata_numeric_value(joker, keys)
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


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


def _basic_consumable_open_slots(state: GameState) -> int:
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


def _is_gold_enhancement(card: Card) -> bool:
    return not card.debuffed and _normalize_card_attr(card.enhancement) in {"gold", "gold card"}


def _is_blue_seal(card: Card) -> bool:
    return not card.debuffed and _normalize_card_attr(card.seal) == "blue"


def _is_gold_seal(card: Card) -> bool:
    return not card.debuffed and _normalize_card_attr(card.seal) == "gold"


def _normalize_card_attr(value: object) -> str:
    return str(value or "").strip().lower().replace("_", " ").replace("-", " ")


def _mystic_summit_setup_discard_action(
    state: GameState,
    best_play: Action,
    score: int,
    remaining_score: int,
    context: _BlindContext | None = None,
) -> Action | None:
    context = context or _BlindContext()
    names = _joker_names(state)
    if (
        "Mystic Summit" not in names
        or "Banner" in names
        or "Delayed Gratification" in names
        or "Green Joker" in names
        or "Ramen" in names
        or state.ante > 2
        or state.current_score > 0
        or remaining_score <= 0
        or score >= remaining_score
        or state.hands_remaining <= 1
        or state.discards_remaining <= 0
    ):
        return None

    active_state = replace(state, discards_remaining=0)
    active_score = _score_play_action(active_state, best_play, context)
    if active_score <= score:
        return None

    current_hands_needed = _estimated_hands_needed(remaining_score, score)
    active_hands_needed = _estimated_hands_needed(remaining_score, active_score)
    if active_score < remaining_score and active_hands_needed >= current_hands_needed:
        return None

    best_discard = _best_discard_action(state, current_best_score=score, context=context)
    if best_discard is None:
        return None

    if state.discards_remaining <= 1:
        projected_score = _projected_score_after_discard(state, best_discard, context)
        projected_hands_needed = _estimated_hands_needed(remaining_score, projected_score)
        if projected_score < remaining_score and projected_hands_needed >= current_hands_needed:
            return None

    return _annotated_action(
        best_discard,
        reason=_mystic_summit_setup_discard_reason(
            state,
            best_play,
            best_discard,
            score,
            active_score,
            context,
        ),
    )


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
    upgrade_discard = _ante_one_near_clear_upgrade_discard_action(
        state,
        best_play,
        best_discard,
        score,
        remaining_score,
        context,
    )
    if upgrade_discard is not None:
        return upgrade_discard
    return _annotated_action(best_discard, reason=_first_blind_discard_reason(state, best_play, best_discard, context))


def _ante_one_near_clear_upgrade_discard_action(
    state: GameState,
    best_play: Action,
    fallback_discard: Action,
    score: int,
    remaining_score: int,
    context: _BlindContext | None = None,
) -> Action | None:
    context = context or _BlindContext()
    if (
        state.ante != 1
        or state.blind != "Small Blind"
        or state.current_score != 0
        or context.discards_taken != 1
        or state.discards_remaining <= 0
        or state.jokers
        or remaining_score <= 0
        or score >= remaining_score
        or len(best_play.card_indices) != 5
        or score < remaining_score * ANTE_ONE_UPGRADE_NEAR_CLEAR_RATIO
    ):
        return None

    fallback_projected_score = _projected_score_after_discard(state, fallback_discard, context)
    if fallback_projected_score >= score * 0.75:
        return None

    evaluation = _evaluate_play_action(state, best_play, context)
    keep_candidates = _ante_one_upgrade_keep_candidates(state, best_play, evaluation.hand_type)
    if not keep_candidates:
        return None

    ranked: list[tuple[tuple[float, float, int, int], Action, int]] = []
    for keep_indices in keep_candidates:
        discard_indices = _ante_one_upgrade_discard_indices(state, keep_indices)
        if not discard_indices:
            continue
        discard = _matching_discard_action(state, discard_indices)
        if discard is None:
            continue

        projected_score = _projected_score_after_discard(state, discard, context)
        if not _ante_one_upgrade_projection_is_good(
            score,
            projected_score,
            remaining_score,
        ):
            continue
        core_score = _ante_one_upgrade_core_score(state, keep_indices)
        ranked.append(
            (
                (float(projected_score), core_score, -len(discard.card_indices), -_action_index_sum(discard)),
                discard,
                projected_score,
            )
        )

    if not ranked:
        return None

    _, discard, projected_score = max(ranked, key=lambda item: item[0])
    return _annotated_action(
        discard,
        reason=_ante_one_upgrade_discard_reason(
            state,
            best_play,
            discard,
            evaluation.hand_type,
            projected_score,
            context,
        ),
    )


def _ante_one_upgrade_keep_candidates(
    state: GameState,
    best_play: Action,
    hand_type: HandType,
) -> tuple[tuple[int, ...], ...]:
    selected_indices = tuple(best_play.card_indices)
    candidates: list[tuple[int, ...]] = []

    if hand_type in {HandType.FLUSH, HandType.STRAIGHT_FLUSH, HandType.FLUSH_HOUSE, HandType.FLUSH_FIVE}:
        flush_core = _ante_one_flush_upgrade_core_indices(state, selected_indices)
        if flush_core:
            candidates.append(flush_core)

    if hand_type in {HandType.STRAIGHT, HandType.STRAIGHT_FLUSH}:
        straight_core = _ante_one_straight_upgrade_core_indices(state, selected_indices)
        if straight_core:
            candidates.append(straight_core)

    return _unique_index_tuples(candidates)


def _ante_one_flush_upgrade_core_indices(
    state: GameState,
    selected_indices: tuple[int, ...],
) -> tuple[int, ...] | None:
    selected_cards = tuple(state.hand[index] for index in selected_indices)
    suit = _dominant_suit_from_cards(selected_cards)
    if suit is None:
        return None

    suited_indices = tuple(
        index
        for index in selected_indices
        if _normalize_suit(state.hand[index].suit) == _normalize_suit(suit)
    )
    if len(suited_indices) < 5:
        return None

    keep = sorted(suited_indices, key=lambda index: _ante_one_card_upgrade_key(state.hand[index]), reverse=True)[:4]
    return tuple(sorted(keep))


def _ante_one_straight_upgrade_core_indices(
    state: GameState,
    selected_indices: tuple[int, ...],
) -> tuple[int, ...] | None:
    best: tuple[tuple[int, int, int, int], tuple[int, ...]] | None = None
    for core in combinations(selected_indices, 4):
        cards = tuple(state.hand[index] for index in core)
        if _straight_draw_potential(cards) < 4:
            continue
        key = (
            _straight_core_high_end(cards),
            sum(STRAIGHT_VALUES[card.rank] for card in cards),
            sum(RANK_VALUES[card.rank] for card in cards),
            -sum(core),
        )
        if best is None or key > best[0]:
            best = (key, tuple(sorted(core)))
    return best[1] if best is not None else None


def _ante_one_upgrade_discard_indices(
    state: GameState,
    keep_core_indices: tuple[int, ...],
) -> tuple[int, ...]:
    keep_indices = set(keep_core_indices)
    extra_keeps_needed = max(0, len(state.hand) - len(keep_indices) - 5)
    if extra_keeps_needed > 0:
        extras = sorted(
            (index for index in range(len(state.hand)) if index not in keep_indices),
            key=lambda index: _ante_one_card_upgrade_key(state.hand[index]),
            reverse=True,
        )
        keep_indices.update(extras[:extra_keeps_needed])
    return tuple(index for index in range(len(state.hand)) if index not in keep_indices)


def _matching_discard_action(state: GameState, discard_indices: tuple[int, ...]) -> Action | None:
    wanted = tuple(sorted(discard_indices))
    for action in state.legal_actions:
        if action.action_type == ActionType.DISCARD and tuple(sorted(action.card_indices)) == wanted:
            return action
    return None


def _ante_one_card_upgrade_key(card: Card) -> tuple[int, int]:
    return (RANK_VALUES[card.rank], STRAIGHT_VALUES[card.rank])


def _ante_one_upgrade_core_score(state: GameState, keep_indices: tuple[int, ...]) -> float:
    cards = tuple(state.hand[index] for index in keep_indices)
    return sum(RANK_VALUES[card.rank] for card in cards) + (_strong_draw_size(cards) * 8.0)


def _ante_one_upgrade_projection_is_good(
    score: int,
    projected_score: int,
    remaining_score: int,
) -> bool:
    if projected_score <= score:
        return False
    if projected_score >= remaining_score:
        return True
    target_score = max(
        score + ANTE_ONE_UPGRADE_MIN_GAIN,
        remaining_score * ANTE_ONE_UPGRADE_TARGET_RATIO,
    )
    return projected_score >= target_score


def _straight_core_high_end(cards: tuple[Card, ...]) -> int:
    values = {STRAIGHT_VALUES[card.rank] for card in cards}
    if any(card.rank == "A" for card in cards):
        values.add(1)

    best = 0
    for start in range(1, 11):
        if sum(1 for value in range(start, start + 5) if value in values) >= 4:
            best = max(best, start + 4)
    return best


def _unique_index_tuples(candidates: list[tuple[int, ...]]) -> tuple[tuple[int, ...], ...]:
    unique: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for candidate in candidates:
        normalized = tuple(sorted(candidate))
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return tuple(unique)


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


def _preferred_hand_hunt_discard_action(
    state: GameState,
    best_play: Action,
    score: int,
    remaining_score: int,
    context: _BlindContext | None = None,
    *,
    clear_line_only: bool = False,
) -> Action | None:
    context = context or _BlindContext()
    preferred = _preferred_hand_type(state)
    if (
        preferred not in PREFERRED_HAND_HUNT_TYPES
        or remaining_score <= 0
        or score >= remaining_score
        or state.hands_remaining <= 0
        or (state.hands_remaining <= 1 and not clear_line_only)
        or state.discards_remaining <= 0
    ):
        return None

    if not _preferred_hand_hunt_allowed(
        state,
        preferred,
        score,
        remaining_score,
        context,
        clear_line_only=clear_line_only,
    ):
        return None

    current_hand_type = _evaluate_play_action(state, best_play, context).hand_type
    if _hand_matches_preferred_family(current_hand_type, preferred):
        return None

    pace_score = remaining_score / max(1, state.hands_remaining)
    if score >= pace_score * _preferred_hunt_play_ceiling(state, preferred):
        return None

    discard_actions = [
        action
        for action in state.legal_actions
        if action.action_type == ActionType.DISCARD and action.card_indices
    ]
    if not discard_actions:
        return None

    keep_scores = _card_keep_scores(state.hand, preferred, state=state)
    protected = _preferred_hunt_protected_indices(state, preferred, keep_scores)
    detailed_actions = _prefilter_discard_actions(
        state,
        discard_actions,
        keep_scores,
        protected,
        preferred,
        limit=_preferred_hunt_discard_detail_limit(state, preferred, clear_line_only=clear_line_only),
    )

    current_needed = _estimated_hands_needed(remaining_score, score)
    ranked: list[tuple[tuple[float, float, int, float, int, int], Action, int, str]] = []
    for action in detailed_actions:
        kept_cards = tuple(card for index, card in enumerate(state.hand) if index not in action.card_indices)
        discarded_cards = tuple(state.hand[index] for index in action.card_indices)
        reason_detail = ""
        projected_score = _projected_score_after_discard(state, action, context)
        side_goal_bonus = _discard_action_playstyle_bonus(
            state,
            action,
            discarded_cards=discarded_cards,
            keep_scores=keep_scores,
            context=context,
        )
        if preferred in {HandType.STRAIGHT, HandType.STRAIGHT_FLUSH}:
            draw_count = _discard_draw_count(state, action, len(kept_cards))
            straight_eval = _straight_draw_evaluation(
                state,
                kept_cards,
                discarded_cards=discarded_cards,
                draw_count=draw_count,
                context=context,
            )
            if straight_eval is None or not _straight_hunt_projection_is_safe(
                state,
                evaluation=straight_eval,
                current_score=score,
                projected_score=projected_score,
                remaining_score=remaining_score,
                pace_score=pace_score,
            ):
                continue
            draw_score = straight_eval.quality
            clears_after_hit = straight_eval.completion_score >= remaining_score
            if clear_line_only and not clears_after_hit:
                continue
            if clears_after_hit:
                draw_score += 900.0 + straight_eval.completion_probability * 500.0
            reason_detail = _straight_draw_reason_detail(straight_eval)
        else:
            draw_count = _discard_draw_count(state, action, len(kept_cards))
            target_eval = _preferred_target_draw_evaluation(
                state,
                preferred,
                kept_cards,
                discarded_cards=discarded_cards,
                draw_count=draw_count,
                context=context,
            )
            if target_eval is None:
                continue
            if target_eval.present_count < _preferred_hunt_min_draw_strength(state, preferred):
                continue
            target_score = target_eval.completion_score
            clears_after_hit = target_eval.completion_score >= remaining_score
            if clear_line_only and not clears_after_hit:
                continue
            if not _preferred_hunt_projection_is_safe(
                state,
                kept_cards=kept_cards,
                current_score=score,
                projected_score=max(projected_score, target_score),
                remaining_score=remaining_score,
                pace_score=pace_score,
            ):
                continue
            draw_score = target_eval.quality
            if clears_after_hit:
                draw_score += 600.0
            projected_score = max(projected_score, target_score)
            reason_detail = _target_draw_reason_detail(target_eval)

        projected_needed = _estimated_hands_needed(remaining_score, projected_score)
        ranked.append(
            (
                (
                    draw_score + min(420.0, max(0.0, side_goal_bonus)) * 0.35,
                    side_goal_bonus,
                    current_needed - projected_needed,
                    min(projected_score, remaining_score),
                    len(action.card_indices),
                    -_action_index_sum(action),
                ),
                action,
                projected_score,
                reason_detail,
            )
        )

    if not ranked:
        return None

    _, action, projected_score, reason_detail = max(ranked, key=lambda item: item[0])
    return _annotated_action(
        action,
        reason=_preferred_hand_hunt_discard_reason(
            state,
            best_play,
            action,
            projected_score,
            preferred,
            current_hand_type,
            context,
            detail=reason_detail,
        ),
    )


def _preferred_hand_hunt_allowed(
    state: GameState,
    preferred: HandType,
    score: int,
    remaining_score: int,
    context: _BlindContext,
    *,
    clear_line_only: bool,
) -> bool:
    if _preferred_hunt_blind_blocks_hunt(state, preferred, context):
        return False
    if state.ante < 4:
        return False
    if state.blind != "Big Blind" and not clear_line_only:
        return False
    if clear_line_only:
        return True

    current_needed = _estimated_hands_needed(remaining_score, score)
    under_pressure = current_needed >= max(1, state.hands_remaining)
    below_pace = not _score_is_on_pace(state, score, remaining_score)

    if state.blind == "Big Blind":
        return state.ante >= 4 or under_pressure or below_pace
    if _is_boss_blind(state):
        return under_pressure or (state.ante >= 4 and below_pace and state.hands_remaining >= 2)
    if state.blind == "Small Blind":
        return under_pressure or (state.ante >= 4 and below_pace and state.hands_remaining >= 3)
    return under_pressure or below_pace


def _preferred_hunt_blind_blocks_hunt(
    state: GameState,
    preferred: HandType,
    context: _BlindContext,
) -> bool:
    if state.blind == "The Water":
        return True
    if state.blind == "The Mouth" and context.played_hand_types:
        return not any(_hand_matches_preferred_family(hand_type, preferred) for hand_type in context.played_hand_types)
    if state.blind == "The Eye":
        return any(_hand_matches_preferred_family(hand_type, preferred) for hand_type in context.played_hand_types)
    return False


def _preferred_hunt_discard_detail_limit(
    state: GameState,
    preferred: HandType,
    *,
    clear_line_only: bool,
) -> int:
    limit = _discard_detail_limit(state)
    if preferred in {HandType.STRAIGHT, HandType.STRAIGHT_FLUSH}:
        limit = max(limit, 48 if clear_line_only else 32)
    return limit


def _preferred_hand_hunt_redraw_play_action(
    state: GameState,
    best_play: Action,
    score: int,
    remaining_score: int,
    context: _BlindContext | None = None,
) -> Action | None:
    context = context or _BlindContext()
    preferred = _preferred_hand_type(state)
    if (
        preferred not in {HandType.STRAIGHT, HandType.STRAIGHT_FLUSH}
        or state.ante < 4
        or remaining_score <= 0
        or score >= remaining_score
        or state.hands_remaining <= 1
        or state.discards_remaining > 0
        or _preferred_hunt_blind_blocks_redraw_play(state, preferred, context)
    ):
        return None

    current_hand_type = _evaluate_play_action(state, best_play, context).hand_type
    if _hand_matches_preferred_family(current_hand_type, preferred):
        return None

    candidates = _play_candidates(state, context)
    if not candidates:
        return None

    ranked: list[tuple[tuple[float, float, int, float, int], _PlayCandidate, _StraightDrawEvaluation]] = []
    for candidate in candidates:
        if _hand_matches_preferred_family(candidate.hand_type, preferred):
            continue
        kept_cards = tuple(card for index, card in enumerate(state.hand) if index not in candidate.action.card_indices)
        if _straight_draw_potential(kept_cards) < 3:
            continue
        draw_count = _discard_draw_count(state, candidate.action, len(kept_cards))
        straight_eval = _straight_draw_evaluation(state, kept_cards, draw_count=draw_count, context=context)
        if straight_eval is None or straight_eval.present_count < 3:
            continue
        next_remaining = max(1, remaining_score - min(candidate.score, remaining_score - 1))
        if straight_eval.completion_score < next_remaining:
            continue
        if state.known_deck and not straight_eval.completes_from_known_draw:
            continue
        if straight_eval.missing_count >= 2 and straight_eval.completion_probability < 0.08:
            continue
        if straight_eval.missing_count == 1 and straight_eval.completion_probability < 0.12:
            continue
        ranked.append(
            (
                (
                    straight_eval.completion_probability,
                    straight_eval.quality,
                    draw_count,
                    min(candidate.score, remaining_score),
                    -_action_index_sum(candidate.action),
                ),
                candidate,
                straight_eval,
            )
        )

    if not ranked:
        return None

    _, candidate, straight_eval = max(ranked, key=lambda item: item[0])
    return _annotated_action(
        candidate.action,
        reason=_preferred_hand_hunt_redraw_play_reason(
            state,
            best_play,
            candidate,
            straight_eval,
            preferred,
            current_hand_type,
            context,
        ),
    )


def _preferred_hunt_blind_blocks_redraw_play(
    state: GameState,
    preferred: HandType,
    context: _BlindContext,
) -> bool:
    if state.blind in {"The Needle", "The Mouth"}:
        return True
    return _preferred_hunt_blind_blocks_hunt(state, preferred, context)


def _preferred_hunt_play_ceiling(state: GameState, preferred: HandType) -> float:
    ceiling = 1.45
    if preferred in {HandType.STRAIGHT, HandType.FLUSH, HandType.FULL_HOUSE}:
        ceiling += 0.10
    if state.ante >= 6:
        ceiling += 0.05
    if state.discards_remaining <= 1:
        ceiling -= 0.15
    return max(1.20, ceiling)


def _preferred_hunt_min_draw_strength(state: GameState, preferred: HandType) -> int:
    if preferred in {HandType.FOUR_OF_A_KIND, HandType.FIVE_OF_A_KIND, HandType.FLUSH_FIVE}:
        return 3
    if preferred in {HandType.STRAIGHT, HandType.FLUSH, HandType.FULL_HOUSE, HandType.STRAIGHT_FLUSH, HandType.FLUSH_HOUSE}:
        return 4 if state.discards_remaining <= 1 else 3
    return 2


def _preferred_hunt_protected_indices(
    state: GameState,
    preferred: HandType,
    keep_scores: tuple[float, ...],
) -> set[int]:
    if preferred in {HandType.STRAIGHT, HandType.STRAIGHT_FLUSH}:
        straight_core = _straight_hunt_core_indices(state)
        if straight_core:
            return straight_core

    keep_count = 4 if preferred in {HandType.STRAIGHT, HandType.FLUSH, HandType.STRAIGHT_FLUSH} else 3
    return {
        index
        for index, _ in sorted(
            enumerate(keep_scores),
            key=lambda item: (item[1], RANK_VALUES[state.hand[item[0]].rank]),
            reverse=True,
        )[: min(keep_count, len(state.hand))]
    }


def _straight_hunt_core_indices(state: GameState) -> set[int]:
    best: tuple[int, int, int, int, tuple[int, ...]] | None = None
    for start in range(1, 11):
        window_values = tuple(range(start, start + 5))
        selected: list[int] = []
        used: set[int] = set()
        for value in window_values:
            candidates = [
                index
                for index, card in enumerate(state.hand)
                if index not in used and _rank_matches_straight_value(card, value)
            ]
            if not candidates:
                continue
            index = max(candidates, key=lambda candidate: RANK_VALUES[state.hand[candidate].rank])
            selected.append(index)
            used.add(index)
        present_count = len(selected)
        if present_count < 3:
            continue
        missing_count = 5 - present_count
        open_ended = int(
            missing_count == 1
            and any(not any(_rank_matches_straight_value(card, edge) for card in state.hand) for edge in (window_values[0], window_values[-1]))
        )
        key = (present_count, -missing_count, open_ended, max(window_values), tuple(sorted(selected)))
        if best is None or key > best:
            best = key
    if best is None:
        return set()
    return set(best[-1])


def _preferred_hunt_projection_is_safe(
    state: GameState,
    *,
    kept_cards: tuple[Card, ...],
    current_score: int,
    projected_score: int,
    remaining_score: int,
    pace_score: float,
) -> bool:
    if projected_score >= remaining_score:
        return True
    if state.known_deck:
        return projected_score >= max(current_score * 0.95, pace_score * 0.75)
    if state.discards_remaining <= 1 and projected_score < max(current_score, pace_score):
        return False
    if projected_score >= max(current_score * 1.08, pace_score * 0.90):
        return True
    return _strong_draw_size(kept_cards) >= 4 and projected_score >= current_score * 0.85


def _straight_hunt_projection_is_safe(
    state: GameState,
    *,
    evaluation: _StraightDrawEvaluation,
    current_score: int,
    projected_score: int,
    remaining_score: int,
    pace_score: float,
) -> bool:
    if evaluation.present_count < 3:
        return False
    if evaluation.missing_count <= 0:
        return True
    if state.known_deck and not evaluation.completes_from_known_draw:
        return False
    if evaluation.missing_count >= 2 and evaluation.completion_probability < 0.08:
        return False
    if evaluation.completion_score >= remaining_score:
        if evaluation.present_count >= 4 and evaluation.completion_probability >= 0.10:
            return True
        if evaluation.present_count >= 3 and evaluation.completion_probability >= 0.08:
            return True
    if evaluation.present_count >= 4 and evaluation.completion_probability >= 0.20:
        return projected_score >= current_score * 0.80 or evaluation.completion_score >= pace_score
    if evaluation.completion_probability >= 0.14 and evaluation.completion_score >= max(pace_score, current_score * 1.10):
        return True
    return evaluation.completion_score >= remaining_score and projected_score >= current_score * 0.85


def _straight_draw_evaluation(
    state: GameState,
    kept_cards: tuple[Card, ...],
    *,
    discarded_cards: tuple[Card, ...] = (),
    draw_count: int,
    context: _BlindContext | None = None,
) -> _StraightDrawEvaluation | None:
    context = context or _BlindContext()
    if not kept_cards or draw_count <= 0:
        return None

    value_to_cards: dict[int, list[Card]] = {}
    for card in kept_cards:
        for value in _straight_values_for_card(card):
            value_to_cards.setdefault(value, []).append(card)

    projected_state = _discard_scoring_state(state, discarded_cards, context)
    known_completion_score = (
        _best_score_from_cards(projected_state, (*kept_cards, *state.known_deck[:draw_count]), context)
        if state.known_deck and draw_count > 0
        else None
    )
    best: _StraightDrawEvaluation | None = None
    for start in range(1, 11):
        window_values = tuple(range(start, start + 5))
        present_values = tuple(value for value in window_values if value in value_to_cards)
        present_count = len(present_values)
        if present_count < 3:
            continue
        missing_values = tuple(value for value in window_values if value not in value_to_cards)
        missing_count = len(missing_values)
        if missing_count > draw_count:
            continue

        out_values = _straight_out_values_for_window(kept_cards, window_values, missing_values)
        out_counts = tuple(_straight_missing_value_out_count(state, value) for value in out_values)
        out_count = sum(out_counts)
        if missing_count > 0 and out_count <= 0:
            continue

        top_draw_out_count = _straight_top_draw_out_count(state, out_values, draw_count)
        completes_from_known_draw = _straight_known_draw_completes(state, missing_values, draw_count)
        completion_probability = _straight_completion_probability(
            state,
            missing_values=missing_values,
            out_values=out_values,
            out_counts=out_counts,
            draw_count=draw_count,
            completes_from_known_draw=completes_from_known_draw,
        )
        completion_score = known_completion_score
        if completion_score is None:
            completion_score = _straight_completion_score(
                projected_state,
                kept_cards,
                missing_values=missing_values,
                draw_count=draw_count,
                context=context,
            )
        duplicate_penalty = _straight_window_duplicate_penalty(value_to_cards, present_values)
        open_ended = _straight_window_is_open_ended(missing_values, window_values)
        gutshot = missing_count == 1 and not open_ended
        window_high = max(window_values)
        quality = _straight_draw_quality(
            present_count=present_count,
            missing_count=missing_count,
            out_count=out_count,
            top_draw_out_count=top_draw_out_count,
            completion_probability=completion_probability,
            completion_score=completion_score,
            window_high=window_high,
            open_ended=open_ended,
            gutshot=gutshot,
            duplicate_penalty=duplicate_penalty,
        )
        evaluation = _StraightDrawEvaluation(
            present_count=present_count,
            missing_count=missing_count,
            missing_values=missing_values,
            out_count=out_count,
            top_draw_out_count=top_draw_out_count,
            completion_probability=completion_probability,
            completion_score=completion_score,
            quality=quality,
            window_high=window_high,
            open_ended=open_ended,
            gutshot=gutshot,
            completes_from_known_draw=completes_from_known_draw,
        )
        if best is None or evaluation.quality > best.quality:
            best = evaluation
    return best


def _straight_draw_quality(
    *,
    present_count: int,
    missing_count: int,
    out_count: int,
    top_draw_out_count: int,
    completion_probability: float,
    completion_score: int,
    window_high: int,
    open_ended: bool,
    gutshot: bool,
    duplicate_penalty: int,
) -> float:
    quality = present_count * 110.0
    quality += out_count * 9.0
    quality += top_draw_out_count * 35.0
    quality += completion_probability * 260.0
    quality += min(120.0, completion_score * 0.012)
    quality += window_high * 2.5
    quality -= missing_count * 95.0
    quality -= duplicate_penalty * 22.0
    if open_ended:
        quality += 55.0
    if gutshot:
        quality -= 35.0
    if missing_count >= 2:
        quality -= 80.0
    return quality


def _straight_out_values_for_window(
    kept_cards: tuple[Card, ...],
    window_values: tuple[int, ...],
    missing_values: tuple[int, ...],
) -> tuple[int, ...]:
    if len(missing_values) != 1:
        return missing_values
    values = _straight_values_present(kept_cards)
    out_values: set[int] = set(missing_values)
    for start in range(1, 11):
        window = set(range(start, start + 5))
        missing = window - values
        if len(missing) == 1 and len(window & values) >= 4:
            out_values.update(missing)
    return tuple(sorted(out_values, key=lambda value: (value not in window_values, value)))


def _straight_values_present(cards: tuple[Card, ...]) -> set[int]:
    values: set[int] = set()
    for card in cards:
        values.update(_straight_values_for_card(card))
    return values


def _straight_values_for_card(card: Card) -> tuple[int, ...]:
    value = STRAIGHT_VALUES[card.rank]
    if card.rank == "A":
        return (1, 14)
    return (value,)


def _straight_missing_value_out_count(state: GameState, value: int) -> int:
    rank = _straight_rank_for_value(value)
    if state.known_deck:
        return sum(1 for card in state.known_deck if _rank_matches_straight_value(card, value))
    seen = sum(1 for card in state.hand if _rank_matches(card.rank, rank))
    return max(0, 4 - seen)


def _straight_top_draw_out_count(state: GameState, out_values: tuple[int, ...], draw_count: int) -> int:
    if not state.known_deck or draw_count <= 0:
        return 0
    top_draw = state.known_deck[:draw_count]
    return sum(1 for card in top_draw if any(_rank_matches_straight_value(card, value) for value in out_values))


def _straight_known_draw_completes(
    state: GameState,
    missing_values: tuple[int, ...],
    draw_count: int,
) -> bool:
    if not state.known_deck:
        return False
    top_values = _straight_values_present(tuple(state.known_deck[:draw_count]))
    return all(value in top_values for value in missing_values)


def _straight_completion_probability(
    state: GameState,
    *,
    missing_values: tuple[int, ...],
    out_values: tuple[int, ...],
    out_counts: tuple[int, ...],
    draw_count: int,
    completes_from_known_draw: bool,
) -> float:
    if not missing_values:
        return 1.0
    if state.known_deck:
        return 1.0 if completes_from_known_draw else 0.0
    if draw_count <= 0 or not out_counts:
        return 0.0
    deck_size = max(1, state.deck_size)
    draw_count = min(draw_count, deck_size)
    if len(missing_values) == 1:
        out_count = min(deck_size, sum(out_counts))
        return _draw_at_least_one_probability(deck_size, draw_count, out_count)
    per_missing_counts = tuple(
        max(0, _straight_missing_value_out_count(state, value))
        for value in missing_values
        if value in out_values
    )
    if len(per_missing_counts) != len(missing_values) or any(count <= 0 for count in per_missing_counts):
        return 0.0
    return _draw_all_groups_probability(deck_size, draw_count, per_missing_counts)


def _draw_at_least_one_probability(deck_size: int, draw_count: int, out_count: int) -> float:
    if out_count <= 0 or draw_count <= 0:
        return 0.0
    if out_count >= deck_size:
        return 1.0
    draw_count = min(draw_count, deck_size)
    total = comb(deck_size, draw_count)
    if total <= 0:
        return 0.0
    misses = comb(max(0, deck_size - out_count), draw_count) if deck_size - out_count >= draw_count else 0
    return max(0.0, min(1.0, 1.0 - (misses / total)))


def _draw_all_groups_probability(deck_size: int, draw_count: int, group_counts: tuple[int, ...]) -> float:
    total = comb(deck_size, draw_count)
    if total <= 0:
        return 0.0
    probability = 1.0
    group_indexes = range(len(group_counts))
    for size in range(1, len(group_counts) + 1):
        sign = -1.0 if size % 2 else 1.0
        for indexes in combinations(group_indexes, size):
            blocked = sum(group_counts[index] for index in indexes)
            remaining = deck_size - blocked
            ways = comb(remaining, draw_count) if remaining >= draw_count else 0
            probability += sign * (ways / total)
    return max(0.0, min(1.0, probability))


def _straight_completion_score(
    state: GameState,
    kept_cards: tuple[Card, ...],
    *,
    missing_values: tuple[int, ...],
    draw_count: int,
    context: _BlindContext,
) -> int:
    if state.known_deck:
        return _best_score_from_cards(state, (*kept_cards, *state.known_deck[:draw_count]), context)
    completion_cards = _straight_completion_cards_for_values(kept_cards, missing_values, draw_count)
    if not completion_cards:
        return 0
    return _best_score_from_cards(state, completion_cards, context)


def _straight_completion_cards_for_values(
    kept_cards: tuple[Card, ...],
    missing_values: tuple[int, ...],
    draw_count: int,
) -> tuple[Card, ...]:
    if len(missing_values) > draw_count:
        return ()
    suit = _dominant_suit_from_cards(kept_cards) or "S"
    fill = tuple(Card(_straight_rank_for_value(value), suit) for value in missing_values)
    remaining_draw = max(0, draw_count - len(fill))
    return (*kept_cards, *fill, *_fill_with_high_cards((), remaining_draw))[: len(kept_cards) + draw_count]


def _straight_window_duplicate_penalty(
    value_to_cards: dict[int, list[Card]],
    present_values: tuple[int, ...],
) -> int:
    return sum(max(0, len(value_to_cards.get(value, ())) - 1) for value in present_values)


def _straight_window_is_open_ended(missing_values: tuple[int, ...], window_values: tuple[int, ...]) -> bool:
    return len(missing_values) == 1 and missing_values[0] in {window_values[0], window_values[-1]}


def _rank_matches_straight_value(card: Card, value: int) -> bool:
    return value in _straight_values_for_card(card)


def _straight_draw_reason_detail(evaluation: _StraightDrawEvaluation) -> str:
    shape = "open" if evaluation.open_ended else "gutshot" if evaluation.gutshot else "made" if evaluation.missing_count == 0 else "two_gap"
    return (
        f"straight_draw={evaluation.present_count}/5 missing={evaluation.missing_count} "
        f"outs={evaluation.out_count} p={evaluation.completion_probability:.2f} "
        f"shape={shape} completion={evaluation.completion_score}"
    )


def _preferred_target_draw_evaluation(
    state: GameState,
    preferred: HandType,
    kept_cards: tuple[Card, ...],
    *,
    discarded_cards: tuple[Card, ...],
    draw_count: int,
    context: _BlindContext,
) -> _TargetDrawEvaluation | None:
    if not kept_cards or draw_count < 0:
        return None

    projected_state = _discard_scoring_state(state, discarded_cards, context)
    candidates: list[_TargetDrawEvaluation] = []
    if preferred == HandType.FLUSH:
        candidates.extend(_flush_target_draw_evaluations(projected_state, kept_cards, draw_count, context))
    elif preferred in {HandType.THREE_OF_A_KIND, HandType.FOUR_OF_A_KIND, HandType.FIVE_OF_A_KIND}:
        target_count = {
            HandType.THREE_OF_A_KIND: 3,
            HandType.FOUR_OF_A_KIND: 4,
            HandType.FIVE_OF_A_KIND: 5,
        }[preferred]
        candidates.extend(
            _rank_target_draw_evaluations(
                projected_state,
                kept_cards,
                draw_count,
                context,
                hand_type=preferred,
                target_count=target_count,
            )
        )
    elif preferred == HandType.FULL_HOUSE:
        candidates.extend(_full_house_target_draw_evaluations(projected_state, kept_cards, draw_count, context))

    if not candidates:
        return None
    return max(candidates, key=lambda item: (item.completion_score, item.completion_probability, item.quality))


def _flush_target_draw_evaluations(
    state: GameState,
    kept_cards: tuple[Card, ...],
    draw_count: int,
    context: _BlindContext,
) -> list[_TargetDrawEvaluation]:
    evaluations: list[_TargetDrawEvaluation] = []
    for suit in ("S", "H", "D", "C"):
        present_count = sum(1 for card in kept_cards if _normalize_suit(card.suit) == suit)
        if present_count < 3:
            continue
        missing_count = max(0, 5 - present_count)
        if missing_count > draw_count:
            continue
        out_count = _flush_suit_out_count(state, suit)
        completion_probability = _flush_completion_probability(
            state,
            suit=suit,
            draw_count=draw_count,
            missing_count=missing_count,
            out_count=out_count,
        )
        if missing_count > 0 and completion_probability <= 0.0:
            continue
        completion_cards = _flush_completion_cards_for_suit(kept_cards, suit, draw_count)
        if not completion_cards:
            continue
        completion_score = _best_score_from_cards(state, completion_cards, context)
        quality = _target_draw_quality(
            present_count=present_count,
            missing_count=missing_count,
            out_count=out_count,
            completion_probability=completion_probability,
            completion_score=completion_score,
            draw_count=draw_count,
        )
        evaluations.append(
            _TargetDrawEvaluation(
                hand_type=HandType.FLUSH,
                label=f"Flush:{suit}",
                present_count=present_count,
                missing_count=missing_count,
                out_count=out_count,
                completion_probability=completion_probability,
                completion_score=completion_score,
                quality=quality,
            )
        )
    return evaluations


def _rank_target_draw_evaluations(
    state: GameState,
    kept_cards: tuple[Card, ...],
    draw_count: int,
    context: _BlindContext,
    *,
    hand_type: HandType,
    target_count: int,
) -> list[_TargetDrawEvaluation]:
    rank_counts = Counter(card.rank for card in kept_cards)
    evaluations: list[_TargetDrawEvaluation] = []
    for rank in _strategy_rank_order():
        present_count = rank_counts.get(rank, 0)
        if present_count <= 0:
            continue
        missing_count = max(0, target_count - present_count)
        if missing_count > draw_count:
            continue
        out_count = _rank_out_count(state, rank)
        completion_probability = _rank_completion_probability(
            state,
            rank=rank,
            draw_count=draw_count,
            missing_count=missing_count,
            out_count=out_count,
        )
        if missing_count > 0 and completion_probability <= 0.0:
            continue
        completion_cards = _rank_completion_cards_for_rank(kept_cards, rank, target_count, draw_count)
        if not completion_cards:
            continue
        completion_score = _best_score_from_cards(state, completion_cards, context)
        quality = _target_draw_quality(
            present_count=present_count,
            missing_count=missing_count,
            out_count=out_count,
            completion_probability=completion_probability,
            completion_score=completion_score,
            draw_count=draw_count,
        )
        evaluations.append(
            _TargetDrawEvaluation(
                hand_type=hand_type,
                label=f"{hand_type.value}:{rank}",
                present_count=present_count,
                missing_count=missing_count,
                out_count=out_count,
                completion_probability=completion_probability,
                completion_score=completion_score,
                quality=quality,
            )
        )
    return evaluations


def _full_house_target_draw_evaluations(
    state: GameState,
    kept_cards: tuple[Card, ...],
    draw_count: int,
    context: _BlindContext,
) -> list[_TargetDrawEvaluation]:
    rank_counts = Counter(card.rank for card in kept_cards)
    evaluations: list[_TargetDrawEvaluation] = []
    ranks = _strategy_rank_order()
    for trip_rank in ranks:
        trip_present = rank_counts.get(trip_rank, 0)
        if trip_present <= 0:
            continue
        for pair_rank in ranks:
            if pair_rank == trip_rank:
                continue
            pair_present = rank_counts.get(pair_rank, 0)
            if pair_present <= 0:
                continue
            trip_missing = max(0, 3 - trip_present)
            pair_missing = max(0, 2 - pair_present)
            missing_count = trip_missing + pair_missing
            if missing_count > draw_count:
                continue
            out_counts = (
                _rank_out_count(state, trip_rank),
                _rank_out_count(state, pair_rank),
            )
            completion_probability = _rank_group_completion_probability(
                state,
                draw_count=draw_count,
                ranks=(trip_rank, pair_rank),
                requirements=(trip_missing, pair_missing),
                out_counts=out_counts,
            )
            if missing_count > 0 and completion_probability <= 0.0:
                continue
            completion_cards = _full_house_completion_cards_for_ranks(
                kept_cards,
                trip_rank=trip_rank,
                pair_rank=pair_rank,
                draw_count=draw_count,
            )
            if not completion_cards:
                continue
            completion_score = _best_score_from_cards(state, completion_cards, context)
            present_count = min(3, trip_present) + min(2, pair_present)
            quality = _target_draw_quality(
                present_count=present_count,
                missing_count=missing_count,
                out_count=sum(out_counts),
                completion_probability=completion_probability,
                completion_score=completion_score,
                draw_count=draw_count,
            )
            evaluations.append(
                _TargetDrawEvaluation(
                    hand_type=HandType.FULL_HOUSE,
                    label=f"Full House:{trip_rank}/{pair_rank}",
                    present_count=present_count,
                    missing_count=missing_count,
                    out_count=sum(out_counts),
                    completion_probability=completion_probability,
                    completion_score=completion_score,
                    quality=quality,
                )
            )
    return evaluations


def _target_draw_quality(
    *,
    present_count: int,
    missing_count: int,
    out_count: int,
    completion_probability: float,
    completion_score: int,
    draw_count: int,
) -> float:
    quality = present_count * 115.0
    quality += completion_probability * 520.0
    quality += min(180.0, completion_score * 0.012)
    quality += out_count * 6.0
    quality += draw_count * 8.0
    quality -= missing_count * 120.0
    return quality


def _target_draw_reason_detail(evaluation: _TargetDrawEvaluation) -> str:
    return (
        f"target={evaluation.label} present={evaluation.present_count} "
        f"missing={evaluation.missing_count} outs={evaluation.out_count} "
        f"p={evaluation.completion_probability:.2f} completion={evaluation.completion_score}"
    )


def _flush_suit_out_count(state: GameState, suit: str) -> int:
    if state.known_deck:
        return sum(1 for card in state.known_deck if _normalize_suit(card.suit) == suit)
    seen = sum(1 for card in state.hand if _normalize_suit(card.suit) == suit)
    return max(0, 13 - seen)


def _flush_completion_probability(
    state: GameState,
    *,
    suit: str,
    draw_count: int,
    missing_count: int,
    out_count: int,
) -> float:
    if missing_count <= 0:
        return 1.0
    if draw_count <= 0:
        return 0.0
    if state.known_deck:
        top_draw = state.known_deck[:draw_count]
        hits = sum(1 for card in top_draw if _normalize_suit(card.suit) == suit)
        return 1.0 if hits >= missing_count else 0.0
    return _draw_at_least_k_probability(max(1, state.deck_size), draw_count, out_count, missing_count)


def _rank_out_count(state: GameState, rank: str) -> int:
    if state.known_deck:
        return sum(1 for card in state.known_deck if _rank_matches(card.rank, rank))
    seen = sum(1 for card in state.hand if _rank_matches(card.rank, rank))
    return max(0, 4 - seen)


def _rank_completion_probability(
    state: GameState,
    *,
    rank: str,
    draw_count: int,
    missing_count: int,
    out_count: int,
) -> float:
    if missing_count <= 0:
        return 1.0
    if draw_count <= 0:
        return 0.0
    if state.known_deck:
        top_draw = state.known_deck[:draw_count]
        hits = sum(1 for card in top_draw if _rank_matches(card.rank, rank))
        return 1.0 if hits >= missing_count else 0.0
    return _draw_at_least_k_probability(max(1, state.deck_size), draw_count, out_count, missing_count)


def _rank_group_completion_probability(
    state: GameState,
    *,
    draw_count: int,
    ranks: tuple[str, ...],
    requirements: tuple[int, ...],
    out_counts: tuple[int, ...],
) -> float:
    needed = tuple(max(0, requirement) for requirement in requirements)
    if all(requirement <= 0 for requirement in needed):
        return 1.0
    if draw_count <= 0:
        return 0.0
    if state.known_deck:
        top_draw = state.known_deck[:draw_count]
        hits = []
        for index, requirement in enumerate(needed):
            if requirement <= 0:
                hits.append(requirement)
                continue
            rank = ranks[index] if index < len(ranks) else ""
            hits.append(sum(1 for card in top_draw if _rank_matches(card.rank, rank)))
        return 1.0 if all(hit >= requirement for hit, requirement in zip(hits, needed, strict=False)) else 0.0
    return _draw_group_min_counts_probability(max(1, state.deck_size), draw_count, needed, out_counts)


def _draw_at_least_k_probability(deck_size: int, draw_count: int, out_count: int, needed: int) -> float:
    if needed <= 0:
        return 1.0
    if out_count < needed or draw_count < needed:
        return 0.0
    draw_count = min(draw_count, deck_size)
    total = comb(deck_size, draw_count)
    if total <= 0:
        return 0.0
    max_hits = min(out_count, draw_count)
    ways = 0
    rest = max(0, deck_size - out_count)
    for hits in range(needed, max_hits + 1):
        misses = draw_count - hits
        if misses > rest:
            continue
        ways += comb(out_count, hits) * comb(rest, misses)
    return max(0.0, min(1.0, ways / total))


def _draw_group_min_counts_probability(
    deck_size: int,
    draw_count: int,
    requirements: tuple[int, ...],
    out_counts: tuple[int, ...],
) -> float:
    draw_count = min(draw_count, deck_size)
    total = comb(deck_size, draw_count)
    if total <= 0:
        return 0.0
    if len(requirements) != 2 or len(out_counts) != 2:
        return 0.0
    first_needed, second_needed = requirements
    first_out, second_out = out_counts
    if first_out < first_needed or second_out < second_needed:
        return 0.0
    rest = max(0, deck_size - first_out - second_out)
    ways = 0
    for first_hits in range(first_needed, min(first_out, draw_count) + 1):
        for second_hits in range(second_needed, min(second_out, draw_count - first_hits) + 1):
            misses = draw_count - first_hits - second_hits
            if misses > rest:
                continue
            ways += comb(first_out, first_hits) * comb(second_out, second_hits) * comb(rest, misses)
    return max(0.0, min(1.0, ways / total))


def _flush_completion_cards_for_suit(
    kept_cards: tuple[Card, ...],
    suit: str,
    draw_count: int,
) -> tuple[Card, ...]:
    present_count = sum(1 for card in kept_cards if _normalize_suit(card.suit) == suit)
    missing = max(0, 5 - present_count)
    if missing > draw_count:
        return ()
    existing_ranks = {card.rank for card in kept_cards if _normalize_suit(card.suit) == suit}
    fill = tuple(Card(rank, suit) for rank in _strategy_rank_order() if rank not in existing_ranks)
    return (*kept_cards, *fill[:missing], *_fill_with_high_cards((), max(0, draw_count - missing)))[: len(kept_cards) + draw_count]


def _rank_completion_cards_for_rank(
    kept_cards: tuple[Card, ...],
    rank: str,
    target_count: int,
    draw_count: int,
) -> tuple[Card, ...]:
    present_count = sum(1 for card in kept_cards if _rank_matches(card.rank, rank))
    missing = max(0, target_count - present_count)
    if missing > draw_count:
        return ()
    used_suits = {_normalize_suit(card.suit) for card in kept_cards if _rank_matches(card.rank, rank)}
    fill_suits = tuple(suit for suit in ("S", "H", "D", "C") if suit not in used_suits)
    fill = tuple(Card(rank, suit) for suit in fill_suits[:missing])
    return (*kept_cards, *fill, *_fill_with_high_cards((), max(0, draw_count - len(fill))))[: len(kept_cards) + draw_count]


def _full_house_completion_cards_for_ranks(
    kept_cards: tuple[Card, ...],
    *,
    trip_rank: str,
    pair_rank: str,
    draw_count: int,
) -> tuple[Card, ...]:
    trip_missing = max(0, 3 - sum(1 for card in kept_cards if _rank_matches(card.rank, trip_rank)))
    pair_missing = max(0, 2 - sum(1 for card in kept_cards if _rank_matches(card.rank, pair_rank)))
    if trip_missing + pair_missing > draw_count:
        return ()
    trip_fill = _rank_fill_cards(kept_cards, trip_rank, trip_missing)
    pair_fill = _rank_fill_cards((*kept_cards, *trip_fill), pair_rank, pair_missing)
    fill = (*trip_fill, *pair_fill)
    return (*kept_cards, *fill, *_fill_with_high_cards((), max(0, draw_count - len(fill))))[: len(kept_cards) + draw_count]


def _rank_fill_cards(existing_cards: tuple[Card, ...], rank: str, count: int) -> tuple[Card, ...]:
    if count <= 0:
        return ()
    used_suits = {_normalize_suit(card.suit) for card in existing_cards if _rank_matches(card.rank, rank)}
    fill_suits = tuple(suit for suit in ("S", "H", "D", "C") if suit not in used_suits)
    return tuple(Card(rank, suit) for suit in fill_suits[:count])


def _strategy_rank_order() -> tuple[str, ...]:
    return ("A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2")


def _preferred_draw_strength(kept_cards: tuple[Card, ...], preferred: HandType) -> int:
    if not kept_cards:
        return 0
    rank_counts = Counter(card.rank for card in kept_cards)
    suit_counts = Counter(card.suit for card in kept_cards)
    max_rank = max(rank_counts.values(), default=0)
    max_suit = max(suit_counts.values(), default=0)

    if preferred in {HandType.STRAIGHT, HandType.STRAIGHT_FLUSH}:
        return _straight_draw_potential(kept_cards)
    if preferred == HandType.FLUSH_HOUSE:
        return min(max_suit, max_rank + 1)
    if preferred in FLUSH_ARCHETYPE_HANDS:
        return max_suit
    if preferred == HandType.FULL_HOUSE:
        pair_count = sum(1 for count in rank_counts.values() if count >= 2)
        return max(max_rank, 2 + pair_count)
    if preferred in {HandType.FOUR_OF_A_KIND, HandType.FIVE_OF_A_KIND, HandType.FLUSH_FIVE}:
        return max_rank
    return max_rank


def _hand_matches_preferred_family(hand_type: HandType, preferred: HandType) -> bool:
    return hand_type in _preferred_hand_family(preferred)


def _preferred_hand_family(preferred: HandType) -> set[HandType]:
    if preferred == HandType.PAIR:
        return set(PAIR_CONTAINS_HANDS)
    if preferred == HandType.TWO_PAIR:
        return set(TWO_PAIR_CONTAINS_HANDS)
    if preferred == HandType.THREE_OF_A_KIND:
        return set(THREE_KIND_CONTAINS_HANDS)
    if preferred == HandType.FULL_HOUSE:
        return {HandType.FULL_HOUSE, HandType.FLUSH_HOUSE}
    if preferred in {HandType.FLUSH, HandType.STRAIGHT_FLUSH, HandType.FLUSH_HOUSE, HandType.FLUSH_FIVE}:
        return set(FLUSH_ARCHETYPE_HANDS)
    if preferred == HandType.STRAIGHT:
        return {HandType.STRAIGHT, HandType.STRAIGHT_FLUSH}
    if preferred == HandType.FOUR_OF_A_KIND:
        return set(FOUR_KIND_CONTAINS_HANDS)
    if preferred == HandType.FIVE_OF_A_KIND:
        return {HandType.FIVE_OF_A_KIND, HandType.FLUSH_FIVE}
    return {preferred}


def _should_chase_discard(
    state: GameState,
    action: Action,
    current_score: int,
    remaining_score: int,
    context: _BlindContext | None = None,
) -> bool:
    kept_cards = tuple(card for index, card in enumerate(state.hand) if index not in action.card_indices)
    projected_score = _projected_score_after_discard(state, action, context)
    if state.known_deck:
        current_hands_needed = _estimated_hands_needed(remaining_score, current_score)
        projected_hands_needed = _estimated_hands_needed(remaining_score, projected_score)
        return projected_score > current_score and projected_hands_needed <= current_hands_needed

    return _unknown_discard_projection_is_trustworthy(
        state,
        kept_cards=kept_cards,
        current_score=current_score,
        projected_score=projected_score,
        remaining_score=remaining_score,
        weak_chase=True,
    )


def _discard_penalty_jokers_active(state: GameState) -> bool:
    return bool(_active_joker_names(state) & DISCARD_PENALTY_JOKERS)


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
    if state.discards_remaining <= 1 and projected_score < current_score:
        return False
    discard_penalty = _discard_penalty_jokers_active(state)
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
        if discard_penalty and projected_score <= current_score:
            return False
        return True

    kept_cards = tuple(card for index, card in enumerate(state.hand) if index not in action.card_indices)
    if state.discards_remaining <= 1 and not state.known_deck:
        return projected_score >= max(current_score * 1.35, pace_score)
    if discard_penalty:
        return projected_score >= max(current_score * 1.2, pace_score * 0.9)

    return projected_score >= max(current_score * 1.2, pace_score * 0.9) or _strong_draw_size(kept_cards) >= 4


def _should_last_hand_hunt_discard(
    state: GameState,
    action: Action,
    current_score: int,
    remaining_score: int,
    context: _BlindContext | None = None,
) -> bool:
    if remaining_score <= 0 or state.hands_remaining > 1 or state.discards_remaining <= 0:
        return False

    projected_score = _projected_score_after_discard(state, action, context)
    if projected_score >= remaining_score and projected_score >= current_score:
        return True
    if current_score < remaining_score and state.discards_remaining > 1:
        return True
    if state.known_deck:
        return projected_score > current_score
    if projected_score > current_score:
        return True
    if _discard_penalty_jokers_active(state):
        return False

    kept_cards = tuple(card for index, card in enumerate(state.hand) if index not in action.card_indices)
    return _strong_draw_size(kept_cards) >= 4


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
    detailed_actions = _prefilter_discard_actions(
        state,
        discard_actions,
        keep_scores,
        protected,
        preferred,
        limit=_discard_detail_limit(state),
    )

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
    names = _active_joker_names(state)
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
        bonus += 520.0 * sum(1 for card in discarded_cards if not card.debuffed and card.rank == "J")
    castle_suit = _castle_target_suit(state)
    if castle_suit is not None:
        bonus += 240.0 * sum(
            1 for card in discarded_cards if not card.debuffed and _normalize_suit(card.suit) == castle_suit
        )
    mail_rank = _mail_in_rebate_rank(state)
    if mail_rank is not None:
        bonus += 180.0 * sum(
            1 for card in discarded_cards if not card.debuffed and _rank_matches(card.rank, mail_rank)
        )
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
    *,
    limit: int = DISCARD_DETAIL_LIMIT,
) -> list[Action]:
    if len(discard_actions) <= limit:
        return discard_actions

    def cheap_score(action: Action) -> tuple[float, int]:
        protected_penalty = sum(1 for index in action.card_indices if index in protected) * 1000
        desirability = sum(100 - keep_scores[index] for index in action.card_indices)
        kept_cards = tuple(card for index, card in enumerate(state.hand) if index not in action.card_indices)
        return (
            desirability + _cheap_kept_hand_potential(kept_cards, preferred) - protected_penalty,
            len(action.card_indices),
        )

    return sorted(discard_actions, key=cheap_score, reverse=True)[:limit]


def _discard_detail_limit(state: GameState) -> int:
    if state.ante >= 3:
        return LATE_DISCARD_DETAIL_LIMIT
    return DISCARD_DETAIL_LIMIT


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
    cache = _state_scoped_cache("projected_score_after_discard", state)
    cache_key = (
        tuple(action.card_indices),
        context.played_hand_types,
        context.discards_taken,
    )
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    kept_cards = tuple(card for index, card in enumerate(state.hand) if index not in action.card_indices)
    discarded_cards = tuple(state.hand[index] for index in action.card_indices)
    draw_count = _discard_draw_count(state, action, len(kept_cards))
    known_draw = tuple(state.known_deck[: min(draw_count, len(state.known_deck))]) if state.known_deck else ()
    projected_state = _state_after_discard_for_projection(
        state,
        action,
        drawn_cards=known_draw,
        context=context,
        decrement_discard=False,
    )
    content_cache = _decision_scoped_cache("projected_score_after_discard_content")
    content_key = _projected_discard_cache_key(
        projected_state,
        kept_cards=kept_cards,
        discarded_cards=discarded_cards,
        draw_count=draw_count,
        drawn_cards=known_draw,
        context=context,
    )
    if content_cache is not None and content_key in content_cache:
        score = content_cache[content_key]
        if cache is not None:
            cache[cache_key] = score
        return int(score)

    if known_draw:
        score = _best_score_from_cards(projected_state, (*kept_cards, *known_draw), context)
        if cache is not None:
            cache[cache_key] = score
        if content_cache is not None:
            content_cache[content_key] = score
        return score

    kept_score = _best_score_from_cards(projected_state, kept_cards, context)
    optimistic_score = _optimistic_completion_score(projected_state, kept_cards, draw_count, context)
    realism = _discard_realism_factor(projected_state, kept_cards, draw_count)
    score = int(max(kept_score, (optimistic_score * realism) + (kept_score * (1.0 - realism))))
    if cache is not None:
        cache[cache_key] = score
    if content_cache is not None:
        content_cache[content_key] = score
    return score


def _projected_discard_cache_key(
    state: GameState,
    *,
    kept_cards: tuple[Card, ...],
    discarded_cards: tuple[Card, ...],
    draw_count: int,
    drawn_cards: tuple[Card, ...],
    context: _BlindContext,
) -> tuple[object, ...]:
    return (
        _hand_multiset_cache_key(kept_cards),
        _hand_multiset_cache_key(discarded_cards),
        draw_count,
        _jokers_cache_key(state.jokers),
        _scoring_state_cache_key(state, context),
        _freeze_for_cache(drawn_cards),
    )


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


def _best_score_from_cards(
    state: GameState,
    cards: tuple[Card, ...],
    context: _BlindContext | None = None,
) -> int:
    context = context or _BlindContext()
    if not cards:
        return 0
    content_cache = _decision_scoped_cache("best_score_from_cards_content")
    content_key = (
        _hand_multiset_cache_key(cards),
        _jokers_cache_key(state.jokers),
        _freeze_for_cache(state.hand_levels),
        _scoring_state_cache_key(state, context),
    )
    if content_cache is not None and content_key in content_cache:
        return int(content_cache[content_key])

    cache = _state_scoped_cache("best_score_from_cards", state)
    cache_key = (
        _freeze_for_cache(cards),
        context.played_hand_types,
        context.discards_taken,
    )
    if cache is not None and cache_key in cache:
        return cache[cache_key]

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
    score = _boss_adjusted_score(state, evaluation.hand_type, evaluation.score, context)
    if cache is not None:
        cache[cache_key] = score
    if content_cache is not None:
        content_cache[content_key] = score
    return score


def _optimistic_completion_score(
    state: GameState,
    kept_cards: tuple[Card, ...],
    draw_count: int,
    context: _BlindContext | None = None,
) -> int:
    context = context or _BlindContext()
    if draw_count <= 0:
        return _best_score_from_cards(state, kept_cards, context)
    cache = _decision_scoped_cache("optimistic_completion_score_content")
    cache_key = (
        _hand_multiset_cache_key(kept_cards),
        draw_count,
        _jokers_cache_key(state.jokers),
        _freeze_for_cache(state.hand_levels),
        _scoring_state_cache_key(state, context),
    )
    if cache is not None and cache_key in cache:
        return int(cache[cache_key])

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

    score = max(_best_score_from_cards(state, candidate, context) for candidate in candidates)
    if cache is not None:
        cache[cache_key] = score
    return score


def _hand_multiset_cache_key(cards: tuple[Card, ...]) -> tuple[object, ...]:
    return tuple(sorted((_card_cache_key(card) for card in cards), key=repr))


def _jokers_cache_key(jokers: tuple[Joker, ...]) -> tuple[object, ...]:
    return tuple(_joker_cache_key(joker) for joker in jokers)


def _scoring_state_cache_key(state: GameState, context: _BlindContext) -> tuple[object, ...]:
    return (
        state.blind,
        state.discards_remaining,
        state.hands_remaining,
        state.deck_size,
        state.money,
        context.played_hand_types,
        context.discards_taken,
        _freeze_for_cache(state.modifiers.get("hands", state.modifiers.get("hand_stats", {}))),
    )


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
    return (*kept_cards, *fill[:draw_count])[: len(kept_cards) + draw_count]


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
    return (*kept_cards, *fill[:draw_count])[: len(kept_cards) + draw_count]


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
    return (*kept_cards, *fill, *_fill_with_high_cards((), max(0, draw_count - len(fill))))[: len(kept_cards) + draw_count]


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
    return (*kept_cards, *fill, *_fill_with_high_cards((), remaining_draw))[: len(kept_cards) + draw_count]


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
    evaluation = _evaluate_play_action(state, action, context)
    context = context or _BlindContext()
    score = _boss_adjusted_score(state, evaluation.hand_type, evaluation.score, context)
    remaining_score = max(0, state.required_score - state.current_score)
    hands_needed = _estimated_hands_needed(remaining_score, score)
    preferred = _preferred_hand_type(state)
    preferred_text = preferred.value if preferred is not None else "-"
    return (
        f"tactical_play hand={evaluation.hand_type.value} preferred={preferred_text} score={score} remaining={remaining_score} "
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


def _preferred_hand_hunt_discard_reason(
    state: GameState,
    best_play: Action,
    discard: Action,
    projected_score: int,
    preferred: HandType,
    current_hand_type: HandType,
    context: _BlindContext | None = None,
    detail: str = "",
) -> str:
    best_score = _score_play_action(state, best_play, context)
    remaining_score = max(0, state.required_score - state.current_score)
    suffix = f" {detail}" if detail else ""
    return (
        f"preferred_hand_hunt preferred={preferred.value} current_hand={current_hand_type.value} "
        f"current_score={best_score} projected_score={projected_score} "
        f"remaining={remaining_score} discarding={len(discard.card_indices)}{suffix}"
    )


def _preferred_hand_hunt_redraw_play_reason(
    state: GameState,
    best_play: Action,
    candidate: _PlayCandidate,
    evaluation: _StraightDrawEvaluation,
    preferred: HandType,
    current_hand_type: HandType,
    context: _BlindContext | None = None,
) -> str:
    best_score = _score_play_action(state, best_play, context)
    remaining_score = max(0, state.required_score - state.current_score)
    return (
        f"preferred_hand_hunt_redraw preferred={preferred.value} current_hand={current_hand_type.value} "
        f"burn_hand={candidate.hand_type.value} burn_score={candidate.score} current_score={best_score} "
        f"remaining={remaining_score} playing={len(candidate.action.card_indices)} "
        f"{_straight_draw_reason_detail(evaluation)}"
    )


def _last_hand_hunt_discard_reason(
    state: GameState,
    best_play: Action,
    discard: Action,
    context: _BlindContext | None = None,
) -> str:
    context = context or _BlindContext()
    best_score = _score_play_action(state, best_play, context)
    remaining_score = max(0, state.required_score - state.current_score)
    projected_score = _projected_score_after_discard(state, discard, context)
    return (
        f"last_hand_hunt current_score={best_score} projected_score={projected_score} "
        f"remaining={remaining_score} discards_left={state.discards_remaining}"
    )


def _winning_economy_hunt_discard_reason(
    state: GameState,
    best_play: Action,
    discard: Action,
    gain: float,
    projected_score: int,
    context: _BlindContext | None = None,
) -> str:
    context = context or _BlindContext()
    best_score = _score_play_action(state, best_play, context)
    drawn_cards = _known_draw_for_discard(state, discard)
    drawn_label = ",".join(card.short_name for card in drawn_cards if _card_is_economy_hunt_target(state, card)) or "-"
    return (
        f"winning_economy_hunt current_score={best_score} projected_score={projected_score} "
        f"gain={gain:.1f} targets={drawn_label} discarding={len(discard.card_indices)}"
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
    names = _active_joker_names(state)
    triggers: set[str] = set()
    if "Trading Card" in names and len(discarded_cards) == 1:
        triggers.add("Trading Card")
    if "Burnt Joker" in names:
        triggers.add("Burnt Joker")
    if "Hit the Road" in names and any(not card.debuffed and card.rank == "J" for card in discarded_cards):
        triggers.add("Hit the Road")
    castle_suit = _castle_target_suit(state)
    if castle_suit is not None and any(
        not card.debuffed and _normalize_suit(card.suit) == castle_suit for card in discarded_cards
    ):
        triggers.add("Castle")
    mail_rank = _mail_in_rebate_rank(state)
    if mail_rank is not None and any(
        not card.debuffed and _rank_matches(card.rank, mail_rank) for card in discarded_cards
    ):
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


def _mystic_summit_setup_discard_reason(
    state: GameState,
    best_play: Action,
    discard: Action,
    score: int,
    active_score: int,
    context: _BlindContext | None = None,
) -> str:
    remaining_score = max(0, state.required_score - state.current_score)
    projected_score = _projected_score_after_discard(state, discard, context)
    return (
        f"mystic_summit_setup current_score={score} active_score={active_score} "
        f"projected_score={projected_score} remaining={remaining_score} "
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


def _ante_one_upgrade_discard_reason(
    state: GameState,
    best_play: Action,
    discard: Action,
    hand_type: HandType,
    projected_score: int,
    context: _BlindContext | None = None,
) -> str:
    best_score = _score_play_action(state, best_play, context)
    remaining_score = max(0, state.required_score - state.current_score)
    return (
        f"ante_one_upgrade hand={hand_type.value} current_score={best_score} "
        f"projected_score={projected_score} target_score={remaining_score} "
        f"discarding={len(discard.card_indices)}"
    )


def _kept_hand_potential(state: GameState, kept_cards: tuple[Card, ...]) -> float:
    if not kept_cards:
        return 0.0
    cache = _state_scoped_cache("kept_hand_potential", state)
    cache_key = (_freeze_for_cache(kept_cards),)
    if cache is not None and cache_key in cache:
        return cache[cache_key]

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

    potential = cheap_potential + (immediate_score * 0.03)
    if cache is not None:
        cache[cache_key] = potential
    return potential


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


def _active_joker_names(state: GameState) -> set[str]:
    return {joker.name for joker in state.jokers if not _joker_is_disabled_for_build(joker)}


def _is_face_card_for_state(state: GameState, card: Card) -> bool:
    if card.debuffed:
        return False
    return card.rank in {"J", "Q", "K"} or any(
        joker.name == "Pareidolia" and not joker.effect.disabled for joker in state.jokers
    )


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
        if joker.name == joker_name and not joker.effect.disabled:
            return _joker_current_plus_value(joker, suffix=suffix)
    return 0


def _castle_target_suit(state: GameState) -> str | None:
    for joker in state.jokers:
        if joker.name != "Castle" or joker.effect.disabled:
            continue
        if joker.effect.discarded_suit or joker.effect.target_suit:
            return joker.effect.discarded_suit or joker.effect.target_suit
    return None


def _mail_in_rebate_rank(state: GameState) -> str | None:
    for joker in state.jokers:
        if joker.name != "Mail-In Rebate" or joker.effect.disabled:
            continue
        return joker.effect.discarded_rank
    return None


def _mail_in_rebate_rank_from_text(text: str) -> str | None:
    match = re.search(
        r"\bdiscarded\s+(Ace|King|Queen|Jack|Ten|Nine|Eight|Seven|Six|Five|Four|Three|Two|K|Q|J|10|[2-9])\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return _normalize_rank(match.group(1))


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
        "k": "K",
        "q": "Q",
        "j": "J",
        "t": "10",
    }.get(value, value.upper())


def _rank_from_text(text: str) -> str | None:
    match = re.search(
        r"\b(Ace|King|Queen|Jack|Ten|Nine|Eight|Seven|Six|Five|Four|Three|Two|K|Q|J|10|[2-9])\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return _normalize_rank(match.group(1))


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
    return _identity_cached_value("preferred_hand_type", state, lambda: _preferred_hand_type_uncached(state))


def _preferred_hand_type_uncached(state: GameState) -> HandType | None:
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
    if level > 0 and hand_type in ADVANCED_HAND_TYPES and not _advanced_hand_level_is_playable(state, hand_type):
        return 0
    if hand_type == HandType.PAIR and not _has_dedicated_pair_plan(state):
        return max(0, level - 1)
    if hand_type == HandType.TWO_PAIR and not _has_dedicated_two_pair_plan(state):
        return max(0, level - 1)
    return level


def _advanced_hand_level_is_playable(state: GameState, hand_type: HandType) -> bool:
    if _hand_archetype_support_count(state, hand_type) > 0:
        return True
    if any(RARE_HAND_JOKER_TARGETS.get(joker.name) == hand_type for joker in state.jokers):
        return True
    return hand_type_is_viable(state, hand_type)


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

PREFERRED_HAND_HUNT_TYPES = {
    HandType.THREE_OF_A_KIND,
    HandType.STRAIGHT,
    HandType.FLUSH,
    HandType.FULL_HOUSE,
    HandType.FOUR_OF_A_KIND,
    HandType.FIVE_OF_A_KIND,
    HandType.STRAIGHT_FLUSH,
    HandType.FLUSH_HOUSE,
    HandType.FLUSH_FIVE,
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


DECAYING_SCORE_JOKERS = frozenset(
    {
        "Ice Cream",
        "Popcorn",
        "Ramen",
        "Seltzer",
        "Turtle Bean",
    }
)


FINITE_SCORE_JOKERS = frozenset(
    {
        "Seltzer",
        "Turtle Bean",
    }
)


ROUND_RESET_SCORE_JOKERS = frozenset(
    {
        "Campfire",
    }
)


TEMPORARY_SCORE_JOKERS = DECAYING_SCORE_JOKERS | ROUND_RESET_SCORE_JOKERS | frozenset({"Gros Michel"})


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
    "Photograph",
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
    "The Emperor": 16,
    "The Hierophant": 18,
    "The Lovers": 20,
    "The Chariot": 24,
    "Justice": 32,
    "The Hermit": 34,
    "The Wheel of Fortune": 16,
    "Strength": 28,
    "The Hanged Man": 28,
    "Death": 34,
    "Temperance": 26,
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


SPECTRAL_CARD_NAMES = {
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


SPECTRAL_SEAL_VALUES = {
    "Deja Vu": 58.0,
    "Trance": 42.0,
    "Medium": 40.0,
    "Talisman": 38.0,
}


SUIT_TAROT_TARGET_SUITS = {
    "The Star": "D",
    "The Moon": "C",
    "The Sun": "H",
    "The World": "S",
}

VOUCHER_BUY_DENYLIST = frozenset(
    {
        "planet merchant",
        "magic trick",
        "directors cut",
        "crystal ball",
        "telescope",
    }
)

VOUCHER_IMMEDIATE_SCORE_NAMES = frozenset(
    {
        "Grabber",
        "Nacho Tong",
        "Wasteful",
        "Recyclomancy",
        "Paint Brush",
        "Palette",
        "Hone",
        "Glow Up",
        "Retcon",
        "Antimatter",
    }
)

VOUCHER_PRESSURE_ALLOWED_NAMES = frozenset(
    {
        "Grabber",
        "Nacho Tong",
        "Wasteful",
        "Recyclomancy",
        "Paint Brush",
        "Palette",
        "Antimatter",
    }
)

DANGEROUS_BOSS_BLINDS = frozenset(
    {
        "The Wall",
        "The Needle",
        "Violet Vessel",
        "Verdant Leaf",
        "Crimson Heart",
        "Amber Acorn",
        "Cerulean Bell",
        "The Eye",
        "The Mouth",
        "The Pillar",
        "The Psychic",
        "The Flint",
        "The Water",
        "The Arm",
        "The Manacle",
        "The Club",
        "The Goad",
        "The Head",
        "The Window",
    }
)

FINAL_BOSS_BLINDS = frozenset(
    {
        "Amber Acorn",
        "Cerulean Bell",
        "Crimson Heart",
        "Verdant Leaf",
        "Violet Vessel",
    }
)

FINAL_BOSS_FRAGILE_JOKERS = frozenset(
    {
        "Ancient Joker",
        "Blackboard",
        "Card Sharp",
        "Driver's License",
        "Flower Pot",
        "Photograph",
        "Seeing Double",
        "The Idol",
    }
)

ORDER_SENSITIVE_JOKERS = frozenset(
    {
        "Ancient Joker",
        "Baseball Card",
        "Baron",
        "Blackboard",
        "Blueprint",
        "Brainstorm",
        "Card Sharp",
        "Flower Pot",
        "Hanging Chad",
        "Photograph",
        "Seeing Double",
        "The Idol",
    }
)


VOUCHER_VALUES = {
    "Overstock": 34,
    "Overstock Plus": 42,
    "Clearance Sale": 38,
    "Liquidation": 46,
    "Hone": 30,
    "Glow Up": 40,
    "Reroll Surplus": 32,
    "Reroll Glut": 40,
    "Tarot Merchant": 24,
    "Tarot Tycoon": 30,
    "Planet Merchant": 0,
    "Planet Tycoon": 24,
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
    "Blank": 22,
    "Magic Trick": 0,
    "Illusion": 18,
    "Antimatter": 58,
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
