"""Shop pressure calculation."""

from __future__ import annotations

import sys

from balatro_ai.api.state import GameState
from balatro_ai.bots.basic_strategy.cache import _identity_cached_value
from balatro_ai.bots.basic_strategy.cards import _normal_joker_open_slots, _normal_joker_slots_used
from balatro_ai.bots.basic_strategy.data import PREFERRED_HAND_HUNT_TYPES, RARE_HAND_TYPES
from balatro_ai.bots.basic_strategy.profile import _BuildProfile, _ShopPressure
from balatro_ai.bots.basic_strategy.rare_hands import _rare_hand_deck_manipulation_need
from balatro_ai.bots.basic_strategy.shop_forecast import (
    _boss_preview_weight,
    _boss_target_safety_bonus,
    _effective_boss_target_multiplier,
    _estimated_shop_planning_required_score,
    _has_planet_investment,
    _shop_pressure_boss_capacity_factor,
    _shop_pressure_effective_hands,
    _shop_pressure_score_state,
    _upcoming_boss_blind_name,
)
from balatro_ai.bots.basic_strategy.shop_safety import _has_money_scaling_joker
from balatro_ai.bots.config import active_config
from balatro_ai.rules.hand_evaluator import HandType


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
    cfg = active_config()
    multiplier = cfg.shop_target_safety_base
    if state.ante >= 3:
        multiplier += cfg.shop_safety_ante3_bonus
    if state.ante >= 4:
        multiplier += cfg.shop_safety_ante4_bonus
    if state.ante >= 5:
        multiplier += cfg.shop_safety_ante5_bonus
    if _normal_joker_open_slots(state) <= 0 and not _has_money_scaling_joker(state):
        multiplier += cfg.shop_safety_full_slots_bonus
    if state.blind in {"The Wall", "The Needle"}:
        multiplier += 0.12
    elif state.blind in {"The Eye", "The Mouth"}:
        multiplier += 0.08
    elif state.blind in {"The Water", "The Arm"}:
        multiplier += 0.06
    boss_name = _upcoming_boss_blind_name(state)
    if boss_name is not None:
        multiplier += _boss_target_safety_bonus(boss_name) * _boss_preview_weight(state)
    return min(cfg.shop_safety_cap, multiplier)


def _shop_capacity_safety_factor(state: GameState) -> float:
    if state.ante < 4:
        return 1.0
    if _has_money_scaling_joker(state):
        return 1.0

    cfg = active_config()
    profile = _build_profile(state)
    factor = 1.0
    if not profile.has_xmult:
        factor -= cfg.capacity_no_xmult_penalty
    if not _has_planet_investment(state):
        factor -= cfg.capacity_no_planet_penalty
    if _normal_joker_open_slots(state) <= 0 and not _has_money_scaling_joker(state):
        factor -= cfg.capacity_full_slots_penalty
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
        factor -= cfg.capacity_rare_hand_penalty
    if state.ante >= 5:
        factor -= cfg.capacity_ante5_penalty
    return max(cfg.capacity_floor, factor)


def _early_build_pressure_floor(state: GameState) -> float:
    joker_count = _normal_joker_slots_used(state)
    if state.ante <= 1 and joker_count == 0:
        return 1.25
    if state.ante <= 2 and joker_count < 2:
        return 1.15
    if state.ante <= 3 and joker_count < 3:
        return 1.05
    return 0.0


def _build_profile(*args, **kwargs) -> _BuildProfile:
    """Honor legacy tests/tools that patch ``basic_strategy_bot._build_profile``."""

    strategy_module = sys.modules.get("balatro_ai.bots.basic_strategy_bot")
    patched = getattr(strategy_module, "_build_profile", None) if strategy_module is not None else None
    if patched is None or patched is _build_profile:
        raise RuntimeError("basic_strategy_bot._build_profile is not available")
    return patched(*args, **kwargs)


def _preferred_hand_type(*args, **kwargs):
    """Honor legacy tests/tools that patch ``basic_strategy_bot._preferred_hand_type``."""

    strategy_module = sys.modules.get("balatro_ai.bots.basic_strategy_bot")
    patched = getattr(strategy_module, "_preferred_hand_type", None) if strategy_module is not None else None
    if patched is None or patched is _preferred_hand_type:
        raise RuntimeError("basic_strategy_bot._preferred_hand_type is not available")
    return patched(*args, **kwargs)


def _sample_build_score(*args, **kwargs) -> float:
    """Honor legacy tests/tools that patch ``basic_strategy_bot._sample_build_score``."""

    strategy_module = sys.modules.get("balatro_ai.bots.basic_strategy_bot")
    patched = getattr(strategy_module, "_sample_build_score", None) if strategy_module is not None else None
    if patched is None or patched is _sample_build_score:
        raise RuntimeError("basic_strategy_bot._sample_build_score is not available")
    return patched(*args, **kwargs)
