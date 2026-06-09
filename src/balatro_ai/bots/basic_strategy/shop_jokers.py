"""Joker valuation, replacement policy, and build-capacity bonuses."""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import replace

from balatro_ai.api.state import GameState, Joker
from balatro_ai.bots.basic_strategy.blind_solver import _blind_capacity_from_score
from balatro_ai.bots.basic_strategy.build_profile import (
    _build_profile,
    _joker_late_durability_factor,
    _owned_role_value,
    _urgent_late_role_hunt,
)
from balatro_ai.bots.basic_strategy.build_scoring import (
    _jokers_after_buy_for_scoring,
    _jokers_after_sell_for_scoring,
    _repeatable_build_score,
    _sample_build_score,
    _sample_score_delta_for_joker,
    _sample_score_gain_for_joker,
)
from balatro_ai.bots.basic_strategy.cards import (
    _card_cost,
    _early_power_bonus,
    _edition_bonus,
    _has_consumable_room,
    _joker_from_shop_card,
    _joker_would_overfill_slots,
    _normal_joker_open_slots,
    _normal_joker_slots_used,
    _uses_normal_joker_slot,
)
from balatro_ai.bots.basic_strategy.data import (
    ANTE_SMALL_BLIND_SCORES,
    DECAYING_SCORE_JOKERS,
    DEDICATED_TWO_PAIR_BUILD_JOKERS,
    GLASS_CANNON_JOKERS,
    JOKER_BASE_VALUES,
    JOKER_ECONOMY_VALUES,
    JOKER_HAND_SYNERGY,
    JOKER_PRIMARY_HAND,
    JOKER_SCALING_VALUES,
    LOW_PRIORITY_JOKERS,
    RARE_HAND_JOKER_TARGETS,
    RARE_HAND_SUPPORT_TAROTS,
    ROLE_MISSING_BONUSES,
    TEMPORARY_SCORE_JOKERS,
    TWO_PAIR_SUPPORT_JOKERS,
)
from balatro_ai.bots.basic_strategy.hand_preferences import _has_dedicated_two_pair_plan, _preferred_hand_type
from balatro_ai.bots.basic_strategy.jokers import (
    _driver_license_readiness_factor,
    _joker_current_plus_value,
    _joker_current_xmult_value,
    _joker_has_sticker,
    _joker_is_disabled_for_build,
    _joker_roles,
)
from balatro_ai.bots.basic_strategy.profile import _BuildProfile, _ShopPressure
from balatro_ai.bots.basic_strategy.rare_hands import (
    _rare_hand_deck_manipulation_need,
    _rare_hand_investment_penalty,
    _unsupported_rare_joker_extra_penalty,
    _unsupported_two_pair_joker_penalty,
)
from balatro_ai.bots.basic_strategy.shop_forecast import (
    _boss_preview_weight,
    _extrapolated_small_blind_score,
    _shop_pressure_boss_discards_remaining,
    _shop_pressure_boss_hands_remaining,
    _shop_pressure_effective_hands,
    _shop_pressure_score_state,
    _upcoming_boss_blind_name,
)
from balatro_ai.bots.basic_strategy.shop_pressure import _shop_hand_realism_factor
from balatro_ai.bots.basic_strategy.shop_safety import _hand_has_natural_support, _has_real_scoring_joker
from balatro_ai.bots.config import active_config
from balatro_ai.rules.hand_evaluator import HandType

_RARE_HAND_COMMITMENT_RELIABILITY_WEIGHT: float = float(
    os.environ.get("BALATRO_RARE_HAND_COMMITMENT_RELIABILITY_W", "1.0")
)
_STENCIL_SLOT_DISCIPLINE_WEIGHT: float = float(
    os.environ.get("BALATRO_STENCIL_SLOT_DISCIPLINE_W", "0.0")
)


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
    sold_roles = _contextual_joker_roles(state, sold_joker)
    candidate_roles = _contextual_joker_roles(state, candidate)
    candidate_relevance = _joker_state_relevance_factor(state, candidate)
    missing = set(profile.missing_roles)
    bonus = 0.0

    if "xmult" in candidate_roles and "xmult" in missing and "xmult" not in sold_roles:
        bonus += 30.0 * max(0.35, profile.role_deficit_ratio("xmult")) * candidate_relevance
    if "scaling" in candidate_roles and "scaling" in missing and "scaling" not in sold_roles:
        bonus += 24.0 * max(0.35, profile.role_deficit_ratio("scaling")) * candidate_relevance
    if _urgent_late_role_hunt(state, pressure, profile) and candidate_roles & {"xmult", "scaling"}:
        bonus += 14.0 * candidate_relevance
    for role in ("xmult", "scaling"):
        if role not in sold_roles or role in candidate_roles:
            continue
        deficit = max(0.35, profile.role_deficit_ratio(role))
        if role == "scaling":
            bonus -= _lost_scaling_replacement_penalty(state, sold_joker, pressure, profile)
        elif _owned_joker_role_source_count(state, role) <= 1:
            bonus -= 70.0 * deficit
        elif role in missing:
            bonus -= (80.0 if role == "xmult" else 65.0) * deficit
        else:
            bonus -= 24.0
    if sold_roles & {"chips", "mult"} and candidate_roles.isdisjoint({"chips", "mult", "xmult", "scaling"}):
        bonus -= 12.0
    if candidate_roles.isdisjoint({"xmult", "scaling"}) and _urgent_late_role_hunt(state, pressure, profile):
        bonus -= 14.0
    return bonus


def _owned_joker_role_source_count(state: GameState, role: str) -> int:
    return sum(1 for joker in state.jokers if role in _contextual_joker_roles(state, joker))


def _lost_scaling_replacement_penalty(
    state: GameState,
    joker: Joker,
    pressure: _ShopPressure,
    profile: _BuildProfile,
) -> float:
    if _owned_joker_role_source_count(state, "scaling") > 1:
        return 24.0
    runway = _remaining_scaling_blinds(state)
    current_mult = max(0, _joker_current_plus_value(joker, suffix="mult"))
    current_chips = max(0, _joker_current_plus_value(joker, suffix="chips"))
    current_value = current_mult * 1.4 + current_chips * 0.18
    future_value = _future_scaling_runway_value(state, joker, pressure, profile, runway)
    if profile.ante >= 7 and future_value <= 8.0:
        return 12.0 + min(16.0, current_value)
    return min(95.0, 14.0 + current_value + future_value)


def _remaining_scaling_blinds(state: GameState) -> int:
    remaining = max(0, 8 - state.ante) * 3
    if state.blind == "Small Blind":
        remaining += 2
    elif state.blind == "Big Blind":
        remaining += 1
    elif state.blind:
        remaining += 1
    return max(0, remaining)


def _future_scaling_runway_value(
    state: GameState,
    joker: Joker,
    pressure: _ShopPressure,
    profile: _BuildProfile,
    runway: int,
) -> float:
    if runway <= 0:
        return 0.0
    name = joker.name
    if name == "Green Joker":
        net_hands = _expected_green_joker_net_hands(pressure) * runway
        return max(0.0, net_hands) * 2.7
    if name == "Ride the Bus":
        return runway * 2.2
    if name == "Spare Trousers":
        multiplier = 3.0 if _has_dedicated_two_pair_plan(state) else 1.4
        return runway * multiplier
    if name in {"Runner", "Square Joker", "Wee Joker", "Castle"}:
        return runway * 1.7
    if name in {"Hologram", "Constellation", "Lucky Cat", "Glass Joker", "Campfire", "Flash Card", "Red Card"}:
        money_factor = 1.2 if profile.rich else 0.75
        return runway * 2.2 * money_factor
    return runway * 1.8


def _expected_green_joker_net_hands(pressure: _ShopPressure) -> float:
    if pressure.ratio >= 1.6 or pressure.raw_ratio >= 1.15:
        return 0.15
    if pressure.ratio >= 1.15 or pressure.raw_ratio >= 0.9:
        return 0.75
    if pressure.ratio >= 0.9:
        return 1.35
    return 2.0


def _joker_card_value(state: GameState, card: object, pressure: _ShopPressure | None = None) -> float:
    joker = _joker_from_shop_card(card)
    if _joker_would_overfill_slots(state, joker):
        return 0.0

    name = joker.name
    if _joker_stencil_would_fill_slots(state, joker):
        return 0.0
    score_state = _joker_score_state(state, pressure)
    durability = _joker_late_durability_factor(state, joker)
    raw_sample_delta = _sample_score_delta_for_joker(score_state, joker)
    relevance = _joker_boss_score_relevance_factor(score_state, joker, pressure, raw_sample_delta)
    preview = _joker_boss_preview_sample(state, joker, pressure, score_state)
    if preview is not None:
        raw_sample_delta, relevance = preview
    relevance *= _joker_state_relevance_factor(state, joker)
    commitment_reliability = _rare_hand_commitment_reliability(score_state, joker)
    sample_delta = (
        raw_sample_delta
        * _joker_sample_reliability(score_state, joker)
        * commitment_reliability
        * durability
    )
    value = sample_delta * active_config().joker_sample_coefficient
    value += _normal_joker_open_slots(state) * 6 * relevance
    value += max(0, 4 - state.ante) * _early_power_bonus(name) * relevance
    value += _joker_heuristic_value(state, joker, pressure=pressure) * relevance
    value += _joker_role_bonus(state, joker, pressure=pressure) * relevance
    value -= _unsupported_two_pair_joker_penalty(state, name)
    value -= _unsupported_rare_joker_extra_penalty(state, name)
    value -= _stencil_slot_discipline_penalty(state, joker)
    value += _edition_bonus(joker.edition)
    value -= _boss_inactive_joker_penalty(pressure, relevance)

    if any(existing.name == name for existing in state.jokers):
        value -= 18
    if _normal_joker_slots_used(state) == 0 and value < 35:
        value += 10
    value = _capped_red_card_candidate_value(state, joker, value)
    if commitment_reliability < 1.0:
        value *= max(0.05, commitment_reliability)
    return value


def _candidate_joker_value_for_replacement(
    state: GameState,
    joker: Joker,
    pressure: _ShopPressure | None = None,
) -> float:
    if _joker_stencil_would_fill_slots(state, joker):
        return 0.0
    score_state = _joker_score_state(state, pressure)
    durability = _joker_late_durability_factor(state, joker)
    raw_sample_gain = _sample_score_gain_for_joker(score_state, joker)
    relevance = _joker_boss_score_relevance_factor(score_state, joker, pressure, raw_sample_gain)
    preview = _joker_boss_preview_sample(state, joker, pressure, score_state, gain_only=True)
    if preview is not None:
        raw_sample_gain, relevance = preview
    relevance *= _joker_state_relevance_factor(state, joker)
    commitment_reliability = _rare_hand_commitment_reliability(score_state, joker)
    sample_gain = (
        raw_sample_gain
        * _joker_sample_reliability(score_state, joker)
        * commitment_reliability
        * durability
    )
    value = sample_gain * active_config().joker_sample_coefficient
    value += _joker_heuristic_value(state, joker, pressure=pressure) * relevance
    value += _joker_role_bonus(state, joker, pressure=pressure) * relevance
    value -= _unsupported_two_pair_joker_penalty(state, joker.name)
    value -= _unsupported_rare_joker_extra_penalty(state, joker.name)
    value += _edition_bonus(joker.edition)
    value -= _boss_inactive_joker_penalty(pressure, relevance)
    value = _capped_red_card_candidate_value(state, joker, value)
    if any(existing.name == joker.name for existing in state.jokers):
        value -= 18
    if commitment_reliability < 1.0:
        value *= max(0.05, commitment_reliability)
    return value


def _owned_joker_value(
    state: GameState,
    joker: Joker,
    *,
    remove_index: int,
    pressure: _ShopPressure | None = None,
) -> float:
    score_state = _joker_score_state(state, pressure)
    without = _jokers_after_sell_for_scoring(score_state, remove_index=remove_index)
    score_loss = max(
        0.0,
        _sample_build_score(score_state, score_state.jokers) - _sample_build_score(score_state, without),
    )
    relevance = _joker_boss_score_relevance_factor(score_state, joker, pressure, score_loss)
    preview = _owned_joker_boss_preview_loss(state, joker, pressure, remove_index)
    if preview is not None:
        score_loss, relevance = preview
    relevance *= _joker_state_relevance_factor(state, joker)
    durability = _joker_late_durability_factor(state, joker)
    commitment_reliability = _rare_hand_commitment_reliability(score_state, joker)
    value = (
        score_loss
        * active_config().joker_sample_coefficient
        * durability
        * commitment_reliability
    )
    value += _joker_heuristic_value(state, joker, pressure=pressure) * 0.75 * relevance
    value += _owned_role_value(state, joker, remove_index=remove_index) * relevance
    value += _edition_bonus(joker.edition)
    value += (joker.sell_value or 0) * 1.5
    if joker.name in LOW_PRIORITY_JOKERS:
        value -= 20
    if joker.name in TWO_PAIR_SUPPORT_JOKERS and not _has_dedicated_two_pair_plan(state):
        value -= 45
    if commitment_reliability < 1.0:
        value *= max(0.05, commitment_reliability)
    return value


def _joker_score_state(state: GameState, pressure: _ShopPressure | None) -> GameState:
    if pressure is None or pressure.boss_name is None:
        return state
    raw_target = pressure.target_score / max(0.1, pressure.safety_multiplier)
    return _shop_pressure_score_state(state, pressure.boss_name, raw_target=raw_target)


def _joker_forced_boss_score_state(state: GameState, pressure: _ShopPressure) -> GameState:
    raw_target = pressure.target_score / max(0.1, pressure.safety_multiplier)
    return replace(
        state,
        blind=str(pressure.boss_name),
        required_score=int(raw_target),
        hands_remaining=_shop_pressure_boss_hands_remaining(state, pressure.boss_name),
        discards_remaining=_shop_pressure_boss_discards_remaining(state, pressure.boss_name),
    )


def _joker_boss_preview_sample(
    state: GameState,
    joker: Joker,
    pressure: _ShopPressure | None,
    score_state: GameState,
    *,
    gain_only: bool = False,
) -> tuple[float, float] | None:
    weight = _joker_boss_preview_weight(state, joker, pressure, score_state)
    if weight <= 0.0 or pressure is None:
        return None

    boss_state = _joker_forced_boss_score_state(state, pressure)
    current_delta = _sample_score_gain_for_joker(state, joker) if gain_only else _sample_score_delta_for_joker(state, joker)
    boss_delta = _sample_score_gain_for_joker(boss_state, joker) if gain_only else _sample_score_delta_for_joker(boss_state, joker)
    current_relevance = _joker_boss_score_relevance_factor(state, joker, pressure, current_delta)
    boss_relevance = _joker_boss_score_relevance_factor(boss_state, joker, pressure, boss_delta)
    return (
        (current_delta * (1.0 - weight)) + (boss_delta * weight),
        (current_relevance * (1.0 - weight)) + (boss_relevance * weight),
    )


def _owned_joker_boss_preview_loss(
    state: GameState,
    joker: Joker,
    pressure: _ShopPressure | None,
    remove_index: int,
) -> tuple[float, float] | None:
    score_state = _joker_score_state(state, pressure)
    weight = _joker_boss_preview_weight(state, joker, pressure, score_state)
    if weight <= 0.0 or pressure is None:
        return None

    boss_state = _joker_forced_boss_score_state(state, pressure)
    current_without = _jokers_after_sell_for_scoring(state, remove_index=remove_index)
    boss_without = _jokers_after_sell_for_scoring(boss_state, remove_index=remove_index)
    current_loss = max(0.0, _sample_build_score(state, state.jokers) - _sample_build_score(state, current_without))
    boss_loss = max(
        0.0,
        _sample_build_score(boss_state, boss_state.jokers) - _sample_build_score(boss_state, boss_without),
    )
    current_relevance = _joker_boss_score_relevance_factor(state, joker, pressure, current_loss)
    boss_relevance = _joker_boss_score_relevance_factor(boss_state, joker, pressure, boss_loss)
    return (
        (current_loss * (1.0 - weight)) + (boss_loss * weight),
        (current_relevance * (1.0 - weight)) + (boss_relevance * weight),
    )


def _joker_boss_preview_weight(
    state: GameState,
    joker: Joker,
    pressure: _ShopPressure | None,
    score_state: GameState,
) -> float:
    if pressure is None or pressure.boss_name is None or score_state is not state:
        return 0.0
    if _upcoming_boss_blind_name(state) != pressure.boss_name:
        return 0.0
    if not _joker_needs_boss_preview_score_context(joker, pressure.boss_name):
        return 0.0
    return max(0.0, min(1.0, _boss_preview_weight(state)))


def _joker_needs_boss_preview_score_context(joker: Joker, boss_name: str | None) -> bool:
    return joker.name == "Banner" and boss_name == "The Water"


def _joker_boss_score_relevance_factor(
    score_state: GameState,
    joker: Joker,
    pressure: _ShopPressure | None,
    sample_delta: float,
) -> float:
    if pressure is None:
        return 1.0
    if sample_delta > 0.0:
        return 1.0
    if score_state.blind == "The Water" and score_state.discards_remaining <= 0 and joker.name == "Banner":
        return 0.0
    return 1.0


def _joker_state_relevance_factor(state: GameState, joker: Joker) -> float:
    return _driver_license_readiness_factor(state, joker)


def _contextual_joker_roles(state: GameState, joker: Joker) -> frozenset[str]:
    roles = _joker_roles(joker)
    if joker.name == "Driver's License" and _joker_state_relevance_factor(state, joker) < 0.5:
        return frozenset(role for role in roles if role != "xmult")
    return roles


def _boss_inactive_joker_penalty(pressure: _ShopPressure | None, relevance: float) -> float:
    if pressure is None or relevance >= 1.0:
        return 0.0
    inactive_weight = 1.0 - max(0.0, relevance)
    if pressure.ratio < 0.95 and pressure.raw_ratio < 0.95:
        return 12.0 * inactive_weight
    return (30.0 + pressure.danger * 16.0) * inactive_weight


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


def _joker_heuristic_value(state: GameState, joker: Joker, pressure: _ShopPressure | None = None) -> float:
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
        value += _scaling_joker_heuristic_value(state, joker, pressure)
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


def _scaling_joker_heuristic_value(
    state: GameState,
    joker: Joker,
    pressure: _ShopPressure | None = None,
) -> float:
    base = float(JOKER_SCALING_VALUES.get(joker.name, 0))
    if base <= 0.0:
        return 0.0
    if pressure is None:
        return base * (1 + max(0, 4 - state.ante) * 0.15)

    profile = _build_profile(state)
    runway = _remaining_scaling_blinds(state)
    current_value = _current_scaling_score_value(joker)
    future_value = _future_scaling_runway_value(state, joker, pressure, profile, runway)
    reasonable_value = current_value + future_value

    capped_value = min(base, 10.0 + reasonable_value)
    if state.ante <= 3 and reasonable_value > base:
        capped_value += min(18.0, (reasonable_value - base) * 0.25)
    elif state.ante <= 4 and reasonable_value > base:
        capped_value += min(10.0, (reasonable_value - base) * 0.18)

    if state.ante >= 6 and (pressure.ratio >= 1.25 or pressure.raw_ratio >= 1.0) and reasonable_value < base * 0.55:
        capped_value *= 0.72
    return max(0.0, capped_value)


def _current_scaling_score_value(joker: Joker) -> float:
    current_mult = max(0, _joker_current_plus_value(joker, suffix="mult"))
    current_chips = max(0, _joker_current_plus_value(joker, suffix="chips"))
    current_xmult = max(1.0, _joker_current_xmult_value(joker))
    return current_mult * 1.25 + current_chips * 0.18 + max(0.0, current_xmult - 1.0) * 24.0


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


def _stencil_slot_discipline_penalty(state: GameState, joker: Joker) -> float:
    weight = _STENCIL_SLOT_DISCIPLINE_WEIGHT
    if weight <= 0.0:
        return 0.0
    if joker.name == "Joker Stencil" or not _uses_normal_joker_slot(joker):
        return 0.0
    if not any(existing.name == "Joker Stencil" for existing in state.jokers):
        return 0.0

    remaining_open_slots = max(0, _normal_joker_open_slots(state) - 1)
    penalty = 10.0 + max(0, 3 - remaining_open_slots) * 18.0
    roles = _contextual_joker_roles(state, joker)
    if roles & {"xmult", "scaling"}:
        penalty *= 0.55
    elif roles & {"mult", "chips"}:
        penalty *= 0.75
    return min(80.0, max(0.0, weight) * penalty)


def _joker_sample_reliability(state: GameState, joker: Joker) -> float:
    primary = JOKER_PRIMARY_HAND.get(joker.name)
    if primary == HandType.TWO_PAIR and not _has_dedicated_two_pair_plan(state):
        return 0.65
    if primary in {HandType.THREE_OF_A_KIND, HandType.FOUR_OF_A_KIND, HandType.FIVE_OF_A_KIND}:
        wanted = set(JOKER_HAND_SYNERGY.get(joker.name, ()))
        if not _hand_has_natural_support((*state.hand, *state.known_deck), wanted):
            return 0.35 if state.ante >= 3 else 0.55
    return 1.0


def _rare_hand_commitment_reliability(state: GameState, joker: Joker) -> float:
    weight = _RARE_HAND_COMMITMENT_RELIABILITY_WEIGHT
    if weight <= 0.0:
        return 1.0
    target = RARE_HAND_JOKER_TARGETS.get(joker.name)
    if target is None:
        return 1.0
    if state.ante <= 2:
        return 1.0

    support = 0.0
    try:
        support += max(0.0, float(state.hand_levels.get(target.value, 1)) - 1.0) * 0.45
    except (TypeError, ValueError, AttributeError):
        pass
    support += sum(1 for name in state.consumables if name in RARE_HAND_SUPPORT_TAROTS) * 0.35
    support += _extra_duplicate_support_for_rare_hand(state, target)
    if support >= 1.0:
        return 1.0

    base = 0.03
    reliability = base + support * (1.0 - base)
    reliability = max(0.05, min(1.0, reliability))
    return 1.0 - min(1.0, weight) * (1.0 - reliability)


def _extra_duplicate_support_for_rare_hand(state: GameState, hand_type: HandType) -> float:
    cards = (*state.hand, *state.known_deck)
    if not cards:
        return 0.0
    rank_counts = Counter(card.rank for card in cards)
    exact_counts = Counter((card.rank, card.suit) for card in cards)
    max_rank = max(rank_counts.values(), default=0)
    max_exact = max(exact_counts.values(), default=0)
    if hand_type == HandType.FOUR_OF_A_KIND:
        return max(0.0, max_rank - 4.0) * 0.35
    if hand_type == HandType.FIVE_OF_A_KIND:
        return max(0.0, max_rank - 4.0) * 0.45
    if hand_type == HandType.FLUSH_FIVE:
        return max(0.0, max_exact - 1.0) * 0.45
    if hand_type == HandType.FLUSH_HOUSE:
        return max(0.0, max_exact - 1.0) * 0.30
    return 0.0


def _joker_role_bonus(state: GameState, joker: Joker, pressure: _ShopPressure | None = None) -> float:
    if _joker_stencil_would_fill_slots(state, joker):
        return -30.0
    profile = _build_profile(state)
    roles = _contextual_joker_roles(state, joker)
    durability = _joker_late_durability_factor(state, joker)
    bonus = 0.0
    for role in profile.missing_roles:
        if role in roles:
            role_bonus = ROLE_MISSING_BONUSES[role] * max(0.35, profile.role_deficit_ratio(role)) * durability
            if role == "scaling" and pressure is not None:
                role_bonus *= _scaling_pressure_role_factor(state, joker, pressure, profile)
            bonus += role_bonus

    if profile.late and profile.rich:
        if "xmult" in roles and not profile.has_xmult:
            bonus += 22 * max(0.4, profile.role_deficit_ratio("xmult")) * durability
        if "scaling" in roles and not profile.has_scaling:
            runway_factor = (
                _scaling_pressure_role_factor(state, joker, pressure, profile)
                if pressure is not None
                else 1.0
            )
            bonus += 16 * max(0.4, profile.role_deficit_ratio("scaling")) * durability * runway_factor
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
    relevance = _joker_state_relevance_factor(state, joker)
    if relevance <= 0.0:
        return 0.0
    rare_target = RARE_HAND_JOKER_TARGETS.get(joker.name)
    if rare_target is not None and _rare_hand_deck_manipulation_need(state, rare_target) > 0:
        return 0.0

    profile = _build_profile(state)
    roles = _contextual_joker_roles(state, joker)
    durability = _joker_late_durability_factor(state, joker)
    bonus = 0.0
    if "scaling" in roles and not profile.has_scaling:
        deficit = max(0.35, profile.role_deficit_ratio("scaling"))
        runway_factor = _scaling_pressure_role_factor(state, joker, pressure, profile)
        bonus += (
            18.0 + pressure.danger * 28.0 + max(0, state.ante - 2) * 4.0
        ) * deficit * durability * runway_factor
        if _urgent_late_role_hunt(state, pressure, profile):
            bonus += 16.0 * deficit * durability * runway_factor
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
    return bonus * relevance


def _scaling_pressure_role_factor(
    state: GameState,
    joker: Joker,
    pressure: _ShopPressure,
    profile: _BuildProfile,
) -> float:
    if joker.name not in JOKER_SCALING_VALUES:
        return 1.0
    runway = _remaining_scaling_blinds(state)
    runway_value = _current_scaling_score_value(joker) + _future_scaling_runway_value(
        state,
        joker,
        pressure,
        profile,
        runway,
    )
    if state.ante <= 3:
        return 1.1
    if runway_value >= 24.0:
        return 1.0
    if runway_value >= 12.0:
        return 0.65
    if pressure.ratio >= 1.15 or pressure.raw_ratio >= 0.95:
        return 0.08
    return 0.18


def _future_score_headroom_joker_bonus(state: GameState, joker: Joker, pressure: _ShopPressure) -> float:
    if state.ante >= 8 or _joker_is_disabled_for_build(joker):
        return 0.0
    if _joker_state_relevance_factor(state, joker) <= 0.0:
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
    score_state = replace(state, jokers=jokers)
    score_state = _joker_score_state(score_state, pressure)
    current_score = _repeatable_build_score(score_state, score_state.jokers) * _shop_hand_realism_factor(
        score_state,
        boss_name=pressure.boss_name,
    )
    effective_hands = _shop_pressure_effective_hands(state, pressure.boss_name)
    return _blind_capacity_from_score(
        current_score,
        effective_hands,
        capacity_safety_factor=pressure.capacity_safety_factor,
    )


def _future_headroom_required_score(state: GameState) -> float:
    ante = min(8, max(1, state.ante) + 2)
    small = ANTE_SMALL_BLIND_SCORES.get(ante, _extrapolated_small_blind_score(ante))
    return small * 2.0


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
        penalty += 28.0
    return penalty
