"""Joker valuation, replacement policy, and build-capacity bonuses."""

from __future__ import annotations

from dataclasses import replace

from balatro_ai.api.state import GameState, Joker
from balatro_ai.bots.basic_strategy.build_profile import (
    _build_profile,
    _joker_late_durability_factor,
    _owned_role_value,
    _urgent_late_role_hunt,
)
from balatro_ai.bots.basic_strategy.build_scoring import (
    _jokers_after_buy_for_scoring,
    _jokers_after_sell_for_scoring,
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
    ROLE_MISSING_BONUSES,
    TEMPORARY_SCORE_JOKERS,
    TWO_PAIR_SUPPORT_JOKERS,
)
from balatro_ai.bots.basic_strategy.hand_preferences import _has_dedicated_two_pair_plan, _preferred_hand_type
from balatro_ai.bots.basic_strategy.jokers import (
    _joker_current_plus_value,
    _joker_has_sticker,
    _joker_is_disabled_for_build,
    _joker_roles,
)
from balatro_ai.bots.basic_strategy.profile import _ShopPressure
from balatro_ai.bots.basic_strategy.rare_hands import (
    _rare_hand_deck_manipulation_need,
    _rare_hand_investment_penalty,
    _unsupported_rare_joker_extra_penalty,
    _unsupported_two_pair_joker_penalty,
)
from balatro_ai.bots.basic_strategy.shop_forecast import (
    _extrapolated_small_blind_score,
    _shop_pressure_effective_hands,
    _shop_pressure_score_state,
)
from balatro_ai.bots.basic_strategy.shop_pressure import _shop_hand_realism_factor
from balatro_ai.bots.basic_strategy.shop_safety import _hand_has_natural_support, _has_real_scoring_joker
from balatro_ai.bots.config import active_config
from balatro_ai.rules.hand_evaluator import HandType


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


def _joker_card_value(state: GameState, card: object) -> float:
    joker = _joker_from_shop_card(card)
    if _joker_would_overfill_slots(state, joker):
        return 0.0

    name = joker.name
    if _joker_stencil_would_fill_slots(state, joker):
        return 0.0
    durability = _joker_late_durability_factor(state, joker)
    sample_delta = _sample_score_delta_for_joker(state, joker) * _joker_sample_reliability(state, joker) * durability
    value = sample_delta * active_config().joker_sample_coefficient
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
    value = sample_gain * active_config().joker_sample_coefficient
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
    value = score_loss * active_config().joker_sample_coefficient * durability
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
