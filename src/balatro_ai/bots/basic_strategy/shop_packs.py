"""Booster pack valuation and late-shop pack opening policy."""

from __future__ import annotations

from dataclasses import replace

from balatro_ai.api.state import GameState
from balatro_ai.bots.basic_strategy.build_profile import _build_profile
from balatro_ai.bots.basic_strategy.build_scoring import _sample_build_score
from balatro_ai.bots.basic_strategy.cards import _card_cost, _card_label, _normal_joker_open_slots
from balatro_ai.bots.basic_strategy.hand_preferences import _flexible_hand_types, _preferred_hand_type
from balatro_ai.bots.basic_strategy.jokers import _active_joker_names
from balatro_ai.bots.basic_strategy.profile import _ShopContext, _ShopPressure
from balatro_ai.bots.basic_strategy.run_plan import _RunPlan
from balatro_ai.bots.basic_strategy.rare_hands import _rare_hand_deck_manipulation_need, _rare_hand_plan
from balatro_ai.bots.basic_strategy.shop_reroll import _late_pressure_closer_mode, _rich_late_role_hunt
from balatro_ai.rules.hand_evaluator import HandType


def _pack_value(state: GameState, pack: object, plan: _RunPlan | None = None) -> float:
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
        value = 6
        if _standard_pack_has_build_payoff(state):
            value += 18
            value += _rare_hand_pack_bonus(state, "standard")
    elif "spectral" in name:
        value = 20 if state.ante <= 2 else 14
        value += _rare_hand_pack_bonus(state, "spectral")
    else:
        value = 16
    if "mega" in name or "jumbo" in name:
        value += 8
    if plan is not None and plan.prioritizes_pack(_pack_kind(pack)):
        value += _run_plan_pack_bonus(plan)
    return float(value)


def _is_buffoon_pack(pack: object) -> bool:
    return "buffoon" in _card_label(pack).lower()


def _is_standard_pack(pack: object) -> bool:
    return "standard" in _card_label(pack).lower()


def _standard_pack_has_build_payoff(state: GameState) -> bool:
    names = _active_joker_names(state)
    if names & {"Hologram", "Vampire", "Midas Mask", "DNA", "Lucky Cat"}:
        return True
    return _rare_hand_pack_bonus(state, "standard") > 0.0


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
    plan: _RunPlan | None = None,
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
    if plan is not None and plan.prioritizes_pack(_pack_kind(pack)):
        floor *= _run_plan_pack_floor_factor(plan)
    if pressure.ratio < 0.65:
        return capacity_gain >= floor
    return capacity_gain >= floor * 0.65


def _pack_kind(pack: object) -> str:
    label = _card_label(pack).lower()
    for kind in ("buffoon", "celestial", "arcana", "standard", "spectral"):
        if kind in label:
            return kind
    return "unknown"


def _run_plan_pack_bonus(plan: _RunPlan) -> float:
    if plan.shop_posture == "panic":
        return 18.0
    if plan.shop_posture == "spend":
        return 14.0
    if plan.shop_posture == "build":
        return 9.0
    return 4.0


def _run_plan_pack_floor_factor(plan: _RunPlan) -> float:
    if plan.shop_posture == "panic":
        return 0.35
    if plan.shop_posture == "spend":
        return 0.50
    if plan.shop_posture == "build":
        return 0.70
    return 0.85


def _late_pack_limit(state: GameState, pressure: _ShopPressure) -> int:
    if state.ante < 5:
        return 99
    if state.ante >= 8 and pressure.boss_name == "Violet Vessel" and pressure.ratio >= 2.0 and state.money >= 25:
        return 5
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
        if not _standard_pack_has_build_payoff(state):
            return -spend_loss
        rate = 0.04
        if _active_joker_names(state) & {"Hologram", "Vampire", "Midas Mask", "DNA", "Lucky Cat"}:
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
