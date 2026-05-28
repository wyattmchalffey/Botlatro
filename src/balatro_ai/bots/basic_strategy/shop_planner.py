"""Auditable calibrated shop/build planner terms."""

from __future__ import annotations

from dataclasses import dataclass

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GameState
from balatro_ai.bots.basic_strategy.build_profile import _build_profile
from balatro_ai.bots.basic_strategy.build_scoring import _normal_slot_joker_card, _repeatable_build_score
from balatro_ai.bots.basic_strategy.cards import (
    _card_label,
    _is_joker_card,
    _joker_from_shop_card,
    _normal_joker_open_slots,
)
from balatro_ai.bots.basic_strategy.data import DANGEROUS_BOSS_BLINDS, FINAL_BOSS_BLINDS
from balatro_ai.bots.basic_strategy.jokers import _joker_roles
from balatro_ai.bots.basic_strategy.profile import CRITICAL_BUILD_ROLES, _BuildProfile, _ShopContext, _ShopPressure
from balatro_ai.bots.basic_strategy.run_plan import _RunPlan, _run_plan
from balatro_ai.bots.basic_strategy.shop_items import _shop_item_for_action
from balatro_ai.bots.basic_strategy.shop_money import _desired_money_reserve, _spendable_money
from balatro_ai.bots.basic_strategy.shop_packs import _is_buffoon_pack
from balatro_ai.bots.basic_strategy.shop_pressure import _shop_pressure
from balatro_ai.bots.basic_strategy.shop_safety import _has_money_scaling_joker
from balatro_ai.bots.basic_strategy.shop_values import _shop_action_cost, _shop_action_value
from balatro_ai.bots.config import active_config
from balatro_ai.search.forward_sim import simulate_buy, simulate_sell, simulate_use_consumable


@dataclass(frozen=True, slots=True)
class _ShopPlannerTerms:
    enabled: bool
    legacy_value: float
    leaf_delta_value: float
    pressure_delta_value: float
    role_fill_value: float
    late_conversion_value: float
    money_reserve_penalty: float
    slot_risk_penalty: float
    boss_risk_penalty: float
    total: float


def _calibrated_shop_action_value(
    state: GameState,
    action: Action,
    pressure: _ShopPressure,
    context: _ShopContext | None = None,
    plan: _RunPlan | None = None,
    *,
    profile: _BuildProfile | None = None,
) -> float:
    return _shop_planner_terms(state, action, pressure, context, plan, profile=profile).total


def _shop_planner_terms(
    state: GameState,
    action: Action,
    pressure: _ShopPressure,
    context: _ShopContext | None = None,
    plan: _RunPlan | None = None,
    *,
    profile: _BuildProfile | None = None,
    legacy_value: float | None = None,
) -> _ShopPlannerTerms:
    context = context or _ShopContext()
    profile = profile or _build_profile(state)
    plan = plan or _run_plan(state, profile, pressure)
    legacy = _shop_action_value(state, action, pressure, context, plan) if legacy_value is None else legacy_value
    cfg = active_config()
    if not cfg.calibrated_shop_planner_enabled or legacy <= 0.0 or state.ante < 5:
        return _legacy_only_planner_terms(enabled=cfg.calibrated_shop_planner_enabled, legacy=legacy)

    after_state = _simulated_after_state(state, action)
    leaf_delta = _leaf_delta_value(state, after_state)
    item = _shop_item_for_action(state, action)
    if action.action_type == ActionType.BUY and not _buying_voucher(action) and (item is None or not _is_joker_card(item)):
        leaf_delta = 0.0
    if _buying_voucher(action):
        if leaf_delta < 0.0 and (legacy < 20.0 or leaf_delta > -50.0):
            leaf_delta = 0.0
        elif leaf_delta > 0.0 and legacy < 10.0:
            leaf_delta = 0.0
    pressure_delta = _pressure_delta_value(state, after_state, pressure)
    if action.action_type == ActionType.BUY and not _buying_voucher(action) and (item is None or not _is_joker_card(item)):
        pressure_delta = 0.0
    role_fill = _role_fill_value(state, profile, after_state, pressure=pressure)
    late_conversion = _late_conversion_value(state, action, pressure, profile, after_state)
    money_reserve_penalty = _money_reserve_penalty(state, action, pressure, after_state)
    slot_risk_penalty = _slot_risk_penalty(state, action, profile, leaf_delta)
    boss_risk_penalty = _boss_risk_penalty(state, action, pressure, leaf_delta)
    total = (
        legacy * cfg.calibrated_shop_legacy_weight
        + leaf_delta * cfg.calibrated_shop_leaf_delta_weight
        + pressure_delta * cfg.calibrated_shop_pressure_delta_weight
        + role_fill * cfg.calibrated_shop_role_fill_weight
        + late_conversion * cfg.calibrated_shop_late_conversion_weight
        - money_reserve_penalty * cfg.calibrated_shop_reserve_risk_weight
        - slot_risk_penalty * cfg.calibrated_shop_slot_risk_weight
        - boss_risk_penalty * cfg.calibrated_shop_boss_risk_weight
    )
    return _ShopPlannerTerms(
        enabled=True,
        legacy_value=legacy,
        leaf_delta_value=leaf_delta,
        pressure_delta_value=pressure_delta,
        role_fill_value=role_fill,
        late_conversion_value=late_conversion,
        money_reserve_penalty=money_reserve_penalty,
        slot_risk_penalty=slot_risk_penalty,
        boss_risk_penalty=boss_risk_penalty,
        total=total,
    )


def _legacy_only_planner_terms(*, enabled: bool, legacy: float) -> _ShopPlannerTerms:
    return _ShopPlannerTerms(
        enabled=enabled,
        legacy_value=legacy,
        leaf_delta_value=0.0,
        pressure_delta_value=0.0,
        role_fill_value=0.0,
        late_conversion_value=0.0,
        money_reserve_penalty=0.0,
        slot_risk_penalty=0.0,
        boss_risk_penalty=0.0,
        total=legacy,
    )


def _shop_planner_terms_payload(terms: _ShopPlannerTerms) -> dict[str, object]:
    return {
        "enabled": terms.enabled,
        "legacy_value": round(terms.legacy_value, 2),
        "leaf_delta_value": round(terms.leaf_delta_value, 2),
        "pressure_delta_value": round(terms.pressure_delta_value, 2),
        "role_fill_value": round(terms.role_fill_value, 2),
        "late_conversion_value": round(terms.late_conversion_value, 2),
        "money_reserve_penalty": round(terms.money_reserve_penalty, 2),
        "slot_risk_penalty": round(terms.slot_risk_penalty, 2),
        "boss_risk_penalty": round(terms.boss_risk_penalty, 2),
        "total": round(terms.total, 2),
    }


def _buying_voucher(action: Action) -> bool:
    return action.action_type == ActionType.BUY and str(action.metadata.get("kind", action.target_id or "")) == "voucher"


def _replacement_planner_adjustment(
    state: GameState,
    *,
    sell_action: Action,
    candidate_action: Action,
    pressure: _ShopPressure,
) -> float:
    return 0.0


def _simulated_after_state(state: GameState, action: Action) -> GameState | None:
    try:
        if action.action_type == ActionType.BUY:
            return simulate_buy(state, action)
        if action.action_type == ActionType.SELL:
            return simulate_sell(state, action)
        if action.action_type == ActionType.USE_CONSUMABLE:
            return simulate_use_consumable(state, action)
    except (ValueError, IndexError, TypeError, AttributeError):
        return None
    return None


def _leaf_delta_value(state: GameState, after_state: GameState | None) -> float:
    if after_state is None:
        return 0.0
    if state.ante < 5:
        return 0.0
    current = _repeatable_build_score(state, state.jokers)
    after = _repeatable_build_score(after_state, after_state.jokers)
    delta = after - current
    scale = 0.014 + min(0.026, max(0, state.ante - 1) * 0.004)
    return max(-55.0, min(70.0, delta * scale))


def _pressure_delta_value(
    state: GameState,
    after_state: GameState | None,
    pressure: _ShopPressure,
) -> float:
    if after_state is None or state.ante < 5:
        return 0.0

    try:
        after_pressure = _shop_pressure(after_state)
    except (ValueError, TypeError, AttributeError, ZeroDivisionError):
        return 0.0

    before = _pressure_score(pressure)
    after = _pressure_score(after_pressure)
    if before <= 0.0 or after <= 0.0:
        return 0.0

    delta = before - after
    value = delta * (10.0 if before < 1.2 else 14.0)
    if before >= 1.0 and after < 1.0:
        value += 10.0
    if before >= 1.5 and after < 1.25:
        value += 8.0
    if after > before:
        value *= 1.25
    return max(-35.0, min(42.0, value))


def _pressure_score(pressure: _ShopPressure) -> float:
    return max(float(pressure.ratio), float(pressure.raw_ratio))


def _role_fill_value(
    state: GameState,
    profile: _BuildProfile,
    after_state: GameState | None,
    *,
    pressure: _ShopPressure,
) -> float:
    if after_state is None:
        return 0.0
    if state.ante < 5:
        return 0.0
    after_profile = _build_profile(after_state)
    before_missing = set(profile.missing_roles)
    after_missing = set(after_profile.missing_roles)
    value = 0.0
    for role in sorted((before_missing - after_missing) & CRITICAL_BUILD_ROLES):
        bonus = 10.0 + profile.role_deficit_ratio(role) * 12.0
        if state.ante >= 5 and role in {"xmult", "scaling"}:
            bonus += 8.0
        value += bonus
    for role in sorted((after_missing - before_missing) & CRITICAL_BUILD_ROLES):
        penalty = 12.0
        if role in {"xmult", "scaling"}:
            penalty += 10.0
        value -= penalty
    return max(-42.0, min(46.0, value))


def _late_conversion_value(
    state: GameState,
    action: Action,
    pressure: _ShopPressure,
    profile: _BuildProfile,
    after_state: GameState | None,
) -> float:
    if state.ante < 5:
        return 0.0
    missing = set(profile.missing_roles) & {"xmult", "scaling"}
    if not missing:
        return 0.0
    spendable = _spendable_money(state, pressure)
    if spendable < 12:
        return 0.0

    base = min(24.0, 6.0 + spendable * 0.28)
    if pressure.ratio >= 1.0 or pressure.raw_ratio >= 0.85:
        base += 5.0
    if pressure.ratio >= 1.75 or pressure.raw_ratio >= 1.2:
        base += 4.0

    item = _shop_item_for_action(state, action)
    if action.action_type == ActionType.BUY and item is not None and _is_joker_card(item):
        roles = _joker_roles(_joker_from_shop_card(item))
        if roles & missing:
            return base
        if roles & CRITICAL_BUILD_ROLES and after_state is not None:
            return base * 0.35
    if action.action_type == ActionType.REROLL:
        if profile.open_joker_slots <= 0:
            return 0.0
        return base * 0.65
    if action.action_type == ActionType.OPEN_PACK and item is not None:
        label = _card_label(item).lower()
        if _is_buffoon_pack(item):
            return base * 0.55
        if "celestial" in label:
            return base * 0.20
        if "arcana" in label or "spectral" in label:
            return base * 0.15
    return 0.0


def _money_reserve_penalty(
    state: GameState,
    action: Action,
    pressure: _ShopPressure,
    after_state: GameState | None,
) -> float:
    cost = _shop_action_cost(state, action)
    if cost <= 0 and after_state is None:
        return 0.0
    if state.ante < 5 and not _has_money_scaling_joker(state):
        return 0.0
    after_money = after_state.money if after_state is not None else state.money - cost
    return _post_state_reserve_penalty(state, after_state or state, pressure, after_money=after_money)


def _post_state_reserve_penalty(
    state: GameState,
    after_state: GameState,
    pressure: _ShopPressure,
    *,
    after_money: int | None = None,
) -> float:
    after_money = after_state.money if after_money is None else after_money
    reserve = _desired_money_reserve(state, pressure)
    before_shortfall = max(0, reserve - state.money)
    after_shortfall = max(0, reserve - after_money)
    new_shortfall = max(0, after_shortfall - before_shortfall)
    if new_shortfall <= 0:
        return 0.0
    weight = 0.85
    if _has_money_scaling_joker(state):
        weight += 0.85
    if state.ante >= 5 and pressure.ratio < 1.0 and pressure.raw_ratio < 0.95:
        weight += 0.45
    penalty = new_shortfall * weight
    if after_money < 4 and state.ante >= 2:
        penalty += 8.0
    return min(45.0, penalty)


def _slot_risk_penalty(
    state: GameState,
    action: Action,
    profile: _BuildProfile,
    leaf_delta_value: float,
) -> float:
    item = _shop_item_for_action(state, action)
    if action.action_type != ActionType.BUY or item is None or not _normal_slot_joker_card(item):
        return 0.0
    if state.ante < 5:
        return 0.0
    if _normal_joker_open_slots(state) > 1:
        return 0.0
    missing = set(profile.missing_roles)
    roles = _joker_roles(_joker_from_shop_card(item))
    if roles & (missing & CRITICAL_BUILD_ROLES):
        return 0.0

    penalty = 0.0
    if state.ante >= 4 and missing & {"xmult", "scaling"} and roles.isdisjoint({"xmult", "scaling"}):
        penalty += 16.0
    if leaf_delta_value <= 0.0:
        penalty += 8.0
    return min(30.0, penalty)


def _boss_risk_penalty(
    state: GameState,
    action: Action,
    pressure: _ShopPressure,
    leaf_delta_value: float,
) -> float:
    if pressure.boss_name not in DANGEROUS_BOSS_BLINDS and pressure.boss_name not in FINAL_BOSS_BLINDS:
        if pressure.ratio < 1.05 and pressure.raw_ratio < 1.0:
            return 0.0
    if action.action_type not in {ActionType.BUY, ActionType.OPEN_PACK, ActionType.REROLL}:
        return 0.0
    if leaf_delta_value >= 0.0:
        return 0.0
    cost = max(0, _shop_action_cost(state, action))
    return min(18.0, pressure.danger * 5.0 + max(0.0, -leaf_delta_value) * 0.25 + cost * 0.35)
