"""Reroll and late shop spend policy."""

from __future__ import annotations

import sys

from balatro_ai.api.actions import ActionType
from balatro_ai.api.state import GameState
from balatro_ai.bots.basic_strategy.cards import (
    _card_cost,
    _card_label,
    _is_joker_card,
    _joker_from_shop_card,
    _normal_joker_open_slots,
)
from balatro_ai.bots.basic_strategy.data import (
    FINAL_BOSS_BLINDS,
    JOKER_ECONOMY_VALUES,
    LOW_PRIORITY_JOKERS,
    MONEY_SCALING_RESERVE_TARGETS,
)
from balatro_ai.bots.basic_strategy.jokers import _joker_roles
from balatro_ai.bots.basic_strategy.profile import CRITICAL_BUILD_ROLES, _BuildProfile, _ShopContext, _ShopPressure
from balatro_ai.bots.basic_strategy.shop_forecast import _shop_has_followup_big_blind_shop
from balatro_ai.bots.basic_strategy.shop_money import (
    _desired_money_reserve,
    _interest_cap_money,
    _late_closing_money_reserve_cap,
    _late_pressure_interest_floor,
    _spendable_money,
)
from balatro_ai.bots.basic_strategy.shop_safety import _has_money_scaling_joker, _has_real_scoring_joker
from balatro_ai.bots.config import active_config


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
            raised_cap_under_severe_pressure = (
                pressure is not None
                and state.ante == 4
                and _interest_cap_money(state) > 25
                and (pressure.ratio >= 3.0 or pressure.raw_ratio >= 1.75)
            )
            if closing_cap is None and not raised_cap_under_severe_pressure:
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
    return visible_value + active_config().shop_value_tolerance < threshold


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
    if state.ante >= 6 and spendable >= 30 and pressure.ratio >= 1.25 and pressure.raw_ratio >= 0.70:
        return True
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


def _rich_late_role_hunt(profile: _BuildProfile) -> bool:
    return profile.rich and profile.late and ("xmult" in profile.missing_roles or "scaling" in profile.missing_roles)


def _build_profile(*args, **kwargs) -> _BuildProfile:
    """Honor legacy tests/tools that patch ``basic_strategy_bot._build_profile``."""

    strategy_module = sys.modules.get("balatro_ai.bots.basic_strategy_bot")
    patched = getattr(strategy_module, "_build_profile", None) if strategy_module is not None else None
    if patched is None or patched is _build_profile:
        raise RuntimeError("basic_strategy_bot._build_profile is not available")
    return patched(*args, **kwargs)


def _shop_action_value(*args, **kwargs) -> float:
    """Honor legacy tests/tools that patch ``basic_strategy_bot._shop_action_value``."""

    strategy_module = sys.modules.get("balatro_ai.bots.basic_strategy_bot")
    patched = getattr(strategy_module, "_shop_action_value", None) if strategy_module is not None else None
    if patched is None or patched is _shop_action_value:
        raise RuntimeError("basic_strategy_bot._shop_action_value is not available")
    return patched(*args, **kwargs)


def _shop_buy_threshold(*args, **kwargs) -> float:
    """Honor legacy tests/tools that patch ``basic_strategy_bot._shop_buy_threshold``."""

    strategy_module = sys.modules.get("balatro_ai.bots.basic_strategy_bot")
    patched = getattr(strategy_module, "_shop_buy_threshold", None) if strategy_module is not None else None
    if patched is None or patched is _shop_buy_threshold:
        raise RuntimeError("basic_strategy_bot._shop_buy_threshold is not available")
    return patched(*args, **kwargs)


def _shop_reroll_cost(*args, **kwargs) -> int:
    """Honor legacy tests/tools that patch ``basic_strategy_bot._shop_reroll_cost``."""

    strategy_module = sys.modules.get("balatro_ai.bots.basic_strategy_bot")
    patched = getattr(strategy_module, "_shop_reroll_cost", None) if strategy_module is not None else None
    if patched is None or patched is _shop_reroll_cost:
        raise RuntimeError("basic_strategy_bot._shop_reroll_cost is not available")
    return patched(*args, **kwargs)


def _urgent_late_role_hunt(*args, **kwargs) -> bool:
    """Honor legacy tests/tools that patch ``basic_strategy_bot._urgent_late_role_hunt``."""

    strategy_module = sys.modules.get("balatro_ai.bots.basic_strategy_bot")
    patched = getattr(strategy_module, "_urgent_late_role_hunt", None) if strategy_module is not None else None
    if patched is None or patched is _urgent_late_role_hunt:
        raise RuntimeError("basic_strategy_bot._urgent_late_role_hunt is not available")
    return patched(*args, **kwargs)
