"""Shop money, reserve, and spend-penalty policy."""

from __future__ import annotations

import sys

from balatro_ai.api.state import GameState, Joker
from balatro_ai.bots.basic_strategy.cards import _card_cost, _is_joker_card, _joker_from_shop_card
from balatro_ai.bots.basic_strategy.data import (
    JOKER_ECONOMY_VALUES,
    MONEY_SCALING_RESERVE_TARGETS,
    VOUCHER_INTEREST_CAP_MONEY,
)
from balatro_ai.bots.basic_strategy.profile import _BuildProfile, _ShopPressure
from balatro_ai.bots.basic_strategy.shop_safety import _has_money_scaling_joker, _has_real_scoring_joker


BASE_INTEREST_CAP_MONEY = 25


def _money_gain_value(state: GameState, amount: int) -> float:
    if amount <= 0:
        return 0.0
    value = float(amount)
    interest_gap = max(0, _interest_cap_money(state) - state.money)
    value += min(amount, interest_gap) * (0.45 if state.ante <= 2 else 0.25)
    if _has_money_scaling_joker(state):
        value += amount * 0.35
    return value


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


def _money_plan_payload(state: GameState, pressure: _ShopPressure | None = None) -> dict[str, int | bool]:
    return {
        "interest_cap_money": _interest_cap_money(state),
        "reserve_money": _desired_money_reserve(state, pressure),
        "spendable_money": _spendable_money(state, pressure),
        "has_money_scaling_joker": _has_money_scaling_joker(state),
    }


def _build_profile(*args, **kwargs) -> _BuildProfile:
    """Honor legacy tests/tools that patch ``basic_strategy_bot._build_profile``."""

    strategy_module = sys.modules.get("balatro_ai.bots.basic_strategy_bot")
    patched = getattr(strategy_module, "_build_profile", None) if strategy_module is not None else None
    if patched is None or patched is _build_profile:
        raise RuntimeError("basic_strategy_bot._build_profile is not available")
    return patched(*args, **kwargs)


def _late_bank_conversion_reserve_cap(*args, **kwargs):
    """Honor legacy tests/tools that patch ``basic_strategy_bot._late_bank_conversion_reserve_cap``."""

    strategy_module = sys.modules.get("balatro_ai.bots.basic_strategy_bot")
    patched = getattr(strategy_module, "_late_bank_conversion_reserve_cap", None) if strategy_module is not None else None
    if patched is None or patched is _late_bank_conversion_reserve_cap:
        raise RuntimeError("basic_strategy_bot._late_bank_conversion_reserve_cap is not available")
    return patched(*args, **kwargs)


def _urgent_late_role_hunt(*args, **kwargs) -> bool:
    """Honor legacy tests/tools that patch ``basic_strategy_bot._urgent_late_role_hunt``."""

    strategy_module = sys.modules.get("balatro_ai.bots.basic_strategy_bot")
    patched = getattr(strategy_module, "_urgent_late_role_hunt", None) if strategy_module is not None else None
    if patched is None or patched is _urgent_late_role_hunt:
        raise RuntimeError("basic_strategy_bot._urgent_late_role_hunt is not available")
    return patched(*args, **kwargs)
