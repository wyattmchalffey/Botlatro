"""Voucher valuation and pressure-aware voucher gating."""

from __future__ import annotations

from balatro_ai.api.state import GameState
from balatro_ai.bots.basic_strategy.build_profile import _build_profile
from balatro_ai.bots.basic_strategy.cards import _card_label, _has_consumable_room
from balatro_ai.bots.basic_strategy.data import (
    DANGEROUS_BOSS_BLINDS,
    PLANET_TO_HAND,
    RARE_HAND_TYPES,
    VOUCHER_BUY_DENYLIST,
    VOUCHER_IMMEDIATE_SCORE_NAMES,
    VOUCHER_INTEREST_CAP_MONEY,
    VOUCHER_PRESSURE_ALLOWED_NAMES,
    VOUCHER_VALUES,
)
from balatro_ai.bots.basic_strategy.profile import _BuildProfile, _ShopPressure
from balatro_ai.bots.basic_strategy.rare_hands import _rare_hand_plan
from balatro_ai.bots.basic_strategy.shop_forecast import (
    _has_planet_investment,
    _shop_cleared_blind_kind,
    _shop_pressure_uses_exact_needle_hand,
)
from balatro_ai.bots.basic_strategy.shop_money import _interest_cap_money, _spendable_money
from balatro_ai.bots.basic_strategy.shop_pressure import _shop_pressure
from balatro_ai.bots.basic_strategy.shop_safety import _has_money_scaling_joker
from balatro_ai.rules.hand_evaluator import HandType


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
