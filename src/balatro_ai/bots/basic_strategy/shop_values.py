"""Shop action, card, pack, and threshold valuation glue."""

from __future__ import annotations

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GameState
from balatro_ai.bots.basic_strategy.build_profile import _build_profile, _reroll_role_hunt_bonus, _urgent_late_role_hunt
from balatro_ai.bots.basic_strategy.build_scoring import _buy_would_overfill_joker_slots, _normal_slot_joker_card
from balatro_ai.bots.basic_strategy.cards import (
    _card_cost,
    _card_label,
    _card_modifier,
    _has_consumable_room,
    _is_black_hole_card,
    _is_joker_card,
    _is_planet_card,
    _is_playing_card,
    _is_spectral_card,
    _is_tarot_card,
    _joker_from_shop_card,
    _normal_joker_open_slots,
    _normal_joker_slots_used,
    _pack_card_requires_targets,
)
from balatro_ai.bots.basic_strategy.data import SPECTRAL_SEAL_VALUES
from balatro_ai.bots.basic_strategy.jokers import _active_joker_names
from balatro_ai.bots.basic_strategy.pack_targets import _target_required_tarot_is_supported
from balatro_ai.bots.basic_strategy.profile import _BuildProfile, _ShopContext, _ShopPressure
from balatro_ai.bots.basic_strategy.shop_cards import (
    _black_hole_card_value,
    _planet_card_value,
    _playing_card_shop_value,
    _tarot_card_value,
)
from balatro_ai.bots.basic_strategy.shop_forecast import _has_planet_investment
from balatro_ai.bots.basic_strategy.shop_items import _shop_item_for_action
from balatro_ai.bots.basic_strategy.shop_jokers import (
    _future_score_headroom_joker_bonus,
    _joker_card_value,
    _pressure_joker_role_bonus,
)
from balatro_ai.bots.basic_strategy.shop_money import (
    _cost_penalty,
    _interest_cap_money,
    _money_after_spend_penalty,
    _money_gain_value,
    _spendable_money,
)
from balatro_ai.bots.basic_strategy.shop_packs import _is_buffoon_pack, _late_pack_is_worth_opening, _pack_value
from balatro_ai.bots.basic_strategy.shop_reroll import (
    _early_reroll_is_allowed,
    _late_bank_conversion_mode,
    _late_reroll_is_worth_it,
    _minimum_reroll_bank,
    _reroll_cost_escalation_penalty,
    _rich_late_role_hunt,
)
from balatro_ai.bots.basic_strategy.shop_safety import _early_shop_safety_adjustment, _has_money_scaling_joker
from balatro_ai.bots.basic_strategy.shop_vouchers import _voucher_value
from balatro_ai.bots.basic_strategy.utils import _int_or_default


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
