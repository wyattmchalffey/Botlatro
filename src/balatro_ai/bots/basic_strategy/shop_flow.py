"""Top-level shop action orchestration and audit payloads."""

from __future__ import annotations

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GameState
from balatro_ai.bots.basic_strategy.actions import _annotated_action, _first_action_of_type
from balatro_ai.bots.basic_strategy.build_profile import _build_profile
from balatro_ai.bots.basic_strategy.build_scoring import _normal_slot_joker_card
from balatro_ai.bots.basic_strategy.cards import (
    _card_cost,
    _card_label,
    _is_joker_card,
    _joker_from_shop_card,
    _normal_joker_open_slots,
)
from balatro_ai.bots.basic_strategy.profile import (
    _BuildProfile,
    _ShopContext,
    _ShopPressure,
    _build_profile_payload,
    _pressure_payload,
)
from balatro_ai.bots.basic_strategy.shop_items import (
    _action_payload,
    _has_shop_decision_surface,
    _item_payload_for_action,
    _shop_item_for_action,
    _shop_option_payload,
)
from balatro_ai.bots.basic_strategy.shop_jokers import (
    _candidate_joker_value_for_replacement,
    _owned_joker_value,
    _replacement_cost_weight,
    _replacement_role_upgrade_bonus,
    _replacement_upgrade_threshold,
)
from balatro_ai.bots.basic_strategy.shop_money import _money_plan_payload, _spendable_money
from balatro_ai.bots.basic_strategy.shop_pressure import _shop_pressure
from balatro_ai.bots.basic_strategy.shop_reroll import _pressure_spend_mode, _pressure_spend_reserve_slack
from balatro_ai.bots.basic_strategy.shop_values import (
    _shop_action_cost,
    _shop_action_reveals_information_before_joker_buy,
    _shop_action_value,
    _shop_buy_threshold,
    _shop_information_action_can_take_joker_slot,
)
from balatro_ai.bots.config import active_config


_SHOP_POLICY_ACTION_TYPES = frozenset(
    {
        ActionType.BUY,
        ActionType.SELL,
        ActionType.REROLL,
        ActionType.OPEN_PACK,
        ActionType.END_SHOP,
        ActionType.USE_CONSUMABLE,
    }
)


def _has_shop_policy_action(state: GameState) -> bool:
    return any(action.action_type in _SHOP_POLICY_ACTION_TYPES for action in state.legal_actions)


def _shop_action(
    state: GameState,
    context: _ShopContext | None = None,
    *,
    pressure: _ShopPressure | None = None,
    profile: _BuildProfile | None = None,
) -> Action | None:
    context = context or _ShopContext()
    pressure = pressure or _shop_pressure(state)
    profile = profile or _build_profile(state)
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
    if best_action is not None and best_value + active_config().shop_value_tolerance >= threshold:
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
                    profile=profile,
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
                profile=profile,
            ),
        )
    forced_action = _pressure_forced_shop_action(
        state,
        pressure,
        context,
        profile=profile,
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
                profile=profile,
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
                profile=profile,
            ),
        )
    return None


def _pressure_forced_shop_action(
    state: GameState,
    pressure: _ShopPressure,
    context: _ShopContext,
    *,
    profile: _BuildProfile,
    best_action: Action | None,
    best_value: float,
    threshold: float,
) -> tuple[Action, float] | None:
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
    if best_value + active_config().shop_value_tolerance < minimum_value:
        return None
    if best_value + active_config().shop_value_tolerance >= threshold:
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
        if value + active_config().shop_value_tolerance < threshold:
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


def _shop_decision_audit(
    state: GameState,
    pressure: _ShopPressure,
    *,
    chosen_action: Action,
    chosen_value: float,
    threshold: float,
    decision: str,
    context: _ShopContext | None = None,
    profile: _BuildProfile | None = None,
) -> dict[str, object]:
    context = context or _ShopContext()
    profile = profile or _build_profile(state)
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
        "build_profile": _build_profile_payload(profile),
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
