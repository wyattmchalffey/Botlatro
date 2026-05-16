"""Shop item lookup and audit payload helpers.

This module only knows how to identify visible shop/pack items and describe
actions/items for traces. It deliberately does not score shop decisions.
"""

from __future__ import annotations

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GameState
from balatro_ai.bots.basic_strategy.actions import _first_action_of_type
from balatro_ai.bots.basic_strategy.cards import _card_cost, _card_label, _card_set


def _has_shop_decision_surface(state: GameState) -> bool:
    return bool(
        state.modifiers.get("shop_cards")
        or state.modifiers.get("voucher_cards")
        or state.modifiers.get("booster_packs")
        or _first_action_of_type(state, ActionType.REROLL)
    )


def _shop_option_payload(state: GameState, action: Action, value: float) -> dict[str, object]:
    payload = _action_payload(action)
    payload["value"] = round(value, 2)
    item = _item_payload_for_action(state, action)
    if item is not None:
        payload["item"] = item
    return payload


def _action_payload(action: Action) -> dict[str, object]:
    return {
        "type": action.action_type.value,
        "kind": str(action.metadata.get("kind", action.target_id or "")),
        "index": action.metadata.get("index", action.amount),
        "target_id": action.target_id,
        "amount": action.amount,
        "stable_key": action.stable_key,
    }


def _item_payload_for_action(state: GameState, action: Action) -> dict[str, object] | None:
    item = _shop_item_for_action(state, action)
    if item is None:
        return None
    return _shop_item_payload(item)


def _shop_item_for_action(state: GameState, action: Action) -> object | None:
    kind = str(action.metadata.get("kind", action.target_id or ""))
    index = action.metadata.get("index", action.amount)
    try:
        item_index = int(index)
    except (TypeError, ValueError):
        item_index = -1
    if action.action_type == ActionType.BUY and kind == "voucher":
        return _indexed_shop_item(state.modifiers.get("voucher_cards", ()), item_index)
    if action.action_type == ActionType.BUY and kind == "card":
        return _indexed_shop_item(state.modifiers.get("shop_cards", ()), item_index)
    if action.action_type == ActionType.OPEN_PACK:
        return _indexed_shop_item(state.modifiers.get("booster_packs", ()), item_index)
    if action.action_type == ActionType.CHOOSE_PACK_CARD:
        return _indexed_shop_item(state.modifiers.get("pack_cards", ()), item_index)
    return None


def _indexed_shop_item(items: tuple[object, ...], index: int) -> object | None:
    if 0 <= index < len(items):
        return items[index]
    return None


def _shop_item_payload(item: object) -> dict[str, object]:
    if not isinstance(item, dict):
        return {"name": str(item)}
    return {
        "name": _card_label(item),
        "set": _card_set(item),
        "cost": _card_cost(item),
        "key": item.get("key"),
        "rarity": item.get("rarity"),
    }
