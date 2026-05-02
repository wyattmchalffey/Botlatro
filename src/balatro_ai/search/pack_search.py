"""Pack-pick search for deterministic booster choices."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GameState
from balatro_ai.search.forward_sim import simulate_choose_pack_card, simulate_sell
from balatro_ai.search.state_value import state_value

ValueFn = Callable[[GameState], float]


@dataclass(frozen=True, slots=True)
class PackSearchConfig:
    leaf_samples: int = 16
    seed: int = 0


def best_pack_action(
    state: GameState,
    *,
    config: PackSearchConfig | None = None,
    value_fn: ValueFn | None = None,
) -> Action | None:
    """Return the deterministic pack choice with the highest leaf value."""

    choose_actions = tuple(action for action in state.legal_actions if action.action_type == ActionType.CHOOSE_PACK_CARD)
    if not choose_actions or not _pack_choices_are_deterministic(state, choose_actions):
        return None

    actions = tuple(action for action in choose_actions if _pack_action_is_legal(state, action))
    search_config = config or PackSearchConfig()
    evaluator = value_fn or _default_value_fn(search_config)
    include_item_value = value_fn is None
    best_action: Action | None = None
    best_value = float("-inf")
    for action in actions:
        value = pack_action_value(state, action, value_fn=evaluator, include_item_value=include_item_value)
        if value > best_value:
            best_action = action
            best_value = value
    sell_action, sell_value = _best_sell_then_pick_action(
        state,
        choose_actions,
        value_fn=evaluator,
        include_item_value=include_item_value,
    )
    if sell_action is not None and sell_value > best_value:
        return sell_action
    if best_action is None:
        return None
    return _annotated_action(best_action, search_value=best_value)


def pack_action_value(
    state: GameState,
    action: Action,
    *,
    value_fn: ValueFn | None = None,
    include_item_value: bool | None = None,
) -> float:
    """Evaluate one deterministic pack choice."""

    if action.action_type != ActionType.CHOOSE_PACK_CARD:
        raise ValueError(f"pack_action_value requires choose_pack_card, got {action.action_type.value}")
    should_include_item_value = value_fn is None if include_item_value is None else include_item_value
    evaluator = value_fn or (lambda leaf: state_value(leaf))
    simulated = simulate_choose_pack_card(state, action)
    value = evaluator(simulated)
    if should_include_item_value:
        value += _pack_item_value(state, action)
    return value


def _best_sell_then_pick_action(
    state: GameState,
    choose_actions: tuple[Action, ...],
    *,
    value_fn: ValueFn,
    include_item_value: bool,
) -> tuple[Action | None, float]:
    sell_actions = tuple(action for action in state.legal_actions if action.action_type == ActionType.SELL)
    if not sell_actions or not choose_actions:
        return None, float("-inf")

    best_action: Action | None = None
    best_value = float("-inf")
    for sell_action in sell_actions:
        sold_index = _action_index(sell_action)
        if sold_index is None or not 0 <= sold_index < len(state.jokers):
            continue
        try:
            sold_state = simulate_sell(state, sell_action)
        except (ValueError, IndexError, TypeError):
            continue
        sold_penalty = _owned_joker_value(state, sold_index)
        for choose_action in choose_actions:
            if _is_skip_action(choose_action):
                continue
            pack_cards = _modifier_items(state.modifiers, "pack_cards")
            pick_index = _action_index(choose_action)
            if pick_index is None or not 0 <= pick_index < len(pack_cards):
                continue
            if _item_set(pack_cards[pick_index]) != "JOKER":
                continue
            if _pack_action_is_legal(state, choose_action):
                continue
            if not _pack_action_is_legal(sold_state, choose_action):
                continue
            try:
                value = pack_action_value(
                    sold_state,
                    choose_action,
                    value_fn=value_fn,
                    include_item_value=include_item_value,
                ) - sold_penalty
            except (ValueError, IndexError, TypeError):
                continue
            if value > best_value:
                best_action = _annotated_sequence_sell_action(sell_action, choose_action, value)
                best_value = value
    return best_action, best_value


def _pack_choices_are_deterministic(state: GameState, actions: tuple[Action, ...]) -> bool:
    pack_cards = _modifier_items(state.modifiers, "pack_cards")
    if not pack_cards:
        return False
    for action in actions:
        if _is_skip_action(action):
            continue
        index = _action_index(action)
        if index is None or not 0 <= index < len(pack_cards):
            return False
        if not _is_deterministic_pack_item(pack_cards[index]):
            return False
    return True


def _pack_action_is_legal(state: GameState, action: Action) -> bool:
    if _is_skip_action(action):
        return True
    pack_cards = _modifier_items(state.modifiers, "pack_cards")
    index = _action_index(action)
    if index is None or not 0 <= index < len(pack_cards):
        return False
    item = pack_cards[index]
    if _item_set(item) == "JOKER" and _joker_item_uses_normal_slot(item):
        return _normal_joker_open_slots(state) > 0
    return True


def _is_deterministic_pack_item(item: object) -> bool:
    item_set = _item_set(item)
    if item_set in {"JOKER", "PLANET", "DEFAULT", "ENHANCED"}:
        return True
    key = _item_key(item)
    return _playing_card_key(key)


def _pack_item_value(state: GameState, action: Action) -> float:
    if _is_skip_action(action):
        return 0.0
    pack_cards = _modifier_items(state.modifiers, "pack_cards")
    index = _action_index(action)
    if index is None or not 0 <= index < len(pack_cards):
        return 0.0
    try:
        from balatro_ai.bots.basic_strategy_bot import _pack_card_value

        return _pack_card_value(state, pack_cards[index])
    except (ImportError, TypeError, ValueError, AttributeError):
        return 0.0


def _owned_joker_value(state: GameState, index: int) -> float:
    if not 0 <= index < len(state.jokers):
        return 0.0
    try:
        from balatro_ai.bots.basic_strategy_bot import _owned_joker_value as basic_owned_joker_value

        return basic_owned_joker_value(state, state.jokers[index], remove_index=index)
    except (ImportError, TypeError, ValueError, AttributeError):
        return float(state.jokers[index].sell_value or 0)


def _is_skip_action(action: Action) -> bool:
    return action.target_id == "skip" or str(action.metadata.get("kind") or "") == "skip"


def _modifier_items(modifiers: dict[str, object], key: str) -> tuple[object, ...]:
    raw = modifiers.get(key, ())
    if isinstance(raw, dict):
        raw = raw.get("cards", ())
    if isinstance(raw, list | tuple):
        return tuple(raw)
    return ()


def _action_index(action: Action) -> int | None:
    raw = action.metadata.get("index", action.amount)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _item_set(item: object) -> str:
    if not isinstance(item, dict):
        return ""
    return str(item.get("set", "")).upper()


def _item_key(item: object) -> str:
    if not isinstance(item, dict):
        return ""
    return str(item.get("key", ""))


def _item_edition(item: object) -> str:
    if not isinstance(item, dict):
        return ""
    edition = item.get("edition")
    modifier = item.get("modifier")
    if edition is None and isinstance(modifier, dict):
        edition = modifier.get("edition")
    if isinstance(edition, dict):
        edition = edition.get("type") or edition.get("key") or edition.get("name") or edition.get("edition")
    return str(edition or "")


def _joker_item_uses_normal_slot(item: object) -> bool:
    return "negative" not in _item_edition(item).lower()


def _normal_joker_open_slots(state: GameState) -> int:
    return max(0, _normal_joker_slot_limit(state) - _normal_joker_slots_used(state))


def _normal_joker_slots_used(state: GameState) -> int:
    return sum(1 for joker in state.jokers if "negative" not in (joker.edition or "").lower())


def _normal_joker_slot_limit(state: GameState) -> int:
    for key in ("joker_slot_limit", "joker_slots"):
        raw = state.modifiers.get(key)
        try:
            if raw is not None:
                return max(0, int(raw))
        except (TypeError, ValueError):
            continue
    return 5


def _playing_card_key(key: str) -> bool:
    parts = key.split("_", maxsplit=1)
    return len(parts) == 2 and parts[0] in {"S", "H", "C", "D"} and parts[1] in {
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "T",
        "J",
        "Q",
        "K",
        "A",
    }


def _default_value_fn(config: PackSearchConfig) -> ValueFn:
    def evaluate(state: GameState) -> float:
        return state_value(state, samples=config.leaf_samples, seed=config.seed)

    return evaluate


def _annotated_action(action: Action, *, search_value: float, sequence: object = None) -> Action:
    metadata = {**action.metadata, "search_value": round(search_value, 6), "search": "pack_value"}
    if sequence is not None:
        metadata["search_sequence"] = sequence
    return Action(
        action.action_type,
        card_indices=action.card_indices,
        target_id=action.target_id,
        amount=action.amount,
        metadata=metadata,
    )


def _annotated_sequence_sell_action(sell_action: Action, choose_action: Action, search_value: float) -> Action:
    return Action(
        sell_action.action_type,
        card_indices=sell_action.card_indices,
        target_id=sell_action.target_id,
        amount=sell_action.amount,
        metadata={
            **sell_action.metadata,
            "search": "pack_value",
            "search_value": round(search_value, 6),
            "search_sequence": (
                _action_summary(sell_action),
                _action_summary(choose_action),
            ),
        },
    )


def _action_summary(action: Action) -> dict[str, object]:
    return {
        "type": action.action_type.value,
        "kind": str(action.metadata.get("kind", action.target_id or "")),
        "index": action.metadata.get("index", action.amount),
    }
