"""Pack-pick search for booster choices."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from random import Random

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GameState
from balatro_ai.search.consumable_search import (
    consumable_action_is_supported,
    sample_consumable_injections,
    simulate_consumable_action,
)
from balatro_ai.search.forward_sim import (
    PLANET_TO_HAND,
    SPECTRAL_NAMES,
    TAROT_ENHANCEMENTS,
    TAROT_NAMES,
    simulate_choose_pack_card,
    simulate_sell,
)
from balatro_ai.search.shop_sampler import ShopSampler
from balatro_ai.search.state_value import state_value

ValueFn = Callable[[GameState], float]


@dataclass(frozen=True, slots=True)
class PackSearchConfig:
    leaf_samples: int = 16
    seed: int = 0
    stochastic_samples: int = 8


def best_pack_action(
    state: GameState,
    *,
    config: PackSearchConfig | None = None,
    value_fn: ValueFn | None = None,
    sampler: ShopSampler | None = None,
) -> Action | None:
    """Return the pack choice with the highest modeled leaf value."""

    choose_actions = tuple(action for action in state.legal_actions if action.action_type == ActionType.CHOOSE_PACK_CARD)
    if not choose_actions:
        return None

    search_config = config or PackSearchConfig()
    shop_sampler = sampler or ShopSampler.from_default_data()
    actions = tuple(action for action in choose_actions if _pack_action_is_legal(state, action) and _pack_action_is_searchable(state, action))
    if not actions:
        return None
    evaluator = value_fn or _default_value_fn(search_config)
    include_item_value = value_fn is None
    best_action: Action | None = None
    best_value = float("-inf")
    for action in actions:
        value = pack_action_value(
            state,
            action,
            config=search_config,
            value_fn=evaluator,
            include_item_value=include_item_value,
            sampler=shop_sampler,
        )
        if value > best_value:
            best_action = action
            best_value = value
    sell_action, sell_value = _best_sell_then_pick_action(
        state,
        choose_actions,
        config=search_config,
        value_fn=evaluator,
        include_item_value=include_item_value,
        sampler=shop_sampler,
    )
    if sell_action is not None and sell_value > best_value:
        return sell_action
    use_action, use_value = _best_use_then_pick_action(
        state,
        choose_actions,
        config=search_config,
        value_fn=evaluator,
        include_item_value=include_item_value,
        sampler=shop_sampler,
    )
    if use_action is not None and use_value > best_value:
        return use_action
    if best_action is None:
        return None
    return _annotated_action(best_action, search_value=best_value)


def pack_action_value(
    state: GameState,
    action: Action,
    *,
    config: PackSearchConfig | None = None,
    value_fn: ValueFn | None = None,
    include_item_value: bool | None = None,
    sampler: ShopSampler | None = None,
) -> float:
    """Evaluate one pack choice."""

    if action.action_type != ActionType.CHOOSE_PACK_CARD:
        raise ValueError(f"pack_action_value requires choose_pack_card, got {action.action_type.value}")
    search_config = config or PackSearchConfig()
    should_include_item_value = value_fn is None if include_item_value is None else include_item_value
    evaluator = value_fn or (lambda leaf: state_value(leaf))
    outcomes = _pack_choice_outcomes(state, action, config=search_config, sampler=sampler)
    if not outcomes:
        raise ValueError("No modeled pack-choice outcomes")
    value = sum(evaluator(simulated) for simulated in outcomes) / len(outcomes)
    if should_include_item_value:
        value += _pack_item_value(state, action)
    return value


def _best_sell_then_pick_action(
    state: GameState,
    choose_actions: tuple[Action, ...],
    *,
    config: PackSearchConfig,
    value_fn: ValueFn,
    include_item_value: bool,
    sampler: ShopSampler,
) -> tuple[Action | None, float]:
    sell_actions = tuple(action for action in state.legal_actions if action.action_type == ActionType.SELL)
    if not sell_actions or not choose_actions:
        return None, float("-inf")

    best_action: Action | None = None
    best_value = float("-inf")
    for sell_action in sell_actions:
        if not _pack_sell_action_is_candidate(state, sell_action, choose_actions):
            continue
        try:
            sold_state = simulate_sell(state, sell_action)
        except (ValueError, IndexError, TypeError):
            continue
        sold_penalty = _sell_penalty(state, sell_action)
        for choose_action in choose_actions:
            if _is_skip_action(choose_action):
                continue
            pack_cards = _modifier_items(state.modifiers, "pack_cards")
            pick_index = _action_index(choose_action)
            if pick_index is None or not 0 <= pick_index < len(pack_cards):
                continue
            if not _pack_action_is_searchable(sold_state, choose_action):
                continue
            if _pack_action_is_legal(state, choose_action) and not _pack_sell_can_improve_pick(state, sell_action, choose_action):
                continue
            if not _pack_action_is_legal(sold_state, choose_action):
                continue
            try:
                value = pack_action_value(
                    sold_state,
                    choose_action,
                    config=config,
                    value_fn=value_fn,
                    include_item_value=include_item_value,
                    sampler=sampler,
                ) - sold_penalty
            except (ValueError, IndexError, TypeError):
                continue
            if value > best_value:
                best_action = _annotated_sequence_sell_action(sell_action, choose_action, value)
                best_value = value
    return best_action, best_value


def _best_use_then_pick_action(
    state: GameState,
    choose_actions: tuple[Action, ...],
    *,
    config: PackSearchConfig,
    value_fn: ValueFn,
    include_item_value: bool,
    sampler: ShopSampler,
) -> tuple[Action | None, float]:
    use_actions = tuple(
        action
        for action in state.legal_actions
        if action.action_type == ActionType.USE_CONSUMABLE and consumable_action_is_supported(state, action)
    )
    if not use_actions or not choose_actions:
        return None, float("-inf")

    best_action: Action | None = None
    best_sequence: tuple[Action, Action] | None = None
    best_value = float("-inf")
    for use_index, use_action in enumerate(use_actions):
        outcomes = _stored_consumable_outcomes(state, use_action, config=config, sampler=sampler, action_index=use_index)
        if not outcomes:
            continue
        value_total = 0.0
        sequence_pick: Action | None = None
        valid_outcomes = 0
        for outcome in outcomes:
            best_pick_value = float("-inf")
            best_pick: Action | None = None
            for choose_action in choose_actions:
                if not _pack_action_is_legal(outcome, choose_action) or not _pack_action_is_searchable(outcome, choose_action):
                    continue
                try:
                    pick_value = pack_action_value(
                        outcome,
                        choose_action,
                        config=config,
                        value_fn=value_fn,
                        include_item_value=include_item_value,
                        sampler=sampler,
                    )
                except (ValueError, IndexError, TypeError, AttributeError):
                    continue
                if pick_value > best_pick_value:
                    best_pick_value = pick_value
                    best_pick = choose_action
            if best_pick is None:
                continue
            valid_outcomes += 1
            value_total += best_pick_value
            sequence_pick = best_pick
        if valid_outcomes <= 0 or sequence_pick is None:
            continue
        value = value_total / valid_outcomes
        if value > best_value:
            best_action = use_action
            best_sequence = (use_action, sequence_pick)
            best_value = value

    if best_action is None or best_sequence is None:
        return None, float("-inf")
    return _annotated_action(best_action, search_value=best_value, sequence=tuple(_action_summary(action) for action in best_sequence)), best_value


def _stored_consumable_outcomes(
    state: GameState,
    action: Action,
    *,
    config: PackSearchConfig,
    sampler: ShopSampler,
    action_index: int,
) -> tuple[GameState, ...]:
    outcomes: list[GameState] = []
    for sample_index in range(max(1, config.stochastic_samples)):
        rng = Random(_sample_seed(config.seed ^ (action_index * 1_000_003), action, sample_index=sample_index))
        try:
            outcomes.append(simulate_consumable_action(state, action, sampler=sampler, rng=rng))
        except (ValueError, IndexError, TypeError, AttributeError):
            continue
        if len(outcomes) == 1 and _stored_consumable_is_deterministic_enough(state, action):
            break
    return tuple(outcomes)


def _stored_consumable_is_deterministic_enough(state: GameState, action: Action) -> bool:
    index = _action_index(action)
    if index is None or not 0 <= index < len(state.consumables):
        return False
    name = state.consumables[index]
    return not _pack_consumable_needs_injections(name)


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


def _pack_action_is_searchable(state: GameState, action: Action) -> bool:
    if _is_skip_action(action):
        return True
    pack_cards = _modifier_items(state.modifiers, "pack_cards")
    index = _action_index(action)
    if index is None or not 0 <= index < len(pack_cards):
        return False
    return _is_searchable_pack_item(pack_cards[index])


def _is_searchable_pack_item(item: object) -> bool:
    item_set = _item_set(item)
    if item_set in {"JOKER", "PLANET", "TAROT", "SPECTRAL", "DEFAULT", "ENHANCED"}:
        return True
    label = _item_label(item)
    if label in TAROT_NAMES or label in SPECTRAL_NAMES or label in PLANET_TO_HAND:
        return True
    key = _item_key(item)
    return _playing_card_key(key)


def _pack_item_value(state: GameState, action: Action) -> float:
    if _is_skip_action(action):
        try:
            from balatro_ai.bots.basic_strategy_bot import _pack_skip_value

            return _pack_skip_value(state)
        except (ImportError, TypeError, ValueError, AttributeError):
            return 0.0
    pack_cards = _modifier_items(state.modifiers, "pack_cards")
    index = _action_index(action)
    if index is None or not 0 <= index < len(pack_cards):
        return 0.0
    try:
        from balatro_ai.bots.basic_strategy_bot import _pack_card_value

        return _pack_card_value(state, pack_cards[index]) + _pack_target_value_bonus(state, action, pack_cards[index])
    except (ImportError, TypeError, ValueError, AttributeError):
        return 0.0


def _pack_choice_outcomes(
    state: GameState,
    action: Action,
    *,
    config: PackSearchConfig,
    sampler: ShopSampler | None,
) -> tuple[GameState, ...]:
    if _is_skip_action(action):
        return (simulate_choose_pack_card(state, action, return_hand_to_deck=False),)
    pack_cards = _modifier_items(state.modifiers, "pack_cards")
    index = _action_index(action)
    if index is None or not 0 <= index < len(pack_cards):
        return ()
    item = pack_cards[index]
    name = _item_label(item)
    if not _pack_item_is_consumable(item):
        return (simulate_choose_pack_card(state, action, return_hand_to_deck=False),)
    if name in PLANET_TO_HAND or not _pack_consumable_needs_injections(name):
        return (simulate_choose_pack_card(state, action, return_hand_to_deck=False),)

    shop_sampler = sampler or ShopSampler.from_default_data()
    outcomes: list[GameState] = []
    for sample_index in range(max(1, config.stochastic_samples)):
        rng = Random(_sample_seed(config.seed, action, sample_index=sample_index))
        injections = sample_consumable_injections(state, name, sampler=shop_sampler, rng=rng, storage_use=False)
        try:
            outcomes.append(simulate_choose_pack_card(state, action, return_hand_to_deck=False, **injections))
        except (ValueError, IndexError, TypeError, AttributeError):
            continue
    return tuple(outcomes)


def _pack_item_is_consumable(item: object) -> bool:
    item_set = _item_set(item)
    label = _item_label(item)
    return item_set in {"TAROT", "PLANET", "SPECTRAL"} or label in TAROT_NAMES or label in SPECTRAL_NAMES or label in PLANET_TO_HAND


def _pack_target_value_bonus(state: GameState, action: Action, item: object) -> float:
    name = _item_label(item)
    enhancement = TAROT_ENHANCEMENTS.get(name)
    if enhancement != "GLASS":
        return 0.0
    bonus = 0.0
    for index in action.card_indices:
        if not 0 <= index < len(state.hand):
            continue
        card = state.hand[index]
        bonus += 18.0 + (_rank_value(card.rank) * 0.8)
        if not card.enhancement:
            bonus += 3.0
        if card.seal == "Red":
            bonus += 5.0
        if card.edition:
            bonus += 3.0
        if card.enhancement == "GLASS":
            bonus -= 8.0
    return bonus


def _rank_value(rank: str) -> int:
    values = {"A": 14, "K": 13, "Q": 12, "J": 11, "10": 10, "T": 10}
    if rank in values:
        return values[rank]
    try:
        return int(rank)
    except (TypeError, ValueError):
        return 5


def _pack_consumable_needs_injections(name: str) -> bool:
    return name in {
        "The Fool",
        "The High Priestess",
        "The Emperor",
        "The Wheel of Fortune",
        "Judgement",
        "Familiar",
        "Grim",
        "Incantation",
        "Aura",
        "Wraith",
        "Sigil",
        "Ouija",
        "Ectoplasm",
        "Immolate",
        "Ankh",
        "Hex",
        "The Soul",
    }


def _pack_sell_action_is_candidate(
    state: GameState,
    sell_action: Action,
    choose_actions: tuple[Action, ...],
) -> bool:
    kind = _action_kind(sell_action, default="joker")
    index = _action_index(sell_action)
    if kind == "joker":
        if index is None or not 0 <= index < len(state.jokers):
            return False
        return any(_pack_pick_needs_joker_slot(state, action) for action in choose_actions)
    if kind == "consumable":
        if index is None or not 0 <= index < len(state.consumables):
            return False
        return _consumable_open_slots(state) <= 0 and any(
            _pack_pick_can_create_consumables(state, action) for action in choose_actions
        )
    return False


def _pack_pick_needs_joker_slot(state: GameState, action: Action) -> bool:
    if _is_skip_action(action) or _pack_action_is_legal(state, action):
        return False
    pack_cards = _modifier_items(state.modifiers, "pack_cards")
    index = _action_index(action)
    if index is None or not 0 <= index < len(pack_cards):
        return False
    return _item_set(pack_cards[index]) == "JOKER" and _joker_item_uses_normal_slot(pack_cards[index])


def _pack_pick_can_create_consumables(state: GameState, action: Action) -> bool:
    if _is_skip_action(action):
        return False
    pack_cards = _modifier_items(state.modifiers, "pack_cards")
    index = _action_index(action)
    if index is None or not 0 <= index < len(pack_cards):
        return False
    return _item_label(pack_cards[index]) in {"The Fool", "The Emperor", "The High Priestess"}


def _pack_sell_can_improve_pick(state: GameState, sell_action: Action, choose_action: Action) -> bool:
    kind = _action_kind(sell_action, default="joker")
    if kind == "joker":
        return _pack_pick_needs_joker_slot(state, choose_action)
    if kind == "consumable":
        return _pack_pick_can_create_consumables(state, choose_action)
    return False


def _sell_penalty(state: GameState, sell_action: Action) -> float:
    kind = _action_kind(sell_action, default="joker")
    index = _action_index(sell_action)
    if kind == "joker":
        if index is None:
            return 0.0
        return _owned_joker_value(state, index)
    if kind == "consumable":
        if index is None or not 0 <= index < len(state.consumables):
            return 0.0
        return _owned_consumable_value(state.consumables[index])
    return 0.0


def _owned_consumable_value(name: str) -> float:
    if name in PLANET_TO_HAND:
        return 6.0
    if name in {"The Hermit", "Temperance"}:
        return 7.0
    if name in {"Death", "The Hanged Man", "The Emperor", "The High Priestess", "The Fool"}:
        return 5.0
    return 3.0


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


def _action_kind(action: Action, *, default: str) -> str:
    return str(action.metadata.get("kind", action.target_id or default))


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


def _item_label(item: object) -> str:
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return str(item)
    return str(item.get("label", item.get("name", item.get("key", ""))))


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


def _consumable_open_slots(state: GameState) -> int:
    return max(0, _consumable_slot_limit(state) - len(state.consumables))


def _consumable_slot_limit(state: GameState) -> int:
    limit = 2
    for key in ("consumable_slot_limit", "consumeable_slot_limit", "consumable_slots", "consumeable_slots"):
        raw = state.modifiers.get(key)
        try:
            if raw is not None:
                limit = max(0, int(raw))
                break
        except (TypeError, ValueError):
            continue
    return limit


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


def _sample_seed(seed: int, action: Action, *, sample_index: int) -> int:
    value = seed ^ (sample_index * 97_409)
    for char in action.stable_key:
        value = ((value * 16_777_619) ^ ord(char)) & 0xFFFFFFFF
    return value


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
