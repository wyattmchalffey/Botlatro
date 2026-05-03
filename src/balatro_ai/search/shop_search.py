"""Beam search over deterministic shop actions for Phase 7."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from random import Random
from typing import Any

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GamePhase, GameState
from balatro_ai.search.forward_sim import simulate_buy, simulate_end_shop, simulate_open_pack, simulate_reroll, simulate_sell
from balatro_ai.search.shop_sampler import ShopSampler

LeafValueFn = Callable[[GameState], float]
ActionValueFn = Callable[[GameState, Action], float]


@dataclass(frozen=True, slots=True)
class ShopSearchConfig:
    beam_width: int = 8
    depth: int = 3
    reroll_samples: int = 32
    seed: int = 0
    leaf_weight: float = 0.35
    min_search_value: float = 0.0


@dataclass(frozen=True, slots=True)
class _BeamNode:
    state: GameState
    first_action: Action | None
    path: tuple[Action, ...]
    score: float
    terminal: bool = False
    protected_jokers: tuple[str, ...] = ()


def best_shop_action(
    state: GameState,
    *,
    config: ShopSearchConfig | None = None,
    sampler: ShopSampler | None = None,
    leaf_value_fn: LeafValueFn | None = None,
    action_value_fn: ActionValueFn | None = None,
    protected_jokers: tuple[str, ...] = (),
) -> Action | None:
    """Return the first action from the best shop beam sequence."""

    if state.phase != GamePhase.SHOP:
        return None
    search_config = config or ShopSearchConfig()
    if search_config.depth <= 0 or search_config.beam_width <= 0:
        return None

    require_replacement_upgrade = leaf_value_fn is None and action_value_fn is None
    shop_sampler = sampler or ShopSampler.from_default_data()
    leaf_value = leaf_value_fn or shop_leaf_value
    action_value = action_value_fn or (lambda current, action: shop_action_search_value(current, action, sampler=shop_sampler, config=search_config))
    root_actions = _legal_shop_actions(
        state,
        shop_sampler,
        root_actions=state.legal_actions,
        protected_jokers=protected_jokers,
        require_replacement_upgrade=require_replacement_upgrade,
    )
    if not root_actions:
        return None

    beams: tuple[_BeamNode, ...] = (_BeamNode(state=state, first_action=None, path=(), score=0.0),)
    completed: list[_BeamNode] = []
    rng = Random(search_config.seed)
    for _ in range(search_config.depth):
        expanded: list[_BeamNode] = []
        for node in beams:
            if node.terminal:
                completed.append(node)
                continue
            actions = _legal_shop_actions(
                node.state,
                shop_sampler,
                root_actions=root_actions if not node.path else (),
                protected_jokers=(*protected_jokers, *node.protected_jokers),
                require_joker_buy_after_sell=bool(node.path and node.path[-1].action_type == ActionType.SELL),
                require_replacement_upgrade=require_replacement_upgrade,
            )
            if not actions:
                completed.append(node)
                continue
            for action in actions:
                child = _expand_action(
                    node,
                    action,
                    shop_sampler,
                    rng=rng,
                    action_value_fn=action_value,
                    leaf_value_fn=leaf_value,
                    leaf_weight=search_config.leaf_weight,
                )
                if child.terminal:
                    completed.append(child)
                else:
                    expanded.append(child)
        if not expanded:
            break
        beams = tuple(sorted(expanded, key=lambda item: item.score, reverse=True)[: search_config.beam_width])

    completed.extend(beams)
    candidates = [node for node in completed if node.first_action is not None]
    if not candidates:
        return None
    best = max(candidates, key=lambda item: item.score)
    if best.score < search_config.min_search_value:
        return None
    return _annotated_action(best.first_action, search_value=best.score, path=best.path)


def shop_action_search_value(
    state: GameState,
    action: Action,
    *,
    sampler: ShopSampler | None = None,
    config: ShopSearchConfig | None = None,
) -> float:
    """Score one shop action using Basic Strategy's current shop heuristics."""

    if action.action_type == ActionType.REROLL:
        shop_sampler = sampler or ShopSampler.from_default_data()
        search_config = config or ShopSearchConfig()
        reroll_cost = shop_sampler.reroll_cost(state)
        reroll_value = shop_sampler.reroll_ev(state, samples=search_config.reroll_samples, rng=Random(search_config.seed))
        return reroll_value - _spend_opportunity_penalty(state, reroll_cost)
    if action.action_type == ActionType.SELL:
        from balatro_ai.bots.basic_strategy_bot import _owned_joker_value

        index = _action_index(action)
        if index is None or not 0 <= index < len(state.jokers):
            return 0.0
        sold = state.jokers[index]
        return max(0.0, float(sold.sell_value or 0) - (_owned_joker_value(state, sold, remove_index=index) * 0.15))
    if action.action_type in {ActionType.BUY, ActionType.OPEN_PACK}:
        from balatro_ai.bots.basic_strategy_bot import _ShopContext, _shop_action_value, _shop_pressure

        try:
            return _shop_action_value(state, action, _shop_pressure(state), _ShopContext())
        except (IndexError, ValueError, TypeError):
            return 0.0
    return 0.0


def shop_leaf_value(state: GameState) -> float:
    """Score an intermediate shop state after deterministic actions."""

    from balatro_ai.bots.basic_strategy_bot import _build_profile, _owned_joker_value

    profile = _build_profile(state)
    owned_value = sum(_owned_joker_value(state, joker, remove_index=index) for index, joker in enumerate(state.jokers))
    role_value = 0.0
    for role in ("chips", "mult", "xmult", "scaling", "economy"):
        requirement = max(1.0, profile.role_requirement(role))
        role_value += min(1.0, profile.role_score(role) / requirement) * 10.0
    missing_penalty = len(profile.missing_roles) * (8.0 + min(8.0, max(0, state.ante - 2) * 1.5))
    money_value = _shop_money_value(state)
    slot_value = _normal_joker_open_slots(state) * 2.5
    consumable_value = len(state.consumables) * 1.5
    return owned_value + role_value + money_value + slot_value + consumable_value - missing_penalty


def _shop_money_value(state: GameState) -> float:
    """Value cash as both spending power and next-cash-out interest."""

    money = max(0, state.money)
    try:
        from balatro_ai.bots.basic_strategy_bot import _desired_money_reserve, _interest_cap_money, _shop_pressure

        pressure = _shop_pressure(state)
        interest_cap = _interest_cap_money(state)
        reserve = _desired_money_reserve(state, pressure)
    except (ImportError, TypeError, ValueError, AttributeError):
        interest_cap = 25
        reserve = 25

    interest_bank = min(money, interest_cap)
    projected_interest = interest_bank // 5
    next_breakpoint_progress = 0.0 if money >= interest_cap else (interest_bank % 5) / 5.0
    reserve_shortfall = max(0, reserve - money)
    above_reserve = max(0, money - reserve)

    return (
        (interest_bank * 0.45)
        + (max(0, money - interest_cap) * 0.22)
        + (projected_interest * 3.5)
        + (next_breakpoint_progress * 0.8)
        - (reserve_shortfall * 0.45)
        + (above_reserve * 0.12)
    )


def _spend_opportunity_penalty(state: GameState, cost: int) -> float:
    if cost <= 0:
        return 0.0
    try:
        from balatro_ai.bots.basic_strategy_bot import _money_after_spend_penalty, _shop_pressure

        return _money_after_spend_penalty(state, cost, _shop_pressure(state))
    except (ImportError, TypeError, ValueError, AttributeError):
        after = state.money - cost
        if after < 0:
            return 1000.0
        before_interest = min(max(0, state.money), 25) // 5
        after_interest = min(max(0, after), 25) // 5
        return max(0, before_interest - after_interest) * 4.0


def _expand_action(
    node: _BeamNode,
    action: Action,
    sampler: ShopSampler,
    *,
    rng: Random,
    action_value_fn: ActionValueFn,
    leaf_value_fn: LeafValueFn,
    leaf_weight: float,
) -> _BeamNode:
    first_action = node.first_action or action
    path = node.path + (action,)
    immediate = action_value_fn(node.state, action)
    terminal = action.action_type in {ActionType.END_SHOP, ActionType.REROLL, ActionType.OPEN_PACK}
    try:
        next_state = _simulate_shop_action(node.state, action, sampler, rng=rng)
    except (ValueError, IndexError, TypeError):
        return _BeamNode(
            state=node.state,
            first_action=first_action,
            path=path,
            score=float("-inf"),
            terminal=True,
        )
    score = node.score + immediate + (leaf_value_fn(next_state) * leaf_weight)
    protected_jokers = node.protected_jokers
    bought_joker = _bought_joker_name(node.state, action)
    if bought_joker is not None:
        protected_jokers = (*protected_jokers, bought_joker)
    return _BeamNode(
        state=next_state,
        first_action=first_action,
        path=path,
        score=score,
        terminal=terminal,
        protected_jokers=protected_jokers,
    )


def _simulate_shop_action(state: GameState, action: Action, sampler: ShopSampler, *, rng: Random) -> GameState:
    if action.action_type == ActionType.BUY:
        bought = simulate_buy(state, action)
        if not _is_overstock_voucher_buy(state, action):
            return bought
        current_shop = _modifier_items(bought.modifiers, "shop_cards")
        filled_shop = sampler.fill_shop_to_slot_count(bought, current_shop, rng=rng)
        return _with_shop_cards(bought, filled_shop)
    if action.action_type == ActionType.SELL:
        return simulate_sell(state, action)
    if action.action_type == ActionType.REROLL:
        return simulate_reroll(state, action, sampler.sample_shop(state, rng=rng))
    if action.action_type == ActionType.END_SHOP:
        return simulate_end_shop(state)
    if action.action_type == ActionType.OPEN_PACK:
        return simulate_open_pack(state, action, ())
    return state


def _bought_joker_name(state: GameState, action: Action) -> str | None:
    if action.action_type != ActionType.BUY or action.target_id != "card":
        return None
    index = _action_index(action)
    shop_cards = _modifier_items(state.modifiers, "shop_cards")
    if index is None or not 0 <= index < len(shop_cards):
        return None
    item = shop_cards[index]
    if not _is_joker_item(item):
        return None
    return _item_label(item)


def _legal_shop_actions(
    state: GameState,
    sampler: ShopSampler,
    *,
    root_actions: tuple[Action, ...] = (),
    protected_jokers: tuple[str, ...] = (),
    require_joker_buy_after_sell: bool = False,
    require_replacement_upgrade: bool = False,
) -> tuple[Action, ...]:
    if require_joker_buy_after_sell:
        return tuple(
            Action(ActionType.BUY, target_id="card", amount=index, metadata={"kind": "card", "index": index})
            for index, item in enumerate(_modifier_items(state.modifiers, "shop_cards"))
            if _is_joker_item(item) and _can_buy_item(state, item) and _shop_card_can_be_bought(state, item)
        )

    if root_actions:
        def root_action_allowed(action: Action) -> bool:
            if not _supported_root_action(action) or _sells_protected_joker(state, action, protected_jokers):
                return False
            if action.action_type != ActionType.SELL:
                return True
            index = _action_index(action)
            return index is not None and _sell_is_search_candidate(
                state,
                index,
                require_upgrade=require_replacement_upgrade,
            )

        return tuple(
            action
            for action in root_actions
            if root_action_allowed(action)
        )

    actions: list[Action] = []
    for index in range(len(state.jokers)):
        action = Action(ActionType.SELL, target_id="joker", amount=index, metadata={"kind": "joker", "index": index})
        if not _sells_protected_joker(state, action, protected_jokers) and _sell_is_search_candidate(
            state,
            index,
            require_upgrade=require_replacement_upgrade,
        ):
            actions.append(action)

    for index, item in enumerate(_modifier_items(state.modifiers, "shop_cards")):
        if _can_buy_item(state, item) and _shop_card_can_be_bought(state, item):
            actions.append(Action(ActionType.BUY, target_id="card", amount=index, metadata={"kind": "card", "index": index}))

    for index, item in enumerate(_modifier_items(state.modifiers, "voucher_cards")):
        if _can_buy_item(state, item):
            actions.append(Action(ActionType.BUY, target_id="voucher", amount=index, metadata={"kind": "voucher", "index": index}))

    for index, item in enumerate(_modifier_items(state.modifiers, "booster_packs")):
        if _can_buy_item(state, item):
            actions.append(Action(ActionType.OPEN_PACK, target_id="pack", amount=index, metadata={"kind": "pack", "index": index}))

    if sampler.reroll_cost(state) <= state.money:
        actions.append(Action(ActionType.REROLL))
    actions.append(Action(ActionType.END_SHOP))
    return tuple(actions)


def _supported_root_action(action: Action) -> bool:
    return action.action_type in {ActionType.BUY, ActionType.SELL, ActionType.REROLL, ActionType.END_SHOP, ActionType.OPEN_PACK}


def _sells_protected_joker(state: GameState, action: Action, protected_jokers: tuple[str, ...]) -> bool:
    if action.action_type != ActionType.SELL or not protected_jokers:
        return False
    index = _action_index(action)
    if index is None or not 0 <= index < len(state.jokers):
        return False
    return state.jokers[index].name in protected_jokers


def _sell_is_search_candidate(state: GameState, index: int, *, require_upgrade: bool = False) -> bool:
    if not 0 <= index < len(state.jokers):
        return False
    sell_value = state.jokers[index].sell_value or 0
    if sell_value <= 0:
        return False
    money_after_sell = state.money + sell_value
    for item_index, item in enumerate(_modifier_items(state.modifiers, "shop_cards")):
        if not _is_joker_item(item):
            continue
        if _item_cost(state, item) > money_after_sell:
            continue
        if _can_buy_item(state, item) and _shop_card_can_be_bought(state, item):
            continue
        if _joker_replacement_has_room_after_sell(state, item):
            if require_upgrade and not _replacement_upgrade_passes(state, index, item_index, item):
                continue
            return True
    return False


def _replacement_upgrade_passes(state: GameState, index: int, item_index: int, item: object) -> bool:
    if not 0 <= index < len(state.jokers):
        return False
    try:
        sell_action = Action(ActionType.SELL, target_id="joker", amount=index, metadata={"kind": "joker", "index": index})
        sold_state = simulate_sell(state, sell_action)
        buy_action = Action(ActionType.BUY, target_id="card", amount=item_index, metadata={"kind": "card", "index": item_index})
        replacement_state = simulate_buy(sold_state, buy_action)
        delta = _replacement_state_value(replacement_state) - _replacement_state_value(state)
        return delta >= _replacement_delta_threshold(state, item)
    except (TypeError, ValueError, IndexError, AttributeError):
        return False


def _replacement_state_value(state: GameState) -> float:
    try:
        from balatro_ai.bots.basic_strategy_bot import _build_profile, _sample_build_score
    except ImportError:
        return shop_leaf_value(state)
    profile = _build_profile(state)
    score_value = min(6000.0, max(0.0, float(_sample_build_score(state, state.jokers)))) * 0.06
    role_value = 0.0
    for role in ("chips", "mult", "xmult", "scaling", "economy"):
        requirement = max(1.0, profile.role_requirement(role))
        role_value += min(1.0, profile.role_score(role) / requirement) * 9.0
    missing_penalty = len(profile.missing_roles) * 7.0
    return score_value + role_value + _shop_money_value(state) + (len(state.consumables) * 1.5) - missing_penalty


def _replacement_delta_threshold(state: GameState, item: object) -> float:
    threshold = 10.0
    if state.ante >= 5:
        threshold += 6.0
    if _is_joker_item(item) and _normal_joker_open_slots(state) <= 0:
        threshold += 4.0
    return threshold


def _joker_replacement_has_room_after_sell(state: GameState, item: object) -> bool:
    if not _joker_item_uses_normal_slot(item):
        return True
    return _normal_joker_slots_used(state) >= _normal_joker_slot_limit(state)


def _can_buy_item(state: GameState, item: object) -> bool:
    return _item_cost(state, item) <= state.money


def _shop_card_can_be_bought(state: GameState, item: object) -> bool:
    if not _is_joker_item(item):
        return not _is_consumable_item(item) or _consumable_open_slots(state) > 0
    if not _joker_item_uses_normal_slot(item):
        return True
    return _normal_joker_open_slots(state) > 0


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
    limit = 2
    for key in ("consumable_slot_limit", "consumeable_slot_limit", "consumable_slots", "consumeable_slots"):
        raw = state.modifiers.get(key)
        try:
            if raw is not None:
                limit = max(0, int(raw))
                break
        except (TypeError, ValueError):
            continue
    return max(0, limit - len(state.consumables))


def _joker_item_uses_normal_slot(item: object) -> bool:
    edition = _item_edition(item)
    return "negative" not in edition.lower()


def _is_joker_item(item: object) -> bool:
    if not isinstance(item, dict):
        return False
    item_set = str(item.get("set", "")).upper()
    key = str(item.get("key", ""))
    label = str(item.get("label", item.get("name", ""))).lower()
    return item_set == "JOKER" or key.startswith("j_") or "joker" in label


def _is_consumable_item(item: object) -> bool:
    if not isinstance(item, dict):
        return False
    item_set = str(item.get("set", "")).upper()
    key = str(item.get("key", "")).lower()
    label = str(item.get("label", item.get("name", ""))).lower()
    return item_set in {"TAROT", "PLANET", "SPECTRAL"} or key.startswith(("c_", "p_")) and "pack" not in label


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


def _item_cost(state: GameState, item: object) -> int:
    if _astronomer_makes_item_free(state, item):
        return 0
    if not isinstance(item, dict):
        return 0
    cost = item.get("cost", {})
    if isinstance(cost, dict):
        return _int_value(cost.get("buy", cost.get("cost", 0)))
    return _int_value(cost)


def _astronomer_makes_item_free(state: GameState, item: object) -> bool:
    if not any(joker.name == "Astronomer" for joker in state.jokers):
        return False
    item_set = str(item.get("set", "")).upper() if isinstance(item, dict) else ""
    if item_set == "PLANET":
        return True
    label = str(item.get("label", item.get("name", ""))).lower() if isinstance(item, dict) else ""
    key = str(item.get("key", "")).lower() if isinstance(item, dict) else ""
    return item_set == "BOOSTER" and ("celestial" in label or key.startswith("p_celestial"))


def _item_label(item: object) -> str:
    if isinstance(item, dict):
        return str(item.get("name") or item.get("label") or item.get("key") or "")
    return str(item)


def _with_shop_cards(state: GameState, shop_cards: tuple[object, ...]) -> GameState:
    modifiers = {**state.modifiers, "shop_cards": shop_cards}
    return replace(state, modifiers=modifiers, shop=tuple(_item_label(item) for item in shop_cards), legal_actions=())


def _is_overstock_voucher_buy(state: GameState, action: Action) -> bool:
    if action.action_type != ActionType.BUY:
        return False
    if str(action.metadata.get("kind", action.target_id or "")) != "voucher":
        return False
    index = _action_index(action)
    voucher_cards = _modifier_items(state.modifiers, "voucher_cards")
    if index is None or not 0 <= index < len(voucher_cards):
        return False
    return _item_label(voucher_cards[index]) in {"Overstock", "Overstock Plus"}


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


def _int_value(raw: object) -> int:
    try:
        return int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _annotated_action(action: Action, *, search_value: float, path: tuple[Action, ...]) -> Action:
    return Action(
        action.action_type,
        card_indices=action.card_indices,
        target_id=action.target_id,
        amount=action.amount,
        metadata={
            **action.metadata,
            "search": "shop_beam",
            "search_value": round(search_value, 6),
            "search_path": tuple(_action_summary(item) for item in path),
        },
    )


def _action_summary(action: Action) -> dict[str, Any]:
    return {
        "type": action.action_type.value,
        "kind": str(action.metadata.get("kind", action.target_id or "")),
        "index": action.metadata.get("index", action.amount),
    }
