"""Phase 7 search bot integrations."""

from __future__ import annotations

from dataclasses import dataclass, field

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GamePhase, GameState
from balatro_ai.bots.basic_strategy_bot import BasicStrategyBot
from balatro_ai.search.discard_search import DiscardSearchConfig, best_discard_action
from balatro_ai.search.pack_search import PackSearchConfig, best_pack_action
from balatro_ai.search.shop_search import ShopSearchConfig, best_shop_action


@dataclass(slots=True)
class SearchBot:
    """SearchBot integrations layered on top of the Basic Strategy fallback."""

    seed: int | None = None
    discard_config: DiscardSearchConfig | None = None
    pack_config: PackSearchConfig | None = None
    shop_config: ShopSearchConfig | None = None
    enable_shop_search: bool = False
    name: str = "search_bot_v0"
    _fallback: BasicStrategyBot = field(init=False, repr=False)
    _shop_key: tuple[int | None, int, str] | None = field(default=None, init=False, repr=False)
    _protected_shop_jokers: tuple[str, ...] = field(default=(), init=False, repr=False)

    def __post_init__(self) -> None:
        self._fallback = BasicStrategyBot(seed=self.seed)
        if self.discard_config is None:
            self.discard_config = DiscardSearchConfig(draw_samples=1, leaf_samples=1, seed=self.seed or 0, max_actions=16)
        if self.pack_config is None:
            self.pack_config = PackSearchConfig(leaf_samples=2, seed=self.seed or 0)
        if self.enable_shop_search and self.shop_config is None:
            self.shop_config = ShopSearchConfig(seed=self.seed or 0)

    def choose_action(self, state: GameState) -> Action:
        self._sync_shop_memory(state)
        if any(action.action_type == ActionType.CHOOSE_PACK_CARD for action in state.legal_actions):
            pack_choice = best_pack_action(state, config=self.pack_config)
            if pack_choice is not None:
                return pack_choice
        if self.enable_shop_search and any(
            action.action_type in {ActionType.BUY, ActionType.SELL, ActionType.REROLL, ActionType.OPEN_PACK, ActionType.END_SHOP}
            for action in state.legal_actions
        ):
            shop_action = best_shop_action(state, config=self.shop_config, protected_jokers=self._protected_shop_jokers)
            if shop_action is not None:
                self._record_shop_action(state, shop_action)
                return shop_action
        if any(action.action_type == ActionType.DISCARD for action in state.legal_actions):
            discard = best_discard_action(state, config=self.discard_config)
            if discard is not None:
                return discard
        return self._fallback.choose_action(state)

    def _sync_shop_memory(self, state: GameState) -> None:
        if state.phase == GamePhase.BOOSTER_OPENED:
            return
        if state.phase != GamePhase.SHOP:
            self._shop_key = None
            self._protected_shop_jokers = ()
            return
        key = (state.seed, state.ante, state.blind)
        if key != self._shop_key:
            self._shop_key = key
            self._protected_shop_jokers = ()
            return
        current_names = {joker.name for joker in state.jokers}
        self._protected_shop_jokers = tuple(name for name in self._protected_shop_jokers if name in current_names)

    def _record_shop_action(self, state: GameState, action: Action) -> None:
        if action.action_type != ActionType.BUY:
            return
        kind = str(action.metadata.get("kind", action.target_id or ""))
        if kind != "card":
            return
        index = _action_index(action)
        shop_cards = _modifier_items(state.modifiers, "shop_cards")
        if index is None or not 0 <= index < len(shop_cards):
            return
        item = shop_cards[index]
        if not _is_joker_item(item):
            return
        name = _item_label(item)
        if name and name not in self._protected_shop_jokers:
            self._protected_shop_jokers = self._protected_shop_jokers + (name,)


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


def _is_joker_item(item: object) -> bool:
    if not isinstance(item, dict):
        return False
    item_set = str(item.get("set", "")).upper()
    key = str(item.get("key", ""))
    label = _item_label(item).lower()
    return item_set == "JOKER" or key.startswith("j_") or "joker" in label


def _item_label(item: object) -> str:
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return str(item)
    return str(item.get("label", item.get("name", item.get("key", ""))))
