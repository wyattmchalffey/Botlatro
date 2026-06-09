"""Phase 7 search bot v2 experiment lane."""

from __future__ import annotations

from dataclasses import dataclass, field
import os

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GamePhase, GameState
from balatro_ai.bots.basic_strategy.actions import _blind_select_action, _first_action_of_type
from balatro_ai.bots.basic_strategy.blind_tactics import _tactical_blind_action
from balatro_ai.bots.basic_strategy.cache import decision_cache_scope
from balatro_ai.bots.basic_strategy.cards import _normal_joker_open_slots
from balatro_ai.bots.basic_strategy.joker_ordering import _joker_rearrange_action
from balatro_ai.bots.basic_strategy.play_scoring import _best_play_action
from balatro_ai.bots.basic_strategy_bot import BasicStrategyBot
from balatro_ai.search.consumable_search import ConsumableSearchConfig, best_consumable_action
from balatro_ai.search.discard_search import DiscardSearchConfig
from balatro_ai.search.hand_search import HandSearchConfig, best_hand_action
from balatro_ai.search.pack_search import PackSearchConfig, best_pack_action
from balatro_ai.search.shop_sampler import ShopSampler
from balatro_ai.search.shop_search import ShopSearchConfig, ShopSearchContext, best_shop_action
from balatro_ai.search.state_value import state_value_cache_scope


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


@dataclass(slots=True)
class SearchBotV2:
    """Fork of search_bot_v1 for independent winrate experiments."""

    seed: int | None = None
    discard_config: DiscardSearchConfig | None = None
    pack_config: PackSearchConfig | None = None
    shop_config: ShopSearchConfig | None = None
    consumable_config: ConsumableSearchConfig | None = None
    hand_config: HandSearchConfig | None = None
    enable_shop_search: bool = True
    name: str = "search_bot_v2"
    _fallback: BasicStrategyBot = field(init=False, repr=False)
    _shop_key: tuple[int | None, int, str] | None = field(default=None, init=False, repr=False)
    _protected_shop_jokers: tuple[str, ...] = field(default=(), init=False, repr=False)
    _rerolls_in_shop: int = field(default=0, init=False, repr=False)
    _packs_opened_in_shop: int = field(default=0, init=False, repr=False)
    _filled_last_joker_slot_in_shop: bool = field(default=False, init=False, repr=False)
    _sampler: ShopSampler = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._fallback = BasicStrategyBot(seed=self.seed)
        self._sampler = ShopSampler.from_default_data()
        if self.discard_config is None:
            self.discard_config = DiscardSearchConfig(draw_samples=2, leaf_samples=2, seed=self.seed or 0, max_actions=8)
        if self.hand_config is None:
            self.hand_config = HandSearchConfig(
                draw_samples=2,
                leaf_samples=2,
                seed=self.discard_config.seed,
                enumerate_draws_up_to=self.discard_config.enumerate_draws_up_to,
                max_play_actions=4,
                max_discard_actions=3,
                beam_depth=3,
                beam_width=_env_int("BALATRO_SEARCH_V2_HAND_BEAM_WIDTH", 2),
                beam_draw_samples=1,
                beam_leaf_samples=1,
            )
        if self.pack_config is None:
            self.pack_config = PackSearchConfig(leaf_samples=1, seed=self.seed or 0, stochastic_samples=4)
        if self.consumable_config is None:
            self.consumable_config = ConsumableSearchConfig(
                leaf_samples=1,
                seed=self.seed or 0,
                stochastic_samples=4,
                max_actions=48,
            )
        if self.enable_shop_search and self.shop_config is None:
            self.shop_config = ShopSearchConfig(seed=self.seed or 0, reroll_samples=8)

    def choose_action(self, state: GameState) -> Action:
        with decision_cache_scope(), state_value_cache_scope():
            return self._choose_action_uncached(state)

    def _choose_action_uncached(self, state: GameState) -> Action:
        self._sync_shop_memory(state)
        blind_select = _blind_select_action(state)
        if blind_select is not None:
            return blind_select
        cash_out = _first_action_of_type(state, ActionType.CASH_OUT)
        if cash_out is not None:
            return cash_out

        shop_search_owns_consumables = self.enable_shop_search and state.phase == GamePhase.SHOP
        if state.phase == GamePhase.BOOSTER_OPENED and any(action.action_type == ActionType.CHOOSE_PACK_CARD for action in state.legal_actions):
            pack_choice = best_pack_action(state, config=self.pack_config, sampler=self._sampler)
            if pack_choice is not None:
                return pack_choice
        if not shop_search_owns_consumables and any(action.action_type == ActionType.USE_CONSUMABLE for action in state.legal_actions):
            consumable_action = best_consumable_action(state, config=self.consumable_config, sampler=self._sampler)
            if consumable_action is not None:
                return consumable_action
        if any(action.action_type == ActionType.CHOOSE_PACK_CARD for action in state.legal_actions):
            pack_choice = best_pack_action(state, config=self.pack_config, sampler=self._sampler)
            if pack_choice is not None:
                return pack_choice
        if self.enable_shop_search and any(
            action.action_type
            in {
                ActionType.BUY,
                ActionType.SELL,
                ActionType.REROLL,
                ActionType.OPEN_PACK,
                ActionType.END_SHOP,
                ActionType.USE_CONSUMABLE,
            }
            for action in state.legal_actions
        ):
            shop_action = best_shop_action(
                state,
                config=self.shop_config,
                sampler=self._sampler,
                protected_jokers=self._protected_shop_jokers,
                shop_context=ShopSearchContext(
                    rerolls_in_shop=self._rerolls_in_shop,
                    packs_opened_in_shop=self._packs_opened_in_shop,
                    filled_last_joker_slot=self._filled_last_joker_slot_in_shop,
                ),
            )
            if shop_action is not None:
                self._record_shop_action(state, shop_action)
                return shop_action
        if any(action.action_type in {ActionType.PLAY_HAND, ActionType.DISCARD} for action in state.legal_actions):
            basic_action = self._basic_blind_action(state)
            hand_action = best_hand_action(state, config=self.hand_config, context=self._basic_blind_context(state))
            if hand_action is not None:
                if (
                    basic_action is not None
                    and _basic_blind_action_should_guard(basic_action, hand_action)
                    and not _v2_beam_can_override_basic(basic_action, hand_action)
                ):
                    self._record_basic_blind_action(state, basic_action)
                    return basic_action
                self._record_basic_blind_action(state, hand_action)
                return hand_action
            if basic_action is not None:
                self._record_basic_blind_action(state, basic_action)
                return basic_action
        return self._fallback.choose_action(state)

    def _basic_blind_context(self, state: GameState):
        self._fallback._sync_blind_memory(state)
        return self._fallback._blind_context(state)

    def _basic_blind_action(self, state: GameState) -> Action | None:
        context = self._basic_blind_context(state)
        rearrange = _joker_rearrange_action(state, context)
        if rearrange is not None:
            return rearrange
        best_play = _best_play_action(state, context)
        if best_play is None:
            return None
        return _tactical_blind_action(state, best_play, context)

    def _record_basic_blind_action(self, state: GameState, action: Action) -> None:
        self._fallback._record_blind_action(state, action, self._basic_blind_context(state))

    def _sync_shop_memory(self, state: GameState) -> None:
        if state.phase == GamePhase.BOOSTER_OPENED:
            return
        if state.phase != GamePhase.SHOP:
            self._shop_key = None
            self._protected_shop_jokers = ()
            self._rerolls_in_shop = 0
            self._packs_opened_in_shop = 0
            self._filled_last_joker_slot_in_shop = False
            return
        key = (state.seed, state.ante, state.blind)
        if key != self._shop_key:
            self._shop_key = key
            self._protected_shop_jokers = ()
            self._rerolls_in_shop = 0
            self._packs_opened_in_shop = 0
            self._filled_last_joker_slot_in_shop = False
            return
        current_names = {joker.name for joker in state.jokers}
        self._protected_shop_jokers = tuple(name for name in self._protected_shop_jokers if name in current_names)

    def _record_shop_action(self, state: GameState, action: Action) -> None:
        if action.action_type == ActionType.REROLL:
            self._rerolls_in_shop += 1
            return
        if action.action_type == ActionType.OPEN_PACK:
            self._packs_opened_in_shop += 1
            return
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
        if _normal_joker_open_slots(state) == 1 and _joker_item_uses_normal_slot(item):
            self._filled_last_joker_slot_in_shop = True
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


def _joker_item_uses_normal_slot(item: object) -> bool:
    return "negative" not in _item_edition(item).lower()


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


def _item_label(item: object) -> str:
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return str(item)
    return str(item.get("label", item.get("name", item.get("key", ""))))


def _basic_blind_action_should_guard(basic_action: Action, search_action: Action) -> bool:
    if basic_action.action_type == ActionType.REARRANGE:
        return True
    if basic_action.action_type == search_action.action_type:
        return False
    return {
        basic_action.action_type,
        search_action.action_type,
    } <= {ActionType.PLAY_HAND, ActionType.DISCARD}


def _v2_beam_can_override_basic(basic_action: Action, search_action: Action) -> bool:
    if basic_action.action_type == ActionType.REARRANGE:
        return False
    return search_action.metadata.get("search") == "hand_beam"
