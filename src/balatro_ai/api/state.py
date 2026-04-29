"""Game state models used by bots, evaluation, and replay logs."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from itertools import combinations
from typing import Any

from balatro_ai.api.actions import Action, ActionType, actions_from_mappings


class GamePhase(StrEnum):
    MENU = "menu"
    BLIND_SELECT = "blind_select"
    SELECTING_HAND = "selecting_hand"
    PLAYING_BLIND = "playing_blind"
    ROUND_EVAL = "round_eval"
    SHOP = "shop"
    PACK = "pack"
    BOOSTER_OPENED = "booster_opened"
    RUN_OVER = "run_over"
    UNKNOWN = "unknown"


class Stake(StrEnum):
    WHITE = "white"
    RED = "red"
    GREEN = "green"
    BLACK = "black"
    BLUE = "blue"
    PURPLE = "purple"
    ORANGE = "orange"
    GOLD = "gold"
    UNKNOWN = "unknown"


PHASE_ALIASES = {
    "MENU": GamePhase.MENU,
    "BLIND_SELECT": GamePhase.BLIND_SELECT,
    "SELECTING_HAND": GamePhase.SELECTING_HAND,
    "ROUND_EVAL": GamePhase.ROUND_EVAL,
    "SHOP": GamePhase.SHOP,
    "SMODS_BOOSTER_OPENED": GamePhase.BOOSTER_OPENED,
    "GAME_OVER": GamePhase.RUN_OVER,
}


@dataclass(frozen=True, slots=True)
class Card:
    rank: str
    suit: str
    enhancement: str | None = None
    seal: str | None = None
    edition: str | None = None
    debuffed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def short_name(self) -> str:
        return f"{self.rank}{self.suit}"

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | str) -> "Card":
        if isinstance(data, str):
            rank = data[:-1]
            suit = data[-1:]
            return cls(rank=rank, suit=suit)
        value = _mapping_or_empty(data.get("value"))
        modifier = _mapping_or_empty(data.get("modifier"))
        state = _mapping_or_empty(data.get("state"))
        return cls(
            rank=str(data.get("rank", value.get("rank"))),
            suit=str(data.get("suit", value.get("suit"))),
            enhancement=data.get("enhancement", modifier.get("enhancement")),
            seal=data.get("seal", modifier.get("seal")),
            edition=data.get("edition", modifier.get("edition")),
            debuffed=bool(data.get("debuffed", state.get("debuff", False))),
            metadata=_raw_metadata(data),
        )


@dataclass(frozen=True, slots=True)
class Joker:
    name: str
    edition: str | None = None
    sell_value: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | str) -> "Joker":
        if isinstance(data, str):
            return cls(name=data)
        modifier = _mapping_or_empty(data.get("modifier"))
        cost = _mapping_or_empty(data.get("cost"))
        return cls(
            name=str(data.get("name", data.get("label", data.get("key", "Unknown Joker")))),
            edition=data.get("edition", modifier.get("edition")),
            sell_value=data.get("sell_value", cost.get("sell")),
            metadata=_raw_metadata(data),
        )


@dataclass(frozen=True, slots=True)
class GameState:
    phase: GamePhase = GamePhase.UNKNOWN
    stake: Stake = Stake.UNKNOWN
    seed: int | None = None
    ante: int = 0
    blind: str = ""
    required_score: int = 0
    current_score: int = 0
    hands_remaining: int = 0
    discards_remaining: int = 0
    money: int = 0
    deck_size: int = 0
    hand: tuple[Card, ...] = ()
    known_deck: tuple[Card, ...] = ()
    jokers: tuple[Joker, ...] = ()
    consumables: tuple[str, ...] = ()
    vouchers: tuple[str, ...] = ()
    shop: tuple[str, ...] = ()
    pack: tuple[str, ...] = ()
    hand_levels: dict[str, int] = field(default_factory=dict)
    modifiers: dict[str, Any] = field(default_factory=dict)
    legal_actions: tuple[Action, ...] = ()
    run_over: bool = False
    won: bool = False

    @property
    def debug_summary(self) -> str:
        hand = " ".join(card.short_name for card in self.hand) or "-"
        jokers = ", ".join(joker.name for joker in self.jokers) or "-"
        return (
            f"phase={self.phase.value} ante={self.ante} blind={self.blind or '-'} "
            f"score={self.current_score}/{self.required_score} money={self.money} "
            f"hands={self.hands_remaining} discards={self.discards_remaining} "
            f"hand=[{hand}] jokers=[{jokers}]"
        )

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "GameState":
        if data is None:
            raise ValueError("Cannot build GameState from None")

        phase = _parse_phase(data.get("phase", data.get("state", GamePhase.UNKNOWN.value)))
        stake = _parse_stake(data.get("stake", Stake.UNKNOWN.value))
        round_data = data.get("round", {})
        current_blind = _current_blind(data.get("blinds", {}))
        hand_cards = _area_cards(data.get("hand", data.get("cards", {})))
        joker_cards = _area_cards(data.get("jokers", {}))
        consumable_cards = _area_cards(data.get("consumables", {}))
        shop_cards = _area_cards(data.get("shop", {}))
        voucher_cards = _area_cards(data.get("vouchers", {}))
        pack_cards = _area_cards(data.get("pack", {}))
        booster_packs = _area_cards(data.get("packs", {}))
        hand_levels = _hand_levels(data.get("hands", data.get("hand_levels", {})))

        state = cls(
            phase=phase,
            stake=stake,
            seed=data.get("seed"),
            ante=int(data.get("ante", data.get("ante_num", 0))),
            blind=str(data.get("blind", current_blind.get("name", ""))),
            required_score=int(data.get("required_score", data.get("score_required", current_blind.get("score", 0)))),
            current_score=int(data.get("current_score", data.get("score", round_data.get("chips", 0)))),
            hands_remaining=int(data.get("hands_remaining", round_data.get("hands_left", 0))),
            discards_remaining=int(data.get("discards_remaining", round_data.get("discards_left", 0))),
            money=int(data.get("money", 0)),
            deck_size=int(data.get("deck_size", _area_count(data.get("cards", {})))),
            hand=tuple(Card.from_mapping(card) for card in hand_cards),
            known_deck=tuple(Card.from_mapping(card) for card in data.get("known_deck", ())),
            jokers=tuple(Joker.from_mapping(joker) for joker in joker_cards),
            consumables=tuple(_card_label(item) for item in consumable_cards),
            vouchers=tuple(_card_label(item) for item in voucher_cards),
            shop=tuple(_card_label(item) for item in shop_cards),
            pack=tuple(_card_label(item) for item in pack_cards),
            hand_levels=hand_levels,
            modifiers={
                **dict(data.get("modifiers", {})),
                "current_blind": current_blind,
                "joker_cards": joker_cards,
                "shop_cards": shop_cards,
                "voucher_cards": voucher_cards,
                "booster_packs": booster_packs,
                "pack_cards": pack_cards,
            },
            legal_actions=actions_from_mappings(tuple(data.get("legal_actions", ()))),
            run_over=bool(data.get("run_over", phase == GamePhase.RUN_OVER)),
            won=bool(data.get("won", False)),
        )
        if state.ante >= 9 and not state.run_over and not state.won:
            return replace(state, ante=8, run_over=True, won=True, legal_actions=())
        if state.legal_actions:
            return _with_augmented_sell_actions(state)
        return _with_derived_legal_actions(state, shop_cards=shop_cards, voucher_cards=voucher_cards, booster_packs=booster_packs)


def _parse_phase(raw: Any) -> GamePhase:
    if isinstance(raw, GamePhase):
        return raw
    text = str(raw)
    if text.upper() in PHASE_ALIASES:
        return PHASE_ALIASES[text.upper()]
    try:
        return GamePhase(text.lower())
    except ValueError:
        return GamePhase.UNKNOWN


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _raw_metadata(data: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(data.get("metadata", {})) if isinstance(data.get("metadata"), dict) else {}
    for key in ("id", "key", "set", "label", "value", "modifier", "state", "cost", "rarity", "rarity_name"):
        if key in data:
            metadata[key] = data[key]
    return metadata


def _parse_stake(raw: Any) -> Stake:
    if isinstance(raw, Stake):
        return raw
    text = str(raw).lower()
    try:
        return Stake(text)
    except ValueError:
        return Stake.UNKNOWN


def _area_cards(area: Any) -> tuple[Any, ...]:
    if isinstance(area, dict):
        return tuple(area.get("cards", ()))
    if isinstance(area, list | tuple):
        return tuple(area)
    return ()


def _area_count(area: Any) -> int:
    if isinstance(area, dict):
        if "count" in area:
            return int(area["count"])
        return len(area.get("cards", ()))
    return 0


def _card_label(card: Any) -> str:
    if isinstance(card, str):
        return card
    return str(card.get("label", card.get("name", card.get("key", "unknown"))))


def _current_blind(blinds: Any) -> dict[str, Any]:
    if not isinstance(blinds, dict):
        return {}
    for value in blinds.values():
        if isinstance(value, dict) and value.get("status") in {"SELECT", "CURRENT"}:
            return value
    for key in ("current", "selected", "small", "big", "boss"):
        value = blinds.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _hand_levels(hands: Any) -> dict[str, int]:
    if not isinstance(hands, dict):
        return {}
    levels: dict[str, int] = {}
    for name, value in hands.items():
        if isinstance(value, dict):
            levels[str(name)] = int(value.get("level", 1))
        else:
            levels[str(name)] = int(value)
    return levels


def _with_derived_legal_actions(
    state: GameState,
    *,
    shop_cards: tuple[Any, ...],
    voucher_cards: tuple[Any, ...],
    booster_packs: tuple[Any, ...],
) -> GameState:
    return GameState(
        phase=state.phase,
        stake=state.stake,
        seed=state.seed,
        ante=state.ante,
        blind=state.blind,
        required_score=state.required_score,
        current_score=state.current_score,
        hands_remaining=state.hands_remaining,
        discards_remaining=state.discards_remaining,
        money=state.money,
        deck_size=state.deck_size,
        hand=state.hand,
        known_deck=state.known_deck,
        jokers=state.jokers,
        consumables=state.consumables,
        vouchers=state.vouchers,
        shop=state.shop,
        pack=state.pack,
        hand_levels=state.hand_levels,
        modifiers=state.modifiers,
        legal_actions=_derive_legal_actions(state, shop_cards, voucher_cards, booster_packs),
        run_over=state.run_over,
        won=state.won,
    )


def _derive_legal_actions(
    state: GameState,
    shop_cards: tuple[Any, ...],
    voucher_cards: tuple[Any, ...],
    booster_packs: tuple[Any, ...],
) -> tuple[Action, ...]:
    actions: list[Action] = []
    if state.phase == GamePhase.BLIND_SELECT:
        actions.append(Action(ActionType.SELECT_BLIND))
        current_blind = _mapping_or_empty(state.modifiers.get("current_blind"))
        if current_blind.get("type") != "BOSS":
            actions.append(Action(ActionType.SKIP_BLIND))
    elif state.phase in {GamePhase.SELECTING_HAND, GamePhase.PLAYING_BLIND}:
        max_cards = min(5, len(state.hand))
        for size in range(1, max_cards + 1):
            for indexes in combinations(range(len(state.hand)), size):
                actions.append(Action(ActionType.PLAY_HAND, card_indices=indexes))
                if state.discards_remaining > 0:
                    actions.append(Action(ActionType.DISCARD, card_indices=indexes))
    elif state.phase == GamePhase.ROUND_EVAL:
        actions.append(Action(ActionType.CASH_OUT))
    elif state.phase == GamePhase.SHOP:
        for index in range(len(state.jokers)):
            actions.append(Action(ActionType.SELL, target_id="joker", amount=index, metadata={"kind": "joker", "index": index}))
        for index in range(len(shop_cards)):
            if _is_affordable(shop_cards[index], state.money):
                actions.append(Action(ActionType.BUY, target_id="card", amount=index, metadata={"kind": "card", "index": index}))
        for index in range(len(voucher_cards)):
            if _is_affordable(voucher_cards[index], state.money):
                actions.append(Action(ActionType.BUY, target_id="voucher", amount=index, metadata={"kind": "voucher", "index": index}))
        for index in range(len(booster_packs)):
            if _is_affordable(booster_packs[index], state.money):
                actions.append(Action(ActionType.OPEN_PACK, target_id="pack", amount=index, metadata={"kind": "pack", "index": index}))
        if state.money >= 5:
            actions.append(Action(ActionType.REROLL))
        actions.append(Action(ActionType.END_SHOP))
    elif state.phase == GamePhase.BOOSTER_OPENED:
        for index in range(len(state.pack)):
            actions.append(Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=index, metadata={"kind": "card", "index": index}))
        actions.append(Action(ActionType.CHOOSE_PACK_CARD, target_id="skip", metadata={"kind": "skip", "index": True}))
    elif state.phase not in {GamePhase.RUN_OVER, GamePhase.UNKNOWN}:
        actions.append(Action(ActionType.NO_OP))
    return tuple(actions)


def _with_augmented_sell_actions(state: GameState) -> GameState:
    if state.phase != GamePhase.SHOP or not state.jokers:
        return state
    if any(action.action_type == ActionType.SELL for action in state.legal_actions):
        return state

    sell_actions = tuple(
        Action(ActionType.SELL, target_id="joker", amount=index, metadata={"kind": "joker", "index": index})
        for index in range(len(state.jokers))
    )
    return GameState(
        phase=state.phase,
        stake=state.stake,
        seed=state.seed,
        ante=state.ante,
        blind=state.blind,
        required_score=state.required_score,
        current_score=state.current_score,
        hands_remaining=state.hands_remaining,
        discards_remaining=state.discards_remaining,
        money=state.money,
        deck_size=state.deck_size,
        hand=state.hand,
        known_deck=state.known_deck,
        jokers=state.jokers,
        consumables=state.consumables,
        vouchers=state.vouchers,
        shop=state.shop,
        pack=state.pack,
        hand_levels=state.hand_levels,
        modifiers=state.modifiers,
        legal_actions=sell_actions + state.legal_actions,
        run_over=state.run_over,
        won=state.won,
    )


def _is_affordable(card: Any, money: int) -> bool:
    if not isinstance(card, dict):
        return True
    cost = _mapping_or_empty(card.get("cost"))
    return int(cost.get("buy", 0)) <= money
