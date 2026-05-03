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

VOUCHER_KEY_TO_NAME = {
    "v_overstock_norm": "Overstock",
    "v_overstock_plus": "Overstock Plus",
    "v_clearance_sale": "Clearance Sale",
    "v_liquidation": "Liquidation",
    "v_hone": "Hone",
    "v_glow_up": "Glow Up",
    "v_reroll_surplus": "Reroll Surplus",
    "v_reroll_glut": "Reroll Glut",
    "v_crystal_ball": "Crystal Ball",
    "v_omen_globe": "Omen Globe",
    "v_telescope": "Telescope",
    "v_observatory": "Observatory",
    "v_grabber": "Grabber",
    "v_nacho_tong": "Nacho Tong",
    "v_wasteful": "Wasteful",
    "v_recyclomancy": "Recyclomancy",
    "v_tarot_merchant": "Tarot Merchant",
    "v_tarot_tycoon": "Tarot Tycoon",
    "v_planet_merchant": "Planet Merchant",
    "v_planet_tycoon": "Planet Tycoon",
    "v_seed_money": "Seed Money",
    "v_money_tree": "Money Tree",
    "v_blank": "Blank",
    "v_antimatter": "Antimatter",
    "v_magic_trick": "Magic Trick",
    "v_illusion": "Illusion",
    "v_hieroglyph": "Hieroglyph",
    "v_petroglyph": "Petroglyph",
    "v_directors_cut": "Director's Cut",
    "v_retcon": "Retcon",
    "v_paint_brush": "Paint Brush",
    "v_palette": "Palette",
}

PLANET_CONSUMABLES = frozenset(
    {
        "Pluto",
        "Mercury",
        "Uranus",
        "Venus",
        "Earth",
        "Mars",
        "Jupiter",
        "Saturn",
        "Neptune",
        "Planet X",
        "Ceres",
        "Eris",
    }
)
TARGETED_CONSUMABLES: dict[str, tuple[int, int]] = {
    "The Magician": (1, 2),
    "The Empress": (1, 2),
    "The Hierophant": (1, 2),
    "The Lovers": (1, 1),
    "The Chariot": (1, 1),
    "Justice": (1, 1),
    "The Devil": (1, 1),
    "The Tower": (1, 1),
    "Strength": (1, 2),
    "The Hanged Man": (1, 2),
    "Death": (2, 2),
    "The Star": (1, 3),
    "The Moon": (1, 3),
    "The Sun": (1, 3),
    "The World": (1, 3),
    "Talisman": (1, 1),
    "Aura": (1, 1),
    "Deja Vu": (1, 1),
    "Trance": (1, 1),
    "Medium": (1, 1),
    "Cryptid": (1, 1),
}
NON_TARGETED_CONSUMABLES = frozenset(
    {
        "The Fool",
        "The High Priestess",
        "The Emperor",
        "The Hermit",
        "The Wheel of Fortune",
        "Temperance",
        "Judgement",
        "Familiar",
        "Grim",
        "Incantation",
        "Wraith",
        "Sigil",
        "Ouija",
        "Ectoplasm",
        "Immolate",
        "Ankh",
        "Hex",
        "The Soul",
        "Black Hole",
    }
)


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
        enhancement = data.get("enhancement", modifier.get("enhancement"))
        return cls(
            rank=str(data.get("rank", value.get("rank"))),
            suit=str(data.get("suit", value.get("suit"))),
            enhancement=enhancement if enhancement is not None else _enhancement_from_label(data),
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
        round_counter_modifiers = _round_counter_modifiers(data, round_data)
        current_blind = _with_round_current_blind(_current_blind(data.get("blinds", {})), round_data, phase)
        hand_cards = _area_cards(data.get("hand", data.get("cards", {})))
        joker_cards = _area_cards(data.get("jokers", {}))
        consumable_cards = _area_cards(data.get("consumables", {}))
        shop_cards = _area_cards(data.get("shop", {}))
        voucher_cards = _area_cards(data.get("vouchers", {}))
        owned_vouchers = _owned_vouchers_from_mapping(data)
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
            vouchers=owned_vouchers,
            shop=tuple(_card_label(item) for item in shop_cards),
            pack=tuple(_card_label(item) for item in pack_cards),
            hand_levels=hand_levels,
            modifiers={
                **dict(data.get("modifiers", {})),
                **round_counter_modifiers,
                "blinds": data.get("blinds", _mapping_or_empty(data.get("modifiers")).get("blinds", {})),
                "current_blind": current_blind,
                "hands": data.get("hands", data.get("hand_levels", {})),
                "joker_cards": joker_cards,
                "owned_vouchers": owned_vouchers,
                "used_vouchers": _raw_used_vouchers(data),
                "shop_cards": shop_cards,
                "voucher_cards": voucher_cards,
                "booster_packs": booster_packs,
                "pack_cards": pack_cards,
            },
            legal_actions=actions_from_mappings(tuple(data.get("legal_actions", ()))),
            run_over=bool(data.get("run_over", phase == GamePhase.RUN_OVER)),
            won=bool(data.get("won", False)),
        )
        if state.ante >= 9:
            won = state.won or not state.run_over
            return replace(state, ante=8, run_over=True, won=won, legal_actions=())
        if state.legal_actions:
            return _with_augmented_consumable_actions(_with_augmented_sell_actions(_with_sanitized_legal_actions(state)))
        return _with_derived_legal_actions(state, shop_cards=shop_cards, voucher_cards=voucher_cards, booster_packs=booster_packs)


def with_derived_legal_actions(
    state: GameState,
    *,
    shop_cards: tuple[Any, ...] | None = None,
    voucher_cards: tuple[Any, ...] | None = None,
    booster_packs: tuple[Any, ...] | None = None,
) -> GameState:
    """Return ``state`` with legal actions derived from its visible surface."""

    derived = _with_derived_legal_actions(
        state,
        shop_cards=_area_cards(state.modifiers.get("shop_cards", ())) if shop_cards is None else tuple(shop_cards),
        voucher_cards=_area_cards(state.modifiers.get("voucher_cards", ())) if voucher_cards is None else tuple(voucher_cards),
        booster_packs=_area_cards(state.modifiers.get("booster_packs", ())) if booster_packs is None else tuple(booster_packs),
    )
    return _with_augmented_consumable_actions(_with_augmented_sell_actions(_with_sanitized_legal_actions(derived)))


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


def _round_counter_modifiers(data: dict[str, Any], round_data: Any) -> dict[str, Any]:
    counters: dict[str, Any] = {}
    round_mapping = _mapping_or_empty(round_data)
    if round_mapping:
        counters["round"] = dict(round_mapping)
    if "discards_used" in round_mapping:
        counters["round_discards_used"] = round_mapping["discards_used"]
        counters["discards_used"] = round_mapping["discards_used"]
    if "hands_played" in round_mapping:
        counters["round_hands_played"] = round_mapping["hands_played"]
        counters["hands_played"] = round_mapping["hands_played"]
    if "reroll_cost" in round_mapping:
        counters["reroll_cost"] = round_mapping["reroll_cost"]
        counters["current_reroll_cost"] = round_mapping["reroll_cost"]
    if "dollars" in round_mapping:
        counters["current_round_dollars"] = round_mapping["dollars"]
        counters["cash_out_money_delta"] = round_mapping["dollars"]
    for key in (
        "blind_reward",
        "blind_score",
        "blind_on_deck",
        "blind_states",
        "money_per_hand",
        "money_per_discard",
        "no_extra_hand_money",
        "no_interest",
        "no_blind_reward",
        "interest_amount",
        "interest_cap",
    ):
        if key in round_mapping:
            counters[key] = round_mapping[key]
    for key in ("round_discards_used", "discards_used", "round_hands_played", "hands_played"):
        if key in data:
            counters[key] = data[key]
    return counters


def _raw_metadata(data: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(data.get("metadata", {})) if isinstance(data.get("metadata"), dict) else {}
    for key, value in data.items():
        if key != "metadata":
            metadata.setdefault(key, value)
    return metadata


def _enhancement_from_label(data: dict[str, Any]) -> str | None:
    label = str(data.get("label", "")).strip().lower()
    card_set = str(data.get("set", "")).strip().lower()
    if card_set not in {"enhanced", "default", ""}:
        return None
    mapping = {
        "bonus card": "BONUS",
        "mult card": "MULT",
        "wild card": "WILD",
        "glass card": "GLASS",
        "steel card": "STEEL",
        "stone card": "STONE",
        "gold card": "GOLD",
        "lucky card": "LUCKY",
    }
    return mapping.get(label)


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


def _owned_vouchers_from_mapping(data: dict[str, Any]) -> tuple[str, ...]:
    for key in ("owned_vouchers", "redeemed_vouchers"):
        labels = _voucher_labels_from_value(data.get(key))
        if labels:
            return labels
    return _voucher_labels_from_used_vouchers(data.get("used_vouchers"))


def _raw_used_vouchers(data: dict[str, Any]) -> Any:
    used_vouchers = data.get("used_vouchers")
    if used_vouchers is not None:
        return used_vouchers
    owned_vouchers = data.get("owned_vouchers")
    return owned_vouchers if owned_vouchers is not None else ()


def _voucher_labels_from_used_vouchers(value: Any) -> tuple[str, ...]:
    if isinstance(value, dict):
        return tuple(_voucher_key_to_name(str(key)) for key in sorted(value))
    return _voucher_labels_from_value(value)


def _voucher_labels_from_value(value: Any) -> tuple[str, ...]:
    if isinstance(value, dict):
        return tuple(_voucher_key_to_name(str(key)) for key in sorted(value))
    if isinstance(value, list | tuple):
        return tuple(_voucher_label(item) for item in value if isinstance(item, str | dict))
    return ()


def _voucher_label(item: Any) -> str:
    if isinstance(item, str):
        return _voucher_key_to_name(item)
    return _voucher_key_to_name(_card_label(item))


def _voucher_key_to_name(value: str) -> str:
    return VOUCHER_KEY_TO_NAME.get(value, value)


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


def _with_round_current_blind(current_blind: dict[str, Any], round_data: Any, phase: GamePhase) -> dict[str, Any]:
    round_mapping = _mapping_or_empty(round_data)
    active_blind_phases = {GamePhase.SELECTING_HAND, GamePhase.PLAYING_BLIND, GamePhase.ROUND_EVAL}
    if phase not in active_blind_phases or "blind_name" not in round_mapping:
        return current_blind
    updated = dict(current_blind)
    updated["name"] = round_mapping["blind_name"]
    if "blind_type" in round_mapping:
        updated["type"] = str(round_mapping["blind_type"]).upper()
    if "blind_reward" in round_mapping:
        updated["dollars"] = round_mapping["blind_reward"]
    if "blind_score" in round_mapping:
        updated["score"] = round_mapping["blind_score"]
    return updated


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
        actions.extend(_consumable_sell_actions(state))
        actions.extend(_consumable_use_actions(state, allow_hand_targets=True))
    elif state.phase == GamePhase.ROUND_EVAL:
        actions.append(Action(ActionType.CASH_OUT))
    elif state.phase == GamePhase.SHOP:
        for index in range(len(state.jokers)):
            actions.append(Action(ActionType.SELL, target_id="joker", amount=index, metadata={"kind": "joker", "index": index}))
        actions.extend(_consumable_sell_actions(state))
        actions.extend(_consumable_use_actions(state, allow_hand_targets=False))
        for index in range(len(shop_cards)):
            if _is_affordable(shop_cards[index], state.money) and _shop_card_can_be_bought(state, shop_cards[index]):
                actions.append(Action(ActionType.BUY, target_id="card", amount=index, metadata={"kind": "card", "index": index}))
        for index in range(len(voucher_cards)):
            if _is_affordable(voucher_cards[index], state.money):
                actions.append(Action(ActionType.BUY, target_id="voucher", amount=index, metadata={"kind": "voucher", "index": index}))
        for index in range(len(booster_packs)):
            if _is_affordable(booster_packs[index], state.money):
                actions.append(Action(ActionType.OPEN_PACK, target_id="pack", amount=index, metadata={"kind": "pack", "index": index}))
        if _reroll_cost(state) <= state.money:
            actions.append(Action(ActionType.REROLL))
        actions.append(Action(ActionType.END_SHOP))
    elif state.phase == GamePhase.BOOSTER_OPENED:
        for index in range(len(state.jokers)):
            actions.append(Action(ActionType.SELL, target_id="joker", amount=index, metadata={"kind": "joker", "index": index}))
        actions.extend(_consumable_sell_actions(state))
        actions.extend(_consumable_use_actions(state, allow_hand_targets=True))
        if state.pack:
            pack_cards = _area_cards(state.modifiers.get("pack_cards", ()))
            for index in range(len(state.pack)):
                item = pack_cards[index] if index < len(pack_cards) else state.pack[index]
                for card_indices in _consumable_target_index_sets(_card_label(item), state, allow_hand_targets=True):
                    actions.append(
                        Action(
                            ActionType.CHOOSE_PACK_CARD,
                            card_indices=card_indices,
                            target_id="card",
                            amount=index,
                            metadata={"kind": "card", "index": index},
                        )
                    )
            actions.append(Action(ActionType.CHOOSE_PACK_CARD, target_id="skip", metadata={"kind": "skip", "index": True}))
        else:
            actions.append(Action(ActionType.NO_OP))
    elif state.phase not in {GamePhase.RUN_OVER, GamePhase.UNKNOWN}:
        actions.append(Action(ActionType.NO_OP))
    return tuple(actions)


def _with_sanitized_legal_actions(state: GameState) -> GameState:
    legal_actions = state.legal_actions
    if state.phase == GamePhase.BOOSTER_OPENED and not state.pack:
        legal_actions = tuple(action for action in legal_actions if action.action_type != ActionType.CHOOSE_PACK_CARD)
        if not legal_actions:
            legal_actions = (Action(ActionType.NO_OP),)
    if state.phase != GamePhase.SHOP:
        if legal_actions == state.legal_actions:
            return state
        return replace(state, legal_actions=legal_actions)

    legal_actions = tuple(action for action in legal_actions if _action_can_be_taken(state, action))
    if legal_actions == state.legal_actions:
        return state
    return replace(state, legal_actions=legal_actions)


def _action_can_be_taken(state: GameState, action: Action) -> bool:
    if action.action_type != ActionType.BUY:
        return True
    kind = str(action.metadata.get("kind", action.target_id or ""))
    if kind != "card":
        return True
    index = _action_index(action)
    shop_cards = state.modifiers.get("shop_cards", ())
    if index is None or not 0 <= index < len(shop_cards):
        return True
    return _shop_card_can_be_bought(state, shop_cards[index])


def _action_index(action: Action) -> int | None:
    raw = action.metadata.get("index", action.amount)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _consumable_sell_actions(state: GameState) -> tuple[Action, ...]:
    return tuple(
        Action(ActionType.SELL, target_id="consumable", amount=index, metadata={"kind": "consumable", "index": index})
        for index in range(len(state.consumables))
    )


def _consumable_use_actions(state: GameState, *, allow_hand_targets: bool) -> tuple[Action, ...]:
    actions: list[Action] = []
    for index, name in enumerate(state.consumables):
        for card_indices in _consumable_target_index_sets(name, state, allow_hand_targets=allow_hand_targets):
            actions.append(
                Action(
                    ActionType.USE_CONSUMABLE,
                    card_indices=card_indices,
                    target_id="consumable",
                    amount=index,
                    metadata={"kind": "consumable", "index": index},
                )
            )
    return tuple(actions)


def _consumable_target_index_sets(
    name: str,
    state: GameState,
    *,
    allow_hand_targets: bool,
) -> tuple[tuple[int, ...], ...]:
    if name in PLANET_CONSUMABLES:
        return ((),)
    if name in NON_TARGETED_CONSUMABLES:
        if name in {"The Wheel of Fortune", "Ankh", "Hex", "Ectoplasm"} and not state.jokers:
            return ()
        if name in {"Hex", "Ectoplasm"} and not any(joker.edition is None for joker in state.jokers):
            return ()
        return ((),)
    bounds = TARGETED_CONSUMABLES.get(name)
    if bounds is None:
        return ((),)
    if not allow_hand_targets:
        return ()
    min_count, max_count = bounds
    max_count = min(max_count, len(state.hand))
    if max_count < min_count:
        return ()
    indexes = range(len(state.hand))
    return tuple(combo for size in range(min_count, max_count + 1) for combo in combinations(indexes, size))


def _shop_card_can_be_bought(state: GameState, card: Any) -> bool:
    if not _is_joker_shop_card(card):
        return True
    if not _shop_joker_uses_normal_slot(card):
        return True
    return _normal_joker_slots_used(state) < _normal_joker_slot_limit(state)


def _shop_joker_uses_normal_slot(card: Any) -> bool:
    if not isinstance(card, dict):
        return True
    edition = card.get("edition")
    modifier = card.get("modifier")
    if edition is None and isinstance(modifier, dict):
        edition = modifier.get("edition")
    if isinstance(edition, dict):
        edition = edition.get("type") or edition.get("key") or edition.get("name") or edition.get("edition")
    return "negative" not in str(edition or "").lower()


def _is_joker_shop_card(card: Any) -> bool:
    if not isinstance(card, dict):
        return False
    label = str(card.get("label", card.get("name", card.get("key", ""))))
    key = str(card.get("key", ""))
    card_set = str(card.get("set", "")).upper()
    return card_set == "JOKER" or "joker" in label.lower() or key.startswith("j_")


def _normal_joker_slots_used(state: GameState) -> int:
    return sum(1 for joker in state.jokers if _joker_uses_normal_slot(joker))


def _joker_uses_normal_slot(joker: Joker) -> bool:
    return "negative" not in (joker.edition or "").lower()


def _normal_joker_slot_limit(state: GameState) -> int:
    for key in ("joker_slot_limit", "joker_slots"):
        raw = state.modifiers.get(key)
        try:
            if raw is not None:
                return max(0, int(raw))
        except (TypeError, ValueError):
            continue
    return 5


def _with_augmented_sell_actions(state: GameState) -> GameState:
    if state.phase not in {GamePhase.SHOP, GamePhase.BOOSTER_OPENED} or (not state.jokers and not state.consumables):
        return state
    existing = {
        (str(action.metadata.get("kind", action.target_id or "")), _action_index(action))
        for action in state.legal_actions
        if action.action_type == ActionType.SELL
    }
    sell_actions = tuple(
        action
        for action in (
            tuple(
                Action(ActionType.SELL, target_id="joker", amount=index, metadata={"kind": "joker", "index": index})
                for index in range(len(state.jokers))
            )
            + _consumable_sell_actions(state)
        )
        if (str(action.metadata.get("kind", action.target_id or "")), _action_index(action)) not in existing
    )
    if not sell_actions:
        return state
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


def _with_augmented_consumable_actions(state: GameState) -> GameState:
    phases = {GamePhase.SELECTING_HAND, GamePhase.PLAYING_BLIND, GamePhase.SHOP, GamePhase.BOOSTER_OPENED}
    if state.phase not in phases or not state.consumables:
        return state
    additions = _consumable_use_actions(state, allow_hand_targets=state.phase != GamePhase.SHOP) + _consumable_sell_actions(state)
    existing = {
        (
            action.action_type,
            tuple(action.card_indices),
            str(action.metadata.get("kind", action.target_id or "")),
            _action_index(action),
        )
        for action in state.legal_actions
    }
    new_actions = tuple(
        action
        for action in additions
        if (
            action.action_type,
            tuple(action.card_indices),
            str(action.metadata.get("kind", action.target_id or "")),
            _action_index(action),
        )
        not in existing
    )
    if not new_actions:
        return state
    return replace(state, legal_actions=state.legal_actions + new_actions)


def _is_affordable(card: Any, money: int) -> bool:
    if not isinstance(card, dict):
        return True
    cost = _mapping_or_empty(card.get("cost"))
    return int(cost.get("buy", 0)) <= money


def _reroll_cost(state: GameState) -> int:
    try:
        if int(state.modifiers.get("free_rerolls", 0)) > 0:
            return 0
    except (TypeError, ValueError):
        pass
    for key in ("reroll_cost", "current_reroll_cost"):
        raw = state.modifiers.get(key)
        try:
            if raw is not None:
                return max(0, int(raw))
        except (TypeError, ValueError):
            continue
    return 5
