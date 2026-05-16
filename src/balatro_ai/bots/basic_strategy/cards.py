"""Card and shop payload adapters used by BasicStrategyBot."""

from __future__ import annotations

import re
from typing import Any

from balatro_ai.api.state import GameState, Joker
from balatro_ai.bots.basic_strategy.data import (
    EARLY_POWER_JOKERS,
    JOKER_SCALING_VALUES,
    PLANET_TO_HAND,
    SPECTRAL_CARD_NAMES,
    TARGET_REQUIRED_TAROTS,
    TAROT_VALUES,
)


def _card_label(card: object) -> str:
    if not isinstance(card, dict):
        return str(card)
    return str(card.get("label", card.get("name", card.get("key", ""))))


def _card_key(card: object) -> str:
    if not isinstance(card, dict):
        return ""
    return str(card.get("key", ""))


def _card_set(card: object) -> str:
    if not isinstance(card, dict):
        return ""
    return str(card.get("set", "")).upper()


def _card_cost(card: object) -> int:
    if not isinstance(card, dict):
        return 0
    cost = card.get("cost", {})
    if isinstance(cost, dict):
        return int(cost.get("buy", cost.get("cost", 0)) or 0)
    return int(cost or 0)


def _card_modifier(card: object) -> dict[str, Any]:
    if not isinstance(card, dict) or not isinstance(card.get("modifier"), dict):
        return {}
    return card["modifier"]


def _card_value(card: object) -> dict[str, Any]:
    if not isinstance(card, dict) or not isinstance(card.get("value"), dict):
        return {}
    return card["value"]


def _card_rank(card: object) -> str:
    if not isinstance(card, dict):
        return ""
    value = _card_value(card)
    rank = str(card.get("rank", value.get("rank", "")))
    return "T" if rank == "10" else rank


def _card_suit(card: object) -> str:
    if not isinstance(card, dict):
        return ""
    value = _card_value(card)
    suit = str(card.get("suit", value.get("suit", "")))
    return {
        "Spades": "S",
        "Spade": "S",
        "Hearts": "H",
        "Heart": "H",
        "Clubs": "C",
        "Club": "C",
        "Diamonds": "D",
        "Diamond": "D",
    }.get(suit, suit)


def _consumable_card_for_name(name: str) -> dict[str, object]:
    if name in PLANET_TO_HAND:
        card_set = "PLANET"
    elif name in TAROT_VALUES:
        card_set = "TAROT"
    elif name in SPECTRAL_CARD_NAMES or name == "Black Hole":
        card_set = "SPECTRAL"
    else:
        card_set = "CONSUMABLE"
    return {"label": name, "set": card_set}


def _has_consumable_room(state: GameState) -> bool:
    return _basic_consumable_open_slots(state) > 0


def _consumable_open_slots_after_storage_use(state: GameState) -> int:
    return _basic_consumable_open_slots(state) + 1


def _basic_consumable_open_slots(state: GameState) -> int:
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


def _pack_card_requires_targets(card: object) -> bool:
    return _is_tarot_card(card) and _card_label(card) in TARGET_REQUIRED_TAROTS


def _is_joker_card(card: object) -> bool:
    if not isinstance(card, dict):
        return False
    label = str(card.get("label", card.get("name", card.get("key", ""))))
    key = str(card.get("key", ""))
    card_set = str(card.get("set", "")).upper()
    return card_set == "JOKER" or "joker" in label.lower() or key.startswith("j_")


def _is_planet_card(card: object) -> bool:
    return _card_set(card) == "PLANET" or _card_label(card) in PLANET_TO_HAND


def _is_black_hole_card(card: object) -> bool:
    return _card_label(card) == "Black Hole" or _card_key(card) == "c_black_hole"


def _is_tarot_card(card: object) -> bool:
    return _card_set(card) == "TAROT" or _card_label(card) in TAROT_VALUES


def _is_spectral_card(card: object) -> bool:
    return _card_set(card) == "SPECTRAL" or _card_label(card) in SPECTRAL_CARD_NAMES


def _is_consumable_card(card: object) -> bool:
    return _is_tarot_card(card) or _is_planet_card(card) or _is_spectral_card(card)


def _is_playing_card(card: object) -> bool:
    if not isinstance(card, dict):
        return False
    card_set = _card_set(card)
    return card_set in {"DEFAULT", "ENHANCED", "PLAYING_CARD"} or (
        _card_rank(card) != "" and _card_suit(card) != ""
    )


def _uses_normal_joker_slot(joker: Joker) -> bool:
    return "negative" not in (joker.edition or "").lower()


def _normal_joker_slots_used(state: GameState) -> int:
    return sum(1 for joker in state.jokers if _uses_normal_joker_slot(joker))


def _normal_joker_slot_limit(state: GameState) -> int:
    for key in ("joker_slot_limit", "joker_slots"):
        raw = state.modifiers.get(key)
        try:
            if raw is not None:
                return max(0, int(raw))
        except (TypeError, ValueError):
            continue
    return 5


def _normal_joker_open_slots(state: GameState) -> int:
    return max(0, _normal_joker_slot_limit(state) - _normal_joker_slots_used(state))


def _joker_would_overfill_slots(state: GameState, joker: Joker) -> bool:
    return _uses_normal_joker_slot(joker) and _normal_joker_open_slots(state) <= 0


def _is_face_card_for_state(state: GameState, card) -> bool:
    if card.debuffed:
        return False
    return card.rank in {"J", "Q", "K"} or any(
        joker.name == "Pareidolia" and not joker.effect.disabled for joker in state.jokers
    )


def _normalize_suit(suit: str) -> str:
    return {
        "Spades": "S",
        "Spade": "S",
        "Hearts": "H",
        "Heart": "H",
        "Clubs": "C",
        "Club": "C",
        "Diamonds": "D",
        "Diamond": "D",
    }.get(suit, suit)


def _rank_matches(card_rank: str, target_rank: str) -> bool:
    if target_rank == "10":
        return card_rank in {"10", "T"}
    return card_rank == target_rank


def _mail_in_rebate_rank_from_text(text: str) -> str | None:
    match = re.search(
        r"\bdiscarded\s+(Ace|King|Queen|Jack|Ten|Nine|Eight|Seven|Six|Five|Four|Three|Two|K|Q|J|10|[2-9])\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return _normalize_rank(match.group(1))


def _normalize_rank(rank: str) -> str:
    value = rank.strip().lower()
    return {
        "ace": "A",
        "king": "K",
        "queen": "Q",
        "jack": "J",
        "ten": "10",
        "nine": "9",
        "eight": "8",
        "seven": "7",
        "six": "6",
        "five": "5",
        "four": "4",
        "three": "3",
        "two": "2",
        "k": "K",
        "q": "Q",
        "j": "J",
        "t": "10",
    }.get(value, value.upper())


def _rank_from_text(text: str) -> str | None:
    match = re.search(
        r"\b(Ace|King|Queen|Jack|Ten|Nine|Eight|Seven|Six|Five|Four|Three|Two|K|Q|J|10|[2-9])\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return _normalize_rank(match.group(1))


def _joker_from_shop_card(card: object) -> Joker:
    if isinstance(card, dict):
        return Joker.from_mapping(card)
    return Joker(str(card))


def _edition_bonus(edition: str | None) -> float:
    if edition is None:
        return 0.0
    text = edition.lower()
    if "negative" in text:
        return 60.0
    if "polychrome" in text:
        return 45.0
    if "holo" in text or "holographic" in text:
        return 24.0
    if "foil" in text:
        return 18.0
    return 0.0


def _edition_chips_value(edition: str | None) -> int:
    if edition is None:
        return 0
    return 50 if "foil" in edition.lower() else 0


def _edition_mult_value(edition: str | None) -> int:
    if edition is None:
        return 0
    text = edition.lower()
    return 10 if "holo" in text or "holographic" in text else 0


def _edition_xmult_value(edition: str | None) -> float:
    if edition is None:
        return 1.0
    return 1.5 if "polychrome" in edition.lower() else 1.0


def _early_power_bonus(name: str) -> float:
    if name in EARLY_POWER_JOKERS:
        return 5.0
    if name in JOKER_SCALING_VALUES:
        return 2.0
    return 0.0


def _is_gold_enhancement(card) -> bool:
    return not card.debuffed and _normalize_card_attr(card.enhancement) in {"gold", "gold card"}


def _is_blue_seal(card) -> bool:
    return not card.debuffed and _normalize_card_attr(card.seal) == "blue"


def _is_gold_seal(card) -> bool:
    return not card.debuffed and _normalize_card_attr(card.seal) == "gold"


def _normalize_card_attr(value: object) -> str:
    return str(value or "").strip().lower().replace("_", " ").replace("-", " ")
