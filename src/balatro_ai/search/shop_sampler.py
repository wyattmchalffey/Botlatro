"""Shop-pool sampling for Phase 7 shop search.

The sampler mirrors Balatro's source-level shop distributions, but it does
not try to replay Lua's seeded pseudorandom stream. Search needs clean
possible outcomes; exact seed prediction belongs in replay validation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from random import Random
from typing import Any, Callable, TypeVar

from balatro_ai.api.state import Card, GameState

SHOP_POOL_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "shop_pools.json"

RARITY_ORDER = ("common", "uncommon", "rare", "legendary")
SHOP_TYPE_TO_DATA_KEY = {
    "Joker": "jokers",
    "Tarot": "tarots",
    "Planet": "planets",
    "Spectral": "spectrals",
}
TYPE_RATE_KEYS = {
    "Joker": "joker_rate",
    "Tarot": "tarot_rate",
    "Planet": "planet_rate",
    "Base": "playing_card_rate",
    "Spectral": "spectral_rate",
}
EDITION_EXTRA_COSTS = {
    "FOIL": 2,
    "HOLOGRAPHIC": 3,
    "POLYCHROME": 5,
    "NEGATIVE": 5,
}
VOUCHER_KEYS_BY_NAME = {
    "Overstock": "v_overstock_norm",
    "Overstock Plus": "v_overstock_plus",
    "Clearance Sale": "v_clearance_sale",
    "Liquidation": "v_liquidation",
    "Hone": "v_hone",
    "Glow Up": "v_glow_up",
    "Reroll Surplus": "v_reroll_surplus",
    "Reroll Glut": "v_reroll_glut",
    "Tarot Merchant": "v_tarot_merchant",
    "Tarot Tycoon": "v_tarot_tycoon",
    "Planet Merchant": "v_planet_merchant",
    "Planet Tycoon": "v_planet_tycoon",
    "Magic Trick": "v_magic_trick",
    "Illusion": "v_illusion",
}
STANDARD_RANKS = ("A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2")
STANDARD_SUITS = ("S", "H", "D", "C")
ENHANCEMENT_BY_CENTER_KEY = {
    "m_bonus": "BONUS",
    "m_mult": "MULT",
    "m_wild": "WILD",
    "m_glass": "GLASS",
    "m_steel": "STEEL",
    "m_stone": "STONE",
    "m_gold": "GOLD",
    "m_lucky": "LUCKY",
}


@dataclass(frozen=True, slots=True)
class ShopSampler:
    """Sample possible shop outcomes from source-derived pools."""

    data: Mapping[str, Any]

    @classmethod
    def from_default_data(cls) -> "ShopSampler":
        return cls(load_shop_pool_data())

    def sample_shop(self, state: GameState, n_slots: int | None = None, rng: Random | None = None) -> tuple[dict[str, Any], ...]:
        """Sample the rerollable shop-card slots."""

        rng = rng or Random()
        slots = self.shop_slot_count(state) if n_slots is None else max(0, n_slots)
        used_jokers = set(_used_joker_identifiers(state))
        allow_duplicates = _allow_duplicate_shop_cards(state)
        cards: list[dict[str, Any]] = []
        for slot_index in range(slots):
            card_type = self.sample_shop_type(state, rng)
            card = self.sample_card_of_type(
                state,
                card_type,
                rng,
                used_jokers=used_jokers,
                key_append=f"sho{slot_index}",
            )
            cards.append(card)
            if not allow_duplicates and _is_joker_payload(card):
                used_jokers.update(_item_identifiers(card))
        return tuple(cards)

    def sample_shop_type(self, state: GameState, rng: Random | None = None) -> str:
        """Sample the type used by create_card_for_shop."""

        rng = rng or Random()
        rates = _shop_type_rates(self.data, state)
        selected = _weighted_choice(tuple(rates.items()), rng, default=("Joker", 1.0))[0]
        if selected == "Base" and _has_voucher(state, "Illusion") and rng.random() > 0.6:
            return "Enhanced"
        return selected

    def sample_card_of_type(
        self,
        state: GameState,
        card_type: str,
        rng: Random | None = None,
        *,
        used_jokers: set[str] | None = None,
        key_append: str = "",
    ) -> dict[str, Any]:
        """Sample one source-center payload for a shop-card type."""

        rng = rng or Random()
        if used_jokers is None:
            used_jokers = set(_used_joker_identifiers(state))
        if card_type == "Joker":
            record = self._sample_joker_record(state, rng, used_jokers=used_jokers)
            edition = _poll_edition(state, rng)
            return _payload_from_record(record, state, edition=edition, key_append=key_append)
        if card_type in {"Base", "Enhanced"}:
            return _sample_playing_card_payload(self.data, state, card_type, rng)

        records = self._pool_records(card_type, state)
        if not records:
            records = _fallback_pool(self.data, card_type)
        record = rng.choice(records)
        return _payload_from_record(record, state, key_append=key_append)

    def sample_voucher(self, state: GameState, rng: Random | None = None) -> dict[str, Any] | None:
        """Sample the fixed voucher slot for a fresh shop."""

        rng = rng or Random()
        records = self._pool_records("Voucher", state)
        if not records:
            return None
        return _payload_from_record(rng.choice(records), state)

    def sample_boosters(self, state: GameState, n_slots: int | None = None, rng: Random | None = None) -> tuple[dict[str, Any], ...]:
        """Sample booster packs for a fresh shop."""

        rng = rng or Random()
        slots = _booster_slot_count(self.data) if n_slots is None else max(0, n_slots)
        records = self._pool_records("Booster", state)
        if not records:
            return ()
        forced_first_buffoon = state.modifiers.get("first_shop_buffoon") is False
        packs: list[dict[str, Any]] = []
        for index in range(slots):
            if index == 0 and forced_first_buffoon:
                candidates = [record for record in records if str(record.get("key", "")).startswith("p_buffoon_normal")]
                record = rng.choice(candidates or records)
            else:
                record = _weighted_choice(tuple((record, float(record.get("weight", 1))) for record in records), rng, default=(records[0], 1))[0]
            packs.append(_payload_from_record(record, state))
        return tuple(packs)

    def reroll_cost(self, state: GameState) -> int:
        """Return the current reroll cost, matching calculate_reroll_cost state."""

        free_rerolls = _int_value(state.modifiers.get("free_rerolls"))
        if free_rerolls > 0:
            return 0
        explicit = _first_int(state.modifiers, ("reroll_cost", "current_reroll_cost"))
        if explicit is not None:
            return max(0, explicit)

        reset_cost = _first_int(state.modifiers, ("round_reset_reroll_cost", "base_reroll_cost"))
        if reset_cost is None:
            reset_cost = int(_mapping(self.data.get("reroll")).get("base_cost", 5))
        increase = _int_value(state.modifiers.get("reroll_cost_increase"))
        return max(0, reset_cost + increase)

    def reroll_ev(
        self,
        state: GameState,
        *,
        samples: int = 64,
        rng: Random | None = None,
        item_value_fn: Callable[[GameState, Mapping[str, Any]], float] | None = None,
    ) -> float:
        """Estimate net EV of one reroll from sampled rerollable shop slots.

        The value function returns the net value of a visible shop item. The
        default delegates to Basic Strategy's existing shop action value, which
        already includes money pressure and item cost.
        """

        if samples <= 0:
            raise ValueError("reroll_ev samples must be positive")
        rng = rng or Random()
        value_fn = item_value_fn or basic_strategy_shop_item_value
        cost = self.reroll_cost(state)
        if cost > state.money:
            return -float(cost)

        from balatro_ai.api.actions import Action, ActionType
        from balatro_ai.search.forward_sim import simulate_reroll

        try:
            post_reroll_state = simulate_reroll(state, Action(ActionType.REROLL), ())
        except ValueError:
            return -float(cost)

        total = 0.0
        for _ in range(samples):
            shop_cards = self.sample_shop(post_reroll_state, rng=rng)
            best = max((value_fn(post_reroll_state, item) for item in shop_cards), default=0.0)
            total += max(0.0, best)
        return (total / samples) - cost

    def shop_slot_count(self, state: GameState) -> int:
        explicit = _first_int(state.modifiers, ("shop_joker_max", "shop_slots", "shop_card_slots"))
        if explicit is not None:
            return max(0, explicit)
        slots = int(_mapping(self.data.get("shop_slots")).get("base_joker_max", 2))
        if _has_voucher(state, "Overstock"):
            slots += 1
        if _has_voucher(state, "Overstock Plus"):
            slots += 1
        return slots

    def _sample_joker_record(self, state: GameState, rng: Random, *, used_jokers: set[str]) -> Mapping[str, Any]:
        rarity = _weighted_choice(tuple(_joker_rarity_rates(self.data).items()), rng, default=("common", 1.0))[0]
        records = [
            record
            for record in self._pool_records("Joker", state)
            if str(record.get("rarity_name", "")) == rarity and (_allow_duplicate_shop_cards(state) or not used_jokers.intersection(_item_identifiers(record)))
        ]
        if not records:
            records = [
                record
                for record in self._pool_records("Joker", state)
                if _allow_duplicate_shop_cards(state) or not used_jokers.intersection(_item_identifiers(record))
            ]
        if not records:
            records = _fallback_pool(self.data, "Joker")
        return rng.choice(records)

    def _pool_records(self, pool_type: str, state: GameState) -> list[Mapping[str, Any]]:
        if pool_type in SHOP_TYPE_TO_DATA_KEY:
            raw = self.data.get(SHOP_TYPE_TO_DATA_KEY[pool_type], ())
        else:
            raw = self.data.get(f"{pool_type.lower()}s", ())
        records = [record for record in raw if isinstance(record, Mapping)]
        if pool_type in {"Tarot", "Planet", "Spectral"}:
            records = [record for record in records if str(record.get("name")) not in {"Black Hole", "The Soul"}]
        return [record for record in records if _record_available(record, state)]


def load_shop_pool_data(path: Path | None = None) -> dict[str, Any]:
    data_path = path or SHOP_POOL_DATA_PATH
    return json.loads(data_path.read_text(encoding="utf-8"))


def basic_strategy_shop_item_value(state: GameState, item: Mapping[str, Any]) -> float:
    """Value one visible shop-card item through the current rule bot heuristic."""

    from balatro_ai.api.actions import Action, ActionType
    from balatro_ai.bots.basic_strategy_bot import _ShopContext, _shop_action_value, _shop_pressure

    if _item_buy_cost(item) > state.money:
        return 0.0

    temp_modifiers = dict(state.modifiers)
    temp_modifiers["shop_cards"] = (item,)
    temp_state = GameState(
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
        shop=(str(item.get("name", item.get("key", "unknown"))),),
        pack=state.pack,
        hand_levels=state.hand_levels,
        modifiers=temp_modifiers,
        legal_actions=state.legal_actions,
        run_over=state.run_over,
        won=state.won,
    )
    action = Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0})
    return _shop_action_value(temp_state, action, _shop_pressure(temp_state), _ShopContext())


def _item_buy_cost(item: Mapping[str, Any]) -> int:
    cost = item.get("cost", {})
    if isinstance(cost, Mapping):
        return _int_value(cost.get("buy", cost.get("cost", 0)))
    return _int_value(cost)


def _shop_type_rates(data: Mapping[str, Any], state: GameState) -> dict[str, float]:
    rates = {str(key): float(value) for key, value in _mapping(data.get("slot_rates")).items()}
    explicit_rates = _mapping(state.modifiers.get("shop_rates"))
    for key, value in explicit_rates.items():
        if _is_number(value):
            rates[str(key)] = float(value)

    for shop_type, modifier_key in TYPE_RATE_KEYS.items():
        if modifier_key in state.modifiers and _is_number(state.modifiers[modifier_key]):
            rates[shop_type] = float(state.modifiers[modifier_key])

    if "tarot_rate" not in state.modifiers:
        if _has_voucher(state, "Tarot Tycoon"):
            rates["Tarot"] = 128.0
        elif _has_voucher(state, "Tarot Merchant"):
            rates["Tarot"] = 38.4
    if "planet_rate" not in state.modifiers:
        if _has_voucher(state, "Planet Tycoon"):
            rates["Planet"] = 128.0
        elif _has_voucher(state, "Planet Merchant"):
            rates["Planet"] = 38.4
    if "playing_card_rate" not in state.modifiers and (_has_voucher(state, "Magic Trick") or _has_voucher(state, "Illusion")):
        rates["Base"] = 4.0
    if "spectral_rate" in state.modifiers and _is_number(state.modifiers["spectral_rate"]):
        rates["Spectral"] = float(state.modifiers["spectral_rate"])
    return {key: max(0.0, value) for key, value in rates.items() if value > 0}


def _joker_rarity_rates(data: Mapping[str, Any]) -> dict[str, float]:
    raw = _mapping(data.get("joker_rarity_rates"))
    return {rarity: float(raw.get(rarity, 0.0)) for rarity in RARITY_ORDER if float(raw.get(rarity, 0.0)) > 0}


def _record_available(record: Mapping[str, Any], state: GameState) -> bool:
    banned = _identifier_set(state.modifiers.get("banned_keys"))
    if _item_identifiers(record).intersection(banned):
        return False

    unlocked = _identifier_set(state.modifiers.get("unlocked_keys"))
    if unlocked and not _item_identifiers(record).intersection(unlocked):
        return False

    pool_flags = _mapping(state.modifiers.get("pool_flags"))
    no_pool_flag = record.get("no_pool_flag")
    if no_pool_flag and pool_flags.get(str(no_pool_flag)):
        return False
    yes_pool_flag = record.get("yes_pool_flag")
    if yes_pool_flag and not pool_flags.get(str(yes_pool_flag)):
        return False

    if str(record.get("set", "")).upper() == "VOUCHER":
        return _voucher_available(record, state)
    if record.get("enhancement_gate"):
        return _state_has_enhancement_gate(state, str(record["enhancement_gate"]))
    return True


def _voucher_available(record: Mapping[str, Any], state: GameState) -> bool:
    owned = _owned_voucher_identifiers(state)
    if _item_identifiers(record).intersection(owned):
        return False
    required = tuple(str(item) for item in record.get("requires", ()) if item)
    if required and not all(req in owned or _voucher_name_for_key(req) in owned for req in required):
        return False
    return record.get("available_default") is not False


def _state_has_enhancement_gate(state: GameState, gate_key: str) -> bool:
    enhancement = ENHANCEMENT_BY_CENTER_KEY.get(gate_key)
    if enhancement is None:
        return False
    cards: Iterable[Card] = state.hand + state.known_deck
    return any(card.enhancement == enhancement for card in cards)


def _payload_from_record(
    record: Mapping[str, Any],
    state: GameState,
    *,
    edition: str | None = None,
    key_append: str = "",
) -> dict[str, Any]:
    base_cost = _int_value(record.get("cost"))
    cost = _effective_cost(state, base_cost, str(record.get("set", "")), edition=edition)
    payload: dict[str, Any] = {
        "key": record.get("key"),
        "name": record.get("name", record.get("key")),
        "label": record.get("name", record.get("key")),
        "set": str(record.get("set", "")).upper(),
        "cost": {"buy": cost, "base": base_cost},
        "metadata": {
            "source_key": record.get("key"),
            "base_cost": base_cost,
            "shop_sampler": True,
        },
    }
    for key in ("rarity", "rarity_name", "kind", "weight", "config"):
        if key in record:
            payload[key] = deepcopy(record[key])
    if edition:
        payload["edition"] = edition
        payload["modifier"] = {"edition": edition}
        payload["metadata"]["edition_key"] = key_append
    return payload


def _sample_playing_card_payload(
    data: Mapping[str, Any],
    state: GameState,
    card_type: str,
    rng: Random,
) -> dict[str, Any]:
    rank = rng.choice(STANDARD_RANKS)
    suit = rng.choice(STANDARD_SUITS)
    enhancement = None
    if card_type == "Enhanced":
        enhancement = rng.choice(tuple(ENHANCEMENT_BY_CENTER_KEY.values()))
    edition = _poll_illusion_playing_card_edition(state, rng) if _has_voucher(state, "Illusion") else None
    base_cost = int(_mapping(data.get("playing_cards")).get("base_cost", 1))
    cost = _effective_cost(state, base_cost, card_type, edition=edition)
    payload: dict[str, Any] = {
        "key": f"{suit}_{_rank_key(rank)}",
        "name": f"{rank}{suit}",
        "label": f"{rank}{suit}",
        "set": card_type.upper(),
        "rank": rank,
        "suit": suit,
        "value": {"rank": rank, "suit": suit},
        "cost": {"buy": cost, "base": base_cost},
        "metadata": {"shop_sampler": True, "base_cost": base_cost},
    }
    if enhancement:
        payload["enhancement"] = enhancement
        payload["modifier"] = {"enhancement": enhancement}
    if edition:
        payload["edition"] = edition
        modifier = dict(payload.get("modifier", {}))
        modifier["edition"] = edition
        payload["modifier"] = modifier
    return payload


def _poll_edition(state: GameState, rng: Random) -> str | None:
    edition_rate = _edition_rate(state)
    poll = rng.random()
    if poll > 1 - 0.003:
        return "NEGATIVE"
    if poll > 1 - (0.006 * edition_rate):
        return "POLYCHROME"
    if poll > 1 - (0.02 * edition_rate):
        return "HOLOGRAPHIC"
    if poll > 1 - (0.04 * edition_rate):
        return "FOIL"
    return None


def _poll_illusion_playing_card_edition(state: GameState, rng: Random) -> str | None:
    if rng.random() <= 0.8:
        return None
    poll = rng.random()
    if poll > 0.85:
        return "POLYCHROME"
    if poll > 0.5:
        return "HOLOGRAPHIC"
    return "FOIL"


def _edition_rate(state: GameState) -> float:
    explicit = state.modifiers.get("edition_rate")
    if _is_number(explicit):
        return max(0.0, float(explicit))
    if _has_voucher(state, "Glow Up"):
        return 4.0
    if _has_voucher(state, "Hone"):
        return 2.0
    return 1.0


def _effective_cost(state: GameState, base_cost: int, card_set: str, *, edition: str | None = None) -> int:
    extra_cost = _int_value(state.modifiers.get("inflation"))
    if edition:
        extra_cost += EDITION_EXTRA_COSTS.get(edition, 0)
    discount_percent = _discount_percent(state)
    cost = int(((base_cost + extra_cost + 0.5) * (100 - discount_percent)) // 100)
    cost = max(1, cost)
    if card_set.upper() == "BOOSTER" and state.modifiers.get("booster_ante_scaling"):
        cost += max(0, state.ante - 1)
    return cost


def _discount_percent(state: GameState) -> int:
    explicit = state.modifiers.get("discount_percent")
    if _is_number(explicit):
        return max(0, min(100, int(explicit)))
    if _has_voucher(state, "Liquidation"):
        return 50
    if _has_voucher(state, "Clearance Sale"):
        return 25
    return 0


def _booster_slot_count(data: Mapping[str, Any]) -> int:
    return int(_mapping(data.get("shop_slots")).get("booster_slots", 2))


def _fallback_pool(data: Mapping[str, Any], pool_type: str) -> list[Mapping[str, Any]]:
    key = SHOP_TYPE_TO_DATA_KEY.get(pool_type, f"{pool_type.lower()}s")
    records = [record for record in data.get(key, ()) if isinstance(record, Mapping)]
    if records:
        return records[:1]
    return [{"key": "j_joker", "name": "Joker", "set": "Joker", "cost": 2, "rarity_name": "common"}]


T = TypeVar("T")


def _weighted_choice(items: Sequence[tuple[T, float]], rng: Random, *, default: tuple[T, float]) -> tuple[T, float]:
    choices = tuple((item, max(0.0, weight)) for item, weight in items if weight > 0)
    if not choices:
        return default
    total = sum(weight for _, weight in choices)
    poll = rng.random() * total
    seen = 0.0
    for item, weight in choices:
        seen += weight
        if poll <= seen:
            return item, weight
    return choices[-1]


def _allow_duplicate_shop_cards(state: GameState) -> bool:
    if state.modifiers.get("allow_duplicate_jokers") or state.modifiers.get("showman"):
        return True
    return any(joker.name == "Showman" for joker in state.jokers)


def _used_joker_identifiers(state: GameState) -> set[str]:
    identifiers: set[str] = set()
    for joker in state.jokers:
        identifiers.add(joker.name)
        identifiers.update(_identifier_set(joker.metadata.get("key")))
    identifiers.update(_identifier_set(state.modifiers.get("used_jokers")))
    return identifiers


def _owned_voucher_identifiers(state: GameState) -> set[str]:
    identifiers = set(state.vouchers)
    for name in state.vouchers:
        key = VOUCHER_KEYS_BY_NAME.get(name)
        if key:
            identifiers.add(key)
    identifiers.update(_identifier_set(state.modifiers.get("owned_vouchers")))
    identifiers.update(_identifier_set(state.modifiers.get("used_vouchers")))
    return identifiers


def _has_voucher(state: GameState, name_or_key: str) -> bool:
    identifiers = _owned_voucher_identifiers(state)
    if name_or_key in identifiers:
        return True
    key = VOUCHER_KEYS_BY_NAME.get(name_or_key)
    return key in identifiers if key else False


def _voucher_name_for_key(key: str) -> str:
    for name, candidate in VOUCHER_KEYS_BY_NAME.items():
        if candidate == key:
            return name
    return key


def _item_identifiers(item: Mapping[str, Any]) -> set[str]:
    identifiers = {str(value) for value in (item.get("key"), item.get("name"), item.get("label")) if value}
    return identifiers


def _is_joker_payload(item: Mapping[str, Any]) -> bool:
    return str(item.get("set", "")).upper() == "JOKER" or str(item.get("key", "")).startswith("j_")


def _identifier_set(raw: object) -> set[str]:
    if raw is None:
        return set()
    if isinstance(raw, str):
        return {raw}
    if isinstance(raw, Mapping):
        identifiers: set[str] = set()
        for key, value in raw.items():
            if value:
                identifiers.add(str(key))
        return identifiers
    if isinstance(raw, Iterable):
        return {str(item) for item in raw if item}
    return {str(raw)}


def _mapping(raw: object) -> Mapping[str, Any]:
    return raw if isinstance(raw, Mapping) else {}


def _first_int(mapping: Mapping[str, Any], keys: Iterable[str]) -> int | None:
    for key in keys:
        if key in mapping and _is_number(mapping[key]):
            return int(mapping[key])
    return None


def _int_value(raw: object) -> int:
    try:
        return int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _is_number(raw: object) -> bool:
    return isinstance(raw, int | float) and not isinstance(raw, bool)


def _rank_key(rank: str) -> str:
    return "T" if rank == "10" else rank
