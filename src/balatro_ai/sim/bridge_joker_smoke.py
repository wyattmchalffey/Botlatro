"""Generate bridge oracle smoke replays that touch every base-game joker.

The bridge side needs the dev ``scenario`` endpoint from
``mods/balatrobot_scenario/scenario.lua`` installed into BalatroBot. The runner
sets up one controlled transition per joker, records the real bridge pre/post
states as replay rows, and can optionally run the local simulator validator over
the resulting JSONL.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Iterable, Sequence

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.client import JsonRpcBalatroClient
from balatro_ai.api.state import GamePhase, GameState
from balatro_ai.data.replay_logger import _state_detail
from balatro_ai.sim.replay_validator import validate_local_sim_replays


SHOP_POOLS_PATH = Path(__file__).resolve().parents[1] / "data" / "shop_pools.json"


@dataclass(frozen=True, slots=True)
class CardSpec:
    key: str
    enhancement: str | None = None
    seal: str | None = None
    edition: str | None = None

    def to_bridge_json(self) -> dict[str, object]:
        payload: dict[str, object] = {"key": self.key}
        if self.enhancement:
            payload["enhancement"] = self.enhancement
        if self.seal:
            payload["seal"] = self.seal
        if self.edition:
            payload["edition"] = self.edition
        return payload


@dataclass(frozen=True, slots=True)
class JokerSpec:
    key: str
    edition: str | None = None
    eternal: bool = False
    perishable: int | None = None
    rental: bool = False
    ability: dict[str, object] = field(default_factory=dict)

    def to_bridge_json(self) -> dict[str, object]:
        payload: dict[str, object] = {"key": self.key}
        if self.edition:
            payload["edition"] = self.edition
        if self.eternal:
            payload["eternal"] = True
        if self.perishable is not None:
            payload["perishable"] = self.perishable
        if self.rental:
            payload["rental"] = True
        if self.ability:
            payload["ability"] = dict(self.ability)
        return payload


@dataclass(frozen=True, slots=True)
class JokerSmokeScenario:
    joker_key: str
    joker_name: str
    action_type: ActionType
    hand: tuple[CardSpec, ...]
    selected_count: int
    jokers: tuple[JokerSpec, ...]
    scenario_id_override: str | None = None
    category: str = "single_joker"
    action_index: int = 0
    action_kind: str = "card"
    shop_cards: tuple[object, ...] = ()
    booster_packs: tuple[object, ...] = ()
    pack_cards: tuple[object, ...] = ()
    consumables: tuple[object, ...] = ()
    blind_key: str | None = None
    blind_score: int | None = None
    probability_normal: int | None = None
    stochastic_outcomes: dict[str, object] = field(default_factory=dict)
    money: int = 100
    chips: int = 0
    hands: int = 4
    discards: int = 3
    note: str = ""

    @property
    def scenario_id(self) -> str:
        if self.scenario_id_override:
            return self.scenario_id_override
        return self.joker_key.removeprefix("j_")

    def setup_params(
        self,
        *,
        include_hand: bool = True,
        include_jokers: bool = True,
        include_surfaces: bool = True,
        include_shop_surfaces: bool = True,
        include_pack_cards: bool = True,
    ) -> dict[str, object]:
        params: dict[str, object] = {
            "money": self.money,
            "chips": self.chips,
            "hands": self.hands,
            "discards": self.discards,
        }
        if self.blind_key:
            params["blind_key"] = self.blind_key
        if self.blind_score is not None:
            params["blind_score"] = self.blind_score
        if self.probability_normal is not None:
            params["probability_normal"] = self.probability_normal
        if include_hand:
            params["clear_hand"] = True
            params["hand"] = [card.to_bridge_json() for card in self.hand]
        if include_jokers:
            params["clear_jokers"] = True
            params["jokers"] = [joker.to_bridge_json() for joker in self.jokers]
        if include_surfaces:
            if include_shop_surfaces and self.shop_cards:
                params["clear_shop"] = True
                params["shop_cards"] = list(self.shop_cards)
            if include_shop_surfaces and self.booster_packs:
                params["clear_shop"] = True
                params["booster_packs"] = list(self.booster_packs)
            if include_pack_cards and self.pack_cards:
                params["clear_pack"] = True
                params["pack_cards"] = list(self.pack_cards)
            if self.consumables:
                params["clear_consumables"] = True
                params["consumables"] = list(self.consumables)
        return params

    def action_metadata(self, base: dict[str, object] | None = None) -> dict[str, object]:
        metadata = dict(base or {})
        if self.stochastic_outcomes:
            metadata["stochastic_outcomes"] = dict(self.stochastic_outcomes)
        return metadata

    def action(self) -> Action:
        if self.action_type == ActionType.PLAY_HAND:
            return Action(ActionType.PLAY_HAND, card_indices=tuple(range(self.selected_count)), metadata=self.action_metadata())
        if self.action_type == ActionType.DISCARD:
            return Action(ActionType.DISCARD, card_indices=tuple(range(self.selected_count)), metadata=self.action_metadata())
        if self.action_type == ActionType.BUY:
            return Action(
                ActionType.BUY,
                target_id=self.action_kind,
                amount=self.action_index,
                metadata=self.action_metadata({"kind": self.action_kind, "index": self.action_index}),
            )
        if self.action_type == ActionType.REROLL:
            return Action(ActionType.REROLL)
        if self.action_type == ActionType.OPEN_PACK:
            return Action(ActionType.OPEN_PACK, target_id="pack", amount=self.action_index, metadata={"kind": "pack", "index": self.action_index})
        if self.action_type == ActionType.CHOOSE_PACK_CARD:
            if self.action_kind == "skip":
                return Action(ActionType.CHOOSE_PACK_CARD, target_id="skip", metadata={"kind": "skip", "index": True})
            return Action(ActionType.CHOOSE_PACK_CARD, target_id="card", amount=self.action_index, metadata={"kind": "card", "index": self.action_index})
        if self.action_type == ActionType.USE_CONSUMABLE:
            return Action(ActionType.USE_CONSUMABLE, target_id="consumable", amount=self.action_index, metadata={"kind": "consumable", "index": self.action_index})
        if self.action_type == ActionType.CASH_OUT:
            return Action(ActionType.CASH_OUT)
        if self.action_type == ActionType.SELECT_BLIND:
            return Action(ActionType.SELECT_BLIND)
        if self.action_type == ActionType.END_SHOP:
            return Action(ActionType.END_SHOP)
        if self.action_type == ActionType.SELL:
            return Action(ActionType.SELL, target_id="joker", amount=self.action_index, metadata={"kind": "joker", "index": self.action_index})
        raise ValueError(f"Unsupported joker smoke action: {self.action_type.value}")

    def manifest_row(self) -> dict[str, object]:
        return {
            "id": self.scenario_id,
            "joker_key": self.joker_key,
            "joker_name": self.joker_name,
            "category": self.category,
            "action_type": self.action_type.value,
            "selected_count": self.selected_count,
            "action_index": self.action_index,
            "action_kind": self.action_kind,
            "money": self.money,
            "hands": self.hands,
            "discards": self.discards,
            "hand": [card.to_bridge_json() for card in self.hand],
            "jokers": [joker.to_bridge_json() for joker in self.jokers],
            "shop_cards": list(self.shop_cards),
            "booster_packs": list(self.booster_packs),
            "pack_cards": list(self.pack_cards),
            "consumables": list(self.consumables),
            "blind_key": self.blind_key,
            "blind_score": self.blind_score,
            "probability_normal": self.probability_normal,
            "stochastic_outcomes": dict(self.stochastic_outcomes),
            "note": self.note,
        }


ROYAL_FLUSH = (
    CardSpec("H_A"),
    CardSpec("H_K"),
    CardSpec("H_Q"),
    CardSpec("H_J"),
    CardSpec("H_T"),
    CardSpec("S_2"),
    CardSpec("C_3"),
    CardSpec("S_4"),
)
SPADE_FLUSH = (
    CardSpec("S_A"),
    CardSpec("S_K"),
    CardSpec("S_Q"),
    CardSpec("S_J"),
    CardSpec("S_T"),
    CardSpec("C_2"),
    CardSpec("S_3"),
    CardSpec("C_4"),
)
FULL_HOUSE = (
    CardSpec("H_A"),
    CardSpec("S_A"),
    CardSpec("D_A"),
    CardSpec("H_K"),
    CardSpec("S_K"),
    CardSpec("C_2"),
    CardSpec("D_3"),
    CardSpec("S_4"),
)
FOUR_KIND = (
    CardSpec("H_A"),
    CardSpec("S_A"),
    CardSpec("D_A"),
    CardSpec("C_A"),
    CardSpec("H_2"),
    CardSpec("S_K"),
    CardSpec("C_Q"),
    CardSpec("S_J"),
)
FIVE_MIXED_SUITS = (
    CardSpec("H_A"),
    CardSpec("D_K"),
    CardSpec("C_Q"),
    CardSpec("S_J"),
    CardSpec("H_9"),
    CardSpec("S_2"),
    CardSpec("C_3"),
    CardSpec("S_4"),
)
FOUR_MIXED_SUITS = (
    CardSpec("H_A"),
    CardSpec("D_K"),
    CardSpec("C_Q"),
    CardSpec("S_J"),
    CardSpec("H_9"),
    CardSpec("S_2"),
    CardSpec("C_3"),
    CardSpec("S_4"),
)
THREE_CARDS = (
    CardSpec("H_A"),
    CardSpec("S_A"),
    CardSpec("D_K"),
    CardSpec("C_2"),
    CardSpec("S_3"),
    CardSpec("C_4"),
    CardSpec("S_5"),
    CardSpec("C_6"),
)
NO_FACE = (
    CardSpec("H_2"),
    CardSpec("D_3"),
    CardSpec("C_4"),
    CardSpec("S_5"),
    CardSpec("H_6"),
    CardSpec("S_7"),
    CardSpec("C_8"),
    CardSpec("S_9"),
)
HELD_BLACK = (
    CardSpec("H_A"),
    CardSpec("D_K"),
    CardSpec("C_Q"),
    CardSpec("S_J"),
    CardSpec("H_9"),
    CardSpec("S_2"),
    CardSpec("C_3"),
    CardSpec("S_4"),
)
HELD_FACE = (
    CardSpec("H_A"),
    CardSpec("D_2"),
    CardSpec("C_3"),
    CardSpec("S_4"),
    CardSpec("H_5"),
    CardSpec("S_K"),
    CardSpec("C_Q"),
    CardSpec("S_J"),
)
HELD_STEEL = (
    CardSpec("H_A"),
    CardSpec("D_K"),
    CardSpec("C_Q"),
    CardSpec("S_J"),
    CardSpec("H_T"),
    CardSpec("S_K", enhancement="STEEL"),
    CardSpec("C_Q", enhancement="STEEL"),
    CardSpec("S_J", enhancement="STEEL"),
)
DISCARD_JACKS = (
    CardSpec("H_J"),
    CardSpec("D_J"),
    CardSpec("C_J"),
    CardSpec("S_J"),
    CardSpec("H_A"),
    CardSpec("S_2"),
    CardSpec("C_3"),
    CardSpec("S_4"),
)


def joker_item(key: str, name: str, *, sell: int = 2, buy: int = 4, edition: str | None = None) -> dict[str, object]:
    item: dict[str, object] = {"key": key, "name": name, "set": "JOKER", "cost": {"buy": buy, "sell": sell}}
    if edition:
        item["edition"] = edition
    return item


def consumable_item(key: str, name: str, card_set: str, *, buy: int = 3) -> dict[str, object]:
    return {"key": key, "name": name, "label": name, "set": card_set, "cost": {"buy": buy, "sell": 1}}


def booster_item(key: str, name: str, *, buy: int = 4) -> dict[str, object]:
    return {"key": key, "name": name, "label": name, "set": "BOOSTER", "cost": {"buy": buy, "sell": 2}}


def playing_item(card: CardSpec, *, buy: int = 1) -> dict[str, object]:
    payload = card.to_bridge_json()
    payload.update({"set": "DEFAULT", "cost": {"buy": buy, "sell": 1}})
    return payload


HAND_OVERRIDES: dict[str, tuple[tuple[CardSpec, ...], int]] = {
    "Bloodstone": (SPADE_FLUSH, 5),
    "Blackboard": (HELD_BLACK, 5),
    "Baron": (HELD_FACE, 5),
    "Shoot the Moon": (HELD_FACE, 5),
    "Reserved Parking": (HELD_FACE, 5),
    "Mime": (HELD_STEEL, 5),
    "Steel Joker": (HELD_STEEL, 5),
    "Half Joker": (THREE_CARDS, 3),
    "Square Joker": (FOUR_MIXED_SUITS, 4),
    "Flower Pot": (FOUR_MIXED_SUITS, 4),
    "Seeing Double": (FIVE_MIXED_SUITS, 5),
    "Ride the Bus": (NO_FACE, 5),
    "Hack": (NO_FACE, 5),
    "Wee Joker": (NO_FACE, 5),
    "Odd Todd": (NO_FACE, 5),
    "Even Steven": (NO_FACE, 5),
    "Scholar": (FIVE_MIXED_SUITS, 5),
    "Business Card": (SPADE_FLUSH, 5),
    "Scary Face": (SPADE_FLUSH, 5),
    "Smiley Face": (SPADE_FLUSH, 5),
    "Sock and Buskin": (SPADE_FLUSH, 5),
    "Photograph": (SPADE_FLUSH, 5),
    "Triboulet": (SPADE_FLUSH, 5),
    "The Family": (FOUR_KIND, 5),
    "Obelisk": (FULL_HOUSE, 5),
    "Spare Trousers": (FULL_HOUSE, 5),
    "Mad Joker": (FULL_HOUSE, 5),
    "Clever Joker": (FULL_HOUSE, 5),
    "The Duo": (FULL_HOUSE, 5),
    "The Trio": (FULL_HOUSE, 5),
    "Jolly Joker": (FULL_HOUSE, 5),
    "Zany Joker": (FULL_HOUSE, 5),
    "Sly Joker": (FULL_HOUSE, 5),
    "Wily Joker": (FULL_HOUSE, 5),
}

RARE_COMBO_SCENARIOS: tuple[JokerSmokeScenario, ...] = (
    JokerSmokeScenario(
        scenario_id_override="combo_sock_chad_photo_face_retriggers",
        joker_key="combo_sock_chad_photo_face_retriggers",
        joker_name="Sock/Chad/Photo retrigger stack",
        category="rare_combo",
        action_type=ActionType.PLAY_HAND,
        hand=(CardSpec("S_K", seal="RED"), CardSpec("S_Q"), CardSpec("S_J"), CardSpec("H_2"), CardSpec("D_3")),
        selected_count=3,
        jokers=(
            JokerSpec("j_sock_and_buskin"),
            JokerSpec("j_hanging_chad"),
            JokerSpec("j_photograph"),
            JokerSpec("j_smiley"),
        ),
        note="Face-card retriggers stack with Red Seal, Hanging Chad, Sock and Buskin, Photograph, and Smiley Face.",
    ),
    JokerSmokeScenario(
        scenario_id_override="combo_mime_baron_shoot_moon_steel_held",
        joker_key="combo_mime_baron_shoot_moon_steel_held",
        joker_name="Mime/Baron held-card stack",
        category="rare_combo",
        action_type=ActionType.PLAY_HAND,
        hand=(
            CardSpec("H_A"),
            CardSpec("D_2"),
            CardSpec("C_3"),
            CardSpec("S_4"),
            CardSpec("H_5"),
            CardSpec("S_K", enhancement="STEEL", seal="RED"),
            CardSpec("C_Q", enhancement="STEEL"),
            CardSpec("D_Q"),
        ),
        selected_count=1,
        jokers=(JokerSpec("j_mime"), JokerSpec("j_baron"), JokerSpec("j_shoot_the_moon")),
        note="Held-card xmult/mult stack with Mime retriggers, Baron, Shoot the Moon, Steel, and Red Seal.",
    ),
    JokerSmokeScenario(
        scenario_id_override="combo_smeared_seeing_double_flower_pot",
        joker_key="combo_smeared_seeing_double_flower_pot",
        joker_name="Smeared suit condition stack",
        category="rare_combo",
        action_type=ActionType.PLAY_HAND,
        hand=(CardSpec("S_K"), CardSpec("C_Q"), CardSpec("H_J"), CardSpec("D_T"), CardSpec("S_9")),
        selected_count=4,
        jokers=(JokerSpec("j_smeared"), JokerSpec("j_seeing_double"), JokerSpec("j_flower_pot")),
        note="Smeared Joker changes suit buckets for Seeing Double and Flower Pot style suit checks.",
    ),
    JokerSmokeScenario(
        scenario_id_override="combo_four_fingers_shortcut_smeared",
        joker_key="combo_four_fingers_shortcut_smeared",
        joker_name="Four Fingers/Shortcut/Smeared straight flush",
        category="rare_combo",
        action_type=ActionType.PLAY_HAND,
        hand=(CardSpec("S_A"), CardSpec("C_K"), CardSpec("S_Q"), CardSpec("C_J"), CardSpec("H_2")),
        selected_count=4,
        jokers=(JokerSpec("j_four_fingers"), JokerSpec("j_shortcut"), JokerSpec("j_smeared")),
        note="Four-card straight flush detection with rank gaps and Smeared black-suit flush behavior.",
    ),
    JokerSmokeScenario(
        scenario_id_override="combo_blueprint_brainstorm_baseball_uncommon",
        joker_key="combo_blueprint_brainstorm_baseball_uncommon",
        joker_name="Copy joker plus Baseball rarity stack",
        category="rare_combo",
        action_type=ActionType.PLAY_HAND,
        hand=HELD_BLACK,
        selected_count=5,
        jokers=(
            JokerSpec("j_blueprint"),
            JokerSpec("j_baseball"),
            JokerSpec("j_blackboard"),
            JokerSpec("j_brainstorm"),
            JokerSpec("j_joker"),
        ),
        note="Blueprint/Brainstorm copy resolution with Baseball Card uncommon xmult and held-card Blackboard.",
    ),
    JokerSmokeScenario(
        scenario_id_override="combo_gold_seal_ticket_rough_bootstraps",
        joker_key="combo_gold_seal_ticket_rough_bootstraps",
        joker_name="Scoring-time money feeds Bootstraps",
        category="rare_combo",
        action_type=ActionType.PLAY_HAND,
        hand=(CardSpec("D_A", enhancement="GOLD", seal="GOLD"), CardSpec("D_K"), CardSpec("D_Q"), CardSpec("D_J"), CardSpec("D_T")),
        selected_count=5,
        jokers=(JokerSpec("j_ticket"), JokerSpec("j_rough_gem"), JokerSpec("j_bootstraps")),
        money=20,
        note="Gold seal, Golden Ticket, and Rough Gem scoring-time dollars should affect Bootstraps in the same hand.",
    ),
    JokerSmokeScenario(
        scenario_id_override="combo_blueprint_bloodstone_forced_hearts",
        joker_key="combo_blueprint_bloodstone_forced_hearts",
        joker_name="Blueprint copied forced Bloodstone",
        category="rare_combo",
        action_type=ActionType.PLAY_HAND,
        hand=(CardSpec("H_A"), CardSpec("H_K"), CardSpec("H_Q"), CardSpec("H_J"), CardSpec("H_T")),
        selected_count=5,
        jokers=(JokerSpec("j_blueprint"), JokerSpec("j_bloodstone")),
        probability_normal=100,
        stochastic_outcomes={"bloodstone_triggers": 10},
        note="Blueprint copying Bloodstone should add another forced X1.5 roll for every scoring Heart.",
    ),
)

SHOP_EVENT_SCENARIOS: tuple[JokerSmokeScenario, ...] = (
    JokerSmokeScenario(
        scenario_id_override="shop_flash_card_reroll",
        joker_key="shop_flash_card_reroll",
        joker_name="Flash Card reroll",
        category="shop_event",
        action_type=ActionType.REROLL,
        hand=ROYAL_FLUSH,
        selected_count=5,
        jokers=(JokerSpec("j_flash", ability={"mult": 2}),),
        shop_cards=(joker_item("j_joker", "Joker"),),
        note="Flash Card scales when the shop is rerolled.",
    ),
    JokerSmokeScenario(
        scenario_id_override="shop_campfire_sell_other_joker",
        joker_key="shop_campfire_sell_other_joker",
        joker_name="Campfire sell trigger",
        category="shop_event",
        action_type=ActionType.SELL,
        hand=ROYAL_FLUSH,
        selected_count=5,
        action_index=1,
        jokers=(JokerSpec("j_campfire", ability={"x_mult": 1.0}), JokerSpec("j_joker")),
        note="Campfire gains XMult when another card is sold.",
    ),
    JokerSmokeScenario(
        scenario_id_override="shop_red_card_pack_skip",
        joker_key="shop_red_card_pack_skip",
        joker_name="Red Card pack skip",
        category="shop_event",
        action_type=ActionType.CHOOSE_PACK_CARD,
        hand=ROYAL_FLUSH,
        selected_count=5,
        action_kind="skip",
        jokers=(JokerSpec("j_red_card", ability={"mult": 0}),),
        booster_packs=(booster_item("p_buffoon_normal_1", "Buffoon Pack"),),
        pack_cards=(joker_item("j_joker", "Joker"),),
        note="Red Card gains Mult when an opened booster is skipped.",
    ),
    JokerSmokeScenario(
        scenario_id_override="shop_hologram_buy_playing_card",
        joker_key="shop_hologram_buy_playing_card",
        joker_name="Hologram shop card add",
        category="shop_event",
        action_type=ActionType.BUY,
        hand=ROYAL_FLUSH,
        selected_count=5,
        jokers=(JokerSpec("j_hologram", ability={"x_mult": 1.0}),),
        shop_cards=(playing_item(CardSpec("H_A")),),
        note="Hologram scales when a playing card is bought from the shop.",
    ),
    JokerSmokeScenario(
        scenario_id_override="shop_hologram_standard_pack_pick",
        joker_key="shop_hologram_standard_pack_pick",
        joker_name="Hologram pack card add",
        category="shop_event",
        action_type=ActionType.CHOOSE_PACK_CARD,
        hand=ROYAL_FLUSH,
        selected_count=5,
        jokers=(JokerSpec("j_hologram", ability={"x_mult": 1.0}),),
        booster_packs=(booster_item("p_standard_normal_1", "Standard Pack"),),
        pack_cards=(playing_item(CardSpec("S_A")),),
        note="Hologram scales when a playing card is selected from a Standard Pack.",
    ),
    JokerSmokeScenario(
        scenario_id_override="shop_constellation_planet_pack_pick",
        joker_key="shop_constellation_planet_pack_pick",
        joker_name="Constellation planet use",
        category="shop_event",
        action_type=ActionType.CHOOSE_PACK_CARD,
        hand=ROYAL_FLUSH,
        selected_count=5,
        jokers=(JokerSpec("j_constellation", ability={"x_mult": 1.0}),),
        booster_packs=(booster_item("p_celestial_normal_1", "Celestial Pack"),),
        pack_cards=(consumable_item("c_mars", "Mars", "PLANET"),),
        note="Constellation scales when a Planet card is used from a Celestial Pack.",
    ),
    JokerSmokeScenario(
        scenario_id_override="shop_fortune_teller_hermit_pack_pick",
        joker_key="shop_fortune_teller_hermit_pack_pick",
        joker_name="Fortune Teller tarot use",
        category="shop_event",
        action_type=ActionType.CHOOSE_PACK_CARD,
        hand=ROYAL_FLUSH,
        selected_count=5,
        money=20,
        jokers=(JokerSpec("j_fortune_teller", ability={"mult": 0}),),
        booster_packs=(booster_item("p_arcana_normal_1", "Arcana Pack"),),
        pack_cards=(consumable_item("c_hermit", "The Hermit", "TAROT"),),
        note="Fortune Teller's tarot-use counter is exercised by using Hermit from an Arcana Pack.",
    ),
    JokerSmokeScenario(
        scenario_id_override="shop_perkeo_end_shop_copy",
        joker_key="shop_perkeo_end_shop_copy",
        joker_name="Perkeo end-shop copy",
        category="shop_event",
        action_type=ActionType.END_SHOP,
        hand=ROYAL_FLUSH,
        selected_count=5,
        jokers=(JokerSpec("j_perkeo"),),
        consumables=(consumable_item("c_mars", "Mars", "PLANET"),),
        note="Perkeo creates a negative consumable copy when leaving the shop.",
    ),
)

EDGE_CASE_SCENARIOS: tuple[JokerSmokeScenario, ...] = (
    JokerSmokeScenario(
        scenario_id_override="edge_midas_vampire_pareidolia_splash_mutation",
        joker_key="edge_midas_vampire_pareidolia_splash_mutation",
        joker_name="Midas/Vampire/Pareidolia mutation order",
        category="edge_case",
        action_type=ActionType.PLAY_HAND,
        hand=(
            CardSpec("S_A", enhancement="BONUS"),
            CardSpec("H_7", enhancement="MULT"),
            CardSpec("D_4", enhancement="WILD"),
            CardSpec("C_2", enhancement="STEEL"),
            CardSpec("S_9"),
        ),
        selected_count=5,
        jokers=(
            JokerSpec("j_splash"),
            JokerSpec("j_pareidolia"),
            JokerSpec("j_midas_mask"),
            JokerSpec("j_vampire", ability={"x_mult": 1.0}),
        ),
        note="Splash scores all cards while Pareidolia lets Midas gold them and Vampire strips enhancements afterward.",
    ),
    JokerSmokeScenario(
        scenario_id_override="edge_final_hand_low_card_retrigger_stack",
        joker_key="edge_final_hand_low_card_retrigger_stack",
        joker_name="Final-hand low-card retrigger stack",
        category="edge_case",
        action_type=ActionType.PLAY_HAND,
        hand=(
            CardSpec("S_2", seal="RED"),
            CardSpec("H_3", seal="RED"),
            CardSpec("D_4"),
            CardSpec("C_5"),
            CardSpec("S_6"),
        ),
        selected_count=5,
        jokers=(
            JokerSpec("j_hack"),
            JokerSpec("j_dusk"),
            JokerSpec("j_selzer", ability={"extra": 2}),
            JokerSpec("j_fibonacci"),
            JokerSpec("j_wee", ability={"chips": 0}),
        ),
        hands=1,
        note="Red Seal, Hack, Dusk, and Seltzer all add scored-card repetitions on the same low-card straight.",
    ),
    JokerSmokeScenario(
        scenario_id_override="edge_blueprint_mime_redseal_steel_held",
        joker_key="edge_blueprint_mime_redseal_steel_held",
        joker_name="Blueprint copied Mime held-card retriggers",
        category="edge_case",
        action_type=ActionType.PLAY_HAND,
        hand=(
            CardSpec("H_A"),
            CardSpec("S_K", enhancement="STEEL", seal="RED"),
            CardSpec("C_Q", enhancement="STEEL"),
            CardSpec("D_2"),
            CardSpec("C_3"),
        ),
        selected_count=1,
        jokers=(JokerSpec("j_blueprint"), JokerSpec("j_mime"), JokerSpec("j_baron")),
        note="Blueprint copying Mime should add another held-card retrigger on top of Red Seal and Baron.",
    ),
    JokerSmokeScenario(
        scenario_id_override="edge_brainstorm_sock_buskin_face_retriggers",
        joker_key="edge_brainstorm_sock_buskin_face_retriggers",
        joker_name="Brainstorm copied Sock and Buskin retriggers",
        category="edge_case",
        action_type=ActionType.PLAY_HAND,
        hand=(CardSpec("S_K", seal="RED"), CardSpec("H_Q"), CardSpec("D_J"), CardSpec("C_9"), CardSpec("S_2")),
        selected_count=3,
        jokers=(JokerSpec("j_sock_and_buskin"), JokerSpec("j_brainstorm"), JokerSpec("j_photograph")),
        note="Brainstorm copies the leftmost Sock and Buskin, stacking face-card retriggers with Photograph and Red Seal.",
    ),
    JokerSmokeScenario(
        scenario_id_override="edge_negative_joker_buy_full_slots",
        joker_key="edge_negative_joker_buy_full_slots",
        joker_name="Negative joker buy at full slots",
        category="edge_case",
        action_type=ActionType.BUY,
        hand=ROYAL_FLUSH,
        selected_count=5,
        jokers=(
            JokerSpec("j_joker"),
            JokerSpec("j_greedy_joker"),
            JokerSpec("j_lusty_joker"),
            JokerSpec("j_wrathful_joker"),
            JokerSpec("j_gluttenous_joker"),
        ),
        shop_cards=(joker_item("j_cavendish", "Cavendish", edition="NEGATIVE"),),
        note="Negative shop jokers can be bought even when all normal joker slots are full.",
    ),
    JokerSmokeScenario(
        scenario_id_override="edge_negative_buffoon_pick_full_slots",
        joker_key="edge_negative_buffoon_pick_full_slots",
        joker_name="Negative Buffoon Pack pick at full slots",
        category="edge_case",
        action_type=ActionType.CHOOSE_PACK_CARD,
        hand=ROYAL_FLUSH,
        selected_count=5,
        jokers=(
            JokerSpec("j_joker"),
            JokerSpec("j_greedy_joker"),
            JokerSpec("j_lusty_joker"),
            JokerSpec("j_wrathful_joker"),
            JokerSpec("j_gluttenous_joker"),
        ),
        booster_packs=(booster_item("p_buffoon_normal_1", "Buffoon Pack"),),
        pack_cards=(joker_item("j_cavendish", "Cavendish", edition="NEGATIVE"),),
        note="Negative jokers from Buffoon Packs should also bypass the normal slot limit.",
    ),
)

BOSS_BLIND_SCENARIOS: tuple[JokerSmokeScenario, ...] = (
    JokerSmokeScenario(
        scenario_id_override="boss_flint_halves_straight_values",
        joker_key="boss_flint_halves_straight_values",
        joker_name="The Flint hand-value modifier",
        category="boss_blind",
        action_type=ActionType.PLAY_HAND,
        hand=(CardSpec("S_Q"), CardSpec("C_J"), CardSpec("D_T"), CardSpec("S_9"), CardSpec("D_8")),
        selected_count=5,
        jokers=(JokerSpec("j_joker"),),
        blind_key="bl_flint",
        blind_score=999999,
        note="The Flint halves base hand chips and Mult using the live boss blind implementation.",
    ),
    JokerSmokeScenario(
        scenario_id_override="boss_psychic_blocks_short_play",
        joker_key="boss_psychic_blocks_short_play",
        joker_name="The Psychic short-hand block",
        category="boss_blind",
        action_type=ActionType.PLAY_HAND,
        hand=(CardSpec("S_K"), CardSpec("C_K"), CardSpec("S_8"), CardSpec("H_8"), CardSpec("D_2")),
        selected_count=4,
        jokers=(JokerSpec("j_joker"),),
        blind_key="bl_psychic",
        blind_score=999999,
        note="The Psychic should score zero for a played hand with fewer than five cards.",
    ),
    JokerSmokeScenario(
        scenario_id_override="boss_hook_discards_held_cards",
        joker_key="boss_hook_discards_held_cards",
        joker_name="The Hook held-card discard",
        category="boss_blind",
        action_type=ActionType.PLAY_HAND,
        hand=(CardSpec("S_A"), CardSpec("D_K"), CardSpec("C_Q"), CardSpec("H_J"), CardSpec("S_9")),
        selected_count=1,
        jokers=(JokerSpec("j_green_joker", ability={"mult": 0}),),
        blind_key="bl_hook",
        blind_score=999999,
        note="The Hook discards two random held cards before scoring; replay-diff infers the actual cards discarded.",
    ),
    JokerSmokeScenario(
        scenario_id_override="boss_head_debuffs_heart_flush",
        joker_key="boss_head_debuffs_heart_flush",
        joker_name="The Head suit debuff",
        category="boss_blind",
        action_type=ActionType.PLAY_HAND,
        hand=(CardSpec("H_A"), CardSpec("H_K"), CardSpec("H_Q"), CardSpec("H_J"), CardSpec("H_T")),
        selected_count=5,
        jokers=(JokerSpec("j_lusty_joker"), JokerSpec("j_joker")),
        blind_key="bl_head",
        blind_score=999999,
        note="The Head debuffs Heart cards; suit jokers and card chips should not fire for debuffed scoring cards.",
    ),
    JokerSmokeScenario(
        scenario_id_override="boss_plant_debuffs_face_jokers",
        joker_key="boss_plant_debuffs_face_jokers",
        joker_name="The Plant face-card debuff",
        category="boss_blind",
        action_type=ActionType.PLAY_HAND,
        hand=(CardSpec("S_K"), CardSpec("H_Q"), CardSpec("D_J"), CardSpec("C_9"), CardSpec("S_2")),
        selected_count=3,
        jokers=(JokerSpec("j_scary_face"), JokerSpec("j_smiley"), JokerSpec("j_photograph"), JokerSpec("j_joker")),
        blind_key="bl_plant",
        blind_score=999999,
        note="The Plant debuffs face cards, suppressing their chips, scored-card joker bonuses, and Photograph.",
    ),
    JokerSmokeScenario(
        scenario_id_override="boss_tooth_money_loss_feeds_bootstraps",
        joker_key="boss_tooth_money_loss_feeds_bootstraps",
        joker_name="The Tooth scoring-time money loss",
        category="boss_blind",
        action_type=ActionType.PLAY_HAND,
        hand=(CardSpec("S_A"), CardSpec("D_K"), CardSpec("C_Q"), CardSpec("H_J"), CardSpec("S_9")),
        selected_count=5,
        jokers=(JokerSpec("j_bootstraps"),),
        blind_key="bl_tooth",
        blind_score=999999,
        money=10,
        note="The Tooth removes $1 per played card before money-scaled joker scoring, which changes Bootstraps immediately.",
    ),
    JokerSmokeScenario(
        scenario_id_override="boss_water_zero_discards_banner",
        joker_key="boss_water_zero_discards_banner",
        joker_name="The Water discard removal",
        category="boss_blind",
        action_type=ActionType.PLAY_HAND,
        hand=ROYAL_FLUSH,
        selected_count=5,
        jokers=(JokerSpec("j_banner"),),
        blind_key="bl_water",
        blind_score=999999,
        discards=3,
        note="The Water sets discards to zero as the blind starts, so Banner should score no discard chips.",
    ),
    JokerSmokeScenario(
        scenario_id_override="boss_needle_one_hand_acrobat",
        joker_key="boss_needle_one_hand_acrobat",
        joker_name="The Needle one-hand setup",
        category="boss_blind",
        action_type=ActionType.PLAY_HAND,
        hand=(CardSpec("S_A"), CardSpec("D_K"), CardSpec("C_Q"), CardSpec("H_J"), CardSpec("S_9")),
        selected_count=1,
        jokers=(JokerSpec("j_acrobat"),),
        blind_key="bl_needle",
        blind_score=1,
        hands=4,
        note="The Needle reduces the blind to one hand; Acrobat should see that hand as the final hand.",
    ),
)

STOCHASTIC_SCENARIOS: tuple[JokerSmokeScenario, ...] = (
    JokerSmokeScenario(
        scenario_id_override="stochastic_bloodstone_all_hearts_forced",
        joker_key="stochastic_bloodstone_all_hearts_forced",
        joker_name="Forced Bloodstone triggers",
        category="stochastic",
        action_type=ActionType.PLAY_HAND,
        hand=(CardSpec("H_A"), CardSpec("H_K"), CardSpec("H_Q"), CardSpec("H_J"), CardSpec("H_T")),
        selected_count=5,
        jokers=(JokerSpec("j_bloodstone"),),
        probability_normal=100,
        stochastic_outcomes={"bloodstone_triggers": 5},
        note="The scenario raises the game's normal probability multiplier so every Bloodstone roll succeeds.",
    ),
    JokerSmokeScenario(
        scenario_id_override="stochastic_space_joker_forced_level",
        joker_key="stochastic_space_joker_forced_level",
        joker_name="Forced Space Joker level-up",
        category="stochastic",
        action_type=ActionType.PLAY_HAND,
        hand=(CardSpec("S_A"), CardSpec("D_K"), CardSpec("C_7"), CardSpec("H_4"), CardSpec("D_2")),
        selected_count=1,
        jokers=(JokerSpec("j_space"),),
        probability_normal=100,
        stochastic_outcomes={"space_joker_triggers": 1},
        note="Space Joker should level the scored hand before score calculation when its probability is forced.",
    ),
    JokerSmokeScenario(
        scenario_id_override="stochastic_lucky_card_mult_money_cat_forced",
        joker_key="stochastic_lucky_card_mult_money_cat_forced",
        joker_name="Forced Lucky Card money/mult",
        category="stochastic",
        action_type=ActionType.PLAY_HAND,
        hand=(CardSpec("S_A", enhancement="LUCKY"), CardSpec("D_K"), CardSpec("C_7"), CardSpec("H_4"), CardSpec("D_2")),
        selected_count=1,
        jokers=(JokerSpec("j_lucky_cat", ability={"x_mult": 1.0}),),
        probability_normal=100,
        stochastic_outcomes={
            "lucky_card_mult_triggers": 1,
            "lucky_card_money_triggers": 1,
            "lucky_card_triggers": 1,
        },
        note="Lucky Card's Mult and money rolls are separate; forced probabilities exercise both plus Lucky Cat scaling.",
    ),
    JokerSmokeScenario(
        scenario_id_override="stochastic_glass_card_shatter_forced",
        joker_key="stochastic_glass_card_shatter_forced",
        joker_name="Forced Glass Card shatter",
        category="stochastic",
        action_type=ActionType.PLAY_HAND,
        hand=(CardSpec("S_A", enhancement="GLASS"), CardSpec("D_K"), CardSpec("C_7"), CardSpec("H_4"), CardSpec("D_2")),
        selected_count=1,
        jokers=(JokerSpec("j_glass", ability={"x_mult": 1.0}),),
        probability_normal=100,
        stochastic_outcomes={
            "shattered_glass_cards": ({"rank": "A", "suit": "S", "enhancement": "GLASS"},),
        },
        note="Glass Card shatter removes the played card and increments Glass Joker when the probability is forced.",
    ),
    JokerSmokeScenario(
        scenario_id_override="stochastic_business_card_bootstraps_forced",
        joker_key="stochastic_business_card_bootstraps_forced",
        joker_name="Forced Business Card money",
        category="stochastic",
        action_type=ActionType.PLAY_HAND,
        hand=(CardSpec("S_K"), CardSpec("D_9"), CardSpec("C_7"), CardSpec("H_4"), CardSpec("D_2")),
        selected_count=1,
        jokers=(JokerSpec("j_business"), JokerSpec("j_bootstraps")),
        probability_normal=100,
        stochastic_outcomes={"business_card_triggers": 1},
        money=4,
        note="A forced Business Card payout should feed Bootstraps during the same scoring calculation.",
    ),
    JokerSmokeScenario(
        scenario_id_override="stochastic_reserved_parking_bootstraps_forced",
        joker_key="stochastic_reserved_parking_bootstraps_forced",
        joker_name="Forced Reserved Parking money",
        category="stochastic",
        action_type=ActionType.PLAY_HAND,
        hand=(
            CardSpec("S_A"),
            CardSpec("D_2"),
            CardSpec("C_3"),
            CardSpec("S_4"),
            CardSpec("H_5"),
            CardSpec("S_K"),
            CardSpec("C_Q"),
            CardSpec("D_J"),
        ),
        selected_count=1,
        jokers=(JokerSpec("j_reserved_parking"), JokerSpec("j_bootstraps")),
        probability_normal=100,
        stochastic_outcomes={"reserved_parking_triggers": 3},
        money=4,
        note="Forced Reserved Parking held-face payouts should also feed Bootstraps on the same hand.",
    ),
)

ACTION_OVERRIDES: dict[str, ActionType] = {
    "Burnt Joker": ActionType.DISCARD,
    "Castle": ActionType.DISCARD,
    "Faceless Joker": ActionType.DISCARD,
    "Green Joker": ActionType.DISCARD,
    "Hit the Road": ActionType.DISCARD,
    "Mail-In Rebate": ActionType.DISCARD,
    "Ramen": ActionType.DISCARD,
    "Trading Card": ActionType.DISCARD,
    "Yorick": ActionType.DISCARD,
    "Campfire": ActionType.CASH_OUT,
    "Cloud 9": ActionType.CASH_OUT,
    "Delayed Gratification": ActionType.CASH_OUT,
    "Egg": ActionType.CASH_OUT,
    "Gift Card": ActionType.CASH_OUT,
    "Golden Joker": ActionType.CASH_OUT,
    "Invisible Joker": ActionType.CASH_OUT,
    "Popcorn": ActionType.CASH_OUT,
    "Rocket": ActionType.CASH_OUT,
    "Satellite": ActionType.CASH_OUT,
    "To Do List": ActionType.CASH_OUT,
    "Troubadour": ActionType.SELECT_BLIND,
    "Turtle Bean": ActionType.CASH_OUT,
    "Burglar": ActionType.SELECT_BLIND,
    "Cartomancer": ActionType.SELECT_BLIND,
    "Ceremonial Dagger": ActionType.SELECT_BLIND,
    "Certificate": ActionType.SELECT_BLIND,
    "Chicot": ActionType.SELECT_BLIND,
    "Madness": ActionType.SELECT_BLIND,
    "Marble Joker": ActionType.SELECT_BLIND,
    "Riff-raff": ActionType.SELECT_BLIND,
    "Diet Cola": ActionType.SELL,
    "Luchador": ActionType.SELL,
}

DISCARD_HAND_OVERRIDES: dict[str, tuple[tuple[CardSpec, ...], int]] = {
    "Faceless Joker": (SPADE_FLUSH, 5),
    "Hit the Road": (DISCARD_JACKS, 4),
    "Mail-In Rebate": (ROYAL_FLUSH, 5),
    "Ramen": (ROYAL_FLUSH, 5),
    "Trading Card": (ROYAL_FLUSH, 1),
    "Yorick": (ROYAL_FLUSH, 5),
}

MONEY_OVERRIDES: dict[str, int] = {
    "Vagabond": 3,
    "Credit Card": 0,
}

HANDS_OVERRIDES: dict[str, int] = {
    "Acrobat": 1,
    "Dusk": 1,
    "Loyalty Card": 1,
}

DISCARDS_OVERRIDES: dict[str, int] = {
    "Mystic Summit": 0,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run bridge oracle smoke scenarios for every joker.")
    parser.add_argument("--endpoint", default="http://127.0.0.1:12346", help="BalatroBot JSON-RPC endpoint.")
    parser.add_argument("--output", default=".data/bridge-joker-smoke/joker_smoke.jsonl", help="Output replay JSONL path.")
    parser.add_argument("--manifest", help="Optional manifest JSON path to write without running the bridge.")
    parser.add_argument("--seed", type=int, default=700001, help="Base seed for scenarios.")
    parser.add_argument("--stake", default="white")
    parser.add_argument("--deck", default="RED")
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--only", action="append", default=[], help="Run one joker name/key. Can be repeated.")
    parser.add_argument("--limit", type=int, help="Run only the first N selected scenarios.")
    parser.add_argument("--skip-validate", action="store_true", help="Do not run local sim replay validation after capture.")
    parser.add_argument("--dry-run", action="store_true", help="Only print/write the generated manifest.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scenarios = list(iter_joker_smoke_scenarios())
    scenarios = _filter_scenarios(scenarios, args.only)
    if args.limit is not None:
        scenarios = scenarios[: max(0, args.limit)]

    if args.manifest:
        write_manifest(scenarios, Path(args.manifest))
    if args.dry_run:
        print(f"Generated {len(scenarios)} joker smoke scenario(s).")
        return 0

    output = Path(args.output)
    client = JsonRpcBalatroClient(endpoint=args.endpoint, deck=args.deck, timeout_seconds=args.timeout_seconds)
    started_at = time.time()
    result = run_bridge_joker_smoke(
        client,
        scenarios,
        output_path=output,
        base_seed=args.seed,
        stake=args.stake,
    )
    print(
        f"Wrote {result['rows_written']} replay rows for {result['scenarios_run']} joker scenario(s) "
        f"to {output} in {time.time() - started_at:.1f}s."
    )
    if result["failures"]:
        print("Scenario failures:")
        for failure in result["failures"]:
            print(f"- {failure}")

    if not args.skip_validate:
        summary = validate_local_sim_replays((output,))
        print(summary.to_text(example_limit=20))
        return 1 if summary.comparable_divergences else 0
    return 1 if result["failures"] else 0


def iter_joker_smoke_scenarios() -> Iterable[JokerSmokeScenario]:
    yield from iter_base_joker_smoke_scenarios()
    yield from RARE_COMBO_SCENARIOS
    yield from SHOP_EVENT_SCENARIOS
    yield from EDGE_CASE_SCENARIOS
    yield from BOSS_BLIND_SCENARIOS
    yield from STOCHASTIC_SCENARIOS


def iter_base_joker_smoke_scenarios() -> Iterable[JokerSmokeScenario]:
    for joker in _load_joker_pool():
        name = str(joker["name"])
        key = str(joker["key"])
        action_type = ACTION_OVERRIDES.get(name, ActionType.PLAY_HAND)
        hand, selected_count = _hand_for_joker(name, action_type)
        yield JokerSmokeScenario(
            joker_key=key,
            joker_name=name,
            action_type=action_type,
            hand=hand,
            selected_count=selected_count,
            jokers=_joker_stack_for(key, name),
            money=MONEY_OVERRIDES.get(name, 100),
            hands=HANDS_OVERRIDES.get(name, 4),
            discards=DISCARDS_OVERRIDES.get(name, 3),
            note=_note_for_joker(name, action_type),
        )


def write_manifest(scenarios: Sequence[JokerSmokeScenario], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": "Generated bridge oracle smoke scenarios: one primary scenario per base-game joker plus rare combo and shop-event cases.",
        "count": len(scenarios),
        "scenarios": [scenario.manifest_row() for scenario in scenarios],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_bridge_joker_smoke(
    client: JsonRpcBalatroClient,
    scenarios: Sequence[JokerSmokeScenario],
    *,
    output_path: Path,
    base_seed: int,
    stake: str,
) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")
    failures: list[str] = []
    rows_written = 0

    for index, scenario in enumerate(scenarios):
        seed = base_seed + index
        try:
            rows = _run_one_scenario(client, scenario, seed=seed, stake=stake)
        except Exception as exc:  # noqa: BLE001 - smoke runner should continue across flaky scenarios.
            failures.append(f"{scenario.joker_name} ({scenario.joker_key}): {type(exc).__name__}: {exc}")
            continue
        with output_path.open("a", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, sort_keys=True) + "\n")
                rows_written += 1

    return {
        "scenarios_run": len(scenarios) - len(failures),
        "rows_written": rows_written,
        "failures": failures,
    }


def _run_one_scenario(
    client: JsonRpcBalatroClient,
    scenario: JokerSmokeScenario,
    *,
    seed: int,
    stake: str,
) -> tuple[dict[str, object], dict[str, object]]:
    state = client.start_run(seed=seed, stake=stake)
    if state.phase == GamePhase.BLIND_SELECT:
        state = _settled_state(client, client.send_action(Action(ActionType.SELECT_BLIND)))
    if state.phase != GamePhase.SELECTING_HAND:
        raise RuntimeError(f"Expected selecting_hand after start/select, got {state.phase.value}")

    state = _state_from_call(client.call("scenario", scenario.setup_params(include_surfaces=False)))
    state = _advance_to_action_phase(client, scenario, state)
    action = scenario.action()
    pre_row = _replay_row(
        scenario=scenario,
        seed=seed,
        state=state,
        action=action,
        stage="pre",
    )
    post_state = _settled_state(client, client.send_action(action))
    post_row = _replay_row(
        scenario=scenario,
        seed=seed,
        state=post_state,
        action=None,
        stage="post",
    )
    return pre_row, post_row


def _advance_to_action_phase(
    client: JsonRpcBalatroClient,
    scenario: JokerSmokeScenario,
    state: GameState,
) -> GameState:
    if scenario.action_type in {ActionType.PLAY_HAND, ActionType.DISCARD}:
        return state

    clear_action = Action(ActionType.PLAY_HAND, card_indices=tuple(range(min(5, len(state.hand)))))
    state = _state_from_call(client.call("set", {"chips": max(0, state.required_score - 1), "hands": 4, "discards": 3}))
    state = _settled_state(client, client.send_action(clear_action))
    if state.phase != GamePhase.ROUND_EVAL:
        raise RuntimeError(f"Setup clear hand did not reach round_eval; got {state.phase.value}")
    if scenario.action_type == ActionType.CASH_OUT:
        return state

    state = _settled_state(client, client.send_action(Action(ActionType.CASH_OUT)))
    if state.phase != GamePhase.SHOP:
        raise RuntimeError(f"Setup cash_out did not reach shop; got {state.phase.value}")
    if _has_surface_setup(scenario):
        state = _state_from_call(
            client.call(
                "scenario",
                scenario.setup_params(include_hand=False, include_jokers=False, include_surfaces=True, include_pack_cards=False),
            )
        )
        state = _settled_state(client, state)
    if scenario.action_type in {
        ActionType.BUY,
        ActionType.SELL,
        ActionType.REROLL,
        ActionType.OPEN_PACK,
        ActionType.USE_CONSUMABLE,
        ActionType.END_SHOP,
    }:
        return state

    if scenario.action_type == ActionType.CHOOSE_PACK_CARD:
        if not scenario.booster_packs:
            raise RuntimeError(f"{scenario.scenario_id} needs booster_packs for choose_pack_card setup")
        state = _settled_state(
            client,
            client.send_action(Action(ActionType.OPEN_PACK, target_id="pack", amount=0, metadata={"kind": "pack", "index": 0})),
        )
        if state.phase != GamePhase.BOOSTER_OPENED:
            raise RuntimeError(f"Setup open_pack did not reach booster_opened; got {state.phase.value}")
        if scenario.pack_cards:
            state = _state_from_call(
                client.call(
                    "scenario",
                    scenario.setup_params(
                        include_hand=False,
                        include_jokers=False,
                        include_surfaces=True,
                        include_shop_surfaces=False,
                    ),
                )
            )
            state = _settled_state(client, state)
        return state

    state = _settled_state(client, client.send_action(Action(ActionType.END_SHOP)))
    if state.phase != GamePhase.BLIND_SELECT:
        raise RuntimeError(f"Setup end_shop did not reach blind_select; got {state.phase.value}")
    return state


def _has_surface_setup(scenario: JokerSmokeScenario) -> bool:
    return bool(scenario.shop_cards or scenario.booster_packs or scenario.pack_cards or scenario.consumables)


def _settled_state(
    client: JsonRpcBalatroClient,
    state: GameState,
    *,
    polls: int = 8,
    delay_seconds: float = 0.05,
) -> GameState:
    stable_polls = 0
    latest = state
    latest_signature = _state_settle_signature(latest)
    for _ in range(polls):
        time.sleep(delay_seconds)
        current = client.get_state()
        current_signature = _state_settle_signature(current)
        if current_signature == latest_signature:
            stable_polls += 1
            if stable_polls >= 2:
                return current
        else:
            stable_polls = 0
            latest = current
            latest_signature = current_signature
    return latest


def _state_settle_signature(state: GameState) -> tuple[object, ...]:
    return (
        state.phase,
        state.current_score,
        state.money,
        state.hands_remaining,
        state.discards_remaining,
        state.deck_size,
        tuple((card.short_name, card.seal, card.enhancement, card.edition, card.debuffed) for card in state.hand),
        tuple((joker.name, joker.sell_value, joker.edition, json.dumps(joker.metadata, sort_keys=True, default=str)) for joker in state.jokers),
        state.consumables,
        state.shop,
        state.pack,
    )


def _replay_row(
    *,
    scenario: JokerSmokeScenario,
    seed: int,
    state: GameState,
    action: Action | None,
    stage: str,
) -> dict[str, object]:
    row: dict[str, object] = {
        "record_type": "bridge_joker_smoke",
        "seed": seed,
        "scenario_id": scenario.scenario_id,
        "joker_key": scenario.joker_key,
        "joker_name": scenario.joker_name,
        "stage": stage,
        "state": state.debug_summary,
        "state_detail": _state_detail(state),
        "score": state.current_score,
        "money": state.money,
        "extra": {
            "bridge_joker_smoke": scenario.manifest_row(),
        },
    }
    if action is not None:
        row["chosen_action"] = action.to_json()
    return row


def _state_from_call(payload: object) -> GameState:
    if not isinstance(payload, dict):
        raise RuntimeError(f"Bridge returned non-state payload: {payload!r}")
    if "message" in payload and "name" in payload and "state" not in payload and "phase" not in payload:
        raise RuntimeError(f"Bridge scenario endpoint returned error: {payload}")
    return GameState.from_mapping(payload)


def _filter_scenarios(
    scenarios: Sequence[JokerSmokeScenario],
    only: Sequence[str],
) -> list[JokerSmokeScenario]:
    if not only:
        return list(scenarios)
    wanted = {item.strip().lower() for item in only if item.strip()}
    return [
        scenario
        for scenario in scenarios
        if scenario.joker_name.lower() in wanted or scenario.joker_key.lower() in wanted or scenario.scenario_id.lower() in wanted
    ]


def _hand_for_joker(name: str, action_type: ActionType) -> tuple[tuple[CardSpec, ...], int]:
    if action_type == ActionType.DISCARD:
        return DISCARD_HAND_OVERRIDES.get(name, (FIVE_MIXED_SUITS, 5))
    if name in HAND_OVERRIDES:
        return HAND_OVERRIDES[name]
    return ROYAL_FLUSH, 5


def _joker_stack_for(key: str, name: str) -> tuple[JokerSpec, ...]:
    if name == "Blueprint":
        return (JokerSpec(key), JokerSpec("j_joker"))
    if name == "Brainstorm":
        return (JokerSpec("j_joker"), JokerSpec(key))
    if name == "Baseball Card":
        return (JokerSpec(key), JokerSpec("j_mime"))
    if name == "Ceremonial Dagger":
        return (JokerSpec(key), JokerSpec("j_joker"))
    if name == "Madness":
        return (JokerSpec(key), JokerSpec("j_joker"))
    if name == "Campfire":
        return (JokerSpec(key), JokerSpec("j_joker"))
    return (JokerSpec(key),)


def _note_for_joker(name: str, action_type: ActionType) -> str:
    if name in {"Bloodstone", "Space Joker", "Gros Michel", "Cavendish"}:
        return "Stochastic effect may require repeated oracle rows for distribution-level validation."
    if action_type != ActionType.PLAY_HAND:
        return f"Uses {action_type.value} because this joker's visible effect is not primarily a single hand score."
    return "Single controlled play-hand transition."


def _load_joker_pool() -> tuple[dict[str, object], ...]:
    data = json.loads(SHOP_POOLS_PATH.read_text(encoding="utf-8"))
    jokers = data.get("jokers")
    if not isinstance(jokers, list):
        raise RuntimeError(f"shop_pools.json has no joker list: {SHOP_POOLS_PATH}")
    return tuple(joker for joker in jokers if isinstance(joker, dict) and joker.get("key") and joker.get("name"))


if __name__ == "__main__":
    raise SystemExit(main())
