"""Pure-Python Balatro run loop for fast bot iteration.

This runner is deliberately built from the same small transition functions used
by Phase 7 search. It is not the source of truth for official benchmarks; the
Balatro bridge still owns that. The local runner is for quick same-simulator
iteration, smoke tests, and generating cheap exploratory trajectories.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
from random import Random
from time import perf_counter
from typing import Any, Iterable, Mapping

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import Card, GamePhase, GameState, Joker, Stake, with_derived_legal_actions
from balatro_ai.bots.base import Bot
from balatro_ai.bots.registry import create_bot
from balatro_ai.eval.metrics import RunResult
from balatro_ai.rules.hand_evaluator import HandType, debuffed_suits_for_blind, evaluate_played_cards
from balatro_ai.rules.joker_compat import is_blueprint_compatible
from balatro_ai.search.forward_sim import (
    StochasticPlayOutcomes,
    simulate_buy,
    simulate_cash_out,
    simulate_choose_pack_card,
    simulate_discard,
    simulate_end_shop,
    simulate_open_pack,
    simulate_play,
    simulate_reroll,
    simulate_select_blind,
    simulate_sell,
    simulate_use_consumable,
    _joker_from_item,
)
from balatro_ai.search.shop_sampler import ShopSampler


RANKS = ("A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2")
SUITS = ("S", "H", "D", "C")
SEALS = ("Red", "Blue", "Gold", "Purple")
BLIND_KINDS = ("SMALL", "BIG", "BOSS")
SOURCE_BOSS_BLINDS = (
    {"key": "bl_hook", "name": "The Hook", "min_ante": 1, "dollars": 5, "mult": 2, "showdown": False},
    {"key": "bl_club", "name": "The Club", "min_ante": 1, "dollars": 5, "mult": 2, "showdown": False},
    {"key": "bl_manacle", "name": "The Manacle", "min_ante": 1, "dollars": 5, "mult": 2, "showdown": False},
    {"key": "bl_psychic", "name": "The Psychic", "min_ante": 1, "dollars": 5, "mult": 2, "showdown": False},
    {"key": "bl_goad", "name": "The Goad", "min_ante": 1, "dollars": 5, "mult": 2, "showdown": False},
    {"key": "bl_head", "name": "The Head", "min_ante": 1, "dollars": 5, "mult": 2, "showdown": False},
    {"key": "bl_pillar", "name": "The Pillar", "min_ante": 1, "dollars": 5, "mult": 2, "showdown": False},
    {"key": "bl_window", "name": "The Window", "min_ante": 1, "dollars": 5, "mult": 2, "showdown": False},
    {"key": "bl_mouth", "name": "The Mouth", "min_ante": 2, "dollars": 5, "mult": 2, "showdown": False},
    {"key": "bl_fish", "name": "The Fish", "min_ante": 2, "dollars": 5, "mult": 2, "showdown": False},
    {"key": "bl_wall", "name": "The Wall", "min_ante": 2, "dollars": 5, "mult": 4, "showdown": False},
    {"key": "bl_house", "name": "The House", "min_ante": 2, "dollars": 5, "mult": 2, "showdown": False},
    {"key": "bl_wheel", "name": "The Wheel", "min_ante": 2, "dollars": 5, "mult": 2, "showdown": False},
    {"key": "bl_arm", "name": "The Arm", "min_ante": 2, "dollars": 5, "mult": 2, "showdown": False},
    {"key": "bl_water", "name": "The Water", "min_ante": 2, "dollars": 5, "mult": 2, "showdown": False},
    {"key": "bl_needle", "name": "The Needle", "min_ante": 2, "dollars": 5, "mult": 1, "showdown": False},
    {"key": "bl_flint", "name": "The Flint", "min_ante": 2, "dollars": 5, "mult": 2, "showdown": False},
    {"key": "bl_mark", "name": "The Mark", "min_ante": 2, "dollars": 5, "mult": 2, "showdown": False},
    {"key": "bl_tooth", "name": "The Tooth", "min_ante": 3, "dollars": 5, "mult": 2, "showdown": False},
    {"key": "bl_eye", "name": "The Eye", "min_ante": 3, "dollars": 5, "mult": 2, "showdown": False},
    {"key": "bl_plant", "name": "The Plant", "min_ante": 4, "dollars": 5, "mult": 2, "showdown": False},
    {"key": "bl_serpent", "name": "The Serpent", "min_ante": 5, "dollars": 5, "mult": 2, "showdown": False},
    {"key": "bl_ox", "name": "The Ox", "min_ante": 6, "dollars": 5, "mult": 2, "showdown": False},
    {"key": "bl_final_bell", "name": "Cerulean Bell", "min_ante": 8, "dollars": 8, "mult": 2, "showdown": True},
    {"key": "bl_final_leaf", "name": "Verdant Leaf", "min_ante": 8, "dollars": 8, "mult": 2, "showdown": True},
    {"key": "bl_final_vessel", "name": "Violet Vessel", "min_ante": 8, "dollars": 8, "mult": 6, "showdown": True},
    {"key": "bl_final_acorn", "name": "Amber Acorn", "min_ante": 8, "dollars": 8, "mult": 2, "showdown": True},
    {"key": "bl_final_heart", "name": "Crimson Heart", "min_ante": 8, "dollars": 8, "mult": 2, "showdown": True},
)
BOSS_METADATA_BY_NAME = {str(record["name"]): record for record in SOURCE_BOSS_BLINDS}
BOSS_KEY_BY_NAME = {str(record["name"]): str(record["key"]) for record in SOURCE_BOSS_BLINDS}
BOSS_NAME_BY_KEY = {str(record["key"]): str(record["name"]) for record in SOURCE_BOSS_BLINDS}
SOURCE_NORMAL_BOSS_POOL = tuple(str(record["name"]) for record in SOURCE_BOSS_BLINDS if not record["showdown"])
SOURCE_SHOWDOWN_BOSS_POOL = tuple(str(record["name"]) for record in SOURCE_BOSS_BLINDS if record["showdown"])
DETERMINISTIC_BOSS_POOL = SOURCE_NORMAL_BOSS_POOL
ANTE_SMALL_BLIND_SCORES = {
    1: 300,
    2: 800,
    3: 2000,
    4: 5000,
    5: 11000,
    6: 20000,
    7: 35000,
    8: 50000,
}
TAG_KEYS_BY_NAME = {
    "Uncommon Tag": "tag_uncommon",
    "Rare Tag": "tag_rare",
    "Negative Tag": "tag_negative",
    "Foil Tag": "tag_foil",
    "Holographic Tag": "tag_holo",
    "Polychrome Tag": "tag_polychrome",
    "Investment Tag": "tag_investment",
    "Voucher Tag": "tag_voucher",
    "Boss Tag": "tag_boss",
    "Standard Tag": "tag_standard",
    "Charm Tag": "tag_charm",
    "Meteor Tag": "tag_meteor",
    "Buffoon Tag": "tag_buffoon",
    "Handy Tag": "tag_handy",
    "Garbage Tag": "tag_garbage",
    "Ethereal Tag": "tag_ethereal",
    "Coupon Tag": "tag_coupon",
    "Double Tag": "tag_double",
    "Juggle Tag": "tag_juggle",
    "D6 Tag": "tag_d_six",
    "Top-up Tag": "tag_top_up",
    "Skip Tag": "tag_skip",
    "Orbital Tag": "tag_orbital",
    "Economy Tag": "tag_economy",
}
TAG_NAMES_BY_KEY = {key: name for name, key in TAG_KEYS_BY_NAME.items()}
TAG_MIN_ANTES = {
    "Negative Tag": 2,
    "Standard Tag": 2,
    "Meteor Tag": 2,
    "Buffoon Tag": 2,
    "Handy Tag": 2,
    "Garbage Tag": 2,
    "Ethereal Tag": 2,
    "Top-up Tag": 2,
    "Orbital Tag": 2,
}
PACK_KIND_BY_TAG = {
    "Standard Tag": "Standard",
    "Charm Tag": "Arcana",
    "Meteor Tag": "Celestial",
    "Buffoon Tag": "Buffoon",
    "Ethereal Tag": "Spectral",
}
EDITION_BY_TAG = {
    "Negative Tag": "NEGATIVE",
    "Foil Tag": "FOIL",
    "Holographic Tag": "HOLOGRAPHIC",
    "Polychrome Tag": "POLYCHROME",
}
EDITION_EXTRA_COST = {"FOIL": 2, "HOLOGRAPHIC": 3, "POLYCHROME": 5, "NEGATIVE": 5}
RARITY_BY_TAG = {
    "Uncommon Tag": "uncommon",
    "Rare Tag": "rare",
}


class LocalSimError(RuntimeError):
    """Raised when the local simulator cannot apply a requested action."""


@dataclass(frozen=True, slots=True)
class LocalSimOptions:
    seed: int
    stake: str = "white"
    max_steps: int = 1000
    initial_deck: tuple[Card, ...] | None = None
    boss_pool: tuple[str, ...] = DETERMINISTIC_BOSS_POOL


@dataclass(slots=True)
class LocalBalatroSimulator:
    """Small stateful simulator with an env-like reset/step API."""

    seed: int
    stake: str = "white"
    sampler: ShopSampler = field(default_factory=ShopSampler.from_default_data)
    initial_deck: tuple[Card, ...] | None = None
    boss_pool: tuple[str, ...] = DETERMINISTIC_BOSS_POOL
    state: GameState | None = None
    _rng: Random = field(init=False, repr=False)
    _boss_by_ante: dict[int, str] = field(default_factory=dict, init=False, repr=False)
    _boss_use_count: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _tag_by_blind: dict[tuple[int, str], str] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = Random(self.seed)

    def reset(self, seed: int | None = None) -> GameState:
        """Start a fresh local White Stake-style run."""

        if seed is not None:
            self.seed = seed
        self._rng = Random(self.seed)
        self._boss_by_ante = {}
        self._boss_use_count = {}
        self._tag_by_blind = {}
        deck = self.initial_deck if self.initial_deck is not None else _shuffled_standard_deck(self._rng)
        state = GameState(
            phase=GamePhase.BLIND_SELECT,
            stake=_parse_stake(self.stake),
            seed=self.seed,
            ante=1,
            money=4,
            deck_size=len(deck),
            known_deck=deck,
            hands_remaining=4,
            discards_remaining=4,
            hand_levels=_base_hand_levels(),
            modifiers=_base_modifiers(),
        )
        self.state = self._with_tagged_blind_selection_surface(
            state,
            ante=1,
            blind_kind="SMALL",
            boss_name=self._boss_for_ante(1),
        )
        return self.state

    def _with_tagged_blind_selection_surface(
        self,
        state: GameState,
        *,
        ante: int,
        blind_kind: str,
        boss_name: str,
    ) -> GameState:
        surfaced = _with_blind_selection_surface(state, ante=ante, blind_kind=blind_kind, boss_name=boss_name)
        return self._with_visible_skip_tags(surfaced)

    def _with_next_tagged_blind_surface(
        self,
        state: GameState,
        *,
        cleared_state: GameState,
        phase: GamePhase,
        boss_name: str,
    ) -> GameState:
        ante, next_kind = _next_blind_position(cleared_state)
        if ante > 8:
            return replace(state, phase=GamePhase.RUN_OVER, run_over=True, won=True, ante=8, legal_actions=())
        surfaced = self._with_tagged_blind_selection_surface(state, ante=ante, blind_kind=next_kind, boss_name=boss_name)
        return replace(surfaced, phase=phase)

    def _with_visible_skip_tags(self, state: GameState) -> GameState:
        current_blind = state.modifiers.get("current_blind")
        if not isinstance(current_blind, Mapping):
            return state
        modifiers = dict(state.modifiers)
        blinds = {
            str(key): dict(value) if isinstance(value, Mapping) else value
            for key, value in _mapping(modifiers.get("blinds")).items()
        }
        for _blind_key, blind in tuple(blinds.items()):
            if not isinstance(blind, dict):
                continue
            kind = str(blind.get("type", "")).upper()
            if kind == "BOSS":
                blind.pop("tag", None)
                blind.pop("tag_key", None)
                blind.pop("skip_tag", None)
                continue
            tag_name = self._tag_for_blind(state.ante, kind)
            tag_payload = _tag_payload(tag_name, kind)
            blind["tag"] = tag_name
            blind["tag_key"] = TAG_KEYS_BY_NAME.get(tag_name, tag_name)
            blind["skip_tag"] = tag_payload
        selected_kind = str(current_blind.get("type", "")).upper()
        selected = dict(current_blind)
        if selected_kind != "BOSS":
            tag_name = self._tag_for_blind(state.ante, selected_kind)
            selected["tag"] = tag_name
            selected["tag_key"] = TAG_KEYS_BY_NAME.get(tag_name, tag_name)
            selected["skip_tag"] = _tag_payload(tag_name, selected_kind)
        else:
            selected.pop("tag", None)
            selected.pop("tag_key", None)
            selected.pop("skip_tag", None)
        modifiers["current_blind"] = selected
        modifiers["blinds"] = blinds
        return replace(state, modifiers=modifiers)

    def _tag_for_blind(self, ante: int, blind_kind: str) -> str:
        key = (ante, blind_kind.upper())
        if key not in self._tag_by_blind:
            self._tag_by_blind[key] = self._sample_skip_tag(ante)
        return self._tag_by_blind[key]

    def _sample_skip_tag(self, ante: int) -> str:
        candidates = tuple(name for name in TAG_KEYS_BY_NAME if ante >= TAG_MIN_ANTES.get(name, 1))
        return self._rng.choice(candidates or tuple(TAG_KEYS_BY_NAME))

    def step(self, action: Action) -> GameState:
        """Apply one bot action and return the next state."""

        if self.state is None:
            raise RuntimeError("Call reset() before step().")
        state = self.state
        next_state = self._apply_action(state, action)
        self.state = _with_local_legal_actions(next_state)
        return self.state

    def _apply_action(self, state: GameState, action: Action) -> GameState:
        if action.action_type == ActionType.SELECT_BLIND:
            return self._select_blind(state)
        if action.action_type == ActionType.SKIP_BLIND:
            return self._skip_blind(state)
        if action.action_type == ActionType.PLAY_HAND:
            return self._play_or_discard(state, action, play=True)
        if action.action_type == ActionType.DISCARD:
            return self._play_or_discard(state, action, play=False)
        if action.action_type == ActionType.CASH_OUT:
            return self._cash_out(state)
        if action.action_type == ActionType.END_SHOP:
            return self._end_shop(state)
        if action.action_type == ActionType.BUY:
            return self._buy(state, action)
        if action.action_type == ActionType.SELL:
            return self._sell(state, action)
        if action.action_type == ActionType.USE_CONSUMABLE:
            return self._use_consumable(state, action)
        if action.action_type == ActionType.REROLL:
            if state.phase == GamePhase.BLIND_SELECT or str(action.metadata.get("kind", "")) == "boss":
                return self._reroll_boss_voucher(state)
            return simulate_reroll(state, action, self.sampler.sample_shop(state, rng=self._rng))
        if action.action_type == ActionType.OPEN_PACK:
            return self._open_pack(state, action)
        if action.action_type == ActionType.CHOOSE_PACK_CARD:
            return self._choose_pack_card(state, action)
        if action.action_type == ActionType.REARRANGE:
            return self._rearrange(state, action)
        if action.action_type == ActionType.NO_OP:
            return self._no_op(state)
        raise LocalSimError(f"Unsupported local action: {action.action_type.value}")

    def _buy(self, state: GameState, action: Action) -> GameState:
        bought = simulate_buy(state, action)
        if not _is_overstock_voucher_buy(state, action):
            return bought
        current_shop = _modifier_items(bought.modifiers, "shop_cards")
        filled_shop = self.sampler.fill_shop_to_slot_count(bought, current_shop, rng=self._rng)
        return _with_shop_cards(bought, filled_shop)

    def _sell(self, state: GameState, action: Action) -> GameState:
        index = _action_index(action)
        if index is None or not 0 <= index < len(state.jokers):
            return simulate_sell(state, action)
        sold = state.jokers[index]
        created_joker = None
        if sold.name == "Invisible Joker" and _invisible_joker_ready(sold):
            candidates = tuple(joker for joker_index, joker in enumerate(state.jokers) if joker_index != index)
            if candidates:
                created_joker = _joker_payload_from_joker(self._rng.choice(candidates))
        return simulate_sell(state, action, created_joker=created_joker)

    def _use_consumable(self, state: GameState, action: Action) -> GameState:
        index = _action_index(action)
        if index is None or not 0 <= index < len(state.consumables):
            return simulate_use_consumable(state, action)
        name = state.consumables[index]
        return simulate_use_consumable(state, action, **self._consumable_injections(state, name, storage_use=True))

    def _select_blind(self, state: GameState) -> GameState:
        state = self._apply_round_start_tags(state)
        state = self._apply_boss_start_effects_before_draw(state)
        state = _with_shuffled_known_deck(state, self._rng)
        draw_count = min(_hand_size(state), state.deck_size)
        drawn_cards = _draw_from_known_deck(state, draw_count)
        drawn_cards = self._drawn_cards_after_blind_flips(state, drawn_cards)
        drawn_cards = _cards_after_blind_debuffs(state, drawn_cards)
        selected = simulate_select_blind(
            state,
            drawn_cards=drawn_cards,
            created_hand_cards=self._certificate_cards(state),
            created_deck_cards=self._marble_cards(state),
            created_consumables=self._cartomancer_consumables(state),
            created_jokers=self._riff_raff_jokers(state),
            next_ancient_suit=self._ancient_suit(state),
            next_idol_card=self._idol_card(state),
            next_castle_suit=self._castle_suit(state),
            next_mail_rank=self._mail_rank(state),
            ceremonial_destroyed_joker_index=self._ceremonial_destroyed_joker_index(state),
            madness_destroyed_joker_index=self._madness_destroyed_joker_index(state),
        )
        selected = self._with_cerulean_forced_selection_for_next_hand(selected)
        return self._with_crimson_heart_disabled_for_next_hand(selected)

    def _skip_blind(self, state: GameState) -> GameState:
        kind = _blind_kind(state)
        if kind == "BOSS":
            raise LocalSimError("Cannot skip a boss blind.")
        modifiers = _with_added_tag(dict(state.modifiers), _current_skip_tag(state))
        modifiers["skips"] = _int_value(modifiers.get("skips")) + 1
        skipped = _jokers_after_skip(state.jokers)
        advanced = self._advance_to_next_blind_selection(replace(state, jokers=skipped, modifiers=modifiers))
        return self._apply_blind_choice_tags(advanced)

    def _play_or_discard(self, state: GameState, action: Action, *, play: bool) -> GameState:
        _validate_card_action(state, action)
        hook_discarded_cards = self._hook_discards_for_play(state, action) if play else ()
        selected = set(action.card_indices)
        held_count = sum(1 for index in range(len(state.hand)) if index not in selected)
        held_count_after_hook = max(0, held_count - len(hook_discarded_cards))
        draw_count = min(max(0, _hand_size(state) - held_count), state.deck_size)
        if play:
            draw_count = min(max(0, _hand_size(state) - held_count_after_hook), state.deck_size)
        if _serpent_draws_three(state):
            draw_count = min(3, state.deck_size)
        drawn_cards = _draw_from_known_deck(state, draw_count)
        drawn_cards = self._drawn_cards_after_blind_flips(state, drawn_cards, after_play=play, after_discard=not play)
        drawn_cards = _cards_after_blind_debuffs(state, drawn_cards)
        if play:
            stochastic_outcomes = self._stochastic_play_outcomes(state, action, hook_discarded_cards)
            next_state = simulate_play(
                state,
                action,
                drawn_cards=drawn_cards,
                created_consumables=self._play_created_consumables(state, action, stochastic_outcomes),
                stochastic_outcomes=stochastic_outcomes,
            )
        else:
            next_state = simulate_discard(state, action, drawn_cards=drawn_cards)
        return self._with_cerulean_forced_selection_for_next_hand(next_state)

    def _hook_discards_for_play(self, state: GameState, action: Action) -> tuple[Card, ...]:
        if state.blind != "The Hook" or _boss_disabled(state):
            return ()
        selected = set(action.card_indices)
        held_cards = tuple(card for index, card in enumerate(state.hand) if index not in selected)
        count = min(2, len(held_cards))
        if count <= 0:
            return ()
        discarded_indexes = set(self._rng.sample(range(len(held_cards)), count))
        return tuple(card for index, card in enumerate(held_cards) if index in discarded_indexes)

    def _stochastic_play_outcomes(
        self,
        state: GameState,
        action: Action,
        hook_discarded_cards: tuple[Card, ...],
    ) -> StochasticPlayOutcomes:
        selected_cards, held_cards = _split_action_cards(state, action)
        held_cards_after_hook = _without_card_instances(held_cards, hook_discarded_cards)
        blind_name = "" if _boss_disabled(state) else state.blind
        evaluation = evaluate_played_cards(
            selected_cards,
            state.hand_levels,
            debuffed_suits=debuffed_suits_for_blind(blind_name),
            blind_name=blind_name,
            jokers=state.jokers,
            discards_remaining=state.discards_remaining,
            hands_remaining=state.hands_remaining,
            held_cards=held_cards_after_hook,
            deck_size=state.deck_size,
            money=state.money,
            played_hand_types_this_round=_played_hand_types_this_round(state),
            played_hand_counts=_played_hand_counts(state),
        )
        scored_triggers = _scored_trigger_cards(selected_cards, evaluation, state.jokers, state.hands_remaining)
        scored_once = tuple(selected_cards[index] for index in evaluation.scoring_indices)
        probability_multiplier = _probability_multiplier(state)
        lucky_mult_triggers = 0
        lucky_money_triggers = 0
        lucky_card_triggers = 0
        for card in scored_triggers:
            if not _is_lucky_card(card):
                continue
            mult_hit = self._roll_odds(5, probability_multiplier=probability_multiplier)
            money_hit = self._roll_odds(15, probability_multiplier=probability_multiplier)
            lucky_mult_triggers += int(mult_hit)
            lucky_money_triggers += int(money_hit)
            lucky_card_triggers += int(mult_hit or money_hit)
        shattered_glass_cards = tuple(
            card
            for card in scored_once
            if _is_glass_card(card) and self._roll_odds(_glass_shatter_odds(card), probability_multiplier=probability_multiplier)
        )
        removed_jokers: list[str] = []
        if self._roll_joker_extinction(state, "Gros Michel", odds=6, probability_multiplier=probability_multiplier):
            removed_jokers.append("Gros Michel")
        if self._roll_joker_extinction(state, "Cavendish", odds=1000, probability_multiplier=probability_multiplier):
            removed_jokers.append("Cavendish")
        return StochasticPlayOutcomes(
            hook_discarded_cards=hook_discarded_cards,
            shattered_glass_cards=shattered_glass_cards,
            misprint_mult=self._misprint_mult_for_play(state),
            bloodstone_triggers=sum(
                1
                for card in scored_triggers
                if _card_has_suit(card, "H", state.jokers) and self._roll_odds(2, probability_multiplier=probability_multiplier)
            ),
            business_card_triggers=sum(
                1
                for card in scored_triggers
                if _is_face_card(card, state.jokers) and self._roll_odds(2, probability_multiplier=probability_multiplier)
            ),
            reserved_parking_triggers=sum(
                1
                for card in held_cards_after_hook
                if _is_face_card(card, state.jokers) and self._roll_odds(2, probability_multiplier=probability_multiplier)
            ),
            space_joker_triggers=sum(
                1
                for _ in range(_active_joker_count(state, "Space Joker"))
                if self._roll_odds(4, probability_multiplier=probability_multiplier)
            ),
            lucky_card_mult_triggers=lucky_mult_triggers,
            lucky_card_money_triggers=lucky_money_triggers,
            lucky_card_triggers=lucky_card_triggers,
            matador_triggered=None,
            removed_jokers=tuple(removed_jokers),
            crimson_heart_disabled_joker_index=self._crimson_heart_disabled_joker_index_for_next_hand(state),
        )

    def _play_created_consumables(
        self,
        state: GameState,
        action: Action,
        stochastic_outcomes: StochasticPlayOutcomes,
    ) -> tuple[Mapping[str, Any], ...]:
        selected_cards, held_cards = _split_action_cards(state, action)
        held_cards_after_hook = _without_card_instances(held_cards, stochastic_outcomes.hook_discarded_cards)
        blind_name = "" if _boss_disabled(state) else state.blind
        evaluation = evaluate_played_cards(
            selected_cards,
            state.hand_levels,
            debuffed_suits=debuffed_suits_for_blind(blind_name),
            blind_name=blind_name,
            jokers=state.jokers,
            discards_remaining=state.discards_remaining,
            hands_remaining=state.hands_remaining,
            held_cards=held_cards_after_hook,
            deck_size=state.deck_size,
            money=state.money,
            played_hand_types_this_round=_played_hand_types_this_round(state),
            played_hand_counts=_played_hand_counts(state),
            stochastic_outcomes=stochastic_outcomes.__dict__ if hasattr(stochastic_outcomes, "__dict__") else {},
        )
        room = _consumable_open_slots(state)
        created: list[Mapping[str, Any]] = []
        if room <= 0:
            return ()
        if _sixth_sense_triggers(state, selected_cards):
            count = min(room, _active_joker_count(state, "Sixth Sense"))
            created.extend(self.sampler.sample_card_of_type(state, "Spectral", self._rng) for _ in range(count))
            room -= count
        if room > 0 and evaluation.hand_type == HandType.STRAIGHT_FLUSH:
            count = min(room, _active_joker_count(state, "Seance"))
            created.extend(self.sampler.sample_card_of_type(state, "Spectral", self._rng) for _ in range(count))
            room -= count
        tarot_count = 0
        if room > 0 and evaluation.hand_type in {HandType.STRAIGHT, HandType.STRAIGHT_FLUSH} and any(
            _normalize_rank(card.rank) == "A" for card in selected_cards
        ):
            tarot_count += _active_joker_count(state, "Superposition")
        if room > 0 and state.money <= 4:
            tarot_count += _active_joker_count(state, "Vagabond")
        eight_ball_candidates = _active_joker_count(state, "8 Ball") * sum(
            1
            for card in _scored_trigger_cards(selected_cards, evaluation, state.jokers, state.hands_remaining)
            if _normalize_rank(card.rank) == "8"
        )
        tarot_count += sum(
            1
            for _ in range(eight_ball_candidates)
            if self._roll_odds(4, probability_multiplier=_probability_multiplier(state))
        )
        tarot_count = min(room, tarot_count)
        created.extend(self.sampler.sample_card_of_type(state, "Tarot", self._rng) for _ in range(tarot_count))
        room -= tarot_count
        if (
            room > 0
            and state.required_score > 0
            and state.current_score + int(evaluation.score) >= state.required_score
        ):
            blue_seal_count = sum(
                1 for card in held_cards_after_hook if not card.debuffed and str(card.seal or "").lower() == "blue"
            )
            count = min(room, blue_seal_count)
            created.extend(self.sampler.sample_card_of_type(state, "Planet", self._rng) for _ in range(count))
        return tuple(created)

    def _roll_odds(self, odds: int | float, *, probability_multiplier: float) -> bool:
        if odds <= 0:
            return True
        return self._rng.random() < min(1.0, probability_multiplier / float(odds))

    def _roll_joker_extinction(
        self,
        state: GameState,
        name: str,
        *,
        odds: int,
        probability_multiplier: float,
    ) -> bool:
        return _active_joker_count(state, name) > 0 and self._roll_odds(odds, probability_multiplier=probability_multiplier)

    def _misprint_mult_for_play(self, state: GameState) -> int:
        if _active_joker_count(state, "Misprint") <= 0:
            return 0
        return self._rng.randint(0, 23)

    def _with_crimson_heart_disabled_for_next_hand(self, state: GameState) -> GameState:
        disabled_index = self._crimson_heart_disabled_joker_index_for_next_hand(state)
        if disabled_index is None:
            return state
        return replace(state, jokers=_jokers_with_only_disabled_index(state.jokers, disabled_index))

    def _crimson_heart_disabled_joker_index_for_next_hand(self, state: GameState) -> int | None:
        if state.blind != "Crimson Heart" or _boss_disabled(state) or not state.jokers:
            return None
        if len(state.jokers) < 2:
            return self._rng.randrange(len(state.jokers))
        candidates = [index for index, joker in enumerate(state.jokers) if not _joker_disabled(joker)]
        if not candidates:
            candidates = list(range(len(state.jokers)))
        return self._rng.choice(candidates)

    def _cash_out(self, state: GameState) -> GameState:
        next_state = simulate_cash_out(
            state,
            next_shop_cards=self.sampler.sample_shop(state, rng=self._rng),
            next_voucher_cards=_optional_tuple(self.sampler.sample_voucher(state, self._rng)),
            next_booster_packs=self.sampler.sample_boosters(state, rng=self._rng),
            next_to_do_targets=self._to_do_targets(state),
        )
        if _blind_kind(state) == "BOSS":
            next_state = _state_after_boss_cleared(next_state, cleared_boss=state.blind)
        next_state = _with_fresh_shop_modifiers(next_state)
        next_state = self._apply_eval_tags(next_state, cleared_state=state)
        next_state = self._apply_shop_tags(next_state)
        if state.ante >= 8 and _blind_kind(state) == "BOSS":
            return replace(next_state, phase=GamePhase.RUN_OVER, run_over=True, won=True, ante=8, legal_actions=())
        surfaced = _with_cash_out_shop_surface(next_state, cleared_state=state, boss_name=self._boss_for_next_surface(state))
        return self._with_visible_skip_tags(surfaced)

    def _end_shop(self, state: GameState) -> GameState:
        cleared_state = _cleared_blind_state_for_progression(state)
        next_state = simulate_end_shop(state, created_consumables=self._perkeo_consumables(state))
        surfaced = self._with_next_tagged_blind_surface(
            next_state,
            cleared_state=cleared_state,
            phase=GamePhase.BLIND_SELECT,
            boss_name=self._boss_for_next_surface(cleared_state),
        )
        return self._apply_blind_choice_tags(surfaced)

    def _advance_to_next_blind_selection(self, state: GameState) -> GameState:
        return self._with_next_tagged_blind_surface(
            state,
            cleared_state=state,
            phase=GamePhase.BLIND_SELECT,
            boss_name=self._boss_for_next_surface(state),
        )

    def _apply_blind_choice_tags(self, state: GameState) -> GameState:
        state = self._apply_immediate_tags(state)
        return self._apply_new_blind_choice_tags(state)

    def _apply_immediate_tags(self, state: GameState) -> GameState:
        tags = _pending_tags(state.modifiers)
        if not tags:
            return state
        remaining: list[str] = []
        money = state.money
        jokers = list(state.jokers)
        hand_levels = dict(state.hand_levels)
        modifiers = dict(state.modifiers)
        for tag in tags:
            if tag == "Economy Tag":
                money += min(40, max(0, money))
            elif tag == "Skip Tag":
                money += 5 * _int_value(modifiers.get("skips"))
            elif tag == "Handy Tag":
                money += _int_value(modifiers.get("hands_played_total", modifiers.get("hands_played")))
            elif tag == "Garbage Tag":
                money += _int_value(modifiers.get("unused_discards", modifiers.get("discards_remaining")))
            elif tag == "Top-up Tag":
                open_slots = max(0, _normal_joker_open_slots(replace(state, jokers=tuple(jokers))))
                for index in range(min(2, open_slots)):
                    payload = self.sampler.sample_joker_by_rarity(state, "common", self._rng, key_append=f"tag_top{index}")
                    jokers.append(_joker_from_item(payload))
            elif tag == "Orbital Tag":
                hand_name = _orbital_hand_for_tag(tag, modifiers) or self._rng.choice(tuple(_base_hand_levels()))
                hand_levels[hand_name] = max(1, hand_levels.get(hand_name, 1)) + 3
                modifiers = _with_hand_level_modifier(modifiers, hand_name, hand_levels[hand_name])
            else:
                remaining.append(tag)
        modifiers["tags"] = tuple(remaining)
        return replace(state, money=money, jokers=tuple(jokers), hand_levels=hand_levels, modifiers=modifiers)

    def _apply_new_blind_choice_tags(self, state: GameState) -> GameState:
        tags = _pending_tags(state.modifiers)
        if not tags:
            return state
        for index, tag in enumerate(tags):
            if tag == "Boss Tag":
                rerolled = self._reroll_boss_tag(state)
                return replace(rerolled, modifiers={**rerolled.modifiers, "tags": tags[:index] + tags[index + 1 :]})
            kind = PACK_KIND_BY_TAG.get(tag)
            if not kind:
                continue
            pack = self.sampler.sample_booster_of_kind(state, kind, self._rng, prefer_mega=kind != "Spectral")
            if pack is None:
                return state
            opened = _open_free_tag_pack(state, pack, self.sampler.sample_pack_contents(state, pack, self._rng))
            # Source truth opens at most one free tag pack per blind-choice event.
            return replace(opened, modifiers={**opened.modifiers, "tags": tags[:index] + tags[index + 1 :]})
        return state

    def _reroll_boss_tag(self, state: GameState) -> GameState:
        return self._reroll_boss_surface(state, cost=0, mark_rerolled=False)

    def _reroll_boss_voucher(self, state: GameState) -> GameState:
        if not _boss_reroll_available(state):
            raise LocalSimError("Boss reroll is not available.")
        cost = _boss_reroll_cost(state)
        rerolled = self._reroll_boss_surface(state, cost=cost, mark_rerolled=True)
        return self._apply_new_blind_choice_tags(rerolled)

    def _reroll_boss_surface(self, state: GameState, *, cost: int, mark_rerolled: bool) -> GameState:
        ante = max(1, state.ante)
        old_boss = _boss_name_from_surface(state) or self._boss_for_ante(ante)
        boss_name = self._choose_boss_for_ante(ante, exclude=(old_boss,))
        self._boss_by_ante[ante] = boss_name

        modifiers = dict(state.modifiers)
        if mark_rerolled:
            modifiers["boss_rerolled"] = True
        blinds = {
            str(key): dict(value) if isinstance(value, Mapping) else value
            for key, value in _mapping(modifiers.get("blinds")).items()
        }
        boss_payload = _blind_payload(ante, "BOSS", boss_name, selected=_blind_kind(state) == "BOSS")
        blinds["boss"] = boss_payload
        modifiers["blinds"] = blinds
        modifiers["upcoming_boss"] = boss_payload
        if _blind_kind(state) != "BOSS":
            return replace(state, money=max(0, state.money - cost), modifiers=modifiers)

        modifiers["current_blind"] = boss_payload
        modifiers["blind_reward"] = boss_payload["dollars"]
        return replace(
            state,
            money=max(0, state.money - cost),
            blind=boss_name,
            required_score=int(boss_payload["score"]),
            modifiers=modifiers,
        )

    def _apply_round_start_tags(self, state: GameState) -> GameState:
        tags = _pending_tags(state.modifiers)
        if "Juggle Tag" not in tags:
            return state
        consumed = False
        remaining: list[str] = []
        modifiers = dict(state.modifiers)
        for tag in tags:
            if tag == "Juggle Tag" and not consumed:
                if "hand_size" in modifiers:
                    modifiers["hand_size"] = _int_value(modifiers.get("hand_size")) + 3
                else:
                    modifiers["hand_size_delta"] = _int_value(modifiers.get("hand_size_delta")) + 3
                modifiers["round_start_hand_size_delta"] = _int_value(modifiers.get("round_start_hand_size_delta")) + 3
                consumed = True
            else:
                remaining.append(tag)
        modifiers["tags"] = tuple(remaining)
        return replace(state, modifiers=modifiers)

    def _apply_boss_start_effects_before_draw(self, state: GameState) -> GameState:
        if not _boss_effect_active(state):
            return state
        modifiers = dict(state.modifiers)
        jokers = state.jokers
        if state.blind == "The Manacle":
            if "hand_size" in modifiers:
                modifiers["hand_size"] = max(1, _int_value(modifiers.get("hand_size")) - 1)
            else:
                modifiers["hand_size_delta"] = _int_value(modifiers.get("hand_size_delta")) - 1
            modifiers["round_start_hand_size_delta"] = _int_value(modifiers.get("round_start_hand_size_delta")) - 1
        elif state.blind == "Amber Acorn" and jokers:
            shuffled = list(jokers)
            self._rng.shuffle(shuffled)
            jokers = tuple(_joker_with_face_down_metadata(joker) for joker in shuffled)
        elif state.blind == "The Ox":
            modifiers["most_played_poker_hand"] = _most_played_hand_name(state)
        return replace(state, jokers=jokers, modifiers=modifiers)

    def _apply_eval_tags(self, state: GameState, *, cleared_state: GameState) -> GameState:
        tags = _pending_tags(state.modifiers)
        if not tags or _blind_kind(cleared_state) != "BOSS":
            return state
        payout_count = sum(1 for tag in tags if tag == "Investment Tag")
        if payout_count <= 0:
            return state
        remaining = tuple(tag for tag in tags if tag != "Investment Tag")
        modifiers = dict(state.modifiers)
        modifiers["tags"] = remaining
        return replace(state, money=state.money + 25 * payout_count, modifiers=modifiers)

    def _apply_shop_tags(self, state: GameState) -> GameState:
        tags = _pending_tags(state.modifiers)
        if not tags:
            return state
        remaining: list[str] = []
        shop_cards = list(_modifier_items(state.modifiers, "shop_cards"))
        voucher_cards = list(_modifier_items(state.modifiers, "voucher_cards"))
        booster_packs = list(_modifier_items(state.modifiers, "booster_packs"))
        modifiers = dict(state.modifiers)

        for tag in tags:
            if tag == "D6 Tag":
                modifiers["reroll_cost"] = 0
                modifiers["current_reroll_cost"] = 0
                modifiers["round_reset_reroll_cost"] = 0
            elif tag == "Voucher Tag":
                voucher = self.sampler.sample_voucher(state, self._rng)
                if voucher is None:
                    remaining.append(tag)
                else:
                    voucher_cards.append(voucher)
            elif tag == "Coupon Tag":
                shop_cards = [_couponed_payload(item) for item in shop_cards]
                booster_packs = [_couponed_payload(item) for item in booster_packs]
                modifiers["shop_free"] = True
            elif tag in RARITY_BY_TAG:
                shop_cards.append(
                    _couponed_payload(
                        self.sampler.sample_joker_by_rarity(
                            state,
                            RARITY_BY_TAG[tag],
                            self._rng,
                            key_append=f"tag_{TAG_KEYS_BY_NAME[tag]}",
                        )
                    )
                )
            elif tag in EDITION_BY_TAG:
                edited, shop_cards = _with_first_eligible_tag_edition(shop_cards, EDITION_BY_TAG[tag])
                if not edited:
                    remaining.append(tag)
            else:
                remaining.append(tag)

        modifiers["tags"] = tuple(remaining)
        modifiers["shop_cards"] = tuple(shop_cards)
        modifiers["voucher_cards"] = tuple(voucher_cards)
        modifiers["booster_packs"] = tuple(booster_packs)
        return replace(
            state,
            shop=tuple(_item_label(item) for item in shop_cards),
            pack=(),
            modifiers=modifiers,
        )

    def _open_pack(self, state: GameState, action: Action) -> GameState:
        pack = _indexed_item(_modifier_items(state.modifiers, "booster_packs"), action)
        contents = self.sampler.sample_pack_contents(state, pack, self._rng)
        opened = simulate_open_pack(state, action, contents, created_consumables=self._hallucination_consumables(state))
        modifiers = dict(opened.modifiers)
        modifiers["pack_choices_remaining"] = _pack_choices(pack)
        return replace(opened, modifiers=modifiers)

    def _choose_pack_card(self, state: GameState, action: Action) -> GameState:
        if _is_skip_action(action):
            chosen = simulate_choose_pack_card(state, action)
            return self._finish_pack_choice(chosen, _without_pack_choice_counter(chosen.modifiers))

        pack_cards = _modifier_items(state.modifiers, "pack_cards")
        index = _action_index(action)
        if index is None or not 0 <= index < len(pack_cards):
            raise LocalSimError(f"Pack choice index is outside pack cards: {action.amount}")

        injections = self._consumable_injections(state, _item_label(pack_cards[index]), storage_use=False)
        remaining_choices = max(1, _int_value(state.modifiers.get("pack_choices_remaining"), default=1))
        remaining_pack_cards = tuple(item for item_index, item in enumerate(pack_cards) if item_index != index)
        if remaining_choices > 1 and remaining_pack_cards:
            chosen = simulate_choose_pack_card(
                state,
                action,
                remaining_pack_cards=remaining_pack_cards,
                **injections,
            )
            modifiers = dict(chosen.modifiers)
            modifiers["pack_choices_remaining"] = remaining_choices - 1
            return replace(chosen, modifiers=modifiers)

        chosen = simulate_choose_pack_card(
            state,
            action,
            **injections,
        )
        return self._finish_pack_choice(chosen, _without_pack_choice_counter(chosen.modifiers))

    def _finish_pack_choice(self, state: GameState, modifiers: Mapping[str, object]) -> GameState:
        if modifiers.get("tag_pack_return_phase") != GamePhase.BLIND_SELECT.value:
            return replace(state, modifiers=dict(modifiers))
        updated_modifiers = dict(modifiers)
        updated_modifiers.pop("tag_pack_return_phase", None)
        return replace(
            state,
            phase=GamePhase.BLIND_SELECT,
            shop=(),
            pack=(),
            modifiers=updated_modifiers,
        )

    def _drawn_cards_after_blind_flips(
        self,
        state: GameState,
        drawn_cards: tuple[Card, ...],
        *,
        after_play: bool = False,
        after_discard: bool = False,
    ) -> tuple[Card, ...]:
        if not _boss_effect_active(state):
            return drawn_cards
        if state.blind == "The House":
            if not after_play and not after_discard and not _played_hand_types_this_round(state) and _discard_used_count(state) <= 0:
                return tuple(_card_with_face_down_metadata(card) for card in drawn_cards)
            return drawn_cards
        if state.blind == "The Mark":
            return tuple(
                _card_with_face_down_metadata(card) if _is_face_card(card, state.jokers) else card
                for card in drawn_cards
            )
        if state.blind == "The Fish":
            return tuple(_card_with_face_down_metadata(card) for card in drawn_cards) if after_play else drawn_cards
        if state.blind != "The Wheel":
            return drawn_cards
        probability_multiplier = _probability_multiplier(state)
        return tuple(
            _card_with_face_down_metadata(card) if self._roll_odds(7, probability_multiplier=probability_multiplier) else card
            for card in drawn_cards
        )

    def _with_cerulean_forced_selection_for_next_hand(self, state: GameState) -> GameState:
        if state.blind != "Cerulean Bell" or not _boss_effect_active(state) or not state.hand:
            return state
        if any(_card_is_forced_selection(card) for card in state.hand):
            return state
        forced_index = self._rng.randrange(len(state.hand))
        hand = tuple(_card_with_forced_selection(card) if index == forced_index else card for index, card in enumerate(state.hand))
        return replace(state, hand=hand)

    def _wheel_of_fortune_outcome(self, state: GameState, item: object) -> tuple[int | None, str | None]:
        if _item_label(item) != "The Wheel of Fortune":
            return None, None
        eligible = tuple(index for index, joker in enumerate(state.jokers) if not joker.edition)
        if not eligible or not self._roll_odds(4, probability_multiplier=_probability_multiplier(state)):
            return None, None
        return self._rng.choice(eligible), self._wheel_of_fortune_edition()

    def _wheel_of_fortune_edition(self) -> str:
        poll = self._rng.random()
        if poll > 1 - (0.006 * 25):
            return "POLYCHROME"
        if poll > 1 - (0.02 * 25):
            return "HOLOGRAPHIC"
        return "FOIL"

    def _consumable_injections(self, state: GameState, name: str, *, storage_use: bool) -> dict[str, object]:
        injections: dict[str, object] = {}
        if name == "The Wheel of Fortune":
            joker_index, edition = self._wheel_of_fortune_outcome(state, name)
            injections["wheel_of_fortune_joker_index"] = joker_index
            injections["wheel_of_fortune_edition"] = edition
        elif name == "The Fool":
            injections["created_consumables"] = self._fool_consumables(state, storage_use=storage_use)
        elif name == "The Emperor":
            injections["created_consumables"] = self._created_consumables_of_type(state, "Tarot", storage_use=storage_use, max_count=2)
        elif name == "The High Priestess":
            injections["created_consumables"] = self._created_consumables_of_type(state, "Planet", storage_use=storage_use, max_count=2)
        elif name in {"Judgement", "The Soul", "Wraith"}:
            if _normal_joker_open_slots(state) > 0:
                injections["created_jokers"] = (self.sampler.sample_card_of_type(state, "Joker", self._rng),)
        elif name == "Aura":
            injections["aura_edition"] = self._wheel_of_fortune_edition()
        elif name in {"Familiar", "Grim", "Incantation"}:
            if state.hand:
                injections["destroyed_hand_card_indices"] = (self._rng.randrange(len(state.hand)),)
            injections["created_hand_cards"] = self._spectral_created_hand_cards(name)
        elif name == "Immolate":
            count = min(5, len(state.hand))
            injections["destroyed_hand_card_indices"] = tuple(self._rng.sample(range(len(state.hand)), count)) if count else ()
        elif name == "Sigil":
            injections["sigil_suit"] = self._rng.choice(SUITS)
        elif name == "Ouija":
            injections["ouija_rank"] = self._rng.choice(RANKS)
        elif name in {"Ankh", "Hex", "Ectoplasm"}:
            eligible = tuple(index for index, joker in enumerate(state.jokers) if name == "Ankh" or not joker.edition)
            if eligible:
                injections["spectral_joker_index"] = self._rng.choice(eligible)
        return injections

    def _fool_consumables(self, state: GameState, *, storage_use: bool) -> tuple[str, ...]:
        if self._consumable_open_slots_after_use(state, storage_use=storage_use) <= 0:
            return ()
        last = _last_tarot_planet_label(state)
        if not last or last == "The Fool":
            return ()
        return (last,)

    def _created_consumables_of_type(
        self,
        state: GameState,
        card_type: str,
        *,
        storage_use: bool,
        max_count: int,
    ) -> tuple[Mapping[str, Any], ...]:
        count = min(max_count, self._consumable_open_slots_after_use(state, storage_use=storage_use))
        return tuple(self.sampler.sample_card_of_type(state, card_type, self._rng) for _ in range(count))

    def _consumable_open_slots_after_use(self, state: GameState, *, storage_use: bool) -> int:
        return max(0, _consumable_open_slots(state) + (1 if storage_use else 0))

    def _spectral_created_hand_cards(self, name: str) -> tuple[Card, ...]:
        if name == "Familiar":
            return tuple(_random_playing_card(self._rng, rank=self._rng.choice(("J", "Q", "K")), enhancement=_random_enhancement(self._rng)) for _ in range(3))
        if name == "Grim":
            return tuple(_random_playing_card(self._rng, rank="A", enhancement=_random_enhancement(self._rng)) for _ in range(2))
        if name == "Incantation":
            return tuple(_random_playing_card(self._rng, rank=self._rng.choice(("2", "3", "4", "5", "6", "7", "8", "9", "10")), enhancement=_random_enhancement(self._rng)) for _ in range(4))
        return ()

    def _rearrange(self, state: GameState, action: Action) -> GameState:
        order = action.card_indices
        if sorted(order) != list(range(len(state.jokers))):
            raise LocalSimError(f"Invalid joker rearrange order: {order}")
        return replace(state, jokers=tuple(state.jokers[index] for index in order))

    def _no_op(self, state: GameState) -> GameState:
        if state.phase == GamePhase.BOOSTER_OPENED and not state.pack:
            return replace(state, phase=GamePhase.SHOP, modifiers=_without_pack_choice_counter(state.modifiers))
        return state

    def _boss_for_ante(self, ante: int) -> str:
        if ante not in self._boss_by_ante:
            self._boss_by_ante[ante] = self._choose_boss_for_ante(ante)
        return self._boss_by_ante[ante]

    def _choose_boss_for_ante(self, ante: int, *, exclude: tuple[str, ...] = ()) -> str:
        pool = self._eligible_boss_pool(ante)
        excluded = set(exclude)
        filtered = tuple(name for name in pool if name not in excluded)
        if filtered:
            pool = filtered
        if not _using_source_boss_pool(self.boss_pool):
            return self._rng.choice(pool)

        least_used = min((self._boss_use_count.get(name, 0) for name in pool), default=0)
        candidates = tuple(name for name in pool if self._boss_use_count.get(name, 0) == least_used)
        boss_name = self._rng.choice(candidates or pool)
        self._boss_use_count[boss_name] = self._boss_use_count.get(boss_name, 0) + 1
        return boss_name

    def _eligible_boss_pool(self, ante: int) -> tuple[str, ...]:
        pool = tuple(self.boss_pool or DETERMINISTIC_BOSS_POOL)
        if not _using_source_boss_pool(pool):
            return pool
        if _is_showdown_ante(ante):
            return SOURCE_SHOWDOWN_BOSS_POOL
        return tuple(
            str(record["name"])
            for record in SOURCE_BOSS_BLINDS
            if not record["showdown"] and int(record["min_ante"]) <= max(1, ante)
        )

    def _boss_for_next_surface(self, state: GameState) -> str:
        ante, next_kind = _next_blind_position(state)
        if next_kind == "BOSS":
            existing = _boss_name_from_surface(state)
            if existing:
                self._boss_by_ante.setdefault(ante, existing)
                return existing
        return self._boss_for_ante(ante)

    def _certificate_cards(self, state: GameState) -> tuple[Card, ...]:
        return tuple(_random_playing_card(self._rng, seal=self._rng.choice(SEALS)) for _ in range(_active_joker_count(state, "Certificate")))

    def _marble_cards(self, state: GameState) -> tuple[Card, ...]:
        return tuple(_random_playing_card(self._rng) for _ in range(_active_joker_count(state, "Marble Joker")))

    def _cartomancer_consumables(self, state: GameState) -> tuple[Mapping[str, Any], ...]:
        room = _consumable_open_slots(state)
        count = min(room, _active_joker_count(state, "Cartomancer"))
        return tuple(self.sampler.sample_card_of_type(state, "Tarot", self._rng) for _ in range(count))

    def _riff_raff_jokers(self, state: GameState) -> tuple[Mapping[str, Any], ...]:
        room = _normal_joker_open_slots(state)
        count = min(room, 2 * _active_joker_count(state, "Riff-raff"))
        return tuple(self.sampler.sample_card_of_type(state, "Joker", self._rng) for _ in range(count))

    def _to_do_targets(self, state: GameState) -> tuple[str, ...]:
        count = _active_joker_count(state, "To Do List")
        hands = tuple(hand_type.value for hand_type in HandType)
        return tuple(self._rng.choice(hands) for _ in range(count))

    def _hallucination_consumables(self, state: GameState) -> tuple[Mapping[str, Any], ...]:
        room = _consumable_open_slots(state)
        if room <= 0:
            return ()
        probability_multiplier = _probability_multiplier(state)
        count = sum(
            1
            for _ in range(_active_joker_count(state, "Hallucination"))
            if self._roll_odds(2, probability_multiplier=probability_multiplier)
        )
        count = min(room, count)
        return tuple(self.sampler.sample_card_of_type(state, "Tarot", self._rng) for _ in range(count))

    def _perkeo_consumables(self, state: GameState) -> tuple[str, ...]:
        if not state.consumables:
            return ()
        count = _active_joker_count(state, "Perkeo")
        return tuple(self._rng.choice(state.consumables) for _ in range(count))

    def _ancient_suit(self, state: GameState) -> str | None:
        if _active_joker_count(state, "Ancient Joker") <= 0:
            return None
        previous = _current_target_suit(state.jokers, "Ancient Joker")
        choices = [suit for suit in ("S", "H", "C", "D") if suit != previous]
        return self._rng.choice(choices or ["S", "H", "C", "D"])

    def _idol_card(self, state: GameState) -> Card | None:
        if _active_joker_count(state, "The Idol") <= 0:
            return None
        candidates = _non_stone_full_deck_cards(state)
        return self._rng.choice(candidates) if candidates else Card("A", "S")

    def _castle_suit(self, state: GameState) -> str | None:
        if _active_joker_count(state, "Castle") <= 0:
            return None
        candidates = _non_stone_full_deck_cards(state)
        return self._rng.choice(candidates).suit if candidates else "S"

    def _mail_rank(self, state: GameState) -> str | None:
        if _active_joker_count(state, "Mail-In Rebate") <= 0:
            return None
        candidates = _non_stone_full_deck_cards(state)
        return self._rng.choice(candidates).rank if candidates else "A"

    def _ceremonial_destroyed_joker_index(self, state: GameState) -> int | None:
        for index, joker in enumerate(state.jokers[:-1]):
            if joker.name != "Ceremonial Dagger" or _joker_disabled(joker):
                continue
            target_index = index + 1
            if _joker_eternal(state.jokers[target_index]):
                continue
            return target_index
        return None

    def _madness_destroyed_joker_index(self, state: GameState) -> int | None:
        if _blind_kind(state) == "BOSS" or _active_joker_count(state, "Madness") <= 0:
            return None
        removed_by_dagger = self._ceremonial_destroyed_joker_index(state)
        remaining = tuple(joker for index, joker in enumerate(state.jokers) if index != removed_by_dagger)
        if not any(joker.name == "Madness" and not _joker_disabled(joker) for joker in remaining):
            return None
        candidates = [
            index
            for index, joker in enumerate(remaining)
            if joker.name != "Madness" and not _joker_disabled(joker) and not _joker_eternal(joker)
        ]
        return self._rng.choice(candidates) if candidates else None


def run_local_seed(*, bot: Bot, options: LocalSimOptions) -> RunResult:
    """Run one bot in the local simulator and return benchmark-style metrics."""

    started_at = perf_counter()
    simulator = LocalBalatroSimulator(
        seed=options.seed,
        stake=options.stake,
        initial_deck=options.initial_deck,
        boss_pool=options.boss_pool,
    )
    state = simulator.reset()
    steps = 0
    error: str | None = None
    while not state.run_over and steps < options.max_steps:
        if not state.legal_actions and state.phase != GamePhase.SELECTING_HAND:
            error = f"no_legal_actions:{state.phase.value}"
            break
        try:
            action = bot.choose_action(state)
            state = simulator.step(action)
        except Exception as exc:  # noqa: BLE001 - benchmarking should record bad local transitions.
            error = f"local_error:{type(exc).__name__}:{exc}"
            break
        steps += 1

    runtime = perf_counter() - started_at
    timed_out = steps >= options.max_steps and not state.run_over
    death_reason = None
    if not state.won:
        if error is not None:
            death_reason = error
        elif timed_out:
            death_reason = "max_steps"
        else:
            death_reason = state.blind or state.phase.value
    return RunResult(
        bot_version=bot.name,
        seed=options.seed,
        stake=options.stake,
        won=state.won,
        ante_reached=state.ante,
        final_score=state.current_score,
        final_money=state.money,
        runtime_seconds=runtime,
        death_reason=death_reason,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one seed in the pure-Python local simulator.")
    parser.add_argument("--bot", default="basic_strategy_bot", help="Bot registry name.")
    parser.add_argument("--seed", type=int, required=True, help="Run seed.")
    parser.add_argument("--stake", default="white", help="Stake label; first pass is tuned for white.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum local simulator steps.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_local_seed(
        bot=create_bot(args.bot, seed=args.seed),
        options=LocalSimOptions(seed=args.seed, stake=args.stake, max_steps=args.max_steps),
    )
    print(
        " ".join(
            (
                f"Bot={result.bot_version}",
                f"seed={result.seed}",
                f"stake={result.stake}",
                f"won={result.won}",
                f"ante={result.ante_reached}",
                f"score={result.final_score}",
                f"money={result.final_money}",
                f"runtime={result.runtime_seconds:.3f}s",
                f"death={result.death_reason or '-'}",
            )
        )
    )
    return 0


def _parse_stake(raw: str) -> Stake:
    try:
        return Stake(raw.lower())
    except ValueError:
        return Stake.UNKNOWN


def _using_source_boss_pool(pool: tuple[str, ...]) -> bool:
    return tuple(pool) == DETERMINISTIC_BOSS_POOL


def _is_showdown_ante(ante: int) -> bool:
    return ante >= 2 and ante % 8 == 0


def _base_modifiers() -> dict[str, Any]:
    return {
        "base_hands": 4,
        "base_discards": 4,
        "hand_size": 8,
        "joker_slot_limit": 5,
        "consumable_slot_limit": 2,
        "reroll_cost": 5,
        "current_reroll_cost": 5,
        "round_reset_reroll_cost": 5,
        "reroll_cost_increase": 0,
        "first_shop_buffoon": False,
        "hands": _base_hands_modifier(),
    }


def _base_hand_levels() -> dict[str, int]:
    return {hand_type.value: 1 for hand_type in HandType}


def _base_hands_modifier() -> dict[str, dict[str, int]]:
    return {
        hand_type.value: {"level": 1, "played": 0, "played_this_round": 0, "order": index}
        for index, hand_type in enumerate(HandType, start=1)
    }


def _with_blind_selection_surface(state: GameState, *, ante: int, blind_kind: str, boss_name: str) -> GameState:
    if ante > 8:
        return replace(state, phase=GamePhase.RUN_OVER, run_over=True, won=True, ante=8, legal_actions=())
    blind = _blind_payload(ante, blind_kind, boss_name)
    modifiers = dict(state.modifiers)
    modifiers["current_blind"] = blind
    modifiers["blinds"] = _blinds_for_ante(ante, selected_kind=blind_kind, boss_name=boss_name)
    modifiers["upcoming_boss"] = modifiers["blinds"]["boss"]
    modifiers["blind_reward"] = blind["dollars"]
    modifiers.pop("boss_disabled", None)
    return _with_local_legal_actions(
        replace(
            state,
            phase=GamePhase.BLIND_SELECT,
            ante=ante,
            blind=str(blind["name"]),
            required_score=int(blind["score"]),
            current_score=0,
            hand=(),
            modifiers=modifiers,
            run_over=False,
            won=False,
        )
    )


def _with_next_blind_surface(
    state: GameState,
    *,
    cleared_state: GameState,
    phase: GamePhase,
    boss_name: str,
) -> GameState:
    ante, next_kind = _next_blind_position(cleared_state)
    if ante > 8:
        return replace(state, phase=GamePhase.RUN_OVER, run_over=True, won=True, ante=8, legal_actions=())
    surfaced = _with_blind_selection_surface(state, ante=ante, blind_kind=next_kind, boss_name=boss_name)
    return replace(surfaced, phase=phase)


def _with_cash_out_shop_surface(state: GameState, *, cleared_state: GameState, boss_name: str) -> GameState:
    cleared_kind = _blind_kind(cleared_state)
    surface_ante = cleared_state.ante + 1 if cleared_kind == "BOSS" else cleared_state.ante
    if surface_ante > 8:
        return replace(state, phase=GamePhase.RUN_OVER, run_over=True, won=True, ante=8, legal_actions=())
    surfaced = _with_blind_selection_surface(state, ante=surface_ante, blind_kind="SMALL", boss_name=boss_name)
    modifiers = dict(surfaced.modifiers)
    modifiers["cleared_blind"] = {
        "ante": cleared_state.ante,
        "blind": cleared_state.blind,
        "required_score": cleared_state.required_score,
        "kind": cleared_kind,
    }
    return _with_local_legal_actions(replace(surfaced, phase=GamePhase.SHOP, modifiers=modifiers))


def _state_after_boss_cleared(state: GameState, *, cleared_boss: str) -> GameState:
    jokers = state.jokers
    if cleared_boss == "Amber Acorn":
        jokers = tuple(_joker_without_face_down_metadata(joker) for joker in jokers)
    return replace(
        state,
        known_deck=tuple(_card_without_played_this_ante(card) for card in state.known_deck),
        hand=tuple(_card_without_played_this_ante(card) for card in state.hand),
        jokers=jokers,
    )


def _cleared_blind_state_for_progression(state: GameState) -> GameState:
    raw = state.modifiers.get("cleared_blind")
    if not isinstance(raw, Mapping):
        return state
    kind = str(raw.get("kind", "")).upper()
    if kind not in BLIND_KINDS:
        return state
    ante = _int_value(raw.get("ante"), default=state.ante)
    blind = str(raw.get("blind", state.blind))
    required_score = _int_value(raw.get("required_score"), default=state.required_score)
    modifiers = dict(state.modifiers)
    modifiers["current_blind"] = {
        "name": blind,
        "type": kind,
        "score": required_score,
        "dollars": _blind_reward(kind, blind),
    }
    return replace(state, ante=ante, blind=blind, required_score=required_score, modifiers=modifiers)


def _next_blind_position(state: GameState) -> tuple[int, str]:
    kind = _blind_kind(state)
    if kind == "SMALL":
        return state.ante, "BIG"
    if kind == "BIG":
        return state.ante, "BOSS"
    return state.ante + 1, "SMALL"


def _blinds_for_ante(ante: int, *, selected_kind: str, boss_name: str) -> dict[str, dict[str, Any]]:
    return {
        "small": _blind_payload(ante, "SMALL", boss_name, selected=selected_kind == "SMALL"),
        "big": _blind_payload(ante, "BIG", boss_name, selected=selected_kind == "BIG"),
        "boss": _blind_payload(ante, "BOSS", boss_name, selected=selected_kind == "BOSS"),
    }


def _blind_payload(ante: int, blind_kind: str, boss_name: str, *, selected: bool = True) -> dict[str, Any]:
    if blind_kind == "SMALL":
        name = "Small Blind"
    elif blind_kind == "BIG":
        name = "Big Blind"
    else:
        name = boss_name
    return {
        "key": BOSS_KEY_BY_NAME.get(name, "bl_small" if blind_kind == "SMALL" else "bl_big" if blind_kind == "BIG" else name),
        "name": name,
        "type": blind_kind,
        "score": _required_score(ante, blind_kind, name),
        "dollars": _blind_reward(blind_kind, name),
        "status": "SELECT" if selected else "UPCOMING",
    }


def _required_score(ante: int, blind_kind: str, blind_name: str) -> int:
    small = ANTE_SMALL_BLIND_SCORES.get(ante, _extrapolated_small_blind_score(ante))
    if blind_kind == "SMALL":
        return small
    if blind_kind == "BIG":
        return int(small * 1.5)
    return int(small * float(BOSS_METADATA_BY_NAME.get(blind_name, {}).get("mult", 2)))


def _extrapolated_small_blind_score(ante: int) -> int:
    last = ANTE_SMALL_BLIND_SCORES[max(ANTE_SMALL_BLIND_SCORES)]
    return int(last * (1.6 ** max(0, ante - 8)))


def _blind_reward(blind_kind: str, blind_name: str = "") -> int:
    if blind_kind == "SMALL":
        return 3
    if blind_kind == "BIG":
        return 4
    return int(BOSS_METADATA_BY_NAME.get(blind_name, {}).get("dollars", 5))


def _with_fresh_shop_modifiers(state: GameState) -> GameState:
    modifiers = dict(state.modifiers)
    hand_size_delta = _int_value(modifiers.pop("round_start_hand_size_delta", 0))
    if hand_size_delta and "hand_size" in modifiers:
        modifiers["hand_size"] = max(1, _int_value(modifiers.get("hand_size")) - hand_size_delta)
    elif hand_size_delta:
        modifiers["hand_size_delta"] = _int_value(modifiers.get("hand_size_delta")) - hand_size_delta
    base_cost = _base_reroll_cost(state)
    modifiers["reroll_cost"] = base_cost
    modifiers["current_reroll_cost"] = base_cost
    modifiers["round_reset_reroll_cost"] = base_cost
    modifiers["reroll_cost_increase"] = 0
    modifiers["first_shop_buffoon"] = True
    return replace(state, modifiers=modifiers)


def _base_reroll_cost(state: GameState) -> int:
    cost = 5
    if "Reroll Surplus" in state.vouchers:
        cost -= 2
    if "Reroll Glut" in state.vouchers:
        cost -= 2
    return max(1, cost)


def _boss_reroll_available(state: GameState) -> bool:
    if state.phase != GamePhase.BLIND_SELECT:
        return False
    if "Retcon" in state.vouchers or state.modifiers.get("boss_rerolls_unlimited"):
        return state.money >= _boss_reroll_cost(state)
    if "Director's Cut" not in state.vouchers:
        return False
    if state.modifiers.get("boss_rerolled"):
        return False
    return state.money >= _boss_reroll_cost(state)


def _boss_reroll_cost(state: GameState) -> int:
    return max(0, _int_value(state.modifiers.get("boss_reroll_cost"), default=10))


def _with_local_legal_actions(state: GameState) -> GameState:
    if state.run_over:
        return replace(state, legal_actions=())
    return with_derived_legal_actions(state)


def _shuffled_standard_deck(rng: Random) -> tuple[Card, ...]:
    deck = [Card(rank, suit) for suit in SUITS for rank in RANKS]
    rng.shuffle(deck)
    return tuple(deck)


def _with_shuffled_known_deck(state: GameState, rng: Random) -> GameState:
    if not state.known_deck:
        return state
    deck = list(state.known_deck)
    rng.shuffle(deck)
    return replace(state, known_deck=tuple(deck))


def _draw_from_known_deck(state: GameState, count: int) -> tuple[Card, ...]:
    if count <= 0:
        return ()
    if state.known_deck:
        return state.known_deck[: min(count, len(state.known_deck))]
    return ()


def _random_playing_card(
    rng: Random,
    *,
    rank: str | None = None,
    suit: str | None = None,
    enhancement: str | None = None,
    seal: str | None = None,
) -> Card:
    return Card(rank or rng.choice(RANKS), suit or rng.choice(SUITS), enhancement=enhancement, seal=seal)


def _random_enhancement(rng: Random) -> str:
    return rng.choice(("BONUS", "MULT", "WILD", "GLASS", "STEEL"))


def _card_with_face_down_metadata(card: Card) -> Card:
    metadata = dict(card.metadata)
    metadata["face_down"] = True
    metadata["flipped"] = True
    state = dict(metadata.get("state")) if isinstance(metadata.get("state"), Mapping) else {}
    state["facing"] = "back"
    metadata["state"] = state
    return replace(card, metadata=metadata)


def _card_with_forced_selection(card: Card) -> Card:
    metadata = dict(card.metadata)
    metadata["forced_selection"] = True
    ability = dict(metadata.get("ability")) if isinstance(metadata.get("ability"), Mapping) else {}
    ability["forced_selection"] = True
    metadata["ability"] = ability
    return replace(card, metadata=metadata)


def _card_is_forced_selection(card: Card) -> bool:
    if card.metadata.get("forced_selection"):
        return True
    ability = card.metadata.get("ability")
    if isinstance(ability, Mapping) and ability.get("forced_selection"):
        return True
    state = card.metadata.get("state")
    return isinstance(state, Mapping) and bool(state.get("forced_selection"))


def _cards_after_blind_debuffs(state: GameState, cards: tuple[Card, ...]) -> tuple[Card, ...]:
    if not _boss_effect_active(state):
        return cards
    if state.blind == "Verdant Leaf":
        return tuple(_card_with_blind_debuff(card) for card in cards)
    if state.blind == "The Plant":
        return tuple(_card_with_blind_debuff(card) if _is_face_card(card, state.jokers) else card for card in cards)
    if state.blind == "The Pillar":
        return tuple(_card_with_blind_debuff(card) if _card_was_played_this_ante(card) else card for card in cards)
    return cards


def _card_with_blind_debuff(card: Card) -> Card:
    metadata = dict(card.metadata)
    metadata["debuffed_by_blind"] = True
    state = dict(metadata.get("state")) if isinstance(metadata.get("state"), Mapping) else {}
    state["debuff"] = True
    metadata["state"] = state
    return replace(card, debuffed=True, metadata=metadata)


def _card_was_played_this_ante(card: Card) -> bool:
    if card.metadata.get("played_this_ante"):
        return True
    ability = card.metadata.get("ability")
    return isinstance(ability, Mapping) and bool(ability.get("played_this_ante"))


def _joker_with_face_down_metadata(joker: Joker) -> Joker:
    metadata = dict(joker.metadata)
    metadata["face_down"] = True
    metadata["flipped"] = True
    state = dict(metadata.get("state")) if isinstance(metadata.get("state"), Mapping) else {}
    state["facing"] = "back"
    metadata["state"] = state
    return Joker(joker.name, edition=joker.edition, sell_value=joker.sell_value, metadata=metadata)


def _joker_without_face_down_metadata(joker: Joker) -> Joker:
    if not joker.metadata.get("face_down") and not joker.metadata.get("flipped"):
        return joker
    metadata = dict(joker.metadata)
    metadata.pop("face_down", None)
    metadata.pop("flipped", None)
    state = dict(metadata.get("state")) if isinstance(metadata.get("state"), Mapping) else {}
    state.pop("facing", None)
    if state:
        metadata["state"] = state
    else:
        metadata.pop("state", None)
    return Joker(joker.name, edition=joker.edition, sell_value=joker.sell_value, metadata=metadata)


def _card_without_played_this_ante(card: Card) -> Card:
    if not _card_was_played_this_ante(card):
        return card
    metadata = dict(card.metadata)
    metadata.pop("played_this_ante", None)
    ability = dict(metadata.get("ability")) if isinstance(metadata.get("ability"), Mapping) else {}
    ability.pop("played_this_ante", None)
    if ability:
        metadata["ability"] = ability
    else:
        metadata.pop("ability", None)
    return replace(card, metadata=metadata)


def _hand_size(state: GameState) -> int:
    explicit = _modifier_int_or_none(state.modifiers, ("hand_size", "hand_size_limit", "hand_size_max"))
    if explicit is not None:
        return max(1, explicit)
    return max(1, 8 + _int_value(state.modifiers.get("hand_size_delta")))


def _blind_kind(state: GameState) -> str:
    current_blind = state.modifiers.get("current_blind")
    if isinstance(current_blind, Mapping):
        kind = str(current_blind.get("type", current_blind.get("kind", ""))).upper()
        if kind in BLIND_KINDS:
            return kind
    if state.blind == "Small Blind":
        return "SMALL"
    if state.blind == "Big Blind":
        return "BIG"
    return "BOSS"


def _boss_effect_active(state: GameState) -> bool:
    return _blind_kind(state) == "BOSS" and not _boss_disabled(state) and _active_joker_count(state, "Chicot") <= 0


def _serpent_draws_three(state: GameState) -> bool:
    return state.blind == "The Serpent" and _boss_effect_active(state)


def _discard_used_count(state: GameState) -> int:
    for key in ("discards_used", "discard_count", "discards_taken"):
        if key in state.modifiers:
            return _int_value(state.modifiers.get(key))
    base = _int_value(state.modifiers.get("base_discards"), default=4)
    return max(0, base - state.discards_remaining)


def _most_played_hand_name(state: GameState) -> str:
    explicit = state.modifiers.get("most_played_poker_hand")
    if explicit:
        return str(explicit)
    hands = state.modifiers.get("hands", {})
    if not isinstance(hands, Mapping):
        return HandType.HIGH_CARD.value
    best_name = HandType.HIGH_CARD.value
    best_played = -1
    best_order = 100
    for name, raw in hands.items():
        if not isinstance(raw, Mapping):
            continue
        played = _int_value(raw.get("played"))
        order = _int_value(raw.get("order")) or 100
        if played > best_played or (played == best_played and order < best_order):
            best_name = str(name)
            best_played = played
            best_order = order
    return best_name


def _boss_name_from_surface(state: GameState) -> str | None:
    blinds = _mapping(state.modifiers.get("blinds"))
    boss = blinds.get("boss")
    if isinstance(boss, Mapping) and boss.get("name"):
        return str(boss["name"])
    upcoming = state.modifiers.get("upcoming_boss")
    if isinstance(upcoming, Mapping) and upcoming.get("name"):
        return str(upcoming["name"])
    if _blind_kind(state) == "BOSS" and state.blind:
        return state.blind
    return None


def _mapping(raw: object) -> Mapping[str, Any]:
    return raw if isinstance(raw, Mapping) else {}


def _tag_payload(name_or_key: object, blind_kind: str | None = None) -> dict[str, Any]:
    name = _normalize_tag_name(name_or_key)
    return {
        "key": TAG_KEYS_BY_NAME.get(name, str(name_or_key)),
        "name": name,
        "label": name,
        "set": "Tag",
        "blind_type": _tag_blind_type(blind_kind),
    }


def _normalize_tag_name(value: object) -> str:
    if isinstance(value, Mapping):
        if value.get("name") or value.get("label"):
            return str(value.get("name", value.get("label")))
        value = value.get("key", "")
    text = str(value or "")
    return TAG_NAMES_BY_KEY.get(text, text)


def _tag_blind_type(blind_kind: str | None) -> str | None:
    if blind_kind is None:
        return None
    normalized = blind_kind.upper()
    if normalized == "SMALL":
        return "Small"
    if normalized == "BIG":
        return "Big"
    if normalized == "BOSS":
        return "Boss"
    return blind_kind


def _pending_tags(modifiers: Mapping[str, object]) -> tuple[str, ...]:
    raw = modifiers.get("tags", ())
    if isinstance(raw, Mapping):
        raw = raw.get("cards", ())
    if isinstance(raw, list | tuple):
        return tuple(_normalize_tag_name(item) for item in raw if _normalize_tag_name(item))
    if raw:
        return (_normalize_tag_name(raw),)
    return ()


def _with_added_tag(modifiers: dict[str, object], tag: object) -> dict[str, object]:
    tag_name = _normalize_tag_name(tag)
    if not tag_name:
        return modifiers
    pending = list(_pending_tags(modifiers))
    if tag_name != "Double Tag":
        double_count = sum(1 for existing in pending if existing == "Double Tag")
        if double_count:
            pending = [existing for existing in pending if existing != "Double Tag"]
            pending.extend(tag_name for _ in range(double_count))
    pending.append(tag_name)
    modifiers["tags"] = tuple(pending)
    return modifiers


def _current_skip_tag(state: GameState) -> str:
    current_blind = state.modifiers.get("current_blind")
    if isinstance(current_blind, Mapping):
        for key in ("skip_tag", "tag", "tag_key"):
            if current_blind.get(key):
                return _normalize_tag_name(current_blind[key])
    return "Skip Tag"


def _couponed_payload(item: Mapping[str, Any]) -> dict[str, Any]:
    payload = {**item}
    cost = dict(payload.get("cost")) if isinstance(payload.get("cost"), Mapping) else {"buy": _int_value(payload.get("cost"))}
    cost["buy"] = 0
    payload["cost"] = cost
    metadata = dict(payload.get("metadata")) if isinstance(payload.get("metadata"), Mapping) else {}
    metadata["couponed"] = True
    payload["metadata"] = metadata
    payload["couponed"] = True
    return payload


def _with_first_eligible_tag_edition(
    shop_cards: Iterable[Mapping[str, Any]],
    edition: str,
) -> tuple[bool, list[Mapping[str, Any]]]:
    updated: list[Mapping[str, Any]] = []
    applied = False
    for item in shop_cards:
        payload = dict(item)
        if not applied and _item_set(payload) == "JOKER" and not payload.get("edition"):
            base_buy_cost = _item_buy_cost(payload)
            payload["edition"] = edition
            modifier = dict(payload.get("modifier")) if isinstance(payload.get("modifier"), Mapping) else {}
            modifier["edition"] = edition
            payload["modifier"] = modifier
            payload = _couponed_payload(payload)
            cost = dict(payload.get("cost")) if isinstance(payload.get("cost"), Mapping) else {}
            sell_value = max(1, (base_buy_cost + EDITION_EXTRA_COST.get(edition, 0)) // 2)
            cost["sell"] = sell_value
            payload["cost"] = cost
            payload["sell_value"] = sell_value
            applied = True
        updated.append(payload)
    return applied, updated


def _open_free_tag_pack(state: GameState, pack: Mapping[str, Any], contents: Iterable[Mapping[str, Any]]) -> GameState:
    modifiers = dict(state.modifiers)
    pack_cards = tuple(contents)
    modifiers["pack_cards"] = pack_cards
    modifiers["pack_choices_remaining"] = _pack_choices(pack)
    modifiers["tag_pack_return_phase"] = GamePhase.BLIND_SELECT.value
    return replace(
        state,
        phase=GamePhase.BOOSTER_OPENED,
        pack=(_item_label(pack),),
        modifiers=modifiers,
        legal_actions=(),
    )


def _orbital_hand_for_tag(tag: str, modifiers: Mapping[str, object]) -> str | None:
    raw = modifiers.get("tag_orbital_hands")
    if isinstance(raw, Mapping):
        value = raw.get(tag)
        if value:
            return str(value)
    value = modifiers.get("orbital_hand")
    return str(value) if value else None


def _with_hand_level_modifier(modifiers: Mapping[str, object], hand_name: str, level: int) -> dict[str, object]:
    updated = dict(modifiers)
    raw_hands = modifiers.get("hands", {})
    hands = dict(raw_hands) if isinstance(raw_hands, Mapping) else {}
    current = dict(hands.get(hand_name)) if isinstance(hands.get(hand_name), Mapping) else {}
    current["level"] = level
    hands[hand_name] = current
    updated["hands"] = hands
    return updated


def _boss_disabled(state: GameState) -> bool:
    value = state.modifiers.get("boss_disabled")
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "none", "nil"}
    return bool(value)


def _validate_card_action(state: GameState, action: Action) -> None:
    if not action.card_indices:
        raise LocalSimError(f"{action.action_type.value} needs at least one card index.")
    invalid = [index for index in action.card_indices if index < 0 or index >= len(state.hand)]
    if invalid:
        raise LocalSimError(f"{action.action_type.value} has invalid hand indexes: {invalid}")


def _indexed_item(items: tuple[Mapping[str, Any], ...], action: Action) -> Mapping[str, Any]:
    index = _action_index(action)
    if index is None or not 0 <= index < len(items):
        raise LocalSimError(f"Action index is outside visible items: {action.amount}")
    return items[index]


def _modifier_items(modifiers: Mapping[str, object], key: str) -> tuple[Mapping[str, Any], ...]:
    raw = modifiers.get(key, ())
    if isinstance(raw, Mapping):
        raw = raw.get("cards", ())
    if isinstance(raw, list | tuple):
        return tuple(item for item in raw if isinstance(item, Mapping))
    return ()


def _with_shop_cards(state: GameState, shop_cards: Iterable[object]) -> GameState:
    cards = tuple(shop_cards)
    modifiers = {**state.modifiers, "shop_cards": cards}
    return replace(state, modifiers=modifiers, shop=tuple(_item_label(item) for item in cards), legal_actions=())


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


def _item_label(item: object) -> str:
    if isinstance(item, Mapping):
        return str(item.get("label", item.get("name", item.get("key", "unknown"))))
    return str(item)


def _item_set(item: object) -> str:
    if isinstance(item, Mapping):
        return str(item.get("set", "")).upper()
    return ""


def _item_buy_cost(item: object) -> int:
    if not isinstance(item, Mapping):
        return 0
    cost = item.get("cost", {})
    if isinstance(cost, Mapping):
        return _int_value(cost.get("buy", cost.get("cost", 0)))
    return _int_value(cost)


def _last_tarot_planet_label(state: GameState) -> str:
    raw = state.modifiers.get("last_tarot_planet_name", state.modifiers.get("last_tarot_planet", ""))
    if isinstance(raw, Mapping):
        return _item_label(raw)
    text = str(raw or "")
    key_map = {
        "c_fool": "The Fool",
        "c_magician": "The Magician",
        "c_high_priestess": "The High Priestess",
        "c_empress": "The Empress",
        "c_emperor": "The Emperor",
        "c_heirophant": "The Hierophant",
        "c_hierophant": "The Hierophant",
        "c_lovers": "The Lovers",
        "c_chariot": "The Chariot",
        "c_justice": "Justice",
        "c_hermit": "The Hermit",
        "c_wheel_of_fortune": "The Wheel of Fortune",
        "c_strength": "Strength",
        "c_hanged_man": "The Hanged Man",
        "c_death": "Death",
        "c_temperance": "Temperance",
        "c_devil": "The Devil",
        "c_tower": "The Tower",
        "c_star": "The Star",
        "c_moon": "The Moon",
        "c_sun": "The Sun",
        "c_judgement": "Judgement",
        "c_world": "The World",
    }
    return key_map.get(text, text)


def _action_index(action: Action) -> int | None:
    raw = action.metadata.get("index", action.amount)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _pack_choices(pack: Mapping[str, Any]) -> int:
    config = pack.get("config")
    if isinstance(config, Mapping) and "choose" in config:
        return max(1, _int_value(config["choose"], default=1))
    key = str(pack.get("key", ""))
    return 2 if "mega" in key else 1


def _without_pack_choice_counter(modifiers: Mapping[str, object]) -> dict[str, object]:
    updated = dict(modifiers)
    updated.pop("pack_choices_remaining", None)
    return updated


def _is_skip_action(action: Action) -> bool:
    return action.target_id == "skip" or str(action.metadata.get("kind", "")) == "skip"


def _jokers_after_skip(jokers: tuple[Joker, ...]) -> tuple[Joker, ...]:
    updated: list[Joker] = []
    for joker in jokers:
        if joker.name == "Throwback":
            current = _joker_float(joker, keys=("current_xmult", "xmult", "current_x_mult", "x_mult"), default=1.0)
            gain = _joker_float(joker, keys=("extra", "xmult_gain"), default=0.25)
            metadata = dict(joker.metadata)
            metadata["current_xmult"] = current + gain
            updated.append(replace(joker, metadata=metadata))
        else:
            updated.append(joker)
    return tuple(updated)


def _active_joker_count(state: GameState, name: str) -> int:
    return _active_joker_name_count(state.jokers, name)


def _joker_disabled(joker: Joker) -> bool:
    state = joker.metadata.get("state")
    if isinstance(state, Mapping) and state.get("debuff"):
        return True
    return bool(joker.metadata.get("debuffed") or joker.metadata.get("disabled")) or (
        "all abilities are disabled" in _joker_effect_text(joker).lower()
    )


def _jokers_with_only_disabled_index(jokers: tuple[Joker, ...], disabled_index: int) -> tuple[Joker, ...]:
    if disabled_index < 0 or disabled_index >= len(jokers):
        raise LocalSimError(f"Crimson Heart disabled joker index is outside jokers: {disabled_index}")
    return tuple(_joker_with_disabled_state(joker, index == disabled_index) for index, joker in enumerate(jokers))


def _joker_with_disabled_state(joker: Joker, disabled: bool) -> Joker:
    metadata = _copy_metadata(joker.metadata)
    state_value = metadata.get("state")
    state_data = dict(state_value) if isinstance(state_value, Mapping) else {}
    if disabled:
        state_data["debuff"] = True
        _set_disabled_effect_text(metadata)
    else:
        state_data.pop("debuff", None)
        _restore_disabled_effect_text(metadata)
    if state_data:
        metadata["state"] = state_data
    elif "state" in metadata:
        metadata["state"] = []
    return replace(joker, metadata=metadata)


def _set_disabled_effect_text(metadata: dict[str, Any]) -> None:
    value = metadata.get("value")
    if not isinstance(value, dict):
        return
    effect = value.get("effect")
    if isinstance(effect, str) and effect and "all abilities are disabled" not in effect.lower():
        metadata.setdefault("_crimson_heart_original_effect", effect)
    value["effect"] = "All abilities are disabled"


def _restore_disabled_effect_text(metadata: dict[str, Any]) -> None:
    value = metadata.get("value")
    if not isinstance(value, dict):
        metadata.pop("_crimson_heart_original_effect", None)
        return
    effect = str(value.get("effect", ""))
    if "all abilities are disabled" not in effect.lower():
        metadata.pop("_crimson_heart_original_effect", None)
        return
    original = metadata.pop("_crimson_heart_original_effect", None)
    if isinstance(original, str) and original:
        value["effect"] = original
    else:
        value.pop("effect", None)


def _joker_effect_text(joker: Joker) -> str:
    value = joker.metadata.get("value")
    if isinstance(value, Mapping):
        return str(value.get("effect", ""))
    return str(joker.metadata.get("effect", ""))


def _copy_metadata(value):
    if isinstance(value, Mapping):
        return {key: _copy_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_metadata(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_metadata(item) for item in value)
    return value


def _normal_joker_open_slots(state: GameState) -> int:
    limit = _modifier_int_or_none(state.modifiers, ("joker_slot_limit", "joker_slots"))
    slot_limit = 5 if limit is None else max(0, limit)
    used = sum(1 for joker in state.jokers if "negative" not in (joker.edition or "").lower())
    return max(0, slot_limit - used)


def _consumable_open_slots(state: GameState) -> int:
    limit = _modifier_int_or_none(
        state.modifiers,
        ("consumable_slot_limit", "consumeable_slot_limit", "consumable_slots", "consumeable_slots"),
    )
    slot_limit = 2 if limit is None else max(0, limit)
    return max(0, slot_limit - len(state.consumables))


def _optional_tuple(item: Mapping[str, Any] | None) -> tuple[Mapping[str, Any], ...]:
    return () if item is None else (item,)


def _modifier_int_or_none(modifiers: Mapping[str, object], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        if key not in modifiers:
            continue
        try:
            return int(modifiers[key])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return None


def _int_value(raw: object, *, default: int = 0) -> int:
    try:
        return int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _joker_float(joker: Joker, *, keys: tuple[str, ...], default: float) -> float:
    for key in keys:
        raw = joker.metadata.get(key)
        try:
            if raw is not None:
                return float(raw)
        except (TypeError, ValueError):
            continue
    return default


def _split_action_cards(state: GameState, action: Action) -> tuple[tuple[Card, ...], tuple[Card, ...]]:
    selected = set(action.card_indices)
    selected_cards = tuple(card for index, card in enumerate(state.hand) if index in selected)
    held_cards = tuple(card for index, card in enumerate(state.hand) if index not in selected)
    return selected_cards, held_cards


def _without_card_instances(cards: tuple[Card, ...], removed_cards: tuple[Card, ...]) -> tuple[Card, ...]:
    remaining = list(cards)
    for removed in removed_cards:
        target = _card_identity_key(removed)
        for index, card in enumerate(remaining):
            if _card_identity_key(card) == target:
                del remaining[index]
                break
    return tuple(remaining)


def _card_identity_key(card: Card) -> tuple[str, str, str | None, str | None, str | None]:
    return (_normalize_rank(card.rank), _normalize_suit(card.suit), card.enhancement, card.seal, card.edition)


def _played_hand_types_this_round(state: GameState) -> tuple[str, ...]:
    hands = state.modifiers.get("hands", {})
    if not isinstance(hands, Mapping):
        return ()
    played: list[tuple[int, str]] = []
    for name, value in hands.items():
        if not isinstance(value, Mapping):
            continue
        played_count = _int_value(value.get("played_this_round"))
        order = _int_value(value.get("order"))
        played.extend((order, str(name)) for _ in range(max(0, played_count)))
    return tuple(name for _, name in sorted(played, key=lambda item: item[0]))


def _played_hand_counts(state: GameState) -> dict[str, int]:
    hands = state.modifiers.get("hands", {})
    if not isinstance(hands, Mapping):
        return {}
    counts: dict[str, int] = {}
    for name, value in hands.items():
        if isinstance(value, Mapping):
            counts[str(name)] = _int_value(value.get("played"))
        else:
            counts[str(name)] = _int_value(value)
    return counts


def _scored_trigger_cards(
    cards: tuple[Card, ...],
    evaluation,
    jokers: tuple[Joker, ...],
    hands_remaining: int,
) -> tuple[Card, ...]:
    if not evaluation.scoring_indices:
        return ()
    first_scored = evaluation.scoring_indices[0]
    triggered: list[Card] = []
    for index in evaluation.scoring_indices:
        card = cards[index]
        triggers = 1
        if _normalized(card.seal) == "red":
            triggers += 1
        if _normalize_rank(card.rank) in {"2", "3", "4", "5"}:
            triggers += _active_joker_name_count(jokers, "Hack")
        if _is_face_card(card, jokers):
            triggers += _active_joker_name_count(jokers, "Sock and Buskin")
        if hands_remaining == 1:
            triggers += _active_joker_name_count(jokers, "Dusk")
        triggers += _active_joker_name_count(jokers, "Seltzer")
        if index == first_scored:
            triggers += 2 * _active_joker_name_count(jokers, "Hanging Chad")
        triggered.extend(card for _ in range(triggers))
    return tuple(triggered)


def _sixth_sense_triggers(state: GameState, played_cards: tuple[Card, ...]) -> bool:
    if len(played_cards) != 1 or _played_hand_types_this_round(state) or _consumable_open_slots(state) <= 0:
        return False
    return _normalize_rank(played_cards[0].rank) == "6" and _active_joker_count(state, "Sixth Sense") > 0


def _probability_multiplier(state: GameState) -> float:
    return _float_value(state.modifiers.get("probability_multiplier"), default=1.0)


def _float_value(raw: object, *, default: float = 0.0) -> float:
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _is_lucky_card(card: Card) -> bool:
    return _normalized(card.enhancement) in {"lucky", "lucky card"}


def _is_glass_card(card: Card) -> bool:
    return _normalized(card.enhancement) in {"glass", "glass card"}


def _glass_shatter_odds(card: Card) -> int:
    for source in _card_metadata_sources(card):
        if "extra" in source:
            value = _int_value(source["extra"])
            if value > 0:
                return value
    return 4


def _card_metadata_sources(card: Card) -> tuple[Mapping[str, Any], ...]:
    sources: list[Mapping[str, Any]] = [card.metadata]
    for key in ("modifier", "value", "ability", "config"):
        value = card.metadata.get(key)
        if isinstance(value, Mapping):
            sources.append(value)
    return tuple(sources)


def _card_has_suit(card: Card, target_suit: str, jokers: tuple[Joker, ...] = ()) -> bool:
    if card.debuffed:
        return False
    if _normalized(card.enhancement) in {"wild", "wild card"}:
        return True
    suit = _normalize_suit(card.suit)
    target = _normalize_suit(target_suit)
    if suit == target:
        return True
    if _active_joker_name_count(jokers, "Smeared Joker"):
        if target in {"H", "D"} and suit in {"H", "D"}:
            return True
        if target in {"S", "C"} and suit in {"S", "C"}:
            return True
    return False


def _is_face_card(card: Card, jokers: tuple[Joker, ...]) -> bool:
    if card.debuffed or _normalized(card.enhancement) in {"stone", "stone card"}:
        return False
    return _normalize_rank(card.rank) in {"J", "Q", "K"} or _active_joker_name_count(jokers, "Pareidolia") > 0


def _active_joker_name_count(jokers: tuple[Joker, ...], name: str) -> int:
    return sum(1 for joker in _active_effective_jokers(jokers) if joker.name == name)


def _active_effective_jokers(jokers: tuple[Joker, ...]) -> tuple[Joker, ...]:
    sources = _effective_ability_joker_indices(jokers)
    effective = tuple(jokers[index] for index in sources)
    return tuple(
        ability
        for physical, ability in zip(jokers, effective, strict=False)
        if not _joker_disabled(physical) and not _joker_disabled(ability)
    )


def _effective_ability_joker_indices(jokers: tuple[Joker, ...]) -> tuple[int, ...]:
    def resolve(index: int, seen: frozenset[int] = frozenset()) -> int:
        joker = jokers[index]
        if index in seen:
            return index
        if joker.name == "Blueprint" and index + 1 < len(jokers) and is_blueprint_compatible(jokers[index + 1]):
            return resolve(index + 1, seen | {index})
        if joker.name == "Brainstorm" and index != 0 and is_blueprint_compatible(jokers[0]):
            return resolve(0, seen | {index})
        return index

    return tuple(resolve(index) for index in range(len(jokers)))


def _non_stone_full_deck_cards(state: GameState) -> tuple[Card, ...]:
    return tuple(card for card in _known_full_deck_cards(state) if _normalized(card.enhancement) not in {"stone", "stone card"})


def _known_full_deck_cards(state: GameState) -> tuple[Card, ...]:
    return state.hand + state.known_deck + _zone_cards(state.modifiers, "played_pile") + _zone_cards(state.modifiers, "discard_pile")


def _zone_cards(modifiers: Mapping[str, object], key: str) -> tuple[Card, ...]:
    raw = modifiers.get(key, ())
    if not isinstance(raw, list | tuple):
        return ()
    return tuple(item if isinstance(item, Card) else Card.from_mapping(item) for item in raw)


def _current_target_suit(jokers: tuple[Joker, ...], name: str) -> str | None:
    for joker in jokers:
        if joker.name != name:
            continue
        for source in _joker_metadata_sources(joker):
            for key in ("current_suit", "target_suit", "suit"):
                value = source.get(key)
                if isinstance(value, str) and value:
                    return _normalize_suit(value)
    return None


def _joker_metadata_sources(joker: Joker) -> tuple[Mapping[str, Any], ...]:
    sources: list[Mapping[str, Any]] = [joker.metadata]
    for key in ("ability", "config", "extra"):
        value = joker.metadata.get(key)
        if isinstance(value, Mapping):
            sources.append(value)
            nested = value.get("extra")
            if isinstance(nested, Mapping):
                sources.append(nested)
    return tuple(sources)


def _invisible_joker_ready(joker: Joker) -> bool:
    rounds = _joker_int(joker, keys=("current_rounds", "invis_rounds", "rounds"), default=0)
    needed = _joker_int(joker, keys=("extra", "rounds_needed"), default=2)
    return rounds >= needed


def _joker_int(joker: Joker, *, keys: tuple[str, ...], default: int) -> int:
    for source in _joker_metadata_sources(joker):
        for key in keys:
            if key in source:
                value = _int_value(source[key], default=default)
                return value
    return default


def _joker_payload_from_joker(joker: Joker) -> dict[str, Any]:
    return {
        "name": joker.name,
        "label": joker.name,
        "edition": joker.edition,
        "sell_value": joker.sell_value,
        "metadata": _copy_metadata(joker.metadata),
    }


def _joker_eternal(joker: Joker) -> bool:
    for source in _joker_metadata_sources(joker):
        if bool(source.get("eternal")):
            return True
    return False


def _copy_metadata(value):
    if isinstance(value, Mapping):
        return {key: _copy_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_metadata(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_metadata(item) for item in value)
    return value


def _normalized(value: object) -> str:
    return str(value or "").removeprefix("m_").replace("_", " ").lower()


def _normalize_suit(suit: str) -> str:
    return {
        "Spades": "S",
        "Spade": "S",
        "S": "S",
        "Hearts": "H",
        "Heart": "H",
        "H": "H",
        "Clubs": "C",
        "Club": "C",
        "C": "C",
        "Diamonds": "D",
        "Diamond": "D",
        "D": "D",
    }.get(str(suit), str(suit))


def _normalize_rank(rank: str) -> str:
    value = str(rank).strip().lower()
    return {
        "ace": "A",
        "a": "A",
        "king": "K",
        "k": "K",
        "queen": "Q",
        "q": "Q",
        "jack": "J",
        "j": "J",
        "ten": "10",
        "t": "10",
        "10": "10",
        "nine": "9",
        "eight": "8",
        "seven": "7",
        "six": "6",
        "five": "5",
        "four": "4",
        "three": "3",
        "two": "2",
    }.get(value, value.upper())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
