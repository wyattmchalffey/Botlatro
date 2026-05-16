"""Consumable-use search for Phase 7 bots."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from random import Random
from typing import Any

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import Card, GamePhase, GameState
from balatro_ai.rules.hand_evaluator import HandType
from balatro_ai.search.forward_sim import (
    PLANET_TO_HAND,
    SPECTRAL_SEALS,
    TAROT_ENHANCEMENTS,
    TAROT_SUIT_CONVERSIONS,
    TAROT_NAMES,
    simulate_use_consumable,
)
from balatro_ai.search.hand_viability import ADVANCED_HAND_TYPES, hand_draw_odds
from balatro_ai.search.shop_sampler import ShopSampler
from balatro_ai.search.state_value import _best_immediate_score, clear_probability, state_value

ValueFn = Callable[[GameState], float]


DETERMINISTIC_NON_TARGETED = frozenset({"The Hermit", "Temperance", "Black Hole"})
DETERMINISTIC_TARGETED = frozenset(
    {
        *TAROT_ENHANCEMENTS,
        *TAROT_SUIT_CONVERSIONS,
        "Strength",
        "The Hanged Man",
        "Death",
        *SPECTRAL_SEALS,
        "Cryptid",
    }
)
STOCHASTIC_NON_TARGETED = frozenset(
    {
        "The Fool",
        "The High Priestess",
        "The Emperor",
        "The Wheel of Fortune",
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
    }
)
STOCHASTIC_TARGETED = frozenset({"Aura"})
SHOP_HAND_DEPENDENT_CONSUMABLES = frozenset(
    {"Familiar", "Grim", "Incantation", "Immolate", "Sigil", "Ouija", "Aura"}
)
RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
SUITS = ("S", "H", "C", "D")
SPECTRAL_RANDOM_ENHANCEMENTS = ("BONUS", "MULT", "WILD", "GLASS", "STEEL")


@dataclass(frozen=True, slots=True)
class ConsumableSearchConfig:
    leaf_samples: int = 16
    seed: int = 0
    stochastic_samples: int = 8
    max_actions: int = 96
    min_shop_delta: float = 2.0
    min_pack_delta: float = 2.0
    min_blind_delta: float = 6.0


def best_consumable_action(
    state: GameState,
    *,
    config: ConsumableSearchConfig | None = None,
    value_fn: ValueFn | None = None,
    sampler: ShopSampler | None = None,
) -> Action | None:
    """Return the best consumable use when modeled outcomes improve the state."""

    use_actions = tuple(action for action in state.legal_actions if action.action_type == ActionType.USE_CONSUMABLE)
    if not use_actions:
        return None

    search_config = config or ConsumableSearchConfig()
    candidates = tuple(action for action in use_actions if consumable_action_is_supported(state, action))
    if not candidates:
        return None
    candidates = _prefer_max_targeted_tarot_actions(state, candidates)
    candidates = _candidate_consumable_actions(state, candidates, limit=search_config.max_actions)

    evaluator = value_fn or _default_value_fn(search_config, root_state=state)
    baseline = evaluator(state)
    best_action: Action | None = None
    best_value = float("-inf")
    best_target_count = -1
    for action_index, action in enumerate(candidates):
        try:
            value = consumable_action_value(
                state,
                action,
                config=search_config,
                value_fn=evaluator,
                baseline_value=baseline,
                action_index=action_index,
                sampler=sampler,
            )
        except (ValueError, IndexError, TypeError, AttributeError):
            continue
        target_count = len(action.card_indices)
        if value > best_value or (value == best_value and target_count > best_target_count):
            best_action = action
            best_value = value
            best_target_count = target_count

    if best_action is None or best_value < _minimum_delta(state, search_config):
        return None
    return _annotated_action(best_action, search_value=best_value)


def _prefer_max_targeted_tarot_actions(state: GameState, actions: tuple[Action, ...]) -> tuple[Action, ...]:
    max_target_counts: dict[int, int] = {}
    for action in actions:
        index = _action_index(action)
        if index is None:
            continue
        name = _action_consumable_name(state, action)
        if name not in TAROT_NAMES:
            continue
        max_target_counts[index] = max(max_target_counts.get(index, -1), len(action.card_indices))

    if not max_target_counts:
        return actions

    kept: list[Action] = []
    for action in actions:
        index = _action_index(action)
        if index is not None and len(action.card_indices) < max_target_counts.get(index, -1):
            continue
        kept.append(action)
    return tuple(kept)


def consumable_action_value(
    state: GameState,
    action: Action,
    *,
    config: ConsumableSearchConfig | None = None,
    value_fn: ValueFn | None = None,
    baseline_value: float | None = None,
    action_index: int = 0,
    sampler: ShopSampler | None = None,
) -> float:
    """Score one consumable use as an improvement over holding it."""

    if action.action_type != ActionType.USE_CONSUMABLE:
        raise ValueError(f"consumable_action_value requires use_consumable, got {action.action_type.value}")
    search_config = config or ConsumableSearchConfig()
    evaluator = value_fn or _default_value_fn(search_config, root_state=state)
    baseline = evaluator(state) if baseline_value is None else baseline_value
    simulated_states = _simulate_consumable_outcomes(
        state,
        action,
        config=search_config,
        action_index=action_index,
        sampler=sampler,
    )
    if not simulated_states:
        raise ValueError("No modeled outcomes for consumable action")
    total = 0.0
    for simulated in simulated_states:
        value_delta = evaluator(simulated) - baseline
        total += value_delta + _consumable_use_bonus(
            state,
            action,
            simulated,
            config=search_config,
            action_index=action_index,
        )
    return total / len(simulated_states)


def simulate_consumable_action(
    state: GameState,
    action: Action,
    *,
    sampler: ShopSampler | None = None,
    rng: Random | None = None,
) -> GameState:
    """Use a consumable, sampling any random outcomes the forward sim needs."""

    if action.action_type != ActionType.USE_CONSUMABLE:
        raise ValueError(f"simulate_consumable_action requires use_consumable, got {action.action_type.value}")
    name = _action_consumable_name(state, action)
    if name is None:
        raise ValueError("Consumable action does not reference a stored consumable")
    if _consumable_action_is_deterministic(state, action):
        return simulate_use_consumable(state, action)
    shop_sampler = sampler or ShopSampler.from_default_data()
    random = rng or Random()
    injections = sample_consumable_injections(state, name, sampler=shop_sampler, rng=random, storage_use=True)
    return simulate_use_consumable(state, action, **injections)


def shop_consumable_actions(state: GameState) -> tuple[Action, ...]:
    """Generate shop-phase consumable actions that do not require a visible hand."""

    actions: list[Action] = []
    for index, _name in enumerate(state.consumables):
        action = Action(
            ActionType.USE_CONSUMABLE,
            target_id="consumable",
            amount=index,
            metadata={"kind": "consumable", "index": index},
        )
        if shop_consumable_action_is_candidate(state, action):
            actions.append(action)
    return tuple(actions)


def shop_consumable_action_is_candidate(state: GameState, action: Action) -> bool:
    """Return whether a stored consumable is useful to model from the shop beam."""

    if action.action_type != ActionType.USE_CONSUMABLE:
        return False
    name = _action_consumable_name(state, action)
    if name is None or not consumable_action_is_supported(state, action):
        return False
    if name in SHOP_HAND_DEPENDENT_CONSUMABLES and not state.hand:
        return False
    if name in DETERMINISTIC_TARGETED or name in STOCHASTIC_TARGETED:
        return bool(action.card_indices)
    if name in {"The Wheel of Fortune", "Ankh"}:
        return bool(state.jokers)
    if name in {"Hex", "Ectoplasm"}:
        return any(not joker.edition for joker in state.jokers)
    if name in {"Judgement", "Wraith", "The Soul"}:
        return _normal_joker_open_slots(state) > 0
    if name in {"The Fool", "The Emperor", "The High Priestess"}:
        return _consumable_open_slots_after_use(state, storage_use=True) > 0
    return name in PLANET_TO_HAND or name in DETERMINISTIC_NON_TARGETED or name in STOCHASTIC_NON_TARGETED


def _default_value_fn(config: ConsumableSearchConfig, *, root_state: GameState) -> ValueFn:
    if root_state.phase == GamePhase.SHOP:
        return _shop_value_fn
    if root_state.phase == GamePhase.BOOSTER_OPENED:
        return lambda state: (_shop_value_fn(state) * 0.25) + (_blind_value_fn(state, config=config) * 0.75)
    return lambda state: _blind_value_fn(state, config=config)


def _shop_value_fn(state: GameState) -> float:
    try:
        from balatro_ai.search.shop_search import shop_leaf_value

        return shop_leaf_value(state)
    except (ImportError, TypeError, ValueError, AttributeError):
        money_value = min(75.0, max(0.0, float(state.money)) * 0.6)
        joker_value = min(40.0, len(state.jokers) * 8.0)
        return money_value + joker_value


def _blind_value_fn(state: GameState, *, config: ConsumableSearchConfig) -> float:
    return state_value(state, samples=config.leaf_samples, seed=config.seed) * 100.0


def _consumable_action_is_deterministic(state: GameState, action: Action) -> bool:
    name = _action_consumable_name(state, action)
    if name is None:
        return False
    if name in PLANET_TO_HAND or name in DETERMINISTIC_NON_TARGETED:
        return True
    return name in DETERMINISTIC_TARGETED


def consumable_action_is_supported(state: GameState, action: Action) -> bool:
    name = _action_consumable_name(state, action)
    if name is None:
        return False
    if _consumable_action_is_deterministic(state, action):
        return True
    if name in STOCHASTIC_NON_TARGETED:
        return True
    return name in STOCHASTIC_TARGETED


def _simulate_consumable_outcomes(
    state: GameState,
    action: Action,
    *,
    config: ConsumableSearchConfig,
    action_index: int,
    sampler: ShopSampler | None,
) -> tuple[GameState, ...]:
    if _consumable_action_is_deterministic(state, action):
        return (simulate_use_consumable(state, action),)

    sample_count = max(1, config.stochastic_samples)
    shop_sampler = sampler or ShopSampler.from_default_data()
    outcomes: list[GameState] = []
    for sample_index in range(sample_count):
        rng = Random(_sample_seed(config.seed, action, action_index=action_index, sample_index=sample_index))
        try:
            outcomes.append(simulate_consumable_action(state, action, sampler=shop_sampler, rng=rng))
        except (ValueError, IndexError, TypeError, AttributeError):
            continue
    return tuple(outcomes)


def sample_consumable_injections(
    state: GameState,
    name: str,
    *,
    sampler: ShopSampler,
    rng: Random,
    storage_use: bool,
) -> dict[str, object]:
    injections: dict[str, object] = {}
    if name == "The Wheel of Fortune":
        joker_index, edition = _wheel_of_fortune_outcome(state, rng)
        injections["wheel_of_fortune_joker_index"] = joker_index
        injections["wheel_of_fortune_edition"] = edition
    elif name == "The Fool":
        injections["created_consumables"] = _fool_consumables(state, storage_use=storage_use)
    elif name == "The Emperor":
        injections["created_consumables"] = _created_consumables_of_type(
            state,
            "Tarot",
            sampler=sampler,
            rng=rng,
            max_count=2,
            storage_use=storage_use,
        )
    elif name == "The High Priestess":
        injections["created_consumables"] = _created_consumables_of_type(
            state,
            "Planet",
            sampler=sampler,
            rng=rng,
            max_count=2,
            storage_use=storage_use,
        )
    elif name in {"Judgement", "The Soul", "Wraith"}:
        if _normal_joker_open_slots(state) > 0:
            injections["created_jokers"] = (sampler.sample_card_of_type(state, "Joker", rng),)
    elif name == "Aura":
        injections["aura_edition"] = _wheel_of_fortune_edition(rng)
    elif name in {"Familiar", "Grim", "Incantation"}:
        if state.hand:
            injections["destroyed_hand_card_indices"] = (rng.randrange(len(state.hand)),)
        injections["created_hand_cards"] = _spectral_created_hand_cards(name, rng)
    elif name == "Immolate":
        count = min(5, len(state.hand))
        injections["destroyed_hand_card_indices"] = tuple(rng.sample(range(len(state.hand)), count)) if count else ()
    elif name == "Sigil":
        injections["sigil_suit"] = rng.choice(SUITS)
    elif name == "Ouija":
        injections["ouija_rank"] = rng.choice(RANKS)
    elif name in {"Ankh", "Hex", "Ectoplasm"}:
        eligible = tuple(index for index, joker in enumerate(state.jokers) if name == "Ankh" or not joker.edition)
        if eligible:
            injections["spectral_joker_index"] = rng.choice(eligible)
    return injections


def _wheel_of_fortune_outcome(state: GameState, rng: Random) -> tuple[int | None, str | None]:
    eligible = tuple(index for index, joker in enumerate(state.jokers) if not joker.edition)
    if not eligible or not _roll_odds(rng, 4, probability_multiplier=_probability_multiplier(state)):
        return None, None
    return rng.choice(eligible), _wheel_of_fortune_edition(rng)


def _wheel_of_fortune_edition(rng: Random) -> str:
    poll = rng.random()
    if poll > 1 - (0.006 * 25):
        return "POLYCHROME"
    if poll > 1 - (0.02 * 25):
        return "HOLOGRAPHIC"
    return "FOIL"


def _roll_odds(rng: Random, odds: int | float, *, probability_multiplier: float) -> bool:
    if odds <= 0:
        return True
    return rng.random() < min(1.0, probability_multiplier / float(odds))


def _probability_multiplier(state: GameState) -> float:
    try:
        return float(state.modifiers.get("probability_multiplier", 1.0))
    except (TypeError, ValueError):
        return 1.0


def _fool_consumables(state: GameState, *, storage_use: bool) -> tuple[str, ...]:
    if _consumable_open_slots_after_use(state, storage_use=storage_use) <= 0:
        return ()
    last = _last_tarot_planet_label(state)
    if not last or last == "The Fool":
        return ()
    return (last,)


def _created_consumables_of_type(
    state: GameState,
    card_type: str,
    *,
    sampler: ShopSampler,
    rng: Random,
    max_count: int,
    storage_use: bool,
) -> tuple[Mapping[str, Any], ...]:
    count = min(max_count, _consumable_open_slots_after_use(state, storage_use=storage_use))
    return tuple(sampler.sample_card_of_type(state, card_type, rng) for _ in range(count))


def _consumable_open_slots_after_use(state: GameState, *, storage_use: bool) -> int:
    used_storage_count = max(0, len(state.consumables) - (1 if storage_use else 0))
    return max(0, _consumable_slot_limit(state) - used_storage_count)


def _spectral_created_hand_cards(name: str, rng: Random) -> tuple[Card, ...]:
    if name == "Familiar":
        return tuple(_random_playing_card(rng, rank=rng.choice(("J", "Q", "K")), enhancement=_random_enhancement(rng)) for _ in range(3))
    if name == "Grim":
        return tuple(_random_playing_card(rng, rank="A", enhancement=_random_enhancement(rng)) for _ in range(2))
    if name == "Incantation":
        return tuple(_random_playing_card(rng, rank=rng.choice(("2", "3", "4", "5", "6", "7", "8", "9", "10")), enhancement=_random_enhancement(rng)) for _ in range(4))
    return ()


def _random_playing_card(rng: Random, *, rank: str | None = None, enhancement: str | None = None) -> Card:
    return Card(rank=rank or rng.choice(RANKS), suit=rng.choice(SUITS), enhancement=enhancement)


def _random_enhancement(rng: Random) -> str:
    return rng.choice(SPECTRAL_RANDOM_ENHANCEMENTS)


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


def _sample_seed(seed: int, action: Action, *, action_index: int, sample_index: int) -> int:
    value = seed ^ (action_index * 1_000_003) ^ (sample_index * 97_409)
    for char in action.stable_key:
        value = ((value * 16_777_619) ^ ord(char)) & 0xFFFFFFFF
    return value


def _candidate_consumable_actions(
    state: GameState,
    actions: tuple[Action, ...],
    *,
    limit: int,
) -> tuple[Action, ...]:
    if limit <= 0 or len(actions) <= limit:
        return actions
    indexed = tuple(enumerate(actions))
    ranked = sorted(indexed, key=lambda item: (_candidate_action_score(state, item[1]), -item[0]), reverse=True)
    return tuple(action for _, action in ranked[:limit])


def _candidate_action_score(state: GameState, action: Action) -> float:
    name = _action_consumable_name(state, action)
    if name is None:
        return float("-inf")
    score = 0.0
    if name in PLANET_TO_HAND:
        score += 18.0 + _planet_alignment_score(state, PLANET_TO_HAND[name])
    elif name == "Black Hole":
        score += 45.0
    elif name in {"The Hermit", "Temperance"}:
        score += 24.0 + max(0, 20 - state.money) * 0.4
    elif name == "Death":
        score += _death_candidate_score(state, action)
    elif name == "The Hanged Man":
        score += _hanged_man_candidate_score(state, action)
    elif name == "Strength":
        score += _strength_candidate_score(state, action)
    elif name in TAROT_SUIT_CONVERSIONS:
        score += _suit_conversion_candidate_score(state, action, TAROT_SUIT_CONVERSIONS[name])
    elif name in TAROT_ENHANCEMENTS:
        score += _enhancement_candidate_score(state, action, TAROT_ENHANCEMENTS[name])
    else:
        score += _selected_card_quality(state, action) + 5.0
    return score + _current_hand_score_delta_bonus(state, action) * 0.35


def _consumable_use_bonus(
    state: GameState,
    action: Action,
    simulated: GameState,
    *,
    config: ConsumableSearchConfig,
    action_index: int,
) -> float:
    name = _action_consumable_name(state, action)
    if name is None:
        return 0.0

    bonus = 0.0
    if _consumable_slots_are_full(state):
        bonus += 2.5
    elif state.phase in {GamePhase.SHOP, GamePhase.BOOSTER_OPENED}:
        bonus += 1.0

    if name in PLANET_TO_HAND:
        bonus += _planet_use_bonus(state, PLANET_TO_HAND[name])
    elif name == "Black Hole":
        bonus += 14.0
    elif name in {"The Hermit", "Temperance"}:
        bonus += _money_consumable_timing_bonus(state, simulated, config=config, action_index=action_index, name=name)
    elif name == "The Emperor":
        created = _created_consumable_count(state, simulated)
        bonus += created * 2.0
        bonus -= max(0, 2 - created) * 5.0
    elif name == "The High Priestess":
        created = _created_consumable_count(state, simulated)
        bonus += created * 4.0
        bonus -= max(0, 2 - created) * 4.0
    elif name == "The Fool":
        bonus += 5.0 if _created_consumable_count(state, simulated) > 0 else -3.0
    elif name in {"Judgement", "The Soul", "Wraith"}:
        bonus += max(0, len(simulated.jokers) - len(state.jokers)) * 8.0
    elif name == "The Wheel of Fortune":
        bonus += _new_edition_count(state, simulated) * 10.0
    elif action.card_indices:
        bonus += _current_hand_score_delta_bonus_from_state(state, simulated)
        bonus += _deck_fixing_shape_bonus(state, action)
        bonus += _hand_viability_delta_bonus(state, simulated)
    elif name == "Immolate":
        bonus += _money_consumable_timing_bonus(state, simulated, config=config, action_index=action_index, name=name)
        bonus -= min(8.0, max(0, len(state.hand) - len(simulated.hand)) * 0.8)
        bonus += _hand_viability_delta_bonus(state, simulated)
    elif name in {"Ankh", "Hex", "Ectoplasm"}:
        bonus += max(0.0, _shop_value_fn(simulated) - _shop_value_fn(state)) * 0.08
    elif name in {"Familiar", "Grim", "Incantation", "Sigil", "Ouija"}:
        bonus += max(0, simulated.deck_size - state.deck_size) * 1.5

    if name in PLANET_TO_HAND and _observatory_should_hold_planet(state, PLANET_TO_HAND[name]):
        bonus -= 12.0
    return bonus


def _created_consumable_count(state: GameState, simulated: GameState) -> int:
    count_after_removing_used = max(0, len(state.consumables) - 1)
    return max(0, len(simulated.consumables) - count_after_removing_used)


def _new_edition_count(state: GameState, simulated: GameState) -> int:
    count = 0
    for before, after in zip(state.jokers, simulated.jokers, strict=False):
        if not before.edition and after.edition:
            count += 1
    return count


def _planet_use_bonus(state: GameState, hand_type: HandType) -> float:
    bonus = 4.0
    preferred = _preferred_hand_type(state)
    if hand_type == preferred:
        bonus += 5.0
    if hand_type == _best_current_hand_type(state):
        bonus += 3.0
    if any(joker.name == "Constellation" for joker in state.jokers):
        bonus += 8.0
    try:
        from balatro_ai.bots.basic_strategy.shop_cards import _planet_capacity_gain

        bonus += min(14.0, _planet_capacity_gain(state, hand_type) * 0.02)
    except (ImportError, TypeError, ValueError, AttributeError):
        pass
    return bonus


def _money_consumable_timing_bonus(
    state: GameState,
    simulated: GameState,
    *,
    config: ConsumableSearchConfig,
    action_index: int,
    name: str,
) -> float:
    money_gain = max(0, simulated.money - state.money)
    if money_gain <= 0:
        return 0.0
    money_premium = 0.0
    if name == "The Hermit":
        money_premium = min(7.0, 2.0 + money_gain * 0.18)
        if state.money >= 10:
            money_premium += 2.0
    elif name == "Temperance" and state.ante <= 2:
        money_premium = -1.5
    if state.phase == GamePhase.SHOP:
        return min(22.0, 4.0 + money_gain * 0.35 + money_premium)
    clear = clear_probability(state, samples=config.leaf_samples, seed=config.seed + (action_index * 10_003))
    if clear >= 0.8 or state.current_score >= state.required_score:
        return min(12.0, money_gain * 0.25 + (money_premium * 0.5))
    return 0.0


def _current_hand_score_delta_bonus(state: GameState, action: Action) -> float:
    if state.phase not in {GamePhase.SELECTING_HAND, GamePhase.PLAYING_BLIND, GamePhase.BOOSTER_OPENED} or not state.hand:
        return 0.0
    try:
        before = _best_immediate_score(state)
        after = _best_immediate_score(simulate_use_consumable(state, action))
    except (ValueError, IndexError, TypeError, AttributeError):
        return 0.0
    score_delta = max(0, after - before)
    if score_delta <= 0:
        return 0.0
    remaining_score = max(1, state.required_score - state.current_score)
    return min(35.0, (score_delta / remaining_score) * 85.0)


def _current_hand_score_delta_bonus_from_state(state: GameState, simulated: GameState) -> float:
    if state.phase not in {GamePhase.SELECTING_HAND, GamePhase.PLAYING_BLIND, GamePhase.BOOSTER_OPENED} or not state.hand:
        return 0.0
    try:
        before = _best_immediate_score(state)
        after = _best_immediate_score(simulated)
    except (ValueError, IndexError, TypeError, AttributeError):
        return 0.0
    score_delta = max(0, after - before)
    if score_delta <= 0:
        return 0.0
    remaining_score = max(1, state.required_score - state.current_score)
    return min(35.0, (score_delta / remaining_score) * 85.0)


def _deck_fixing_shape_bonus(state: GameState, action: Action) -> float:
    name = _action_consumable_name(state, action)
    if name is None:
        return 0.0
    if name == "The Hanged Man":
        return min(8.0, sum(15 - _rank_value(state.hand[index]) for index in action.card_indices if 0 <= index < len(state.hand)) * 0.35)
    if name == "Death":
        return _death_candidate_score(state, action) * 0.12
    if name in TAROT_SUIT_CONVERSIONS:
        return _suit_conversion_candidate_score(state, action, TAROT_SUIT_CONVERSIONS[name]) * 0.12
    if name == "Strength":
        return _strength_candidate_score(state, action) * 0.12
    if name in TAROT_ENHANCEMENTS:
        return _enhancement_candidate_score(state, action, TAROT_ENHANCEMENTS[name]) * 0.18
    return 0.0


def _hand_viability_delta_bonus(state: GameState, simulated: GameState) -> float:
    targets = _viability_targets(state)
    if not targets:
        return 0.0
    try:
        before = hand_draw_odds(_future_draw_state(state))
        after = hand_draw_odds(_future_draw_state(simulated), draw_size=before.draw_size)
    except (ValueError, TypeError, AttributeError):
        return 0.0

    bonus = 0.0
    for hand_type in targets:
        delta = max(0.0, after.probability(hand_type) - before.probability(hand_type))
        if delta <= 0.0:
            continue
        scale = 120.0 if hand_type in ADVANCED_HAND_TYPES else 42.0
        bonus += min(10.0, delta * scale)
    return min(14.0, bonus)


def _viability_targets(state: GameState) -> tuple[HandType, ...]:
    targets: list[HandType] = []
    preferred = _preferred_hand_type(state)
    if preferred is not None:
        targets.append(preferred)
    for hand_type in ADVANCED_HAND_TYPES:
        if state.hand_levels.get(hand_type.value, 1) > 1:
            targets.append(hand_type)
    if not targets:
        targets.extend((HandType.FLUSH, HandType.FULL_HOUSE, HandType.THREE_OF_A_KIND))
    deduped: list[HandType] = []
    for hand_type in targets:
        if hand_type not in deduped:
            deduped.append(hand_type)
    return tuple(deduped[:4])


def _future_draw_state(state: GameState) -> GameState:
    if not state.hand:
        return state
    known_deck = state.known_deck + state.hand if state.known_deck else ()
    deck_size = max(state.deck_size + len(state.hand), len(known_deck))
    return replace(state, hand=(), known_deck=known_deck, deck_size=deck_size)


def _enhancement_candidate_score(state: GameState, action: Action, enhancement: str) -> float:
    score = 0.0
    for index in action.card_indices:
        if not 0 <= index < len(state.hand):
            continue
        card = state.hand[index]
        rank_value = _rank_value(card)
        if not card.enhancement:
            score += 4.0
        if card.seal:
            score += 3.0
        if card.edition:
            score += 3.0
        if enhancement == "GLASS":
            score += 12.0 + (rank_value * 0.8)
            if card.seal == "Red":
                score += 6.0
            if card.enhancement == "GLASS":
                score -= 10.0
        elif enhancement == "STEEL":
            score += 8.0 + (rank_value * 0.35)
        elif enhancement == "MULT":
            score += 7.0 + (rank_value * 0.2)
        elif enhancement == "BONUS":
            score += 5.0 + (rank_value * 0.12)
        elif enhancement == "LUCKY":
            score += 6.0 + (4.0 if any(joker.name == "Lucky Cat" for joker in state.jokers) else 0.0)
        elif enhancement == "GOLD":
            score += 5.0
        elif enhancement == "STONE":
            score += 3.0 if rank_value <= 8 else -2.0
        elif enhancement == "WILD":
            score += 4.0
    return score


def _death_candidate_score(state: GameState, action: Action) -> float:
    if len(action.card_indices) != 2:
        return 0.0
    source_index = max(action.card_indices)
    target_index = min(action.card_indices)
    if source_index >= len(state.hand) or target_index >= len(state.hand):
        return 0.0
    source = state.hand[source_index]
    target = state.hand[target_index]
    score = max(0, _rank_value(source) - _rank_value(target)) * 1.5
    same_rank_count = sum(1 for card in state.hand if card.rank == source.rank)
    same_suit_count = sum(1 for card in state.hand if card.suit == source.suit)
    score += same_rank_count * 5.0
    score += same_suit_count * 1.5
    if source.enhancement or source.seal or source.edition:
        score += 8.0
    return score


def _hanged_man_candidate_score(state: GameState, action: Action) -> float:
    score = 0.0
    selected = set(action.card_indices)
    kept = tuple(card for index, card in enumerate(state.hand) if index not in selected)
    for index in action.card_indices:
        if 0 <= index < len(state.hand):
            card = state.hand[index]
            score += max(0, 15 - _rank_value(card)) * 0.9
            if not card.enhancement and not card.seal and not card.edition:
                score += 2.0
    score += _duplicate_support_score(kept) * 1.5
    score += _suit_support_score(kept) * 0.8
    return score


def _strength_candidate_score(state: GameState, action: Action) -> float:
    score = 0.0
    for index in action.card_indices:
        if not 0 <= index < len(state.hand):
            continue
        card = state.hand[index]
        score += 2.0
        if card.rank in {"9", "10", "T", "Q", "K"}:
            score += 5.0
        if _near_duplicate_after_strength(state, index):
            score += 7.0
    return score


def _suit_conversion_candidate_score(state: GameState, action: Action, target_suit: str) -> float:
    score = 0.0
    current_count = sum(1 for card in state.hand if card.suit == target_suit)
    converted_count = sum(1 for index in action.card_indices if 0 <= index < len(state.hand) and state.hand[index].suit != target_suit)
    score += current_count * 1.5
    score += converted_count * 5.0
    if _preferred_hand_type(state) in {HandType.FLUSH, HandType.STRAIGHT_FLUSH, HandType.FLUSH_HOUSE, HandType.FLUSH_FIVE}:
        score += converted_count * 3.0
    return score


def _selected_card_quality(state: GameState, action: Action) -> float:
    score = 0.0
    for index in action.card_indices:
        if not 0 <= index < len(state.hand):
            continue
        card = state.hand[index]
        score += _rank_value(card) * 0.4
        if card.enhancement:
            score += 3.0
        if card.seal:
            score += 5.0
        if card.edition:
            score += 4.0
    return score


def _planet_alignment_score(state: GameState, hand_type: HandType) -> float:
    score = 0.0
    if hand_type == _preferred_hand_type(state):
        score += 18.0
    if hand_type == _best_current_hand_type(state):
        score += 8.0
    level = max(1, int(state.hand_levels.get(hand_type.value, 1)))
    if level >= 2:
        score += min(12.0, (level - 1) * 3.0)
    return score


def _best_current_hand_type(state: GameState) -> HandType | None:
    if not state.hand:
        return None
    try:
        from balatro_ai.rules.hand_evaluator import best_play_from_hand

        return best_play_from_hand(state.hand, state.hand_levels).hand_type
    except (TypeError, ValueError, AttributeError):
        return None


def _preferred_hand_type(state: GameState) -> HandType | None:
    try:
        from balatro_ai.bots.basic_strategy.hand_preferences import _preferred_hand_type as basic_preferred_hand_type

        return basic_preferred_hand_type(state)
    except (ImportError, TypeError, ValueError, AttributeError):
        return None


def _observatory_should_hold_planet(state: GameState, hand_type: HandType) -> bool:
    if "Observatory" not in state.vouchers:
        return False
    preferred = _preferred_hand_type(state)
    return preferred is None or preferred == hand_type


def _minimum_delta(state: GameState, config: ConsumableSearchConfig) -> float:
    if state.phase == GamePhase.SHOP:
        return config.min_shop_delta
    if state.phase == GamePhase.BOOSTER_OPENED:
        return config.min_pack_delta
    return config.min_blind_delta


def _action_consumable_name(state: GameState, action: Action) -> str | None:
    index = _action_index(action)
    if index is None or not 0 <= index < len(state.consumables):
        return None
    return state.consumables[index]


def _action_index(action: Action) -> int | None:
    raw = action.metadata.get("index", action.amount)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _consumable_slots_are_full(state: GameState) -> bool:
    return len(state.consumables) >= _consumable_slot_limit(state)


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


def _item_label(item: object) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, Mapping):
        return str(item.get("label", item.get("name", item.get("key", ""))))
    return str(item)


def _near_duplicate_after_strength(state: GameState, index: int) -> bool:
    if not 0 <= index < len(state.hand):
        return False
    next_rank = {
        "2": "3",
        "3": "4",
        "4": "5",
        "5": "6",
        "6": "7",
        "7": "8",
        "8": "9",
        "9": "10",
        "T": "J",
        "10": "J",
        "J": "Q",
        "Q": "K",
        "K": "A",
        "A": "2",
    }.get(state.hand[index].rank)
    return next_rank is not None and any(card.rank == next_rank for card_index, card in enumerate(state.hand) if card_index != index)


def _duplicate_support_score(cards: tuple[object, ...]) -> float:
    counts: dict[str, int] = {}
    for card in cards:
        rank = getattr(card, "rank", "")
        counts[rank] = counts.get(rank, 0) + 1
    return float(sum(count * count for count in counts.values() if count >= 2))


def _suit_support_score(cards: tuple[object, ...]) -> float:
    counts: dict[str, int] = {}
    for card in cards:
        suit = getattr(card, "suit", "")
        counts[suit] = counts.get(suit, 0) + 1
    return float(max(counts.values(), default=0))


def _rank_value(card: object) -> int:
    rank = getattr(card, "rank", str(card))
    return {
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "10": 10,
        "T": 10,
        "J": 11,
        "Q": 12,
        "K": 13,
        "A": 14,
    }.get(rank, 0)


def _annotated_action(action: Action, *, search_value: float) -> Action:
    return Action(
        action.action_type,
        card_indices=action.card_indices,
        target_id=action.target_id,
        amount=action.amount,
        metadata={**action.metadata, "search": "consumable_use", "search_value": round(search_value, 6)},
    )
