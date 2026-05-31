"""Beam search over deterministic shop actions for Phase 7."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from random import Random
from typing import Any

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GamePhase, GameState
from balatro_ai.search.consumable_search import (
    ConsumableSearchConfig,
    consumable_action_value,
    shop_consumable_action_is_candidate,
    shop_consumable_actions,
    simulate_consumable_action,
)
from balatro_ai.search.forward_sim import PLANET_TO_HAND, simulate_buy, simulate_end_shop, simulate_open_pack, simulate_reroll, simulate_sell
from balatro_ai.search.shop_sampler import ShopSampler

LeafValueFn = Callable[[GameState], float]
ActionValueFn = Callable[[GameState, Action], float]
LeafTermsFn = Callable[[GameState], "ShopLeafTerms"]
_DEFAULT_ACTION_POTENTIAL_WEIGHT = 0.65
# How much of a sold joker's owned value to charge against the sell action's
# search value (anti-churn). Old behavior was 0.15 with a >=0 floor; see the
# SELL branch of shop_action_search_value for why that caused sell->rebuy churn.
# Env-tunable for winrate A/B.
_SELL_OWNED_VALUE_COEFF: float = float(os.environ.get("BALATRO_SELL_OWNED_COEFF", "0.45"))

_CONDITIONAL_CHIP_JOKERS = frozenset(
    {
        "Sly Joker",
        "Wily Joker",
        "Clever Joker",
        "Devious Joker",
        "Crafty Joker",
        "Arrowhead",
        "Square Joker",
        "Runner",
    }
)
_CONDITIONAL_MULT_ROLE_CREDITS = {
    "Jolly Joker": 8.0,
    "Zany Joker": 8.0,
    "Mad Joker": 8.0,
    "Crazy Joker": 8.0,
    "Droll Joker": 8.0,
    "Greedy Joker": 8.0,
    "Lusty Joker": 8.0,
    "Wrathful Joker": 8.0,
    "Gluttonous Joker": 8.0,
}
_CONDITIONAL_PRESSURE_BOSSES = frozenset(
    {
        "The Water",
        "The Manacle",
        "The Needle",
        "The Eye",
        "The Mouth",
        "The Psychic",
    }
)


@dataclass(frozen=True, slots=True)
class ShopSearchConfig:
    beam_width: int = 8
    depth: int = 3
    reroll_samples: int = 32
    seed: int = 0
    leaf_weight: float = 0.35
    min_search_value: float = float("-inf")
    trace_top_paths: int = 0


@dataclass(frozen=True, slots=True)
class ShopSearchContext:
    rerolls_in_shop: int = 0
    packs_opened_in_shop: int = 0
    filled_last_joker_slot: bool = False


@dataclass(frozen=True, slots=True)
class ShopLeafTerms:
    raw_owned_value: float
    owned_value: float
    role_value: float
    survival_value: float
    build_capacity_value: float
    headroom_value: float
    money_value: float
    slot_value: float
    consumable_value: float
    leveling_value: float
    coherence_value: float
    missing_penalty: float
    money_floor_penalty: float
    clear_gate_penalty: float
    conditional_penalty: float
    build_score: float
    root_build_score: float
    pressure_ratio: float
    raw_pressure_ratio: float
    capacity_ratio: float

    @property
    def total(self) -> float:
        return (
            self.owned_value
            + self.role_value
            + self.survival_value
            + self.build_capacity_value
            + self.headroom_value
            + self.money_value
            + self.slot_value
            + self.consumable_value
            + self.leveling_value
            + self.coherence_value
            - self.missing_penalty
            - self.money_floor_penalty
            - self.clear_gate_penalty
            - self.conditional_penalty
        )

    def to_trace_dict(self) -> dict[str, float]:
        return {
            "owned_raw": round(self.raw_owned_value, 3),
            "owned": round(self.owned_value, 3),
            "roles": round(self.role_value, 3),
            "survival": round(self.survival_value, 3),
            "build_delta": round(self.build_capacity_value, 3),
            "headroom": round(self.headroom_value, 3),
            "money": round(self.money_value, 3),
            "slots": round(self.slot_value, 3),
            "consumables": round(self.consumable_value, 3),
            "leveling": round(self.leveling_value, 3),
            "coherence": round(self.coherence_value, 3),
            "missing_penalty": round(self.missing_penalty, 3),
            "money_floor_penalty": round(self.money_floor_penalty, 3),
            "clear_gate_penalty": round(self.clear_gate_penalty, 3),
            "conditional_penalty": round(self.conditional_penalty, 3),
            "build_score": round(self.build_score, 3),
            "root_build_score": round(self.root_build_score, 3),
            "pressure_ratio": round(self.pressure_ratio, 3),
            "raw_pressure_ratio": round(self.raw_pressure_ratio, 3),
            "capacity_ratio": round(self.capacity_ratio, 3),
            "total": round(self.total, 3),
        }


@dataclass(frozen=True, slots=True)
class _BeamNode:
    state: GameState
    first_action: Action | None
    path: tuple[Action, ...]
    action_score: float
    leaf_score: float
    score: float
    leaf_terms: ShopLeafTerms | None = None
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
    shop_context: ShopSearchContext | None = None,
) -> Action | None:
    """Return the first action from the best shop beam sequence."""

    if state.phase != GamePhase.SHOP:
        return None
    search_config = config or ShopSearchConfig()
    if search_config.depth <= 0 or search_config.beam_width <= 0:
        return None

    shop_sampler = sampler or ShopSampler.from_default_data()
    runtime_context = shop_context or ShopSearchContext()
    if leaf_value_fn is None:
        root_build_score = _shop_build_score(state)
        leaf_terms: LeafTermsFn | None = lambda leaf_state: shop_leaf_terms(
            leaf_state,
            root_state=state,
            root_build_score=root_build_score,
        )
        leaf_value = lambda leaf_state: leaf_terms(leaf_state).total if leaf_terms is not None else 0.0
    else:
        leaf_terms = None
        leaf_value = leaf_value_fn
    action_potential_weight = (
        _DEFAULT_ACTION_POTENTIAL_WEIGHT if leaf_value_fn is None and action_value_fn is None else 0.0
    )
    action_value = action_value_fn or (
        lambda current, action: shop_action_search_value(
            current,
            action,
            sampler=shop_sampler,
            config=search_config,
            shop_context=runtime_context,
        )
    )
    root_actions = _legal_shop_actions(
        state,
        shop_sampler,
        root_actions=state.legal_actions,
        protected_jokers=protected_jokers,
    )
    if not root_actions:
        return None

    beams: tuple[_BeamNode, ...] = (_BeamNode(state=state, first_action=None, path=(), action_score=0.0, leaf_score=0.0, score=0.0),)
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
                    leaf_terms_fn=leaf_terms,
                    leaf_weight=search_config.leaf_weight,
                    action_potential_weight=action_potential_weight,
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
    trace_candidates = _trace_candidates(candidates, limit=search_config.trace_top_paths)
    if best.score < search_config.min_search_value:
        return None
    return _annotated_action(
        best.first_action,
        search_value=best.score,
        path=best.path,
        snapshot=_shop_trace_snapshot(state) if trace_candidates else None,
        candidates=trace_candidates,
    )


def shop_action_search_value(
    state: GameState,
    action: Action,
    *,
    sampler: ShopSampler | None = None,
    config: ShopSearchConfig | None = None,
    shop_context: ShopSearchContext | None = None,
) -> float:
    """Score one shop action using Basic Strategy's current shop heuristics."""

    if action.action_type == ActionType.REROLL:
        shop_sampler = sampler or ShopSampler.from_default_data()
        search_config = config or ShopSearchConfig()
        runtime_context = shop_context or ShopSearchContext()
        reroll_cost = shop_sampler.reroll_cost(state)
        basic_value = _basic_shop_action_value(state, action, runtime_context)
        spend_penalty = _spend_opportunity_penalty(state, reroll_cost)
        reroll_value = shop_sampler.reroll_ev(state, samples=search_config.reroll_samples, rng=Random(search_config.seed))
        sampled_value = reroll_value - spend_penalty
        if basic_value <= 0.0 and _safe_shop_blocks_reroll_override(state):
            return min(0.0, sampled_value)
        if basic_value <= 0.0:
            return sampled_value * 0.35
        blended_value = (basic_value * 0.55) + (sampled_value * 0.45)
        return min(sampled_value, blended_value)
    if action.action_type == ActionType.SELL:
        kind = _action_kind(action, default="joker")
        index = _action_index(action)
        if kind == "consumable":
            return _consumable_sell_search_value(state, action)
        if index is None or not 0 <= index < len(state.jokers):
            return 0.0
        from balatro_ai.bots.basic_strategy.shop_jokers import _owned_joker_value

        sold = state.jokers[index]
        # Selling a joker DESTROYS its owned value. The old formula floored at 0
        # and only subtracted 15% of owned value, so selling a useful joker was
        # scored as a free (>=0) money gain. Combined with `_basic_shop_action_value`
        # for BUY being state-relative (a freed slot inflates the next buy's
        # heuristic value), the beam learned to SELL a good joker then RE-BUY into
        # the slot it freed — netting positive action-score despite a strictly
        # worse build (confirmed via scripts/shop_decision_trace.py beam dump:
        # "sell Scary -> buy Blue" beat "buy Blue -> end" 383 vs 361 with a LOWER
        # leaf). That churn caps the build and destroys scaling jokers' progress.
        # Fix: charge the sale the value it destroys (no >=0 floor), so selling a
        # HIGH-value joker is net-negative (blocks churn) while dumping a LOW-value
        # / dead joker still nets ~ its sell price (legitimate upgrades survive).
        # Coefficient env-tunable for winrate A/B (scripts/shop_paired_ab.py).
        return float(sold.sell_value or 0) - (_owned_joker_value(state, sold, remove_index=index) * _SELL_OWNED_VALUE_COEFF)
    if action.action_type in {ActionType.BUY, ActionType.OPEN_PACK}:
        value = _basic_shop_action_value(state, action, shop_context or ShopSearchContext())
        if action.action_type == ActionType.BUY:
            value -= _unresolved_pressure_buy_penalty(state, action)
            value -= _conditional_joker_buy_penalty(state, action)
        value -= _clear_first_speculative_spend_penalty(state, action)
        return value
    if action.action_type == ActionType.USE_CONSUMABLE:
        search_config = config or ShopSearchConfig()
        try:
            return consumable_action_value(
                state,
                action,
                config=ConsumableSearchConfig(
                    leaf_samples=1,
                    seed=search_config.seed,
                    stochastic_samples=max(4, min(12, search_config.reroll_samples)),
                    min_shop_delta=0.0,
                ),
                value_fn=shop_leaf_value,
                baseline_value=shop_leaf_value(state),
                sampler=sampler,
            ) * 0.55
        except (ValueError, IndexError, TypeError, AttributeError):
            return 0.0
    return 0.0


def _basic_shop_action_value(state: GameState, action: Action, context: ShopSearchContext) -> float:
    from balatro_ai.bots.basic_strategy.profile import _ShopContext
    from balatro_ai.bots.basic_strategy.shop_pressure import _shop_pressure
    from balatro_ai.bots.basic_strategy.shop_values import _shop_action_value

    return _shop_action_value(
        state,
        action,
        _shop_pressure(state),
        _ShopContext(
            rerolls_in_shop=context.rerolls_in_shop,
            packs_opened_in_shop=context.packs_opened_in_shop,
            filled_last_joker_slot=context.filled_last_joker_slot,
        ),
    )


def _safe_shop_blocks_reroll_override(state: GameState) -> bool:
    try:
        from balatro_ai.bots.basic_strategy.shop_pressure import _shop_pressure

        pressure = _shop_pressure(state)
        return pressure.ratio < 1.0 and pressure.raw_ratio < 1.05
    except (ImportError, TypeError, ValueError, AttributeError):
        return state.ante >= 2


def shop_leaf_value(
    state: GameState,
    *,
    root_state: GameState | None = None,
    root_build_score: float | None = None,
) -> float:
    """Score an intermediate shop state after deterministic actions."""

    return shop_leaf_terms(state, root_state=root_state, root_build_score=root_build_score).total


def shop_leaf_terms(
    state: GameState,
    *,
    root_state: GameState | None = None,
    root_build_score: float | None = None,
) -> ShopLeafTerms:
    """Break the shop leaf value into auditable terms."""

    from balatro_ai.bots.basic_strategy.build_profile import _build_profile
    from balatro_ai.bots.basic_strategy.shop_jokers import _owned_joker_value

    profile = _build_profile(state)
    raw_owned_value = sum(_owned_joker_value(state, joker, remove_index=index) for index, joker in enumerate(state.jokers))
    survival_value = shop_survival_value(state)
    role_value = 0.0
    for role in ("chips", "mult", "xmult", "scaling", "economy"):
        requirement = max(1.0, profile.role_requirement(role))
        role_value += min(1.0, profile.role_score(role) / requirement) * 10.0
    missing_penalty = len(profile.missing_roles) * (8.0 + min(8.0, max(0, state.ante - 2) * 1.5))
    money_value = _shop_money_value(state)
    leaf_build_score = _shop_build_score(state)
    baseline = root_build_score
    if baseline is None and root_state is not None:
        baseline = _shop_build_score(root_state)
    baseline = float(baseline or 0.0)
    build_capacity_value = shop_build_capacity_delta_value(
        state,
        root_state=root_state,
        root_build_score=baseline,
        leaf_build_score=leaf_build_score,
    )
    headroom_value = shop_capacity_headroom_value(state)
    slot_value = _normal_joker_open_slots(state) * 2.5
    consumable_value = _shop_consumable_inventory_value(state)
    leveling_value = _shop_leveling_value(state)
    coherence_value = _shop_archetype_coherence_value(state)
    money_floor_penalty = _shop_money_floor_penalty(state)
    pressure_ratio, raw_pressure_ratio, capacity_ratio = _shop_pressure_metrics(state)
    clear_gate_penalty = _shop_clear_gate_penalty(
        state,
        root_state=root_state,
        root_build_score=root_build_score,
        leaf_build_score=leaf_build_score,
        pressure_ratio=pressure_ratio,
        raw_pressure_ratio=raw_pressure_ratio,
        capacity_ratio=capacity_ratio,
    )
    conditional_penalty = _conditional_joker_reliability_penalty(
        state,
        profile,
        pressure_ratio=pressure_ratio,
        raw_pressure_ratio=raw_pressure_ratio,
        capacity_ratio=capacity_ratio,
    )
    return ShopLeafTerms(
        raw_owned_value=raw_owned_value,
        owned_value=shop_owned_joker_leaf_value(
            state,
            raw_owned_value,
            pressure_ratio=pressure_ratio,
            raw_pressure_ratio=raw_pressure_ratio,
            capacity_ratio=capacity_ratio,
        ),
        role_value=role_value,
        survival_value=survival_value,
        build_capacity_value=build_capacity_value,
        headroom_value=headroom_value,
        money_value=money_value,
        slot_value=slot_value,
        consumable_value=consumable_value,
        leveling_value=leveling_value,
        coherence_value=coherence_value,
        missing_penalty=missing_penalty,
        money_floor_penalty=money_floor_penalty,
        clear_gate_penalty=clear_gate_penalty,
        conditional_penalty=conditional_penalty,
        build_score=leaf_build_score,
        root_build_score=baseline,
        pressure_ratio=pressure_ratio,
        raw_pressure_ratio=raw_pressure_ratio,
        capacity_ratio=capacity_ratio,
    )


def shop_build_capacity_delta_value(
    state: GameState,
    *,
    root_state: GameState | None = None,
    root_build_score: float | None = None,
    leaf_build_score: float | None = None,
) -> float:
    """Reward shop lines that leave the build with higher scoring capacity."""

    if root_state is None and root_build_score is None:
        return 0.0
    baseline = root_build_score
    if baseline is None and root_state is not None:
        baseline = _shop_build_score(root_state)
    if baseline is None:
        return 0.0

    leaf_score = _shop_build_score(state) if leaf_build_score is None else leaf_build_score
    delta = leaf_score - float(baseline)
    ante = max(1, state.ante)
    scale = 0.018 + min(0.024, max(0, ante - 1) * 0.004)
    weighted = delta * scale
    return max(-75.0, min(95.0, weighted))


def shop_survival_value(state: GameState) -> float:
    """Estimate how well this shop leaf survives the upcoming score ramp."""

    try:
        from balatro_ai.bots.basic_strategy.build_profile import _build_profile
        from balatro_ai.bots.basic_strategy.shop_pressure import _shop_pressure

        pressure = _shop_pressure(state)
        profile = _build_profile(state)
    except (ImportError, TypeError, ValueError, AttributeError):
        return 0.0

    safe_clear = _probability_from_pressure_ratio(pressure.ratio)
    raw_clear = _probability_from_pressure_ratio(pressure.raw_ratio)
    boss_ready = _probability_from_pressure_ratio(pressure.raw_ratio * max(1.0, pressure.boss_target_multiplier))
    role_completion = 1.0 - (len(profile.missing_roles) / 5.0)
    late_role_penalty = 0.0
    if state.ante >= 5 and ("xmult" in profile.missing_roles or "scaling" in profile.missing_roles):
        late_role_penalty = min(18.0, (state.ante - 4) * 4.5)

    return (
        (safe_clear * 48.0)
        + (raw_clear * 18.0)
        + (boss_ready * 12.0)
        + (max(0.0, role_completion) * 20.0)
        - late_role_penalty
    )


def shop_capacity_headroom_value(state: GameState) -> float:
    """Reward scoring cushion beyond the next shop-pressure target."""

    try:
        from balatro_ai.bots.basic_strategy.shop_pressure import _shop_pressure

        pressure = _shop_pressure(state)
        capacity_ratio = pressure.build_capacity / max(1.0, pressure.target_score)
        raw_cushion = max(0.0, capacity_ratio - 1.0)
        value = min(42.0, raw_cushion * 22.0)
        if state.ante <= 2:
            value *= 0.65
        elif state.ante >= 5:
            value *= 1.12
        if pressure.raw_ratio >= 1.05:
            value *= 0.65
        return value
    except (ImportError, TypeError, ValueError, AttributeError):
        return 0.0


def shop_owned_joker_leaf_value(
    state: GameState,
    raw_owned_value: float,
    *,
    pressure_ratio: float | None = None,
    raw_pressure_ratio: float | None = None,
    capacity_ratio: float | None = None,
) -> float:
    """Value owned jokers without letting weak, broke leaves coast on inventory value alone."""

    raw_value = max(0.0, float(raw_owned_value))
    if raw_value <= 0.0:
        return 0.0

    if pressure_ratio is None or raw_pressure_ratio is None or capacity_ratio is None:
        pressure_ratio, raw_pressure_ratio, capacity_ratio = _shop_pressure_metrics(state)

    scale = 0.72
    if capacity_ratio < 0.55:
        scale *= 0.50
    elif capacity_ratio < 0.80:
        scale *= 0.66
    elif capacity_ratio < 1.00:
        scale *= 0.84
    elif capacity_ratio >= 1.60 and raw_pressure_ratio <= 0.85:
        scale *= 1.05

    floor_shortfall = max(0, _minimum_safe_shop_money(state) - state.money)
    if floor_shortfall > 0 and (pressure_ratio >= 1.0 or raw_pressure_ratio >= 1.0):
        scale *= max(0.70, 1.0 - min(0.30, floor_shortfall * 0.035))

    value = raw_value * scale
    if capacity_ratio < 1.0:
        cap = 140.0 + (max(1, state.ante) * 26.0)
        value = min(value, cap)
    return value


def _conditional_joker_reliability_penalty(
    state: GameState,
    profile: Any,
    *,
    pressure_ratio: float | None = None,
    raw_pressure_ratio: float | None = None,
    capacity_ratio: float | None = None,
) -> float:
    """Discount high-capacity leaves that need narrow hands but lack stable mult."""

    conditional_chips = [joker.name for joker in state.jokers if joker.name in _CONDITIONAL_CHIP_JOKERS]
    if not conditional_chips:
        return 0.0

    if pressure_ratio is None or raw_pressure_ratio is None or capacity_ratio is None:
        pressure_ratio, raw_pressure_ratio, capacity_ratio = _shop_pressure_metrics(state)

    chip_count = len(conditional_chips)
    conditional_mult_count = sum(1 for joker in state.jokers if joker.name in _CONDITIONAL_MULT_ROLE_CREDITS)
    stable_mult_ratio = _stable_mult_ratio_after_conditional_credit(state, profile)
    missing_stable_mult = stable_mult_ratio < 0.80

    penalty = 0.0
    if chip_count >= 2:
        penalty += 8.0 + ((chip_count - 1) * 6.0)
    if missing_stable_mult:
        penalty += chip_count * (6.0 + ((0.80 - stable_mult_ratio) * 8.0))
        if chip_count >= 2:
            penalty += 10.0

    if (
        state.ante >= 2
        and "Clever Joker" in conditional_chips
        and "Square Joker" in conditional_chips
        and missing_stable_mult
    ):
        penalty += 10.0

    pressure = max(float(pressure_ratio), float(raw_pressure_ratio))
    capacity_ratio = float(capacity_ratio)
    if pressure >= 0.90 or capacity_ratio < 1.18:
        penalty *= 1.0 + min(0.35, max(0.0, pressure - 0.85) * 0.7)

    boss_name = _upcoming_shop_boss_name(state)
    if boss_name in _CONDITIONAL_PRESSURE_BOSSES:
        narrow_count = chip_count + conditional_mult_count
        penalty += 10.0 + (narrow_count * 8.0)
        if missing_stable_mult:
            penalty += 12.0

    if state.ante <= 1:
        penalty *= 0.25
    elif state.ante >= 4:
        penalty *= 1.10
    return min(110.0, max(0.0, penalty))


def _conditional_joker_buy_penalty(state: GameState, action: Action) -> float:
    if action.action_type != ActionType.BUY:
        return 0.0
    item = _buy_action_item(state, action)
    name = _item_label(item)
    if name not in _CONDITIONAL_CHIP_JOKERS:
        return 0.0

    try:
        next_state = simulate_buy(state, action)
        from balatro_ai.bots.basic_strategy.build_profile import _build_profile

        profile = _build_profile(next_state)
    except (ImportError, TypeError, ValueError, IndexError, AttributeError):
        return 0.0

    stable_mult_ratio = _stable_mult_ratio_after_conditional_credit(next_state, profile)
    missing_stable_mult = stable_mult_ratio < 0.80
    pressure_ratio, raw_pressure_ratio, capacity_ratio = _shop_pressure_metrics(next_state)
    pressure = max(pressure_ratio, raw_pressure_ratio)
    conditional_chip_count = sum(1 for joker in next_state.jokers if joker.name in _CONDITIONAL_CHIP_JOKERS)

    penalty = 0.0
    if missing_stable_mult:
        penalty += 12.0 + ((0.80 - stable_mult_ratio) * 18.0)
        if conditional_chip_count >= 2:
            penalty += 12.0 + ((conditional_chip_count - 2) * 6.0)

    if pressure >= 0.90 or capacity_ratio < 1.12:
        penalty += max(0.0, 1.12 - capacity_ratio) * 70.0
        penalty += max(0.0, 1.00 - capacity_ratio) * 105.0
        penalty += max(0.0, pressure - 0.90) * 34.0

    boss_name = _upcoming_shop_boss_name(next_state)
    if boss_name in _CONDITIONAL_PRESSURE_BOSSES:
        penalty += 8.0
        if missing_stable_mult:
            penalty += 8.0

    if state.ante <= 1:
        penalty *= 0.35
    elif state.ante >= 4:
        penalty *= 1.10
    return min(160.0, max(0.0, penalty))


def _stable_mult_ratio_after_conditional_credit(state: GameState, profile: Any) -> float:
    requirement = max(1.0, float(profile.role_requirement("mult")))
    conditional_credit = sum(_CONDITIONAL_MULT_ROLE_CREDITS.get(joker.name, 0.0) for joker in state.jokers)
    stable_score = max(0.0, float(profile.role_score("mult")) - conditional_credit)
    return stable_score / requirement


def _upcoming_shop_boss_name(state: GameState) -> str | None:
    try:
        from balatro_ai.bots.basic_strategy.shop_forecast import _upcoming_boss_blind_name

        return _upcoming_boss_blind_name(state)
    except (ImportError, TypeError, ValueError, AttributeError):
        return None


# Per-hand-level reward in the shop leaf value. Hand leveling (planet cards /
# Celestial packs) is the dominant scaling mechanic, but the depth-2 shop search
# is MYOPIC about its compounding payoff over future antes, so it under-invests
# vs a leveling-aware build (basic_strategy_bot levels aggressively and wins ~23%
# while the solver caps ~ante 5). This term biases the search toward buying/using
# planet cards and opening Celestial packs. Tuned via winrate.
_LEVELING_VALUE_PER_LEVEL: float = 3.0


def _shop_leveling_value(state: GameState) -> float:
    levels = getattr(state, "hand_levels", None) or {}
    total = 0.0
    for lvl in levels.values():
        try:
            total += max(0.0, float(lvl) - 1.0)
        except (TypeError, ValueError):
            continue
    return total * _LEVELING_VALUE_PER_LEVEL


# --- Build coherence: focused (concentrated) hand leveling ---------------
# Codifies researched Balatro principle #2 (memory/reference_balatro_strategy):
# "CONCENTRATE planet cards on ONE committed hand; do NOT spread — it compounds
# across the run." A build audit (scripts/solver_build_audit.py) showed the
# solver's BEST runs reach ante 8 by pushing a single hand very high (e.g. Pair
# lvl 8) while early deaths spread a few levels thin. The generic `leveling_value`
# rewards every level equally, so it does NOT distinguish a concentrated build
# from a scattered one. This term adds an EXTRA reward for the single most-leveled
# hand, on top of uniform leveling, so the depth-2 shop beam prefers buying/using
# planets that compound the hand it has already committed to.
#
# Deliberately hand-AGNOSTIC rather than joker-archetype based: the audit found
# only ~35% of owned jokers fall in any archetype key list, and the #1 leveled
# hand (Full House, 26/60 games) plus Straight (17) are in NO archetype — so a
# joker-archetype coherence term barely engages and mis-steers away from the
# hands the solver actually builds. Concentration is coverage-independent: it
# rewards committing to whichever hand the deck/jokers favor. Weight via winrate.
#
# DEFAULT 0.0 (INERT / disabled): a 40-seed PAIRED A/B (scripts/shop_paired_ab.py,
# weight 0->4) left 38/40 games byte-identical (2 worse, 0 better). Root cause:
# this term depends only on owned `hand_levels`, which is CONSTANT across nearly
# every shop action (buying a joker / rerolling / ending shop don't change hand
# levels), so it never moves the search argmax — it only matters for the rare
# planet decision. Leveling-based shop-leaf terms are structurally inert. Kept
# env-toggleable for experiments, but OFF in production. To steer BUILDS, change
# how JOKER PURCHASES are valued (owned/role/build terms), not leveling.
_COHERENCE_LEVEL_WEIGHT: float = float(os.environ.get("BALATRO_COHERENCE_LEVEL_W", "0.0"))


def _shop_archetype_coherence_value(state: GameState) -> float:
    if _COHERENCE_LEVEL_WEIGHT == 0.0:
        return 0.0
    levels = getattr(state, "hand_levels", None) or {}
    peak = 0.0
    for lvl in levels.values():
        try:
            extra = max(0.0, float(lvl) - 1.0)
        except (TypeError, ValueError):
            continue
        if extra > peak:
            peak = extra
    return peak * _COHERENCE_LEVEL_WEIGHT


def _shop_build_score(state: GameState) -> float:
    try:
        from balatro_ai.bots.basic_strategy.build_scoring import _sample_build_score

        return max(0.0, float(_sample_build_score(state, state.jokers)))
    except (ImportError, TypeError, ValueError, AttributeError):
        return 0.0


def _probability_from_pressure_ratio(ratio: float) -> float:
    try:
        value = float(ratio)
    except (TypeError, ValueError):
        return 0.0
    if value <= 0:
        return 1.0
    if value <= 0.55:
        return 0.99
    if value <= 0.8:
        return 0.88 + ((0.8 - value) * 0.44)
    if value <= 1.0:
        return 0.62 + ((1.0 - value) * 1.3)
    if value <= 1.25:
        return 0.35 + ((1.25 - value) * 1.08)
    if value <= 1.7:
        return 0.08 + ((1.7 - value) * 0.6)
    return max(0.0, 0.08 - ((value - 1.7) * 0.08))


def _shop_pressure_metrics(state: GameState) -> tuple[float, float, float]:
    try:
        from balatro_ai.bots.basic_strategy.shop_pressure import _shop_pressure

        pressure = _shop_pressure(state)
        capacity_ratio = float(pressure.build_capacity) / max(1.0, float(pressure.target_score))
        return float(pressure.ratio), float(pressure.raw_ratio), capacity_ratio
    except (ImportError, TypeError, ValueError, AttributeError):
        return 0.0, 0.0, 0.0


def _shop_clear_gate_penalty(
    state: GameState,
    *,
    root_state: GameState | None = None,
    root_build_score: float | None = None,
    leaf_build_score: float | None = None,
    pressure_ratio: float | None = None,
    raw_pressure_ratio: float | None = None,
    capacity_ratio: float | None = None,
) -> float:
    """Keep value lines behind the next-blind clear check."""

    if pressure_ratio is None or raw_pressure_ratio is None or capacity_ratio is None:
        pressure_ratio, raw_pressure_ratio, capacity_ratio = _shop_pressure_metrics(state)
    pressure = max(float(pressure_ratio), float(raw_pressure_ratio))
    capacity_ratio = float(capacity_ratio)
    if pressure <= 0.0 and capacity_ratio <= 0.0:
        return 0.0

    leaf_score = _shop_build_score(state) if leaf_build_score is None else float(leaf_build_score)
    baseline_known = root_state is not None or root_build_score is not None
    baseline = root_build_score
    if baseline is None and root_state is not None:
        baseline = _shop_build_score(root_state)
    if baseline is None:
        baseline = leaf_score
    baseline = float(baseline)
    build_gain = max(0.0, leaf_score - baseline)
    meaningful_gain = max(45.0, baseline * 0.30)
    gain_credit = min(1.0, build_gain / meaningful_gain) if baseline_known else 0.0

    pressure_gap = max(0.0, float(pressure_ratio) - 1.0)
    raw_gap = max(0.0, float(raw_pressure_ratio) - 1.0)
    capacity_gap = max(0.0, 1.0 - capacity_ratio)
    near_clear_gap = 0.0
    if pressure >= 0.90 or capacity_ratio < 1.12:
        near_clear_gap = max(0.0, 1.12 - capacity_ratio) * 9.0

    penalty = (
        (pressure_gap * 82.0)
        + (raw_gap * 46.0)
        + (capacity_gap * 74.0)
        + near_clear_gap
    )
    unresolved_gain = 1.0 - gain_credit
    penalty *= unresolved_gain

    floor_shortfall = _safe_money_shortfall(state)
    if floor_shortfall > 0:
        if pressure >= 1.0 or capacity_ratio < 1.0:
            floor_weight = 1.9 + min(1.0, max(0, state.ante - 2) * 0.25)
            penalty += floor_shortfall * floor_weight * unresolved_gain
        elif pressure >= 0.90 or capacity_ratio < 1.12:
            penalty += floor_shortfall * 0.85 * unresolved_gain

    if pressure < 0.85 and capacity_ratio >= 1.20:
        penalty *= 0.25
    elif pressure < 0.95 and capacity_ratio >= 1.05:
        penalty *= 0.55
    if state.ante <= 1:
        penalty *= 0.70
    return min(260.0, max(0.0, penalty))


def _safe_money_shortfall(state: GameState, *, after_spend: int | None = None) -> int:
    money = state.money if after_spend is None else after_spend
    return max(0, _minimum_safe_shop_money(state) - money - _economy_floor_credit(state))


def _clear_first_speculative_spend_penalty(state: GameState, action: Action) -> float:
    """Charge value-only spending until the shop line passes the clear bar."""

    spend_kind = _speculative_spend_kind(state, action)
    if spend_kind is None:
        return 0.0
    cost = _shop_action_cost(state, action)
    if cost <= 0:
        return 0.0

    pressure_ratio, raw_pressure_ratio, capacity_ratio = _shop_pressure_metrics(state)
    pressure = max(pressure_ratio, raw_pressure_ratio)
    after_money = state.money - cost
    if after_money < 0:
        return 1000.0

    current_shortfall = _safe_money_shortfall(state)
    after_shortfall = _safe_money_shortfall(state, after_spend=after_money)
    new_shortfall = max(0, after_shortfall - current_shortfall)
    clear_unsettled = pressure >= 0.92 or raw_pressure_ratio >= 0.96 or capacity_ratio < 1.12
    if not clear_unsettled and new_shortfall <= 0:
        return 0.0

    pressure_weight = max(0.0, pressure - 0.88) * 1.35
    capacity_weight = max(0.0, 1.12 - capacity_ratio) * 0.85
    floor_weight = 3.1 + min(1.0, max(0, state.ante - 2) * 0.25)
    penalty = (cost * (0.55 + pressure_weight + capacity_weight)) + (new_shortfall * floor_weight)
    if after_money < 5 and state.ante >= 2:
        penalty += (5 - after_money) * 3.0

    if clear_unsettled and capacity_ratio < 1.0:
        capacity_shortfall = 1.0 - capacity_ratio
        if spend_kind == "pack":
            penalty += capacity_shortfall * 105.0
        elif spend_kind == "voucher":
            penalty += capacity_shortfall * 55.0
        elif spend_kind == "consumable":
            penalty += capacity_shortfall * 35.0

    if spend_kind == "pack":
        penalty *= 1.25
        pack = _pack_item_for_action(state, action)
        pack_label = _item_label(pack).lower() if pack is not None else ""
        if "buffoon" in pack_label and _normal_joker_open_slots(state) > 0 and capacity_ratio < 1.0:
            penalty *= 0.55
    elif spend_kind == "voucher":
        penalty *= 1.10
    elif spend_kind == "consumable":
        penalty *= 0.82
    if state.ante <= 1:
        penalty *= 0.65
    return min(150.0, max(0.0, penalty))


def _speculative_spend_kind(state: GameState, action: Action) -> str | None:
    if action.action_type == ActionType.OPEN_PACK:
        return "pack"
    if action.action_type != ActionType.BUY:
        return None
    kind = _action_kind(action, default="card")
    if kind == "voucher":
        return "voucher"
    item = _buy_action_item(state, action)
    if item is None or _is_joker_item(item):
        return None
    if _is_consumable_item(item):
        return "consumable"
    return "speculative"


def _shop_action_cost(state: GameState, action: Action) -> int:
    if action.action_type == ActionType.OPEN_PACK:
        pack = _pack_item_for_action(state, action)
        return _item_cost(state, pack)
    if action.action_type == ActionType.BUY:
        item = _buy_action_item(state, action)
        return _item_cost(state, item)
    return 0


def _shop_money_value(state: GameState) -> float:
    """Value cash as both spending power and next-cash-out interest."""

    money = max(0, state.money)
    try:
        from balatro_ai.bots.basic_strategy.shop_money import _desired_money_reserve, _interest_cap_money
        from balatro_ai.bots.basic_strategy.shop_pressure import _shop_pressure

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


def _shop_consumable_inventory_value(state: GameState) -> float:
    if not state.consumables:
        return 0.0
    total = sum(_held_consumable_leaf_value(state, name) for name in state.consumables)
    limit = _consumable_slot_limit(state)
    if limit > 0 and len(state.consumables) >= limit:
        total -= 1.5
    return max(0.0, total)


def _held_consumable_leaf_value(state: GameState, name: str) -> float:
    if name in PLANET_TO_HAND:
        value = 4.0
        try:
            from balatro_ai.bots.basic_strategy.hand_preferences import _preferred_hand_type

            if _preferred_hand_type(state) == PLANET_TO_HAND[name]:
                value += 3.0
        except (ImportError, TypeError, ValueError, AttributeError):
            pass
        return value
    values = {
        "Justice": 8.0,
        "Death": 8.0,
        "The Hanged Man": 7.0,
        "Strength": 6.5,
        "The Chariot": 6.5,
        "The Hermit": 8.5,
        "Temperance": 5.5,
        "The Empress": 5.5,
        "The Magician": 5.0,
        "The Hierophant": 4.8,
        "The Devil": 4.8,
        "The Lovers": 4.5,
        "The Tower": 4.5,
        "The Star": 4.5,
        "The Moon": 4.5,
        "The Sun": 4.5,
        "The World": 4.5,
        "The Fool": 4.0,
        "The High Priestess": 4.0,
        "The Wheel of Fortune": 3.8,
        "Judgement": 3.8,
        "The Emperor": 3.5,
    }
    return values.get(name, 3.0)


def _shop_money_floor_penalty(state: GameState) -> float:
    """Make ending a shop broke visibly worse than merely losing interest."""

    floor = _minimum_safe_shop_money(state)
    try:
        from balatro_ai.bots.basic_strategy.shop_money import _desired_money_reserve
        from balatro_ai.bots.basic_strategy.shop_pressure import _shop_pressure
        from balatro_ai.bots.basic_strategy.shop_safety import _has_money_scaling_joker

        pressure = _shop_pressure(state)
        if pressure.raw_ratio >= 1.25:
            floor = max(5, floor - 10)
        elif pressure.raw_ratio >= 1.05:
            floor = max(5, floor - 5)
        elif pressure.ratio <= 0.75:
            floor = min(_shop_interest_cap_money(state), floor + 5)
        if _has_money_scaling_joker(state):
            floor = max(
                floor,
                _interest_breakpoint_target(
                    _desired_money_reserve(state, pressure),
                    maximum=max(35, _shop_interest_cap_money(state)),
                ),
            )
    except (ImportError, TypeError, ValueError, AttributeError):
        pass

    shortfall = max(0, floor - state.money - _economy_floor_credit(state))
    if shortfall <= 0:
        return 0.0
    weight = 2.6 + min(1.4, max(0, state.ante - 2) * 0.35)
    penalty = shortfall * weight
    if state.ante >= 2 and state.money < 5:
        penalty += (5 - state.money) * 3.0
    return penalty


_ECONOMY_FLOOR_CREDITS = {
    "Rocket": 14,
    "Golden Joker": 12,
    "To the Moon": 12,
    "Hallucination": 10,
    "Trading Card": 10,
    "Golden Ticket": 9,
    "Mail-In Rebate": 9,
    "Business Card": 8,
    "Cloud 9": 8,
    "Reserved Parking": 7,
    "Delayed Gratification": 6,
    "Satellite": 6,
    "Faceless Joker": 5,
}


def _economy_floor_credit(state: GameState) -> int:
    try:
        from balatro_ai.bots.basic_strategy.shop_pressure import _shop_pressure

        pressure = _shop_pressure(state)
        if pressure.raw_ratio >= 1.0 or pressure.ratio >= 0.95:
            return 0
        pressure_scale = 1.0 if pressure.ratio <= 0.75 else 0.65
    except (ImportError, TypeError, ValueError, AttributeError):
        pressure_scale = 0.65
    credit = sum(_ECONOMY_FLOOR_CREDITS.get(joker.name, 0) for joker in state.jokers)
    if credit <= 0:
        return 0
    return int(min(max(0, _minimum_safe_shop_money(state) - 5), credit * pressure_scale))


def _minimum_safe_shop_money(state: GameState) -> int:
    ante = max(1, state.ante)
    if ante <= 1:
        base_floor = 10
    elif ante == 2:
        base_floor = 20
    else:
        base_floor = 25

    interest_cap = _shop_interest_cap_money(state)
    if interest_cap <= 25 or ante <= 3:
        return min(base_floor, interest_cap)

    late_ante_steps = ante - 3
    voucher_step = max(5, ((interest_cap - 25 + 3) // 4))
    voucher_floor = 25 + (late_ante_steps * voucher_step)
    return min(interest_cap, max(base_floor, _interest_breakpoint_target(voucher_floor, maximum=interest_cap)))


def _shop_interest_cap_money(state: GameState) -> int:
    cap = 25
    modifier_cap = state.modifiers.get("interest_cap_money", state.modifiers.get("interest_cap"))
    if isinstance(modifier_cap, int | float):
        cap = max(cap, int(modifier_cap))
    if "Seed Money" in state.vouchers:
        cap = max(cap, 50)
    if "Money Tree" in state.vouchers:
        cap = max(cap, 100)
    return max(0, cap)


def _interest_breakpoint_target(value: int, *, maximum: int) -> int:
    clamped = max(5, min(maximum, int(value)))
    return min(maximum, ((clamped + 4) // 5) * 5)


def _spend_opportunity_penalty(state: GameState, cost: int) -> float:
    if cost <= 0:
        return 0.0
    try:
        from balatro_ai.bots.basic_strategy.shop_money import _money_after_spend_penalty
        from balatro_ai.bots.basic_strategy.shop_pressure import _shop_pressure

        return _money_after_spend_penalty(state, cost, _shop_pressure(state))
    except (ImportError, TypeError, ValueError, AttributeError):
        after = state.money - cost
        if after < 0:
            return 1000.0
        interest_cap = _shop_interest_cap_money(state)
        before_interest = min(max(0, state.money), interest_cap) // 5
        after_interest = min(max(0, after), interest_cap) // 5
        return max(0, before_interest - after_interest) * 4.0


def _unresolved_pressure_buy_penalty(state: GameState, action: Action) -> float:
    """Penalize joker buys that leave a pressured shop broke without raising scoring capacity."""

    if action.action_type != ActionType.BUY:
        return 0.0
    item = _buy_action_item(state, action)
    if item is None or not _is_joker_item(item):
        return 0.0

    try:
        next_state = simulate_buy(state, action)
    except (TypeError, ValueError, IndexError, AttributeError):
        return 0.0

    build_before = _shop_build_score(state)
    build_after = _shop_build_score(next_state)
    build_gain = max(0.0, build_after - build_before)
    pressure_ratio, raw_pressure_ratio, capacity_ratio = _shop_pressure_metrics(next_state)
    pressure = max(pressure_ratio, raw_pressure_ratio)
    if pressure < 1.35 and capacity_ratio >= 0.75:
        return 0.0

    meaningful_gain = max(45.0, build_before * 0.35)
    unresolved = 1.0 - min(1.0, build_gain / meaningful_gain)
    if unresolved <= 0.0:
        return 0.0

    floor_shortfall = max(0, _minimum_safe_shop_money(next_state) - next_state.money)
    pressure_gap = max(0.0, pressure - 1.15)
    capacity_gap = max(0.0, 0.85 - capacity_ratio)
    penalty = (pressure_gap * 32.0) + (capacity_gap * 60.0) + (floor_shortfall * 5.0)
    penalty = min(170.0, penalty) * unresolved
    if state.ante <= 1:
        penalty *= 0.75
    return penalty


def _expand_action(
    node: _BeamNode,
    action: Action,
    sampler: ShopSampler,
    *,
    rng: Random,
    action_value_fn: ActionValueFn,
    leaf_value_fn: LeafValueFn,
    leaf_terms_fn: LeafTermsFn | None,
    leaf_weight: float,
    action_potential_weight: float,
) -> _BeamNode:
    first_action = node.first_action or action
    path = node.path + (action,)
    immediate = action_value_fn(node.state, action)
    terminal = action.action_type in {ActionType.END_SHOP, ActionType.REROLL, ActionType.OPEN_PACK} or _is_overstock_voucher_buy(
        node.state,
        action,
    )
    try:
        next_state = _simulate_shop_action(node.state, action, sampler, rng=rng)
    except (ValueError, IndexError, TypeError):
        return _BeamNode(
            state=node.state,
            first_action=first_action,
            path=path,
            action_score=node.action_score,
            leaf_score=0.0,
            score=float("-inf"),
            leaf_terms=None,
            terminal=True,
        )
    leaf_terms = leaf_terms_fn(next_state) if leaf_terms_fn is not None else None
    leaf_score = leaf_terms.total if leaf_terms is not None else leaf_value_fn(next_state)
    immediate += _discount_voucher_ordering_bonus(node.state, action, next_state)
    immediate += _action_potential_shaping(
        node,
        action,
        leaf_score,
        leaf_value_fn=leaf_value_fn,
        leaf_terms_fn=leaf_terms_fn,
        weight=action_potential_weight,
    )
    action_score = node.action_score + immediate
    score = action_score + (leaf_score * leaf_weight)
    protected_jokers = node.protected_jokers
    bought_joker = _bought_joker_name(node.state, action)
    if bought_joker is not None:
        protected_jokers = (*protected_jokers, bought_joker)
    return _BeamNode(
        state=next_state,
        first_action=first_action,
        path=path,
        action_score=action_score,
        leaf_score=leaf_score,
        score=score,
        leaf_terms=leaf_terms,
        terminal=terminal,
        protected_jokers=protected_jokers,
    )


def _discount_voucher_ordering_bonus(state: GameState, action: Action, next_state: GameState) -> float:
    if action.action_type != ActionType.BUY or _action_kind(action, default="") != "voucher":
        return 0.0
    voucher = _buy_action_item(state, action)
    if _item_label(voucher) not in {"Clearance Sale", "Liquidation"}:
        return 0.0

    savings = 0
    for key in ("shop_cards", "booster_packs"):
        before_items = _modifier_items(state.modifiers, key)
        after_items = _modifier_items(next_state.modifiers, key)
        for before_item, after_item in zip(before_items, after_items, strict=False):
            savings += max(0, _item_cost(state, before_item) - _item_cost(next_state, after_item))
    if savings <= 0:
        return 0.0
    return min(36.0, savings * 24.0)


def _action_potential_shaping(
    node: _BeamNode,
    action: Action,
    next_leaf_score: float,
    *,
    leaf_value_fn: LeafValueFn,
    leaf_terms_fn: LeafTermsFn | None,
    weight: float,
) -> float:
    if weight <= 0.0 or action.action_type not in {ActionType.BUY, ActionType.SELL, ActionType.USE_CONSUMABLE}:
        return 0.0
    if node.path:
        current_leaf_score = node.leaf_score
    else:
        current_terms = leaf_terms_fn(node.state) if leaf_terms_fn is not None else None
        current_leaf_score = current_terms.total if current_terms is not None else leaf_value_fn(node.state)
    return (next_leaf_score - current_leaf_score) * weight


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
        return simulate_end_shop(state, created_consumables=_perkeo_consumables(state, rng))
    if action.action_type == ActionType.OPEN_PACK:
        pack = _pack_item_for_action(state, action)
        contents = sampler.sample_pack_contents(state, pack, rng) if isinstance(pack, dict) else ()
        return simulate_open_pack(
            state,
            action,
            contents,
            created_consumables=_hallucination_consumables(state, sampler=sampler, rng=rng),
        )
    if action.action_type == ActionType.USE_CONSUMABLE:
        return simulate_consumable_action(state, action, sampler=sampler, rng=rng)
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
) -> tuple[Action, ...]:
    if root_actions:
        def root_action_allowed(action: Action) -> bool:
            if not _supported_root_action(action) or _sells_protected_joker(state, action, protected_jokers):
                return False
            if action.action_type == ActionType.USE_CONSUMABLE:
                return shop_consumable_action_is_candidate(state, action)
            if action.action_type != ActionType.SELL:
                return True
            if _action_kind(action, default="joker") == "consumable":
                index = _action_index(action)
                return index is not None and _consumable_sell_is_search_candidate(state, index)
            index = _action_index(action)
            return index is not None and _sell_is_search_candidate(state, index)

        return tuple(
            action
            for action in root_actions
            if root_action_allowed(action)
        )

    actions: list[Action] = []
    actions.extend(shop_consumable_actions(state))
    for index in range(len(state.consumables)):
        if _consumable_sell_is_search_candidate(state, index):
            actions.append(Action(ActionType.SELL, target_id="consumable", amount=index, metadata={"kind": "consumable", "index": index}))

    for index in range(len(state.jokers)):
        action = Action(ActionType.SELL, target_id="joker", amount=index, metadata={"kind": "joker", "index": index})
        if not _sells_protected_joker(state, action, protected_jokers) and _sell_is_search_candidate(state, index):
            actions.append(action)

    for index, item in enumerate(_modifier_items(state.modifiers, "shop_cards")):
        if not _can_buy_item(state, item):
            continue
        if _shop_card_can_be_bought(state, item):
            actions.append(Action(ActionType.BUY, target_id="card", amount=index, metadata={"kind": "card", "index": index}))
        elif _shop_card_can_be_bought_and_used(state, item):
            actions.append(
                Action(
                    ActionType.BUY,
                    target_id="card",
                    amount=index,
                    metadata={"kind": "card", "index": index, "buy_and_use": True},
                )
            )

    for index, item in enumerate(_modifier_items(state.modifiers, "voucher_cards")):
        if _can_buy_item(state, item):
            actions.append(Action(ActionType.BUY, target_id="voucher", amount=index, metadata={"kind": "voucher", "index": index}))

    for index, item in enumerate(_modifier_items(state.modifiers, "booster_packs")):
        if _can_buy_item(state, item):
            actions.append(Action(ActionType.OPEN_PACK, target_id="pack", amount=index, metadata={"kind": "pack", "index": index}))

    if _can_afford_cost(state, sampler.reroll_cost(state)):
        actions.append(Action(ActionType.REROLL))
    actions.append(Action(ActionType.END_SHOP))
    return tuple(actions)


def _supported_root_action(action: Action) -> bool:
    return action.action_type in {
        ActionType.BUY,
        ActionType.SELL,
        ActionType.REROLL,
        ActionType.END_SHOP,
        ActionType.OPEN_PACK,
        ActionType.USE_CONSUMABLE,
    }


def _sells_protected_joker(state: GameState, action: Action, protected_jokers: tuple[str, ...]) -> bool:
    if action.action_type != ActionType.SELL or not protected_jokers:
        return False
    if _action_kind(action, default="joker") != "joker":
        return False
    index = _action_index(action)
    if index is None or not 0 <= index < len(state.jokers):
        return False
    return state.jokers[index].name in protected_jokers


def _consumable_sell_search_value(state: GameState, action: Action) -> float:
    index = _action_index(action)
    if index is None or not 0 <= index < len(state.consumables):
        return 0.0
    try:
        sold_state = simulate_sell(state, action)
    except (ValueError, IndexError, TypeError, AttributeError):
        return 0.0
    gain = sold_state.money - state.money
    return float(gain) - (_held_consumable_value(state.consumables[index]) * 0.35)


def _consumable_sell_is_search_candidate(state: GameState, index: int) -> bool:
    if not 0 <= index < len(state.consumables):
        return False
    if _consumable_open_slots(state) > 0:
        return False
    remaining = tuple(name for item_index, name in enumerate(state.consumables) if item_index != index)
    if any(_consumable_creates_consumables(name) for name in remaining):
        return True
    if any(_is_consumable_item(item) and _can_buy_item(state, item) for item in _modifier_items(state.modifiers, "shop_cards")):
        return True
    try:
        sold_state = simulate_sell(state, Action(ActionType.SELL, target_id="consumable", amount=index, metadata={"kind": "consumable", "index": index}))
    except (ValueError, IndexError, TypeError, AttributeError):
        return False
    gain = sold_state.money - state.money
    return gain > 1 and state.money < _minimum_safe_shop_money(state)


def _consumable_creates_consumables(name: str) -> bool:
    return name in {"The Fool", "The Emperor", "The High Priestess"}


def _held_consumable_value(name: str) -> float:
    if name in {"The Hermit", "Temperance"}:
        return 8.0
    if name in {"Death", "The Hanged Man", "The Emperor", "The High Priestess", "The Fool"}:
        return 6.0
    if name in {"Black Hole", "The Soul"}:
        return 12.0
    if name in {"Wheel of Fortune", "The Wheel of Fortune"}:
        return 4.0
    return 3.0


def _sell_is_search_candidate(state: GameState, index: int) -> bool:
    if not 0 <= index < len(state.jokers):
        return False
    sell_value = state.jokers[index].sell_value or 0
    return sell_value > 0


def _can_buy_item(state: GameState, item: object) -> bool:
    return _can_afford_cost(state, _item_cost(state, item))


def _can_afford_cost(state: GameState, cost: int) -> bool:
    return state.money - max(0, cost) >= _bankrupt_at(state)


def _bankrupt_at(state: GameState) -> int:
    try:
        return min(0, int(state.modifiers.get("bankrupt_at", 0)))
    except (TypeError, ValueError):
        return 0


def _buy_action_item(state: GameState, action: Action) -> object | None:
    kind = str(action.metadata.get("kind", action.target_id or ""))
    index = _action_index(action)
    if index is None:
        return None
    if kind == "card":
        items = _modifier_items(state.modifiers, "shop_cards")
    elif kind == "voucher":
        items = _modifier_items(state.modifiers, "voucher_cards")
    else:
        return None
    if not 0 <= index < len(items):
        return None
    return items[index]


def _shop_card_can_be_bought(state: GameState, item: object) -> bool:
    if not _is_joker_item(item):
        return not _is_consumable_item(item) or _consumable_open_slots(state) > 0
    if not _joker_item_uses_normal_slot(item):
        return True
    return _normal_joker_open_slots(state) > 0


def _shop_card_can_be_bought_and_used(state: GameState, item: object) -> bool:
    if not _is_consumable_item(item):
        return False
    name = _item_label(item)
    if name in PLANET_TO_HAND or name in {"The Hermit", "Temperance", "Black Hole"}:
        return True
    if name == "The Wheel of Fortune":
        return bool(state.jokers)
    if name in {"Judgement", "The Soul", "Wraith"}:
        return _normal_joker_open_slots(state) > 0
    if name == "Ankh":
        return bool(state.jokers) and _normal_joker_slot_limit(state) > 1
    if name in {"Hex", "Ectoplasm"}:
        return any(joker.edition is None for joker in state.jokers)
    return False


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


def _pack_item_for_action(state: GameState, action: Action) -> object | None:
    if action.action_type != ActionType.OPEN_PACK:
        return None
    index = _action_index(action)
    packs = _modifier_items(state.modifiers, "booster_packs")
    if index is None or not 0 <= index < len(packs):
        return None
    return packs[index]


def _hallucination_consumables(state: GameState, *, sampler: ShopSampler, rng: Random) -> tuple[object, ...]:
    room = _consumable_open_slots(state)
    if room <= 0:
        return ()
    probability_multiplier = _probability_multiplier(state)
    count = sum(
        1
        for _ in range(_active_joker_count(state, "Hallucination"))
        if _roll_odds(rng, 2, probability_multiplier=probability_multiplier)
    )
    count = min(room, count)
    used_consumables = set(state.consumables)
    created: list[object] = []
    for _ in range(count):
        card = sampler.sample_card_of_type(state, "Tarot", rng, used_consumables=used_consumables)
        created.append(card)
        used_consumables.update(_sampled_item_identifiers(card))
    return tuple(created)


def _sampled_item_identifiers(item: object) -> set[str]:
    if not isinstance(item, Mapping):
        return {str(item)}
    identifiers = {str(item.get("label", item.get("name", item.get("key", ""))))}
    key = str(item.get("key", ""))
    if key:
        identifiers.add(key)
    return {identifier for identifier in identifiers if identifier}


def _perkeo_consumables(state: GameState, rng: Random) -> tuple[str, ...]:
    if not state.consumables:
        return ()
    count = _active_joker_count(state, "Perkeo")
    return tuple(rng.choice(state.consumables) for _ in range(count))


def _active_joker_count(state: GameState, name: str) -> int:
    return sum(1 for joker in state.jokers if joker.name == name and not _joker_is_disabled(joker))


def _joker_is_disabled(joker: object) -> bool:
    effect = getattr(joker, "effect", None)
    disabled = getattr(effect, "disabled", None)
    if isinstance(disabled, bool):
        return disabled
    metadata = getattr(joker, "metadata", {})
    if not isinstance(metadata, dict):
        return False
    value = metadata.get("value")
    effect = value.get("effect", "") if isinstance(value, dict) else metadata.get("effect", "")
    return "all abilities are disabled" in str(effect).lower()


def _roll_odds(rng: Random, odds: int | float, *, probability_multiplier: float) -> bool:
    if odds <= 0:
        return True
    return rng.random() < min(1.0, probability_multiplier / float(odds))


def _probability_multiplier(state: GameState) -> float:
    try:
        return float(state.modifiers.get("probability_multiplier", 1.0))
    except (TypeError, ValueError):
        return 1.0


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


def _action_kind(action: Action, *, default: str) -> str:
    return str(action.metadata.get("kind", action.target_id or default))


def _int_value(raw: object) -> int:
    try:
        return int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _annotated_action(
    action: Action,
    *,
    search_value: float,
    path: tuple[Action, ...],
    snapshot: dict[str, Any] | None = None,
    candidates: tuple[dict[str, Any], ...] = (),
) -> Action:
    metadata = {
        **action.metadata,
        "search": "shop_beam",
        "search_value": round(search_value, 6),
        "search_path": tuple(_action_summary(item) for item in path),
    }
    if snapshot is not None:
        metadata["search_shop_snapshot"] = snapshot
    if candidates:
        metadata["search_candidates"] = candidates
    return Action(
        action.action_type,
        card_indices=action.card_indices,
        target_id=action.target_id,
        amount=action.amount,
        metadata=metadata,
    )


def _trace_candidates(nodes: list[_BeamNode], *, limit: int) -> tuple[dict[str, Any], ...]:
    if limit <= 0:
        return ()
    return tuple(_node_trace_summary(node) for node in sorted(nodes, key=lambda item: item.score, reverse=True)[:limit])


def _node_trace_summary(node: _BeamNode) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "score": round(node.score, 6),
        "action_score": round(node.action_score, 6),
        "leaf_score": round(node.leaf_score, 6),
        "terminal": node.terminal,
        "path": tuple(_action_summary(action) for action in node.path),
        "result": {
            "money": node.state.money,
            "jokers": tuple(joker.name for joker in node.state.jokers),
            "consumables": tuple(str(item) for item in node.state.consumables),
        },
    }
    if node.leaf_terms is not None:
        summary["leaf_terms"] = node.leaf_terms.to_trace_dict()
    return summary


def _shop_trace_snapshot(state: GameState) -> dict[str, Any]:
    return {
        "ante": state.ante,
        "blind": state.blind,
        "money": state.money,
        "jokers": tuple(joker.name for joker in state.jokers),
        "shop_cards": tuple(_item_label(item) for item in _modifier_items(state.modifiers, "shop_cards")),
        "booster_packs": tuple(_item_label(item) for item in _modifier_items(state.modifiers, "booster_packs")),
        "voucher_cards": tuple(_item_label(item) for item in _modifier_items(state.modifiers, "voucher_cards")),
    }


def _action_summary(action: Action) -> dict[str, Any]:
    return {
        "type": action.action_type.value,
        "kind": str(action.metadata.get("kind", action.target_id or "")),
        "index": action.metadata.get("index", action.amount),
    }
