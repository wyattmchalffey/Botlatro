"""Composed solver policy (Milestone M5).

`SolverPolicy` is the solver's full policy across game phases. It dispatches
each `GameState` to whichever sub-policy handles that phase:

- `SELECTING_HAND` -> `SearchV2PlayPolicy` by default, with the legacy M4
  `PlaySearchPolicy` still available through `play_backend="legacy"`.
- `SHOP` -> shop beam search via existing `search/shop_search.py`.
- Everything else (BLIND_SELECT, BOOSTER_OPENED, ROUND_EVAL, ...) ->
  fallback callable, defaulting to `basic_strategy_bot.choose_action`.

The search_v2 play path is the first Tier 1 optimization from
`SOLVER_OPTIMIZATION_PLAN.md`: keep the solver's whole-blind planning shape,
but remove the live-bot preprocessing that dominated play-decision runtime.

Status: M5 plus Tier 1 search_v2 work from `SOLVER_OPTIMIZATION_PLAN.md`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GamePhase, GameState
from balatro_ai.bots.basic_strategy.cards import _normal_joker_open_slots
from balatro_ai.bots.basic_strategy_bot import BasicStrategyBot
from balatro_ai.search.shop_sampler import ShopSampler
from balatro_ai.search.shop_search import (
    ShopSearchConfig,
    ShopSearchContext,
    best_shop_action,
    shop_leaf_terms,
)
from balatro_ai.solver.archetypes import Archetype
from balatro_ai.solver.play_search import PlaySearchPolicy
from balatro_ai.solver.search_v2.leaf_value import (
    ArchetypeAwareLeaf,
    PlanningValueLeaf,
)
from balatro_ai.solver.search_v2.play import SearchV2PlayPolicy


# Defaults tuned for the M5.5 throughput pass.
#
# `search_bot_v0`'s shop defaults are width=8 / depth=3 / reroll_samples=32.
# After profiling we drop:
#  - reroll_samples 32 -> 8: shop sampling for variance reduction, 8 is
#    plenty when the beam re-evaluates leaves at each ply.
#  - beam_width 8 -> 4: halves the leaf count per ply. The shop tree is
#    narrow (BUY/SELL/REROLL/OPEN_PACK/END_SHOP) so width 4 still keeps
#    enough alternatives in flight.
#  - depth 3 -> 2: shop chains of 3 actions are rare; depth 2 covers
#    BUY+END or BUY+BUY+END which is the common case.
#
# These three knobs compound — projected per-shop-call savings ~3-5x.
# 2026-06-08: further trimmed reroll_samples 8->4 and beam_width 4->3 after a 160-seed
# A/B (rust on, ECON_W=1.5) held winrate (35 vs 34) at ~1.4x faster shop (758s->543s);
# mean ante dipped slightly (6.60->6.44, within noise). Env overrides:
# BALATRO_SOLVER_SHOP_{BEAM_WIDTH,DEPTH,REROLL_SAMPLES}.
DEFAULT_SHOP_BEAM_WIDTH = 3
DEFAULT_SHOP_DEPTH = 2
DEFAULT_SHOP_REROLL_SAMPLES = 4
# Default reverted to "legacy" after 4-seed measurement (2026-05-26)
# showed v2 averages ante 2.0 vs legacy ante 4.5 across canonical
# seeds. v2 is ~4x faster (32s vs 129s per seed effective on 4
# cores) but the quality hit isn't worth shipping by default. v2
# stays available for deep-search experiments and as the iteration
# target for Tier 2 work (Cython port of evaluate_played_cards
# would close most of the gap by letting v2 use samples=16 leaves
# affordably). See SOLVER_OPTIMIZATION_PLAN.md "Honest assessment
# of Tier 1 speed gains" for the full measurement.
DEFAULT_PLAY_BACKEND = "legacy"
DEFAULT_V2_PLAY_DEPTH = 3
DEFAULT_V2_PLAY_WIDTH = 2


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


@dataclass(slots=True)
class SolverPolicy:
    """End-to-end solver policy across all game phases.

    Parameters
    ----------
    play_policy: handles SELECTING_HAND. Default: a fresh search_v2 play
                policy unless `play_backend="legacy"` is selected.
    play_backend: either "v2" for the solver-specific beam or "legacy" for
                the wrapped M4 `PlaySearchPolicy`.
    shop_config: ShopSearchConfig used at SHOP phase.
    fallback:   any `GameState -> Action` callable used for phases neither
                play nor shop search owns. Default: `BasicStrategyBot`.
    seed:       passed to all sub-components for reproducibility (where
                they accept it).
    """

    play_policy: object | None = None
    play_backend: str = DEFAULT_PLAY_BACKEND
    play_depth: int = DEFAULT_V2_PLAY_DEPTH
    play_width: int = DEFAULT_V2_PLAY_WIDTH
    shop_config: ShopSearchConfig | None = None
    fallback: object | None = None
    seed: int = 0
    archetype: Archetype | None = None
    allow_shop_sells: bool = True
    prefer_fallback_info_first_shop: bool = False
    fallback_negative_shop_sells: bool = False
    prefer_opening_buffoon_pack: bool = False
    fallback_opening_buffoon_over_non_joker_buy: bool = False
    fallback_unfunded_open_slot_sells: bool = False
    fallback_negative_end_shop: bool = False
    fallback_ante1_single_joker_planet_buy: bool = False
    fallback_negative_shop_actions: bool = False
    # Optional injected shop-leaf value factory: (root_state) -> (leaf_state -> float).
    # When set it replaces the heuristic shop leaf for the SHOP beam — the economy-side
    # analog of `play_policy=` for plays — so a learned value head can be A/B'd as the
    # shop evaluator with no other change. Takes precedence over the archetype leaf.
    shop_leaf_value_fn: object | None = None
    _sampler: ShopSampler = field(init=False, repr=False)
    _shop_key: tuple[int | None, int, str] | None = field(default=None, init=False, repr=False)
    _protected_shop_jokers: tuple[str, ...] = field(default=(), init=False, repr=False)
    _rerolls_in_shop: int = field(default=0, init=False, repr=False)
    _packs_opened_in_shop: int = field(default=0, init=False, repr=False)
    _filled_last_joker_slot_in_shop: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.fallback is None:
            self.fallback = BasicStrategyBot(seed=self.seed)
        if self.play_policy is None:
            self.play_policy = self._build_default_play_policy()
        if self.shop_config is None:
            self.shop_config = ShopSearchConfig(
                beam_width=_env_int("BALATRO_SOLVER_SHOP_BEAM_WIDTH", DEFAULT_SHOP_BEAM_WIDTH),
                depth=_env_int("BALATRO_SOLVER_SHOP_DEPTH", DEFAULT_SHOP_DEPTH),
                reroll_samples=_env_int("BALATRO_SOLVER_SHOP_REROLL_SAMPLES", DEFAULT_SHOP_REROLL_SAMPLES),
                seed=self.seed,
                min_search_value=_env_float("BALATRO_SOLVER_SHOP_MIN_SEARCH_VALUE", float("-inf")),
            )
        self._sampler = ShopSampler.from_default_data()

    def __call__(self, state: GameState) -> Action:
        return self.choose_action(state)

    def choose_action(self, state: GameState) -> Action:
        """Dispatch by game phase to the right sub-policy."""

        # Wrap the per-decision work in a basic_strategy decision cache
        # scope so identity-keyed sub-results (joker freezes, card
        # freezes, sample-build keys) dedupe across the many helpers
        # invoked during one decision. Without this scope, every
        # `_identity_cached_value` falls through to its factory ⇒
        # `_freeze_for_cache` runs millions of times per trajectory.
        # The other bots (basic_strategy_bot, search_bot_v2) already
        # use this pattern.
        from balatro_ai.bots.basic_strategy.cache import decision_cache_scope
        with decision_cache_scope():
            self._sync_shop_memory(state)
            if state.phase == GamePhase.SELECTING_HAND:
                return self.play_policy.choose_action(state)

            if state.phase == GamePhase.SHOP and _has_shop_action(state):
                if self.prefer_opening_buffoon_pack:
                    buffoon_action = self._opening_buffoon_pack_action(state)
                    if buffoon_action is not None:
                        self._record_shop_action(state, buffoon_action)
                        self._record_fallback_shop_action(state, buffoon_action)
                        return buffoon_action
                if self.prefer_fallback_info_first_shop:
                    fallback_action = self._fallback_info_first_shop_action(state)
                    if fallback_action is not None:
                        self._record_shop_action(state, fallback_action)
                        self._record_fallback_shop_action(state, fallback_action)
                        return fallback_action
                try:
                    if self.shop_leaf_value_fn is not None:
                        leaf_value_fn = self.shop_leaf_value_fn(state)
                    elif self.archetype is not None:
                        leaf_value_fn = self._archetype_leaf_value_fn(state)
                    else:
                        leaf_value_fn = None
                    action = best_shop_action(
                        state,
                        config=self.shop_config,
                        sampler=self._sampler,
                        protected_jokers=self._protected_shop_jokers,
                        shop_context=ShopSearchContext(
                            rerolls_in_shop=self._rerolls_in_shop,
                            packs_opened_in_shop=self._packs_opened_in_shop,
                            filled_last_joker_slot=self._filled_last_joker_slot_in_shop,
                        ),
                        leaf_value_fn=leaf_value_fn,
                    )
                except (ValueError, IndexError, KeyError, TypeError, AttributeError):
                    # Shop search occasionally errors on unusual states
                    # (consumable-slot-full, voucher legality, etc.). Fall
                    # back rather than crash the trajectory.
                    action = None
                if action is not None:
                    if (
                        self.fallback_opening_buffoon_over_non_joker_buy
                        and _action_is_non_joker_shop_card_buy(state, action)
                    ):
                        buffoon_action = self._opening_buffoon_pack_action(state)
                        if buffoon_action is not None:
                            self._record_shop_action(state, buffoon_action)
                            self._record_fallback_shop_action(state, buffoon_action)
                            return buffoon_action
                    if action.action_type == ActionType.SELL and not self.allow_shop_sells:
                        fallback_action = self.fallback.choose_action(state)
                        self._record_shop_action(state, fallback_action)
                        return fallback_action
                    if (
                        action.action_type == ActionType.SELL
                        and self.fallback_unfunded_open_slot_sells
                        and _normal_joker_open_slots(state) > 0
                        and not _sell_funds_planned_buy(state, action)
                    ):
                        fallback_action = self.fallback.choose_action(state)
                        self._record_shop_action(state, fallback_action)
                        return fallback_action
                    if (
                        action.action_type == ActionType.SELL
                        and self.fallback_negative_shop_sells
                        and _action_search_value(action) < 0.0
                        and not _sell_funds_planned_buy(state, action)
                    ):
                        fallback_action = self.fallback.choose_action(state)
                        self._record_shop_action(state, fallback_action)
                        return fallback_action
                    if (
                        action.action_type == ActionType.END_SHOP
                        and self.fallback_negative_end_shop
                        and _action_search_value(action) < 0.0
                    ):
                        fallback_action = self.fallback.choose_action(state)
                        if fallback_action.action_type != ActionType.END_SHOP:
                            self._record_shop_action(state, fallback_action)
                            self._record_fallback_shop_action(state, fallback_action)
                            return fallback_action
                    if (
                        self.fallback_ante1_single_joker_planet_buy
                        and _action_is_ante1_single_joker_planet_buy(state, action)
                    ):
                        fallback_action = self.fallback.choose_action(state)
                        if _action_is_joker_shop_card_buy(state, fallback_action) or _action_opens_pack_named(
                            state,
                            fallback_action,
                            "celestial",
                        ):
                            self._record_shop_action(state, fallback_action)
                            self._record_fallback_shop_action(state, fallback_action)
                            return fallback_action
                    if self.fallback_negative_shop_actions and _action_search_value(action) < 0.0:
                        fallback_action = self.fallback.choose_action(state)
                        if fallback_action.action_type != ActionType.END_SHOP:
                            self._record_shop_action(state, fallback_action)
                            return fallback_action
                    self._record_shop_action(state, action)
                    self._record_fallback_shop_action(state, action)
                    return action

            return self.fallback.choose_action(state)

    def _build_default_play_policy(self):
        backend = self.play_backend.strip().lower()
        if backend == "legacy":
            return PlaySearchPolicy(
                beam_depth=self.play_depth,
                beam_width=self.play_width,
                seed=self.seed,
                fallback=self.fallback,
            )
        if backend != "v2":
            raise ValueError(f"Unsupported solver play_backend: {self.play_backend!r}")

        # PlanningValueLeaf (rollout-backed) is the M4-parity leaf. The
        # earlier draft defaulted to FastHeuristicLeaf for speed but a
        # single-seed AAAAAAA measurement showed it loses ~3 antes at
        # d3w2 vs the M4 baseline — the rollout-free leaf can't drive
        # setup discards because it sees no progress signal from them.
        # FastHeuristicLeaf remains available for deeper-search
        # experiments via an explicit `play_policy=` injection.
        leaf = PlanningValueLeaf()
        if self.archetype is not None:
            leaf = ArchetypeAwareLeaf(base=leaf, archetype=self.archetype)
        return SearchV2PlayPolicy(
            depth=self.play_depth,
            width=self.play_width,
            leaf_evaluator=leaf,
            seed=self.seed,
            fallback=self.fallback,
        )

    def _archetype_leaf_value_fn(self, root_state: GameState):
        """Build the leaf_value_fn that adds the archetype-fit bonus.

        The shop search calls this for every leaf state in its beam. We
        delegate to the default `shop_leaf_terms` for the baseline score
        (capacity, build value, etc.) and add the archetype's per-match
        bonus on top — a soft bias, not a hard override.
        """

        archetype = self.archetype
        if archetype is None:
            return None
        # The default leaf_value_fn relies on the root state's build
        # baseline; we replicate that here so the archetype bonus is
        # additive rather than replacing the baseline scoring.
        from balatro_ai.search.shop_search import _shop_build_score  # local import to avoid widening top-level deps
        root_build_score = _shop_build_score(root_state)

        def _value(leaf_state: GameState) -> float:
            baseline = shop_leaf_terms(
                leaf_state,
                root_state=root_state,
                root_build_score=root_build_score,
            ).total
            return baseline + archetype.archetype_fit_score(leaf_state)

        return _value

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

    def _record_fallback_shop_action(self, state: GameState, action: Action) -> None:
        sync = getattr(self.fallback, "_sync_shop_memory", None)
        record = getattr(self.fallback, "_record_shop_action", None)
        if not callable(sync) or not callable(record):
            return
        try:
            sync(state)
            record(state, action)
        except (ValueError, IndexError, KeyError, TypeError, AttributeError):
            return

    def _fallback_info_first_shop_action(self, state: GameState) -> Action | None:
        try:
            from balatro_ai.bots.basic_strategy.actions import _annotated_action
            from balatro_ai.bots.basic_strategy.build_profile import _build_profile
            from balatro_ai.bots.basic_strategy.profile import _ShopContext
            from balatro_ai.bots.basic_strategy.run_plan import _run_plan
            from balatro_ai.bots.basic_strategy.shop_flow import _shop_information_first_action
            from balatro_ai.bots.basic_strategy.shop_planner import _calibrated_shop_action_value
            from balatro_ai.bots.basic_strategy.shop_pressure import _shop_pressure
            from balatro_ai.bots.basic_strategy.shop_values import _shop_buy_threshold
            from balatro_ai.bots.config import active_config

            context = _ShopContext(
                rerolls_in_shop=self._rerolls_in_shop,
                packs_opened_in_shop=self._packs_opened_in_shop,
                filled_last_joker_slot=self._filled_last_joker_slot_in_shop,
            )
            pressure = _shop_pressure(state)
            profile = _build_profile(state)
            run_plan = _run_plan(state, profile, pressure)
            best_action: Action | None = None
            best_value = 0.0
            for action in state.legal_actions:
                value = _calibrated_shop_action_value(state, action, pressure, context, run_plan, profile=profile)
                if value > best_value:
                    best_action = action
                    best_value = value
            threshold = _shop_buy_threshold(state, pressure)
            if best_action is None or best_value + active_config().shop_value_tolerance < threshold:
                return None
            sequenced = _shop_information_first_action(
                state,
                pressure,
                context,
                best_action=best_action,
                best_value=best_value,
                threshold=threshold,
                run_plan=run_plan,
                profile=profile,
            )
        except (ValueError, IndexError, KeyError, TypeError, AttributeError):
            return None
        if sequenced is None:
            return None
        info_action, info_value, planned_item = sequenced
        if info_action.action_type == ActionType.OPEN_PACK:
            return _annotated_action(
                info_action,
                reason=(
                    f"shop_sequence_info_first value={info_value:.1f} "
                    f"planned_buy={_item_label(planned_item)} buy_value={best_value:.1f} "
                    f"pressure={pressure.ratio:.2f}"
                ),
            )
        return None

    def _opening_buffoon_pack_action(self, state: GameState) -> Action | None:
        if state.ante > 1 or state.jokers or self._packs_opened_in_shop > 0:
            return None
        packs = _modifier_items(state.modifiers, "booster_packs")
        for action in state.legal_actions:
            if action.action_type != ActionType.OPEN_PACK:
                continue
            index = _action_index(action)
            if index is None or not 0 <= index < len(packs):
                continue
            pack = packs[index]
            if "buffoon" not in _item_label(pack).lower():
                continue
            try:
                from balatro_ai.bots.basic_strategy.actions import _annotated_action

                return _annotated_action(action, reason="shop_opening_buffoon_first")
            except (ValueError, IndexError, KeyError, TypeError, AttributeError):
                return action
        return None


def _has_shop_action(state: GameState) -> bool:
    shop_action_types = {
        ActionType.BUY,
        ActionType.SELL,
        ActionType.REROLL,
        ActionType.OPEN_PACK,
        ActionType.END_SHOP,
        ActionType.USE_CONSUMABLE,
    }
    for action in state.legal_actions:
        if action.action_type in shop_action_types:
            return True
    return False


def _action_search_value(action: Action) -> float:
    try:
        return float(action.metadata.get("search_value", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _action_is_non_joker_shop_card_buy(state: GameState, action: Action) -> bool:
    if action.action_type != ActionType.BUY:
        return False
    kind = str(action.metadata.get("kind", action.target_id or "")).lower()
    if kind != "card":
        return False
    index = _action_index(action)
    shop_cards = _modifier_items(state.modifiers, "shop_cards")
    if index is None or not 0 <= index < len(shop_cards):
        return False
    return not _is_joker_item(shop_cards[index])


def _action_is_joker_shop_card_buy(state: GameState, action: Action) -> bool:
    if action.action_type != ActionType.BUY:
        return False
    kind = str(action.metadata.get("kind", action.target_id or "")).lower()
    if kind != "card":
        return False
    index = _action_index(action)
    shop_cards = _modifier_items(state.modifiers, "shop_cards")
    return index is not None and 0 <= index < len(shop_cards) and _is_joker_item(shop_cards[index])


def _action_is_ante1_single_joker_planet_buy(state: GameState, action: Action) -> bool:
    if state.ante != 1 or len(state.jokers) != 1 or action.action_type != ActionType.BUY:
        return False
    kind = str(action.metadata.get("kind", action.target_id or "")).lower()
    if kind != "card":
        return False
    index = _action_index(action)
    shop_cards = _modifier_items(state.modifiers, "shop_cards")
    if index is None or not 0 <= index < len(shop_cards):
        return False
    item = shop_cards[index]
    return isinstance(item, dict) and str(item.get("set", "")).upper() == "PLANET"


def _action_opens_pack_named(state: GameState, action: Action, name_part: str) -> bool:
    if action.action_type != ActionType.OPEN_PACK:
        return False
    index = _action_index(action)
    packs = _modifier_items(state.modifiers, "booster_packs")
    if index is None or not 0 <= index < len(packs):
        return False
    return name_part.lower() in _item_label(packs[index]).lower()


def _sell_funds_planned_buy(state: GameState, action: Action) -> bool:
    sell_value = _sell_action_value(state, action)
    if sell_value <= 0:
        return False
    path = action.metadata.get("search_path")
    if not isinstance(path, list | tuple) or len(path) < 2:
        return False
    for step in path[1:]:
        if not isinstance(step, dict) or str(step.get("type", "")) != "buy":
            continue
        item = _planned_buy_item(state, step)
        cost = _item_buy_cost(item)
        return state.money < cost <= state.money + sell_value
    return False


def _sell_action_value(state: GameState, action: Action) -> int:
    index = _action_index(action)
    if index is None:
        return 0
    kind = str(action.metadata.get("kind", action.target_id or "")).lower()
    if kind in {"joker", "jokers"} and 0 <= index < len(state.jokers):
        return int(state.jokers[index].sell_value or 0)
    return 0


def _planned_buy_item(state: GameState, step: dict[str, object]) -> object | None:
    try:
        index = int(step.get("index"))
    except (TypeError, ValueError):
        return None
    kind = str(step.get("kind", "")).lower()
    bucket = {
        "card": "shop_cards",
        "pack": "booster_packs",
        "voucher": "voucher_cards",
    }.get(kind)
    if bucket is None:
        return None
    items = _modifier_items(state.modifiers, bucket)
    if not 0 <= index < len(items):
        return None
    return items[index]


def _item_buy_cost(item: object) -> int:
    if not isinstance(item, dict):
        return 0
    cost = item.get("cost", 0)
    if isinstance(cost, dict):
        cost = cost.get("buy", 0)
    try:
        return int(cost or 0)
    except (TypeError, ValueError):
        return 0


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
