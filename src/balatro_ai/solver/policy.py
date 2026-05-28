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

from dataclasses import dataclass, field

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GamePhase, GameState
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
DEFAULT_SHOP_BEAM_WIDTH = 4
DEFAULT_SHOP_DEPTH = 2
DEFAULT_SHOP_REROLL_SAMPLES = 8
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
    _sampler: ShopSampler = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.fallback is None:
            self.fallback = BasicStrategyBot(seed=self.seed)
        if self.play_policy is None:
            self.play_policy = self._build_default_play_policy()
        if self.shop_config is None:
            self.shop_config = ShopSearchConfig(
                beam_width=DEFAULT_SHOP_BEAM_WIDTH,
                depth=DEFAULT_SHOP_DEPTH,
                reroll_samples=DEFAULT_SHOP_REROLL_SAMPLES,
                seed=self.seed,
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
            if state.phase == GamePhase.SELECTING_HAND:
                return self.play_policy.choose_action(state)

            if state.phase == GamePhase.SHOP and _has_shop_action(state):
                try:
                    leaf_value_fn = self._archetype_leaf_value_fn(state) if self.archetype is not None else None
                    action = best_shop_action(
                        state,
                        config=self.shop_config,
                        sampler=self._sampler,
                        shop_context=ShopSearchContext(),
                        leaf_value_fn=leaf_value_fn,
                    )
                except (ValueError, IndexError, KeyError, TypeError, AttributeError):
                    # Shop search occasionally errors on unusual states
                    # (consumable-slot-full, voucher legality, etc.). Fall
                    # back rather than crash the trajectory.
                    action = None
                if action is not None:
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


def _has_shop_action(state: GameState) -> bool:
    shop_action_types = {
        ActionType.BUY,
        ActionType.SELL,
        ActionType.REROLL,
        ActionType.OPEN_PACK,
        ActionType.END_SHOP,
    }
    for action in state.legal_actions:
        if action.action_type in shop_action_types:
            return True
    return False
