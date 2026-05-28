"""Whole-blind beam search policy (Milestone M4).

`PlaySearchPolicy` is the solver's first non-trivial policy. For
`SELECTING_HAND` decisions it runs `best_hand_action` from the existing
`balatro_ai.search.hand_search` module with a beam-search configuration
(``beam_depth > 1`` triggers the whole-blind expectimax beam in
`best_blind_beam_action`). For every other phase (BLIND_SELECT, SHOP,
BOOSTER_OPENED, etc.) the policy delegates to a fallback —
`basic_strategy_bot` by default — until M5/M6 replace shop/pack
decisions with their own search.

Why wrap instead of reimplement: the existing beam-search machinery in
`hand_search.py` is already exercised by `search_bot_v2` plus the
solver/sim integration tests. Reusing it keeps M4 small and lets us
focus subsequent milestones on shop search (M5) and archetype branching
(M6), which are the changes that actually move the win-rate needle.

Status: M4 of `SOLVER_PLAN.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GamePhase, GameState
from balatro_ai.bots.basic_strategy_bot import BasicStrategyBot
from balatro_ai.search.hand_search import (
    HandSearchConfig,
    best_blind_beam_action,
    best_hand_action,
)


# Defaults match search_bot_v2's `beam_depth=3, beam_width=2`, which is
# the known-working setting in the live bot. Higher values (depth 8 +
# width 4) blow up per-decision cost ~10x without proportional gain
# from a single beam pass — archetype branching in M6 is the right
# place to spend more compute. Tune in M3-style throughput measurements
# before any larger dataset run.
DEFAULT_BEAM_DEPTH = 3
DEFAULT_BEAM_WIDTH = 2


@dataclass(slots=True)
class PlaySearchPolicy:
    """Policy that runs whole-blind beam search at SELECTING_HAND.

    Parameters
    ----------
    beam_depth: maximum plies of the blind to expand. The blind usually
                terminates earlier (clear or bust) and the search bails
                out then; this is the upper bound.
    beam_width: number of branches kept after each ply. Higher = better
                action quality, linearly more compute.
    max_play_actions / max_discard_actions: per-state action enumeration
                ceilings before ranking — caps the explosion of 5-card
                hand combinations.
    seed: passed into `HandSearchConfig` for any sampled-draws path it
          still exposes (the beam uses `beam_draw_samples` instead).
    skip_preprocessing: when True (default), call `best_blind_beam_action`
                directly and skip the live-bot's heuristic preprocessing
                (`_opening_first_blind_hunt_action` + first-blind hunts).
                The M4 profile showed that preprocessing was ~90% of
                per-call cost; skipping it is the M5.5 optimization (1).
                Set False to mirror live-bot behavior exactly (for
                debugging / parity testing only).
    fallback: any `GameState -> Action` callable used for non-play/discard
              decisions. Defaults to a fresh `BasicStrategyBot`.
    """

    beam_depth: int = DEFAULT_BEAM_DEPTH
    beam_width: int = DEFAULT_BEAM_WIDTH
    max_play_actions: int = 16
    max_discard_actions: int = 8
    seed: int = 0
    # M5.5 finding: setting True actually slows the play path. The
    # heuristic preprocessing was acting as a useful shortcut — when it
    # finds a clear winning discard it skips the beam entirely. Bypassing
    # it makes the beam run on every state, increasing per-decision cost
    # from ~1.5s to ~8s for typical opening hands. Default stays False
    # (use the preprocessing) and we look for cost wins elsewhere.
    skip_preprocessing: bool = False
    fallback: object | None = None
    _config: HandSearchConfig | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.fallback is None:
            self.fallback = BasicStrategyBot(seed=self.seed)
        self._config = HandSearchConfig(
            draw_samples=1,
            leaf_samples=1,
            seed=self.seed,
            max_play_actions=self.max_play_actions,
            max_discard_actions=self.max_discard_actions,
            beam_depth=self.beam_depth,
            beam_width=self.beam_width,
            beam_draw_samples=1,
            beam_leaf_samples=1,
        )

    def __call__(self, state: GameState) -> Action:
        return self.choose_action(state)

    def choose_action(self, state: GameState) -> Action:
        """Return an action: beam search for play/discard, fallback otherwise."""

        if state.phase == GamePhase.SELECTING_HAND and _has_play_or_discard(state):
            try:
                if self.skip_preprocessing:
                    # Direct beam call — skips first-blind heuristic and the
                    # `_basic_best_discard_action` preprocessing that
                    # M4 profiling showed dominated per-call cost.
                    action = best_blind_beam_action(state, config=self._config)
                else:
                    action = best_hand_action(state, config=self._config)
            except (ValueError, IndexError, KeyError, TypeError, AttributeError):
                # Search machinery occasionally fails on unusual states
                # (bridge-style validation errors, missing context). Fall
                # back rather than crash the trajectory.
                action = None
            if action is not None:
                return action
        return self.fallback.choose_action(state)


def _has_play_or_discard(state: GameState) -> bool:
    for action in state.legal_actions:
        if action.action_type in (ActionType.PLAY_HAND, ActionType.DISCARD):
            return True
    return False
