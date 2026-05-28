"""Seed materialization for the offline solver (Milestone M1).

A `SeedGame` is the perfect-information view of a Balatro run that the
offline solver searches over. Given a seed string, it predicts:

- Initial deck shuffle order (via `rng.luajit_pseudoshuffle`).
- Per-ante boss blind, current voucher, and Small/Big tag pair
  (via `rng.surfaces.predict_initial_surface`).
- Shop and pack RNG outputs lazily, on demand, when the solver actually
  reaches a decision that needs them.

This is the thin orchestrator between the existing RNG predictors and the
existing `forward_sim`. It deliberately does NOT reimplement game logic —
its only job is to construct a fresh `GameState` that `forward_sim` can
drive forward without a live bridge.

Status: M1 of `SOLVER_PLAN.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from balatro_ai.api.state import Card, GamePhase, GameState, Stake
from balatro_ai.rng.balatro_rng import BalatroRNG
from balatro_ai.rng.deck import build_standard_red_deck
from balatro_ai.rng.luajit_prng import luajit_pseudoshuffle
from balatro_ai.rng.surfaces import BOSS_NAMES, InitialRunSurface, predict_initial_surface
from balatro_ai.rules.hand_evaluator import HandType
from balatro_ai.sim.local_runner import _with_blind_selection_surface


STARTING_MONEY = 4
STARTING_HANDS = 4
STARTING_DISCARDS = 4
DEFAULT_HAND_SIZE = 8


def deck_for_seed(seed: str) -> tuple[Card, ...]:
    """Predict the initial shuffled deck for `seed` as `api.state.Card` tuples.

    Verified to match captured bridge fixtures for the four canonical seeds
    on the full 52-card deck (see `tests/test_rng_against_bridge.py`).
    """

    rng = BalatroRNG(seed=seed, mix_hashed_seed=True)
    deck_keys = list(build_standard_red_deck())
    seed_float = rng.random("shuffle")
    luajit_pseudoshuffle(deck_keys, seed_float)
    return tuple(Card(rank=card.rank, suit=card.suit) for card in deck_keys)


def _base_hand_levels() -> dict[str, int]:
    return {hand_type.value: 1 for hand_type in HandType}


def _base_hands_modifier() -> dict[str, dict[str, int]]:
    return {
        hand_type.value: {"level": 1, "played": 0, "played_this_round": 0, "order": index}
        for index, hand_type in enumerate(HandType, start=1)
    }


def _base_modifiers() -> dict[str, Any]:
    return {
        "base_hands": STARTING_HANDS,
        "base_discards": STARTING_DISCARDS,
        "hand_size": DEFAULT_HAND_SIZE,
        "joker_slot_limit": 5,
        "consumable_slot_limit": 2,
        "reroll_cost": 5,
        "current_reroll_cost": 5,
        "round_reset_reroll_cost": 5,
        "reroll_cost_increase": 0,
        "first_shop_buffoon": True,  # vanilla: first shop has guaranteed Buffoon pack
        "hands": _base_hands_modifier(),
    }


@dataclass(slots=True)
class SeedGame:
    """Perfect-information view of a Balatro run for the offline solver.

    Construction is cheap (a few RNG calls). Deeper predictions (shop
    contents, pack contents, boss for ante 2+) are produced lazily by
    helper methods so the search doesn't pay for branches it never
    visits.

    Attributes
    ----------
    seed: the run seed string (e.g. "AAAAAAA").
    stake: stake name, defaults to "white".
    """

    seed: str
    stake: str = "white"
    _initial_surface: InitialRunSurface | None = field(default=None, init=False, repr=False)
    _deck: tuple[Card, ...] | None = field(default=None, init=False, repr=False)
    _boss_by_ante: dict[int, str] = field(default_factory=dict, init=False, repr=False)

    def initial_surface(self) -> InitialRunSurface:
        """Ante-1 boss/voucher/Small-tag/Big-tag (memoized)."""

        if self._initial_surface is None:
            self._initial_surface = predict_initial_surface(self.seed, ante=1)
        return self._initial_surface

    def deck(self) -> tuple[Card, ...]:
        """Seed-faithful initial deck (memoized)."""

        if self._deck is None:
            self._deck = deck_for_seed(self.seed)
        return self._deck

    def boss_for_ante(self, ante: int) -> str:
        """Predicted boss key for the given ante (1..8). Memoized per ante."""

        if ante not in self._boss_by_ante:
            if ante == 1:
                self._boss_by_ante[ante] = self.initial_surface().boss_key
            else:
                # Ante 2+ boss prediction needs the running boss-use counts.
                # Defer until shop/multi-ante work lands — M1 only needs ante 1.
                raise NotImplementedError(
                    f"boss_for_ante({ante}) requires multi-ante state tracking; "
                    "not implemented yet in M1. Use rng.surfaces.predict_boss "
                    "directly with the running boss_use_counts dict."
                )
        return self._boss_by_ante[ante]

    def initial_state(self) -> GameState:
        """Fresh BLIND_SELECT state for ante 1, ready for `simulate_select_blind`.

        The returned state is what the bridge would emit immediately after
        `start_run(seed=...)`: ante 1 BLIND_SELECT with the predicted deck
        ordered top-to-bottom in `known_deck`, $4 starting money, 4 hands,
        4 discards, hand-level 1 for every hand type, and the ante-1 boss
        / voucher / tag surface populated in `modifiers`.
        """

        surface = self.initial_surface()
        deck = self.deck()

        modifiers = _base_modifiers()
        # Stash surface predictions where downstream code expects them.
        # Use simple flat keys for now; can promote to the full bridge-style
        # nested "blinds" / "current_round.voucher" shape later if needed.
        modifiers["seed_game_initial_surface"] = {
            "boss_key": surface.boss_key,
            "voucher_key": surface.voucher_key,
            "small_tag_key": surface.small_tag_key,
            "big_tag_key": surface.big_tag_key,
        }

        bare = GameState(
            phase=GamePhase.BLIND_SELECT,
            stake=_stake_from_name(self.stake),
            ante=1,
            money=STARTING_MONEY,
            deck_size=len(deck),
            known_deck=deck,
            hands_remaining=STARTING_HANDS,
            discards_remaining=STARTING_DISCARDS,
            hand_levels=_base_hand_levels(),
            modifiers=modifiers,
        )
        # Use the local_runner helper to populate the blind-selection surface
        # (current_blind, blinds dict, required_score, blind name). This is the
        # same code path the local simulator uses, so downstream forward_sim
        # behavior is consistent.
        boss_name = BOSS_NAMES.get(surface.boss_key, surface.boss_key)
        return _with_blind_selection_surface(
            bare, ante=1, blind_kind="SMALL", boss_name=boss_name
        )


def _stake_from_name(name: str) -> Stake:
    name = (name or "white").strip().lower()
    try:
        return Stake(name)
    except ValueError:
        return Stake.WHITE
