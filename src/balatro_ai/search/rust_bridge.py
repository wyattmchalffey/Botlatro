"""Shared Rust-extension wire-in helpers (Phase 4 of RUST_PORT_PLAN.md).

Multiple Python callers (state_value, build_scoring, score_projection,
play_scoring) now route hand-scoring through `balatro_core`. Rather
than duplicating the joker-extraction + bail-check boilerplate, this
module centralizes the shared scaffolding:

- `RUST_BLIND_SAFE`: the blinds the Rust evaluator handles without
  needing Python-side boss adjustments (Eye / Mouth / Flint / Arm /
  Tooth / Psychic / Plant all bail to Python).
- `rust_joker_data(jokers)`: identity-cached extraction of the
  14-list of joker fields needed by `evaluate_simple_with_levels`.
  Cache value carries the original tuple reference so id() collisions
  after GC can't return stale data — that hazard caused a real
  trajectory divergence (130→165 steps) during Phase 4f development.
- `rust_evaluate_score(state, played, held, jokers)`: full wire-in
  returning just the integer score, or None to signal Python
  fallback. Use this when the caller only needs `evaluation.score`.

Bail behavior is conservative: anything outside the fast path
returns None (caller falls through to its own Python implementation).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from balatro_ai.api.state import Card, GameState, Joker


RUST_BLIND_SAFE: frozenset[str] = frozenset({
    "", "Small Blind", "Big Blind",
    # Suit-debuff bosses — handled via the debuffed_suits list passed
    # to evaluate_simple_with_levels, no Python adjustment needed.
    "The Club", "The Goad", "The Head", "The Window",
    # The Eye / The Mouth: Rust returns the un-adjusted score; callers
    # that need the boss adjustment (zero-out on duplicate hand type
    # for Eye / non-first hand type for Mouth) wrap with
    # `_boss_adjusted_score` Python-side.
    "The Eye", "The Mouth",
    # The Manacle: -1 hand size — affects which cards available to
    # play, not the score of a given play. Safe for evaluate.
    "The Manacle",
    # The Wheel: 1-in-7 face-down per card is a SIMULATE_PLAY effect
    # (forward_sim), not an evaluator effect. The Python evaluator
    # (`evaluate_played_cards`) doesn't apply face-down — that
    # happens later in simulate_play when actually executing the
    # play. So Rust evaluate returns the same score as Python.
    "The Wheel",
    # The Hook: removes 2 random cards from HELD before scoring
    # (see simulate_play: hook_discarded_cards). The play cards
    # themselves are unaffected, so evaluator score is unchanged.
    # Python's evaluate_played_cards doesn't reference "The Hook"
    # — the hook-discarded held cards are removed by the caller
    # before the evaluator runs.
    "The Hook",
    # The Psychic: Python's evaluator zeros score if len(cards) != 5.
    # The bridge handles this as a SHORT-CIRCUIT (returns 0 directly)
    # in both `rust_evaluate_score` and
    # `rust_evaluate_score_and_hand_type` BEFORE calling Rust, so
    # Psychic is safe in the blind set.
    "The Psychic",
    # Score-neutral bosses (audited 2026-05-28): effects are
    # money / hands / forward-sim / card-debuff, none touch
    # chips/mult/score. Parity verified Rust==Python on 495 samples
    # including Bull/Bootstraps money jokers + debuffed cards (Pillar).
    # NOT The Tooth — its $1/card deduction feeds money-scaling jokers
    # within the same evaluation, so Rust (pre-deduction money)
    # diverges from Python (post-deduction).
    "The Ox", "The Needle", "The Mark", "The Pillar",
})

# Identity-keyed cache for the Rust-format joker data. The jokers
# tuple is frozen, so identity is a safe content key for the
# tuple object's lifetime. Cache value is (jokers_ref, data) so we
# can verify identity AND keep the original tuple alive to prevent
# id() recycling. Bounded size to prevent runaway memory in long
# processes.
_JOKER_CACHE: dict[int, tuple[tuple, tuple[object, ...]]] = {}
_JOKER_CACHE_LIMIT = 512

_RARITY_TO_INT = {"common": 0, "uncommon": 1, "rare": 2, "legendary": 3}


def rust_joker_data(jokers: tuple) -> tuple[object, ...] | None:
    """Identity-cached extraction of joker fields for the Rust
    evaluator. Returns the 14-tuple expected by
    `evaluate_simple_with_levels`'s joker_* params, or None if the
    extraction fails."""

    key = id(jokers)
    entry = _JOKER_CACHE.get(key)
    if entry is not None and entry[0] is jokers:
        return entry[1]
    try:
        from balatro_ai.rules.hand_evaluator import (
            _joker_current_plus, _joker_current_xmult,
            _joker_leading_plus,
            _loyalty_card_ready, _drivers_license_active,
            _joker_rarity, _joker_target_suit, _joker_target_rank_suit,
            _obelisk_xmult_gain,
        )
    except ImportError:
        return None
    try:
        joker_names = [j.name for j in jokers]
        joker_editions = [getattr(j, "edition", None) for j in jokers]
        joker_plus_mult = [int(_joker_current_plus(j, suffix="mult")) for j in jokers]
        joker_plus_chips = [int(_joker_current_plus(j, suffix="chips")) for j in jokers]
        joker_xmult = [float(_joker_current_xmult(j)) for j in jokers]
        joker_loyalty_ready = [
            bool(_loyalty_card_ready(j)) if j.name == "Loyalty Card" else False
            for j in jokers
        ]
        joker_drivers_active = [
            bool(_drivers_license_active(j)) if j.name == "Driver's License" else False
            for j in jokers
        ]
        joker_leading_plus_mult = [
            int(_joker_leading_plus(j, suffix="mult")) if j.name == "Popcorn" else 0
            for j in jokers
        ]
        joker_leading_plus_chips = [
            int(_joker_leading_plus(j, suffix="chips")) if j.name == "Ice Cream" else 0
            for j in jokers
        ]
        joker_sell_value = [int(getattr(j, "sell_value", 0) or 0) for j in jokers]
        joker_rarity = [_RARITY_TO_INT.get(_joker_rarity(j), 0) for j in jokers]
        joker_target_suit: list[str | None] = []
        joker_target_rank: list[str | None] = []
        for j in jokers:
            if j.name == "Ancient Joker":
                joker_target_suit.append(_joker_target_suit(j))
                joker_target_rank.append(None)
            elif j.name == "The Idol":
                ts = _joker_target_rank_suit(j)
                if ts is None:
                    joker_target_suit.append(None)
                    joker_target_rank.append(None)
                else:
                    rk, st = ts
                    joker_target_rank.append(rk)
                    joker_target_suit.append(st)
            else:
                joker_target_suit.append(None)
                joker_target_rank.append(None)
        joker_obelisk_gain = [
            float(_obelisk_xmult_gain(j)) if j.name == "Obelisk" else 0.0
            for j in jokers
        ]
        # Swashbuckler precompute: sum of OTHER jokers' sell values.
        if "Swashbuckler" in joker_names:
            total_sell = sum(joker_sell_value)
            for i, n in enumerate(joker_names):
                if n == "Swashbuckler":
                    joker_plus_mult[i] = total_sell - joker_sell_value[i]
    except Exception:  # noqa: BLE001
        return None
    result = (
        joker_names, joker_editions, joker_plus_mult, joker_plus_chips,
        joker_xmult, joker_loyalty_ready, joker_drivers_active,
        joker_leading_plus_mult, joker_leading_plus_chips,
        joker_sell_value, joker_rarity, joker_target_suit,
        joker_target_rank, joker_obelisk_gain,
    )
    if len(_JOKER_CACHE) > _JOKER_CACHE_LIMIT:
        _JOKER_CACHE.clear()
    _JOKER_CACHE[key] = (jokers, result)
    return result


# Identity-keyed cache for RustCard conversion of frozen card
# tuples. Sample hands in shop valuation are mostly module-level
# constants (WHITE_STAKE_SAMPLE_HANDS) so the same tuple object
# gets reused thousands of times — re-converting wastes Python
# attribute lookups. Tuple-ref guard prevents id() recycling stale
# data. Bounded size.
_RUST_CARDS_CACHE: dict[int, tuple[tuple, list]] = {}
_RUST_CARDS_CACHE_LIMIT = 1024


def _rust_cards_from_tuple(cards: tuple) -> list | None:
    """Identity-cached conversion of a frozen card tuple to a
    RustCard list. Returns None if balatro_core can't extract any
    card (e.g. unknown enhancement)."""

    if not cards:
        return []
    key = id(cards)
    entry = _RUST_CARDS_CACHE.get(key)
    if entry is not None and entry[0] is cards:
        return entry[1]
    try:
        import balatro_core
    except ImportError:
        return None
    try:
        out = [balatro_core.RustCard.from_python(c) for c in cards]
    except (AttributeError, TypeError, ValueError):
        return None
    if len(_RUST_CARDS_CACHE) > _RUST_CARDS_CACHE_LIMIT:
        _RUST_CARDS_CACHE.clear()
    _RUST_CARDS_CACHE[key] = (cards, out)
    return out


def rust_best_play_scores(
    cards: tuple,
    hand_levels: dict | None,
    blind_name: str,
    jokers: tuple,
    *,
    discards_remaining: int = 0,
    hands_remaining: int = 0,
    deck_size: int = 0,
    money: int = 0,
    played_hand_types: tuple = (),
    max_cards: int = 5,
) -> tuple[list[tuple[int, ...]], list[int]] | None:
    """Score EVERY 1..max_cards-card subset of ``cards`` via the Rust
    batched scorer, returning ``(action_indices_list, scores)`` in the
    SAME enumeration order as ``itertools.combinations`` (so callers can
    apply their own tie-break), or None to signal Python fallback.

    Bails (returns None) on: non-vanilla scoring blinds, The Psychic
    (Python zeroes !=5-card plays — handled by Python loop), empty hand,
    balatro_core unavailable, card/joker extraction failure, or any Rust
    internal bail (Wild cards / unsupported jokers -> batch returns None).
    """

    if not cards:
        return None
    try:
        import balatro_core
    except ImportError:
        return None
    if blind_name not in RUST_BLIND_SAFE or blind_name == "The Psychic":
        return None
    rust_hand = _rust_cards_from_tuple(tuple(cards))
    if rust_hand is None:
        return None
    joker_data = rust_joker_data(tuple(jokers))
    if joker_data is None:
        return None
    (joker_names, joker_editions, joker_plus_mult, joker_plus_chips,
     joker_xmult, joker_loyalty_ready, joker_drivers_active,
     joker_leading_plus_mult, joker_leading_plus_chips,
     joker_sell_value, joker_rarity, joker_target_suit,
     joker_target_rank, joker_obelisk_gain) = joker_data

    if any(n in {"Card Sharp", "Supernova", "Obelisk"} for n in joker_names):
        played_types_strs = [
            ht.value if hasattr(ht, "value") else str(ht) for ht in played_hand_types
        ]
    else:
        played_types_strs = []

    from itertools import combinations

    from balatro_ai.rules.hand_evaluator import debuffed_suits_for_blind
    debuffed = [s for s in debuffed_suits_for_blind(blind_name) if len(s) == 1]
    n = len(cards)
    max_size = min(max_cards, n)
    action_indices_list = [
        list(idxs) for size in range(1, max_size + 1)
        for idxs in combinations(range(n), size)
    ]
    if not action_indices_list:
        return None
    try:
        scores = balatro_core.score_play_actions_batch(
            rust_hand, action_indices_list,
            hand_levels or {}, debuffed,
            joker_names=joker_names, joker_editions=joker_editions,
            joker_current_plus_mult=joker_plus_mult,
            joker_current_plus_chips=joker_plus_chips,
            joker_current_xmult=joker_xmult,
            joker_loyalty_ready=joker_loyalty_ready,
            joker_drivers_active=joker_drivers_active,
            joker_leading_plus_mult=joker_leading_plus_mult,
            joker_leading_plus_chips=joker_leading_plus_chips,
            joker_sell_value=joker_sell_value, joker_rarity=joker_rarity,
            joker_target_suit=joker_target_suit, joker_target_rank=joker_target_rank,
            joker_obelisk_gain=joker_obelisk_gain,
            money=int(money), joker_slot_limit=5,
            discards_remaining=int(max(0, discards_remaining)),
            hands_remaining=int(max(0, hands_remaining)),
            played_hand_types=played_types_strs,
            deck_size=int(max(0, deck_size)),
        )
    except Exception:  # noqa: BLE001
        return None
    if scores is None or len(scores) != len(action_indices_list):
        return None
    return [tuple(a) for a in action_indices_list], [int(s) for s in scores]


def rust_evaluate_score(
    state: "GameState",
    played_cards: tuple,
    held_cards: tuple,
    jokers: tuple,
    *,
    played_hand_types: tuple = (),
) -> int | None:
    """Full Rust-fast-path evaluator returning just the score int,
    or None to bail to Python. Bails on:
    - Non-vanilla blinds (boss scoring effects)
    - Unsupported jokers (handled by Rust returning None internally)
    - Empty played_cards
    - balatro_core not loadable / card extraction failures
    """

    if not played_cards:
        return None
    try:
        import balatro_core  # noqa: F401
    except ImportError:
        return None
    if state.blind not in RUST_BLIND_SAFE:
        return None
    # The Psychic: Python's evaluator zeros score for !=5-card plays.
    # Short-circuit here BEFORE building Rust inputs (saves the
    # extraction work too). 5-card plays under Psychic fall through
    # to the normal Rust path.
    if state.blind == "The Psychic" and len(played_cards) != 5:
        return 0  # Note: for rust_evaluate_score_and_hand_type the
                  # caller still expects a tuple; that branch handles
                  # its own short-circuit below.
    rust_played = _rust_cards_from_tuple(played_cards)
    if rust_played is None:
        return None
    rust_held = _rust_cards_from_tuple(held_cards)
    if rust_held is None:
        return None
    import balatro_core
    joker_data = rust_joker_data(jokers)
    if joker_data is None:
        return None
    (joker_names, joker_editions, joker_plus_mult, joker_plus_chips,
     joker_xmult, joker_loyalty_ready, joker_drivers_active,
     joker_leading_plus_mult, joker_leading_plus_chips,
     joker_sell_value, joker_rarity, joker_target_suit,
     joker_target_rank, joker_obelisk_gain) = joker_data

    # played_hand_types is only relevant when Card Sharp / Supernova /
    # Obelisk are present. Compute strs lazily.
    if any(n in {"Card Sharp", "Supernova", "Obelisk"} for n in joker_names):
        played_types_strs = [
            ht.value if hasattr(ht, "value") else str(ht)
            for ht in played_hand_types
        ]
    else:
        played_types_strs = []

    from balatro_ai.rules.hand_evaluator import debuffed_suits_for_blind
    debuffed = [s for s in debuffed_suits_for_blind(state.blind) if len(s) == 1]
    slot_limit_raw = state.modifiers.get("joker_slot_limit", 5)
    try:
        joker_slot_limit = int(slot_limit_raw) if slot_limit_raw is not None else 5
    except (TypeError, ValueError):
        joker_slot_limit = 5
    try:
        result = balatro_core.evaluate_simple_with_levels(
            rust_played, state.hand_levels, debuffed_suits=debuffed,
            joker_names=joker_names, joker_editions=joker_editions,
            held_cards=rust_held,
            joker_current_plus_mult=joker_plus_mult,
            joker_current_plus_chips=joker_plus_chips,
            joker_current_xmult=joker_xmult,
            joker_loyalty_ready=joker_loyalty_ready,
            joker_drivers_active=joker_drivers_active,
            joker_leading_plus_mult=joker_leading_plus_mult,
            joker_leading_plus_chips=joker_leading_plus_chips,
            joker_sell_value=joker_sell_value, joker_rarity=joker_rarity,
            joker_target_suit=joker_target_suit, joker_target_rank=joker_target_rank,
            joker_obelisk_gain=joker_obelisk_gain,
            money=int(state.money), joker_slot_limit=joker_slot_limit,
            discards_remaining=int(max(0, state.discards_remaining)),
            hands_remaining=int(max(1, state.hands_remaining)),
            played_hand_types=played_types_strs,
            deck_size=int(max(0, state.deck_size)),
        )
    except Exception:  # noqa: BLE001
        return None
    if result is None:
        return None
    _chips, _mult, score, _ht = result
    return int(score)


def rust_evaluate_score_and_hand_type(
    state: "GameState",
    played_cards: tuple,
    held_cards: tuple,
    jokers: tuple,
    *,
    played_hand_types: tuple = (),
) -> tuple[int, str] | None:
    """Same as `rust_evaluate_score` but also returns the hand type
    string. Used by callers that need `evaluation.hand_type` (for
    boss adjustments, etc.)."""

    if not played_cards:
        return None
    try:
        import balatro_core  # noqa: F401
    except ImportError:
        return None
    if state.blind not in RUST_BLIND_SAFE:
        return None
    # The Psychic: Python's evaluator zeros score for !=5-card plays.
    # Short-circuit here. We still need the hand_type for downstream
    # boss-adjusted-score logic, but it doesn't matter what we return
    # since score=0 makes _boss_adjusted_score early-out via its
    # `if score <= 0` clause.
    if state.blind == "The Psychic" and len(played_cards) != 5:
        return 0, "High Card"
    rust_played = _rust_cards_from_tuple(played_cards)
    if rust_played is None:
        return None
    rust_held = _rust_cards_from_tuple(held_cards)
    if rust_held is None:
        return None
    import balatro_core
    joker_data = rust_joker_data(jokers)
    if joker_data is None:
        return None
    (joker_names, joker_editions, joker_plus_mult, joker_plus_chips,
     joker_xmult, joker_loyalty_ready, joker_drivers_active,
     joker_leading_plus_mult, joker_leading_plus_chips,
     joker_sell_value, joker_rarity, joker_target_suit,
     joker_target_rank, joker_obelisk_gain) = joker_data

    if any(n in {"Card Sharp", "Supernova", "Obelisk"} for n in joker_names):
        played_types_strs = [
            ht.value if hasattr(ht, "value") else str(ht)
            for ht in played_hand_types
        ]
    else:
        played_types_strs = []

    from balatro_ai.rules.hand_evaluator import debuffed_suits_for_blind
    debuffed = [s for s in debuffed_suits_for_blind(state.blind) if len(s) == 1]
    slot_limit_raw = state.modifiers.get("joker_slot_limit", 5)
    try:
        joker_slot_limit = int(slot_limit_raw) if slot_limit_raw is not None else 5
    except (TypeError, ValueError):
        joker_slot_limit = 5
    try:
        result = balatro_core.evaluate_simple_with_levels(
            rust_played, state.hand_levels, debuffed_suits=debuffed,
            joker_names=joker_names, joker_editions=joker_editions,
            held_cards=rust_held,
            joker_current_plus_mult=joker_plus_mult,
            joker_current_plus_chips=joker_plus_chips,
            joker_current_xmult=joker_xmult,
            joker_loyalty_ready=joker_loyalty_ready,
            joker_drivers_active=joker_drivers_active,
            joker_leading_plus_mult=joker_leading_plus_mult,
            joker_leading_plus_chips=joker_leading_plus_chips,
            joker_sell_value=joker_sell_value, joker_rarity=joker_rarity,
            joker_target_suit=joker_target_suit, joker_target_rank=joker_target_rank,
            joker_obelisk_gain=joker_obelisk_gain,
            money=int(state.money), joker_slot_limit=joker_slot_limit,
            discards_remaining=int(max(0, state.discards_remaining)),
            hands_remaining=int(max(1, state.hands_remaining)),
            played_hand_types=played_types_strs,
            deck_size=int(max(0, state.deck_size)),
        )
    except Exception:  # noqa: BLE001
        return None
    if result is None:
        return None
    _chips, _mult, score, ht_str = result
    return int(score), ht_str
