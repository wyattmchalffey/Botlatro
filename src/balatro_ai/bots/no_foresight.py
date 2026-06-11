"""Information-set ablation: hide the sim's future-draw order from bots.

In sim states, ``state.known_deck`` is the TRUE post-shuffle draw order
(``local_runner`` deals ``known_deck[:count]``), while the live bridge
supplies ``known_deck=()``. Any heuristic that reads ``known_deck[:n]`` is
therefore clairvoyant in sim and blind on the bridge, so sim winrates are an
upper bound on bridge winrates and every sim-based decision audit certifies a
clairvoyant player (see SUPERHUMAN_ROADMAP.md, P0.1).

``BALATRO_NO_FORESIGHT`` selects the information set the bot observes:

- ``shuffle`` (or ``1``): the remaining-deck MULTISET is preserved — that is
  legitimately visible in-game via the deck view — but the order is replaced
  with a deterministic belief permutation independent of the true order.
- ``hide``: ``known_deck=()`` — what the bridge provides today.
- unset / ``0`` / ``off``: no-op (current clairvoyant behavior).

Blinding happens once at the bot's observation boundary (``choose_action``),
so the play heuristic, the solver's shop search, and every rollout forked from
the observed state plan against the same honest belief. The runner's true
state is never touched.
"""

from __future__ import annotations

import hashlib
import os
import random
from collections import OrderedDict
from dataclasses import replace

from balatro_ai.api.state import GameState

_MODE_ENV = "BALATRO_NO_FORESIGHT"

# Identity pin for already-blinded outputs: composed bots blind at the wrapper
# AND again in the delegated BasicStrategyBot.choose_action. The second call's
# output is provably identical (the belief seed is multiset-invariant and the
# canonical sort erases the first shuffle), so just return the same object.
# `is` guard makes an id-reuse-after-GC collision impossible.
_BLINDED_PIN: "OrderedDict[int, GameState]" = OrderedDict()
_BLINDED_PIN_MAX = 16


def no_foresight_mode() -> str:
    """Return '', 'shuffle', or 'hide' from the env (validated)."""
    raw = os.environ.get(_MODE_ENV, "").strip().lower()
    if raw in ("", "0", "off", "none"):
        return ""
    if raw in ("1", "shuffle"):
        return "shuffle"
    if raw == "hide":
        return "hide"
    raise ValueError(f"{_MODE_ENV} must be unset, 'shuffle', or 'hide'; got {raw!r}")


def blind_known_deck(state: GameState) -> GameState:
    """Return ``state`` with the future-draw order hidden per the env mode.

    No-op (returns the same object) when the mode is off or there is nothing
    to hide.
    """
    mode = no_foresight_mode()
    if not mode or not state.known_deck:
        return state
    if _BLINDED_PIN.get(id(state)) is state:
        return state
    if mode == "hide":
        return replace(state, known_deck=())
    # Sort to a canonical order first so the belief permutation is a pure
    # function of the multiset — no trace of the true order survives.
    cards = sorted(state.known_deck, key=repr)
    random.Random(_belief_seed(state)).shuffle(cards)
    blinded = replace(state, known_deck=tuple(cards))
    _BLINDED_PIN[id(blinded)] = blinded
    if len(_BLINDED_PIN) > _BLINDED_PIN_MAX:
        _BLINDED_PIN.popitem(last=False)
    return blinded


def _belief_seed(state: GameState) -> int:
    """Deterministic, PYTHONHASHSEED-independent seed for the belief order.

    Keyed on decision-visible features (not the true order) so repeated calls
    on the same decision agree, while successive decisions get fresh
    permutations.
    """
    key = "|".join(
        (
            str(state.phase),
            str(state.ante),
            state.blind,
            str(state.current_score),
            str(state.required_score),
            str(state.money),
            str(state.hands_remaining),
            str(state.discards_remaining),
            str(state.deck_size),
            ",".join(sorted(repr(card) for card in state.hand)),
            ",".join(sorted(repr(card) for card in state.known_deck)),
        )
    )
    return int.from_bytes(hashlib.md5(key.encode("utf-8")).digest()[:8], "big")
