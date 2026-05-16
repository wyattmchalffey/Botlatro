"""Small blind-state helpers shared by blind decision modules."""

from __future__ import annotations

from balatro_ai.api.state import GameState


def _is_boss_blind(state: GameState) -> bool:
    if not state.blind:
        return False
    return state.blind not in {"Small Blind", "Big Blind"}
