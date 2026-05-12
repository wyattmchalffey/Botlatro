"""Bot interface."""

from __future__ import annotations

from typing import Protocol

from balatro_ai.api.actions import Action
from balatro_ai.api.state import GameState


class Bot(Protocol):
    name: str

    def choose_action(self, state: GameState) -> Action:
        """Choose one legal action for the current state."""

