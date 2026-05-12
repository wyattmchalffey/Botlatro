"""Random legal-action baseline bot."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from balatro_ai.api.actions import Action
from balatro_ai.api.state import GameState


@dataclass(slots=True)
class RandomBot:
    """Choose uniformly from the provided legal actions."""

    seed: int | None = None
    name: str = "random_bot"
    rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def choose_action(self, state: GameState) -> Action:
        if not state.legal_actions:
            raise ValueError("RandomBot cannot choose an action without legal actions")
        return self.rng.choice(state.legal_actions)

