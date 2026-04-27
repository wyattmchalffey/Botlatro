"""Small Gym-like environment wrapper around a local Balatro client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from balatro_ai.api.actions import Action
from balatro_ai.api.client import BalatroClient
from balatro_ai.api.state import GameState
from balatro_ai.env.observations import vector_from_state
from balatro_ai.env.rewards import default_reward


@dataclass(slots=True)
class BalatroEnv:
    """A minimal reset/step wrapper that does not require Gymnasium yet."""

    client: BalatroClient
    stake: str = "white"
    state: GameState | None = None

    def reset(self, seed: int | None = None) -> tuple[tuple[float, ...], dict[str, Any]]:
        self.state = self.client.start_run(seed=seed, stake=self.stake)
        return vector_from_state(self.state), {"state": self.state}

    def step(self, action: Action) -> tuple[tuple[float, ...], float, bool, bool, dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("Call reset() before step()")

        previous = self.state
        self.state = self.client.send_action(action)
        reward = default_reward(previous, self.state)
        terminated = self.state.run_over
        truncated = False
        return vector_from_state(self.state), reward, terminated, truncated, {"state": self.state}

    def legal_actions(self) -> tuple[Action, ...]:
        if self.state is None:
            raise RuntimeError("Call reset() before legal_actions()")
        return self.state.legal_actions

