"""Reward helpers for the first environment wrapper."""

from __future__ import annotations

from balatro_ai.api.state import GameState


def default_reward(previous: GameState, current: GameState) -> float:
    """A conservative early reward for smoke tests and baseline runs."""

    reward = 0.0
    reward += max(0, current.current_score - previous.current_score) / 1000.0
    reward += max(0, current.ante - previous.ante) * 1.0
    reward += max(0, current.money - previous.money) * 0.02

    if current.run_over and current.won:
        reward += 10.0
    elif current.run_over:
        reward -= 1.0

    return reward

