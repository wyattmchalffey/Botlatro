from __future__ import annotations

import unittest

import context  # noqa: F401
from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GamePhase, GameState
from balatro_ai.env.balatro_env import BalatroEnv


class FakeClient:
    def __init__(self) -> None:
        self.action = Action(ActionType.NO_OP)

    def start_run(self, seed: int | None = None, stake: str | None = None) -> GameState:
        return GameState(
            phase=GamePhase.PLAYING_BLIND,
            seed=seed,
            stake=stake or "white",
            current_score=0,
            legal_actions=(self.action,),
        )

    def get_state(self) -> GameState:
        return GameState(phase=GamePhase.PLAYING_BLIND, legal_actions=(self.action,))

    def send_action(self, action: Action) -> GameState:
        return GameState(
            phase=GamePhase.RUN_OVER,
            current_score=1000,
            run_over=True,
            won=True,
            legal_actions=(),
        )


class EnvTests(unittest.TestCase):
    def test_reset_and_step(self) -> None:
        env = BalatroEnv(client=FakeClient())
        observation, info = env.reset(seed=123)

        self.assertEqual(info["state"].seed, 123)
        self.assertGreater(len(observation), 0)

        action = env.legal_actions()[0]
        _, reward, terminated, truncated, step_info = env.step(action)

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(step_info["state"].won)
        self.assertGreater(reward, 0)


if __name__ == "__main__":
    unittest.main()

