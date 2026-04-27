from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import context  # noqa: F401
from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.client import BalatroBridgeError
from balatro_ai.api.state import GamePhase, GameState
from balatro_ai.bots.random_bot import RandomBot
from balatro_ai.eval.run_seed import RunSeedOptions, run_single_seed


class TwoStepClient:
    def __init__(self) -> None:
        self.action = Action(ActionType.NO_OP)
        self.steps = 0

    def start_run(self, seed: int | None = None, stake: str | None = None) -> GameState:
        return GameState(
            phase=GamePhase.PLAYING_BLIND,
            seed=seed,
            ante=1,
            current_score=0,
            money=4,
            legal_actions=(self.action,),
        )

    def get_state(self) -> GameState:
        return GameState(phase=GamePhase.PLAYING_BLIND, legal_actions=(self.action,))

    def send_action(self, action: Action) -> GameState:
        self.steps += 1
        if self.steps == 1:
            return GameState(
                phase=GamePhase.PLAYING_BLIND,
                ante=1,
                current_score=500,
                money=4,
                legal_actions=(self.action,),
            )
        return GameState(
            phase=GamePhase.RUN_OVER,
            ante=2,
            current_score=1200,
            money=8,
            run_over=True,
            won=True,
        )


class StaleStateClient:
    def __init__(self) -> None:
        self.select = Action(ActionType.SELECT_BLIND)
        self.no_op = Action(ActionType.NO_OP)
        self.refreshed = False

    def start_run(self, seed: int | None = None, stake: str | None = None) -> GameState:
        return GameState(
            phase=GamePhase.BLIND_SELECT,
            seed=seed,
            ante=1,
            legal_actions=(self.select,),
        )

    def get_state(self) -> GameState:
        self.refreshed = True
        return GameState(
            phase=GamePhase.PLAYING_BLIND,
            ante=1,
            money=2,
            legal_actions=(self.no_op,),
        )

    def send_action(self, action: Action) -> GameState:
        if not self.refreshed:
            raise BalatroBridgeError(
                {
                    "data": {"name": "INVALID_STATE"},
                    "code": -32002,
                    "message": "Method 'select' requires one of these states: BLIND_SELECT",
                }
            )
        return GameState(
            phase=GamePhase.RUN_OVER,
            ante=1,
            current_score=100,
            money=2,
            run_over=True,
        )


class RunSeedTests(unittest.TestCase):
    def test_run_single_seed_returns_result(self) -> None:
        result = run_single_seed(
            bot=RandomBot(seed=123),
            client=TwoStepClient(),
            options=RunSeedOptions(seed=123, max_steps=10),
        )

        self.assertTrue(result.won)
        self.assertEqual(result.seed, 123)
        self.assertEqual(result.ante_reached, 2)
        self.assertEqual(result.final_money, 8)

    def test_run_single_seed_recovers_from_stale_invalid_state(self) -> None:
        result = run_single_seed(
            bot=RandomBot(seed=123),
            client=StaleStateClient(),
            options=RunSeedOptions(seed=123, max_steps=10),
        )

        self.assertFalse(result.won)
        self.assertEqual(result.seed, 123)
        self.assertEqual(result.ante_reached, 1)

    def test_replay_mode_off_does_not_write_replay_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory) / "run.jsonl"
            run_single_seed(
                bot=RandomBot(seed=123),
                client=TwoStepClient(),
                options=RunSeedOptions(seed=123, max_steps=10, replay_path=replay_path, replay_mode="off"),
            )

            self.assertFalse(replay_path.exists())

    def test_light_replay_writes_without_score_audit_extra(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory) / "run.jsonl"
            run_single_seed(
                bot=RandomBot(seed=123),
                client=TwoStepClient(),
                options=RunSeedOptions(seed=123, max_steps=10, replay_path=replay_path, replay_mode="light"),
            )

            text = replay_path.read_text(encoding="utf-8")
            self.assertIn('"chosen_action"', text)
            self.assertNotIn('"score_audit"', text)

    def test_unknown_replay_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            run_single_seed(
                bot=RandomBot(seed=123),
                client=TwoStepClient(),
                options=RunSeedOptions(seed=123, max_steps=10, replay_mode="loud"),
            )


if __name__ == "__main__":
    unittest.main()
