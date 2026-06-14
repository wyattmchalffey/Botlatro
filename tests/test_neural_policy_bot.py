"""Tests for the deployed decision-shaped policy bot (Phase B component 5)."""

from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace

from balatro_ai.api.actions import ActionType
from balatro_ai.api.state import with_derived_legal_actions
from balatro_ai.bots.config import DEFAULT_CONFIG, bot_config_scope
from balatro_ai.bots.registry import create_bot
from balatro_ai.sim.local_runner import LocalBalatroSimulator
from balatro_ai.solver.seed_game import SeedGame
from balatro_ai.solver.trajectory import _stable_seed_int


def _run(bot, seed: str, steps: int = 60) -> int:
    sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
    sim.state = with_derived_legal_actions(SeedGame(seed, stake="white").initial_state())
    n = 0
    with bot_config_scope(replace(DEFAULT_CONFIG, shop_audit_enabled=False)):
        for _ in range(steps):
            st = sim.state
            if st.run_over:
                break
            action = bot.choose_action(st)
            if action.action_type == ActionType.NO_OP:
                break
            # every returned action must be legal for the current state
            assert action.stable_key in {a.stable_key for a in st.legal_actions} or \
                st.phase.value in ("round_eval",), action.action_type
            sim.step(action)
            n += 1
    return n


class NeuralPolicyBotTests(unittest.TestCase):
    def test_fallback_without_checkpoint_runs(self) -> None:
        os.environ.pop("BALATRO_POLICY_CKPT", None)
        bot = create_bot("neural_policy_bot", seed=0)
        self.assertGreater(_run(bot, "0000009"), 0)  # falls back to heuristic, completes

    def test_bad_checkpoint_path_falls_back(self) -> None:
        os.environ["BALATRO_POLICY_CKPT"] = os.path.join(tempfile.gettempdir(), "_nope_xyz.pt")
        try:
            bot = create_bot("neural_policy_bot", seed=0)
            self.assertGreater(_run(bot, "0000009"), 0)
        finally:
            os.environ.pop("BALATRO_POLICY_CKPT", None)

    def test_trained_checkpoint_deploys_and_plays_legally(self) -> None:
        from balatro_ai.ml.dataset import capture_run, examples_from_capture
        from balatro_ai.ml.policy_net import PolicyConfig, save_policy, train_decision_policy

        src = create_bot("solver_shop_basic_play_bot", seed=0)
        exs = examples_from_capture(
            capture_run("0000003", src.choose_action, stake="white", max_steps=4000)
        )
        cfg = PolicyConfig(epochs=2, seed=0)
        net, _ = train_decision_policy(exs, cfg)
        ckpt = os.path.join(tempfile.gettempdir(), "_npb_test.pt")
        save_policy(net, ckpt, config=cfg)
        os.environ["BALATRO_POLICY_CKPT"] = ckpt
        try:
            bot = create_bot("neural_policy_bot", seed=0)
            self.assertGreater(_run(bot, "0000009"), 0)  # deploys, plays only legal actions
        finally:
            os.environ.pop("BALATRO_POLICY_CKPT", None)


if __name__ == "__main__":
    unittest.main()
