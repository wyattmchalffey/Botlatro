from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import context  # noqa: F401
import torch

from balatro_ai.api.state import GamePhase, GameState
from balatro_ai.bots.basic_strategy.cache import decision_cache_scope
from balatro_ai.bots.basic_strategy import value_guidance


def joker_card(name: str, *, cost: int = 4) -> dict[str, object]:
    return {
        "key": f"j_{name.lower().replace(' ', '_')}",
        "name": name,
        "label": name,
        "set": "JOKER",
        "cost": {"buy": cost, "base": cost},
    }


class _FakeValueNet:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def eval(self) -> None:
        return None

    def ante_value(self, _batch) -> torch.Tensor:
        return torch.tensor([self._values.pop(0)], dtype=torch.float32)

    def win_prob(self, _batch) -> torch.Tensor:
        return self.ante_value(_batch)

    def clear_value(self, _batch) -> torch.Tensor:
        return self.ante_value(_batch)


class ValueGuidanceTests(unittest.TestCase):
    def tearDown(self) -> None:
        value_guidance._VALUE_NET_CACHE = None

    def test_value_guidance_is_off_by_default(self) -> None:
        state = GameState(phase=GamePhase.SHOP, money=10)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(value_guidance.value_bonus_for_card(state, joker_card("Joker")), 0.0)

    def test_value_guidance_can_use_current_value_net_checkpoint(self) -> None:
        state = GameState(phase=GamePhase.SHOP, money=10)
        model = _FakeValueNet([0.20, 0.35])
        env = {
            "BALATRO_VALUE_MODEL_CKPT": "dummy.pt",
            "BALATRO_VALUE_MODEL_HEAD": "ante",
            "BALATRO_VALUE_SCALE": "100",
        }

        with (
            patch.dict(os.environ, env, clear=True),
            patch("balatro_ai.ml.train.load_checkpoint", return_value=model),
        ):
            value_guidance._VALUE_NET_CACHE = None
            bonus = value_guidance.value_bonus_for_card(state, joker_card("Joker"))

        self.assertAlmostEqual(bonus, 15.0, places=5)

    def test_value_net_prediction_is_decision_cached_by_state_identity(self) -> None:
        state = GameState(phase=GamePhase.SHOP, money=10)
        model = _FakeValueNet([0.42])
        env = {
            "BALATRO_VALUE_MODEL_CKPT": "dummy.pt",
            "BALATRO_VALUE_MODEL_HEAD": "ante",
        }

        with (
            patch.dict(os.environ, env, clear=True),
            patch("balatro_ai.ml.train.load_checkpoint", return_value=model),
            decision_cache_scope(),
        ):
            value_guidance._VALUE_NET_CACHE = None
            first = value_guidance._value_net_predict(state)
            second = value_guidance._value_net_predict(state)

        self.assertEqual(first, second)
        self.assertEqual(model._values, [])


if __name__ == "__main__":
    unittest.main()
