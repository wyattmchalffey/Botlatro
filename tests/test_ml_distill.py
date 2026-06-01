"""Tests for rollout distillation (Stage 1.3, Option A).

Gated on torch. Verifies the distillation training/eval wiring overfits a tiny
synthetic set whose `clear_probability` label depends on the joker set — so
fitting it proves `clear_head` + `clear_value` + `train_distill` + `eval_distill`
are wired end-to-end. The beam-driven `collect_distill_pairs` is exercised by the
`scripts/phase8_distill.py` run, not here (it needs the full solver).
"""

from __future__ import annotations

import unittest

from balatro_ai.api.state import Card, GamePhase, GameState, Joker
from balatro_ai.ml.encoding import encode_state

try:
    import torch  # noqa: F401

    from balatro_ai.ml.distill import eval_distill, train_distill

    _HAS_TORCH = True
except Exception:  # noqa: BLE001
    _HAS_TORCH = False


def _pairs(n: int = 16):
    pairs = []
    for i in range(n):
        bp = i % 2 == 0
        jokers = (Joker(name="Blueprint"),) if bp else (Joker(name="Joker"),)
        state = GameState(
            phase=GamePhase.SELECTING_HAND, ante=2, money=5 + 2 * i,
            required_score=1000, hands_remaining=3, discards_remaining=2,
            jokers=jokers, hand=(Card("A", "Spades"), Card("K", "Hearts")),
            deck_size=52, hand_levels={"Pair": 2})
        pairs.append((encode_state(state), 0.9 if bp else 0.2))
    return pairs


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestDistill(unittest.TestCase):
    def test_train_distill_overfits(self) -> None:
        pairs = _pairs(16)
        model = train_distill(pairs, epochs=250, lr=1e-2, dropout=0.0, weight_decay=0.0)
        metrics = eval_distill(model, pairs)
        self.assertGreater(metrics["corr"], 0.95)
        self.assertLess(metrics["mse"], 0.02)

    def test_clear_value_range(self) -> None:
        # clear_value is bounded to [0, 2] (the leaf convention).
        from balatro_ai.ml.model import ValueNet, collate_states
        out = ValueNet().clear_value(collate_states([p[0] for p in _pairs(4)]))
        self.assertTrue(bool((out >= 0).all()) and bool((out <= 2).all()))


if __name__ == "__main__":
    unittest.main()
