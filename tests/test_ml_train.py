"""Tests for the value net + training harness (Stage 0.3).

Gated on torch (the only torch-dependent part of the ml layer). The core gate is
`overfit_check`: a tiny synthetic set whose label depends *only* on whether the
joker set contains "Blueprint" (money is varied but uninformative), so fitting it
proves the whole stack wires up — embeddings, masked set-pooling, trunk, loss,
and backward.
"""

from __future__ import annotations

import os
import tempfile
import unittest

from balatro_ai.api.state import Card, GamePhase, GameState, Joker
from balatro_ai.ml.dataset import TrainingExample, ValueTarget
from balatro_ai.ml.encoding import ENCODING_VERSION, encode_state, encoding_spec

try:
    import torch

    from balatro_ai.ml.model import ValueNet, collate_states
    from balatro_ai.ml.train import (
        TrainConfig,
        evaluate,
        load_checkpoint,
        overfit_check,
        save_checkpoint,
        train,
    )

    _HAS_TORCH = True
except Exception:  # noqa: BLE001
    _HAS_TORCH = False


def _example(has_blueprint: bool, money: int, won: bool) -> TrainingExample:
    jokers = (Joker(name="Blueprint"),) if has_blueprint else (Joker(name="Joker"),)
    state = GameState(
        phase=GamePhase.SHOP,
        ante=2,
        money=money,
        required_score=1000,
        current_score=0,
        hands_remaining=3,
        discards_remaining=2,
        jokers=jokers,
        hand=(Card("A", "Spades"), Card("K", "Hearts")),
        deck_size=52,
        hand_levels={"Pair": 2},
    )
    return TrainingExample(
        step=0,
        phase=state.phase.value,
        encoded_state=encode_state(state),
        action={"type": "end_shop"},
        value=ValueTarget(won=won, final_ante=8 if won else 3, final_score=1000),
        steps_to_end=1,
    )


def _synthetic(n: int = 16) -> list[TrainingExample]:
    # label == has Blueprint; money varied 5..(5+2n) as uninformative noise.
    return [_example(i % 2 == 0, money=5 + 2 * i, won=i % 2 == 0) for i in range(n)]


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestCollate(unittest.TestCase):
    def test_shapes(self) -> None:
        spec = encoding_spec()
        b = collate_states([ex.encoded_state for ex in _synthetic(4)])
        self.assertEqual(b.size, 4)
        self.assertEqual(tuple(b.scalars.shape), (4, spec["scalar_dim"]))
        self.assertEqual(tuple(b.hand_levels.shape), (4, spec["hand_levels_dim"]))
        self.assertEqual(b.joker_cat.shape[0], 4)
        self.assertEqual(b.joker_cat.shape[2], 2)
        self.assertEqual(b.joker_mask.dtype, torch.bool)
        self.assertEqual(tuple(b.boss.shape), (4,))

    def test_forward_finite_including_empty_sets(self) -> None:
        # An empty state (no jokers/hand/shop) must pool to zero, not NaN.
        empty = collate_states([encode_state(GameState(phase=GamePhase.BLIND_SELECT))])
        out = ValueNet()(empty)
        self.assertEqual(tuple(out.shape), (1,))
        self.assertTrue(bool(torch.isfinite(out).all()))


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestTraining(unittest.TestCase):
    def test_overfit_gate(self) -> None:
        """THE GATE: the net fits a tiny synthetic set (wiring works end-to-end)."""
        ok, loss, acc = overfit_check(_synthetic(16), TrainConfig(epochs=400, lr=1e-2))
        self.assertTrue(ok, msg=f"loss={loss:.4f} acc={acc:.3f}")
        self.assertEqual(acc, 1.0)
        self.assertLess(loss, 0.05)

    def test_train_is_deterministic(self) -> None:
        cfg = TrainConfig(epochs=20, seed=7)
        h1 = train(_synthetic(8), cfg).history
        h2 = train(_synthetic(8), cfg).history
        self.assertEqual(h1, h2)

    def test_eval_split(self) -> None:
        examples = _synthetic(16)
        tr, va = next(iter([(examples[4:], examples[:4])]))
        res = train(tr, TrainConfig(epochs=10), val_examples=va)
        self.assertIsNotNone(res.final_val_loss)
        self.assertEqual(len(res.history), 10)


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestCheckpoint(unittest.TestCase):
    def test_roundtrip(self) -> None:
        res = train(_synthetic(8), TrainConfig(epochs=5))
        probe = collate_states([ex.encoded_state for ex in _synthetic(3)])
        res.model.eval()
        before = res.model.win_prob(probe)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ckpt.pt")
            save_checkpoint(res.model, path)
            loaded = load_checkpoint(path)
        self.assertEqual(loaded.encoding_version, ENCODING_VERSION)
        torch.testing.assert_close(loaded.win_prob(probe), before)


if __name__ == "__main__":
    unittest.main()
