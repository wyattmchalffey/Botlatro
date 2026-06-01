"""Tests for the candidate-subset play policy (Stage 2.2). Gated on torch.

Overfits a tiny set where the chosen play is a flush in a flush-available hand —
the policy should rank that 5-card subset above random alternatives, proving the
candidate scorer (pooled subset embeddings + hand-type + context) wires up.
"""

from __future__ import annotations

import unittest

from balatro_ai.api.state import Card, GamePhase, GameState, Joker
from balatro_ai.ml.dataset import TrainingExample, ValueTarget
from balatro_ai.ml.encoding import encode_state

try:
    import torch  # noqa: F401

    from balatro_ai.ml.play_policy import (
        PlayPolicyConfig,
        eval_play_policy,
        train_play_policy,
    )

    _HAS_TORCH = True
except Exception:  # noqa: BLE001
    _HAS_TORCH = False

_HEARTS = ["2", "5", "7", "9", "J"]


def _flush_example(off_rank: str) -> TrainingExample:
    hand = (
        *[Card(r, "Hearts") for r in _HEARTS],
        Card(off_rank, "Spades"), Card("4", "Clubs"), Card("8", "Diamonds"),
    )
    state = GameState(
        phase=GamePhase.SELECTING_HAND, ante=2, money=10, required_score=1000,
        hands_remaining=3, discards_remaining=2, jokers=(Joker(name="Joker"),),
        hand=hand, deck_size=52, hand_levels={"Pair": 2})
    action = {"type": "play_hand", "card_indices": [0, 1, 2, 3, 4]}  # the flush
    return TrainingExample(0, state.phase.value, encode_state(state), action,
                           ValueTarget(False, 3, 1000), 1)


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestPlayPolicy(unittest.TestCase):
    def test_overfits_candidate_ranking(self) -> None:
        data = [_flush_example(r) for r in ("3", "6", "10", "Q", "K", "A", "3", "6")]
        model = train_play_policy(data, PlayPolicyConfig(epochs=200, lr=1e-2,
                                                         dropout=0.0, weight_decay=0.0,
                                                         n_neg=15))
        m = eval_play_policy(model, data, n_neg=15, seed=1)
        self.assertGreater(m["top1_acc"], 0.85)            # ranks the flush #1
        self.assertGreater(m["top1_acc"], m["random_baseline"] * 5)


if __name__ == "__main__":
    unittest.main()
