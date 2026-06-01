"""Tests for the policy head (Stage 2). Gated on torch.

Overfits a tiny synthetic set where the action TYPE depends on the joker set
(Blueprint -> play_hand, else -> discard) — fitting it proves the type head +
per-card pointer + imitation training wire end-to-end.
"""

from __future__ import annotations

import unittest

from balatro_ai.api.state import Card, GamePhase, GameState, Joker
from balatro_ai.ml.dataset import TrainingExample, ValueTarget
from balatro_ai.ml.encoding import encode_state

try:
    import torch  # noqa: F401

    from balatro_ai.ml.policy import PolicyConfig, eval_policy, train_policy

    _HAS_TORCH = True
except Exception:  # noqa: BLE001
    _HAS_TORCH = False


def _example(blueprint: bool) -> TrainingExample:
    jokers = (Joker(name="Blueprint"),) if blueprint else (Joker(name="Joker"),)
    state = GameState(
        phase=GamePhase.SELECTING_HAND, ante=2, money=10, required_score=1000,
        hands_remaining=3, discards_remaining=2, jokers=jokers,
        hand=(Card("A", "Spades"), Card("K", "Hearts"), Card("3", "Clubs")),
        deck_size=52, hand_levels={"Pair": 2})
    action = ({"type": "play_hand", "card_indices": [0, 1]} if blueprint
              else {"type": "discard", "card_indices": [2]})
    return TrainingExample(0, state.phase.value, encode_state(state), action,
                           ValueTarget(blueprint, 8 if blueprint else 3, 1000), 1)


def _synthetic(n: int = 16):
    return [_example(i % 2 == 0) for i in range(n)]


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestPolicy(unittest.TestCase):
    def test_overfits_type_and_cards(self) -> None:
        data = _synthetic(16)
        model = train_policy(data, PolicyConfig(epochs=300, lr=1e-2, dropout=0.0,
                                                weight_decay=0.0))
        m = eval_policy(model, data)
        self.assertEqual(m["type_acc"], 1.0)             # separable by joker
        self.assertGreater(m["card_pos_acc"], 0.9)       # learned the card pattern
        self.assertIn("subset_exact", m)


if __name__ == "__main__":
    unittest.main()
