"""Test the neural candidate provider wires into the v2 beam (Stage 2.3). Torch-gated."""

from __future__ import annotations

import unittest

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import Card, GamePhase, GameState

try:
    import torch  # noqa: F401

    from balatro_ai.ml.model import ValueNet
    from balatro_ai.ml.neural_search import PolicyCandidateProvider

    _HAS_TORCH = True
except Exception:  # noqa: BLE001
    _HAS_TORCH = False


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestNeuralSearch(unittest.TestCase):
    def test_provider_prunes_to_top_k(self) -> None:
        hand = tuple(Card(r, "Hearts") for r in ("2", "3", "4", "5", "6", "7", "8", "9"))
        plays = tuple(
            Action(ActionType.PLAY_HAND, card_indices=idx)
            for idx in [(0,), (1,), (2,), (0, 1), (0, 1, 2), (2, 3, 4),
                        (0, 1, 2, 3), (0, 1, 2, 3, 4), (3, 4, 5, 6, 7)])
        state = GameState(
            phase=GamePhase.SELECTING_HAND, hand=hand, legal_actions=plays,
            required_score=1000, current_score=0, hands_remaining=3, discards_remaining=2)
        provider = PolicyCandidateProvider(ValueNet(), play_oversample=2)
        out = provider(state, width=2)
        self.assertGreater(len(out), 0)
        self.assertLessEqual(len(out), 2 * 2)  # capped to width * play_oversample
        self.assertTrue(all(a.action_type == ActionType.PLAY_HAND for a in out))
        self.assertTrue(all(a in plays for a in out))


if __name__ == "__main__":
    unittest.main()
