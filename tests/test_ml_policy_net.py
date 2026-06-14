"""Tests for the decision-shaped policy net (Phase B component 4)."""

from __future__ import annotations

import unittest

from balatro_ai.ml.dataset import CandidateToken, TrainingExample, ValueTarget
from balatro_ai.ml.encoding import encode_state
from balatro_ai.ml.policy_net import (
    PolicyConfig,
    collate_candidates,
    evaluate,
    train_decision_policy,
)
from balatro_ai.api.state import GamePhase, GameState


def _example(chosen: int, n_cands: int, won: bool, *, scored_first: bool = True) -> TrainingExample:
    """A synthetic example: n_cands candidates, the chosen one optionally
    carrying the highest play-score (a learnable signal)."""
    cands = []
    for j in range(n_cands):
        cands.append(
            CandidateToken(
                action_type_index=j % 3,
                n_cards=0.5,
                amount=0.0,
                has_target=0.0,
                play_score=(1.0 if j == chosen and scored_first else 0.1),
                has_play_score=1.0,
                heuristic_choice=(1.0 if j == chosen and scored_first else 0.0),
            )
        )
    return TrainingExample(
        step=0,
        phase="selecting_hand",
        encoded_state=encode_state(GameState(phase=GamePhase.SELECTING_HAND, ante=1)),
        action={"type": "play_hand"},
        value=ValueTarget(won=won, final_ante=8 if won else 4, final_score=0),
        steps_to_end=1,
        candidates=tuple(cands),
        chosen_index=chosen,
    )


class CollateTests(unittest.TestCase):
    def test_pads_and_masks_variable_candidate_counts(self) -> None:
        exs = [_example(0, 3, True), _example(2, 6, False)]
        cb = collate_candidates(exs)
        self.assertEqual(cb.cand_type.shape, (2, 6))          # padded to max
        self.assertEqual(int(cb.cand_mask[0].sum()), 3)       # first has 3 real
        self.assertEqual(int(cb.cand_mask[1].sum()), 6)
        self.assertEqual(cb.chosen.tolist(), [0, 2])


class TrainTests(unittest.TestCase):
    def test_learns_a_planted_signal_above_chance(self) -> None:
        # The chosen candidate always has the top play-score -> the net should
        # learn to rank it first well above 1/n_cands chance.
        exs = [_example(i % 5, 5, i % 2 == 0) for i in range(120)]
        _, metrics = train_decision_policy(exs, PolicyConfig(epochs=25, seed=0))
        self.assertGreater(metrics["top1"], metrics["chance"] * 2)

    def test_ablation_reported(self) -> None:
        exs = [_example(i % 4, 4, True) for i in range(40)]
        net, _ = train_decision_policy(exs, PolicyConfig(epochs=5, seed=0))
        m = evaluate(net, exs)
        self.assertIn("top1_no_heuristic", m)   # anti-shortcut ablation present
        self.assertIn("value_auc", m)

    def test_drops_unlabelled_examples(self) -> None:
        good = _example(0, 3, True)
        bad = _example(0, 3, True)
        object.__setattr__(bad, "chosen_index", -1)  # frozen dataclass
        net, m = train_decision_policy([good, bad], PolicyConfig(epochs=2, seed=0))
        self.assertEqual(m["n_examples"], 1)


if __name__ == "__main__":
    unittest.main()
