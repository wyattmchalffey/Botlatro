"""Tests for the Stage 0.2 training-data pipeline (`ml/dataset.py`).

The core gate: a captured *thin* action log, when re-simulated, reproduces the
run exactly (so we can store action logs, not states) — and that property
survives JSON persistence. Uses `GreedyBot` (deterministic, scores hands, no
search) so runs clear early blinds and reach shops/packs, exercising the
metadata-carrying actions (`BUY`, `CHOOSE_PACK_CARD`) that the lossy
`StepRecord` could not replay.
"""

from __future__ import annotations

import json
import unittest

from balatro_ai.api.actions import Action, ActionType
from balatro_ai.bots.greedy_bot import GreedyBot
from balatro_ai.ml import dataset as ds
from balatro_ai.ml.encoding import ENCODING_VERSION
from balatro_ai.solver.trajectory import generate_trajectory

_SEED = "AAAAAAA"
_MAX_STEPS = 150


class _BuyGreedy:
    """Greedy, but buys the first affordable shop item so captures exercise the
    `BUY` action (amount + metadata) — the case `StepRecord` can't replay."""

    def __init__(self) -> None:
        self._greedy = GreedyBot(seed=0)

    def choose_action(self, state) -> Action:
        buys = [a for a in state.legal_actions if a.action_type == ActionType.BUY]
        if buys:
            return buys[0]
        return self._greedy.choose_action(state)


def _capture(policy=None):
    policy = policy or _BuyGreedy()
    return ds.capture_run(_SEED, policy.choose_action, max_steps=_MAX_STEPS)


class TestCapture(unittest.TestCase):
    def test_capture_produces_replay_complete_log(self) -> None:
        cap = _capture()
        self.assertGreater(cap.n_steps, 0)
        self.assertIn(cap.terminated_reason, ds.TERMINAL_REASONS)
        self.assertEqual(len(cap.step_summaries), cap.n_steps)
        # actions serialize with a type field.
        self.assertTrue(all("type" in a for a in cap.actions))

    def test_determinism(self) -> None:
        self.assertEqual(_capture().to_json_dict(), _capture().to_json_dict())

    def test_multi_phase_coverage(self) -> None:
        cap = _capture()
        types = {a["type"] for a in cap.actions}
        self.assertIn(ActionType.SELECT_BLIND.value, types)
        self.assertIn(ActionType.PLAY_HAND.value, types)
        # Reached the shop/cash-out machinery, not just the opening blind.
        self.assertGreaterEqual(len(types), 4)


class TestRoundTrip(unittest.TestCase):
    def test_roundtrip_exact(self) -> None:
        """THE GATE: re-simulating the log reproduces the run exactly."""
        result = ds.verify_capture_roundtrip(_capture())
        self.assertTrue(result.ok, msg=f"mismatches: {result.mismatches[:5]}")
        self.assertGreater(result.n_steps, 0)

    def test_roundtrip_survives_json(self) -> None:
        cap = _capture()
        rehydrated = ds.RunCapture.from_json_dict(json.loads(json.dumps(cap.to_json_dict())))
        self.assertEqual(rehydrated, cap)
        # The persisted thin log is still replay-complete.
        self.assertTrue(ds.verify_capture_roundtrip(rehydrated).ok)

    def test_replay_states_yields_states_and_actions(self) -> None:
        cap = _capture()
        pairs = list(ds.replay_states(cap.seed, cap.actions, stake=cap.stake))
        self.assertEqual(len(pairs), cap.n_steps)
        state0, action0 = pairs[0]
        self.assertIsInstance(action0, Action)
        self.assertEqual(action0.to_json(), cap.actions[0])


class TestExamples(unittest.TestCase):
    def test_examples_encode_and_label(self) -> None:
        cap = _capture()
        examples = ds.examples_from_capture(cap)
        self.assertEqual(len(examples), cap.n_steps)
        # Value target = run outcome, shared by every step.
        for ex in examples:
            self.assertEqual(ex.encoded_state.version, ENCODING_VERSION)
            self.assertEqual(ex.value.won, cap.won)
            self.assertEqual(ex.value.final_ante, cap.final_ante)
            self.assertIn("type", ex.action)
        # steps_to_end counts down n..1.
        self.assertEqual([ex.steps_to_end for ex in examples],
                         list(range(cap.n_steps, 0, -1)))
        # build_examples is the capture+expand shortcut.
        self.assertEqual(len(ds.build_examples(_SEED, _BuyGreedy().choose_action,
                                               max_steps=_MAX_STEPS)),
                         cap.n_steps)


class TestCandidatesSchemaV2(unittest.TestCase):
    """Schema v2: every example carries the legal-candidate set + the index of
    the taken action — the decision-shaped policy's training input."""

    def test_candidates_populated_and_chosen_matches_action(self) -> None:
        examples = ds.examples_from_capture(_capture())
        matched = 0
        for ex in examples:
            if ex.chosen_index >= 0:
                matched += 1
                self.assertLess(ex.chosen_index, len(ex.candidates))
                chosen = ex.candidates[ex.chosen_index]
                self.assertEqual(
                    chosen.action_type_index,
                    ds._ACTION_TYPE_INDEX[ActionType(ex.action["type"])],
                )
        self.assertGreater(matched / max(1, len(examples)), 0.9)

    def test_missing_feature_flag_is_honest(self) -> None:
        for ex in ds.examples_from_capture(_capture()):
            for c in ex.candidates:
                if not c.has_play_score:
                    self.assertEqual(c.play_score, 0.0)


class TestFaithfulToGenerator(unittest.TestCase):
    def test_capture_matches_generate_trajectory(self) -> None:
        """capture_run reproduces the canonical generate_trajectory run."""
        cap = ds.capture_run(_SEED, GreedyBot(seed=0).choose_action, max_steps=_MAX_STEPS)
        traj = generate_trajectory(_SEED, GreedyBot(seed=0).choose_action, max_steps=_MAX_STEPS)
        self.assertEqual([a["type"] for a in cap.actions],
                         [s.action_type for s in traj.steps])
        self.assertEqual(cap.won, traj.won)
        self.assertEqual(cap.final_ante, traj.final_ante)
        self.assertEqual(cap.final_score, traj.final_score)


if __name__ == "__main__":
    unittest.main()
