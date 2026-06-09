from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

import context  # noqa: F401
from balatro_ai.api.actions import Action, ActionType
from balatro_ai.api.state import GamePhase, GameState
from balatro_ai.ml.encoding import encode_state
from balatro_ai.ml.shop_candidate_dataset import action_key
from balatro_ai.ml import shop_ranker as sr


def _shop_state() -> GameState:
    return GameState(
        phase=GamePhase.SHOP,
        ante=8,
        money=24,
        modifiers={
            "shop_cards": (
                {"key": "j_joker", "name": "Joker", "set": "JOKER", "cost": {"buy": 4}},
            ),
            "booster_packs": (
                {"key": "p_buffoon_normal_1", "name": "Buffoon Pack", "set": "BOOSTER", "cost": {"buy": 4}},
            ),
        },
    )


def _candidate(action: Action, values: list[float], *, heuristic: bool = False) -> dict:
    half = len(values) // 2
    return {
        "action_key": action_key(action),
        "action": action.to_json(),
        "shop_token_index": 1 if action.action_type == ActionType.OPEN_PACK else -1,
        "rollout_values": values,
        "mean_value": sum(values) / len(values),
        "first_half_mean": sum(values[:half]) / half,
        "second_half_mean": sum(values[half:]) / half,
        "rank": 1,
        "is_heuristic_action": heuristic,
    }


def _record(
    candidates: list[dict],
    *,
    seed: str = "0000001",
    state_index: int = 3,
    source_bot: str = "solver_shop_basic_play_bot",
) -> dict:
    state = _shop_state()
    best = max(candidates, key=lambda candidate: candidate["mean_value"])
    return {
        "seed": seed,
        "state_index": state_index,
        "source_bot": source_bot,
        "ante": state.ante,
        "money": state.money,
        "encoded_state": asdict(encode_state(state)),
        "heuristic_action_key": next(
            (candidate["action_key"] for candidate in candidates if candidate["is_heuristic_action"]),
            None,
        ),
        "best_action_key": best["action_key"],
        "candidates": candidates,
    }


def _clear_record(**kwargs) -> dict:
    baseline = Action(ActionType.END_SHOP)
    pack = Action(ActionType.OPEN_PACK, target_id="pack", amount=0, metadata={"kind": "pack", "index": 0})
    buy = Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0})
    return _record(
        [
            _candidate(baseline, [3.0, 3.0, 3.0, 3.0], heuristic=True),
            _candidate(pack, [4.0, 4.0, 4.0, 4.0]),
            _candidate(buy, [2.0, 2.0, 2.0, 2.0]),
        ],
        **kwargs,
    )


def _ambiguous_record() -> dict:
    baseline = Action(ActionType.END_SHOP)
    positive = Action(ActionType.OPEN_PACK, target_id="pack", amount=0, metadata={"kind": "pack", "index": 0})
    negative = Action(ActionType.BUY, target_id="card", amount=0, metadata={"kind": "card", "index": 0})
    ambiguous = Action(ActionType.REROLL)
    return _record(
        [
            _candidate(baseline, [3.0, 3.0, 3.0, 3.0], heuristic=True),
            _candidate(positive, [4.0, 4.0, 4.0, 4.0]),
            _candidate(negative, [2.0, 2.0, 2.0, 2.0]),
            _candidate(ambiguous, [3.3, 2.7, 3.3, 2.7]),
        ]
    )


class FixedLogitModel(nn.Module):
    def __init__(self, rows: list[list[float]]) -> None:
        super().__init__()
        self.rows = rows

    def forward(self, batch: sr.ShopRankerBatch) -> torch.Tensor:
        logits = torch.full(batch.candidate_mask.shape, -1e9, dtype=torch.float32)
        for row, values in enumerate(self.rows):
            logits[row, : len(values)] = torch.tensor(values, dtype=torch.float32)
        return logits


class ShopRankerTests(unittest.TestCase):
    def test_example_from_record_populates_baseline_and_confidence_features(self) -> None:
        example = sr.example_from_record(_clear_record())

        self.assertIsNotNone(example)
        assert example is not None
        self.assertEqual(example.target_index, 1)
        self.assertEqual(example.baseline_index, 0)
        self.assertEqual(example.baseline_value, 3.0)
        self.assertEqual(example.advantages, (0.0, 1.0, -1.0))
        self.assertEqual(example.advantage_rollout_counts, (4, 4, 4))
        self.assertEqual(example.advantage_positive_rates, (0.0, 1.0, 0.0))
        self.assertAlmostEqual(example.best_vs_baseline_lcb, 1.0)
        self.assertEqual(example.high_conf_override_candidates, 1)
        self.assertEqual(example.high_conf_practical_override_candidates, 1)

    def test_examples_from_jsonl_paths_merges_deeper_duplicate_candidate_labels(self) -> None:
        shallow = _clear_record()
        deep = _clear_record()
        deep["candidates"][1] = {
            **deep["candidates"][1],
            "rollout_values": [5.0] * 8,
            "mean_value": 5.0,
            "first_half_mean": 5.0,
            "second_half_mean": 5.0,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            first = Path(tmpdir) / "shallow.jsonl"
            second = Path(tmpdir) / "deep.jsonl"
            first.write_text(json.dumps(shallow), encoding="utf-8")
            second.write_text(json.dumps(deep), encoding="utf-8")

            examples = sr.examples_from_jsonl_paths([first, second])

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].mean_values[1], 5.0)
        self.assertEqual(examples[0].advantage_rollout_counts[1], 4)

    def test_examples_from_jsonl_paths_keeps_undeduped_records_without_seed(self) -> None:
        first = _clear_record(seed="")
        second = _clear_record(seed="")

        with tempfile.TemporaryDirectory() as tmpdir:
            first_path = Path(tmpdir) / "first.jsonl"
            second_path = Path(tmpdir) / "second.jsonl"
            first_path.write_text(json.dumps(first), encoding="utf-8")
            second_path.write_text(json.dumps(second), encoding="utf-8")

            examples = sr.examples_from_jsonl_paths([first_path, second_path])

        self.assertEqual(len(examples), 2)

    def test_collate_shop_ranker_examples_pads_candidates_and_keeps_masks(self) -> None:
        example_a = sr.example_from_record(_ambiguous_record())
        example_b = sr.example_from_record(_clear_record())
        assert example_a is not None
        assert example_b is not None

        batch = sr.collate_shop_ranker_examples([example_a, example_b])

        self.assertEqual(tuple(batch.action_type.shape), (2, 4))
        self.assertEqual(tuple(batch.numeric.shape), (2, 4, 3))
        self.assertTrue(bool(batch.candidate_mask[0, 3]))
        self.assertFalse(bool(batch.candidate_mask[1, 3]))
        self.assertEqual(batch.baseline_index.tolist(), [0, 0])
        self.assertEqual(batch.target.tolist(), [1, 1])

    def test_confidence_advantage_label_summary_counts_clear_and_ambiguous_labels(self) -> None:
        example = sr.example_from_record(_ambiguous_record())
        assert example is not None

        summary = sr.confidence_advantage_label_summary([example], margin=0.10)

        self.assertEqual(summary["candidate_labels"], 3)
        self.assertEqual(summary["positive_labels"], 1)
        self.assertEqual(summary["negative_labels"], 1)
        self.assertEqual(summary["ambiguous_labels"], 1)
        self.assertEqual(summary["records_with_positive_rate"], 1.0)

    def test_filter_examples_by_label_quality_uses_confidence_fields(self) -> None:
        clear = sr.example_from_record(_clear_record())
        baseline = Action(ActionType.END_SHOP)
        marginal = Action(ActionType.OPEN_PACK, target_id="pack", amount=0, metadata={"kind": "pack", "index": 0})
        flat = sr.example_from_record(
            _record(
                [
                    _candidate(baseline, [3.0, 3.0, 3.0, 3.0], heuristic=True),
                    _candidate(marginal, [3.05, 3.05, 3.05, 3.05]),
                ],
                seed="0000002",
            )
        )
        assert clear is not None
        assert flat is not None

        filtered = sr.filter_examples_by_label_quality(
            [clear, flat],
            min_best_vs_baseline_lcb=0.9,
            min_high_conf_practical_override_candidates=1,
            max_actions_within_0_10=1,
        )

        self.assertEqual(filtered, [clear])

    def test_heuristic_baseline_reports_regret_and_near_best_rates(self) -> None:
        example = sr.example_from_record(_clear_record())
        assert example is not None

        metrics = sr.evaluate_heuristic_baseline([example])

        self.assertEqual(metrics["covered_n"], 1)
        self.assertEqual(metrics["top1_acc"], 0.0)
        self.assertEqual(metrics["mean_regret"], 1.0)
        self.assertEqual(metrics["near_best_acc_0_10"], 0.0)

    def test_evaluate_advantage_override_tracks_helpful_and_harmful_overrides(self) -> None:
        helpful = sr.example_from_record(_clear_record())
        baseline = Action(ActionType.END_SHOP)
        worse = Action(ActionType.REROLL)
        harmful = sr.example_from_record(
            _record(
                [
                    _candidate(baseline, [4.0, 4.0, 4.0, 4.0], heuristic=True),
                    _candidate(worse, [3.0, 3.0, 3.0, 3.0]),
                ],
                seed="0000002",
            )
        )
        assert helpful is not None
        assert harmful is not None

        metrics = sr.evaluate_advantage_override(
            FixedLogitModel([[0.0, 1.0, -1.0], [0.0, 1.0]]),
            [helpful, harmful],
            threshold=0.5,
        )

        self.assertEqual(metrics["covered_n"], 2)
        self.assertEqual(metrics["override_rate"], 1.0)
        self.assertEqual(metrics["helpful_override_rate"], 0.5)
        self.assertEqual(metrics["harmful_override_rate"], 0.5)
        self.assertEqual(metrics["helpful_covered_rate"], 0.5)
        self.assertEqual(metrics["harmful_covered_rate"], 0.5)

    def test_evaluate_advantage_overrides_keys_thresholds_stably(self) -> None:
        example = sr.example_from_record(_clear_record())
        assert example is not None

        metrics = sr.evaluate_advantage_overrides(FixedLogitModel([[0.0, 1.0, -1.0]]), [example], thresholds=(0, 0.1))

        self.assertEqual(set(metrics), {"0.000", "0.100"})

    def test_split_examples_by_seed_keeps_seed_groups_together(self) -> None:
        examples = []
        for seed in ("a", "a", "b", "b", "c", "c"):
            example = sr.example_from_record(_clear_record(seed=seed, state_index=len(examples)))
            assert example is not None
            examples.append(example)

        train, val = sr.split_examples(examples, 0.34, seed=1, split_by="seed")
        train_seeds = {example.seed for example in train}
        val_seeds = {example.seed for example in val}

        self.assertFalse(train_seeds & val_seeds)
        self.assertTrue(train)
        self.assertTrue(val)

    def test_parse_helpers_accept_csv_values(self) -> None:
        self.assertEqual(
            {action.value for action in sr.parse_action_types_csv("buy, OPEN_PACK, buy")},
            {"buy", "open_pack"},
        )
        self.assertEqual(sr.parse_action_kinds_csv("card, pack, card"), frozenset({"card", "pack"}))

    def test_train_shop_ranker_rejects_empty_examples(self) -> None:
        with self.assertRaises(ValueError):
            sr.train_shop_ranker([])


if __name__ == "__main__":
    unittest.main()
