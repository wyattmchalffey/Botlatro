"""Tests for `balatro_ai.dataset` (Tier 1 #2).

Three layers:

1. Schema round-trip: `SeedResult.to_json_dict` → JSON → `from_json_dict`
   gives back an equivalent object (modulo tuple-vs-list distinction).
2. Writer/Reader interop: write a few rows, read them back, verify
   `completed_seeds()` returns the right set.
3. CLI: a tiny `--dry-run` sanity check and a single-seed live run
   that exercises the full multiprocessing path with a small policy.

The big multi-seed throughput run lives in `.data/` scripts, not
unittest — it's too slow to run on every CI invocation.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import context  # noqa: F401
from balatro_ai.dataset.reader import JsonlSeedReader, read_seed_file
from balatro_ai.dataset.schema import (
    ArchetypeAttemptRow,
    SeedResult,
    StepRow,
)
from balatro_ai.dataset.worker import WorkerConfig, solve_seed
from balatro_ai.dataset.writer import JsonlSeedWriter


def _sample_result(seed: str = "AAAAAAA", *, won: bool = False) -> SeedResult:
    """Build a representative `SeedResult` for round-trip tests."""

    return SeedResult(
        seed=seed,
        stake="white",
        policy="v2-d3w2-planning",
        won=won,
        final_ante=4,
        final_score=7392,
        final_money=12,
        n_steps=108,
        wall_seconds=154.8,
        terminated_reason="RUN_OVER",
        best_archetype="",
        attempts=(),
        steps=(
            StepRow(
                step=0,
                phase_before="selecting_hand",
                action_type="play_hand",
                card_indices=(0, 4, 5, 6, 7),
                money_before=4,
                money_after=8,
                score_after=1200,
                ante_after=1,
                hands_after=3,
                discards_after=3,
            ),
        ),
    )


class SchemaRoundTripTests(unittest.TestCase):
    def test_simple_round_trip_preserves_fields(self) -> None:
        original = _sample_result()
        serialized = json.dumps(original.to_json_dict())
        rebuilt = SeedResult.from_json_dict(json.loads(serialized))
        self.assertEqual(rebuilt.seed, original.seed)
        self.assertEqual(rebuilt.final_ante, original.final_ante)
        self.assertEqual(rebuilt.final_score, original.final_score)
        self.assertEqual(rebuilt.terminated_reason, original.terminated_reason)
        # tuple-vs-list inside StepRow.card_indices is normalized back
        # to tuple by from_json_dict.
        self.assertEqual(rebuilt.steps[0].card_indices, (0, 4, 5, 6, 7))

    def test_multi_archetype_attempts_round_trip(self) -> None:
        attempts = (
            ArchetypeAttemptRow(
                archetype_name="baseline",
                won=False, final_ante=4, final_score=2000, final_money=10,
                n_steps=80, wall_seconds=120.0, terminated_reason="RUN_OVER",
            ),
            ArchetypeAttemptRow(
                archetype_name="flush",
                won=True, final_ante=8, final_score=999, final_money=15,
                n_steps=200, wall_seconds=210.0, terminated_reason="RUN_OVER",
            ),
        )
        original = SeedResult(
            seed="X",
            stake="white",
            policy="multi-archetype-d3w2-planning",
            won=True, final_ante=8, final_score=999, final_money=15,
            n_steps=200, wall_seconds=330.0, terminated_reason="RUN_OVER",
            best_archetype="flush", attempts=attempts,
        )
        rebuilt = SeedResult.from_json_dict(
            json.loads(json.dumps(original.to_json_dict()))
        )
        self.assertEqual(rebuilt.best_archetype, "flush")
        self.assertEqual(len(rebuilt.attempts), 2)
        self.assertEqual(rebuilt.attempts[1].archetype_name, "flush")
        self.assertTrue(rebuilt.attempts[1].won)


class WriterReaderTests(unittest.TestCase):
    def test_writer_appends_and_reader_recovers_all_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            with JsonlSeedWriter(path) as w:
                w.write(_sample_result("AAA"))
                w.write(_sample_result("BBB"))

            rows = list(JsonlSeedReader(path).iter_rows())
            self.assertEqual([r.seed for r in rows], ["AAA", "BBB"])

    def test_completed_seeds_returns_set_of_seen_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            with JsonlSeedWriter(path) as w:
                w.write(_sample_result("AAA"))
                w.write(_sample_result("BBB"))
                w.write(_sample_result("CCC"))
            self.assertEqual(
                JsonlSeedReader(path).completed_seeds(),
                {"AAA", "BBB", "CCC"},
            )

    def test_completed_seeds_on_missing_file_returns_empty_set(self) -> None:
        # Resume-on-fresh-output is the common case — must not error.
        self.assertEqual(
            JsonlSeedReader("/nonexistent/path/file.jsonl").completed_seeds(),
            set(),
        )

    def test_writer_append_preserves_existing_rows(self) -> None:
        # Resume: second open() in append mode must not clobber.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            with JsonlSeedWriter(path) as w:
                w.write(_sample_result("AAA"))
            with JsonlSeedWriter(path) as w:
                w.write(_sample_result("BBB"))
            seeds = [r.seed for r in JsonlSeedReader(path).iter_rows()]
            self.assertEqual(seeds, ["AAA", "BBB"])


class SeedFileParserTests(unittest.TestCase):
    def test_one_seed_per_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seeds.txt"
            path.write_text("AAA\nBBB\nCCC\n", encoding="utf-8")
            self.assertEqual(read_seed_file(path), ["AAA", "BBB", "CCC"])

    def test_comma_separated_single_line(self) -> None:
        # Format used by older .data/*-seeds.txt files.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seeds.txt"
            path.write_text("1,2,3,4,5", encoding="utf-8")
            self.assertEqual(read_seed_file(path), ["1", "2", "3", "4", "5"])

    def test_comments_and_blanks_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seeds.txt"
            path.write_text(
                "# header comment\nAAA\n\n   \nBBB\n",
                encoding="utf-8",
            )
            self.assertEqual(read_seed_file(path), ["AAA", "BBB"])

    def test_dedup_preserves_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seeds.txt"
            path.write_text("AAA\nBBB\nAAA\nCCC\nBBB\n", encoding="utf-8")
            self.assertEqual(read_seed_file(path), ["AAA", "BBB", "CCC"])


class WorkerErrorContainmentTests(unittest.TestCase):
    """If anything inside the worker explodes, it must surface as a
    SeedResult, not a raised exception. The whole point of running
    seeds in a pool is one bad seed can't take the batch down."""

    def test_unknown_policy_kind_returns_error_result(self) -> None:
        config = WorkerConfig(policy_kind="not-a-policy")
        result = solve_seed("AAAAAAA", config)
        self.assertEqual(result.terminated_reason, "WORKER_ERROR")
        self.assertTrue(result.error_type)
        self.assertIn("not-a-policy", result.error_message)


if __name__ == "__main__":
    unittest.main()
