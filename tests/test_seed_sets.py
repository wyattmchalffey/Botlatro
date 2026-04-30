from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import context  # noqa: F401
from balatro_ai.eval.seed_sets import SeedSet, make_explicit_seed_set, make_seed_set, parse_seed_values


class SeedSetTests(unittest.TestCase):
    def test_make_seed_set_is_deterministic(self) -> None:
        first = make_seed_set("white:fast", 10)
        second = make_seed_set("white:fast", 10)

        self.assertEqual(first, second)
        self.assertEqual(len(set(first.seeds)), 10)

    def test_primary_benchmark_seed_label_is_stable(self) -> None:
        seed_set = make_seed_set("white:primary-score-audit-100", 5)

        self.assertEqual(
            seed_set.seeds,
            (349774307, 380312572, 2059837045, 685623659, 2097732365),
        )

    def test_seed_set_round_trip(self) -> None:
        seed_set = SeedSet(label="demo", seeds=(1, 2, 3))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seeds.json"
            seed_set.save(path)
            loaded = SeedSet.load(path)

        self.assertEqual(loaded, seed_set)

    def test_parse_seed_values_accepts_common_separators(self) -> None:
        self.assertEqual(parse_seed_values("123, 456\n789;10"), (123, 456, 789, 10))

    def test_make_explicit_seed_set_rejects_duplicates(self) -> None:
        with self.assertRaises(ValueError):
            make_explicit_seed_set("demo", (1, 1))


if __name__ == "__main__":
    unittest.main()
