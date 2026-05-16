from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import context  # noqa: F401
from balatro_ai.eval.seed_sets import (
    CANONICAL_200_SEED_SET_LABEL,
    CANONICAL_200_SOURCE_LABEL,
    SeedSet,
    make_benchmark_seed_set,
    make_canonical_200_seed_set,
    make_explicit_seed_set,
    make_seed_set,
    parse_seed_values,
)


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

    def test_200_seed_benchmark_set_is_canonical_regardless_of_label(self) -> None:
        first = make_seed_set("white:strict200", 200)
        second = make_seed_set("white:refactor_pre200", 200)
        source_first_200 = make_seed_set(CANONICAL_200_SOURCE_LABEL, 1000).seeds[:200]

        self.assertEqual(first, second)
        self.assertEqual(first, make_canonical_200_seed_set())
        self.assertEqual(first.label, CANONICAL_200_SEED_SET_LABEL)
        self.assertEqual(first.seeds, source_first_200)
        self.assertEqual(first.seeds[0], 2132984486)

    def test_alternate_200_seed_windows_come_from_same_1000_seed_anchor(self) -> None:
        source = make_seed_set(CANONICAL_200_SOURCE_LABEL, 1000).seeds

        second_window = make_canonical_200_seed_set(2)

        self.assertEqual(second_window.seeds, source[200:400])
        self.assertEqual(second_window.label, f"{CANONICAL_200_SOURCE_LABEL}:201-400")
        self.assertEqual(
            make_benchmark_seed_set("white:any-label", 200, seed_window=2),
            second_window,
        )

    def test_seed_window_requires_200_seed_benchmark_size(self) -> None:
        with self.assertRaises(ValueError):
            make_benchmark_seed_set("white:fast", 100, seed_window=2)

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
