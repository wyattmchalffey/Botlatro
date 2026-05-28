"""Tests for `balatro_ai.solver.trajectory` (Milestone M2).

Single-seed end-to-end smoke that `generate_trajectory` drives
`basic_strategy_bot` from a `SeedGame`'s initial state through `RUN_OVER`
without crashing or getting stuck on legitimate state-signature repeats.
"""

from __future__ import annotations

import unittest

import context  # noqa: F401
from balatro_ai.bots.basic_strategy_bot import BasicStrategyBot
from balatro_ai.solver.trajectory import generate_trajectory


class TrajectoryGenerationTests(unittest.TestCase):
    def test_single_seed_runs_to_completion(self) -> None:
        bot = BasicStrategyBot(seed=0)
        traj = generate_trajectory("AAAAAAA", bot.choose_action, max_steps=2000)
        # The bot must finish naturally — not via the stuck detector or
        # step-limit guard. If this regresses, the signature/threshold in
        # generate_trajectory needs another look.
        self.assertEqual(
            traj.terminated_reason,
            "RUN_OVER",
            f"Expected RUN_OVER, got {traj.terminated_reason} after {traj.n_steps} steps",
        )
        self.assertGreater(traj.n_steps, 50, "trajectory cut off too early")
        self.assertGreaterEqual(traj.final_ante, 1)

    def test_trajectory_records_step_count(self) -> None:
        bot = BasicStrategyBot(seed=0)
        traj = generate_trajectory(
            "AAAAAAA", bot.choose_action, max_steps=2000, record_steps=True
        )
        self.assertEqual(len(traj.steps), traj.n_steps)

    def test_record_steps_false_skips_recording(self) -> None:
        bot = BasicStrategyBot(seed=0)
        traj = generate_trajectory(
            "AAAAAAA", bot.choose_action, max_steps=2000, record_steps=False
        )
        self.assertEqual(traj.steps, ())
        self.assertGreater(traj.sim_step_count, 0)

    def test_consecutive_discards_do_not_trigger_stuck_detection(self) -> None:
        # AAAAAAA's first blind opens with multiple discards in a row from
        # basic_strategy_bot. The signature must include enough state that
        # those discards register as progress.
        bot = BasicStrategyBot(seed=0)
        traj = generate_trajectory("AAAAAAA", bot.choose_action, max_steps=2000)
        self.assertNotEqual(traj.terminated_reason, "STUCK")


class StableSeedIntTests(unittest.TestCase):
    """Regression check: seed-string -> int derivation must be process-stable.

    Python's built-in hash() is randomized per process via PYTHONHASHSEED,
    which caused cross-process trajectory divergence during the M6
    archetype evaluation. The fix swapped hash() for an MD5-based stable
    derivation; this test pins the values so the regression can't return
    silently.
    """

    def test_canonical_seeds_have_pinned_stable_ints(self) -> None:
        from balatro_ai.solver.trajectory import _stable_seed_int

        # Values produced by md5(seed)[:4] big-endian. Pinned now so a
        # future hash-fn change is loud rather than silent.
        expected = {
            "AAAAAAA": 2217773388,
            "BBBBBBB": 3753761556,
            "CCCCCCC": 3278790901,
            "1234567": 4243231247,
        }
        for seed, want in expected.items():
            self.assertEqual(
                _stable_seed_int(seed),
                want,
                f"stable seed for {seed!r} drifted — dataset reproducibility would break",
            )

    def test_distinct_seeds_produce_distinct_ints(self) -> None:
        # Doesn't have to be a unique mapping forever, but with MD5 and
        # 4 short canonical seeds the collision rate is essentially zero.
        from balatro_ai.solver.trajectory import _stable_seed_int

        seeds = ("AAAAAAA", "BBBBBBB", "CCCCCCC", "1234567", "XKCD0001")
        ints = {_stable_seed_int(s) for s in seeds}
        self.assertEqual(len(ints), len(seeds))


if __name__ == "__main__":
    unittest.main()
