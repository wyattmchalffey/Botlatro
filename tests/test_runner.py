from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import context  # noqa: F401
import balatro_ai.eval.runner as runner
from balatro_ai.eval.metrics import RunResult
from balatro_ai.eval.runner import BenchmarkOptions, _seed_set_from_options, endpoint_urls, run_benchmark


class RunnerTests(unittest.TestCase):
    def test_endpoint_urls_uses_consecutive_ports(self) -> None:
        self.assertEqual(
            endpoint_urls("127.0.0.1", 12346, 3),
            (
                "http://127.0.0.1:12346",
                "http://127.0.0.1:12347",
                "http://127.0.0.1:12348",
            ),
        )

    def test_endpoint_urls_requires_worker(self) -> None:
        with self.assertRaises(ValueError):
            endpoint_urls("127.0.0.1", 12346, 0)

    def test_explicit_seed_values_override_seed_count(self) -> None:
        seed_set = _seed_set_from_options(
            BenchmarkOptions(bot="random_bot", seeds=100, seed_values=(42,), label="manual")
        )

        self.assertEqual(seed_set.seeds, (42,))
        self.assertIn("explicit", seed_set.label)

    def test_run_benchmark_stops_scheduling_single_worker(self) -> None:
        calls: list[int] = []
        original_run_seed = runner._run_seed

        def fake_run_seed(*, seed: int, endpoint: str, options: BenchmarkOptions) -> RunResult:
            calls.append(seed)
            return RunResult(
                bot_version=options.bot,
                seed=seed,
                stake=options.stake,
                won=False,
                ante_reached=1,
                final_score=0,
                final_money=0,
                runtime_seconds=0.0,
            )

        try:
            runner._run_seed = fake_run_seed
            result = run_benchmark(
                BenchmarkOptions(bot="random_bot", seed_values=(1, 2, 3)),
                should_stop=lambda: bool(calls),
            )
        finally:
            runner._run_seed = original_run_seed

        self.assertEqual(calls, [1])
        self.assertEqual(tuple(run.seed for run in result.results), (1,))
        self.assertEqual(result.summary.deck, "RED")
        self.assertEqual(result.summary.profile_name, "P1")
        self.assertEqual(result.summary.unlock_state, "all")

    def test_run_benchmark_records_custom_metadata(self) -> None:
        original_run_seed = runner._run_seed

        def fake_run_seed(*, seed: int, endpoint: str, options: BenchmarkOptions) -> RunResult:
            self.assertEqual(options.replay_mode, "light")
            return RunResult(
                bot_version=options.bot,
                seed=seed,
                stake=options.stake,
                won=True,
                ante_reached=8,
                final_score=1000,
                final_money=10,
                runtime_seconds=1.0,
            )

        try:
            runner._run_seed = fake_run_seed
            result = run_benchmark(
                BenchmarkOptions(
                    bot="random_bot",
                    deck="BLUE",
                    profile_name="P2",
                    unlock_state="profile-default",
                    seed_values=(99,),
                    replay_mode="light",
                )
            )
        finally:
            runner._run_seed = original_run_seed

        text = result.summary.to_text()
        self.assertIn("Deck: BLUE", text)
        self.assertIn("Profile: P2", text)
        self.assertIn("Unlocks: profile-default", text)

    def test_run_benchmark_reports_seed_metadata(self) -> None:
        original_run_seed = runner._run_seed
        messages: list[str] = []

        def fake_run_seed(*, seed: int, endpoint: str, options: BenchmarkOptions) -> RunResult:
            return RunResult(
                bot_version=options.bot,
                seed=seed,
                stake=options.stake,
                won=False,
                ante_reached=1,
                final_score=100,
                final_money=5,
                runtime_seconds=1.0,
            )

        try:
            runner._run_seed = fake_run_seed
            run_benchmark(
                BenchmarkOptions(bot="random_bot", seed_values=(42,), label="manual"),
                progress=messages.append,
            )
        finally:
            runner._run_seed = original_run_seed

        self.assertIn("Seed label: white:manual:explicit", messages)
        self.assertIn("First seed: 42", messages)

    def test_run_benchmark_can_stop_before_any_seed(self) -> None:
        result = run_benchmark(
            BenchmarkOptions(bot="random_bot", seed_values=(1, 2, 3)),
            should_stop=lambda: True,
        )

        self.assertEqual(result.results, ())
        self.assertEqual(result.summary.run_count, 0)

    def test_run_benchmark_retries_failed_seed_and_replaces_replay(self) -> None:
        original_run_seed = runner._run_seed
        original_healthy_endpoints = runner._healthy_endpoints
        calls: list[tuple[int, str]] = []
        attempts: dict[int, int] = {}

        with TemporaryDirectory() as tmp:
            replay_dir = Path(tmp)

            def fake_run_seed(*, seed: int, endpoint: str, options: BenchmarkOptions) -> RunResult:
                calls.append((seed, endpoint))
                attempts[seed] = attempts.get(seed, 0) + 1
                replay_path = replay_dir / f"{options.bot}_{options.stake}_{seed}.jsonl"
                if seed == 1 and attempts[seed] == 1:
                    replay_path.write_text("first failure\n", encoding="utf-8")
                    return RunResult(
                        bot_version=options.bot,
                        seed=seed,
                        stake=options.stake,
                        won=False,
                        ante_reached=0,
                        final_score=0,
                        final_money=0,
                        runtime_seconds=1.0,
                        death_reason="error:ConnectionError: bridge down",
                    )
                if seed == 1:
                    self.assertFalse(replay_path.exists())
                    replay_path.write_text("retry success\n", encoding="utf-8")
                return RunResult(
                    bot_version=options.bot,
                    seed=seed,
                    stake=options.stake,
                    won=seed == 1,
                    ante_reached=8 if seed == 1 else 2,
                    final_score=100,
                    final_money=5,
                    runtime_seconds=1.0,
                )

            try:
                runner._run_seed = fake_run_seed
                runner._healthy_endpoints = lambda options: ("http://127.0.0.1:12347",)
                result = run_benchmark(
                    BenchmarkOptions(
                        bot="random_bot",
                        seed_values=(1, 2),
                        endpoints=("http://127.0.0.1:12346", "http://127.0.0.1:12347"),
                        replay_dir=replay_dir,
                        replay_mode="summary",
                        retry_failed_seeds=1,
                    )
                )
            finally:
                runner._run_seed = original_run_seed
                runner._healthy_endpoints = original_healthy_endpoints

            self.assertEqual(attempts[1], 2)
            self.assertEqual(calls[-1], (1, "http://127.0.0.1:12347"))
            self.assertTrue(result.results[0].won)
            self.assertEqual((replay_dir / "random_bot_white_1.jsonl").read_text(encoding="utf-8"), "retry success\n")

    def test_run_benchmark_retires_unhealthy_parallel_endpoint(self) -> None:
        original_run_seed = runner._run_seed
        original_endpoint_is_healthy = runner._endpoint_is_healthy
        original_healthy_endpoints = runner._healthy_endpoints
        calls: list[tuple[int, str]] = []

        def fake_run_seed(*, seed: int, endpoint: str, options: BenchmarkOptions) -> RunResult:
            calls.append((seed, endpoint))
            if endpoint.endswith("12346"):
                return RunResult(
                    bot_version=options.bot,
                    seed=seed,
                    stake=options.stake,
                    won=False,
                    ante_reached=0,
                    final_score=0,
                    final_money=0,
                    runtime_seconds=1.0,
                    death_reason="error:ConnectionError: bridge down",
                )
            return RunResult(
                bot_version=options.bot,
                seed=seed,
                stake=options.stake,
                won=False,
                ante_reached=2,
                final_score=100,
                final_money=5,
                runtime_seconds=1.0,
            )

        try:
            runner._run_seed = fake_run_seed
            runner._endpoint_is_healthy = lambda endpoint, timeout: not endpoint.endswith("12346")
            runner._healthy_endpoints = lambda options: ("http://127.0.0.1:12347",)
            run_benchmark(
                BenchmarkOptions(
                    bot="random_bot",
                    seed_values=(1, 2, 3, 4),
                    endpoints=("http://127.0.0.1:12346", "http://127.0.0.1:12347"),
                    retry_failed_seeds=0,
                )
            )
        finally:
            runner._run_seed = original_run_seed
            runner._endpoint_is_healthy = original_endpoint_is_healthy
            runner._healthy_endpoints = original_healthy_endpoints

        self.assertEqual(sum(1 for _, endpoint in calls if endpoint.endswith("12346")), 1)


if __name__ == "__main__":
    unittest.main()
