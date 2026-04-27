"""Benchmark result summaries."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True, slots=True)
class RunResult:
    bot_version: str
    seed: int
    stake: str
    won: bool
    ante_reached: int
    final_score: int
    final_money: int
    runtime_seconds: float
    death_reason: str | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    bot_version: str
    stake: str
    run_count: int
    win_rate: float
    average_ante: float
    average_final_score: float
    average_final_money: float
    average_runtime_seconds: float
    deck: str = "unknown"
    profile_name: str = "unknown"
    unlock_state: str = "unknown"

    def to_text(self) -> str:
        return "\n".join(
            (
                f"Bot: {self.bot_version}",
                f"Stake: {self.stake}",
                f"Deck: {self.deck}",
                f"Profile: {self.profile_name}",
                f"Unlocks: {self.unlock_state}",
                f"Runs: {self.run_count}",
                f"Win rate: {self.win_rate:.1%}",
                f"Average ante: {self.average_ante:.2f}",
                f"Average final score: {self.average_final_score:.1f}",
                f"Average final money: {self.average_final_money:.1f}",
                f"Average runtime: {self.average_runtime_seconds:.2f} sec/run",
            )
        )


def summarize_runs(results: list[RunResult] | tuple[RunResult, ...]) -> BenchmarkSummary:
    if not results:
        raise ValueError("Cannot summarize an empty benchmark")

    first = results[0]
    return BenchmarkSummary(
        bot_version=first.bot_version,
        stake=first.stake,
        run_count=len(results),
        win_rate=sum(1 for result in results if result.won) / len(results),
        average_ante=mean(result.ante_reached for result in results),
        average_final_score=mean(result.final_score for result in results),
        average_final_money=mean(result.final_money for result in results),
        average_runtime_seconds=mean(result.runtime_seconds for result in results),
    )
