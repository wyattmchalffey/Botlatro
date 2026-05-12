"""Benchmark result summaries."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    economy: dict[str, float] = field(default_factory=dict)
    tarot_usage: dict[str, int] = field(default_factory=dict)


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
    average_economy: dict[str, float] = field(default_factory=dict)
    average_tarot_usage: dict[str, float] = field(default_factory=dict)
    deck: str = "unknown"
    profile_name: str = "unknown"
    unlock_state: str = "unknown"

    def to_text(self) -> str:
        lines = [
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
        ]
        if self.average_economy:
            lines.extend(
                (
                    f"Average effective money: {self.average_economy.get('money_effective_purchase_power', 0.0):.1f}",
                    f"Average income: {self.average_economy.get('money_income_total', 0.0):.1f}",
                    f"Average spent: {self.average_economy.get('money_spent_total', 0.0):.1f}",
                    "Average spend by type: "
                    f"jokers={self.average_economy.get('money_spent_jokers', 0.0):.1f}, "
                    f"consumables={self.average_economy.get('money_spent_consumables', 0.0):.1f}, "
                    f"packs={self.average_economy.get('money_spent_booster_packs', 0.0):.1f}, "
                    f"vouchers={self.average_economy.get('money_spent_vouchers', 0.0):.1f}, "
                    f"rerolls={self.average_economy.get('money_spent_rerolls', 0.0):.1f}, "
                    f"cards={self.average_economy.get('money_spent_playing_cards', 0.0):.1f}, "
                    f"other={self.average_economy.get('money_spent_other', 0.0):.1f}",
                )
            )
        if self.average_tarot_usage:
            total = sum(self.average_tarot_usage.values())
            top_tarots = sorted(self.average_tarot_usage.items(), key=lambda item: (-item[1], item[0]))[:8]
            tarot_text = ", ".join(f"{name}={value:.2f}" for name, value in top_tarots)
            lines.append(f"Average tarot uses: total={total:.2f}" + (f", {tarot_text}" if tarot_text else ""))
        lines.append(f"Average runtime: {self.average_runtime_seconds:.2f} sec/run")
        return "\n".join(lines)


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
        average_economy=_average_economy(results),
        average_tarot_usage=_average_tarot_usage(results),
    )


def _average_economy(results: list[RunResult] | tuple[RunResult, ...]) -> dict[str, float]:
    keys = sorted({key for result in results for key in result.economy})
    if not keys:
        return {}
    return {key: mean(float(result.economy.get(key, 0.0)) for result in results) for key in keys}


def _average_tarot_usage(results: list[RunResult] | tuple[RunResult, ...]) -> dict[str, float]:
    keys = sorted({key for result in results for key in result.tarot_usage})
    if not keys:
        return {}
    return {key: mean(float(result.tarot_usage.get(key, 0)) for result in results) for key in keys}
