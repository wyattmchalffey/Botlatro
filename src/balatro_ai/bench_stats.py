"""Statistical inference helpers for winrate benches and paired A/Bs.

Pure stdlib. Every bench that reports a winrate or a paired delta should also
report these, so adoption/rejection decisions stop happening on noise-level
reads (see SUPERHUMAN_ROADMAP.md P0.2: 100-seed unpaired MDE is ~11pp).
"""

from __future__ import annotations

from math import comb, sqrt


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% (default) Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 1.0)
    phat = wins / n
    denom = 1.0 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * sqrt(phat * (1.0 - phat) / n + z * z / (4.0 * n * n))
    return (
        max(0.0, (centre - margin) / denom),
        min(1.0, (centre + margin) / denom),
    )


def exact_binomial_two_sided_p(k: int, n: int) -> float:
    """Two-sided exact p-value for k successes from X ~ Bin(n, 0.5).

    Symmetric null, so the two-sided p is twice the smaller tail, capped at 1.
    """
    if n <= 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2.0**n
    return min(1.0, 2.0 * tail)


def mcnemar_exact_p(flips_for_b: int, flips_for_a: int) -> float:
    """Exact McNemar test (= sign test) on discordant paired outcomes.

    flips_for_b: pairs where A lost and B won; flips_for_a: the reverse.
    Concordant pairs carry no information about the difference and are ignored.
    """
    return exact_binomial_two_sided_p(
        min(flips_for_b, flips_for_a), flips_for_b + flips_for_a
    )


def paired_delta_ci(
    flips_for_b: int, flips_for_a: int, n_pairs: int, z: float = 1.96
) -> tuple[float, float]:
    """95% (default) Wald CI for the paired winrate difference (B - A).

    Uses the standard paired-proportion variance, which only the discordant
    counts drive: se = sqrt(b + c - (b - c)^2 / n) / n.
    """
    if n_pairs <= 0:
        return (-1.0, 1.0)
    b, c = flips_for_a, flips_for_b
    delta = (c - b) / n_pairs
    se = sqrt(max(0.0, (b + c) - (b - c) ** 2 / n_pairs)) / n_pairs
    return (max(-1.0, delta - z * se), min(1.0, delta + z * se))
