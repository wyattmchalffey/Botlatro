"""Tests for the bench statistical-inference helpers."""

from __future__ import annotations

import pytest

from balatro_ai.bench_stats import (
    exact_binomial_two_sided_p,
    mcnemar_exact_p,
    paired_delta_ci,
    wilson_ci,
)


def test_wilson_ci_hand_checked():
    # 39/200 = 19.5% — the canonical held-out winrate.
    lo, hi = wilson_ci(39, 200)
    assert lo == pytest.approx(0.1461, abs=2e-3)
    assert hi == pytest.approx(0.2554, abs=2e-3)


def test_wilson_ci_edges():
    assert wilson_ci(0, 0) == (0.0, 1.0)
    lo, hi = wilson_ci(0, 100)
    assert lo == 0.0 and 0.0 < hi < 0.05
    lo, hi = wilson_ci(100, 100)
    assert 0.95 < lo < 1.0 and hi == pytest.approx(1.0)


def test_mcnemar_hand_checked():
    # The 2026-06-08 skip A/B: lost 15 / gained 5 -> p ~= 0.0414 (significant).
    assert mcnemar_exact_p(5, 15) == pytest.approx(0.041389, abs=1e-5)
    # Symmetric flips -> p = 1.
    assert mcnemar_exact_p(7, 7) == 1.0
    # No discordant pairs -> no evidence.
    assert mcnemar_exact_p(0, 0) == 1.0


def test_exact_binomial_symmetry():
    assert exact_binomial_two_sided_p(3, 10) == exact_binomial_two_sided_p(7, 10)
    assert exact_binomial_two_sided_p(0, 10) == pytest.approx(2 / 1024)


def test_paired_delta_ci():
    # +5 / -15 over 100 pairs: delta = -0.10, CI excludes 0.
    lo, hi = paired_delta_ci(5, 15, 100)
    assert lo < hi < 0.0
    assert (lo + hi) / 2 == pytest.approx(-0.10, abs=1e-9)
    # Balanced flips: CI straddles 0.
    lo, hi = paired_delta_ci(10, 10, 100)
    assert lo < 0.0 < hi
