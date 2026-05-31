"""Paired A/B for a shop-value change — cancels seed variance.

At ~0 wins the only signal is ante/score, and independent runs on different
seed subsets are too noisy (+-0.2 ante) to detect a small build-quality effect.
This harness runs EACH seed TWICE in the same worker — once with the coherence
weight at the BASELINE value, once at the TEST value — and reports the PAIRED
per-seed delta (ante/score) plus win counts for each arm. Pairing removes the
dominant seed-to-seed variance, so a real effect shows at far fewer seeds.

The coherence weight is a module global read at call time, so we set it between
the two runs in-process (no rebuild, no separate pool).

    PYTHONPATH=src python scripts/shop_paired_ab.py [n_seeds] [jobs] [baseline_w] [test_w]
"""

from __future__ import annotations

import statistics
import sys
from concurrent.futures import ProcessPoolExecutor


def _run_one(seed: str, weight: float) -> dict:
    import balatro_ai.search.hand_search as hs
    import balatro_ai.search.shop_search as ss
    from balatro_ai.solver.policy import SolverPolicy
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int
    from balatro_ai.sim.local_runner import LocalBalatroSimulator

    hs._NATIVE_BEAM_ENABLED = False
    ss._SELL_OWNED_VALUE_COEFF = float(weight)  # toggle the anti-churn sell penalty
    sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
    sim.state = SeedGame(seed, stake="white").initial_state()
    pol = SolverPolicy(seed=0)
    for _ in range(6000):
        st = sim.state
        if st.run_over:
            break
        a = pol.choose_action(st)
        if a.action_type.value == "no_op":
            break
        sim.step(a)
    s = sim.state
    return {"ante": s.ante, "score": int(s.current_score),
            "won": bool(getattr(s, "won", False)) or s.ante >= 9}


def run_task(arg: tuple) -> dict:
    seed, base_w, test_w = arg
    base = _run_one(seed, base_w)
    test = _run_one(seed, test_w)
    return {"seed": seed, "base": base, "test": test,
            "d_ante": test["ante"] - base["ante"],
            "d_score": test["score"] - base["score"]}


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    base_w = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15
    test_w = float(sys.argv[4]) if len(sys.argv) > 4 else 0.85
    seeds = [f"{i:07d}" for i in range(1, n + 1)]
    args = [(s, base_w, test_w) for s in seeds]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        rows = list(ex.map(run_task, args))

    d_ante = [r["d_ante"] for r in rows]
    d_score = [r["d_score"] for r in rows]
    base_ante = statistics.mean(r["base"]["ante"] for r in rows)
    test_ante = statistics.mean(r["test"]["ante"] for r in rows)
    base_wins = sum(r["base"]["won"] for r in rows)
    test_wins = sum(r["test"]["won"] for r in rows)
    better = sum(1 for d in d_ante if d > 0)
    worse = sum(1 for d in d_ante if d < 0)
    same = sum(1 for d in d_ante if d == 0)
    print(f"=== paired A/B: coherence weight {base_w} -> {test_w}, {n} seeds ===")
    print(f"  base ante mean={base_ante:.2f}  wins={base_wins}/{n}")
    print(f"  test ante mean={test_ante:.2f}  wins={test_wins}/{n}")
    print(f"  PAIRED d_ante: mean={statistics.mean(d_ante):+.3f} "
          f"median={statistics.median(d_ante):+.1f}  (better {better} / worse {worse} / same {same})")
    print(f"  PAIRED d_score: mean={statistics.mean(d_score):+.0f} "
          f"median={statistics.median(d_score):+.0f}")
    # show the seeds with the biggest swings (both directions)
    rows_sorted = sorted(rows, key=lambda r: r["d_ante"])
    print("  biggest regressions:", [(r["seed"], r["d_ante"]) for r in rows_sorted[:4]])
    print("  biggest gains:      ", [(r["seed"], r["d_ante"]) for r in rows_sorted[-4:]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
