"""Paired A/B for SHOP search DEPTH (myopia lever).

The churn bug (project_datagen_speed.md) was a depth-2 myopia exploit. Deeper
shop search trusts the leaf value over multi-step action-heuristic gaming and can
plan more coherent multi-joker build sequences. This pairs depth=base vs depth=test
on the SAME seed (cancels seed variance) and reports the win/ante deltas. Depth 3
is slower, so this is the speed/winrate tradeoff test the user cares about.

    PYTHONPATH=src python scripts/shop_depth_ab.py [n_seeds] [jobs] [base_depth] [test_depth]
"""

from __future__ import annotations

import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor


def _run_one(seed: str, depth: int) -> dict:
    import time as _t
    import balatro_ai.search.hand_search as hs
    from balatro_ai.search.shop_search import ShopSearchConfig
    from balatro_ai.solver.policy import (
        SolverPolicy, DEFAULT_SHOP_BEAM_WIDTH, DEFAULT_SHOP_REROLL_SAMPLES,
    )
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int
    from balatro_ai.sim.local_runner import LocalBalatroSimulator

    hs._NATIVE_BEAM_ENABLED = False
    sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
    sim.state = SeedGame(seed, stake="white").initial_state()
    cfg = ShopSearchConfig(
        beam_width=DEFAULT_SHOP_BEAM_WIDTH, depth=int(depth),
        reroll_samples=DEFAULT_SHOP_REROLL_SAMPLES, seed=0,
    )
    pol = SolverPolicy(seed=0, shop_config=cfg)
    t0 = _t.process_time()
    for _ in range(6000):
        st = sim.state
        if st.run_over:
            break
        a = pol.choose_action(st)
        if a.action_type.value == "no_op":
            break
        sim.step(a)
    elapsed = _t.process_time() - t0
    s = sim.state
    return {"ante": s.ante, "score": int(s.current_score),
            "won": bool(getattr(s, "won", False)) or s.ante >= 9, "cpu": elapsed}


def run_task(arg: tuple) -> dict:
    seed, base_d, test_d = arg
    base = _run_one(seed, base_d)
    test = _run_one(seed, test_d)
    return {"seed": seed, "base": base, "test": test,
            "d_ante": test["ante"] - base["ante"]}


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    base_d = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    test_d = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    seeds = [f"{i:07d}" for i in range(1, n + 1)]
    args = [(s, base_d, test_d) for s in seeds]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        rows = list(ex.map(run_task, args))
    wall = time.perf_counter() - t0

    d_ante = [r["d_ante"] for r in rows]
    base_ante = statistics.mean(r["base"]["ante"] for r in rows)
    test_ante = statistics.mean(r["test"]["ante"] for r in rows)
    base_wins = sum(r["base"]["won"] for r in rows)
    test_wins = sum(r["test"]["won"] for r in rows)
    base_cpu = statistics.mean(r["base"]["cpu"] for r in rows)
    test_cpu = statistics.mean(r["test"]["cpu"] for r in rows)
    better = sum(1 for d in d_ante if d > 0)
    worse = sum(1 for d in d_ante if d < 0)
    print(f"=== paired shop-depth A/B: {base_d} -> {test_d}, {n} seeds, wall={wall:.0f}s ===")
    print(f"  base d{base_d}: ante {base_ante:.2f}  wins {base_wins}/{n}  cpu/run {base_cpu:.1f}s")
    print(f"  test d{test_d}: ante {test_ante:.2f}  wins {test_wins}/{n}  cpu/run {test_cpu:.1f}s")
    print(f"  PAIRED d_ante mean={statistics.mean(d_ante):+.3f}  (better {better} / worse {worse})")
    print(f"  speed cost: {test_cpu/max(0.01,base_cpu):.2f}x CPU/run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
