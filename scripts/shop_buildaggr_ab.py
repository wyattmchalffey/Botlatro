"""Paired A/B for the build-aggression + boss-aware changes (3 toggles together).

Treatment = the user's two directives:
  - buy/keep score jokers early: flat-min anti-churn + raised early build-capacity
    scale (_SELL_FLAT_MIN, _BUILD_CAP_BASE)
  - boss-aware build scoring (_BOSS_AWARE_EVAL)
Baseline = current committed behavior (flat_min 0, build_cap 0.018, boss_aware off;
sell coeff 0.45 stays in both arms). Paired per seed cancels seed variance.

    PYTHONPATH=src python scripts/shop_buildaggr_ab.py [n_seeds] [jobs]
"""

from __future__ import annotations

import statistics
import sys
from concurrent.futures import ProcessPoolExecutor

import os as _os
BASE = {"flat": 0.0, "cap": 0.018, "boss": False}
# Ablation via env: BALATRO_AB_FLAT / _CAP / _BOSS override the TEST arm.
TEST = {
    "flat": float(_os.environ.get("BALATRO_AB_FLAT", "5.0")),
    "cap": float(_os.environ.get("BALATRO_AB_CAP", "0.030")),
    "boss": _os.environ.get("BALATRO_AB_BOSS", "1") != "0",
}


def _run_one(seed: str, cfg: dict) -> dict:
    import balatro_ai.search.hand_search as hs
    import balatro_ai.search.shop_search as ss
    from balatro_ai.solver.policy import SolverPolicy
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int
    from balatro_ai.sim.local_runner import LocalBalatroSimulator

    hs._NATIVE_BEAM_ENABLED = False
    ss._SELL_FLAT_MIN = cfg["flat"]
    ss._BUILD_CAP_BASE = cfg["cap"]
    ss._BOSS_AWARE_EVAL = cfg["boss"]
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


def run_task(seed: str) -> dict:
    base = _run_one(seed, BASE)
    test = _run_one(seed, TEST)
    return {"seed": seed, "base": base, "test": test,
            "d_ante": test["ante"] - base["ante"]}


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    seeds = [f"{i:07d}" for i in range(1, n + 1)]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        rows = list(ex.map(run_task, seeds))

    d_ante = [r["d_ante"] for r in rows]
    base_ante = statistics.mean(r["base"]["ante"] for r in rows)
    test_ante = statistics.mean(r["test"]["ante"] for r in rows)
    base_wins = sum(r["base"]["won"] for r in rows)
    test_wins = sum(r["test"]["won"] for r in rows)
    better = sum(1 for d in d_ante if d > 0)
    worse = sum(1 for d in d_ante if d < 0)
    print(f"=== build-aggression + boss-aware paired A/B, {n} seeds ===")
    print(f"  base: ante {base_ante:.2f}  wins {base_wins}/{n}")
    print(f"  test: ante {test_ante:.2f}  wins {test_wins}/{n}")
    print(f"  PAIRED d_ante mean={statistics.mean(d_ante):+.3f}  (better {better} / worse {worse})")
    rs = sorted(rows, key=lambda r: r["d_ante"])
    print("  regressions:", [(r["seed"], r["d_ante"]) for r in rs[:4]])
    print("  gains:      ", [(r["seed"], r["d_ante"]) for r in rs[-4:]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
