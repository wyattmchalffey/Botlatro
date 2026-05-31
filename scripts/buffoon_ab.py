"""Paired A/B for the first-shop Buffoon-pack fix (isolated), at play_width=1.

Real Balatro guarantees a Buffoon pack in the very first shop (free early joker).
The sampler fallback used by data-gen never produced it (inverted check), so every
run was denied that joker. This pairs the fix ON vs OFF on the SAME seed (cancels
variance). Build-aggression toggles are forced OFF in both arms to isolate the
Buffoon effect. Uses play_width=1 to MATCH the committed data-gen config (and run
~1.85x faster than the width=2 default).

    PYTHONPATH=src python scripts/buffoon_ab.py [n_seeds] [jobs]
"""

from __future__ import annotations

import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor


def _run_one(seed: str, buffoon: bool) -> dict:
    import balatro_ai.search.hand_search as hs
    import balatro_ai.search.shop_search as ss
    import balatro_ai.search.shop_sampler as smp
    from balatro_ai.solver.policy import SolverPolicy
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int
    from balatro_ai.sim.local_runner import LocalBalatroSimulator

    hs._NATIVE_BEAM_ENABLED = False
    # isolate: build-aggression OFF in both arms (committed baseline values)
    ss._SELL_FLAT_MIN = 0.0
    ss._BUILD_CAP_BASE = 0.018
    ss._BOSS_AWARE_EVAL = False
    smp._FORCE_FIRST_BUFFOON = bool(buffoon)
    sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
    sim.state = SeedGame(seed, stake="white").initial_state()
    pol = SolverPolicy(seed=0, play_width=1)  # MATCH data-gen (fast + representative)
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
    off = _run_one(seed, False)
    on = _run_one(seed, True)
    return {"seed": seed, "off": off, "on": on, "d_ante": on["ante"] - off["ante"]}


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 96
    jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    seeds = [f"{i:07d}" for i in range(1, n + 1)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        rows = list(ex.map(run_task, seeds))
    wall = time.perf_counter() - t0

    d = [r["d_ante"] for r in rows]
    off_ante = statistics.mean(r["off"]["ante"] for r in rows)
    on_ante = statistics.mean(r["on"]["ante"] for r in rows)
    off_w = sum(r["off"]["won"] for r in rows)
    on_w = sum(r["on"]["won"] for r in rows)
    better = sum(1 for x in d if x > 0)
    worse = sum(1 for x in d if x < 0)
    print(f"=== first-shop Buffoon fix paired A/B (width=1), {n} seeds, wall={wall:.0f}s ===")
    print(f"  buffoon OFF: ante {off_ante:.2f}  wins {off_w}/{n}")
    print(f"  buffoon ON:  ante {on_ante:.2f}  wins {on_w}/{n}")
    print(f"  PAIRED d_ante mean={statistics.mean(d):+.3f}  (better {better} / worse {worse})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
