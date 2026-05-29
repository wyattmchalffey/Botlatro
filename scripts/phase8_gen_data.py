"""Generate a value-function dataset: (features, won) at shop-entry states.

Runs the heuristic bot across many seeds (parallel), records the feature
vector at each shop-entry decision, and labels every state with the run's
eventual outcome (won). The value model learns P(win | state). One feature
vector per shop VISIT (first shop action) to avoid over-weighting shops with
many buys/rerolls.

    PYTHONPATH=src python scripts/phase8_gen_data.py [n_seeds] [jobs] [out.npz] [faithful]
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def gen_seed(args):
    seed, faithful, seed_idx = args
    from balatro_ai.api.state import GamePhase, with_derived_legal_actions
    from balatro_ai.bots.registry import create_bot
    from balatro_ai.ml.features import features_from_state
    from balatro_ai.sim.local_runner import LocalBalatroSimulator
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int

    kw = {"seed": _stable_seed_int(seed), "stake": "white"}
    if faithful:
        kw["balatro_seed"] = seed
    sim = LocalBalatroSimulator(**kw)
    sim.state = with_derived_legal_actions(SeedGame(seed, stake="white").initial_state())
    bot = create_bot("basic_strategy_bot", seed=0)
    rows = []  # (features, ante)
    was_shop = False
    for _ in range(4000):
        st = sim.state
        if st.run_over:
            break
        in_shop = st.phase == GamePhase.SHOP
        if in_shop and not was_shop:  # shop entry
            rows.append((features_from_state(st), int(st.ante)))
        was_shop = in_shop
        a = bot.choose_action(st)
        if a.action_type.value == "no_op":
            break
        sim.step(a)
    won = int(bool(sim.state.won))
    return [(f, ante, won, seed_idx) for (f, ante) in rows]


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    out = sys.argv[3] if len(sys.argv) > 3 else ".data/phase8_value_dataset.npz"
    faithful = bool(int(sys.argv[4])) if len(sys.argv) > 4 else False
    seeds = [f"{i:07d}" for i in range(1, n + 1)]
    t0 = time.perf_counter()
    all_rows = []
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for seed_rows in ex.map(gen_seed, [(s, faithful, idx) for idx, s in enumerate(seeds)]):
            all_rows.extend(seed_rows)
    dt = time.perf_counter() - t0
    if not all_rows:
        print("no rows generated")
        return 1
    X = np.array([r[0] for r in all_rows], dtype=np.float64)
    ante = np.array([r[1] for r in all_rows], dtype=np.int64)
    y = np.array([r[2] for r in all_rows], dtype=np.float64)
    seed_idx = np.array([r[3] for r in all_rows], dtype=np.int64)
    np.savez(out, X=X, y=y, ante=ante, seed_idx=seed_idx)
    print(f"generated {len(all_rows)} states from {n} seeds in {dt:.1f}s -> {out}", flush=True)
    print(f"win rate of states: {y.mean():.1%}; feature dim: {X.shape[1]}", flush=True)
    # per-ante state counts + win fraction
    import collections
    by_ante = collections.defaultdict(lambda: [0, 0])
    for a, w in zip(ante, y):
        by_ante[int(a)][0] += 1
        by_ante[int(a)][1] += int(w)
    for a in sorted(by_ante):
        c, w = by_ante[a]
        print(f"  ante {a}: {c} states, {w/c:.0%} eventually won", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
