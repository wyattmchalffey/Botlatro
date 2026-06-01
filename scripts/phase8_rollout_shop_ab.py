"""Engine-path first brick A/B: rollout shop policy vs the heuristic shop.

Both conditions use the SAME play backend; the only difference is the shop:
- heuristic : SolverPolicy's best_shop_action (the 18.8% baseline).
- rollout   : RolloutShopPolicy (forward-model rollout per candidate).
Measures true winrate + mean ante + CPU/shop-decision.

    PYTHONPATH=src python scripts/phase8_rollout_shop_ab.py \
        --seeds 24 --jobs 6 --horizon 2 --samples 2 --metrics .data/phase8_rollout_shop_ab.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _mk_sim(seed):
    from balatro_ai.sim.local_runner import LocalBalatroSimulator
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int

    sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
    sim.state = SeedGame(seed, stake="white").initial_state()
    return sim


def _run_seed(arg):
    seed, condition, horizon, samples, d, w = arg
    from balatro_ai.api.actions import ActionType
    from balatro_ai.api.state import GamePhase
    from balatro_ai.solver.policy import SolverPolicy, _has_shop_action

    base = SolverPolicy(play_backend="v2", play_depth=d, play_width=w, seed=0)
    shop = None
    if condition == "rollout":
        from balatro_ai.search.rollout_shop import RolloutShopPolicy
        shop = RolloutShopPolicy(horizon=horizon, samples=samples, base_seed=0)

    shop_dec, shop_cpu = 0, 0.0
    sim = _mk_sim(seed)
    for _ in range(2000):
        st = sim.state
        if st.run_over or st.phase == GamePhase.RUN_OVER:
            break
        is_shop = st.phase == GamePhase.SHOP and _has_shop_action(st)
        t0 = time.process_time()
        action = shop.choose_action(st) if (is_shop and shop is not None) else base.choose_action(st)
        dt = time.process_time() - t0
        if is_shop:
            shop_dec += 1
            shop_cpu += dt
        if action.action_type == ActionType.NO_OP:
            break
        sim.step(action)
    s = sim.state
    return {"seed": seed, "condition": condition, "won": bool(s.won), "ante": s.ante,
            "shop_decisions": shop_dec, "shop_cpu": shop_cpu}


def _agg(rows, name):
    antes = [r["ante"] for r in rows]
    wins = sum(int(r["won"]) for r in rows)
    dec = sum(r["shop_decisions"] for r in rows)
    cpu = sum(r["shop_cpu"] for r in rows)
    return {
        "condition": name, "n": len(rows),
        "mean_ante": round(statistics.mean(antes), 3),
        "median_ante": statistics.median(antes),
        "wins": wins, "winrate": round(wins / max(1, len(rows)), 3),
        "ms_per_shop_decision": round(1000 * cpu / max(1, dec), 1),
        "shop_decisions": dec,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--width", type=int, default=2)
    ap.add_argument("--metrics", required=True)
    args = ap.parse_args()

    seeds = [f"{i:07d}" for i in range(1, args.seeds + 1)]
    jobs = [(s, cond, args.horizon, args.samples, args.depth, args.width)
            for cond in ("heuristic", "rollout") for s in seeds]

    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            rows = list(ex.map(_run_seed, jobs))
    else:
        rows = [_run_seed(j) for j in jobs]

    results = [_agg([r for r in rows if r["condition"] == c], c)
               for c in ("heuristic", "rollout")]
    for r in results:
        print("[rollout-shop-ab]", json.dumps(r), flush=True)

    with open(args.metrics, "w", encoding="utf-8") as fh:
        json.dump({"horizon": args.horizon, "samples": args.samples,
                   "results": results, "rows": rows}, fh, indent=2)
    h = next(r for r in results if r["condition"] == "heuristic")
    ro = next(r for r in results if r["condition"] == "rollout")
    open(f".data/_ROLLOUTSHOP_h_a{h['mean_ante']:.2f}_w{h['winrate']:.2f}"
         f"_r_a{ro['mean_ante']:.2f}_w{ro['winrate']:.2f}", "w").close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
