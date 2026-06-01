"""Stage 2.4 A/B: value-guided shop leaf vs the heuristic shop leaf.

Injects the learned value head as the shop beam's `leaf_value_fn` (z-score
calibrated to the heuristic leaf scale) via `SolverPolicy(shop_leaf_value_fn=...)`
— the only change between conditions is the shop leaf. Measures true winrate +
mean ante + shop-decision CPU. Play config is fixed + identical across conditions.

    PYTHONPATH=src python scripts/phase8_shop_value_ab.py \
        --ckpt .data/phase8_value_v0.pt --seeds 24 --jobs 6 \
        --metrics .data/phase8_shop_value_ab.json
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


def _collect_calib_states(seeds, d, w, cap):
    from balatro_ai.api.state import GamePhase
    from balatro_ai.solver.policy import SolverPolicy, _has_shop_action

    driver = SolverPolicy(play_backend="v2", play_depth=d, play_width=w, seed=0)
    states = []
    for seed in seeds:
        sim = _mk_sim(seed)
        for _ in range(2000):
            st = sim.state
            if st.run_over or st.phase == GamePhase.RUN_OVER:
                break
            if st.phase == GamePhase.SHOP and _has_shop_action(st):
                states.append(st)
                if len(states) >= cap:
                    return states
            sim.step(driver.choose_action(st))
    return states


def _run_seed(arg):
    seed, condition, ckpt, calib, d, w = arg
    from balatro_ai.api.actions import ActionType
    from balatro_ai.api.state import GamePhase
    from balatro_ai.solver.policy import SolverPolicy, _has_shop_action

    if condition == "neural":
        from balatro_ai.ml.shop_value import NeuralShopLeaf
        from balatro_ai.ml.train import load_checkpoint
        leaf = NeuralShopLeaf(load_checkpoint(ckpt), calib)
        policy = SolverPolicy(play_backend="v2", play_depth=d, play_width=w,
                              seed=0, shop_leaf_value_fn=leaf)
    else:
        policy = SolverPolicy(play_backend="v2", play_depth=d, play_width=w, seed=0)

    shop_dec, shop_cpu = 0, 0.0
    sim = _mk_sim(seed)
    for _ in range(2000):
        st = sim.state
        if st.run_over or st.phase == GamePhase.RUN_OVER:
            break
        is_shop = st.phase == GamePhase.SHOP and _has_shop_action(st)
        t0 = time.process_time()
        action = policy.choose_action(st)
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
    ap.add_argument("--ckpt", default=".data/phase8_value_v0.pt")
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--width", type=int, default=2)
    ap.add_argument("--calib-seeds", type=int, default=3)
    ap.add_argument("--calib-cap", type=int, default=60)
    ap.add_argument("--metrics", required=True)
    args = ap.parse_args()

    from balatro_ai.ml.shop_value import calibrate_scale
    from balatro_ai.ml.train import load_checkpoint

    # Calibrate on a disjoint seed block (900001+) so scale isn't fit to A/B seeds.
    calib_seeds = [f"{900000 + i:07d}" for i in range(1, args.calib_seeds + 1)]
    print(f"[shop-ab] collecting calibration states from {len(calib_seeds)} seeds...", flush=True)
    calib_states = _collect_calib_states(calib_seeds, args.depth, 1, args.calib_cap)
    model = load_checkpoint(args.ckpt)
    calib = calibrate_scale(model, calib_states)
    print(f"[shop-ab] calib (n={calib['n']}): "
          f"heuristic mean={calib['mean_h']:.1f} std={calib['std_h']:.1f} | "
          f"neural mean={calib['mean_n']:.3f} std={calib['std_n']:.3f}", flush=True)

    seeds = [f"{i:07d}" for i in range(1, args.seeds + 1)]
    jobs = [(s, cond, args.ckpt, calib, args.depth, args.width)
            for cond in ("heuristic", "neural") for s in seeds]

    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            rows = list(ex.map(_run_seed, jobs))
    else:
        rows = [_run_seed(j) for j in jobs]

    results = [_agg([r for r in rows if r["condition"] == c], c)
               for c in ("heuristic", "neural")]
    for r in results:
        print("[shop-ab]", json.dumps(r), flush=True)

    with open(args.metrics, "w", encoding="utf-8") as fh:
        json.dump({"depth": args.depth, "width": args.width, "calib": calib,
                   "results": results, "rows": rows}, fh, indent=2)
    h = next(r for r in results if r["condition"] == "heuristic")
    nu = next(r for r in results if r["condition"] == "neural")
    open(f".data/_SHOPAB_h_a{h['mean_ante']:.2f}_w{h['winrate']:.2f}"
         f"_n_a{nu['mean_ante']:.2f}_w{nu['winrate']:.2f}", "w").close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
