"""Stage 1.3 A/B: learned value leaf vs the rollout leaf, in the v2 beam.

Runs the v2 solver play search with each leaf evaluator over the same seeds and
compares trajectory quality (final ante / wins) and the fair speed metric
(CPU per play-decision, process_time so it's contention-immune). No production
code is touched — `SolverPolicy(play_policy=...)` injects a `SearchV2PlayPolicy`
built with the chosen leaf.

    PYTHONPATH=src python scripts/phase8_leaf_ab.py --ckpt .data/phase8_value_v3_bootstrap.pt \
        --seeds 16 --depth 3 --width 2 --metrics .data/phase8_leaf_ab.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _run_condition(name, make_leaf, seeds, depth, width):
    from balatro_ai.api.actions import ActionType
    from balatro_ai.api.state import GamePhase
    from balatro_ai.bots.basic_strategy_bot import BasicStrategyBot
    from balatro_ai.sim.local_runner import LocalBalatroSimulator
    from balatro_ai.solver.policy import SolverPolicy
    from balatro_ai.solver.search_v2.play import SearchV2PlayPolicy
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int

    play_policy = SearchV2PlayPolicy(
        depth=depth, width=width, leaf_evaluator=make_leaf(),
        seed=0, fallback=BasicStrategyBot(seed=0))
    solver = SolverPolicy(play_policy=play_policy, play_backend="v2",
                          play_depth=depth, play_width=width, seed=0)

    antes, wins, cpu_total, decisions = [], 0, 0.0, 0
    t_wall = time.perf_counter()
    for seed in seeds:
        sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
        sim.state = SeedGame(seed, stake="white").initial_state()
        for _ in range(2000):
            st = sim.state
            if st.run_over or st.phase == GamePhase.RUN_OVER:
                break
            if st.phase == GamePhase.SELECTING_HAND:
                decisions += 1
            t0 = time.process_time()
            action = solver.choose_action(st)
            cpu_total += time.process_time() - t0
            if action.action_type == ActionType.NO_OP:
                break
            sim.step(action)
        final = sim.state
        antes.append(final.ante)
        wins += int(bool(final.won) or final.ante >= 9)
    wall = time.perf_counter() - t_wall
    return {
        "condition": name,
        "n": len(seeds),
        "mean_ante": round(statistics.mean(antes), 3),
        "median_ante": statistics.median(antes),
        "wins": wins,
        "ms_per_play_decision": round(1000 * cpu_total / max(1, decisions), 1),
        "play_decisions": decisions,
        "wall_s": round(wall, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--width", type=int, default=2)
    ap.add_argument("--head", default="ante")
    ap.add_argument("--metrics", required=True)
    args = ap.parse_args()

    from balatro_ai.ml.leaf import ValueNetLeaf
    from balatro_ai.solver.search_v2.leaf_value import ClearProbabilityLeaf, PlanningValueLeaf

    seeds = [f"{i:07d}" for i in range(1, args.seeds + 1)]
    conditions = [
        ("planning_rollout", lambda: PlanningValueLeaf()),
        ("clear_prob_rollout", lambda: ClearProbabilityLeaf()),
        ("value_net", lambda: ValueNetLeaf(args.ckpt, head=args.head)),
    ]
    results = []
    for name, make_leaf in conditions:
        r = _run_condition(name, make_leaf, seeds, args.depth, args.width)
        results.append(r)
        print("[leaf_ab]", json.dumps(r), flush=True)

    with open(args.metrics, "w", encoding="utf-8") as fh:
        json.dump({"depth": args.depth, "width": args.width, "head": args.head,
                   "results": results}, fh, indent=2)
    vn = next(r for r in results if r["condition"] == "value_net")
    pl = next(r for r in results if r["condition"] == "planning_rollout")
    marker = (f".data/_LEAFAB_vnante{vn['mean_ante']:.2f}_plante{pl['mean_ante']:.2f}"
              f"_vnms{vn['ms_per_play_decision']:.0f}_plms{pl['ms_per_play_decision']:.0f}")
    open(marker, "w").close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
