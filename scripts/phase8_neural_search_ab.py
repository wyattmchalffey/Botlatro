"""Stage 2.3 A/B: neural-guided v2 beam vs the heuristic v2 beam.

Three conditions (same seeds), comparing final ante + CPU/play-decision. The
decomposition isolates the two neural changes independently:
- heuristic     : default v2 (TopKByImmediateScore + PlanningValueLeaf rollout).
- leaf_only     : TopKByImmediateScore + distilled clear leaf  (isolates LEAF).
- neural_full   : policy candidate provider + distilled clear leaf (adds POLICY).
  => leaf effect = leaf_only - heuristic ; policy-prune effect = neural_full - leaf_only.

    PYTHONPATH=src python scripts/phase8_neural_search_ab.py \
        --value-ckpt .data/phase8_clear_v0.pt --policy-ckpt .data/phase8_playpolicy_v0.pt \
        --seeds 12 --metrics .data/phase8_neural_search_ab.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _run(name, make_solver, seeds):
    from balatro_ai.api.actions import ActionType
    from balatro_ai.api.state import GamePhase
    from balatro_ai.sim.local_runner import LocalBalatroSimulator
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int

    solver = make_solver()
    antes, wins, cpu, decisions = [], 0, 0.0, 0
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
            cpu += time.process_time() - t0
            if action.action_type == ActionType.NO_OP:
                break
            sim.step(action)
        final = sim.state
        antes.append(final.ante)
        wins += int(bool(final.won) or final.ante >= 9)
    return {
        "condition": name, "n": len(seeds),
        "mean_ante": round(statistics.mean(antes), 3),
        "median_ante": statistics.median(antes), "wins": wins,
        "ms_per_play_decision": round(1000 * cpu / max(1, decisions), 1),
        "play_decisions": decisions, "wall_s": round(time.perf_counter() - t_wall, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--value-ckpt", required=True)
    ap.add_argument("--policy-ckpt", required=True)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--width", type=int, default=2)
    ap.add_argument("--metrics", required=True)
    args = ap.parse_args()

    from balatro_ai.bots.basic_strategy_bot import BasicStrategyBot
    from balatro_ai.ml.leaf import ValueNetLeaf
    from balatro_ai.ml.neural_search import PolicyCandidateProvider
    from balatro_ai.ml.train import load_checkpoint
    from balatro_ai.solver.policy import SolverPolicy
    from balatro_ai.solver.search_v2.leaf_value import PlanningValueLeaf
    from balatro_ai.solver.search_v2.play import SearchV2PlayPolicy

    d, w = args.depth, args.width
    policy_model = load_checkpoint(args.policy_ckpt)
    seeds = [f"{i:07d}" for i in range(1, args.seeds + 1)]

    def _wrap(play_policy):
        return SolverPolicy(play_policy=play_policy, play_backend="v2",
                            play_depth=d, play_width=w, seed=0)

    def heuristic():
        return SolverPolicy(play_backend="v2", play_depth=d, play_width=w, seed=0)

    def leaf_only():
        return _wrap(SearchV2PlayPolicy(
            depth=d, width=w, leaf_evaluator=ValueNetLeaf(args.value_ckpt, head="clear"),
            candidate_provider=None, seed=0, fallback=BasicStrategyBot(seed=0)))

    def neural_full():
        return _wrap(SearchV2PlayPolicy(
            depth=d, width=w, leaf_evaluator=ValueNetLeaf(args.value_ckpt, head="clear"),
            candidate_provider=PolicyCandidateProvider(policy_model),
            seed=0, fallback=BasicStrategyBot(seed=0)))

    results = []
    for name, mk in [("heuristic", heuristic), ("leaf_only", leaf_only),
                     ("neural_full", neural_full)]:
        r = _run(name, mk, seeds)
        results.append(r)
        print("[neural-ab]", json.dumps(r), flush=True)

    with open(args.metrics, "w", encoding="utf-8") as fh:
        json.dump({"depth": d, "width": w, "results": results}, fh, indent=2)
    h = next(r for r in results if r["condition"] == "heuristic")
    nf = next(r for r in results if r["condition"] == "neural_full")
    open(f".data/_NEURALAB_h{h['mean_ante']:.2f}_nf{nf['mean_ante']:.2f}"
         f"_hms{h['ms_per_play_decision']:.0f}_nfms{nf['ms_per_play_decision']:.0f}", "w").close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
