"""Archetype planner Phase 1: conservative live selector, A/B vs baseline.

Phase 0 oracle showed +6% ceiling (flush-dominated) but standalone commit HURTS, so the
selector must be conservative. This commits to the archetype the build is already leaning into
(>= threshold owned key jokers) with hysteresis (keep while >=1 key joker remains; else drop to
baseline). Sets SolverPolicy.archetype per-decision (drives ArchetypeAwareLeaf). Same deployed
composition (solver shop + BasicStrategy play) as solver_shop_basic_play_bot.

    PYTHONPATH=src py -3.12 scripts/phase8_archetype_planner_ab.py --seeds 150 --jobs 8 \
        --out .data/archetype_planner_ab.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _worker(arg):
    seed, condition = arg
    from dataclasses import replace
    from balatro_ai.bots.basic_strategy_bot import BasicStrategyBot
    from balatro_ai.bots.config import DEFAULT_CONFIG, bot_config_scope
    from balatro_ai.solver.archetypes import BUILT_IN_ARCHETYPES, FLUSH_ARCHETYPE, _joker_key
    from balatro_ai.solver.policy import SolverPolicy
    from balatro_ai.solver.trajectory import generate_trajectory

    # Oracle (Phase 0): flush is the only consistently-useful archetype; committing to
    # broad-list scaling/pair HURTS. So focus on flush; "general_t2" kept as a control.
    cond = {
        "baseline": (None, -1),
        "flush_t1": ([FLUSH_ARCHETYPE], 1),
        "flush_t2": ([FLUSH_ARCHETYPE], 2),
        "general_t2": (list(BUILT_IN_ARCHETYPES), 2),
    }[condition]
    candidates, threshold = cond
    fb = BasicStrategyBot(seed=0)
    policy = SolverPolicy(play_policy=fb, fallback=fb, seed=0,
                          prefer_fallback_info_first_shop=True, fallback_negative_shop_sells=True)
    committed = [None]
    commits = Counter()

    def _owned(state, arch):
        keys = arch.key_joker_keys
        return sum(1 for j in state.jokers if _joker_key(j) in keys)

    def _select(state):
        if candidates is None:
            return None
        scored = [(a, _owned(state, a)) for a in candidates]
        best_a, best_n = max(scored, key=lambda x: x[1])
        if best_n >= threshold:
            committed[0] = best_a
        if committed[0] is not None:
            if _owned(state, committed[0]) >= 1:
                return committed[0]
            committed[0] = None
        return None

    def choose(state):
        sel = _select(state)
        policy.archetype = sel
        if sel is not None:
            commits[sel.name] += 1
        return policy.choose_action(state)

    with bot_config_scope(replace(DEFAULT_CONFIG, shop_audit_enabled=False)):
        traj = generate_trajectory(seed, choose, stake="white", max_steps=5000, record_steps=False)
    return {"seed": seed, "condition": condition, "won": bool(traj.won),
            "ante": int(traj.final_ante), "commits": dict(commits)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=150)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    seeds = [f"{args.seed_offset + i:07d}" for i in range(1, args.seeds + 1)]
    conds = ["baseline", "flush_t1", "flush_t2", "general_t2"]
    jobs = [(s, c) for c in conds for s in seeds]
    print(f"[arch-planner] {len(seeds)} seeds x {len(conds)} conditions (rust default ON)", flush=True)

    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            rows = list(ex.map(_worker, jobs))
    else:
        rows = [_worker(j) for j in jobs]

    out = {"n": len(seeds), "results": {}}
    for c in conds:
        cr = [r for r in rows if r["condition"] == c]
        wins = sum(r["won"] for r in cr)
        commits = Counter()
        for r in cr:
            commits.update(r["commits"])
        out["results"][c] = {
            "winrate": round(wins / len(cr), 3), "wins": wins,
            "mean_ante": round(statistics.mean(r["ante"] for r in cr), 3),
            "total_commit_decisions": dict(commits.most_common()),
        }
    for c in conds:
        print(f"[arch-planner] {c}: {json.dumps(out['results'][c])}", flush=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
