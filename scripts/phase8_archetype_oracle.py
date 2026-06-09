"""Archetype planner Phase 0: oracle leverage gate.

For each held-out seed, run the DEPLOYED composition (SolverPolicy shop + BasicStrategy play +
the deployed shop flags) once per archetype in {None(baseline), flush, scaling_joker,
high_card_mult, pair_retrigger} and keep the best (won > ante > score). Compares:
  - baseline winrate (archetype=None, ~ the live 19.5% bot) vs
  - oracle best-of-archetypes winrate (upper bound on perfect live archetype selection).
Also records which archetype wins each seed -> labels for a future neural selector.

A big oracle>baseline gap => correct archetype commitment has real headroom => build the
live selector. A small gap => the archetype leaf is too weak; strengthen fit-score first.

    PYTHONPATH=src py -3.12 scripts/phase8_archetype_oracle.py --seeds 48 --jobs 8 \
        --out .data/archetype_oracle.json
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
    seed, stake = arg
    from dataclasses import replace
    from balatro_ai.bots.basic_strategy_bot import BasicStrategyBot
    from balatro_ai.bots.config import DEFAULT_CONFIG, bot_config_scope
    from balatro_ai.solver.archetypes import BUILT_IN_ARCHETYPES
    from balatro_ai.solver.policy import SolverPolicy
    from balatro_ai.solver.trajectory import generate_trajectory

    cfg = replace(DEFAULT_CONFIG, shop_audit_enabled=False)
    results = {}
    for arch in [None, *BUILT_IN_ARCHETYPES]:
        fb = BasicStrategyBot(seed=0)
        kwargs = dict(play_policy=fb, fallback=fb, seed=0,
                      prefer_fallback_info_first_shop=True, fallback_negative_shop_sells=True)
        if arch is not None:
            kwargs["archetype"] = arch
        policy = SolverPolicy(**kwargs)
        with bot_config_scope(cfg):
            traj = generate_trajectory(seed, policy.choose_action, stake=stake,
                                       max_steps=5000, record_steps=False)
        name = "baseline" if arch is None else arch.name
        results[name] = [bool(traj.won), int(traj.final_ante), int(traj.final_score)]
    best_name = max(results, key=lambda k: (results[k][0], results[k][1], results[k][2]))
    return {"seed": seed, "results": results, "best_name": best_name}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=48)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--stake", default="white")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    seeds = [f"{args.seed_offset + i:07d}" for i in range(1, args.seeds + 1)]
    print(f"[arch-oracle] {len(seeds)} seeds x 5 attempts (baseline + 4 archetypes), basic play, stake={args.stake}", flush=True)

    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            rows = list(ex.map(_worker, [(s, args.stake) for s in seeds]))
    else:
        rows = [_worker((s, args.stake)) for s in seeds]

    n = len(rows)
    base_wins = sum(r["results"]["baseline"][0] for r in rows)
    oracle_wins = sum(r["results"][r["best_name"]][0] for r in rows)
    base_ante = statistics.mean(r["results"]["baseline"][1] for r in rows)
    oracle_ante = statistics.mean(r["results"][r["best_name"]][1] for r in rows)
    # an archetype "helped" a seed if a non-baseline attempt strictly beats baseline by the sort key
    def _key(v): return (v[0], v[1], v[2])
    helped = sum(1 for r in rows
                 if max((_key(v) for k, v in r["results"].items() if k != "baseline"), default=(0, 0, 0))
                 > _key(r["results"]["baseline"]))
    # per-archetype: how often each is the unique best / wins
    best_dist = Counter(r["best_name"] for r in rows)
    # archetype win-rates standalone (each archetype's own winrate)
    arch_names = [k for k in rows[0]["results"] if k != "baseline"]
    arch_wr = {a: round(sum(r["results"][a][0] for r in rows) / n, 3) for a in arch_names}

    out = {
        "n": n,
        "baseline_winrate": round(base_wins / n, 3), "baseline_wins": base_wins,
        "oracle_winrate": round(oracle_wins / n, 3), "oracle_wins": oracle_wins,
        "winrate_lift": round((oracle_wins - base_wins) / n, 3),
        "baseline_mean_ante": round(base_ante, 3), "oracle_mean_ante": round(oracle_ante, 3),
        "seeds_archetype_helped": helped, "seeds_archetype_helped_rate": round(helped / n, 3),
        "best_archetype_distribution": dict(best_dist),
        "standalone_archetype_winrates": arch_wr,
        "labels": {r["seed"]: r["best_name"] for r in rows},
        "rows": rows,
    }
    print(json.dumps({k: v for k, v in out.items() if k != "labels"}, indent=2), flush=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
