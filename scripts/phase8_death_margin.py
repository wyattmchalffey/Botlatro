"""Quick pass: where do runs die, and by how much did they miss?

For every LOSS, read the failed blind's score/required at RUN_OVER (the state retains
both). The ratio distribution — bucketed by ante — is a cheap proxy for the build-vs-play
fork: deaths at ~0.9 of required "barely missed" (a better play line likely clears them →
PLAY-limited); deaths at <0.5 "missed by a mile" (no play fixes that → BUILD-limited).

    PYTHONPATH=src python scripts/phase8_death_margin.py --seeds 300 --jobs 8 \
        --metrics .data/phase8_death_margin.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _run_seed(seed):
    from dataclasses import replace
    from balatro_ai.api.actions import ActionType
    from balatro_ai.api.state import GamePhase
    from balatro_ai.bots.config import DEFAULT_CONFIG, bot_config_scope
    from balatro_ai.bots.registry import create_bot
    from balatro_ai.sim.local_runner import LocalBalatroSimulator
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int

    with bot_config_scope(replace(DEFAULT_CONFIG, shop_audit_enabled=False)):
        sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
        sim.state = SeedGame(seed, stake="white").initial_state()
        bot = create_bot("basic_strategy_bot", seed=0)
        for _ in range(3000):
            s = sim.state
            if s.run_over or s.phase == GamePhase.RUN_OVER:
                break
            a = bot.choose_action(s)
            if a is None or a.action_type == ActionType.NO_OP:
                break
            try:
                sim.step(a)
            except (ValueError, IndexError, KeyError, TypeError, AttributeError):
                break
    f = sim.state
    return {
        "seed": seed,
        "won": bool(f.won),
        "ante": int(f.ante),
        "cur": int(f.current_score),
        "req": int(f.required_score),
        "ratio": (f.current_score / f.required_score) if f.required_score > 0 else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--metrics", required=True)
    args = ap.parse_args()

    seeds = [f"{i:07d}" for i in range(1, args.seeds + 1)]
    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            results = list(ex.map(_run_seed, seeds))
    else:
        results = [_run_seed(s) for s in seeds]

    wins = [r for r in results if r["won"]]
    losses = [r for r in results if not r["won"] and r["ratio"] is not None]
    print(f"[death] {len(results)} runs | winrate {len(wins)}/{len(results)} "
          f"({100*len(wins)/len(results):.1f}%) | losses {len(losses)}", flush=True)

    # death-ante histogram (losses)
    by_ante = {}
    for r in losses:
        by_ante.setdefault(r["ante"], []).append(r["ratio"])

    def _buckets(ratios):
        n = len(ratios)
        if not n:
            return {}
        b = {">=0.9": 0, "0.8-0.9": 0, "0.7-0.8": 0, "0.5-0.7": 0, "<0.5": 0}
        for x in ratios:
            if x >= 0.9: b[">=0.9"] += 1
            elif x >= 0.8: b["0.8-0.9"] += 1
            elif x >= 0.7: b["0.7-0.8"] += 1
            elif x >= 0.5: b["0.5-0.7"] += 1
            else: b["<0.5"] += 1
        return {k: round(v / n, 3) for k, v in b.items()}

    all_ratios = [r["ratio"] for r in losses]
    print("\n[death] LOSS death-ante distribution + median miss ratio:")
    ante_summary = {}
    for ante in sorted(by_ante):
        rs = by_ante[ante]
        ante_summary[ante] = {"n": len(rs), "median_ratio": round(statistics.median(rs), 3),
                              "buckets": _buckets(rs)}
        print(f"   ante {ante}: n={len(rs):3d}  median={statistics.median(rs):.2f}  {_buckets(rs)}")

    # late deaths (ante>=5) are what gate reaching the win
    late = [r["ratio"] for r in losses if r["ante"] >= 5]
    out = {
        "n_runs": len(results),
        "winrate": round(len(wins) / max(1, len(results)), 4),
        "n_losses": len(losses),
        "overall_median_loss_ratio": round(statistics.median(all_ratios), 4) if all_ratios else None,
        "overall_buckets": _buckets(all_ratios),
        "late_deaths_ante5plus": {
            "n": len(late),
            "median_ratio": round(statistics.median(late), 4) if late else None,
            "buckets": _buckets(late),
        },
        "by_ante": ante_summary,
    }
    print("\n[death] SUMMARY:", json.dumps({k: out[k] for k in
          ("winrate", "overall_median_loss_ratio", "overall_buckets", "late_deaths_ante5plus")}, indent=2), flush=True)
    with open(args.metrics, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
