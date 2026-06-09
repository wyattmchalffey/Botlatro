"""Contrast winning vs losing solver builds at terminal.

The death-margin pass shows the solver is build-limited at antes 5-8 (engine does
not scale). This pass asks *what* the winning builds have that the losing builds
lack -- joker count, hand-leveling investment, money, death blind -- so a better
shop-search leaf value knows what to reward.

    PYTHONPATH=src py -3.12 scripts/build_profile.py --seeds 150 --jobs 8 \
        --bot solver_shop_basic_play_bot --out .data/build_profile_solver.json
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _run_seed(seed, bot_name="solver_shop_basic_play_bot"):
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
        bot = create_bot(bot_name, seed=0)
        for _ in range(4000):
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
    s = sim.state
    levels = dict(s.hand_levels or {})
    level_invest = sum(max(0, int(v) - 1) for v in levels.values())
    max_level = max((int(v) for v in levels.values()), default=1)
    return {
        "seed": seed,
        "won": bool(s.won),
        "ante": int(s.ante),
        "blind": str(s.blind or ""),
        "n_jokers": len(s.jokers),
        "jokers": [j.name for j in s.jokers],
        "level_invest": int(level_invest),
        "max_hand_level": int(max_level),
        "money": int(s.money),
        "deck_size": int(s.deck_size),
    }


def _grp(rows):
    if not rows:
        return {}
    jk = Counter(j for r in rows for j in r["jokers"])
    return {
        "n": len(rows),
        "mean_n_jokers": round(statistics.mean(r["n_jokers"] for r in rows), 2),
        "mean_level_invest": round(statistics.mean(r["level_invest"] for r in rows), 2),
        "mean_max_hand_level": round(statistics.mean(r["max_hand_level"] for r in rows), 2),
        "mean_money": round(statistics.mean(r["money"] for r in rows), 1),
        "mean_deck_size": round(statistics.mean(r["deck_size"] for r in rows), 1),
        "top_jokers": jk.most_common(15),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=150)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--bot", default="solver_shop_basic_play_bot")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    seeds = [f"{i:07d}" for i in range(1, args.seeds + 1)]
    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            rows = list(ex.map(_run_seed, seeds, [args.bot] * len(seeds)))
    else:
        rows = [_run_seed(s, args.bot) for s in seeds]

    wins = [r for r in rows if r["won"]]
    losses = [r for r in rows if not r["won"]]
    late = [r for r in losses if r["ante"] >= 5]
    out = {
        "bot": args.bot,
        "n": len(rows),
        "winrate": round(len(wins) / max(1, len(rows)), 4),
        "wins": _grp(wins),
        "losses": _grp(losses),
        "late_losses_ante5plus": _grp(late),
        "death_blind_counts": dict(Counter(r["blind"] for r in losses).most_common(20)),
    }
    print(json.dumps(out, indent=2), flush=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
