"""S-pre Part 2: per-ante instrumented capture of the DEPLOYED bot, to profile winners vs losers.

Distinguishes thesis A (scaling basin: winners own COMPOUNDER jokers losers don't; build-capacity
grows multiplicatively) from thesis B (concentration+economy: winners level ONE hand type + bank
interest; the joker gap is additive). Captures at every blind-select via a policy wrapper inside
generate_trajectory (deployed SolverPolicy shop + BasicStrategy play).

    PYTHONPATH=src py -3.12 scripts/s_pre_winloss_capture.py --seeds 150 --seed-offset 3000000 \
        --jobs 8 --out .data/s_pre_winloss_150.json
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

DECAY = {"Gros Michel", "Ice Cream", "Popcorn", "Ramen", "Turtle Bean", "Luchador",
         "Diet Cola", "Invisible Joker", "Hallucination", "Mr. Bones", "Egg"}
RETRIGGER = {"Hanging Chad", "Mime", "Sock and Buskin", "Dusk", "Hack", "Seltzer"}
HAND_TYPES = ("Flush", "Straight", "Two Pair", "Pair", "Three of a Kind", "High Card",
              "Four of a Kind", "Straight Flush", "Full House", "Five of a Kind")


def _worker(arg):
    seed, stake = arg
    from balatro_ai.api.actions import Action, ActionType
    from balatro_ai.api.state import GamePhase
    from balatro_ai.bots.basic_strategy_bot import BasicStrategyBot
    from balatro_ai.bots.config import DEFAULT_CONFIG, bot_config_scope
    from balatro_ai.solver.policy import SolverPolicy
    from balatro_ai.solver.trajectory import generate_trajectory
    import balatro_ai.bots.basic_strategy.data as d
    from dataclasses import replace as dcr
    try:
        from balatro_ai.bots.basic_strategy.build_scoring import _sample_build_score
    except Exception:
        _sample_build_score = None

    SCALING = set(d.SCALING_JOKERS); XMULT = set(d.XMULT_JOKERS)
    ECON = set(getattr(d, "JOKER_ECONOMY_VALUES", {}))

    def jtype(name):
        if name in RETRIGGER: return "retrigger"
        if name in DECAY: return "decay"
        if name in SCALING or name in XMULT: return "compounder"
        if name in ECON: return "economy"
        return "additive"

    fb = BasicStrategyBot(seed=0)
    policy = SolverPolicy(play_policy=fb, fallback=fb, seed=0,
                          prefer_fallback_info_first_shop=True, fallback_negative_shop_sells=True)
    snaps = {}

    def feats(state):
        jt = {"compounder": 0, "retrigger": 0, "decay": 0, "economy": 0, "additive": 0}
        for j in state.jokers:
            jt[jtype(j.name)] += 1
        hl = state.hand_levels or {}
        levels = {ht: int(hl.get(ht, 1) or 1) for ht in HAND_TYPES}
        leveled = {ht: lv for ht, lv in levels.items() if lv > 1}
        max_lv = max(levels.values()) if levels else 1
        top_lv_type = max(levels, key=levels.get) if levels else None
        # played-hand concentration
        phc = {}
        for key in ("played_hand_counts", "hand_counts", "hands_played"):
            v = state.modifiers.get(key)
            if isinstance(v, dict) and v:
                phc = {str(k): int(x) for k, x in v.items() if isinstance(x, (int, float))}
                break
        tot = sum(phc.values()) or 1
        top_played_frac = max(phc.values()) / tot if phc else 0.0
        bscore = 0.0
        if _sample_build_score is not None:
            try:
                bscore = float(_sample_build_score(state, state.jokers))
            except Exception:
                bscore = 0.0
        return {
            "ante": int(state.ante), "money": int(state.money), "n_jokers": len(state.jokers),
            "compounder": jt["compounder"], "retrigger": jt["retrigger"], "decay": jt["decay"],
            "economy": jt["economy"], "additive": jt["additive"],
            "max_hand_level": max_lv, "n_types_leveled": len(leveled),
            "sum_levels": sum(levels.values()), "top_level_type": top_lv_type,
            "top_played_frac": round(top_played_frac, 3),
            "build_score": round(bscore, 1),
        }

    def wrapped(state):
        if state.phase == GamePhase.BLIND_SELECT:
            a = int(state.ante)
            if a not in snaps:
                snaps[a] = feats(state)
        return policy.choose_action(state)

    with bot_config_scope(dcr(DEFAULT_CONFIG, shop_audit_enabled=False)):
        traj = generate_trajectory(seed, wrapped, stake=stake, max_steps=5000, record_steps=False)
    return {"seed": seed, "won": bool(traj.won), "final_ante": int(traj.final_ante), "snaps": snaps}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=150)
    ap.add_argument("--seed-offset", type=int, default=3000000)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--stake", default="white")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    seeds = [f"{args.seed_offset + i:07d}" for i in range(1, args.seeds + 1)]
    print(f"[s-pre p2] {len(seeds)} deployed-bot runs, per-ante instrumented capture, stake={args.stake}", flush=True)
    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            rows = list(ex.map(_worker, [(s, args.stake) for s in seeds]))
    else:
        rows = [_worker((s, args.stake)) for s in seeds]
    wins = sum(r["won"] for r in rows)
    print(f"winrate {wins}/{len(rows)} ({wins/len(rows):.1%})", flush=True)
    json.dump(rows, open(args.out, "w", encoding="utf-8"), indent=1)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
