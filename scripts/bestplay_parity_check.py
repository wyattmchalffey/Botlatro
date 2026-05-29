"""Prove the Rust best_play_from_hand fast path is decision-identical.

Runs games with BALATRO_BESTPLAY_PARITY=1 so every best_play_from_hand call
computes BOTH the Rust fast path and the pure-Python loop, comparing the
chosen play. Reports total calls, how often Rust bailed to Python, and any
mismatches (with examples). MUST be run single-process (stats live in a
module global).

    BALATRO_BESTPLAY_PARITY=1 PYTHONPATH=src python scripts/bestplay_parity_check.py [n_seeds]
"""

from __future__ import annotations

import sys


def main() -> int:
    import balatro_ai.rules.hand_evaluator as he
    if not he._BESTPLAY_PARITY_CHECK:
        print("ERROR: set BALATRO_BESTPLAY_PARITY=1 to enable parity mode", flush=True)
        return 1

    from balatro_ai.api.state import with_derived_legal_actions
    from balatro_ai.bots.registry import create_bot
    from balatro_ai.sim.local_runner import LocalBalatroSimulator
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    seeds = [f"{i:07d}" for i in range(1, n + 1)]
    for seed in seeds:
        sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
        sim.state = with_derived_legal_actions(SeedGame(seed, stake="white").initial_state())
        bot = create_bot("basic_strategy_bot", seed=0)
        for _ in range(4000):
            st = sim.state
            if st.run_over:
                break
            a = bot.choose_action(st)
            if a.action_type.value == "no_op":
                break
            sim.step(a)
        s = he._BESTPLAY_PARITY_STATS
        print(f"after seed {seed}: calls={s['n']} fast_bail={s['fast_none']} vector_div={s['vector_div']}", flush=True)

    s = he._BESTPLAY_PARITY_STATS
    print("\n=== PARITY SUMMARY (full-vector) ===", flush=True)
    print(f"total best_play_from_hand calls: {s['n']}", flush=True)
    print(f"  rust fast path used:  {s['n'] - s['fast_none']} ({(s['n']-s['fast_none'])/max(1,s['n']):.1%})", flush=True)
    print(f"  rust bailed to python:{s['fast_none']} ({s['fast_none']/max(1,s['n']):.1%})", flush=True)
    print(f"  VECTOR DIVERGENCES:   {s['vector_div']} ({s['vector_div']/max(1,s['n']-s['fast_none']):.1%} of fast-path calls)", flush=True)
    print("\n  divergence rate by joker (divergent calls / fast-path calls present):", flush=True)
    present, dv = s["present"], s["div_jokers"]
    rows = sorted(dv.items(), key=lambda kv: -(kv[1] / max(1, present.get(kv[0], 1))))
    for name, c in rows:
        p = present.get(name, 0)
        print(f"    {name:22} {c:5} div / {p:6} present  ({c/max(1,p):.1%})", flush=True)
    print("\n  divergence count by blind:", dict(s["div_blinds"]), flush=True)
    print("\n  worst-divergence examples:", flush=True)
    for ex in s["examples"]:
        print("   ", ex, flush=True)
    return 0 if s["vector_div"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
