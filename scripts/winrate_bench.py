"""Fast fixed winrate benchmark (generic sim) for before/after tuning.

Runs a bot across a FIXED seed set on the generic (fast) sim and prints
winrate + ante histogram + avg final-score-fraction. Use the SAME seed set
before and after a change to attribute the delta. Generic sim is fine for
build/shop-strategy tuning (strategy quality transfers); validate winners
on the deck-faithful sim afterward.

    PYTHONPATH=src python scripts/winrate_bench.py [bot] [n]
"""

from __future__ import annotations

import sys
from collections import Counter

from balatro_ai.api.state import with_derived_legal_actions
from balatro_ai.bots.registry import create_bot
from balatro_ai.sim.local_runner import LocalBalatroSimulator
from balatro_ai.solver.seed_game import SeedGame
from balatro_ai.solver.trajectory import _stable_seed_int


def run_seed(bot_name: str, seed: str, max_steps: int = 4000) -> dict:
    # generic sim: do NOT pass balatro_seed (faster; random shops/hands).
    sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
    sim.state = with_derived_legal_actions(SeedGame(seed, stake="white").initial_state())
    bot = create_bot(bot_name, seed=0)
    for _ in range(max_steps):
        st = sim.state
        if st.run_over:
            break
        a = bot.choose_action(st)
        if a.action_type.value == "no_op":
            break
        sim.step(a)
    s = sim.state
    return {"won": bool(s.won), "ante": s.ante, "frac": s.current_score / max(1, s.required_score)}


def main() -> int:
    bot_name = sys.argv[1] if len(sys.argv) > 1 else "basic_strategy_bot"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    seeds = [f"{i:07d}" for i in range(1, n + 1)]
    rows = [run_seed(bot_name, s) for s in seeds]
    wins = sum(r["won"] for r in rows)
    print(f"{bot_name}: winrate {wins}/{len(rows)} ({wins/len(rows):.1%})", flush=True)
    print("ante reached:", dict(sorted(Counter(r["ante"] for r in rows).items())), flush=True)
    losses = [r for r in rows if not r["won"]]
    if losses:
        import statistics
        print(f"loss frac: median={statistics.median(r['frac'] for r in losses):.2f} mean={statistics.mean(r['frac'] for r in losses):.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
