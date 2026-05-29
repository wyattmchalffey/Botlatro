"""Richer winrate diagnostic: per-seed outcome + joker roles -> JSONL.

Runs a bot across a seed set (generic sim by default for speed; set
BALATRO_SEED_FAITHFUL=1 for accuracy), records each run's final state with
joker role classification, writes JSONL, and prints a wins-vs-losses role
comparison so we can see what separates winning builds from losing ones.

    PYTHONPATH=src python scripts/winrate_diag.py [bot] [n] [out.jsonl]
"""

from __future__ import annotations

import json
import sys
from collections import Counter

from balatro_ai.api.state import with_derived_legal_actions
from balatro_ai.bots.basic_strategy.jokers import (
    _static_chip_role_score,
    _static_mult_role_score,
    _static_xmult_role_score,
)
from balatro_ai.bots.registry import create_bot
from balatro_ai.sim.local_runner import LocalBalatroSimulator
from balatro_ai.solver.seed_game import SeedGame
from balatro_ai.solver.trajectory import _stable_seed_int


def _roles_for(name: str) -> dict:
    return {
        "chips": _static_chip_role_score(name),
        "mult": _static_mult_role_score(name),
        "xmult": _static_xmult_role_score(name),
    }


def run_seed(bot_name: str, seed: str, max_steps: int = 4000) -> dict:
    sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white", balatro_seed=seed)
    sim.state = with_derived_legal_actions(SeedGame(seed, stake="white").initial_state())
    bot = create_bot(bot_name, seed=0)
    last_blind = ""
    for _ in range(max_steps):
        st = sim.state
        if st.run_over:
            break
        if st.blind:
            last_blind = st.blind
        a = bot.choose_action(st)
        if a.action_type.value == "no_op":
            break
        sim.step(a)
    s = sim.state
    jokers = [j.name for j in s.jokers]
    role_totals: dict[str, float] = {}
    for jn in jokers:
        for role, val in _roles_for(jn).items():
            role_totals[role] = role_totals.get(role, 0.0) + float(val)
    return {
        "seed": seed, "won": bool(s.won), "ante": s.ante, "blind": last_blind,
        "score": s.current_score, "required": s.required_score, "money": s.money,
        "jokers": jokers, "role_totals": role_totals,
        "has_xmult": role_totals.get("xmult", 0) > 0,
    }


def main() -> int:
    bot_name = sys.argv[1] if len(sys.argv) > 1 else "basic_strategy_bot"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    out = sys.argv[3] if len(sys.argv) > 3 else ".data/winrate_diag.jsonl"
    seeds = [f"{i:07d}" for i in range(1, n + 1)]
    rows = []
    with open(out, "w", encoding="utf-8") as fh:
        for seed in seeds:
            r = run_seed(bot_name, seed)
            rows.append(r)
            fh.write(json.dumps(r) + "\n")
            print(f"{'WIN ' if r['won'] else 'loss'} {seed} ante{r['ante']} xmult={r['has_xmult']} jk={r['jokers']}", flush=True)
    wins = [r for r in rows if r["won"]]
    losses = [r for r in rows if not r["won"]]
    print(f"\n=== {bot_name}: {len(wins)}/{len(rows)} ({len(wins)/len(rows):.1%}) ===", flush=True)
    print(f"xmult present: wins {sum(r['has_xmult'] for r in wins)}/{len(wins)} | losses {sum(r['has_xmult'] for r in losses)}/{len(losses)}", flush=True)
    def avg_role(rows_, role):
        return sum(r["role_totals"].get(role, 0) for r in rows_) / max(1, len(rows_))
    for role in ("chips", "mult", "xmult"):
        print(f"  avg {role}: wins {avg_role(wins, role):.1f} | losses {avg_role(losses, role):.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
