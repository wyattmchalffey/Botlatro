"""Forensic: hand-leveling + build investment in wins vs losses.

For each seed, records the bot's final hand levels (primary/most-leveled
hand + total leveling), joker count, and outcome. Leveling one hand high
via planets is the dominant scaling mechanic; if losing builds under-level,
that's a systematic lever.

    PYTHONPATH=src python scripts/level_forensic.py [n]
"""

from __future__ import annotations

import sys
import statistics

from balatro_ai.api.state import with_derived_legal_actions
from balatro_ai.bots.registry import create_bot
from balatro_ai.sim.local_runner import LocalBalatroSimulator
from balatro_ai.solver.seed_game import SeedGame
from balatro_ai.solver.trajectory import _stable_seed_int


def run_seed(seed: str, max_steps: int = 4000) -> dict:
    sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
    sim.state = with_derived_legal_actions(SeedGame(seed, stake="white").initial_state())
    bot = create_bot("basic_strategy_bot", seed=0)
    for _ in range(max_steps):
        st = sim.state
        if st.run_over:
            break
        a = bot.choose_action(st)
        if a.action_type.value == "no_op":
            break
        sim.step(a)
    s = sim.state
    levels = {k: int(v) for k, v in s.hand_levels.items()}
    primary = max(levels, key=lambda k: levels[k]) if levels else None
    return {
        "won": bool(s.won), "ante": s.ante,
        "primary": primary, "primary_level": levels.get(primary, 1) if primary else 1,
        "total_leveling": sum(max(0, v - 1) for v in levels.values()),
        "n_jokers": len(s.jokers),
        "money": s.money,
    }


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    rows = [run_seed(f"{i:07d}") for i in range(1, n + 1)]
    wins = [r for r in rows if r["won"]]
    losses = [r for r in rows if not r["won"]]
    def avg(rows_, key):
        return statistics.mean(r[key] for r in rows_) if rows_ else 0.0
    print(f"winrate {len(wins)}/{len(rows)}", flush=True)
    print(f"primary hand level: wins {avg(wins,'primary_level'):.1f} | losses {avg(losses,'primary_level'):.1f}", flush=True)
    print(f"total leveling:     wins {avg(wins,'total_leveling'):.1f} | losses {avg(losses,'total_leveling'):.1f}", flush=True)
    print(f"n_jokers:           wins {avg(wins,'n_jokers'):.1f} | losses {avg(losses,'n_jokers'):.1f}", flush=True)
    print(f"final money:        wins {avg(wins,'money'):.1f} | losses {avg(losses,'money'):.1f}", flush=True)
    from collections import Counter
    print("primary hand (wins):", Counter(r["primary"] for r in wins).most_common(), flush=True)
    print("primary hand (losses):", Counter(r["primary"] for r in losses).most_common(6), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
