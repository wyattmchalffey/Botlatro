"""Parallel fixed winrate benchmark (multiprocessing across seeds).

Per-seed runs are independent, so we fan them out across worker processes.
On a 16-core box this turns a ~30-50 min single-threaded 100-seed benchmark
into a few minutes. Use the SAME seed set before/after a change.

    PYTHONPATH=src python scripts/winrate_bench_par.py [bot] [n] [jobs] [faithful]
        faithful=1 -> deck-faithful sim (slower, accurate); default 0 (generic)
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace


def run_seed(args) -> dict:
    bot_name, seed, faithful = args
    # Imports inside the worker so each process initializes cleanly.
    from balatro_ai.api.state import with_derived_legal_actions
    from balatro_ai.bots.config import DEFAULT_CONFIG, bot_config_scope
    from balatro_ai.bots.registry import create_bot
    from balatro_ai.sim.local_runner import LocalBalatroSimulator
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int

    kw = {"seed": _stable_seed_int(seed), "stake": "white"}
    if faithful:
        kw["balatro_seed"] = seed
    sim = LocalBalatroSimulator(**kw)
    sim.state = with_derived_legal_actions(SeedGame(seed, stake="white").initial_state())
    bot = create_bot(bot_name, seed=0)
    with bot_config_scope(replace(DEFAULT_CONFIG, shop_audit_enabled=False)):
        for _ in range(4000):
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
    jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    faithful = bool(int(sys.argv[4])) if len(sys.argv) > 4 else False
    seeds = [f"{i:07d}" for i in range(1, n + 1)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        rows = list(ex.map(run_seed, [(bot_name, s, faithful) for s in seeds]))
    dt = time.perf_counter() - t0
    wins = sum(r["won"] for r in rows)
    from balatro_ai.bench_stats import wilson_ci
    lo, hi = wilson_ci(wins, len(rows))
    print(f"{bot_name} ({'faithful' if faithful else 'generic'}, {jobs} jobs): "
          f"winrate {wins}/{len(rows)} ({wins/len(rows):.1%}, 95% CI {lo:.1%}..{hi:.1%}) "
          f"in {dt:.1f}s ({dt/len(rows):.2f}s/seed wall)", flush=True)
    print("ante reached:", dict(sorted(Counter(r["ante"] for r in rows).items())), flush=True)
    import statistics as _st
    print(f"mean ante: {_st.mean(r['ante'] for r in rows):.3f}", flush=True)
    losses = [r for r in rows if not r["won"]]
    if losses:
        import statistics
        print(f"loss frac: median={statistics.median(r['frac'] for r in losses):.2f} mean={statistics.mean(r['frac'] for r in losses):.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
