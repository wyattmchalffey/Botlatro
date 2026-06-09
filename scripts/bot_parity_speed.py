"""Parity + speed harness for behavior-preserving optimizations.

Runs the deployed winrate bot (solver_shop_basic_play_bot) on a fixed seed set
and emits, per seed, a SIGNATURE that captures the full trajectory (every chosen
action's content + the final game state) plus the per-run process_time CPU cost.

Use it as an A/B: capture a baseline digest, apply a change, re-capture. If the
per-seed signatures are IDENTICAL the change is behavior-preserving (same
decisions, same outcomes); the CPU totals give the speed delta. Any float-reorder
or logic drift shows up as a signature mismatch.

    PYTHONPATH=src python scripts/bot_parity_speed.py [n] [jobs] [bot] [faithful] > digest.txt
"""

from __future__ import annotations

import hashlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor


def _action_sig(state, action) -> str:
    at = action.action_type.value
    parts = [str(state.phase.value), at]
    for attr in ("card_ids", "target_id", "index", "amount", "shop_index", "pack_index", "card_index"):
        v = getattr(action, attr, None)
        if v is not None:
            parts.append(f"{attr}={v}")
    return "|".join(parts)


def run_task(args):
    seed, bot_name, faithful = args
    import time as _t
    from dataclasses import replace

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
    h = hashlib.blake2b(digest_size=16)
    nsteps = 0
    cpu0 = _t.process_time()
    with bot_config_scope(replace(DEFAULT_CONFIG, shop_audit_enabled=False)):
        for _ in range(4000):
            st = sim.state
            if st.run_over:
                break
            a = bot.choose_action(st)
            h.update(_action_sig(st, a).encode())
            nsteps += 1
            if a.action_type.value == "no_op":
                break
            sim.step(a)
    cpu = _t.process_time() - cpu0
    s = sim.state
    final = f"won={bool(s.won)};ante={s.ante};score={s.current_score};money={getattr(s,'money',None)};njok={len(s.jokers)};steps={nsteps}"
    h.update(final.encode())
    return {"seed": seed, "sig": h.hexdigest(), "final": final, "cpu": cpu, "steps": nsteps,
            "won": bool(s.won), "ante": s.ante}


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    bot_name = sys.argv[3] if len(sys.argv) > 3 else "solver_shop_basic_play_bot"
    faithful = bool(int(sys.argv[4])) if len(sys.argv) > 4 else False
    seeds = [f"{i:07d}" for i in range(1, n + 1)]
    wall0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        rows = list(ex.map(run_task, [(s, bot_name, faithful) for s in seeds]))
    wall = time.perf_counter() - wall0
    rows.sort(key=lambda r: r["seed"])
    total_cpu = sum(r["cpu"] for r in rows)
    wins = sum(r["won"] for r in rows)
    combined = hashlib.blake2b(digest_size=16)
    for r in rows:
        combined.update(r["seed"].encode())
        combined.update(r["sig"].encode())
    print(f"# bot={bot_name} faithful={faithful} n={n} jobs={jobs}")
    print(f"# wall={wall:.1f}s total_cpu={total_cpu:.1f}s ({total_cpu/n:.2f}s/run) wins={wins}/{n}")
    print(f"# COMBINED_SIG={combined.hexdigest()}")
    for r in rows:
        print(f"{r['seed']} {r['sig']} cpu={r['cpu']:.2f} {r['final']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
