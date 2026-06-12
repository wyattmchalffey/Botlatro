"""Generic paired A/B for a single env-gated knob (read at call time by the bot).

Runs each seed twice in the same worker -- baseline (env unset) and treatment (env=VAL)
-- on identical RNG, so the comparison is paired. Reports winrate, mean ante, and flips.

    PYTHONPATH=src py -3.12 scripts/env_paired_ab.py ENVVAR VALUE [n] [jobs] [seed_offset] [rows_jsonl]

rows_jsonl (optional): persist each completed pair as a JSONL line and RESUME
from it on restart — interrupted runs lose at most the in-flight pairs.
    e.g. scripts/env_paired_ab.py BALATRO_DECAY_W 1.6 128 8 4000000
"""
from __future__ import annotations
import os, sys, time
from concurrent.futures import ProcessPoolExecutor


def _run_one(seed: str) -> dict:
    from dataclasses import replace
    from balatro_ai.bots.config import DEFAULT_CONFIG, bot_config_scope
    from balatro_ai.bots.registry import create_bot
    from balatro_ai.sim.local_runner import LocalBalatroSimulator
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int
    sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
    sim.state = SeedGame(seed, stake="white").initial_state()
    bot = create_bot("solver_shop_basic_play_bot", seed=0)
    with bot_config_scope(replace(DEFAULT_CONFIG, shop_audit_enabled=False)):
        for _ in range(6000):
            st = sim.state
            if st.run_over:
                break
            a = bot.choose_action(st)
            if a.action_type.value == "no_op":
                break
            sim.step(a)
    s = sim.state
    return {"won": bool(s.won), "ante": s.ante}


def _worker(arg):
    seed, env, val = arg
    os.environ.pop(env, None)
    base = _run_one(seed)
    os.environ[env] = val
    treat = _run_one(seed)
    os.environ.pop(env, None)
    return {"seed": seed, "base": base, "treat": treat}


def main():
    env = sys.argv[1]
    val = sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    jobs = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    off = int(sys.argv[5]) if len(sys.argv) > 5 else 4000000
    rows_path = sys.argv[6] if len(sys.argv) > 6 else None
    seeds = [f"{off + i:07d}" for i in range(1, n + 1)]
    rows = []
    if rows_path:
        import json as _json
        import os as _os
        if _os.path.exists(rows_path):
            with open(rows_path, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        rows.append(_json.loads(line))
            done_seeds = {r["seed"] for r in rows}
            seeds = [s for s in seeds if s not in done_seeds]
            print(f"  [resume] {len(rows)} pairs loaded from {rows_path}; {len(seeds)} to run", flush=True)
    t0 = time.perf_counter()
    fresh = 0
    if seeds:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            from concurrent.futures import as_completed
            futs = {ex.submit(_worker, (s, env, val)): s for s in seeds}
            for fut in as_completed(futs):
                row = fut.result()
                rows.append(row)
                fresh += 1
                if rows_path:
                    import json as _json
                    with open(rows_path, "a", encoding="utf-8") as fh:
                        fh.write(_json.dumps(row) + "\n")
                if fresh % 16 == 0 or len(rows) == n:
                    el = time.perf_counter() - t0
                    print(f"  [pairs {len(rows)}/{n}] {el:.0f}s elapsed (~{el/max(1,fresh):.1f}s/pair this session)", flush=True)
    dt = time.perf_counter() - t0
    import statistics as st
    bw = sum(r["base"]["won"] for r in rows)
    tw = sum(r["treat"]["won"] for r in rows)
    gained = [r["seed"] for r in rows if r["treat"]["won"] and not r["base"]["won"]]
    lost = [r["seed"] for r in rows if r["base"]["won"] and not r["treat"]["won"]]
    print(f"{env}={val}  n={n} jobs={jobs} offset={off}  {dt:.0f}s", flush=True)
    print(f"  baseline : {bw}/{n} ({bw/n:.1%})  mean ante {st.mean(r['base']['ante'] for r in rows):.3f}", flush=True)
    print(f"  treat    : {tw}/{n} ({tw/n:.1%})  mean ante {st.mean(r['treat']['ante'] for r in rows):.3f}", flush=True)
    print(f"  net wins : {tw-bw:+d}   gained {len(gained)} {gained}   lost {len(lost)} {lost}", flush=True)
    from balatro_ai.bench_stats import mcnemar_exact_p, paired_delta_ci
    pval = mcnemar_exact_p(len(gained), len(lost))
    lo, hi = paired_delta_ci(len(gained), len(lost), n)
    print(f"  McNemar exact p={pval:.4f}   d_winrate {(len(gained)-len(lost))/n:+.4f} (95% CI {lo:+.4f}..{hi:+.4f})", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
