"""Native-beam QUALITY gate (Phase 4d): the metric that actually matters.

Byte-identity to the Python beam is sufficient but NOT necessary — the beam is
a noisy sampling search. For data-gen, what matters is whether the native beam
plays AS WELL as the Python beam, not whether it picks identical actions.

Runs the full SolverPolicy to completion on a seed batch BOTH ways
(BALATRO_NATIVE_BEAM on/off, toggled per-task inside each worker) and compares
the outcome distribution (final ante, score, win) plus per-run compute time.
Parallel across cores via ProcessPoolExecutor (same pattern as
winrate_bench_par.py). Each worker times its own run loop so the ON-vs-OFF
speedup is measured as mean per-run compute, independent of pool scheduling.

    PYTHONPATH=src python scripts/native_beam_quality.py [n_seeds] [start] [jobs]
"""

from __future__ import annotations

import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor


def run_task(args) -> dict:
    seed, native = args
    # Imports inside the worker so each process initializes cleanly (Windows spawn).
    import time as _t

    import balatro_ai.search.hand_search as hs
    from balatro_ai.solver.policy import SolverPolicy
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int
    from balatro_ai.sim.local_runner import LocalBalatroSimulator

    hs._NATIVE_BEAM_ENABLED = bool(native)
    sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
    sim.state = SeedGame(seed, stake="white").initial_state()
    pol = SolverPolicy(seed=0)
    steps = 0
    t0 = _t.perf_counter()
    for _ in range(6000):
        st = sim.state
        if st.run_over:
            break
        a = pol.choose_action(st)
        if a.action_type.value == "no_op":
            break
        sim.step(a)
        steps += 1
    elapsed = _t.perf_counter() - t0
    s = sim.state
    won = bool(getattr(s, "won", False)) or s.ante >= 9
    return {"seed": seed, "native": bool(native), "ante": s.ante,
            "score": int(s.current_score), "won": won, "steps": steps,
            "elapsed": elapsed}


def seed_family(n: int, start: int):
    from balatro_ai.solver.trajectory import _stable_seed_int
    alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    out = []
    h = _stable_seed_int(f"BEAMQ{start}")
    for i in range(n):
        h = (h * 1103515245 + 12345 + i * 7919) & 0xFFFFFFFFFFFF
        out.append("".join(alpha[(h >> (k * 5)) % len(alpha)] for k in range(7)))
    return out


def summarize(tag: str, rows: list[dict]) -> dict:
    antes = [r["ante"] for r in rows]
    scores = [r["score"] for r in rows]
    wins = sum(1 for r in rows if r["won"])
    compute = sum(r["elapsed"] for r in rows)
    print(f"\n[{tag}]  n={len(rows)}  compute={compute:.1f}s ({compute/max(1,len(rows)):.2f}s/run mean)")
    print(f"  ante:  mean={statistics.mean(antes):.2f}  median={statistics.median(antes)}  max={max(antes)}")
    print(f"  score: mean={statistics.mean(scores):.0f}  median={statistics.median(scores):.0f}")
    print(f"  wins:  {wins}/{len(rows)} ({100*wins/len(rows):.1f}%)")
    return {"ante_mean": statistics.mean(antes), "score_mean": statistics.mean(scores),
            "wins": wins, "compute": compute, "rows": {r["seed"]: r for r in rows}}


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    seeds = seed_family(n, start)

    # Fan out ALL tasks (both ON and OFF for every seed) across the pool so
    # every core stays saturated. Per-run compute time is measured inside the
    # worker, so interleaving ON/OFF doesn't distort the speedup number.
    tasks = [(s, False) for s in seeds] + [(s, True) for s in seeds]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        rows = list(ex.map(run_task, tasks))
    wall = time.perf_counter() - t0

    off = [r for r in rows if not r["native"]]
    on = [r for r in rows if r["native"]]
    print(f"total wall: {wall:.1f}s for {len(tasks)} runs on {jobs} workers "
          f"({wall/len(tasks):.2f}s/run wall)")
    a = summarize("native OFF (Python beam)", off)
    b = summarize("native ON  (Rust beam)  ", on)

    print("\n=== DELTA (ON - OFF) ===")
    print(f"  ante_mean:  {b['ante_mean'] - a['ante_mean']:+.3f}")
    print(f"  score_mean: {b['score_mean'] - a['score_mean']:+.0f}")
    print(f"  wins:       {b['wins'] - a['wins']:+d}")
    print(f"  compute speedup (mean per-run): "
          f"{a['compute'] / max(0.001, b['compute']):.2f}x  "
          f"(OFF {a['compute']/len(off):.2f}s vs ON {b['compute']/len(on):.2f}s per run)")
    same_win = sum(1 for s in (r["seed"] for r in off)
                   if a["rows"][s]["won"] == b["rows"][s]["won"])
    print(f"  same win/loss outcome: {same_win}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
