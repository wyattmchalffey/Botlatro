"""Time the production data-gen path (native beam OFF) across a seed batch.

Used to measure the speedup from un-bailing rollout jokers/blinds. Reports
mean per-run compute (timed inside each worker, so parallel scheduling does
not distort it) plus the ante/score distribution (so a speed change can be
checked for any quality drift). Run before AND after a binary change on the
SAME seeds for a controlled speedup number.

    PYTHONPATH=src python scripts/datagen_speed.py [n_seeds] [jobs]
"""

from __future__ import annotations

import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor


def run_task(seed: str) -> dict:
    import os
    import time as _t

    import balatro_ai.search.hand_search as hs
    import balatro_ai.search.state_value as sv
    from balatro_ai.solver.policy import SolverPolicy
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int
    from balatro_ai.sim.local_runner import LocalBalatroSimulator

    hs._NATIVE_BEAM_ENABLED = False  # production path
    # A/B toggle (Python-only, no rebuild): drop a blind from the rollout
    # set to measure its un-bail's speedup. Comma-separated blind names.
    drop = os.environ.get("BALATRO_DROP_ROLLOUT_BLINDS", "")
    if drop:
        sv._RUST_ROLLOUT_BLIND_SAFE = sv._RUST_ROLLOUT_BLIND_SAFE - set(drop.split(","))
    # Beam-param A/B: override play search depth/width to test the
    # speed/quality tradeoff of a shallower or narrower beam.
    pol_kw = {"seed": 0}
    if os.environ.get("BALATRO_PLAY_DEPTH"):
        pol_kw["play_depth"] = int(os.environ["BALATRO_PLAY_DEPTH"])
    if os.environ.get("BALATRO_PLAY_WIDTH"):
        pol_kw["play_width"] = int(os.environ["BALATRO_PLAY_WIDTH"])
    sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
    sim.state = SeedGame(seed, stake="white").initial_state()
    pol = SolverPolicy(**pol_kw)
    # process_time = CPU time of THIS worker process. Between these two
    # calls the worker runs only this task (single-threaded Python + Rust),
    # so the delta is contention-immune per-run CPU — unlike wall-clock,
    # which inflates under the other 13 workers' load.
    t0 = _t.process_time()
    play_decisions = 0
    for _ in range(6000):
        st = sim.state
        if st.run_over:
            break
        if str(st.phase.value) in ("selecting_hand", "playing_blind"):
            play_decisions += 1
        a = pol.choose_action(st)
        if a.action_type.value == "no_op":
            break
        sim.step(a)
    elapsed = _t.process_time() - t0
    s = sim.state
    return {"seed": seed, "ante": s.ante, "score": int(s.current_score),
            "won": bool(getattr(s, "won", False)) or s.ante >= 9, "elapsed": elapsed,
            "play_decisions": play_decisions}


def main() -> int:
    import os as _os
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    # BALATRO_SEED_SET=numeric -> standard winrate seeds (winnable, countable
    # wins) for a quality gate; default BEAMQ family (hard, for speed timing).
    if _os.environ.get("BALATRO_SEED_SET") == "numeric":
        seeds = [f"{i:07d}" for i in range(1, n + 1)]
    else:
        seeds = [f"BEAMQ{i}" for i in range(n)]  # same family as the quality gate
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        rows = list(ex.map(run_task, seeds))
    wall = time.perf_counter() - t0

    cpu = [r["elapsed"] for r in rows]
    antes = [r["ante"] for r in rows]
    scores = [r["score"] for r in rows]
    wins = sum(r["won"] for r in rows)
    decisions = sum(r.get("play_decisions", 0) for r in rows)
    print(f"n={n} jobs={jobs}  wall={wall:.1f}s")
    print(f"  CPU/run (process_time): mean={statistics.mean(cpu):.2f}s "
          f"median={statistics.median(cpu):.2f}s total={sum(cpu):.1f}s")
    print(f"  play decisions: {decisions}  CPU/play-decision: "
          f"{1000*sum(cpu)/max(1,decisions):.1f}ms  (fair speed metric)")
    print(f"  ante  mean={statistics.mean(antes):.2f} median={statistics.median(antes)} max={max(antes)}")
    print(f"  score mean={statistics.mean(scores):.0f} median={statistics.median(scores):.0f}")
    print(f"  wins  {wins}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
