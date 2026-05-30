"""Accurate cost breakdown of a data-gen run, WITHOUT cProfile distortion.

cProfile over-weights Python functions with many small calls and is blind to
time spent inside the Rust extension, so it wildly over-attributed runtime to
the Python rollout. This instead wraps the real cost centers with
process_time() accumulators (CPU time, including nested Rust) and runs a few
games single-threaded. The wrapped centers are mutually non-nesting, so their
sum is a clean leaf breakdown; the remainder is sim-stepping / enumeration /
shop / misc.

    PYTHONPATH=src python scripts/phase_timing.py [n_seeds]
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict


def main() -> int:
    import balatro_ai.search.state_value as sv
    from balatro_ai.solver.policy import SolverPolicy
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int
    from balatro_ai.sim.local_runner import LocalBalatroSimulator

    t = defaultdict(float)
    c = defaultdict(int)

    # --- wrap the Rust rollout bridge: time + success/bail count ---
    _orig_rust_clear = sv._try_rust_clear_probability
    def timed_rust_clear(state, samples, seed):
        t0 = time.process_time()
        r = _orig_rust_clear(state, samples, seed)
        t["rust_rollout"] += time.process_time() - t0
        c["rust_rollout_calls"] += 1
        if r is None:
            c["rust_rollout_bail"] += 1
        return r
    sv._try_rust_clear_probability = timed_rust_clear

    # --- wrap the Python fallback rollout (whole subtree) ---
    _orig_py_roll = sv._greedy_rollout_clears
    def timed_py_roll(state, rng):
        t0 = time.process_time()
        r = _orig_py_roll(state, rng)
        t["python_rollout"] += time.process_time() - t0
        c["python_rollout_calls"] += 1
        return r
    sv._greedy_rollout_clears = timed_py_roll

    # --- wrap headroom best-immediate-score (separate from rollout) ---
    _orig_bis = sv._best_immediate_score
    def timed_bis(state):
        t0 = time.process_time()
        r = _orig_bis(state)
        t["headroom_bestscore"] += time.process_time() - t0
        c["headroom_calls"] += 1
        return r
    sv._best_immediate_score = timed_bis

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    seeds = [f"BEAMQ{i}" for i in range(n)]
    run_t0 = time.process_time()
    for seed in seeds:
        sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
        sim.state = SeedGame(seed, stake="white").initial_state()
        pol = SolverPolicy(seed=0)
        for _ in range(6000):
            st = sim.state
            if st.run_over:
                break
            a = pol.choose_action(st)
            if a.action_type.value == "no_op":
                break
            sim.step(a)
    total = time.process_time() - run_t0

    print(f"n={n}  total CPU={total:.1f}s ({total/n:.1f}s/run)")
    measured = t["rust_rollout"] + t["python_rollout"] + t["headroom_bestscore"]
    other = total - measured
    def line(label, key):
        v = t[key]
        print(f"  {label:22s} {v:7.1f}s  ({100*v/total:4.1f}%)  calls={c.get(key+'_calls', 0)}")
    line("Rust rollout", "rust_rollout")
    print(f"      └ bail to Python:  {c['rust_rollout_bail']}/{c['rust_rollout_calls']} "
          f"({100*c['rust_rollout_bail']/max(1,c['rust_rollout_calls']):.0f}%)")
    line("Python rollout (bail)", "python_rollout")
    line("Headroom best-score", "headroom_bestscore")
    print(f"  {'OTHER (sim/enum/shop)':22s} {other:7.1f}s  ({100*other/total:4.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
