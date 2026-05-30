"""Accurate, REPRESENTATIVE cost breakdown of data-gen, aggregated over seeds.

cProfile over-weights Python's many small calls and is blind to Rust time, so
it mis-attributed runtime. A 3-seed single-threaded sample also misled (it
over-sampled boss/Wall-heavy games). This wraps the real cost centers with
process_time() (CPU time, contention-immune even under the pool) and aggregates
across N seeds in parallel, so the breakdown reflects a representative
population, not a cherry-picked few.

Wrapped centers are mutually non-nesting; their sum is a clean leaf breakdown,
remainder is beam expand/enum.

    PYTHONPATH=src python scripts/phase_timing.py [n_seeds] [jobs]
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor


def run_task(seed: str):
    import time as _t

    import balatro_ai.search.hand_search as hs
    import balatro_ai.search.state_value as sv
    from balatro_ai.solver.policy import SolverPolicy
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int
    from balatro_ai.sim.local_runner import LocalBalatroSimulator

    hs._NATIVE_BEAM_ENABLED = False
    t = defaultdict(float)
    c = defaultdict(int)

    _orig_rust_clear = sv._try_rust_clear_probability
    def timed_rust_clear(state, samples, seed):
        t0 = _t.process_time()
        r = _orig_rust_clear(state, samples, seed)
        t["rust_rollout"] += _t.process_time() - t0
        c["rust_rollout_calls"] += 1
        if r is None:
            c["rust_rollout_bail"] += 1
        return r
    sv._try_rust_clear_probability = timed_rust_clear

    _orig_py_roll = sv._greedy_rollout_clears
    def timed_py_roll(state, rng):
        t0 = _t.process_time()
        r = _orig_py_roll(state, rng)
        t["python_rollout"] += _t.process_time() - t0
        c["python_rollout_calls"] += 1
        return r
    sv._greedy_rollout_clears = timed_py_roll

    _orig_bis = sv._best_immediate_score
    def timed_bis(state):
        t0 = _t.process_time()
        r = _orig_bis(state)
        t["headroom_bestscore"] += _t.process_time() - t0
        return r
    sv._best_immediate_score = timed_bis

    # --- beam-expand sub-breakdown (the 41.7% bucket) ---
    _orig_sim = hs._simulate_hand_action
    def timed_sim(state, action, drawn):
        t0 = _t.process_time()
        r = _orig_sim(state, action, drawn)
        t["beam_simulate"] += _t.process_time() - t0
        return r
    hs._simulate_hand_action = timed_sim

    _orig_wdla = hs.with_derived_legal_actions
    def timed_wdla(state):
        t0 = _t.process_time()
        r = _orig_wdla(state)
        t["beam_legal_actions"] += _t.process_time() - t0
        return r
    hs.with_derived_legal_actions = timed_wdla

    _orig_bfa = hs._beam_future_actions
    def timed_bfa(state, config):
        t0 = _t.process_time()
        r = _orig_bfa(state, config)
        t["beam_enum"] += _t.process_time() - t0
        return r
    hs._beam_future_actions = timed_bfa

    _orig_bra = hs._beam_root_actions
    def timed_bra(state, config, context):
        t0 = _t.process_time()
        r = _orig_bra(state, config, context)
        t["beam_root_enum"] += _t.process_time() - t0
        return r
    hs._beam_root_actions = timed_bra

    sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
    sim.state = SeedGame(seed, stake="white").initial_state()
    pol = SolverPolicy(seed=0)
    run_t0 = _t.process_time()
    for _ in range(6000):
        st = sim.state
        if st.run_over:
            break
        phase = str(st.phase.value)
        ca0 = _t.process_time()
        a = pol.choose_action(st)
        dt = _t.process_time() - ca0
        if phase in ("selecting_hand", "playing_blind"):
            t["ca_play"] += dt
        elif phase == "shop":
            t["ca_shop"] += dt
        else:
            t["ca_other"] += dt
        if a.action_type.value == "no_op":
            break
        s0 = _t.process_time()
        sim.step(a)
        t["sim_step"] += _t.process_time() - s0
    t["total"] = _t.process_time() - run_t0
    return dict(t), dict(c)


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    seeds = [f"BEAMQ{i}" for i in range(n)]
    wall0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        results = list(ex.map(run_task, seeds))
    wall = time.perf_counter() - wall0

    t = defaultdict(float)
    c = defaultdict(int)
    for tt, cc in results:
        for k, v in tt.items():
            t[k] += v
        for k, v in cc.items():
            c[k] += v
    total = t["total"]
    pct = lambda v: f"{100*v/total:4.1f}%"
    print(f"n={n} jobs={jobs}  wall={wall:.1f}s  total CPU={total:.0f}s ({total/n:.1f}s/run)")
    print("TOP-LEVEL (by call site):")
    print(f"  play choose_action     {t['ca_play']:7.0f}s  ({pct(t['ca_play'])})")
    print(f"  shop choose_action     {t['ca_shop']:7.0f}s  ({pct(t['ca_shop'])})")
    print(f"  other choose_action    {t['ca_other']:7.0f}s  ({pct(t['ca_other'])})")
    print(f"  sim.step               {t['sim_step']:7.0f}s  ({pct(t['sim_step'])})")
    print("PLAY-SEARCH SUB-BREAKDOWN (subset of play choose_action):")
    print(f"  Rust rollout           {t['rust_rollout']:7.0f}s  ({pct(t['rust_rollout'])})  "
          f"bail {c['rust_rollout_bail']}/{c['rust_rollout_calls']} "
          f"({100*c['rust_rollout_bail']/max(1,c['rust_rollout_calls']):.0f}%)")
    print(f"  Python rollout (bail)  {t['python_rollout']:7.0f}s  ({pct(t['python_rollout'])})  "
          f"calls={c['python_rollout_calls']}")
    print(f"  Headroom best-score    {t['headroom_bestscore']:7.0f}s  ({pct(t['headroom_bestscore'])})")
    beam_exp = t["ca_play"] - t["rust_rollout"] - t["python_rollout"] - t["headroom_bestscore"]
    print(f"  beam expand+enum (rem) {beam_exp:7.0f}s  ({pct(beam_exp)})")
    print("  BEAM-EXPAND SUB-BREAKDOWN:")
    print(f"    simulate_play/disc   {t['beam_simulate']:7.0f}s  ({pct(t['beam_simulate'])})")
    print(f"    legal_actions derive {t['beam_legal_actions']:7.0f}s  ({pct(t['beam_legal_actions'])})")
    print(f"    future-action enum   {t['beam_enum']:7.0f}s  ({pct(t['beam_enum'])})")
    print(f"    root-action enum     {t['beam_root_enum']:7.0f}s  ({pct(t['beam_root_enum'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
