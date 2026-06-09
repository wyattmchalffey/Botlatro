"""Parity + speed harness for the DATA-GEN agent (SolverPolicy full play+shop).

Mirrors bot_parity_speed.py but drives the SolverPolicy used for offline data
generation (the play-beam + rollout workload that bot_parity_speed's deployed
winrate bot does NOT exercise). Captures a per-seed trajectory signature + the
per-run process_time so behaviour-preserving rollout/beam changes can be A/B'd.

    PYTHONHASHSEED=0 PYTHONPATH=src python scripts/datagen_parity.py [n] [jobs]
"""

from __future__ import annotations

import hashlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor


def run_task(seed: str):
    import time as _t

    import balatro_ai.search.hand_search as hs
    from balatro_ai.solver.policy import SolverPolicy
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int
    from balatro_ai.sim.local_runner import LocalBalatroSimulator

    hs._NATIVE_BEAM_ENABLED = False
    sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
    sim.state = SeedGame(seed, stake="white").initial_state()
    pol = SolverPolicy(seed=0)
    h = hashlib.blake2b(digest_size=16)
    n = 0
    cpu0 = _t.process_time()
    for _ in range(6000):
        st = sim.state
        if st.run_over:
            break
        a = pol.choose_action(st)
        sig = f"{st.phase.value}|{a.action_type.value}|{getattr(a,'card_indices',None)}|{getattr(a,'target_id',None)}|{getattr(a,'amount',None)}"
        h.update(sig.encode())
        n += 1
        if a.action_type.value == "no_op":
            break
        sim.step(a)
    cpu = _t.process_time() - cpu0
    s = sim.state
    h.update(f"won={bool(s.won)};ante={s.ante};score={s.current_score};steps={n}".encode())
    return {"seed": seed, "sig": h.hexdigest(), "cpu": cpu, "won": bool(s.won), "ante": s.ante, "steps": n}


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    seeds = [f"BEAMQ{i}" for i in range(n)]
    wall0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        rows = list(ex.map(run_task, seeds))
    wall = time.perf_counter() - wall0
    rows.sort(key=lambda r: r["seed"])
    total_cpu = sum(r["cpu"] for r in rows)
    combined = hashlib.blake2b(digest_size=16)
    for r in rows:
        combined.update(r["seed"].encode())
        combined.update(r["sig"].encode())
    print(f"# datagen SolverPolicy n={n} jobs={jobs} wall={wall:.1f}s total_cpu={total_cpu:.1f}s ({total_cpu/n:.2f}s/run)")
    print(f"# COMBINED_SIG={combined.hexdigest()}")
    for r in rows:
        print(f"{r['seed']} {r['sig']} cpu={r['cpu']:.2f} won={r['won']} ante={r['ante']} steps={r['steps']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
