"""cProfile a single SolverPolicy data-gen run to find the real bottleneck.

The native-beam quality gate showed the native beam is speed-neutral (0.92x):
the expensive leaf rollout is already in Rust (clear_probability_native), so
offloading the tree-walk doesn't help and per-call FFI translation offsets it.

Before re-architecting the beam, find where the ~60s/run actually goes. Prints
top functions by cumulative AND total (self) time so we can see whether the
play beam, shop search, sim, or FFI translation dominates.

    PYTHONPATH=src python scripts/profile_datagen_run.py [seed]
"""

from __future__ import annotations

import cProfile
import io
import pstats
import sys


def main() -> int:
    from balatro_ai.solver.policy import SolverPolicy
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int
    from balatro_ai.sim.local_runner import LocalBalatroSimulator

    seed = sys.argv[1] if len(sys.argv) > 1 else "BEAMQ0"

    def one_run():
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
        return sim.state

    pr = cProfile.Profile()
    pr.enable()
    final = one_run()
    pr.disable()
    print(f"seed={seed} final ante={final.ante} score={final.current_score}")

    for sort_key, label in (("cumulative", "CUMULATIVE (incl. callees)"),
                            ("tottime", "TOTTIME (self only)")):
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(sort_key)
        ps.print_stats(28)
        print(f"\n===== TOP BY {label} =====")
        # Trim pstats header noise, keep the function rows.
        for line in s.getvalue().splitlines():
            if line.strip() and ("function calls" not in line) and ("Ordered by" not in line):
                print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
