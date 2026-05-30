"""Native-beam LEAF parity check (Phase 4d component #5-6).

The Rust beam at depth=0 returns `leaf_value(root)`. That leaf now mirrors
Python `_planning_value_uncached` exactly:
  - headroom via best-play score + money + jokers + hands
  - clear via the SHARED discard-aware rollout (rollout.rs), 1 sample,
    XoshiroRng::new(config.seed + seed_offset)
  - planning_value formula

This harness drives a real SeedGame trajectory and, at every selecting_hand
state the bridge does NOT bail on, compares:
    rust = _try_rust_beam_plan_value(state, depth=0, config, seed_offset=0)
    py   = _beam_leaf_value(state, config, seed_offset=0)
           == planning_value(state, samples=1, seed=config.seed)

A byte-identical leaf means rust == py (to float tolerance) on every state.
Any gap pinpoints a residual parity bug (best-score, rollout-state, or formula).

    PYTHONPATH=src python scripts/native_beam_leaf_parity.py [seed] [n_seeds]
"""

from __future__ import annotations

import sys

TOL = 1e-9


def main() -> int:
    from balatro_ai.search.hand_search import (
        HandSearchConfig, _try_rust_beam_plan_value, _beam_leaf_value,
    )
    from balatro_ai.solver.policy import SolverPolicy
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int
    from balatro_ai.sim.local_runner import LocalBalatroSimulator

    base_seed = sys.argv[1] if len(sys.argv) > 1 else "AAAAAAA"
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    config = HandSearchConfig()  # seed=0, beam_leaf_samples=1, beam_width=0

    def seed_strs(base: str, n: int):
        # Deterministic family of 7-char seeds derived from base.
        alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        out = []
        h = _stable_seed_int(base)
        for i in range(n):
            h = (h * 1103515245 + 12345 + i * 7919) & 0xFFFFFFFFFFFF
            s = "".join(alpha[(h >> (k * 5)) % len(alpha)] for k in range(7))
            out.append(s)
        return out

    compared = 0
    bailed = 0
    mism = 0
    worst = 0.0
    worst_info = None
    for seed in seed_strs(base_seed, n_seeds):
        sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
        sim.state = SeedGame(seed, stake="white").initial_state()
        pol = SolverPolicy(seed=0)
        for _ in range(4000):
            st = sim.state
            if st.run_over:
                break
            if str(st.phase.value) in ("selecting_hand", "playing_blind"):
                rust = _try_rust_beam_plan_value(st, 0, config, 0)
                if rust is None:
                    bailed += 1
                else:
                    py = _beam_leaf_value(st, config=config, seed_offset=0)
                    gap = abs(rust - py)
                    compared += 1
                    if gap > TOL:
                        mism += 1
                        if gap > worst:
                            worst = gap
                            worst_info = (seed, st.ante, st.blind, st.current_score,
                                          st.required_score, st.hands_remaining,
                                          st.discards_remaining, len(st.hand or ()),
                                          rust, py)
                        if mism <= 10:
                            print(f"MISMATCH seed={seed} ante={st.ante} blind={st.blind!r} "
                                  f"score={st.current_score}/{st.required_score} "
                                  f"hands={st.hands_remaining} disc={st.discards_remaining} "
                                  f"njok={len(st.jokers)} | rust={rust:.6f} py={py:.6f} gap={gap:.2e}")
            a = pol.choose_action(st)
            if a.action_type.value == "no_op":
                break
            sim.step(a)

    print(f"\ncompared (bridge-active): {compared}")
    print(f"bailed (bridge None):     {bailed}")
    print(f"mismatches (gap>{TOL:g}):  {mism} ({100*mism/max(1,compared):.2f}%)")
    print(f"worst gap: {worst:.3e}")
    if worst_info:
        s, an, bl, cs, rs, hr, dr, nh, ru, py = worst_info
        print(f"  worst @ seed={s} ante={an} blind={bl!r} score={cs}/{rs} "
              f"hands={hr} disc={dr} handlen={nh} rust={ru:.6f} py={py:.6f}")
    print("\nVERDICT:", "LEAF PARITY OK" if mism == 0 else "LEAF DIVERGES — investigate")
    return 0 if mism == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
