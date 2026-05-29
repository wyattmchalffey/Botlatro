"""Side-by-side native-beam divergence finder (Phase 4d groundwork).

Drives ONE simulator down the Python-beam trajectory, but at every step asks
BOTH the Python beam (BALATRO_NATIVE_BEAM off) and the Rust beam (on) for their
action on the IDENTICAL state. The first disagreement is the exact point where
the native beam's value estimates flip a decision — the thing the prior two
Phase-4d attempts never pinned down. Advances using the Python action so the
comparison stays on the reference trajectory.

    BALATRO_NATIVE_BEAM=1 PYTHONPATH=src python scripts/native_beam_divergence.py [seed]
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("BALATRO_NATIVE_BEAM", "1")  # ensure the bridge path loads


def _sig(a) -> tuple:
    return (a.action_type.value, tuple(getattr(a, "card_indices", ()) or ()))


def main() -> int:
    import balatro_ai.search.hand_search as hs
    from balatro_ai.solver.policy import SolverPolicy
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int
    from balatro_ai.sim.local_runner import LocalBalatroSimulator

    seed = sys.argv[1] if len(sys.argv) > 1 else "AAAAAAA"
    sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
    sim.state = SeedGame(seed, stake="white").initial_state()
    pol = SolverPolicy(seed=0)

    play_decisions = 0
    diverged = 0
    for i in range(5000):
        st = sim.state
        if st.run_over:
            break
        phase = str(st.phase.value)

        hs._NATIVE_BEAM_ENABLED = False
        a_py = pol.choose_action(st)
        hs._NATIVE_BEAM_ENABLED = True
        a_rust = pol.choose_action(st)
        hs._NATIVE_BEAM_ENABLED = False

        if phase in ("selecting_hand", "playing_blind"):
            play_decisions += 1
            if _sig(a_py) != _sig(a_rust):
                diverged += 1
                if diverged <= 8:
                    hand = [(c.get("rank") if isinstance(c, dict) else getattr(c, "rank", "?"),
                             c.get("suit") if isinstance(c, dict) else getattr(c, "suit", "?"))
                            for c in (st.hand or ())]
                    print(f"DIVERGENCE #{diverged} @ step {i}: ante={st.ante} blind={st.blind!r} "
                          f"req={st.required_score} score={st.current_score} "
                          f"hands={st.hands_remaining} disc={st.discards_remaining}")
                    print(f"    hand: {hand}")
                    print(f"    Python beam -> {_sig(a_py)}")
                    print(f"    Rust   beam -> {_sig(a_rust)}")

        if a_py.action_type.value == "no_op":
            break
        sim.step(a_py)

    print(f"\nplay decisions compared: {play_decisions}")
    print(f"diverged: {diverged} ({100*diverged/max(1,play_decisions):.1f}% of play decisions)")
    print(f"final (Python-beam ref): ante={sim.state.ante} score={sim.state.current_score}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
