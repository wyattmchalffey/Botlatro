"""Audit WHY the Rust clear_probability rollout bails to Python.

The profile shows the play beam spends ~56% of runtime in clear_probability,
almost all of it in the PYTHON `_greedy_rollout_clears` fallback (the Rust
`clear_probability_native` fast path is bailing). Each bail is ~92ms of Python
hand evaluation; moving them to Rust is a ~2x data-gen lever.

This monkeypatches `_try_rust_clear_probability` to tally, for every leaf
clear-probability call on a real trajectory, the bail reason:
  - early-return states never reach the bridge (handled before it)
  - blind-not-safe
  - glass/blue-seal card
  - simple-bail joker (Vampire etc.)
  - unsupported joker (which ones)
  - rust-success
So we see exactly which jokers/blinds to add support for.

    PYTHONPATH=src python scripts/rollout_bail_audit.py [n_seeds]
"""

from __future__ import annotations

import sys
from collections import Counter


def main() -> int:
    import balatro_ai.search.state_value as sv
    from balatro_ai.search.state_value import (
        _RUST_ROLLOUT_BLIND_SAFE, _PLAYED_STATE_JOKERS,
    )
    from balatro_ai.solver.policy import SolverPolicy
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int
    from balatro_ai.sim.local_runner import LocalBalatroSimulator

    import balatro_core

    reasons = Counter()
    bail_jokers = Counter()      # precise culprit jokers (via joker_bail_reason)
    bail_blinds = Counter()      # which boss blinds force the fallback
    orig = sv._try_rust_clear_probability

    def classify(state) -> str:
        # Precise attribution using the EXACT Rust predicates.
        if state.blind not in _RUST_ROLLOUT_BLIND_SAFE:
            bail_blinds[state.blind or "(none)"] += 1
            return "blind-not-safe"
        for c in list(state.hand or ()) + list(state.known_deck or ()):
            enh = getattr(c, "enhancement", None)
            seal = getattr(c, "seal", None)
            if str(getattr(enh, "value", enh)) == "glass" or str(getattr(seal, "value", seal)) == "blue":
                return "glass-or-blue-seal"
        culprits = []
        for j in state.jokers:
            reason = balatro_core.joker_bail_reason(j.name)
            if reason is not None:
                culprits.append((j.name, reason))
        if culprits:
            for name, reason in culprits:
                bail_jokers[f"{name} [{reason}]"] += 1
            # Reason bucket = the first culprit's reason (the one the Rust loop hits).
            return f"{culprits[0][1]}-joker"
        return "other"  # bailed despite passing all cheap checks (investigate)

    def wrapped(state, samples, seed):
        result = orig(state, samples, seed)
        if result is None:
            reasons[classify(state)] += 1
        else:
            reasons["rust-success"] += 1
        return result

    sv._try_rust_clear_probability = wrapped

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    seeds = [f"BEAMQ{i}" for i in range(n)]
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

    total = sum(reasons.values())
    print(f"clear_probability bridge calls: {total}")
    for reason, cnt in reasons.most_common():
        print(f"  {reason:24s} {cnt:7d} ({100*cnt/max(1,total):.1f}%)")
    print("\nTop bail-causing jokers:")
    for name, cnt in bail_jokers.most_common(20):
        print(f"  {name:34s} {cnt}")
    print("\nTop bail-causing blinds (blind-not-safe):")
    for name, cnt in bail_blinds.most_common(20):
        print(f"  {name:24s} {cnt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
