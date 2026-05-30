"""Validate a rollout un-bail: Rust clear-prob vs Python ground truth, no bias.

After widening the Rust rollout's coverage (a boss blind added to
_RUST_ROLLOUT_BLIND_SAFE, or a joker scoring port added), this checks the
un-bailed Rust `clear_probability_native` against the Python
`_greedy_rollout_clears` ground truth on REAL states matching a filter.

Both sides are noise-averaged over K samples. The rollout is a stochastic
estimator (Rust xoshiro draws != Python random.Random draws), so per-state
estimates differ by sampling noise; what must NOT happen is a SYSTEMATIC bias
(Rust consistently over/under-estimating clear probability), which would skew
the beam's decisions. Reports mean signed bias, mean abs diff, and the worst
divergences.

    PYTHONPATH=src python scripts/rollout_parity_check.py <filter> [n_seeds] [K]
        <filter> = "blind:The Needle"  | "joker:Showman" | "any-bailed"
"""

from __future__ import annotations

import statistics
import sys
from random import Random


def main() -> int:
    import balatro_ai.search.state_value as sv
    from balatro_ai.search.state_value import _greedy_rollout_clears, _try_rust_clear_probability
    from balatro_ai.solver.policy import SolverPolicy
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int
    from balatro_ai.sim.local_runner import LocalBalatroSimulator

    filt = sys.argv[1] if len(sys.argv) > 1 else "any-bailed"
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 128

    kind, _, val = filt.partition(":")

    def matches(state) -> bool:
        if state.current_score >= state.required_score or state.hands_remaining <= 0 or not state.hand:
            return False
        if kind == "blind":
            return state.blind == val
        if kind == "joker":
            return any(j.name == val for j in state.jokers)
        return True  # any-bailed: collect every non-trivial leaf state

    # Collect candidate states from real trajectories.
    states = []
    seeds = [f"BEAMQ{i}" for i in range(n_seeds)]
    for seed in seeds:
        sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
        sim.state = SeedGame(seed, stake="white").initial_state()
        pol = SolverPolicy(seed=0)
        for _ in range(6000):
            st = sim.state
            if st.run_over:
                break
            if matches(st):
                states.append(st)
            a = pol.choose_action(st)
            if a.action_type.value == "no_op":
                break
            sim.step(a)
        if len(states) >= 400:
            break

    if not states:
        print(f"no states matched filter {filt!r}")
        return 1

    # Dedupe by identity-ish key to avoid over-weighting repeated states.
    seen = set()
    uniq = []
    for s in states:
        key = (s.blind, s.current_score, s.required_score, s.hands_remaining,
               s.discards_remaining, tuple((c.rank, c.suit) for c in s.hand),
               tuple(j.name for j in s.jokers))
        if key not in seen:
            seen.add(key)
            uniq.append(s)
    uniq = uniq[:300]

    rust_bailed = 0
    diffs = []
    rows = []
    for i, st in enumerate(uniq):
        rust = _try_rust_clear_probability(st, K, 12345 + i)
        if rust is None:
            rust_bailed += 1
            continue
        rng = Random(98765 + i)
        clears = sum(1 for _ in range(K) if _greedy_rollout_clears(st, rng))
        py = clears / K
        diffs.append(rust - py)
        rows.append((abs(rust - py), rust, py, st))

    print(f"filter={filt!r}  matched-unique={len(uniq)}  compared={len(diffs)}  "
          f"rust-still-bailed={rust_bailed}")
    if not diffs:
        print("  Rust bailed on ALL matched states — un-bail not active for this filter.")
        return 1
    bias = statistics.mean(diffs)
    mad = statistics.mean(abs(d) for d in diffs)
    print(f"  signed bias (rust-py): {bias:+.4f}   (|bias| small = no systematic skew)")
    print(f"  mean abs diff:         {mad:.4f}   (sampling noise floor)")
    print(f"  stdev of diff:         {statistics.pstdev(diffs):.4f}")
    rows.sort(key=lambda r: r[0], reverse=True)
    print("  worst 6 divergences (|diff|, rust, py):")
    for ad, ru, py, st in rows[:6]:
        print(f"    |{ad:.3f}| rust={ru:.3f} py={py:.3f}  blind={st.blind!r} "
              f"score={st.current_score}/{st.required_score} hands={st.hands_remaining} "
              f"jokers={[j.name for j in st.jokers]}")
    # Verdict: bias within ~1 sample-stderr of 0 is clean.
    stderr = statistics.pstdev(diffs) / (len(diffs) ** 0.5)
    verdict = "CLEAN (no systematic bias)" if abs(bias) <= max(0.02, 2 * stderr) else "BIASED — investigate"
    print(f"  VERDICT: {verdict}  (bias {bias:+.4f}, 2*stderr {2*stderr:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
