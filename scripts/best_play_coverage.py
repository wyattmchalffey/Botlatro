"""Validate the best-play-finder's candidate GENERATOR (Phase 1, isolated).

The finder generates ~10-20 canonical candidate subsets instead of all 218.
For it to be CORRECT, the brute-force best play must ALWAYS be in that set.
This drives real states, and for each play decision compares:
  - brute best play (joker-aware, _best_greedy_play_action) card indices
  - my candidate set (balatro_core.candidate_plays(hand))
and reports COVERAGE (how often brute-best is in the candidate set). Misses
reveal which subsets the generator omits (joker / Splash / Four Fingers edge
cases). 100% coverage on the simple case = generator correct.

    PYTHONPATH=src python scripts/best_play_coverage.py [n_seeds]
"""

from __future__ import annotations

import sys
from collections import Counter


def main() -> int:
    import balatro_core
    from balatro_ai.search.state_value import _best_greedy_play_action
    from balatro_ai.solver.policy import SolverPolicy
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int
    from balatro_ai.sim.local_runner import LocalBalatroSimulator

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    seeds = [f"BEAMQ{i}" for i in range(n)]

    total = 0
    covered = 0
    bailed = 0
    miss_reasons = Counter()
    miss_examples = []

    # Jokers the finder will BAIL on (canonical poker subsets aren't enough):
    #  - identification jokers change hand-type detection
    #  - held-card-pass jokers make the optimal play depend on what's HELD
    #    (Raised Fist: lowest held; Baron: Kings held; Shoot the Moon: Queens
    #    held; Mime: retrigger held) -> "play low, hold high" plays we don't gen
    HARD = {"Smeared Joker", "Four Fingers", "Shortcut",
            "Raised Fist", "Baron", "Shoot the Moon", "Mime"}

    def has_hard_joker(state) -> bool:
        return bool({j.name for j in state.jokers} & HARD)

    def has_wild(state) -> bool:
        for c in state.hand:
            enh = getattr(c, "enhancement", None)
            if str(getattr(enh, "value", enh)) == "wild":
                return True
        return False

    for seed in seeds:
        sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
        sim.state = SeedGame(seed, stake="white").initial_state()
        pol = SolverPolicy(seed=0)
        for _ in range(6000):
            st = sim.state
            if st.run_over:
                break
            if str(st.phase.value) in ("selecting_hand", "playing_blind") and st.hand:
                if has_hard_joker(st) or has_wild(st):
                    bailed += 1  # finder bails -> brute-force fallback (correct)
                    a = pol.choose_action(st)
                    if a.action_type.value == "no_op":
                        break
                    sim.step(a)
                    continue
                brute = _best_greedy_play_action(st)
                if brute is not None and brute.card_indices:
                    try:
                        hand_rust = [balatro_core.RustCard.from_python(c) for c in st.hand]
                        cands = balatro_core.candidate_plays(hand_rust)
                    except Exception as e:  # noqa: BLE001
                        miss_reasons[f"err:{type(e).__name__}"] += 1
                        cands = []
                    cand_sets = {tuple(sorted(c)) for c in cands}
                    key = tuple(sorted(int(i) for i in brute.card_indices))
                    total += 1
                    if key in cand_sets:
                        covered += 1
                    else:
                        # Classify the miss.
                        if has_wild(st):
                            r = "wild-card"
                        elif has_hard_joker(st):
                            r = "hard-joker(bail: id/held-card)"
                        elif any(j.name == "Splash" for j in st.jokers):
                            r = "splash"
                        else:
                            r = "OTHER(simple-case-miss)"
                        miss_reasons[r] += 1
                        if r.startswith("OTHER") and len(miss_examples) < 8:
                            hand_str = [(c.rank, c.suit) for c in st.hand]
                            miss_examples.append(
                                (seed, st.blind, key, hand_str,
                                 [j.name for j in st.jokers], len(cands)))
            a = pol.choose_action(st)
            if a.action_type.value == "no_op":
                break
            sim.step(a)

    print(f"play decisions (finder-eligible): {total}   bailed (hard joker/wild): {bailed}")
    print(f"covered: {covered} ({100*covered/max(1,total):.2f}%)")
    print("miss reasons:")
    for r, c in miss_reasons.most_common():
        print(f"  {r:40s} {c}")
    if miss_examples:
        print("\nOTHER (simple-case) miss examples — generator bugs to fix:")
        for seed, blind, key, hand, jokers, ncand in miss_examples:
            print(f"  seed={seed} blind={blind!r} brute={key} ncand={ncand}")
            print(f"    hand={hand} jokers={jokers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
