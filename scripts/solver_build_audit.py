"""Audit what BUILDS the solver actually makes — to check archetype coverage.

For each seed, run the SolverPolicy to game end and report the final jokers
(by key), the hand levels, and which BUILT_IN_ARCHETYPE best fits (plus its
coherence value). This answers: does the coherence term ENGAGE (are owned
jokers in any archetype's key list?), and is the solver building uncovered
hands (e.g. Full House) that coherence would mis-steer away from?

    PYTHONPATH=src python scripts/solver_build_audit.py [n_seeds] [jobs]
"""

from __future__ import annotations

import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor


def run_task(seed: str) -> dict:
    import balatro_ai.search.hand_search as hs
    from balatro_ai.solver.archetypes import BUILT_IN_ARCHETYPES, _joker_key
    from balatro_ai.solver.policy import SolverPolicy
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int
    from balatro_ai.sim.local_runner import LocalBalatroSimulator

    hs._NATIVE_BEAM_ENABLED = False
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
    st = sim.state
    joker_keys = [_joker_key(j) for j in (getattr(st, "jokers", None) or ())]
    levels = {k: v for k, v in (getattr(st, "hand_levels", None) or {}).items() if v and v > 1}
    # best-fit archetype
    best_name, best_fit, best_match = "none", 0.0, 0
    for arch in BUILT_IN_ARCHETYPES:
        matching = sum(1 for k in joker_keys if k and k in arch.key_joker_keys)
        target_levels = sum(max(0.0, float(levels.get(ht.value, 1)) - 1.0) for ht in arch.target_hand_types)
        fit = matching * 4.0 + target_levels * 2.0
        if fit > best_fit:
            best_name, best_fit, best_match = arch.name, fit, matching
    covered = sum(1 for k in joker_keys if any(k in a.key_joker_keys for a in BUILT_IN_ARCHETYPES))
    return {
        "seed": seed, "ante": st.ante, "score": int(st.current_score),
        "joker_keys": joker_keys, "levels": levels,
        "best_arch": best_name, "best_fit": best_fit, "best_match": best_match,
        "n_jokers": len(joker_keys), "covered_jokers": covered,
    }


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    seeds = [f"{i:07d}" for i in range(1, n + 1)]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        rows = list(ex.map(run_task, seeds))

    arch_counts: Counter = Counter()
    level_hand_counts: Counter = Counter()
    joker_counts: Counter = Counter()
    total_jokers = covered_jokers = 0
    for r in rows:
        arch_counts[r["best_arch"]] += 1
        for k in r["joker_keys"]:
            if k:
                joker_counts[k] += 1
        for h in r["levels"]:
            level_hand_counts[h] += 1
        total_jokers += r["n_jokers"]
        covered_jokers += r["covered_jokers"]

    print(f"=== build audit, {n} numeric seeds ===")
    print(f"best-fit archetype distribution: {dict(arch_counts)}")
    print(f"joker coverage: {covered_jokers}/{total_jokers} owned jokers are in SOME archetype key list "
          f"({100*covered_jokers/max(1,total_jokers):.0f}%)")
    print(f"leveled hands (count of games leveling each): {dict(level_hand_counts.most_common())}")
    print(f"top 15 owned jokers: {dict(joker_counts.most_common(15))}")
    # a few sample builds
    print("--- sample final builds ---")
    for r in sorted(rows, key=lambda x: -x["ante"])[:8]:
        lv = {k: v for k, v in r["levels"].items()}
        print(f"  seed {r['seed']} ante={r['ante']} arch={r['best_arch']}(match={r['best_match']}) "
              f"levels={lv} jokers={[k for k in r['joker_keys'] if k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
