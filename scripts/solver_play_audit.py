"""Audit: does the SolverPolicy underplay (pick a lower-scoring hand than best)?

The ante-1-death traces show the solver playing Three of a Kind when a Full
House is available (seed 3: TTT+J instead of TTTJJ). This drives the
ante-1-death seeds and, at every play decision, compares the CHOSEN play's
immediate score to the GREEDY-BEST play's score. A chosen score far below the
best (and below what's needed to clear) is a misplay that loses runs.

    PYTHONPATH=src python scripts/solver_play_audit.py [seeds...]
"""

from __future__ import annotations

import sys


def main() -> int:
    from balatro_ai.search.state_value import _best_greedy_play_action, _score_action
    from balatro_ai.solver.policy import SolverPolicy
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int
    from balatro_ai.sim.local_runner import LocalBalatroSimulator
    from balatro_ai.api.actions import ActionType

    seeds = sys.argv[1:] or ["0000003", "0000006", "0000010"]
    total_plays = 0
    underplays = 0
    big_underplays = 0  # chosen < 60% of best AND best would have helped clear

    for seed in seeds:
        sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
        sim.state = SeedGame(seed, stake="white").initial_state()
        pol = SolverPolicy(seed=0)
        print(f"\n=== seed {seed} ===")
        for _ in range(4000):
            st = sim.state
            if st.run_over:
                break
            a = pol.choose_action(st)
            if (str(st.phase.value) in ("selecting_hand", "playing_blind")
                    and a.action_type == ActionType.PLAY_HAND and a.card_indices):
                chosen_score = _score_action(st, a)
                best = _best_greedy_play_action(st)
                best_score = _score_action(st, best) if best is not None else 0
                total_plays += 1
                tag = ""
                if best_score > chosen_score + 1:
                    underplays += 1
                    ratio = chosen_score / max(1, best_score)
                    needed = max(0, st.required_score - st.current_score)
                    # "big" = chosen misses, but best would have cleared or
                    # gotten much closer
                    if ratio < 0.7:
                        big_underplays += 1
                        tag = f"  <<< UNDERPLAY chosen={chosen_score} best={best_score} needed={needed}"
                        ci_chosen = tuple(a.card_indices)
                        ci_best = tuple(best.card_indices) if best else ()
                        print(f"  a{st.ante} {st.blind!r} score={st.current_score}/{st.required_score} "
                              f"h={st.hands_remaining}{tag}")
                        print(f"      chosen={ci_chosen} ({chosen_score})  best={ci_best} ({best_score})")
            if a.action_type.value == "no_op":
                break
            sim.step(a)

    print(f"\n--- summary ---")
    print(f"play decisions: {total_plays}")
    print(f"underplays (chosen < best): {underplays} ({100*underplays/max(1,total_plays):.0f}%)")
    print(f"BIG underplays (chosen < 70% of best): {big_underplays} "
          f"({100*big_underplays/max(1,total_plays):.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
