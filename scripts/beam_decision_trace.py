"""Instrument best_blind_beam_action on the seed-12 opening misplay.

Drives seed 0000012 to the first play decision (Small Blind 300, score 0,
needs a 316 full house that clears) and prints, for every root candidate
action: immediate score, whether it clears, and the beam's _beam_action_value.
This shows WHY the beam ranks a 15-point single card above the 316 full house.

    PYTHONPATH=src python scripts/beam_decision_trace.py [seed]
"""

from __future__ import annotations

import sys


def main() -> int:
    import balatro_ai.search.hand_search as hs
    from balatro_ai.search.state_value import _score_action, clear_probability, planning_value, headroom_value
    from balatro_ai.solver.policy import SolverPolicy
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int
    from balatro_ai.sim.local_runner import LocalBalatroSimulator
    from balatro_ai.api.actions import ActionType

    seed = sys.argv[1] if len(sys.argv) > 1 else "0000012"
    sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
    sim.state = SeedGame(seed, stake="white").initial_state()
    pol = SolverPolicy(seed=0)

    # advance to first selecting_hand
    st = sim.state
    while str(st.phase.value) not in ("selecting_hand", "playing_blind"):
        a = pol.choose_action(st)
        sim.step(a)
        st = sim.state

    cfg = pol.play_policy._config
    print(f"seed={seed} blind={st.blind!r} score={st.current_score}/{st.required_score} "
          f"hands={st.hands_remaining} disc={st.discards_remaining}")
    print(f"config: beam_depth={cfg.beam_depth} beam_width={cfg.beam_width} seed={cfg.seed}")
    print(f"chosen by solver: {pol.choose_action(st).action_type.value}"
          f"{tuple(pol.choose_action(st).card_indices or ())}")

    ctx = hs._context_from_state(st)
    actions = hs._beam_root_actions(st, cfg, ctx)
    rows = []
    for i, a in enumerate(actions):
        if a.action_type != ActionType.PLAY_HAND:
            sc = -1
        else:
            sc = _score_action(st, a)
        val = hs._beam_action_value(
            st, a, config=cfg, depth=max(0, cfg.beam_depth - 1),
            action_index=i, seed_offset=i * 1_000_003, memo=None,
        )
        clears = sc >= (st.required_score - st.current_score)
        rows.append((val, sc, clears, a.action_type.value, tuple(a.card_indices or ())))
    rows.sort(reverse=True)
    print(f"\n{'beam_value':>11} {'imm_score':>9} {'clears':>6}  action")
    for val, sc, clears, atype, idx in rows[:18]:
        print(f"{val:11.4f} {sc:9d} {str(clears):>6}  {atype}{idx}")

    # Sanity: planning_value of the cleared child vs a non-cleared child.
    print("\nsanity — planning_value building blocks:")
    print(f"  required-current = {st.required_score - st.current_score}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
