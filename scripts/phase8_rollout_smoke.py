"""De-risk: can we roll out from a mid-run SHOP state with a fresh simulator?

The rollout-shop idea needs to take a shop leaf state, set a fresh sim to it, and
play forward a couple blinds with a cheap policy. `step()` depends on the sim's
internal RNG/deck (not just .state), so this checks the mechanic is SANE for a
rollout estimate: hands draw ~8 cards, antes advance, the run terminates.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main() -> int:
    from balatro_ai.api.state import GamePhase
    from balatro_ai.bots.registry import create_bot
    from balatro_ai.sim.local_runner import LocalBalatroSimulator
    from balatro_ai.solver.policy import SolverPolicy, _has_shop_action
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int

    # Step seed 1 to its first SHOP decision; capture that state.
    drv = SolverPolicy(play_backend="v2", play_depth=2, play_width=1, seed=0)
    sim = LocalBalatroSimulator(seed=_stable_seed_int("0000001"), stake="white")
    sim.state = SeedGame("0000001", stake="white").initial_state()
    shop_state = None
    for _ in range(400):
        st = sim.state
        if st.run_over:
            break
        if st.phase == GamePhase.SHOP and _has_shop_action(st):
            shop_state = st
            break
        sim.step(drv.choose_action(st))
    if shop_state is None:
        print("no shop state reached")
        return 1
    print(f"captured SHOP state: ante={shop_state.ante} money={shop_state.money} "
          f"jokers={len(shop_state.jokers)} deck~={len(getattr(shop_state, 'deck', ()) or ())}")

    # Fresh sim, set to the captured shop state, roll out 2 blinds with basic bot.
    for rseed in (101, 202):
        sim2 = LocalBalatroSimulator(seed=rseed, stake="white")
        sim2.state = shop_state
        bot = create_bot("basic_strategy_bot", seed=rseed)
        start_ante = shop_state.ante
        hands_seen, steps = [], 0
        trace = []
        for _ in range(300):
            s = sim2.state
            if s.run_over or s.won:
                break
            if s.ante - start_ante >= 2:
                break
            if s.phase == GamePhase.SELECTING_HAND:
                hands_seen.append(len(s.hand))
            a = bot.choose_action(s)
            if a is None or a.action_type.value == "no_op":
                break
            if not trace or trace[-1] != s.phase.value:
                trace.append(s.phase.value)
            sim2.step(a)
            steps += 1
        f = sim2.state
        print(f"  rseed={rseed}: steps={steps} start_ante={start_ante} -> end_ante={f.ante} "
              f"won={f.won} over={f.run_over} hand_sizes={sorted(set(hands_seen))} "
              f"phases={trace[:8]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
