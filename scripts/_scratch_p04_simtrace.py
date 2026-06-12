"""Scratch: verbose sim trajectory for a seed (no bridge). P04 root-cause aid."""
from __future__ import annotations

import os
import sys

os.environ["BALATRO_SEED_FAITHFUL"] = "1"

from balatro_ai.api.actions import ActionType  # noqa: E402
from balatro_ai.api.state import with_derived_legal_actions  # noqa: E402
from balatro_ai.bots.registry import create_bot  # noqa: E402
from balatro_ai.sim.local_runner import LocalBalatroSimulator  # noqa: E402
from balatro_ai.solver.seed_game import SeedGame  # noqa: E402
from balatro_ai.solver.trajectory import _stable_seed_int  # noqa: E402


def hand_ms(state):
    return sorted(f"{c.suit}_{('T' if c.rank == '10' else c.rank)}" for c in state.hand)


def main():
    seed = sys.argv[1]
    max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white", balatro_seed=seed)
    sim.state = with_derived_legal_actions(SeedGame(seed, stake="white").initial_state())
    bot = create_bot("basic_strategy_bot", seed=0)
    for i in range(max_steps):
        s = sim.state
        if s.run_over:
            print(f"RUN OVER at step {i}")
            break
        action = bot.choose_action(s)
        if action.action_type == ActionType.NO_OP:
            print(f"NO_OP at step {i}")
            break
        jokers = [getattr(j, 'name', str(j)) for j in (s.jokers or [])]
        sel = [f"{s.hand[k].suit}_{s.hand[k].rank}" for k in action.card_indices] if action.card_indices and s.hand else list(action.card_indices)
        print(f"[{i:02d}] phase={s.phase.value:15s} ante={s.ante} blind={getattr(s, 'blind_name', None) or getattr(s,'blind',None)} "
              f"score={s.current_score}/{s.required_score} money={s.money} hands={s.hands_remaining} disc={s.discards_remaining} "
              f"jokers={jokers}")
        print(f"     hand={hand_ms(s)}")
        print(f"     ACTION={action.action_type.value} idx={list(action.card_indices)} sel={sel}")
        sim.step(action)
    s = sim.state
    print(f"FINAL phase={s.phase.value} ante={s.ante} score={s.current_score} money={s.money} run_over={s.run_over}")


if __name__ == "__main__":
    main()
