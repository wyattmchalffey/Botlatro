"""Scratch: sim-only trace printing shop slots/voucher/packs + owned vouchers per step."""
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from _scratch_p04_shopdump import _shop_dump  # noqa: E402


def main():
    seed = sys.argv[1]
    max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 115
    lo = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white", balatro_seed=seed)
    sim.state = with_derived_legal_actions(SeedGame(seed, stake="white").initial_state())
    bot = create_bot("basic_strategy_bot", seed=0)
    for i in range(max_steps):
        s = sim.state
        if s.run_over:
            print(f"RUN OVER at {i}")
            break
        action = bot.choose_action(s)
        if action.action_type == ActionType.NO_OP:
            print(f"NO_OP at {i}")
            break
        if i >= lo:
            dump = _shop_dump(s)
            cons = [getattr(c, "name", c) for c in (s.consumables or [])]
            print(f"[{i:03d}] {action.action_type.value}{tuple(action.card_indices)} meta={action.metadata} "
                  f"phase={s.phase.value} ante={s.ante} money={s.money}")
            print(f"      vouchers={list(s.vouchers)} consumables={cons}")
            print(f"      shop={dump}")
        sim.step(action)


if __name__ == "__main__":
    main()
