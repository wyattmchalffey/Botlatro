"""Scratch (P04 r5): replay seed on bridge, after each step dump the bridge's
joker list (from state) alongside the sim's, flag the first step where the
joker MULTISET diverges. Localizes a joker-acquisition divergence.

Usage: PYTHONPATH=src python scripts/_scratch_p04_jokerprobe.py 0000014 [maxstep]
"""
from __future__ import annotations

import os
import sys

os.environ["BALATRO_SEED_FAITHFUL"] = "1"

from balatro_ai.api.client import JsonRpcBalatroClient  # noqa: E402
from balatro_ai.api.state import GameState, with_derived_legal_actions  # noqa: E402
from balatro_ai.rng.capture_shop_sequence import _settled_raw_state, _bridge_state  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from sim_bridge_replay_check import run_sim  # noqa: E402


def _sim_jokers_at(seed, target):
    from balatro_ai.api.state import with_derived_legal_actions as wd
    from balatro_ai.bots.registry import create_bot
    from balatro_ai.sim.local_runner import LocalBalatroSimulator
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int
    sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white", balatro_seed=seed)
    sim.state = wd(SeedGame(seed, stake="white").initial_state())
    bot = create_bot("basic_strategy_bot", seed=0)
    out = {}
    for i in range(target + 1):
        s = sim.state
        a = bot.choose_action(s)
        out[i] = sorted(j.name for j in s.jokers)
        sim.step(a)
    return out


def main():
    seed = sys.argv[1]
    maxstep = int(sys.argv[2]) if len(sys.argv) > 2 else 118
    _, records = run_sim(seed)
    sim_jokers = _sim_jokers_at(seed, maxstep)

    client = JsonRpcBalatroClient(endpoint="http://127.0.0.1:12346", timeout_seconds=120.0)
    _bridge_state(client, "menu", None)
    raw = _settled_raw_state(client, _bridge_state(client, "start", {"deck": "RED", "stake": "WHITE", "seed": seed}))
    state = with_derived_legal_actions(GameState.from_mapping(raw))

    for i, (action, sim_pre) in enumerate(records):
        if i > maxstep:
            break
        bj = sorted(j.name for j in state.jokers)
        sj = sim_jokers.get(i, [])
        if bj != sj:
            print(f">>> JOKER DIVERGE at step {i} (pre {action.action_type.value}{tuple(action.card_indices)}) ante={sim_pre['ante']}")
            print(f"    sim   ={sj}")
            print(f"    bridge={bj}")
            print(f"    only_in_bridge={sorted(set(bj)-set(sj))} only_in_sim={sorted(set(sj)-set(bj))}")
            return
        method, params = client._action_to_rpc(action)
        raw = _settled_raw_state(client, _bridge_state(client, method, params), polls=16)
        state = with_derived_legal_actions(GameState.from_mapping(raw))
        if state.run_over:
            print("run_over")
            return
    print("no joker divergence in range")


if __name__ == "__main__":
    main()
