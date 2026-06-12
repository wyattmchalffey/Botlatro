"""Scratch: replay 0000015 to step 85 on the bridge; dump hands/discards/blind
both sides for steps 79-85 and the raw bridge state around the cash_out.

P04 round-3 aid for the $1 cash_out gap at step 84->85.
Usage: PYTHONPATH=src python scripts/_scratch_p04_cashout15.py 0000015 85
"""
from __future__ import annotations

import json
import os
import sys

os.environ["BALATRO_SEED_FAITHFUL"] = "1"

from balatro_ai.api.actions import ActionType  # noqa: E402
from balatro_ai.api.client import JsonRpcBalatroClient  # noqa: E402
from balatro_ai.api.state import GameState, with_derived_legal_actions  # noqa: E402
from balatro_ai.bots.registry import create_bot  # noqa: E402
from balatro_ai.rng.capture_shop_sequence import _settled_raw_state, _bridge_state  # noqa: E402
from balatro_ai.sim.local_runner import LocalBalatroSimulator  # noqa: E402
from balatro_ai.solver.seed_game import SeedGame  # noqa: E402
from balatro_ai.solver.trajectory import _stable_seed_int  # noqa: E402


def _brief(state: GameState) -> dict:
    return {
        "phase": state.phase.value,
        "ante": state.ante,
        "blind": state.blind,
        "score": state.current_score,
        "money": state.money,
        "hands": state.hands_remaining,
        "discards": state.discards_remaining,
    }


def main():
    seed = sys.argv[1]
    max_replay = int(sys.argv[2]) if len(sys.argv) > 2 else 85
    lo = int(sys.argv[3]) if len(sys.argv) > 3 else 79

    sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white", balatro_seed=seed)
    sim.state = with_derived_legal_actions(SeedGame(seed, stake="white").initial_state())
    bot = create_bot("basic_strategy_bot", seed=0)
    records = []
    for _ in range(max_replay + 1):
        state = sim.state
        if state.run_over:
            break
        action = bot.choose_action(state)
        if action.action_type == ActionType.NO_OP:
            break
        records.append((action, _brief(state)))
        sim.step(action)

    print(f"sim recorded {len(records)} steps; replaying on bridge")
    client = JsonRpcBalatroClient(endpoint="http://127.0.0.1:12346", timeout_seconds=120.0)
    _bridge_state(client, "menu", None)
    raw = _settled_raw_state(client, _bridge_state(client, "start", {"deck": "RED", "stake": "WHITE", "seed": seed}))
    state = with_derived_legal_actions(GameState.from_mapping(raw))

    for i, (action, sim_pre) in enumerate(records):
        bridge_pre = _brief(state)
        if i >= lo:
            marks = [k for k, v in sim_pre.items() if bridge_pre.get(k) != v]
            print(f"\n[{i:02d}] ACTION={action.action_type.value}{tuple(action.card_indices)}  DIFF={marks or 'none'}")
            print(f"  sim_pre   = {sim_pre}")
            print(f"  bridge_pre= {bridge_pre}")
            if sim_pre["phase"] == "round_eval":
                ce = raw.get("current_eval") or raw.get("round_eval") or {}
                rnd = raw.get("round") or {}
                print(f"  bridge_round_raw={json.dumps({k: rnd.get(k) for k in ('hands_left', 'discards_left', 'dollars', 'reroll_cost', 'dollars_to_be_earned')})}")
                if ce:
                    print(f"  bridge_eval_raw={json.dumps(ce)[:600]}")
        method, params = client._action_to_rpc(action)
        raw = _settled_raw_state(client, _bridge_state(client, method, params), polls=14)
        state = with_derived_legal_actions(GameState.from_mapping(raw))
        if state.run_over:
            print("bridge run_over")
            break
    print(f"\nFINAL bridge_post={_brief(state)}")


if __name__ == "__main__":
    main()
