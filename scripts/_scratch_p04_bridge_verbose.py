"""Scratch: ONE instrumented bridge replay for P04 root-cause (seed 0000048).

Replays the sim's recorded actions on the live bridge, printing FULL state
both sides each step, including raw bridge joker entries (to read Popcorn's
displayed mult and Rocket's displayed dollars).

Usage: PYTHONPATH=src python scripts/_scratch_p04_bridge_verbose.py 0000048 15
"""
from __future__ import annotations

import json
import os
import sys

os.environ["BALATRO_SEED_FAITHFUL"] = "1"

from balatro_ai.api.client import JsonRpcBalatroClient  # noqa: E402
from balatro_ai.api.state import GameState, with_derived_legal_actions  # noqa: E402
from balatro_ai.rng.capture_shop_sequence import _settled_raw_state, _bridge_state  # noqa: E402
from balatro_ai.sim.local_runner import LocalBalatroSimulator  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from sim_bridge_replay_check import run_sim, _summary  # noqa: E402


def _joker_brief(raw: dict) -> list:
    out = []
    for j in raw.get("jokers", []) or []:
        if isinstance(j, dict):
            keep = {k: j.get(k) for k in ("label", "name", "key", "edition") if j.get(k)}
            val = j.get("value")
            if isinstance(val, dict):
                keep["value"] = {k: v for k, v in val.items() if k in ("mult", "chips", "x_mult", "dollars", "effect", "extra", "current_mult")}
            out.append(keep)
        else:
            out.append(j)
    return out


def main():
    seed = sys.argv[1]
    max_replay = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    _, records = run_sim(seed)
    print(f"sim recorded {len(records)} steps; replaying first {max_replay + 1}")

    client = JsonRpcBalatroClient(endpoint="http://127.0.0.1:12346", timeout_seconds=120.0)
    _bridge_state(client, "menu", None)
    raw = _settled_raw_state(client, _bridge_state(client, "start", {"deck": "RED", "stake": "WHITE", "seed": seed}))
    state = with_derived_legal_actions(GameState.from_mapping(raw))

    for i, (action, sim_pre) in enumerate(records[: max_replay + 1]):
        bridge_pre = _summary(state)
        marks = []
        for key in ("phase", "ante", "score", "money", "hand"):
            if bridge_pre[key] != sim_pre[key]:
                marks.append(key)
        print(f"\n[{i:02d}] ACTION={action.action_type.value}{tuple(action.card_indices)}  DIFF={marks or 'none'}")
        print(f"  sim_pre   = {sim_pre}")
        print(f"  bridge_pre= {bridge_pre}")
        print(f"  bridge_jokers={json.dumps(_joker_brief(raw))}")
        method, params = client._action_to_rpc(action)
        raw = _settled_raw_state(client, _bridge_state(client, method, params), polls=14)
        state = with_derived_legal_actions(GameState.from_mapping(raw))
        if state.run_over:
            print("bridge run_over")
            break
    bridge_post = _summary(state)
    print(f"\nFINAL bridge_post={bridge_post}")
    print(f"FINAL bridge_jokers={json.dumps(_joker_brief(raw))}")


if __name__ == "__main__":
    main()
