"""Scratch: bridge replay that reads G.GAME.pseudorandom stream states from
the live game's save.jkr after every step (P04 round-4 misprint probe).

The save file is the ground truth for per-key stream positions: comparing it
with the sim's BalatroRNG state after N rolls pinpoints WHERE the real game
consumed a roll the sim didn't (or vice versa).

Usage: PYTHONPATH=src python scripts/_scratch_p04_saveprobe.py 0000064 47 misprint [gros_michel ...]
"""
from __future__ import annotations

import os
import re
import sys
import zlib

os.environ["BALATRO_SEED_FAITHFUL"] = "1"

from balatro_ai.api.client import JsonRpcBalatroClient  # noqa: E402
from balatro_ai.api.state import GameState, with_derived_legal_actions  # noqa: E402
from balatro_ai.rng.capture_shop_sequence import _settled_raw_state, _bridge_state  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from sim_bridge_replay_check import run_sim, _summary  # noqa: E402

SAVE_PATH = r"C:/Users/Wyatt/AppData/Roaming/Balatro/1/save.jkr"


def read_save_streams(keys: list[str]) -> dict[str, str]:
    try:
        raw = open(SAVE_PATH, "rb").read()
        out = zlib.decompress(raw, -15).decode("latin-1")
    except Exception as exc:  # noqa: BLE001
        return {k: f"<save read failed: {exc}>" for k in keys}
    i = out.find('["pseudorandom"]')
    table = out[i : i + 4000] if i >= 0 else ""
    result = {}
    for key in keys:
        m = re.search(r'\["' + re.escape(key) + r'"\]=([0-9.]+)', table)
        result[key] = m.group(1) if m else "<absent>"
    return result


def main():
    seed = sys.argv[1]
    max_replay = int(sys.argv[2]) if len(sys.argv) > 2 else 47
    keys = sys.argv[3:] or ["misprint"]
    _, records = run_sim(seed)
    print(f"sim recorded {len(records)} steps; replaying first {max_replay + 1}; probing {keys}")

    client = JsonRpcBalatroClient(endpoint="http://127.0.0.1:12346", timeout_seconds=120.0)
    _bridge_state(client, "menu", None)
    raw = _settled_raw_state(client, _bridge_state(client, "start", {"deck": "RED", "stake": "WHITE", "seed": seed}))
    state = with_derived_legal_actions(GameState.from_mapping(raw))

    for i, (action, sim_pre) in enumerate(records[: max_replay + 1]):
        bridge_pre = _summary(state)
        marks = [k for k in ("phase", "ante", "score", "money", "hand") if bridge_pre[k] != sim_pre[k]]
        streams = read_save_streams(keys)
        print(f"[{i:02d}] pre  ACTION={action.action_type.value}{tuple(action.card_indices)} DIFF={marks or 'none'} save_streams={streams}", flush=True)
        method, params = client._action_to_rpc(action)
        raw = _settled_raw_state(client, _bridge_state(client, method, params), polls=14)
        state = with_derived_legal_actions(GameState.from_mapping(raw))
        if state.run_over:
            print("bridge run_over")
            break
    print(f"FINAL bridge_post={_summary(state)}")
    print(f"FINAL save_streams={read_save_streams(keys)}")


if __name__ == "__main__":
    main()
