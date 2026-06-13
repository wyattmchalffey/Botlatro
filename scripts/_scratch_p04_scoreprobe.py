"""Scratch (P04 round5): replay seed to a play step, apply it on the bridge,
read the resulting round chip_total / score from the bridge state AND the save.
Pinpoints whether a play that has IDENTICAL inputs (hand+levels+jokers) still
scores differently on the bridge vs sim.

Usage: PYTHONPATH=src python scripts/_scratch_p04_scoreprobe.py 0000014 117
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


def read_save():
    try:
        raw = open(SAVE_PATH, "rb").read()
        return zlib.decompress(raw, -15).decode("latin-1")
    except Exception as exc:  # noqa: BLE001
        return f"<save read failed: {exc}>"


def main():
    seed = sys.argv[1]
    play_step = int(sys.argv[2])
    _, records = run_sim(seed)

    client = JsonRpcBalatroClient(endpoint="http://127.0.0.1:12346", timeout_seconds=120.0)
    _bridge_state(client, "menu", None)
    raw = _settled_raw_state(client, _bridge_state(client, "start", {"deck": "RED", "stake": "WHITE", "seed": seed}))
    state = with_derived_legal_actions(GameState.from_mapping(raw))

    for i, (action, sim_pre) in enumerate(records):
        if i == play_step:
            print(f"PRE-play bridge={_summary(state)}")
            print(f"PRE-play sim   ={sim_pre}")
            method, params = client._action_to_rpc(action)
            raw = _settled_raw_state(client, _bridge_state(client, method, params), polls=30)
            state = with_derived_legal_actions(GameState.from_mapping(raw))
            print(f"\nPOST-play bridge={_summary(state)}")
            sim_post = records[i + 1][1] if i + 1 < len(records) else "<end>"
            print(f"POST-play sim   ={sim_post}")
            out = read_save()
            cr = re.search(r'\["current_round"\]=\{(.*?)\["dollars"\]', out, re.S)
            hp = re.search(r'\["hands_played"\]=([0-9]+)', out)
            chips = re.findall(r'\["chips"\]=([0-9.]+)', out)
            print(f"SAVE hands_played={hp.group(1) if hp else '?'} (sample chips fields={chips[:6]})")
            return
        method, params = client._action_to_rpc(action)
        raw = _settled_raw_state(client, _bridge_state(client, method, params), polls=14)
        state = with_derived_legal_actions(GameState.from_mapping(raw))
        if state.run_over:
            print("bridge run_over")
            return


if __name__ == "__main__":
    main()
