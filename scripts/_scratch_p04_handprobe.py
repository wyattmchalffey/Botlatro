"""Scratch (P04 r5): at a play step, dump the BRIDGE's hand in its native
display order (with seals/enhancements/editions) and which cards the play
indices select, vs the sim's. Catches sort-order / per-card-state divergences
that make an identical-multiset play score differently.

Usage: PYTHONPATH=src python scripts/_scratch_p04_handprobe.py 0000014 117
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


def _card_desc(c):
    bits = [f"{c.suit}_{c.rank}"]
    seal = getattr(c, "seal", None)
    enh = getattr(c, "enhancement", None)
    ed = getattr(c, "edition", None)
    if seal:
        bits.append(f"seal={seal}")
    if enh:
        bits.append(f"enh={enh}")
    if ed:
        bits.append(f"ed={ed}")
    return "/".join(bits)


def main():
    seed = sys.argv[1]
    target = int(sys.argv[2])
    _, records = run_sim(seed)

    client = JsonRpcBalatroClient(endpoint="http://127.0.0.1:12346", timeout_seconds=120.0)
    _bridge_state(client, "menu", None)
    raw = _settled_raw_state(client, _bridge_state(client, "start", {"deck": "RED", "stake": "WHITE", "seed": seed}))
    state = with_derived_legal_actions(GameState.from_mapping(raw))

    for i, (action, sim_pre) in enumerate(records):
        if i == target:
            idx = tuple(action.card_indices)
            print(f"=== STEP {target} action={action.action_type.value}{idx} ===")
            print("BRIDGE hand (native order):")
            for k, c in enumerate(state.hand):
                mark = " <== SELECTED" if k in idx else ""
                print(f"  [{k}] {_card_desc(c)}{mark}")
            print(f"BRIDGE selected by idx {idx}: {[_card_desc(state.hand[k]) for k in idx if k < len(state.hand)]}")
            print(f"BRIDGE money={state.money}")
            print(f"sim_pre money={sim_pre['money']} hand={sim_pre['hand']}")
            return
        method, params = client._action_to_rpc(action)
        raw = _settled_raw_state(client, _bridge_state(client, method, params), polls=18)
        state = with_derived_legal_actions(GameState.from_mapping(raw))
        if state.run_over:
            print("run_over")
            return


if __name__ == "__main__":
    main()
