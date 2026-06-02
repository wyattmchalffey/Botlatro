"""Audit early-losing seeds: find runs that die by ante <= N and dump the decision
stream (shop buys/skips + reasons, plays, the death blind) to spot bad decisions.

    PYTHONPATH=src python scripts/phase8_early_loss_audit.py --seeds 40 --max-ante 3 --show 4
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _item_label(state, action):
    """Best-effort name of the shop item an action targets."""
    md = action.metadata or {}
    for key in ("item", "name", "card_name"):
        if md.get(key):
            return str(md[key])
    tid = getattr(action, "target_id", None)
    mods = state.modifiers or {}
    for bucket in ("shop_cards", "booster_packs"):
        for it in mods.get(bucket, []) or []:
            if isinstance(it, dict) and (str(it.get("id")) == str(tid)):
                return str(it.get("name") or it.get("key") or tid)
    return str(tid) if tid is not None else ""


def _shop_offer(state):
    mods = state.modifiers or {}
    names = []
    for it in (mods.get("shop_cards") or []):
        if isinstance(it, dict):
            names.append(str(it.get("name") or it.get("key") or "?"))
    packs = [str(it.get("name") or "?") for it in (mods.get("booster_packs") or []) if isinstance(it, dict)]
    return names, packs


def _summarize_step(state, action):
    from balatro_ai.api.actions import ActionType
    from balatro_ai.api.state import GamePhase

    p = state.phase
    reason = (action.metadata or {}).get("reason", "")
    at = action.action_type
    jok = "[" + ",".join(j.name for j in state.jokers) + "]"
    if p == GamePhase.SHOP:
        offer, packs = _shop_offer(state)
        head = f"  a{state.ante} SHOP ${state.money} jok={jok} offer={offer} packs={packs}"
        act = f"-> {at.value}:{_item_label(state, action)}"
        return f"{head}\n      {act}  [{reason}]"
    if p == GamePhase.SELECTING_HAND:
        return (f"  a{state.ante} PLAY {state.blind} {state.current_score}/{state.required_score} "
                f"hands={state.hands_remaining} disc={state.discards_remaining} jok={jok} "
                f"-> {at.value} {list(action.card_indices)}  [{reason}]")
    if p == GamePhase.BLIND_SELECT:
        bt = (state.modifiers or {}).get("current_blind", {})
        return f"  a{state.ante} BLIND_SELECT {bt.get('type','?')}/{bt.get('name','?')} -> {at.value}  [{reason}]"
    if at in (ActionType.USE_CONSUMABLE, ActionType.SELL):
        return f"  a{state.ante} {p.value} -> {at.value}:{_item_label(state, action)}  [{reason}]"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--max-ante", type=int, default=3)
    ap.add_argument("--show", type=int, default=4)
    args = ap.parse_args()

    from balatro_ai.api.actions import ActionType
    from balatro_ai.api.state import GamePhase
    from balatro_ai.bots.registry import create_bot
    from balatro_ai.sim.local_runner import LocalBalatroSimulator
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int

    early = []
    for i in range(1, args.seeds + 1):
        seed = f"{i:07d}"
        sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
        sim.state = SeedGame(seed, stake="white").initial_state()
        bot = create_bot("basic_strategy_bot", seed=0)
        steps = []
        for _ in range(2000):
            st = sim.state
            if st.run_over or st.phase == GamePhase.RUN_OVER:
                break
            a = bot.choose_action(st)
            if a.action_type == ActionType.NO_OP:
                break
            line = _summarize_step(st, a)
            if line is not None:
                steps.append(line)
            sim.step(a)
        final = sim.state
        if not final.won and final.ante <= args.max_ante:
            early.append((seed, final, steps))

    print(f"[early-loss] {len(early)}/{args.seeds} seeds died by ante {args.max_ante}\n")
    for seed, final, steps in early[:args.show]:
        print(f"===== seed {seed}: DIED ante {final.ante}, "
              f"final {final.current_score}/{final.required_score} "
              f"({final.current_score / max(1, final.required_score):.0%}), "
              f"jokers={[j.name for j in final.jokers]} money=${final.money} =====")
        for s in steps:
            print(s)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
