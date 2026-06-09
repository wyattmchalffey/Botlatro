"""Out-test: how many ante-8 losses had an available joker that would clear the failing blind.

For each ante-8 loss (from cached solver captures): replay the run; at every shop, fork+buy
each AFFORDABLE joker offer to obtain a properly-constructed joker object; then graft each such
joker (not already in the final build) onto the failing ante-8 blind (forking the live sim so the
exact RNG/draws are preserved) and run the bot's normal play to see if it now clears.

An "out" = an affordable, offered joker that flips the failing blind to a clear. This is the
best-case (slot-permitting) reading of "a joker of their choice": the joker is ADDED to the build.

    PYTHONPATH=src py -3.12 scripts/endgame_out_test.py \
        --caps .data/onpolicy_solver_caps_384.jsonl --jobs 8 --out .data/endgame_out_test.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _worker(cap_dict):
    from dataclasses import replace as dcr
    from balatro_ai.api.actions import Action, ActionType
    from balatro_ai.api.state import GamePhase, with_derived_legal_actions
    from balatro_ai.bots.config import DEFAULT_CONFIG, bot_config_scope
    from balatro_ai.bots.registry import create_bot
    from balatro_ai.sim.local_runner import LocalBalatroSimulator
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int

    if cap_dict.get("won"):
        return None
    active = {GamePhase.SELECTING_HAND, GamePhase.PLAYING_BLIND, GamePhase.ROUND_EVAL}
    left = {GamePhase.SHOP, GamePhase.BLIND_SELECT, GamePhase.RUN_OVER}
    seed = cap_dict["seed"]

    sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
    sim.state = SeedGame(seed, stake="white").initial_state()
    prev = sim.state.phase
    last_blind_sim = None
    candidates = {}  # name -> Joker object (freshly bought, base config)

    for ad in cap_dict.get("actions", ()):
        st = sim.state
        if st.phase == GamePhase.SHOP:
            cards = st.modifiers.get("shop_cards", [])
            for i, sc in enumerate(cards):
                if str(sc.get("set", "")).upper() != "JOKER":
                    continue
                nm = sc.get("name")
                if not nm or nm in candidates:
                    continue
                cost = sc.get("cost", {})
                c = cost.get("buy", cost.get("base", 999)) if isinstance(cost, dict) else 999
                if c > st.money:
                    continue
                f = copy.deepcopy(sim)
                before = {id(j) for j in f.state.jokers}
                try:
                    f.step(Action(ActionType.BUY, target_id="card", amount=i,
                                  metadata={"kind": "card", "index": i}))
                except Exception:
                    continue
                new = [j for j in f.state.jokers if id(j) not in before] or \
                      [j for j in f.state.jokers if j.name == nm]
                if new:
                    candidates[nm] = new[-1]
        try:
            sim.step(Action.from_mapping(ad))
        except Exception:
            break
        if prev == GamePhase.BLIND_SELECT and sim.state.phase in active:
            last_blind_sim = copy.deepcopy(sim)
        prev = sim.state.phase

    if last_blind_sim is None:
        return None
    bs = last_blind_sim.state
    if int(bs.ante) != 8 or int(bs.required_score) <= 0:
        return None
    required = int(bs.required_score)
    owned = {j.name for j in bs.jokers}
    cands = {n: j for n, j in candidates.items() if n not in owned}
    bot = create_bot("solver_shop_basic_play_bot", 0)

    def _peak_after_play(base_sim, extra_joker=None):
        s = copy.deepcopy(base_sim)
        if extra_joker is not None:
            s.state = with_derived_legal_actions(dcr(s.state, jokers=(*s.state.jokers, extra_joker)))
        peak = int(s.state.current_score)
        with bot_config_scope(dcr(DEFAULT_CONFIG, shop_audit_enabled=False)):
            for _ in range(250):
                stx = s.state
                peak = max(peak, int(stx.current_score))
                if stx.run_over or stx.phase in left or int(stx.ante) > 8:
                    break
                if stx.phase not in active:
                    break
                try:
                    a = bot.choose_action(stx)
                except Exception:
                    break
                if a is None or a.action_type == ActionType.NO_OP:
                    break
                try:
                    s.step(a)
                except Exception:
                    break
        return max(peak, int(s.state.current_score))

    base_peak = _peak_after_play(last_blind_sim)
    base_clears = base_peak >= required
    outs = []
    for nm, jk in cands.items():
        if _peak_after_play(last_blind_sim, jk) >= required:
            outs.append(nm)
    return {
        "seed": seed, "required": required,
        "ratio": round(cap_dict.get("final_score", 0) / required, 3),
        "base_peak_ratio": round(base_peak / required, 3),
        "n_candidates": len(cands), "n_outs": len(outs),
        "had_out": len(outs) > 0, "base_clears": base_clears, "outs": outs[:12],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="cap losses processed (0=all)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    caps = [json.loads(l) for l in open(args.caps, encoding="utf-8") if l.strip()]
    losses = [c for c in caps if not c.get("won")]
    if args.limit:
        losses = losses[: args.limit]
    print(f"[out-test] {len(losses)} losses to scan for ante-8 outs", flush=True)
    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            rows = [r for r in ex.map(_worker, losses) if r is not None]
    else:
        rows = [r for r in (_worker(c) for c in losses) if r is not None]

    # restrict to faithful baselines (reload/fork reproduces the loss)
    faithful = [r for r in rows if not r["base_clears"]]
    n = len(faithful)
    out = {
        "n_ante8_losses": len(rows),
        "n_base_clears_excluded": len(rows) - n,
        "n_faithful": n,
        "had_out_rate": round(sum(r["had_out"] for r in faithful) / n, 3) if n else None,
        "n_with_out": sum(r["had_out"] for r in faithful),
        "mean_n_outs": round(statistics.mean(r["n_outs"] for r in faithful), 2) if n else None,
        "mean_n_candidates": round(statistics.mean(r["n_candidates"] for r in faithful), 1) if n else None,
        "rows": sorted(rows, key=lambda r: -r["n_outs"]),
    }
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2), flush=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
