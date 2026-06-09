"""Gate: does the on-policy value net discriminate build value at ante-8 shops?

Uses the out-test's ground truth (which grafted jokers actually clear the failing blind).
For each ante-8 loss: re-derive the failing build state + the affordable candidate jokers,
compute the value net's win-probability for base and for base+each-candidate, and label each
candidate by whether it was an OUT (clears, from endgame_out_test.json).

If V ranks clearing graftings above non-clearing ones (AUC >> 0.5) and adding an out RAISES
win_prob, the build-construction lane has a learnable foothold near the terminal (propagatable
by TD/search). If AUC ~ 0.5 / flat, V is blind to build construction -> lane is hard.

No bot rollouts (just V forward passes), so it's fast.

    PYTHONPATH=src py -3.12 scripts/value_buildgate.py \
        --caps .data/onpolicy_solver_caps_384.jsonl --outtest .data/endgame_out_test.json \
        --ckpt .data/value_onpolicy_attn_v1.pt --jobs 8 --out .data/value_buildgate.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

_MODEL = None
_CKPT = None


def _winprob(model, state):
    import torch
    from balatro_ai.ml.encoding import encode_state
    from balatro_ai.ml.model import collate_states
    batch = collate_states([encode_state(state)])
    with torch.no_grad():
        return float(torch.sigmoid(model.forward(batch))[0])


def _worker(arg):
    global _MODEL, _CKPT
    cap_dict, ckpt, out_names = arg
    from dataclasses import replace as dcr
    from balatro_ai.api.actions import Action, ActionType
    from balatro_ai.api.state import GamePhase, with_derived_legal_actions
    from balatro_ai.ml.train import load_checkpoint
    from balatro_ai.sim.local_runner import LocalBalatroSimulator
    from balatro_ai.solver.seed_game import SeedGame
    from balatro_ai.solver.trajectory import _stable_seed_int

    if _MODEL is None or _CKPT != ckpt:
        _MODEL = load_checkpoint(ckpt)
        _MODEL.eval()
        _CKPT = ckpt

    active = {GamePhase.SELECTING_HAND, GamePhase.PLAYING_BLIND, GamePhase.ROUND_EVAL}
    seed = cap_dict["seed"]
    sim = LocalBalatroSimulator(seed=_stable_seed_int(seed), stake="white")
    sim.state = SeedGame(seed, stake="white").initial_state()
    prev = sim.state.phase
    last_blind = None
    candidates = {}
    for ad in cap_dict.get("actions", ()):
        st = sim.state
        if st.phase == GamePhase.SHOP:
            for i, sc in enumerate(st.modifiers.get("shop_cards", [])):
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
            last_blind = copy.deepcopy(sim.state)
        prev = sim.state.phase

    if last_blind is None or int(last_blind.ante) != 8:
        return None
    owned = {j.name for j in last_blind.jokers}
    base_v = _winprob(_MODEL, last_blind)
    rows = []
    for nm, jk in candidates.items():
        if nm in owned:
            continue
        aug = with_derived_legal_actions(dcr(last_blind, jokers=(*last_blind.jokers, jk)))
        rows.append({"name": nm, "v": _winprob(_MODEL, aug), "clears": nm in out_names})
    return {"seed": seed, "base_v": base_v, "candidates": rows}


def _auc(pos, neg):
    # rank-based AUC = P(V_pos > V_neg)
    if not pos or not neg:
        return None
    wins = ties = 0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return round((wins + 0.5 * ties) / (len(pos) * len(neg)), 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", required=True)
    ap.add_argument("--outtest", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    caps = {json.loads(l)["seed"]: json.loads(l) for l in open(args.caps, encoding="utf-8") if l.strip()}
    ot = json.load(open(args.outtest, encoding="utf-8"))
    out_by_seed = {r["seed"]: set(r.get("outs", [])) for r in ot.get("rows", []) if int(r.get("required", 0)) > 0}
    jobs = [(caps[s], args.ckpt, out_by_seed.get(s, set())) for s in out_by_seed if s in caps]
    print(f"[gate] {len(jobs)} ante-8 loss runs to score with the value net", flush=True)

    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            results = [r for r in ex.map(_worker, jobs) if r is not None]
    else:
        results = [r for r in (_worker(j) for j in jobs) if r is not None]

    out_v, nonout_v, base_v, raise_out, raise_nonout = [], [], [], [], []
    per_loss_std = []
    for r in results:
        base_v.append(r["base_v"])
        cv = [c["v"] for c in r["candidates"]]
        if len(cv) >= 2:
            per_loss_std.append(statistics.pstdev(cv))
        for c in r["candidates"]:
            (out_v if c["clears"] else nonout_v).append(c["v"])
            (raise_out if c["clears"] else raise_nonout).append(c["v"] - r["base_v"])

    summary = {
        "n_losses_scored": len(results),
        "n_out_graftings": len(out_v),
        "n_nonout_graftings": len(nonout_v),
        "AUC_winprob_predicts_clear": _auc(out_v, nonout_v),
        "mean_winprob_base": round(statistics.mean(base_v), 4) if base_v else None,
        "mean_winprob_out_graft": round(statistics.mean(out_v), 4) if out_v else None,
        "mean_winprob_nonout_graft": round(statistics.mean(nonout_v), 4) if nonout_v else None,
        "mean_winprob_rise_from_out": round(statistics.mean(raise_out), 4) if raise_out else None,
        "mean_winprob_rise_from_nonout": round(statistics.mean(raise_nonout), 4) if raise_nonout else None,
        "mean_per_loss_candidate_winprob_std": round(statistics.mean(per_loss_std), 4) if per_loss_std else None,
    }
    print(json.dumps(summary, indent=2), flush=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "results": results}, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
